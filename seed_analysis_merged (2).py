import os
import cv2
import numpy as np
import pandas as pd
import math
from io import BytesIO

# =====================================================
# SEED ANALYSIS — Color (Lab) + Shape per image
# =====================================================
# Combines color measurement (SAM2 + gray world WB +
# ColorThief dominant color) with shape metrics
# (area, roundness) into a single per-image CSV.
#
# Color outputs per image:
#   - L_dominant, a_dominant, b_dominant  (ColorThief → standard Lab)
#   - L_std, a_std, b_std
#   - Chroma    = sqrt(a² + b²)           (from ColorThief dominant)
#   - Hue_angle = atan2(b, a) in degrees  (from ColorThief dominant, 0–360)
#   - C0_L, C0_a, C0_b                   (lighter cluster, standard Lab)
#   - C1_L, C1_a, C1_b                   (darker cluster, standard Lab)
#   - DeltaE    = sqrt((C0_L-C1_L)² + (C0_a-C1_a)² + (C0_b-C1_b)²)
#
# No per-seed output. No debug images.
# =====================================================

# =====================================================
# SETUP CELL — run this first in a separate Colab cell:
#
#   !pip install git+https://github.com/facebookresearch/sam2.git
#   !pip install transformers accelerate colorthief
#
# Runtime → Change runtime type → T4 GPU
# Upload images via Files panel → /content/
# =====================================================

# ─────────────────────────────────────────────────
# USER SETTINGS — only edit this section
# ─────────────────────────────────────────────────

INPUT_DIR      = "/content"
SAM_CHECKPOINT = "facebook/sam2-hiera-base-plus"

CROP_TOP_PX    = 120
FULL_WIDTH_MM  = 100.0

# Blob detection
THRESH_VALUE   = 200
MIN_BLOB_PX    = 300
MAX_BLOB_FRAC  = 0.25

# SAM2 mask sanity check
MIN_MASK_PX    = 200
MAX_MASK_FRAC  = 0.30

# ColorThief quality: 1 = highest quality (slower), 10 = faster
COLORTHIEF_QUALITY = 1

# Max pixels to reconstruct for ColorThief
MAX_CT_PX = 30000

# K-means clustering
K_CLUSTERS      = 2
MAX_KMEANS_PX   = 25000   # downsample pixels for speed/stability

# ─────────────────────────────────────────────────

OUT_DIR = os.path.join(INPUT_DIR, "python_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────
# WHITE BALANCE
# ─────────────────────────────────────────────────

def gray_world_white_balance(bgr):
    b, g, r   = cv2.split(bgr.astype(np.float32))
    mean_gray = (b.mean() + g.mean() + r.mean()) / 3.0
    b = np.clip(b * (mean_gray / (b.mean() + 1e-6)), 0, 255)
    g = np.clip(g * (mean_gray / (g.mean() + 1e-6)), 0, 255)
    r = np.clip(r * (mean_gray / (r.mean() + 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


# ─────────────────────────────────────────────────
# SAM2
# ─────────────────────────────────────────────────

def load_sam():
    import torch
    from transformers import Sam2Processor, Sam2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 Base Plus on {device.upper()} ...")
    print("(First run downloads ~300MB — please wait...)\n")

    processor = Sam2Processor.from_pretrained(SAM_CHECKPOINT)
    model     = Sam2Model.from_pretrained(SAM_CHECKPOINT).to(device)
    model.eval()

    print("SAM2 loaded.\n")
    return processor, model


def segment_point(processor, model, pil_img, cx, cy, img_w, img_h):
    import torch

    device = next(model.parameters()).device

    inputs = processor(
        images=pil_img,
        input_points=[[[[cx, cy]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_masks = outputs.pred_masks[0, 0]
    iou_scores = outputs.iou_scores[0, 0]
    best_mask_tensor = pred_masks[int(iou_scores.argmax())]

    if best_mask_tensor.dtype == torch.bool:
        best_mask = best_mask_tensor.cpu().numpy()
    else:
        best_mask = (best_mask_tensor.cpu().numpy() > 0.0)

    mH, mW = best_mask.shape
    if mH != img_h or mW != img_w:
        best_mask_u8 = best_mask.astype(np.uint8) * 255
        best_mask_u8 = cv2.resize(best_mask_u8, (img_w, img_h),
                                   interpolation=cv2.INTER_NEAREST)
        best_mask = best_mask_u8 > 0

    return best_mask


# ─────────────────────────────────────────────────
# BLOB DETECTION
# ─────────────────────────────────────────────────

def find_seed_blobs(cropped, min_px, max_px):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    centers = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_px <= area <= max_px:
            centers.append((int(centroids[i][0]), int(centroids[i][1])))

    return centers


# ─────────────────────────────────────────────────
# COLORTHIEF
# ─────────────────────────────────────────────────

def pixels_to_pil(rgb_pixels):
    from PIL import Image as PILImage

    N = len(rgb_pixels)
    if N > MAX_CT_PX:
        idx        = np.random.choice(N, MAX_CT_PX, replace=False)
        rgb_pixels = rgb_pixels[idx]
        N          = MAX_CT_PX

    img_array = rgb_pixels.reshape(1, N, 3).astype(np.uint8)
    return PILImage.fromarray(img_array, mode="RGB")


def colorthief_dominant(rgb_pixels):
    from colorthief import ColorThief

    pil_img = pixels_to_pil(rgb_pixels)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    ct = ColorThief(buf)
    return ct.get_color(quality=COLORTHIEF_QUALITY)


def rgb_to_lab(r, g, b):
    """Convert a single RGB color to standard Lab (L 0–100, a/b –128 to +127)."""
    pixel_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)[0, 0]

    # OpenCV Lab → standard Lab
    L     = float(pixel_lab[0]) * (100.0 / 255.0)
    a     = float(pixel_lab[1]) - 128.0
    b_val = float(pixel_lab[2]) - 128.0

    return L, a, b_val


def pooled_dominant_lab(corrected_img, masks):
    """
    Pool all seed pixels, run ColorThief, return dominant color in standard Lab
    plus per-pixel std for QC.
    """
    all_ys, all_xs = [], []
    for m in masks:
        ys, xs = np.where(m)
        all_ys.append(ys)
        all_xs.append(xs)

    all_ys = np.concatenate(all_ys)
    all_xs = np.concatenate(all_xs)

    if len(all_ys) == 0:
        return (np.nan,) * 6

    rgb_img    = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
    pixels_rgb = rgb_img[all_ys, all_xs, :]

    dominant_rgb         = colorthief_dominant(pixels_rgb)
    L_dom, a_dom, b_dom  = rgb_to_lab(*dominant_rgb)

    lab_img    = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2LAB)
    pixels_lab = lab_img[all_ys, all_xs, :].astype(np.float32)
    L_all = pixels_lab[:, 0] * (100.0 / 255.0)
    a_all = pixels_lab[:, 1] - 128.0
    b_all = pixels_lab[:, 2] - 128.0

    return (
        L_dom, a_dom, b_dom,
        float(np.std(L_all)), float(np.std(a_all)), float(np.std(b_all)),
    )


# ─────────────────────────────────────────────────
# CHROMA AND HUE ANGLE
# Computed from ColorThief dominant standard Lab values
# ─────────────────────────────────────────────────

def compute_chroma(a, b):
    """
    Chroma C = sqrt(a² + b²)
    Measures color saturation/intensity.
    """
    return math.sqrt(a ** 2 + b ** 2)


def compute_hue_angle(a, b):
    """
    Hue angle h = atan2(b, a) converted to degrees.
    Adjusted to 0–360 range.
    """
    h = math.degrees(math.atan2(b, a))
    if h < 0:
        h += 360.0
    return h


# ─────────────────────────────────────────────────
# K-MEANS CLUSTERING (OpenCV) ON LAB PIXELS
# Returns two cluster centers sorted by Lightness
# (C0 = lighter, C1 = darker) converted to standard Lab
# ─────────────────────────────────────────────────

def run_kmeans_lab(lab_pixels_uint8, K):
    """
    lab_pixels_uint8: (N, 3) uint8 in OpenCV Lab scale
    Returns centers (K, 3) float32 in OpenCV Lab scale,
    sorted so C0 is lighter (higher L) and C1 is darker.
    """
    Z = lab_pixels_uint8.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, _, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Sort by L descending: C0 = lighter, C1 = darker
    order   = np.argsort(-centers[:, 0])
    centers = centers[order]

    return centers.astype(np.float32)


def opencv_lab_center_to_standard(center):
    """
    Convert a single OpenCV Lab center [L, a, b] (all 0–255)
    to standard Lab: L 0–100, a/b –128 to +127.
    Mirrors the same conversion used in rgb_to_lab() and Document 4.
    """
    L = float(center[0]) * (100.0 / 255.0)
    a = float(center[1]) - 128.0
    b = float(center[2]) - 128.0
    return L, a, b


def cluster_centers_from_masks(corrected_img, masks):
    """
    Pool Lab pixels from all seed masks, run K-means (K=2),
    return C0 and C1 in standard Lab (sorted lighter first).
    """
    all_ys, all_xs = [], []
    for m in masks:
        ys, xs = np.where(m)
        all_ys.append(ys)
        all_xs.append(xs)

    all_ys = np.concatenate(all_ys)
    all_xs = np.concatenate(all_xs)

    if len(all_ys) < K_CLUSTERS:
        return (np.nan,) * 6

    lab_img    = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2LAB)
    pixels_lab = lab_img[all_ys, all_xs, :]   # (N, 3) uint8 OpenCV Lab

    # Downsample for speed/stability
    N = pixels_lab.shape[0]
    if N > MAX_KMEANS_PX:
        idx        = np.random.choice(N, MAX_KMEANS_PX, replace=False)
        pixels_lab = pixels_lab[idx]

    centers = run_kmeans_lab(pixels_lab, K_CLUSTERS)   # (2, 3) OpenCV Lab

    C0_L, C0_a, C0_b = opencv_lab_center_to_standard(centers[0])
    C1_L, C1_a, C1_b = opencv_lab_center_to_standard(centers[1])

    return C0_L, C0_a, C0_b, C1_L, C1_a, C1_b


# ─────────────────────────────────────────────────
# DELTA E (CIE76)
# Euclidean distance between C0 and C1 in standard Lab
# ─────────────────────────────────────────────────

def compute_delta_e(C0_L, C0_a, C0_b, C1_L, C1_a, C1_b):
    """
    ΔE = sqrt((C0_L - C1_L)² + (C0_a - C1_a)² + (C0_b - C1_b)²)
    """
    return math.sqrt(
        (C0_L - C1_L) ** 2 +
        (C0_a - C1_a) ** 2 +
        (C0_b - C1_b) ** 2
    )


# ─────────────────────────────────────────────────
# SHAPE — roundness only
# ─────────────────────────────────────────────────

def compute_roundness(contour):
    area = float(cv2.contourArea(contour))
    if area <= 0 or len(contour) < 5:
        return 0.0
    try:
        (_, _), (ew, eh), _ = cv2.fitEllipse(contour)
        major = float(max(ew, eh))
        if major > 0:
            return (4.0 * area) / (math.pi * major * major)
    except cv2.error:
        pass
    return 0.0


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    from PIL import Image

    processor, model = load_sam()

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print("No images found in:", INPUT_DIR)
        return

    print(f"Found {len(image_files)} images. Processing...\n")

    per_image_rows = []

    for fname in image_files:
        fpath = os.path.join(INPUT_DIR, fname)
        img   = cv2.imread(fpath)
        if img is None:
            print(f"Skipped (cannot read): {fname}")
            continue

        H_img, W_img = img.shape[:2]
        mm_per_px    = FULL_WIDTH_MM / float(W_img)
        mm2_per_px2  = mm_per_px ** 2
        img_area_px  = W_img * (H_img - CROP_TOP_PX)
        max_blob_px  = MAX_BLOB_FRAC * img_area_px
        max_mask_px  = MAX_MASK_FRAC * img_area_px

        # Step 1: crop ruler
        cropped  = img[CROP_TOP_PX:H_img, 0:W_img].copy()
        H_crop   = cropped.shape[0]

        # Step 2: gray world white balance
        corrected = gray_world_white_balance(cropped)

        # Step 3: blob detection
        centers = find_seed_blobs(cropped, MIN_BLOB_PX, max_blob_px)

        if not centers:
            print(f"{fname} | WARNING: no seeds found — try lowering THRESH_VALUE")
            continue

        pil_img = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))

        # Step 4: SAM2 → one mask per seed
        seed_masks  = []
        seen_masks  = []
        roundnesses = []

        for (cx, cy) in centers:
            mask = segment_point(processor, model, pil_img, cx, cy, W_img, H_crop)
            if mask is None:
                continue

            mask_area = mask.sum()
            if mask_area < MIN_MASK_PX or mask_area > max_mask_px:
                continue

            duplicate = False
            for prev in seen_masks:
                inter   = np.logical_and(mask, prev).sum()
                smaller = min(mask_area, prev.sum())
                if (inter / smaller if smaller > 0 else 0.0) > 0.60:
                    duplicate = True
                    break
            if duplicate:
                continue

            seen_masks.append(mask)
            seed_masks.append(mask)

            # Roundness from contour
            seg_u8  = mask.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                roundnesses.append(compute_roundness(c))

        n_seeds = len(seed_masks)
        if n_seeds == 0:
            print(f"{fname} | WARNING: SAM2 produced no valid masks")
            continue

        # Step 5: ColorThief dominant Lab
        L_dom, a_dom, b_dom, L_std, a_std, b_std = pooled_dominant_lab(
            corrected, seed_masks)

        # Step 6: Chroma and Hue from ColorThief dominant
        chroma    = compute_chroma(a_dom, b_dom)   if not np.isnan(a_dom) else np.nan
        hue_angle = compute_hue_angle(a_dom, b_dom) if not np.isnan(a_dom) else np.nan

        # Step 7: K-means cluster centers (C0 lighter, C1 darker) + Delta E
        C0_L, C0_a, C0_b, C1_L, C1_a, C1_b = cluster_centers_from_masks(
            corrected, seed_masks)

        if not np.isnan(C0_L):
            delta_e = compute_delta_e(C0_L, C0_a, C0_b, C1_L, C1_a, C1_b)
        else:
            delta_e = np.nan

        # Step 8: area
        areas_mm2 = [float(m.sum()) * mm2_per_px2 for m in seed_masks]

        per_image_rows.append({
            "Image":             fname,
            "BlobsDetected":     len(centers),
            "SeedsSegmented":    n_seeds,
            # ── ColorThief dominant color ──────────────
            "L_dominant":        round(L_dom,      3),
            "a_dominant":        round(a_dom,      3),
            "b_dominant":        round(b_dom,      3),
            "L_std":             round(L_std,      3),
            "a_std":             round(a_std,      3),
            "b_std":             round(b_std,      3),
            "Chroma":            round(chroma,     3),
            "Hue_angle":         round(hue_angle,  3),
            # ── K-means cluster centers ────────────────
            "C0_L":              round(C0_L,       3),
            "C0_a":              round(C0_a,       3),
            "C0_b":              round(C0_b,       3),
            "C1_L":              round(C1_L,       3),
            "C1_a":              round(C1_a,       3),
            "C1_b":              round(C1_b,       3),
            # ── Delta E between clusters ───────────────
            "DeltaE":            round(delta_e,    3),
            # ── Area ──────────────────────────────────
            "MeanSeedArea_mm2":  round(float(np.mean(areas_mm2)), 4),
            "MeanSeedArea_cm2":  round(float(np.mean(areas_mm2)) / 100.0, 6),
            "TotalSeedArea_mm2": round(float(np.sum(areas_mm2)),  4),
            "TotalSeedArea_cm2": round(float(np.sum(areas_mm2))  / 100.0, 6),
            # ── Shape ─────────────────────────────────
            "MeanRoundness":     round(float(np.mean(roundnesses)) if roundnesses else 0.0, 4),
            "mm_per_px":         round(mm_per_px, 6),
        })

        print(
            f"{fname} | blobs={len(centers)} seeds={n_seeds} | "
            f"L={L_dom:.1f} a={a_dom:.1f} b={b_dom:.1f} | "
            f"C={chroma:.1f} h={hue_angle:.1f}° | "
            f"ΔE={delta_e:.2f} | "
            f"mean_area={np.mean(areas_mm2):.2f}mm²"
        )

    pd.DataFrame(per_image_rows).to_csv(
        os.path.join(OUT_DIR, "results_per_image.csv"), index=False
    )

    print("\n✅ DONE — zipping and downloading...\n")

    import subprocess
    zip_path = "/content/seed_results.zip"
    subprocess.run(["zip", "-r", zip_path, OUT_DIR], check=True)

    from google.colab import files
    files.download(zip_path)
    print("Download started: seed_results.zip")


main()
