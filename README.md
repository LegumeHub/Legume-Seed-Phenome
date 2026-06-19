# Legume Seed Phenome

Automated pipeline for legume seed phenotyping from images — measuring color, shape, and texture traits using computer vision and deep learning.

## What it does

For each seed image, the pipeline:

1. **Crops** the ruler region from the top of the image
2. **Applies gray-world white balance** for consistent color measurement
3. **Detects seed blobs** using OpenCV thresholding
4. **Segments each seed** using [SAM2](https://github.com/facebookresearch/sam2) (Facebook's Segment Anything Model 2)
5. **Extracts dominant color** per image using ColorThief, converted to CIE Lab color space
6. **Clusters seed pixels** into two groups (lighter C0, darker C1) via K-means
7. **Computes shape metrics** - area (mm², cm²) and roundness via ellipse fitting

## Output columns (per image)

| Column | Description |
|--------|-------------|
| `L_dominant`, `a_dominant`, `b_dominant` | Dominant color in standard CIE Lab |
| `L_std`, `a_std`, `b_std` | Per-channel standard deviation |
| `Chroma` | Color saturation — `sqrt(a² + b²)` |
| `Hue_angle` | Hue direction in degrees (0–360) |
| `C0_L/a/b` | Lighter cluster center (standard Lab) |
| `C1_L/a/b` | Darker cluster center (standard Lab) |
| `DeltaE` | CIE76 color distance between clusters |
| `MeanSeedArea_mm2` | Mean seed area in mm² |
| `MeanSeedArea_cm2` | Mean seed area in cm² |
| `TotalSeedArea_mm2` | Total seed area in mm² |
| `MeanRoundness` | Roundness score (0–1, 1 = perfect circle) |

## Setup (Google Colab)

```bash
!pip install git+https://github.com/facebookresearch/sam2.git
!pip install transformers accelerate colorthief
```

Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU), then upload images to `/content/`.

## Requirements

- Python 3.8+
- `opencv-python`
- `numpy`
- `pandas`
- `colorthief`
- `Pillow`
- `transformers`
- `torch` (CUDA recommended)
- `sam2`

## Configuration

Edit the **USER SETTINGS** section at the top of `seed_analysis_merged.py`:

```python
INPUT_DIR      = "/content"       # folder containing your images
FULL_WIDTH_MM  = 100.0            # physical width of the image frame in mm
CROP_TOP_PX    = 120              # pixels to crop from top (ruler)
THRESH_VALUE   = 200              # blob detection threshold
```

## Authors

- **Shubh Yadav** - domain expert, legume genetics & genomics, Tennessee State University
- **Abhisek Gupta** - computer vision pipeline & implementation, Tennessee State University
