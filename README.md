# SpaceNet 2 Paris — Building Segmentation Pipeline

## Project Purpose

This project builds a **building segmentation + AI explainability pipeline** using the SpaceNet 2 Paris satellite imagery dataset.

**Overall pipeline:**

```
Satellite Images → Building Segmentation Model → Explainability (Grad-CAM / SHAP) → LLM Urban Planning Report
```

**Planned model:** EfficientNetB3 + Attention U-Net

## Team Roles

| Person | Role | Responsibility |
|--------|------|----------------|
| Person 1 | **Data Lead** | Data preprocessing pipeline (this deliverable) |
| Person 2 | Model Lead | Model architecture, training, evaluation |
| Person 3 | Explainability Lead | Grad-CAM, SHAP, LLM report generation |

---

## Dataset Source

- **Dataset:** SpaceNet 2 — Building Detection (AOI 3: Paris)
- **Kaggle:** [https://www.kaggle.com/datasets/amerii/spacenet-2-building-detection-v2](https://www.kaggle.com/datasets/amerii/spacenet-2-building-detection-v2)
- **Imagery:** WorldView-3 RGB-PanSharpened GeoTIFF satellite images
- **Labels:** Building footprint polygons in GeoJSON format

---

## Directory Structure

```
Segmentation_Project/
├── archive/                          # Original downloaded dataset (DO NOT MODIFY)
│   ├── AOI_3_Paris_Test_public/
│   └── AOI_3_Paris_Train/
│       └── AOI_3_Paris_Train/
│           ├── RGB-PanSharpen/       # 1,148 satellite images (.tif)
│           ├── geojson/
│           │   └── buildings/        # 1,148 building annotation files (.geojson)
│           └── summaryData/
│
├── Train/                            # Symlinks to archive data (DO NOT BREAK)
│   ├── RGB-PanSharpen/ → archive/.../RGB-PanSharpen/
│   └── geojson/
│       └── buildings/  → archive/.../geojson/buildings/
│
├── outputs/                          # Preprocessed .npy datasets
│   ├── X_train.npy
│   ├── Y_train.npy
│   ├── X_val.npy
│   ├── Y_val.npy
│   ├── X_test.npy
│   ├── Y_test.npy
│   └── sanity_check.png
│
├── preprocessing.py                  # Data preprocessing pipeline script
├── README.md                         # This file
└── .gitignore
```

---

## Image / GeoJSON Pairing

Each satellite image has a corresponding building GeoJSON file. The naming convention:

| Type | Filename Pattern | Example |
|------|-----------------|---------|
| Satellite image | `RGB-PanSharpen_<ID>.tif` | `RGB-PanSharpen_AOI_3_Paris_img100.tif` |
| Building labels | `buildings_<ID>.geojson` | `buildings_AOI_3_Paris_img100.geojson` |

The common `<ID>` (e.g., `AOI_3_Paris_img100`) is extracted by stripping the respective prefix and extension. **All 1,148 images have a matching GeoJSON — zero unmatched files.**

---

## Mask Generation Methodology

The `geojson_to_mask()` function in `preprocessing.py`:

1. Opens the satellite GeoTIFF with `rasterio` to read spatial metadata (width, height, CRS, affine transform)
2. Reads the building GeoJSON with `geopandas`
3. Handles empty GeoJSON files gracefully (returns an all-zero mask)
4. Reprojects building geometries to the image CRS if they differ
5. Fixes invalid geometries using `shapely.validation.make_valid()`
6. Rasterizes building polygons onto the exact image grid using `rasterio.features.rasterize()`
7. Produces a binary mask: **0 = background, 1 = building**

---

## Mask Generation Statistics (Actual Run)

| Metric | Count |
|--------|-------|
| Satellite images found | 1,148 |
| GeoJSON files found | 1,148 |
| Matched pairs | 1,148 |
| Valid masks generated | 1,148 |
| Empty masks (no buildings) | 515 |
| Missing GeoJSON | 0 |
| Processing failures | 0 |
| Invalid geometries fixed | 3 |

> **Note:** 515 images have no building footprints (empty mask). These are valid samples (e.g., parks, water, forests) and are included in the dataset.

---

## Preprocessing Details

### Image Preprocessing
- **Read:** RGB-PanSharpened GeoTIFF (first 3 bands)
- **Resize:** 128 × 128 pixels (bilinear interpolation)
- **Dtype:** `uint16` (original raw pixel values preserved)
- **Normalization:** **NOT applied** — Model Lead should normalize during training (e.g., `X.astype('float32') / X.max()` or per-channel normalization)
- **Why uint16?** The source TIFFs are 16-bit with pixel values ranging 0–2000+. Casting to uint8 would destroy ~78% of pixel values that exceed 255.

### Mask Preprocessing
- **Resize:** 128 × 128 pixels (**nearest-neighbor** interpolation to preserve binary values)
- **Dtype:** `uint8`
- **Values:** Binary — 0 (background) or 1 (building)

### Image Dimensions
- Input images: 650 × 650 × 3 (original)
- Output images: 128 × 128 × 3 (resized)
- Output masks: 128 × 128 (resized)

---

## Train / Validation / Test Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 803 | 69.9% |
| Validation | 172 | 15.0% |
| Test | 173 | 15.1% |
| **Total** | **1,148** | **100%** |

- **Method:** `sklearn.model_selection.train_test_split`
- **Random seed:** `42`
- **Split ratios:** 70% / 15% / 15%
- Split is at the **image level** — each image stays paired with its mask

---

## Output .npy Files

| File | Shape | Dtype | Size |
|------|-------|-------|------|
| `X_train.npy` | (803, 128, 128, 3) | uint16 | 75.3 MB |
| `Y_train.npy` | (803, 128, 128) | uint8 | 12.5 MB |
| `X_val.npy` | (172, 128, 128, 3) | uint16 | 16.1 MB |
| `Y_val.npy` | (172, 128, 128) | uint8 | 2.7 MB |
| `X_test.npy` | (173, 128, 128, 3) | uint16 | 16.2 MB |
| `Y_test.npy` | (173, 128, 128) | uint8 | 2.7 MB |

---

## How to Load the Data (For Persons 2 & 3)

```python
import numpy as np

# Load training data
X_train = np.load("outputs/X_train.npy")  # (803, 128, 128, 3) uint16
Y_train = np.load("outputs/Y_train.npy")  # (803, 128, 128)    uint8

# Load validation data
X_val = np.load("outputs/X_val.npy")      # (172, 128, 128, 3) uint16
Y_val = np.load("outputs/Y_val.npy")      # (172, 128, 128)    uint8

# Load test data
X_test = np.load("outputs/X_test.npy")    # (173, 128, 128, 3) uint16
Y_test = np.load("outputs/Y_test.npy")    # (173, 128, 128)    uint8

# Normalize images for model training
# Option A: divide by global max (preserves relative intensities)
X_train = X_train.astype("float32") / 2000.0
# Option B: per-image normalization
# X_train = X_train.astype("float32") / X_train.max()

X_val   = X_val.astype("float32") / 2000.0
X_test  = X_test.astype("float32") / 2000.0

# Add channel dimension to masks if needed (for model input)
Y_train = Y_train[..., np.newaxis]  # (803, 128, 128, 1)
Y_val   = Y_val[..., np.newaxis]    # (172, 128, 128, 1)
Y_test  = Y_test[..., np.newaxis]   # (173, 128, 128, 1)
```

---

## Google Drive Handoff

The `.npy` output files (~126 MB total) are too large for Git. Share via Google Drive:

**Google Drive dataset outputs:**
> [ADD SHARED GOOGLE DRIVE LINK HERE]

Upload the entire `outputs/` folder (~126 MB total) to Google Drive and share the link with Persons 2 and 3.

---

## Rerunning the Preprocessing Pipeline

### Required Python Packages

```
numpy
opencv-python-headless
geopandas
rasterio
scikit-learn
matplotlib
```

### Setup & Run

```bash
# Create virtual environment (if not already present)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy opencv-python-headless geopandas rasterio scikit-learn matplotlib

# Run preprocessing
python preprocessing.py
```

The script will output all six `.npy` files to `outputs/` and print a complete statistics summary.

---

## Notes for Teammates

1. **Do NOT modify `archive/`** — it contains the original SpaceNet dataset
2. **Do NOT break the symlinks in `Train/`** — they point to `archive/`
3. **Images are uint16 (0–2000+)** — you MUST normalize before training (e.g., `/ 2000.0` or per-channel)
4. **Do NOT use `/ 255.0`** — that's for uint8 data; these are 16-bit satellite images
5. **Masks are binary uint8** — values are 0 (background) and 1 (building)
6. **515 out of 1,148 images have empty masks** — these are valid (no buildings in those tiles)
7. **3 invalid geometries** were auto-repaired during mask generation — no manual intervention needed
8. The sanity check image at `outputs/sanity_check.png` shows sample image/mask pairs
