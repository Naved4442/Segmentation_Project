#!/usr/bin/env python3
"""
SpaceNet 2 Paris — Building Segmentation Preprocessing Pipeline
================================================================
Data Lead: Person 1
Model: EfficientNetB3 + Attention U-Net (downstream)

This script:
  1. Matches RGB-PanSharpen satellite images with building GeoJSON annotations
  2. Generates binary building masks via rasterization
  3. Resizes images to 128×128 and masks to 128×128 (nearest-neighbor)
  4. Splits into 70% train / 15% val / 15% test (random_state=42)
  5. Saves X_train, Y_train, X_val, Y_val, X_test, Y_test as .npy files
  6. Runs data-quality sanity checks

Usage:
    python preprocessing.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import cv2
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.validation import make_valid
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "Train"
RGB_DIR = BASE_DIR / "RGB-PanSharpen"
GEOJSON_DIR = BASE_DIR / "geojson" / "buildings"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMG_SIZE = (128, 128)  # (height, width)
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def verify_paths():
    """Verify that required data directories exist."""
    print("=" * 60)
    print("STEP 1: Verifying data paths")
    print("=" * 60)
    for name, path in [("RGB_DIR", RGB_DIR), ("GEOJSON_DIR", GEOJSON_DIR)]:
        exists = path.exists()
        print(f"  {name}: {path}")
        print(f"    exists={exists}, is_symlink={path.is_symlink()}")
        if not exists:
            sys.exit(f"ERROR: {name} does not exist: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  OUTPUT_DIR: {OUTPUT_DIR} (created/verified)")
    print()


def match_files():
    """Match satellite images with their corresponding GeoJSON files."""
    print("=" * 60)
    print("STEP 2: Matching images with GeoJSON files")
    print("=" * 60)

    rgb_files = sorted(RGB_DIR.glob("*.tif"))
    geo_files = sorted(GEOJSON_DIR.glob("*.geojson"))

    print(f"  Total satellite .tif images found: {len(rgb_files)}")
    print(f"  Total GeoJSON files found:         {len(geo_files)}")

    # Build lookup: image_id -> geojson_path
    geo_lookup = {}
    for gf in geo_files:
        # buildings_AOI_3_Paris_img10.geojson -> AOI_3_Paris_img10
        img_id = gf.stem.replace("buildings_", "", 1)
        geo_lookup[img_id] = gf

    matched = []
    missing_geojson = []

    for rf in rgb_files:
        # RGB-PanSharpen_AOI_3_Paris_img10.tif -> AOI_3_Paris_img10
        img_id = rf.stem.replace("RGB-PanSharpen_", "", 1)
        gf = geo_lookup.get(img_id)
        if gf is not None:
            matched.append((rf, gf, img_id))
        else:
            missing_geojson.append(rf.name)

    unmatched_geo = set(geo_lookup.keys()) - {m[2] for m in matched}

    print(f"  Matched image/GeoJSON pairs:       {len(matched)}")
    print(f"  Images missing GeoJSON:            {len(missing_geojson)}")
    print(f"  Unmatched GeoJSON (no image):      {len(unmatched_geo)}")

    if missing_geojson:
        print(f"    Examples: {missing_geojson[:5]}")
    if unmatched_geo:
        print(f"    Examples: {sorted(unmatched_geo)[:5]}")

    print()
    return matched, len(rgb_files), len(geo_files), len(missing_geojson)


def geojson_to_mask(image_path, geojson_path):
    """
    Generate a binary building mask from a GeoJSON file, aligned to the
    satellite image grid.

    Parameters
    ----------
    image_path : Path
        Path to the RGB-PanSharpen GeoTIFF.
    geojson_path : Path
        Path to the building footprint GeoJSON.

    Returns
    -------
    mask : np.ndarray of uint8, shape (height, width)
        Binary mask: 0 = background, 1 = building.
    n_invalid : int
        Number of invalid geometries that were repaired.
    """
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform

    gdf = gpd.read_file(geojson_path)

    # Handle empty GeoJSON
    if gdf.empty or len(gdf) == 0:
        return np.zeros((height, width), dtype=np.uint8), 0

    # Drop null geometries
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        return np.zeros((height, width), dtype=np.uint8), 0

    # Reproject to image CRS if both are defined and they differ
    if crs is None:
        raise ValueError(f"Image has no CRS: {image_path}")
    if gdf.crs is None:
        # Assume GeoJSON uses same CRS as image (common in SpaceNet)
        gdf = gdf.set_crs(crs)
    elif gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Fix invalid geometries
    n_invalid = 0
    valid_geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
            n_invalid += 1
        if not geom.is_empty:
            valid_geoms.append(geom)

    if not valid_geoms:
        return np.zeros((height, width), dtype=np.uint8), n_invalid

    # Rasterize building polygons
    shapes = [(geom, 1) for geom in valid_geoms]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    return mask, n_invalid


def generate_all_masks(matched_pairs):
    """Generate masks for the full dataset and collect statistics."""
    print("=" * 60)
    print("STEP 3: Generating binary masks for full dataset")
    print("=" * 60)

    valid_pairs = []      # (image_path, mask_array)
    empty_mask_count = 0
    failure_count = 0
    total_invalid_geoms = 0

    total = len(matched_pairs)
    t0 = time.time()

    for i, (img_path, geo_path, img_id) in enumerate(matched_pairs):
        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            print(f"  Processing {i + 1}/{total}  ({elapsed:.1f}s elapsed)")

        try:
            mask, n_inv = geojson_to_mask(img_path, geo_path)
            total_invalid_geoms += n_inv

            # Verify dimensions by reading image metadata
            with rasterio.open(img_path) as src:
                assert mask.shape == (src.height, src.width), (
                    f"Mask shape {mask.shape} != image shape "
                    f"({src.height}, {src.width})"
                )

            # Verify binary values
            unique_vals = np.unique(mask)
            assert all(v in (0, 1) for v in unique_vals), (
                f"Non-binary mask values: {unique_vals}"
            )

            is_empty = mask.max() == 0
            if is_empty:
                empty_mask_count += 1

            valid_pairs.append((img_path, mask))

        except Exception as e:
            failure_count += 1
            print(f"  FAILURE: {img_path.name} — {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    print(f"\n  Mask generation completed in {elapsed:.1f}s")
    print(f"  Valid masks generated:    {len(valid_pairs)}")
    print(f"  Empty masks (no bldgs):   {empty_mask_count}")
    print(f"  Processing failures:      {failure_count}")
    print(f"  Invalid geometries fixed: {total_invalid_geoms}")
    print()

    return valid_pairs, empty_mask_count, failure_count, total_invalid_geoms


def preprocess_images_and_masks(valid_pairs):
    """Resize images to 128×128 (RGB) and masks to 128×128 (nearest)."""
    print("=" * 60)
    print("STEP 4: Preprocessing — resize to 128×128")
    print("=" * 60)

    X_list = []
    Y_list = []
    total = len(valid_pairs)
    t0 = time.time()

    for i, (img_path, mask) in enumerate(valid_pairs):
        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            print(f"  Resizing {i + 1}/{total}  ({elapsed:.1f}s elapsed)")

        with rasterio.open(img_path) as src:
            n_bands = src.count
            if n_bands < 3:
                raise ValueError(
                    f"{img_path.name}: expected >= 3 bands, got {n_bands}"
                )
            img = src.read()  # shape: (bands, H, W)

        # Take first 3 bands as RGB, transpose to (H, W, 3)
        img = img[:3].transpose(1, 2, 0)  # (H, W, 3)

        # Resize image — bilinear
        img_resized = cv2.resize(
            img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR
        )

        # Resize mask — nearest neighbor to preserve binary values
        mask_resized = cv2.resize(
            mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST
        )

        # Ensure mask stays binary
        mask_resized = (mask_resized > 0).astype(np.uint8)

        X_list.append(img_resized)
        Y_list.append(mask_resized)

    # Source TIFFs are uint16 (values 0–2000+). Preserve native dtype to
    # avoid lossy truncation. Model Lead normalizes during training.
    X = np.array(X_list, dtype=np.uint16)
    Y = np.array(Y_list, dtype=np.uint8)

    elapsed = time.time() - t0
    print(f"\n  Resizing completed in {elapsed:.1f}s")
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  Y shape: {Y.shape}, dtype: {Y.dtype}")
    print(f"  Image normalization: NOT applied (raw uint16 values)")
    print(f"    → Model Lead should normalize, e.g. X.astype('float32') / X.max()")
    print()

    return X, Y


def split_dataset(X, Y):
    """Split into 70% train, 15% val, 15% test with fixed random seed."""
    print("=" * 60)
    print("STEP 5: Train / Validation / Test split (70/15/15)")
    print("=" * 60)

    # First split: 70% train, 30% temp
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
    )

    # Second split: 50% of temp = 15% val, 50% of temp = 15% test
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=RANDOM_SEED
    )

    total = len(X)
    for name, arr in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        pct = len(arr) / total * 100
        print(f"  {name:5s}: {len(arr):5d} samples ({pct:.1f}%)")

    print()
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def save_arrays(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """Save all six .npy files to the outputs/ directory."""
    print("=" * 60)
    print("STEP 6: Saving .npy files")
    print("=" * 60)

    arrays = {
        "X_train": X_train, "Y_train": Y_train,
        "X_val": X_val, "Y_val": Y_val,
        "X_test": X_test, "Y_test": Y_test,
    }

    for name, arr in arrays.items():
        path = OUTPUT_DIR / f"{name}.npy"
        np.save(path, arr)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  Saved {name}.npy — shape={arr.shape}, "
              f"dtype={arr.dtype}, size={size_mb:.1f} MB")

    print()


def verify_saved_files():
    """Reload and verify all six .npy files."""
    print("=" * 60)
    print("STEP 7: Verification — reload and validate .npy files")
    print("=" * 60)

    files = ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"]
    results = {}

    for name in files:
        path = OUTPUT_DIR / f"{name}.npy"
        assert path.exists(), f"Missing: {path}"
        arr = np.load(path)
        results[name] = arr

        # Check for NaN/Inf
        has_nan = np.isnan(arr.astype(float)).any()
        has_inf = np.isinf(arr.astype(float)).any()

        print(f"  {name:8s}: shape={str(arr.shape):22s} dtype={arr.dtype}"
              f"  NaN={has_nan}  Inf={has_inf}")

    # Cross-checks
    print("\n  Cross-validation:")
    for split in ["train", "val", "test"]:
        x = results[f"X_{split}"]
        y = results[f"Y_{split}"]
        assert x.shape[0] == y.shape[0], f"X/Y count mismatch in {split}"
        assert x.shape[1:3] == (128, 128), f"X image size wrong in {split}"
        assert y.shape[1:3] == (128, 128), f"Y mask size wrong in {split}"
        assert len(x) > 0, f"Empty {split} set"

        # Verify masks are binary
        y_unique = np.unique(y)
        assert set(y_unique).issubset({0, 1}), (
            f"{split} masks contain non-binary values: {y_unique}"
        )

        print(f"    {split:5s}: X={x.shape}, Y={y.shape}, "
              f"masks binary={set(y_unique).issubset({0, 1})}")

    print("\n  ✓ All verifications passed!")
    print()
    return results


def generate_visual_sanity_check(results):
    """Save a visual grid of sample images with their masks."""
    print("=" * 60)
    print("STEP 8: Visual sanity check")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_train = results["X_train"]
    Y_train = results["Y_train"]

    n_samples = min(5, len(X_train))
    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    for i in range(n_samples):
        # Image — the TIF stores 16-bit, rescale for display
        img = X_train[i].astype(np.float32)
        # Simple percentile stretch for display
        for c in range(3):
            lo, hi = np.percentile(img[:, :, c], [2, 98])
            if hi > lo:
                img[:, :, c] = np.clip((img[:, :, c] - lo) / (hi - lo), 0, 1)
            else:
                img[:, :, c] = 0

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(Y_train[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Mask {i}", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Sanity Check: Train Images and Building Masks", fontsize=12)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "sanity_check.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved visual sanity check to: {out_path}")
    print()


def print_final_summary(
    n_images, n_geojsons, n_matched, n_missing_geo,
    n_valid, n_empty, n_failures, n_invalid_geoms,
    results
):
    """Print the final pipeline summary."""
    print("=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    print(f"  Images found:            {n_images}")
    print(f"  GeoJSON files found:     {n_geojsons}")
    print(f"  Matched pairs:           {n_matched}")
    print(f"  Valid masks generated:   {n_valid}")
    print(f"  Empty masks (no bldgs):  {n_empty}")
    print(f"  Missing GeoJSON:         {n_missing_geo}")
    print(f"  Processing failures:     {n_failures}")
    print(f"  Invalid geoms fixed:     {n_invalid_geoms}")
    print()
    print(f"  Image size:   {IMG_SIZE[0]}×{IMG_SIZE[1]}×3")
    print(f"  Mask size:    {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"  Image dtype:  uint16 (raw values, no normalization)")
    print(f"  Mask dtype:   uint8 (binary: 0 or 1)")
    print(f"  Random seed:  {RANDOM_SEED}")
    print(f"  Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print()

    print("  Split counts:")
    for split in ["train", "val", "test"]:
        x = results[f"X_{split}"]
        total = sum(len(results[f"X_{s}"]) for s in ["train", "val", "test"])
        pct = len(x) / total * 100
        print(f"    {split:5s}: {len(x):5d} ({pct:.1f}%)")
    print()

    print("  Output files:")
    for name in ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"]:
        arr = results[name]
        path = OUTPUT_DIR / f"{name}.npy"
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"    {name}.npy — {arr.shape} — {size_mb:.1f} MB")

    print()
    print("  Pipeline status: SUCCESS ✓")
    print("=" * 60)


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def main():
    t_start = time.time()

    # Step 1: Verify paths
    verify_paths()

    # Step 2-3: Match files
    matched_pairs, n_images, n_geojsons, n_missing_geo = match_files()
    n_matched = len(matched_pairs)

    # Step 4: Generate masks
    valid_pairs, n_empty, n_failures, n_invalid_geoms = generate_all_masks(
        matched_pairs
    )
    n_valid = len(valid_pairs)

    # Step 5: Preprocess (resize)
    X, Y = preprocess_images_and_masks(valid_pairs)

    # Step 6: Split
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    # Step 7: Save
    save_arrays(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    # Step 8: Verify
    results = verify_saved_files()

    # Step 9: Visual sanity check
    generate_visual_sanity_check(results)

    # Final summary
    print_final_summary(
        n_images, n_geojsons, n_matched, n_missing_geo,
        n_valid, n_empty, n_failures, n_invalid_geoms,
        results,
    )

    elapsed = time.time() - t_start
    print(f"  Total pipeline time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
