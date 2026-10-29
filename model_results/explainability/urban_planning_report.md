# Satellite Building Segmentation — Explainability Report

## Scope

This report summarizes model predictions and post-hoc explanation maps for selected SpaceNet 2 Paris test images. It is a technical screening aid, not a substitute for field checks or authoritative GIS analysis.

## Model evaluation metrics

- **Iou:** 0.1743
- **Dice:** 0.2968
- **Precision:** 0.2732
- **Recall:** 0.3249
- **F1 Score:** 0.2968

## Selected test-image observations

- Images explained: **3**
- Grad-CAM highlights feature-map regions influencing aggregate predicted building probability.
- Occlusion sensitivity measures how patch replacement changes mean predicted building probability.
- Explanation maps describe model behavior; they do not prove a highlighted region is a building.

| Test index | Predicted building coverage | Grad-CAM overlap diagnostic | Occlusion overlap diagnostic |
|---:|---:|---:|---:|
| 1 | 0.00% | 0.00% | 0.00% |
| 2 | 0.00% | 0.00% | 0.00% |
| 3 | 34.67% | 34.23% | 56.97% |

## Interpretation and limitations

Building coverage is the fraction of pixels predicted as buildings in a selected tile. Coverage alone does not establish real-world building density, land-use quality, or planning risk. Validate findings against ground-truth labels, georeferenced data, and local context. The overlap diagnostics are custom project diagnostics, not standardized explainability metrics or accuracy measures.
