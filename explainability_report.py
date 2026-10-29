#!/usr/bin/env python3
"""
Person 3: Explainability and urban-planning report for SpaceNet 2 Paris.

Inputs (produced by Person 1 and Person 2):
  outputs/X_test.npy
  model_results/models/best_model.keras

Outputs:
  model_results/explainability/gradcam_*.png
  model_results/explainability/occlusion_*.png
  model_results/explainability/summary.json
  model_results/explainability/urban_planning_report.md

Grad-CAM is applied to a segmentation model by differentiating the sum of
building probabilities with respect to a selected encoder feature map.
Occlusion sensitivity measures the drop in mean predicted building probability
when image patches are replaced with the image's channel-wise mean.

The optional Gemini report uses only statistics calculated by this script.
Without GEMINI_API_KEY (or the optional google-genai package), a deterministic
template report is generated instead. No API key is stored in source code.

Run from the repository root:
  python explainability_report.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "model_results" / "models" / "best_model.keras"
METRICS_PATH = ROOT / "model_results" / "segmentation_metrics.txt"
OUT_DIR = ROOT / "model_results" / "explainability"

NORM_DIVISOR = 2000.0
DEFAULT_SAMPLES = 3
PATCH_SIZE = 16
PATCH_STRIDE = 16
EXPLANATION_TOP_FRACTION = 0.20


def load_metrics(path):
    """Read the metrics file if it exists; do not invent missing metrics."""
    metrics = {}
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8")
    for key in ("IoU", "Dice", "Precision", "Recall", "F1 Score"):
        match = re.search(
            rf"^\s*{re.escape(key)}\s*:\s*([0-9]*\.?[0-9]+)",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if match:
            metrics[key.lower().replace(" ", "_")] = float(match.group(1))
    return metrics


def load_test_images():
    path = DATA_DIR / "X_test.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the preprocessed outputs/ folder from "
            "the team's shared Google Drive and place it in the repository."
        )
    images = np.load(path)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Expected X_test with shape (N,H,W,3); received {images.shape}."
        )
    # Match train_model.py exactly: uint16 -> float32, divide by 2000, clip.
    images = np.clip(images.astype(np.float32) / NORM_DIVISOR, 0.0, 1.0)
    return images


def choose_target_layer(model, requested=None):
    """Pick a connected convolutional feature layer for Grad-CAM."""
    if requested:
        try:
            return model.get_layer(requested)
        except ValueError as exc:
            raise ValueError(
                f"Target layer '{requested}' was not found in the model."
            ) from exc

    # EfficientNetB3's final encoder activation is a useful default when present.
    for name in ("top_activation", "block6a_expand_activation"):
        try:
            return model.get_layer(name)
        except ValueError:
            pass

    # Fallback for model variants: last 4-D layer with a single output tensor.
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer
        except (AttributeError, TypeError):
            continue
    raise ValueError("Could not find a 4-D feature layer for Grad-CAM.")


def make_gradcam(model, image, target_layer):
    """Return a normalized Grad-CAM heatmap resized to input image size."""
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )
    image_batch = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps, prediction = grad_model(image_batch, training=False)
        # Segmentation has one probability per pixel. Summing gives a scalar
        # target representing total predicted building confidence.
        target_score = tf.reduce_sum(prediction)

    gradients = tape.gradient(target_score, feature_maps)
    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable for the selected layer. "
            "Try --target-layer with another connected convolutional layer."
        )

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam.numpy()
    maximum = float(cam.max()) if cam.size else 0.0
    if maximum > 0:
        cam = cam / maximum

    height, width = image.shape[:2]
    cam = tf.image.resize(
        cam[..., np.newaxis], (height, width), method="bilinear"
    ).numpy()[..., 0]
    maximum = float(cam.max()) if cam.size else 0.0
    if maximum > 0:
        cam = cam / maximum
    return np.clip(cam, 0.0, 1.0)


def make_occlusion_map(model, image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    """
    Estimate patch importance by measuring the drop in mean building probability.
    A larger positive value means the occluded patch had more influence.
    """
    base_prediction = model.predict(image[np.newaxis, ...], verbose=0)[0, ..., 0]
    baseline = float(base_prediction.mean())
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    replacement = image.mean(axis=(0, 1), keepdims=True)

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)
            occluded = image.copy()
            occluded[top:bottom, left:right, :] = replacement
            prediction = model.predict(occluded[np.newaxis, ...], verbose=0)[0, ..., 0]
            drop = baseline - float(prediction.mean())
            heatmap[top:bottom, left:right] += drop
            counts[top:bottom, left:right] += 1.0

    heatmap = heatmap / np.maximum(counts, 1.0)
    # Display only positive influence; keep raw signed range in the summary.
    positive = np.maximum(heatmap, 0.0)
    maximum = float(positive.max()) if positive.size else 0.0
    if maximum > 0:
        positive = positive / maximum
    return np.clip(positive, 0.0, 1.0), baseline


def save_explanation_figure(image, heatmap, title, output_path):
    """Save image, heatmap, and overlay in one figure."""
    image_display = np.clip(image, 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_display)
    axes[0].set_title("Satellite image")
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Importance heatmap")
    axes[2].imshow(image_display)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    fig.suptitle(title)
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def explanation_mask_overlap(heatmap, predicted_mask):
    """
    Project-defined diagnostic, not a standardized published metric:
    fraction of top-20%-importance pixels that overlap predicted building pixels.
    It describes alignment with the model's own prediction, not correctness.
    """
    threshold = float(np.quantile(heatmap, 1.0 - EXPLANATION_TOP_FRACTION))
    important = heatmap >= threshold
    denominator = int(important.sum())
    if denominator == 0:
        return 0.0
    return float(np.logical_and(important, predicted_mask).sum() / denominator)


def make_template_report(summary):
    """Generate a cautious report grounded only in calculated values."""
    metrics = summary.get("test_metrics", {})
    lines = [
        "# Satellite Building Segmentation — Explainability Report",
        "",
        "## Scope",
        "",
        "This report summarizes model predictions and post-hoc explanation maps "
        "for selected SpaceNet 2 Paris test images. It is a technical screening "
        "aid, not a substitute for field checks or authoritative GIS analysis.",
        "",
        "## Model evaluation metrics",
        "",
    ]
    if metrics:
        for name, value in metrics.items():
            lines.append(f"- **{name.replace('_', ' ').title()}:** {value:.4f}")
    else:
        lines.append(
            "No saved evaluation metrics file was found. Run the model evaluation "
            "step before making performance claims."
        )

    lines.extend([
        "",
        "## Selected test-image observations",
        "",
        f"- Images explained: **{len(summary['samples'])}**",
        "- Grad-CAM highlights feature-map regions that influence the model's "
        "aggregate building probability.",
        "- Occlusion sensitivity highlights patches whose replacement changes "
        "the model's mean predicted building probability.",
        "- Explanation maps are model-behavior diagnostics; they do not prove "
        "that a highlighted region is a building.",
        "",
    ])

    if summary["samples"]:
        lines.extend([
            "| Test index | Predicted building coverage | Grad-CAM overlap diagnostic | "
            "Occlusion overlap diagnostic |",
            "|---:|---:|---:|---:|",
        ])
        for sample in summary["samples"]:
            lines.append(
                f"| {sample['test_index']} | "
                f"{sample['predicted_building_coverage']:.2%} | "
                f"{sample['gradcam_top20_overlap']:.2%} | "
                f"{sample['occlusion_top20_overlap']:.2%} |"
            )

    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "Building coverage is the fraction of pixels predicted as building in a "
        "selected tile. High or low coverage alone does not establish land-use "
        "quality, building density across a city, or planning risk. Validate "
        "important findings against ground-truth labels, georeferenced data, and "
        "local context. The overlap diagnostics are custom project diagnostics, "
        "not standardized explainability metrics and not accuracy measures.",
        "",
    ])
    return "\n".join(lines)


def maybe_generate_llm_report(summary, output_path):
    """Use Gemini only when explicitly configured; otherwise write a local report."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                "Write a cautious, concise technical report for an urban-planning "
                "research prototype using ONLY the JSON facts below. Do not invent "
                "numbers, locations, causes, or recommendations. Explain that "
                "Grad-CAM and occlusion maps describe model behavior, not proof of "
                "correctness. Distinguish predicted building pixel coverage from "
                "real-world building density. State that the custom overlap "
                "diagnostic is not a standard metric. Include limitations and "
                "validation steps. Return Markdown.\n\nDATA:\n"
                + json.dumps(summary, indent=2)
            )
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=prompt,
            )
            if response.text:
                output_path.write_text(response.text.strip() + "\n", encoding="utf-8")
                return "gemini"
        except Exception as exc:
            print(f"LLM report unavailable ({exc}); using local template.")

    output_path.write_text(make_template_report(summary) + "\n", encoding="utf-8")
    return "template"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run train_model.py first, or copy Person 2's best_model.keras into "
            "model_results/models/."
        )

    images = load_test_images()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    target_layer = choose_target_layer(model, os.getenv("GRADCAM_TARGET_LAYER"))
    metrics = load_metrics(METRICS_PATH)

    sample_count = min(DEFAULT_SAMPLES, len(images))
    summary = {
        "project": "SpaceNet 2 Paris building segmentation",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "gradcam_target_layer": target_layer.name,
        "test_images_available": int(len(images)),
        "samples_explained": sample_count,
        "test_metrics": metrics,
        "diagnostic_definition": (
            "top-20% explanation pixels overlapping predicted building pixels; "
            "custom diagnostic only, not a standardized metric or accuracy score"
        ),
        "samples": [],
    }

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Grad-CAM target layer: {target_layer.name}")
    print(f"Explaining {sample_count} test images...")

    for index in range(sample_count):
        image = images[index]
        prediction = model.predict(image[np.newaxis, ...], verbose=0)[0, ..., 0]
        predicted_mask = prediction >= 0.5
        gradcam = make_gradcam(model, image, target_layer)
        occlusion, baseline = make_occlusion_map(model, image)

        gradcam_path = OUT_DIR / f"gradcam_test_{index + 1}.png"
        occlusion_path = OUT_DIR / f"occlusion_test_{index + 1}.png"
        save_explanation_figure(
            image, gradcam, f"Grad-CAM — test image {index + 1}", gradcam_path
        )
        save_explanation_figure(
            image, occlusion, f"Occlusion sensitivity — test image {index + 1}",
            occlusion_path
        )

        summary["samples"].append({
            "test_index": index + 1,
            "predicted_building_coverage": float(predicted_mask.mean()),
            "mean_predicted_building_probability": float(prediction.mean()),
            "gradcam_top20_overlap": explanation_mask_overlap(gradcam, predicted_mask),
            "occlusion_top20_overlap": explanation_mask_overlap(occlusion, predicted_mask),
            "occlusion_baseline_mean_probability": baseline,
            "gradcam_figure": str(gradcam_path.relative_to(ROOT)),
            "occlusion_figure": str(occlusion_path.relative_to(ROOT)),
        })
        print(
            f"  Image {index + 1}: predicted coverage "
            f"{predicted_mask.mean():.2%}; saved explanation maps."
        )

    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = OUT_DIR / "urban_planning_report.md"
    report_source = maybe_generate_llm_report(summary, report_path)

    print("\nPerson 3 explainability run complete.")
    print(f"Summary JSON : {json_path}")
    print(f"Report       : {report_path} (source: {report_source})")
    print(f"Visual maps  : {OUT_DIR}")


if __name__ == "__main__":
    main()
