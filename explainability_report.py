#!/usr/bin/env python3
"""Person 3: Grad-CAM, occlusion sensitivity, and report for SpaceNet 2 Paris."""
import builtins
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

# Saved Lambda functions refer to the global name `tf`. Keras may deserialize
# them without that name in their function globals, so expose it for this
# Python process. This does not modify the system or the saved model.
builtins.tf = tf

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


def load_compatible_model(model_path):
    """Patch missing Lambda output shapes in a temporary .keras archive."""
    with tempfile.TemporaryDirectory(prefix="spacenet_model_") as temp_dir:
        patched_path = Path(temp_dir) / "model_patched.keras"
        with zipfile.ZipFile(model_path, "r") as original:
            config = json.loads(original.read("config.json"))
            count = 0
            for layer in config["config"]["layers"]:
                if layer.get("class_name") != "Lambda":
                    continue
                nodes = layer.get("inbound_nodes", [])
                if not nodes:
                    continue
                try:
                    shape = nodes[0]["args"][0]["config"]["shape"]
                except (KeyError, IndexError, TypeError):
                    continue
                if len(shape) != 4:
                    continue
                layer["config"]["output_shape"] = [shape[1], shape[2], 1]
                count += 1
            with zipfile.ZipFile(
                patched_path, "w", zipfile.ZIP_DEFLATED
            ) as patched:
                for name in original.namelist():
                    if name == "config.json":
                        patched.writestr(name, json.dumps(config).encode("utf-8"))
                    else:
                        patched.writestr(name, original.read(name))
        print(f"Prepared temporary model ({count} Lambda layers).")
        model = tf.keras.models.load_model(
            patched_path,
            custom_objects={
                "tf": tf,
                "reduce_mean": tf.reduce_mean,
                "reduce_max": tf.reduce_max,
            },
            compile=False,
            safe_mode=False,
        )
    print("Model loaded successfully.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print(f"Parameters: {model.count_params():,}")
    return model


def load_metrics(path):
    metrics = {}
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8")
    for key in ("IoU", "Dice", "Precision", "Recall", "F1 Score"):
        match = re.search(
            rf"^\s*{re.escape(key)}\s*:\s*([0-9]*\.?[0-9]+)",
            text, flags=re.MULTILINE | re.IGNORECASE
        )
        if match:
            metrics[key.lower().replace(" ", "_")] = float(match.group(1))
    return metrics


def load_test_images():
    path = DATA_DIR / "X_test.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing test data: {path}")
    images = np.load(path)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected X_test shape (N,H,W,3); got {images.shape}.")
    return np.clip(images.astype(np.float32) / NORM_DIVISOR, 0.0, 1.0)


def choose_target_layer(model, requested=None):
    if requested:
        try:
            return model.get_layer(requested)
        except ValueError as exc:
            raise ValueError(f"Target layer '{requested}' was not found.") from exc
    for name in ("top_activation", "block6a_expand_activation"):
        try:
            return model.get_layer(name)
        except ValueError:
            pass
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except (AttributeError, TypeError):
            pass
    raise ValueError("Could not find a 4-D feature layer for Grad-CAM.")


def make_gradcam(model, image, target_layer):
    grad_model = tf.keras.Model(
        inputs=model.inputs, outputs=[target_layer.output, model.output]
    )
    batch = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        features, prediction = grad_model(batch, training=False)
        score = tf.reduce_sum(prediction)
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients unavailable for this target layer.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1)[0]).numpy()
    maximum = float(cam.max()) if cam.size else 0.0
    if maximum > 0:
        cam /= maximum
    cam = tf.image.resize(
        cam[..., np.newaxis], image.shape[:2], method="bilinear"
    ).numpy()[..., 0]
    maximum = float(cam.max()) if cam.size else 0.0
    if maximum > 0:
        cam /= maximum
    return np.clip(cam, 0.0, 1.0)


def make_occlusion_map(model, image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    base = model.predict(image[np.newaxis, ...], verbose=0)[0, ..., 0]
    baseline = float(base.mean())
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    replacement = image.mean(axis=(0, 1), keepdims=True)
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom, right = min(top + patch_size, height), min(left + patch_size, width)
            occluded = image.copy()
            occluded[top:bottom, left:right, :] = replacement
            pred = model.predict(occluded[np.newaxis, ...], verbose=0)[0, ..., 0]
            drop = baseline - float(pred.mean())
            heatmap[top:bottom, left:right] += drop
            counts[top:bottom, left:right] += 1.0
    heatmap /= np.maximum(counts, 1.0)
    positive = np.maximum(heatmap, 0.0)
    maximum = float(positive.max()) if positive.size else 0.0
    if maximum > 0:
        positive /= maximum
    return np.clip(positive, 0.0, 1.0), baseline


def save_explanation_figure(image, heatmap, title, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("Satellite image")
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Importance heatmap")
    axes[2].imshow(np.clip(image, 0.0, 1.0))
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def explanation_mask_overlap(heatmap, predicted_mask):
    """Custom diagnostic, not a standardized metric or accuracy score."""
    threshold = float(np.quantile(heatmap, 1.0 - EXPLANATION_TOP_FRACTION))
    important = heatmap >= threshold
    denominator = int(important.sum())
    if denominator == 0:
        return 0.0
    return float(np.logical_and(important, predicted_mask).sum() / denominator)


def make_template_report(summary):
    metrics = summary.get("test_metrics", {})
    lines = [
        "# Satellite Building Segmentation — Explainability Report", "",
        "## Scope", "",
        "This report summarizes model predictions and post-hoc explanation "
        "maps for selected SpaceNet 2 Paris test images. It is a technical "
        "screening aid, not a substitute for field checks or authoritative GIS analysis.",
        "", "## Model evaluation metrics", ""
    ]
    if metrics:
        for name, value in metrics.items():
            lines.append(f"- **{name.replace('_', ' ').title()}:** {value:.4f}")
    else:
        lines.append("No saved evaluation metrics were found.")
    lines.extend([
        "", "## Selected test-image observations", "",
        f"- Images explained: **{len(summary['samples'])}**",
        "- Grad-CAM highlights feature-map regions influencing aggregate predicted building probability.",
        "- Occlusion sensitivity measures how patch replacement changes mean predicted building probability.",
        "- Explanation maps describe model behavior; they do not prove a highlighted region is a building.", ""
    ])
    if summary["samples"]:
        lines.extend([
            "| Test index | Predicted building coverage | Grad-CAM overlap diagnostic | Occlusion overlap diagnostic |",
            "|---:|---:|---:|---:|"
        ])
        for s in summary["samples"]:
            lines.append(
                f"| {s['test_index']} | {s['predicted_building_coverage']:.2%} | "
                f"{s['gradcam_top20_overlap']:.2%} | {s['occlusion_top20_overlap']:.2%} |"
            )
    lines.extend([
        "", "## Interpretation and limitations", "",
        "Building coverage is the fraction of pixels predicted as buildings in a selected tile. "
        "Coverage alone does not establish real-world building density, land-use quality, or "
        "planning risk. Validate findings against ground-truth labels, georeferenced data, "
        "and local context. The overlap diagnostics are custom project diagnostics, not "
        "standardized explainability metrics or accuracy measures.", ""
    ])
    return "\n".join(lines)


def maybe_generate_llm_report(summary, output_path):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                "Write a cautious technical report using ONLY the JSON facts below. Do not invent "
                "numbers, locations, causes, or recommendations. Explain that Grad-CAM and "
                "occlusion maps describe model behavior, not proof of correctness. Distinguish "
                "predicted pixel coverage from real-world building density. State that the custom "
                "overlap diagnostic is not a standard metric. Include limitations and validation "
                "steps. Return Markdown.\n\nDATA:\n" + json.dumps(summary, indent=2)
            )
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt
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
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    images = load_test_images()
    if len(images) == 0:
        raise ValueError("X_test.npy contains no test images.")
    model = load_compatible_model(MODEL_PATH)
    target_layer = choose_target_layer(model, os.getenv("GRADCAM_TARGET_LAYER"))
    summary = {
        "project": "SpaceNet 2 Paris building segmentation",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "gradcam_target_layer": target_layer.name,
        "test_images_available": int(len(images)),
        "samples_explained": min(DEFAULT_SAMPLES, len(images)),
        "test_metrics": load_metrics(METRICS_PATH),
        "diagnostic_definition": (
            "Top 20% explanation pixels overlapping predicted building pixels; "
            "custom diagnostic only, not a standardized metric or accuracy score."
        ),
        "samples": [],
    }
    sample_count = min(DEFAULT_SAMPLES, len(images))
    print(f"Grad-CAM target layer: {target_layer.name}")
    print(f"Explaining {sample_count} test images...")
    for i in range(sample_count):
        image = images[i]
        prediction = model.predict(image[np.newaxis, ...], verbose=0)[0, ..., 0]
        predicted_mask = prediction >= 0.5
        gradcam = make_gradcam(model, image, target_layer)
        occlusion, baseline = make_occlusion_map(model, image)
        grad_path = OUT_DIR / f"gradcam_test_{i + 1}.png"
        occ_path = OUT_DIR / f"occlusion_test_{i + 1}.png"
        save_explanation_figure(image, gradcam, f"Grad-CAM — test image {i + 1}", grad_path)
        save_explanation_figure(image, occlusion, f"Occlusion sensitivity — test image {i + 1}", occ_path)
        summary["samples"].append({
            "test_index": i + 1,
            "predicted_building_coverage": float(predicted_mask.mean()),
            "mean_predicted_building_probability": float(prediction.mean()),
            "gradcam_top20_overlap": explanation_mask_overlap(gradcam, predicted_mask),
            "occlusion_top20_overlap": explanation_mask_overlap(occlusion, predicted_mask),
            "occlusion_baseline_mean_probability": baseline,
            "gradcam_figure": str(grad_path.relative_to(ROOT)),
            "occlusion_figure": str(occ_path.relative_to(ROOT)),
        })
        print(f"Image {i + 1}/{sample_count}: predicted building coverage {predicted_mask.mean():.2%}; maps saved.")
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = OUT_DIR / "urban_planning_report.md"
    source = maybe_generate_llm_report(summary, report_path)
    print("\nPerson 3 explainability run complete.")
    print(f"Summary JSON : {json_path}")
    print(f"Report       : {report_path} (source: {source})")
    print(f"Visual maps  : {OUT_DIR}")


if __name__ == "__main__":
    main()
