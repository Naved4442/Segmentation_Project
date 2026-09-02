#!/usr/bin/env python3
"""
SpaceNet 2 Paris — Building Segmentation Model Training
=========================================================
Model Lead: Person 2
Architecture: EfficientNetB3 encoder + Attention U-Net decoder
              (channel + spatial attention blocks, attention-gated skips)

Expects the six .npy files produced by Person 1's preprocessing.py:
  X_train.npy  (N, 128, 128, 3)  uint16  — raw pixel values, NOT normalized
  Y_train.npy  (N, 128, 128)     uint8   — binary mask, 0/1, no channel dim
  X_val.npy / Y_val.npy, X_test.npy / Y_test.npy — same format

This script:
  1. Loads the .npy files and normalizes/reshapes them correctly
  2. Builds the Urban-Attention EfficientNetB3-U-Net
  3. Trains with BCE+Dice loss, checkpointing, early stopping, LR reduction
  4. Evaluates on the test set: IoU, Dice, Precision, Recall, F1
  5. Saves the best model, metrics report, and sample prediction visuals

Usage:
    python train_model.py
"""

import os
import time
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, UpSampling2D,
    Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense,
    Add, Multiply, Reshape, Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ──────────────────────────────────────────────
# CONFIGURATION — adjust DATA_DIR to wherever you
# downloaded Person 1's outputs/ folder from Drive
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "outputs"          # the six .npy files live here
RESULTS_DIR = PROJECT_ROOT / "model_results"
MODEL_DIR = RESULTS_DIR / "models"
VIS_DIR = RESULTS_DIR / "visual_outputs"

for d in [RESULTS_DIR, MODEL_DIR, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16          # 128x128 + EfficientNetB3 is memory-heavy; drop to 8 if you OOM
EPOCHS = 100
LEARNING_RATE = 1e-4
NORM_DIVISOR = 2000.0     # per README: raw uint16 values are 0-2000+

# Set to "imagenet" if you have internet access on the training machine,
# otherwise leave as None (README's default assumption for HPC/offline runs)
ENCODER_WEIGHTS = None


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────
def load_split(name):
    X = np.load(DATA_DIR / f"X_{name}.npy")   # (N, 128, 128, 3) uint16
    Y = np.load(DATA_DIR / f"Y_{name}.npy")   # (N, 128, 128)    uint8

    # Normalize images: uint16 raw values -> float32 [0, ~1]
    X = X.astype(np.float32) / NORM_DIVISOR
    X = np.clip(X, 0.0, 1.0)

    # Add channel dim to masks: (N, 128, 128) -> (N, 128, 128, 1)
    Y = Y.astype(np.float32)[..., np.newaxis]

    return X, Y


def load_all_data():
    print("=" * 60)
    print("STEP 1: Loading preprocessed data")
    print("=" * 60)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found. Download the outputs/ folder from the "
            f"team's Google Drive (see README.md) and place it next to this script."
        )

    X_train, Y_train = load_split("train")
    X_val, Y_val = load_split("val")
    X_test, Y_test = load_split("test")

    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_val  : {X_val.shape}  Y_val  : {Y_val.shape}")
    print(f"  X_test : {X_test.shape}  Y_test : {Y_test.shape}")
    print(f"  X value range after normalization: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Y unique values: {np.unique(Y_train)}")
    print()

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ──────────────────────────────────────────────
# 2. LOSSES AND METRICS
# ──────────────────────────────────────────────
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return 1.0 - dice


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def binary_iou_np(y_true, y_pred):
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return inter / (union + 1e-8)


def dice_score_np(y_true, y_pred):
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    inter = np.sum(y_true * y_pred)
    return (2.0 * inter) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)


# ──────────────────────────────────────────────
# 3. URBAN ATTENTION MODULES
# ──────────────────────────────────────────────
def channel_attention(x, ratio=8):
    channels = int(x.shape[-1])

    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)

    dense1 = Dense(max(channels // ratio, 1), activation="relu")
    dense2 = Dense(channels)

    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))

    out = Add()([avg_out, max_out])
    out = Activation("sigmoid")(out)
    out = Reshape((1, 1, channels))(out)

    return Multiply()([x, out])


def spatial_attention(x):
    avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    att = Conv2D(1, 7, padding="same", activation="sigmoid")(concat)

    return Multiply()([x, att])


def urban_attention_block(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x


def attention_gate(skip, gating, filters):
    theta = Conv2D(filters, 1, padding="same")(skip)
    phi = Conv2D(filters, 1, padding="same")(gating)

    add = Add()([theta, phi])
    act = Activation("relu")(add)
    psi = Conv2D(1, 1, padding="same", activation="sigmoid")(act)

    return Multiply()([skip, psi])


def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = urban_attention_block(x)

    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = urban_attention_block(x)

    return x


# ──────────────────────────────────────────────
# 4. MODEL — EfficientNetB3 encoder + Attention U-Net decoder
# ──────────────────────────────────────────────
def build_model(img_size=IMG_SIZE, encoder_weights=ENCODER_WEIGHTS):
    inputs = Input((img_size, img_size, 3))

    encoder = EfficientNetB3(
        include_top=False,
        weights=encoder_weights,
        input_tensor=inputs,
    )

    s1 = encoder.get_layer("block2a_expand_activation").output
    s2 = encoder.get_layer("block3a_expand_activation").output
    s3 = encoder.get_layer("block4a_expand_activation").output
    s4 = encoder.get_layer("block6a_expand_activation").output

    bridge = encoder.output
    bridge = urban_attention_block(bridge)

    d1 = UpSampling2D()(bridge)
    s4 = attention_gate(s4, d1, 256)
    d1 = Concatenate()([d1, s4])
    d1 = conv_block(d1, 256)

    d2 = UpSampling2D()(d1)
    s3 = attention_gate(s3, d2, 128)
    d2 = Concatenate()([d2, s3])
    d2 = conv_block(d2, 128)

    d3 = UpSampling2D()(d2)
    s2 = attention_gate(s2, d3, 64)
    d3 = Concatenate()([d3, s2])
    d3 = conv_block(d3, 64)

    d4 = UpSampling2D()(d3)
    s1 = attention_gate(s1, d4, 32)
    d4 = Concatenate()([d4, s1])
    d4 = conv_block(d4, 32)

    d5 = UpSampling2D()(d4)
    d5 = conv_block(d5, 16)

    outputs = Conv2D(1, 1, activation="sigmoid", name="building_mask")(d5)

    return Model(inputs, outputs, name="UrbanAttention_EfficientNetB3_UNet")


# ──────────────────────────────────────────────
# 5. TRAINING
# ──────────────────────────────────────────────
def train(model, X_train, Y_train, X_val, Y_val):
    print("=" * 60)
    print("STEP 3: Training")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint = ModelCheckpoint(
        str(MODEL_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    earlystop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    )

    t0 = time.time()
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, earlystop, reduce_lr],
    )
    print(f"\n  Training completed in {time.time() - t0:.1f}s")
    print()

    # Save training curves
    save_training_curves(history)

    return history


def save_training_curves(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (BCE + Dice)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=120)
    plt.close()
    print(f"  Saved training curves to {RESULTS_DIR / 'training_curves.png'}")


# ──────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────
def evaluate(model, X_test, Y_test):
    print("=" * 60)
    print("STEP 4: Evaluation on held-out test set")
    print("=" * 60)

    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    preds_bin = (preds > 0.5).astype(np.uint8)

    iou = binary_iou_np(Y_test, preds_bin)
    dice = dice_score_np(Y_test, preds_bin)

    y_true_flat = Y_test.flatten().astype(np.uint8)
    y_pred_flat = preds_bin.flatten().astype(np.uint8)

    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

    metrics_text = f"""
Segmentation Evaluation Results (Test Set, n={len(X_test)})
=============================================================
IoU        : {iou:.4f}
Dice       : {dice:.4f}
Precision  : {precision:.4f}
Recall     : {recall:.4f}
F1 Score   : {f1:.4f}
"""
    print(metrics_text)

    with open(RESULTS_DIR / "segmentation_metrics.txt", "w") as f:
        f.write(metrics_text)

    save_prediction_visuals(X_test, Y_test, preds_bin)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "f1": f1}, preds_bin


def save_prediction_visuals(X_test, Y_test, preds_bin, n=10):
    for i in range(min(n, len(X_test))):
        img = X_test[i]
        gt = Y_test[i].squeeze()
        pred = preds_bin[i].squeeze()

        img_uint8 = (img * 255).astype(np.uint8)

        overlay = img_uint8.copy()
        overlay[pred > 0] = [255, 0, 0]

        cv2.imwrite(str(VIS_DIR / f"test_original_{i+1}.png"),
                    cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(VIS_DIR / f"test_groundtruth_{i+1}.png"), gt * 255)
        cv2.imwrite(str(VIS_DIR / f"test_prediction_{i+1}.png"), pred * 255)
        cv2.imwrite(str(VIS_DIR / f"test_overlay_{i+1}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"  Saved {min(n, len(X_test))} prediction visualizations to {VIS_DIR}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_all_data()

    print("=" * 60)
    print("STEP 2: Building model")
    print("=" * 60)
    model = build_model()
    print(f"  Total params: {model.count_params():,}")
    print()

    train(model, X_train, Y_train, X_val, Y_val)
    metrics, preds_bin = evaluate(model, X_test, Y_test)

    print("=" * 60)
    print("DONE — hand off to Person 3 (XAI/LLM Lead)")
    print("=" * 60)
    print(f"  Trained model : {MODEL_DIR / 'best_model.keras'}")
    print(f"  Metrics       : {RESULTS_DIR / 'segmentation_metrics.txt'}")
    print(f"  Visuals       : {VIS_DIR}")
    print(f"  IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
