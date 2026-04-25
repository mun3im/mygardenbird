#!/usr/bin/env python3
"""
Stage 9b: Find Most Confused Samples for Dataset Balancing

This script identifies the most confused samples in classes with > 600 samples
to enable dataset balancing to exactly 600 samples per class.

Strategy:
1. Count samples per class
   - Classes with < 600: Exclude from training/evaluation
   - Classes with = 600: Include in training (already balanced)
   - Classes with > 600: Include in training (identify samples to remove)

2. 5-Fold Cross-Validation with Source-Based MIP Splitting
   - Create folds across ALL eligible classes (>= 600 samples)
   - Each fold: ~20% of data per class (source-based to prevent leakage)
   - Train on 4 folds, validate on 1 fold
   - Every sample evaluated exactly once

3. Multi-Class Classification
   - Train across all eligible classes to capture inter-class confusion
   - Track predictions: filename, source, true_label, pred_label, confidences
   - Calculate confusion score per sample
   - Accumulate confusion matrix across all 5 folds

4. Output
   - CSV per class with > 600: Samples sorted by confusion score (most confused first)
   - Composite confusion matrix heatmap (all 5 folds aggregated)

Model: MobileNetV3Small (fastest)
Epochs: 25 per fold (sufficient for confusion detection)
Sample Rate: 16 kHz
Feature: Mel spectrogram
Augmentation: None (not needed for confusion detection)

Usage:
    python Stage9b_find_most_confused.py --dataset_root /path/to/mygardenbird16khz
"""

import os
import json
import csv
import platform
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime
import argparse
import time

from tqdm import tqdm

# Startup progress bar
pbar = tqdm(
    total=100,
    desc="Starting up...",
    bar_format='{desc} {bar}',
    colour='red',
    dynamic_ncols=True
)
for i in range(100):
    time.sleep(0.01)
    pbar.update(1)
pbar.close()

import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import MYGARDENBIRD_16K, METADATA_16K

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, losses

# Check TensorFlow version
if not tf.__version__.startswith("2.15."):
    print("⚠️  Warning: This script was tested with TensorFlow 2.15.x")
    print(f"   Current version: {tf.__version__}")

tf.keras.backend.set_image_data_format('channels_last')

# Import MIP optimization from Stage8a
try:
    from pulp import (
        LpProblem, LpMinimize, LpVariable, lpSum,
        PULP_CBC_CMD, LpBinary, value
    )
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("\n❌ ERROR: PuLP not installed (required for MIP splitting)")
    print("Install with: pip install pulp\n")
    exit(1)

# Global variables
SAMPLE_RATE = 16000
TARGET_LENGTH_SAMPLES = int(SAMPLE_RATE * 3.0)
HOP_LENGTH = TARGET_LENGTH_SAMPLES // 224
N_FFT = 2048
N_MELS = 224
TARGET_SPEC_WIDTH = 224
TARGET_SPEC_HEIGHT = 224
PREPROCESS_FN = None

# Target sample count per class
TARGET_SAMPLES_PER_CLASS = 600

# 5-Fold CV configuration
NUM_FOLDS = 5

# Training configuration
MAX_EPOCHS_PER_FOLD = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ============================================================================
# Dataset Analysis
# ============================================================================

def count_samples_per_class(dataset_root: Path) -> Dict[str, int]:
    """Count .wav files in each class directory."""
    counts = {}
    for class_dir in dataset_root.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        wav_files = list(class_dir.glob('*.wav'))
        counts[class_dir.name] = len(wav_files)
    return counts


def load_dataset_structure(dataset_root: Path, eligible_classes: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Load dataset structure for eligible classes: class -> source -> [files].

    Args:
        dataset_root: Path to dataset directory
        eligible_classes: List of class names to include

    Returns:
        Nested dict: class_name -> source_id -> [filenames]
    """
    structure = {}

    for class_name in eligible_classes:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue

        structure[class_name] = defaultdict(list)

        for audio_file in class_dir.glob('*.wav'):
            # Extract source ID: xc402013_27465.wav -> xc402013
            source = audio_file.stem.split('_')[0]
            structure[class_name][source].append(audio_file.name)

    return structure


# ============================================================================
# MIP-Based 5-Fold Splitting (Multi-Class)
# ============================================================================

def create_multiclass_5fold_mip_splits(structure: Dict[str, Dict[str, List[str]]],
                                       verbose: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Use MIP to create 5 balanced folds (~20% each per class) with source-based separation.

    Args:
        structure: Dataset structure (class -> source -> files)
        verbose: Print progress messages

    Returns:
        assignment: Dict[class][source] = 'fold0' | 'fold1' | ... | 'fold4'
    """
    if verbose:
        print(f"\nCreating 5-fold MIP splits across {len(structure)} classes...")

    # Create MIP problem
    prob = LpProblem("MultiClass_5Fold_Split", LpMinimize)

    # Decision variables: x[class][source][fold] = 1 if source assigned to fold
    x = {}
    for class_name, sources in structure.items():
        x[class_name] = {}
        # Sanitize class name and source for PuLP variable names (alphanumeric + underscore only)
        safe_cls = class_name.replace(' ', '_').replace('-', '_').replace('.', '_')
        for source in sources.keys():
            safe_src = source.replace(' ', '_').replace('-', '_').replace('.', '_')
            x[class_name][source] = {
                f'fold{i}': LpVariable(f"x_{safe_cls}_{safe_src}_fold{i}", cat=LpBinary)
                for i in range(NUM_FOLDS)
            }

    # Constraint 1: Each source assigned to exactly one fold
    for class_name, sources in structure.items():
        for source in sources.keys():
            prob += lpSum([x[class_name][source][f'fold{i}'] for i in range(NUM_FOLDS)]) == 1

    # Calculate target counts per class per fold
    class_totals = {cls: sum(len(files) for files in sources.values())
                    for cls, sources in structure.items()}

    # Slack variables for fold size deviations per class
    slack_pos = {}
    slack_neg = {}
    for class_name in structure.keys():
        # Sanitize class name for PuLP variable names
        safe_cls = class_name.replace(' ', '_').replace('-', '_').replace('.', '_')
        slack_pos[class_name] = {
            f'fold{i}': LpVariable(f"slack_pos_{safe_cls}_fold{i}", lowBound=0)
            for i in range(NUM_FOLDS)
        }
        slack_neg[class_name] = {
            f'fold{i}': LpVariable(f"slack_neg_{safe_cls}_fold{i}", lowBound=0)
            for i in range(NUM_FOLDS)
        }

    # Constraint 2: Each class should have ~20% per fold (soft constraint)
    for class_name, sources in structure.items():
        total = class_totals[class_name]
        target_per_fold = total // NUM_FOLDS

        for i in range(NUM_FOLDS):
            fold_samples = lpSum([
                x[class_name][source][f'fold{i}'] * len(sources[source])
                for source in sources.keys()
            ])
            prob += (fold_samples - slack_pos[class_name][f'fold{i}'] +
                     slack_neg[class_name][f'fold{i}'] == target_per_fold)

    # Objective: Minimize total deviation from target fold sizes
    prob += lpSum([
        slack_pos[class_name][f'fold{i}'] + slack_neg[class_name][f'fold{i}']
        for class_name in structure.keys()
        for i in range(NUM_FOLDS)
    ])

    # Solve
    if verbose:
        print("  Solving MIP problem (may take a few minutes)...")
    solver = PULP_CBC_CMD(msg=0, timeLimit=600)
    prob.solve(solver)

    if prob.status != 1:
        raise RuntimeError(f"MIP solver failed: status={prob.status}")

    # Extract assignment
    assignment = {}
    for class_name, sources in structure.items():
        assignment[class_name] = {}
        for source in sources.keys():
            for i in range(NUM_FOLDS):
                if value(x[class_name][source][f'fold{i}']) > 0.5:
                    assignment[class_name][source] = f'fold{i}'
                    break

    # Print fold statistics
    if verbose:
        print("\n  Fold statistics per class:")
        print(f"  {'Class':<30} {'fold0':>6} {'fold1':>6} {'fold2':>6} {'fold3':>6} {'fold4':>6} {'Total':>7}")
        print("  " + "-" * 80)

        for class_name in sorted(structure.keys()):
            fold_counts = defaultdict(int)
            for source, fold in assignment[class_name].items():
                fold_counts[fold] += len(structure[class_name][source])

            counts_str = [f"{fold_counts[f'fold{i}']:>6}" for i in range(NUM_FOLDS)]
            total = sum(fold_counts.values())
            print(f"  {class_name:<30} " + " ".join(counts_str) + f" {total:>7}")
        print()

    return assignment


# ============================================================================
# Feature Extraction (Mel Spectrogram)
# ============================================================================

def audio_to_melspec(audio, label, augment=False):
    """Convert audio to mel spectrogram (224x224x3)."""
    # Ensure audio is numpy array (tf.py_function should already provide numpy arrays)
    audio = np.asarray(audio, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=SAMPLE_RATE // 2
    )
    # top_db clips extreme negative values and stabilizes normalization
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Resize to 224x224 using scipy (avoid TF ops inside py_function)
    from scipy.ndimage import zoom
    h_scale = TARGET_SPEC_HEIGHT / mel_db.shape[0]
    w_scale = TARGET_SPEC_WIDTH / mel_db.shape[1]
    mel_resized = zoom(mel_db, (h_scale, w_scale), order=1)

    # Crop or pad to ensure exact 224x224 (zoom might be slightly off due to rounding)
    h, w = mel_resized.shape
    if h > TARGET_SPEC_HEIGHT:
        mel_resized = mel_resized[:TARGET_SPEC_HEIGHT, :]
    elif h < TARGET_SPEC_HEIGHT:
        pad_h = TARGET_SPEC_HEIGHT - h
        mel_resized = np.pad(mel_resized, ((0, pad_h), (0, 0)), mode='edge')

    if w > TARGET_SPEC_WIDTH:
        mel_resized = mel_resized[:, :TARGET_SPEC_WIDTH]
    elif w < TARGET_SPEC_WIDTH:
        pad_w = TARGET_SPEC_WIDTH - w
        mel_resized = np.pad(mel_resized, ((0, 0), (0, pad_w)), mode='edge')

    # Add channel dimension and convert to 3-channel
    mel_rgb = np.stack([mel_resized] * 3, axis=-1).astype(np.float32)

    # MobileNetV3 preprocessing: expects [0, 255] range, then scales to [-1, 1]
    # mel_rgb is currently [0, 1], so scale to [0, 255] first
    mel_scaled = mel_rgb * 255.0

    # Apply MobileNetV3 preprocessing manually (equivalent to preprocess_input)
    # MobileNetV3 uses: (x / 127.5) - 1.0 which maps [0, 255] to [-1, 1]
    mel_preprocessed = (mel_scaled / 127.5) - 1.0

    return mel_preprocessed.astype(np.float32), label


# ============================================================================
# Dataset Building
# ============================================================================

def build_multiclass_dataset(dataset_root: Path,
                             structure: Dict[str, Dict[str, List[str]]],
                             assignment: Dict[str, Dict[str, str]],
                             class_to_idx: Dict[str, int],
                             fold_names: List[str],
                             batch_size: int = 32,
                             shuffle: bool = False,
                             seed: int = 42):
    """
    Build dataset from specified folds across all classes.

    Args:
        dataset_root: Path to dataset directory
        structure: class -> source -> files mapping
        assignment: class -> source -> fold_name mapping
        class_to_idx: class_name -> integer label mapping
        fold_names: List of fold names to include (e.g., ['fold0', 'fold1', ...])
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        (dataset, file_info_list)
        file_info_list: List of (filename, class_name, source, label_idx) for tracking
    """
    paths = []
    labels = []
    file_info = []

    fold_set = set(fold_names)

    for class_name in sorted(structure.keys()):
        class_dir = dataset_root / class_name
        label_idx = class_to_idx[class_name]

        for source, fold in assignment[class_name].items():
            if fold not in fold_set:
                continue

            for filename in structure[class_name][source]:
                file_path = class_dir / filename
                if file_path.exists():
                    paths.append(str(file_path))
                    labels.append(label_idx)
                    file_info.append((filename, class_name, source, label_idx))

    if not paths:
        raise ValueError(f"No samples found for folds {fold_names}")

    def load_audio_pyfunc(path, label):
        # Decode path to string
        if isinstance(path, bytes):
            path_str = path.decode('utf-8')
        elif hasattr(path, 'numpy'):
            path_bytes = path.numpy()
            path_str = path_bytes.decode('utf-8') if isinstance(path_bytes, bytes) else str(path_bytes)
        else:
            path_str = str(path)

        audio, _ = librosa.load(path_str, sr=SAMPLE_RATE, duration=3.0)
        if len(audio) < TARGET_LENGTH_SAMPLES:
            audio = np.pad(audio, (0, TARGET_LENGTH_SAMPLES - len(audio)))
        else:
            audio = audio[:TARGET_LENGTH_SAMPLES]
        return audio.astype(np.float32), label

    def process_audio(path, label):
        audio, label = tf.py_function(
            load_audio_pyfunc,
            [path, label],
            [tf.float32, tf.int32]
        )
        audio.set_shape([TARGET_LENGTH_SAMPLES])
        label.set_shape([])

        spec, label = tf.py_function(
            lambda a, l: audio_to_melspec(a, l, augment=False),
            [audio, label],
            [tf.float32, tf.int32]
        )
        spec.set_shape([TARGET_SPEC_HEIGHT, TARGET_SPEC_WIDTH, 3])
        label.set_shape([])
        return spec, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Reduce parallelism to avoid librosa FFTW/OpenMP thread deadlocks in tf.py_function
    dataset = dataset.map(process_audio, num_parallel_calls=2)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000, seed=seed)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, file_info


# ============================================================================
# Model Creation
# ============================================================================

def create_mobilenetv3s_multiclass(num_classes: int, use_pretrained: bool = True):
    """Create MobileNetV3Small for multi-class classification."""
    weights = 'imagenet' if use_pretrained else None

    base = applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=weights,
        pooling='avg'
    )

    model = keras.Sequential([
        base,
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base


# ============================================================================
# 5-Fold Cross-Validation
# ============================================================================

def run_5fold_cv(dataset_root: Path,
                structure: Dict[str, Dict[str, List[str]]],
                assignment: Dict[str, Dict[str, str]],
                classes: List[str],
                seed: int = 42):
    """
    Run 5-fold cross-validation and collect all predictions.

    Returns:
        all_predictions: List of prediction dicts
        confusion_matrix_accumulated: Confusion matrix summed across all folds
    """
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    num_classes = len(classes)

    all_predictions = []
    confusion_matrix_accumulated = np.zeros((num_classes, num_classes), dtype=int)

    print(f"\nRunning 5-fold cross-validation...")
    print(f"Classes: {num_classes}")
    print(f"Max epochs per fold: {MAX_EPOCHS_PER_FOLD}")
    print()

    for fold_idx in range(NUM_FOLDS):
        print(f"{'='*80}")
        print(f"Fold {fold_idx+1}/{NUM_FOLDS}")
        print(f"{'='*80}")

        # Train folds: all except current fold
        train_fold_names = [f'fold{i}' for i in range(NUM_FOLDS) if i != fold_idx]
        val_fold_name = f'fold{fold_idx}'

        # Build datasets
        print("Building train dataset...")
        train_ds, _ = build_multiclass_dataset(
            dataset_root, structure, assignment, class_to_idx,
            train_fold_names, batch_size=BATCH_SIZE, shuffle=True, seed=seed
        )

        print("Building validation dataset...")
        val_ds, val_file_info = build_multiclass_dataset(
            dataset_root, structure, assignment, class_to_idx,
            [val_fold_name], batch_size=BATCH_SIZE, shuffle=False
        )

        print(f"Validation samples: {len(val_file_info)}")

        # Create model
        print("Creating model...")
        model, base = create_mobilenetv3s_multiclass(num_classes, use_pretrained=True)

        # Use legacy Adam optimizer on macOS for better performance
        if platform.system().lower() == 'darwin':
            try:
                optimizer = optimizers.legacy.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
            except AttributeError:
                optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        else:
            optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

        loss_fn = losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Train
        print("Training...")
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=3, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=8,
                mode='max', restore_best_weights=True, verbose=1
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MAX_EPOCHS_PER_FOLD,
            callbacks=callbacks,
            verbose=1
        )

        epochs_trained = len(history.history['loss'])
        best_val_acc = max(history.history['val_accuracy'])
        print(f"\nCompleted in {epochs_trained} epochs | Best val_acc: {best_val_acc:.4f}")

        # Get predictions
        print("Generating predictions...")
        y_pred_probs = model.predict(val_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.array([info[3] for info in val_file_info])

        # Accumulate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        confusion_matrix_accumulated += cm

        # Store detailed predictions
        for i, (filename, class_name, source, true_label_idx) in enumerate(val_file_info):
            pred_label_idx = y_pred[i]
            pred_probs = y_pred_probs[i]

            pred_class_name = idx_to_class[pred_label_idx]
            correct = (pred_label_idx == true_label_idx)

            # Confusion score: 1 - confidence in true class
            true_confidence = float(pred_probs[true_label_idx])
            pred_confidence = float(pred_probs[pred_label_idx])
            confusion_score = 1.0 - true_confidence

            all_predictions.append({
                'filename': filename,
                'source': source,
                'class': class_name,
                'true_label': class_name,
                'pred_label': pred_class_name,
                'true_confidence': true_confidence,
                'pred_confidence': pred_confidence,
                'correct': correct,
                'confusion_score': confusion_score,
                'fold': fold_idx
            })

        acc = np.sum(y_true == y_pred) / len(y_true)
        print(f"Fold {fold_idx+1} validation accuracy: {acc:.4f}")
        print()

        # Clean up
        del model, base, train_ds, val_ds
        tf.keras.backend.clear_session()

    return all_predictions, confusion_matrix_accumulated, classes


# ============================================================================
# Output Generation
# ============================================================================

def save_class_confusion_csvs(all_predictions: List[dict],
                              classes_to_trim: List[str],
                              output_dir: Path):
    """Save per-class CSVs for classes that need trimming (> 600 samples)."""
    print(f"\nSaving per-class confusion CSVs...")

    # Group predictions by class
    by_class = defaultdict(list)
    for pred in all_predictions:
        by_class[pred['class']].append(pred)

    for class_name in classes_to_trim:
        if class_name not in by_class:
            continue

        # Sort by confusion score (most confused first)
        class_preds = sorted(by_class[class_name],
                            key=lambda x: x['confusion_score'],
                            reverse=True)

        csv_path = output_dir / f"{class_name}_confusion_scores.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['filename', 'source', 'class', 'true_label', 'pred_label',
                         'true_confidence', 'pred_confidence', 'correct',
                         'confusion_score', 'fold']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(class_preds)

        accuracy = sum(p['correct'] for p in class_preds) / len(class_preds)
        print(f"  {class_name}: {len(class_preds)} samples, accuracy={accuracy:.2%}, saved to {csv_path.name}")


def plot_composite_confusion_matrix(cm: np.ndarray,
                                    classes: List[str],
                                    output_dir: Path):
    """Plot confusion matrix accumulated from all 5 folds."""
    print(f"\nGenerating composite confusion matrix...")

    # Normalize by row (true class)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # Prevent division by zero for empty rows
    cm_normalized = cm.astype('float') / row_sums

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Proportion'})

    plt.title('Confusion Matrix (Accumulated from 5 Folds, Row-Normalized)', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = output_dir / "confusion_matrix_5fold_composite.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  ✓ Saved: {plot_path}")

    # Also save raw counts
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix (Accumulated from 5 Folds, Raw Counts)', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path_raw = output_dir / "confusion_matrix_5fold_composite_raw_counts.png"
    plt.savefig(plot_path_raw, dpi=150)
    plt.close()

    print(f"  ✓ Saved: {plot_path_raw}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 9b: Find most confused samples for dataset balancing"
    )
    parser.add_argument('--dataset_root', default=str(MYGARDENBIRD_16K),
                        help=f'Path to 16kHz dataset (default: {MYGARDENBIRD_16K})')
    parser.add_argument('--output_dir', default='./confusion_analysis',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU mode (disable GPU)')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set preprocessing function
    global PREPROCESS_FN
    PREPROCESS_FN = applications.mobilenet_v3.preprocess_input

    if args.force_cpu:
        print("Forcing CPU mode (CUDA disabled)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print startup info
    print("=" * 80)
    print("STAGE 9b: FIND MOST CONFUSED SAMPLES")
    print("=" * 80)
    print("OBJECTIVE:")
    print(f"  Identify most confused samples in classes with > {TARGET_SAMPLES_PER_CLASS} samples")
    print(f"  to enable balancing to exactly {TARGET_SAMPLES_PER_CLASS} samples per class")
    print()
    print("METHOD:")
    print(f"  - 5-Fold Cross-Validation with MIP source-based splitting")
    print(f"  - Model: MobileNetV3Small (pretrained)")
    print(f"  - Max epochs per fold: {MAX_EPOCHS_PER_FOLD}")
    print(f"  - Feature: Mel spectrogram (224x224)")
    print(f"  - Sample rate: {SAMPLE_RATE} Hz")
    print()
    print("INPUT:")
    print(f"  Dataset root: {dataset_root}")
    print()
    print("OUTPUT:")
    print(f"  Results directory: {output_dir}")
    print(f"  - <class>_confusion_scores.csv (for classes with > 600 samples)")
    print(f"  - confusion_matrix_5fold_composite.png (normalized)")
    print(f"  - confusion_matrix_5fold_composite_raw_counts.png")
    print("=" * 80)
    print()

    # Count samples per class
    print("Analyzing dataset...")
    sample_counts = count_samples_per_class(dataset_root)

    classes_below = {cls: cnt for cls, cnt in sample_counts.items()
                     if cnt < TARGET_SAMPLES_PER_CLASS}
    classes_exact = {cls: cnt for cls, cnt in sample_counts.items()
                     if cnt == TARGET_SAMPLES_PER_CLASS}
    classes_above = {cls: cnt for cls, cnt in sample_counts.items()
                     if cnt > TARGET_SAMPLES_PER_CLASS}

    # Classes to include in training: >= 600 samples
    eligible_classes = sorted(list(classes_exact.keys()) + list(classes_above.keys()))

    print(f"\nDataset Summary:")
    print(f"  Classes with < {TARGET_SAMPLES_PER_CLASS} samples: {len(classes_below)} (excluded)")
    print(f"  Classes with = {TARGET_SAMPLES_PER_CLASS} samples: {len(classes_exact)} (included)")
    print(f"  Classes with > {TARGET_SAMPLES_PER_CLASS} samples: {len(classes_above)} (included, need trimming)")
    print(f"  Total eligible classes for training: {len(eligible_classes)}")
    print()

    if classes_below:
        print("Classes with insufficient samples (< 600, excluded):")
        for cls, cnt in sorted(classes_below.items()):
            print(f"  {cls}: {cnt} samples")
        print()

    if classes_exact:
        print("Classes already balanced (= 600, included):")
        for cls in sorted(classes_exact.keys()):
            print(f"  {cls}: {TARGET_SAMPLES_PER_CLASS} samples")
        print()

    if classes_above:
        print("Classes to trim (> 600, included + need trimming):")
        for cls, cnt in sorted(classes_above.items()):
            excess = cnt - TARGET_SAMPLES_PER_CLASS
            print(f"  {cls}: {cnt} samples (+{excess} to remove)")
        print()

    if not eligible_classes:
        print("❌ No classes have >= 600 samples. Cannot proceed.")
        return

    # Load dataset structure for eligible classes
    print("Loading dataset structure...")
    structure = load_dataset_structure(dataset_root, eligible_classes)

    # Create 5-fold MIP splits
    assignment = create_multiclass_5fold_mip_splits(structure)

    # Run 5-fold cross-validation
    start_time = time.time()
    all_predictions, cm_accumulated, classes = run_5fold_cv(
        dataset_root, structure, assignment, eligible_classes, seed=args.seed
    )
    elapsed = time.time() - start_time

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    # Save per-class CSVs for classes that need trimming
    if classes_above:
        save_class_confusion_csvs(all_predictions, list(classes_above.keys()), output_dir)

    # Save composite confusion matrix
    plot_composite_confusion_matrix(cm_accumulated, classes, output_dir)

    # Summary statistics
    total_accuracy = sum(p['correct'] for p in all_predictions) / len(all_predictions)

    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Overall accuracy: {total_accuracy:.2%}")
    print(f"Results saved to: {output_dir}")
    print()

    if classes_above:
        print("Next steps for dataset balancing:")
        print(f"  1. Review <class>_confusion_scores.csv for each class with > 600 samples")
        print(f"  2. Remove top N most confused samples (where N = excess samples)")
        print(f"     Example: If Javan_Myna has 850 samples, remove top 250 from CSV")
        print(f"  3. Re-run dataset splitting (Stage 8) with balanced classes")
        print()

    print("Confusion matrix shows inter-class confusion patterns.")
    print("=" * 80)


if __name__ == "__main__":
    main()
