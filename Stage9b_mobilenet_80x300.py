#!/usr/bin/env python3
"""
MobileNetV3-Small on SEAbird with a native 80 mel × 300 time-step spectrogram.

Unlike Stage9 (which resizes everything to 224×224), this script keeps the
audio-native spectrogram shape: 80 mel bins × 300 time frames (hop = 160 samples,
i.e. 10 ms at 16 kHz).  This matches the 80-mel front-end used by MynaNet
(mynanet_v1.py) so results can be directly compared.

Spectrogram parameters
----------------------
  n_mels      = 80
  time_steps  = 300   (hop_length = 48 000 // 300 = 160 samples)
  duration    = 3 s at 16 kHz → 48 000 samples → exactly 300 frames
  input shape = (80, 300, 3)  fed to MobileNetV3-Small with include_top=False

Usage examples
--------------
  # Minimal (CSV splits, pretrained):
  python Stage9b_mobilenet_80x300.py \\
      --splits_csv /Volumes/Evo/SEABIRD/splits/splits_mip_75_10_15_16khz.csv \\
      --use_pretrained

  # Full example:
  python Stage9b_mobilenet_80x300.py \\
      --splits_csv /Volumes/Evo/SEABIRD/splits/splits_mip_75_10_15_16khz.csv \\
      --dataset_root /Volumes/Evo/SEABIRD/extracted_segments \\
      --use_pretrained \\
      --seed 42 \\
      --num_epochs 50 \\
      --output_dir ./results_80x300

Defaults
--------
  --n_mels        80
  --time_steps    300
  --sample_rate   16000
  --batch_size    32
  --num_epochs    50
  --learning_rate 0.001
  --seed          42
  --output_dir    ./results_80x300
"""

import os
import csv
import json
import platform
import random
import time
import socket
from io import StringIO
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List

import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with 'pip install psutil' for resource info.")

from config import MYGARDENBIRD_16K, MYGARDENBIRD_44K, METADATA_16K, METADATA_44K

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, losses
from tensorflow.keras.applications import mobilenet_v3

if not tf.__version__.startswith("2.15."):
    print("This script requires TensorFlow 2.15.x")
    exit(1)

tf.keras.backend.set_image_data_format('channels_last')

# ── Global audio / spectrogram parameters (set in main()) ───────────────────
SAMPLE_RATE = 16000
TARGET_LENGTH_SAMPLES = 48000   # 3 s × 16 kHz
N_MELS = 80
TIME_STEPS = 300
HOP_LENGTH = TARGET_LENGTH_SAMPLES // TIME_STEPS   # 160
N_FFT = 2048


# ── Spectrogram conversion ───────────────────────────────────────────────────

def audio_to_mel80x300(audio: np.ndarray, label, augment: bool = False):
    """
    Convert a 48 000-sample audio array to an (80, 300, 3) mel spectrogram.

    Steps:
      1. Log-mel spectrogram with librosa (shape 80 × 300).
      2. Optional SpecAugment: random time / frequency masking (p=0.5 each).
      3. Clip to [p2, p98], scale to [0, 255] for ImageNet preprocessing.
      4. Replicate single channel → 3-channel tensor.
      5. Apply mobilenet_v3.preprocess_input (maps [0,255] to [-1, 1]).
    """
    audio = np.squeeze(audio).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0.0, fmax=SAMPLE_RATE // 2,
        center=True, pad_mode='constant'
    )                                           # (80, ~300)
    log_mel = librosa.power_to_db(mel, top_db=None)

    # Crop / pad time axis to exactly TIME_STEPS
    if log_mel.shape[1] >= TIME_STEPS:
        log_mel = log_mel[:, :TIME_STEPS]
    else:
        pad_w = TIME_STEPS - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_w)), mode='constant',
                         constant_values=log_mel.min())

    # SpecAugment
    if augment:
        if np.random.rand() < 0.5:
            w = np.random.randint(0, TIME_STEPS // 10)
            t0 = np.random.randint(0, max(1, TIME_STEPS - w))
            log_mel[:, t0:t0 + w] = log_mel.min()
        if np.random.rand() < 0.5:
            h = np.random.randint(0, N_MELS // 10)
            f0 = np.random.randint(0, max(1, N_MELS - h))
            log_mel[f0:f0 + h, :] = log_mel.min()

    # Normalise to [0, 255]
    p2, p98 = np.percentile(log_mel, (2, 98))
    if p98 > p2 + 1e-8:
        log_mel = np.clip(log_mel, p2, p98)
        log_mel = ((log_mel - p2) / (p98 - p2) * 255.0).astype(np.float32)
    else:
        log_mel = np.full_like(log_mel, 128.0, dtype=np.float32)

    # (80, 300) → (80, 300, 3)
    img = np.stack([log_mel] * 3, axis=-1)

    # ImageNet preprocessing for MobileNetV3 (maps [0,255] → [-1,1])
    img = mobilenet_v3.preprocess_input(img)
    return img.astype(np.float32), label


# ── CSV split helpers (identical to Stage9) ──────────────────────────────────

def load_splits_from_csv(csv_path: str, split: str = 'train') -> List[str]:
    files = []
    with open(csv_path, 'r') as f:
        content = ''.join(line for line in f if not line.startswith('#'))
    reader = csv.DictReader(StringIO(content))
    for row in reader:
        if row['split'] != split:
            continue
        if 'file_id' in row:
            fid = row['file_id']
            files.append('xc' + fid[2:] + '.wav')
        else:
            files.append(row['filename'])
    return files


def print_per_class_breakdown(csv_path: str, dataset_root: str, classes: list):
    file_to_split = {}
    with open(csv_path, 'r') as f:
        content = ''.join(line for line in f if not line.startswith('#'))
    reader = csv.DictReader(StringIO(content))
    for row in reader:
        sp = row['split']
        if 'file_id' in row:
            wav = 'xc' + row['file_id'][2:] + '.wav'
        else:
            wav = row['filename']
        file_to_split[wav] = sp

    clip_counts = defaultdict(lambda: defaultdict(int))
    src_sets    = defaultdict(lambda: defaultdict(set))

    for cls in classes:
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.wav', '.mp3')):
                continue
            sp = file_to_split.get(fname)
            if sp is None:
                continue
            clip_counts[cls][sp] += 1
            stem = os.path.splitext(fname)[0]
            src_id = stem.split('_')[0]
            src_sets[cls][sp].add(src_id)

    splits = ['train', 'val', 'test']
    print()
    print("Per-Class Breakdown:")
    print(f"{'Class':<33} {'Train':>7} {'Val':>6} {'Test':>6} {'Total':>7} | Sources")
    print("-" * 80)
    tot_clips = defaultdict(int)
    tot_srcs  = defaultdict(set)
    for cls in sorted(classes):
        tr = clip_counts[cls].get('train', 0)
        va = clip_counts[cls].get('val',   0)
        te = clip_counts[cls].get('test',  0)
        total = tr + va + te
        s_tr = len(src_sets[cls].get('train', set()))
        s_va = len(src_sets[cls].get('val',   set()))
        s_te = len(src_sets[cls].get('test',  set()))
        print(f"{cls:<33} {tr:>7} {va:>6} {te:>6} {total:>7} | {s_tr:3d}/{s_va:2d}/{s_te:2d}")
        for sp in splits:
            tot_clips[sp] += clip_counts[cls].get(sp, 0)
            tot_srcs[sp].update(src_sets[cls].get(sp, set()))
    print("-" * 80)
    tr_t = tot_clips['train']; va_t = tot_clips['val']; te_t = tot_clips['test']
    grand = tr_t + va_t + te_t
    s_tr_t = len(tot_srcs['train']); s_va_t = len(tot_srcs['val']); s_te_t = len(tot_srcs['test'])
    print(f"{'OVERALL':<33} {tr_t:>7} {va_t:>6} {te_t:>6} {grand:>7} | {s_tr_t:3d}/{s_va_t:2d}/{s_te_t:2d}")
    print()


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_dataset(root_dir, classes=None, augment=False, shuffle=False,
                  batch_size=32, num_parallel=4, seed=42,
                  csv_path=None, split_name=None):

    if classes is None:
        classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    paths, labels = [], []

    if csv_path and split_name:
        csv_files = set(load_splits_from_csv(csv_path, split_name))
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.wav', '.mp3')) and fname in csv_files:
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])
    else:
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.wav', '.mp3')):
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])

    if not paths:
        raise ValueError(f"No audio files found in {root_dir}")

    def load_audio(path, label):
        audio, _ = librosa.load(path.numpy().decode('utf-8'),
                                sr=SAMPLE_RATE, duration=3.0)
        if len(audio) < TARGET_LENGTH_SAMPLES:
            audio = np.pad(audio, (0, TARGET_LENGTH_SAMPLES - len(audio)))
        else:
            audio = audio[:TARGET_LENGTH_SAMPLES]
        return audio.astype(np.float32), label

    def process(path, label):
        audio, label = tf.py_function(load_audio, [path, label],
                                      [tf.float32, tf.int32])
        audio.set_shape([TARGET_LENGTH_SAMPLES])
        label.set_shape([])

        spec, label = tf.py_function(
            lambda a, l: audio_to_mel80x300(a, l, augment=augment),
            [audio, label],
            [tf.float32, tf.int32]
        )
        spec.set_shape([N_MELS, TIME_STEPS, 3])
        label.set_shape([])
        return spec, label

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(paths),
        tf.data.Dataset.from_tensor_slices(labels)
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(process, num_parallel_calls=num_parallel)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, classes


# ── Model ────────────────────────────────────────────────────────────────────

def create_model(num_classes: int, use_pretrained: bool):
    """
    MobileNetV3-Small with a native (80, 300, 3) input.

    ImageNet weights are compatible: depthwise+pointwise conv kernels
    are position-agnostic; global average pooling handles the non-square
    spatial dimensions (5×19 at the deepest feature map).
    """
    weights = 'imagenet' if use_pretrained else None

    base = applications.MobileNetV3Small(
        input_shape=(N_MELS, TIME_STEPS, 3),
        include_top=False,
        weights=weights,
        pooling='avg'         # → flat (576,) feature vector
    )

    model = keras.Sequential([
        base,
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model, base


# ── Plotting & logging helpers ───────────────────────────────────────────────

def plot_history(history, outdir: Path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=150)
    plt.close()


def save_system_config(outdir: Path):
    from importlib import metadata
    lines = [
        "=" * 80, "SYSTEM CONFIGURATION", "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "Platform Information:",
        f"  OS: {platform.system()} {platform.release()}",
        f"  Platform: {platform.platform()}",
        f"  Architecture: {platform.machine()}",
        f"  Processor: {platform.processor()}",
        f"  Hostname: {socket.gethostname()}", "",
    ]
    if PSUTIL_AVAILABLE:
        lines += [
            "CPU Information:",
            f"  Physical cores: {psutil.cpu_count(logical=False)}",
            f"  Logical cores: {psutil.cpu_count(logical=True)}", "",
            "Memory Information:",
        ]
        mem = psutil.virtual_memory()
        lines += [
            f"  Total: {mem.total / (1024**3):.2f} GB",
            f"  Available: {mem.available / (1024**3):.2f} GB",
            f"  Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)", "",
        ]

    def pkg(p):
        try: return metadata.version(p)
        except: return "unknown"

    lines += [
        "TensorFlow Configuration:",
        f"  TensorFlow version: {tf.__version__}",
        f"  Keras version: {pkg('keras')}",
        f"  GPUs available: {len(tf.config.list_physical_devices('GPU'))}",
        "",
        "Python Environment:",
        f"  Python version: {platform.python_version()}",
        f"  NumPy version: {np.__version__}",
        f"  Librosa version: {librosa.__version__}", "",
    ]
    (outdir / "system_config.txt").write_text("\n".join(lines))


def save_hyperparameters(outdir: Path, args, model,
                         early_stop_patience, lr_plateau_patience,
                         monitor_metric, weight_decay):
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    lines = [
        "=" * 80, "TRAINING HYPERPARAMETERS", "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "Model Architecture:",
        "  Model: MobileNetV3-Small (audio-native 80x300 input)",
        f"  Pretrained: {args.use_pretrained}",
        f"  Total parameters: {model.count_params():,}",
        f"  Trainable parameters: {trainable:,}",
        "  Dropout rates: [0.3, 0.2]", "",
        "Spectrogram Parameters:",
        f"  Feature: mel-spectrogram",
        f"  n_mels: {N_MELS}",
        f"  time_steps: {TIME_STEPS}",
        f"  hop_length: {HOP_LENGTH} samples ({HOP_LENGTH/args.sample_rate*1000:.1f} ms)",
        f"  n_fft: {N_FFT}",
        f"  Sample rate: {args.sample_rate} Hz",
        f"  Input shape: ({N_MELS}, {TIME_STEPS}, 3)", "",
        "Training Configuration:",
        f"  Batch size: {args.batch_size}",
        f"  Max epochs: {args.num_epochs}",
        f"  Learning rate: {args.learning_rate}",
    ]
    if platform.system().lower() == 'linux':
        lines.append(f"  Optimizer: AdamW (weight_decay={weight_decay})")
    else:
        lines.append("  Optimizer: Adam.legacy (macOS/Metal)")
    lines += [
        "  Gradient clipping: clipnorm=1.0", "",
        "Callbacks:",
        f"  EarlyStopping: monitor={monitor_metric}, patience={early_stop_patience}",
        f"  ReduceLROnPlateau: monitor=val_loss, patience={lr_plateau_patience}, factor=0.5", "",
        "Dataset:",
        f"  splits_csv: {args.splits_csv}",
        f"  dataset_root: {args.dataset_root}", "",
        "Reproducibility:",
        f"  Random seed: {args.seed}", "",
    ]
    (outdir / "hyperparameters.txt").write_text("\n".join(lines))


def save_runtime_info(outdir: Path, train_time: float, history, test_acc: float, test_loss: float):
    h = history.history
    best_val_ep = int(np.argmax(h['val_accuracy'])) + 1
    final_ep = len(h['loss'])
    overfitting_gap = max(h['val_accuracy']) - h['val_accuracy'][-1]
    tv_gap = h['accuracy'][-1] - h['val_accuracy'][-1]

    lines = [
        "=" * 80, "RUNTIME INFORMATION", "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "Training Duration:",
        f"  Total time: {train_time:.2f} minutes ({train_time*60:.1f} seconds)",
        f"  Epochs completed: {final_ep}",
        f"  Time per epoch: {train_time / final_ep:.2f} minutes", "",
        "Training History:",
        f"  Final train accuracy: {h['accuracy'][-1]:.4f}",
        f"  Final train loss: {h['loss'][-1]:.4f}",
        f"  Best train accuracy: {max(h['accuracy']):.4f} (epoch {int(np.argmax(h['accuracy']))+1})",
        f"  Best train loss: {min(h['loss']):.4f} (epoch {int(np.argmin(h['loss']))+1})", "",
        "Validation History:",
        f"  Final val accuracy: {h['val_accuracy'][-1]:.4f}",
        f"  Final val loss: {h['val_loss'][-1]:.4f}",
        f"  Best val accuracy: {max(h['val_accuracy']):.4f} (epoch {best_val_ep})",
        f"  Best val loss: {min(h['val_loss']):.4f} (epoch {int(np.argmin(h['val_loss']))+1})", "",
        "Overfitting Analysis:",
        f"  Best validation accuracy at epoch: {best_val_ep}",
        f"  Training stopped at epoch: {final_ep}",
        f"  Epochs after peak: {final_ep - best_val_ep}",
        f"  Val accuracy degradation from peak: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)",
        f"  Train-val accuracy gap (final): {tv_gap:.4f} ({tv_gap*100:.2f}%)", "",
        "Test Results:",
        f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)",
        f"  Test loss: {test_loss:.4f}", "",
        "System Resources (at completion):",
    ]
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        lines += [
            f"  Memory used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)",
            f"  Memory available: {mem.available / (1024**3):.2f} GB",
            f"  CPU usage: {psutil.cpu_percent(interval=1)}%",
        ]
    else:
        lines.append("  (psutil not available)")
    lines.append("")
    (outdir / "runtime_info.txt").write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small on SEAbird with native 80×300 mel spectrogram"
    )
    # Data
    parser.add_argument('--splits_csv', default=None, type=str,
                        help='Stage-8 splits CSV (recommended). If omitted, uses --train/val/test_dir.')
    parser.add_argument('--dataset_root', default=str(MYGARDENBIRD_16K), type=str,
                        help=f'Audio root dir for CSV-based splits. Default: {MYGARDENBIRD_16K}')
    parser.add_argument('--train_dir', default=str(MYGARDENBIRD_16K / 'train'))
    parser.add_argument('--val_dir',   default=str(MYGARDENBIRD_16K / 'val'))
    parser.add_argument('--test_dir',  default=str(MYGARDENBIRD_16K / 'test'))
    # Spectrogram
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_mels',      type=int, default=80,
                        help='Mel bins (default 80 to match MynaNet)')
    parser.add_argument('--time_steps',  type=int, default=300,
                        help='Time frames; hop_length = (sr*3) // time_steps (default 300 → hop=160)')
    parser.add_argument('--n_fft',       type=int, default=2048)
    # Training
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--num_epochs',     type=int,   default=50)
    parser.add_argument('--learning_rate',  type=float, default=0.001)
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--force_cpu',      action='store_true',
                        help='Disable all GPUs (use CPU only)')
    parser.add_argument('--output_dir', default='./results_80x300')
    args = parser.parse_args()

    # ── Seeds ──────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # ── GPU / CPU ──────────────────────────────────────────────────────────
    if args.force_cpu:
        print("Forcing CPU mode")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')

    # ── Update globals from args ───────────────────────────────────────────
    global SAMPLE_RATE, TARGET_LENGTH_SAMPLES, N_MELS, TIME_STEPS, HOP_LENGTH, N_FFT
    SAMPLE_RATE            = args.sample_rate
    TARGET_LENGTH_SAMPLES  = int(SAMPLE_RATE * 3.0)
    N_MELS                 = args.n_mels
    TIME_STEPS             = args.time_steps
    HOP_LENGTH             = TARGET_LENGTH_SAMPLES // TIME_STEPS
    N_FFT                  = args.n_fft

    # ── Header ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SEAbird — MobileNetV3-Small  |  Mel 80 × 300 (audio-native)")
    print("=" * 80)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Input shape: ({N_MELS}, {TIME_STEPS}, 3)   hop={HOP_LENGTH} samples")
    print(f"Pretrained: {args.use_pretrained}   Seed: {args.seed}")
    print()

    # ── Datasets ───────────────────────────────────────────────────────────
    print("Building datasets...")
    if args.splits_csv:
        print(f"Using CSV splits: {args.splits_csv}")
        print(f"Dataset root:     {args.dataset_root}")
        train_ds, classes = build_dataset(
            args.dataset_root, augment=True, shuffle=True,
            batch_size=args.batch_size, num_parallel=args.num_workers,
            seed=args.seed, csv_path=args.splits_csv, split_name='train'
        )
        val_ds, _ = build_dataset(
            args.dataset_root, augment=False, shuffle=False,
            batch_size=args.batch_size, num_parallel=args.num_workers,
            csv_path=args.splits_csv, split_name='val'
        )
        test_ds, _ = build_dataset(
            args.dataset_root, augment=False, shuffle=False,
            batch_size=args.batch_size, num_parallel=args.num_workers,
            csv_path=args.splits_csv, split_name='test'
        )
    else:
        train_ds, classes = build_dataset(
            args.train_dir, augment=True, shuffle=True,
            batch_size=args.batch_size, num_parallel=args.num_workers,
            seed=args.seed
        )
        val_ds, _ = build_dataset(
            args.val_dir, augment=False, shuffle=False,
            batch_size=args.batch_size, num_parallel=args.num_workers
        )
        test_ds, _ = build_dataset(
            args.test_dir, augment=False, shuffle=False,
            batch_size=args.batch_size, num_parallel=args.num_workers
        )

    print(f"Classes ({len(classes)}): {classes}")
    if args.splits_csv:
        print_per_class_breakdown(args.splits_csv, args.dataset_root, classes)

    # ── Model ──────────────────────────────────────────────────────────────
    print("Creating model...")
    model, base = create_model(len(classes), args.use_pretrained)
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")

    # ── Optimiser ──────────────────────────────────────────────────────────
    weight_decay     = 1e-4
    early_stop_pat   = 15
    lr_plateau_pat   = 5
    monitor_metric   = 'val_accuracy'
    monitor_mode     = 'max'

    def make_optimizer(lr):
        if platform.system().lower() == 'linux':
            return optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=1.0)
        try:
            return optimizers.legacy.Adam(learning_rate=lr, clipnorm=1.0)
        except AttributeError:
            return optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    loss_fn = losses.SparseCategoricalCrossentropy()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=lr_plateau_pat,
            min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric, patience=early_stop_pat,
            mode=monitor_mode, restore_best_weights=True, verbose=1
        ),
    ]

    # ── Warm-up: train head only ───────────────────────────────────────────
    start_time = time.time()

    if args.use_pretrained:
        warmup_epochs = 10
        print(f"Warm-up: {warmup_epochs} epochs with frozen base...")
        base.trainable = False
        model.compile(optimizer=make_optimizer(args.learning_rate),
                      loss=loss_fn, metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds,
                  epochs=warmup_epochs, verbose=1)

        print("Fine-tuning: unfreezing all layers...")
        base.trainable = True
        model.compile(optimizer=make_optimizer(1e-4),
                      loss=loss_fn, metrics=['accuracy'])

    else:
        model.compile(optimizer=make_optimizer(args.learning_rate),
                      loss=loss_fn, metrics=['accuracy'])

    # ── Full training ──────────────────────────────────────────────────────
    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.num_epochs,
        callbacks=callbacks,
        verbose=1
    )
    train_time = (time.time() - start_time) / 60

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)

    y_true, y_pred = [], []
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    # ── Output directory ───────────────────────────────────────────────────
    pretrained_str = "pretrained" if args.use_pretrained else "scratch"
    run_name = f"mobilenetv3s_mel{N_MELS}x{TIME_STEPS}_{pretrained_str}_seed{args.seed}"
    out_platform = platform.platform().split('-')[0].lower()
    output_dir = Path(f"{args.output_dir}_{out_platform}") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    (output_dir / "classification_report.txt").write_text(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix — MobileNetV3s mel{N_MELS}×{TIME_STEPS} seed{args.seed}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\nClassification Report:\n", report)
    plot_history(history, output_dir)

    # Detailed logs
    save_system_config(output_dir)
    save_hyperparameters(output_dir, args, model,
                         early_stop_pat, lr_plateau_pat,
                         monitor_metric, weight_decay)
    save_runtime_info(output_dir, train_time, history, test_acc, test_loss)

    # results.json (compatible with Stage9 format for easy comparison)
    results = {
        'model': 'mobilenetv3s',
        'feature': f'mel{N_MELS}x{TIME_STEPS}',
        'input_shape': [N_MELS, TIME_STEPS, 3],
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_time_minutes': train_time,
        'config': vars(args)
    }
    (output_dir / 'results.json').write_text(json.dumps(results, indent=2))

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Input shape: ({N_MELS}, {TIME_STEPS}, 3)  hop={HOP_LENGTH}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Training Time: {train_time:.1f} min")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
