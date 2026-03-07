#!/usr/bin/env python3
"""
Train CNNs for MyGardenBird audio classification with multiple feature types.

Supports: Mel Spectrogram, STFT, MFCC+deltas
Models: MobileNetV3Small, VGG16, EfficientNetB0, ResNet50

Augmentation
------------
Default: no augmentation
--specaug         Enable SpecAugment (time + frequency masking)
--mixup <alpha>   Enable Mixup (e.g. --mixup 0.2). Overrides --specaug.

Usage Examples
--------------
# No augmentation (pretrained weights used by default):
python Stage9_train_mygardenbird_multifeature.py

# With SpecAugment:
python Stage9_train_mygardenbird_multifeature.py --specaug

# With Mixup (alpha=0.2):
python Stage9_train_mygardenbird_multifeature.py --mixup 0.2

# Full example:
python Stage9_train_mygardenbird_multifeature.py \\
    --model mobilenetv3s \\
    --feature mel \\
    --splits_csv /Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv \\
    --dataset_root /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz \\
    --batch_size 32 \\
    --num_epochs 50 \\
    --seed 42 \\
    --mixup 0.2

Defaults
--------
--model          mobilenetv3s
--feature        mel
--seed           42
--batch_size     32
--num_epochs     50
--learning_rate  0.001
--sample_rate    Auto-detected from first audio file (fallback: 16000)
--n_mels         224
--n_fft          2048
--num_workers    4
--output_dir     ./results
--dataset_root   /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz
--use_pretrained True (ImageNet pretrained weights)

Note: Sample rate is automatically detected from the first .wav file in dataset_root.
      You can override by explicitly specifying --sample_rate.
"""

import os
import json
import platform
from tqdm import tqdm
import time

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

import socket
from string import digits

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List
import argparse
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with 'pip install psutil' for detailed system info.")

from config import MYGARDENBIRD_16K, MYGARDENBIRD_44K, METADATA_16K, METADATA_44K

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, losses, metrics
from tensorflow.keras.applications import (
    mobilenet_v3,
    resnet50,
    vgg16,
    efficientnet
)

if not tf.__version__.startswith("2.15."):
    print("❌ This script requires TensorFlow 2.15.x")
    exit(1)

tf.keras.backend.set_image_data_format('channels_last')

# Global variables (set after arg parsing)
SAMPLE_RATE = None
TARGET_LENGTH_SAMPLES = None
HOP_LENGTH = None
N_FFT = 2048
N_MELS = 128
N_MFCC = 80
TARGET_SPEC_WIDTH = 224
TARGET_SPEC_HEIGHT = 224
PREPROCESS_FN = None
MIXUP_ALPHA = None   # Set from --mixup arg when mixup is active


# ============================================================================
# Mixup augmentation
# ============================================================================

def mixup_batch(images, labels_one_hot):
    """Apply Mixup to a batch of (images, one-hot labels)."""
    lam = tf.cast(
        tf.py_function(
            lambda: np.float32(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)),
            [],
            tf.float32
        ),
        tf.float32
    )
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels_one_hot, indices)
    mixed_images = lam * images + (1.0 - lam) * images_shuffled
    mixed_labels = lam * labels_one_hot + (1.0 - lam) * labels_shuffled

    # Explicitly set shapes to avoid "unknown TensorShape" errors in metrics
    mixed_images.set_shape(images.shape)
    mixed_labels.set_shape(labels_one_hot.shape)

    return mixed_images, mixed_labels


# ============================================================================
# Feature extraction
# ============================================================================

def audio_to_melspec(audio, label, augment=False):
    """Convert audio to mel spectrogram."""
    audio = np.squeeze(audio).astype(np.float32)

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0.0, fmax=SAMPLE_RATE // 2,
        center=True, pad_mode='constant'
    )

    log_mel_spec = librosa.power_to_db(mel_spec, top_db=None)

    if augment:
        if np.random.rand() < 0.5:
            time_mask_width = np.random.randint(0, log_mel_spec.shape[1] // 10)
            start = np.random.randint(0, max(1, log_mel_spec.shape[1] - time_mask_width))
            log_mel_spec[:, start:start + time_mask_width] = np.min(log_mel_spec)

        if np.random.rand() < 0.5:
            freq_mask_width = np.random.randint(0, log_mel_spec.shape[0] // 10)
            start = np.random.randint(0, max(1, log_mel_spec.shape[0] - freq_mask_width))
            log_mel_spec[start:start + freq_mask_width, :] = np.min(log_mel_spec)

    # Crop or pad both axes to (TARGET_SPEC_HEIGHT, TARGET_SPEC_WIDTH).
    h, w = log_mel_spec.shape
    if h > TARGET_SPEC_HEIGHT:
        log_mel_spec = log_mel_spec[:TARGET_SPEC_HEIGHT, :]
    elif h < TARGET_SPEC_HEIGHT:
        log_mel_spec = np.pad(log_mel_spec,
                              ((0, TARGET_SPEC_HEIGHT - h), (0, 0)),
                              constant_values=log_mel_spec.min())
    if w > TARGET_SPEC_WIDTH:
        log_mel_spec = log_mel_spec[:, :TARGET_SPEC_WIDTH]
    elif w < TARGET_SPEC_WIDTH:
        log_mel_spec = np.pad(log_mel_spec,
                              ((0, 0), (0, TARGET_SPEC_WIDTH - w)),
                              constant_values=log_mel_spec.min())
    log_mel_spec = np.stack([log_mel_spec] * 3, axis=-1)

    p2, p98 = np.percentile(log_mel_spec, (2, 98))
    if p98 > p2 + 1e-8:
        log_mel_spec = np.clip(log_mel_spec, p2, p98)
        log_mel_spec = ((log_mel_spec - p2) / (p98 - p2) * 255.0).astype(np.float32)
    else:
        log_mel_spec = np.full_like(log_mel_spec, 128.0, dtype=np.float32)

    log_mel_spec = PREPROCESS_FN(log_mel_spec)
    return log_mel_spec.astype(np.float32), label


def audio_to_stft(audio, label, augment=False):
    """Convert audio to STFT magnitude spectrogram."""
    audio = np.squeeze(audio).astype(np.float32)

    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_mag = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_mag, top_db=None)

    if augment:
        if np.random.rand() < 0.5:
            time_mask_width = np.random.randint(0, stft_db.shape[1] // 10)
            start = np.random.randint(0, max(1, stft_db.shape[1] - time_mask_width))
            stft_db[:, start:start + time_mask_width] = np.min(stft_db)

        if np.random.rand() < 0.5:
            freq_mask_width = np.random.randint(0, stft_db.shape[0] // 10)
            start = np.random.randint(0, max(1, stft_db.shape[0] - freq_mask_width))
            stft_db[start:start + freq_mask_width, :] = np.min(stft_db)

    stft_db = np.expand_dims(stft_db, axis=-1)
    stft_db = np.repeat(stft_db, 3, axis=-1)
    stft_db = tf.image.resize(stft_db, [TARGET_SPEC_HEIGHT, TARGET_SPEC_WIDTH]).numpy()

    p2, p98 = np.percentile(stft_db, (2, 98))
    if p98 > p2 + 1e-8:
        stft_db = np.clip(stft_db, p2, p98)
        stft_db = ((stft_db - p2) / (p98 - p2) * 255.0).astype(np.float32)
    else:
        stft_db = np.full_like(stft_db, 128.0, dtype=np.float32)

    stft_db = PREPROCESS_FN(stft_db)
    return stft_db.astype(np.float32), label


def audio_to_mfcc_deltas(audio, label, augment=False):
    """Convert audio to MFCC (repeated 3×) as RGB."""
    audio = np.squeeze(audio).astype(np.float32)

    mfccs = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    if augment:
        for feat in [mfccs, mfcc_delta, mfcc_delta2]:
            if np.random.rand() < 0.5:
                time_mask_width = np.random.randint(0, feat.shape[1] // 10)
                start = np.random.randint(0, max(1, feat.shape[1] - time_mask_width))
                feat[:, start:start + time_mask_width] = np.min(feat)

    mfccs_normalized = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-9)
    mfcc_rgb = np.stack([mfccs_normalized] * 3, axis=-1)
    mfcc_rgb = tf.image.resize(mfcc_rgb, [TARGET_SPEC_HEIGHT, TARGET_SPEC_WIDTH], method='bicubic').numpy()

    p2, p98 = np.percentile(mfcc_rgb, (2, 98))
    if p98 > p2 + 1e-8:
        mfcc_rgb = np.clip(mfcc_rgb, p2, p98)
        mfcc_rgb = ((mfcc_rgb - p2) / (p98 - p2) * 255.0).astype(np.float32)
    else:
        mfcc_rgb = np.full_like(mfcc_rgb, 128.0, dtype=np.float32)

    mfcc_rgb = PREPROCESS_FN(mfcc_rgb)
    return mfcc_rgb.astype(np.float32), label


# ============================================================================
# Dataset utilities
# ============================================================================

def load_splits_from_csv(csv_path: str, split: str = 'train') -> List[str]:
    import csv
    from io import StringIO

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
    import csv as _csv
    from io import StringIO
    from collections import defaultdict

    file_to_split = {}
    with open(csv_path, 'r') as f:
        content = ''.join(line for line in f if not line.startswith('#'))
    reader = _csv.DictReader(StringIO(content))
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
    tr_t = tot_clips['train']
    va_t = tot_clips['val']
    te_t = tot_clips['test']
    grand = tr_t + va_t + te_t
    s_tr_t = len(tot_srcs['train'])
    s_va_t = len(tot_srcs['val'])
    s_te_t = len(tot_srcs['test'])
    print(f"{'OVERALL':<33} {tr_t:>7} {va_t:>6} {te_t:>6} {grand:>7} | {s_tr_t:3d}/{s_va_t:2d}/{s_te_t:2d}")
    print()


def build_dataset(
    root_dir,
    classes=None,
    feature='mel',
    augment=False,
    shuffle=False,
    batch_size=32,
    num_parallel=4,
    seed=42,
    csv_path=None,
    split_name=None
):
    if feature == 'mel':
        feature_func = audio_to_melspec
    elif feature == 'stft':
        feature_func = audio_to_stft
    elif feature == 'mfcc':
        feature_func = audio_to_mfcc_deltas
    else:
        raise ValueError(f"Unknown feature type: {feature}")

    if classes is None:
        classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    paths = []
    labels = []

    if csv_path and split_name:
        csv_files = load_splits_from_csv(csv_path, split_name)
        csv_files_set = set(csv_files)

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.wav', '.mp3')) and file in csv_files_set:
                    paths.append(os.path.join(cls_dir, file))
                    labels.append(class_to_idx[cls])
    else:
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.wav', '.mp3')):
                    paths.append(os.path.join(cls_dir, file))
                    labels.append(class_to_idx[cls])

    if not paths:
        raise ValueError(f"No audio files found in {root_dir}")

    def load_audio_pyfunc(path, label):
        audio, _ = librosa.load(path.numpy().decode('utf-8'),
                                 sr=SAMPLE_RATE,
                                 duration=3.0)
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
            lambda a, l: feature_func(a, l, augment=augment),
            [audio, label],
            [tf.float32, tf.int32]
        )
        spec.set_shape([TARGET_SPEC_HEIGHT, TARGET_SPEC_WIDTH, 3])
        label.set_shape([])
        return spec, label

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(process_audio, num_parallel_calls=num_parallel)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, classes


# ============================================================================
# Model
# ============================================================================

def create_model(model_name='mobilenetv3s', num_classes=10, use_pretrained=True,
                 dropout1=None, dropout2=None):
    weights = 'imagenet' if use_pretrained else None

    if model_name == 'mobilenetv3s':
        base = applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=weights,
            pooling='avg'
        )
    elif model_name == 'resnet50':
        base = applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=weights,
            pooling='avg'
        )
    elif model_name == 'vgg16':
        base = applications.VGG16(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=weights,
            pooling='avg'
        )
    elif model_name == 'efficientnetb0':
        base = applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=weights,
            pooling='avg'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    d1 = dropout1 if dropout1 is not None else 0.3
    d2 = dropout2 if dropout2 is not None else 0.2
    hidden_units = 512 if model_name == 'mobilenetv3s' else 256

    model = keras.Sequential([
        base,
        layers.Dropout(d1),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(d2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base


# ============================================================================
# Reporting utilities
# ============================================================================

def plot_history(history, outdir):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=150)
    plt.close()


def save_system_config(outdir):
    from importlib import metadata as _meta

    def pkg(name):
        try:
            return _meta.version(name)
        except Exception:
            return "unknown"

    lines = [
        "=" * 80, "SYSTEM CONFIGURATION", "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "Platform:", f"  {platform.platform()}",
        f"  Hostname: {socket.gethostname()}", "",
        "CPU:",
    ]
    if PSUTIL_AVAILABLE:
        lines += [f"  Physical cores: {psutil.cpu_count(logical=False)}",
                  f"  Logical cores: {psutil.cpu_count(logical=True)}"]
    else:
        lines.append("  (psutil not available)")

    lines += ["", "Memory:"]
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        lines += [f"  Total: {mem.total/(1024**3):.2f} GB",
                  f"  Available: {mem.available/(1024**3):.2f} GB",
                  f"  Used: {mem.used/(1024**3):.2f} GB ({mem.percent}%)"]
    else:
        lines.append("  (psutil not available)")

    lines += ["", "TensorFlow:",
              f"  Version: {tf.__version__}",
              f"  Keras: {pkg('keras')}"]
    try:
        gpus = tf.config.list_physical_devices('GPU')
        lines.append(f"  GPUs: {len(gpus)}")
    except Exception:
        lines.append("  GPUs: unknown")

    lines += ["", "Python:", f"  {platform.python_version()}",
              f"  NumPy: {np.__version__}",
              f"  Librosa: {librosa.__version__}", ""]

    try:
        with open(outdir / "system_config.txt", "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print("[WARN] Could not write system_config.txt:", e)


def save_hyperparameters(outdir, args, model, learning_rate, dropout1, dropout2,
                         early_stopping_patience, lr_plateau_patience,
                         monitor_metric, weight_decay, aug_mode):
    lines = [
        "=" * 80, "TRAINING HYPERPARAMETERS", "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "Model Architecture:",
        f"  Model: {args.model}",
        f"  Pretrained: {args.use_pretrained}",
        f"  Total parameters: {model.count_params():,}",
        f"  Trainable parameters: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}",
        f"  Dropout rates: [{dropout1}, {dropout2}]", "",
        "Feature Extraction:",
        f"  Feature type: {args.feature}",
        f"  Sample rate: {args.sample_rate} Hz",
        f"  N_FFT: {args.n_fft}",
        f"  N_Mels: {args.n_mels}",
        f"  Target size: {TARGET_SPEC_HEIGHT}x{TARGET_SPEC_WIDTH}", "",
        "Augmentation:",
        f"  Mode: {aug_mode}",
    ]
    if aug_mode == 'mixup':
        lines.append(f"  Mixup alpha: {args.mixup}")
    lines += [
        "",
        "Training:",
        f"  Batch size: {args.batch_size}",
        f"  Max epochs: {args.num_epochs}",
        f"  Learning rate: {learning_rate}",
        f"  Weight decay: {weight_decay}",
        f"  Gradient clipping: clipnorm=1.0", "",
        "Callbacks:",
        f"  Early stopping: patience={early_stopping_patience}, monitor={monitor_metric}",
        f"  ReduceLROnPlateau: patience={lr_plateau_patience}, factor=0.5", "",
        "Dataset:",
    ]
    if args.splits_csv:
        lines += [f"  Splits CSV: {args.splits_csv}",
                  f"  Dataset root: {args.dataset_root}"]
        import re
        m = re.search(r'(\d+)_(\d+)_(\d+)', Path(args.splits_csv).stem)
        if m:
            lines.append(f"  Split ratio: {m.group(1)}:{m.group(2)}:{m.group(3)}")
    else:
        lines += [f"  Train dir: {args.train_dir}",
                  f"  Val dir: {args.val_dir}",
                  f"  Test dir: {args.test_dir}"]
    lines += [f"  Num workers: {args.num_workers}", "",
              "Reproducibility:",
              f"  Random seed: {args.seed}",
              f"  Force CPU: {args.force_cpu}", ""]

    with open(outdir / "hyperparameters.txt", "w") as f:
        f.write("\n".join(lines))


def save_runtime_info(outdir, train_time, history, test_acc, test_loss):
    total_seconds = int(train_time * 60)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60

    lines = [
        "=" * 80, "RUNTIME INFORMATION", "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "Training Duration:",
        f"  Total time: {train_time:.2f} min ({train_time*60:.1f} s)",
        f"  Time delta: {hh:02d}:{mm:02d}:{ss:02d}",
        f"  Epochs completed: {len(history.history['loss'])}",
        f"  Time per epoch: {train_time/len(history.history['loss']):.2f} min", "",
        "Training History:",
        f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}",
        f"  Best val accuracy: {max(history.history['val_accuracy']):.4f} "
        f"(epoch {np.argmax(history.history['val_accuracy'])+1})", "",
        "Test Results:",
        f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)",
        f"  Test loss: {test_loss:.4f}", "",
        "System Resources (at completion):",
    ]
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        lines += [f"  Memory used: {mem.used/(1024**3):.2f} GB ({mem.percent}%)"]
    else:
        lines.append("  (psutil not available)")
    lines.append("")

    with open(outdir / "runtime_info.txt", "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mobilenetv3s',
                        choices=['mobilenetv3s', 'resnet50', 'vgg16', 'efficientnetb0'])
    parser.add_argument('--feature', default='mel', choices=['mel', 'stft', 'mfcc'])
    parser.add_argument('--train_dir', default=str(MYGARDENBIRD_16K / 'train'))
    parser.add_argument('--val_dir',   default=str(MYGARDENBIRD_16K / 'val'))
    parser.add_argument('--test_dir',  default=str(MYGARDENBIRD_16K / 'test'))
    parser.add_argument('--splits_csv', default=None, type=str)
    parser.add_argument('--dataset_root', default=str(MYGARDENBIRD_16K), type=str)
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate (auto-detected from first file if not specified)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='Use pretrained ImageNet weights (default: True). Set to False for from-scratch training.')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_mels', type=int, default=224)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--specaug', action='store_true',
                        help='Enable SpecAugment (time + frequency masking)')
    parser.add_argument('--mixup', type=float, default=None, metavar='ALPHA',
                        help='Enable Mixup augmentation with given alpha (e.g. 0.2). Overrides --specaug.')
    parser.add_argument('--dropout1', type=float, default=None)
    parser.add_argument('--dropout2', type=float, default=None)
    parser.add_argument('--force_cpu', action='store_true')

    args = parser.parse_args()

    # Determine augmentation mode: mixup > specaug > none
    if args.mixup is not None:
        aug_mode = 'mixup'
    elif args.specaug:
        aug_mode = 'specaug'
    else:
        aug_mode = 'noaug'

    import random
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    global PREPROCESS_FN
    if args.model == 'mobilenetv3s':
        PREPROCESS_FN = mobilenet_v3.preprocess_input
    elif args.model == 'resnet50':
        PREPROCESS_FN = resnet50.preprocess_input
    elif args.model == 'vgg16':
        PREPROCESS_FN = vgg16.preprocess_input
    elif args.model == 'efficientnetb0':
        PREPROCESS_FN = efficientnet.preprocess_input
    else:
        PREPROCESS_FN = lambda x: x

    if args.force_cpu:
        print("Forcing CPU mode (CUDA disabled)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')

    # Auto-detect sample rate from first audio file if not explicitly specified
    if args.sample_rate == 16000:  # Default value
        try:
            # Find first audio file
            import glob
            audio_files = glob.glob(os.path.join(args.dataset_root, "*.wav"))
            if audio_files:
                # Detect sample rate using librosa
                detected_sr = librosa.get_samplerate(audio_files[0])
                if detected_sr != 16000:
                    print(f"⚠️  Auto-detected sample rate: {detected_sr} Hz (from {os.path.basename(audio_files[0])})")
                    print(f"   Overriding default --sample_rate 16000 → {detected_sr}")
                    args.sample_rate = detected_sr
        except Exception as e:
            print(f"⚠️  Could not auto-detect sample rate: {e}")
            print(f"   Using default --sample_rate 16000")

    global SAMPLE_RATE, TARGET_LENGTH_SAMPLES, HOP_LENGTH, N_MELS, N_FFT, MIXUP_ALPHA
    SAMPLE_RATE = args.sample_rate
    TARGET_LENGTH_SAMPLES = int(SAMPLE_RATE * 3.0)
    N_MELS = args.n_mels
    N_FFT = args.n_fft
    HOP_LENGTH = TARGET_LENGTH_SAMPLES // 224
    if aug_mode == 'mixup':
        MIXUP_ALPHA = args.mixup

    print("=" * 80)
    print(f"MYGARDENBIRD CNN TRAINING - {args.feature.upper()} Features")
    print("=" * 80)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Model: {args.model}  |  Pretrained: {args.use_pretrained}")
    print(f"Feature: {args.feature}  |  Augmentation: {aug_mode}"
          + (f" (alpha={args.mixup})" if aug_mode == 'mixup' else ""))
    print(f"Seed: {args.seed}  |  Epochs: {args.num_epochs}")
    print()

    print("Building datasets...")

    # SpecAugment is applied inside the feature function (per sample).
    # Mixup is applied post-batch; train_ds is built without per-sample augmentation.
    augment_train = (aug_mode == 'specaug')

    if args.splits_csv:
        print(f"Using CSV splits: {args.splits_csv}")
        train_ds, classes = build_dataset(
            args.dataset_root, feature=args.feature,
            augment=augment_train, shuffle=True, batch_size=args.batch_size,
            seed=args.seed, csv_path=args.splits_csv, split_name='train'
        )
        val_ds, _ = build_dataset(
            args.dataset_root, feature=args.feature,
            augment=False, shuffle=False, batch_size=args.batch_size,
            csv_path=args.splits_csv, split_name='val'
        )
        test_ds, _ = build_dataset(
            args.dataset_root, feature=args.feature,
            augment=False, shuffle=False, batch_size=args.batch_size,
            csv_path=args.splits_csv, split_name='test'
        )
    else:
        train_ds, classes = build_dataset(
            args.train_dir, feature=args.feature,
            augment=augment_train, shuffle=True, batch_size=args.batch_size,
            seed=args.seed
        )
        val_ds, _ = build_dataset(
            args.val_dir, feature=args.feature,
            augment=False, shuffle=False, batch_size=args.batch_size
        )
        test_ds, _ = build_dataset(
            args.test_dir, feature=args.feature,
            augment=False, shuffle=False, batch_size=args.batch_size
        )

    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # Apply mixup post-batch: convert integer labels → one-hot, then mix
    if aug_mode == 'mixup':
        def to_one_hot(x, y):
            one_hot = tf.one_hot(tf.cast(y, tf.int32), num_classes)
            # Explicitly set shape to avoid unknown TensorShape errors
            one_hot.set_shape([None, num_classes])
            return x, one_hot

        train_ds = train_ds.map(to_one_hot)
        train_ds = train_ds.map(mixup_batch)
        # val_ds also needs one-hot labels so CategoricalCrossentropy can score it
        val_ds = val_ds.map(to_one_hot)

    if args.splits_csv:
        print_per_class_breakdown(args.splits_csv, args.dataset_root, classes)

    print("Creating model...")
    model, base = create_model(args.model, num_classes, args.use_pretrained,
                               dropout1=args.dropout1, dropout2=args.dropout2)

    if args.model == 'vgg16':
        learning_rate = args.learning_rate * 0.1
    else:
        learning_rate = args.learning_rate

    weight_decay = 1e-4 if args.model == 'mobilenetv3s' else 1e-5

    if platform.system().lower() == 'linux':
        optimizer = optimizers.AdamW(learning_rate=learning_rate,
                                     weight_decay=weight_decay, clipnorm=1.0)
    else:
        try:
            optimizer = optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0)
        except AttributeError:
            optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    # Loss: CategoricalCrossentropy for mixup (soft one-hot labels),
    #       SparseCategoricalCrossentropy otherwise (integer labels).
    if aug_mode == 'mixup':
        loss_fn = losses.CategoricalCrossentropy()
    else:
        loss_fn = losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    print(f"Model: {args.model}  |  Parameters: {model.count_params():,}")
    print()

    if args.model == 'mobilenetv3s':
        early_stopping_patience = 15
        lr_plateau_patience = 5
        monitor_metric = 'val_accuracy'
        monitor_mode = 'max'
    else:
        early_stopping_patience = 10
        lr_plateau_patience = 5
        monitor_metric = 'val_loss'
        monitor_mode = 'min'

    dropout1 = args.dropout1 if args.dropout1 is not None else 0.3
    dropout2 = args.dropout2 if args.dropout2 is not None else 0.2

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=lr_plateau_patience, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric, patience=early_stopping_patience,
            mode=monitor_mode, restore_best_weights=True, verbose=1
        )
    ]

    print("Training...")
    start_time = time.time()

    if args.use_pretrained and args.model in ['vgg16', 'mobilenetv3s']:
        warmup_epochs = 10 if args.model == 'mobilenetv3s' else 5
        print(f"Warmup: freezing base for {warmup_epochs} epochs...")
        base.trainable = False
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=warmup_epochs, verbose=1)

        print("Fine-tuning: unfreezing base...")
        base.trainable = True

        fine_tune_lr = 1e-4 if args.model == 'mobilenetv3s' else learning_rate * 0.1
        fine_tune_wd = 1e-5

        if platform.system().lower() == 'linux':
            fine_tune_optimizer = tf.keras.optimizers.AdamW(
                learning_rate=fine_tune_lr, weight_decay=fine_tune_wd, clipnorm=1.0)
        else:
            try:
                fine_tune_optimizer = tf.keras.optimizers.legacy.Adam(
                    learning_rate=fine_tune_lr, clipnorm=1.0)
            except AttributeError:
                fine_tune_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=fine_tune_lr, clipnorm=1.0)

        model.compile(optimizer=fine_tune_optimizer, loss=loss_fn, metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.num_epochs,
        callbacks=callbacks,
        verbose=1
    )

    train_time = (time.time() - start_time) / 60

    # Recompile with sparse loss before evaluating on integer-labelled test_ds
    if aug_mode == 'mixup':
        model.compile(
            optimizer=model.optimizer,
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)

    y_true, y_pred = [], []
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    # Build run name: always include aug tag
    pretrained_str = "pretrained" if args.use_pretrained else "scratch"
    tags = [aug_mode]
    if aug_mode == 'mixup':
        tags[0] = f"mixup{args.mixup}"
    if args.n_mels != 224:
        tags.append(f"mel{args.n_mels}")
    if args.dropout1 is not None or args.dropout2 is not None:
        d1 = args.dropout1 if args.dropout1 is not None else 0.3
        d2 = args.dropout2 if args.dropout2 is not None else 0.2
        tags.append(f"drop{d1}_{d2}")
    run_name = f"{args.model}_{args.feature}_{pretrained_str}_{'_'.join(tags)}_seed{args.seed}"

    output_dir = f"{args.output_dir}_{platform.platform().split('-')[0].lower()}"
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(output_path / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\nClassification Report:\n", report)
    plot_history(history, output_path)

    save_system_config(output_path)
    save_hyperparameters(output_path, args, model, learning_rate, dropout1, dropout2,
                         early_stopping_patience, lr_plateau_patience,
                         monitor_metric, weight_decay, aug_mode)
    save_runtime_info(output_path, train_time, history, test_acc, test_loss)

    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_time_minutes': train_time,
        'config': vars(args)
    }
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Feature: {args.feature}  |  Augmentation: {aug_mode}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Training Time: {train_time:.1f} min")
    print(f"Compute:       {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
