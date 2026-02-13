#!/usr/bin/env python3
"""
Train CNNs for seabird audio classification with multiple feature types.

Supports: Mel Spectrogram, STFT, MFCC+deltas
Models: MobileNetV3Small, VGG16, EfficientNetB0, ResNet50

Usage Examples
--------------
# Run with all defaults (mobilenetv3s, mel, seed=42, batch_size=32, epochs=50):
python Stage9_train_seabird_multifeature.py --use_pretrained

# Select different models:
python Stage9_train_seabird_multifeature.py --model efficientnetb0 --use_pretrained
python Stage9_train_seabird_multifeature.py --model resnet50 --use_pretrained
python Stage9_train_seabird_multifeature.py --model vgg16 --use_pretrained

# Select different feature types:
python Stage9_train_seabird_multifeature.py --feature mel --use_pretrained   # Mel spectrogram (default)
python Stage9_train_seabird_multifeature.py --feature stft --use_pretrained  # STFT magnitude
python Stage9_train_seabird_multifeature.py --feature mfcc --use_pretrained  # MFCC with deltas

# Specify random seed for reproducibility:
python Stage9_train_seabird_multifeature.py --seed 42 --use_pretrained
python Stage9_train_seabird_multifeature.py --seed 100 --use_pretrained

# Use CSV-based splits (recommended):
python Stage9_train_seabird_multifeature.py \\
    --splits_csv ./seabird_splits_mip_75_10_15.csv \\
    --dataset_root /path/to/audio/files \\
    --use_pretrained

# Full example with all options:
python Stage9_train_seabird_multifeature.py \\
    --model efficientnetb0 \\
    --feature mel \\
    --splits_csv ./seabird_splits_mip_75_10_15.csv \\
    --dataset_root /Volumes/Evo/seabird16khz_flat \\
    --batch_size 32 \\
    --num_epochs 50 \\
    --learning_rate 0.001 \\
    --seed 42 \\
    --use_pretrained \\
    --output_dir ./results

# Force CPU training (when GPU is unavailable):
python Stage9_train_seabird_multifeature.py --force_cpu --use_pretrained

Defaults
--------
--model          mobilenetv3s
--feature        mel
--seed           42
--batch_size     32
--num_epochs     50
--learning_rate  0.001
--sample_rate    16000
--n_mels         224
--n_fft          2048
--num_workers    4
--output_dir     ./results
--dataset_root   /Volumes/Evo/seabird16khz_flat
"""

import os
import json
import platform
import time
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

# Optional psutil for detailed system info
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with 'pip install psutil' for detailed system info.")

from config import EXTRACTED_SEGS, SPLITS_DIR

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

# Force channels-last
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
PREPROCESS_FN = None   # Will be set in main() based on chosen model

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

    log_mel_spec = log_mel_spec[:224, :224]
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

def load_splits_from_csv(csv_path: str, split: str = 'train') -> List[str]:
    """
    Load file paths for a specific split from CSV file.

    Args:
        csv_path: Path to seabird_splits.csv
        split: 'train', 'val', or 'test'

    Returns:
        List of filenames for the specified split

    Note:
        Handles comment lines starting with '#' which are produced
        by Stage8 splitter scripts (MIP, genetic algorithm, simulated annealing).
    """
    import csv
    from io import StringIO

    files = []
    with open(csv_path, 'r') as f:
        # Filter out comment lines (Stage8 scripts add metadata comments)
        content = ''.join(line for line in f if not line.startswith('#'))

    reader = csv.DictReader(StringIO(content))
    for row in reader:
        if row['split'] == split:
            files.append(row['filename'])

    return files


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

    # If CSV path is provided, use it to filter files
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
        # Original behavior: use all files in directory
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

def create_model(model_name='mobilenetv3s', num_classes=10, use_pretrained=True):
    if use_pretrained:
        weights = 'imagenet'
    else:
        weights = None

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

    # MobileNetV3s was underfitting - reduce dropout and add more capacity
    if model_name == 'mobilenetv3s':
        dropout1 = 0.3
        dropout2 = 0.2
        hidden_units = 512  # More capacity for audio features
    else:
        dropout1 = 0.3
        dropout2 = 0.2
        hidden_units = 256

    model = keras.Sequential([
        base,
        layers.Dropout(dropout1),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(dropout2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base

def plot_history(history, outdir):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=150)
    plt.close()

def save_system_config(outdir):
    from datetime import datetime
    from importlib import metadata
    import platform, socket

    system_info = []

    def pkg(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return "unknown"

    system_info += [
        "="*80,
        "SYSTEM CONFIGURATION",
        "="*80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Platform Information:",
        f"  OS: {platform.system()} {platform.release()}",
        f"  Platform: {platform.platform()}",
        f"  Architecture: {platform.machine()}",
        f"  Processor: {platform.processor()}",
        f"  Hostname: {socket.gethostname()}",
        "",
        "CPU Information:",
    ]

    if PSUTIL_AVAILABLE:
        system_info += [
            f"  Physical cores: {psutil.cpu_count(logical=False)}",
            f"  Logical cores: {psutil.cpu_count(logical=True)}",
        ]
    else:
        system_info.append("  (psutil not available)")

    system_info += [
        "",
        "Memory Information:",
    ]

    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        system_info += [
            f"  Total: {mem.total / (1024**3):.2f} GB",
            f"  Available: {mem.available / (1024**3):.2f} GB",
            f"  Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)",
        ]
    else:
        system_info.append("  (psutil not available)")

    system_info += [
        "",
        "TensorFlow Configuration:",
        f"  TensorFlow version: {tf.__version__}",
        f"  Keras version: {pkg('keras')}",
    ]

    try:
        gpus = tf.config.list_physical_devices('GPU')
        system_info.append(f"  GPUs available: {len(gpus)}")
    except Exception:
        system_info.append("  GPUs available: unknown")

    system_info += [
        "",
        "Python Environment:",
        f"  Python version: {platform.python_version()}",
        f"  NumPy version: {np.__version__}",
        f"  Librosa version: {librosa.__version__}",
        "",
    ]

    try:
        with open(outdir / "system_config.txt", "w") as f:
            f.write("\n".join(system_info))
    except Exception as e:
        print("[WARN] Could not write system_config.txt:", e)

def save_hyperparameters(outdir, args, model, learning_rate, dropout1, dropout2,
                        early_stopping_patience, lr_plateau_patience,
                        monitor_metric, weight_decay, label_smoothing):
    """Save detailed hyperparameters to file."""
    hp_info = []

    hp_info.append("="*80)
    hp_info.append("TRAINING HYPERPARAMETERS")
    hp_info.append("="*80)
    hp_info.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hp_info.append("")

    # Model architecture
    hp_info.append("Model Architecture:")
    hp_info.append(f"  Model: {args.model}")
    hp_info.append(f"  Pretrained: {args.use_pretrained}")
    hp_info.append(f"  Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    hp_info.append(f"  Trainable parameters: {trainable_params:,}")
    hp_info.append(f"  Dropout rates: [{dropout1}, {dropout2}]")
    hp_info.append("")

    # Feature extraction
    hp_info.append("Feature Extraction:")
    hp_info.append(f"  Feature type: {args.feature}")
    hp_info.append(f"  Sample rate: {args.sample_rate} Hz")
    hp_info.append(f"  N_FFT: {args.n_fft}")
    hp_info.append(f"  N_Mels: {args.n_mels}")
    hp_info.append(f"  Target size: {TARGET_SPEC_HEIGHT}x{TARGET_SPEC_WIDTH}")
    hp_info.append("")

    # Training configuration
    hp_info.append("Training Configuration:")
    hp_info.append(f"  Batch size: {args.batch_size}")
    hp_info.append(f"  Max epochs: {args.num_epochs}")
    hp_info.append(f"  Learning rate: {learning_rate}")
    if platform.system().lower() == 'linux':
        hp_info.append(f"  Optimizer: AdamW (with weight_decay={weight_decay})")
    else:
        hp_info.append(f"  Optimizer: Adam.legacy (macOS/Metal optimized)")
    hp_info.append(f"  Gradient clipping: clipnorm=1.0")
    hp_info.append("")

    # Regularization
    hp_info.append("Regularization:")
    hp_info.append(f"  Dropout: [{dropout1}, {dropout2}]")
    hp_info.append(f"  Label smoothing: {label_smoothing} (not supported by SparseCategoricalCrossentropy)")
    hp_info.append(f"  Data augmentation: {'Yes (SpecAugment)' if args.feature in ['mel', 'stft', 'mfcc'] else 'No'}")
    hp_info.append("")

    # Callbacks
    hp_info.append("Callbacks:")
    hp_info.append(f"  Early stopping:")
    hp_info.append(f"    Monitor: {monitor_metric}")
    hp_info.append(f"    Patience: {early_stopping_patience}")
    hp_info.append(f"    Restore best weights: True")
    hp_info.append(f"  ReduceLROnPlateau:")
    hp_info.append(f"    Monitor: val_loss")
    hp_info.append(f"    Patience: {lr_plateau_patience}")
    hp_info.append(f"    Factor: 0.5")
    hp_info.append(f"    Min LR: 1e-7")
    hp_info.append("")

    # Dataset
    hp_info.append("Dataset:")
    hp_info.append(f"  Train dir: {args.train_dir}")
    hp_info.append(f"  Val dir: {args.val_dir}")
    hp_info.append(f"  Test dir: {args.test_dir}")
    hp_info.append(f"  Num workers: {args.num_workers}")
    hp_info.append("")

    # Reproducibility
    hp_info.append("Reproducibility:")
    hp_info.append(f"  Random seed: {args.seed}")
    hp_info.append(f"  Force CPU: {args.force_cpu}")
    hp_info.append("")

    with open(outdir / "hyperparameters.txt", "w") as f:
        f.write("\n".join(hp_info))

def save_runtime_info(outdir, train_time, history, test_acc, test_loss):
    """Save runtime information and training statistics."""
    runtime_info = []

    runtime_info.append("="*80)
    runtime_info.append("RUNTIME INFORMATION")
    runtime_info.append("="*80)
    runtime_info.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    runtime_info.append("")

    # Training duration
    runtime_info.append("Training Duration:")
    runtime_info.append(f"  Total time: {train_time:.2f} minutes ({train_time*60:.1f} seconds)")
    runtime_info.append(f"  Epochs completed: {len(history.history['loss'])}")
    runtime_info.append(f"  Time per epoch: {train_time / len(history.history['loss']):.2f} minutes")
    runtime_info.append("")

    # Training history
    runtime_info.append("Training History:")
    runtime_info.append(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    runtime_info.append(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    runtime_info.append(f"  Best train accuracy: {max(history.history['accuracy']):.4f} (epoch {np.argmax(history.history['accuracy'])+1})")
    runtime_info.append(f"  Best train loss: {min(history.history['loss']):.4f} (epoch {np.argmin(history.history['loss'])+1})")
    runtime_info.append("")

    # Validation history
    runtime_info.append("Validation History:")
    runtime_info.append(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    runtime_info.append(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
    runtime_info.append(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f} (epoch {np.argmax(history.history['val_accuracy'])+1})")
    runtime_info.append(f"  Best val loss: {min(history.history['val_loss']):.4f} (epoch {np.argmin(history.history['val_loss'])+1})")
    runtime_info.append("")

    # Overfitting check
    best_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
    final_epoch = len(history.history['val_accuracy'])
    overfitting_gap = max(history.history['val_accuracy']) - history.history['val_accuracy'][-1]
    runtime_info.append("Overfitting Analysis:")
    runtime_info.append(f"  Best validation accuracy at epoch: {best_val_acc_epoch}")
    runtime_info.append(f"  Training stopped at epoch: {final_epoch}")
    runtime_info.append(f"  Epochs after peak: {final_epoch - best_val_acc_epoch}")
    runtime_info.append(f"  Val accuracy degradation from peak: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
    train_val_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
    runtime_info.append(f"  Train-val accuracy gap (final): {train_val_gap:.4f} ({train_val_gap*100:.2f}%)")
    runtime_info.append("")

    # Test results
    runtime_info.append("Test Results:")
    runtime_info.append(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    runtime_info.append(f"  Test loss: {test_loss:.4f}")
    runtime_info.append("")

    # System resource usage at end
    runtime_info.append("System Resources (at completion):")
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        runtime_info.append(f"  Memory used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
        runtime_info.append(f"  Memory available: {mem.available / (1024**3):.2f} GB")
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            runtime_info.append(f"  CPU usage: {cpu_percent}%")
        except:
            pass
    else:
        runtime_info.append("  (psutil not available - install for resource monitoring)")
    runtime_info.append("")

    with open(outdir / "runtime_info.txt", "w") as f:
        f.write("\n".join(runtime_info))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mobilenetv3s', choices=['mobilenetv3s', 'resnet50', 'vgg16', 'efficientnetb0'])
    parser.add_argument('--feature', default='mel', choices=['mel', 'stft', 'mfcc'],
                       help='Feature type: mel (spectrogram), stft, or mfcc (with deltas)')
    parser.add_argument('--train_dir', default=str(EXTRACTED_SEGS / 'train'))
    parser.add_argument('--val_dir', default=str(EXTRACTED_SEGS / 'val'))
    parser.add_argument('--test_dir', default=str(EXTRACTED_SEGS / 'test'))
    parser.add_argument('--splits_csv', default=None, type=str,
                       help='Path to splits CSV file (if provided, uses CSV-based splits instead of directories). Can be relative to current directory.')
    parser.add_argument('--dataset_root', default=str(EXTRACTED_SEGS), type=str,
                       help=f'Root directory containing all audio files (used with --splits_csv). Default: {EXTRACTED_SEGS}')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_mels', type=int, default=224)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU only (disable all GPUs)')

    args = parser.parse_args()

    import random

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    global PREPROCESS_FN
    if args.model == 'mobilenetv3s':
        PREPROCESS_FN = mobilenet_v3.preprocess_input  # Maps [0,255] -> [-1,1]
    elif args.model == 'resnet50':
        PREPROCESS_FN = resnet50.preprocess_input
    elif args.model == 'vgg16':
        PREPROCESS_FN = vgg16.preprocess_input
    elif args.model == 'efficientnetb0':
        PREPROCESS_FN = efficientnet.preprocess_input
    else:
        PREPROCESS_FN = lambda x: x

    if args.force_cpu:
        print("⚠ Forcing CPU mode (CUDA disabled)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')

    global SAMPLE_RATE, TARGET_LENGTH_SAMPLES, HOP_LENGTH, N_MELS, N_FFT
    SAMPLE_RATE = args.sample_rate
    TARGET_LENGTH_SAMPLES = int(SAMPLE_RATE * 3.0)
    N_MELS = args.n_mels
    N_FFT = args.n_fft
    HOP_LENGTH = TARGET_LENGTH_SAMPLES // 224

    print("="*80)
    print(f"SEABIRD CNN TRAINING - {args.feature.upper()} Features")
    print("="*80)
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Force CPU: {args.force_cpu}")
    print(f"Model: {args.model}")
    print(f"Feature Type: {args.feature}")
    print(f"Sample Rate: {args.sample_rate} Hz")
    print(f"Pretrained: {args.use_pretrained}")
    print(f"Epochs: {args.num_epochs}")
    print()

    print("Building datasets...")

    # Check if using CSV-based splits
    if args.splits_csv:
        print(f"Using CSV splits from: {args.splits_csv}")
        print(f"Dataset root: {args.dataset_root}")

        train_ds, classes = build_dataset(
            args.dataset_root, feature=args.feature,
            augment=True, shuffle=True, batch_size=args.batch_size,
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
        # Original directory-based approach
        train_ds, classes = build_dataset(
            args.train_dir, feature=args.feature,
            augment=True, shuffle=True, batch_size=args.batch_size,
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

    print(f"✓ Datasets built with {args.feature} features")
    print(f"✓ Classes: {len(classes)}")
    print(classes)

    print("Creating model...")
    model, base = create_model(args.model, len(classes), args.use_pretrained)

    # Adjust learning rates per model
    if args.model == 'vgg16':
        learning_rate = args.learning_rate * 0.1
        print(f"  Using reduced learning rate for VGG16: {learning_rate}")
    elif args.model == 'mobilenetv3s':
        learning_rate = args.learning_rate  # Use full learning rate - was underfitting
        print(f"  Using full learning rate for MobileNetV3s: {learning_rate}")
    else:
        learning_rate = args.learning_rate

    # Add weight decay for regularization (especially important for MobileNet)
    weight_decay = 1e-4 if args.model == 'mobilenetv3s' else 1e-5

    if platform.system().lower() == 'linux':
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=1.0)
    else:
        # Use legacy Adam optimizer for M-series Macs (better performance)
        # Standard Adam doesn't have weight_decay, using clipnorm for stability
        try:
            optimizer = optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0)
        except AttributeError:
            optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    label_smoothing = 0.0
    loss_fn = losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    print(f"✓ Model: {args.model}")
    print(f"  Parameters: {model.count_params():,}")
    print()

    # MobileNetV3S needs longer training - was underfitting
    if args.model == 'mobilenetv3s':
        early_stopping_patience = 15  # More patience to find good solution
        lr_plateau_patience = 5
        monitor_metric = 'val_accuracy'
        monitor_mode = 'max'
        dropout1 = 0.3
        dropout2 = 0.2
        print(f"  MobileNet-specific: Early stopping patience={early_stopping_patience}, monitoring {monitor_metric}")
    else:
        early_stopping_patience = 10
        lr_plateau_patience = 5
        monitor_metric = 'val_loss'
        monitor_mode = 'min'
        dropout1 = 0.3
        dropout2 = 0.2

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=lr_plateau_patience, min_lr=1e-7, verbose=1
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
        print(f"{args.model} warmup: training classifier head with frozen base ({warmup_epochs} epochs)...")
        base.trainable = False

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            verbose=1
        )

        print(f"{args.model} fine-tuning: unfreezing top layers with lower learning rate...")

        if args.model == 'mobilenetv3s':
            # MobileNetV3S: unfreeze all layers for full adaptation to audio domain
            # ImageNet features don't transfer well to spectrograms - need full fine-tuning
            base.trainable = True

            fine_tune_learning_rate = 1e-4  # Higher LR for better adaptation
            fine_tune_weight_decay = 1e-5  # Less regularization - was underfitting
        else:
            # VGG: unfreeze entire base
            base.trainable = True
            fine_tune_learning_rate = learning_rate * 0.1
            fine_tune_weight_decay = 1e-5

        if platform.system().lower() == 'linux':
            fine_tune_optimizer = tf.keras.optimizers.AdamW(
                learning_rate=fine_tune_learning_rate,
                weight_decay=fine_tune_weight_decay,
                clipnorm=1.0
            )
        else:
            # Use legacy Adam optimizer for M-series Macs (better performance)
            try:
                fine_tune_optimizer = tf.keras.optimizers.legacy.Adam(
                    learning_rate=fine_tune_learning_rate,
                    clipnorm=1.0
                )
            except AttributeError:
                fine_tune_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=fine_tune_learning_rate,
                    clipnorm=1.0
                )

        # Recompile with fine-tune optimizer (keep same loss with label smoothing)
        model.compile(
            optimizer=fine_tune_optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.num_epochs,
        callbacks=callbacks,
        verbose=1
    )

    train_time = (time.time() - start_time) / 60

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)

    y_true = []
    y_pred = []

    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pretrained_str = "pretrained" if args.use_pretrained else "scratch"
    run_name = f"{args.model}_{args.feature}_{pretrained_str}_seed{args.seed}"
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

    # Save detailed configuration and runtime information
    print("\nSaving detailed configuration files...")
    save_system_config(output_path)
    save_hyperparameters(output_path, args, model, learning_rate, dropout1, dropout2,
                        early_stopping_patience, lr_plateau_patience,
                        monitor_metric, weight_decay, label_smoothing)
    save_runtime_info(output_path, train_time, history, test_acc, test_loss)
    print("✓ Saved: system_config.txt, hyperparameters.txt, runtime_info.txt")

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
    print(f"Feature: {args.feature.lower()}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Training Time: {train_time:.1f} min")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()