#!/usr/bin/env python3
"""
Train CNNs for seabird audio classification with multiple feature types.
Supports: Mel Spectrogram, STFT, MFCC+deltas
Models: MobileNetV3Small, VGG16, EfficientNetB0, ResNet50
"""

import os
import json
import platform
import time
from string import digits

import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


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

def build_dataset(
    root_dir,
    classes=None,
    feature='mel',
    augment=False,
    shuffle=False,
    batch_size=32,
    num_parallel=4,
    seed=42
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

    model = keras.Sequential([
        base,
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mobilenetv3s', choices=['mobilenetv3s', 'resnet50', 'vgg16', 'efficientnetb0'])
    parser.add_argument('--feature', default='mel', choices=['mel', 'stft', 'mfcc'],
                       help='Feature type: mel (spectrogram), stft, or mfcc (with deltas)')
    parser.add_argument('--train_dir', default='/Volumes/Evo/seabird16k/train')
    parser.add_argument('--val_dir', default='/Volumes/Evo/seabird16k/val')
    parser.add_argument('--test_dir', default='/Volumes/Evo/seabird16k/test')
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

    if args.model == 'vgg16':
        learning_rate = args.learning_rate * 0.1
        print(f"  Using reduced learning rate for VGG16: {learning_rate}")
    else:
        learning_rate = args.learning_rate

    if platform.system().lower() == 'linux':
        optimizer = optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
    else:
        optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    loss_fn = losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    print(f"✓ Model: {args.model}")
    print(f"  Parameters: {model.count_params():,}")
    print()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        )
    ]

    print("Training...")
    start_time = time.time()

    if args.use_pretrained and args.model in ['vgg16', 'mobilenetv3s']:
        print(f"{args.model} warmup: training classifier head with frozen base...")
        base.trainable = False

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=5,
            verbose=1
        )

        print(f"{args.model} fine-tuning: unfreezing top layers with lower learning rate...")

        if args.model == 'mobilenetv3s':
            # >>> FIX 2: Unfreeze top 50% of layers <<<
            n = int(len(base.layers) * 0.5)
            for layer in base.layers[:n]:
                layer.trainable = False
            for layer in base.layers[n:]:
                layer.trainable = True

            # >>> FIX 3: Use lower fine-tuning LR <<<
            fine_tune_learning_rate = 1e-5
        else:
            # VGG: unfreeze entire base
            base.trainable = True
            fine_tune_learning_rate = learning_rate * 0.1

        if platform.system().lower() == 'linux':
            fine_tune_optimizer = tf.keras.optimizers.AdamW(
                learning_rate=fine_tune_learning_rate,
                clipnorm=1.0
            )
        else:
            fine_tune_optimizer = tf.keras.optimizers.Adam(
                learning_rate=fine_tune_learning_rate,
                clipnorm=1.0
            )

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
    run_name = f"{args.model}_{args.feature}_{pretrained_str}_seed{args.seed}_{platform.system().lower()}"
    output_path = Path(args.output_dir) / run_name
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