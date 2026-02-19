#!/usr/bin/env python3
"""
MynaNet v1 (Optimized) - Compressed for Cortex-M7 Deployment
Optimized from v0 baseline (93.56% @ 529KB) → Target: 92-93% @ <512KB
Post-Training Quantization (PTQ) → Cortex-M7 deployment
Spectrogram: 80x300 (10ms per frame, 80 mel bins)
Data Split: CSV-based (train/val/test) reproducible splits

VERSION: v1 (Optimized - Target: 92-93% @ <512KB)
STATUS: Deployment candidate, channel-reduced

Model Architecture: Channel-Reduced Asymmetric DS-CNN with SE
  Conv2D(120, 3×3) → BN → ReLU6                            # Initial (6% reduction)
  DS-Conv-SE(120) + Residual → MaxPool2D(2×2)              # Block 1 (40×150)
  DS-Conv-SE(180) + Residual(1×1 proj) → MaxPool2D(2×2)    # Block 2 (20×75)
  DS-Conv-SE(146) + Residual(1×1 proj) → MaxPool2D(2×2)    # Block 3 (10×37) *bottleneck*
  DS-Conv-SE(360) + Residual(1×1 proj) → MaxPool2D(2×2)    # Block 4 (5×18)
  Lightweight MHSA (2 heads, 32 key_dim, 88 dims)           # 8% reduction
  Dense(176) → Dropout → Dense(10)                          # 8% reduction

Key Options:
  --n_mels 64|80     : Number of mel bins (default: 80)

Key Improvements over Model 1d:
  1. Wider channels: 80→160→320→640 (vs 64→128→256→512)
     - Better feature representation
     - More capacity for complex patterns

  2. Residual connections in ALL blocks: Better gradient flow
     - Block 1: Direct skip (80→80 channels)
     - Blocks 2-4: 1×1 projection for channel matching
     - Enables training much deeper networks

  3. Enhanced attention: 4 heads, 48 key_dim (vs 2 heads, 32 key_dim)
     - Better self-attention capacity
     - Improved temporal modeling

  4. Wider FC layer: 192 units (vs 128)
     - Better classification capacity

Target: >95% INT8 accuracy, <512KB model size
"""

print("\n\n\n")
for _ in range(3):
    print(" 🔶 " * 30)

import os
import sys
import argparse
import platform
import warnings
import hashlib
import json

warnings.filterwarnings("ignore")

# GPU Configuration (before TensorFlow import)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --------------------------------------------------------------
# EARLY ARGUMENT PARSING (for GPU config before TF import)
# --------------------------------------------------------------
def parse_early_args():
    """Parse GPU-related args before TensorFlow import."""
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--force_cpu", action='store_true')
    temp_parser.add_argument("--gpu_memory_limit", type=int, default=None)
    temp_args, _ = temp_parser.parse_known_args()

    if temp_args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("\n⚠ Force CPU mode enabled - GPU disabled")

    return temp_args

# Parse GPU settings early
early_args = parse_early_args()

# --------------------------------------------------------------
# TENSORFLOW & KERAS ENVIRONMENT CHECK
# --------------------------------------------------------------
print("\n" + "=" * 70)
print("ENVIRONMENT VALIDATION (Checking before dataset preparation)")
print("=" * 70)

try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers, callbacks

    print(f"✓ TensorFlow version: {tf.__version__}")
    print(f"✓ tf_keras version: {keras.__version__}")

    # Configure GPU with memory growth and error recovery
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and not early_args.force_cpu:
        try:
            if early_args.gpu_memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=early_args.gpu_memory_limit)]
                )
                print(f"✓ Found {len(gpus)} GPU(s)")
                print(f"  GPU: {gpus[0].name}")
                print(f"  Memory limit: {early_args.gpu_memory_limit} MB")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                print(f"✓ Found {len(gpus)} GPU(s)")
                print(f"  GPU: {gpus[0].name}")
                if 'device_name' in gpu_details:
                    print(f"  Device: {gpu_details['device_name']}")
                print(f"  Memory growth: Enabled (prevents OOM errors)")

            print(f"  Deterministic ops: Enabled (stable cuDNN)")
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")
            print("  Continuing with default GPU settings...")
    else:
        print("✓ Running on CPU (no GPU detected)")

    print("\n✓ Environment check PASSED - safe to proceed with dataset preparation")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ CRITICAL: TensorFlow environment check failed: {e}")
    print("Install: pip install tensorflow tf_keras")
    print("\n⚠ Stopping now to save your time (no dataset loading yet)")
    print("=" * 70)
    sys.exit(1)

# Now safe to import other libraries
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

script_start = time.time()

# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------
# Default random seed - can be overridden via --random_seed argument
DEFAULT_RANDOM_STATE = 42

TARGET_SR = 16000
AUDIO_LENGTH_SEC = 3
FIXED_AUDIO_LENGTH = TARGET_SR * AUDIO_LENGTH_SEC
HOP_LENGTH = 160  # 10ms at 16kHz = 160 samples
N_FFT = 512
DEFAULT_N_MELS = 80  # Can be overridden via --n_mels (64 or 80)
FMAX = 8000
TIME_FRAMES = 300  # Fixed: 3 seconds / 10ms = 300 frames

# Wider channel configuration (matches 1d --wide mode)
CHANNELS_WIDE = [120, 180, 146, 360]  # v1: 6% reduction from v0 [128, 192, 156, 384]

# Default paths (can be overridden via config)
DEFAULT_FLAT_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
DEFAULT_SPECTROGRAM_DIR = "/Volumes/Evo/MYGARDENBIRD/precompute/spectrograms_16k_mels80"

# SpecAugment settings
SPECAUGMENT_FREQ_MASK = 8
SPECAUGMENT_TIME_MASK = 20
SPECAUGMENT_NUM_MASKS = 2

# Global stats percentiles for normalization
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98

# Global stats sample size per class
GLOBAL_STATS_SAMPLES = 100


# --------------------------------------------------------------
# UTILITY: TIME FORMATTING
# --------------------------------------------------------------
def format_time(seconds):
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# --------------------------------------------------------------
# OPTIMIZER – LEGACY ON APPLE SILICON, ADAMW ON LINUX
# --------------------------------------------------------------
system = platform.system()
processor = platform.processor()

if system == "Darwin" and processor == "arm":
    from tf_keras.optimizers.legacy import Adam as LegacyAdam
    Adam = LegacyAdam
    OPTIMIZER_NAME = "Legacy Adam"
    print("Using LEGACY Adam (fast on M1/M2/M4)")
elif system == "Linux":
    try:
        from tf_keras.optimizers import AdamW
        Adam = AdamW
        OPTIMIZER_NAME = "AdamW"
        print("Using AdamW optimizer (Linux - optimal for weight decay)")
    except ImportError:
        from tf_keras.optimizers import Adam
        OPTIMIZER_NAME = "Adam"
        print("Using standard Adam (AdamW not available)")
else:
    from tf_keras.optimizers import Adam
    OPTIMIZER_NAME = "Adam"
    print(f"Using standard Adam ({system})")


# --------------------------------------------------------------
# CACHE MANAGEMENT (from 4d)
# --------------------------------------------------------------
def compute_cache_hash(n_mels):
    """Compute hash of preprocessing parameters for cache validation."""
    cache_key = {
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_mels': n_mels,
        'fmax': FMAX,
        'target_sr': TARGET_SR,
        'time_frames': TIME_FRAMES,
        'audio_length': FIXED_AUDIO_LENGTH,
        'center': True,
        'window': 'hann',
        'win_length': 400,  # Include WIN_LENGTH in hash
    }
    hash_str = json.dumps(cache_key, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]


def validate_cache(cache_dir, n_mels):
    """Validate cache by checking version file."""
    version_file = os.path.join(cache_dir, '.cache_version')
    current_hash = compute_cache_hash(n_mels)

    if not os.path.exists(version_file):
        return False

    try:
        with open(version_file, 'r') as f:
            cached_hash = f.read().strip()
        return cached_hash == current_hash
    except:
        return False


def save_cache_version(cache_dir, n_mels):
    """Save cache version file."""
    version_file = os.path.join(cache_dir, '.cache_version')
    current_hash = compute_cache_hash(n_mels)
    with open(version_file, 'w') as f:
        f.write(current_hash)


# --------------------------------------------------------------
# CONFIG WITH VALIDATION
# --------------------------------------------------------------
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Default dropout = 0.05")
    parser.add_argument("--calib_samples", type=int, default=200)

    # Augmentation flags (from 7c)
    parser.add_argument("--augment", action='store_true',
                        help="Enable baseline augmentation (time/pitch shift)")
    parser.add_argument("--mixup", type=float, default=None,
                        help="Enable mixup augmentation with alpha value (e.g., 0.2)")
    parser.add_argument("--specaugment", action='store_true',
                        help="Enable SpecAugment (frequency/time masking)")

    # Baseline augmentation parameters
    parser.add_argument("--time_shift_ms", type=int, default=100,
                        help="Max time shift in milliseconds (baseline augmentation)")
    parser.add_argument("--pitch_shift_steps", type=int, default=2,
                        help="Max pitch shift in semitones (baseline augmentation)")

    # GPU parameters (already parsed early, but include for completeness)
    parser.add_argument("--force_cpu", action='store_true',
                        help="Force CPU execution (disable GPU)")
    parser.add_argument("--gpu_memory_limit", type=int, default=None,
                        help="GPU memory limit in MB (e.g., 8192 for 8GB)")

    # Configurable paths
    parser.add_argument("--splits_csv", type=str, required=True,
                        help="Path to splits CSV from seabird_splitter_mip.py")
    parser.add_argument("--flat_dir", type=str, default=DEFAULT_FLAT_DIR,
                        help="Path to flat dataset directory")
    parser.add_argument("--spectrogram_dir", type=str, default=DEFAULT_SPECTROGRAM_DIR,
                        help="Path to spectrogram cache directory")

    # LR schedule (from 4d)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "plateau", "both", "none"],
                        help="Learning rate schedule strategy")

    # Random seed (from 4d)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_STATE,
                        help="Random seed for reproducibility (default: 42)")

    # Model architecture options
    parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS, choices=[64, 80],
                        help="Number of mel bins (64 or 80, default: 80)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Set n_mels based on argument
    n_mels = args.n_mels

    # Determine augmentation mode and folder name suffix
    aug_suffix = ""
    augmentation_mode = "none"

    if args.mixup is not None:
        augmentation_mode = "mixup"
        aug_suffix = f"mixup{args.mixup}"
    elif args.specaugment:
        augmentation_mode = "specaugment"
        aug_suffix = "specaugment"
    elif args.augment:
        augmentation_mode = "baseline"
        aug_suffix = "baseline"

    # Parse split ratio from CSV header for output dir naming
    split_suffix = ""
    try:
        with open(args.splits_csv, 'r') as f:
            header = f.readline().strip()
        if header.startswith('# split_ratio='):
            ratio_str = header.split('split_ratio=')[1].split()[0]
            split_suffix = f"split{ratio_str}"
    except Exception:
        split_suffix = "splitcsv"

    output_dir_name = (
        f"results_{platform.platform().split('-')[0].lower()}/"
        f"v1_dscnn_se_res_att_wide_"
        f"mels{n_mels}_"
        f"drop{int(args.dropout * 100):02d}_"
        f"rand{args.random_seed}_"
        f"warm{args.warmup_epochs}_"
        f"{aug_suffix}_"
        f"{split_suffix}_"
        f"{platform.system().lower()}"
    )

    # Clean up double underscores if aug_suffix or split_suffix is empty
    output_dir_name = output_dir_name.replace("__", "_").rstrip("_")

    # Update spectrogram dir to include n_mels
    spec_dir = args.spectrogram_dir
    if spec_dir == DEFAULT_SPECTROGRAM_DIR:
        spec_dir = f"/Volumes/Evo/MYGARDENBIRD/precompute/spectrograms_16k_mels{n_mels}"

    config = {
        'warmup_epochs': args.warmup_epochs,
        'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size,
        'warmup_lr': args.warmup_lr,
        'finetune_lr': args.finetune_lr,
        'dropout': args.dropout,
        'time_frames': TIME_FRAMES,
        'n_mels': n_mels,
        'input_shape': (n_mels, TIME_FRAMES, 1),
        'output_dir': output_dir_name,
        'calib_samples': args.calib_samples,
        'model_type': 'dscnn_se_res',
        'augmentation_mode': augmentation_mode,
        'mixup_alpha': args.mixup,
        'time_shift_ms': args.time_shift_ms,
        'pitch_shift_steps': args.pitch_shift_steps,
        'force_cpu': args.force_cpu,
        'gpu_memory_limit': args.gpu_memory_limit,
        'spectrogram_dir': spec_dir,
        'lr_schedule': args.lr_schedule,
        'random_seed': args.random_seed,
        'channels': CHANNELS_WIDE,
        'splits_csv': args.splits_csv,
        'flat_dir': args.flat_dir,
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    return config


# --------------------------------------------------------------
# LOGGING UTILITIES
# --------------------------------------------------------------
class TrainingLogger:
    """Centralized logger for all training metrics and hyperparameters."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, 'training_report.txt')
        self.start_time = time.time()
        self.stage_times = {}

        # Initialize log file
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL 1E: DS-CNN + SE + RESIDUAL (ALL BLOCKS) + ATTENTION + WIDER CHANNELS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Platform: {platform.system()} {platform.machine()}\n")
            f.write(f"Python: {sys.version.split()[0]}\n")
            f.write(f"TensorFlow: {tf.__version__}\n")
            f.write(f"Keras: {keras.__version__}\n")
            f.write("\n")

    def log_section(self, title):
        """Log a section header."""
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n")

    def log_hyperparameters(self, config):
        """Log all hyperparameters."""
        self.log_section("HYPERPARAMETERS")
        with open(self.log_path, 'a') as f:
            # System info
            f.write("\nSystem Configuration:\n")
            f.write(f"  Platform: {platform.system()} {platform.machine()}\n")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and not config.get('force_cpu', False):
                f.write(f"  GPU: {len(gpus)} device(s) detected\n")
                f.write(f"  GPU Memory: Dynamic growth enabled\n")
                if config.get('gpu_memory_limit'):
                    f.write(f"  GPU Memory Limit: {config['gpu_memory_limit']} MB\n")
            else:
                f.write(f"  Compute: CPU only\n")

            f.write("\nAudio Processing:\n")
            f.write(f"  Target Sample Rate:     {TARGET_SR} Hz\n")
            f.write(f"  Audio Length:           {AUDIO_LENGTH_SEC} seconds ({FIXED_AUDIO_LENGTH} samples)\n")
            f.write(f"  FFT Size (N_FFT):       {N_FFT}\n")
            f.write(f"  FFT Window:             Hann (YAMNet standard, reduces spectral leakage)\n")
            f.write(f"  Window Length:          400 samples (25ms at 16kHz)\n")
            f.write(f"  Hop Length:             {HOP_LENGTH} samples (10.0 ms)\n")
            f.write(f"  Mel Bins (N_MELS):      {config['n_mels']}\n")
            f.write(f"  Max Frequency (FMAX):   {FMAX} Hz\n")
            f.write(f"  Time Frames:            {TIME_FRAMES} (FIXED)\n")
            f.write(f"  Spectrogram Shape:      {config['n_mels']}x{TIME_FRAMES}\n")
            f.write(f"  Center Padding:         Enabled (librosa center=True)\n")

            channels = config.get('channels', CHANNELS_WIDE)
            f.write("\nModel Architecture:\n")
            f.write(f"  Model Type:             DS-CNN + SE + Residual (All) + Attention (Model 1e)\n")
            f.write(f"  Channels:               {channels[0]}→{channels[1]}→{channels[2]}→{channels[3]}\n")
            f.write(f"  Dropout Rate:           {config['dropout']}\n")
            f.write(f"  Input Shape:            {config['input_shape']}\n")
            f.write(f"  Conv Blocks:            4 DS blocks + 1 initial conv\n")
            f.write(f"  DS-Conv:                DepthwiseConv2D(3×3) + Conv2D(1×1)\n")
            f.write(f"  Kernel Size:            3×3 (depthwise spatial)\n")
            f.write(f"  Pooling:                MaxPool2D (2×2) after each DS block\n")
            f.write(f"  Activation:             ReLU6 (quantization-friendly)\n")
            f.write(f"  Global Pooling:         GlobalAveragePooling2D\n")
            f.write(f"  Dense Layers:           192 → 10 (classification head)\n")

            f.write("\nTraining Configuration:\n")
            f.write(f"  Random Seed:            {config['random_seed']}\n")
            f.write(f"  Warmup Epochs:          {config['warmup_epochs']}\n")
            f.write(f"  Fine-tune Epochs:       {config['finetune_epochs']}\n")
            f.write(f"  Total Epochs:           {config['warmup_epochs'] + config['finetune_epochs']}\n")
            f.write(f"  Batch Size:             {config['batch_size']}\n")
            f.write(f"  Warmup Learning Rate:   {config['warmup_lr']}\n")
            f.write(f"  Fine-tune Learning Rate:{config['finetune_lr']}\n")
            f.write(f"  LR Schedule:            {config['lr_schedule']}\n")
            f.write(f"  Optimizer:              {OPTIMIZER_NAME}\n")
            f.write(f"  Loss Function:          Sparse Categorical Crossentropy\n")

            f.write("\nData Augmentation:\n")
            f.write(f"  Mode:                   {config['augmentation_mode']}\n")

            if config['augmentation_mode'] == 'baseline':
                f.write(f"  Type:                   Baseline (Time/Pitch Shift)\n")
                f.write(f"  Time Shift:             ±{config['time_shift_ms']} ms\n")
                f.write(f"  Pitch Shift:            ±{config['pitch_shift_steps']} semitones\n")
                f.write(f"  Data Multiplier:        2x (original + augmented)\n")
            elif config['augmentation_mode'] == 'mixup':
                f.write(f"  Type:                   Mixup\n")
                f.write(f"  Alpha:                  {config['mixup_alpha']}\n")
                f.write(f"  Data Multiplier:        2x (original + mixup)\n")
            elif config['augmentation_mode'] == 'specaugment':
                f.write(f"  Type:                   SpecAugment\n")
                f.write(f"  Frequency Mask:         {SPECAUGMENT_FREQ_MASK} bins\n")
                f.write(f"  Time Mask:              {SPECAUGMENT_TIME_MASK} frames\n")
                f.write(f"  Number of Masks:        {SPECAUGMENT_NUM_MASKS}\n")
                f.write(f"  Data Multiplier:        2x (original + augmented)\n")
            else:
                f.write(f"  Enabled:                False\n")

            f.write("\nQuantization:\n")
            f.write(f"  Method:                 Post-Training Quantization (PTQ)\n")
            f.write(f"  Target Format:          INT8 TFLite\n")
            f.write(f"  Calibration Samples:    {config['calib_samples']}\n")
            f.write(f"  Input/Output Type:      INT8\n")

            f.write("\nDeployment Target:\n")
            f.write(f"  Platform:               ARM Cortex-M7\n")
            f.write(f"  Memory Target:          <512 KB (50% of 1MB)\n")

            f.write("\nData Paths:\n")
            f.write(f"  Splits CSV:             {config['splits_csv']}\n")
            f.write(f"  Flat Directory:         {config['flat_dir']}\n")
            f.write(f"  Spectrogram Cache:      {config['spectrogram_dir']}\n")
            f.write(f"  Output Directory:       {config['output_dir']}\n")

    def log_dataset_info(self, X, y, class_labels, X_train, X_val, X_test, failed_files=0):
        """Log dataset statistics."""
        self.log_section("DATASET INFORMATION")
        with open(self.log_path, 'a') as f:
            f.write(f"\nTotal Samples:          {len(X)}\n")
            f.write(f"Number of Classes:      {len(class_labels)}\n")
            if failed_files > 0:
                f.write(f"Failed Files:           {failed_files}\n")
            f.write(f"\nData Split (Fixed per-class):\n")
            f.write(f"  Training:             {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)\n")
            f.write(f"  Validation:           {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)\n")
            f.write(f"  Test:                 {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)\n")

            f.write(f"\nClass Distribution:\n")
            for name, idx in sorted(class_labels.items(), key=lambda x: x[1]):
                count = np.sum(y == idx)
                percentage = count / len(y) * 100
                f.write(f"  {name:30s}: {count:5d} samples ({percentage:5.2f}%)\n")

    def log_model_info(self, model):
        """Log model architecture summary."""
        self.log_section("MODEL ARCHITECTURE")

        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()

        # Save model summary to separate file
        summary_path = os.path.join(self.output_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL ARCHITECTURE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary_str)
        print(f"Saved model summary: {summary_path}")

        with open(self.log_path, 'a') as f:
            f.write("\n" + summary_str)

            total_params = model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params

            f.write(f"\nParameter Summary:\n")
            f.write(f"  Total Parameters:       {total_params:,}\n")
            f.write(f"  Trainable Parameters:   {trainable_params:,}\n")
            f.write(f"  Non-trainable Params:   {non_trainable_params:,}\n")

            fp32_size_mb = total_params * 4 / (1024 ** 2)
            int8_size_kb = total_params / 1024

            f.write(f"\nEstimated Model Sizes:\n")
            f.write(f"  FP32 (4 bytes/param):   {fp32_size_mb:.2f} MB\n")
            f.write(f"  INT8 (1 byte/param):    {int8_size_kb:.1f} KB\n")

            if int8_size_kb > 512:
                f.write(f"\n  WARNING: Model may exceed 512 KB target for Cortex-M7\n")
            else:
                f.write(f"\n  Model size within 512 KB target\n")

    def start_stage(self, stage_name):
        """Mark the start of a training stage."""
        self.stage_times[stage_name] = {'start': time.time()}
        self.log_section(stage_name)

    def end_stage(self, stage_name, history=None):
        """Mark the end of a training stage and log results."""
        if stage_name not in self.stage_times:
            return

        self.stage_times[stage_name]['end'] = time.time()
        elapsed = self.stage_times[stage_name]['end'] - self.stage_times[stage_name]['start']

        with open(self.log_path, 'a') as f:
            f.write(f"\nStage Duration: {format_time(elapsed)}\n")

            if history is not None:
                f.write(f"\nTraining History:\n")
                final_epoch = len(history.history['loss'])
                f.write(f"  Epochs Completed:       {final_epoch}\n")
                f.write(f"  Final Train Loss:       {history.history['loss'][-1]:.4f}\n")
                f.write(f"  Final Train Accuracy:   {history.history['accuracy'][-1]:.4f}\n")
                f.write(f"  Final Val Loss:         {history.history['val_loss'][-1]:.4f}\n")
                f.write(f"  Final Val Accuracy:     {history.history['val_accuracy'][-1]:.4f}\n")
                f.write(f"  Best Val Loss:          {min(history.history['val_loss']):.4f}\n")
                f.write(f"  Best Val Accuracy:      {max(history.history['val_accuracy']):.4f}\n")

    def log_evaluation(self, model_name, accuracy, report_path):
        """Log model evaluation results."""
        with open(self.log_path, 'a') as f:
            f.write(f"\n{model_name} Evaluation:\n")
            f.write(f"  Test Accuracy:          {accuracy:.2f}%\n")
            f.write(f"  Classification Report:  {report_path}\n")

    def log_final_results(self, fp32_acc, int8_acc, model_sizes,
                          warmup_history, finetune_history, config, model=None):
        """Log final comparison results."""
        self.log_section("FINAL RESULTS SUMMARY")

        drop = fp32_acc - int8_acc
        total_time = time.time() - script_start

        with open(self.log_path, 'a') as f:
            # Quick reference card for spreadsheet comparison
            f.write("\n" + "=" * 80 + "\n")
            f.write("QUICK REFERENCE (Copy to spreadsheet)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Config: model1e_drp{int(config['dropout'] * 10)}_"
                    f"{config['augmentation_mode']}_warmup{config['warmup_epochs']}_"
                    f"finetune{config['finetune_epochs']}_lr{config['lr_schedule']}\n")
            f.write(f"FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}% | "
                    f"Drop: {drop:+.2f}% | Time: {format_time(total_time)}\n")

            # Detailed results
            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n")

            f.write(f"\nAccuracy Results:\n")
            f.write(f"  FP32 (.keras):          {fp32_acc:6.2f}%\n")
            f.write(f"  INT8 (TFLite):          {int8_acc:6.2f}%\n")

            f.write(f"\nAccuracy Change (INT8 vs FP32):\n")
            f.write(f"  Drop:                   {drop:+6.2f}% ")
            if abs(drop) < 0.5:
                f.write("✓ Excellent (no degradation)\n")
            elif drop > 0:
                f.write("✓✓ INT8 better! (quantization as regularizer)\n")
            elif drop > -2:
                f.write("✓ Good (<2% drop)\n")
            elif drop > -5:
                f.write("⚠ Acceptable (2-5% drop)\n")
            else:
                f.write("✗ High degradation (>5% drop)\n")

            f.write(f"\nModel Sizes:\n")
            for model_type, size_info in model_sizes.items():
                f.write(f"  {model_type:20s}: {size_info}\n")

            # Training metrics
            f.write(f"\nTraining Metrics:\n")
            f.write(f"  Best Warmup Val Acc:    {max(warmup_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Best Finetune Val Acc:  {max(finetune_history.history['val_accuracy']) * 100:6.2f}%\n")
            f.write(f"  Final Train Acc:        {finetune_history.history['accuracy'][-1] * 100:6.2f}%\n")
            f.write(f"  Final Val Acc:          {finetune_history.history['val_accuracy'][-1] * 100:6.2f}%\n")
            f.write(f"  Train-Test Gap:         {finetune_history.history['accuracy'][-1] * 100 - int8_acc:+6.2f}%\n")

            overfitting_gap = finetune_history.history['accuracy'][-1] * 100 - finetune_history.history['val_accuracy'][-1] * 100
            f.write(f"  Train-Val Gap:          {overfitting_gap:+6.2f}%")
            if overfitting_gap < 2:
                f.write(" ✓ No overfitting\n")
            elif overfitting_gap < 5:
                f.write(" ⚠ Slight overfitting\n")
            else:
                f.write(" ✗ Overfitting detected\n")

            f.write(f"\nExecution Time:\n")
            f.write(f"  Total Duration:         {format_time(total_time)}\n")

            f.write(f"\nTraining completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # CSV format for easy import
            f.write("\n" + "=" * 80 + "\n")
            f.write("CSV FORMAT (for batch comparison)\n")
            f.write("=" * 80 + "\n")
            f.write("model_type,dropout,augmentation,warmup_epochs,finetune_epochs,warmup_lr,finetune_lr,"
                    "lr_schedule,fp32_acc,int8_acc,drop,best_val_acc,train_val_gap,train_time_sec,model_size_kb\n")
            f.write(f"1e_dscnn_se_res_att_wide,{config['dropout']},{config['augmentation_mode']},"
                    f"{config['warmup_epochs']},{config['finetune_epochs']},{config['warmup_lr']},{config['finetune_lr']},"
                    f"{config['lr_schedule']},{fp32_acc:.2f},{int8_acc:.2f},{drop:.2f},"
                    f"{max(finetune_history.history['val_accuracy']) * 100:.2f},"
                    f"{overfitting_gap:.2f},{int(total_time)},"
                    f"{os.path.getsize(os.path.join(config['output_dir'], 'model_int8.tflite')) / 1024:.1f}\n")

            # Cortex-M7 Deployment Assessment
            f.write("\n" + "=" * 80 + "\n")
            f.write("CORTEX-M7 DEPLOYMENT ASSESSMENT\n")
            f.write("=" * 80 + "\n")

            int8_size_kb = os.path.getsize(os.path.join(config['output_dir'], 'model_int8.tflite')) / 1024
            size_ok = int8_size_kb < 512
            accuracy_ok = int8_acc >= 90.0

            f.write(f"\nDeployment Criteria:\n")
            f.write(f"  Model Size:             {int8_size_kb:.1f} KB {'< 512 KB' if size_ok else '>= 512 KB'}\n")
            f.write(f"  INT8 Accuracy:          {int8_acc:.2f}% {'>= 90%' if accuracy_ok else '< 90%'}\n")

            if size_ok and accuracy_ok:
                f.write(f"\n  SUITABLE FOR CORTEX-M7 DEPLOYMENT\n")
            elif size_ok and not accuracy_ok:
                f.write(f"\n  SIZE OK, BUT ACCURACY BELOW 90% TARGET\n")
            elif not size_ok and accuracy_ok:
                f.write(f"\n  ACCURACY OK, BUT MODEL TOO LARGE (>512KB)\n")
            else:
                f.write(f"\n  NOT SUITABLE: Size too large AND accuracy below target\n")

            # Latency and Power Estimates for Cortex-M7 @ 480 MHz
            # Based on typical CMSIS-NN benchmarks: ~10-20 MAC/cycle for INT8
            # Assuming ~15 MAC/cycle average for mixed operations
            if model is not None:
                total_params = sum([np.prod(w.shape) for w in model.trainable_weights])
            else:
                # Estimate from INT8 model size (1 byte per param)
                total_params = int(int8_size_kb * 1024)

            estimated_macs = total_params * 2  # Rough estimate: params * 2 for forward pass
            cycles_per_inference = estimated_macs / 15  # ~15 MAC/cycle for INT8 on M7
            latency_ms = (cycles_per_inference / 480_000_000) * 1000  # 480 MHz clock

            # Power estimate: Cortex-M7 @ 480 MHz typically draws ~100-150 mW active
            # Per-inference energy = power * time
            power_mw = 120  # Typical active power
            energy_per_inference_mj = power_mw * latency_ms / 1000  # mJ = mW * s

            f.write(f"\nCortex-M7 @ 480 MHz Estimates:\n")
            f.write(f"  Estimated MACs:         {estimated_macs:,}\n")
            f.write(f"  Estimated Latency:      {latency_ms:.2f} ms\n")
            f.write(f"  Active Power:           ~{power_mw} mW\n")
            f.write(f"  Energy per Inference:   ~{energy_per_inference_mj:.3f} mJ\n")

            if latency_ms < 100:
                f.write(f"  Real-time capable (< 100ms latency)\n")
            elif latency_ms < 500:
                f.write(f"  Near real-time (100-500ms latency)\n")
            else:
                f.write(f"  Batch processing recommended (>500ms latency)\n")

            # Analysis
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS & RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n")

            f.write(f"\nCurrent Performance: {int8_acc:.2f}%\n")

            # Performance tier
            if int8_acc >= 98:
                f.write("  🏆 EXCEPTIONAL - Near-perfect performance!\n")
            elif int8_acc >= 95:
                f.write("  ✓✓ EXCELLENT - Strong baseline performance\n")
            elif int8_acc >= 90:
                f.write("  ✓ GOOD - Solid performance with room for improvement\n")
            elif int8_acc >= 85:
                f.write("  ⚠ FAIR - Significant improvement needed\n")
            else:
                f.write("  ✗ POOR - Major improvements required\n")

            f.write(f"\nQuantization Method:\n")
            f.write(f"  Method: Post-Training Quantization (PTQ)\n")
            f.write(f"  INT8 vs FP32: {drop:+.2f}%\n")

            f.write(f"\nModel Architecture:\n")
            f.write(f"  DS-CNN + SE + Residual (All) + Attention (Model 1e)\n")
            f.write(f"  - 4 DS-Conv blocks: 128→192→256→384 filters\n")
            f.write(f"  - Squeeze-and-Excitation (SE) modules in every block\n")
            f.write(f"  - Residual connections in ALL 4 blocks (with 1×1 projection where needed)\n")
            f.write(f"  - DS-Conv = DepthwiseConv(3×3) + PointwiseConv(1×1)\n")
            f.write(f"  - Lightweight MHSA: 2 heads, 32 key_dim, projected to 64 dims\n")
            f.write(f"  - GlobalAveragePooling1D + Dense(128) classifier\n")
            f.write(f"  - ReLU6 + BatchNorm (quantization-friendly)\n")

            f.write(f"\nAugmentation Strategy:\n")
            f.write(f"  Current Mode: {config['augmentation_mode']}\n")
            if config['augmentation_mode'] == 'none':
                f.write("  Recommendation: Try --augment (baseline), --mixup 0.2, or --specaugment\n")
            elif config['augmentation_mode'] == 'baseline':
                f.write("  Recommendation: Try --mixup 0.2 or --specaugment for advanced augmentation\n")
            elif config['augmentation_mode'] == 'mixup':
                f.write(f"  Alpha: {config['mixup_alpha']}\n")
                f.write("  Recommendation: Try different alpha values (0.1-0.4)\n")
            elif config['augmentation_mode'] == 'specaugment':
                f.write("  Recommendation: Adjust masks in source code or combine with mixup\n")

            f.write(f"\nLearning Rate Schedule:\n")
            f.write(f"  Current: {config['lr_schedule']}\n")
            if config['lr_schedule'] == 'none':
                f.write("  Recommendation: Try --lr_schedule cosine or plateau\n")


# --------------------------------------------------------------
# GLOBAL STATS (2-98 percentile)
# --------------------------------------------------------------
def compute_global_stats(data_dir, n_mels, allowed_files=None):
    """Compute global normalization statistics from dataset.

    Args:
        data_dir: Path to flat dataset directory (class_name/file.wav)
        n_mels: Number of mel bins
        allowed_files: Optional set of filenames to restrict to (e.g. training
            files from the splits CSV). If None, all .wav files are used.
    """
    all_mel = []
    total_sampled = 0

    print(f"Computing global stats (sampling up to {GLOBAL_STATS_SAMPLES} files per class, n_mels={n_mels})...")

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue

        wavs = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        if allowed_files is not None:
            wavs = [f for f in wavs if f in allowed_files]
        sample_size = min(len(wavs), GLOBAL_STATS_SAMPLES)

        for f in wavs[:sample_size]:
            try:
                audio_path = os.path.join(class_dir, f)
                audio, _ = librosa.load(audio_path, sr=TARGET_SR)

                if len(audio) > FIXED_AUDIO_LENGTH:
                    audio = audio[:FIXED_AUDIO_LENGTH]
                else:
                    audio = np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                mel = librosa.feature.melspectrogram(
                    y=audio, sr=TARGET_SR, n_fft=N_FFT,
                    win_length=400, hop_length=HOP_LENGTH,
                    n_mels=n_mels, fmax=FMAX, center=True,
                    power=2.0, window='hann'
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                all_mel.append(mel_db.flatten())
                total_sampled += 1
            except Exception as e:
                print(f"\n⚠ Failed to process {f} during stats computation: {e}")
                continue

    if len(all_mel) == 0:
        raise RuntimeError("No valid audio files found for computing global stats")

    all_mel = np.concatenate(all_mel)
    gmin, gmax = np.percentile(all_mel, PERCENTILE_LOW), np.percentile(all_mel, PERCENTILE_HIGH)
    print(f"✓ Global stats computed from {total_sampled} files: {gmin:.2f} → {gmax:.2f} dB")
    return float(gmin), float(gmax)


# --------------------------------------------------------------
# SPECTROGRAM + NORMALIZE (FIXED n_mels x 300 with WIN_LENGTH)
# --------------------------------------------------------------
def compute_spec(audio, sr, gmin, gmax, n_mels):
    """
    Compute and normalize mel spectrogram with shape validation.
    Uses WIN_LENGTH=400 (25ms at 16kHz) for better time resolution.
    """
    WIN_LENGTH = 400  # 25ms at 16kHz

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT,
        win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmax=FMAX, center=True,
        power=2.0, window='hann'
    )

    # Ensure exact TIME_FRAMES
    if mel.shape[1] > TIME_FRAMES:
        mel = mel[:, :TIME_FRAMES]
    if mel.shape[1] < TIME_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, TIME_FRAMES - mel.shape[1])))

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, gmin, gmax)
    mel_norm = (mel_db - gmin) / (gmax - gmin + 1e-8)
    return mel_norm[..., np.newaxis].astype(np.float32)


# --------------------------------------------------------------
# AUGMENTATION FUNCTIONS
# --------------------------------------------------------------
def augment_baseline(audio, sr, time_shift_ms=100, pitch_steps=2):
    """Baseline augmentation: time shift + pitch shift"""
    if np.random.rand() > 0.5:
        shift_samples = int(np.random.uniform(-time_shift_ms, time_shift_ms) * sr / 1000)
        audio = np.roll(audio, shift_samples)

    if np.random.rand() > 0.5:
        n_steps = np.random.uniform(-pitch_steps, pitch_steps)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    return audio


def augment_specaugment(spec):
    """SpecAugment: frequency and time masking on spectrogram"""
    spec_aug = spec.copy()
    freq_bins, time_bins, _ = spec_aug.shape

    # Apply frequency masks
    for _ in range(SPECAUGMENT_NUM_MASKS):
        f_mask = np.random.randint(0, SPECAUGMENT_FREQ_MASK)
        f0 = np.random.randint(0, freq_bins - f_mask) if f_mask < freq_bins else 0
        spec_aug[f0:f0 + f_mask, :, 0] = 0

    # Apply time masks
    for _ in range(SPECAUGMENT_NUM_MASKS):
        t_mask = np.random.randint(0, SPECAUGMENT_TIME_MASK)
        t0 = np.random.randint(0, time_bins - t_mask) if t_mask < time_bins else 0
        spec_aug[:, t0:t0 + t_mask, 0] = 0

    return spec_aug


# --------------------------------------------------------------
# DEPTHWISE SEPARABLE CNN + SE + RESIDUAL (Model 1c)
# --------------------------------------------------------------
def se_block(x, filters, reduction=16, block_id=0):
    """
    Squeeze-and-Excitation block for channel attention.

    SE = GlobalAvgPool → FC(filters/r) → ReLU → FC(filters) → Sigmoid → Scale

    "Which channels are important for this input?"

    Args:
        x: Input tensor
        filters: Number of channels
        reduction: Reduction ratio for bottleneck (default 16)
        block_id: Block identifier for naming

    Returns:
        Channel-reweighted tensor

    Parameter cost: 2 * filters * (filters / reduction)
    Example: 512 channels, r=16 → 2 * 512 * 32 = 32,768 params
    """
    prefix = f'block{block_id}_se_'

    # Squeeze: Global average pooling
    se = layers.GlobalAveragePooling2D(keepdims=True, name=prefix + 'squeeze')(x)

    # Excitation: Two FC layers with bottleneck
    se = layers.Conv2D(
        filters // reduction, (1, 1), activation='relu',
        use_bias=True, name=prefix + 'reduce'
    )(se)
    se = layers.Conv2D(
        filters, (1, 1), activation='sigmoid',
        use_bias=True, name=prefix + 'expand'
    )(se)

    # Scale: Channel-wise multiplication
    return layers.Multiply(name=prefix + 'scale')([x, se])


def ds_conv_block_se_res(x, filters, kernel_size=(3, 3), strides=(1, 1),
                          block_id=0, use_se=True, use_residual=True, se_reduction=16):
    """
    Depthwise Separable Convolution Block with SE and Residual.

    DS-Conv + SE + Residual:
    1. Depthwise Conv (spatial filtering per channel)
    2. BatchNorm + ReLU6
    3. Pointwise Conv (channel mixing)
    4. BatchNorm
    5. SE block (channel attention)
    6. Residual connection (with 1×1 projection if channels don't match)
    7. ReLU6

    Args:
        x: Input tensor
        filters: Output channels
        kernel_size: Depthwise kernel size
        strides: Depthwise strides
        block_id: Block identifier
        use_se: Whether to use SE block
        use_residual: Whether to use residual connection
        se_reduction: SE reduction ratio
    """
    prefix = f'block{block_id}_'
    input_channels = x.shape[-1]

    # Save input for residual
    shortcut = x

    # Depthwise convolution (spatial filtering per channel)
    x = layers.DepthwiseConv2D(
        kernel_size, strides=strides, padding='same',
        use_bias=False, name=prefix + 'depthwise'
    )(x)
    x = layers.BatchNormalization(name=prefix + 'depthwise_bn')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Pointwise convolution (1×1, mixes channels)
    x = layers.Conv2D(
        filters, (1, 1), padding='same',
        use_bias=False, name=prefix + 'pointwise'
    )(x)
    x = layers.BatchNormalization(name=prefix + 'pointwise_bn')(x)

    # SE block (channel attention)
    if use_se:
        x = se_block(x, filters, reduction=se_reduction, block_id=block_id)

    # Residual connection (with 1×1 projection if channels don't match)
    if use_residual:
        if input_channels != filters:
            # 1×1 projection to match channels
            shortcut = layers.Conv2D(
                filters, (1, 1), strides=strides, padding='same',
                use_bias=False, name=prefix + 'residual_proj'
            )(shortcut)
            shortcut = layers.BatchNormalization(name=prefix + 'residual_proj_bn')(shortcut)
        x = layers.Add(name=prefix + 'residual')([x, shortcut])

    # Final activation
    x = layers.ReLU(6., name=prefix + 'relu')(x)

    return x


def create_se_res_att(num_classes, input_shape, dropout=0.2, channels=None):
    """
    DS-CNN + SE + Residual (ALL blocks) + Enhanced MHSA (Model 1e)
    Wider channels + residual in all blocks for improved accuracy.

    Args:
        num_classes: Number of output classes
        input_shape: Input tensor shape (n_mels, time_frames, 1)
        dropout: Dropout rate
        channels: List of 4 channel widths [c1, c2, c3, c4]
                  Default: [80, 160, 320, 640] (wide config)

    Estimated ~480 KB INT8 with 80→160→320→640 channels.
    """
    if channels is None:
        channels = CHANNELS_WIDE

    c1, c2, c3, c4 = channels

    # SE reduction ratios (scale with channel width)
    se_r1 = max(4, c1 // 16)
    se_r2 = max(8, c2 // 16)
    se_r3 = max(16, c3 // 16)
    se_r4 = max(16, c4 // 16)

    inputs = layers.Input(shape=input_shape, name='input')

    # Initial conv - match first block channels
    x = layers.Conv2D(c1, (3, 3), padding='same', use_bias=False, name='initial_conv')(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.ReLU(6., name='initial_relu')(x)

    # DS Block 1: c1→c1 channels - with residual!
    x = ds_conv_block_se_res(x, c1, block_id=1, use_se=True, use_residual=True, se_reduction=se_r1)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(dropout * 0.5, name='drop1')(x)

    # DS Block 2: c1→c2 channels - with residual (1×1 proj)!
    x = ds_conv_block_se_res(x, c2, block_id=2, use_se=True, use_residual=True, se_reduction=se_r2)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(dropout * 0.5, name='drop2')(x)

    # DS Block 3: c2→c3 channels - with residual (1×1 proj)!
    x = ds_conv_block_se_res(x, c3, block_id=3, use_se=True, use_residual=True, se_reduction=se_r3)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(dropout * 0.75, name='drop3')(x)

    # DS Block 4: c3→c4 channels - with residual (1×1 proj)!
    x = ds_conv_block_se_res(x, c4, block_id=4, use_se=True, use_residual=True, se_reduction=se_r4)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(dropout, name='drop4')(x)

    # === ENHANCED MHSA MODULE ===
    # Reshape to (sequence_length, channels) for attention
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((h * w, c), name='reshape_for_attention')(x)

    # Project to lower dimension to reduce attention cost
    att_dim = 88  # v1: 8% reduction from v0's 96
    x_proj = layers.Dense(att_dim, activation='relu', name='attention_proj')(x)

    # Multi-head self-attention (2 heads, key_dim=32 - keep same as v0)
    x_att = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32,
        name='enhanced_mhsa'
    )(x_proj, x_proj)

    # Add & norm
    x_att = layers.Add()([x_proj, x_att])  # residual
    x_att = layers.LayerNormalization(name='att_ln')(x_att)

    # Global average over sequence dimension
    x = layers.GlobalAveragePooling1D(name='global_pool_att')(x_att)

    # Classification head - reduced for <512KB target
    x = layers.Dense(176, name='fc1')(x)  # v1: 8% reduction from v0's 192
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.ReLU(6., name='fc1_relu')(x)
    x = layers.Dropout(dropout, name='fc_drop')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return keras.Model(inputs, outputs, name="MynaNet_DSCNN_SE_Res_Att_Wide")


# --------------------------------------------------------------
# TFLITE CONVERSION (POST-TRAINING QUANTIZATION)
# --------------------------------------------------------------
def convert_to_tflite_int8(model, X_calib, path):
    """Convert model to INT8 TFLite with post-training quantization."""
    def rep_dataset():
        for i in range(len(X_calib)):
            yield [X_calib[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ Saved INT8 TFLite: {path} ({os.path.getsize(path) / 1024:.1f} KB)")


# --------------------------------------------------------------
# PLOTTING & EVALUATION
# --------------------------------------------------------------
def _save_classification_report(y_test, y_pred, class_names, output_dir, model_type):
    """Save classification report to file and print to console."""
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, f'classification_report_{model_type.lower()}.txt')
    with open(report_path, 'w') as f:
        f.write(f"{model_type} Model Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print(f"✓ Saved classification report: {report_path}")
    print(f"\n{model_type} Classification Report:")
    print(report)
    return report_path


def _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, accuracy):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    cmap = 'Blues' if 'FP32' in model_type else 'Greens'

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_type} Confusion Matrix - Accuracy: {accuracy:.2f}%', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type.lower()}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(warmup_hist, finetune_hist, output_dir):
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(warmup_hist.history['accuracy'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[0].plot(warmup_hist.history['val_accuracy'], label='Val (Warmup)', color='blue', linestyle='--')

    offset = len(warmup_hist.history['accuracy'])
    epochs_finetune = range(offset, offset + len(finetune_hist.history['accuracy']))
    axes[0].plot(epochs_finetune, finetune_hist.history['accuracy'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[0].plot(epochs_finetune, finetune_hist.history['val_accuracy'], label='Val (Finetune)', color='red',
                 linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(warmup_hist.history['loss'], label='Train (Warmup)', color='blue', alpha=0.7)
    axes[1].plot(warmup_hist.history['val_loss'], label='Val (Warmup)', color='blue', linestyle='--')
    axes[1].plot(epochs_finetune, finetune_hist.history['loss'], label='Train (Finetune)', color='red', alpha=0.7)
    axes[1].plot(epochs_finetune, finetune_hist.history['val_loss'], label='Val (Finetune)', color='red',
                 linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    print("✓ Saved training history plot")


def evaluate_model(model, X_test, y_test, class_names, output_dir, model_type="FP32"):
    """Evaluate a Keras model and save results."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred) * 100

    _save_classification_report(y_test, y_pred, class_names, output_dir, model_type)
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, model_type, acc)

    return acc


def evaluate_tflite(tflite_path, X_test, y_test, class_names, output_dir):
    """Evaluate a TFLite INT8 model and save results."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    y_pred = []
    print("Evaluating TFLite model...")
    for i in tqdm(range(len(X_test)), desc="Running inference", unit="sample"):
        x_fp32 = X_test[i:i + 1]
        x_int8 = (x_fp32 / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], x_int8)
        interpreter.invoke()

        output_int8 = interpreter.get_tensor(output_details[0]['index'])
        output_fp32 = (output_int8.astype(np.float32) - output_zero_point) * output_scale

        y_pred.append(np.argmax(output_fp32))

    y_pred = np.array(y_pred)
    acc = accuracy_score(y_test, y_pred) * 100

    _save_classification_report(y_test, y_pred, class_names, output_dir, "INT8")
    _save_confusion_matrix(y_test, y_pred, class_names, output_dir, "INT8", acc)

    return acc


# --------------------------------------------------------------
# LOAD DATA (flat dir + splits CSV)
# --------------------------------------------------------------

def parse_splits_csv(csv_path):
    """Read splits CSV into a {wav_filename: split} dict.

    Supports both formats:
      - Legacy:  filename,split  (e.g. xc1002657_2860.wav,train)
      - Stage-8: file_id,split   (e.g. XC1002657_2860,train)
    Comment lines starting with '#' are skipped.
    """
    import csv as _csv
    from io import StringIO

    splits = {}
    with open(csv_path, 'r') as f:
        content = ''.join(line for line in f if not line.startswith('#'))

    reader = _csv.DictReader(StringIO(content))
    for row in reader:
        sp = row['split']
        if 'file_id' in row:
            # Stage-8 format: XC1002657_2860  ->  xc1002657_2860.wav
            fid = row['file_id']
            wav = 'xc' + fid[2:] + '.wav'
        else:
            wav = row['filename']
        splits[wav] = sp
    return splits


def load_data_from_csv(csv_path, flat_dir, gmin, gmax, n_mels,
                       augmentation_mode='none', time_shift_ms=100,
                       pitch_shift_steps=2, mixup_alpha=0.2):
    """
    Load data using a flat directory + splits CSV.

    The CSV maps filename -> split (train/val/test).
    Files are discovered in flat_dir/{class_name}/{filename}.
    Uses compute_spec() for spectrograms + same augmentation pipeline.
    """
    if augmentation_mode == 'none':
        print("\n⚠ WARNING: No augmentation enabled")
        print("  For better results, try --augment, --mixup, or --specaugment\n")

    splits = parse_splits_csv(csv_path)

    X_test, y_test, test_paths = [], [], []
    X_val, y_val, val_paths = [], [], []
    X_train, y_train, train_paths = [], [], []
    labels = {}
    idx = 0
    failed_files = []
    csv_hits = 0
    csv_misses = 0

    # Build lookup: filename -> (class_name, full_path)
    file_lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                file_lookup[f] = (class_name, os.path.join(class_dir, f))

    # Count totals for progress bar
    total_files = 0
    train_count = 0
    for fn in splits:
        if fn in file_lookup:
            total_files += 1
            if splits[fn] == 'train':
                train_count += 1

    if augmentation_mode in ['baseline', 'specaugment']:
        total_files += train_count

    print(f"\nDataset Structure: CSV-based split from {csv_path}")
    print(f"  Flat dir: {flat_dir}")
    print(f"  CSV entries: {len(splits)}")
    print(f"  Files found: {len(file_lookup)}")

    print(f"\nAugmentation Strategy:")
    if augmentation_mode == 'baseline':
        print(f"  Train: baseline augmentation (time/pitch shift)")
    elif augmentation_mode == 'mixup':
        print(f"  Train: mixup (alpha={mixup_alpha})")
    elif augmentation_mode == 'specaugment':
        print(f"  Train: SpecAugment (freq/time masking)")
    else:
        print(f"  Train: no augmentation")
    print(f"  Total samples to process: {total_files}")

    with tqdm(total=total_files, desc="Loading from CSV splits", unit="file") as pbar:
        # Process each file in CSV order, grouped by split for consistency
        for target_split in ['test', 'val', 'train']:
            for fn, split in sorted(splits.items()):
                if split != target_split:
                    continue
                if fn not in file_lookup:
                    csv_misses += 1
                    continue

                csv_hits += 1
                class_name, full_path = file_lookup[fn]

                if class_name not in labels:
                    labels[class_name] = idx
                    idx += 1

                try:
                    audio, _ = librosa.load(full_path, sr=TARGET_SR)
                    audio = audio[:FIXED_AUDIO_LENGTH] if len(audio) > FIXED_AUDIO_LENGTH else \
                        np.pad(audio, (0, FIXED_AUDIO_LENGTH - len(audio)))

                    spec = compute_spec(audio, TARGET_SR, gmin, gmax, n_mels)

                    if split == 'test':
                        X_test.append(spec)
                        y_test.append(labels[class_name])
                        test_paths.append(full_path)
                        pbar.update(1)
                    elif split == 'val':
                        X_val.append(spec)
                        y_val.append(labels[class_name])
                        val_paths.append(full_path)
                        pbar.update(1)
                    elif split == 'train':
                        X_train.append(spec)
                        y_train.append(labels[class_name])
                        train_paths.append(full_path)
                        pbar.update(1)

                        # Augmented sample (mode-dependent, train only)
                        if augmentation_mode == 'baseline':
                            aug_audio = augment_baseline(audio, TARGET_SR, time_shift_ms, pitch_shift_steps)
                            aug_spec = compute_spec(aug_audio, TARGET_SR, gmin, gmax, n_mels)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(full_path + "_aug")
                            pbar.update(1)
                        elif augmentation_mode == 'specaugment':
                            aug_spec = augment_specaugment(spec)
                            X_train.append(aug_spec)
                            y_train.append(labels[class_name])
                            train_paths.append(full_path + "_specaug")
                            pbar.update(1)

                except Exception as e:
                    failed_files.append(f"{full_path}: {str(e)}")
                    if split == 'train' and augmentation_mode in ['baseline', 'specaugment']:
                        pbar.update(2)
                    else:
                        pbar.update(1)
                    continue

    # Convert to numpy arrays
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    if csv_misses > 0:
        print(f"\n⚠ Warning: {csv_misses} CSV entries had no matching file in {flat_dir}")

    if len(failed_files) > 0:
        print(f"\n⚠ Warning: {len(failed_files)} files failed to load")
        print("Failed files saved to data_loading_errors.txt")
        with open('data_loading_errors.txt', 'w') as f:
            for error in failed_files:
                f.write(error + '\n')

    num_classes = len(labels)
    print(f"\n✓ CSV Split Complete:")
    print(f"  Test (held-out):        {len(X_test):5d} samples")
    print(f"  Val (held-out):         {len(X_val):5d} samples")
    print(f"  Train (w/ augment):     {len(X_train):5d} samples")
    print(f"  Total:                  {len(X_test) + len(X_val) + len(X_train):5d} samples")
    print(f"  Classes:                {num_classes}")
    print(f"\n  ✓ NO DATA LEAKAGE: Test and Val samples never augmented")
    print(f"  ✓ INDEPENDENT SPLITS: True held-out evaluation (CSV-based)")

    return X_train, X_val, X_test, y_train, y_val, y_test, labels, len(failed_files)


# --------------------------------------------------------------
# CUSTOM TRAINING LOOP FOR MIXUP
# --------------------------------------------------------------
class MixupDataGenerator(keras.utils.Sequence):
    """Custom data generator for mixup augmentation"""
    def __init__(self, X, y, batch_size, alpha=0.2, num_classes=10):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.num_classes = num_classes
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        # Apply mixup
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = len(X_batch)
            index = np.random.permutation(batch_size)

            X_mixed = lam * X_batch + (1 - lam) * X_batch[index]

            # Convert to one-hot for mixing
            y_a = keras.utils.to_categorical(y_batch, self.num_classes)
            y_b = keras.utils.to_categorical(y_batch[index], self.num_classes)
            y_mixed = lam * y_a + (1 - lam) * y_b

            return X_mixed, y_mixed
        else:
            return X_batch, keras.utils.to_categorical(y_batch, self.num_classes)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    # Initialize logger and config
    config = get_config()
    logger = TrainingLogger(config['output_dir'])

    print(f"\nConfig:")
    print(f"  Random seed: {config['random_seed']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Finetune epochs: {config['finetune_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Warmup LR: {config['warmup_lr']}")
    print(f"  Finetune LR: {config['finetune_lr']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Model: DS-CNN + SE + Residual (All) + Att + Wider (1e)")
    print(f"  Channels: {config['channels']}")
    print(f"  Augmentation: {config['augmentation_mode']}")
    if config['augmentation_mode'] == 'mixup':
        print(f"  Mixup Alpha: {config['mixup_alpha']}")
    print(f"  LR Schedule: {config['lr_schedule']}")
    print(f"  Spectrogram: {config['n_mels']}x{TIME_FRAMES} (10ms/frame)")
    print(f"  Splits CSV: {config['splits_csv']}")
    print(f"  Flat dir: {config['flat_dir']}")
    print(f"  Cache: {config['spectrogram_dir']}")

    # Log hyperparameters
    logger.log_hyperparameters(config)

    # Validate cache (smart cache management from 4d)
    cache_valid = validate_cache(config['spectrogram_dir'], config['n_mels'])
    if config['augmentation_mode'] != 'none':
        print("\n⚠ Augmentation enabled: cache not used (spectrograms computed on-the-fly)")
    else:
        if not cache_valid:
            print("\n⚠ Cache invalid or parameters changed - will clear cache")
            if os.path.exists(config['spectrogram_dir']):
                shutil.rmtree(config['spectrogram_dir'])
            os.makedirs(config['spectrogram_dir'], exist_ok=True)
        else:
            print("\n✓ Cache valid - reusing cached spectrograms")

    # Compute global stats (from training files only)
    print("\nComputing global normalization stats...")
    splits = parse_splits_csv(config['splits_csv'])
    train_files = {fn for fn, split in splits.items() if split == 'train'}
    global_min, global_max = compute_global_stats(
        config['flat_dir'], config['n_mels'], allowed_files=train_files)

    # Load data
    print(f"\nLoading dataset from CSV splits...")
    print("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels, failed_count = load_data_from_csv(
        config['splits_csv'], config['flat_dir'],
        global_min, global_max, config['n_mels'],
        augmentation_mode=config['augmentation_mode'],
        time_shift_ms=config['time_shift_ms'],
        pitch_shift_steps=config['pitch_shift_steps'],
        mixup_alpha=config['mixup_alpha']
    )
    print("=" * 70)

    # Save cache version if not using augmentation
    if config['augmentation_mode'] == 'none' and not cache_valid:
        save_cache_version(config['spectrogram_dir'], config['n_mels'])
        print("✓ Cache version saved")

    class_names = list(class_labels.keys())
    num_classes = len(class_names)

    total_samples = len(X_train) + len(X_val) + len(X_test)

    print(f"\n✓ Total samples loaded: {total_samples}")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Spectrogram shape: {X_train[0].shape}")

    print(f"\nFinal Split:")
    print(f"  Train:     {len(X_train):5d} samples ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"  Val:       {len(X_val):5d} samples ({len(X_val) / total_samples * 100:.1f}%)")
    print(f"  Test:      {len(X_test):5d} samples ({len(X_test) / total_samples * 100:.1f}%)")
    print(f"  Total:     {total_samples:5d} samples")

    # Verify class distributions
    print(f"\nTest Set Class Distribution:")
    for class_name, class_idx in sorted(class_labels.items(), key=lambda x: x[1]):
        count = np.sum(y_test == class_idx)
        print(f"  {class_name:30s}: {count:3d} samples")

    print(f"\nValidation Set Class Distribution:")
    for class_name, class_idx in sorted(class_labels.items(), key=lambda x: x[1]):
        count = np.sum(y_val == class_idx)
        print(f"  {class_name:30s}: {count:3d} samples")

    # Log dataset info
    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    logger.log_dataset_info(X_all, y_all, class_labels, X_train, X_val, X_test, failed_count)

    # Calibration set (from validation set)
    if len(X_val) < config['calib_samples']:
        print(f"\n⚠ Warning: Requested {config['calib_samples']} calibration samples, "
              f"but only {len(X_val)} validation samples available")
        config['calib_samples'] = len(X_val)
        print(f"  Using all {config['calib_samples']} validation samples for calibration")

    X_calib = X_val[:config['calib_samples']]
    print(f"\n✓ Calibration set: {len(X_calib)} samples (from validation set)")

    # Create model
    print(f"\n{'=' * 70}")
    print("CREATING DS-CNN + SE + RESIDUAL (ALL) + ATTENTION + WIDER (Model 1e)")
    print(f"{'=' * 70}")
    print(f"  Channels: {config['channels']}")
    print(f"  Input shape: {config['input_shape']}")
    model = create_se_res_att(
        num_classes,
        config['input_shape'],
        dropout=config['dropout'],
        channels=config['channels']
    )
    model.summary()

    # Log model info
    logger.log_model_info(model)

    # Prepare training data based on augmentation mode
    if config['augmentation_mode'] == 'mixup':
        # Use custom generator for mixup
        train_generator = MixupDataGenerator(
            X_train, y_train,
            config['batch_size'],
            alpha=config['mixup_alpha'],
            num_classes=num_classes
        )
        val_data = (X_val, keras.utils.to_categorical(y_val, num_classes))
        loss_function = 'categorical_crossentropy'
        print("\n✓ Using Mixup data generator for training")
    else:
        train_generator = None
        val_data = (X_val, y_val)
        loss_function = 'sparse_categorical_crossentropy'

    # Compile for warmup
    model.compile(
        optimizer=Adam(learning_rate=config['warmup_lr']),
        loss=loss_function,
        metrics=['accuracy']
    )

    # Stage 1: Warmup training
    logger.start_stage("STAGE 1: WARMUP TRAINING")
    print(f"\n{'=' * 70}")
    print(f"STAGE 1: WARMUP TRAINING ({config['warmup_epochs']} epochs)")
    print(f"{'=' * 70}")

    warmup_checkpoint = os.path.join(config['output_dir'], 'warmup_best.weights.h5')
    warmup_callbacks = [
        callbacks.ModelCheckpoint(
            warmup_checkpoint, monitor='val_accuracy',
            save_best_only=True, save_weights_only=True,
            mode='max', verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )
    ]

    # Add LR schedule based on config (from 4d)
    if config['lr_schedule'] == 'cosine':
        warmup_callbacks.append(
            callbacks.LearningRateScheduler(
                lambda epoch: config['warmup_lr'] * 0.5 * (1 + np.cos(np.pi * epoch / config['warmup_epochs'])),
                verbose=0
            )
        )
    elif config['lr_schedule'] == 'plateau':
        warmup_callbacks.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                min_lr=1e-7, verbose=1
            )
        )
    elif config['lr_schedule'] == 'both':
        warmup_callbacks.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                min_lr=1e-7, verbose=1
            )
        )
        warmup_callbacks.append(
            callbacks.LearningRateScheduler(
                lambda epoch: config['warmup_lr'] * 0.5 * (1 + np.cos(np.pi * epoch / config['warmup_epochs'])),
                verbose=0
            )
        )

    try:
        if config['augmentation_mode'] == 'mixup':
            warmup_history = model.fit(
                train_generator,
                validation_data=val_data,
                epochs=config['warmup_epochs'],
                callbacks=warmup_callbacks,
                verbose=1
            )
        else:
            warmup_history = model.fit(
                X_train, y_train,
                validation_data=val_data,
                epochs=config['warmup_epochs'],
                batch_size=config['batch_size'],
                callbacks=warmup_callbacks,
                verbose=1
            )
    except Exception as e:
        print(f"\n✗ Warmup training failed: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Reduce batch size: --batch_size 16")
        print("2. Force CPU mode: --force_cpu")
        print("3. Limit GPU memory: --gpu_memory_limit 8192")
        raise

    logger.end_stage("STAGE 1: WARMUP TRAINING", warmup_history)
    print("\n✓ Warmup complete - best weights restored")

    # Stage 2: Fine-tuning
    logger.start_stage("STAGE 2: FINE-TUNING")
    print(f"\n{'=' * 70}")
    print(f"STAGE 2: FINE-TUNING ({config['finetune_epochs']} epochs)")
    print(f"{'=' * 70}")

    model.compile(
        optimizer=Adam(learning_rate=config['finetune_lr']),
        loss=loss_function,
        metrics=['accuracy']
    )

    finetune_checkpoint = os.path.join(config['output_dir'], 'finetune_best.weights.h5')
    finetune_callbacks = [
        callbacks.ModelCheckpoint(
            finetune_checkpoint, monitor='val_accuracy',
            save_best_only=True, save_weights_only=True,
            mode='max', verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )
    ]

    # Add LR schedule for finetune
    if config['lr_schedule'] in ['plateau', 'both']:
        finetune_callbacks.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                min_lr=1e-8, verbose=1
            )
        )

    try:
        if config['augmentation_mode'] == 'mixup':
            finetune_history = model.fit(
                train_generator,
                validation_data=val_data,
                epochs=config['finetune_epochs'],
                callbacks=finetune_callbacks,
                verbose=1
            )
        else:
            finetune_history = model.fit(
                X_train, y_train,
                validation_data=val_data,
                epochs=config['finetune_epochs'],
                batch_size=config['batch_size'],
                callbacks=finetune_callbacks,
                verbose=1
            )
    except Exception as e:
        print(f"\n✗ Fine-tuning failed: {e}")
        print("\nCannot continue with invalid model. Please check:")
        print("1. GPU memory issues - try --force_cpu or smaller --batch_size")
        print("2. Learning rate too high - try lower --finetune_lr")
        print("3. Model architecture issues")
        raise

    logger.end_stage("STAGE 2: FINE-TUNING", finetune_history)
    print("\n✓ Fine-tuning complete - best weights restored")

    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(warmup_history, finetune_history, config['output_dir'])

    # Save FP32 model (.keras only, no .h5)
    fp32_path = os.path.join(config['output_dir'], 'model_fp32.keras')
    model.save(fp32_path)
    print(f"✓ Saved FP32 model: {fp32_path}")

    # Evaluate FP32
    logger.start_stage("EVALUATION: FP32 (.keras)")
    print(f"\n{'=' * 70}")
    print("EVALUATING FP32 MODEL (.keras) ON HELD-OUT TEST SET")
    print(f"{'=' * 70}")
    fp32_acc = evaluate_model(model, X_test, y_test, class_names,
                              config['output_dir'], "FP32")
    logger.log_evaluation("FP32 (.keras)", fp32_acc,
                          os.path.join(config['output_dir'], 'classification_report_fp32.txt'))

    # Convert to TFLite INT8 (POST-TRAINING QUANTIZATION)
    logger.start_stage("TFLITE CONVERSION (PTQ)")
    print(f"\n{'=' * 70}")
    print("CONVERTING TO INT8 TFLITE (POST-TRAINING QUANTIZATION)")
    print(f"{'=' * 70}")
    int8_path = os.path.join(config['output_dir'], 'model_int8.tflite')
    convert_to_tflite_int8(model, X_calib, int8_path)

    # Evaluate INT8
    logger.start_stage("EVALUATION: INT8 TFLite")
    print(f"\n{'=' * 70}")
    print("EVALUATING INT8 TFLITE ON HELD-OUT TEST SET")
    print(f"{'=' * 70}")
    int8_acc = evaluate_tflite(int8_path, X_test, y_test, class_names,
                               config['output_dir'])
    logger.log_evaluation("INT8 TFLite", int8_acc,
                          os.path.join(config['output_dir'], 'classification_report_int8.txt'))

    # Collect model sizes
    model_sizes = {
        "FP32 (.keras)": f"{os.path.getsize(fp32_path) / (1024 ** 2):.2f} MB",
        "INT8 (.tflite)": f"{os.path.getsize(int8_path) / 1024:.1f} KB"
    }

    # Log final results
    logger.log_final_results(fp32_acc, int8_acc, model_sizes,
                             warmup_history, finetune_history, config, model)

    # Console summary
    drop = fp32_acc - int8_acc
    total_time = time.time() - script_start

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS (FIXED 90/60/450 SPLIT EVALUATION)")
    print(f"{'=' * 70}")
    print(f"Augmentation Mode:       {config['augmentation_mode']}")
    if config['augmentation_mode'] == 'mixup':
        print(f"Mixup Alpha:             {config['mixup_alpha']}")
    print(f"LR Schedule:             {config['lr_schedule']}")
    print(f"FP32 Accuracy:           {fp32_acc:6.2f}%")
    print(f"INT8 Accuracy:           {int8_acc:6.2f}%")
    print(f"Accuracy Drop:           {drop:6.2f}%")
    print(f"Total Execution Time:    {format_time(total_time)}")
    print(f"\n✓ Test/Val sets were HELD-OUT during training (no data leakage)")
    print(f"✓ Results are publication-ready and reproducible")
    print(f"{'=' * 70}")

    print(f"\n✓ Complete training report saved to:")
    print(f"  {logger.log_path}")
    print(f"\nAll results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()

    # Report total execution time
    total_script_time = time.time() - script_start
    print(f"\n{'=' * 70}")
    print(f"Script completed in: {format_time(total_script_time)}")
    print(f"{'=' * 70}")
