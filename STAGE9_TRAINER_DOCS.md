# Stage 9: CNN Training for Bird Audio Classification

Train CNN models on spectral features extracted from bird audio recordings.

## Supported Models

| Model | Parameters | MFLOPs | Best For |
|-------|------------|--------|----------|
| `efficientnetb0` | 4.0M | 390 | Highest accuracy (93.4%) |
| `mobilenetv3s` | 1.2M | 60 | Edge deployment (90.1% at 4.6x less compute) |
| `resnet50` | 23.5M | 4100 | Baseline comparison |
| `vgg16` | 134M | 15500 | Legacy/stability |

## Supported Features

| Feature | Description | MFLOPs | Recommendation |
|---------|-------------|--------|----------------|
| `mel` | Mel spectrogram (224 bins) | 32 | Best for bird audio |
| `stft` | STFT magnitude | 26 | Alternative |
| `mfcc` | MFCC with delta/delta-delta | 36 | Not recommended for birds |

## Quick Start

### Run with Defaults

```bash
# Default: mobilenetv3s + mel + seed=42
python Stage9_train_seabird_multifeature.py --use_pretrained
```

### Select Model

```bash
python Stage9_train_seabird_multifeature.py --model efficientnetb0 --use_pretrained
python Stage9_train_seabird_multifeature.py --model resnet50 --use_pretrained
python Stage9_train_seabird_multifeature.py --model vgg16 --use_pretrained
python Stage9_train_seabird_multifeature.py --model mobilenetv3s --use_pretrained
```

### Select Feature Type

```bash
python Stage9_train_seabird_multifeature.py --feature mel --use_pretrained   # Mel spectrogram
python Stage9_train_seabird_multifeature.py --feature stft --use_pretrained  # STFT magnitude
python Stage9_train_seabird_multifeature.py --feature mfcc --use_pretrained  # MFCC with deltas
```

### Reproducibility with Seeds

```bash
python Stage9_train_seabird_multifeature.py --seed 42 --use_pretrained
python Stage9_train_seabird_multifeature.py --seed 100 --use_pretrained
python Stage9_train_seabird_multifeature.py --seed 786 --use_pretrained
```

## Using CSV-Based Splits (Recommended)

CSV splits ensure no data leakage between train/val/test sets by keeping segments
from the same source recording together.

```bash
python Stage9_train_seabird_multifeature.py \
    --splits_csv ./seabird_splits_mip_75_10_15.csv \
    --dataset_root /path/to/audio/files \
    --use_pretrained
```

Splits CSVs use `file_id` as the key column (e.g. `XC1002657_2860`). Stage 9
derives the WAV filename automatically (`xc1002657_2860.wav`) — no change is
needed in how you invoke the script.

### Available Pre-generated Splits

| File | Train | Val | Test |
|------|-------|-----|------|
| `seabird_splits_mip_75_10_15.csv` | 75% | 10% | 15% |
| `seabird_splits_mip_80_10_10.csv` | 80% | 10% | 10% |
| `seabird_splits_mip_70_15_15.csv` | 70% | 15% | 15% |

## Full Example

```bash
python Stage9_train_seabird_multifeature.py \
    --model efficientnetb0 \
    --feature mel \
    --splits_csv ./seabird_splits_mip_75_10_15.csv \
    --dataset_root /Volumes/Evo/seabird16khz_flat \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --seed 42 \
    --use_pretrained \
    --output_dir ./results
```

## CPU-Only Training

When GPU is unavailable or occupied:

```bash
python Stage9_train_seabird_multifeature.py --force_cpu --use_pretrained
```

## All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `mobilenetv3s` | CNN architecture |
| `--feature` | `mel` | Feature extraction method |
| `--seed` | `42` | Random seed for reproducibility |
| `--batch_size` | `32` | Training batch size |
| `--num_epochs` | `50` | Maximum training epochs |
| `--learning_rate` | `0.001` | Initial learning rate |
| `--sample_rate` | `16000` | Audio sample rate (Hz) |
| `--n_mels` | `224` | Number of mel bins |
| `--n_fft` | `2048` | FFT window size |
| `--num_workers` | `4` | Data loader workers |
| `--use_pretrained` | `False` | Use ImageNet pretrained weights |
| `--force_cpu` | `False` | Disable GPU |
| `--output_dir` | `./results` | Results directory |
| `--splits_csv` | `None` | Path to splits CSV file |
| `--dataset_root` | `/Volumes/Evo/seabird16khz_flat` | Audio files directory |

## Training Strategy

### Standard Models (EfficientNetB0, ResNet50, VGG16)

1. **Warmup phase** (5 epochs): Train classifier head with frozen base
2. **Fine-tuning phase**: Unfreeze top 20% of base layers, reduce LR to 5e-5
3. **Early stopping**: Patience=7, monitoring validation accuracy

### MobileNetV3S (Optimized Strategy)

MobileNetV3S requires a less conservative training approach:

| Setting | Standard | MobileNetV3S |
|---------|----------|--------------|
| Warmup epochs | 5 | 10 |
| Fine-tuning scope | Top 20% | All layers |
| Fine-tune LR | 5e-5 | 1e-4 |
| Weight decay | 1e-4 | 1e-5 |
| Dropout | 0.5/0.4 | 0.3/0.2 |
| Hidden units | 256 | 512 |
| Early stopping patience | 7 | 15 |

## Output

Training produces:
- Model checkpoints in `--output_dir`
- Training logs with loss/accuracy curves
- Final test evaluation metrics

## Benchmark Results

> **Note:** Results below are from the finalised 10-class / 6,000-clip dataset (Barbet + Waterhen replacing Myna + Zebra Dove).
> EfficientNetB0, ResNet50, and VGG16 multi-seed results are from an earlier 10-class configuration and will be updated.
> Full re-training on the geographically-filtered dataset (lon ≥ 60°E) is pending.

Test accuracy (%) with 75:10:15 split, averaged over seeds 42, 100, 786:

| Model | Mel | STFT | MFCC |
|-------|-----|------|------|
| EfficientNetB0 | **93.4 ± 2.6** | 91.0 ± 1.5 | 89.4 ± 1.3 |
| MobileNetV3S | **87.22 ± 0.24** | - | - |
| ResNet50 | 88.8 ± 2.9 | 91.0 ± 1.1 | 86.0 ± 1.6 |
| VGG16 | 88.2 ± 0.8 | 86.7 ± 1.8 | 81.9 ± 3.4 |

### MobileNetV3S + Mel per-seed breakdown (10-class, seeds 42/100/786)

| Seed | Test Acc | Common Iora F1 | Spotted Dove F1 | LT Nightjar F1 |
|------|----------|----------------|-----------------|----------------|
| 42   | 87.44%   | 0.781          | 0.995           | 0.929          |
| 100  | 86.89%   | 0.762          | 0.994           | 0.954          |
| 786  | 87.33%   | 0.742          | 0.989           | 0.960          |
| **Mean** | **87.22 ± 0.24%** | **0.762** | **0.993** | **0.948** |

**Recommendation:** Use EfficientNetB0 + Mel for best accuracy, or MobileNetV3S + Mel for edge deployment.
