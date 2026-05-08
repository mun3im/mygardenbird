# MyGardenBird

Dataset creation and validation pipeline for Malaysian garden bird audio classification.

**Dataset:** The MyGardenBird dataset (7,200 annotated 3-second segments, 12 species) is available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18694053.svg)](https://doi.org/10.5281/zenodo.18694053)

## Overview

End-to-end pipeline from Xeno-Canto downloads to trained CNN classifiers:

1. **Download** recordings from Xeno-Canto (FLAC format)
2. **Annotate** bird vocalizations with interactive GUI
3. **Extract** 3-second segments
4. **Quality control** and filtering
5. **Validate** labels with BirdNET zero-shot evaluation
6. **Split** with MIP optimization (prevents data leakage)
7. **Train** CNN classifiers

## Dataset

- **12 Malaysian garden bird species**, 600 clips per species = **7,200 total clips**
- All clips manually verified via interactive annotation GUI
- Available at two sample rates: 16 kHz and 44.1 kHz
- Train/val/test split: **80:10:10** (MIP-optimized, source-separated)

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `Stage1_xc_fetch_metadata.py` | Fetch recording metadata from Xeno-Canto |
| 2 | `Stage2_xc_dload_all_from_species_list.py` | Download FLAC recordings |
| 2a | `Stage2a_xc_dload_delta_by_id.py` | Download specific recording IDs |
| 3 | `Stage3_audit_downloads.py` | Audit downloaded FLACs against metadata |
| 4 | `Stage4_find_segments_interactive.py` | Interactive annotation GUI |
| 5 | `Stage5_extract_annotated_segments.py` | Extract 3-second WAV segments |
| 5a | `Stage5a_find_most_confused.py` | Select best 600 clips per class |
| 6 | `Stage6_clip_qc_manifest.py` | QC metrics and dataset manifest CSV |
| 7 | `Stage7_eval_birdnet.py` | BirdNET zero-shot label quality validation |
| 8 | `Stage8_splitter_mip.py` | **MIP-based splitting (recommended)** |
| 8a | `Stage8a_splitter_genetic_algorithm.py` | GA-based splitting |
| 8b | `Stage8b_splitter_simulated_annealing.py` | SA-based splitting |
| 9 | `Stage9_train_mygardenbird_multifeature.py` | Train CNN classifiers |

## MIP Splitter (Stage 8)

Generates CSV-based splits with configurable ratios:

```bash
python Stage8_splitter_mip.py /path/to/dataset \
    --train_ratio 0.80 --val_ratio 0.10 --test_ratio 0.10 \
    --output ./metadata16khz/splits_mip_80_10_10.csv
```

**Output format:**
```csv
# split_ratio=80:10:10 seed=42 objective=0 solver=mip_cbc
filename,split
xc1002657_2860.wav,test
xc1003831_2642.wav,train
```

**Key features:**
- Source-based separation (all clips from the same XC recording go to the same split)
- Perfect class balance (objective=0 means exact ratios achieved)
- Reproducible and deterministic
- CSV output for use with any training framework

## Splitter Performance Comparison

Benchmark on 7,200-sample dataset (12 classes, 80:10:10 split):

| Algorithm | Time | Solution Quality |
|-----------|------|------------------|
| **MIP** | **1.7s** | Optimal (objective=0) |
| Genetic Algorithm | ~24s | Optimal (objective=0) |
| Simulated Annealing | ~952s | Optimal (objective=0) |

**Recommendation:** Use MIP (Stage 8) — 564× faster than SA with guaranteed optimality.

## Pre-generated Splits

Ready-to-use splits in `metadata16khz/` and `metadata44khz/` (seed=42, all objective=0):

| File | Train | Val | Test |
|------|-------|-----|------|
| `splits_mip_80_10_10.csv` | 80% | 10% | 10% |
| `splits_mip_70_15_15.csv` | 70% | 15% | 15% |

**Canonical split:** 80:10:10 (maximises training data; all results below use this split).

## Training Results

All results use ImageNet pretrained weights, Mel spectrogram (224×224), 3 random seeds (42, 100, 786), 80:10:10 MIP split.

### BirdNET Zero-Shot Baseline (no fine-tuning)

| Sample Rate | Clips | Accuracy | Macro AUC |
|-------------|-------|----------|-----------|
| 16 kHz | 7,200 | **97.94%** | 0.9913 |
| 44.1 kHz | 6,950 | **98.06%** | 0.9922 |

### 16 kHz Results — 12 Species

| Model | No Aug | SpecAugment | Mixup 0.2 | **Best** |
|-------|-------:|------------:|----------:|-------:|
| EfficientNet-B0 | 94.91 ± 1.04% | 94.91 ± 1.04% | **96.39 ± 0.84%** | **96.39%** |
| ResNet-50 | 93.06 ± 0.84% | 94.07 ± 0.58% | 94.63 ± 0.08% | 94.63% |
| VGG16 | 91.85 ± 1.55% | 92.87 ± 0.42% | 94.03 ± 0.24% | 94.03% |
| MobileNetV3-Small | 89.40 ± 1.66% | 92.27 ± 0.21% | 92.41 ± 0.79% | 92.41% |

**Best overall:** EfficientNet-B0 + Mixup 0.2 → **96.39% ± 0.84%**

#### Detailed 16 kHz Results

| Model | Augmentation | Seed 42 | Seed 100 | Seed 786 | Mean | ±SD |
|-------|-------------|--------:|---------:|---------:|-----:|----:|
| EfficientNet-B0 | noaug | 94.31% | 94.31% | 96.11% | 94.91% | 1.04% |
| EfficientNet-B0 | specaug | 94.86% | 93.89% | 95.97% | 94.91% | 1.04% |
| EfficientNet-B0 | mixup0.2 | **96.94%** | **95.42%** | **96.81%** | **96.39%** | **0.84%** |
| ResNet-50 | noaug | 92.08% | 93.47% | 93.61% | 93.06% | 0.84% |
| ResNet-50 | specaug | 94.72% | 93.61% | 93.89% | 94.07% | 0.58% |
| ResNet-50 | mixup0.2 | 94.58% | 94.72% | 94.58% | 94.63% | 0.08% |
| VGG16 | noaug | 91.25% | 90.69% | 93.61% | 91.85% | 1.55% |
| VGG16 | specaug | 92.78% | 92.50% | 93.33% | 92.87% | 0.42% |
| VGG16 | mixup0.2 | 94.17% | 93.75% | 94.17% | 94.03% | 0.24% |
| MobileNetV3-Small | noaug | 90.14% | 87.50% | 90.56% | 89.40% | 1.66% |
| MobileNetV3-Small | specaug | 92.08% | 92.50% | 92.22% | 92.27% | 0.21% |
| MobileNetV3-Small | mixup0.2 | 93.06% | 91.53% | 92.64% | 92.41% | 0.79% |

### 44.1 kHz Results — 12 Species

| Model | No Aug | SpecAugment | Mixup 0.2 | **Best** |
|-------|-------:|------------:|----------:|-------:|
| EfficientNet-B0 | 93.19 ± 0.47% | 93.91 ± 1.16% | **94.24 ± 0.94%** | **94.24%** |
| ResNet-50 | 92.04 ± 1.37% | 91.80 ± 1.66% | 93.09 ± 0.76% | 93.09% |
| VGG16 | 89.73 ± 0.72% | 91.27 ± 1.37% | 92.09 ± 0.76% | 92.09% |
| MobileNetV3-Small | 85.66 ± 1.26% | 89.69 ± 0.42% | 90.70 ± 0.36% | 90.70% |

**Best overall:** EfficientNet-B0 + Mixup 0.2 → **94.24% ± 0.94%**

#### Detailed 44.1 kHz Results

| Model | Augmentation | Seed 42 | Seed 100 | Seed 786 | Mean | ±SD |
|-------|-------------|--------:|---------:|---------:|-----:|----:|
| EfficientNet-B0 | noaug | 93.38% | 92.66% | 93.53% | 93.19% | 0.47% |
| EfficientNet-B0 | specaug | 94.10% | 92.66% | 94.96% | 93.91% | 1.16% |
| EfficientNet-B0 | mixup0.2 | **95.25%** | **94.10%** | **93.38%** | **94.24%** | **0.94%** |
| ResNet-50 | noaug | 92.09% | 90.65% | 93.38% | 92.04% | 1.37% |
| ResNet-50 | specaug | 90.07% | 91.94% | 93.38% | 91.80% | 1.66% |
| ResNet-50 | mixup0.2 | 93.67% | 92.23% | 93.38% | 93.09% | 0.76% |
| VGG16 | noaug | 90.50% | 89.06% | 89.64% | 89.73% | 0.72% |
| VGG16 | specaug | 92.66% | 91.22% | 89.93% | 91.27% | 1.37% |
| VGG16 | mixup0.2 | 91.51% | 92.95% | 91.80% | 92.09% | 0.76% |
| MobileNetV3-Small | noaug | 87.05% | 84.60% | 85.32% | 85.66% | 1.26% |
| MobileNetV3-Small | specaug | 89.93% | 89.93% | 89.21% | 89.69% | 0.42% |
| MobileNetV3-Small | mixup0.2 | 90.65% | 90.36% | 91.08% | 90.70% | 0.36% |

### Key Findings

- **Best model:** EfficientNet-B0 + Mixup 0.2 (96.39% at 16 kHz; 94.24% at 44.1 kHz)
- **Best augmentation:** Mixup 0.2 consistently outperforms no-aug and SpecAugment across all models
- **Sample rate:** 16 kHz outperforms 44.1 kHz (~2 pp gap) — high-frequency noise in 44.1 kHz recordings hurts training
- **Most efficient:** MobileNetV3-Small + Mixup 0.2 (92.41% at 16 kHz, 92 MFLOPs — 4.6× less compute than EfficientNet-B0)
- **BirdNET ceiling:** Zero-shot BirdNET reaches 97.94–98.06% with no task-specific training

### Compute vs Accuracy (16 kHz, best augmentation per model)

| Model | Best Aug | Accuracy | CNN MFLOPs | Feature MFLOPs | Total |
|-------|----------|----------|------------|----------------|-------|
| EfficientNet-B0 | mixup0.2 | **96.39%** | 390 | 32 | **422** |
| ResNet-50 | mixup0.2 | 94.63% | 4,100 | 32 | **4,132** |
| VGG16 | mixup0.2 | 94.03% | 15,500 | 32 | **15,532** |
| MobileNetV3-Small | mixup0.2 | 92.41% | 60 | 32 | **92** |

**Feature MFLOPs** (16 kHz × 3s → 224×224 Mel spectrogram): N_FFT=2048, hop=214, 224 frames, 224 mel bins → 32 MFLOPs.

### MobileNetV3-Small Hyperparameters

MobileNetV3-Small requires different settings than other CNNs to avoid underfitting:

| Setting | Other CNNs | MobileNetV3-Small |
|---------|------------|-------------------|
| Warmup epochs | 5 | 10 |
| Fine-tuning scope | Top 20% layers | All layers |
| Fine-tune learning rate | 5e-5 | 1e-4 |
| Weight decay | 1e-4 | 1e-5 |
| Dropout (classifier) | 0.5 / 0.4 | 0.3 / 0.2 |
| Hidden units | 256 | 512 |
| Early stopping patience | 7 | 15 |

### Training Commands

```bash
# Basic (no augmentation)
python Stage9_train_mygardenbird_multifeature.py \
    --model efficientnetb0 --feature mel \
    --splits_csv ./metadata16khz/splits_mip_80_10_10.csv \
    --seed 42

# With SpecAugment
python Stage9_train_mygardenbird_multifeature.py \
    --model efficientnetb0 --feature mel --specaug \
    --splits_csv ./metadata16khz/splits_mip_80_10_10.csv \
    --seed 42

# With Mixup (recommended)
python Stage9_train_mygardenbird_multifeature.py \
    --model efficientnetb0 --feature mel --mixup 0.2 \
    --splits_csv ./metadata16khz/splits_mip_80_10_10.csv \
    --seed 42
```

## Setup

### 1. Configure paths

Edit the top of `config.py`:

```python
PROJECT_ROOT = "/Volumes/Evo"     # your storage mount point
DATASET_NAME = "MYGARDENBIRD"     # top-level folder under PROJECT_ROOT
```

All stages derive their input/output paths from these two constants.

### 2. Place `target_species.csv`

Copy `target_species.csv` into `<PROJECT_ROOT>/<DATASET_NAME>/project_csv/` before running any stage. Set `active=yes` for species to include.

## Installation

```bash
pip install numpy scipy librosa soundfile requests tqdm matplotlib sounddevice pulp tensorflow psutil seaborn scikit-learn pandas
```

**Required for MIP splitting:** `pulp`

## License

MIT
