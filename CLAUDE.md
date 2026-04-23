# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MyGardenBird is an end-to-end pipeline for creating bird audio classification datasets from Xeno-Canto recordings. The pipeline downloads audio, annotates bird vocalizations using an interactive GUI, extracts 3-second clips, applies quality control, generates optimal train/val/test splits using Mixed Integer Programming, and trains CNN classifiers.

**Dataset**: 6000 annotated 3-second segments across 10 Malaysian bird species, available on Zenodo: https://doi.org/10.5281/zenodo.18112566

## Core Configuration

All paths and species configuration are centralized in `config.py`. **Before running any script, edit these two constants**:

```python
PROJECT_ROOT = "/Volumes/Evo"     # Your storage mount point
DATASET_NAME = "MYGARDENBIRD"     # Dataset folder name
```

All Stage scripts automatically derive their paths from these constants. The pipeline creates this directory structure:

```
PROJECT_ROOT/DATASET_NAME/
├── project_csv/              # Centralized metadata (target_species.csv, recordings.csv)
├── per_species_csv/          # Stage 1: XC metadata per species
├── per_species_flacs/        # Stage 2-3: Downloaded FLACs + annotations (.txt)
├── mygardenbird16khz/        # Stage 6: Extracted 3-sec clips (16kHz)
├── mygardenbird44khz/        # Stage 6: Extracted 3-sec clips (44kHz)
├── metadata16khz/            # Stage 7: clips.csv, qc_report.csv, splits CSVs
└── metadata44khz/            # Stage 7: clips.csv, qc_report.csv, splits CSVs
```

**Important**: Place `target_species.csv` in `<PROJECT_ROOT>/<DATASET_NAME>/project_csv/` before running any pipeline stage. The CSV controls which species are active (`active=yes` column).

## Pipeline Stages

The pipeline is sequential - each Stage depends on outputs from previous stages:

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `Stage1_xc_fetch_metadata.py` | Download XC metadata for active species |
| 2 | `Stage2_xc_dload_all_from_species_list.py` | Download FLAC recordings |
| 3 | `Stage3_xc_dload_delta_by_id.py` | Download specific recording IDs |
| 4 | `Stage4_audit_downloads.py` | Verify FLAC files against metadata |
| 5 | `Stage5_find_segments_interactive.py` | **Interactive GUI annotator** |
| 6 | `Stage6_extract_annotated_segments.py` | Extract 3-second WAV clips |
| 7 | `Stage7_clip_qc_manifest.py` | QC metrics and clips.csv manifest |
| 8a | `Stage8a_splitter_mip.py` | **MIP-based splitting (recommended)** |
| 8b | `Stage8b_splitter_genetic_algorithm.py` | GA-based splitting (alternative) |
| 8c | `Stage8c_splitter_simulated_annealing.py` | SA-based splitting (alternative) |
| 9 | `Stage9_train_mygardenbird_multifeature.py` | Train CNNs (multiple architectures/features) |

### Stage 5: Interactive Annotation GUI

The annotation GUI (`Stage5_find_segments_interactive.py`) is a critical manual step:

- Uses spectrogram-based blob detection to suggest bird vocalization segments
- User reviews/adjusts/deletes suggested segments via drag-and-drop interface
- Saves annotations as `.txt` files alongside FLACs (same filename, different extension)
- **Format**: Each line is `start_ms,duration_ms,label` where label is from `{song, call, other, noise, birdsong}`
- Prevents overlapping segments (max 10% overlap = 0.3 seconds)
- Fixed segment duration: 3.0 seconds

**Annotation file locations**: `per_species_flacs/<Species Name>/A/xc####.txt`

### Stage 7: Quality Control Metrics

`Stage7_clip_qc_manifest.py` computes audio quality metrics for each 3-second clip:

- **SNR (dB)**: Signal-to-noise ratio using vocal envelope detection
- **RMS (dB)**: Root mean square energy level
- **Peak amplitude**: Maximum absolute sample value
- **Clipping detection**: Flags clips with >1% samples at ±1.0

**Output**: `metadata16khz/clips.csv` or `metadata44khz/clips.csv` with columns:
```
file_id,source_id,onset_ms,sampling_rate,snr_db,rms_db,peak_amplitude,is_clipped
```

This CSV serves as the ground truth manifest for splitting and training.

### Stage 8: Data Splitting

**Use MIP splitter (Stage8a)** - it's 564× faster than simulated annealing and 8× faster than genetic algorithm, with guaranteed optimal solutions.

```bash
python Stage8a_splitter_mip.py /path/to/dataset \
    --train_ratio 0.80 --val_ratio 0.10 --test_ratio 0.10 \
    --output ./metadata16khz/splits_mip_80_10_10.csv
```

**Critical constraint**: Source-based separation - all clips from the same XC recording ID go into the same split (train/val/test) to prevent data leakage.

**Output format** (CSV with header comment):
```csv
# split_ratio=80:10:10 seed=42 objective=0 solver=mip_cbc
filename,split
xc1002657_2860.wav,test
xc1003831_2642.wav,train
```

Pre-generated optimal splits are in `metadata16khz/` and `metadata44khz/` directories.

## Training Architecture

`Stage9_train_mygardenbird_multifeature.py` supports:

**Models**: MobileNetV3Small, EfficientNetB0, ResNet50, VGG16 (ImageNet pretrained by default)

**Features**:
- Mel spectrogram (recommended - best accuracy)
- STFT (magnitude spectrogram)
- MFCC + deltas (not recommended for bird audio)

**Augmentation**:
- `--specaug`: SpecAugment (time + frequency masking)
- `--mixup <alpha>`: Mixup augmentation (e.g., `--mixup 0.2`)

**Key hyperparameters**:
- Fixed CNN input: 224×224 (regardless of sample rate)
- Dynamic hop length computation: `hop = (sample_rate × 3.0s) / 224` ensures consistent time resolution
  - 16kHz: hop=214
  - 44kHz: hop=590

### Sample Rate Handling

The training script **auto-detects sample rate** from the first WAV file in `--dataset_root`. To override:

```bash
python Stage9_train_mygardenbird_multifeature.py --sample_rate 44100
```

**Important**: Dataset root path determines both sample rate and data location:
- 16kHz: `--dataset_root /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz`
- 44kHz: `--dataset_root /Volumes/Evo/MYGARDENBIRD/mygardenbird44khz`

### MobileNetV3Small Training

MobileNetV3S requires different hyperparameters than other CNNs (see `MOBILENETV3S_BASELINE_ANALYSIS.md`):

- Warmup: 10 epochs (vs 5 for others)
- Fine-tuning: All layers (vs top 20%)
- Fine-tune LR: 1e-4 (vs 5e-5)
- Weight decay: 1e-5 (vs 1e-4)
- Dropout: 0.3/0.2 (vs 0.5/0.4)
- Hidden units: 512 (vs 256)
- Early stopping patience: 15 (vs 7)

## Training Commands

### Basic training (no augmentation)
```bash
python Stage9_train_mygardenbird_multifeature.py \
    --model efficientnetb0 \
    --feature mel \
    --splits_csv ./metadata16khz/splits_mip_80_10_10.csv \
    --seed 42
```

### With augmentation
```bash
# SpecAugment
python Stage9_train_mygardenbird_multifeature.py --model mobilenetv3s --feature mel --specaug

# Mixup (recommended: alpha=0.2)
python Stage9_train_mygardenbird_multifeature.py --model mobilenetv3s --feature mel --mixup 0.2
```

### Multi-seed robustness testing
```bash
# Run with 3 seeds: 42, 100, 786
for seed in 42 100 786; do
    python Stage9_train_mygardenbird_multifeature.py \
        --model efficientnetb0 --feature mel --seed $seed
done
```

### 44kHz training
```bash
python Stage9_train_mygardenbird_multifeature.py \
    --dataset_root /Volumes/Evo/MYGARDENBIRD/mygardenbird44khz \
    --splits_csv ./metadata44khz/splits_mip_80_10_10.csv \
    --model efficientnetb0 --feature mel
```

## Results Directory Structure

Training outputs are organized by platform:

```
results_16k_linux/
  <model>_<feature>_<pretrained>_<aug>_seed<N>/
    ├── test_metrics.json          # Final test accuracy, loss
    ├── confusion_matrix.png
    ├── training_history.png
    ├── classification_report.txt
    └── config.json               # Full hyperparameter record

results_44k_linux/
  (same structure)

results_16k_macos/
  (same structure for Mac local runs)
```

**Naming convention**: `mobilenetv3s_mel_pretrained_mixup0.2_seed42`
- Model: `{mobilenetv3s|efficientnetb0|resnet50|vgg16}`
- Feature: `{mel|stft|mfcc}`
- Pretrained: `pretrained` (ImageNet) or `scratch`
- Augmentation: `{noaug|specaug|mixup<alpha>}`
- Seed: `seed{42|100|786}`

## Remote Training via Dropbox Trigger System

For training on a Linux machine behind NAT (no direct SSH from Mac):

### Mac side (trigger training)
```bash
cd ~/Dropbox/Conda/MyGardenBird
./TRIGGER_FROM_MAC.sh
```

### Linux side (auto-responds via cron)
The `linux_auto_trainer.sh` script runs every minute via cron, checking for trigger files synced through Dropbox.

**Setup** (one-time on Linux):
```bash
crontab -e
# Add: * * * * * /bin/bash $HOME/Dropbox/Conda/MyGardenBird/linux_auto_trainer.sh
```

**Status monitoring**: Check `LINUX_STATUS.txt` (synced via Dropbox) for Linux agent heartbeat and training progress.

See `TRIGGER_SYSTEM_GUIDE.md` for detailed setup.

## Data Leakage Prevention

**Critical**: Clips from the same XC recording (source_id) must never appear in different splits.

**How it's enforced**:
1. Stage 8 splitters extract source_id from filename: `xc1002657_2860.wav` → source_id `1002657`
2. MIP optimization assigns entire sources to splits, not individual clips
3. Validation: Run `extras/verify_structural_integrity.py` to check for leakage

**Filename format**: `xc<source_id>_<onset_ms>.wav`
- Example: `xc1002657_2860.wav` = recording XC1002657, segment starts at 2860ms

## Species Configuration

Edit `project_csv/target_species.csv`:

```csv
Common name,Scientific name,eBird code,active
Javan Myna,Acridotheres javanicus,javmyn,yes
Asian Koel,Eudynamys scolopaceus,asikoe1,yes
Pink-necked Green Pigeon,Treron vernans,pngpig1,no
```

**Active species**: Only rows with `active=yes` are processed by Stage 1 metadata fetch.

**Current dataset**: 10 active species (6000 clips total, 600 per species)

**Expansion**: Set additional species to `active=yes` to expand the dataset. See `DATASET_EXPANSION_GUIDE.md`.

## Testing Commands

No formal test suite exists. Verify pipeline stages with:

```bash
# Verify config loads correctly
python config.py

# Check annotation file format
python Stage6_extract_annotated_segments.py --dry-run

# Validate splits (no data leakage)
python extras/verify_structural_integrity.py

# Benchmark splitters
python benchmark_splitters.py --dataset /path/to/mygardenbird16khz
```

## Dependencies

```bash
pip install numpy scipy librosa soundfile requests tqdm matplotlib sounddevice pulp tensorflow psutil seaborn scikit-learn pandas
```

**Critical**: `pulp` is required for MIP-based splitting (Stage 8a)

## Key Files to Understand

- `config.py`: All path configuration and species catalog
- `Stage5_find_segments_interactive.py`: GUI annotator (manual review)
- `Stage7_clip_qc_manifest.py`: QC metrics computation
- `Stage8a_splitter_mip.py`: Optimal splitting algorithm
- `Stage9_train_mygardenbird_multifeature.py`: Multi-model CNN trainer
- `utils.py`: Shared helper functions (currently minimal - just time formatting)

## Common Issues

**"Cannot find target_species.csv"**: Ensure CSV is in `<PROJECT_ROOT>/<DATASET_NAME>/project_csv/`, not in the script directory.

**"Sample rate mismatch"**: Ensure `--dataset_root` points to correct directory (16khz vs 44khz) or explicitly set `--sample_rate`.

**"Data leakage detected"**: Run `Stage8a_splitter_mip.py` again - never manually edit split assignments.

**"MobileNetV3S underfitting"**: Use MobileNetV3S-specific hyperparameters (see MOBILENETV3S_BASELINE_ANALYSIS.md).

**"Shape mismatch in training"**: Verify `n_mels=224` (not 128) for 224×224 CNN input.

## Reproducibility

All random operations are seeded for reproducibility:
- Splitting: MIP is deterministic (no seed needed)
- Training: `--seed <N>` controls TF/NumPy random state
- Recommended seeds for multi-seed experiments: 42, 100, 786

## Performance Benchmarks

**Best model**: EfficientNetB0 + Mel + Mixup0.2 → 93.4% test accuracy (422 MFLOPs)

**Edge deployment**: MobileNetV3S + Mel → 90.1% accuracy (92 MFLOPs, 4.6× more efficient)

**Splitter speed**: MIP solves 80:10:10 split in 1.7 seconds (vs 952s for Simulated Annealing)

See README.md for full benchmark results.
