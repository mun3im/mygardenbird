# MyGardenBird

End-to-end pipeline for building and validating a Malaysian garden-bird audio
classification dataset — from Xeno-Canto download to trained CNN classifiers.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20306877.svg)](https://doi.org/10.5281/zenodo.20306877)

## The dataset at a glance

| | |
|---|---|
| Species | **12** Malaysian garden birds (600 clips each) |
| Clips | **7,200** manually verified 3-second segments |
| Sample rates | 16 kHz and 44.1 kHz |
| Split | **80:10:10**, MIP-optimised, source-separated (no leakage) |
| Sources | 1,381 unique Xeno-Canto recordings |

A **14-species "Plus" edition** (8,400 clips, adds Common Myna & Zebra Dove) is
documented in [`mygardenbirdplus16khz/`](mygardenbirdplus16khz/README.md).

## Headline results

**BirdNET v2.4 zero-shot** (no fine-tuning — a label-quality check):

| Sample rate | Accuracy | Macro AUC |
|---|---:|---:|
| 16 kHz | **97.94%** | 0.9913 |
| 44.1 kHz | **98.06%** | 0.9922 |

**Best fine-tuned CNN** — EfficientNet-B0 + Mixup 0.2, Mel spectrogram, 3 seeds:

| Sample rate | Test accuracy |
|---|---:|
| 16 kHz | **96.39% ± 0.84%** |
| 44.1 kHz | **94.24% ± 0.94%** |

- Smallest deployable model: **MobileNetV3-Small** → 92.4% at 16 kHz, **~1.5 MB**
  INT8 TFLite (1.24 M params).
- MIP splitter solves the 80:10:10 split in **1.7 s** — 564× faster than
  simulated annealing, with guaranteed-optimal class balance.

Full per-seed / per-model / per-augmentation tables:
12-species in [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md);
14-species in [`benchmark_summaries/plus_16k_14species.md`](benchmark_summaries/plus_16k_14species.md).
(Raw per-run training outputs stay local and are not published.)

## Pipeline

Xeno-Canto → annotate → extract → QC → BirdNET validate → MIP split → train.
The nine numbered `pipeline/StageN_*.py` scripts run in sequence; each derives
its paths from `pipeline/config.py`.

**Full stage-by-stage guide, hyperparameters, and reproduction commands:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).**

## Quick start

```bash
# 1. Install (pulp is required for MIP splitting)
pip install -r requirements.txt

# 2. Point config at your storage, then place target_species.csv
#    in <PROJECT_ROOT>/<DATASET_NAME>/project_csv/
$EDITOR pipeline/config.py          # set PROJECT_ROOT and DATASET_NAME

# 3. Train the best configuration
python pipeline/Stage9_train_mygardenbird_multifeature.py \
    --model efficientnetb0 --feature mel --mixup 0.2 \
    --splits_csv ./metadata16khz/splits_mip_80_10_10.csv --seed 42
```

## Where things live

| Topic | Location |
|---|---|
| Pipeline guide, training matrix, hyperparameters | [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) |
| Metadata schema / data dictionary | [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md), [`project_csv/`](project_csv/README.md) |
| Pre-generated splits & QC manifests | [`metadata16khz/`](metadata16khz/README.md), [`metadata44khz/`](metadata44khz/README.md) |
| Stage-specific notes | `STAGE4_ANNOTATOR_DOCS.md`, `STAGE6_QC_MANIFEST_DOCS.md`, `STAGE7_SPLITTER_DOCS.md`, `STAGE8_TRAINER_DOCS.md` |
| CNN benchmark summaries | [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) (12-species), [`benchmark_summaries/`](benchmark_summaries/plus_16k_14species.md) (14-species) |
| 14-species Plus edition | [`mygardenbirdplus16khz/`](mygardenbirdplus16khz/README.md) |
| Helper / archived scripts | [`extras/`](extras/README.md) |

## License

Code: MIT. Dataset: CC BY-NC-SA 4.0 (per-recording licences in `recordings.csv`).
