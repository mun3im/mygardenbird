# MyGardenBird

End-to-end pipeline for building and validating a Malaysian garden-bird audio
classification dataset — from Xeno-Canto download to trained CNN classifiers.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20306876.svg)](https://doi.org/10.5281/zenodo.20306876)

## The dataset at a glance

The dataset itself is published on Zenodo under the concept DOI
[10.5281/zenodo.20306876](https://doi.org/10.5281/zenodo.20306876), which always
resolves to the current version. This repository holds the pipeline that builds
it, not the data.

| | |
|---|---|
| Release | **v2.0** (2026-11-14) |
| Species | **15** Malaysian garden birds, **438–600** clips each |
| Clips | **8,659** manually verified 3-second segments (16 kHz); 8,287 at 44.1 kHz |
| Sample rates | 16 kHz, and a 44.1 kHz subset — not an upsampling |
| Splits | 80:10:10, 75:10:15, 70:15:15, plus a **site-grouped** 80:10:10 |
| Sources | **1,732** unique Xeno-canto recordings, each verified individually |

Class sizes are uneven because recordings licensed CC BY-NC-ND are excluded
throughout — segmentation produces a derivative work — and four species run out
of usable regional material before reaching 600.

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

Full per-seed / per-model / per-augmentation tables are in
[`benchmark_summaries/`](benchmark_summaries/README.md). **Those results were
produced on the superseded 12- and 14-species releases and are not comparable
with v2.0**; the models will be retrained on the current dataset.
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
    --splits_csv /path/to/metadata16khz/splits_mip_80_10_10.csv --seed 42
```

## Where things live

| Topic | Location |
|---|---|
| Pipeline guide, training matrix, hyperparameters | [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) |
| Metadata schema / data dictionary | [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) |
| Pre-generated splits & QC manifests | ship with the [dataset deposit](https://doi.org/10.5281/zenodo.20306876), not this repository |
| Stage-specific notes | [`docs/STAGE4_ANNOTATOR_DOCS.md`](docs/STAGE4_ANNOTATOR_DOCS.md), [`docs/STAGE6_QC_MANIFEST_DOCS.md`](docs/STAGE6_QC_MANIFEST_DOCS.md), [`docs/STAGE8_SPLITTER_DOCS.md`](docs/STAGE8_SPLITTER_DOCS.md), [`docs/STAGE9_TRAINER_DOCS.md`](docs/STAGE9_TRAINER_DOCS.md) |
| CNN benchmark summaries | [`benchmark_summaries/`](benchmark_summaries/README.md) — **predate v2.0; retained for reference, to be retrained** |
| Helper / archived scripts | [`extras/`](extras/README.md) |

## License

Code: MIT. Dataset: CC BY-NC-SA 4.0 (per-recording licences in `recordings.csv`).
