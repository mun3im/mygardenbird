# SEAbird

Dataset creation and validation pipeline for Southeast Asian bird audio classification.

## Overview

End-to-end pipeline from Xeno-Canto downloads to optimized train/val/test splits:

1. **Download** recordings from Xeno-Canto (FLAC format)
2. **Annotate** bird vocalizations with interactive GUI
3. **Extract** 3-second segments
4. **Quality control** and filtering
5. **Split** with MIP optimization (prevents data leakage)

## Pipeline Stages

| Stage | Script                                     | Description                           |
| ----- | ------------------------------------------ | ------------------------------------- |
| 1     | `Stage1_xc_fetch_metadata.py`              | Fetch recording metadata              |
| 2     | `Stage2_xc_dload_all_from_species_list.py` | Download recordings                   |
| 3     | `Stage3_xc_dload_delta_by_id.py`           | Download specific IDs                 |
| 4     | `Stage4_eda_downloads.py`                  | Exploratory data analysis             |
| 5     | `Stage5_find_segments_interactive.py`      | Interactive annotation GUI            |
| 6     | `Stage6_extract_annotated_segments.py`     | Extract WAV segments                  |
| 7     | `Stage7_quality_control_selection.py`      | Quality control                       |
| 8a    | `Stage8a_splitter_mip.py`                  | **MIP-based splitting (recommended)** |
| 8b    | `Stage8b_splitter_genetic_algorithm.py`    | GA-based splitting                    |
| 8c    | `Stage8c_splitter_simulated_annealing.py`  | SA-based splitting                    |
| 9     | `Stage9_train_seabird_multifeature.py`       | Train 4 CNN models                    |

## MIP Splitter (Stage 8a)

Generates CSV-based splits with configurable ratios:

```bash
python Stage8a_splitter_mip.py /path/to/dataset \
    --train_ratio 0.80 --val_ratio 0.10 --test_ratio 0.10 \
    --output /path/to/splits.csv
```

**Output format:**
```csv
# split_ratio=80:10:10 seed=42 objective=0 solver=mip_cbc
filename,split
xc1002657_2860.wav,test
xc1003831_2642.wav,train
...
```

**Key features:**
- Source-based separation (same recording never in multiple splits)
- Perfect class balance (objective=0 means exact ratios achieved)
- Reproducible via seed parameter
- CSV output for use with any training framework

## Pre-generated Splits

Ready-to-use splits for 6000-sample dataset (seed=42, all objective=0):

```
splits_csv/
  seabird_splits_70_15_15_seed42.csv
  seabird_splits_75_10_15_seed42.csv
  seabird_splits_80_10_10_seed42.csv
```

## Training

For audio-focused CNN training, see [mun3im/mynanet](https://github.com/mun3im/mynanet).

## Installation

```bash
pip install numpy scipy librosa soundfile requests tqdm matplotlib sounddevice pulp
```

## License

MIT
