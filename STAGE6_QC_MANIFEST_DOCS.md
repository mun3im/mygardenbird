# Stage 7: Clip Quality Control and Manifest Generation

`Stage7_clip_qc_manifest.py` analyses every extracted WAV clip, runs quality
checks, and writes two normalised CSV tables that serve as the dataset catalogue
for all downstream stages.

## Outputs

| File | Description |
|------|-------------|
| `qc_report.csv` | Per-clip diagnostic metrics (not used by training) |
| `recordings.csv` | One row per source recording — PK: `source_id` |
| `clips.csv` | One row per WAV clip — PK: `file_id`, FK: `source_id` |

### Schema

#### `recordings.csv`

| Column | Type | Description |
|--------|------|-------------|
| `source_id` | str | Xeno-canto recording ID (e.g. `1002657`) — **primary key** |
| `species_common` | str | Common name (e.g. `Asian koel`) |
| `species_scientific` | str | Scientific name (e.g. `Eudynamys scolopaceus`) |
| `quality_grade` | str | Xeno-canto grade A–E |
| `type_label` | str | Normalised vocalisation type: `song`, `call`, or `other` |
| `latitude` | float | Recording latitude |
| `longitude` | float | Recording longitude |
| `country` | str | Recording country |

#### `clips.csv`

| Column | Type | Description |
|--------|------|-------------|
| `file_id` | str | Clip identifier `XC{source_id}_{clip_index}` — **primary key** |
| `source_id` | str | Parent recording ID — **foreign key → recordings.csv** |
| `clip_index` | int | Onset in milliseconds from start of source FLAC |
| `sampling_rate` | int | Sample rate of extracted WAV (Hz) |
| `snr_db` | float | Signal-to-noise ratio estimate (dB) |
| `rms_db` | float | RMS level (dB) |
| `peak_amplitude` | float | Peak absolute amplitude [0–1] |
| `is_clipped` | bool | True if peak amplitude > 0.99 |

> **`wav_filename` is not stored** — it is always derivable as
> `xc{source_id}_{clip_index}.wav` (lowercase `xc` prefix).

## Entity-Relationship Diagram

```
┌──────────────────────────────┐         ┌─────────────────────────────────┐
│         recordings.csv       │         │           clips.csv             │
│  (one row per XC recording)  │         │  (one row per 3-second clip)    │
├──────────────────────────────┤         ├─────────────────────────────────┤
│ PK  source_id                │1      N │ PK  file_id  (XC{id}_{onset})  │
│     species_common           ├─────────┤ FK  source_id                   │
│     species_scientific       │         │     clip_index  (onset ms)      │
│     quality_grade            │         │     sampling_rate               │
│     type_label               │         │     snr_db                      │
│     latitude                 │         │     rms_db                      │
│     longitude                │         │     peak_amplitude              │
│     country                  │         │     is_clipped                  │
└──────────────────────────────┘         └────────────────┬────────────────┘
                                                          │ 1
                                                          │
                                                          │ N
                                         ┌────────────────┴────────────────┐
                                         │    seabird_splits_*.csv         │
                                         │  (one row per clip per split)   │
                                         ├─────────────────────────────────┤
                                         │ FK  file_id                     │
                                         │     split  (train/val/test)     │
                                         └─────────────────────────────────┘
```

Each `source_id` in `recordings.csv` has 1–10 clips in `clips.csv`.
Each `file_id` in `clips.csv` appears in exactly one row of each splits file.

## Joining the tables

```python
import pandas as pd

clips      = pd.read_csv("clips.csv")
recordings = pd.read_csv("recordings.csv")
splits     = pd.read_csv("seabird_splits_mip_75_10_15.csv", comment="#")

# Full dataset with all metadata
full = clips.merge(recordings, on="source_id")

# Train clips only, with species and SNR
train = (splits[splits.split == "train"]
         .merge(clips, on="file_id")
         .merge(recordings, on="source_id"))
```

## Quality checks

| Check | Criterion | Action |
|-------|-----------|--------|
| Duration | abs(duration − 3.0 s) < 0.1 s | Flagged in `qc_report.csv`; not filtered |
| Clipping | peak amplitude > 0.99 | Recorded in `is_clipped`; not filtered |
| Silence | RMS < −40 dB | Flagged; not filtered (informational) |
| Corruption | librosa load failure | Logged in `qc_report.csv` |

Stage 6 is expected to deliver well-formed 3-second clips; these checks are
diagnostic rather than gatekeeping.

## Running Stage 7

```bash
# Default paths from config.py
python Stage7_clip_qc_manifest.py

# Custom paths
python Stage7_clip_qc_manifest.py /path/to/extracted_segments \
    --output-dir /path/to/dataset \
    --metadata-dir /path/to/per_species_csv
```

## Prerequisites

```bash
pip install librosa soundfile numpy tqdm
```

## Relationship to other stages

| Stage | Dependency |
|-------|------------|
| Stage 6 | Produces the WAV clips that Stage 7 analyses |
| Stage 8 | Reads the WAV directory; outputs `seabird_splits_*.csv` keyed on `file_id` |
| Stage 9 | Reads `seabird_splits_*.csv`; derives wav filenames from `file_id` |

Stage 7 does not need to be re-run when splits change — splits are independent
CSV files produced by Stage 8.
