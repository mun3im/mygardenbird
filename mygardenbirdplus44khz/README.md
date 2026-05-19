# MyGardenBird Addendum — Common Myna & Zebra Dove (44.1 kHz)

This folder contains metadata for the 44.1 kHz subset of the two supplementary
species that extend the core MyGardenBird dataset from 12 to 14 species
("MyGardenBird-Plus").

## Relationship to the 16 kHz addendum

This subset is a **strict subset** of `mygardenbirdplus16khz/`: every clip here
comes from a source recording that was also used for the 16 kHz addendum.
Only recordings with a native sample rate ≥ 44.1 kHz are included (`--no-upsample`
was applied during extraction), so the clip counts are lower than 600 per species.

| Species | 16 kHz clips | 44.1 kHz clips | Sources dropped (native SR < 44.1 kHz) |
|---|---:|---:|---:|
| Common Myna | 600 | 590 | 2 clips (1 source, 24 kHz) |
| Zebra Dove | 600 | 577 | 23 clips (5 sources, ≤32 kHz) |
| **Total** | **1,200** | **1,167** | |

## Why these species are supplementary

Common Myna (*Acridotheres tristis*) and Zebra Dove (*Geopelia striata*) are
prominent garden birds in Peninsular Malaysia, but insufficient recordings were
available within the ASEAN/Indo-Malayan geographic region to reach the 600-clip
target under the strict regional provenance constraint applied to the core
dataset. They are provided here without that constraint for use cases where
regional provenance is less critical.

## Contents

```
mygardenbirdplus44khz/
├── metadata/
│   ├── clips.csv                  # 1,167 clips (590 CM + 577 ZD)
│   ├── recordings.csv             # 220 source recordings
│   ├── splits_mip_80_10_10.csv    # MIP 80:10:10 split (source-separated)
│   └── qc_report.csv             # Per-clip QC metrics
└── README.md
```

> **Zenodo**: https://doi.org/10.5281/zenodo.18694053 *(addendum included)*

Place the downloaded species folders at the same level as the core species:

```
mygardenbird44khz/
├── Asian Koel/
├── ...
├── White-throated Kingfisher/
├── Common Myna/        ← addendum
└── Zebra Dove/         ← addendum
```

## Metadata schema

### `metadata/clips.csv`

One row per 3-second WAV clip. Schema identical to the core `metadata44khz/clips.csv`.

| Field | Type | Description |
|---|---|---|
| `file_id` | string | Clip identifier: `XC{source_id}_{onset_ms}` (PK) |
| `source_id` | string | Xeno-canto recording identifier (FK → recordings.csv) |
| `onset_ms` | integer | Clip start time within source recording (ms) |
| `sampling_rate` | integer | Sample rate (44100 Hz) |
| `snr_db` | float | Signal-to-noise ratio (dB) |
| `rms_db` | float | RMS energy level (dB) |
| `peak_amplitude` | float | Maximum absolute sample value |
| `is_clipped` | boolean | True if peak amplitude > 0.99 |

Filename derivable as: `xc{source_id}_{onset_ms}.wav`

### `metadata/recordings.csv`

One row per Xeno-canto source recording. Schema identical to the core
`project_csv/recordings.csv`.

| Field | Type | Description |
|---|---|---|
| `source_id` | string | Xeno-canto recording identifier (PK) |
| `species_common` | string | English common name |
| `species_scientific` | string | Binomial scientific name |
| `quality_grade` | string | XC quality grade (A–E) |
| `cc_license` | string | Creative Commons licence (SPDX, e.g. `CC-BY-NC-SA-4.0`) |
| `type_label` | string | Normalised vocalisation type: song / call / other |
| `latitude` | float | Recording latitude (WGS84; blank if unknown) |
| `longitude` | float | Recording longitude (WGS84; blank if unknown) |
| `country` | string | Country of recording |

### `metadata/splits_mip_80_10_10.csv`

MIP-optimal 80:10:10 source-separated split for the addendum clips.

| Split | Clips | Sources |
|---|---|---|
| train | 934 | 159 |
| val | 116 | 22 |
| test | 117 | 39 |

All clips from the same source recording appear in the same split (no leakage).

## Combining with the core dataset (MyGardenBird-Plus, 14 species, 44.1 kHz)

```python
import pandas as pd

# Clips
clips_core = pd.read_csv("metadata44khz/clips.csv")
clips_add  = pd.read_csv("mygardenbirdplus44khz/metadata/clips.csv")
clips_plus = pd.concat([clips_core, clips_add], ignore_index=True)

# Recordings
rec_core = pd.read_csv("project_csv/recordings.csv")
rec_add  = pd.read_csv("mygardenbirdplus44khz/metadata/recordings.csv")
rec_plus = pd.concat([rec_core, rec_add], ignore_index=True)

# Splits — concatenate for full-dataset training
splits_core = pd.read_csv("metadata44khz/splits_mip_80_10_10.csv", comment="#")
splits_add  = pd.read_csv("mygardenbirdplus44khz/metadata/splits_mip_80_10_10.csv", comment="#")
splits_plus = pd.concat([splits_core, splits_add], ignore_index=True)
```

The combined dataset has **8,117 clips** across **14 species** at 44.1 kHz,
derived from **1,601 unique source recordings**.

## Licence

All source recordings were obtained from Xeno-canto under Creative Commons
licences. Per-recording licence identifiers are in `recordings.csv`
(`cc_license` column). The addendum is released under **CC BY-NC-SA 4.0**,
consistent with the core dataset.
