# MyGardenBird-Plus (16 kHz) — 14-Species Edition

**MyGardenBird-Plus** extends the 12-species core MyGardenBird dataset with two
supplementary garden birds — **Common Myna** (*Acridotheres tristis*) and
**Zebra Dove** (*Geopelia striata*) — for a total of **14 species / 8,400
3-second clips** (600 per species) at 16 kHz.

## Why these two species are supplementary

Common Myna and Zebra Dove are prominent garden birds in Peninsular Malaysia,
but insufficient recordings were available within the ASEAN/Indo-Malayan
geographic region to reach the 600-clip target under the strict regional
provenance constraint applied to the 12 core species. They are provided as an
addendum **without that regional constraint**, so users who need strict regional
provenance can keep to the 12-species core, while users who want broader garden
coverage can train on the full 14-species Plus dataset.

## What you download

| Artifact (Zenodo) | Unzips to | Contents |
|---|---|---|
| `mygardenbird16khz.zip` | `mygardenbird16khz/` | 12 core species (7,200 clips) |
| `mygardenbirdplus16khz.zip` | `mygardenbirdplus16khz/` | 2 addendum species (1,200 clips) + `metadata/` |

- **Core (12 species)**: https://doi.org/10.5281/zenodo.20306877
- **Plus addendum (this folder)**: https://doi.org/10.5281/zenodo.18694053

The `metadata/` folder in this addendum already contains the **combined
14-species** `recordings.csv` (1,610 recordings) and the combined 14-species
`splits_mip_80_10_10.csv` (8,400 clips), so no re-splitting is required.

## Constructing the 14-species dataset

### Option A — merge the species folders (training-ready)

Download and unzip both archives, then move the two addendum species into the
core folder so all 14 species sit side by side:

```bash
# 1. Download both zips from Zenodo, then unzip
unzip mygardenbird16khz.zip          # -> mygardenbird16khz/   (12 species)
unzip mygardenbirdplus16khz.zip      # -> mygardenbirdplus16khz/ (2 species + metadata)

# 2. Merge the two addendum species into the core folder
mv "mygardenbirdplus16khz/Common Myna" mygardenbird16khz/
mv "mygardenbirdplus16khz/Zebra Dove"  mygardenbird16khz/

# 3. (optional) rename the merged root to make the 14-species edition explicit
mv mygardenbird16khz mygardenbirdplus16khz_full
```

Result — 14 species directories, 8,400 WAV clips:

```
mygardenbirdplus16khz_full/
├── Asian Koel/                  ┐
├── Collared Kingfisher/         │
├── Common Iora/                 │
├── Common Tailorbird/           │
├── Coppersmith Barbet/          │ 12 core species
├── Large-tailed Nightjar/       │ (regional provenance)
├── Olive-backed Sunbird/        │
├── Pied Fantail/                │
├── Spotted Dove/                │
├── White-breasted Waterhen/     │
├── White-throated Kingfisher/   │
├── Yellow-vented Bulbul/        ┘
├── Common Myna/                 ┐ 2 addendum species
└── Zebra Dove/                  ┘ (no regional constraint)
```

### Option B — symlink without copying (saves disk)

If you keep the core and addendum unpacked separately, build a combined root of
symlinks instead of moving files:

```bash
ROOT=mygardenbirdplus16khz_full
mkdir -p "$ROOT"
ln -s "$PWD"/mygardenbird16khz/*/        "$ROOT"/
ln -s "$PWD"/mygardenbirdplus16khz/"Common Myna" "$ROOT"/
ln -s "$PWD"/mygardenbirdplus16khz/"Zebra Dove"   "$ROOT"/
```

### Verify the construction

```bash
# 14 species directories
ls mygardenbirdplus16khz_full | wc -l        # -> 14

# 8,400 clips total (follow symlinks with -L if you used Option B)
find -L mygardenbirdplus16khz_full -name '*.wav' | wc -l   # -> 8400

# every clip in the split CSV resolves to a file, and vice-versa
python - <<'PY'
import csv, io, os, glob
root = "mygardenbirdplus16khz_full"
csv_path = "mygardenbirdplus16khz/metadata/splits_mip_80_10_10.csv"
rows = [l for l in open(csv_path) if not l.startswith('#')]
want = {'xc'+r['file_id'][2:]+'.wav' for r in csv.DictReader(io.StringIO(''.join(rows)))}
have = {os.path.basename(p) for p in glob.glob(os.path.join(root, '*', '*.wav'))}
print("split clips:", len(want), "| files:", len(have))
print("missing files:", len(want - have), "| files not in split:", len(have - want))
PY
```

A correct build prints `split clips: 8400 | files: 8400` with `0` missing and
`0` extra.

## Combined split (80:10:10, source-separated)

`metadata/splits_mip_80_10_10.csv` is the MIP-optimal 80:10:10 split over all
14 species. All clips from the same Xeno-canto recording (`source_id`) are kept
in the same split, so there is **no source-level leakage**.

| Split | Clips | Sources |
|---|---:|---:|
| train | 6,720 | 1,169 |
| val   |   840 |   166 |
| test  |   840 |   275 |
| **total** | **8,400** | **1,610** |

## Training on the Plus dataset

`pipeline/Stage9_train_mygardenbird_multifeature.py` reads `dataset_root/<species>/*.wav`
and keeps only the files listed in `--splits_csv`. Classes are auto-derived from
the 14 subdirectories. Point it at the merged root and the combined split:

```bash
python pipeline/Stage9_train_mygardenbird_multifeature.py \
    --model efficientnetb0 --feature mel --seed 42 \
    --dataset_root  /path/to/mygardenbirdplus16khz_full \
    --splits_csv    mygardenbirdplus16khz/metadata/splits_mip_80_10_10.csv \
    --output_dir    ./resultsplus_16k
```

The script appends the platform name to `--output_dir`, so on Linux the run
lands in `resultsplus_16k_linux/` (kept local, not published). The benchmark
summary for the 3-models × 3-augmentations × 3-seeds grid on this 14-species
dataset is in
[`benchmark_summaries/plus_16k_14species.md`](../benchmark_summaries/plus_16k_14species.md).

## Metadata schema

### `metadata/recordings.csv` (14 species, 1,610 recordings)

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

### `metadata/splits_mip_80_10_10.csv` (14 species, 8,400 clips)

| Field | Type | Description |
|---|---|---|
| `file_id` | string | Clip identifier `XC{source_id}_{onset_ms}`; WAV is `xc{source_id}_{onset_ms}.wav` |
| `split` | string | `train` / `val` / `test` |

### `metadata/clips.csv` (addendum-only, 1,200 clips)

Per-clip QC metrics for the two addendum species. Schema identical to the core
`metadata16khz/clips.csv` (`file_id, source_id, onset_ms, sampling_rate,
snr_db, rms_db, peak_amplitude, is_clipped`). Concatenate with the core
`metadata16khz/clips.csv` for full 14-species per-clip QC.

## Licence

All source recordings were obtained from Xeno-canto under Creative Commons
licences; per-recording identifiers are in `recordings.csv` (`cc_license`
column). MyGardenBird-Plus is released under **CC BY-NC-SA 4.0**, consistent
with the core dataset.
