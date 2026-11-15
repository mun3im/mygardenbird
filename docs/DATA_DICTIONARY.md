# Data dictionary

Every CSV the pipeline reads or writes, and how they relate.

Four of these are published with the dataset; two are working files that stay
on local storage. All paths are relative to
`$PROJECT_ROOT/$DATASET_NAME/` as set in `pipeline/config.py`.

| File | Role | Published? |
|---|---|---|
| [`project_csv/recordings.csv`](#1-project_csvrecordingscsv-source-recordings) | one row per source recording | yes |
| [`metadata{16,44}khz/clips.csv`](#2-metadata1644khzclipscsv-extracted-clips) | one row per extracted clip | yes |
| [`metadata{16,44}khz/splits_mip_*.csv`](#3-metadata1644khzsplits_mip_csv-partitions) | partition assignment | yes |
| [`metadata{16,44}khz/qc_report.csv`](#4-metadata1644khzqc_reportcsv-qc-diagnostics) | per-clip QC diagnostics | yes |
| [`project_csv/target_species.csv`](#5-project_csvtarget_speciescsv-species-list) | the species list every stage resolves against | no |
| [`per_species_csv/<Species>.csv`](#6-per_species_csvspecies_namecsv-xeno-canto-catalogue) | raw Xeno-canto catalogue | no |
| [`source_verification.csv`](#7-source_verificationcsv-provenance-audit) | per-source provenance audit | yes |

---

## Entity–relationship

```
  project_csv/recordings.csv
  ┌─────────────────────────────┐
  │ PK  source_id               │   one row per Xeno-canto recording
  │     species_common          │   provenance: who recorded it, where,
  │     species_scientific      │   under what licence, at what grade
  │     quality_grade           │
  │     cc_license              │
  │     latitude, longitude     │
  │     country, recordist      │
  │     xc_url, subspecies      │
  │     ebird_code, taxonomy_*  │
  └──────────────┬──────────────┘
                 │ 1
                 │
                 │ N        a recording yields one or more 3-second clips
                 ▼
  metadata{16,44}khz/clips.csv
  ┌─────────────────────────────┐
  │ PK  file_id                 │   = xc{source_id}_{onset_ms}
  │ FK  source_id   ────────────┼─► recordings.source_id
  │     onset_ms                │
  │     sampling_rate           │   per-clip signal measurements
  │     snr_db, rms_db          │
  │     peak_amplitude          │
  │     is_clipped              │
  └──────────────┬──────────────┘
                 │ 1
                 │
                 │ 1        every clip is assigned to exactly one partition
                 ▼
  metadata{16,44}khz/splits_mip_*.csv
  ┌─────────────────────────────┐
  │ FK  file_id     ────────────┼─► clips.file_id
  │     split                   │   train | val | test
  └─────────────────────────────┘

  qc_report.csv maps 1:1 onto clips.csv through `file` = file_id + ".wav".
  It is diagnostic output, not part of the relational schema.
```

**The foreign key is satisfied by construction, not enforced.**
`recordings.csv` is *derived* from the clips that exist, so it cannot contain a
source with no clips — but dropping clips without regenerating it leaves orphan
rows. An earlier release accumulated 506 stale rows exactly this way.

> ### The join trap
>
> `recordings.csv` stores `source_id` as **bare digits** (`1002657`), while
> `clips.csv` stores `file_id` with an **`xc` prefix** (`xc1002657_2860`) and
> `source_id` as bare digits again. Joining `clips.source_id` to
> `recordings.source_id` works directly; parsing the identifier out of
> `file_id` requires stripping the prefix first.
>
> ```python
> clips = pd.read_csv("metadata16khz/clips.csv", dtype={"source_id": str})
> recs  = pd.read_csv("project_csv/recordings.csv", dtype={"source_id": str})
> df = clips.merge(recs, on="source_id", how="left")
> ```
>
> Read `source_id` as a **string**. As an integer it is still unique, but it
> will not match the `xc`-prefixed forms used in filenames.

---

## 1. `project_csv/recordings.csv` — source recordings

One row per Xeno-canto recording that contributes at least one clip. **1,732
rows in v2.0.** Primary key `source_id`; the functional dependency
`source_id → species_common` holds, so a recording contributes to exactly one
class.

| Field | Type | Description | Example |
|---|---|---|---|
| `source_id` | str | Xeno-canto identifier, bare digits (**PK**) | `1000004` |
| `species_common` | str | Class label, English common name | `Scarlet-backed Flowerpecker` |
| `species_scientific` | str | Binomial under the declared taxonomy | `Dicaeum cruentatum` |
| `quality_grade` | str | Xeno-canto quality rating, `A`–`E` | `B` |
| `cc_license` | str | Creative Commons licence, SPDX-style | `CC-BY-NC-SA-4.0` |
| `type_label` | str | Vocalisation category | `call`, `song` |
| `latitude` | float | WGS84, blank where not supplied | `19.3442` |
| `longitude` | float | WGS84, blank where not supplied | `109.6846` |
| `country` | str | Country of recording | `China` |
| `recordist` | str | Contributor name — **attribution is due to this person** | `Dave Guo` |
| `xc_url` | str | Canonical page for the source recording | `https://xeno-canto.org/1000004` |
| `subspecies` | str | Archive-supplied `ssp` tag; **blank for 67.1% of rows** | `tristis` |
| `ebird_code` | str | eBird species code | `scbflo1` |
| `taxonomy_authority` | str | Checklist the labels follow | `eBird/Clements Checklist` |
| `taxonomy_version` | str | Checklist version | `v2024` |
| `taxonomy_access_date` | str | When the checklist was consulted | `2026-02-23` |

**On `subspecies`:** blank means the archive supplied no tag, not that the
subspecies is unknown to be absent. Two thirds of rows are blank, so
subspecific identity in this dataset is generally implied by locality rather
than verified — relevant to anyone filtering on it.

**Site grouping** uses the compound key `(recordist, round(latitude, 2),
round(longitude, 2))`, which is not a column but derived from three. It yields
905 sites over 1,708 fully attributed recordings. Rows missing `recordist` or
coordinates have no site and fall back to source-level grouping, which is
correct relationally: a missing value must not match another missing value.

---

## 2. `metadata{16,44}khz/clips.csv` — extracted clips

One row per released clip. **8,659 rows at 16 kHz, 8,287 at 44.1 kHz.** Same
schema for both.

| Field | Type | Description | Example |
|---|---|---|---|
| `file_id` | str | Clip identifier (**PK**), `xc{source_id}_{onset_ms}` | `xc1002657_2860` |
| `source_id` | str | Source recording (**FK** → `recordings.source_id`) | `1002657` |
| `onset_ms` | int | Clip start, ms from the beginning of the source FLAC | `2860` |
| `sampling_rate` | int | Hz | `16000` |
| `snr_db` | float | Estimated SNR; see the caveat below | `4.28` |
| `rms_db` | float | Whole-clip RMS level, dBFS | `-36.16` |
| `peak_amplitude` | float | Peak sample amplitude, 0–1 | `0.1111` |
| `is_clipped` | bool | `True` if `peak_amplitude > 0.99` | `False` |

`file_id` encodes both its own foreign key and the clip onset, so the filename
and the metadata cannot disagree. Every clip is exactly 3.000 s.

> **`snr_db` measures duty cycle as much as recording quality.** The noise
> floor is the 10th percentile of 50 ms frame energies, which assumes
> vocalisation-active frames are a minority of the clip. Where a call fills
> most of the clip that percentile lands on signal and SNR is understated —
> measured correlation with duty cycle is *r* = −0.50. Values are not
> comparable across clips of differing duty cycle.
>
> **A blank `snr_db` means a digitally silent clip**, which is a defect rather
> than a formatting problem. `grep ',,' clips.csv` is the cheapest detector.

**The 44.1 kHz file is a subset, not a resampling.** Sources recorded below
44.1 kHz are omitted rather than upsampled, so no clip in it has interpolated
high-frequency content.

---

## 3. `metadata{16,44}khz/splits_mip_*.csv` — partitions

One row per clip. Four partitions ship per sampling rate: `80_10_10`,
`75_10_15`, `70_15_15`, and `80_10_10_sitegrouped`.

| Field | Type | Description | Example |
|---|---|---|---|
| `file_id` | str | Clip (**FK** → `clips.file_id`) | `xc1002657_2860` |
| `split` | str | `train`, `val` or `test` | `val` |

**The first line is a `#` comment, not the header**, recording the ratio,
solver, seed and objective:

```
# split_ratio=80:10:10 seed=42 objective=1 solver=mip_cbc
file_id,split
```

Skip it when parsing — `pandas.read_csv(..., comment="#")`, or `csv` readers
need the line filtered first. A bare `DictReader` will read the comment as the
header and return zero usable rows.

**Use `splits_mip_80_10_10_sitegrouped.csv` for any published evaluation.** It
is the only partition free of site-level leakage: the others group by source
recording only, so two recordings made by the same person at the same place can
land on opposite sides of the split.

---

## 4. `metadata{16,44}khz/qc_report.csv` — QC diagnostics

One row per WAV file, mapping 1:1 onto `clips.csv` through `file` = `file_id` +
`.wav`. Diagnostic output, deliberately **not** part of the relational schema.

| Field | Type | Description |
|---|---|---|
| `species` | str | Species folder, English common name |
| `file` | str | WAV filename, `xc{source_id}_{onset_ms}.wav` |
| `valid` | bool | `True` if the file could be read and analysed |
| `error` | str | Read/analysis error message, blank if `valid` |
| `duration` | float | Clip duration, seconds |
| `sample_rate` | int | Hz |
| `num_samples` | int | Sample count |
| `rms_db` | float | Whole-clip RMS level, dBFS — unrounded |
| `peak_amplitude` | float | Peak sample amplitude, 0–1 — unrounded |
| `snr_db` | float | Estimated SNR, dB |
| `is_silent` | bool | `True` if RMS level < −40 dBFS |
| `is_clipped` | bool | `True` if peak amplitude > 0.99 |
| `duration_ok` | bool | `True` if duration is within 0.1 s of 3.0 s |

`snr_db` and `is_clipped` are identical to the values in `clips.csv`; `rms_db`
and `peak_amplitude` are the unrounded versions of them.

**QC flags are informational. Nothing is filtered on them** — roughly 12.7% of
clips are flagged silent and about 2% clipped, and all of them are in the
release. Filter yourself if your application needs it.

---

## 5. `project_csv/target_species.csv` — species list

What `config.py` reads, and therefore what every stage resolves species names
against. **Not published**; it is configuration. 50 rows.

| Field | Type | Description | Example |
|---|---|---|---|
| `Common name` | str | Resolves in `--species` arguments | `Ashy Tailorbird` |
| `Scientific name` | str | Binomial | `Orthotomus ruficeps` |
| `eBird code` | str | eBird species code | `ashtai1` |
| `active` | str | `yes` if a dataset class, `no` otherwise | `no` |

A species absent from this file cannot be resolved: `resolve_species()` returns
`None` and the stage exits with `Unknown species`. Adding a species means
adding a row here first.

---



---

## 6. `per_species_csv/<Species_Name>.csv` — Xeno-canto catalogue

Produced by `pipeline/Stage1_xc_fetch_metadata.py`, one file per species, from
the XC API v3 (`https://xeno-canto.org/api/3/recordings`). These are the raw
catalogue records the pipeline selects from — upstream input, not released
data. Field names are Xeno-canto's own.

Note the filename convention: species **folders** use spaces (`Asian Koel/`)
while these CSVs use underscores (`Asian_Koel.csv`).

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | int | XC recording ID | `1072882` |
| `gen` | str | Genus | `Acridotheres` |
| `sp` | str | Species epithet | `tristis` |
| `ssp` | str | Subspecies (often blank) | `tristis` |
| `grp` | str | Taxonomic group | `birds` |
| `en` | str | English common name | `Common Myna` |
| `rec` | str | Recordist name | `Albert Lastukhin` |
| `cnt` | str | Country | `Uzbekistan` |
| `loc` | str | Locality description | `Kitab, Kitab District, Kashkadarya Province` |
| `lat` | float | Latitude (WGS84) | `39.1382` |
| `lon` | float | Longitude (WGS84) | `66.8735` |
| `alt` | str | Altitude in metres (sometimes blank) | `650` |
| `type` | str | Vocalisation type | `call`, `song`, `alarm call`, `flight call` |
| `sex` | str | Sex of individual (often blank) | `male`, `female` |
| `stage` | str | Life stage (often blank) | `adult`, `juvenile` |
| `method` | str | Recording method | `field recording` |
| `url` | str | XC page URL (protocol-relative) | `//xeno-canto.org/1072882` |
| `file` | str | Download URL. **Blank/null for blocked species** | `https://xeno-canto.org/1072882/download` |
| `file-name` | str | Original uploaded filename | `XC1072882-Acridotheres-tristis-140510_018-B.wav` |
| `lic` | str | Creative Commons licence URL | `//creativecommons.org/licenses/by-nc-nd/4.0/` |
| `q` | str | Quality rating | `A`, `B`, `C`, `D`, `E` |
| `length` | str | Duration as `M:SS` or `H:MM:SS` | `0:08`, `1:23`, `1:02:15` |
| `time` | str | Time of day recorded (`HH:MM`) | `07:00` |
| `date` | str | Date recorded (`YYYY-MM-DD`) | `2014-05-10` |
| `uploaded` | str | Date uploaded (`YYYY-MM-DD`) | `2026-01-16` |
| `also` | str | Other species heard (JSON list) | `["Pycnonotus goiavier"]` or `[]` |
| `rmk` | str | Recordist remarks (often blank) | `fighting` |
| `animal-seen` | str | Was the animal seen? | `yes`, `no`, `unknown` |
| `playback-used` | str | Was playback used? | `yes`, `no` |
| `temp` | float | Temperature in C (rarely populated) | |
| `regnr` | float | Registration number (rarely populated) | |
| `auto` | str | Automatic recording? | `yes`, `no` |
| `dvc` | str | Recording device (often blank) | `sound devices 702` |
| `mic` | str | Microphone (often blank) | `Telinga pro 9 mkII` |
| `smp` | int | Sample rate in Hz | `44100`, `48000` |
| `length_seconds` | float | Duration in seconds (computed from `length`) | `8.0` |

### Notes on the catalogue fields

- **Blocked species**: Oriental Magpie-Robin, Javan Myna, Black-naped Oriole have all metadata fields populated *except* `file` is null/blank. These cannot be downloaded.
- **`length_seconds`**: Added by Stage 1 during save. Parsed from the `length` field (`M:SS` -> seconds).
- **`sono`/`osci`**: Sonogram and oscillogram fields are dropped during save to reduce CSV size.
- **Quality ratings**: A (best) through E (worst). Assigned by XC community.

---

## 7. `source_verification.csv` — provenance audit

Produced by the per-identifier audit and published in
`mygardenbird_validation.zip`. One row per source FLAC on disk at curation time
— **3,152 rows**, more than `recordings.csv` because it includes recordings
that were evaluated and rejected.

| Field | Type | Description |
|---|---|---|
| `source_id` | str | Xeno-canto identifier |
| `folder_species` | str | The class the file is filed under on disk |
| `grade` | str | Quality-grade subfolder it sits in |
| `xc_scientific` | str | What the archive says it is — `gen` + `sp` |
| `xc_english` | str | Archive's English name |
| `country` | str | Country of recording |
| `lat`, `lon` | float | Coordinates, for the regional check |
| `licence` | str | Licence as the archive reports it |
| `quality` | str | Quality rating as the archive reports it |
| `ssp` | str | Archive-supplied subspecies tag |
| `segments` | int | Clips this source contributes |

The point of the table is the comparison between `folder_species` and
`xc_scientific`: a mismatch is a misfiled recording. **This check is why the
audit queries each identifier individually** rather than reading the
per-species catalogues — those are built with the regional filter applied, so a
recording from outside the region is absent from them entirely and a
catalogue-keyed sweep cannot see it.
