# Fetching Xeno-canto metadata and extending the dataset

How to pull catalogue metadata, where everything is cached, how the ND licence
filter is controlled, and what is involved in adding material to the published
v2.0 release.

If you only want the released data, you do not need any of this — download it
from <https://doi.org/10.5281/zenodo.20306876>.

---

## 1. Before anything: an API key

Xeno-canto's v3 API requires a key. The fetch stages look for one in three
places, in order:

1. `--api-key <key>` on the command line
2. the `XENO_API_KEY` environment variable
3. `pipeline/xc_key.txt`, a file containing the key and nothing else

The third is easiest for repeated use and is already in `.gitignore`. Request a
key from your Xeno-canto account page.

```bash
echo 'your-key-here' > pipeline/xc_key.txt
```

---

## 2. Where everything is stored

All paths derive from two constants at the top of `pipeline/config.py`:

```python
PROJECT_ROOT = "/Volumes/Evo"      # your storage root
DATASET_NAME = "MYGARDENBIRD"      # top-level folder inside it
```

Set those once and every stage follows. Nothing is written inside the
repository — the code and the data stay separate, which is why this repository
is small and the dataset lives on Zenodo.

```
$PROJECT_ROOT/$DATASET_NAME/
├── project_csv/
│   ├── target_species.csv        the species list every stage resolves against
│   ├── recordings.csv            one row per contributing source (Stage 6)
│   └── regional_ranking{.csv,.md}  availability ranking (Stage 1)
├── per_species_csv/              catalogue metadata, ONE CSV PER SPECIES
│   ├── Asian_Koel.csv            underscores, not spaces
│   └── global_stats.json         pre-filter totals for every species
├── per_species_flacs/            downloaded audio + annotations
│   └── Asian Koel/               spaces, not underscores
│       ├── A/  xc1002657.flac    one subfolder per Xeno-canto quality grade
│       │       xc1002657.txt     the annotation sidecar, beside its audio
│       └── B/  ...
├── mygardenbird16khz/            extracted clips (Stage 5)
│   └── Asian Koel/xc1002657_2860.wav
├── mygardenbird44khz/
└── metadata{16,44}khz/           clips.csv, qc_report.csv, splits (Stages 6, 8)
```

Three conventions worth knowing, because they trip people up:

- **Species folders use spaces; species CSVs use underscores.** `Asian Koel/`
  but `Asian_Koel.csv`. `config.folder_name()` does the conversion.
- **The clip tree is flat; the FLAC tree is not.** Audio is filed by grade
  (`A/`, `B/`, …) but extracted clips are not — all of a species' clips sit in
  one directory. A clip filename, `xc{source_id}_{onset_ms}.wav`, encodes
  everything needed to trace it back.
- **An annotation sidecar lives beside its FLAC**, named identically but
  `.txt`. Its presence and size carry meaning: **absent** means not yet
  evaluated, **zero-length** means the recording was listened to and rejected,
  and **non-empty** means accepted, one `start<TAB>end<TAB>label` line per
  segment. A zero-length file is a decision, not a failure — do not delete
  them.

---

## 3. Fetching catalogue metadata (Stage 1)

Stage 1 queries the API for every species in `target_species.csv` and writes one
CSV per species into `per_species_csv/`.

```bash
# everything in the species list
python pipeline/Stage1_xc_fetch_metadata.py --species all

# one species, or a few
python pipeline/Stage1_xc_fetch_metadata.py --species "Asian Koel" "Common Iora"

# see what would be fetched without calling the API
python pipeline/Stage1_xc_fetch_metadata.py --species all --dry-run
```

The written CSVs hold the **regionally filtered** records: longitude
60°–140°E, latitude north of 11°S, or an ASEAN country of origin where
coordinates are missing, and at least 3 seconds long. The filter bounds are
flags (`--lon-min`, `--lon-max`, `--lat-min`, `--min-duration`), and
`--no-region-filter` disables geography entirely.

> **`--species` with a subset overwrites `global_stats.json`.**
> That file holds pre-filter totals for *all* species, and a partial run
> rewrites it with only the species it fetched — after which the ranking
> reports 0 global hours for everything else. If you fetch a subset, either
> keep a copy of `global_stats.json` first or re-run with `--species all`
> afterwards. `--rank-only` rebuilds the ranking from existing CSVs without
> touching the API.

**There is no northern latitude bound.** The −11° floor exists to exclude
northern Australia; Chinese and Taiwanese recordings are inside the filter by
design. Do not add a northern bound without checking which species that
silently removes.

### A per-species catalogue cannot always be built

Xeno-canto has not adopted every taxonomic split this dataset follows. For the
Ornate Sunbird, `sp:"Cinnyris ornatus"` returns nothing — all material is still
filed under *Cinnyris jugularis* with the daughter taxa recorded only as `ssp`
tags. A Stage 1 run for such a species writes an empty CSV and ranks it at zero
hours. The workaround is to fetch the parent species and filter by `ssp` and
locality afterwards.

---

## 4. The ND licence filter

**Segmenting a recording produces a derivative work.** Recordings licensed
CC BY-NC-ND do not permit a derivative to be *distributed*, so they are
excluded from the released dataset.

The exclusion happens at download, not at fetch: Stage 1 writes every regional
record to the catalogue CSVs, including ND ones, and Stages 2 and 2a skip them
when selecting what to download.

```bash
# default: ND recordings are skipped, and the count is printed
python pipeline/Stage2_xc_dload_all_from_species_list.py --species "Asian Koel"

# include them (see the warning below)
python pipeline/Stage2_xc_dload_all_from_species_list.py --species "Asian Koel" --no-skip-nd
```

Stage 2a takes the same flag. Both print how many recordings were skipped and
their identifiers, so the decision is visible in the log rather than silent.

> **If you pass `--no-skip-nd`, the resulting clips cannot be redistributed.**
> NoDerivatives permits a derivative to be made but not distributed, so
> material obtained this way is yours to train on privately. It must not be
> deposited, published, or shared, and a dataset containing it must not be
> passed on. See `NOTICE` in the repository root.
>
> This is a real option rather than a hypothetical one: the four classes that
> fall short of 600 clips in v2.0 would all pass it with ND material included.
> That is a licensing ceiling, not a shortage of audio.

Downstream stages do not inspect licences. Once a recording is on disk, nothing
distinguishes it, so the decision taken here is the only one that matters —
which is why the flag is on the download step rather than somewhere later.

---

## 5. Downloading audio (Stages 2 and 2a)

**Stage 2** downloads everything in the catalogue CSVs for the species you
name, converting each to mono FLAC:

```bash
python pipeline/Stage2_xc_dload_all_from_species_list.py \
    --species "Asian Koel" --quality A B
```

**Stage 2a** downloads specific identifiers — the right tool for topping up a
species after a catalogue refresh:

```bash
python pipeline/Stage2a_xc_dload_delta_by_id.py \
    --species "Asian Koel" --id-file new_ids.txt
```

Both skip files already on disk, so re-running is safe and cheap. Both take
`--dry-run`.

> **One recording, one grade folder.** Xeno-canto grades can change after
> upload. If a recording is re-downloaded after its grade changed, it lands in
> a second folder while the first copy remains, and because the clip tree is
> flat both copies' clips then coexist under one namespace — the same audio
> segmented twice, which is a leakage risk across a train/test split. Check
> after any bulk download:
>
> ```bash
> find "per_species_flacs/<species>" -name '*.flac' \
>   | xargs -n1 basename | sort | uniq -d
> ```

---

## 6. The per-identifier metadata cache

The per-species CSVs are built *with the regional filter applied*, so a
recording outside the region is simply absent from them. A sweep keyed on those
catalogues therefore cannot see such a recording at all — which is how one
Australian recording survived two licence audits while filed under a Malaysian
species.

The remedy is a per-identifier cache: one `nr:<id>` query per source, which is
unfiltered and always answers.

```
$PROJECT_ROOT/$DATASET_NAME/xc_metadata_YYYYMMDD/
├── by_id/1002657.json         one file per source, straight from the API
└── source_verification.csv    one row per FLAC on disk
```

`source_verification.csv` is the table deposited with the release. Each row
records four checks: species (the folder against the archive's `gen` + `sp`),
region (the same filter Stage 1 applies), licence (no CC `*-ND`), and whether
the identifier resolves at all.

**Re-run this after any promotion or bulk download.** The species check is the
one that catches misfiling, and it is the only check that works regardless of
region — several wrong-species recordings in this corpus were *inside* the
regional filter, so a geography-only audit would have passed them.

Fetching is resumable: a rerun requests only the identifiers it does not
already have. Budget roughly 0.4 s per request.

---

## 7. Extending the released v2.0 dataset

Adding material means republishing, so it is worth being deliberate.

**1. Fetch, download, annotate.** Refresh the catalogue (§3), download what is
missing (§5), then annotate with `pipeline/Stage4_annotate_qt.py`. A source
contributes nothing until its sidecar exists.

**2. Re-run the chain in order.** Each stage reads the previous one's output:

```bash
python pipeline/Stage5_extract_annotated_segments.py . --sample-rate 16000 --clean
python pipeline/Stage6_clip_qc_manifest.py
python pipeline/Stage8_splitter_mip.py --group-by-site
python pipeline/Stage7_eval_birdnet.py
```

Run Stage 5 for each sampling rate. The 44.1 kHz tree is a *subset*: sources
recorded below 44.1 kHz are skipped rather than upsampled, so it is always
smaller, and that is correct rather than a sync failure.

**3. Verify before publishing.** Check that no source spans two partitions,
that no site does either in the site-grouped split, and that the sites number
well below the sources — a site leak of zero is only meaningful if site
grouping actually merges anything.

**4. Deposit as a new version**, not as edits to the existing one. Use
Zenodo's "New version" on the existing record so the concept DOI
(`10.5281/zenodo.20306876`) continues to resolve to the current release, and
earlier versions stay citable at their own DOIs.

> **Anything already published stays published.** A Zenodo version cannot be
> retracted, only superseded. If a release turns out to contain material that
> should not have been distributed, the correct response is a new version that
> removes it plus a disclosure — not a silent replacement, which would conceal
> a distribution that happened. v2.0 supersedes an earlier release for exactly
> this reason.

### Things that break quietly

- **Stage 6 writes `recordings.csv` only if it does not already exist.** Move
  the old file aside to force a rebuild, or it will silently describe the
  previous release.
- **Stage 6 does not emit the taxonomy columns.** `xc_url`, `subspecies`,
  `ebird_code`, `taxonomy_authority`, `taxonomy_version` and
  `taxonomy_access_date` are added by a separate pass that must be re-run after
  every Stage 6 rebuild.
- **A blank `snr_db` means a digitally silent clip**, not a formatting problem.
  It is the cheapest detector for zero-length annotations that the extraction
  step padded out to a full window.
- **Check the species map in `Stage7_eval_birdnet.py` after any class change.**
  It silently skips species directories it does not know, so a new class is
  absent from the evaluation rather than scored badly.
- **Do not run Stage 5 while Stage 5a is running.** `--clean` deletes every
  clip before re-extracting, and the confusion run will fail on the files it
  was reading.
