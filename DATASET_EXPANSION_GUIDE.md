# Dataset Expansion Guide: 10 → 12 Species

This guide documents the addition of **Yellow-vented Bulbul** (*Pycnonotus goiavier*) and
**Pied Fantail** (*Rhipidura javanica*) to the MyGardenBird dataset, expanding it from
10 to **12 species** (6,000 → **7,200 clips**).

Both species appear in the Malaysia Nature Society MY Garden Birdwatch 30-species
identification guide and have sufficient regional Xeno-canto coverage (see
`regional_ranking.csv`, ranks 9 and 13 respectively).

---

## New Species Summary

| Species | Scientific name | eBird code | Regional hrs | Regional files | Grade A | Grade B |
|---|---|---|---|---|---|---|
| Yellow-vented Bulbul | *Pycnonotus goiavier* | `yevbul1` | 1.76 | 136 | 19 | 58 |
| Pied Fantail | *Rhipidura javanica* | `piefan1` | 1.32 | 142 | 28 | 96 |

Target: **600 clips per species** (same as existing species), max 10 clips per source
recording, manual spectrogram review of every clip in Audacity via Stage 5.

---

## Pipeline Stages

### Stage 1 — Fetch Xeno-canto metadata *(if not done)*

```bash
python Stage1_xc_fetch_metadata.py
```

Metadata for both species is already present in `regional_ranking.csv`. Run Stage 1
only if you want a fresh pull to capture any new uploads since the original scrape.
Filter by ASEAN region / longitudes 60°–140°.

---

### Stage 2 — Download recordings

```bash
python Stage2_xc_dload_all_from_species_list.py
```

Add `yevbul1` and `piefan1` to the species list in `config.py` before running.
Downloads are saved as FLAC (converted from MP3 immediately on download to preserve
lossless audio for all downstream stages).

Prioritise **Grade A then Grade B** recordings. Grade C is acceptable to reach the
600-clip target but should be used last.

---

### Stage 3 — Download delta by ID *(optional)*

```bash
python Stage3_xc_dload_delta_by_id.py
```

Use this if Stage 2 missed specific XC IDs that you want to add individually.

---

### Stage 4 — Audit downloads

```bash
python Stage4_audit_downloads.py
```

Confirms all downloaded FLACs match their metadata entries. Fix any corrupt or
zero-byte files before proceeding to annotation.

---

### Stage 5 — Interactive segmentation GUI

```bash
python Stage5_find_segments_interactive.py --sound-dir /path/to/new_species_flacs
```

**Annotation protocol** (same as existing species):

- Tune *Median Threshold* and *Max Segment Gap* sliders per recording
- Accept a segment only when the target vocalisation is **clearly audible**, species
  identity is **unambiguous**, and there is **no severe heterospecific overlap**
- Extract no more than **10 clips per source recording**
- For long vocalisations (> 3 s) take consecutive 3-second clips
- For recordings with multiple short calls, prefer **acoustic diversity** (different call
  types, song variants) over quantity — do not repeat the same motif
- Reject clips with digital clipping (peak > 0.99) only if the vocalisation is
  unintelligible; otherwise retain and let the QC manifest flag them

Save the Audacity-format label file for each recording before moving to the next.

> **Yellow-vented Bulbul note:** Vocalisations are variable (melodic phrases + harsh
> churring calls). Select a diverse mix of call types to maximise repertoire coverage.
>
> **Pied Fantail note:** Often recorded near water/mangroves; fan-tail displays
> accompanied by soft chattering calls. Exclude clips dominated by water/insect noise
> with no audible bird vocalisation.

---

### Stage 6 — Extract annotated clips

```bash
python Stage6_extract_annotated_segments.py
```

Extracts the labelled regions as 3.000-second, 16-bit PCM mono WAV files at:

- **16 kHz** (master set, all clips) → `MyGardenBird-16k/<species_name>/`
- **44.1 kHz** (high-resolution subset, only if native sample rate ≥ 44.1 kHz) →
  `MyGardenBird-44k/<species_name>/`

Filenames follow the existing convention: `xc{source_id}_{onset_ms}.wav`

Species folder names to use:
- `Yellow-vented Bulbul/`
- `Pied Fantail/`

---

### Stage 7 — QC manifest

```bash
python Stage7_clip_qc_manifest.py
```

Run over the **full dataset** (all 12 species) to regenerate `clips.csv` and
`recordings.csv` with updated per-clip SNR, RMS, peak amplitude, and `is_clipped`
fields. Verify:

- Each new species reaches exactly **600 clips** in the 16 kHz set
- No duplicate `file_id` values
- SNR distribution for new species is comparable to existing species (median > 10 dB)

---

### Stage 8a — Re-run MIP splitter on the full 12-species dataset

```bash
python Stage8a_splitter_mip.py /path/to/MyGardenBird-16k \
    --train_ratio 0.80 --val_ratio 0.10 --test_ratio 0.10 \
    --output splits_mip_80_10_10.csv
```

> **Important:** Re-run on the *complete* 12-species dataset, not just the two new
> species. The MIP solver reassigns all 1,200+ source recordings jointly to guarantee
> exact class balance and zero cross-partition leakage.

Expected output:
- 5,760 train / 720 val / 720 test clips (480/60/60 per class × 12 classes)
- Solver objective = 0 (exact ratios achieved)
- Run with seeds 42, 100, 786 to produce three alternative splits

---

### Stage 9 — Retrain CNN baselines

```bash
bash runsweep.sh        # seed 42
bash runsweep100.sh     # seed 100
bash runsweep786.sh     # seed 786
```

Retrain all three architectures (MobileNetV3-Small, EfficientNet-B0, ResNet-50) on
the new 12-species split. Use the same hyperparameters as the original run
(see `TRAINING_CONFIG_GUIDE.md`): Mixup α=0.2, AdamW lr=1e-3, 50 epochs max,
early stopping patience 10, ImageNet initialisation.

Report mean ± s.d. test accuracy across the three seeds for both 16 kHz and 44.1 kHz
subsets.

---

## Paper Update Checklist

Once all stages are complete, update `zabidi2026mygardenbird.tex`:

### Numbers to change throughout

| Current | Replace with |
|---|---|
| `ten common bird species` | `twelve common bird species` |
| `6,000` clips (dataset total) | `7,200` |
| `5.0 hours` | `6.0 hours` |
| `$N = 10 \times 600 = 6{,}000$` | `$N = 12 \times 600 = 7{,}200$` |
| `10 species` (all occurrences) | `12 species` |

### Specific sections

- **Abstract** — update clip count, hours, species count, and class-balance sentence
- **`tab:species`** — add rows for Yellow-vented Bulbul (`yevbul1`) and Pied Fantail
  (`piefan1`) with 16 kHz clips, 44.1 kHz clips, and XC file counts; update Total row
- **`tab:shortlist`** — move both species from *Pending inclusion* block → *Selected*
  block; update block headers to `Selected (12 species)` and `Excluded (7 species)`;
  remove the `‡` footnote
- **`tab:splits`** — replace with new MIP output (5,760/720/720 total;
  480/60/60 per class)
- **`tab:accuracies`** — replace with new 3-seed CNN results for both sampling rates
- **Background & Summary** — update "6,000 class-balanced clips", "ten common…
  species", and "Table 1" description
- **Comparison table (`tab:otherds`)** — update MyGardenBird row: size 6,000 → 7,200
- **FAIR / Code Availability** — update any hardcoded clip counts

### Figure that may need regenerating

- `fig/example_spectrograms.pdf` — add panels for Yellow-vented Bulbul and Pied Fantail
- `fig/snr_distribution.pdf` — replot with 12 species
- `fig/quality_country.pdf` — replot with 12 species
- `fig/confusion_classification_composite.pdf` — replot with new CNN results
- `fig/accuracy_vs_compute.pdf` — replot if accuracy values change materially

---

## Zenodo Deposit

After the paper is accepted:

1. Upload the updated 16 kHz and 44.1 kHz WAV folders (with new species subfolders)
2. Upload updated `clips.csv`, `recordings.csv`, and `splits_mip_80_10_10.csv`
3. Update the Zenodo record description to reflect 12 species / 7,200 clips
4. The DOI remains the same (new version under the same concept DOI)
