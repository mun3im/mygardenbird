# Extras

Additional utility scripts for the MyGardenBird dataset analysis, verification, and visualization.

## Dataset Verification Scripts

### `verify_structural_integrity_final.py`

Verifies all structural integrity claims from Section 5.2 (Technical Validation) of the paper.

**Verifies:**
1. All ten species contribute exactly 600 clips
2. No duplicate file_id values
3. No duplicate (source_id, clip_index) pairs
4. Zero overlap between train/val/test sets at source-recording level
5. No source_id appears in more than one partition

**Usage:**
```bash
python verify_structural_integrity_final.py
```

**Requirements:**
- pandas
- Paths to clips.csv, recordings.csv, and splits_mip_80_10_10.csv

**Output:**
```
✅ ✅ ✅  ALL CLAIMS VERIFIED  ✅ ✅ ✅

All structural integrity claims in Section 5.2 are accurate:
  ✓ All ten species have exactly 600 clips
  ✓ No duplicate file_id values
  ✓ No duplicate (source_id, clip_index) pairs
  ✓ Zero overlap between train/val/test at source level
  ✓ No source_id appears in multiple partitions
```

### `reverse_annotate.py`

Converts MyGardenBird clip metadata back to Audacity label format for the original Xeno-canto source recordings.

**Purpose:**
Useful for reviewing which segments were extracted from each source recording, or for creating new annotation workflows.

**Usage:**
```bash
python reverse_annotate.py <dataset_root> [--output_dir <dir>] [--clip_ms <duration>]
```

**Arguments:**
- `dataset_root`: Path to dataset root containing species subdirectories
- `--output_dir`: Output directory (default: `<dataset_root>/annotations`)
- `--clip_ms`: Clip duration in milliseconds (default: 3000)

**Output:**
Generates Audacity-compatible label files (`.txt`) showing the extracted clip positions within each source recording.

**Example output (`xc19400.txt`):**
```
1.044    4.044    1
29.855   32.855   2
57.109   60.109   3
```

## Analysis Scripts

### `analyze_all_results.py`

Aggregates experimental results across all training runs to generate summary statistics.

**Purpose:**
- Collects test accuracy from all completed experiments
- Computes mean ± std across multiple random seeds
- Identifies missing experiments
- Generates completion status reports

**Usage:**
```bash
python analyze_all_results.py
```

**Output:**
- Per-model, per-augmentation accuracy statistics
- Experiment completion status (X/Y complete)
- List of missing experiments

### `analyze_licenses_fixed.py`

Analyzes Xeno-canto Creative Commons license distribution across all source recordings.

**Purpose:**
Documents the licensing constraints of source recordings to ensure dataset compliance.

**Findings:**
- 62.8% CC BY-NC-SA (allows derivatives)
- 36.5% CC BY-NC-ND (restricts derivatives)
- 0.7% more permissive (CC BY-SA, CC BY, CC0)

**Usage:**
```bash
python analyze_licenses_fixed.py
```

**Output:**
```
OVERALL LICENSE DISTRIBUTION
  CC BY-NC-SA 4.0: 705 (62.8%)
  CC BY-NC-ND 4.0: 410 (36.5%)
  CC BY-SA 4.0:      6 (0.5%)
  ...
```

## Plotting Scripts

### `plot_accuracy_vs_compute.py`

Generates Figure 8 from the paper: Accuracy vs computational cost (MFLOPs).

**Output:**
- `accuracy_vs_compute.pdf`: Scatter plot with error bars
- Shows efficiency frontier for MobileNetV3-Small, EfficientNet-B0, ResNet-50

**Usage:**
```bash
python plot_accuracy_vs_compute.py
```

### `plot_quality_country.py`

Generates Figure 3 from the paper: (a) Quality grade distribution, (b) Country of origin.

**Output:**
- `quality_country.pdf`: Dual-panel figure
  - Panel (a): Stacked bar chart of Xeno-canto quality grades per species
  - Panel (b): Bar chart of source recordings by country

**Usage:**
```bash
python plot_quality_country.py
```

### `plot_spectrograms.py`

Generates Figure 5 from the paper: Representative mel-spectrograms for all 10 species.

**Purpose:**
Creates a 2×5 grid showing median-SNR spectrograms (typical examples, not best-case).

**Output:**
- `example_spectrograms.pdf`: Multi-panel mel-spectrogram figure

**Usage:**
```bash
python plot_spectrograms.py
```

**Requirements:**
- librosa
- matplotlib
- numpy

---

## Notes

These scripts are provided as supplementary tools for dataset verification, analysis, and visualization. They are not required for using the dataset but may be helpful for:

- **Researchers**: Verifying dataset integrity and reproducing paper figures
- **Developers**: Understanding the curation workflow and data structure
- **Annotators**: Creating custom annotation pipelines or reviewing segmentation
- **Reviewers**: Validating all technical claims in the paper

All scripts assume the standard MyGardenBird directory structure with `metadata16khz/`, `metadata44khz/`, and `project_csv/` folders.
