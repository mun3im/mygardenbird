# Extras

Additional utility scripts for the MyGardenBird dataset.

## Scripts

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

### `reverse_annotate.py`

Converts MyGardenBird clip metadata back to Audacity label format for the original Xeno-canto source recordings.

**Purpose:**
Useful for reviewing which segments were extracted from each source recording, or for creating new annotation workflows.

**Usage:**
```bash
python reverse_annotate.py
```

**Output:**
Generates Audacity-compatible label files showing the extracted clip positions within each source recording.

---

## Notes

These scripts are provided as supplementary tools for dataset verification and annotation management. They are not required for using the dataset but may be helpful for:
- Verifying dataset integrity
- Understanding the segmentation workflow
- Creating custom annotation pipelines
