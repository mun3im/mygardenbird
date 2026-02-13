"""
config.py — single shared configuration for the SEAbird pipeline.

Contains:
  - Storage paths  (edit PROJECT_ROOT / DATASET_NAME for your setup)
  - Species catalogue  (loaded from DATASET_ROOT/target_species.csv)
  - Shared helpers used across stages

Directory layout produced by the pipeline:

  PROJECT_ROOT/
  └── DATASET_NAME/                     (e.g. /Volumes/Evo/SEABIRD)
      ├── target_species.csv
      ├── per_species_csv/              Stage 1 — XC metadata CSVs
      │   ├── Zebra Dove.csv
      │   └── ...
      ├── per_species_flacs/            Stage 2/3 — downloaded mono FLACs
      │   ├── Zebra Dove/
      │   │   ├── A/xc####.flac
      │   │   └── ...
      │   └── ...
      ├── extracted_segments/           Stage 6 — 3-second WAV clips
      │   ├── Zebra Dove/xc####_0.wav
      │   └── ...
      ├── dataset/                      Stage 7 — QC report + manifest
      │   ├── qc_report.csv
      │   └── seabird_dataset_manifest.csv
      └── splits/                       Stage 8 — train/val/test split CSVs
          └── seabird_splits_*.csv
"""

import csv
from pathlib import Path

# =============================================================================
# USER CONFIGURATION — change these two lines once for your setup
# =============================================================================

PROJECT_ROOT = "/Volumes/Evo"   # Root mount point / storage root
DATASET_NAME = "SEABIRD"        # Top-level folder inside PROJECT_ROOT

# =============================================================================
# DERIVED PATHS — do not edit below this line
# =============================================================================

DATASET_ROOT      = Path(PROJECT_ROOT) / DATASET_NAME

PER_SPECIES_CSV   = DATASET_ROOT / "per_species_csv"
PER_SPECIES_FLACS = DATASET_ROOT / "per_species_flacs"
EXTRACTED_SEGS    = DATASET_ROOT / "extracted_segments"
DATASET_DIR       = DATASET_ROOT / "dataset"
SPLITS_DIR        = DATASET_ROOT / "splits"

# target_species.csv lives in the dataset root, not alongside the scripts
_SPECIES_CSV = DATASET_ROOT / "target_species.csv"

# =============================================================================
# SPECIES CATALOGUE
# =============================================================================

VALID_QUALITIES = ["A", "B", "C", "D", "E"]


def _load_species(csv_path: Path):
    """Load species from target_species.csv.

    Returns (all_species, active_species) — each a list of
    (common_name, scientific_name, ebird_code) tuples.
    """
    all_species    = []
    active_species = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            common     = row["Common name"].strip()
            scientific = row["Scientific name"].strip()
            code       = row["eBird code"].strip()
            if not (common and scientific and code):
                continue
            entry = (common, scientific, code)
            all_species.append(entry)
            if row.get("active", "").strip().lower() == "yes":
                active_species.append(entry)
    return all_species, active_species


SPECIES, ACTIVE_SPECIES = _load_species(_SPECIES_CSV)


def folder_name(common_name: str) -> str:
    """Return a filesystem-safe folder name from a common name.

    Replaces characters that are unsafe on common OSes.
    """
    return common_name.replace("/", "-").replace(":", "-")


def resolve_species(name: str):
    """Resolve a species by common name, scientific name, or eBird code (case-insensitive).

    Returns (common_name, scientific_name, ebird_code) or None.
    """
    lower = name.lower()
    for common, scientific, code in SPECIES:
        if lower in (common.lower(), scientific.lower(), code.lower()):
            return (common, scientific, code)
    return None


if __name__ == "__main__":
    active_codes = {code for _, _, code in ACTIVE_SPECIES}
    print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"DATASET_NAME  : {DATASET_NAME}")
    print(f"DATASET_ROOT  : {DATASET_ROOT}")
    print()
    print(f"{'#':<4} {'Common Name':<35} {'Scientific Name':<30} {'Code':<10} {'Active'}")
    print("-" * 90)
    for i, (common, scientific, code) in enumerate(SPECIES, 1):
        flag = "yes" if code in active_codes else ""
        print(f"{i:<4} {common:<35} {scientific:<30} {code:<10} {flag}")
    print(f"\nTotal: {len(SPECIES)} species, {len(ACTIVE_SPECIES)} active")
