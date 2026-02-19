"""
config.py — single shared configuration for the SEAbird pipeline.

Contains:
  - Storage paths  (edit PROJECT_ROOT / DATASET_NAME for your setup)
  - Species catalogue  (loaded from DATASET_ROOT/target_species.csv)
  - Shared helpers used across stages

Directory layout produced by the pipeline:

  PROJECT_ROOT/
  └── DATASET_NAME/                     (e.g. /Volumes/Evo/MYGARDENBIRD)
      ├── target_species.csv
      ├── recordings.csv                Top-level — source recording metadata (shared)
      ├── per_species_csv/              Stage 1 — XC metadata CSVs
      │   ├── Species Name.csv
      │   └── ...
      ├── per_species_flacs/            Stage 2/3 — downloaded mono FLACs + annotations
      │   ├── Species Name/
      │   │   ├── A/xc####.flac
      │   │   │   xc####.txt
      │   │   └── ...
      │   └── ...
      ├── mygardenbird16khz/            Stage 6 (16kHz) — 3-second WAV clips
      │   ├── Species Name/xc####_0.wav
      │   └── ...
      ├── mygardenbird44khz/            Stage 6 (44kHz) — 3-second WAV clips
      │   ├── Species Name/xc####_0.wav
      │   └── ...
      ├── metadata16khz/                Stage 7 (16kHz) — QC + manifest + splits
      │   ├── clips.csv
      │   ├── qc_report.csv
      │   └── splits_mip_75_10_15.csv
      └── metadata44khz/                Stage 7 (44kHz) — QC + manifest + splits
          ├── clips.csv
          ├── qc_report.csv
          └── splits_mip_75_10_15.csv
"""

import csv
from pathlib import Path

# =============================================================================
# USER CONFIGURATION — change these two lines once for your setup
# =============================================================================

PROJECT_ROOT = "/Volumes/Evo"   # Root mount point / storage root
DATASET_NAME = "MYGARDENBIRD"        # Top-level folder inside PROJECT_ROOT

# =============================================================================
# DERIVED PATHS — do not edit below this line
# =============================================================================

DATASET_ROOT      = Path(PROJECT_ROOT) / DATASET_NAME

# Source data directories
PER_SPECIES_CSV   = DATASET_ROOT / "per_species_csv"
PER_SPECIES_FLACS = DATASET_ROOT / "per_species_flacs"

# Extracted audio clips (training-ready, organized by species)
MYGARDENBIRD_16K  = DATASET_ROOT / "mygardenbird16khz"
MYGARDENBIRD_44K  = DATASET_ROOT / "mygardenbird44khz"

# Metadata CSVs (clips.csv, qc_report.csv, splits)
METADATA_16K      = DATASET_ROOT / "metadata16khz"
METADATA_44K      = DATASET_ROOT / "metadata44khz"

# Top-level shared metadata
RECORDINGS_CSV    = DATASET_ROOT / "recordings.csv"

# Legacy aliases (for backward compatibility with existing scripts)
EXTRACTED_SEGS    = MYGARDENBIRD_16K
DATASET_DIR       = METADATA_16K
SPLITS_DIR        = DATASET_ROOT / "splits"  # Deprecated; splits now live in metadata dirs

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


_SONG_PREFIXES   = ("song", "dawn song", "subsong", "sub-song", "duet")
_CALL_SUBSTRINGS = ("call",)


def normalise_type(raw_type: str) -> str:
    """Map a raw XC 'type' field to one of: song | call | other.

    Takes the first comma-separated token (the primary type), lowercases it,
    then classifies:
      - starts with a song prefix          → "song"
      - contains the word "call"           → "call"
      - everything else (wing beats, etc.) → "other"
    Falls back to "other" if the field is blank.
    """
    if not raw_type:
        return "other"
    primary = raw_type.split(",")[0].strip().lower()
    if any(primary.startswith(p) for p in _SONG_PREFIXES):
        return "song"
    if any(s in primary for s in _CALL_SUBSTRINGS):
        return "call"
    return "other"


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
