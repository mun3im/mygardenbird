"""
config.py — single shared configuration for the SEAbird pipeline.

Contains:
  - Storage paths  (edit PROJECT_ROOT / DATASET_NAME for your setup)
  - Species catalogue  (loaded from DATASET_ROOT/target_species.csv)
  - Shared helpers used across stages

Directory layout produced by the pipeline:

  PROJECT_ROOT/
  └── DATASET_NAME/                     (e.g. /Volumes/Evo/MYGARDENBIRD)
      ├── project_csv/                  Centralized project metadata
      │   ├── target_species.csv
      │   ├── recordings.csv            Source recording metadata (shared)
      │   ├── regional_ranking.csv      Stage 1 — species ranking
      │   └── regional_ranking.md
      ├── per_species_csv/              Stage 1 — XC metadata CSVs (per species)
      │   ├── Species_Name.csv
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
from dataclasses import dataclass, field
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

# Project-level metadata (centralized)
PROJECT_CSV       = DATASET_ROOT / "project_csv"

# Source data directories
PER_SPECIES_CSV   = DATASET_ROOT / "per_species_csv"
PER_SPECIES_FLACS = DATASET_ROOT / "per_species_flacs"

# Extracted audio clips (training-ready, organized by species)
MYGARDENBIRD_16K  = DATASET_ROOT / "mygardenbird16khz"
MYGARDENBIRD_44K  = DATASET_ROOT / "mygardenbird44khz"

# Metadata CSVs (clips.csv, qc_report.csv, splits)
METADATA_16K      = DATASET_ROOT / "metadata16khz"
METADATA_44K      = DATASET_ROOT / "metadata44khz"

# Top-level shared metadata (centralized in project_csv/)
RECORDINGS_CSV    = PROJECT_CSV / "recordings.csv"
REGIONAL_RANKING_CSV = PROJECT_CSV / "regional_ranking.csv"
REGIONAL_RANKING_MD  = PROJECT_CSV / "regional_ranking.md"

# Legacy aliases (for backward compatibility with existing scripts)
EXTRACTED_SEGS    = MYGARDENBIRD_16K
DATASET_DIR       = METADATA_16K
SPLITS_DIR        = DATASET_ROOT / "splits"  # Deprecated; splits now live in metadata dirs

# target_species.csv lives in project_csv/ (centralized metadata)
_SPECIES_CSV = PROJECT_CSV / "target_species.csv"

# =============================================================================
# DATASET PROFILES — multi-dataset support
# =============================================================================
#
# The pipeline originally supported only one dataset (MyGardenBird, above).
# A second, newer dataset (SEA-BIRD30, a superset with more species) needed
# the same stages, which used to mean forking scripts with hand-edited
# paths. DatasetProfile makes each script's path resolution keyed off
# `--dataset {mygardenbird,sea-bird30}` instead, so one canonical copy of
# each script serves both. See each dataset's own README/CLAUDE.md for
# what it actually contains; this is just the path/naming shape.


@dataclass
class DatasetProfile:
    name: str
    dataset_root: Path
    project_csv_dir: Path
    per_species_csv: Path | None   # None if this dataset has no per-species XC CSV dir
    per_species_flacs: Path
    species_csv: Path
    recordings_csv: Path
    clips_dirs: dict                # {16000: Path, 44100: Path, ...}
    metadata_dirs: dict             # same shape as clips_dirs
    folder_styles: dict             # role -> "space" | "underscore", e.g. {"per_species_flacs": "underscore"}
    species_overrides: dict = field(default_factory=dict)  # e.g. {"Pied Fantail": "Malaysian_Pied_Fantail"}


DATASET_PROFILES = {
    "mygardenbird": DatasetProfile(
        name="mygardenbird",
        dataset_root=DATASET_ROOT,
        project_csv_dir=PROJECT_CSV,
        per_species_csv=PER_SPECIES_CSV,
        per_species_flacs=PER_SPECIES_FLACS,
        species_csv=_SPECIES_CSV,
        recordings_csv=RECORDINGS_CSV,
        clips_dirs={16000: MYGARDENBIRD_16K, 44100: MYGARDENBIRD_44K},
        metadata_dirs={16000: METADATA_16K, 44100: METADATA_44K},
        folder_styles={"per_species_flacs": "space", "clips": "space"},
    ),
    "sea-bird30": DatasetProfile(
        name="sea-bird30",
        dataset_root=Path("/Volumes/Evo/SEA-BIRD30"),
        project_csv_dir=Path("/Volumes/Evo/SEA-BIRD30/metadata"),
        per_species_csv=None,   # SEA-BIRD30 has no per-species XC CSV dir
        per_species_flacs=Path("/Volumes/Evo/SEA-BIRD30/flacs"),
        species_csv=Path("/Volumes/Evo/SEA-BIRD30/metadata/target_species.csv"),
        recordings_csv=Path("/Volumes/Evo/SEA-BIRD30/metadata/recordings.csv"),
        clips_dirs={16000: Path("/Volumes/Evo/SEA-BIRD30/wavs")},
        metadata_dirs={16000: Path("/Volumes/Evo/SEA-BIRD30/metadata")},
        folder_styles={"per_species_flacs": "underscore", "clips": "space"},
        species_overrides={"Pied Fantail": "Malaysian_Pied_Fantail"},
    ),
}


def get_profile(name: str) -> DatasetProfile:
    """Look up a DatasetProfile by --dataset value. Raises ValueError with
    the valid choices listed if `name` isn't a known profile."""
    if name not in DATASET_PROFILES:
        raise ValueError(f"Unknown --dataset '{name}'. Choices: {list(DATASET_PROFILES)}")
    return DATASET_PROFILES[name]


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


def resolve_species_dir(base_dir: Path, common_name: str, style: str, overrides: dict | None = None) -> Path:
    """Find a species' folder under base_dir, generalized from SEA-BIRD30's
    regenerate_species_wavs.py find_flac_dir(). Three-tier resolution:
      1. explicit override (species_overrides dict)
      2. mechanical transform per `style` ("space" -> folder_name(), "underscore" -> spaces->underscores)
      3. case-insensitive directory scan fallback
    Raises FileNotFoundError if nothing matches.
    """
    overrides = overrides or {}
    if style == "underscore":
        candidate = overrides.get(common_name, common_name.replace(" ", "_"))
    else:  # "space" (default)
        candidate = overrides.get(common_name, folder_name(common_name))
    d = base_dir / candidate
    if d.is_dir():
        return d
    for other in base_dir.iterdir():
        if other.is_dir() and other.name.lower() == candidate.lower():
            return other
    raise FileNotFoundError(f"No folder for species '{common_name}' under {base_dir}")


def resolve_species(name: str, species_list=None):
    """Resolve a species by common name, scientific name, or eBird code (case-insensitive).

    Looks up against `species_list` if given (e.g. a profile-specific list
    loaded via `_load_species(profile.species_csv)`), else the default
    MyGardenBird module-level SPECIES list -- unchanged default behavior for
    any existing caller that doesn't pass species_list.

    Returns (common_name, scientific_name, ebird_code) or None.
    """
    if species_list is None:
        species_list = SPECIES
    lower = name.lower()
    for common, scientific, code in species_list:
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

    # --- DatasetProfile self-test: confirm both profiles load and resolve ---
    print()
    print("=" * 90)
    print("DATASET PROFILES")
    print("=" * 90)
    for profile_name, profile in DATASET_PROFILES.items():
        print(f"\n[{profile_name}] dataset_root={profile.dataset_root}")
        if not profile.species_csv.exists():
            print(f"  species_csv not found: {profile.species_csv} (skipping species load)")
            continue
        all_sp, active_sp = _load_species(profile.species_csv)
        print(f"  species_csv: {profile.species_csv}")
        print(f"  {len(all_sp)} species total, {len(active_sp)} active")
        # Exercise resolve_species_dir against this profile's per_species_flacs,
        # if it exists on disk, using a species known to need an override.
        test_species = "Pied Fantail"
        if profile.per_species_flacs.is_dir():
            try:
                resolved = resolve_species_dir(
                    profile.per_species_flacs, test_species,
                    profile.folder_styles.get("per_species_flacs", "space"),
                    profile.species_overrides,
                )
                print(f"  resolve_species_dir('{test_species}') -> {resolved}")
            except FileNotFoundError as e:
                print(f"  resolve_species_dir('{test_species}') -> not found: {e}")
        else:
            print(f"  per_species_flacs not found on disk: {profile.per_species_flacs} (skipping resolve check)")
