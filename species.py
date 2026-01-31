"""
species.py — shared species module for all Stage scripts.

Reads target_species.csv (same directory as this module) and provides:
    SPECIES         — list of (common_name, scientific_name, ebird_code) tuples (all 48)
    ACTIVE_SPECIES  — subset where the 'active' column is 'yes'
    VALID_QUALITIES — ["A", "B", "C", "D", "E"]
    resolve_species(name) -> tuple or None  (case-insensitive by common/scientific/code)
"""

import csv
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_SCRIPT_DIR, "target_species.csv")


def _load_species(csv_path):
    """Load species from CSV. Returns (all, active) tuple of lists.

    Each entry is (common_name, scientific_name, ebird_code).
    """
    all_species = []
    active_species = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            common = row["Common name"].strip()
            scientific = row["Scientific name"].strip()
            code = row["eBird code"].strip()
            if not (common and scientific and code):
                continue
            entry = (common, scientific, code)
            all_species.append(entry)
            if row.get("active", "").strip().lower() == "yes":
                active_species.append(entry)
    return all_species, active_species


SPECIES, ACTIVE_SPECIES = _load_species(_CSV_PATH)

VALID_QUALITIES = ["A", "B", "C", "D", "E"]


def folder_name(common_name):
    """Convert common name to a filesystem-safe folder name.

    E.g. "White-throated Kingfisher" -> "White-throated Kingfisher" (kept as-is;
    only characters unsafe on common OSes are replaced).
    """
    return common_name.replace("/", "-").replace(":", "-")


def resolve_species(name):
    """Resolve a species by common name, scientific name, or eBird code (case-insensitive).

    Returns (common_name, scientific_name, ebird_code) or None.
    """
    lower = name.lower()
    for common, scientific, code in SPECIES:
        if lower in (common.lower(), scientific.lower(), code.lower()):
            return (common, scientific, code)
    return None


if __name__ == "__main__":
    _active_codes = {code for _, _, code in ACTIVE_SPECIES}
    print(f"{'#':<4} {'Common Name':<35} {'Scientific Name':<30} {'Code':<10} {'Active'}")
    print("-" * 90)
    for i, (common, scientific, code) in enumerate(SPECIES, 1):
        flag = "yes" if code in _active_codes else ""
        print(f"{i:<4} {common:<35} {scientific:<30} {code:<10} {flag}")
    print(f"\nTotal: {len(SPECIES)} species, {len(ACTIVE_SPECIES)} active")
