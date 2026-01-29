import argparse
import os
import re
import sys

import requests

# Same species table as Stage1 for folder-structure consistency
SEABIRD_SPECIES = [
    ("Asian Koel",                "Eudynamys scolopaceus",   "asikoe2"),
    ("Collared Kingfisher",       "Todiramphus chloris",     "colkin1"),
    ("Common Iora",               "Aegithina tiphia",        "comior1"),
    ("Common Myna",               "Acridotheres tristis",    "commyn"),
    ("Common Tailorbird",         "Orthotomus sutorius",     "comtai1"),
    ("Large-tailed Nightjar",     "Caprimulgus macrurus",    "latnig2"),
    ("Olive-backed Sunbird",      "Cinnyris jugularis",      "olbsun4"),
    ("Spotted Dove",              "Spilopelia chinensis",    "spodov2"),
    ("White-throated Kingfisher", "Halcyon smyrnensis",      "whtkin2"),
    ("Zebra Dove",                "Geopelia striata",        "zebdov"),
]

VALID_QUALITIES = ["A", "B", "C", "D", "E"]


def resolve_species(name):
    """Resolve a species by common name, scientific name, or eBird code (case-insensitive)."""
    lower = name.lower()
    for common, scientific, code in SEABIRD_SPECIES:
        if lower in (common.lower(), scientific.lower(), code.lower()):
            return (common, scientific, code)
    return None


def parse_ids(raw_ids):
    """Extract numeric XC IDs from strings like 'XC123456', '123456', 'xc789'."""
    clean = []
    for raw in raw_ids:
        match = re.search(r"\d+", raw.strip())
        if match:
            clean.append(match.group(0))
        else:
            print(f"Warning: Skipping invalid ID '{raw}'")
    return clean


def load_ids_from_file(filepath):
    """Read XC IDs from a text file (one per line, blank lines and # comments ignored)."""
    ids = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def download_recording(xc_id, save_folder):
    """Download a single recording by XC ID into save_folder."""
    download_url = f"https://xeno-canto.org/{xc_id}/download"
    local_filename = os.path.join(save_folder, f"xc{xc_id}.mp3")
    if os.path.exists(local_filename):
        print(f"  Already exists: {local_filename}")
        return False
    try:
        print(f"  Downloading XC{xc_id} ...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        os.makedirs(save_folder, exist_ok=True)
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"  Saved: {local_filename}")
        return True
    except Exception as e:
        print(f"  Error downloading XC{xc_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Download specific Xeno-Canto recordings by ID (delta downloads).",
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help='XC IDs to download (e.g. 1048463 XC1047135). '
             'Use --id-file to read from a file instead.',
    )
    parser.add_argument(
        "--id-file",
        help="Text file with one XC ID per line (# comments and blank lines ignored).",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Species (common name, scientific name, or eBird code). "
             "When provided, files are saved to <output-dir>/<ebird_code>/<quality>/.",
    )
    parser.add_argument(
        "--quality",
        default="A",
        choices=VALID_QUALITIES,
        help="Quality subfolder to save into (only used with --species). Default: A.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Base output directory. Default: current directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    # Collect IDs from positional args and/or file
    raw_ids = list(args.ids) if args.ids else []
    if args.id_file:
        raw_ids.extend(load_ids_from_file(args.id_file))

    if not raw_ids:
        parser.error("No IDs provided. Pass IDs as arguments or use --id-file.")

    xc_ids = parse_ids(raw_ids)
    if not xc_ids:
        print("Error: No valid IDs found.")
        sys.exit(1)

    # Determine save folder
    if args.species:
        species_name = " ".join(args.species)
        result = resolve_species(species_name)
        if not result:
            print(f"Error: Unknown species '{species_name}'")
            print("Known species: " + ", ".join(code for _, _, code in SEABIRD_SPECIES))
            sys.exit(1)
        english, scientific, ebird_code = result
        save_folder = os.path.join(args.output_dir, ebird_code, args.quality)
        print(f"Species: {english} ({scientific})")
        print(f"Saving to: {save_folder}")
    else:
        save_folder = args.output_dir

    if args.dry_run:
        print(f"\n[DRY RUN] No files will be downloaded.\n")

    print(f"Processing {len(xc_ids)} ID(s)...\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, xc_id in enumerate(xc_ids, 1):
        dest = os.path.join(save_folder, f"xc{xc_id}.mp3")
        if args.dry_run:
            if os.path.exists(dest):
                print(f"  [{i}/{len(xc_ids)}] [dry-run] Already exists: {dest}")
                skipped += 1
            else:
                print(f"  [{i}/{len(xc_ids)}] [dry-run] Would download XC{xc_id} -> {dest}")
                downloaded += 1
        else:
            if download_recording(xc_id, save_folder):
                downloaded += 1
            elif os.path.exists(dest):
                skipped += 1
            else:
                failed += 1

    print(f"\n[DONE] {downloaded} downloaded, {skipped} already existed, {failed} failed.")


if __name__ == "__main__":
    main()
