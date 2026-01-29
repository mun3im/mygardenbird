import argparse
import os
import sys
import time

import requests

# The 10 SEA-Bird species: (common name, scientific name, eBird code)
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


def get_recordings(scientific_name, quality, page=1):
    """Fetch a page of recordings from Xeno-Canto API for given species and quality."""
    url = "https://xeno-canto.org/api/2/recordings"
    params = {
        "query": f"{scientific_name} q:{quality}",
        "page": page,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    js = resp.json()
    return js["recordings"], js["numPages"] > page


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


def download_species(scientific_name, english_name, ebird_code, qualities, output_dir, dry_run):
    """Download all recordings for one species across the given quality levels."""
    counts = {}
    for quality in qualities:
        save_folder = os.path.join(output_dir, ebird_code, quality)
        page = 1
        more = True
        quality_count = 0

        print(f"  Quality {quality}:")
        while more:
            recordings, more = get_recordings(scientific_name, quality, page=page)
            if not recordings:
                break

            for rec in recordings:
                xc_id = rec.get("id")
                if not xc_id:
                    continue
                if dry_run:
                    dest = os.path.join(save_folder, f"xc{xc_id}.mp3")
                    print(f"    [dry-run] Would download XC{xc_id} -> {dest}")
                else:
                    download_recording(xc_id, save_folder)
                quality_count += 1

            page += 1
            if more:
                time.sleep(1)

        counts[quality] = quality_count

    total = sum(counts.values())
    breakdown = ", ".join(f"{c} {q}" for q, c in counts.items() if c > 0)
    action = "Found" if dry_run else "Downloaded"
    print(f"  {action} {total} files for {english_name} ({breakdown})")
    return counts


def list_species():
    """Print the 10 SEA-Bird species and exit."""
    print(f"{'Common Name':<30} {'Scientific Name':<30} {'eBird Code'}")
    print("-" * 75)
    for common, scientific, code in SEABIRD_SPECIES:
        print(f"{common:<30} {scientific:<30} {code}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Xeno-Canto recordings for SEA-Bird species.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["all"],
        help='Species to download (common name, scientific name, or eBird code). '
             'Use "all" for all 10 species. Default: all.',
    )
    parser.add_argument(
        "--quality",
        nargs="+",
        default=["A", "B", "C"],
        choices=VALID_QUALITIES,
        help="Quality grades to download. Default: A B C.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Base output directory. Default: current directory.",
    )
    parser.add_argument(
        "--list-species",
        action="store_true",
        help="Print the 10 SEA-Bird species and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    if args.list_species:
        list_species()
        sys.exit(0)

    # Resolve species list
    if len(args.species) == 1 and args.species[0].lower() == "all":
        species_list = list(SEABIRD_SPECIES)
    else:
        species_list = []
        # Rejoin args in case a multi-word name was passed as separate tokens
        # e.g. --species Zebra Dove -> ["Zebra", "Dove"]
        # Try to match joined tokens greedily
        tokens = args.species
        i = 0
        while i < len(tokens):
            matched = False
            # Try longest possible match first (up to 3 tokens for names like "Large-tailed Nightjar")
            for length in range(min(3, len(tokens) - i), 0, -1):
                candidate = " ".join(tokens[i:i + length])
                result = resolve_species(candidate)
                if result:
                    species_list.append(result)
                    i += length
                    matched = True
                    break
            if not matched:
                print(f"Error: Unknown species '{tokens[i]}'")
                print("Use --list-species to see available species.")
                sys.exit(1)

    qualities = args.quality
    output_dir = args.output_dir

    if args.dry_run:
        print("[DRY RUN] No files will be downloaded.\n")

    grand_total = 0
    for idx, (english, scientific, code) in enumerate(species_list, 1):
        print(f"[{idx}/{len(species_list)}] Downloading {english} ({scientific})...")
        counts = download_species(scientific, english, code, qualities, output_dir, args.dry_run)
        grand_total += sum(counts.values())
        print()

    action = "found" if args.dry_run else "downloaded"
    print(f"[DONE] Total {action}: {grand_total} files across {len(species_list)} species.")


if __name__ == "__main__":
    main()
