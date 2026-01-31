import argparse
import os
import subprocess
import sys
import tempfile
import time

import requests

from species import ACTIVE_SPECIES, VALID_QUALITIES, folder_name, resolve_species


def get_recordings(scientific_name, quality, page=1, api_key=None):
    """Fetch a page of recordings from Xeno-Canto API for given species and quality."""
    url = "https://xeno-canto.org/api/3/recordings"
    # API v3 requires tag-based queries: use sp: tag for species
    params = {
        "query": f'sp:"{scientific_name}" q:{quality}',
        "page": page,
    }
    if api_key:
        params["key"] = api_key
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    js = resp.json()
    return js["recordings"], js["numPages"] > page


def _mp3_to_mono_flac(mp3_path, flac_path):
    """Convert an MP3 file to mono FLAC using ffmpeg. Returns True on success."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", mp3_path,
                "-ac", "1",        # mono
                "-c:a", "flac",    # FLAC codec
                flac_path,
            ],
            capture_output=True, timeout=120,
        )
        return os.path.isfile(flac_path) and os.path.getsize(flac_path) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ffmpeg error: {e}")
        return False


def download_recording(xc_id, save_folder):
    """Download a single recording by XC ID, convert to mono FLAC in save_folder."""
    flac_filename = os.path.join(save_folder, f"xc{xc_id}.flac")
    if os.path.exists(flac_filename):
        print(f"  Already exists: {flac_filename}")
        return False
    try:
        print(f"  Downloading XC{xc_id} ...")
        download_url = f"https://xeno-canto.org/{xc_id}/download"
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        os.makedirs(save_folder, exist_ok=True)

        # Download MP3 to temp file, convert to mono FLAC, delete temp
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)

        if _mp3_to_mono_flac(tmp_path, flac_filename):
            print(f"  Saved: {flac_filename}")
            return True
        else:
            print(f"  Error converting XC{xc_id} to FLAC")
            return False
    except Exception as e:
        print(f"  Error downloading XC{xc_id}: {e}")
        return False
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_species(scientific_name, english_name, ebird_code, qualities, output_dir, dry_run, api_key=None):
    """Download all recordings for one species across the given quality levels."""
    species_folder = folder_name(english_name)
    counts = {}
    for quality in qualities:
        save_folder = os.path.join(output_dir, species_folder, quality)
        page = 1
        more = True
        quality_count = 0

        print(f"  Quality {quality}:")
        while more:
            recordings, more = get_recordings(scientific_name, quality, page=page, api_key=api_key)
            if not recordings:
                break

            for rec in recordings:
                xc_id = rec.get("id")
                if not xc_id:
                    continue
                if dry_run:
                    dest = os.path.join(save_folder, f"xc{xc_id}.flac")
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
    """Print active species and exit."""
    print(f"{'Common Name':<35} {'Scientific Name':<30} {'eBird Code'}")
    print("-" * 80)
    for common, scientific, code in ACTIVE_SPECIES:
        print(f"{common:<35} {scientific:<30} {code}")
    print(f"\n{len(ACTIVE_SPECIES)} active species (edit 'active' column in target_species.csv to change)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Download Xeno-Canto recordings for active species.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["all"],
        help='Species to download (common name, scientific name, or eBird code). '
             'Use "all" for all active species. Default: all.',
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
        help="Print active species and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Xeno-Canto API key (optional for API v3, but recommended). Will auto-load from xc_key.txt if present.",
    )
    args = parser.parse_args()

    if args.list_species:
        list_species()
        sys.exit(0)

    # Auto-load API key from xc_key.txt if not provided via command line
    api_key = args.api_key
    if not api_key:
        key_file = os.path.join(os.path.dirname(__file__), "xc_key.txt")
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                api_key = f.read().strip()
            if api_key:
                print(f"Loaded API key from {key_file}\n")

    # Resolve species list
    if len(args.species) == 1 and args.species[0].lower() == "all":
        species_list = list(ACTIVE_SPECIES)
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
        counts = download_species(scientific, english, code, qualities, output_dir, args.dry_run, api_key)
        grand_total += sum(counts.values())
        print()

    action = "found" if args.dry_run else "downloaded"
    print(f"[DONE] Total {action}: {grand_total} files across {len(species_list)} species.")


if __name__ == "__main__":
    main()
