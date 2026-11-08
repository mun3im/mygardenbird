import argparse
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd
import requests

from config import (
    ACTIVE_SPECIES, VALID_QUALITIES, folder_name, resolve_species,
    PER_SPECIES_FLACS, PER_SPECIES_CSV, get_profile, _load_species,
    resolve_species_dir,
)

from xc_download import (
    MIN_DURATION_S, mp3_to_mono_flac, probe_audio, resolve_api_key,
)

# Stage 1 already applies regional filtering (lon 60-140 OR ASEAN) and duration filtering (>=3s)
# Stage 2 just downloads what's in the CSV


def _is_nd_license(lic) -> bool:
    """True if an XC `lic` URL denotes a No-Derivatives Creative Commons licence.

    Covers both the `-nd` licence-code form (e.g. .../licenses/by-nc-nd/4.0/)
    and a bare `/nd/` segment, case-insensitively.
    """
    if not isinstance(lic, str):
        return False
    lic = lic.lower()
    return "-nd" in lic or "/nd/" in lic


def get_xc_ids_from_csv(common_name, quality, per_species_csv_dir=PER_SPECIES_CSV, skip_nd=True):
    """Load XC IDs from Stage1 CSV for given species and quality.

    Stage 1 CSVs now contain ONLY regional-filtered records (lon 60-140 OR ASEAN, >=3s).
    No additional filtering needed - just read and filter by quality.

    Uses English common name to match Stage 1 CSV naming convention.
    `per_species_csv_dir` defaults to MyGardenBird's PER_SPECIES_CSV for
    backward compatibility; pass a different profile's per_species_csv to
    look elsewhere (None if that dataset has no such directory).

    When `skip_nd` is True (default), recordings with a No-Derivatives
    licence are excluded here so Stage 2 never downloads them in the first
    place -- avoids the recurring manual cleanup of ND flac+txt pairs after
    the fact. Returns (xc_ids, nd_skipped_count).
    """
    safe = common_name.replace(" ", "_").replace("/", "-").replace(":", "-")
    csv_path = os.path.join(per_species_csv_dir, f"{safe}.csv")

    if not os.path.isfile(csv_path):
        print(f"    WARNING: No Stage1 metadata found at {csv_path}")
        print(f"    Run Stage1 first to generate regional metadata.")
        return None, 0

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"    WARNING: Empty or missing CSV: {csv_path}")
        return None, 0

    # CSV already contains regional-filtered records
    # Only filter by quality if specified
    if "q" in df.columns and quality:
        df = df[df["q"] == quality]

    nd_skipped = 0
    if skip_nd and "lic" in df.columns:
        is_nd = df["lic"].apply(_is_nd_license)
        nd_skipped = int(is_nd.sum())
        df = df[~is_nd]

    # Extract XC IDs
    if "id" not in df.columns:
        print(f"    WARNING: No 'id' column in {csv_path}")
        return set(), nd_skipped

    xc_ids = set(str(xid) for xid in df["id"].dropna())
    return xc_ids, nd_skipped


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

        if mp3_to_mono_flac(tmp_path, flac_filename):
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


def download_species(scientific_name, english_name, ebird_code, qualities, output_dir, dry_run,
                      api_key=None, per_species_csv_dir=PER_SPECIES_CSV,
                      folder_style="space", species_overrides=None, skip_nd=True):
    """Download recordings for one species across the given quality levels.

    Reads XC IDs from Stage 1 CSVs (already filtered for regional + duration criteria).
    No additional filtering needed - just download what's in the CSV.

    `folder_style`/`species_overrides` select how the species subfolder name
    under output_dir is derived -- "space" (default, folder_name(), matches
    MyGardenBird) or "underscore" (matches SEA-BIRD30's flacs/ convention),
    with explicit per-species overrides (e.g. "Pied Fantail" ->
    "Malaysian_Pied_Fantail") checked first either way.

    `skip_nd` (default True) excludes No-Derivatives-licensed recordings
    before downloading; set False to download everything in the Stage1 CSV.
    """
    overrides = species_overrides or {}
    if folder_style == "underscore":
        species_folder = overrides.get(english_name, english_name.replace(" ", "_"))
    else:
        species_folder = overrides.get(english_name, folder_name(english_name))
    counts = {}
    nd_skipped_total = 0
    for quality in qualities:
        save_folder = os.path.join(output_dir, species_folder, quality)

        # Load XC IDs from Stage1 metadata (already regional-filtered)
        xc_ids, nd_skipped = get_xc_ids_from_csv(english_name, quality, per_species_csv_dir, skip_nd=skip_nd)
        nd_skipped_total += nd_skipped
        if xc_ids is None:
            print(f"    Skipping quality {quality}: no Stage1 metadata available")
            counts[quality] = 0
            continue

        if nd_skipped:
            print(f"  Quality {quality}: skipping {nd_skipped} ND-licensed recording(s)")

        if not xc_ids:
            print(f"  Quality {quality}: 0 recordings to download")
            counts[quality] = 0
            continue

        print(f"  Quality {quality}: {len(xc_ids)} recordings to download")

        page = 1
        more = True
        quality_count = 0

        while more:
            recordings, more = get_recordings(scientific_name, quality, page=page, api_key=api_key)
            if not recordings:
                break

            for rec in recordings:
                xc_id = rec.get("id")
                if not xc_id:
                    continue

                # Only download IDs from Stage1 CSV (already regional-filtered)
                if str(xc_id) not in xc_ids:
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
    nd_note = f", skipped {nd_skipped_total} ND-licensed" if nd_skipped_total else ""
    print(f"  {action} {total} files for {english_name} ({breakdown}{nd_note})")
    return counts, nd_skipped_total


def list_species(active_species=ACTIVE_SPECIES):
    """Print active species and exit."""
    print(f"{'Common Name':<35} {'Scientific Name':<30} {'eBird Code'}")
    print("-" * 80)
    for common, scientific, code in active_species:
        print(f"{common:<35} {scientific:<30} {code}")
    print(f"\n{len(active_species)} active species (edit 'active' column in target_species.csv to change)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Download Xeno-Canto recordings for active species.",
    )
    parser.add_argument(
        "--dataset", choices=["mygardenbird", "sea-bird30"],
        default=os.environ.get("PIPELINE_DATASET", "mygardenbird"),
        help="Which dataset's species catalogue/paths to use. Default: mygardenbird "
             "(or $PIPELINE_DATASET if set).",
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
        default=None,
        help=f"Base output directory. Default: the selected dataset's per_species_flacs dir "
             f"(MyGardenBird: {PER_SPECIES_FLACS}).",
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
    parser.add_argument(
        "--skip-nd",
        dest="skip_nd",
        action="store_true",
        default=True,
        help="Skip No-Derivatives (CC ...-ND) licensed recordings (default: on).",
    )
    parser.add_argument(
        "--no-skip-nd",
        dest="skip_nd",
        action="store_false",
        help="Download ND-licensed recordings too (disables the default ND filter).",
    )
    args = parser.parse_args()

    # Resolve dataset profile, then defer path/species-list defaults until
    # now (argparse evaluates default= before --dataset itself is parsed).
    profile = get_profile(args.dataset)
    if args.dataset == "mygardenbird":
        active_catalogue = list(ACTIVE_SPECIES)
        full_catalogue = None  # resolve_species(..., None) falls back to module-level SPECIES
    else:
        full_catalogue, active_catalogue = _load_species(profile.species_csv)

    if args.output_dir is None:
        args.output_dir = str(profile.per_species_flacs)

    if args.list_species:
        list_species(active_catalogue)
        sys.exit(0)

    api_key = resolve_api_key(args.api_key,
                              script_dir=os.path.dirname(__file__))

    # Resolve species list
    if len(args.species) == 1 and args.species[0].lower() == "all":
        species_list = list(active_catalogue)
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
                result = resolve_species(candidate, full_catalogue)
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

    # Print startup information
    print("=" * 80)
    print("STAGE 2: DOWNLOAD XENO-CANTO RECORDINGS")
    print("=" * 80)
    print("WHAT THIS DOES:")
    print("  - Downloads FLAC recordings from Xeno-Canto for active species")
    print("  - Reads pre-filtered XC IDs from Stage 1 CSVs (already regional + >=3s)")
    print("  - Converts MP3 to mono FLAC (preserving original sample rate)")
    print("  - NO filtering needed - Stage 1 already did regional/duration filtering")
    nd_status = "skipped" if args.skip_nd else "downloaded (not filtered)"
    print(f"  - No-Derivatives (ND) licensed recordings are {nd_status}")
    print()
    print("NOTE:")
    print("  - Only downloads species with active=yes in target_species.csv")
    print(f"  - To shortlist species: Edit {profile.species_csv}")
    print(f"      Set active=yes for species you want")
    print(f"      Set active=no for species you want to skip")
    print()
    print("INPUT:")
    if profile.per_species_csv is not None:
        print(f"  - Species metadata: {profile.per_species_csv}/<English_name>.csv")
        print(f"      (Generated by Stage 1 using English names)")
    else:
        print(f"  - Dataset '{args.dataset}' has no per_species_csv directory; "
              f"pass --output-dir/species metadata some other way if needed.")
    print(f"  - Species to download: {len(species_list)} species (active=yes)")
    if len(species_list) <= 5:
        for common, sci, code in species_list:
            print(f"      • {common} ({sci})")
    else:
        print(f"      • {species_list[0][0]} ({species_list[0][1]})")
        print(f"      • {species_list[1][0]} ({species_list[1][1]})")
        print(f"      ... and {len(species_list) - 2} more")
    print(f"  - Quality grades: {', '.join(qualities)}")
    print()
    print("OUTPUT:")
    print(f"  - FLAC files: {output_dir}/<Species Name>/<Quality>/xc####.flac")
    print(f"      Example: {output_dir}/Javan Myna/A/xc123456.flac")
    print("=" * 80)
    print()

    if args.dry_run:
        print("[DRY RUN] No files will be downloaded.\n")

    grand_total = 0
    grand_nd_skipped = 0
    for idx, (english, scientific, code) in enumerate(species_list, 1):
        print(f"[{idx}/{len(species_list)}] Downloading {english} ({scientific})...")
        counts, nd_skipped = download_species(
            scientific, english, code, qualities, output_dir, args.dry_run, api_key,
            per_species_csv_dir=profile.per_species_csv,
            folder_style=profile.folder_styles.get("per_species_flacs", "space"),
            species_overrides=profile.species_overrides,
            skip_nd=args.skip_nd,
        )
        grand_total += sum(counts.values())
        grand_nd_skipped += nd_skipped
        print()

    action = "found" if args.dry_run else "downloaded"
    nd_summary = f" ({grand_nd_skipped} ND-licensed recordings skipped)" if grand_nd_skipped else ""
    print(f"[DONE] Total {action}: {grand_total} files across {len(species_list)} species.{nd_summary}")


if __name__ == "__main__":
    main()
