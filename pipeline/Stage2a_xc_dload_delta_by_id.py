import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from config import (
    SPECIES, VALID_QUALITIES, folder_name, resolve_species, PER_SPECIES_FLACS,
    get_profile, _load_species, resolve_species_dir,
)

from xc_download import (
    MIN_DURATION_S, is_nd_licence, mp3_to_mono_flac, probe_audio,
    resolve_api_key,
)


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


def licence_for_ids(xc_ids, profile=None):
    """Map XC id -> licence URL using whatever metadata is on disk.

    Stage 2 filters ND out at the catalogue level, but this stage takes
    explicit ids and so has no catalogue to filter. Without this lookup a
    delta download silently reintroduces the ND recordings the project has
    gone to some trouble to remove.
    """
    import csv
    import glob

    out = {}
    roots = []
    if profile is not None:
        for attr in ("per_species_csv", "project_csv"):
            d = getattr(profile, attr, None)
            if d:
                roots.append(str(d))
    base = PER_SPECIES_FLACS.parent
    # per_species_csv_old holds ids the current catalogue has dropped. A
    # delta download is exactly the case where an old id gets named, so the
    # superseded catalogue has to be consulted too or its ND rows read as
    # "licence unknown" and sail through.
    for extra in ("per_species_csv", "per_species_csv_old", "project_csv"):
        d = str(base / extra)
        if d not in roots:
            roots.append(d)

    wanted = set(xc_ids)
    patterns = []
    for r in roots:
        patterns += [os.path.join(r, "*.csv")]
    for path in sorted({p for pat in patterns for p in glob.glob(pat)}):
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    rid = (row.get("id") or row.get("file_id") or "").strip()
                    rid = rid.lower().replace("xc", "")
                    if rid in wanted and rid not in out:
                        lic = (row.get("lic") or row.get("license")
                               or row.get("licence") or "").strip()
                        if lic:
                            out[rid] = lic
        except (OSError, csv.Error):
            continue
    return out


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


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Download specific Xeno-Canto recordings by ID (delta downloads).",
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
        "--dataset", choices=["mygardenbird", "sea-bird30"],
        default=os.environ.get("PIPELINE_DATASET", "mygardenbird"),
        help="Which dataset's species catalogue/paths to use. Default: mygardenbird "
             "(or $PIPELINE_DATASET if set). Also controls how --species resolves the "
             "destination folder (space-named for mygardenbird, underscore-named with "
             "overrides for sea-bird30).",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Species (common name, scientific name, or eBird code). "
             "When provided, files are saved to <output-dir>/<species folder>/<quality>/.",
    )
    parser.add_argument(
        "--quality",
        default="A",
        choices=VALID_QUALITIES,
        help="Quality subfolder to save into (only used with --species). Default: A.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Base output directory. Default: the selected dataset's per_species_flacs dir "
             f"(MyGardenBird: {PER_SPECIES_FLACS}).",
    )
    parser.add_argument(
        "--no-skip-nd",
        action="store_true",
        help="Download ND-licensed recordings too (disables the default ND "
             "filter). ND forbids derivative works and 3-second segmentation "
             "is one, so these cannot be redistributed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    # Resolve dataset profile, then defer path/species-catalogue defaults
    # until now (argparse evaluates default= before --dataset is parsed).
    profile = get_profile(args.dataset)
    if args.dataset == "mygardenbird":
        catalogue = None  # resolve_species(..., None) falls back to module-level SPECIES
    else:
        catalogue, _ = _load_species(profile.species_csv)

    if args.output_dir is None:
        args.output_dir = str(profile.per_species_flacs)

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

    # Determine save folder. Uses resolve_species_dir() (rather than a plain
    # folder_name() call) so --species correctly resolves SEA-BIRD30's
    # underscore-named / overridden species folders (e.g. "Pied Fantail" ->
    # flacs/Malaysian_Pied_Fantail) instead of creating a new, wrong,
    # space-named duplicate folder alongside the real one.
    if args.species:
        species_name = " ".join(args.species)
        result = resolve_species(species_name, catalogue)
        if not result:
            print(f"Error: Unknown species '{species_name}'")
            all_species = catalogue if catalogue is not None else SPECIES
            print("Known species: " + ", ".join(code for _, _, code in all_species))
            sys.exit(1)
        english, scientific, ebird_code = result
        style = profile.folder_styles.get("per_species_flacs", "space")
        base_dir = Path(args.output_dir)
        try:
            species_dir = resolve_species_dir(base_dir, english, style, profile.species_overrides)
        except FileNotFoundError:
            # Folder doesn't exist yet (first download for this species) --
            # compute where it SHOULD go using the same style/override rules.
            if style == "underscore":
                folder = profile.species_overrides.get(english, english.replace(" ", "_"))
            else:
                folder = profile.species_overrides.get(english, folder_name(english))
            species_dir = base_dir / folder
        save_folder = os.path.join(species_dir, args.quality)
    else:
        save_folder = args.output_dir

    # Print startup information
    print("=" * 80)
    print("STAGE 3: DOWNLOAD SPECIFIC RECORDINGS BY ID (DELTA DOWNLOAD)")
    print("=" * 80)
    print("WHAT THIS DOES:")
    print("  - Downloads specific XC recordings by ID number")
    print("  - Useful for filling gaps or adding specific recordings")
    print("  - Converts MP3 to mono FLAC (preserving original sample rate)")
    print("  - Skips recordings <3 seconds")
    print()
    print("INPUT:")
    if args.id_file:
        print(f"  - ID file: {args.id_file}")
    else:
        print(f"  - IDs from command line: {len(xc_ids)} ID(s)")
    if args.species:
        print(f"  - Species: {english} ({scientific})")
        print(f"  - Quality: {args.quality}")
    print()
    print("OUTPUT:")
    print(f"  - FLAC files: {save_folder}/xc####.flac")
    if args.species:
        print(f"      Full path: {save_folder}/")
    print("=" * 80)
    print()

    if args.dry_run:
        print(f"[DRY RUN] No files will be downloaded.\n")

    if not args.no_skip_nd:
        licences = licence_for_ids(xc_ids, profile)
        nd_ids = [i for i in xc_ids if is_nd_licence(licences.get(i, ""))]
        if nd_ids:
            print(f"Skipping {len(nd_ids)} ND-licensed recording(s): "
                  f"{', '.join('XC' + i for i in nd_ids)}")
            print("  (pass --no-skip-nd to override)\n")
            xc_ids = [i for i in xc_ids if i not in set(nd_ids)]
        unknown = [i for i in xc_ids if i not in licences]
        if unknown:
            print(f"Warning: licence unknown for {len(unknown)} id(s): "
                  f"{', '.join('XC' + i for i in unknown[:10])}"
                  f"{' ...' if len(unknown) > 10 else ''}")
            print("  These are downloaded anyway -- verify before publishing.\n")

    print(f"Processing {len(xc_ids)} ID(s)...\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, xc_id in enumerate(xc_ids, 1):
        dest = os.path.join(save_folder, f"xc{xc_id}.flac")
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
