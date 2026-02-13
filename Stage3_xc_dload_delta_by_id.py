import argparse
import os
import re
import subprocess
import sys
import tempfile

import requests

from config import SPECIES, VALID_QUALITIES, folder_name, resolve_species, PER_SPECIES_FLACS


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


MIN_DURATION_S = 3.0   # clips shorter than this are skipped at download time


def _probe_audio(path):
    """Return (sample_rate, duration_s) of an audio file via ffprobe.

    Either value is None if it cannot be determined.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,duration",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        parts = result.stdout.strip().split(",")
        sr  = int(parts[0])   if len(parts) > 0 and parts[0] else None
        dur = float(parts[1]) if len(parts) > 1 and parts[1] else None
        return sr, dur
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return None, None


def _mp3_to_mono_flac(mp3_path, flac_path):
    """Convert an MP3 file to mono FLAC using ffmpeg.

    Pre-conversion:  recordings shorter than MIN_DURATION_S are skipped.
    Post-conversion: asserts the output FLAC preserves the original sample rate.
                     Resampling is intentionally deferred to Stage 6.

    Returns True if the file was converted and passes all checks.
    Returns False (and deletes any partial output) otherwise.
    """
    src_sr, src_dur = _probe_audio(mp3_path)

    # Skip recordings that are too short
    if src_dur is not None and src_dur < MIN_DURATION_S:
        print(f"  Skipping XC{os.path.basename(mp3_path)}: {src_dur:.2f}s < {MIN_DURATION_S}s")
        return False

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-ac", "1",      # mono — channel change only, no -ar flag
             "-c:a", "flac",
             flac_path],
            capture_output=True, timeout=120,
        )
        if not (os.path.isfile(flac_path) and os.path.getsize(flac_path) > 0):
            return False

        # Assert sample rate was preserved
        dst_sr, _ = _probe_audio(flac_path)
        if src_sr is not None and dst_sr is not None and src_sr != dst_sr:
            print(f"  ERROR: sample rate changed {src_sr} Hz -> {dst_sr} Hz in {flac_path}")
            print(f"  Deleting corrupted output.")
            os.unlink(flac_path)
            return False

        return True
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
        "--species",
        nargs="+",
        help="Species (common name, scientific name, or eBird code). "
             "When provided, files are saved to <output-dir>/<English name>/<quality>/.",
    )
    parser.add_argument(
        "--quality",
        default="A",
        choices=VALID_QUALITIES,
        help="Quality subfolder to save into (only used with --species). Default: A.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PER_SPECIES_FLACS),
        help=f"Base output directory. Default: {PER_SPECIES_FLACS}",
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
            print("Known species: " + ", ".join(code for _, _, code in SPECIES))
            sys.exit(1)
        english, scientific, ebird_code = result
        save_folder = os.path.join(args.output_dir, folder_name(english), args.quality)
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
