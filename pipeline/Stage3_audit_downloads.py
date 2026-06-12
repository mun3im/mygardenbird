import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

from tqdm import tqdm

from config import ACTIVE_SPECIES, VALID_QUALITIES, folder_name, PER_SPECIES_FLACS, PER_SPECIES_CSV


def scan_downloads(input_dir):
    """Scan input_dir for {English name}/{quality}/xc*.flac structure.

    Returns dict keyed by English name: {english: {quality: [filepath, ...]}}.
    Only considers active species and valid quality letters.
    """
    results = {}
    for english, _, _ in tqdm(ACTIVE_SPECIES, desc="Scanning downloads", unit="species"):
        results[english] = {}
        species_dir = os.path.join(input_dir, folder_name(english))
        if not os.path.isdir(species_dir):
            continue
        for quality in VALID_QUALITIES:
            qual_dir = os.path.join(species_dir, quality)
            if not os.path.isdir(qual_dir):
                continue
            flacs = sorted(
                os.path.join(qual_dir, f)
                for f in os.listdir(qual_dir)
                if f.lower().endswith(".flac")
            )
            if flacs:
                results[english][quality] = flacs
    return results


def load_metadata(metadata_dir):
    """Read per-species CSVs and count available recordings per quality.

    Returns dict keyed by English name: {english: {quality: count}}.
    CSV filenames match English name with spaces replaced by underscores,
    e.g. Asian_Koel.csv.
    """
    if not os.path.isdir(metadata_dir):
        return None

    # Build lookup: underscore_english -> english
    english_lookup = {
        eng.replace(" ", "_"): eng
        for eng, _, _ in ACTIVE_SPECIES
    }

    results = {}
    for fname in os.listdir(metadata_dir):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]  # e.g. "Asian_Koel"
        english = english_lookup.get(stem)
        if not english:
            continue

        quality_counts = defaultdict(int)
        path = os.path.join(metadata_dir, fname)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("q", "").strip()
                if q in VALID_QUALITIES:
                    quality_counts[q] += 1
        results[english] = dict(quality_counts)
    return results


def count_tiny_files(file_list, min_size):
    """Count files smaller than min_size bytes."""
    return sum(1 for f in file_list if os.path.getsize(f) < min_size)


def get_total_duration(file_list, desc="Processing"):
    """Get total duration in seconds for a list of audio files using ffprobe."""
    total = 0.0
    for fpath in tqdm(file_list, desc=desc, unit="file", leave=False):
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    fpath,
                ],
                capture_output=True, text=True, timeout=10,
            )
            dur = result.stdout.strip()
            if dur:
                total += float(dur)
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
    return total


def format_duration(seconds):
    """Format duration as hours:min or min:sec for readability."""
    if seconds == 0:
        return "-"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    elif minutes > 0:
        return f"{minutes}m{secs:02d}s"
    else:
        return f"{secs}s"


def print_table(rows):
    """Print a formatted summary table with duration per quality."""
    # Header - always show duration (removed Tiny column)
    hdr = f"{'Species':<35} {'Qual':>4} {'Files':>6} {'Avail':>6} {'Done%':>6} {'Duration':>12}"
    sep = "-" * len(hdr)

    print(sep)
    print(hdr)
    print(sep)

    total_tiny = 0
    for row in rows:
        line = (
            f"{row['species']:<35} {row['quality']:>4} "
            f"{row['files']:>6} {row['avail']:>6} "
            f"{row['pct']:>6} {row['duration']:>12}"
        )
        print(line)

        # Track tiny files for warning
        total_tiny += row.get('tiny', 0)

        # Print separator after last quality of each species
        if row.get("species_end"):
            print(sep)

    # Grand total
    print(sep)

    # Warning if any tiny files found (sanity check)
    if total_tiny > 0:
        print(f"\n⚠️  WARNING: Found {total_tiny} files <1KB (possible corruption)")
        print("   This should be 0 if Stage 1 filtering (>=3s) worked correctly.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: EDA on downloaded Xeno-Canto FLACs — how many usable files per species/quality?",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(PER_SPECIES_FLACS),
        help=f"Base directory with {{English name}}/{{quality}}/ structure from Stage 2. Default: {PER_SPECIES_FLACS}",
    )
    parser.add_argument(
        "--metadata-dir",
        default=str(PER_SPECIES_CSV),
        help=f"Path to per-species CSV directory for completeness check. Default: {PER_SPECIES_CSV}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Default: <project_csv>/stage4_eda_report.csv.",
    )
    parser.add_argument(
        "--skip-durations",
        action="store_true",
        help="Skip duration extraction (faster but less useful).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum file size in bytes for tiny file detection. Default: 1024.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)

    metadata_dir = args.metadata_dir

    output_path = args.output
    if output_path is None:
        from config import PROJECT_CSV
        output_path = os.path.join(str(PROJECT_CSV), "stage4_eda_report.csv")

    # Print startup information
    print("=" * 80)
    print("STAGE 4: AUDIT DOWNLOADED RECORDINGS")
    print("=" * 80)
    print("WHAT THIS DOES:")
    print("  - Scans downloaded FLAC files for each species and quality")
    print("  - Checks completeness against Stage1 metadata")
    print("  - Extracts total duration per quality (helps assess sample size)")
    print("  - Checks for tiny/corrupted files (should be 0 after Stage1 filtering)")
    print()
    print("INPUT:")
    print(f"  - Downloaded FLACs: {input_dir}/<Species Name>/<Quality>/")
    print(f"  - Stage1 metadata: {metadata_dir}/<Scientific_name>.csv")
    print()
    print("OUTPUT:")
    print(f"  - Audit report CSV: {output_path}")
    print(f"  - Terminal summary table")
    print("=" * 80)
    print()

    # Scan downloads
    print(f"Scanning downloads in: {input_dir}")
    downloads = scan_downloads(input_dir)

    # Load metadata for completeness check
    metadata = load_metadata(metadata_dir)
    if metadata is None:
        print(f"Warning: Metadata directory '{metadata_dir}' not found. Completeness will show N/A.")
    else:
        print(f"Loaded metadata from: {metadata_dir}")

    # Duration extraction is on by default (can be disabled with --skip-durations)
    extract_durations = not args.skip_durations
    if extract_durations:
        # Check ffprobe availability
        try:
            subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
            print("Duration extraction: ENABLED (use --skip-durations to disable)")
        except FileNotFoundError:
            print("Warning: ffprobe not found. Durations will show N/A.")
            extract_durations = False
    else:
        print("Duration extraction: DISABLED")

    print()

    # Compute stats
    table_rows = []
    csv_rows = []

    total_files = 0
    total_avail = 0
    total_tiny = 0
    total_duration = 0.0

    # Progress bar for species processing
    species_pbar = tqdm(ACTIVE_SPECIES, desc="Processing species", unit="species")
    for english, scientific, code in species_pbar:
        # Update progress bar with current species
        species_pbar.set_postfix_str(f"{english[:20]}")

        species_downloads = downloads.get(english, {})
        species_metadata = metadata.get(english, {}) if metadata else {}

        # Collect qualities that appear in either downloads or metadata
        qualities_present = sorted(
            set(list(species_downloads.keys()) + list(species_metadata.keys())),
            key=lambda q: VALID_QUALITIES.index(q),
        )
        if not qualities_present:
            qualities_present = ["A", "B", "C"]  # show defaults even if empty

        for qi, quality in enumerate(qualities_present):
            file_list = species_downloads.get(quality, [])
            n_files = len(file_list)
            n_avail = species_metadata.get(quality) if species_metadata else None
            tiny = count_tiny_files(file_list, args.min_size) if file_list else 0

            if n_avail is not None and n_avail > 0:
                pct = f"{n_files / n_avail * 100:.1f}%"
                pct_val = n_files / n_avail * 100
            elif n_avail == 0 and n_files == 0:
                pct = "-"
                pct_val = ""
            else:
                pct = "N/A"
                pct_val = ""

            avail_str = str(n_avail) if n_avail is not None else "N/A"

            duration = 0.0
            dur_str = ""
            if extract_durations and file_list:
                # Create descriptive label for progress bar
                desc = f"{english[:15]}_{quality}"
                duration = get_total_duration(file_list, desc=desc)
                dur_str = format_duration(duration)
            elif extract_durations:
                dur_str = "-"
            else:
                dur_str = "N/A"

            is_last = qi == len(qualities_present) - 1

            table_rows.append({
                "species": english if qi == 0 else "",
                "quality": quality,
                "files": str(n_files),
                "avail": avail_str,
                "pct": pct,
                "tiny": tiny,
                "duration": dur_str,
                "species_end": is_last,
            })

            csv_rows.append({
                "species": english,
                "ebird_code": code,
                "quality": quality,
                "files_downloaded": n_files,
                "files_available": n_avail if n_avail is not None else "",
                "completeness_pct": f"{pct_val:.1f}" if isinstance(pct_val, float) else "",
                "tiny_files": tiny,
                "total_duration_s": f"{duration:.1f}" if extract_durations else "",
            })

            total_files += n_files
            total_avail += n_avail if n_avail is not None else 0
            total_tiny += tiny
            total_duration += duration

    # Close progress bar
    species_pbar.close()

    # Grand total row
    if total_avail > 0:
        total_pct = f"{total_files / total_avail * 100:.1f}%"
        total_pct_val = f"{total_files / total_avail * 100:.1f}"
    else:
        total_pct = "N/A"
        total_pct_val = ""

    total_dur_str = format_duration(total_duration) if extract_durations else "N/A"

    table_rows.append({
        "species": "TOTAL",
        "quality": "",
        "files": str(total_files),
        "avail": str(total_avail) if metadata else "N/A",
        "pct": total_pct,
        "tiny": total_tiny,
        "duration": total_dur_str,
    })

    csv_rows.append({
        "species": "TOTAL",
        "ebird_code": "",
        "quality": "",
        "files_downloaded": total_files,
        "files_available": total_avail if metadata else "",
        "completeness_pct": total_pct_val,
        "tiny_files": total_tiny,
        "total_duration_s": f"{total_duration:.1f}" if extract_durations else "",
    })

    # Print table
    print_table(table_rows)

    # Save CSV
    fieldnames = [
        "species", "ebird_code", "quality",
        "files_downloaded", "files_available", "completeness_pct",
        "tiny_files", "total_duration_s",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
