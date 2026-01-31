import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

from species import ACTIVE_SPECIES, VALID_QUALITIES, folder_name

# Derived lookups
_SCIENTIFIC_TO_SPECIES = {
    sci.replace(" ", "_"): (eng, sci, code)
    for eng, sci, code in ACTIVE_SPECIES
}


def scan_downloads(input_dir):
    """Scan input_dir for {English name}/{quality}/xc*.flac structure.

    Returns dict keyed by English name: {english: {quality: [filepath, ...]}}.
    Only considers active species and valid quality letters.
    """
    results = {}
    for english, _, _ in ACTIVE_SPECIES:
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
    """Read active_species/ CSVs and count available recordings per quality.

    Returns dict keyed by English name: {english: {quality: count}}.
    CSV filenames are like Geopelia_striata.csv (Genus_species.csv).
    """
    if not os.path.isdir(metadata_dir):
        return None

    results = {}
    for fname in os.listdir(metadata_dir):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]  # e.g. "Geopelia_striata"
        species_info = _SCIENTIFIC_TO_SPECIES.get(stem)
        if not species_info:
            continue
        english, _, _ = species_info

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


def get_total_duration(file_list):
    """Get total duration in seconds for a list of audio files using ffprobe."""
    total = 0.0
    for fpath in file_list:
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


def print_table(rows, show_durations):
    """Print a formatted summary table."""
    # Header
    hdr = f"{'Species':<25} {'Qual':>4} {'Files':>6} {'Avail':>6} {'Done%':>6} {'Tiny':>5}"
    if show_durations:
        hdr += f" {'Duration':>10}"
    sep = "-" * len(hdr)

    print(sep)
    print(hdr)
    print(sep)

    for row in rows:
        line = (
            f"{row['species']:<25} {row['quality']:>4} "
            f"{row['files']:>6} {row['avail']:>6} "
            f"{row['pct']:>6} {row['tiny']:>5}"
        )
        if show_durations:
            line += f" {row['duration']:>10}"
        print(line)

        # Print separator after last quality of each species
        if row.get("species_end"):
            print(sep)

    # Grand total
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: EDA on downloaded Xeno-Canto FLACs — how many usable files per species/quality?",
    )
    parser.add_argument(
        "input_dir",
        help="Base directory with {English name}/{quality}/ structure from Stage 2.",
    )
    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="Path to active_species/ CSV directory for completeness check. "
             "Default: <script_dir>/active_species.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Default: {input_dir}/stage3_eda_report.csv.",
    )
    parser.add_argument(
        "--durations",
        action="store_true",
        help="Extract audio durations via ffprobe (slower).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum file size in bytes to consider usable. Default: 1024.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)

    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_species")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(input_dir, "stage4_eda_report.csv")

    # Scan downloads
    print(f"Scanning downloads in: {input_dir}")
    downloads = scan_downloads(input_dir)

    # Load metadata for completeness check
    metadata = load_metadata(metadata_dir)
    if metadata is None:
        print(f"Warning: Metadata directory '{metadata_dir}' not found. Completeness will show N/A.")
    else:
        print(f"Loaded metadata from: {metadata_dir}")

    if args.durations:
        # Check ffprobe availability
        try:
            subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            print("Warning: ffprobe not found. Durations will show N/A.")
            args.durations = False

    print()

    # Compute stats
    table_rows = []
    csv_rows = []

    total_files = 0
    total_avail = 0
    total_tiny = 0
    total_duration = 0.0

    for english, scientific, code in ACTIVE_SPECIES:
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
            if args.durations and file_list:
                duration = get_total_duration(file_list)
                dur_str = f"{duration:.1f}s"
            elif args.durations:
                dur_str = "-"

            is_last = qi == len(qualities_present) - 1

            table_rows.append({
                "species": english if qi == 0 else "",
                "quality": quality,
                "files": str(n_files),
                "avail": avail_str,
                "pct": pct,
                "tiny": str(tiny),
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
                "total_duration_s": f"{duration:.1f}" if args.durations else "",
            })

            total_files += n_files
            total_avail += n_avail if n_avail is not None else 0
            total_tiny += tiny
            total_duration += duration

    # Grand total row
    if total_avail > 0:
        total_pct = f"{total_files / total_avail * 100:.1f}%"
        total_pct_val = f"{total_files / total_avail * 100:.1f}"
    else:
        total_pct = "N/A"
        total_pct_val = ""

    total_dur_str = f"{total_duration:.1f}s" if args.durations else ""

    table_rows.append({
        "species": "TOTAL",
        "quality": "",
        "files": str(total_files),
        "avail": str(total_avail) if metadata else "N/A",
        "pct": total_pct,
        "tiny": str(total_tiny),
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
        "total_duration_s": f"{total_duration:.1f}" if args.durations else "",
    })

    # Print table
    print_table(table_rows, args.durations)

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
