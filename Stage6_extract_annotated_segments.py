#!/usr/bin/env python3
"""
Stage 6: Extract annotated segments from FLAC files to 3-second WAV files.

Scans the input directory for .txt annotation files matching .flac files,
then extracts the annotated segments as 3-second WAV files.

Annotation file format (one segment per line):
    start_time\tend_time\tlabel\tindex

Example:
    2.345\t5.345\tsong\t0
    7.890\t10.890\tsong\t1
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

from config import PER_SPECIES_FLACS, EXTRACTED_SEGS


def find_annotation_files(input_dir, recursive=True):
    """
    Find all .txt annotation files in the input directory.

    Returns:
        List of tuples: [(annotation_path, flac_path), ...]
    """
    input_path = Path(input_dir)
    pairs = []

    if recursive:
        txt_files = input_path.rglob("*.txt")
    else:
        txt_files = input_path.glob("*.txt")

    for txt_file in txt_files:
        # Find corresponding FLAC file (same name, different extension)
        flac_file = txt_file.with_suffix(".flac")

        if flac_file.exists():
            pairs.append((txt_file, flac_file))
        else:
            print(f"Warning: No matching FLAC file for {txt_file}")

    return pairs


def parse_annotation_file(annotation_path):
    """
    Parse annotation file and return list of segments.

    Format: start\tend\tlabel\tindex

    Returns:
        List of tuples: [(start_time, end_time, label, index), ...]
    """
    segments = []

    with open(annotation_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"Warning: Skipping malformed line {line_num} in {annotation_path}: {line}")
                    continue

                start_time = float(parts[0])
                end_time = float(parts[1])
                label = parts[2] if len(parts) > 2 else "unknown"
                index = parts[3] if len(parts) > 3 else str(line_num - 1)

                # Validate segment duration (should be close to 3 seconds)
                duration = end_time - start_time
                if abs(duration - 3.0) > 0.1:
                    print(f"Warning: Segment {index} in {annotation_path} has duration {duration:.2f}s (expected ~3.0s)")

                segments.append((start_time, end_time, label, index))

            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {line_num} in {annotation_path}: {e}")
                continue

    return segments


def extract_segment(audio, sr, start_time, end_time, target_duration=3.0):
    """
    Extract a segment from audio and ensure it's exactly target_duration seconds.

    Returns:
        numpy array of audio samples
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract segment
    segment = audio[start_sample:end_sample]

    # Ensure exact target duration
    target_samples = int(target_duration * sr)

    if len(segment) < target_samples:
        # Pad if too short
        segment = np.pad(segment, (0, target_samples - len(segment)), mode='constant')
    elif len(segment) > target_samples:
        # Trim if too long
        segment = segment[:target_samples]

    return segment


def get_species_from_path(flac_path, input_dir):
    """
    Extract species name from the directory structure.

    Assumes structure: {input_dir}/{Species name}/{quality}/file.flac

    Returns:
        species name (str) or "unknown"
    """
    try:
        # Get relative path from input_dir
        rel_path = Path(flac_path).relative_to(input_dir)
        # First directory is species name
        species = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
        return species
    except (ValueError, IndexError):
        return "unknown"


def process_annotation_file(annotation_path, flac_path, output_dir, input_dir,
                            target_sr=16000, audio_format='wav'):
    """
    Process a single annotation file and extract all segments.

    Returns:
        dict with extraction statistics
    """
    stats = {
        'total_segments': 0,
        'extracted': 0,
        'skipped': 0,
        'errors': 0
    }

    # Parse annotations
    segments = parse_annotation_file(annotation_path)
    stats['total_segments'] = len(segments)

    if not segments:
        print(f"No segments found in {annotation_path}")
        return stats

    # Load audio
    try:
        audio, sr = librosa.load(flac_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading {flac_path}: {e}")
        stats['errors'] = stats['total_segments']
        return stats

    # Get species name from directory structure
    species = get_species_from_path(flac_path, input_dir)

    # Create output directory for this species
    species_output_dir = Path(output_dir) / species
    species_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract each segment
    flac_basename = flac_path.stem  # e.g., "xc123456"

    for start_time, end_time, label, index in segments:
        try:
            # Extract segment
            segment = extract_segment(audio, sr, start_time, end_time)

            # Resample if needed
            if sr != target_sr:
                segment = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)

            # Generate output filename: {xc_id}_{index}.wav
            output_filename = f"{flac_basename}_{index}.{audio_format}"
            output_path = species_output_dir / output_filename

            # Skip if already exists
            if output_path.exists():
                stats['skipped'] += 1
                continue

            # Save as WAV
            sf.write(output_path, segment, target_sr, subtype='PCM_16')
            stats['extracted'] += 1

        except Exception as e:
            print(f"Error extracting segment {index} from {flac_path}: {e}")
            stats['errors'] += 1
            continue

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Extract annotated segments from FLAC files to WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all annotated segments
  python Stage6_extract_annotated_segments.py /Volumes/Evo/xc-mygarden-flac --output-dir ./extracted_segments

  # Extract with custom sample rate
  python Stage6_extract_annotated_segments.py /Volumes/Evo/xc-mygarden-flac --output-dir ./extracted_segments --sample-rate 22050

  # Non-recursive (only search top-level directory)
  python Stage6_extract_annotated_segments.py /Volumes/Evo/xc-mygarden-flac --output-dir ./extracted_segments --no-recursive
        """
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(PER_SPECIES_FLACS),
        help=f"Input directory containing FLAC files and .txt annotation files. Default: {PER_SPECIES_FLACS}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXTRACTED_SEGS),
        help=f"Output directory for extracted WAV segments. Default: {EXTRACTED_SEGS}",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for output WAV files. Default: 16000"
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "flac"],
        help="Output audio format. Default: wav"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories for annotation files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without actually extracting."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)

    print("="*80)
    print("STAGE 6: EXTRACT ANNOTATED SEGMENTS")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target sample rate: {args.sample_rate} Hz")
    print(f"Output format: {args.format}")
    print(f"Recursive search: {not args.no_recursive}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Find annotation files
    print("Scanning for annotation files...")
    annotation_pairs = find_annotation_files(input_dir, recursive=not args.no_recursive)

    if not annotation_pairs:
        print("No annotation files (.txt) with matching FLAC files found.")
        sys.exit(0)

    print(f"Found {len(annotation_pairs)} annotation files with matching FLAC files.")
    print()

    if args.dry_run:
        print("[DRY RUN] Would process the following files:")
        for txt_file, flac_file in annotation_pairs:
            segments = parse_annotation_file(txt_file)
            species = get_species_from_path(flac_file, input_dir)
            print(f"  {txt_file.name} -> {flac_file.name} ({len(segments)} segments, species: {species})")
        print()
        print(f"Total segments that would be extracted: {sum(len(parse_annotation_file(t)) for t, _ in annotation_pairs)}")
        sys.exit(0)

    # Process each annotation file
    total_stats = defaultdict(int)
    species_stats = defaultdict(lambda: defaultdict(int))

    print("Extracting segments...")
    for annotation_path, flac_path in tqdm(annotation_pairs, desc="Processing files"):
        stats = process_annotation_file(
            annotation_path, flac_path, output_dir, input_dir,
            target_sr=args.sample_rate, audio_format=args.format
        )

        # Update totals
        for key, value in stats.items():
            total_stats[key] += value

        # Update species-specific stats
        species = get_species_from_path(flac_path, input_dir)
        for key, value in stats.items():
            species_stats[species][key] += value

    # Print summary
    print()
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total annotation files processed: {len(annotation_pairs)}")
    print(f"Total segments found: {total_stats['total_segments']}")
    print(f"Successfully extracted: {total_stats['extracted']}")
    print(f"Skipped (already exist): {total_stats['skipped']}")
    print(f"Errors: {total_stats['errors']}")
    print()

    print("Per-species breakdown:")
    print(f"{'Species':<30} {'Total':>8} {'Extracted':>10} {'Skipped':>8} {'Errors':>8}")
    print("-"*80)
    for species in sorted(species_stats.keys()):
        s = species_stats[species]
        print(f"{species:<30} {s['total_segments']:>8} {s['extracted']:>10} {s['skipped']:>8} {s['errors']:>8}")
    print()

    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
