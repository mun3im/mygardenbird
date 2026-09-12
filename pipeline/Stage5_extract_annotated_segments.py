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

from config import PER_SPECIES_FLACS, MYGARDENBIRD_16K, MYGARDENBIRD_44K, get_profile, folder_name


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
                # Always derive the filename suffix from onset_ms so output
                # filenames consistently follow xc{id}_{onset_ms}.wav.
                index = str(int(round(start_time * 1000)))

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
    Extract the RAW species folder name from the directory structure (as it
    literally appears on disk under input_dir -- may be underscore-styled,
    e.g. SEA-BIRD30's flacs/ convention).

    Assumes structure: {input_dir}/{Species name}/{quality}/file.flac

    Returns:
        species folder name (str) or "unknown"
    """
    try:
        # Get relative path from input_dir
        rel_path = Path(flac_path).relative_to(input_dir)
        # First directory is species name
        species = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
        return species
    except (ValueError, IndexError):
        return "unknown"


def species_output_name(raw_species_folder, profile):
    """
    Translate a raw input species folder name (as found under
    profile.per_species_flacs, e.g. SEA-BIRD30's underscore-styled
    "Malaysian_Pied_Fantail") to the output folder name matching
    profile.clips_dirs' convention (e.g. SEA-BIRD30's wavs/ is
    space-styled: "Pied Fantail").

    Uses profile.species_overrides as a reverse lookup first (so
    "Malaysian_Pied_Fantail" correctly maps back to "Pied Fantail", not a
    naive "Malaysian Pied Fantail"), then falls back to a plain
    underscore<->space swap for the (more common) unambiguous case.
    If input and output folder styles already match (MyGardenBird: both
    "space"), this is a no-op.
    """
    in_style = profile.folder_styles.get("per_species_flacs", "space")
    out_style = profile.folder_styles.get("clips", "space")
    if in_style == out_style:
        return raw_species_folder

    # Reverse-lookup: does this raw folder name match an override's value?
    for common_name, override_folder in profile.species_overrides.items():
        if override_folder == raw_species_folder:
            return common_name if out_style == "space" else common_name.replace(" ", "_")

    if out_style == "space":
        return raw_species_folder.replace("_", " ")
    return raw_species_folder.replace(" ", "_")


def process_annotation_file(annotation_path, flac_path, output_dir, input_dir,
                            target_sr=16000, audio_format='wav', no_upsample=False,
                            profile=None):
    """
    Process a single annotation file and extract all segments.

    If no_upsample=True, files whose native sample rate is below target_sr are
    skipped entirely (all their segments counted as 'upsample_skip').

    `profile` (a config.DatasetProfile), if given, is used to translate the
    raw input species folder name to the output naming convention via
    species_output_name() -- needed when input_dir's and output_dir's
    folder styles differ (e.g. SEA-BIRD30's underscore-styled flacs/ vs
    space-styled wavs/). If None, the raw folder name is used unchanged
    (MyGardenBird's historical behavior, where input/output styles match).

    Returns:
        dict with extraction statistics
    """
    stats = {
        'total_segments': 0,
        'extracted': 0,
        'skipped': 0,
        'upsample_skip': 0,
        'errors': 0
    }

    # Parse annotations
    segments = parse_annotation_file(annotation_path)
    if len(segments) > 10:
        segments = segments[:10]   # hard limit: max 10 clips per source file
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

    # Skip source files that would require upsampling when --no-upsample is set
    if no_upsample and sr < target_sr:
        stats['upsample_skip'] = stats['total_segments']
        return stats

    # Get species name from directory structure, translated to the output
    # naming convention if the two differ (see species_output_name()).
    species = get_species_from_path(flac_path, input_dir)
    output_species = species_output_name(species, profile) if profile is not None else species

    # Create output directory for this species
    species_output_dir = Path(output_dir) / output_species
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
  python Stage5_extract_annotated_segments.py /Volumes/Evo/xc-mygarden-flac --output-dir ./extracted_segments

  # Extract with custom sample rate
  python Stage5_extract_annotated_segments.py /Volumes/Evo/xc-mygarden-flac --output-dir ./extracted_segments --sample-rate 22050

  # Non-recursive (only search top-level directory)
  python Stage5_extract_annotated_segments.py /Volumes/Evo/xc-mygarden-flac --output-dir ./extracted_segments --no-recursive
        """
    )

    parser.add_argument(
        "--dataset", choices=["mygardenbird", "sea-bird30"],
        default=os.environ.get("PIPELINE_DATASET", "mygardenbird"),
        help="Which dataset's default paths to use. Default: mygardenbird "
             "(or $PIPELINE_DATASET if set).",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help=f"Input directory containing FLAC files and .txt annotation files. Default: "
             f"the selected dataset's per_species_flacs dir (MyGardenBird: {PER_SPECIES_FLACS}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for extracted WAV segments. If not specified, auto-detects "
             "based on --sample-rate and the selected dataset's clips_dirs "
             f"(MyGardenBird: 16kHz → {MYGARDENBIRD_16K}, 44.1kHz → {MYGARDENBIRD_44K}).",
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
        "--no-upsample",
        action="store_true",
        help="Skip source files whose native sample rate is below --sample-rate. "
             "Use with --sample-rate 44100 to produce a high-fidelity subset that "
             "contains only recordings originally captured at ≥44.1 kHz."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without actually extracting."
    )

    args = parser.parse_args()

    profile = get_profile(args.dataset)

    if args.input_dir is None:
        args.input_dir = str(profile.per_species_flacs)
    input_dir = Path(args.input_dir)

    # Auto-select output directory based on sample rate if not explicitly
    # specified -- looked up from the dataset profile's clips_dirs instead
    # of hardcoded 16k/44k branches + a "mygardenbird{khz}" string-built
    # fallback, so an unsupported rate fails clearly instead of silently
    # inventing a wrong path (e.g. SEA-BIRD30 has no 44kHz variant at all).
    if args.output_dir is None:
        if args.sample_rate not in profile.clips_dirs:
            supported = sorted(profile.clips_dirs)
            print(f"Error: dataset '{args.dataset}' has no {args.sample_rate} Hz variant. "
                  f"Supported sample rates: {supported}. Pass --output-dir to override.")
            sys.exit(1)
        output_dir = profile.clips_dirs[args.sample_rate]
    else:
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
    print("WHAT THIS DOES:")
    print("  - Reads .txt annotation files created in Stage 5")
    print("  - Extracts 3-second segments from FLAC recordings")
    print("  - Resamples audio to target sample rate (16kHz or 44kHz)")
    print("  - Saves as WAV files organized by species")
    print()
    print("INPUT:")
    print(f"  - FLAC recordings: {input_dir}/<Species Name>/<Quality>/xc####.flac")
    print(f"  - Annotations: {input_dir}/<Species Name>/<Quality>/xc####.txt")
    print()
    print("OUTPUT:")
    print(f"  - WAV segments: {output_dir}/<Species Name>/xc####_<onset_ms>.wav")
    print(f"      Example: {output_dir}/Javan Myna/xc123456_2860.wav")
    print()
    print("CONFIGURATION:")
    print(f"  - Target sample rate: {args.sample_rate} Hz")
    print(f"  - No-upsample filter: {args.no_upsample}")
    print(f"  - Output format: {args.format}")
    print(f"  - Recursive search: {not args.no_recursive}")
    print(f"  - Dry run: {args.dry_run}")
    print("="*80)
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
            output_species = species_output_name(species, profile)
            print(f"  {txt_file.name} -> {flac_file.name} ({len(segments)} segments, species: {output_species})")
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
            target_sr=args.sample_rate, audio_format=args.format,
            no_upsample=args.no_upsample, profile=profile,
        )

        # Update totals
        for key, value in stats.items():
            total_stats[key] += value

        # Update species-specific stats (output naming convention, matching
        # the folders actually created above)
        species = get_species_from_path(flac_path, input_dir)
        output_species = species_output_name(species, profile)
        for key, value in stats.items():
            species_stats[output_species][key] += value
        species_stats[output_species]['source_files'] += 1

    # Print summary
    print()
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    total_src = len(annotation_pairs)
    overall_avg = total_stats['total_segments'] / total_src if total_src > 0 else 0.0
    print(f"Total source files processed: {total_src}")
    print(f"Average clips per source file: {overall_avg:.1f}")
    print(f"Total segments found: {total_stats['total_segments']}")
    print(f"Successfully extracted: {total_stats['extracted']}")
    print(f"Skipped (already exist): {total_stats['skipped']}")
    if total_stats['upsample_skip']:
        print(f"Skipped (native rate < target, --no-upsample): {total_stats['upsample_skip']}")
    print(f"Errors: {total_stats['errors']}")
    print()

    print("Per-species breakdown:")
    print(f"{'Species':<30} {'SrcFiles':>9} {'Avg Clips/Src':>14} {'Total':>8} {'Extracted':>10} {'Skipped':>8} {'Errors':>8}")
    print("-"*93)
    for species in sorted(species_stats.keys()):
        s = species_stats[species]
        src = s['source_files']
        avg = s['total_segments'] / src if src > 0 else 0.0
        print(f"{species:<30} {src:>9} {avg:>14.1f} {s['total_segments']:>8} {s['extracted']:>10} {s['skipped']:>8} {s['errors']:>8}")
    print()

    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
