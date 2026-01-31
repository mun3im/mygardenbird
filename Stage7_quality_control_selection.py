#!/usr/bin/env python3
"""
Stage 7: Quality control and dataset preparation for training.

Analyzes extracted WAV segments, provides quality metrics, and creates
train/val/test splits for ML training.

Quality checks:
- Audio duration validation
- Sample rate verification
- Silence detection
- Clipping detection
- File corruption detection
- Class balance analysis

Output:
- QC report (CSV + TXT)
- Train/val/test splits (with configurable ratios)
- Dataset statistics
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


def check_audio_quality(audio_path, target_sr=16000, target_duration=3.0,
                        silence_threshold=-40, clipping_threshold=0.99):
    """
    Perform quality checks on an audio file.

    Returns:
        dict with quality metrics
    """
    metrics = {
        'file': audio_path.name,
        'valid': True,
        'error': None,
        'duration': 0.0,
        'sample_rate': 0,
        'num_samples': 0,
        'rms_db': 0.0,
        'peak_amplitude': 0.0,
        'is_silent': False,
        'is_clipped': False,
        'duration_ok': False,
        'sample_rate_ok': False
    }

    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

        # Basic metrics
        metrics['sample_rate'] = sr
        metrics['num_samples'] = len(audio)
        metrics['duration'] = len(audio) / sr
        metrics['sample_rate_ok'] = (sr == target_sr)
        metrics['duration_ok'] = abs(metrics['duration'] - target_duration) < 0.1

        # RMS (dB)
        rms = np.sqrt(np.mean(audio ** 2))
        metrics['rms_db'] = 20 * np.log10(rms + 1e-10)

        # Peak amplitude
        metrics['peak_amplitude'] = np.max(np.abs(audio))

        # Silence detection
        metrics['is_silent'] = metrics['rms_db'] < silence_threshold

        # Clipping detection
        metrics['is_clipped'] = metrics['peak_amplitude'] > clipping_threshold

    except Exception as e:
        metrics['valid'] = False
        metrics['error'] = str(e)

    return metrics


def analyze_dataset(input_dir, target_sr=16000, target_duration=3.0):
    """
    Analyze all audio files in the dataset.

    Returns:
        dict mapping species -> list of quality metrics
    """
    input_path = Path(input_dir)
    results = defaultdict(list)

    if not input_path.exists():
        print(f"Error: Directory does not exist: {input_dir}")
        return results

    # Find all species directories
    species_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    print(f"Found {len(species_dirs)} species directories")
    print()

    for species_dir in tqdm(species_dirs, desc="Analyzing species"):
        species = species_dir.name

        # Find all audio files
        audio_files = list(species_dir.glob("*.wav")) + list(species_dir.glob("*.flac"))

        for audio_file in audio_files:
            metrics = check_audio_quality(audio_file, target_sr, target_duration)
            metrics['species'] = species
            results[species].append(metrics)

    return results


def print_summary_report(results):
    """Print a summary report to console."""
    total_files = sum(len(files) for files in results.values())
    total_valid = sum(sum(1 for m in files if m['valid']) for files in results.values())
    total_silent = sum(sum(1 for m in files if m['is_silent']) for files in results.values())
    total_clipped = sum(sum(1 for m in files if m['is_clipped']) for files in results.values())
    total_duration_ok = sum(sum(1 for m in files if m['duration_ok']) for files in results.values())
    total_sr_ok = sum(sum(1 for m in files if m['sample_rate_ok']) for files in results.values())

    print("="*80)
    print("QUALITY CONTROL SUMMARY")
    print("="*80)
    print(f"Total files: {total_files}")
    print(f"Valid files: {total_valid} ({total_valid/total_files*100:.1f}%)")
    print(f"Invalid/corrupt files: {total_files - total_valid} ({(total_files-total_valid)/total_files*100:.1f}%)")
    print()
    print(f"Silent files (RMS < -40dB): {total_silent} ({total_silent/total_files*100:.1f}%)")
    print(f"Clipped files (peak > 0.99): {total_clipped} ({total_clipped/total_files*100:.1f}%)")
    print(f"Correct duration (~3s): {total_duration_ok} ({total_duration_ok/total_files*100:.1f}%)")
    print(f"Correct sample rate: {total_sr_ok} ({total_sr_ok/total_files*100:.1f}%)")
    print()

    print("Per-species breakdown:")
    print(f"{'Species':<30} {'Total':>8} {'Valid':>8} {'Silent':>8} {'Clipped':>8}")
    print("-"*80)

    for species in sorted(results.keys()):
        files = results[species]
        n_total = len(files)
        n_valid = sum(1 for m in files if m['valid'])
        n_silent = sum(1 for m in files if m['is_silent'])
        n_clipped = sum(1 for m in files if m['is_clipped'])
        print(f"{species:<30} {n_total:>8} {n_valid:>8} {n_silent:>8} {n_clipped:>8}")

    print("="*80)


def save_detailed_report(results, output_path):
    """Save detailed QC report to CSV."""
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'species', 'file', 'valid', 'error',
            'duration', 'sample_rate', 'num_samples',
            'rms_db', 'peak_amplitude',
            'is_silent', 'is_clipped',
            'duration_ok', 'sample_rate_ok'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for species, files in sorted(results.items()):
            for metrics in files:
                writer.writerow(metrics)

    print(f"Detailed report saved to: {output_path}")


def filter_files(results, remove_silent=True, remove_clipped=True,
                remove_invalid=True, remove_wrong_duration=False,
                remove_wrong_sr=False):
    """
    Filter files based on quality criteria.

    Returns:
        dict mapping species -> list of valid file paths
    """
    filtered = defaultdict(list)

    for species, files in results.items():
        for metrics in files:
            # Apply filters
            if remove_invalid and not metrics['valid']:
                continue
            if remove_silent and metrics['is_silent']:
                continue
            if remove_clipped and metrics['is_clipped']:
                continue
            if remove_wrong_duration and not metrics['duration_ok']:
                continue
            if remove_wrong_sr and not metrics['sample_rate_ok']:
                continue

            filtered[species].append(metrics['file'])

    return filtered


def create_splits(filtered_files, input_dir, output_dir,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 min_samples_per_class=10, seed=42):
    """
    Create train/val/test splits.

    Returns:
        dict with split statistics
    """
    random.seed(seed)
    np.random.seed(seed)

    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"

    # Create directories
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_files': 0,
        'total_species': 0,
        'skipped_species': [],
        'train_files': 0,
        'val_files': 0,
        'test_files': 0,
        'species_counts': {}
    }

    print()
    print("Creating train/val/test splits...")
    print(f"Ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
    print(f"Minimum samples per class: {min_samples_per_class}")
    print()

    for species, files in tqdm(sorted(filtered_files.items()), desc="Splitting species"):
        n_files = len(files)

        # Skip species with too few samples
        if n_files < min_samples_per_class:
            print(f"Skipping {species}: only {n_files} samples (< {min_samples_per_class})")
            stats['skipped_species'].append(species)
            continue

        # Shuffle files
        files_shuffled = list(files)
        random.shuffle(files_shuffled)

        # Calculate split sizes
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        n_test = n_files - n_train - n_val  # Remainder goes to test

        # Ensure at least 1 sample in each split
        if n_train == 0 or n_val == 0 or n_test == 0:
            # Rebalance to ensure at least 1 per split
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_files - n_train - n_val)

        train_files = files_shuffled[:n_train]
        val_files = files_shuffled[n_train:n_train + n_val]
        test_files = files_shuffled[n_train + n_val:]

        # Create species subdirectories
        (train_dir / species).mkdir(exist_ok=True)
        (val_dir / species).mkdir(exist_ok=True)
        (test_dir / species).mkdir(exist_ok=True)

        # Copy files to splits
        for file in train_files:
            src = Path(input_dir) / species / file
            dst = train_dir / species / file
            if src.exists():
                shutil.copy2(src, dst)

        for file in val_files:
            src = Path(input_dir) / species / file
            dst = val_dir / species / file
            if src.exists():
                shutil.copy2(src, dst)

        for file in test_files:
            src = Path(input_dir) / species / file
            dst = test_dir / species / file
            if src.exists():
                shutil.copy2(src, dst)

        # Update stats
        stats['total_files'] += n_files
        stats['total_species'] += 1
        stats['train_files'] += len(train_files)
        stats['val_files'] += len(val_files)
        stats['test_files'] += len(test_files)
        stats['species_counts'][species] = {
            'total': n_files,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }

    return stats


def print_split_summary(stats):
    """Print summary of dataset splits."""
    print()
    print("="*80)
    print("DATASET SPLIT SUMMARY")
    print("="*80)
    print(f"Total species included: {stats['total_species']}")
    print(f"Species skipped (too few samples): {len(stats['skipped_species'])}")
    if stats['skipped_species']:
        print(f"  Skipped: {', '.join(stats['skipped_species'])}")
    print()
    print(f"Total files: {stats['total_files']}")
    print(f"Train files: {stats['train_files']} ({stats['train_files']/stats['total_files']*100:.1f}%)")
    print(f"Val files: {stats['val_files']} ({stats['val_files']/stats['total_files']*100:.1f}%)")
    print(f"Test files: {stats['test_files']} ({stats['test_files']/stats['total_files']*100:.1f}%)")
    print()

    print("Per-species split sizes:")
    print(f"{'Species':<30} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-"*80)

    for species in sorted(stats['species_counts'].keys()):
        counts = stats['species_counts'][species]
        print(f"{species:<30} {counts['total']:>8} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Quality control and dataset preparation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset and create splits
  python Stage7_quality_control_selection.py ./extracted_segments --output-dir ./dataset

  # Custom split ratios
  python Stage7_quality_control_selection.py ./extracted_segments --output-dir ./dataset --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # QC only (no splitting)
  python Stage7_quality_control_selection.py ./extracted_segments --qc-only
        """
    )

    parser.add_argument(
        "input_dir",
        help="Input directory containing extracted WAV segments (organized by species)."
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory for train/val/test splits. Default: ./dataset"
    )
    parser.add_argument(
        "--qc-only",
        action="store_true",
        help="Only run quality control analysis, don't create splits."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio. Default: 0.7"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio. Default: 0.15"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio. Default: 0.15"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per class to include. Default: 10"
    )
    parser.add_argument(
        "--keep-silent",
        action="store_true",
        help="Don't filter out silent files."
    )
    parser.add_argument(
        "--keep-clipped",
        action="store_true",
        help="Don't filter out clipped files."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Expected sample rate. Default: 16000"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits. Default: 42"
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    print("="*80)
    print("STAGE 7: QUALITY CONTROL & DATASET PREPARATION")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"QC only: {args.qc_only}")
    print()

    # Run quality analysis
    print("Running quality analysis...")
    results = analyze_dataset(input_dir, target_sr=args.sample_rate)

    if not results:
        print("No audio files found.")
        sys.exit(0)

    # Print summary
    print()
    print_summary_report(results)

    # Save detailed report
    qc_report_path = output_dir / "qc_report.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_detailed_report(results, qc_report_path)

    if args.qc_only:
        print()
        print("QC analysis complete. Exiting (--qc-only specified).")
        sys.exit(0)

    # Filter files based on quality
    print()
    print("Filtering files based on quality criteria...")
    filtered = filter_files(
        results,
        remove_silent=not args.keep_silent,
        remove_clipped=not args.keep_clipped,
        remove_invalid=True
    )

    total_filtered = sum(len(files) for files in filtered.values())
    total_original = sum(len(files) for files in results.values())
    print(f"Kept {total_filtered}/{total_original} files after filtering ({total_filtered/total_original*100:.1f}%)")

    # Create splits
    split_stats = create_splits(
        filtered, input_dir, output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_samples_per_class=args.min_samples,
        seed=args.seed
    )

    # Print split summary
    print_split_summary(split_stats)

    # Save split stats
    split_stats_path = output_dir / "split_statistics.txt"
    with open(split_stats_path, 'w') as f:
        f.write(f"Dataset Split Statistics\n")
        f.write(f"{'='*80}\n")
        f.write(f"Generated: {Path(__file__).name}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Total species: {split_stats['total_species']}\n")
        f.write(f"Total files: {split_stats['total_files']}\n")
        f.write(f"Train: {split_stats['train_files']} ({split_stats['train_files']/split_stats['total_files']*100:.1f}%)\n")
        f.write(f"Val: {split_stats['val_files']} ({split_stats['val_files']/split_stats['total_files']*100:.1f}%)\n")
        f.write(f"Test: {split_stats['test_files']} ({split_stats['test_files']/split_stats['total_files']*100:.1f}%)\n")
        f.write(f"\n")
        f.write(f"Per-species counts:\n")
        f.write(f"{'Species':<30} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}\n")
        f.write(f"{'-'*80}\n")
        for species in sorted(split_stats['species_counts'].keys()):
            counts = split_stats['species_counts'][species]
            f.write(f"{species:<30} {counts['total']:>8} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8}\n")

    print(f"\nSplit statistics saved to: {split_stats_path}")
    print()
    print("Dataset preparation complete!")
    print(f"Train/val/test directories: {output_dir}")


if __name__ == "__main__":
    main()
