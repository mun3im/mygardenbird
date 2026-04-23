#!/usr/bin/env python3
"""
Benchmark BirdNET-Analyzer on MyGardenBird test set.

This script evaluates BirdNET's performance on the same test set used for
MynaNet and other custom models, providing a baseline comparison.

Usage Examples
--------------
# Basic usage with default settings:
python Stage10_benchmark_birdnet.py \\
    --splits_csv /Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_75_10_15.csv \\
    --dataset_root /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz

# Adjust BirdNET confidence threshold:
python Stage10_benchmark_birdnet.py \\
    --splits_csv /Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_75_10_15.csv \\
    --dataset_root /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz \\
    --min_conf 0.25

# Specify custom location for better regional predictions:
python Stage10_benchmark_birdnet.py \\
    --splits_csv /Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_75_10_15.csv \\
    --dataset_root /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz \\
    --lat 3.139 \\
    --lon 101.687

Defaults
--------
--min_conf       0.1       (BirdNET confidence threshold)
--lat            3.139     (Kuala Lumpur latitude)
--lon            101.687   (Kuala Lumpur longitude)
--output_dir     ./birdnet_benchmark

Requirements
------------
- BirdNET-Analyzer installed (see setup_birdnet.sh)
- Conda environment 'birdnet' activated
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import config if available
try:
    from config import MYGARDENBIRD_16K, METADATA_16K
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config.py not found. Using default paths.")
    MYGARDENBIRD_16K = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
    METADATA_16K = "/Volumes/Evo/MYGARDENBIRD/metadata16khz"


# ---------------------------------------------------------------------------
# SPECIES MAPPING
# ---------------------------------------------------------------------------
def create_species_mapping():
    """
    Create mapping from BirdNET common names to MyGardenBird directory names.

    BirdNET uses eBird common names, which may differ slightly from
    directory names in MyGardenBird.
    """
    return {
        'Asian Koel': 'Asian Koel',
        'Collared Kingfisher': 'Collared Kingfisher',
        'Common Iora': 'Common Iora',
        'Common Tailorbird': 'Common Tailorbird',
        'Coppersmith Barbet': 'Coppersmith Barbet',
        'Large-tailed Nightjar': 'Large-tailed Nightjar',
        'Olive-backed Sunbird': 'Olive-backed Sunbird',
        'Spotted Dove': 'Spotted Dove',
        'White-breasted Waterhen': 'White-breasted Waterhen',
        'White-throated Kingfisher': 'White-throated Kingfisher',
    }


def map_birdnet_to_mygardenbird(birdnet_name, species_mapping):
    """
    Map BirdNET species prediction to MyGardenBird species name.

    Args:
        birdnet_name: Common name returned by BirdNET
        species_mapping: Dictionary mapping BirdNET names to MyGardenBird names

    Returns:
        Mapped species name, or None if no match found
    """
    # Try direct match
    if birdnet_name in species_mapping:
        return species_mapping[birdnet_name]

    # Try case-insensitive match
    for key, value in species_mapping.items():
        if key.lower() == birdnet_name.lower():
            return value

    return None


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def parse_splits_csv(csv_path):
    """
    Read splits CSV into a {filename: split} dict.

    Supports Stage8 format:
      - file_id,split (e.g., xc1002657_2860,test)

    Comment lines starting with '#' are skipped.
    """
    import csv as _csv
    from io import StringIO

    splits = {}
    with open(csv_path, 'r') as f:
        content = ''.join(line for line in f if not line.startswith('#'))

    reader = _csv.DictReader(StringIO(content))
    for row in reader:
        sp = row['split']
        if 'file_id' in row:
            # Stage8 format: xc1002657_2860 -> xc1002657_2860.wav
            fid = row['file_id']
            if fid.startswith('XC') or fid.startswith('xc'):
                wav = fid.lower() + '.wav'
            else:
                wav = fid + '.wav'
        else:
            wav = row['filename']
        splits[wav] = sp
    return splits


def load_test_set(splits_csv, dataset_root):
    """
    Load test set file paths and ground truth labels.

    Args:
        splits_csv: Path to splits CSV from Stage8
        dataset_root: Root directory containing species subdirectories

    Returns:
        List of dicts with 'filepath', 'filename', 'ground_truth'
    """
    # Parse splits CSV
    splits = parse_splits_csv(splits_csv)

    # Build file lookup from dataset
    test_files = [fn for fn, split in splits.items() if split == 'test']

    # Find actual files and extract labels
    test_data = []
    dataset_root = Path(dataset_root)

    for species_dir in dataset_root.iterdir():
        if not species_dir.is_dir() or species_dir.name.startswith('.'):
            continue

        species_name = species_dir.name

        for audio_file in species_dir.iterdir():
            if audio_file.name in test_files:
                test_data.append({
                    'filepath': str(audio_file),
                    'filename': audio_file.name,
                    'ground_truth': species_name
                })

    print(f"Loaded {len(test_data)} test files")
    print(f"  Species: {len(set([d['ground_truth'] for d in test_data]))}")
    return test_data


# ---------------------------------------------------------------------------
# BIRDNET INFERENCE
# ---------------------------------------------------------------------------
def run_birdnet_analyze(audio_file, min_conf=0.1, lat=3.139, lon=101.687):
    """
    Run BirdNET analysis on a single audio file.

    Args:
        audio_file: Path to audio file
        min_conf: Minimum confidence threshold (0.0-1.0)
        lat: Latitude for location-based filtering
        lon: Longitude for location-based filtering

    Returns:
        Dict with 'scientific_name', 'common_name', 'confidence'
        or None if no detection above threshold
    """
    try:
        # Import BirdNET's analyze module
        from birdnet_analyzer import analyze
        import tempfile
        import csv

        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run analysis - BirdNET v2.4.0 saves results to file
            analyze(
                audio_input=audio_file,
                output=tmpdir,
                min_conf=min_conf,
                lat=lat,
                lon=lon,
                rtype='csv'
            )

            # Read results from CSV file
            result_file = Path(tmpdir) / Path(audio_file).with_suffix('.csv').name
            if not result_file.exists():
                return None

            # Parse CSV results
            detections = []
            with open(result_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    detections.append({
                        'scientific_name': row.get('Scientific name', ''),
                        'common_name': row.get('Common name', ''),
                        'confidence': float(row.get('Confidence', 0))
                    })

            # Return top prediction by confidence
            if detections:
                return max(detections, key=lambda x: x['confidence'])
            else:
                return None

    except Exception as e:
        print(f"\n  Error analyzing {audio_file}: {e}")
        return None


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def evaluate_birdnet(test_data, species_mapping, min_conf=0.1, lat=3.139, lon=101.687):
    """
    Run BirdNET on all test files and compute accuracy.

    Args:
        test_data: List of test file dicts
        species_mapping: Mapping from BirdNET names to MyGardenBird names
        min_conf: BirdNET confidence threshold
        lat: Latitude for location filtering
        lon: Longitude for location filtering

    Returns:
        Tuple of (predictions, ground_truths, stats_dict)
    """
    predictions = []
    ground_truths = []
    no_detection_count = 0
    unknown_species_count = 0

    print(f"\nRunning BirdNET on {len(test_data)} test files...")
    print(f"  Min confidence: {min_conf}")
    print(f"  Location: ({lat}, {lon})")

    for item in tqdm(test_data, desc="BirdNET inference"):
        # Run BirdNET
        result = run_birdnet_analyze(
            item['filepath'],
            min_conf=min_conf,
            lat=lat,
            lon=lon
        )

        if result:
            # Map BirdNET prediction to MyGardenBird species
            predicted_species = map_birdnet_to_mygardenbird(
                result['common_name'],
                species_mapping
            )

            if predicted_species:
                predictions.append(predicted_species)
            else:
                # BirdNET predicted a species not in MyGardenBird
                predictions.append(f"UNKNOWN_{result['common_name']}")
                unknown_species_count += 1
        else:
            # No prediction (low confidence)
            predictions.append("NO_DETECTION")
            no_detection_count += 1

        ground_truths.append(item['ground_truth'])

    stats = {
        'total_samples': len(test_data),
        'no_detections': no_detection_count,
        'unknown_species': unknown_species_count
    }

    return predictions, ground_truths, stats


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
def generate_report(predictions, ground_truths, stats, args, output_dir):
    """
    Generate classification report and confusion matrix.

    Args:
        predictions: List of predicted species names
        ground_truths: List of ground truth species names
        stats: Dict with evaluation statistics
        args: Command-line arguments
        output_dir: Directory to save results
    """
    # Filter out NO_DETECTION and UNKNOWN for accuracy calculation
    valid_preds = []
    valid_truths = []
    for pred, truth in zip(predictions, ground_truths):
        if not pred.startswith('UNKNOWN_') and pred != 'NO_DETECTION':
            valid_preds.append(pred)
            valid_truths.append(truth)

    # Check if we have any valid predictions
    if len(valid_preds) == 0:
        print("\n" + "="*70)
        print("WARNING: No valid predictions found!")
        print("="*70)
        print(f"Total samples: {stats['total_samples']}")
        print(f"No detections: {stats['no_detections']}")
        print(f"Unknown species: {stats['unknown_species']}")
        print("\nSample of predictions:")
        unique_preds = list(set(predictions))[:20]
        for pred in unique_preds:
            count = predictions.count(pred)
            print(f"  {pred}: {count} samples")

        # Save summary
        results = {
            'accuracy': 0.0,
            'total_samples': stats['total_samples'],
            'valid_predictions': 0,
            'no_detections': stats['no_detections'],
            'unknown_species': stats['unknown_species'],
            'min_confidence': args.min_conf,
            'latitude': args.lat,
            'longitude': args.lon,
            'dataset_root': str(args.dataset_root),
            'splits_csv': str(args.splits_csv),
            'timestamp': datetime.now().isoformat(),
            'error': 'No valid predictions - all predictions were NO_DETECTION or UNKNOWN'
        }

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return 0.0

    # Calculate accuracy
    accuracy = accuracy_score(valid_truths, valid_preds)

    # Generate classification report
    species_names = sorted(set(ground_truths))
    report = classification_report(
        valid_truths,
        valid_preds,
        labels=species_names,
        target_names=species_names,
        zero_division=0
    )

    # Generate confusion matrix
    cm = confusion_matrix(valid_truths, valid_preds, labels=species_names)

    # Save results JSON
    results = {
        'accuracy': float(accuracy),
        'total_samples': stats['total_samples'],
        'valid_predictions': len(valid_preds),
        'no_detections': stats['no_detections'],
        'unknown_species': stats['unknown_species'],
        'min_confidence': args.min_conf,
        'latitude': args.lat,
        'longitude': args.lon,
        'dataset_root': str(args.dataset_root),
        'splits_csv': str(args.splits_csv),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save classification report
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(f"BirdNET Benchmark on MyGardenBird Test Set\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Min Confidence: {args.min_conf}\n")
        f.write(f"  Location: ({args.lat}, {args.lon})\n")
        f.write(f"  Splits CSV: {args.splits_csv}\n")
        f.write(f"  Dataset Root: {args.dataset_root}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total Test Samples: {stats['total_samples']}\n")
        f.write(f"  Valid Predictions: {len(valid_preds)}\n")
        f.write(f"  No Detections: {stats['no_detections']}\n")
        f.write(f"  Unknown Species: {stats['unknown_species']}\n")
        f.write(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(f"Classification Report:\n")
        f.write(f"{'-' * 70}\n")
        f.write(report)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=species_names,
        yticklabels=species_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted', fontsize=10, fontfamily='serif')
    plt.ylabel('Ground Truth', fontsize=10, fontfamily='serif')
    plt.title(f'BirdNET Confusion Matrix (Accuracy: {accuracy*100:.2f}%)',
              fontsize=12, fontfamily='serif')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    print(f"\n{'=' * 70}")
    print("BirdNET Benchmark Results")
    print(f"{'=' * 70}")
    print(f"Total Test Samples:    {stats['total_samples']}")
    print(f"Valid Predictions:     {len(valid_preds)}")
    print(f"No Detections:         {stats['no_detections']}")
    print(f"Unknown Species:       {stats['unknown_species']}")
    print(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nResults saved to: {output_dir}/")
    print(f"{'=' * 70}")

    return accuracy


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark BirdNET on MyGardenBird test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--splits_csv', type=str, required=True,
                        help='Path to splits CSV from Stage8 (e.g., splits_mip_75_10_15.csv)')
    parser.add_argument('--dataset_root', type=str,
                        default=str(MYGARDENBIRD_16K) if CONFIG_AVAILABLE else None,
                        required=not CONFIG_AVAILABLE,
                        help=f'Root directory with species subdirectories (default: {MYGARDENBIRD_16K if CONFIG_AVAILABLE else "required"})')

    # BirdNET parameters
    parser.add_argument('--min_conf', type=float, default=0.1,
                        help='Minimum confidence threshold for BirdNET (default: 0.1)')
    parser.add_argument('--lat', type=float, default=3.139,
                        help='Latitude for location-based filtering (default: 3.139 - Kuala Lumpur)')
    parser.add_argument('--lon', type=float, default=101.687,
                        help='Longitude for location-based filtering (default: 101.687 - Kuala Lumpur)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./birdnet_benchmark',
                        help='Output directory for results (default: ./birdnet_benchmark)')

    return parser.parse_args()


def main():
    """Main benchmark function."""
    # Parse arguments
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("BirdNET Benchmark on MyGardenBird")
    print(f"{'=' * 70}")
    print(f"Splits CSV:     {args.splits_csv}")
    print(f"Dataset Root:   {args.dataset_root}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"{'=' * 70}\n")

    # Load test set
    start_time = time.time()
    print("Loading test set...")
    test_data = load_test_set(args.splits_csv, args.dataset_root)

    # Create species mapping
    species_mapping = create_species_mapping()

    # Run BirdNET evaluation
    predictions, ground_truths, stats = evaluate_birdnet(
        test_data,
        species_mapping,
        min_conf=args.min_conf,
        lat=args.lat,
        lon=args.lon
    )

    # Generate report
    accuracy = generate_report(predictions, ground_truths, stats, args, output_dir)

    # Print timing
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.1f}s ({elapsed_time/len(test_data):.2f}s per file)")


if __name__ == '__main__':
    main()
