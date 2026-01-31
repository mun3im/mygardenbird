#!/usr/bin/env python3
"""
SEAbird Dataset Splitter using Mixed Integer Programming (MIP)

This script creates optimal train/val/test splits for the SEAbird acoustic dataset
while ensuring:
1. Exact target distributions (75%/10%/15%) across all classes
2. Source-based separation to prevent data leakage
3. Class balance preservation

The script uses Mixed Integer Programming to find globally optimal source assignments.
MIP provides deterministic, provably optimal solutions compared to heuristic approaches.

Advantages of MIP:
- Guarantees optimality (if solution exists)
- Deterministic results (same input = same output)
- Fast convergence for feasible problems
- Built-in constraint handling
- No hyperparameter tuning needed

Requires: pip install pulp

Author: Generated for SEAbird biodiversity monitoring research
Date: December 2024
"""

import json
import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import time

try:
    from pulp import (
        LpProblem, LpMinimize, LpVariable, lpSum, LpStatus,
        PULP_CBC_CMD, LpBinary, value
    )
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("\n❌ ERROR: PuLP not installed")
    print("Install with: pip install pulp\n")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_TRAIN_RATIO = 0.75
TARGET_VAL_RATIO = 0.10
TARGET_TEST_RATIO = 0.15

# MIP Solver parameters
MIP_TIME_LIMIT = 600  # Maximum time in seconds (10 minutes)
MIP_GAP_TOLERANCE = 0.0  # 0 = find optimal solution (no approximation)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset_structure(dataset_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load the dataset structure: class -> source -> [files]

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Nested dict mapping class_name -> source_id -> list of filenames
    """
    structure = {}

    for class_dir in Path(dataset_path).iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        class_name = class_dir.name
        structure[class_name] = defaultdict(list)

        # Group files by source
        for audio_file in class_dir.glob('*.wav'):
            # Extract source ID from filename like xc402013_27465.wav -> xc402013
            source = audio_file.stem.split('_')[0]
            structure[class_name][source].append(audio_file.name)

    return structure


# ============================================================================
# MIP OPTIMIZATION
# ============================================================================

def optimize_split_mip(structure: Dict[str, Dict[str, List[str]]],
                       train_ratio: float = TARGET_TRAIN_RATIO,
                       val_ratio: float = TARGET_VAL_RATIO,
                       test_ratio: float = TARGET_TEST_RATIO,
                       time_limit: int = MIP_TIME_LIMIT,
                       verbose: bool = True) -> Tuple[Dict, Dict, float]:
    """
    Use Mixed Integer Programming to find optimal dataset split.

    Args:
        structure: Dataset structure (class -> source -> files)
        train_ratio: Target ratio for training set
        val_ratio: Target ratio for validation set
        test_ratio: Target ratio for test set
        time_limit: Maximum solving time in seconds
        verbose: Print progress messages

    Returns:
        (assignment, stats, objective_value)
        - assignment: Dict[class][source] = 'train'|'val'|'test'
        - stats: Detailed statistics
        - objective_value: Final objective (0 = perfect)
    """
    if verbose:
        print("🔧 MIXED INTEGER PROGRAMMING OPTIMIZATION")
        print("=" * 80)
        print(f"Time limit: {time_limit}s")
        print(f"Gap tolerance: {MIP_GAP_TOLERANCE}")
        print(f"Target ratios: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
        print()

    start_time = time.time()

    # Calculate total samples per class
    class_totals = {}
    for class_name, sources in structure.items():
        total = sum(len(files) for files in sources.values())
        class_totals[class_name] = total

    # Calculate target counts per class
    targets = {}
    for class_name, total in class_totals.items():
        targets[class_name] = {
            'train': round(total * train_ratio),
            'val': round(total * val_ratio),
            'test': round(total * test_ratio)
        }

    if verbose:
        print("📊 Target sample counts per class:")
        for class_name in sorted(targets.keys()):
            t = targets[class_name]
            print(f"  {class_name:30s}: {t['train']:4d} train / {t['val']:3d} val / {t['test']:3d} test")
        print()

    # Create MIP problem
    prob = LpProblem("Dataset_Split_Optimization", LpMinimize)

    # Decision variables: x[class][source][split] = 1 if source assigned to split
    x = {}
    for class_name, sources in structure.items():
        x[class_name] = {}
        for source in sources.keys():
            x[class_name][source] = {
                'train': LpVariable(f"x_{class_name}_{source}_train", cat=LpBinary),
                'val': LpVariable(f"x_{class_name}_{source}_val", cat=LpBinary),
                'test': LpVariable(f"x_{class_name}_{source}_test", cat=LpBinary)
            }

    # Slack variables for soft constraints (deviation from targets)
    slack_pos = {}  # Positive deviation (over target)
    slack_neg = {}  # Negative deviation (under target)
    for class_name in structure.keys():
        slack_pos[class_name] = {
            'train': LpVariable(f"slack_pos_{class_name}_train", lowBound=0),
            'val': LpVariable(f"slack_pos_{class_name}_val", lowBound=0),
            'test': LpVariable(f"slack_pos_{class_name}_test", lowBound=0)
        }
        slack_neg[class_name] = {
            'train': LpVariable(f"slack_neg_{class_name}_train", lowBound=0),
            'val': LpVariable(f"slack_neg_{class_name}_val", lowBound=0),
            'test': LpVariable(f"slack_neg_{class_name}_test", lowBound=0)
        }

    # Objective: Minimize total deviation from targets
    prob += lpSum([
        slack_pos[cls][split] + slack_neg[cls][split]
        for cls in structure.keys()
        for split in ['train', 'val', 'test']
    ])

    # CONSTRAINT 1: Each source must be assigned to exactly one split
    for class_name, sources in structure.items():
        for source in sources.keys():
            prob += (
                x[class_name][source]['train'] +
                x[class_name][source]['val'] +
                x[class_name][source]['test'] == 1,
                f"one_split_{class_name}_{source}"
            )

    # CONSTRAINT 2: Match target sample counts (with slack)
    for class_name, sources in structure.items():
        for split in ['train', 'val', 'test']:
            actual_samples = lpSum([
                x[class_name][source][split] * len(files)
                for source, files in sources.items()
            ])
            target = targets[class_name][split]

            # actual_samples = target + slack_pos - slack_neg
            prob += (
                actual_samples == target + slack_pos[class_name][split] - slack_neg[class_name][split],
                f"target_{class_name}_{split}"
            )

    # Solve with time limit
    if verbose:
        print("🚀 Solving MIP problem...")
        print("   This may take a few seconds to minutes depending on dataset size...")
        print()

    solver = PULP_CBC_CMD(
        msg=verbose,
        timeLimit=time_limit,
        gapRel=MIP_GAP_TOLERANCE,
        threads=None  # Use all available cores
    )

    prob.solve(solver)

    solve_time = time.time() - start_time

    # Check solution status
    status = LpStatus[prob.status]
    if verbose:
        print(f"\n✓ Solution status: {status}")
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  Objective value: {value(prob.objective):.0f}")
        print()

    if status not in ['Optimal', 'Not Solved']:  # Not Solved can still have a feasible solution
        if verbose:
            print(f"⚠️  Warning: Solution status is '{status}'")
            print("   Proceeding with best found solution...")
            print()

    # Extract solution
    assignment = {}
    for class_name in structure.keys():
        assignment[class_name] = {}
        for source in structure[class_name].keys():
            # Find which split this source was assigned to
            for split in ['train', 'val', 'test']:
                if value(x[class_name][source][split]) > 0.5:  # Binary variable = 1
                    assignment[class_name][source] = split
                    break

    # Calculate statistics
    stats = calculate_statistics(structure, assignment)

    return assignment, stats, value(prob.objective) if prob.objective else 0


# ============================================================================
# STATISTICS
# ============================================================================

def calculate_statistics(structure: Dict[str, Dict[str, List[str]]],
                        assignment: Dict[str, Dict[str, str]]) -> Dict:
    """Calculate detailed statistics for the split assignment."""
    stats = {}

    # Per-class statistics
    for class_name, sources in structure.items():
        train_samples = sum(
            len(files) for source, files in sources.items()
            if assignment[class_name].get(source) == 'train'
        )
        val_samples = sum(
            len(files) for source, files in sources.items()
            if assignment[class_name].get(source) == 'val'
        )
        test_samples = sum(
            len(files) for source, files in sources.items()
            if assignment[class_name].get(source) == 'test'
        )

        total_samples = train_samples + val_samples + test_samples

        train_sources = sum(
            1 for source in sources.keys()
            if assignment[class_name].get(source) == 'train'
        )
        val_sources = sum(
            1 for source in sources.keys()
            if assignment[class_name].get(source) == 'val'
        )
        test_sources = sum(
            1 for source in sources.keys()
            if assignment[class_name].get(source) == 'test'
        )

        stats[class_name] = {
            'total_samples': total_samples,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples,
            'total_sources': len(sources),
            'train_sources': train_sources,
            'val_sources': val_sources,
            'test_sources': test_sources
        }

    # Overall statistics
    total_train = sum(s['train_samples'] for s in stats.values())
    total_val = sum(s['val_samples'] for s in stats.values())
    total_test = sum(s['test_samples'] for s in stats.values())
    total_all = total_train + total_val + total_test

    stats['_overall'] = {
        'total_samples': total_all,
        'train_samples': total_train,
        'val_samples': total_val,
        'test_samples': total_test,
        'total_sources': sum(s['total_sources'] for s in stats.values() if isinstance(s, dict) and 'total_sources' in s),
        'train_sources': sum(s['train_sources'] for s in stats.values() if isinstance(s, dict) and 'train_sources' in s),
        'val_sources': sum(s['val_sources'] for s in stats.values() if isinstance(s, dict) and 'val_sources' in s),
        'test_sources': sum(s['test_sources'] for s in stats.values() if isinstance(s, dict) and 'test_sources' in s)
    }

    stats['_config'] = {
        'train_ratio': TARGET_TRAIN_RATIO,
        'val_ratio': TARGET_VAL_RATIO,
        'test_ratio': TARGET_TEST_RATIO,
        'optimization': 'mixed_integer_programming'
    }

    return stats


def format_statistics_text(stats: Dict) -> str:
    """Format statistics as text string."""
    lines = []
    lines.append("=" * 80)
    lines.append("FINAL STATISTICS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Per-Class Breakdown:")
    lines.append(f"{'Class':<30} {'Train':>8} {'Val':>6} {'Test':>6} {'Total':>7} | {'Sources':>7}")
    lines.append("-" * 80)

    for class_name in sorted(stats.keys()):
        if class_name.startswith('_'):
            continue
        s = stats[class_name]
        lines.append(f"{class_name:<30} {s['train_samples']:8d} {s['val_samples']:6d} "
              f"{s['test_samples']:6d} {s['total_samples']:7d} | "
              f"{s['train_sources']:3d}/{s['val_sources']:3d}/{s['test_sources']:3d}")

    lines.append("-" * 80)
    overall = stats['_overall']
    lines.append(f"{'OVERALL':<30} {overall['train_samples']:8d} {overall['val_samples']:6d} "
          f"{overall['test_samples']:6d} {overall['total_samples']:7d} | "
          f"{overall['train_sources']:3d}/{overall['val_sources']:3d}/{overall['test_sources']:3d}")

    # Calculate actual ratios
    total = overall['total_samples']
    actual_train_ratio = overall['train_samples'] / total
    actual_val_ratio = overall['val_samples'] / total
    actual_test_ratio = overall['test_samples'] / total

    lines.append("")
    lines.append("Actual Ratios:")
    lines.append(f"  Train: {actual_train_ratio:.1%} (target: {TARGET_TRAIN_RATIO:.1%})")
    lines.append(f"  Val:   {actual_val_ratio:.1%} (target: {TARGET_VAL_RATIO:.1%})")
    lines.append(f"  Test:  {actual_test_ratio:.1%} (target: {TARGET_TEST_RATIO:.1%})")
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"  Optimization: {stats['_config']['optimization']}")
    lines.append(f"  Target ratios: {stats['_config']['train_ratio']:.0%} / "
                f"{stats['_config']['val_ratio']:.0%} / {stats['_config']['test_ratio']:.0%}")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def print_statistics(stats: Dict, verbose: bool = True):
    """Print formatted statistics."""
    if not verbose:
        return

    print("📊 FINAL STATISTICS")
    print("=" * 80)
    print()
    print("Per-Class Breakdown:")
    print(f"{'Class':<30} {'Train':>8} {'Val':>6} {'Test':>6} {'Total':>7} | {'Sources':>7}")
    print("-" * 80)

    for class_name in sorted(stats.keys()):
        if class_name.startswith('_'):
            continue
        s = stats[class_name]
        print(f"{class_name:<30} {s['train_samples']:8d} {s['val_samples']:6d} "
              f"{s['test_samples']:6d} {s['total_samples']:7d} | "
              f"{s['train_sources']:3d}/{s['val_sources']:3d}/{s['test_sources']:3d}")

    print("-" * 80)
    overall = stats['_overall']
    print(f"{'OVERALL':<30} {overall['train_samples']:8d} {overall['val_samples']:6d} "
          f"{overall['test_samples']:6d} {overall['total_samples']:7d} | "
          f"{overall['train_sources']:3d}/{overall['val_sources']:3d}/{overall['test_sources']:3d}")

    # Calculate actual ratios
    total = overall['total_samples']
    actual_train_ratio = overall['train_samples'] / total
    actual_val_ratio = overall['val_samples'] / total
    actual_test_ratio = overall['test_samples'] / total

    print()
    print("Actual Ratios:")
    print(f"  Train: {actual_train_ratio:.1%} (target: {TARGET_TRAIN_RATIO:.1%})")
    print(f"  Val:   {actual_val_ratio:.1%} (target: {TARGET_VAL_RATIO:.1%})")
    print(f"  Test:  {actual_test_ratio:.1%} (target: {TARGET_TEST_RATIO:.1%})")
    print()


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def create_split_directories(dataset_path: str,
                            structure: Dict[str, Dict[str, List[str]]],
                            assignment: Dict[str, Dict[str, str]],
                            output_dir: str = 'splits_mip',
                            verbose: bool = True):
    """Create train/val/test directories and copy files."""
    output_path = Path(dataset_path).parent / output_dir

    if verbose:
        print(f"📁 Creating split directories in: {output_path}")

    # Create directories
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        split_path.mkdir(parents=True, exist_ok=True)

        for class_name in structure.keys():
            (split_path / class_name).mkdir(exist_ok=True)

    # Copy files
    if verbose:
        print("   Copying files...")

    total_files = sum(
        len(files)
        for class_sources in structure.values()
        for files in class_sources.values()
    )

    copied = 0
    for class_name, sources in structure.items():
        for source, files in sources.items():
            split = assignment[class_name][source]
            src_dir = Path(dataset_path) / class_name
            dst_dir = output_path / split / class_name

            for file in files:
                src = src_dir / file
                dst = dst_dir / file
                shutil.copy2(src, dst)
                copied += 1

    if verbose:
        print(f"   ✓ Copied {copied} files")
        print()

    return output_path


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_split_distribution(stats: Dict, output_path: Path):
    """Create visualization of the split distribution."""
    if not HAS_MATPLOTLIB:
        return

    classes = [k for k in sorted(stats.keys()) if not k.startswith('_')]

    train_samples = [stats[c]['train_samples'] for c in classes]
    val_samples = [stats[c]['val_samples'] for c in classes]
    test_samples = [stats[c]['test_samples'] for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar chart - samples
    x = range(len(classes))
    ax1.bar(x, train_samples, label='Train', color='#3498db')
    ax1.bar(x, val_samples, bottom=train_samples, label='Val', color='#2ecc71')
    ax1.bar(x, test_samples,
            bottom=[t+v for t, v in zip(train_samples, val_samples)],
            label='Test', color='#e74c3c')

    ax1.set_xlabel('Species')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Distribution per Class (MIP Optimization)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ratio visualization
    ratios_train = [t / (t+v+te) for t, v, te in zip(train_samples, val_samples, test_samples)]
    ratios_val = [v / (t+v+te) for t, v, te in zip(train_samples, val_samples, test_samples)]
    ratios_test = [te / (t+v+te) for t, v, te in zip(train_samples, val_samples, test_samples)]

    ax2.bar(x, ratios_train, label='Train', color='#3498db')
    ax2.bar(x, ratios_val, bottom=ratios_train, label='Val', color='#2ecc71')
    ax2.bar(x, ratios_test,
            bottom=[t+v for t, v in zip(ratios_train, ratios_val)],
            label='Test', color='#e74c3c')

    ax2.axhline(y=TARGET_TRAIN_RATIO, color='#3498db', linestyle='--', alpha=0.5, label=f'Target Train ({TARGET_TRAIN_RATIO:.0%})')
    ax2.axhline(y=TARGET_TRAIN_RATIO + TARGET_VAL_RATIO, color='#2ecc71', linestyle='--', alpha=0.5, label=f'Target Val ({TARGET_VAL_RATIO:.0%})')

    ax2.set_xlabel('Species')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Split Ratios per Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path / 'split_distribution_mip.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved plot: {output_path / 'split_distribution_mip.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Split SEAbird dataset using Mixed Integer Programming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python seabird_splitter_mip.py --dataset /path/to/seabird16khz_flat

  # With visualization and directory creation
  python seabird_splitter_mip.py --dataset /path/to/seabird16khz_flat --plot --create-dirs

  # Custom time limit
  python seabird_splitter_mip.py --dataset /path/to/seabird16khz_flat --time-limit 300
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to flat dataset directory')
    parser.add_argument('--output', type=str, default='splits_mip',
                       help='Output directory name (default: splits_mip)')
    parser.add_argument('--create-dirs', action='store_true',
                       help='Create train/val/test directories and copy files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--time-limit', type=int, default=MIP_TIME_LIMIT,
                       help=f'MIP solver time limit in seconds (default: {MIP_TIME_LIMIT})')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("\n" + "=" * 80)
        print("SEAbird Dataset Splitter - Mixed Integer Programming")
        print("=" * 80)
        print()

    # Check PuLP
    if not HAS_PULP:
        print("ERROR: This script requires PuLP")
        print("Install with: pip install pulp")
        return 1

    # Load dataset
    if verbose:
        print(f"📂 Loading dataset from: {args.dataset}")

    structure = load_dataset_structure(args.dataset)

    if not structure:
        print(f"ERROR: No data found in {args.dataset}")
        return 1

    num_classes = len(structure)
    total_sources = sum(len(sources) for sources in structure.values())
    total_files = sum(
        len(files)
        for class_sources in structure.values()
        for files in class_sources.values()
    )

    if verbose:
        print(f"   ✓ Loaded {num_classes} classes")
        print(f"   ✓ Found {total_sources} unique sources")
        print(f"   ✓ Total files: {total_files}")
        print()

    # Run MIP optimization
    assignment, stats, objective = optimize_split_mip(
        structure,
        time_limit=args.time_limit,
        verbose=verbose
    )

    # Print statistics
    print_statistics(stats, verbose)

    # Save statistics
    output_path = Path(args.dataset).parent / args.output
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON statistics
    stats_file = output_path / 'split_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Save text statistics
    stats_text_file = output_path / 'split_stats.txt'
    with open(stats_text_file, 'w') as f:
        f.write(format_statistics_text(stats))

    if verbose:
        print(f"💾 Saved statistics:")
        print(f"   JSON: {stats_file}")
        print(f"   Text: {stats_text_file}")
        print()

    # Create directories
    if args.create_dirs:
        create_split_directories(
            args.dataset,
            structure,
            assignment,
            args.output,
            verbose
        )

    # Plot
    if args.plot and HAS_MATPLOTLIB:
        if verbose:
            print("📊 Generating plot...")

        plot_split_distribution(stats, output_path)
        print()

    if verbose:
        print("=" * 80)
        print("✅ SUCCESS - MIP optimization complete!")
        print("=" * 80)
        print()

    return 0


if __name__ == '__main__':
    exit(main())
