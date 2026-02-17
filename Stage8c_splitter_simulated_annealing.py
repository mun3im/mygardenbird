#!/usr/bin/env python3
"""
SEAbird Dataset Splitter using Simulated Annealing Optimization

This script creates optimal train/val/test splits for the SEAbird acoustic dataset
while ensuring:
1. Exact target distributions (75%/10%/15%) across all classes
2. Source-based separation to prevent data leakage
3. Class balance preservation

The script uses simulated annealing optimization to find the best source assignments
that satisfy all constraints.

Author: Generated for SEAbird biodiversity monitoring research
Date: December 2024
"""

import json
import os
import random
import math
import copy
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from config import EXTRACTED_SEGS, SPLITS_DIR

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_TRAIN_RATIO = 0.75
TARGET_VAL_RATIO = 0.10
TARGET_TEST_RATIO = 0.15

# Simulated Annealing parameters
INITIAL_TEMPERATURE = 100.0
COOLING_RATE = 0.9995
MIN_TEMPERATURE = 0.01
ITERATIONS_PER_TEMP = 100


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
# SCORING AND STATISTICS
# ============================================================================

def calculate_split_score(structure: Dict[str, Dict[str, List[str]]],
                          assignment: Dict[str, Dict[str, str]]) -> float:
    """
    Calculate how close a split assignment is to the target ratios.

    Args:
        structure: Dataset structure
        assignment: Current source assignments to splits

    Returns:
        Total error (lower is better, 0 is perfect)
    """
    total_error = 0.0

    for class_name, sources in structure.items():
        total_samples = sum(len(files) for files in sources.values())

        # Target counts for this class
        target_train = int(total_samples * TARGET_TRAIN_RATIO)
        target_val = int(total_samples * TARGET_VAL_RATIO)
        target_test = int(total_samples * TARGET_TEST_RATIO)

        # Actual counts based on assignment
        train_count = sum(len(sources[src]) for src in assignment[class_name]
                         if assignment[class_name][src] == 'train')
        val_count = sum(len(sources[src]) for src in assignment[class_name]
                       if assignment[class_name][src] == 'val')
        test_count = sum(len(sources[src]) for src in assignment[class_name]
                        if assignment[class_name][src] == 'test')

        # Calculate error (absolute difference from target)
        error = (abs(train_count - target_train) +
                abs(val_count - target_val) +
                abs(test_count - target_test))

        total_error += error

    return total_error


def get_detailed_stats(structure: Dict[str, Dict[str, List[str]]],
                      assignment: Dict[str, Dict[str, str]]) -> Dict:
    """Get detailed statistics for the current assignment."""
    stats = {}

    for class_name, sources in structure.items():
        total_samples = sum(len(files) for files in sources.values())

        # Target counts
        target_train = int(total_samples * TARGET_TRAIN_RATIO)
        target_val = int(total_samples * TARGET_VAL_RATIO)
        target_test = int(total_samples * TARGET_TEST_RATIO)

        # Actual counts
        train_count = sum(len(sources[src]) for src in assignment[class_name]
                         if assignment[class_name][src] == 'train')
        val_count = sum(len(sources[src]) for src in assignment[class_name]
                       if assignment[class_name][src] == 'val')
        test_count = sum(len(sources[src]) for src in assignment[class_name]
                        if assignment[class_name][src] == 'test')

        error = (abs(train_count - target_train) +
                abs(val_count - target_val) +
                abs(test_count - target_test))

        stats[class_name] = {
            'total': total_samples,
            'target': {'train': target_train, 'val': target_val, 'test': target_test},
            'actual': {'train': train_count, 'val': val_count, 'test': test_count},
            'error': error
        }

    return stats


# ============================================================================
# SIMULATED ANNEALING OPTIMIZATION
# ============================================================================

def generate_initial_solution(structure: Dict[str, Dict[str, List[str]]],
                              seed: int = None) -> Dict[str, Dict[str, str]]:
    """Generate initial solution using greedy approach."""
    if seed is not None:
        random.seed(seed)

    assignment = {}

    for class_name, sources in structure.items():
        assignment[class_name] = {}

        # Get source sizes and shuffle for randomness
        source_sizes = [(src, len(files)) for src, files in sources.items()]
        random.shuffle(source_sizes)
        source_sizes.sort(key=lambda x: x[1])  # Sort by size

        total_samples = sum(size for _, size in source_sizes)
        target_train = int(total_samples * TARGET_TRAIN_RATIO)
        target_val = int(total_samples * TARGET_VAL_RATIO)
        target_test = int(total_samples * TARGET_TEST_RATIO)

        train_count = 0
        val_count = 0
        test_count = 0

        # Greedy assignment
        for src, size in source_sizes:
            train_dist = abs((train_count + size) - target_train)
            val_dist = abs((val_count + size) - target_val)
            test_dist = abs((test_count + size) - target_test)

            train_over = (train_count >= target_train)
            val_over = (val_count >= target_val)
            test_over = (test_count >= target_test)

            options = []
            if not train_over:
                options.append(('train', train_dist))
            if not val_over:
                options.append(('val', val_dist))
            if not test_over:
                options.append(('test', test_dist))

            if not options:
                options = [('train', train_dist), ('val', val_dist), ('test', test_dist)]

            chosen_split = min(options, key=lambda x: x[1])[0]

            assignment[class_name][src] = chosen_split
            if chosen_split == 'train':
                train_count += size
            elif chosen_split == 'val':
                val_count += size
            else:
                test_count += size

    return assignment


def generate_neighbor(structure: Dict[str, Dict[str, List[str]]],
                     assignment: Dict[str, Dict[str, str]],
                     temperature: float) -> Dict[str, Dict[str, str]]:
    """
    Generate a neighboring solution by making a small random change.
    Higher temperature = more aggressive changes.
    """
    new_assignment = copy.deepcopy(assignment)

    # Choose number of changes based on temperature
    num_changes = 1 if temperature < 10 else (2 if random.random() < 0.3 else 1)

    for _ in range(num_changes):
        class_name = random.choice(list(structure.keys()))
        sources = list(structure[class_name].keys())

        # Choose a move type
        move_type = random.choice(['swap_sources', 'move_source', 'swap_between_classes'])

        if move_type == 'swap_sources':
            # Swap two sources between different splits
            if len(sources) >= 2:
                src1, src2 = random.sample(sources, 2)
                split1 = new_assignment[class_name][src1]
                split2 = new_assignment[class_name][src2]

                if split1 != split2:
                    new_assignment[class_name][src1] = split2
                    new_assignment[class_name][src2] = split1

        elif move_type == 'move_source':
            # Move a single source to a different split
            src = random.choice(sources)
            current_split = new_assignment[class_name][src]
            other_splits = [s for s in ['train', 'val', 'test'] if s != current_split]
            new_split = random.choice(other_splits)
            new_assignment[class_name][src] = new_split

        elif move_type == 'swap_between_classes':
            # Swap sources between two classes
            if len(structure) >= 2:
                class2 = random.choice([c for c in structure.keys() if c != class_name])
                sources2 = list(structure[class2].keys())

                if sources and sources2:
                    src1 = random.choice(sources)
                    src2 = random.choice(sources2)

                    split1 = new_assignment[class_name][src1]
                    split2 = new_assignment[class2][src2]

                    new_assignment[class_name][src1] = split2
                    new_assignment[class2][src2] = split1

    return new_assignment


def acceptance_probability(current_score: float, new_score: float, temperature: float) -> float:
    """
    Calculate probability of accepting a worse solution.
    Uses the Metropolis criterion from statistical mechanics.
    """
    if new_score < current_score:
        return 1.0
    else:
        delta = new_score - current_score
        return math.exp(-delta / temperature)


def simulated_annealing(structure: Dict[str, Dict[str, List[str]]],
                       initial_temp: float = INITIAL_TEMPERATURE,
                       cooling_rate: float = COOLING_RATE,
                       min_temp: float = MIN_TEMPERATURE,
                       iterations_per_temp: int = ITERATIONS_PER_TEMP,
                       seed: int = None,
                       verbose: bool = True) -> Tuple[Dict, Dict, float, List]:
    """
    Optimize splits using Simulated Annealing.

    Returns:
        (best_assignment, best_stats, best_score, history)
    """
    if verbose:
        print("🔥 SIMULATED ANNEALING OPTIMIZATION")
        print("="*80)
        print(f"Initial temperature: {initial_temp}")
        print(f"Cooling rate: {cooling_rate}")
        print(f"Minimum temperature: {min_temp}")
        print(f"Iterations per temperature: {iterations_per_temp}")
        print()

    # Generate initial solution
    if verbose:
        print("Generating initial solution...")
    current_assignment = generate_initial_solution(structure, seed=seed)
    current_score = calculate_split_score(structure, current_assignment)

    # Track best solution
    best_assignment = copy.deepcopy(current_assignment)
    best_score = current_score
    best_stats = get_detailed_stats(structure, best_assignment)

    # Calculate estimated maximum iterations
    estimated_temps = int(math.log(min_temp / initial_temp) / math.log(cooling_rate)) + 1
    max_iterations = estimated_temps * iterations_per_temp

    # Temperature and iteration tracking
    temperature = initial_temp
    total_iterations = 0
    accepted_moves = 0
    rejected_moves = 0
    history = []

    if verbose:
        print(f"Initial solution score: {current_score:.0f}")
        print(f"Estimated max iterations: ~{max_iterations:,}")
        print()

    # Main annealing loop with progress bar
    pbar = tqdm(total=max_iterations,
                desc="Optimizing",
                disable=not verbose,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Best: {postfix}')

    pbar.set_postfix_str(f"Score={best_score:.0f}, Temp={temperature:.2f}, Accept={0:.1f}%")

    try:
        while temperature > min_temp:
            temp_accepted = 0

            for i in range(iterations_per_temp):
                total_iterations += 1

                # Generate neighbor
                neighbor_assignment = generate_neighbor(structure, current_assignment, temperature)
                neighbor_score = calculate_split_score(structure, neighbor_assignment)

                # Decide whether to accept
                prob = acceptance_probability(current_score, neighbor_score, temperature)

                if random.random() < prob:
                    current_assignment = neighbor_assignment
                    current_score = neighbor_score
                    accepted_moves += 1
                    temp_accepted += 1

                    # Check if this is the best solution so far
                    if current_score < best_score:
                        best_assignment = copy.deepcopy(current_assignment)
                        best_score = current_score
                        best_stats = get_detailed_stats(structure, best_assignment)
                else:
                    rejected_moves += 1

                # Record history
                history.append({
                    'iteration': total_iterations,
                    'temperature': temperature,
                    'current_score': current_score,
                    'best_score': best_score
                })

                # Update progress bar
                accept_rate = (accepted_moves / total_iterations) * 100
                pbar.set_postfix_str(f"Score={best_score:.0f}, Temp={temperature:.2f}, Accept={accept_rate:.1f}%")
                pbar.update(1)

                # If we found a perfect solution, stop
                if best_score == 0:
                    pbar.set_postfix_str(f"🎉 PERFECT! Score={best_score:.0f}")
                    break

            # Cool down
            temperature *= cooling_rate

            if best_score == 0:
                break

    finally:
        pbar.close()

    if verbose:
        print()
        if best_score == 0:
            print("🎉 PERFECT SOLUTION FOUND!")
        print("="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Total iterations: {total_iterations:,}")
        print(f"Moves accepted: {accepted_moves:,} ({(accepted_moves/total_iterations)*100:.1f}%)")
        print(f"Moves rejected: {rejected_moves:,} ({(rejected_moves/total_iterations)*100:.1f}%)")
        print(f"Final temperature: {temperature:.4f}")
        print(f"Best score achieved: {best_score:.0f}")
        print()

    return best_assignment, best_stats, best_score, history


# ============================================================================
# SAVING SPLITS
# ============================================================================

def create_splits_csv(structure: Dict[str, Dict[str, List[str]]],
                      assignment: Dict[str, Dict[str, str]],
                      output_path: Path,
                      train_ratio: float = 0.75,
                      val_ratio: float = 0.10,
                      test_ratio: float = 0.15,
                      objective: float = 0,
                      seed: int = 42,
                      verbose: bool = True):
    """Write a CSV with columns: file_id,split.

    file_id is the normalised clip primary key: XC{source_id}_{clip_index}
    (matches the file_id column in clips.csv produced by Stage 7).
    wav_filename is derivable as: xc{source_id}_{clip_index}.wav
    """
    if verbose:
        print(f"📄 Writing splits CSV: {output_path}")

    train_pct = int(round(train_ratio * 100))
    val_pct = int(round(val_ratio * 100))
    test_pct = int(round(test_ratio * 100))

    rows = []
    for class_name in sorted(structure.keys()):
        sources = structure[class_name]
        for source in sorted(sources.keys()):
            split = assignment[class_name][source]
            for filename in sorted(sources[source]):
                stem = Path(filename).stem          # e.g. xc1002657_2860
                file_id = "XC" + stem[2:] if stem.lower().startswith("xc") else stem
                rows.append((file_id, split))

    with open(output_path, 'w') as f:
        f.write(f"# split_ratio={train_pct}:{val_pct}:{test_pct} seed={seed} objective={objective:.0f} solver=simulated_annealing\n")
        f.write("file_id,split\n")
        for file_id, split in rows:
            f.write(f"{file_id},{split}\n")

    if verbose:
        counts = {'train': 0, 'val': 0, 'test': 0}
        for _, split in rows:
            counts[split] += 1
        print(f"   ✓ Wrote {len(rows)} rows (train={counts['train']}, val={counts['val']}, test={counts['test']})")
        print()

    return output_path


def save_splits(structure: Dict[str, Dict[str, List[str]]],
               assignment: Dict[str, Dict[str, str]],
               output_dir: str = './splits') -> Dict:
    """Save the optimized splits to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create split files
    splits = {'train': [], 'val': [], 'test': []}
    split_stats = {}

    for class_name, sources in structure.items():
        class_stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'total_sources': len(sources),
            'train_sources': 0,
            'val_sources': 0,
            'test_sources': 0
        }

        for src, files in sources.items():
            split = assignment[class_name][src]
            class_stats[f'{split}_samples'] += len(files)
            class_stats[f'{split}_sources'] += 1
            class_stats['total_samples'] += len(files)

            for file in files:
                splits[split].append(f"{class_name}/{file}")

        split_stats[class_name] = class_stats

    # Write split files
    for split_name, files in splits.items():
        files.sort()
        with open(output_path / f'{split_name}.txt', 'w') as f:
            f.write('\n'.join(files) + '\n')
        print(f"Wrote {len(files)} files to {split_name}.txt")

    # Calculate overall stats
    overall = {
        'total_samples': sum(s['total_samples'] for s in split_stats.values()),
        'train_samples': sum(s['train_samples'] for s in split_stats.values()),
        'val_samples': sum(s['val_samples'] for s in split_stats.values()),
        'test_samples': sum(s['test_samples'] for s in split_stats.values()),
        'total_sources': sum(s['total_sources'] for s in split_stats.values()),
        'train_sources': sum(s['train_sources'] for s in split_stats.values()),
        'val_sources': sum(s['val_sources'] for s in split_stats.values()),
        'test_sources': sum(s['test_sources'] for s in split_stats.values())
    }

    split_stats['_overall'] = overall
    split_stats['_config'] = {
        'train_ratio': TARGET_TRAIN_RATIO,
        'val_ratio': TARGET_VAL_RATIO,
        'test_ratio': TARGET_TEST_RATIO,
        'optimization': 'simulated_annealing'
    }

    # Write stats
    with open(output_path / 'split_stats.json', 'w') as f:
        json.dump(split_stats, f, indent=2)
    print(f"Wrote statistics to split_stats.json")

    return split_stats


# ============================================================================
# VERIFICATION
# ============================================================================

def extract_source_id(filename: str) -> str:
    """Extract source ID from path like Asian Koel/xc402013_27465.wav -> xc402013"""
    try:
        fname = Path(filename).name
        return fname.split('_')[0]
    except:
        return None


def verify_no_leakage(splits_dir: str, verbose: bool = True) -> bool:
    """Verify no source leakage between splits."""
    splits_dir = Path(splits_dir)

    # Load splits
    def load_split(split_file):
        with open(split_file, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        sources = defaultdict(list)
        for path in paths:
            source_id = extract_source_id(path)
            if source_id:
                sources[source_id].append(path)
        return paths, sources

    train_paths, train_sources = load_split(splits_dir / "train.txt")
    val_paths, val_sources = load_split(splits_dir / "val.txt")
    test_paths, test_sources = load_split(splits_dir / "test.txt")

    if verbose:
        print("\n" + "="*80)
        print("VERIFYING DATA LEAKAGE")
        print("="*80)
        print(f"Train: {len(train_paths)} samples, {len(train_sources)} sources")
        print(f"Val:   {len(val_paths)} samples, {len(val_sources)} sources")
        print(f"Test:  {len(test_paths)} samples, {len(test_sources)} sources")

    # Check for source overlap
    train_source_set = set(train_sources.keys())
    val_source_set = set(val_sources.keys())
    test_source_set = set(test_sources.keys())

    train_val_overlap = train_source_set & val_source_set
    train_test_overlap = train_source_set & test_source_set
    val_test_overlap = val_source_set & test_source_set

    # Check file-level duplicates
    train_path_set = set(train_paths)
    val_path_set = set(val_paths)
    test_path_set = set(test_paths)

    train_val_dup = train_path_set & val_path_set
    train_test_dup = train_path_set & test_path_set
    val_test_dup = val_path_set & test_path_set

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap or
                       train_val_dup or train_test_dup or val_test_dup)

    if verbose:
        if train_val_overlap:
            print(f"\n❌ LEAKAGE: {len(train_val_overlap)} sources overlap between train and val")
        else:
            print(f"\n✓ No source overlap between train and val")

        if train_test_overlap:
            print(f"❌ LEAKAGE: {len(train_test_overlap)} sources overlap between train and test")
        else:
            print(f"✓ No source overlap between train and test")

        if val_test_overlap:
            print(f"❌ LEAKAGE: {len(val_test_overlap)} sources overlap between val and test")
        else:
            print(f"✓ No source overlap between val and test")

        print("\n" + "="*80)
        if has_leakage:
            print("❌ LEAKAGE DETECTED")
        else:
            print("✅ NO LEAKAGE - Splits are clean")
        print("="*80)

    return not has_leakage


# ============================================================================
# PHYSICAL DIRECTORY CREATION
# ============================================================================

def create_split_directories(dataset_path: str, splits_dir: str,
                            output_dir: str, verbose: bool = True):
    """Create physical train/val/test directories with actual audio files."""
    dataset_path = Path(dataset_path)
    splits_dir = Path(splits_dir)
    output_path = Path(output_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created directory: {split_dir}")

    # Process each split
    for split_name in ['train', 'val', 'test']:
        if verbose:
            print(f"\nProcessing {split_name} split...")

        split_file = splits_dir / f"{split_name}.txt"
        split_dir = output_path / split_name

        # Read split file
        with open(split_file, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]

        if verbose:
            print(f"Found {len(file_paths)} files")

        # Copy files with progress bar
        iterator = tqdm(file_paths, desc=f"Copying {split_name}") if verbose else file_paths
        for rel_path in iterator:
            # Parse path: "ClassName/filename.wav"
            class_name, filename = rel_path.split('/')

            # Source and destination
            src_file = dataset_path / class_name / filename
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            dst_file = class_dir / filename

            # Copy file
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: Source file not found: {src_file}")

    if verbose:
        print(f"\n✓ Split directories created in: {output_dir}")
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        for split_name in ['train', 'val', 'test']:
            split_dir = output_path / split_name
            total_files = sum(1 for _ in split_dir.rglob('*.wav'))
            print(f"{split_name.capitalize():5s}: {total_files:4d} files")
        print("="*80)


# ============================================================================
# VISUALIZATION
# ============================================================================

def print_split_summary(stats: Dict, detailed_stats: Dict):
    """Print a detailed summary of the splits."""
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for class_name, class_stats in detailed_stats.items():
        actual = class_stats['actual']
        target = class_stats['target']
        total = class_stats['total']

        train_pct = (actual['train'] / total) * 100
        val_pct = (actual['val'] / total) * 100
        test_pct = (actual['test'] / total) * 100

        status = "✓ PERFECT" if class_stats['error'] == 0 else f"Error: {class_stats['error']}"

        print(f"\n{class_name}: {status}")
        print(f"  Total: {total} samples")
        print(f"  Train: {actual['train']:3d} (target: {target['train']:3d}, {train_pct:5.2f}%)")
        print(f"  Val:   {actual['val']:3d} (target: {target['val']:3d}, {val_pct:5.2f}%)")
        print(f"  Test:  {actual['test']:3d} (target: {target['test']:3d}, {test_pct:5.2f}%)")

    # Overall stats
    overall = stats['_overall']
    total = overall['total_samples']
    train_pct = (overall['train_samples'] / total) * 100
    val_pct = (overall['val_samples'] / total) * 100
    test_pct = (overall['test_samples'] / total) * 100

    print(f"\n{'='*80}")
    print("OVERALL")
    print(f"{'='*80}")
    print(f"Total: {total} samples")
    print(f"Train: {overall['train_samples']} samples ({train_pct:.2f}%)")
    print(f"Val:   {overall['val_samples']} samples ({val_pct:.2f}%)")
    print(f"Test:  {overall['test_samples']} samples ({test_pct:.2f}%)")
    print(f"{'='*80}")


# ============================================================================
# PLOTTING
# ============================================================================

def plot_optimization_history(history: List[Dict], output_path: str = './optimization_history.png'):
    """
    Plot the optimization history similar to deep learning training curves.

    Args:
        history: List of dicts with iteration, temperature, current_score, best_score
        output_path: Where to save the plot
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
        return

    if not history:
        print("Warning: No history data to plot")
        return

    # Extract data
    iterations = [h['iteration'] for h in history]
    temperatures = [h['temperature'] for h in history]
    current_scores = [h['current_score'] for h in history]
    best_scores = [h['best_score'] for h in history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Simulated Annealing Optimization History', fontsize=16, fontweight='bold')

    # Plot 1: Best Score over Iterations
    ax1 = axes[0, 0]
    ax1.plot(iterations, best_scores, linewidth=2, color='#2E86AB', label='Best Score')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Error (samples)', fontsize=11)
    ax1.set_title('Best Score Progress', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)

    # Highlight perfect solution if achieved
    if best_scores[-1] == 0:
        perfect_iter = next(i for i, s in enumerate(best_scores) if s == 0)
        ax1.axvline(x=iterations[perfect_iter], color='green', linestyle='--',
                   linewidth=2, alpha=0.7, label='Perfect Solution')
        ax1.scatter([iterations[perfect_iter]], [0], color='green', s=100,
                   zorder=5, marker='*', label='Zero Error')
        ax1.legend(fontsize=10)

    # Plot 2: Current vs Best Score
    ax2 = axes[0, 1]
    ax2.plot(iterations, current_scores, linewidth=1, color='#A23B72',
            alpha=0.6, label='Current Score')
    ax2.plot(iterations, best_scores, linewidth=2, color='#2E86AB', label='Best Score')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Error (samples)', fontsize=11)
    ax2.set_title('Current vs Best Score', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)

    # Plot 3: Temperature over Iterations
    ax3 = axes[1, 0]
    ax3.plot(iterations, temperatures, linewidth=2, color='#F18F01')
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Temperature', fontsize=11)
    ax3.set_title('Temperature Schedule', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_yscale('log')  # Log scale for better visualization

    # Plot 4: Score Improvement Rate
    ax4 = axes[1, 1]
    # Calculate improvement: how much best score decreased
    improvements = []
    window = max(1, len(best_scores) // 100)  # Adaptive window size
    for i in range(window, len(best_scores)):
        improvement = best_scores[i-window] - best_scores[i]
        improvements.append(improvement)

    if improvements:
        ax4.plot(iterations[window:], improvements, linewidth=1, color='#06A77D')
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Score Reduction', fontsize=11)
        ax4.set_title(f'Improvement Rate (window={window})', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add stats text box
    stats_text = f"Total Iterations: {len(iterations):,}\n"
    stats_text += f"Initial Score: {best_scores[0]:.0f}\n"
    stats_text += f"Final Score: {best_scores[-1]:.0f}\n"
    stats_text += f"Improvement: {best_scores[0] - best_scores[-1]:.0f} ({(1 - best_scores[-1]/best_scores[0])*100:.1f}%)"

    fig.text(0.99, 0.01, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Optimization history plot saved to: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SEAbird Dataset Splitter using Simulated Annealing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - optimize splits only
  python seabird_dataset_splitter.py --dataset /path/to/seabird16k

  # With optimization history plot
  python seabird_dataset_splitter.py --dataset /path/to/seabird16k --plot

  # Create physical directories too
  python seabird_dataset_splitter.py --dataset /path/to/seabird16k --create-dirs

  # Full workflow with plotting
  python seabird_dataset_splitter.py --dataset /path/to/seabird16k --plot --create-dirs

  # Create directories from existing splits (auto-detects dataset path)
  python seabird_dataset_splitter.py --from-splits ./splits --dirs-output ./my_dataset_splits

  # Or specify dataset path explicitly if needed
  python seabird_dataset_splitter.py --from-splits ./splits --dataset /path/to/data --dirs-output ./output

Simulated Annealing:
  The algorithm mimics metallurgical annealing to find optimal splits:
  - High temperature: Explores widely, accepts many moves
  - Cooling down: Gradually becomes more selective
  - Low temperature: Only accepts improvements
        """
    )

    parser.add_argument('--dataset', type=str, default=str(EXTRACTED_SEGS),
                       help='Path to dataset directory')
    parser.add_argument('--dataset-label', type=str, default=None,
                       help='Short label identifying the dataset variant, included in the output '
                            'filename (e.g. "16khz", "44khz"). Auto-derived from --dataset path '
                            'when not given: "44khz" if the path contains "44100", else "16khz".')
    parser.add_argument('--output', type=str, default=str(SPLITS_DIR),
                       help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                       help='Target train ratio (default: 0.75)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                       help='Target validation ratio (default: 0.10)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Target test ratio (default: 0.15)')
    parser.add_argument('--create-dirs', action='store_true',
                       help='Create physical split directories with copied files')
    parser.add_argument('--dirs-output', type=str, default='./dataset_splits',
                       help='Output directory for physical splits (if --create-dirs)')
    parser.add_argument('--from-splits', type=str, default=None,
                       help='Skip optimization and create directories from existing split files (auto-detects dataset path from split_stats.json)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--initial-temp', type=float, default=INITIAL_TEMPERATURE,
                       help='Starting temperature (higher = more exploration)')
    parser.add_argument('--cooling-rate', type=float, default=COOLING_RATE,
                       help='Temperature decay rate (0-1, closer to 1 = slower)')
    parser.add_argument('--min-temp', type=float, default=MIN_TEMPERATURE,
                       help='Minimum temperature (stopping criterion)')
    parser.add_argument('--iterations-per-temp', type=int, default=ITERATIONS_PER_TEMP,
                       help='Number of iterations at each temperature')
    parser.add_argument('--plot', action='store_true',
                       help='Generate optimization history plot (requires matplotlib)')
    parser.add_argument('--plot-output', type=str, default='./optimization_history.png',
                       help='Output path for optimization plot')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()
    verbose = not args.quiet

    # Build dataset label and auto-name the splits CSV
    _label = args.dataset_label or ('44khz' if '44100' in str(args.dataset) else '16khz')
    _t  = int(round(args.train_ratio * 100))
    _v  = int(round(args.val_ratio   * 100))
    _te = int(round(args.test_ratio  * 100))
    _auto_csv = f"seabird_splits_sa_{_t}_{_v}_{_te}_{_label}.csv"

    # Validate ratios sum to 1.0
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        print(f"ERROR: Split ratios must sum to 1.0, got {ratio_sum:.6f} "
              f"({args.train_ratio} + {args.val_ratio} + {args.test_ratio})")
        return

    # Update global ratio constants from command line arguments
    global TARGET_TRAIN_RATIO, TARGET_VAL_RATIO, TARGET_TEST_RATIO
    TARGET_TRAIN_RATIO = args.train_ratio
    TARGET_VAL_RATIO = args.val_ratio
    TARGET_TEST_RATIO = args.test_ratio

    # Check if using existing splits (skip optimization mode)
    if args.from_splits:
        if verbose:
            print("="*80)
            print("MODE: Creating directories from existing splits")
            print("="*80)
            print(f"Reading splits from: {args.from_splits}")
            print()

        splits_dir = args.from_splits

        # Check if split files exist
        splits_path = Path(splits_dir)
        required_files = ['train.txt', 'val.txt', 'test.txt']
        missing_files = [f for f in required_files if not (splits_path / f).exists()]

        if missing_files:
            print(f"❌ Error: Missing split files in {splits_dir}:")
            for f in missing_files:
                print(f"  - {f}")
            return

        # Try to auto-detect dataset path from split_stats.json
        dataset_path = args.dataset
        stats_file = splits_path / 'split_stats.json'

        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            config = stats.get('_config', {})
            saved_dataset_path = config.get('dataset_path')

            if saved_dataset_path:
                # Check if user provided a different dataset path
                if args.dataset != '/Volumes/Evo/seabird16k':  # Not the default
                    # User explicitly specified a dataset path
                    dataset_path = args.dataset
                    if verbose:
                        print(f"Using dataset path from command line: {dataset_path}")
                else:
                    # Use saved path from split_stats.json
                    dataset_path = saved_dataset_path
                    if verbose:
                        print(f"Auto-detected dataset path from split_stats.json: {dataset_path}")
            else:
                if verbose:
                    print(f"Using dataset path: {dataset_path}")
        else:
            if verbose:
                print(f"⚠️  split_stats.json not found, using dataset path: {dataset_path}")

        # Verify dataset path exists
        if not Path(dataset_path).exists():
            print(f"\n❌ Error: Dataset path does not exist: {dataset_path}")
            print(f"\nPlease specify the correct dataset path with --dataset")
            return

        if verbose:
            print()

        # Verify no leakage
        if verbose:
            print("Verifying splits...")
        is_clean = verify_no_leakage(splits_dir, verbose=verbose)

        if not is_clean and verbose:
            print("\n⚠️  Warning: Leakage detected in splits!")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return

        # Create physical directories
        if verbose:
            print("\nCreating physical split directories...")
        create_split_directories(
            dataset_path,
            splits_dir,
            args.dirs_output,
            verbose=verbose
        )

        # Load and print stats if available
        stats_file = splits_path / 'split_stats.json'
        if stats_file.exists() and verbose:
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            overall = stats.get('_overall', {})
            if overall:
                print("\n" + "="*80)
                print("SPLIT STATISTICS")
                print("="*80)
                total = overall.get('total_samples', 0)
                train = overall.get('train_samples', 0)
                val = overall.get('val_samples', 0)
                test = overall.get('test_samples', 0)

                print(f"Total: {total:,} samples")
                print(f"Train: {train:,} samples ({train/total*100:.1f}%)")
                print(f"Val:   {val:,} samples ({val/total*100:.1f}%)")
                print(f"Test:  {test:,} samples ({test/total*100:.1f}%)")
                print("="*80)

        if verbose:
            print("\n✅ All done!")
        return

    # Normal mode: Run optimization
    if verbose:
        print("="*80)
        print("MODE: Optimize splits using Simulated Annealing")
        print("="*80)
        print()

    # Load dataset
    if verbose:
        print("Loading dataset structure...")
    structure = load_dataset_structure(args.dataset)

    if verbose:
        print(f"\nLoaded {len(structure)} classes:")
        for class_name, sources in structure.items():
            total = sum(len(files) for files in sources.values())
            print(f"  {class_name:30s}: {len(sources):3d} sources, {total:4d} samples")
        print()

    # Run simulated annealing
    best_assignment, best_stats, best_score, history = simulated_annealing(
        structure,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        min_temp=args.min_temp,
        iterations_per_temp=args.iterations_per_temp,
        seed=args.seed,
        verbose=verbose
    )

    # Save results
    if verbose:
        print("\nSaving optimized splits...")
    stats = save_splits(structure, best_assignment, args.output)

    # Write splits CSV
    csv_path = Path(args.output) / _auto_csv
    create_splits_csv(structure, best_assignment, csv_path,
                      train_ratio=TARGET_TRAIN_RATIO,
                      val_ratio=TARGET_VAL_RATIO,
                      test_ratio=TARGET_TEST_RATIO,
                      objective=best_score,
                      seed=args.seed if args.seed else 42,
                      verbose=verbose)

    # Plot optimization history if requested
    if args.plot:
        plot_output = Path(args.output) / Path(args.plot_output).name
        plot_optimization_history(history, str(plot_output))

    # Print summary
    if verbose:
        print_split_summary(stats, best_stats)

        if best_score == 0:
            print("\n🎉🎉🎉 PERFECT SPLIT ACHIEVED! 🎉🎉🎉")
            print("All classes have exactly 75%/10%/15% splits!")

    # Verify no leakage
    verify_no_leakage(args.output, verbose=verbose)

    # Create physical directories if requested
    if args.create_dirs:
        if verbose:
            print("\nCreating physical split directories...")
        create_split_directories(
            args.dataset,
            args.output,
            args.dirs_output,
            verbose=verbose
        )

    if verbose:
        print("\n✅ All done!")


if __name__ == '__main__':
    main()
