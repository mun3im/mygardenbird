#!/usr/bin/env python3
"""
SEAbird Dataset Splitter using Genetic Algorithm Optimization

This script creates optimal train/val/test splits for the SEAbird acoustic dataset
while ensuring:
1. Exact target distributions (75%/10%/15%) across all classes
2. Source-based separation to prevent data leakage
3. Class balance preservation

The script uses genetic algorithm optimization to find the best source assignments
that satisfy all constraints. The algorithm evolves a population of candidate
solutions through selection, crossover, and mutation operations.

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

from config import MYGARDENBIRD_16K, MYGARDENBIRD_44K, METADATA_16K, METADATA_44K

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

# Genetic Algorithm parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 2000
MUTATION_RATE = 0.10  # Lower initial rate
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3  # Smaller tournament = more diversity
ELITISM_COUNT = 5  # Keep more elite individuals
EARLY_STOP_PATIENCE = 300  # More patience before early stopping


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
    This is the fitness function for the genetic algorithm (lower is better).

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
# GENETIC ALGORITHM OPTIMIZATION
# ============================================================================

def create_random_individual(structure: Dict[str, Dict[str, List[str]]],
                             seed: int = None) -> Dict[str, Dict[str, str]]:
    """
    Create a random individual (split assignment) with greedy initialization.
    This creates better initial solutions than pure random assignment.

    Args:
        structure: Dataset structure
        seed: Random seed

    Returns:
        Individual (assignment dictionary)
    """
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

        # Greedy assignment with randomness
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

            # Add randomness: 80% greedy, 20% random
            if random.random() < 0.8:
                chosen_split = min(options, key=lambda x: x[1])[0]
            else:
                chosen_split = random.choice(options)[0]

            assignment[class_name][src] = chosen_split
            if chosen_split == 'train':
                train_count += size
            elif chosen_split == 'val':
                val_count += size
            else:
                test_count += size

    return assignment


def initialize_population(structure: Dict[str, Dict[str, List[str]]],
                         pop_size: int,
                         seed: int = None) -> List[Dict]:
    """
    Initialize population of individuals.

    Args:
        structure: Dataset structure
        pop_size: Population size
        seed: Random seed

    Returns:
        List of individuals (assignments)
    """
    if seed is not None:
        random.seed(seed)

    population = []
    for i in range(pop_size):
        individual = create_random_individual(structure, seed=None)
        population.append(individual)

    return population


def tournament_selection(population: List[Dict],
                        fitness_scores: List[float],
                        tournament_size: int) -> Dict:
    """
    Select an individual using tournament selection.

    Args:
        population: List of individuals
        fitness_scores: Fitness scores (lower is better)
        tournament_size: Number of individuals in tournament

    Returns:
        Selected individual
    """
    # Randomly select tournament_size individuals
    indices = random.sample(range(len(population)), tournament_size)

    # Find the best individual in tournament (lowest fitness)
    best_idx = min(indices, key=lambda i: fitness_scores[i])

    return copy.deepcopy(population[best_idx])


def crossover(parent1: Dict[str, Dict[str, str]],
             parent2: Dict[str, Dict[str, str]],
             structure: Dict[str, Dict[str, List[str]]]) -> Tuple[Dict, Dict]:
    """
    Perform crossover between two parents to create two offspring.
    Uses class-level crossover: for each class, randomly choose parent.

    Args:
        parent1: First parent
        parent2: Second parent
        structure: Dataset structure

    Returns:
        Two offspring
    """
    child1 = {}
    child2 = {}

    for class_name in structure.keys():
        # Randomly choose which parent contributes to which child
        if random.random() < 0.5:
            child1[class_name] = copy.deepcopy(parent1[class_name])
            child2[class_name] = copy.deepcopy(parent2[class_name])
        else:
            child1[class_name] = copy.deepcopy(parent2[class_name])
            child2[class_name] = copy.deepcopy(parent1[class_name])

    return child1, child2


def mutate(individual: Dict[str, Dict[str, str]],
          structure: Dict[str, Dict[str, List[str]]],
          mutation_rate: float,
          guided: bool = True) -> Dict:
    """
    Mutate an individual by changing source assignments.
    Uses guided mutation that considers current split sizes.

    Args:
        individual: Individual to mutate
        structure: Dataset structure
        mutation_rate: Probability of mutation per class
        guided: Use guided mutation (70% of the time)

    Returns:
        Mutated individual
    """
    mutated = copy.deepcopy(individual)

    for class_name, sources in structure.items():
        if random.random() >= mutation_rate:
            continue

        # Calculate current split sizes for this class
        total_samples = sum(len(files) for files in sources.values())
        target_train = int(total_samples * TARGET_TRAIN_RATIO)
        target_val = int(total_samples * TARGET_VAL_RATIO)
        target_test = int(total_samples * TARGET_TEST_RATIO)

        train_count = sum(len(sources[src]) for src in mutated[class_name]
                         if mutated[class_name][src] == 'train')
        val_count = sum(len(sources[src]) for src in mutated[class_name]
                       if mutated[class_name][src] == 'val')
        test_count = sum(len(sources[src]) for src in mutated[class_name]
                        if mutated[class_name][src] == 'test')

        # Determine which splits are over/under target
        train_diff = train_count - target_train
        val_diff = val_count - target_val
        test_diff = test_count - target_test

        # 70% guided mutation, 30% random
        if guided and random.random() < 0.7:
            # Guided: Move sources from oversized splits to undersized splits
            from_split = None
            to_split = None

            # Find most oversized split
            if train_diff > 0 and train_diff > max(val_diff, test_diff):
                from_split = 'train'
            elif val_diff > 0 and val_diff > max(train_diff, test_diff):
                from_split = 'val'
            elif test_diff > 0:
                from_split = 'test'

            # Find most undersized split
            if from_split:
                if train_diff < 0 and train_diff < min(val_diff, test_diff):
                    to_split = 'train'
                elif val_diff < 0 and val_diff < min(train_diff, test_diff):
                    to_split = 'val'
                elif test_diff < 0:
                    to_split = 'test'

            # Apply guided mutation
            if from_split and to_split and from_split != to_split:
                # Find sources in from_split
                candidates = [src for src, split in mutated[class_name].items()
                            if split == from_split]
                if candidates:
                    # Pick smallest source to move (minimize disruption)
                    src_to_move = min(candidates, key=lambda s: len(sources[s]))
                    mutated[class_name][src_to_move] = to_split
        else:
            # Random mutation: pick random source and reassign
            src = random.choice(list(sources.keys()))
            current_split = mutated[class_name][src]
            other_splits = [s for s in ['train', 'val', 'test'] if s != current_split]
            mutated[class_name][src] = random.choice(other_splits)

    return mutated


def local_search(individual: Dict[str, Dict[str, str]],
                structure: Dict[str, Dict[str, List[str]]],
                max_iterations: int = 50) -> Tuple[Dict, float]:
    """
    Perform local search hill climbing to refine an individual.
    Similar to SA's local refinement but deterministic.

    Args:
        individual: Individual to refine
        structure: Dataset structure
        max_iterations: Maximum number of improvement attempts

    Returns:
        (improved_individual, improved_score)
    """
    current = copy.deepcopy(individual)
    current_score = calculate_split_score(structure, current)

    for _ in range(max_iterations):
        improved = False

        # Try swapping sources within each class
        for class_name, sources in structure.items():
            source_list = list(sources.keys())
            if len(source_list) < 2:
                continue

            # Try swapping pairs
            for i in range(len(source_list)):
                for j in range(i + 1, len(source_list)):
                    src1, src2 = source_list[i], source_list[j]
                    split1 = current[class_name][src1]
                    split2 = current[class_name][src2]

                    if split1 == split2:
                        continue

                    # Try swap
                    current[class_name][src1] = split2
                    current[class_name][src2] = split1

                    new_score = calculate_split_score(structure, current)

                    if new_score < current_score:
                        # Keep the improvement
                        current_score = new_score
                        improved = True
                        break
                    else:
                        # Revert
                        current[class_name][src1] = split1
                        current[class_name][src2] = split2

                if improved:
                    break
            if improved:
                break

        # If no improvement found, stop
        if not improved:
            break

        # If perfect score, stop
        if current_score == 0:
            break

    return current, current_score


def genetic_algorithm(structure: Dict[str, Dict[str, List[str]]],
                     pop_size: int = POPULATION_SIZE,
                     max_generations: int = MAX_GENERATIONS,
                     mutation_rate: float = MUTATION_RATE,
                     crossover_rate: float = CROSSOVER_RATE,
                     tournament_size: int = TOURNAMENT_SIZE,
                     elitism_count: int = ELITISM_COUNT,
                     early_stop_patience: int = EARLY_STOP_PATIENCE,
                     seed: int = None,
                     verbose: bool = True) -> Tuple[Dict, Dict, float, List]:
    """
    Optimize splits using Genetic Algorithm.

    Args:
        structure: Dataset structure
        pop_size: Population size
        max_generations: Maximum number of generations
        mutation_rate: Probability of mutation per gene
        crossover_rate: Probability of crossover
        tournament_size: Size of tournament selection
        elitism_count: Number of best individuals to preserve
        seed: Random seed
        verbose: Print progress

    Returns:
        (best_assignment, best_stats, best_score, history)
    """
    if verbose:
        print("🧬 GENETIC ALGORITHM OPTIMIZATION")
        print("="*80)
        print(f"Population size: {pop_size}")
        print(f"Max generations: {max_generations}")
        print(f"Mutation rate: {mutation_rate:.1%}")
        print(f"Crossover rate: {crossover_rate:.1%}")
        print(f"Tournament size: {tournament_size}")
        print(f"Elitism: {elitism_count} individuals")
        print()

    if seed is not None:
        random.seed(seed)

    # Initialize population
    if verbose:
        print("Initializing population...")
    population = initialize_population(structure, pop_size, seed=seed)

    # Track best solution
    best_assignment = None
    best_score = float('inf')
    best_stats = None
    history = []

    # Main evolution loop
    if verbose:
        print(f"Evolving population for {max_generations} generations...")
        print()

    pbar = tqdm(range(max_generations),
                desc="Evolving",
                disable=not verbose,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Best: {postfix}')

    generation = 0
    generations_without_improvement = 0

    try:
        for generation in pbar:
            # Calculate fitness for all individuals
            fitness_scores = [calculate_split_score(structure, ind) for ind in population]

            # Find best individual in current generation
            min_fitness = min(fitness_scores)
            min_idx = fitness_scores.index(min_fitness)

            # Update global best
            if min_fitness < best_score:
                best_score = min_fitness
                best_assignment = copy.deepcopy(population[min_idx])
                best_stats = get_detailed_stats(structure, best_assignment)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Calculate stats
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            max_fitness = max(fitness_scores)

            # Record history
            history.append({
                'generation': generation + 1,
                'best_fitness': best_score,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'max_fitness': max_fitness
            })

            # Update progress bar
            pbar.set_postfix_str(f"Score={best_score:.0f}, Avg={avg_fitness:.0f}, Gen w/o improv={generations_without_improvement}")

            # Check if perfect solution found
            if best_score == 0:
                pbar.set_postfix_str(f"🎉 PERFECT! Score={best_score:.0f}")
                break

            # Early stopping if no improvement for early_stop_patience generations
            if generations_without_improvement >= early_stop_patience:
                if verbose:
                    pbar.set_postfix_str(f"Early stop (no improvement for {early_stop_patience} gen)")
                break

            # Create new population
            new_population = []

            # Elitism: Keep best individuals
            sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i])
            for i in range(elitism_count):
                elite = copy.deepcopy(population[sorted_indices[i]])

                # Apply local search to elite individuals every 50 generations
                if (generation + 1) % 50 == 0 and i == 0:  # Only on best individual
                    elite, _ = local_search(elite, structure, max_iterations=30)

                new_population.append(elite)

            # Generate offspring
            while len(new_population) < pop_size:
                # Selection
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)

                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2, structure)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                # Mutation
                child1 = mutate(child1, structure, mutation_rate)
                child2 = mutate(child2, structure, mutation_rate)

                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)

            # Replace population
            population = new_population

    finally:
        pbar.close()

    if verbose:
        print()
        if best_score == 0:
            print("🎉 PERFECT SOLUTION FOUND!")
        print("="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Generations: {generation + 1:,}")
        print(f"Best score achieved: {best_score:.0f}")
        if history:
            print(f"Average fitness (final): {history[-1]['avg_fitness']:.0f}")
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
        f.write(f"# split_ratio={train_pct}:{val_pct}:{test_pct} seed={seed} objective={objective:.0f} solver=genetic_algorithm\n")
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
        'optimization': 'genetic_algorithm'
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
    Plot the GA optimization history.

    Args:
        history: List of dicts with generation, best_fitness, avg_fitness, etc.
        output_path: Where to save the plot
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
        return

    if not history:
        print("Warning: No history data to plot")
        return

    # Extract data
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    min_fitness = [h['min_fitness'] for h in history]
    max_fitness = [h['max_fitness'] for h in history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Genetic Algorithm Optimization History', fontsize=16, fontweight='bold')

    # Plot 1: Best Fitness over Generations
    ax1 = axes[0, 0]
    ax1.plot(generations, best_fitness, linewidth=2, color='#2E86AB', label='Best Fitness')
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Fitness (Error)', fontsize=11)
    ax1.set_title('Best Fitness Progress', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)

    # Highlight perfect solution if achieved
    if best_fitness[-1] == 0:
        perfect_gen = next(i for i, f in enumerate(best_fitness) if f == 0)
        ax1.axvline(x=generations[perfect_gen], color='green', linestyle='--',
                   linewidth=2, alpha=0.7, label='Perfect Solution')
        ax1.scatter([generations[perfect_gen]], [0], color='green', s=100,
                   zorder=5, marker='*', label='Zero Error')
        ax1.legend(fontsize=10)

    # Plot 2: Population Fitness Distribution
    ax2 = axes[0, 1]
    ax2.fill_between(generations, min_fitness, max_fitness, alpha=0.2, color='gray', label='Min-Max Range')
    ax2.plot(generations, avg_fitness, linewidth=2, color='#A23B72', label='Average Fitness')
    ax2.plot(generations, best_fitness, linewidth=2, color='#2E86AB', label='Best Fitness')
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Fitness (Error)', fontsize=11)
    ax2.set_title('Population Fitness Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)

    # Plot 3: Fitness Improvement
    ax3 = axes[1, 0]
    # Calculate improvement: how much best fitness decreased
    improvements = []
    for i in range(1, len(best_fitness)):
        improvement = best_fitness[i-1] - best_fitness[i]
        improvements.append(improvement)

    if improvements:
        ax3.bar(generations[1:], improvements, width=1.0, color='#06A77D', alpha=0.7)
        ax3.set_xlabel('Generation', fontsize=11)
        ax3.set_ylabel('Fitness Reduction', fontsize=11)
        ax3.set_title('Improvement per Generation', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 4: Convergence Measure (Avg - Best)
    ax4 = axes[1, 1]
    convergence = [avg - best for avg, best in zip(avg_fitness, best_fitness)]
    ax4.plot(generations, convergence, linewidth=2, color='#F18F01')
    ax4.set_xlabel('Generation', fontsize=11)
    ax4.set_ylabel('Avg - Best Fitness', fontsize=11)
    ax4.set_title('Population Diversity (Convergence)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Add stats text box
    stats_text = f"Total Generations: {len(generations):,}\n"
    stats_text += f"Initial Best: {best_fitness[0]:.0f}\n"
    stats_text += f"Final Best: {best_fitness[-1]:.0f}\n"
    stats_text += f"Improvement: {best_fitness[0] - best_fitness[-1]:.0f} ({(1 - best_fitness[-1]/max(best_fitness[0], 1))*100:.1f}%)"

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
        description='SEAbird Dataset Splitter using Genetic Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - optimize splits only
  python seabird_splitter_genetic_algorithm.py --dataset /path/to/seabird16khz_flat

  # With optimization history plot
  python seabird_splitter_genetic_algorithm.py --dataset /path/to/seabird16khz_flat --plot

  # Create physical directories too
  python seabird_splitter_genetic_algorithm.py --dataset /path/to/seabird16khz_flat --create-dirs

  # Full workflow with custom parameters
  python seabird_splitter_genetic_algorithm.py --dataset /path/to/seabird16khz_flat \\
      --population 150 --generations 2000 --mutation-rate 0.2 --plot --create-dirs

  # Create directories from existing splits
  python seabird_splitter_genetic_algorithm.py --from-splits ./splits --dirs-output ./output

Genetic Algorithm:
  The algorithm evolves a population of candidate solutions:
  - Selection: Tournament selection chooses parents
  - Crossover: Parents combine to create offspring
  - Mutation: Random changes introduce diversity
  - Elitism: Best solutions carry to next generation
        """
    )

    parser.add_argument('--dataset', type=str, default=str(MYGARDENBIRD_16K),
                       help='Path to dataset directory')
    parser.add_argument('--dataset-label', type=str, default=None,
                       help='Short label identifying the dataset variant, included in the output '
                            'filename (e.g. "16khz", "44khz"). Auto-derived from --dataset path '
                            'when not given: "44khz" if the path contains "44", else "16khz".')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for split files (default: metadata directory corresponding to --dataset)')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                       help='Target train ratio (default: 0.75)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                       help='Target validation ratio (default: 0.10)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Target test ratio (default: 0.15)')
    parser.add_argument('--create-dirs', action='store_true',
                       help='Create physical split directories with copied files')
    parser.add_argument('--dirs-output', type=str, default='./dataset_splits_ga',
                       help='Output directory for physical splits (if --create-dirs)')
    parser.add_argument('--from-splits', type=str, default=None,
                       help='Skip optimization and create directories from existing split files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--population', type=int, default=POPULATION_SIZE,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=MAX_GENERATIONS,
                       help='Maximum number of generations')
    parser.add_argument('--mutation-rate', type=float, default=MUTATION_RATE,
                       help='Mutation rate (0-1)')
    parser.add_argument('--crossover-rate', type=float, default=CROSSOVER_RATE,
                       help='Crossover rate (0-1)')
    parser.add_argument('--tournament-size', type=int, default=TOURNAMENT_SIZE,
                       help='Tournament selection size')
    parser.add_argument('--elitism', type=int, default=ELITISM_COUNT,
                       help='Number of elite individuals to preserve')
    parser.add_argument('--plot', action='store_true',
                       help='Generate optimization history plot (requires matplotlib)')
    parser.add_argument('--plot-output', type=str, default='./optimization_history_ga.png',
                       help='Output path for optimization plot')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()
    verbose = not args.quiet

    # Build dataset label and auto-name the splits CSV
    _label = args.dataset_label or ('44khz' if '44' in str(args.dataset) else '16khz')
    _t  = int(round(args.train_ratio * 100))
    _v  = int(round(args.val_ratio   * 100))
    _te = int(round(args.test_ratio  * 100))
    _auto_csv = f"splits_ga_{_t}_{_v}_{_te}.csv"

    # Auto-select output directory based on dataset path if not specified
    if args.output is None:
        if '44' in str(args.dataset):
            args.output = str(METADATA_44K)
        else:
            args.output = str(METADATA_16K)

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
                if args.dataset != '/Volumes/Evo/seabird16khz_flat':  # Not the default
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
        print("MODE: Optimize splits using Genetic Algorithm")
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

    # Run genetic algorithm
    best_assignment, best_stats, best_score, history = genetic_algorithm(
        structure,
        pop_size=args.population,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        elitism_count=args.elitism,
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
