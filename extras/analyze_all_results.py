#!/usr/bin/env python3
"""
Analyze all completed experiments and generate summary statistics
"""
import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_results(base_dir):
    """Load all results.json files from a directory"""
    results = []
    for exp_dir in sorted(Path(base_dir).iterdir()):
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            results_file = exp_dir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    # Parse experiment name
                    parts = exp_dir.name.split('_')
                    model = parts[0]
                    aug = 'noaug' if 'noaug' in exp_dir.name else (
                        'specaug' if 'specaug' in exp_dir.name else 'mixup'
                    )
                    seed = int(parts[-1].replace('seed', ''))
                    
                    results.append({
                        'model': model,
                        'augmentation': aug,
                        'seed': seed,
                        'accuracy': data['test_accuracy'] * 100,  # Convert to percentage
                        'loss': data['test_loss'],
                        'training_time': data['training_time_minutes'],
                        'sample_rate': data['config']['sample_rate'],
                        'exp_name': exp_dir.name
                    })
    return results

# Load all results
results_16k = load_results('results_16k_linux')
results_44k = load_results('results_44k_linux')

print("=" * 80)
print("COMPLETED EXPERIMENTS SUMMARY")
print("=" * 80)

# Aggregate by model and augmentation
def aggregate_by_config(results, label):
    print(f"\n{label}")
    print("-" * 80)
    
    grouped = defaultdict(list)
    for r in results:
        key = (r['model'], r['augmentation'])
        grouped[key].append(r['accuracy'])
    
    # Sort by model and augmentation
    model_order = ['mobilenetv3s', 'efficientnetb0', 'resnet50', 'vgg16']
    aug_order = ['noaug', 'specaug', 'mixup']
    
    for model in model_order:
        for aug in aug_order:
            key = (model, aug)
            if key in grouped:
                accs = grouped[key]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0
                n_seeds = len(accs)
                print(f"{model:20s} | {aug:10s} | {mean_acc:6.2f}% ± {std_acc:4.2f}% | n={n_seeds}")

aggregate_by_config(results_16k, "16 kHz Results")
aggregate_by_config(results_44k, "44.1 kHz Results")

# Best configurations
print("\n" + "=" * 80)
print("BEST CONFIGURATIONS")
print("=" * 80)

def find_best(results, label):
    grouped = defaultdict(list)
    for r in results:
        key = (r['model'], r['augmentation'])
        grouped[key].append(r['accuracy'])
    
    configs = []
    for key, accs in grouped.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0
        configs.append((key[0], key[1], mean_acc, std_acc, len(accs)))
    
    configs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{label} - Top 5:")
    for i, (model, aug, mean, std, n) in enumerate(configs[:5], 1):
        print(f"  {i}. {model:20s} + {aug:10s}: {mean:6.2f}% ± {std:4.2f}% (n={n})")

find_best(results_16k, "16 kHz")
find_best(results_44k, "44.1 kHz")

# Per-model statistics (averaged across augmentations)
print("\n" + "=" * 80)
print("PER-MODEL STATISTICS (all augmentations)")
print("=" * 80)

def per_model_stats(results, label):
    print(f"\n{label}:")
    grouped = defaultdict(list)
    for r in results:
        grouped[r['model']].append(r['accuracy'])
    
    model_order = ['mobilenetv3s', 'efficientnetb0', 'resnet50', 'vgg16']
    for model in model_order:
        if model in grouped:
            accs = grouped[model]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0
            min_acc = np.min(accs)
            max_acc = np.max(accs)
            n = len(accs)
            print(f"  {model:20s}: {mean_acc:6.2f}% ± {std_acc:4.2f}% | range: [{min_acc:6.2f}%, {max_acc:6.2f}%] | n={n}")

per_model_stats(results_16k, "16 kHz")
per_model_stats(results_44k, "44.1 kHz")

print("\n" + "=" * 80)
print(f"Total experiments completed: {len(results_16k) + len(results_44k)}")
print(f"  16 kHz: {len(results_16k)}")
print(f"  44.1 kHz: {len(results_44k)}")
print("=" * 80)
