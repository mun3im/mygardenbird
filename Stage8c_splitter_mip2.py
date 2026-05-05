#!/usr/bin/env python3
"""
MyGardenBird Dataset Splitter — MIP with dual-objective (clip + source balance)

Variant of Stage8a_splitter_mip.py that adds a secondary source-count balance
term to the optimisation objective.

Stage8a problem:
  The original solver minimises only clip-count deviation from the target ratio.
  It is blind to how many source recordings land in each partition.  Dense
  sources (many clips per recording) can satisfy an entire partition quota with
  very few recordings, leaving val/test with poor source diversity even when
  clip counts are exact.  Common Iora illustrates this: 9 val sources vs 22
  test sources, despite both partitions containing exactly 60 clips.

This variant's objective:
  minimise  Σ clip_slack  +  ε · Σ source_slack(val, test)

where ε is a small weight (default 0.001) chosen so that the source term can
never outweigh a single clip deviation:

  ε × max_possible_source_slack < 1.0

With ≤200 sources per class, ε = 0.001 gives a maximum source contribution of
0.2 — always below 1 clip — so clip balance is guaranteed unchanged.  Among
the (often many) degenerate solutions that achieve zero clip deviation, the
solver will prefer those with the most balanced source counts.

New CLI argument:
  --source-weight FLOAT   Weight ε for the source-balance term (default: 0.001)

All other behaviour is identical to Stage8a_splitter_mip.py.

Requires: pip install pulp
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

# Progress bar (startup animation inherited from original)
pbar = tqdm(
    total=100,
    desc="Starting up...",
    bar_format='{desc} {bar}',
    colour='red',
    dynamic_ncols=True
)
for i in range(100):
    time.sleep(0.01)
pbar.close()


from config import MYGARDENBIRD_16K, MYGARDENBIRD_44K, METADATA_16K, METADATA_44K

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

TARGET_TRAIN_RATIO = 0.80
TARGET_VAL_RATIO   = 0.10
TARGET_TEST_RATIO  = 0.10

MIP_TIME_LIMIT    = 600   # seconds
MIP_GAP_TOLERANCE = 0.0   # require optimal solution

# Default weight for the secondary source-balance term.
# Must satisfy: SOURCE_BALANCE_WEIGHT × max_sources_per_class < 1.0
# (so source terms never outweigh a single clip deviation)
DEFAULT_SOURCE_WEIGHT = 0.001


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset_structure(dataset_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load dataset structure: class -> source -> [files]."""
    structure = {}

    for class_dir in Path(dataset_path).iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        class_name = class_dir.name
        structure[class_name] = defaultdict(list)

        for audio_file in class_dir.glob('*.wav'):
            source = audio_file.stem.split('_')[0]
            structure[class_name][source].append(audio_file.name)

    return structure


# ============================================================================
# MIP OPTIMIZATION — dual objective
# ============================================================================

def optimize_split_mip(structure: Dict[str, Dict[str, List[str]]],
                       train_ratio: float = TARGET_TRAIN_RATIO,
                       val_ratio:   float = TARGET_VAL_RATIO,
                       test_ratio:  float = TARGET_TEST_RATIO,
                       time_limit:  int   = MIP_TIME_LIMIT,
                       source_weight: float = DEFAULT_SOURCE_WEIGHT,
                       verbose: bool = True) -> Tuple[Dict, Dict, float]:
    """
    MIP dataset split with dual-objective: clip balance (primary) + source
    balance in val/test (secondary, weight = source_weight).

    Returns (assignment, stats, objective_value).
    """
    if verbose:
        print("🔧 MIXED INTEGER PROGRAMMING OPTIMIZATION  (dual-objective)")
        print("=" * 80)
        print(f"Time limit:      {time_limit}s")
        print(f"Gap tolerance:   {MIP_GAP_TOLERANCE}")
        print(f"Target ratios:   {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
        print(f"Source weight ε: {source_weight}  (secondary objective)")

        max_src = max(len(s) for s in structure.values())
        max_src_contrib = source_weight * max_src
        print(f"  Max source contribution: {source_weight} × {max_src} = {max_src_contrib:.4f}  "
              f"{'✓ < 1 clip' if max_src_contrib < 1.0 else '⚠️ WARNING: may affect clip balance!'}")
        print()

    start_time = time.time()

    # ── Clip targets ──────────────────────────────────────────────────────────
    class_totals = {
        cls: sum(len(f) for f in srcs.values())
        for cls, srcs in structure.items()
    }
    clip_targets = {
        cls: {
            'train': round(total * train_ratio),
            'val':   round(total * val_ratio),
            'test':  round(total * test_ratio),
        }
        for cls, total in class_totals.items()
    }

    # ── Source targets (val and test only; train is the residual) ─────────────
    source_targets = {
        cls: {
            'val':  round(len(srcs) * val_ratio),
            'test': round(len(srcs) * test_ratio),
        }
        for cls, srcs in structure.items()
    }

    if verbose:
        print("📊 Clip targets per class:")
        for cls in sorted(clip_targets):
            t = clip_targets[cls]
            st = source_targets[cls]
            n_src = len(structure[cls])
            print(f"  {cls:30s}: clips {t['train']:4d}/{t['val']:3d}/{t['test']:3d}  "
                  f"sources target val={st['val']:3d}/{n_src} test={st['test']:3d}/{n_src}")
        print()

    # ── Problem ───────────────────────────────────────────────────────────────
    prob = LpProblem("Dataset_Split_DualObj", LpMinimize)

    # Decision variables: x[cls][src][split] ∈ {0,1}
    x = {
        cls: {
            src: {
                split: LpVariable(f"x_{cls}_{src}_{split}", cat=LpBinary)
                for split in ('train', 'val', 'test')
            }
            for src in srcs
        }
        for cls, srcs in structure.items()
    }

    # Clip slack variables (deviation from clip targets)
    clip_sp = {}
    clip_sn = {}
    for cls in structure:
        clip_sp[cls] = {s: LpVariable(f"csp_{cls}_{s}", lowBound=0) for s in ('train', 'val', 'test')}
        clip_sn[cls] = {s: LpVariable(f"csn_{cls}_{s}", lowBound=0) for s in ('train', 'val', 'test')}

    # Source slack variables (deviation from source targets — val and test only)
    src_sp = {}
    src_sn = {}
    for cls in structure:
        src_sp[cls] = {s: LpVariable(f"ssp_{cls}_{s}", lowBound=0) for s in ('val', 'test')}
        src_sn[cls] = {s: LpVariable(f"ssn_{cls}_{s}", lowBound=0) for s in ('val', 'test')}

    # ── Objective: primary (clips) + ε × secondary (sources) ─────────────────
    prob += (
        lpSum(clip_sp[cls][sp] + clip_sn[cls][sp]
              for cls in structure for sp in ('train', 'val', 'test'))
        + source_weight * lpSum(src_sp[cls][sp] + src_sn[cls][sp]
                                for cls in structure for sp in ('val', 'test'))
    )

    # ── Constraint 1: each source goes to exactly one partition ───────────────
    for cls, srcs in structure.items():
        for src in srcs:
            prob += (
                x[cls][src]['train'] + x[cls][src]['val'] + x[cls][src]['test'] == 1,
                f"one_split_{cls}_{src}"
            )

    # ── Constraint 2: clip count matches target (with slack) ──────────────────
    for cls, srcs in structure.items():
        for sp in ('train', 'val', 'test'):
            actual_clips = lpSum(x[cls][src][sp] * len(files)
                                 for src, files in srcs.items())
            prob += (
                actual_clips == clip_targets[cls][sp] + clip_sp[cls][sp] - clip_sn[cls][sp],
                f"clip_target_{cls}_{sp}"
            )

    # ── Constraint 3: source count matches target (with slack, val/test only) ─
    for cls, srcs in structure.items():
        for sp in ('val', 'test'):
            actual_srcs = lpSum(x[cls][src][sp] for src in srcs)
            prob += (
                actual_srcs == source_targets[cls][sp] + src_sp[cls][sp] - src_sn[cls][sp],
                f"src_target_{cls}_{sp}"
            )

    # ── Solve ─────────────────────────────────────────────────────────────────
    if verbose:
        print("🚀 Solving MIP problem (dual-objective)...")
        print()

    solver = PULP_CBC_CMD(
        msg=verbose,
        timeLimit=time_limit,
        gapRel=MIP_GAP_TOLERANCE,
        threads=None
    )
    prob.solve(solver)

    solve_time = time.time() - start_time
    status = LpStatus[prob.status]

    if verbose:
        obj_val = value(prob.objective)
        clip_component   = sum(value(clip_sp[cls][sp]) + value(clip_sn[cls][sp])
                               for cls in structure for sp in ('train', 'val', 'test'))
        source_component = source_weight * sum(
                               value(src_sp[cls][sp]) + value(src_sn[cls][sp])
                               for cls in structure for sp in ('val', 'test'))
        print(f"\n✓ Solution status: {status}")
        print(f"  Solve time:         {solve_time:.2f}s")
        print(f"  Objective (total):  {obj_val:.4f}")
        print(f"    clip component:   {clip_component:.4f}")
        print(f"    source component: {source_component:.4f}")
        print()

    # ── Extract assignment ────────────────────────────────────────────────────
    assignment = {}
    for cls, srcs in structure.items():
        assignment[cls] = {}
        for src in srcs:
            for sp in ('train', 'val', 'test'):
                if value(x[cls][src][sp]) > 0.5:
                    assignment[cls][src] = sp
                    break

    stats = calculate_statistics(structure, assignment,
                                 train_ratio=train_ratio,
                                 val_ratio=val_ratio,
                                 test_ratio=test_ratio)

    return assignment, stats, value(prob.objective) if prob.objective else 0


# ============================================================================
# STATISTICS  (unchanged from Stage8a)
# ============================================================================

def calculate_statistics(structure, assignment,
                         train_ratio=TARGET_TRAIN_RATIO,
                         val_ratio=TARGET_VAL_RATIO,
                         test_ratio=TARGET_TEST_RATIO):
    stats = {}

    for cls, srcs in structure.items():
        def clip_count(sp):
            return sum(len(f) for src, f in srcs.items() if assignment[cls].get(src) == sp)
        def src_count(sp):
            return sum(1 for src in srcs if assignment[cls].get(src) == sp)

        stats[cls] = {
            'total_samples':  sum(len(f) for f in srcs.values()),
            'train_samples':  clip_count('train'),
            'val_samples':    clip_count('val'),
            'test_samples':   clip_count('test'),
            'total_sources':  len(srcs),
            'train_sources':  src_count('train'),
            'val_sources':    src_count('val'),
            'test_sources':   src_count('test'),
        }

    overall_tr  = sum(s['train_samples']  for s in stats.values())
    overall_val = sum(s['val_samples']    for s in stats.values())
    overall_te  = sum(s['test_samples']   for s in stats.values())
    stats['_overall'] = {
        'total_samples':  overall_tr + overall_val + overall_te,
        'train_samples':  overall_tr,
        'val_samples':    overall_val,
        'test_samples':   overall_te,
        'total_sources':  sum(s['total_sources']  for s in stats.values() if 'total_sources' in s),
        'train_sources':  sum(s['train_sources']  for s in stats.values() if 'train_sources' in s),
        'val_sources':    sum(s['val_sources']    for s in stats.values() if 'val_sources' in s),
        'test_sources':   sum(s['test_sources']   for s in stats.values() if 'test_sources' in s),
    }
    stats['_config'] = {
        'train_ratio': train_ratio,
        'val_ratio':   val_ratio,
        'test_ratio':  test_ratio,
        'optimization': 'mip_dual_objective_clip_source',
    }
    return stats


def format_statistics_text(stats):
    lines = ["=" * 80, "FINAL STATISTICS", "=" * 80, "",
             "Per-Class Breakdown:",
             f"{'Class':<30} {'Train':>8} {'Val':>6} {'Test':>6} {'Total':>7} | {'Sources':>7}",
             "-" * 80]

    for cls in sorted(k for k in stats if not k.startswith('_')):
        s = stats[cls]
        lines.append(f"{cls:<30} {s['train_samples']:8d} {s['val_samples']:6d} "
                     f"{s['test_samples']:6d} {s['total_samples']:7d} | "
                     f"{s['train_sources']:3d}/{s['val_sources']:3d}/{s['test_sources']:3d}")

    lines.append("-" * 80)
    o = stats['_overall']
    lines.append(f"{'OVERALL':<30} {o['train_samples']:8d} {o['val_samples']:6d} "
                 f"{o['test_samples']:6d} {o['total_samples']:7d} | "
                 f"{o['train_sources']:3d}/{o['val_sources']:3d}/{o['test_sources']:3d}")
    total = o['total_samples']
    cfg = stats['_config']
    lines += ["", "Actual Ratios:",
              f"  Train: {o['train_samples']/total:.1%} (target: {cfg['train_ratio']:.1%})",
              f"  Val:   {o['val_samples']  /total:.1%} (target: {cfg['val_ratio']  :.1%})",
              f"  Test:  {o['test_samples'] /total:.1%} (target: {cfg['test_ratio'] :.1%})",
              "", f"  Optimisation: {cfg['optimization']}", "", "=" * 80]
    return "\n".join(lines)


def print_statistics(stats, verbose=True):
    if not verbose:
        return
    print(format_statistics_text(stats))


# ============================================================================
# FILE OPERATIONS  (unchanged from Stage8a)
# ============================================================================

def create_splits_csv(structure, assignment, output_path,
                      train_ratio=0.80, val_ratio=0.10, test_ratio=0.10,
                      objective=0, seed=42, verbose=True):
    if verbose:
        print(f"📄 Writing splits CSV: {output_path}")

    tp  = int(round(train_ratio * 100))
    vp  = int(round(val_ratio   * 100))
    tep = int(round(test_ratio  * 100))

    rows = []
    for cls in sorted(structure):
        for src in sorted(structure[cls]):
            sp = assignment[cls][src]
            for filename in sorted(structure[cls][src]):
                stem    = Path(filename).stem
                file_id = "XC" + stem[2:] if stem.lower().startswith("xc") else stem
                rows.append((file_id, sp))

    with open(output_path, 'w') as f:
        f.write(f"# split_ratio={tp}:{vp}:{tep} seed={seed} objective={objective:.4f} "
                f"solver=mip_cbc_dual_obj\n")
        f.write("file_id,split\n")
        for file_id, sp in rows:
            f.write(f"{file_id},{sp}\n")

    if verbose:
        counts = {'train': 0, 'val': 0, 'test': 0}
        for _, sp in rows:
            counts[sp] += 1
        print(f"   ✓ Wrote {len(rows)} rows "
              f"(train={counts['train']}, val={counts['val']}, test={counts['test']})")
        print()

    return output_path


# ============================================================================
# VISUALIZATION  (unchanged from Stage8a)
# ============================================================================

def plot_split_distribution(stats, output_path):
    if not HAS_MATPLOTLIB:
        return

    classes = [k for k in sorted(stats) if not k.startswith('_')]
    tr = [stats[c]['train_samples'] for c in classes]
    va = [stats[c]['val_samples']   for c in classes]
    te = [stats[c]['test_samples']  for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = range(len(classes))

    ax1.bar(x, tr, label='Train', color='#3498db')
    ax1.bar(x, va, bottom=tr,                              label='Val',  color='#2ecc71')
    ax1.bar(x, te, bottom=[a+b for a,b in zip(tr, va)],   label='Test', color='#e74c3c')
    ax1.set_xticks(x); ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.set_ylabel('Samples'); ax1.set_title('Sample Distribution (MIP dual-obj)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    rtrain = [a/(a+b+c) for a,b,c in zip(tr, va, te)]
    rval   = [b/(a+b+c) for a,b,c in zip(tr, va, te)]
    rtest  = [c/(a+b+c) for a,b,c in zip(tr, va, te)]
    ax2.bar(x, rtrain, label='Train', color='#3498db')
    ax2.bar(x, rval,   bottom=rtrain,                         label='Val',  color='#2ecc71')
    ax2.bar(x, rtest,  bottom=[a+b for a,b in zip(rtrain, rval)], label='Test', color='#e74c3c')
    cfg = stats['_config']
    ax2.axhline(cfg['train_ratio'],                     color='#3498db', linestyle='--', alpha=0.5)
    ax2.axhline(cfg['train_ratio']+cfg['val_ratio'],    color='#2ecc71', linestyle='--', alpha=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_ylabel('Ratio'); ax2.set_title('Split Ratios per Class')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim([0, 1])

    plt.tight_layout()
    out = output_path / 'split_distribution_mip_dualobj.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved plot: {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Split dataset using MIP with dual clip+source-balance objective',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Stage8a1_splitter_mip.py --dataset /path/to/mygardenbird16khz
  python Stage8a1_splitter_mip.py --dataset /path/to/mygardenbird16khz --source-weight 0.002
        """
    )

    parser.add_argument('--dataset', type=str, default=str(MYGARDENBIRD_16K),
                        help=f'Path to dataset directory. Default: {MYGARDENBIRD_16K}')
    parser.add_argument('--dataset-label', type=str, default=None,
                        help='Short label for the dataset variant (e.g. "16khz", "44khz"). '
                             'Auto-derived from --dataset path when not given.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: auto-named in metadata directory)')
    parser.add_argument('--train_ratio', type=float, default=0.80,
                        help='Target train ratio (default: 0.80)')
    parser.add_argument('--val_ratio',   type=float, default=0.10,
                        help='Target validation ratio (default: 0.10)')
    parser.add_argument('--test_ratio',  type=float, default=0.10,
                        help='Target test ratio (default: 0.10)')
    parser.add_argument('--source-weight', type=float, default=DEFAULT_SOURCE_WEIGHT,
                        help=f'Weight ε for the secondary source-balance term '
                             f'(default: {DEFAULT_SOURCE_WEIGHT}). '
                             'Must satisfy ε × max_sources_per_class < 1.0 to preserve '
                             'clip balance as the primary objective.')
    parser.add_argument('--plot',       action='store_true', help='Generate visualisation plots')
    parser.add_argument('--time-limit', type=int, default=MIP_TIME_LIMIT,
                        help=f'MIP solver time limit in seconds (default: {MIP_TIME_LIMIT})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed recorded in CSV header (default: 42)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')

    args = parser.parse_args()
    verbose = not args.quiet

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("ERROR: Split ratios must sum to 1.0")
        return 1

    # Auto-name output CSV
    _label = args.dataset_label or ('44khz' if '44' in str(args.dataset) else '16khz')
    _t  = int(round(args.train_ratio * 100))
    _v  = int(round(args.val_ratio   * 100))
    _te = int(round(args.test_ratio  * 100))
    _auto_csv = f"splits_mip2_{_t}_{_v}_{_te}.csv"

    if args.output is None:
        meta_dir = METADATA_44K if '44' in str(args.dataset) else METADATA_16K
        args.output = str(meta_dir / _auto_csv)

    if verbose:
        print("\n" + "=" * 80)
        print("MyGardenBird Splitter — MIP dual-objective (clip + source balance)")
        print("=" * 80 + "\n")

    if not HAS_PULP:
        print("ERROR: PuLP not installed.  pip install pulp")
        return 1

    if verbose:
        print(f"📂 Loading dataset from: {args.dataset}")

    structure = load_dataset_structure(args.dataset)
    if not structure:
        print(f"ERROR: No data found in {args.dataset}")
        return 1

    num_classes   = len(structure)
    total_sources = sum(len(s) for s in structure.values())
    total_files   = sum(len(f) for s in structure.values() for f in s.values())

    if verbose:
        print(f"   ✓ {num_classes} classes, {total_sources} sources, {total_files} clips\n")

    # Validate source_weight safety bound
    max_src = max(len(s) for s in structure.values())
    if args.source_weight * max_src >= 1.0:
        print(f"WARNING: source-weight {args.source_weight} × {max_src} sources = "
              f"{args.source_weight * max_src:.3f} ≥ 1.0")
        print("  This may allow source-balance to override clip balance.")
        print("  Consider reducing --source-weight below "
              f"{0.99/max_src:.4f} for this dataset.")

    assignment, stats, objective = optimize_split_mip(
        structure,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        time_limit=args.time_limit,
        source_weight=args.source_weight,
        verbose=verbose,
    )

    print_statistics(stats, verbose)

    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    create_splits_csv(structure, assignment, csv_path,
                      train_ratio=args.train_ratio,
                      val_ratio=args.val_ratio,
                      test_ratio=args.test_ratio,
                      objective=objective,
                      seed=args.seed,
                      verbose=verbose)

    stats_dir = csv_path.parent
    with open(stats_dir / 'split_stats_srcbal.json', 'w') as f:
        json.dump(stats, f, indent=2)
    with open(stats_dir / 'split_stats_srcbal.txt', 'w') as f:
        f.write(format_statistics_text(stats))

    if verbose:
        print(f"💾 Statistics saved to {stats_dir}")
        print()

    if args.plot and HAS_MATPLOTLIB:
        plot_split_distribution(stats, stats_dir)
        print()

    if verbose:
        print("=" * 80)
        print("✅ Done — MIP dual-objective split complete!")
        print(f"   CSV: {csv_path}")
        print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
