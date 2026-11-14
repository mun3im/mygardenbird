#!/usr/bin/env python3
"""Stage 5b: pick clips to drop, from repeated Stage 5a cross-validation runs.

Stage 5a runs one 5-fold cross-validation and scores every clip by how badly
the model misreads it. A single run is noisy: which clips look worst depends on
the fold partition and on weight initialisation, so a clip can score badly once
and cleanly the next time. Repeating the run and keeping the clips that fail
*repeatedly* separates genuinely ambiguous audio from one unlucky draw.

Rank by, in order:
  1. `times_wrong`  -- how many repeats misclassified the clip outright.
  2. `mean_confusion` -- average confusion score (1 - confidence in the true
     class), which breaks ties and orders clips never misclassified at all.

The clips at the top are the most consistently confusing, and the ones to drop
when a class sits above the target count.

Only classes over the target are considered; the rest are already short.
Usage:
    python Stage5b_consolidate_confusion.py run1/ run2/ run3/ --target 600
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path


def load_run(run_dir):
    """Read one Stage 5a output directory -> {class: {clip: row}}."""
    out = {}
    for path in sorted(Path(run_dir).glob("*_confusion_scores.csv")):
        cls = path.name[: -len("_confusion_scores.csv")]
        rows = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = r.get("filename") or r.get("clip") or r.get("file")
                if key:
                    rows[key] = r
        if rows:
            out[cls] = rows
    return out


def _is_wrong(row):
    t = (row.get("true_label") or "").strip()
    p = (row.get("pred_label") or "").strip()
    return bool(t) and bool(p) and t != p


def _score(row):
    try:
        return float(row.get("confusion_score", "nan"))
    except (TypeError, ValueError):
        return float("nan")


def consolidate(run_dirs, target, output_dir, apply_list=True):
    runs = [load_run(d) for d in run_dirs]
    present = [d for d, r in zip(run_dirs, runs) if r]
    if not present:
        raise SystemExit("No *_confusion_scores.csv found in the given directories.")
    if len(present) < len(run_dirs):
        missing = set(map(str, run_dirs)) - set(map(str, present))
        print(f"Warning: no results in {sorted(missing)} -- using {len(present)} run(s).")

    classes = sorted({c for r in runs for c in r})
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for cls in classes:
        per_clip = defaultdict(list)
        for r in runs:
            for clip, row in r.get(cls, {}).items():
                per_clip[clip].append(row)

        n_runs = max(len(v) for v in per_clip.values()) if per_clip else 0
        total = len(per_clip)
        if total <= target:
            summary.append((cls, total, 0, 0.0))
            continue

        rows = []
        for clip, got in sorted(per_clip.items()):
            wrong = sum(1 for g in got if _is_wrong(g))
            scores = [s for s in (_score(g) for g in got) if s == s]
            mean_s = statistics.fmean(scores) if scores else 0.0
            # A clip missing from a run was never scored there; record it so a
            # partial run cannot look like agreement.
            rows.append({
                "clip": clip,
                "source_id": clip.split("_")[0],
                "times_wrong": wrong,
                "runs_seen": len(got),
                "mean_confusion": round(mean_s, 6),
                "max_confusion": round(max(scores), 6) if scores else "",
                "true_label": (got[0].get("true_label") or "").strip(),
            })

        rows.sort(key=lambda r: (-r["times_wrong"], -r["mean_confusion"], r["clip"]))
        drop_n = total - target
        for i, r in enumerate(rows):
            r["rank"] = i + 1
            r["action"] = "DROP" if i < drop_n else "keep"

        path = Path(output_dir) / f"{cls}_consolidated.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "rank", "action", "clip", "source_id", "times_wrong",
                "runs_seen", "mean_confusion", "max_confusion", "true_label"])
            w.writeheader()
            w.writerows(rows)

        dropped = rows[:drop_n]
        consistent = sum(1 for r in dropped if r["times_wrong"] == n_runs)
        summary.append((cls, total, drop_n,
                        100.0 * consistent / drop_n if drop_n else 0.0))
        print(f"{cls}: {total} clips, drop {drop_n} -> {target}  "
              f"({consistent}/{drop_n} wrong in all {n_runs} runs)  -> {path.name}")

    if apply_list:
        path = Path(output_dir) / "clips_to_drop.txt"
        with open(path, "w") as f:
            for cls, total, drop_n, _ in summary:
                if drop_n <= 0:
                    continue
                src = Path(output_dir) / f"{cls}_consolidated.csv"
                with open(src, newline="") as g:
                    for r in csv.DictReader(g):
                        if r["action"] == "DROP":
                            f.write(f"{cls}/{r['clip']}\n")
        print(f"\nDrop list: {path}")

    print(f"\n{'class':30s}{'clips':>7}{'drop':>6}{'consistent%':>13}")
    for cls, total, drop_n, pct in summary:
        print(f"{cls:30s}{total:7d}{drop_n:6d}{pct:12.0f}%")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+",
                   help="Stage 5a output directories, one per repeat")
    p.add_argument("--target", type=int, default=600,
                   help="Clips to keep per class (default: 600)")
    p.add_argument("--output_dir", default="./confusion_consolidated")
    p.add_argument("--no-drop-list", action="store_true",
                   help="Skip writing clips_to_drop.txt")
    a = p.parse_args()
    consolidate(a.run_dirs, a.target, a.output_dir, apply_list=not a.no_drop_list)


if __name__ == "__main__":
    main()
