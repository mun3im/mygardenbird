#!/usr/bin/env python3
"""Accuracy vs params for the completed v2.0 full sweep (36/36 runs:
4 models x 3 augmentations x 3 seeds).

Reads classification_report.txt + hyperparameters.txt from each
results_v2_16k_<model>_<aug>_seed<N>_linux/ directory (Stage 9 models),
and classification_report_int8.txt from each
results_linux/v1_dscnn_se_res_att_wide_mels80_drop05_rand<seed>_warm50[_<aug>]_split80:10:10_linux/
directory (MynaNet v1, INT8 TFLite).
"""
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = {
    "mobilenetv3s": "#2ca02c",
    "efficientnetb0": "#1f77b4",
    "resnet50": "#d62728",
    "mynanet": "#9467bd",
}
MODEL_LABELS = {
    "mobilenetv3s": "MobileNetV3-Small",
    "efficientnetb0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "mynanet": "MynaNet v1 (INT8)",
}
AUG_MARKERS = {
    "noaug": "o",
    "specaug": "s",
    "mixup0.2": "^",
}
AUG_LABELS = {
    "noaug": "No Augmentation",
    "specaug": "SpecAugment",
    "mixup0.2": "Mixup α=0.2",
}

MYNANET_PARAMS = 328_800


def collect():
    rows = []
    for d in glob.glob("../results_v2_16k_*_linux"):
        m = re.match(
            r"\.\./results_v2_16k_(mobilenetv3s|efficientnetb0|resnet50)_"
            r"(noaug|specaug|mixup0\.2)_seed(\d+)_linux",
            d,
        )
        if not m:
            continue
        model, aug, seed = m.groups()
        subdirs = [p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)]
        if not subdirs:
            continue
        cr = os.path.join(subdirs[0], "classification_report.txt")
        hp = os.path.join(subdirs[0], "hyperparameters.txt")
        if not (os.path.isfile(cr) and os.path.isfile(hp)):
            continue
        acc = params = None
        with open(cr) as f:
            for line in f:
                if line.strip().startswith("accuracy"):
                    acc = float(line.split()[-2])
        with open(hp) as f:
            for line in f:
                if "Total parameters" in line:
                    params = int(line.split(":")[1].strip().replace(",", ""))
        if acc is not None and params is not None:
            rows.append((model, aug, int(seed), params, acc))

    for d in glob.glob("../results_linux/v1_dscnn_se_res_att_wide_mels80_drop05_rand*"):
        m = re.match(
            r"\.\./results_linux/v1_dscnn_se_res_att_wide_mels80_drop05_rand(\d+)"
            r"_warm50(_specaugment|_mixup0\.2)?_split",
            d,
        )
        if not m:
            continue
        seed, augsuf = m.groups()
        aug = {"_specaugment": "specaug", "_mixup0.2": "mixup0.2", None: "noaug"}[augsuf]
        cr = os.path.join(d, "classification_report_int8.txt")
        if not os.path.isfile(cr):
            continue
        acc = None
        with open(cr) as f:
            for line in f:
                if line.strip().startswith("accuracy"):
                    acc = float(line.split()[-2])
        if acc is not None:
            rows.append(("mynanet", aug, int(seed), MYNANET_PARAMS, acc))

    return rows


def main():
    rows = collect()
    if not rows:
        print("No completed runs found.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    by_model_aug = defaultdict(list)
    for model, aug, seed, params, acc in rows:
        by_model_aug[(model, aug)].append((params, acc, seed))

    seen_labels = set()
    for (model, aug), pts in sorted(by_model_aug.items()):
        color = MODEL_COLORS.get(model, "gray")
        marker = AUG_MARKERS.get(aug, "x")
        params_vals = [p for p, _, _ in pts]
        accs = [a for _, a, _ in pts]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_params = np.mean(params_vals)

        for p, a, seed in pts:
            ax.scatter(p / 1e6, a * 100, marker=marker, s=40, color=color,
                       alpha=0.35, zorder=2)

        label = f"{MODEL_LABELS.get(model, model)} — {AUG_LABELS.get(aug, aug)}"
        ax.errorbar(mean_params / 1e6, mean_acc * 100, yerr=std_acc * 100,
                     marker=marker, markersize=11, color=color,
                     markeredgecolor="black", capsize=4, linestyle="none",
                     zorder=3, label=label if label not in seen_labels else None)
        seen_labels.add(label)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    n_runs = len(rows)
    ax.set_title(
        f"Accuracy vs. Model Size — MyGardenBird v2.0 full sweep\n"
        f"({n_runs}/36 runs; mean ± std over 3 seeds)",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="center right", ncol=1)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.7, xmax * 1.6)

    fig.tight_layout()
    fig.savefig("accuracy_vs_params_full_sweep.pdf")
    fig.savefig("accuracy_vs_params_full_sweep.png", dpi=150)
    fig.savefig("../figures/accuracy_vs_params_full_sweep.pdf")
    print(f"Plotted {n_runs} completed runs.")


if __name__ == "__main__":
    main()
