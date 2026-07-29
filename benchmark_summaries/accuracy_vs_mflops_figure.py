#!/usr/bin/env python3
"""Regenerate accuracy_vs_mflops.svg from the benchmark summary tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model, MFLOPs, params_M, 12-species Mixup acc, 14-species Mixup acc)
DATA = [
    ("EfficientNet-B0", 390, 4.4, 96.39, 95.91),
    ("ResNet-50", 4100, 24.1, 94.63, 93.89),
    ("MobileNetV3-Small", 56, 1.2, 92.41, 91.35),
]
VGG16 = ("VGG16", 15300, 14.8, 94.03, None)  # 12-species only

COLORS = {
    "EfficientNet-B0": "#1f77b4",
    "ResNet-50": "#d62728",
    "MobileNetV3-Small": "#2ca02c",
    "VGG16": "#9467bd",
}


def main():
    all_points = DATA + [VGG16]
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, mflops, params, acc12, acc14 in DATA:
        c = COLORS[name]
        ax.scatter(mflops, acc12, marker="o", s=110, color=c, edgecolor="black",
                   zorder=3, label=f"{name} (12-sp)")
        ax.scatter(mflops, acc14, marker="^", s=110, color=c, edgecolor="black",
                   zorder=3, facecolor="white", linewidth=1.8, label=f"{name} (14-sp)")
        ax.plot([mflops, mflops], [acc12, acc14], color=c, linestyle="--",
                linewidth=1, alpha=0.6, zorder=1)

    name, mflops, params, acc12, _ = VGG16
    ax.scatter(mflops, acc12, marker="o", s=110, color=COLORS[name], edgecolor="black",
               zorder=3, label=f"{name} (12-sp)")

    ax.set_xscale("log")
    ax.set_xlabel("MFLOPs (log scale)", fontsize=11)
    ax.set_ylabel("Mean Test Accuracy (%), Mixup α=0.2, 3 seeds", fontsize=11)
    ax.set_title("Accuracy vs. Compute — 12-species (●) vs 14-species (▲) MyGardenBird",
                 fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    for name, mflops, params, acc12, acc14 in all_points:
        ax.annotate(name, (mflops, acc12), textcoords="offset points",
                    xytext=(8, -10), fontsize=8.5)

    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    ax.legend(h2, l2, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig("accuracy_vs_mflops.svg")


if __name__ == "__main__":
    main()
