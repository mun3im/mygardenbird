#!/usr/bin/env python3
"""Generate the v2.0 (15-species) accuracy-vs-params comparison figure.

Params and accuracies are measured directly from this run's training
artifacts (hyperparameters.txt / classification_report.txt per model,
model_summary.txt for MynaNet), not from literature/architecture tables.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model, total_params, test_accuracy_pct, marker, color)
DATA = [
    ("MynaNet v1 (INT8)",      328_793,    90.76, "D", "#ff7f0e"),
    ("MobileNetV3-Small",    1_242_239,    88.80, "o", "#2ca02c"),
    ("EfficientNet-B0",      4_381_362,    92.15, "s", "#1f77b4"),
    ("ResNet-50",           24_116_111,    89.38, "^", "#d62728"),
]


def main():
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, params, acc, marker, color in DATA:
        ax.scatter(params / 1e6, acc, marker=marker, s=130, color=color,
                   edgecolor="black", zorder=3, label=name)
        ax.annotate(name, (params / 1e6, acc), textcoords="offset points",
                    xytext=(8, -4), fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%), Mixup α=0.2, seed=42", fontsize=11)
    ax.set_title("Accuracy vs. Model Size — MyGardenBird v2.0 (15 species, 16 kHz)",
                 fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig("accuracy_vs_params_v2.pdf")
    fig.savefig("../figures/accuracy_vs_params_v2.pdf")


if __name__ == "__main__":
    main()
