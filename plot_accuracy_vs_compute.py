#!/usr/bin/env python3
"""
plot_accuracy_vs_compute.py

Scatter plot: mean test accuracy vs MFLOPs (log scale) for three CNN architectures.
Data: 16 kHz, Mel features, SpecAugment, ImageNet init, 80:10:10 split, 3 seeds.
Output: fig/accuracy_vs_compute.pdf + fig/accuracy_vs_compute.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR = Path("/Users/mun3im/Dropbox/Paper Scientific Data/fig")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
# mean ± sd from Table 5 (16 kHz, SpecAugment, 3 seeds)
models = [
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet-50",
]
means  = np.array([94.61, 96.94, 95.06])
stds   = np.array([ 0.51,  1.11,  0.67])
mflops = np.array([  56,    390,  4100])   # from keras/torchinfo @ 224×224 input

colours = ["#E69F00", "#56B4E9", "#CC79A7"]   # Okabe-Ito (yellow, blue, pink)
markers = ["o", "s", "^"]
offsets_x = [0, 0, 0]           # label nudge (log units)
offsets_y = [0.35, -0.55, 0.35] # label nudge (accuracy units)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

for i, (model, mean, std, mf, col, mk) in enumerate(
        zip(models, means, stds, mflops, colours, markers)):
    ax.errorbar(
        mf, mean,
        yerr=std,
        fmt=mk,
        color=col,
        markersize=9,
        capsize=4,
        elinewidth=1.2,
        capthick=1.2,
        markeredgewidth=0.8,
        markeredgecolor="white",
        zorder=3,
        label=model,
    )
    ax.annotate(
        model,
        xy=(mf, mean),
        xytext=(mf * (1.18 if i != 1 else 0.52),
                mean + offsets_y[i]),
        fontsize=8,
        fontfamily="serif",
        ha="left" if i != 1 else "right",
        va="center",
        color="#333333",
    )

ax.set_xscale("log")
ax.set_xlabel("Computational cost (MFLOPs, log scale)", fontsize=9,
              fontfamily="serif")
ax.set_ylabel("Mean test accuracy (%)", fontsize=9, fontfamily="serif")
ax.set_xlim(20, 12000)
ax.set_ylim(90, 97)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))
ax.grid(axis="both", linewidth=0.3, alpha=0.45, which="major")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily("serif")

ax.set_title(
    "Accuracy vs. computational cost\n"
    "(16\u202fkHz, Mel, SpecAugment, mean\u202f\u00b1\u202fs.d., 3 seeds)",
    fontsize=9, fontfamily="serif", pad=8,
)

plt.tight_layout()
for ext in ("pdf", "png"):
    out = OUT_DIR / f"accuracy_vs_compute.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()
