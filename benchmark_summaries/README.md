# Benchmark Summaries

Pretrained-CNN benchmarks for the MyGardenBird 16 kHz datasets. Each summary
covers 3 seeds (42, 100, 786) × 3 augmentation strategies (No Augmentation,
SpecAugment, Mixup α=0.2) on ImageNet-pretrained backbones fine-tuned on mel
spectrograms.

| Summary | Dataset | Species | Models | Experiments |
|---|---|---|---|---|
| [`16k_12species.md`](16k_12species.md) | MyGardenBird (core) | 12 | EfficientNet-B0, ResNet-50, VGG16, MobileNetV3-Small | 36/36 complete |
| [`plus_16k_14species.md`](plus_16k_14species.md) | MyGardenBird-Plus | 14 | EfficientNet-B0, ResNet-50, MobileNetV3-Small | 27/27 complete |

> VGG16 was only benchmarked on the 12-species core set; the 14-species Plus
> sweep covers the other three architectures.

## Accuracy vs. Compute

Best-configuration (Mixup α=0.2) mean test accuracy vs. MFLOPs for both
datasets, log-scaled on compute:

![Accuracy vs MFLOPs](accuracy_vs_mflops.svg)

- ● = 12-species core, ▲ = 14-species Plus (dashed line connects the same
  architecture across datasets)
- **EfficientNet-B0** dominates the accuracy/compute trade-off on both
  datasets — highest accuracy at a fraction of ResNet-50/VGG16's MFLOPs.
- Adding 2 species (12→14) costs every shared architecture ≈0.5–1.1 pp,
  with the compute-accuracy ordering unchanged.
- **VGG16** sits far to the right (15,300 MFLOPs) yet still trails
  EfficientNet-B0 and ResNet-50 in accuracy, making it the least
  compute-efficient of the four.

| Model | MFLOPs | Params | 12-sp Acc (Mixup) | 14-sp Acc (Mixup) | Δ (12→14) |
|---|---|---|---|---|---|
| EfficientNet-B0 | 390 | 4.4M | **96.39%** | **95.91%** | −0.48 pp |
| ResNet-50 | 4,100 | 24.1M | 94.63% | 93.89% | −0.74 pp |
| VGG16 | 15,300 | 14.8M | 94.03% | — (not run) | — |
| MobileNetV3-Small | 56 | 1.2M | 92.41% | 91.35% | −1.06 pp |

## Regenerating the Plot

```bash
cd benchmark_summaries
python3 - <<'EOF'
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model, MFLOPs, params_M, 12-species Mixup acc, 14-species Mixup acc)
data = [
    ("EfficientNet-B0", 390,   4.4, 96.39, 95.91),
    ("ResNet-50",       4100, 24.1, 94.63, 93.89),
    ("MobileNetV3-Small", 56,  1.2, 92.41, 91.35),
]
vgg16 = ("VGG16", 15300, 14.8, 94.03, None)  # 12-species only
all_points = data + [vgg16]

fig, ax = plt.subplots(figsize=(7, 5.5))
colors = {"EfficientNet-B0": "#1f77b4", "ResNet-50": "#d62728",
          "MobileNetV3-Small": "#2ca02c", "VGG16": "#9467bd"}

for name, mflops, params, acc12, acc14 in data:
    c = colors[name]
    ax.scatter(mflops, acc12, marker="o", s=110, color=c, edgecolor="black", zorder=3,
               label=f"{name} (12-sp)")
    ax.scatter(mflops, acc14, marker="^", s=110, color=c, edgecolor="black", zorder=3,
               facecolor="white", linewidth=1.8, label=f"{name} (14-sp)")
    ax.plot([mflops, mflops], [acc12, acc14], color=c, linestyle="--", linewidth=1, alpha=0.6, zorder=1)

name, mflops, params, acc12, _ = vgg16
ax.scatter(mflops, acc12, marker="o", s=110, color=colors[name], edgecolor="black",
           zorder=3, label=f"{name} (12-sp)")

ax.set_xscale("log")
ax.set_xlabel("MFLOPs (log scale)", fontsize=11)
ax.set_ylabel("Mean Test Accuracy (%), Mixup α=0.2, 3 seeds", fontsize=11)
ax.set_title("Accuracy vs. Compute — 12-species (●) vs 14-species (▲) MyGardenBird", fontsize=12)
ax.grid(True, which="both", linestyle=":", alpha=0.5)
for name, mflops, params, acc12, acc14 in all_points:
    ax.annotate(name, (mflops, acc12), textcoords="offset points", xytext=(8, -10), fontsize=8.5)

handles, labels = ax.get_legend_handles_labels()
seen, h2, l2 = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); h2.append(h); l2.append(l)
ax.legend(h2, l2, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig("accuracy_vs_mflops.svg")
EOF
```

Source figures are regenerated from the per-run `results.json` files in
[`../results_16k_12sp_linux/`](../results_16k_12sp_linux/) and
[`../resultsplus_16k_linux/`](../resultsplus_16k_linux/).
