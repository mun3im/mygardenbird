import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Data: [total_mflops, accuracy_mean, accuracy_std, feature_type, model, is_best]
data = [
    # MFLOPs, Acc, Std, Feature, Model, Best?
    [62.7, 90.1, 0.8, 'STFT', 'MobileNetV3S', True],
    [69.1, 90.0, 0.9, 'Mel', 'MobileNetV3S', False],
    [71.2, 83.0, 1.0, 'MFCC', 'MobileNetV3S', False],

    [396.7, 91.0, 1.5, 'STFT', 'EfficientNetB0', False],
    [403.1, 93.4, 2.6, 'Mel', 'EfficientNetB0', True],
    [405.2, 89.4, 1.3, 'MFCC', 'EfficientNetB0', False],

    [4106.7, 91.0, 1.1, 'STFT', 'ResNet50', True],
    [4113.1, 88.8, 2.9, 'Mel', 'ResNet50', False],
    [4115.2, 86.0, 1.6, 'MFCC', 'ResNet50', False],

    [15306.7, 86.7, 1.8, 'STFT', 'VGG16', False],
    [15313.1, 88.2, 0.8, 'Mel', 'VGG16', True],
    [15315.2, 81.9, 3.4, 'MFCC', 'VGG16', False],
]

# Organize data by feature type for coloring
features = {'Mel': [], 'STFT': [], 'MFCC': []}
for row in data:
    mflops, acc, std, feat, model, best = row
    features[feat].append((mflops, acc, std, model, best))

fig, ax = plt.subplots(figsize=(12, 7.5))

# Color scheme and model markers
colors = {'Mel': '#2ca02c', 'STFT': '#1f77b4', 'MFCC': '#d62728'}
markers = {
    'MobileNetV3S': 'o',  # circle
    'EfficientNetB0': 's',  # square
    'ResNet50': '^',  # triangle
    'VGG16': 'D'  # diamond
}

# Plot all points grouped by feature type
for feat in ['Mel', 'STFT', 'MFCC']:
    for mflops, acc, std, model, best in features[feat]:
        # Plot base point with error bars
        ax.errorbar(mflops, acc, yerr=std,
                    fmt=markers[model], color=colors[feat],
                    markersize=11, capsize=5, elinewidth=2,
                    alpha=0.85, zorder=3, linewidth=1.5)

        # Highlight best-per-model with gold star overlay
        if best:
            ax.plot(mflops, acc, marker='*', color='gold',
                    markersize=22, markeredgecolor='black',
                    markeredgewidth=1.8, zorder=6)

# Pareto frontier (non-dominated points: best accuracy at each compute tier)
pareto_points = [(62.7, 90.1), (403.1, 93.4)]
pareto_x, pareto_y = zip(*sorted(pareto_points, key=lambda x: x[0]))
ax.plot(pareto_x, pareto_y, 'k--', linewidth=2.5, alpha=0.7,
        label='Pareto frontier', zorder=2)

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Total Compute Complexity (MFLOPs)\n[TFR + CNN Inference]',
              fontsize=14, fontweight='bold', labelpad=12)
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Accuracy vs. Compute Complexity\n(3s @ 16kHz → 224×224, FP32)',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.35, which='both', linestyle='--', linewidth=0.8)
ax.set_ylim(78, 96)
ax.set_xlim(50, 20000)

# Custom legend with feature types and model markers
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Mel'],
           markersize=13, label='Mel spectrogram'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['STFT'],
           markersize=13, label='STFT'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['MFCC'],
           markersize=13, label='MFCC'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
           markersize=18, markeredgecolor='black', markeredgewidth=1.5,
           label='Best per model'),
    Line2D([0], [0], color='k', linestyle='--', linewidth=2.5, label='Pareto frontier')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
          framealpha=0.95, shadow=True, ncol=1)

# Annotate key Pareto points
ax.annotate('EfficientNetB0\n+ Mel ★',
            xy=(403.1, 93.4), xytext=(250, 95.5),
            fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.8, alpha=0.8))

ax.annotate('MobileNetV3S\n+ STFT ★',
            xy=(62.7, 90.1), xytext=(120, 87),
            fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.8, alpha=0.8))

# Add vertical lines separating compute tiers
ax.axvline(x=100, color='gray', linestyle=':', alpha=0.4, linewidth=1)
ax.axvline(x=1000, color='gray', linestyle=':', alpha=0.4, linewidth=1)
ax.text(75, 79, 'Ultra-low\n(<100 MFLOPs)', fontsize=9, ha='center', style='italic', alpha=0.7)
ax.text(400, 79, 'Low\n(100–1k MFLOPs)', fontsize=9, ha='center', style='italic', alpha=0.7)
ax.text(3000, 79, 'Medium\n(1k–10k MFLOPs)', fontsize=9, ha='center', style='italic', alpha=0.7)
ax.text(12000, 79, 'High\n(>10k MFLOPs)', fontsize=9, ha='center', style='italic', alpha=0.7)

plt.tight_layout()
plt.savefig('cnn-vs-feature.svg', bbox_inches='tight', facecolor='white')
plt.savefig('cnn-vs-feature.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Plot saved as 'cnn-vs-feature.svg' and 'cnn-vs-feature.png'")
plt.show()