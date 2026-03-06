# Results: 44.1 kHz MyGardenBird (80:10:10 Split)

Benchmark of 3 pretrained CNN architectures with 3 augmentation strategies for 10-class bird species classification. Each configuration was trained with 3 random seeds (42, 100, 786) to assess stability.

## Experimental Setup

- **Dataset**: MyGardenBird (6,000 clips, 600 per species)
- **Dataset split**: 80% train / 10% val / 10% test (source-level split)
- **Classes**: 10 species (Asian Koel, Collared Kingfisher, Common Iora, Common Tailorbird, Coppersmith Barbet, Large-tailed Nightjar, Olive-backed Sunbird, Pink-necked Green Pigeon, Spotted Dove, White-throated Kingfisher)
- **Test set**: 600 samples (60 per class)
- **Sample rate**: 44.1 kHz
- **Features**: Mel spectrograms (128 bins, N_FFT=2048, hop=512)
- **Input size**: 224×224
- **Framework**: PyTorch 2.x
- **Hardware**: Linux x86_64, GPU-accelerated
- **Training**: AdamW optimizer, batch size 32, max 50 epochs, early stopping (patience 10), ReduceLROnPlateau (patience 5)
- **Initialization**: ImageNet pretrained weights

## Test Accuracy by Model and Augmentation

### SpecAugment (Complete - 9/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| MobileNetV3-Small | 91.33% | 92.00% | 90.17% | **91.17%** | 0.94% |
| EfficientNet-B0 | 93.67% | 95.17% | 93.83% | **94.22%** | 0.84% |
| ResNet-50 | 92.83% | 92.00% | 91.83% | **92.22%** | 0.55% |

### Mixup α=0.2 (Complete - 9/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| MobileNetV3-Small | 91.17% | 90.17% | 92.17% | **91.17%** | 1.00% |
| EfficientNet-B0 | 93.50% | 94.17% | 93.50% | **93.72%** | 0.39% |
| ResNet-50 | 93.33% | 91.83% | 92.83% | **92.67%** | 0.76% |

### No Augmentation (Incomplete - 6/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| MobileNetV3-Small | — | — | — | — | — |
| EfficientNet-B0 | 94.00% | 95.17% | 92.67% | **93.94%** | 1.25% |
| ResNet-50 | 92.00% | 91.50% | 91.17% | **91.56%** | 0.42% |

## Mean Test Accuracy: Model vs Augmentation (Complete experiments only)

| Model | Mixup α=0.2 | SpecAugment |
|---|---|---|
| EfficientNet-B0 | 93.72% ± 0.39% | **94.22% ± 0.84%** |
| ResNet-50 | 92.67% ± 0.76% | 92.22% ± 0.55% |
| MobileNetV3-Small | 91.17% ± 1.00% | 91.17% ± 0.94% |

## Rankings

### By Model (SpecAugment only, 3 seeds each)

| Rank | Model | Parameters | MFLOPs | Mean Accuracy | Std |
|---|---|---|---|---|---|---|
| 1 | EfficientNet-B0 | 5.3M | 390 | **94.22%** | 0.84% |
| 2 | ResNet-50 | 25.6M | 4,100 | 92.22% | 0.55% |
| 3 | MobileNetV3-Small | 2.9M | 56 | 91.17% | 0.94% |

### By Augmentation Strategy (averaged across all models with complete data)

| Rank | Strategy | Mean Accuracy | Std |
|---|---|---|---|
| 1 | SpecAugment | **92.54%** | 1.59% |
| 2 | Mixup α=0.2 | 92.52% | 1.37% |
| 3 | No Augmentation | Incomplete | — |

## Experiment Status

- **Total expected**: 27 experiments (3 models × 3 augmentations × 3 seeds)
- **Completed**: 24 experiments (88.9%)
- **Pending**: 3 experiments (No Augmentation for MobileNetV3-Small)

### Completed Experiments
- ⚠️ No Augmentation: 6/9 (66.7%)
- ✅ SpecAugment: 9/9 (100%)
- ✅ Mixup α=0.2: 9/9 (100%)

## Key Findings

1. **EfficientNet-B0 + SpecAugment achieves the best performance**, with 94.22% ± 0.84% mean test accuracy.

2. **44.1 kHz results are 2-3% lower than 16 kHz** across all models, suggesting that higher sampling rates may not benefit this classification task (likely due to most bird vocalizations being concentrated below 8 kHz).

3. **SpecAugment and Mixup perform nearly identically** (92.54% vs 92.52%), with Mixup showing slightly lower variance (1.37% vs 1.59%).

4. **ResNet-50 shows the most stable training** with SpecAugment (0.55% std), despite not achieving the highest accuracy.

5. **MobileNetV3-Small performance is consistent** across both augmentation strategies (91.17% for both), suggesting it may have reached its capacity limit for this task.

6. **EfficientNet-B0 maintains the best accuracy-to-compute ratio** at 44.1 kHz, achieving 94.22% with only 390 MFLOPs.

## Comparison: 44.1 kHz vs 16 kHz (SpecAugment)

| Model | 44.1 kHz | 16 kHz | Δ |
|---|---|---|---|
| MobileNetV3-Small | 91.17% ± 0.94% | 94.61% ± 0.51% | -3.44% |
| EfficientNet-B0 | 94.22% ± 0.84% | 96.94% ± 1.11% | -2.72% |
| ResNet-50 | 92.22% ± 0.55% | 95.06% ± 0.67% | -2.84% |

**Recommendation**: Use 16 kHz sampling for this bird species classification task to achieve better accuracy with lower computational cost.

## Notes

- Results reported in the MyGardenBird paper use SpecAugment at 16 kHz (Table 5).
- 44.1 kHz experiments demonstrate that higher sampling rates are unnecessary for this task.
- All experiments use source-level splitting to prevent data leakage.
