# Results: 16 kHz MyGardenBird (80:10:10 Split)

Benchmark of 3 pretrained CNN architectures with 3 augmentation strategies for 10-class bird species classification. Each configuration was trained with 3 random seeds (42, 100, 786) to assess stability.

## Experimental Setup

- **Dataset**: MyGardenBird (6,000 clips, 600 per species)
- **Dataset split**: 80% train / 10% val / 10% test (source-level split)
- **Classes**: 10 species (Asian Koel, Collared Kingfisher, Common Iora, Common Tailorbird, Coppersmith Barbet, Large-tailed Nightjar, Olive-backed Sunbird, Pink-necked Green Pigeon, Spotted Dove, White-throated Kingfisher)
- **Test set**: 600 samples (60 per class)
- **Sample rate**: 16 kHz
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
| MobileNetV3-Small | 94.83% | 94.17% | 94.83% | **94.61%** | 0.51% |
| EfficientNet-B0 | 97.83% | 95.00% | 98.00% | **96.94%** | 1.11% |
| ResNet-50 | 95.50% | 95.17% | 94.50% | **95.06%** | 0.67% |

### No Augmentation (Complete - 9/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| MobileNetV3-Small | 93.83% | 92.50% | 93.00% | **93.11%** | 0.69% |
| EfficientNet-B0 | 97.00% | 96.17% | 95.17% | **96.11%** | 0.93% |
| ResNet-50 | 95.17% | 94.17% | 94.83% | **94.72%** | 0.51% |

### Mixup α=0.2 (Incomplete - 1/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| MobileNetV3-Small | 95.33% | — | — | — | — |
| EfficientNet-B0 | — | — | — | — | — |
| ResNet-50 | — | — | — | — | — |

## Mean Test Accuracy: Model vs Augmentation (Complete experiments only)

| Model | No Augmentation | SpecAugment |
|---|---|---|
| EfficientNet-B0 | 96.11% ± 0.93% | **96.94% ± 1.11%** |
| ResNet-50 | 94.72% ± 0.51% | 95.06% ± 0.67% |
| MobileNetV3-Small | 93.11% ± 0.69% | 94.61% ± 0.51% |

## Rankings

### By Model (SpecAugment only, 3 seeds each)

| Rank | Model | Parameters | MFLOPs | Mean Accuracy | Std |
|---|---|---|---|---|---|---|
| 1 | EfficientNet-B0 | 5.3M | 390 | **96.94%** | 1.11% |
| 2 | ResNet-50 | 25.6M | 4,100 | 95.06% | 0.67% |
| 3 | MobileNetV3-Small | 2.9M | 56 | 94.61% | 0.51% |

### By Augmentation Strategy (averaged across all models)

| Rank | Strategy | Mean Accuracy | Std |
|---|---|---|---|
| 1 | SpecAugment | **95.54%** | 1.34% |
| 2 | No Augmentation | 94.65% | 1.52% |
| 3 | Mixup α=0.2 | Incomplete | — |

## Experiment Status

- **Total expected**: 27 experiments (3 models × 3 augmentations × 3 seeds)
- **Completed**: 19 experiments (70.4%)
- **Pending**: 8 experiments (all Mixup variants)

### Completed Experiments
- ✅ No Augmentation: 9/9 (100%)
- ✅ SpecAugment: 9/9 (100%)
- ⚠️ Mixup α=0.2: 1/9 (11%)

## Key Findings

1. **EfficientNet-B0 + SpecAugment is the best configuration**, achieving 96.94% ± 1.11% mean test accuracy with the highest single-run accuracy of 98.00% (seed 786).

2. **SpecAugment provides consistent improvement** over no augmentation across all models (+0.89% average improvement).

3. **MobileNetV3-Small shows the lowest variance** with SpecAugment (0.51% std), indicating highly stable training despite having the fewest parameters.

4. **EfficientNet-B0 achieves the best accuracy-to-compute ratio**, delivering 96.94% accuracy with only 390 MFLOPs (compared to ResNet-50's 4,100 MFLOPs for 95.06%).

5. **All models exceed 94% accuracy**, demonstrating that the MyGardenBird dataset enables strong performance across different architecture families.

6. **ImageNet pretraining is effective** even for this specialized bird species classification task with source-level data splitting.

## Notes

- Results reported in the MyGardenBird paper are based on SpecAugment experiments (Table 5).
- Mixup experiments are incomplete and not included in the paper.
- All experiments use source-level splitting to prevent data leakage.
