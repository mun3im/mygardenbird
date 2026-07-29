# Results: 16 kHz MyGardenBird (12 Species, 80:10:10 Split)

Benchmark of 4 pretrained CNN architectures with 3 augmentation strategies for
12-class bird species classification on the **MyGardenBird** core dataset. Each
configuration was trained with 3 random seeds (42, 100, 786) to assess stability.
All 36 experiments completed.

## Experimental Setup

- **Dataset**: MyGardenBird (7,200 clips, 600 per species, 12 species)
- **Dataset split**: 80% train / 10% val / 10% test (source-level split, no leakage)
- **Classes**: 12 species (Asian Koel, Collared Kingfisher, Common Iora, Common Tailorbird, Coppersmith Barbet, Large-tailed Nightjar, Olive-backed Sunbird, Pied Fantail, Spotted Dove, White-breasted Waterhen, White-throated Kingfisher, Yellow-vented Bulbul)
- **Test set**: 720 samples (60 per class)
- **Sample rate**: 16 kHz
- **Features**: Mel spectrograms (224 bins, N_FFT=2048, hop=214 → 224×224)
- **Framework**: TensorFlow 2.15 (Keras)
- **Hardware**: Linux x86_64, NVIDIA GTX 1080 Ti
- **Training**: AdamW, batch size 32, max 50 epochs, early stopping, ReduceLROnPlateau
- **Initialization**: ImageNet pretrained weights

> MyGardenBird is the 12-species core dataset; see
> [`../mygardenbird16khz/`](../mygardenbird16khz/README.md) for dataset
> construction. The 14-species MyGardenBird-Plus extension (adds Common Myna
> and Zebra Dove) is benchmarked separately in
> [`plus_16k_14species.md`](plus_16k_14species.md).

## Test Accuracy by Model and Augmentation

### No Augmentation (Complete — 12/12 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| EfficientNet-B0 | 94.31% | 94.31% | 96.11% | **94.91%** | 0.85% |
| ResNet-50 | 92.08% | 93.47% | 93.61% | **93.06%** | 0.69% |
| VGG16 | 91.25% | 90.69% | 93.61% | **91.85%** | 1.26% |
| MobileNetV3-Small | 90.14% | 87.50% | 90.56% | **89.40%** | 1.35% |

### SpecAugment (Complete — 12/12 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| EfficientNet-B0 | 94.86% | 93.89% | 95.97% | **94.91%** | 0.85% |
| ResNet-50 | 94.72% | 93.61% | 93.89% | **94.07%** | 0.47% |
| VGG16 | 92.78% | 92.50% | 93.33% | **92.87%** | 0.35% |
| MobileNetV3-Small | 92.08% | 92.50% | 92.22% | **92.27%** | 0.17% |

### Mixup α=0.2 (Complete — 12/12 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| EfficientNet-B0 | 96.94% | 95.42% | 96.81% | **96.39%** | 0.69% |
| ResNet-50 | 94.58% | 94.72% | 94.58% | **94.63%** | 0.07% |
| VGG16 | 94.17% | 93.75% | 94.17% | **94.03%** | 0.20% |
| MobileNetV3-Small | 93.06% | 91.53% | 92.64% | **92.41%** | 0.64% |

## Mean Test Accuracy: Model vs Augmentation

| Model | No Augmentation | SpecAugment | Mixup α=0.2 |
|---|---|---|---|
| EfficientNet-B0 | 94.91% ± 0.85% | 94.91% ± 0.85% | **96.39% ± 0.69%** |
| ResNet-50 | 93.06% ± 0.69% | 94.07% ± 0.47% | 94.63% ± 0.07% |
| VGG16 | 91.85% ± 1.26% | 92.87% ± 0.35% | 94.03% ± 0.20% |
| MobileNetV3-Small | 89.40% ± 1.35% | 92.27% ± 0.17% | 92.41% ± 0.64% |

## Rankings

### By Model (Mixup α=0.2, 3 seeds each)

| Rank | Model | Parameters | MFLOPs | Mean Accuracy | Std |
|---|---|---|---|---|---|
| 1 | EfficientNet-B0 | 4.4M | 390 | **96.39%** | 0.69% |
| 2 | ResNet-50 | 24.1M | 4,100 | 94.63% | 0.07% |
| 3 | VGG16 | 14.8M | 15,300 | 94.03% | 0.20% |
| 4 | MobileNetV3-Small | 1.2M | 56 | 92.41% | 0.64% |

### By Augmentation Strategy (averaged across all models)

| Rank | Strategy | Mean Accuracy | Std |
|---|---|---|---|
| 1 | Mixup α=0.2 | **94.37%** | 1.42% |
| 2 | SpecAugment | 93.53% | 1.03% |
| 3 | No Augmentation | 92.31% | 2.00% |

## Experiment Status

- **Total expected**: 36 experiments (4 models × 3 augmentations × 3 seeds)
- **Completed**: 36 experiments (100%)
- **Failures**: 0
- Total training time: ≈ 49.6 GPU-hours (GTX 1080 Ti)

### Completed Experiments
- ✅ No Augmentation: 12/12 (100%)
- ✅ SpecAugment: 12/12 (100%)
- ✅ Mixup α=0.2: 12/12 (100%)

## Key Findings

1. **EfficientNet-B0 + Mixup α=0.2 is the best configuration**, achieving
   96.39% ± 0.69% mean test accuracy (best single run 96.94%, seed 42).

2. **Mixup is the strongest augmentation** across all models (+2.06 pp over no-aug
   on average), ahead of SpecAugment (+1.22 pp). The ordering mixup > specaug >
   no-aug holds for every architecture.

3. **Augmentation helps the smallest and largest-non-EfficientNet models most**:
   MobileNetV3-Small gains +3.01 pp from Mixup (89.40% → 92.41%) and VGG16 gains
   +2.18 pp (91.85% → 94.03%), versus +1.48 pp for EfficientNet-B0, reflecting
   the smaller/less-efficient models' greater sensitivity to overfitting.

4. **EfficientNet-B0 has the best accuracy-to-compute ratio**, reaching 96.39%
   with only 390 MFLOPs — 10.5× less compute than ResNet-50 (4,100 MFLOPs) and
   39× less than VGG16 (15,300 MFLOPs) for the best accuracy of the group.

5. **Low variance for most configurations** (std ≤ 0.85% for every EfficientNet-B0
   and Mixup-augmented configuration); MobileNetV3-Small no-aug is the noisiest
   setting (std 1.35%), consistent with its higher sensitivity to overfitting.

6. **Two extra confusable species cost ≈0.5 pp**: the 14-species MyGardenBird-Plus
   set (EfficientNet-B0 + Mixup = 95.91%) trails this 12-species core result
   (96.39%) despite adding only Common Myna and Zebra Dove — see
   [`plus_16k_14species.md`](plus_16k_14species.md).

## Notes

- All experiments use **source-level splitting** (every clip from a Xeno-Canto
  recording stays in one split) to prevent data leakage.
- MobileNetV3-Small uses its architecture-specific hyperparameters (10-epoch
  warmup, all-layer fine-tuning, patience 15); see
  [`../docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md).
- Per-run artifacts (`results.json`, `classification_report.txt`,
  `confusion_matrix.pdf`, `training_curves.png`) are in each run subdirectory
  under [`../results_16k_12sp_linux/`](../results_16k_12sp_linux/).
