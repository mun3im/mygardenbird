# Results: 16 kHz MyGardenBird-Plus (14 Species, 80:10:10 Split)

Benchmark of 3 pretrained CNN architectures with 3 augmentation strategies for
14-class bird species classification on the **MyGardenBird-Plus** dataset. Each
configuration was trained with 3 random seeds (42, 100, 786) to assess stability.
All 27 experiments completed.

## Experimental Setup

- **Dataset**: MyGardenBird-Plus (8,400 clips, 600 per species, 14 species)
- **Dataset split**: 80% train / 10% val / 10% test (source-level split, no leakage)
- **Classes**: 14 species (Asian Koel, Collared Kingfisher, Common Iora, Common Myna, Common Tailorbird, Coppersmith Barbet, Large-tailed Nightjar, Olive-backed Sunbird, Pied Fantail, Spotted Dove, White-breasted Waterhen, White-throated Kingfisher, Yellow-vented Bulbul, Zebra Dove)
- **Test set**: 840 samples (60 per class)
- **Sample rate**: 16 kHz
- **Features**: Mel spectrograms (224 bins, N_FFT=2048, hop=214 → 224×224)
- **Framework**: TensorFlow 2.15 (Keras)
- **Hardware**: Linux x86_64, NVIDIA GTX 1080 Ti
- **Training**: AdamW, batch size 32, max 50 epochs, early stopping, ReduceLROnPlateau
- **Initialization**: ImageNet pretrained weights

> MyGardenBird-Plus extends the 12-species core dataset with Common Myna and
> Zebra Dove. See [`../mygardenbirdplus16khz/`](../mygardenbirdplus16khz/README.md)
> for dataset construction. Core 12-species results are in
> [`../docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md).

## Test Accuracy by Model and Augmentation

### No Augmentation (Complete — 9/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| EfficientNet-B0 | 95.00% | 94.88% | 95.71% | **95.20%** | 0.37% |
| ResNet-50 | 92.14% | 93.21% | 92.86% | **92.74%** | 0.45% |
| MobileNetV3-Small | 88.57% | 88.81% | 89.17% | **88.85%** | 0.24% |

### SpecAugment (Complete — 9/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| EfficientNet-B0 | 95.24% | 94.88% | 95.60% | **95.24%** | 0.29% |
| ResNet-50 | 93.57% | 93.57% | 92.86% | **93.33%** | 0.34% |
| MobileNetV3-Small | 91.07% | 91.31% | 90.48% | **90.95%** | 0.35% |

### Mixup α=0.2 (Complete — 9/9 experiments)

| Model | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|
| EfficientNet-B0 | 95.60% | 95.71% | 96.43% | **95.91%** | 0.37% |
| ResNet-50 | 93.81% | 93.45% | 94.40% | **93.89%** | 0.39% |
| MobileNetV3-Small | 92.26% | 90.83% | 90.95% | **91.35%** | 0.65% |

## Mean Test Accuracy: Model vs Augmentation

| Model | No Augmentation | SpecAugment | Mixup α=0.2 |
|---|---|---|---|
| EfficientNet-B0 | 95.20% ± 0.37% | 95.24% ± 0.29% | **95.91% ± 0.37%** |
| ResNet-50 | 92.74% ± 0.45% | 93.33% ± 0.34% | 93.89% ± 0.39% |
| MobileNetV3-Small | 88.85% ± 0.24% | 90.95% ± 0.35% | 91.35% ± 0.65% |

## Rankings

### By Model (Mixup α=0.2, 3 seeds each)

| Rank | Model | Parameters | MFLOPs | Mean Accuracy | Std |
|---|---|---|---|---|---|
| 1 | EfficientNet-B0 | 5.3M | 390 | **95.91%** | 0.37% |
| 2 | ResNet-50 | 25.6M | 4,100 | 93.89% | 0.39% |
| 3 | MobileNetV3-Small | 2.9M | 56 | 91.35% | 0.65% |

### By Augmentation Strategy (averaged across all models)

| Rank | Strategy | Mean Accuracy | Std |
|---|---|---|---|
| 1 | Mixup α=0.2 | **93.72%** | 1.86% |
| 2 | SpecAugment | 93.17% | 1.78% |
| 3 | No Augmentation | 92.26% | 2.62% |

## Experiment Status

- **Total expected**: 27 experiments (3 models × 3 augmentations × 3 seeds)
- **Completed**: 27 experiments (100%)
- **Failures**: 0
- Total training time: ≈ 42.4 GPU-hours (GTX 1080 Ti)

### Completed Experiments
- ✅ No Augmentation: 9/9 (100%)
- ✅ SpecAugment: 9/9 (100%)
- ✅ Mixup α=0.2: 9/9 (100%)

## Key Findings

1. **EfficientNet-B0 + Mixup α=0.2 is the best configuration**, achieving
   95.91% ± 0.37% mean test accuracy (best single run 96.43%, seed 786).

2. **Mixup is the strongest augmentation** across all models (+1.46 pp over no-aug
   on average), ahead of SpecAugment (+0.91 pp). The ordering mixup > specaug >
   no-aug holds for every architecture.

3. **Augmentation helps the smallest model most**: MobileNetV3-Small gains
   +2.50 pp from Mixup (88.85% → 91.35%), versus +0.71 pp for EfficientNet-B0,
   reflecting its lower capacity and greater sensitivity to overfitting.

4. **EfficientNet-B0 has the best accuracy-to-compute ratio**, reaching 95.91%
   with only 390 MFLOPs — 10.5× less compute than ResNet-50 (4,100 MFLOPs) for
   +2.0 pp higher accuracy.

5. **Low variance throughout** (std ≤ 0.65% for every configuration), indicating
   stable, reproducible training under source-level splitting.

6. **Adding two species costs little**: versus the 12-species core
   (EfficientNet-B0 + Mixup = 96.39%), the 14-species Plus set reaches 95.91% —
   a ≈0.5 pp dip despite two additional confusable garden species.

## Notes

- All experiments use **source-level splitting** (every clip from a Xeno-Canto
  recording stays in one split) to prevent data leakage.
- MobileNetV3-Small uses its architecture-specific hyperparameters (10-epoch
  warmup, all-layer fine-tuning, patience 15); see
  [`../docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md).
- Per-run artifacts (`results.json`, `classification_report.txt`,
  `confusion_matrix.pdf`, `training_curves.png`) are in each run subdirectory.
