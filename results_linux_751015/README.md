# Results: 75/10/15 Split (Train/Val/Test)

Benchmark of 4 pretrained CNN architectures across 3 audio feature representations for 10-class seabird species classification. Each configuration was trained 3 times with different random seeds (42, 100, 786) to assess stability.

## Experimental Setup

- **Dataset split**: 75% train / 10% val / 15% test
- **Classes**: 10 species (Asian Koel, Collared Kingfisher, Common Iora, Common Myna, Common Tailorbird, Large-tailed Nightjar, Olive-backed Sunbird, Spotted Dove, White-throated Kingfisher, Zebra Dove)
- **Test set**: 900 samples (90 per class)
- **Input size**: 224x224 spectrograms
- **Sample rate**: 16 kHz, N_FFT=2048
- **Framework**: TensorFlow 2.15 / Keras 2.15
- **Hardware**: Linux x86_64, 8 logical cores, 31.27 GB RAM, 1 GPU
- **Training**: AdamW (lr=0.001, weight_decay=1e-5), batch size 32, max 50 epochs, early stopping (patience 10), ReduceLROnPlateau (patience 5), SpecAugment, dropout [0.3, 0.2]

## Test Accuracy by Model and Feature

| Model | Feature | Seed 42 | Seed 100 | Seed 786 | Mean | Std |
|---|---|---|---|---|---|---|
| EfficientNetB0 | Mel | **95.89%** | 90.67% | 93.56% | **93.37%** | 2.62% |
| EfficientNetB0 | MFCC | 90.33% | 90.00% | 87.89% | 89.41% | 1.33% |
| EfficientNetB0 | STFT | 89.33% | 91.33% | 92.33% | 91.00% | 1.53% |
| MobileNetV3S | Mel | 90.33% | 90.67% | 88.89% | 89.96% | 0.94% |
| MobileNetV3S | MFCC | 82.56% | 82.22% | 84.11% | 82.96% | 1.01% |
| MobileNetV3S | STFT | 90.22% | 90.89% | 89.33% | 90.15% | 0.78% |
| ResNet50 | Mel | 88.89% | 91.67% | 85.89% | 88.81% | 2.89% |
| ResNet50 | MFCC | 84.33% | 86.22% | 87.56% | 86.04% | 1.62% |
| ResNet50 | STFT | 91.78% | 91.44% | 89.67% | 90.96% | 1.13% |
| VGG16 | Mel | 89.11% | 87.67% | 87.89% | 88.22% | 0.78% |
| VGG16 | MFCC | 80.33% | 79.56% | 85.78% | 81.89% | 3.39% |
| VGG16 | STFT | 84.56% | 87.56% | 87.89% | 86.67% | 1.84% |

## Mean Test Accuracy: Model vs Feature

| Model | Mel | MFCC | STFT |
|---|---|---|---|
| EfficientNetB0 | **93.37%** | 89.41% | 91.00% |
| MobileNetV3S | 89.96% | 82.96% | 90.15% |
| ResNet50 | 88.81% | 86.04% | 90.96% |
| VGG16 | 88.22% | 81.89% | 86.67% |

## Rankings

### By Model (averaged across all features and seeds)

| Rank | Model | Mean Accuracy | Std |
|---|---|---|---|
| 1 | EfficientNetB0 | 91.26% | 2.39% |
| 2 | ResNet50 | 88.60% | 2.76% |
| 3 | MobileNetV3S | 87.69% | 3.63% |
| 4 | VGG16 | 85.59% | 3.47% |

### By Feature (averaged across all models and seeds)

| Rank | Feature | Mean Accuracy | Std |
|---|---|---|---|
| 1 | Mel | 90.09% | 2.71% |
| 2 | STFT | 89.69% | 2.20% |
| 3 | MFCC | 85.07% | 3.53% |

### Top 5 Configurations (by mean accuracy)

| Rank | Configuration | Mean Accuracy | Best Single Run |
|---|---|---|---|
| 1 | EfficientNetB0 + Mel | 93.37% | 95.89% (seed 42) |
| 2 | EfficientNetB0 + STFT | 91.00% | 92.33% (seed 786) |
| 3 | ResNet50 + STFT | 90.96% | 91.78% (seed 42) |
| 4 | MobileNetV3S + STFT | 90.15% | 90.89% (seed 100) |
| 5 | MobileNetV3S + Mel | 89.96% | 90.67% (seed 100) |

## Mean Training Time (minutes)

| Model | Mel | MFCC | STFT |
|---|---|---|---|
| EfficientNetB0 | 41.6 | 39.5 | 25.2 |
| MobileNetV3S | 74.2 | 76.3 | 41.9 |
| ResNet50 | 43.3 | 39.1 | 27.8 |
| VGG16 | 40.5 | 111.2 | 19.0 |

## Key Findings

1. **EfficientNetB0 + Mel is the best configuration**, achieving 93.37% mean test accuracy and the highest single-run accuracy of 95.89%.
2. **EfficientNetB0 is the top-performing model overall** (91.26%), followed by ResNet50 (88.60%).
3. **Mel spectrograms yield the best results** (90.09%), closely followed by STFT (89.69%), while MFCC lags behind (85.07%).
4. **MFCC underperforms consistently** across all architectures, with the largest gap seen in MobileNetV3S (82.96% vs 90.15% for STFT).
5. **STFT features train fastest** across all models, likely due to earlier convergence.
6. **MobileNetV3S shows the lowest variance** across seeds for Mel and STFT features, suggesting more stable training despite lower peak accuracy.
