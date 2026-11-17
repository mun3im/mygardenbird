# MyGardenBird v2.0 — 16 kHz, 15 species

Benchmarks on the current dataset release
([v2.0](https://doi.org/10.5281/zenodo.24217498), DOI 10.5281/zenodo.24217498):
15 species, CC BY-NC-ND sources removed throughout, source/site-disjoint
80:10:10 MIP split (`metadata16khz/splits_mip_80_10_10.csv`), class sizes
438–600 (not uniform — 11/15 classes reach 600; 4 fall short for lack of
further non-ND regional recordings).

Single seed (42) per model so far — mel spectrogram features, Mixup
α=0.2, 866 test clips. Full 3-seed × 3-augmentation sweep in progress.

## Results

| Model | Params | Test Accuracy | Macro F1 | Notes |
|---|---:|---:|---:|---|
| MynaNet v1 (INT8 TFLite) | 328.8K | **90.76%** | 0.9061 | 434.8 KB on-device model; FP32 was 90.53% (INT8 quantization *improved* accuracy by +0.23 pp) |
| EfficientNet-B0 | 4.38M | **92.15%** (best) | 0.9197 | ImageNet-pretrained, fine-tuned |
| ResNet-50 | 24.12M | 89.38% | 0.8934 | ImageNet-pretrained, fine-tuned |
| MobileNetV3-Small | 1.24M | 88.80% | 0.8853 | ImageNet-pretrained, fine-tuned |

![Accuracy vs params](accuracy_vs_params_v2.pdf)

- **EfficientNet-B0** is the strongest of the four architectures.
- **MynaNet v1**, a from-scratch depthwise-separable CNN designed for
  Cortex-M7 deployment, needs only 328.8K params (13× fewer than
  MobileNetV3-Small, 73× fewer than ResNet-50) yet beats both of them
  and sits within 1.4 pp of EfficientNet-B0 — the best accuracy-per-parameter
  trade-off of the four.
- The three hardest classes across every model are **Pied Fantail**,
  **Common Myna**, and **Ornate Sunbird**, which also have the smallest
  sample counts or the fewest unique source recordings (see per-class
  `support` in each model's classification report) — consistent with the
  BirdNET zero-shot pass flagging many of the same species for review
  (see `../birdnet_results/v2_16khz_full/flagged_for_manual_review.csv`).

## Per-model classification reports

- `../results_v2_16k_mobilenetv3s_linux/mobilenetv3s_mel_pretrained_mixup0.2_seed42/classification_report.txt`
- `../results_v2_16k_efficientnetb0_linux/efficientnetb0_mel_pretrained_mixup0.2_seed42/classification_report.txt`
- `../results_v2_16k_resnet50_linux/resnet50_mel_pretrained_mixup0.2_seed42/classification_report.txt`
- `../results_linux/v1_dscnn_se_res_att_wide_mels80_drop05_rand42_warm50_mixup0.2_split80:10:10_linux/classification_report_{fp32,int8}.txt`

## Reproduction

```bash
# CNN backbones (Stage 9)
python pipeline/Stage9_train_mygardenbird_multifeature.py \
    --model {mobilenetv3s,efficientnetb0,resnet50} --feature mel --mixup 0.2 --seed 42 \
    --splits_csv /Volumes/Evo/MYGARDENBIRD_v2/metadata16khz/splits_mip_80_10_10.csv \
    --dataset_root /Volumes/Evo/MYGARDENBIRD_v2/mygardenbird16khz

# MynaNet v1
python mynanet_v1.py \
    --splits_csv /Volumes/Evo/MYGARDENBIRD_v2/metadata16khz/splits_mip_80_10_10.csv \
    --flat_dir /Volumes/Evo/MYGARDENBIRD_v2/mygardenbird16khz \
    --mixup 0.2 --random_seed 42
```
