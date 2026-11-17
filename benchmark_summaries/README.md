# Benchmark Summaries

Pretrained-CNN and from-scratch-CNN benchmarks for the MyGardenBird v2.0
16 kHz dataset (15 species, DOI
[10.5281/zenodo.24217498](https://doi.org/10.5281/zenodo.24217498)).

| Summary | Dataset | Species | Models | Experiments |
|---|---|---|---|---|
| [`v2_16khz_15species.md`](v2_16khz_15species.md) | MyGardenBird v2.0 | 15 | EfficientNet-B0, ResNet-50, MobileNetV3-Small, MynaNet v1 | 4/4 (seed 42 only — full 3-seed × 3-augmentation sweep in progress) |

## Accuracy vs. Model Size

Measured params (not literature/architecture tables) vs. test accuracy,
single seed (42), Mixup α=0.2:

![Accuracy vs params (v2.0)](accuracy_vs_params_v2.pdf)

| Model | Params | Test Acc |
|---|---:|---:|
| EfficientNet-B0 | 4.38M | **92.15%** |
| MynaNet v1 (INT8) | 0.33M | 90.76% |
| ResNet-50 | 24.12M | 89.38% |
| MobileNetV3-Small | 1.24M | 88.80% |

MynaNet v1 — a from-scratch depthwise-separable CNN sized for Cortex-M7
deployment — gets within 1.4 pp of EfficientNet-B0 at 13× fewer params than
MobileNetV3-Small and 73× fewer than ResNet-50, the best accuracy-per-param
trade-off of the four. Full details and per-class breakdowns in
[`v2_16khz_15species.md`](v2_16khz_15species.md).

## Regenerating the Plot

```bash
cd benchmark_summaries
python3 accuracy_vs_params_v2_figure.py
```

## Regenerating the Underlying Data

Each CNN row is one run of
[`pipeline/Stage9_train_mygardenbird_multifeature.py`](../pipeline/Stage9_train_mygardenbird_multifeature.py),
which trains a single (model, augmentation, seed) configuration end-to-end and
writes `results.json` + `classification_report.txt` + `confusion_matrix.png`
+ `training_curves.png` to `--output_dir`. For example, to reproduce the
EfficientNet-B0 + Mixup α=0.2, seed 42 run:

```bash
python3 pipeline/Stage9_train_mygardenbird_multifeature.py \
    --model efficientnetb0 \
    --feature mel \
    --splits_csv metadata16khz/splits_mip_80_10_10.csv \
    --dataset_root /path/to/mygardenbird16khz \
    --mixup 0.2 \
    --seed 42 \
    --output_dir ./results_v2_16k_efficientnetb0
```

`--model` accepts `mobilenetv3s`, `resnet50`, `efficientnetb0`; omit
`--mixup` for the no-augmentation runs or pass `--specaug` for SpecAugment.
The full 36-run sweep (4 models × 3 augmentations × 3 seeds — MynaNet v1
via `mynanet_v1.py` with `--specaugment`/`--mixup`, the other three via
Stage9) is driven by a shell loop over those cells.

Note that the splits file referenced above ships with the dataset deposit
rather than with this repository; see the deposit's `metadata16khz/`.
