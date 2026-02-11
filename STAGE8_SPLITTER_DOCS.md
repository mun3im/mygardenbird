# Stage 8: Dataset Splitting for Bird Audio Classification

Split audio datasets into train/val/test sets while preventing data leakage from same-source recordings.

## Algorithm Overview

Three splitting algorithms are available, all achieving optimal solutions (objective=0):

| Algorithm | Stage | Description | Recommendation |
|-----------|-------|-------------|----------------|
| **MIP** | 8a | Mixed Integer Programming (CBC solver) | **Recommended** - fastest, guaranteed optimal |
| **GA** | 8b | Genetic Algorithm | Good alternative, slower |
| **SA** | 8c | Simulated Annealing | Slowest, useful for exploration |

### MIP (Mixed Integer Programming) - Stage 8a

Uses the CBC solver to find optimal splits via integer linear programming.

**Advantages:**
- Guaranteed optimal solution
- Fastest execution (~1 second)
- Deterministic results

**How it works:**
1. Formulates splitting as an optimization problem
2. Decision variables: source → split assignment
3. Constraints: exact split ratios per class
4. Objective: minimize class imbalance

### GA (Genetic Algorithm) - Stage 8b

Evolutionary approach with population-based search.

**Advantages:**
- Good for exploring solution space
- Can handle complex constraints
- Parallelizable

**How it works:**
1. Initialize population of random splits
2. Evaluate fitness (class balance + no leakage)
3. Select, crossover, mutate
4. Repeat until convergence

### SA (Simulated Annealing) - Stage 8c

Physics-inspired optimization with temperature-based acceptance.

**Advantages:**
- Simple implementation
- Good at escaping local minima
- Anytime algorithm (can stop early)

**How it works:**
1. Start with random split
2. Make random moves (swap sources between splits)
3. Accept improvements; accept worse moves with decreasing probability
4. Cool temperature until frozen

## Performance Comparison

Benchmark on 6000-sample dataset (10 classes, 1074 sources):

| Algorithm | Time | Speed vs SA | Optimal? | Deterministic? |
|-----------|------|-------------|----------|----------------|
| **MIP (CBC solver)** | **1.7 s** | **564×** | Proven | Yes |
| **Genetic Algorithm** | **13.6 s** | **70×** | Achieved | No |
| **Simulated Annealing** | **952 s (15.9 min)** | **1×** | Achieved | No |

**Key Finding:** MIP is ~564× faster than SA while guaranteeing the same optimal result with deterministic output.

## Quick Start

### MIP Splitter (Recommended)

```bash
python Stage8a_splitter_mip.py /path/to/dataset \
    --train_ratio 0.75 --val_ratio 0.10 --test_ratio 0.15 \
    --output ./seabird_splits_mip_75_10_15.csv \
    --seed 42
```

### Genetic Algorithm Splitter

```bash
python Stage8b_splitter_genetic_algorithm.py /path/to/dataset \
    --train_ratio 0.75 --val_ratio 0.10 --test_ratio 0.15 \
    --output ./seabird_splits_ga_75_10_15.csv \
    --seed 42
```

### Simulated Annealing Splitter

```bash
python Stage8c_splitter_simulated_annealing.py /path/to/dataset \
    --train_ratio 0.75 --val_ratio 0.10 --test_ratio 0.15 \
    --output ./seabird_splits_sa_75_10_15.csv \
    --seed 42
```

## Output Format

All splitters produce CSV files with the same format:

```csv
# split_ratio=75:10:15 seed=42 objective=0 solver=mip_cbc
filename,split
xc1002657_2860.wav,test
xc1003831_2642.wav,train
xc1004325_1203.wav,val
...
```

**Header metadata:**
- `split_ratio`: Train:Val:Test percentages
- `seed`: Random seed for reproducibility
- `objective`: Solution quality (0 = perfect)
- `solver`: Algorithm used

## Pre-generated Splits

Ready-to-use splits for 6000-sample SEAbird dataset (seed=42, all objective=0):

| File | Algorithm | Train | Val | Test |
|------|-----------|-------|-----|------|
| `seabird_splits_mip_75_10_15.csv` | MIP | 75% | 10% | 15% |
| `seabird_splits_mip_80_10_10.csv` | MIP | 80% | 10% | 10% |
| `seabird_splits_mip_70_15_15.csv` | MIP | 70% | 15% | 15% |
| `seabird_splits_ga_75_10_15.csv` | Genetic Algorithm | 75% | 10% | 15% |
| `seabird_splits_ga_80_10_10.csv` | Genetic Algorithm | 80% | 10% | 10% |
| `seabird_splits_ga_70_15_15.csv` | Genetic Algorithm | 70% | 15% | 15% |
| `seabird_splits_sa_75_10_15.csv` | Simulated Annealing | 75% | 10% | 15% |
| `seabird_splits_sa_80_10_10.csv` | Simulated Annealing | 80% | 10% | 10% |
| `seabird_splits_sa_70_15_15.csv` | Simulated Annealing | 70% | 15% | 15% |

## Data Leakage Prevention

All splitters enforce **source-based separation**:

- Audio segments from the same Xeno-Canto recording share a source ID (e.g., `xc402013`)
- Sources are assigned atomically to one split only
- This prevents the model from memorizing recording-specific artifacts

**Example:**
```
xc402013_27465.wav → train  (source: xc402013)
xc402013_30210.wav → train  (same source, same split)
xc402013_33041.wav → train  (same source, same split)
xc789456_12000.wav → test   (different source, can be in different split)
```

## SEAbird Dataset Statistics

The 75:10:15 split distributes 1,074 Xeno-Canto sources across train/val/test:

| Species | Scientific Name | Train | Val | Test | Clips | XC Sources (Train/Val/Test) |
|---------|-----------------|-------|-----|------|-------|----------------------------|
| Asian Koel | *Eudynamys scolopaceus* | 450 | 60 | 90 | 600 | 86 / 13 / 32 |
| Collared Kingfisher | *Todiramphus chloris* | 450 | 60 | 90 | 600 | 73 / 18 / 26 |
| Common Iora | *Aegithina tiphia* | 450 | 60 | 90 | 600 | 75 / 9 / 21 |
| Common Myna | *Acridotheres tristis* | 450 | 60 | 90 | 600 | 81 / 10 / 32 |
| Common Tailorbird | *Orthotomus sutorius* | 450 | 60 | 90 | 600 | 65 / 10 / 21 |
| Large-tailed Nightjar | *Caprimulgus macrurus* | 450 | 60 | 90 | 600 | 56 / 11 / 16 |
| Olive-backed Sunbird | *Cinnyris jugularis* | 450 | 60 | 90 | 600 | 68 / 10 / 17 |
| Spotted Dove | *Spilopelia chinensis* | 450 | 60 | 90 | 600 | 68 / 9 / 18 |
| White-throated Kingfisher | *Halcyon smyrnensis* | 450 | 60 | 90 | 600 | 81 / 17 / 25 |
| Zebra Dove | *Geopelia striata* | 450 | 60 | 90 | 600 | 69 / 10 / 27 |
| **Total** | | **4500** | **600** | **900** | **6000** | **722 / 117 / 235** |

Each species has exactly 600 clips (perfect class balance). The MIP solver achieves objective=0, meaning exact target ratios per class.

## Using Splits with Training

```bash
python Stage9_train_seabird_multifeature.py \
    --splits_csv ./seabird_splits_mip_75_10_15.csv \
    --dataset_root /path/to/audio/files \
    --model efficientnetb0 \
    --feature mel \
    --use_pretrained
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--train_ratio` | Training set proportion | 0.75 |
| `--val_ratio` | Validation set proportion | 0.10 |
| `--test_ratio` | Test set proportion | 0.15 |
| `--seed` | Random seed | 42 |
| `--output` | Output CSV path | `./splits.csv` |

## Justification for the 75:10:15 Split Ratio


> We adopt a 75:10:15 train/validation/test split with source-level separation enforced by mixed integer programming. The larger test partition (15%) provides approximately 90 samples per class, yielding more reliable per-class evaluation metrics than a conventional 80:10:10 split (~60 per class). Joseph (2022) demonstrates that the optimal splitting ratio is not fixed but depends on effective model complexity; for transfer-learned CNNs with strong regularisation, the marginal value of additional training data is low. We verified empirically that reducing the training proportion from 80% to 75% produced no statistically significant change in test accuracy across 9 paired runs (MobileNetV3S, 3 features × 3 seeds; p > 0.05), while reducing seed-to-seed variance by up to 3.5×.

### Rationale

The 75:10:15 train/validation/test split was selected over the more conventional 80:10:10 ratio to increase test set reliability for a constrained dataset of 6,000 samples across 10 classes. With 15% allocated to testing, each class receives ~90 test samples compared to ~60 under an 80:10:10 split — a 50% increase in per-class evaluation data that yields more reliable F1 and accuracy estimates.

There is no universally optimal split ratio; the best choice depends on dataset size, model complexity, and whether transfer learning is used (Joseph, 2022). For transfer-learned models (e.g., ImageNet-pretrained MobileNetV3S), the strong inductive bias means the 5% reduction in training data has negligible impact on learned representations.

### Empirical Validation (MobileNetV3S Ablation)

A direct comparison of MobileNetV3S across 3 feature types (Mel, MFCC, STFT) and 3 seeds (42, 100, 786) confirmed that the 75:10:15 split produces equivalent or better results:

| Feature | 80:10:10 Mean Acc (std) | 75:10:15 Mean Acc (std) | Difference |
|---------|-------------------------|-------------------------|------------|
| Mel     | 87.22% (3.22%)          | 89.96% (0.93%)          | +2.74%     |
| MFCC    | 83.89% (1.53%)          | 82.96% (0.99%)          | -0.93%     |
| STFT    | 89.72% (1.40%)          | 90.15% (0.80%)          | +0.43%     |
| **Overall** | **86.94% (3.26%)**  | **87.69% (3.60%)**      | +0.75%     |

Key findings:

- **No accuracy penalty**: overall mean test accuracy differed by only +0.75% (not statistically significant).
- **Substantially lower seed-to-seed variance**: within-feature standard deviation was 1.5–3.5× lower under 75:10:15 (e.g., Mel std dropped from 3.22% to 0.93%).
- **Better per-class reliability**: difficult classes (Asian Koel, Zebra Dove) showed F1 improvements of +0.06–0.09 with the larger test set.

----------------------------------------------


## Tips

1. **Use MIP** unless you have a specific reason to explore alternatives
2. **Always set `--seed`** for reproducible splits
3. **Check objective=0** in the output to confirm optimal solution
4. **Use pre-generated splits** from this repository for consistency
5. **Verify no leakage** by checking that same-source files are in same split

## Troubleshooting

### MIP solver not found

```bash
pip install pulp
# CBC solver is bundled with PuLP
```

### Infeasible solution

If splits are impossible with exact ratios:
- Check that each class has enough sources
- Try slightly different ratios (e.g., 74:11:15 instead of 75:10:15)

### Slow GA/SA performance

- Reduce population size (GA) or iterations (SA)
- Consider using MIP instead for guaranteed fast results

--------

### References

1. **Joseph, V. R. (2022).** Optimal ratio for data splitting. *Statistical Analysis and Data Mining: The ASA Data Science Journal*, 15(4), 531–538. https://doi.org/10.1002/sam.11583
   — Derives a closed-form optimal split ratio dependent on model complexity; shows that 80:10:10 is not universally optimal.

2. **Ramezan, C. A., Warner, T. A., & Maxwell, A. E. (2019).** Evaluation of sampling and cross-validation tuning strategies for regional-scale machine learning classification. *Remote Sensing*, 11(2), 185. https://doi.org/10.3390/rs11020185
   — Demonstrates that single holdout splits are sensitive to random partitioning and that larger evaluation sets reduce variance in performance estimates.

3. **Oala, L., et al. (2024).** Trade-off between training and testing ratio in machine learning for medical image processing. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11419616/
   — Studies how split ratio affects evaluation reliability in medical imaging; analogous small-dataset transfer learning context.

4. **Vabalas, A., Gowen, E., Poliakoff, E., & Casson, A. J. (2019).** Machine learning algorithm validation with a limited sample size. *PLoS ONE*, 14(11), e0224365. https://doi.org/10.1371/journal.pone.0224365
   — Discusses sample size requirements for reliable ML evaluation; supports larger test proportions when total data is limited.

5. **Stowell, D., et al. (2022).** Computational bioacoustics with deep learning: a review and roadmap. *PeerJ*, 10, e13152. https://doi.org/10.7717/peerj.13152
   — Reviews data splitting practices in bioacoustics; emphasises source-level separation to prevent data leakage.

