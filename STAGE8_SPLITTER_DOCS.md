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

Benchmark on 6000-sample dataset (10 classes, 1074 sources) with 3 split ratios:

| Algorithm | Avg Time | 75:10:15 | 80:10:10 | 70:15:15 | Solution Quality |
|-----------|----------|----------|----------|----------|------------------|
| **MIP** | **1.1s** | 1.17s | 1.24s | 1.03s | Optimal (objective=0) |
| GA | 7.5s | 3.72s | 3.04s | 15.59s | Optimal (objective=0) |
| SA | ~19 min | 19.2 min | 19.6 min | 17.6 min | Optimal (objective=0) |

**Key Finding:** MIP is ~7x faster than GA and ~1000x faster than SA while achieving the same optimal result.

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
