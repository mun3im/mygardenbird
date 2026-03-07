# Metadata for 16 kHz MyGardenBird Dataset

This folder contains per-clip metadata and train/validation/test splits for the **16 kHz version** of the MyGardenBird dataset (6,000 clips, 600 per species).

## Files

### `clips.csv`
Per-clip metadata for all 6,000 three-second audio clips.

**Fields:**
- `file_id`: Unique clip identifier (e.g., "xc123456_2860")
- `source_id`: Xeno-canto source recording ID (foreign key to `recordings.csv`)
- `clip_index`: Sequential index within source recording (0-based)
- `start_ms`: Start time within source recording (milliseconds)
- `duration_ms`: Clip duration (always 3000 ms)
- `snr_db`: Estimated signal-to-noise ratio (dB)
- `peak_amplitude`: Maximum absolute sample value (normalized units)
- `vocalization_type`: Song, call, or other (from Xeno-canto metadata)

**Usage:** Links clips to source recordings and provides quality metrics for each segment.

### Split Files

All splits guarantee **source-level separation**: clips from the same source recording never appear in multiple partitions (train/val/test). This prevents data leakage.

#### Primary Split (used in paper)
- **`splits_mip_80_10_10.csv`**: 80% train / 10% val / 10% test using Mixed Integer Programming
  - Train: 4,800 clips (480 per class)
  - Val: 600 clips (60 per class)
  - Test: 600 clips (60 per class)

#### Alternative Splits (for comparison)
- **`splits_mip_70_15_15.csv`**: 70/15/15 split (MIP algorithm)
- **`splits_ga_80_10_10.csv`**: 80/10/10 split (Genetic Algorithm)
- **`splits_ga_70_15_15.csv`**: 70/15/15 split (Genetic Algorithm)
- **`splits_sa_80_10_10.csv`**: 80/10/10 split (Simulated Annealing)
- **`splits_sa_70_15_15.csv`**: 70/15/15 split (Simulated Annealing)

**Fields:**
- `file_id`: Clip identifier (primary key)
- `split`: Partition assignment ("train", "val", or "test")

## Dataset Statistics

- **Total clips**: 6,000
- **Clips per species**: 600 (perfectly balanced)
- **Clip duration**: 3.0 seconds
- **Sampling rate**: 16 kHz
- **Bit depth**: 16-bit PCM
- **Source recordings**: 1,123 unique Xeno-canto files
- **SNR range**: 0.85–59.18 dB (mean: 16.36 dB, median: 15.49 dB)

## Splitting Algorithms

### Mixed Integer Programming (MIP)
Formulates splitting as an optimization problem with hard constraints:
- Each source recording assigned to exactly one partition
- Exact class balance per partition (480/60/60 per species)
- Globally optimal solution guaranteed

### Genetic Algorithm (GA)
Evolutionary approach with:
- Population size: 100
- Generations: 500
- Mutation rate: 0.05
- Near-optimal solutions, fast convergence

### Simulated Annealing (SA)
Stochastic optimization with:
- Temperature schedule: exponential cooling
- Iterations: 10,000
- Acceptance probability based on Boltzmann distribution

## Usage Example

```python
import pandas as pd

# Load clip metadata
clips = pd.read_csv('metadata16khz/clips.csv')

# Load primary split
splits = pd.read_csv('metadata16khz/splits_mip_80_10_10.csv')

# Merge
data = clips.merge(splits, on='file_id')

# Get training clips
train_clips = data[data['split'] == 'train']
print(f"Training clips: {len(train_clips)}")  # 4800
```

## Source-Level Splitting Verification

To verify no source recording appears in multiple partitions:

```python
import pandas as pd

clips = pd.read_csv('metadata16khz/clips.csv')
splits = pd.read_csv('metadata16khz/splits_mip_80_10_10.csv')
data = clips.merge(splits, on='file_id')

# Group by source_id and check for multiple partitions
source_splits = data.groupby('source_id')['split'].nunique()
assert (source_splits == 1).all(), "Data leakage detected!"
print("✓ Source-level separation verified")
```

## References

See `extras/verify_structural_integrity_final.py` for automated verification of all dataset integrity claims.
