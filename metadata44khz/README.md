# Metadata for 44.1 kHz MyGardenBird Dataset

This folder contains per-clip metadata and train/validation/test splits for the **44.1 kHz high-resolution version** of the MyGardenBird dataset (5,792 clips).

## Files

### `clips.csv`
Per-clip metadata for 5,792 three-second audio clips at 44.1 kHz.

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
  - Split maintains same source-level assignments as 16 kHz version where possible

#### Alternative Splits (for comparison)
- **`splits_mip_70_15_15.csv`**: 70/15/15 split (MIP algorithm)
- **`splits_ga_80_10_10.csv`**: 80/10/10 split (Genetic Algorithm)
- **`splits_ga_70_15_15.csv`**: 70/15/15 split (Genetic Algorithm)
- **`splits_sa_70_15_15.csv`**: 70/15/15 split (Simulated Annealing)

**Fields:**
- `file_id`: Clip identifier (primary key)
- `split`: Partition assignment ("train", "val", or "test")

## Dataset Statistics

- **Total clips**: 5,792 (vs 6,000 at 16 kHz)
- **Clips per species**: Variable (574–599 per species)
- **Clip duration**: 3.0 seconds
- **Sampling rate**: 44.1 kHz
- **Bit depth**: 16-bit PCM
- **Source recordings**: Subset of sources with native sampling rate ≥44.1 kHz

## Why Fewer Clips?

The 44.1 kHz dataset contains 5,792 clips (208 fewer than the 16 kHz version) because:

1. **No upsampling**: Only sources recorded at ≥44.1 kHz are included
2. **Avoids artifacts**: Upsampling from lower rates would introduce interpolation artifacts
3. **Maintains quality**: All clips represent authentic high-frequency content

Per-species counts range from 574 to 599 clips, with slight imbalance due to varying availability of high-sample-rate recordings across species.

## Use Case

The 44.1 kHz subset is intended for:
- Applications requiring high spectro-temporal resolution
- Species with significant acoustic energy above 8 kHz
- Transfer learning from high-resolution models
- Research on the impact of sampling rate on classification performance

## Performance Comparison

Experimental results (SpecAugment, 3 seeds):

| Model | 16 kHz | 44.1 kHz | Δ |
|---|---|---|---|
| MobileNetV3-Small | 94.61% ± 0.51% | 91.17% ± 0.94% | -3.44% |
| EfficientNet-B0 | 96.94% ± 1.11% | 94.22% ± 0.84% | -2.72% |
| ResNet-50 | 95.06% ± 0.67% | 92.22% ± 0.55% | -2.84% |

**Finding:** 16 kHz provides better accuracy for these species, as most vocalizations have dominant energy below 8 kHz. The 44.1 kHz version introduces additional high-frequency noise without improving discriminability.

## Usage Example

```python
import pandas as pd

# Load clip metadata
clips = pd.read_csv('metadata44khz/clips.csv')

# Load primary split
splits = pd.read_csv('metadata44khz/splits_mip_80_10_10.csv')

# Merge
data = clips.merge(splits, on='file_id')

# Check class balance
print(data.groupby(['species_common', 'split']).size())
```

## Relationship to 16 kHz Dataset

- The 44.1 kHz dataset is a **subset** of the 16 kHz dataset (same source recordings)
- File IDs are consistent across both versions
- Source-level splitting ensures no leakage in either version
- Recordings recorded below 44.1 kHz are excluded from the 44.1 kHz version

## References

See `extras/verify_structural_integrity_final.py` for automated verification of dataset integrity claims. The script works with both 16 kHz and 44.1 kHz metadata.
