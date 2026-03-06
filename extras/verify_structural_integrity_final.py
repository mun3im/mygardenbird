#!/usr/bin/env python3
"""
Verify all claims in Section 5.2 - Structural Integrity

Paper claims:
1. All ten species contribute exactly 600 clips
2. No duplicate file_id values
3. No duplicate (source_id, clip_index) pairs  
4. Zero overlap between train/val/test sets at source-recording level
5. No source_id appears in more than one partition
"""

import pandas as pd
from pathlib import Path
import sys

CLIPS_CSV = Path("/Volumes/Evo/MYGARDENBIRD/metadata16khz/clips.csv")
RECORDINGS_CSV = Path("/Volumes/Evo/MYGARDENBIRD/project_csv/recordings.csv")
SPLITS_CSV = Path("/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv")

print("=" * 80)
print("STRUCTURAL INTEGRITY VERIFICATION - Section 5.2")
print("=" * 80)

clips_df = pd.read_csv(CLIPS_CSV)
recordings_df = pd.read_csv(RECORDINGS_CSV)
splits_df = pd.read_csv(SPLITS_CSV, comment='#')

clips_with_source = clips_df.merge(recordings_df[['source_id', 'species_common']], on='source_id')
clips_with_splits = clips_df.merge(splits_df, on='file_id')

all_passed = True

# Claim 1: Class balance
print("\n✓ CLAIM 1: All ten species contribute exactly 600 clips")
species_counts = clips_with_source['species_common'].value_counts()
assert len(species_counts) == 10 and all(c == 600 for c in species_counts), "Class balance failed"

# Claim 2: No duplicate file_ids
print("✓ CLAIM 2: No duplicate file_id values")
assert not clips_df['file_id'].duplicated().any(), "Duplicate file_ids found"

# Claim 3: No duplicate (source_id, clip_index) pairs
print("✓ CLAIM 3: No duplicate (source_id, clip_index) pairs")
assert not clips_df.duplicated(subset=['source_id', 'clip_index']).any(), "Duplicate pairs found"

# Claims 4 & 5: No source-level overlap
print("✓ CLAIM 4: Zero overlap between partitions at source level")
print("✓ CLAIM 5: No source_id appears in multiple partitions")
source_partitions = clips_with_splits.groupby('source_id')['split'].unique()
assert all(len(p) == 1 for p in source_partitions), "Source appears in multiple partitions"

print("\n" + "=" * 80)
print("✅  ALL 5 CLAIMS VERIFIED")
print("=" * 80)
print(f"\nDataset: {len(clips_df)} clips, {len(recordings_df)} sources, {len(species_counts)} species")
print(f"Split: {clips_with_splits['split'].value_counts()['train']} train, "
      f"{clips_with_splits['split'].value_counts()['val']} val, "
      f"{clips_with_splits['split'].value_counts()['test']} test")

