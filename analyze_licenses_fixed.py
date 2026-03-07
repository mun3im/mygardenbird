#!/usr/bin/env python3
"""
Analyze licenses from per-species CSV files, filtering by recordings.csv
"""
import pandas as pd
from pathlib import Path

# Load the recordings.csv to get the source IDs we actually use
RECORDINGS_CSV = Path("/Volumes/Evo/MYGARDENBIRD/project_csv/recordings.csv")
PER_SPECIES_DIR = Path("/Volumes/Evo/MYGARDENBIRD/per_species_csv")

print("=" * 80)
print("XENO-CANTO LICENSE ANALYSIS")
print("=" * 80)

# Load recordings to get source IDs we actually use
recordings_df = pd.read_csv(RECORDINGS_CSV)
used_source_ids = set(recordings_df['source_id'].astype(str))

# Find all per-species CSV files
per_species_files = list(PER_SPECIES_DIR.glob("*.csv"))

# Aggregate all metadata
all_records = []
for csv_file in sorted(per_species_files):
    try:
        df = pd.read_csv(csv_file)
        if 'id' not in df.columns:
            continue
        df['id'] = df['id'].astype(str)
        df_filtered = df[df['id'].isin(used_source_ids)]
        if len(df_filtered) > 0:
            all_records.append(df_filtered)
    except Exception as e:
        pass

combined_df = pd.concat(all_records, ignore_index=True)
combined_df['id'] = combined_df['id'].astype(str)

# Merge with species names
recordings_df['source_id'] = recordings_df['source_id'].astype(str)
combined_df = combined_df.merge(
    recordings_df[['source_id', 'species_common']],
    left_on='id',
    right_on='source_id',
    how='left'
)

print(f"\nTotal recordings analyzed: {len(combined_df)}")
print(f"Species: {combined_df['species_common'].nunique()}")

# License analysis
print(f"\n{'=' * 80}")
print("OVERALL LICENSE DISTRIBUTION")
print(f"{'=' * 80}")

license_counts = combined_df['lic'].value_counts()
total = len(combined_df)

for license_url, count in license_counts.items():
    # Extract license type from URL
    if 'by-nc-sa' in license_url:
        lic_type = 'CC BY-NC-SA'
    elif 'by-nc-nd' in license_url:
        lic_type = 'CC BY-NC-ND'
    elif 'by-sa' in license_url and 'nc' not in license_url:
        lic_type = 'CC BY-SA'
    elif 'by/4.0' in license_url or 'by/3.0' in license_url:
        lic_type = 'CC BY'
    elif 'publicdomain' in license_url or 'zero' in license_url:
        lic_type = 'CC0 (Public Domain)'
    else:
        lic_type = 'Other'
    
    version = ''
    if '4.0' in license_url:
        version = '4.0'
    elif '3.0' in license_url:
        version = '3.0'
    elif '2.5' in license_url:
        version = '2.5'
    
    pct = (count / total) * 100
    full_name = f"{lic_type} {version}".strip()
    print(f"  {full_name:30s}: {count:5d} ({pct:5.1f}%)")

# Summary by license type
print(f"\n{'=' * 80}")
print("SUMMARY BY LICENSE TYPE")
print(f"{'=' * 80}")

nc_sa = combined_df['lic'].str.contains('by-nc-sa').sum()
nc_nd = combined_df['lic'].str.contains('by-nc-nd').sum()
by_sa = (combined_df['lic'].str.contains('by-sa') & ~combined_df['lic'].str.contains('nc')).sum()
by_only = (combined_df['lic'].str.contains('by/') & ~combined_df['lic'].str.contains('nc|nd|sa')).sum()
cc0 = combined_df['lic'].str.contains('publicdomain|zero').sum()

print(f"  CC BY-NC-SA (NonCommercial, ShareAlike): {nc_sa:5d} ({nc_sa/total*100:5.1f}%)")
print(f"  CC BY-NC-ND (NonCommercial, NoDerivs):  {nc_nd:5d} ({nc_nd/total*100:5.1f}%)")
print(f"  CC BY-SA (ShareAlike, commercial OK):    {by_sa:5d} ({by_sa/total*100:5.1f}%)")
print(f"  CC BY (Attribution only):                {by_only:5d} ({by_only/total*100:5.1f}%)")
print(f"  CC0 (Public Domain):                     {cc0:5d} ({cc0/total*100:5.1f}%)")

# Redistribution compatibility
print(f"\n{'=' * 80}")
print("REDISTRIBUTION COMPATIBILITY ASSESSMENT")
print(f"{'=' * 80}")

nc_count = combined_df['lic'].str.contains('nc').sum()
nd_count = combined_df['lic'].str.contains('nd').sum()
compatible = total - max(nc_count, nd_count)  # Simplification

print(f"\nRecordings with 'NC' (NonCommercial): {nc_count} ({nc_count/total*100:.1f}%)")
print(f"Recordings with 'ND' (NoDerivatives): {nd_count} ({nd_count/total*100:.1f}%)")
print(f"\n⚠️  CRITICAL: {nc_count + nd_count - (nc_count if nd_count > nc_count else nd_count)} recordings have NC and/or ND restrictions")
print(f"✅ Only {total - nc_count} recordings ({(total-nc_count)/total*100:.1f}%) allow commercial use")
print(f"✅ Only {total - nd_count} recordings ({(total-nd_count)/total*100:.1f}%) explicitly allow derivatives")

# Per-species breakdown
print(f"\n{'=' * 80}")
print("PER-SPECIES LICENSE BREAKDOWN (simplified)")
print(f"{'=' * 80}")

for species in sorted(combined_df['species_common'].dropna().unique()):
    species_df = combined_df[combined_df['species_common'] == species]
    total_species = len(species_df)
    nc_sa_count = species_df['lic'].str.contains('by-nc-sa').sum()
    nc_nd_count = species_df['lic'].str.contains('by-nc-nd').sum()
    permissive = total_species - nc_sa_count - nc_nd_count
    
    print(f"\n{species}:")
    print(f"  Total: {total_species}")
    print(f"  CC BY-NC-SA: {nc_sa_count} ({nc_sa_count/total_species*100:.1f}%)")
    print(f"  CC BY-NC-ND: {nc_nd_count} ({nc_nd_count/total_species*100:.1f}%)")
    print(f"  More permissive (BY-SA/BY/CC0): {permissive} ({permissive/total_species*100:.1f}%)")

