#!/usr/bin/env python3
"""
Reverse-annotate SEAbird dataset: regenerate Audacity label files from WAV filenames.

For each Xeno-canto source recording, produces one .txt file with tab-separated
columns:  start_sec  end_sec  sequence_number

Example: xc19400_1044.wav, xc19400_29855.wav, xc19400_57109.wav
  → xc19400.txt:
      1.044	4.044	1
      29.855	32.855	2
      57.109	60.109	3
"""

import os
import re
import argparse
from collections import defaultdict

CLIP_DURATION_MS = 3000
FILENAME_RE = re.compile(r'^(xc\d+)_(\d+)\.wav$', re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description='Reverse-annotate SEAbird WAVs to Audacity labels')
    parser.add_argument('dataset_root', help='Path to dataset root (contains species subdirs)')
    parser.add_argument('--output_dir', help='Output directory (default: <dataset_root>/annotations)')
    parser.add_argument('--clip_ms', type=int, default=CLIP_DURATION_MS, help='Clip duration in ms (default: 3000)')
    args = parser.parse_args()

    root = args.dataset_root
    out_root = args.output_dir or os.path.join(root, 'annotations')

    if not os.path.isdir(root):
        print(f'Error: {root} is not a directory')
        return 1

    species_dirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d != 'annotations'
    )

    total_files = 0
    total_labels = 0

    for species in species_dirs:
        species_path = os.path.join(root, species)
        sources = defaultdict(list)

        for fname in os.listdir(species_path):
            m = FILENAME_RE.match(fname)
            if m:
                source_id = m.group(1).lower()
                start_ms = int(m.group(2))
                sources[source_id].append(start_ms)

        if not sources:
            continue

        species_out = os.path.join(out_root, species)
        os.makedirs(species_out, exist_ok=True)

        for source_id in sorted(sources):
            timestamps = sorted(sources[source_id])
            label_path = os.path.join(species_out, f'{source_id}.txt')

            with open(label_path, 'w') as f:
                for seq, start_ms in enumerate(timestamps, 1):
                    start_sec = start_ms / 1000.0
                    end_sec = (start_ms + args.clip_ms) / 1000.0
                    f.write(f'{start_sec:.6f}\t{end_sec:.6f}\t{seq}\n')

            total_files += 1
            total_labels += len(timestamps)

    print(f'Done: {total_labels} labels across {total_files} files for {len(species_dirs)} species')
    print(f'Output: {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
