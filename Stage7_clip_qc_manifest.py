#!/usr/bin/env python3
"""
Stage 7: Quality control and manifest generation.

Analyzes extracted WAV segments and produces:
  - qc_report.csv        — per-clip quality metrics
  - seabird_dataset_manifest.csv — rich per-clip metadata for downstream use
                                   (split column left empty; filled at Stage 8)

Quality checks performed:
  - Audio duration validation (~3 s)
  - Sample rate detection (whatever Stage 6 produced; reported, not filtered)
  - Clipping detection
  - File corruption detection
  - SNR estimation

Silent files are not expected at this stage (Stage 6 filters them);
silence is detected and reported but not used as a rejection criterion.

Splits are created at Stage 8.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from config import ACTIVE_SPECIES, EXTRACTED_SEGS, DATASET_DIR, PER_SPECIES_CSV, PER_SPECIES_FLACS

_COMMON_TO_INFO = {common: (common, scientific, code)
                   for common, scientific, code in ACTIVE_SPECIES}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def compute_snr(audio, sr, noise_percentile=10):
    """
    Estimate SNR from a clip using a percentile-based noise floor.

    Divides the clip into 50 ms frames, computes per-frame RMS, and treats
    the lowest `noise_percentile`% of frames as the noise floor.

    Returns SNR in dB, or None if the estimate cannot be computed.
    """
    frame_len = int(sr * 0.05)   # 50 ms
    hop = frame_len // 2
    if len(audio) < frame_len:
        return None
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop)
    rms_frames = np.sqrt(np.mean(frames ** 2, axis=0))
    rms_frames = rms_frames[rms_frames > 0]
    if len(rms_frames) == 0:
        return None
    noise_floor = np.percentile(rms_frames, noise_percentile)
    if noise_floor < 1e-10:
        return None
    signal_rms = np.sqrt(np.mean(audio ** 2))
    return float(20 * np.log10(signal_rms / noise_floor))


def check_audio_quality(audio_path, target_duration=3.0,
                        silence_threshold=-40, clipping_threshold=0.99):
    """
    Load an audio file and compute quality metrics.

    Sample rate is detected from the file itself — no expected rate is assumed.

    Returns a dict of metrics.
    """
    metrics = {
        'file': audio_path.name,
        'valid': True,
        'error': None,
        'duration': 0.0,
        'sample_rate': 0,
        'num_samples': 0,
        'rms_db': 0.0,
        'peak_amplitude': 0.0,
        'snr_db': None,
        'is_silent': False,
        'is_clipped': False,
        'duration_ok': False,
    }

    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

        metrics['sample_rate'] = sr
        metrics['num_samples'] = len(audio)
        metrics['duration'] = len(audio) / sr
        metrics['duration_ok'] = abs(metrics['duration'] - target_duration) < 0.1

        rms = np.sqrt(np.mean(audio ** 2))
        metrics['rms_db'] = float(20 * np.log10(rms + 1e-10))
        metrics['peak_amplitude'] = float(np.max(np.abs(audio)))
        metrics['snr_db'] = compute_snr(audio, sr)

        metrics['is_silent'] = metrics['rms_db'] < silence_threshold
        metrics['is_clipped'] = metrics['peak_amplitude'] > clipping_threshold

    except Exception as e:
        metrics['valid'] = False
        metrics['error'] = str(e)

    return metrics


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(input_dir, target_duration=3.0):
    """
    Walk {input_dir}/{species}/*.wav|flac and run quality checks on every file.

    Returns dict: species -> list of metrics dicts.
    """
    input_path = Path(input_dir)
    results = defaultdict(list)

    if not input_path.exists():
        print(f"Error: directory does not exist: {input_dir}")
        return results

    species_dirs = sorted(d for d in input_path.iterdir() if d.is_dir())
    print(f"Found {len(species_dirs)} species directories")
    print()

    for species_dir in tqdm(species_dirs, desc="Analysing species"):
        species = species_dir.name
        audio_files = (list(species_dir.glob("*.wav")) +
                       list(species_dir.glob("*.flac")))
        for audio_file in sorted(audio_files):
            metrics = check_audio_quality(audio_file, target_duration)
            metrics['species'] = species
            results[species].append(metrics)

    return results


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_summary_report(results):
    """Print a per-species QC summary to stdout."""
    total = sum(len(f) for f in results.values())
    if total == 0:
        print("No files found.")
        return

    n_valid    = sum(sum(1 for m in f if m['valid'])       for f in results.values())
    n_silent   = sum(sum(1 for m in f if m['is_silent'])   for f in results.values())
    n_clipped  = sum(sum(1 for m in f if m['is_clipped'])  for f in results.values())
    n_dur_ok   = sum(sum(1 for m in f if m['duration_ok']) for f in results.values())

    print("=" * 80)
    print("QUALITY CONTROL SUMMARY")
    print("=" * 80)
    print(f"Total files   : {total}")
    print(f"Valid         : {n_valid}  ({n_valid/total*100:.1f}%)")
    print(f"Corrupt       : {total - n_valid}  ({(total-n_valid)/total*100:.1f}%)")
    print(f"Silent        : {n_silent}  ({n_silent/total*100:.1f}%)  [informational — not filtered]")
    print(f"Clipped       : {n_clipped}  ({n_clipped/total*100:.1f}%)")
    print(f"Duration ~3 s : {n_dur_ok}  ({n_dur_ok/total*100:.1f}%)")
    print()

    # Detect unique sample rates present
    all_srs = sorted({m['sample_rate'] for f in results.values() for m in f if m['valid']})
    print(f"Sample rates detected: {all_srs}")
    print()

    print(f"{'Species':<30} {'Total':>7} {'Valid':>7} {'Silent':>7} {'Clipped':>7} {'Dur OK':>7}")
    print("-" * 80)
    for species in sorted(results.keys()):
        files = results[species]
        print(f"{species:<30} "
              f"{len(files):>7} "
              f"{sum(1 for m in files if m['valid']):>7} "
              f"{sum(1 for m in files if m['is_silent']):>7} "
              f"{sum(1 for m in files if m['is_clipped']):>7} "
              f"{sum(1 for m in files if m['duration_ok']):>7}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# QC CSV
# ---------------------------------------------------------------------------

def save_qc_report(results, output_path):
    """Save per-file quality metrics to qc_report.csv."""
    fieldnames = [
        'species', 'file', 'valid', 'error',
        'duration', 'sample_rate', 'num_samples',
        'rms_db', 'peak_amplitude', 'snr_db',
        'is_silent', 'is_clipped', 'duration_ok',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for species in sorted(results):
            for metrics in results[species]:
                row = dict(metrics)
                if row.get('snr_db') is not None:
                    row['snr_db'] = f"{row['snr_db']:.2f}"
                writer.writerow(row)
    print(f"QC report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Annotation index (Stage 5 .txt files)
# ---------------------------------------------------------------------------

def load_annotation_index(annotation_dir):
    """
    Build lookup: xc_id (str) -> {sequential_index (int) -> onset_ms (int)}.

    Stage 5 annotation format (tab-separated):
        start_time  end_time  label  index
    where start_time is seconds from start of the source FLAC.
    """
    index = defaultdict(dict)
    ann_dir = Path(annotation_dir)
    if not ann_dir.exists():
        return index
    for txt_file in ann_dir.rglob("*.txt"):
        stem = txt_file.stem.lower()
        if not stem.startswith("xc"):
            continue
        xc_id = stem[2:]
        try:
            with open(txt_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue
                    onset_ms = int(round(float(parts[0]) * 1000))
                    seq_idx  = int(parts[3])
                    index[xc_id][seq_idx] = onset_ms
        except Exception:
            continue
    return index


# ---------------------------------------------------------------------------
# Stage 1 metadata
# ---------------------------------------------------------------------------

def load_xc_metadata(metadata_dir):
    """
    Load Stage 1 per-species CSVs.

    Returns dict: xc_id (str) -> {lat, lon, quality_grade, recording_type, country}

    Note: sampling_rate for the manifest comes from the WAV file itself (detected
    in check_audio_quality), not from the Stage 1 `smp` field (which is the
    original recorder rate, not the extracted clip rate).
    """
    meta = {}
    for csv_file in Path(metadata_dir).glob("*.csv"):
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    xc_id = str(row.get("id", "")).strip()
                    if not xc_id:
                        continue
                    meta[xc_id] = {
                        "lat":            row.get("lat",  ""),
                        "lon":            row.get("lon",  ""),
                        "quality_grade":  row.get("q",    ""),
                        "recording_type": row.get("type", ""),
                        "country":        row.get("cnt",  ""),
                    }
        except Exception:
            continue
    return meta


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def generate_manifest(results, output_path, annotation_index, xc_metadata):
    """
    Write seabird_dataset_manifest.csv — one row per WAV clip.

    file_id format: XC{source_id}_{onset_ms}
    clip_index    : onset in milliseconds from start of source FLAC
    split         : empty column, populated at Stage 8
    """
    fieldnames = [
        "file_id", "source_id", "clip_index",
        "species_common", "species_scientific",
        "latitude", "longitude",
        "sampling_rate", "num_samples",
        "snr_db", "rms_db", "peak_amplitude", "is_clipped",
        "quality_grade", "recording_type", "country",
        "wav_filename", "split",
    ]

    rows = []
    for species in sorted(results):
        info = _COMMON_TO_INFO.get(species)
        species_common     = info[0] if info else species
        species_scientific = info[1] if info else ""

        for metrics in results[species]:
            wav_name = metrics["file"]          # e.g. xc1068442_3.wav
            stem = Path(wav_name).stem.lower()  # xc1068442_3

            xc_id   = ""
            seq_idx = None
            try:
                if stem.startswith("xc"):
                    parts = stem[2:].rsplit("_", 1)
                    xc_id   = parts[0]
                    seq_idx = int(parts[1]) if len(parts) == 2 else None
            except Exception:
                pass

            onset_ms = ""
            if xc_id and seq_idx is not None:
                onset_ms = annotation_index.get(xc_id, {}).get(seq_idx, "")

            if xc_id and onset_ms != "":
                file_id = f"XC{xc_id}_{onset_ms}"
            elif xc_id:
                file_id = f"XC{xc_id}_{seq_idx if seq_idx is not None else ''}"
            else:
                file_id = stem

            meta = xc_metadata.get(xc_id, {})
            snr  = metrics.get("snr_db")

            rows.append({
                "file_id":            file_id,
                "source_id":          xc_id,
                "clip_index":         onset_ms,
                "species_common":     species_common,
                "species_scientific": species_scientific,
                "latitude":           meta.get("lat", ""),
                "longitude":          meta.get("lon", ""),
                "sampling_rate":      metrics.get("sample_rate", ""),
                "num_samples":        metrics.get("num_samples", ""),
                "snr_db":             f"{snr:.2f}" if snr is not None else "",
                "rms_db":             f"{metrics.get('rms_db', 0.0):.2f}",
                "peak_amplitude":     f"{metrics.get('peak_amplitude', 0.0):.4f}",
                "is_clipped":         metrics.get("is_clipped", False),
                "quality_grade":      meta.get("quality_grade", ""),
                "recording_type":     meta.get("recording_type", ""),
                "country":            meta.get("country", ""),
                "wav_filename":       wav_name,
                "split":              "",   # Stage 8 fills this
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset manifest saved to: {output_path}  ({len(rows)} clips)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Quality control and manifest generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Stage7_quality_control_selection.py ./extracted_segments \\
      --output-dir ./dataset \\
      --annotation-dir ./sounds \\
      --metadata-dir ./xc_metadata_v3
        """
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(EXTRACTED_SEGS),
        help=f"Directory of extracted WAV segments organised by species subfolder. Default: {EXTRACTED_SEGS}",
    )
    parser.add_argument(
        "--output-dir", default=str(DATASET_DIR),
        help=f"Output directory for QC report and manifest. Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--annotation-dir", default=str(PER_SPECIES_FLACS),
        help=f"Directory containing Stage 5 .txt annotation files (searched recursively). "
             f"Used to resolve clip onset times (ms). Default: {PER_SPECIES_FLACS}",
    )
    parser.add_argument(
        "--metadata-dir", default=str(PER_SPECIES_CSV),
        help=f"Directory containing Stage 1 per-species CSV files. "
             f"Provides lat/lon/quality_grade/recording_type/country. Default: {PER_SPECIES_CSV}",
    )

    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    print("=" * 80)
    print("STAGE 7: QUALITY CONTROL & MANIFEST GENERATION")
    print("=" * 80)
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print()

    results = analyze_dataset(input_dir)

    if not results:
        print("No audio files found.")
        sys.exit(0)

    print()
    print_summary_report(results)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_qc_report(results, output_dir / "qc_report.csv")

    print()
    print("Loading annotation index and XC metadata...")
    annotation_index = load_annotation_index(args.annotation_dir) if args.annotation_dir else {}
    xc_metadata      = load_xc_metadata(args.metadata_dir)        if args.metadata_dir  else {}

    if not annotation_index:
        print("  Warning: no annotation dir / no .txt files found — clip_index will be empty.")
    if not xc_metadata:
        print("  Warning: no metadata dir / no CSVs found — lat/lon/quality/type/country will be empty.")

    generate_manifest(results, output_dir / "seabird_dataset_manifest.csv",
                      annotation_index, xc_metadata)

    print()
    print("Stage 7 complete.")


if __name__ == "__main__":
    main()
