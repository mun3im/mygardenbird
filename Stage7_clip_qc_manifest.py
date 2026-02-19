#!/usr/bin/env python3
"""
Stage 7: Quality control and manifest generation.

Analyzes extracted WAV segments and produces:
  - qc_report.csv    — per-clip quality metrics (diagnostic)
  - recordings.csv   — one row per source recording (PK: source_id)
                         fields: source_id, species_common, species_scientific,
                                 quality_grade, type_label, latitude, longitude, country
  - clips.csv        — one row per clip (PK: file_id, FK: source_id)
                         fields: file_id, source_id, clip_index,
                                 sampling_rate, snr_db, rms_db, peak_amplitude, is_clipped
                         wav_filename is derivable: xc{source_id}_{clip_index}.wav

Split assignments are created at Stage 8 as separate split CSV files keyed on file_id.

Quality checks performed:
  - Audio duration validation (~3 s)
  - Sample rate detection (whatever Stage 6 produced; reported, not filtered)
  - Clipping detection
  - File corruption detection
  - SNR estimation

Silent files are not expected at this stage (Stage 6 filters them);
silence is detected and reported but not used as a rejection criterion.
"""

import argparse
import csv
import sys
from collections import defaultdict

from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from config import ACTIVE_SPECIES, MYGARDENBIRD_16K, METADATA_16K, PER_SPECIES_CSV, RECORDINGS_CSV, normalise_type

_COMMON_TO_INFO = {common.lower(): (common, scientific, code)
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
# Stage 1 metadata
# ---------------------------------------------------------------------------

def load_xc_metadata(metadata_dir):
    """
    Load Stage 1 per-species CSVs.

    Returns dict: xc_id (str) -> {lat, lon, quality_grade, type_label, country}

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
                        "lat":           row.get("lat", ""),
                        "lon":           row.get("lon", ""),
                        "quality_grade": row.get("q",   ""),
                        "type_label":    normalise_type(row.get("type", "")),
                        "country":       row.get("cnt", ""),
                    }
        except Exception:
            continue
    return meta


# ---------------------------------------------------------------------------
# Normalised output tables
# ---------------------------------------------------------------------------

def _parse_stem(wav_name):
    """
    Extract (xc_id, onset_ms, file_id) from a wav filename stem.
    Returns ("", "", stem) on failure.
    """
    stem = Path(wav_name).stem.lower()   # e.g. "xc1002657_2860"
    xc_id = onset_ms = ""
    try:
        if stem.startswith("xc"):
            parts = stem[2:].rsplit("_", 1)
            if len(parts) == 2:
                xc_id, onset_ms = parts[0], parts[1]
    except Exception:
        pass
    file_id = f"XC{xc_id}_{onset_ms}" if (xc_id and onset_ms) else (f"XC{xc_id}" if xc_id else stem)
    return xc_id, onset_ms, file_id


def generate_recordings_csv(results, output_path, xc_metadata):
    """
    Write recordings.csv — one row per unique source recording (PK: source_id).

    Columns: source_id, species_common, species_scientific,
             quality_grade, type_label, latitude, longitude, country
    """
    fieldnames = [
        "source_id",
        "species_common", "species_scientific",
        "quality_grade", "type_label",
        "latitude", "longitude", "country",
    ]

    seen = {}   # source_id -> row (first clip wins; all clips from same source share metadata)
    for species in sorted(results):
        info = _COMMON_TO_INFO.get(species.lower())
        species_common     = info[0] if info else species
        species_scientific = info[1] if info else ""

        for metrics in results[species]:
            xc_id, _, _ = _parse_stem(metrics["file"])
            if not xc_id or xc_id in seen:
                continue
            meta = xc_metadata.get(xc_id, {})
            seen[xc_id] = {
                "source_id":          xc_id,
                "species_common":     species_common,
                "species_scientific": species_scientific,
                "quality_grade":      meta.get("quality_grade", ""),
                "type_label":         meta.get("type_label", ""),
                "latitude":           meta.get("lat", ""),
                "longitude":          meta.get("lon", ""),
                "country":            meta.get("country", ""),
            }

    rows = [seen[k] for k in sorted(seen)]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Recordings table saved to: {output_path}  ({len(rows)} source recordings)")


def generate_clips_csv(results, output_path):
    """
    Write clips.csv — one row per WAV clip (PK: file_id, FK: source_id).

    Columns: file_id, source_id, clip_index,
             sampling_rate, snr_db, rms_db, peak_amplitude, is_clipped

    wav_filename is derivable as: xc{source_id}_{clip_index}.wav
    Split assignments are stored in separate Stage 8 split files keyed on file_id.
    """
    fieldnames = [
        "file_id", "source_id", "clip_index",
        "sampling_rate",
        "snr_db", "rms_db", "peak_amplitude", "is_clipped",
    ]

    rows = []
    for species in sorted(results):
        for metrics in results[species]:
            xc_id, onset_ms, file_id = _parse_stem(metrics["file"])
            snr = metrics.get("snr_db")
            rows.append({
                "file_id":        file_id,
                "source_id":      xc_id,
                "clip_index":     onset_ms,
                "sampling_rate":  metrics.get("sample_rate", ""),
                "snr_db":         f"{snr:.2f}" if snr is not None else "",
                "rms_db":         f"{metrics.get('rms_db', 0.0):.2f}",
                "peak_amplitude": f"{metrics.get('peak_amplitude', 0.0):.4f}",
                "is_clipped":     metrics.get("is_clipped", False),
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Clips table saved to: {output_path}  ({len(rows)} clips)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Quality control and manifest generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Stage7_clip_qc_manifest.py
  python Stage7_clip_qc_manifest.py /path/to/extracted_segments --output-dir ./dataset
        """
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(MYGARDENBIRD_16K),
        help=f"Directory of extracted WAV segments organised by species subfolder. Default: {MYGARDENBIRD_16K}",
    )
    parser.add_argument(
        "--output-dir", default=str(METADATA_16K),
        help=f"Output directory for metadata CSVs (clips.csv, qc_report.csv). Default: {METADATA_16K}",
    )
    parser.add_argument(
        "--metadata-dir", default=str(PER_SPECIES_CSV),
        help=f"Directory containing Stage 1 per-species CSV files. "
             f"Provides lat/lon/quality_grade/recording_type/country. Default: {PER_SPECIES_CSV}",
    )
    parser.add_argument(
        "--recordings-csv", default=str(RECORDINGS_CSV),
        help=f"Path to top-level recordings.csv (shared by both 16kHz and 44kHz variants). Default: {RECORDINGS_CSV}",
    )

    args = parser.parse_args()

    input_dir       = Path(args.input_dir)
    output_dir      = Path(args.output_dir)
    recordings_path = Path(args.recordings_csv)

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
    print("Loading XC metadata...")
    xc_metadata = load_xc_metadata(args.metadata_dir) if args.metadata_dir else {}

    if not xc_metadata:
        print("  Warning: no metadata dir / no CSVs found — lat/lon/quality/type/country will be empty.")

    # Write recordings.csv to top level (shared by both 16kHz and 44kHz variants)
    # Only generate if it doesn't exist, to avoid re-parsing XC metadata repeatedly
    if not recordings_path.exists():
        print()
        print(f"Generating top-level recordings.csv at {recordings_path}...")
        generate_recordings_csv(results, recordings_path, xc_metadata)
    else:
        print()
        print(f"Using existing recordings.csv at {recordings_path}")

    # Write variant-specific metadata (clips.csv, qc_report.csv) to output_dir
    generate_clips_csv(results, output_dir / "clips.csv")

    print()
    print("Stage 7 complete.")


if __name__ == "__main__":
    main()
