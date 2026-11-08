#!/usr/bin/env python3
"""
stage4_core.py — Shared, GUI-free annotation logic for Stage 4.

Extracted verbatim from Stage4_annotate_segments.py so that the matplotlib
annotator and the Qt annotator (Stage4b_annotate_qt.py) run identical
detection, SNR and annotation-I/O code. Contains no GUI dependencies.

Blob detection follows Sprengel et al. (2016) §2.1: STFT normalised to [0,1],
row×column median threshold, 4×4 erosion+dilation, column indicator vector,
two 4×1 smoothing dilation passes.
"""

import os
import warnings

import pandas as pd
import numpy as np
import librosa
from scipy.ndimage import binary_dilation, binary_erosion

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# ── Configuration ─────────────────────────────────────────────────────────────
FREQ_MIN               = 200      # Hz - minimum frequency for display / analysis
DEFAULT_FREQ_CUTOFF    = 8000     # Hz — default max frequency for spectrogram display
# Nyquist frequencies for standard MP3 sample rates (8, 11.025, 12, 16, 22.05, 24, 32, 44.1, 48 kHz)
MP3_NYQUIST_VALUES     = [4000, 5512, 6000, 8000, 11025, 12000, 16000, 22050, 24000]

# Sprengel (2016) §2.1 detection STFT: window 512, 75% overlap
SPRENGEL_N_FFT         = 512
SPRENGEL_HOP           = 128      # 75% overlap

# Display STFT (larger window, better frequency resolution for the viewer)
DISPLAY_N_FFT          = 2048
DISPLAY_HOP            = DISPLAY_N_FFT // 4

SEGMENT_DURATION       = 3.0      # seconds — fixed clip length
MAX_SEGMENTS           = 10

# Sprengel threshold multiplier (cells must exceed N × row_median AND N × col_median)
INITIAL_THRESHOLD_MULT = 3.0      # lower = more sensitive

# Handle config import gracefully
try:
    from config import PER_SPECIES_FLACS, PER_SPECIES_CSV, normalise_type, get_profile
    DEFAULT_SOUND_DIR = str(PER_SPECIES_FLACS)
except ImportError:
    DEFAULT_SOUND_DIR = os.getcwd()
    get_profile = None
    def normalise_type(t):
        return t.lower() if t else "birdsong"


# ── XC metadata type lookup ──────────────────────────────────────────────────
def _build_xc_type_map():
    xc_map = {}
    try:
        csv_dir = PER_SPECIES_CSV
        if csv_dir and csv_dir.exists():
            for csv_file in csv_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, usecols=["id", "type"])
                    for _, row in df.iterrows():
                        try:
                            xc_map[int(row["id"])] = normalise_type(str(row["type"]) if pd.notna(row["type"]) else "")
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    pass
    except NameError:
        pass
    return xc_map

_XC_TYPE_MAP = _build_xc_type_map()

def lookup_xc_type(audio_path):
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    if stem.startswith("xc"):
        try:
            xc_id = int(stem[2:])
            return _XC_TYPE_MAP.get(xc_id, "birdsong")
        except ValueError:
            pass
    return "birdsong"


# ── SNR Calculation ──────────────────────────────────────────────────────────
def calculate_enhanced_snr(y, sr, segments, fade_buffer=0.1):
    """Calculate SNR for detected segments"""
    if not segments:
        return {"average_snr": float('-inf'), "peak_snr": float('-inf'), "segment_snrs": []}
    
    segment_indices = []
    fade_samples = int(fade_buffer * sr)
    for start_time, end_time in segments:
        start_idx = int(start_time * sr)
        end_idx = min(int(end_time * sr), len(y))
        if end_idx - start_idx <= 2 * fade_samples:
            segment_indices.append((start_idx, end_idx))
        else:
            segment_indices.append((start_idx + fade_samples, end_idx - fade_samples))

    signal_mask = np.zeros_like(y, dtype=bool)
    for s, e in segment_indices:
        if s < e: 
            signal_mask[s:e] = True

    noise_samples = y[~signal_mask]
    if len(noise_samples) == 0:
        return {"average_snr": float('inf'), "peak_snr": float('inf'), "segment_snrs": []}

    noise_power = np.mean(noise_samples ** 2)
    if noise_power == 0:
        noise_power = 1e-10
        
    segment_powers, segment_snrs = [], []
    for s, e in segment_indices:
        if s >= e: 
            continue
        seg = y[s:e]
        if len(seg) == 0: 
            continue
        p = np.mean(seg ** 2)
        segment_powers.append(p)
        segment_snrs.append(10 * np.log10((p + 1e-10) / (noise_power + 1e-10)))

    if not segment_powers:
        return {"average_snr": float('-inf'), "peak_snr": float('-inf'), "segment_snrs": []}

    return {
        "average_snr": 10 * np.log10((np.mean(segment_powers) + 1e-10) / (noise_power + 1e-10)),
        "peak_snr": 10 * np.log10((max(segment_powers) + 1e-10) / (noise_power + 1e-10)),
        "segment_snrs": segment_snrs,
    }


# ── Spectrogram computation ───────────────────────────────────────────────────
def compute_spectrograms(y, sr):
    """
    Return two spectrograms:
    - S_det:      Sprengel detection spectrogram (n_fft=512, hop=128, normalised to [0,1])
    - S_db_full:  Display spectrogram, full bandwidth FREQ_MIN–fs/2 (n_fft=2048, log-power dB)
    - freqs_full: Frequency axis for S_db_full (Hz)
    """
    # Detection spectrogram — Sprengel §2.1 parameters
    S_det = np.abs(librosa.stft(y, n_fft=SPRENGEL_N_FFT, hop_length=SPRENGEL_HOP))
    max_val = S_det.max()
    if max_val > 0:
        S_det /= max_val  # normalise to [0, 1] — do NOT log

    # Display spectrogram — full bandwidth so the freq slider can pan freely
    S_disp = np.abs(librosa.stft(y, n_fft=DISPLAY_N_FFT, hop_length=DISPLAY_HOP))
    disp_freqs = librosa.fft_frequencies(sr=sr, n_fft=DISPLAY_N_FFT)
    freq_mask = disp_freqs >= FREQ_MIN
    S_db_full = librosa.power_to_db(S_disp[freq_mask, :] ** 2, ref=np.max)
    freqs_full = disp_freqs[freq_mask]

    return S_det, S_db_full, freqs_full


# ── BLOB-BASED DETECTION — Sprengel (2016) §2.1 ──────────────────────────────
def detect_sound_blobs(S_det, sr, threshold_mult=INITIAL_THRESHOLD_MULT,
                       erode_cycles=1, dilate_cycles=1):
    """
    Sprengel et al. (2016) §2.1 signal/noise separation:

    1. Each cell is 1 iff value > threshold_mult × row_median
                           AND value > threshold_mult × col_median.
    2. Binary erosion with 4×4 element (erode_cycles passes).
    3. Binary dilation with 4×4 element (dilate_cycles passes).
    4. Column indicator vector: col active if any row is 1.
    5. Two smoothing binary dilation passes with 4×1 (time-only) element.
    6. Extract contiguous active runs → (start_s, end_s) events.

    Returns (events_with_bounds, binary_mask_2d) where events_with_bounds is
    a list of (start_s, end_s, fmin_hz, fmax_hz).
    binary_mask_2d has the same shape as S_det and is used for the blue overlay
    (re-projected onto the display spectrogram time axis by the caller).
    """
    # 1. Row × column median threshold
    row_med = np.median(S_det, axis=1, keepdims=True)  # shape (F, 1)
    col_med = np.median(S_det, axis=0, keepdims=True)  # shape (1, T)
    binary_mask = (S_det > threshold_mult * row_med) & (S_det > threshold_mult * col_med)

    # 2–3. Morphological cleaning with 4×4 structuring element
    struct_4x4 = np.ones((4, 4), dtype=bool)
    for _ in range(erode_cycles):
        binary_mask = binary_erosion(binary_mask, structure=struct_4x4)
    for _ in range(dilate_cycles):
        binary_mask = binary_dilation(binary_mask, structure=struct_4x4)

    # 4. Column indicator vector
    col_active = binary_mask.any(axis=0)  # shape (T,)

    # 5. Two smoothing dilation passes with 4×1 (time-only) element
    struct_4x1 = np.ones((4,), dtype=bool)
    for _ in range(2):
        col_active = binary_dilation(col_active, structure=struct_4x1)

    # 6. Extract contiguous runs of active columns → time intervals
    time_frames = S_det.shape[1]
    time_axis = np.arange(time_frames) * SPRENGEL_HOP / sr

    raw_events = []
    in_run = False
    run_start = 0
    for t, active in enumerate(col_active):
        if active and not in_run:
            run_start = t
            in_run = True
        elif not active and in_run:
            raw_events.append((time_axis[run_start], time_axis[t - 1]))
            in_run = False
    if in_run:
        raw_events.append((time_axis[run_start], time_axis[min(time_frames - 1, time_frames - 1)]))

    # Frequency bounds: use the 2-D binary mask rows for each event
    events_with_bounds = []
    det_freqs = librosa.fft_frequencies(sr=sr, n_fft=SPRENGEL_N_FFT)

    for start, end in raw_events:
        t0 = max(0, int(start * sr / SPRENGEL_HOP))
        t1 = min(time_frames, int(end * sr / SPRENGEL_HOP) + 1)
        if t0 >= t1:
            events_with_bounds.append((start, end, FREQ_MIN, sr / 2))
            continue

        active_rows = binary_mask[:, t0:t1].any(axis=1)
        active_freqs = det_freqs[active_rows]
        if len(active_freqs) == 0:
            events_with_bounds.append((start, end, FREQ_MIN, sr / 2))
        else:
            flo = max(float(active_freqs[0]), FREQ_MIN)
            fhi = min(float(active_freqs[-1]), sr / 2)
            pad = (fhi - flo) * 0.1
            events_with_bounds.append((start, end,
                                        max(FREQ_MIN, flo - pad),
                                        min(sr / 2, fhi + pad)))

    return events_with_bounds, binary_mask

def compute_display_bounds(S_db_full, freqs_full, sr, start_s, end_s,
                           drop_db=20):
    """
    Return (t_lo, t_hi, flo, fhi) tightly enclosing vocal energy in [start_s, end_s].
    Uses the display spectrogram (n_fft=2048, ref=global max, so values are ≤ 0 dB).
    Threshold = window_peak - drop_db (relative to the local window peak, not global).
    Frequency: rows whose max across the window is within drop_db of the window peak.
    Time:      columns whose max across frequency is within drop_db of the window peak.
    """
    total_frames = S_db_full.shape[1]
    t0 = max(0, int(start_s * sr / DISPLAY_HOP))
    t1 = min(total_frames, int(end_s * sr / DISPLAY_HOP) + 1)
    if t0 >= t1:
        return start_s, end_s, FREQ_MIN, float(freqs_full[-1])

    window = S_db_full[:, t0:t1]           # shape (F, T_win)
    win_peak = window.max()                 # local peak (≤ 0 dB)
    thresh = win_peak - drop_db             # e.g. peak=-3 → thresh=-23

    # Frequency bounds
    freq_active = window.max(axis=1) >= thresh
    freq_idx = np.where(freq_active)[0]
    if len(freq_idx) == 0:
        flo, fhi = FREQ_MIN, float(freqs_full[-1])
    else:
        flo = max(float(freqs_full[freq_idx[0]]), FREQ_MIN)
        fhi = float(freqs_full[freq_idx[-1]])
        pad = (fhi - flo) * 0.1
        flo = max(FREQ_MIN, flo - pad)
        fhi = min(float(freqs_full[-1]), fhi + pad)

    # Time bounds
    time_active = window.max(axis=0) >= thresh
    col_idx = np.where(time_active)[0]
    if len(col_idx) == 0:
        t_lo, t_hi = start_s, end_s
    else:
        t_lo = (t0 + col_idx[0])  * DISPLAY_HOP / sr
        t_hi = (t0 + col_idx[-1]) * DISPLAY_HOP / sr

    return t_lo, t_hi, flo, fhi


def compute_freq_bounds(S_db_full, freqs_full, sr, start_s, end_s, drop_db=20):
    """Return (flo, fhi) only — used by create_fixed_segments for yellow box height."""
    _, _, flo, fhi = compute_display_bounds(
        S_db_full, freqs_full, sr, start_s, end_s, drop_db)
    return flo, fhi


def spread_segments(segments, bounds, limit=MAX_SEGMENTS):
    """Thin segments down to `limit`, spread evenly across the recording.

    Taking the first `limit` candidates would cluster every clip at the start:
    a long recording can yield dozens of non-overlapping events, and the
    earliest ten of them often span only the opening fraction of the audio
    even when the bird calls throughout. Clips drawn from one short stretch
    share the same background, distance and conditions, which is the
    correlation that sampling across the recording is meant to avoid.

    Indices are picked evenly across the candidate list, always including the
    first and last. Fewer candidates than `limit` are returned untouched.
    """
    n = len(segments)
    if n <= limit:
        return segments, bounds
    if limit <= 0:
        return [], []
    if limit == 1:
        return segments[:1], bounds[:1]

    picks = sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})
    # Rounding can collide on short lists; backfill so exactly `limit` come
    # back whenever the pool is big enough to supply them.
    if len(picks) < limit:
        for i in range(n):
            if len(picks) >= limit:
                break
            if i not in picks:
                picks.append(i)
        picks = sorted(picks)

    out_segs = [segments[i] for i in picks]
    out_bnds = [bounds[i] for i in picks if i < len(bounds)]
    return out_segs, out_bnds


def create_fixed_segments(events_with_bounds, audio_duration,
                          S_db_full=None, freqs_full=None, sr=None):
    """Create fixed 3s segments centered on detected events."""
    half_seg = SEGMENT_DURATION / 2.0
    fixed_segs = []
    fixed_bounds = []

    for start, end, fmin, fmax in events_with_bounds:
        center = (start + end) / 2.0
        bs = max(0, center - half_seg)
        be = bs + SEGMENT_DURATION
        if be > audio_duration:
            be = audio_duration
            bs = max(0, be - SEGMENT_DURATION)

        if S_db_full is not None:
            fmin, fmax = compute_freq_bounds(S_db_full, freqs_full, sr, bs, be)

        fixed_segs.append((bs, be))
        fixed_bounds.append((fmin, fmax))

    # Deduplicate & enforce non-overlap
    fixed_segs.sort(key=lambda x: x[0])
    final_segments, final_bounds = [], []
    last_end = -SEGMENT_DURATION

    for i, (s, e) in enumerate(fixed_segs):
        if s >= last_end and abs(e - s - SEGMENT_DURATION) < 0.01:
            final_segments.append((s, e))
            if i < len(fixed_bounds):
                final_bounds.append(fixed_bounds[i])
            last_end = e

    # Detection runs over the whole recording, then the candidates are thinned
    # to MAX_SEGMENTS by spreading rather than truncating -- see
    # spread_segments() for why the first N would be the wrong N.
    return spread_segments(final_segments, final_bounds)


# ── Annotation I/O ───────────────────────────────────────────────────────────
def annotation_path_for(audio_path):
    return os.path.splitext(audio_path)[0] + '.txt'

def load_annotation(txt_path):
    segments = []
    try:
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): 
                    continue
                parts = line.split('\t')
                if len(parts) < 2: 
                    continue
                segments.append((float(parts[0]), float(parts[1])))
    except Exception as e:
        print(f"  Warning: could not read annotation {txt_path}: {e}")
    return segments


