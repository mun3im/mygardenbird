#!/usr/bin/env python3
"""
Stage4_annotate_segments.py — Interactive bird vocalization annotator.

Blob detection follows Sprengel et al. (2016) §2.1: STFT normalised to [0,1],
row×column median threshold, 4×4 erosion+dilation, column indicator vector,
two 4×1 smoothing dilation passes.
"""

import argparse
import json
import os
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import librosa
import librosa.display
from scipy.ndimage import binary_dilation, binary_erosion, label, find_objects

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation
import sounddevice as sd

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
    from config import PER_SPECIES_FLACS, PER_SPECIES_CSV, normalise_type
    DEFAULT_SOUND_DIR = str(PER_SPECIES_FLACS)
except ImportError:
    DEFAULT_SOUND_DIR = os.getcwd()
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
                           energy_thresh_db=-20):
    """
    Return (t_lo, t_hi, flo, fhi) tightly enclosing vocal energy in [start_s, end_s].
    Uses the display spectrogram (n_fft=2048) for accurate time and frequency resolution.
    Frames/bins within energy_thresh_db dB of the window peak are considered active.
    """
    total_frames = S_db_full.shape[1]
    t0 = max(0, int(start_s * sr / DISPLAY_HOP))
    t1 = min(total_frames, int(end_s * sr / DISPLAY_HOP) + 1)
    if t0 >= t1:
        return start_s, end_s, FREQ_MIN, float(freqs_full[-1])

    window = S_db_full[:, t0:t1]
    peak = window.max()
    thresh = peak + energy_thresh_db

    # Frequency bounds — rows with any active frame
    freq_active = (window.max(axis=1) >= thresh)
    freq_idx = np.where(freq_active)[0]
    if len(freq_idx) == 0:
        flo, fhi = FREQ_MIN, float(freqs_full[-1])
    else:
        flo = max(float(freqs_full[freq_idx[0]]), FREQ_MIN)
        fhi = float(freqs_full[freq_idx[-1]])
        pad = (fhi - flo) * 0.1
        flo = max(FREQ_MIN, flo - pad)
        fhi = min(float(freqs_full[-1]), fhi + pad)

    # Time bounds — columns with any active frequency bin
    time_active = (window.max(axis=0) >= thresh)
    col_idx = np.where(time_active)[0]
    if len(col_idx) == 0:
        t_lo, t_hi = start_s, end_s
    else:
        t_lo = (t0 + col_idx[0])  * DISPLAY_HOP / sr
        t_hi = (t0 + col_idx[-1]) * DISPLAY_HOP / sr

    return t_lo, t_hi, flo, fhi


def compute_freq_bounds(S_db_full, freqs_full, sr, start_s, end_s,
                        energy_thresh_db=-20):
    """Return (flo, fhi) only — used by create_fixed_segments for yellow box height."""
    _, _, flo, fhi = compute_display_bounds(
        S_db_full, freqs_full, sr, start_s, end_s, energy_thresh_db)
    return flo, fhi


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

        if len(final_segments) >= MAX_SEGMENTS:
            break

    return final_segments, final_bounds


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


# ── Main Interactive Function ─────────────────────────────────────────────────
def interactive_segment_detector(audio_path):
    """Main interactive visualization and annotation interface"""
    
    print(f"\nLoading: {os.path.basename(audio_path)}")
    y, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr
    
    if audio_duration < 3:
        print(f"File {audio_path} is too short ({audio_duration:.2f}s < 3s). Skipping.")
        return

    # Compute spectrograms
    print("\nComputing spectrograms...")
    S_det, S_db_full, freqs_full = compute_spectrograms(y, sr)

    # Initial detection (Sprengel §2.1)
    print("Running blob detection...")
    events_with_bounds, binary_mask = detect_sound_blobs(S_det, sr)
    detected_segs, detected_bounds = create_fixed_segments(
        events_with_bounds, audio_duration,
        S_db_full=S_db_full, freqs_full=freqs_full, sr=sr)
    print(f"Found {len(detected_segs)} candidate segments")

    # Load existing annotation if available
    ann_path = annotation_path_for(audio_path)
    has_annotation = os.path.exists(ann_path)
    title_suffix = ""

    if has_annotation:
        loaded = load_annotation(ann_path)
        if loaded:
            current_segments = list(loaded)
            current_bounds = [
                compute_freq_bounds(S_db_full, freqs_full, sr, s, e)
                for s, e in loaded
            ]
            title_suffix = "  [loaded from annotation]"
            print(f"Loaded {len(current_segments)} segment(s) from {ann_path}")
        else:
            current_segments = list(detected_segs)
            current_bounds = list(detected_bounds)
    else:
        current_segments = list(detected_segs)
        current_bounds = list(detected_bounds)

    # ── Setup Plot ────────────────────────────────────────────────────────────
    # Detect screen width and fill it
    _root = tk.Tk()
    _root.withdraw()
    _screen_w_px = _root.winfo_screenwidth()
    _screen_dpi  = _root.winfo_fpixels('1i')  # pixels per inch
    _root.destroy()
    _fig_w_in = _screen_w_px / _screen_dpi

    times_waveform = np.linspace(0, audio_duration, len(y))
    fig, (ax_spec, ax_wave) = plt.subplots(
        2, 1, figsize=(_fig_w_in, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True
    )
    fig.canvas.manager.set_window_title(os.path.basename(audio_path) + title_suffix)

    # Waveform plot
    ax_wave.plot(times_waveform, y, color='black', linewidth=0.5)
    ax_wave.set_title('Waveform — drag a segment to reposition', fontsize=10)
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_xlim(0, audio_duration)
    ax_wave.set_xlabel('Time (s)')
    ax_wave.grid(True, alpha=0.3)

    # Spectrogram — display slice up to DEFAULT_FREQ_CUTOFF initially
    _freq_cutoff = min(DEFAULT_FREQ_CUTOFF, sr / 2)
    _freq_slice = freqs_full <= _freq_cutoff
    img = ax_spec.imshow(S_db_full[_freq_slice, :], aspect='auto', origin='lower',
                         extent=[0, audio_duration, freqs_full[0], freqs_full[_freq_slice].max()],
                         cmap='plasma', vmin=-60, vmax=0)
    ax_spec.set_title('Spectrogram — click a yellow box to delete a segment', fontsize=10)
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_xlim(0, audio_duration)
    ax_spec.set_ylim(FREQ_MIN, _freq_cutoff)
    

    plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.36)

    # ── Widgets ──────────────────────────────────────────────────────────────
    # Layout: info bar at top, then 4 sliders spaced 0.055 apart, then buttons row
    SL_LEFT, SL_W, SL_H = 0.37, 0.43, 0.025
    ax_info          = plt.axes([0.37, 0.04, 0.43, 0.04])
    ax_energy_thresh = plt.axes([SL_LEFT, 0.275, SL_W, SL_H])
    ax_erode         = plt.axes([SL_LEFT, 0.220, SL_W, SL_H])
    ax_dilate        = plt.axes([SL_LEFT, 0.165, SL_W, SL_H])
    ax_freq_cutoff   = plt.axes([SL_LEFT, 0.110, SL_W, SL_H])
    ax_save      = plt.axes([0.82, 0.04, 0.08, 0.04])
    ax_quit      = plt.axes([0.01, 0.04, 0.08, 0.04])
    ax_play_stop = plt.axes([0.12, 0.04, 0.08, 0.04])

    slider_thresh = widgets.Slider(ax_energy_thresh, 'Threshold × median (lower=more sensitive)', 1.0, 6.0,
                                   valinit=INITIAL_THRESHOLD_MULT, valstep=0.25)
    slider_erode  = widgets.Slider(ax_erode,  'Erosion cycles (remove noise)', 0, 4,
                                   valinit=1, valstep=1)
    slider_dilate = widgets.Slider(ax_dilate, 'Dilation cycles (reconnect blobs)', 0, 4,
                                   valinit=1, valstep=1)
    _freq_steps = [v for v in MP3_NYQUIST_VALUES if FREQ_MIN < v <= sr / 2]
    if not _freq_steps:
        _freq_steps = [int(sr / 2)]
    _freq_init = min(_freq_steps, key=lambda v: abs(v - DEFAULT_FREQ_CUTOFF))
    slider_freq_cutoff = widgets.Slider(ax_freq_cutoff, 'Max Freq Display (Hz)',
                                        _freq_steps[0], _freq_steps[-1],
                                        valinit=_freq_init, valstep=_freq_steps)
    save_button = widgets.Button(ax_save, 'Save')
    quit_button = widgets.Button(ax_quit, 'Quit')
    play_stop_button = widgets.Button(ax_play_stop, 'Play')
    ax_info.axis('off')

    # Information display
    snr_metrics = calculate_enhanced_snr(y, sr, current_segments)
    info_text = ax_info.text(
        0.5, 0.5,
        f"Avg SNR: {snr_metrics['average_snr']:.2f} dB | "
        f"Peak SNR: {snr_metrics['peak_snr']:.2f} dB | "
        f"Segments: {len(current_segments)}",
        ha='center', va='center', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7))

    # ── Drawing Helpers ──────────────────────────────────────────────────────
    wave_segment_patches = []
    spec_segment_rects = []
    event_rects = []

    def update_detection(val=None):
        nonlocal binary_mask, events_with_bounds

        print("  Re-running blob detection...")
        events_with_bounds, binary_mask = detect_sound_blobs(
            S_det, sr,
            threshold_mult=slider_thresh.val,
            erode_cycles=int(slider_erode.val),
            dilate_cycles=int(slider_dilate.val),
        )

        new_segs, new_bnds = create_fixed_segments(
            events_with_bounds, audio_duration,
            S_db_full=S_db_full, freqs_full=freqs_full, sr=sr)
        current_segments[:] = new_segs
        current_bounds[:] = new_bnds
        update_segments()
        print(f"  Found {len(new_segs)} segments")

    def update_segments():
        """Redraw segments on waveform and spectrogram"""
        for p in wave_segment_patches:
            p.remove()
        for r in spec_segment_rects:
            r.remove()
        for r in event_rects:
            r.remove()
        wave_segment_patches.clear()
        spec_segment_rects.clear()
        event_rects.clear()

        # Cyan boxes: Sprengel blob extents refined via display STFT
        for (start, end, _fmin, _fmax) in events_with_bounds:
            t_lo, t_hi, flo, fhi = compute_display_bounds(
                S_db_full, freqs_full, sr, start, end)
            r = patches.Rectangle((t_lo, flo), t_hi - t_lo, fhi - flo,
                                   linewidth=1.2, edgecolor='cyan', facecolor='none',
                                   linestyle='--', alpha=0.8, zorder=4)
            ax_spec.add_patch(r)
            event_rects.append(r)

        # Draw final segments
        for i, (start, end) in enumerate(current_segments):
            # Waveform highlight
            span = ax_wave.axvspan(start, end, alpha=0.4, color='deepskyblue')
            span.segment_idx = i
            wave_segment_patches.append(span)
            
            # Spectrogram frequency bounds
            if i < len(current_bounds):
                fmin, fmax = current_bounds[i]
                # Fill
                rect = patches.Rectangle((start, fmin), SEGMENT_DURATION, fmax - fmin,
                                        linewidth=2, edgecolor='yellow', facecolor='yellow', 
                                        alpha=0.3)
                ax_spec.add_patch(rect)
                spec_segment_rects.append(rect)
                # Outline
                outline = patches.Rectangle((start, fmin), SEGMENT_DURATION, fmax - fmin,
                                           linewidth=2, edgecolor='yellow', facecolor='none',
                                           alpha=0.9)
                ax_spec.add_patch(outline)
                spec_segment_rects.append(outline)

        snr = calculate_enhanced_snr(y, sr, current_segments)
        info_text.set_text(
            f"Avg SNR: {snr['average_snr']:.2f} dB | "
            f"Peak SNR: {snr['peak_snr']:.2f} dB | "
            f"Segments: {len(current_segments)}"
        )
        fig.canvas.draw_idle()

    # Initial display
    update_segments()

    # ── Interactive Editing ──────────────────────────────────────────────────
    drag_state = {'idx': None, 'start_x': None, 'original_start': None, 'axis': None}

    def _find_segment_at(x):
        for i, (s, e) in enumerate(current_segments):
            if s <= x <= e: 
                return i
        return None

    def on_press(event):
        if event.inaxes not in (ax_wave, ax_spec) or event.button != 1 or event.xdata is None: 
            return
        idx = _find_segment_at(event.xdata)
        if idx is not None:
            drag_state.update(idx=idx, start_x=event.xdata,
                              original_start=current_segments[idx][0], axis=event.inaxes)

    def on_motion(event):
        idx = drag_state['idx']
        if idx is None or drag_state['axis'] != ax_wave or event.xdata is None: 
            return
        dx = event.xdata - drag_state['start_x']
        new_start = drag_state['original_start'] + dx
        
        left_limit = current_segments[idx - 1][1] if idx > 0 else 0.0
        right_limit = (current_segments[idx + 1][0] - SEGMENT_DURATION
                       if idx < len(current_segments) - 1 else audio_duration - SEGMENT_DURATION)
        new_start = max(left_limit, min(new_start, right_limit))
        
        if idx < len(wave_segment_patches):
            wave_segment_patches[idx].set_x(new_start)
            # Update corresponding spectrogram rectangles (2 per segment)
            rect_idx = idx * 2
            if rect_idx + 1 < len(spec_segment_rects):
                spec_segment_rects[rect_idx].set_x(new_start)
                spec_segment_rects[rect_idx + 1].set_x(new_start)
        
        fig.canvas.draw_idle()

    def on_release(event):
        idx = drag_state['idx']
        if idx is None: 
            return
            
        total_move = (abs(event.xdata - drag_state['start_x'])
                      if drag_state['axis'] == ax_wave and event.xdata is not None else 0.0)
        
        if total_move < 0.05 and drag_state['axis'] == ax_spec:
            print(f"Removing segment {idx}")
            del current_segments[idx]
            del current_bounds[idx]
            update_segments()
        elif total_move >= 0.05 and drag_state['axis'] == ax_wave:
            dx = (event.xdata - drag_state['start_x'] if event.xdata is not None else 0.0)
            new_start = drag_state['original_start'] + dx
            left_limit = current_segments[idx - 1][1] if idx > 0 else 0.0
            right_limit = (current_segments[idx + 1][0] - SEGMENT_DURATION
                           if idx < len(current_segments) - 1 else audio_duration - SEGMENT_DURATION)
            new_start = max(left_limit, min(new_start, right_limit))
            current_segments[idx] = (new_start, new_start + SEGMENT_DURATION)
            print(f"Moved segment {idx} to {new_start:.3f}s")
            update_segments()
            
        drag_state.update(idx=None, start_x=None, original_start=None, axis=None)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # ── Playback ─────────────────────────────────────────────────────────────
    progress_line_spec = ax_spec.axvline(x=0, color='red', linewidth=2, visible=False)
    progress_line_wave = ax_wave.axvline(x=0, color='red', linewidth=2, visible=False)
    is_playing = False
    play_start_time = None
    progress_animation = None

    def stop_playback():
        nonlocal is_playing, progress_animation
        sd.stop()
        is_playing = False
        play_stop_button.label.set_text('Play')
        progress_line_spec.set_visible(False)
        progress_line_wave.set_visible(False)
        if progress_animation:
            progress_animation.event_source.stop()
            progress_animation = None
        fig.canvas.draw_idle()

    def update_frame(frame):
        if not is_playing: 
            return []
        elapsed = time.time() - play_start_time
        if elapsed >= audio_duration: 
            stop_playback()
            return []
        progress_line_spec.set_xdata([elapsed, elapsed])
        progress_line_wave.set_xdata([elapsed, elapsed])
        return [progress_line_spec, progress_line_wave]

    def toggle_play_stop(event):
        nonlocal is_playing, play_start_time, progress_animation
        if is_playing:
            stop_playback()
        else:
            sd.play(y, sr)
            play_start_time = time.time()
            play_stop_button.label.set_text('Stop')
            is_playing = True
            progress_line_spec.set_visible(True)
            progress_line_wave.set_visible(True)
            progress_line_spec.set_xdata([0, 0])
            progress_line_wave.set_xdata([0, 0])
            progress_animation = FuncAnimation(fig, update_frame, interval=50, blit=True, cache_frame_data=False)
            fig.canvas.draw_idle()

    play_stop_button.on_clicked(toggle_play_stop)

    # ── Slider Callbacks ─────────────────────────────────────────────────────
    slider_thresh.on_changed(update_detection)
    slider_erode.on_changed(update_detection)
    slider_dilate.on_changed(update_detection)
    def update_freq_cutoff(val):
        cutoff = slider_freq_cutoff.val
        fs = freqs_full <= cutoff
        img.set_data(S_db_full[fs, :])
        img.set_extent([0, audio_duration, freqs_full[0], freqs_full[fs].max()])
        ax_spec.set_ylim(FREQ_MIN, cutoff)
        fig.canvas.draw_idle()

    slider_freq_cutoff.on_changed(update_freq_cutoff)

    # ── Save / Quit ──────────────────────────────────────────────────────────
    def save_segments(event):
        txt = annotation_path_for(audio_path)
        final_snr = calculate_enhanced_snr(y, sr, current_segments)
        xc_type = lookup_xc_type(audio_path)
        
        with open(txt, 'w') as f:
            for start, end in current_segments[:MAX_SEGMENTS]:
                f.write(f"{start:.6f}\t{end:.6f}\t{xc_type}\n")
        
        print(f"\n✓ Saved {len(current_segments)} segment(s) to {txt}")
        print(f"  Avg SNR: {final_snr['average_snr']:.2f} dB")
        print(f"  Peak SNR: {final_snr['peak_snr']:.2f} dB")
        plt.close(fig)

    def quit_without_saving(event):
        print("\nCancelled without saving.")
        plt.close(fig)

    save_button.on_clicked(save_segments)
    quit_button.on_clicked(quit_without_saving)
    
    print("\n" + "="*60)
    print("INTERACTIVE CONTROLS:")
    print("  • Threshold × median: Lower = more sensitive (Sprengel multiplier N)")
    print("  • Erosion cycles:     Remove noise / shrink blobs (0=none, 4=aggressive)")
    print("  • Dilation cycles:    Reconnect / expand blobs   (0=none, 4=aggressive)")
    print("  • Max Freq Display:   Pan the spectrogram frequency axis")
    print("  • Waveform:     drag a segment to reposition")
    print("  • Spectrogram:  click a yellow box to delete a segment")
    print("  • Press 'Play' to listen to the recording")
    print("="*60 + "\n")
    
    plt.show()


# ── Entry Point ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive bird vocalization segment detector using blob analysis")
    parser.add_argument("--sound-dir", default=DEFAULT_SOUND_DIR,
                        help=f"Path to sound files folder (default: '{DEFAULT_SOUND_DIR}')")
    parser.add_argument("--file", type=str, help="Direct path to audio file (skip file dialog)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("="*80)
    print("SPRENGEL (2016) BLOB DETECTION — Stage4_annotate_segments.py")
    print("="*80)
    print("HOW IT WORKS (Sprengel et al. 2016, §2.1):")
    print("  1. STFT (Hanning, n_fft=512, hop=128) normalised to [0,1]")
    print("  2. Binary mask: cell=1 iff value > N×row_median AND > N×col_median")
    print("  3. Erosion (4×4) + Dilation (4×4) to remove noise / reconnect blobs")
    print("  4. Column indicator vector: active if any row in that frame is 1")
    print("  5. Two 4×1 smoothing dilation passes to fill short gaps")
    print("  6. Contiguous active columns → sound events (blue overlay)")
    print("  7. Each event padded to fixed 3s segment (yellow boxes)")
    print("="*80)

    _STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".stage4_state.json")

    def _load_last_dir():
        try:
            with open(_STATE_FILE) as _f:
                return json.load(_f).get("last_dir", "")
        except (FileNotFoundError, ValueError):
            return ""

    def _save_last_dir(path):
        try:
            with open(_STATE_FILE, "w") as _f:
                json.dump({"last_dir": path}, _f)
        except OSError:
            pass

    if args.file and os.path.isfile(args.file):
        audio_file = args.file
        _save_last_dir(os.path.dirname(os.path.abspath(audio_file)))
        print(f"Processing: {audio_file}")
        interactive_segment_detector(audio_file)
    else:
        sound_dir = os.path.abspath(args.sound_dir) if os.path.exists(args.sound_dir) else os.getcwd()
        last_dir = _load_last_dir()
        initial_dir = last_dir if last_dir and os.path.isdir(last_dir) else sound_dir

        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)

        audio_file = filedialog.askopenfilename(
            title="Select audio file", 
            initialdir=initial_dir,
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma")]
        )

        root.update_idletasks()
        root.quit()
        root.destroy()

        if audio_file:
            _save_last_dir(os.path.dirname(os.path.abspath(audio_file)))
            print(f"Processing: {audio_file}")
            interactive_segment_detector(audio_file)
        else:
            print("No file selected.")