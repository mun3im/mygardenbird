#!/usr/bin/env python3
"""
STAGE 5: BLOB-BASED SEGMENT DETECTION (Fixed Index Bug)
========================================================
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
from scipy.signal import medfilt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

import sounddevice as sd

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# ── Configuration ─────────────────────────────────────────────────────────────
FREQ_MIN               = 200      # Hz - minimum frequency for analysis
FREQ_MAX               = 8000     # Hz - maximum frequency for analysis
DEFAULT_FREQ_CUTOFF    = 8000     # Hz — default max frequency for spectrogram display
STFT_FRAME_SIZE        = 2048
HOP_LENGTH             = STFT_FRAME_SIZE // 4  # 512
SEGMENT_DURATION       = 3.0      # seconds — fixed clip length
MAX_SEGMENTS           = 10

# Default thresholds
INITIAL_ENERGY_THRESHOLD = 0.3    # 0-1, higher = less sensitive
INITIAL_MIN_BLOB_DURATION = 0.15  # seconds
INITIAL_MAX_BLOB_GAP = 0.6        # seconds - max gap to merge blobs

# PCEN parameters
PCEN_TIME_CONSTANT     = 0.09     # seconds
PCEN_EPS               = 1e-6
PCEN_POWER             = 0.25
PCEN_GAIN              = 0.98

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


# ── PCEN Spectrogram ─────────────────────────────────────────────────────────
def compute_pcen_spectrogram(y, sr, n_fft=STFT_FRAME_SIZE, hop_length=HOP_LENGTH,
                              fmin=FREQ_MIN, fmax=FREQ_MAX):
    """Compute spectrogram: PCEN for detection, plain dB for display."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # PCEN — used only for blob detection
    S_pcen = librosa.pcen(S, sr=sr, hop_length=hop_length,
                          time_constant=PCEN_TIME_CONSTANT,
                          eps=PCEN_EPS, power=PCEN_POWER, gain=PCEN_GAIN)

    # Slice to frequency range of interest
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    S_pcen_masked = S_pcen[freq_mask, :]
    freqs_masked = freqs[freq_mask]

    # Plain log-power dB for display — matches Audacity's appearance
    S_db = librosa.power_to_db(S[freq_mask, :] ** 2, ref=np.max)

    return S_pcen_masked, S_db, freqs_masked, freq_mask


# ── BLOB-BASED DETECTION ─────────────────────────────────────────────────────
def detect_sound_blobs(S_pcen, sr, freqs_sliced=None, hop_length=HOP_LENGTH,
                       energy_threshold=INITIAL_ENERGY_THRESHOLD,
                       erode_cycles=2, dilate_cycles=2):
    """
    Detect sound events using per-band adaptive thresholding + spectral flux.

    Process:
    1. Per-frequency-band median subtraction (removes steady noise per band)
    2. Spectral flux (frame-to-frame difference) highlights onsets
    3. Combine: band-normalised PCEN weighted by flux
    4. Binary threshold (user controlled)
    5. Erode N / Dilate M
    6. Find connected components
    """

    # freqs_sliced corresponds to the rows of S_pcen (already frequency-masked)
    if freqs_sliced is None:
        freqs_sliced = librosa.fft_frequencies(sr=sr, n_fft=STFT_FRAME_SIZE)
        freq_mask = (freqs_sliced >= FREQ_MIN) & (freqs_sliced <= FREQ_MAX)
        freqs_sliced = freqs_sliced[freq_mask]

    # 1. Per-band median subtraction — each frequency row has its own baseline removed
    band_median = np.median(S_pcen, axis=1, keepdims=True)
    S_denoised = np.maximum(S_pcen - band_median, 0)

    # 2. Spectral flux — forward difference across time, per band
    flux = np.diff(S_denoised, axis=1, prepend=S_denoised[:, :1])
    flux = np.maximum(flux, 0)  # keep only increases (onsets)

    # 3. Combine denoised energy with flux to favour transient events
    S_combined = S_denoised * (1.0 + flux)

    # 4. Normalize to 0-1 and threshold
    S_norm = (S_combined - S_combined.min()) / (S_combined.max() - S_combined.min() + 1e-8)
    binary_mask = S_norm > energy_threshold

    # 5. Morphological operations — erode and dilate independently
    structure = np.ones((3, 3))

    for _ in range(erode_cycles):
        binary_mask = binary_erosion(binary_mask, structure=structure)
    for _ in range(dilate_cycles):
        binary_mask = binary_dilation(binary_mask, structure=structure)

    # 4. Find connected components (blobs)
    labeled_mask, num_blobs = label(binary_mask)
    blob_slices = find_objects(labeled_mask)

    # 5. Extract temporal bounds from each blob
    time_frames = S_pcen.shape[1]
    time_axis = np.arange(time_frames) * hop_length / sr

    raw_events = []

    for blob_slice in blob_slices:
        if blob_slice is None:
            continue

        t_start = max(0, blob_slice[1].start)
        t_end = min(time_frames, blob_slice[1].stop)

        start_time = time_axis[t_start]
        end_time = time_axis[t_end - 1] if t_end > t_start else time_axis[t_start]
        raw_events.append((start_time, end_time))

    # 6. Merge events that are adjacent (blobs touching in time)
    if len(raw_events) > 1:
        raw_events.sort(key=lambda x: x[0])
        merged_events = []
        current_start, current_end = raw_events[0]

        for start, end in raw_events[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_events.append((current_start, current_end))
                current_start, current_end = start, end
        merged_events.append((current_start, current_end))
        raw_events = merged_events

    # 7. Refine frequency bounds for each event
    events_with_bounds = []

    for start, end in raw_events:
        # Find time indices
        t_start = max(0, int(start * sr / hop_length))
        t_end = min(time_frames, int(end * sr / hop_length) + 1)

        if t_start >= t_end:
            events_with_bounds.append((start, end, FREQ_MIN, FREQ_MAX))
            continue

        # Use denoised energy (per-band median removed) for frequency bounds
        energy_per_freq = np.mean(S_denoised[:, t_start:t_end], axis=1)
        peak_energy = np.max(energy_per_freq)

        if peak_energy <= 0:
            events_with_bounds.append((start, end, FREQ_MIN, FREQ_MAX))
            continue

        # Find frequency bins with significant energy (above 25% of peak)
        threshold = peak_energy * 0.25
        active_bins = np.where(energy_per_freq > threshold)[0]

        if len(active_bins) == 0:
            events_with_bounds.append((start, end, FREQ_MIN, FREQ_MAX))
        else:
            # freqs_sliced rows match S_pcen rows directly
            active_freqs = freqs_sliced[active_bins]

            if len(active_freqs) > 0:
                fmin = max(active_freqs[0], FREQ_MIN)
                fmax = min(active_freqs[-1], FREQ_MAX)
            else:
                fmin, fmax = FREQ_MIN, FREQ_MAX

            # Add 10% padding
            freq_padding = (fmax - fmin) * 0.1
            fmin = max(FREQ_MIN, fmin - freq_padding)
            fmax = min(FREQ_MAX, fmax + freq_padding)

            events_with_bounds.append((start, end, fmin, fmax))

    return events_with_bounds, binary_mask

def create_fixed_segments(events_with_bounds, audio_duration):
    """Create fixed 3s segments centered on detected events"""
    half_seg = SEGMENT_DURATION / 2.0
    fixed_segs = []
    fixed_bounds = []
    
    for start, end, fmin, fmax in events_with_bounds:
        # Center on the middle of the event
        center = (start + end) / 2.0
        bs = max(0, center - half_seg)
        be = bs + SEGMENT_DURATION
        
        # Adjust if exceeds boundaries
        if be > audio_duration:
            be = audio_duration
            bs = max(0, be - SEGMENT_DURATION)
        
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

    # Compute PCEN spectrogram
    print("\nComputing PCEN spectrogram...")
    S_pcen, S_db, freqs, freq_mask = compute_pcen_spectrogram(y, sr)
    
    # Initial detection
    print("Running blob detection...")
    events_with_bounds, binary_mask = detect_sound_blobs(S_pcen, sr, freqs_sliced=freqs)
    detected_segs, detected_bounds = create_fixed_segments(events_with_bounds, audio_duration)
    print(f"Found {len(detected_segs)} candidate segments")

    # Load existing annotation if available
    ann_path = annotation_path_for(audio_path)
    has_annotation = os.path.exists(ann_path)
    title_suffix = ""

    if has_annotation:
        loaded = load_annotation(ann_path)
        if loaded:
            current_segments = list(loaded)
            current_bounds = [(FREQ_MIN, FREQ_MAX) for _ in loaded]
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
    ax_wave.set_title('Waveform — 3 s segments (drag to reposition; click on spectrogram to delete)', fontsize=10)
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_xlim(0, audio_duration)
    ax_wave.set_xlabel('Time (s)')
    ax_wave.grid(True, alpha=0.3)

    # Spectrogram — freqs[0]/freqs[-1] are the actual bin edges of the sliced matrix
    img = ax_spec.imshow(S_db, aspect='auto', origin='lower',
                         extent=[0, audio_duration, freqs[0], freqs[-1]],
                         cmap='plasma', vmin=-60, vmax=0)
    ax_spec.set_title('PCEN Spectrogram — Blue: binary mask overlay, Yellow: detected segments', fontsize=10)
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_xlim(0, audio_duration)
    ax_spec.set_ylim(FREQ_MIN, min(DEFAULT_FREQ_CUTOFF, sr/2))
    
    # Store binary overlay reference
    binary_overlay = None
    
    def update_binary_overlay():
        nonlocal binary_overlay
        if binary_overlay is not None:
            binary_overlay.remove()
        # binary_mask rows already correspond to the freq-sliced S_pcen
        mask_display = np.ma.masked_where(~binary_mask, binary_mask)
        binary_overlay = ax_spec.imshow(mask_display, aspect='auto', origin='lower',
                                        extent=[0, audio_duration, freqs[0], freqs[-1]],
                                        cmap=LinearSegmentedColormap.from_list('blue_alpha', [(0,0,1,0), (0,0,1,0.3)]),
                                        alpha=0.3, vmin=0, vmax=1)
    
    update_binary_overlay()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.36)

    # ── Widgets ──────────────────────────────────────────────────────────────
    # Layout: info bar at top, then 4 sliders spaced 0.055 apart, then buttons row
    SL_LEFT, SL_W, SL_H = 0.22, 0.55, 0.025
    ax_info       = plt.axes([SL_LEFT, 0.335, SL_W, 0.030])
    ax_energy_thresh = plt.axes([SL_LEFT, 0.275, SL_W, SL_H])
    ax_erode      = plt.axes([SL_LEFT, 0.220, SL_W, SL_H])
    ax_dilate     = plt.axes([SL_LEFT, 0.165, SL_W, SL_H])
    ax_freq_cutoff   = plt.axes([SL_LEFT, 0.110, SL_W, SL_H])
    ax_save      = plt.axes([0.82, 0.04, 0.08, 0.04])
    ax_quit      = plt.axes([0.01, 0.04, 0.08, 0.04])
    ax_play_stop = plt.axes([0.12, 0.04, 0.08, 0.04])

    slider_energy = widgets.Slider(ax_energy_thresh, 'Energy Threshold (lower=more sensitive)', 0.05, 0.95,
                                   valinit=INITIAL_ENERGY_THRESHOLD, valstep=0.01)
    slider_erode  = widgets.Slider(ax_erode,  'Erosion cycles (remove noise)', 0, 4,
                                   valinit=2, valstep=1)
    slider_dilate = widgets.Slider(ax_dilate, 'Dilation cycles (reconnect blobs)', 0, 4,
                                   valinit=2, valstep=1)
    slider_freq_cutoff = widgets.Slider(ax_freq_cutoff, 'Max Freq Display (Hz)', FREQ_MIN, sr/2,
                                        valinit=min(DEFAULT_FREQ_CUTOFF, sr/2), valstep=100)
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
        """Update detection based on current slider values"""
        nonlocal binary_mask, events_with_bounds, binary_overlay
        
        print("  Re-running blob detection...")
        events_with_bounds, binary_mask = detect_sound_blobs(
            S_pcen, sr,
            freqs_sliced=freqs,
            energy_threshold=slider_energy.val,
            erode_cycles=int(slider_erode.val),
            dilate_cycles=int(slider_dilate.val),
        )
        
        # Update binary mask overlay
        update_binary_overlay()
        
        # Create fixed segments
        new_segs, new_bnds = create_fixed_segments(events_with_bounds, audio_duration)
        
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

        # Draw detected events as cyan outlines
        for (start, end, fmin, fmax) in events_with_bounds:
            event_rect = patches.Rectangle((start, fmin), end - start, fmax - fmin,
                                          linewidth=1.5, edgecolor='cyan', facecolor='none',
                                          linestyle='--', alpha=0.7)
            ax_spec.add_patch(event_rect)
            event_rects.append(event_rect)

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
    slider_energy.on_changed(update_detection)
    slider_erode.on_changed(update_detection)
    slider_dilate.on_changed(update_detection)
    slider_freq_cutoff.on_changed(lambda val: ax_spec.set_ylim(FREQ_MIN, val) or fig.canvas.draw_idle())

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
    print("  • Energy Threshold: Lower = more sensitive, Higher = fewer blobs")
    print("  • Erosion cycles:   Remove noise / shrink blobs (0=none, 4=aggressive)")
    print("  • Dilation cycles:  Reconnect / expand blobs   (0=none, 4=aggressive)")
    print("  • Max Freq Display: Pan the spectrogram frequency axis")
    print("  • Drag blue segments horizontally to reposition")
    print("  • Click on a segment in the spectrogram to delete it")
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
    print("BLOB-BASED SEGMENT DETECTION (Binary Mask + Morphological Operations)")
    print("="*80)
    print("HOW IT WORKS:")
    print("  1. PCEN spectrogram — noise-robust representation for detection")
    print("  2. Per-band median subtraction removes steady noise per frequency row")
    print("  3. Spectral flux weights onset frames, suppressing sustained tones")
    print("  4. Binary threshold on combined score (blue overlay)")
    print("  5. Erode N / Dilate M to clean and reconnect blobs")
    print("  6. Connected components become sound events (cyan outlines)")
    print("  7. Fixed 3s segments created around events (yellow boxes)")
    print("="*80)

    _STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".stage5_blob_state.json")

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