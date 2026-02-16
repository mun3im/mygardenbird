import argparse
import json
import os
import time
import tkinter as tk
from config import PER_SPECIES_FLACS, PER_SPECIES_CSV, normalise_type
from tkinter import filedialog
import warnings

import pandas as pd

import numpy as np
import scipy.ndimage as ndi
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
import sounddevice as sd
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# ── Detection configuration ───────────────────────────────────────────────────
LOW_PASS_CUTOFF        = 8000    # Hz
HIGH_PASS_CUTOFF       = 200     # Hz
STFT_FRAME_SIZE        = 2048
HOP_LENGTH             = STFT_FRAME_SIZE // 3
INITIAL_MAX_SEGMENT_GAP = 0.7   # seconds
MIN_SEGMENT_LENGTH     = 0.2    # seconds
SEGMENT_DURATION       = 3.0    # seconds — fixed clip length
MAX_SEGMENTS           = 10
MAX_BLOB_DURATION      = 30.0   # seconds

# Maximum overlap allowed when dragging segments (fraction of SEGMENT_DURATION)
MAX_OVERLAP_FRACTION   = 0.10   # 10 %
MAX_OVERLAP_S          = SEGMENT_DURATION * MAX_OVERLAP_FRACTION   # 0.3 s

DEFAULT_SOUND_DIR = str(PER_SPECIES_FLACS)


# ── XC metadata type lookup ────────────────────────────────────────────────────

def _build_xc_type_map():
    """
    Load all per-species CSVs once and return a dict mapping int XC id → normalised type.
    Falls back gracefully if the CSV directory is missing or a CSV is malformed.
    """
    xc_map = {}
    csv_dir = PER_SPECIES_CSV
    if not csv_dir.exists():
        return xc_map
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
    return xc_map

_XC_TYPE_MAP = _build_xc_type_map()


def lookup_xc_type(audio_path):
    """
    Return the normalised XC 'type' (song / call / other) for the recording,
    or 'birdsong' if the id cannot be resolved.
    Expects filename stem of the form xc{id}.flac / xc{id}.wav.
    """
    stem = os.path.splitext(os.path.basename(audio_path))[0]  # e.g. "xc161194"
    if stem.startswith("xc"):
        try:
            xc_id = int(stem[2:])
            return _XC_TYPE_MAP.get(xc_id, "birdsong")
        except ValueError:
            pass
    return "birdsong"


# ── SNR ───────────────────────────────────────────────────────────────────────

def calculate_enhanced_snr(y, sr, segments, fade_buffer=0.1):
    """Average and peak SNR, ignoring fade_buffer seconds at each segment edge."""
    segment_indices = []
    fade_samples = int(fade_buffer * sr)
    for start_time, end_time in segments:
        start_idx = int(start_time * sr)
        end_idx   = min(int(end_time * sr), len(y))
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
        "peak_snr":    10 * np.log10((max(segment_powers)      + 1e-10) / (noise_power + 1e-10)),
        "segment_snrs": segment_snrs,
    }


# ── Audio processing ──────────────────────────────────────────────────────────

def pre_process_audio(y, sr, low_cutoff=LOW_PASS_CUTOFF, high_cutoff=HIGH_PASS_CUTOFF,
                      n_fft=STFT_FRAME_SIZE):
    """Band-pass filter via FFT zeroing."""
    S     = librosa.stft(y, n_fft=n_fft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask  = (freqs <= low_cutoff) & (freqs >= high_cutoff)
    S_f   = S.copy()
    S_f[~mask, :] = 0
    return librosa.istft(S_f)


def auto_tune_parameters(y, sr):
    """Suggest initial median threshold from signal statistics."""
    S = np.abs(librosa.stft(y, n_fft=STFT_FRAME_SIZE, hop_length=HOP_LENGTH))
    ratio = np.mean(S) / np.median(S)
    if ratio > 5:   return 3.0
    if ratio > 2:   return 4.0
    return 5.0


# ── Segment detection ─────────────────────────────────────────────────────────

def detect_segments(y, sr, median_threshold, max_segment_gap,
                    min_segment_length=MIN_SEGMENT_LENGTH):
    """
    Blob-finding + fixed-length segment extraction.

    Returns:
        final_segments  — list of (start, end) 3-second boxes  [for waveform]
        final_bounds    — matching list of (min_freq, max_freq) per segment
        merged_blobs    — list of (start, end, min_freq, max_freq) raw blobs
                          [for spectrogram overlay]
        S               — STFT magnitude matrix
    """
    audio_duration = len(y) / sr
    half_seg       = SEGMENT_DURATION / 2.0
    S              = np.abs(librosa.stft(y, n_fft=STFT_FRAME_SIZE, hop_length=HOP_LENGTH))
    freqs          = librosa.fft_frequencies(sr=sr, n_fft=STFT_FRAME_SIZE)

    # Dual-median thresholding
    row_median  = np.median(S, axis=1, keepdims=True)
    col_median  = np.median(S, axis=0, keepdims=True)
    binary_mask = ((S >= median_threshold * row_median) &
                   (S >= median_threshold * col_median))
    binary_mask = ndi.binary_opening(binary_mask, structure=np.ones((3, 3)))
    labeled, num = ndi.label(binary_mask)
    sizes = ndi.sum(binary_mask, labeled, range(num + 1))
    binary_mask[sizes[labeled] < 5] = 0

    temporal = ndi.binary_dilation(
        (np.sum(binary_mask, axis=0) > 0).astype(int), structure=np.ones(5))
    sr_per_frame = HOP_LENGTH / sr
    indices = np.where(temporal == 1)[0]

    raw_segments, raw_bounds = [], []

    if len(indices) > 0:
        start = indices[0]
        cur_frames = [start]
        for i in range(1, len(indices)):
            if (indices[i] - indices[i - 1]) * sr_per_frame > max_segment_gap:
                t0, t1 = start * sr_per_frame, indices[i - 1] * sr_per_frame
                if t1 - t0 >= min_segment_length:
                    raw_segments.append((t0, t1))
                    af = np.where(np.any(binary_mask[:, cur_frames], axis=1))[0]
                    raw_bounds.append((
                        freqs[af.min()] if len(af) else 0,
                        freqs[af.max()] if len(af) else sr / 2,
                    ))
                start, cur_frames = indices[i], [indices[i]]
            else:
                cur_frames.append(indices[i])

        t0, t1 = start * sr_per_frame, indices[-1] * sr_per_frame
        if t1 - t0 >= min_segment_length:
            raw_segments.append((t0, t1))
            af = np.where(np.any(binary_mask[:, cur_frames], axis=1))[0]
            raw_bounds.append((
                freqs[af.min()] if len(af) else 0,
                freqs[af.max()] if len(af) else sr / 2,
            ))

        # Merge close blobs
        merged_segs, merged_bounds = [], []
        for seg, bnd in zip(raw_segments, raw_bounds):
            if merged_segs and (seg[0] - merged_segs[-1][1]) <= max_segment_gap:
                merged_segs[-1]   = (merged_segs[-1][0], seg[1])
                merged_bounds[-1] = (min(merged_bounds[-1][0], bnd[0]),
                                     max(merged_bounds[-1][1], bnd[1]))
            else:
                merged_segs.append(seg)
                merged_bounds.append(bnd)

        # Build merged_blobs list for spectrogram display
        merged_blobs = [(s, e, lo, hi)
                        for (s, e), (lo, hi) in zip(merged_segs, merged_bounds)]

        # Expand blobs into fixed-length non-overlapping 3s segments
        fixed_segs, fixed_bounds = [], []
        for (blob_start, blob_end), blob_bnd in zip(merged_segs, merged_bounds):
            dur = min(blob_end - blob_start, MAX_BLOB_DURATION)
            blob_end = blob_start + dur

            if dur < SEGMENT_DURATION:
                center  = (blob_start + blob_end) / 2
                bs = max(0, center - half_seg)
                be = bs + SEGMENT_DURATION
                if be > audio_duration:
                    be = audio_duration
                    bs = max(0, be - SEGMENT_DURATION)
                fixed_segs.append((bs, be))
                fixed_bounds.append(blob_bnd)

            elif dur <= 2 * SEGMENT_DURATION:
                bs = max(0, blob_start)
                be = min(audio_duration, bs + SEGMENT_DURATION)
                if be - bs == SEGMENT_DURATION:
                    fixed_segs.append((bs, be))
                    fixed_bounds.append(blob_bnd)
                bs2 = be
                be2 = min(audio_duration, bs2 + SEGMENT_DURATION)
                if be2 - bs2 == SEGMENT_DURATION and bs2 < blob_end:
                    fixed_segs.append((bs2, be2))
                    fixed_bounds.append(blob_bnd)

            else:
                cur = blob_start
                while cur < blob_end and len(fixed_segs) < MAX_SEGMENTS:
                    be = min(blob_end, cur + SEGMENT_DURATION)
                    if be - cur == SEGMENT_DURATION and be <= audio_duration:
                        fixed_segs.append((cur, be))
                        fixed_bounds.append(blob_bnd)
                    cur += SEGMENT_DURATION

        # Deduplicate / enforce strict non-overlap from detection
        final_segments, final_bounds = [], []
        last_end = -SEGMENT_DURATION
        for (s, e), bnd in sorted(zip(fixed_segs, fixed_bounds)):
            if e - s == SEGMENT_DURATION and s >= last_end:
                final_segments.append((s, e))
                final_bounds.append(bnd)
                last_end = e
            if len(final_segments) >= MAX_SEGMENTS:
                break

        # Refine frequency bounds per final segment
        for i, (st, et) in enumerate(final_segments):
            si = int(st / sr_per_frame)
            ei = min(int(et / sr_per_frame) + 1, binary_mask.shape[1])
            af = np.where(np.any(binary_mask[:, si:ei], axis=1))[0]
            if len(af):
                final_bounds[i] = (freqs[af.min()], freqs[af.max()])
            else:
                final_bounds[i] = (0, sr / 2)

        return final_segments, final_bounds, merged_blobs, S

    return [], [], [], S


# ── Annotation I/O ────────────────────────────────────────────────────────────

def annotation_path_for(audio_path):
    """Return the expected .txt annotation path for an audio file."""
    return os.path.splitext(audio_path)[0] + '.txt'


def load_annotation(txt_path):
    """
    Load segments from an existing Stage 5 annotation file.

    Returns list of (start_s, end_s) tuples, or empty list on failure.
    """
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


# ── Main interactive function ─────────────────────────────────────────────────

def interactive_segment_detector(audio_path):
    y, sr          = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr
    if audio_duration < 3:
        print(f"File {audio_path} is too short ({audio_duration:.2f}s < 3s). Skipping.")
        return

    y_filtered        = pre_process_audio(y, sr)
    initial_threshold = auto_tune_parameters(y, sr)
    detected_segs, detected_bounds, merged_blobs, _ = detect_segments(
        y_filtered, sr, initial_threshold, INITIAL_MAX_SEGMENT_GAP)

    # ── Check for existing annotation ────────────────────────────────────────
    ann_path       = annotation_path_for(audio_path)
    has_annotation = os.path.exists(ann_path)

    if has_annotation:
        loaded = load_annotation(ann_path)
        if loaded:
            current_segments = list(loaded)
            # Synthesise bounds from audio (needed for spectrogram display)
            current_bounds = []
            for _ in loaded:
                current_bounds.append((HIGH_PASS_CUTOFF, LOW_PASS_CUTOFF))
            title_suffix = "  [loaded from annotation]"
            print(f"Loaded {len(current_segments)} segment(s) from {ann_path}")
        else:
            current_segments = list(detected_segs)
            current_bounds   = list(detected_bounds)
            title_suffix     = ""
    else:
        current_segments = list(detected_segs)
        current_bounds   = list(detected_bounds)
        title_suffix     = ""

    # ── Layout ───────────────────────────────────────────────────────────────
    times_waveform = np.linspace(0, audio_duration, len(y))
    fig, (ax_spec, ax_wave) = plt.subplots(
        2, 1, figsize=(20, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True
    )
    fig.canvas.manager.set_window_title(
        os.path.basename(audio_path) + title_suffix)
    plt.subplots_adjust(bottom=0.25)

    # Waveform
    ax_wave.plot(times_waveform, y, color='black')
    ax_wave.set_title('Waveform — 3 s segments (drag to reposition; 10 % overlap allowed)')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_xlim(0, audio_duration)

    # Spectrogram
    S_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_filtered, n_fft=STFT_FRAME_SIZE, hop_length=HOP_LENGTH)),
        ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                             y_axis='log', x_axis='time', ax=ax_spec, cmap='plasma')
    ax_spec.set_title('Spectrogram — blob outlines (click blob to remove segment)')
    ax_spec.set_xlabel("")
    ax_spec.set_xlim(0, audio_duration)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # ── Widgets ───────────────────────────────────────────────────────────────
    ax_threshold  = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_gap        = plt.axes([0.2, 0.10, 0.6, 0.03])
    ax_save       = plt.axes([0.8, 0.05, 0.1,  0.04])
    ax_quit       = plt.axes([0.01, 0.05, 0.08, 0.04])
    ax_info       = plt.axes([0.2, 0.05, 0.5,  0.04])
    ax_play_stop  = plt.axes([0.1, 0.05, 0.08, 0.04])

    slider_threshold = widgets.Slider(ax_threshold, 'Median Threshold', 1, 10,
                                      valinit=initial_threshold, valstep=0.1)
    slider_gap       = widgets.Slider(ax_gap, 'Max Segment Gap (s)', 0.1, 2.0,
                                      valinit=INITIAL_MAX_SEGMENT_GAP, valstep=0.1)
    save_button      = widgets.Button(ax_save, 'Save')
    quit_button      = widgets.Button(ax_quit, 'Quit')
    play_stop_button = widgets.Button(ax_play_stop, 'Play')
    ax_info.axis('off')

    snr_metrics = calculate_enhanced_snr(y_filtered, sr, current_segments)
    info_text = ax_info.text(
        0.5, 0.5,
        f"Avg SNR: {snr_metrics['average_snr']:.2f} dB | "
        f"Peak SNR: {snr_metrics['peak_snr']:.2f} dB | "
        f"Segments: {len(current_segments)}",
        ha='center', va='center', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7))

    # ── Drawing helpers ───────────────────────────────────────────────────────
    wave_segment_patches = []   # pink axvspan fills on waveform (3 s boxes)
    spec_blob_rects      = []   # pink outlines on spectrogram (raw blobs)

    def _draw_blob_rects(blobs):
        """Draw raw blob outlines on the spectrogram."""
        for r in spec_blob_rects:
            r.remove()
        spec_blob_rects.clear()
        for (bs, be, lo, hi) in blobs:
            r = patches.Rectangle(
                (bs, lo), be - bs, hi - lo,
                linewidth=1.5, edgecolor='cyan', facecolor='none',
                linestyle='--', alpha=0.7)
            ax_spec.add_patch(r)
            spec_blob_rects.append(r)

    def update_segments(new_blobs=None):
        """Redraw waveform segment fills and (optionally) spectrogram blob boxes."""
        for p in wave_segment_patches:
            p.remove()
        wave_segment_patches.clear()

        for i, (start, end) in enumerate(current_segments):
            span = ax_wave.axvspan(start, end, alpha=0.4, color='deepskyblue')
            span.segment_idx = i
            wave_segment_patches.append(span)

        if new_blobs is not None:
            _draw_blob_rects(new_blobs)

        snr = calculate_enhanced_snr(y_filtered, sr, current_segments)
        info_text.set_text(
            f"Avg SNR: {snr['average_snr']:.2f} dB | "
            f"Peak SNR: {snr['peak_snr']:.2f} dB | "
            f"Segments: {len(current_segments)}"
        )
        fig.canvas.draw_idle()

    # Initial draw — blob rects from detection result
    _draw_blob_rects(merged_blobs)
    update_segments()   # waveform fills only (blobs already drawn)

    # ── Drag / click interaction ───────────────────────────────────────────────
    drag_state = {'idx': None, 'start_x': None, 'original_start': None, 'axis': None}

    def _find_segment_at(x):
        for i, (s, e) in enumerate(current_segments):
            if s <= x <= e:
                return i
        return None

    def on_press(event):
        if event.inaxes not in (ax_wave, ax_spec) or event.button != 1:
            return
        if event.xdata is None:
            return
        idx = _find_segment_at(event.xdata)
        if idx is not None:
            drag_state.update(idx=idx, start_x=event.xdata,
                              original_start=current_segments[idx][0],
                              axis=event.inaxes)

    def on_motion(event):
        idx = drag_state['idx']
        if idx is None or drag_state['axis'] != ax_wave or event.xdata is None:
            return

        dx        = event.xdata - drag_state['start_x']
        new_start = drag_state['original_start'] + dx

        # Boundaries allow up to MAX_OVERLAP_S into the neighbouring segment
        left_limit = (current_segments[idx - 1][1] - MAX_OVERLAP_S
                      if idx > 0 else 0.0)
        right_limit = (current_segments[idx + 1][0] - SEGMENT_DURATION + MAX_OVERLAP_S
                       if idx < len(current_segments) - 1
                       else audio_duration - SEGMENT_DURATION)

        new_start = max(left_limit, min(new_start, right_limit))
        new_end   = new_start + SEGMENT_DURATION

        if idx < len(wave_segment_patches):
            wave_segment_patches[idx].set_x(new_start)
            wave_segment_patches[idx].set_width(SEGMENT_DURATION)

        fig.canvas.draw_idle()

    def on_release(event):
        idx = drag_state['idx']
        if idx is None:
            return

        total_move = (abs(event.xdata - drag_state['start_x'])
                      if drag_state['axis'] == ax_wave and event.xdata is not None
                      else 0.0)

        if total_move < 0.05 and drag_state['axis'] == ax_spec:
            # Click on spectrogram blob → remove matching segment
            print(f"Removing segment {idx} at {current_segments[idx][0]:.3f}s")
            del current_segments[idx]
            del current_bounds[idx]
            update_segments()

        elif total_move >= 0.05 and drag_state['axis'] == ax_wave:
            dx        = (event.xdata - drag_state['start_x']
                         if event.xdata is not None else 0.0)
            new_start = drag_state['original_start'] + dx

            left_limit = (current_segments[idx - 1][1] - MAX_OVERLAP_S
                          if idx > 0 else 0.0)
            right_limit = (current_segments[idx + 1][0] - SEGMENT_DURATION + MAX_OVERLAP_S
                           if idx < len(current_segments) - 1
                           else audio_duration - SEGMENT_DURATION)

            new_start = max(left_limit, min(new_start, right_limit))
            new_end   = new_start + SEGMENT_DURATION
            current_segments[idx] = (new_start, new_end)
            print(f"Moved segment {idx} to {new_start:.6f}s – {new_end:.6f}s")
            update_segments()

        drag_state.update(idx=None, start_x=None, original_start=None, axis=None)

    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('motion_notify_event',  on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # ── Playback ──────────────────────────────────────────────────────────────
    progress_line_spec = ax_spec.axvline(x=0, color='red', linewidth=2, visible=False)
    progress_line_wave = ax_wave.axvline(x=0, color='red', linewidth=2, visible=False)
    is_playing        = False
    play_start_time   = None
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
        nonlocal is_playing
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
            progress_animation = FuncAnimation(
                fig, update_frame, interval=50, blit=True, cache_frame_data=False)
            fig.canvas.draw_idle()

    play_stop_button.on_clicked(toggle_play_stop)

    # ── Slider callbacks — re-detect and redraw everything ───────────────────
    def update_plot(val):
        new_segs, new_bnds, new_blobs, _ = detect_segments(
            y_filtered, sr, slider_threshold.val, slider_gap.val)
        current_segments[:] = new_segs
        current_bounds[:]   = new_bnds
        update_segments(new_blobs=new_blobs)

    slider_threshold.on_changed(update_plot)
    slider_gap.on_changed(update_plot)

    # ── Save / Quit ───────────────────────────────────────────────────────────
    def save_segments(event):
        txt = annotation_path_for(audio_path)
        final_snr = calculate_enhanced_snr(y_filtered, sr, current_segments)
        xc_type = lookup_xc_type(audio_path)
        with open(txt, 'w') as f:
            for start, end in current_segments[:MAX_SEGMENTS]:
                f.write(f"{start:.6f}\t{end:.6f}\t{xc_type}\n")
        print(f"Saved {len(current_segments)} segment(s) to {txt}")
        print(f"Avg SNR: {final_snr['average_snr']:.2f} dB  "
              f"Peak SNR: {final_snr['peak_snr']:.2f} dB")
        plt.close(fig)

    def quit_without_saving(event):
        print("Cancelled without saving.")
        plt.close(fig)

    save_button.on_clicked(save_segments)
    quit_button.on_clicked(quit_without_saving)
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive audio segment detector with spectrogram visualization.")
    parser.add_argument(
        "--sound-dir", default=DEFAULT_SOUND_DIR,
        help=(f"Path to sound files folder (default: '{DEFAULT_SOUND_DIR}'). "
              "The file dialog opens here."),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".stage5_state.json")

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

    sound_dir = os.path.abspath(args.sound_dir)
    last_dir  = _load_last_dir()

    if last_dir and os.path.isdir(last_dir):
        initial_dir = last_dir
    elif os.path.isdir(sound_dir):
        initial_dir = sound_dir
    elif os.path.isdir(os.path.dirname(sound_dir)):
        initial_dir = os.path.dirname(sound_dir)
    else:
        initial_dir = os.getcwd()

    root = tk.Tk()
    root.withdraw()
    audio_file = filedialog.askopenfilename(
        title="Select audio file",
        initialdir=initial_dir,
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma"),
            ("WAV files",   "*.wav"),
            ("MP3 files",   "*.mp3"),
            ("M4A files",   "*.m4a"),
            ("FLAC files",  "*.flac"),
            ("OGG files",   "*.ogg"),
            ("AAC files",   "*.aac"),
            ("WMA files",   "*.wma"),
        ],
    )
    root.destroy()

    if audio_file:
        _save_last_dir(os.path.dirname(os.path.abspath(audio_file)))

    if audio_file:
        print(f"Processing: {audio_file}")
        interactive_segment_detector(audio_file)
    else:
        print("No file selected.")
