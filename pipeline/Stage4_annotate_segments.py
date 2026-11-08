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

# ── Shared core ───────────────────────────────────────────────────────────────
# Detection, SNR and annotation I/O live in stage4_core so that this annotator
# and Stage4b_annotate_qt.py behave identically.
from stage4_core import (
    FREQ_MIN, DEFAULT_FREQ_CUTOFF, MP3_NYQUIST_VALUES,
    SPRENGEL_N_FFT, SPRENGEL_HOP, DISPLAY_N_FFT, DISPLAY_HOP,
    SEGMENT_DURATION, MAX_SEGMENTS, INITIAL_THRESHOLD_MULT,
    DEFAULT_SOUND_DIR, get_profile, normalise_type,
    lookup_xc_type, calculate_enhanced_snr, compute_spectrograms,
    detect_sound_blobs, compute_display_bounds, compute_freq_bounds,
    create_fixed_segments, annotation_path_for, load_annotation,
)


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
    ax_wave.set_title('Waveform — click/drag a segment to select & reposition it; Play plays only the selected (lightly highlighted) segment', fontsize=10)
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

    # Tracks whether current_segments reflects hand edits or a loaded .txt,
    # so re-detection can warn before throwing that work away.
    segments_dirty = {'edited': False, 'from_annotation': bool(has_annotation and title_suffix)}

    # ── Drawing Helpers ──────────────────────────────────────────────────────
    wave_segment_patches = []
    spec_segment_rects = []
    spec_segment_labels = []
    # segment index -> the spectrogram rects drawn for it, so a drag can move
    # exactly those without assuming a fixed rects-per-segment stride.
    spec_rects_by_segment = {}
    spec_label_by_segment = {}

    def update_detection(val=None):
        nonlocal binary_mask, events_with_bounds

        # Warn before discarding anything the user did by hand, or anything
        # that came from an existing annotation file on disk.
        if segments_dirty['from_annotation']:
            print(f"  ⚠ Re-detection replaces {len(current_segments)} segment(s) loaded "
                  f"from {os.path.basename(ann_path)} — press Quit (not Save) to keep "
                  f"the file as it is on disk.")
        elif segments_dirty['edited']:
            print(f"  ⚠ Re-detection discards your manual edits "
                  f"({len(current_segments)} segment(s) replaced).")

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
        segments_dirty['edited'] = False
        segments_dirty['from_annotation'] = False
        update_segments()
        print(f"  Found {len(new_segs)} segments")

    def update_segments():
        """Redraw segments on waveform and spectrogram"""
        # Single enforcement point for the cap, so the on-screen count, the SNR
        # figures and the saved .txt can never disagree about how many segments
        # there are.
        if len(current_segments) > MAX_SEGMENTS:
            print(f"  Capping at {MAX_SEGMENTS} segments "
                  f"(dropped {len(current_segments) - MAX_SEGMENTS})")
            del current_segments[MAX_SEGMENTS:]
            del current_bounds[MAX_SEGMENTS:]

        for p in wave_segment_patches:
            p.remove()
        for r in spec_segment_rects:
            r.remove()
        for t in spec_segment_labels:
            t.remove()
        wave_segment_patches.clear()
        spec_segment_rects.clear()
        spec_segment_labels.clear()
        spec_rects_by_segment.clear()
        spec_label_by_segment.clear()

        # Clamp selection to a valid index whenever the segment list changes
        if not current_segments:
            selected_idx['val'] = None
        elif selected_idx['val'] is None or not (0 <= selected_idx['val'] < len(current_segments)):
            selected_idx['val'] = 0

        # Draw final segments
        for i, (start, end) in enumerate(current_segments):
            is_selected = (i == selected_idx['val'])

            # Waveform highlight — selected segment gets a lighter, distinct tint
            wave_color = 'lightgreen' if is_selected else 'deepskyblue'
            wave_alpha = 0.55 if is_selected else 0.4
            span = ax_wave.axvspan(start, end, alpha=wave_alpha, color=wave_color)
            span.segment_idx = i
            wave_segment_patches.append(span)

            # Spectrogram frequency bounds
            if i < len(current_bounds):
                fmin, fmax = current_bounds[i]
                edge_color = 'lightgreen' if is_selected else 'yellow'
                fill_alpha = 0.45 if is_selected else 0.3
                # Fill
                rect = patches.Rectangle((start, fmin), SEGMENT_DURATION, fmax - fmin,
                                        linewidth=2, edgecolor=edge_color, facecolor=edge_color,
                                        alpha=fill_alpha)
                ax_spec.add_patch(rect)
                spec_segment_rects.append(rect)
                # Outline
                outline = patches.Rectangle((start, fmin), SEGMENT_DURATION, fmax - fmin,
                                           linewidth=2 if not is_selected else 3,
                                           edgecolor=edge_color, facecolor='none',
                                           alpha=0.9)
                ax_spec.add_patch(outline)
                spec_segment_rects.append(outline)
                spec_rects_by_segment[i] = (rect, outline)

                # Segment number in the box's top-right corner
                label = ax_spec.text(
                    start + SEGMENT_DURATION, fmax, f"{i + 1}",
                    ha='right', va='top', fontsize=7, fontweight='bold',
                    color='black', clip_on=True,
                    bbox=dict(boxstyle='square,pad=0.15', facecolor=edge_color,
                              edgecolor='none', alpha=0.85))
                spec_segment_labels.append(label)
                spec_label_by_segment[i] = label

        snr = calculate_enhanced_snr(y, sr, current_segments)
        sel_label = (f"Segment {selected_idx['val'] + 1}/{len(current_segments)}"
                     if selected_idx['val'] is not None else "no segment")
        info_text.set_text(
            f"Avg SNR: {snr['average_snr']:.2f} dB | "
            f"Peak SNR: {snr['peak_snr']:.2f} dB | "
            f"Segments: {len(current_segments)} | "
            f"Playing: {sel_label}"
        )
        fig.canvas.draw_idle()

    # ── Segment selection (for playback) ─────────────────────────────────────
    selected_idx = {'val': 0 if current_segments else None}

    def select_segment(idx):
        """Select segment idx for playback and refresh its highlight."""
        if idx is None or not (0 <= idx < len(current_segments)):
            idx = None
        selected_idx['val'] = idx
        update_segments()

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
            if event.inaxes == ax_wave:
                # Selecting via the waveform (click or start of a reposition drag)
                # also makes this the segment Play will play.
                select_segment(idx)

    def _clamp_start(idx, new_start):
        """Clamp a dragged segment's start time.

        Neighbours bound the move, but a segment that is already flush against a
        neighbour would otherwise have left_limit == right_limit and be frozen;
        in that case fall back to the audio bounds so the drag can still move it
        (overlap is resolved on the next detection/redraw rather than blocking
        the interaction).
        """
        hard_lo = 0.0
        hard_hi = max(0.0, audio_duration - SEGMENT_DURATION)

        left_limit = current_segments[idx - 1][1] if idx > 0 else hard_lo
        right_limit = (current_segments[idx + 1][0] - SEGMENT_DURATION
                       if idx < len(current_segments) - 1 else hard_hi)

        if right_limit <= left_limit:
            left_limit, right_limit = hard_lo, hard_hi

        return max(left_limit, min(new_start, right_limit))

    def on_motion(event):
        idx = drag_state['idx']
        if idx is None or drag_state['axis'] != ax_wave or event.xdata is None:
            return
        dx = event.xdata - drag_state['start_x']
        new_start = _clamp_start(idx, drag_state['original_start'] + dx)

        if idx < len(wave_segment_patches):
            wave_segment_patches[idx].set_x(new_start)
        # Move whichever spectrogram rects belong to this segment (may be none
        # if the segment had no frequency bounds).
        for rect in spec_rects_by_segment.get(idx, ()):
            rect.set_x(new_start)
        label = spec_label_by_segment.get(idx)
        if label is not None:
            label.set_x(new_start + SEGMENT_DURATION)

        fig.canvas.draw_idle()

    def on_release(event):
        idx = drag_state['idx']
        if idx is None: 
            return
            
        total_move = (abs(event.xdata - drag_state['start_x'])
                      if drag_state['axis'] == ax_wave and event.xdata is not None else 0.0)
        
        if total_move < 0.05 and drag_state['axis'] == ax_spec:
            # Editing invalidates the coordinates playback was started with.
            if is_playing:
                stop_playback()
            print(f"Removing segment {idx}")
            del current_segments[idx]
            if idx < len(current_bounds):
                del current_bounds[idx]
            # Remap the playback selection around the removed index.
            sel = selected_idx['val']
            if sel is not None:
                if not current_segments:
                    selected_idx['val'] = None
                elif sel == idx:
                    selected_idx['val'] = min(idx, len(current_segments) - 1)
                elif sel > idx:
                    selected_idx['val'] = sel - 1
            segments_dirty['edited'] = True
            update_segments()
        elif total_move >= 0.05 and drag_state['axis'] == ax_wave:
            if is_playing:
                stop_playback()
            dx = (event.xdata - drag_state['start_x'] if event.xdata is not None else 0.0)
            new_start = _clamp_start(idx, drag_state['original_start'] + dx)
            current_segments[idx] = (new_start, new_start + SEGMENT_DURATION)
            segments_dirty['edited'] = True
            print(f"Moved segment {idx} to {new_start:.3f}s")
            update_segments()

        drag_state.update(idx=None, start_x=None, original_start=None, axis=None)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # ── Playback — plays only the selected 3s segment, not the full file ────
    progress_line_spec = ax_spec.axvline(x=0, color='red', linewidth=2, visible=False)
    progress_line_wave = ax_wave.axvline(x=0, color='red', linewidth=2, visible=False)
    is_playing = False
    play_start_time = None
    play_seg_start = 0.0
    play_seg_end = 0.0
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
        elapsed_in_seg = time.time() - play_start_time
        abs_t = play_seg_start + elapsed_in_seg
        if abs_t >= play_seg_end:
            stop_playback()
            return []
        progress_line_spec.set_xdata([abs_t, abs_t])
        progress_line_wave.set_xdata([abs_t, abs_t])
        return [progress_line_spec, progress_line_wave]

    def toggle_play_stop(event):
        nonlocal is_playing, play_start_time, progress_animation, play_seg_start, play_seg_end
        if is_playing:
            stop_playback()
            return

        idx = selected_idx['val']
        if idx is None or not (0 <= idx < len(current_segments)):
            print("  No segment selected to play.")
            return

        seg_start, seg_end = current_segments[idx]
        start_sample = max(0, int(seg_start * sr))
        end_sample = min(len(y), int(seg_end * sr))
        if end_sample <= start_sample:
            print("  Selected segment is empty; nothing to play.")
            return

        sd.play(y[start_sample:end_sample], sr)
        play_seg_start = seg_start
        play_seg_end = seg_end
        play_start_time = time.time()
        play_stop_button.label.set_text('Stop')
        is_playing = True
        progress_line_spec.set_visible(True)
        progress_line_wave.set_visible(True)
        progress_line_spec.set_xdata([seg_start, seg_start])
        progress_line_wave.set_xdata([seg_start, seg_start])
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
            for start, end in current_segments:
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
    print("  • Waveform:     click or drag a segment to select it (reposition selector) and reposition it")
    print("  • Spectrogram:  click a yellow box to delete a segment")
    print("  • Press 'Play' to listen to only the selected segment (lightly highlighted in green; default = segment 1)")
    print(f"  • Note: moving a slider re-runs detection and replaces the current segments (max {MAX_SEGMENTS})")
    print("="*60 + "\n")
    
    plt.show()


# ── Entry Point ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive bird vocalization segment detector using blob analysis")
    if get_profile is not None:
        parser.add_argument("--dataset", choices=["mygardenbird", "sea-bird30"],
                            default=os.environ.get("PIPELINE_DATASET", "mygardenbird"),
                            help="Which dataset's default sound dir to use. Default: mygardenbird "
                                 "(or $PIPELINE_DATASET if set). Only affects --sound-dir's default; "
                                 "pass --sound-dir explicitly to point at any folder directly, "
                                 "as before.")
    parser.add_argument("--sound-dir", default=None,
                        help=f"Path to sound files folder (default: the selected dataset's "
                             f"per_species_flacs dir; MyGardenBird: '{DEFAULT_SOUND_DIR}')")
    parser.add_argument("--file", type=str, help="Direct path to audio file (skip file dialog)")
    args = parser.parse_args()
    if args.sound_dir is None:
        if get_profile is not None:
            args.sound_dir = str(get_profile(args.dataset).per_species_flacs)
        else:
            args.sound_dir = DEFAULT_SOUND_DIR
    return args


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