import argparse
import os
import time
import tkinter as tk
from tkinter import filedialog
import warnings

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

# Detection Configuration Parameters
LOW_PASS_CUTOFF = 8000       # Hz, frequency cutoff for low-pass filter
HIGH_PASS_CUTOFF = 200       # Hz, frequency cutoff for high-pass filter
STFT_FRAME_SIZE = 2048       # FFT frame size
HOP_LENGTH = STFT_FRAME_SIZE // 3  # Hop length between frames
INITIAL_MAX_SEGMENT_GAP = 0.7  # Maximum gap between segments to merge (seconds)
MIN_SEGMENT_LENGTH = 0.2    # Minimum segment length in seconds
SEGMENT_DURATION = 3.0       # Fixed segment length in seconds
MAX_SEGMENTS = 10            # Maximum number of segments to produce
MAX_BLOB_DURATION = 30.0     # Cap blob duration at this value (seconds)

DEFAULT_SOUND_DIR = "sandbox"

def calculate_enhanced_snr(y, sr, segments, fade_buffer=0.1):
    """
    Calculate multiple SNR metrics using both average and peak power approaches.
    Ignores the first and last fade_buffer seconds of each segment to account for fade in/fade out effects.
    """
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
    for start_idx, end_idx in segment_indices:
        if start_idx < end_idx:
            signal_mask[start_idx:end_idx] = True

    noise_samples = y[~signal_mask]
    if len(noise_samples) == 0:
        return {
            "average_snr": float('inf'),
            "peak_snr": float('inf'),
            "segment_snrs": []
        }

    noise_power_avg = np.mean(noise_samples ** 2)
    segment_powers = []
    segment_snrs = []

    for start_idx, end_idx in segment_indices:
        if start_idx >= end_idx:
            continue
        segment = y[start_idx:end_idx]
        if len(segment) == 0:
            continue
        segment_power_avg = np.mean(segment ** 2)
        segment_powers.append(segment_power_avg)
        segment_snr = 10 * np.log10((segment_power_avg + 1e-10) / (noise_power_avg + 1e-10))
        segment_snrs.append(segment_snr)

    if not segment_powers:
        return {
            "average_snr": float('-inf'),
            "peak_snr": float('-inf'),
            "segment_snrs": []
        }

    average_snr = 10 * np.log10((np.mean(segment_powers) + 1e-10) / (noise_power_avg + 1e-10))
    peak_snr = 10 * np.log10((max(segment_powers) + 1e-10) / (noise_power_avg + 1e-10))

    return {
        "average_snr": average_snr,
        "peak_snr": peak_snr,
        "segment_snrs": segment_snrs
    }


def pre_process_audio(y, sr, low_cutoff=LOW_PASS_CUTOFF, high_cutoff=HIGH_PASS_CUTOFF, n_fft=STFT_FRAME_SIZE):
    """
    Apply band-pass filtering to audio signal using FFT-based approach.
    """
    S = librosa.stft(y, n_fft=n_fft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_pass_mask = freqs <= low_cutoff
    high_pass_mask = freqs >= high_cutoff
    band_pass_mask = low_pass_mask & high_pass_mask
    S_filtered = S.copy()
    S_filtered[~band_pass_mask, :] = 0
    y_filtered = librosa.istft(S_filtered)
    return y_filtered


def detect_segments(y, sr, median_threshold, max_segment_gap, min_segment_length=MIN_SEGMENT_LENGTH):
    """
    Blob finding algorithm with noise reduction and non-overlapping segment creation.

    Ensures all final segments are exactly SEGMENT_DURATION long and non-overlapping.
    Rules:
    - Blobs < SEGMENT_DURATION: One segment centered around blob.
    - Blobs up to 2x SEGMENT_DURATION: Two segments (non-overlapping).
    - Blobs > 2x SEGMENT_DURATION: Multiple segments (non-overlapping, max MAX_BLOB_DURATION total).
    - Max MAX_SEGMENTS segments total.
    """
    audio_duration = len(y) / sr
    half_seg = SEGMENT_DURATION / 2.0
    S = np.abs(librosa.stft(y, n_fft=STFT_FRAME_SIZE, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=STFT_FRAME_SIZE)

    # Noise reduction via dual-median thresholding
    row_median = np.median(S, axis=1, keepdims=True)
    col_median = np.median(S, axis=0, keepdims=True)
    binary_mask = (S >= median_threshold * row_median) & (S >= median_threshold * col_median)
    binary_mask = ndi.binary_opening(binary_mask, structure=np.ones((3, 3)))
    labeled_components, num_components = ndi.label(binary_mask)
    component_sizes = ndi.sum(binary_mask, labeled_components, range(num_components + 1))
    small_noise_mask = component_sizes < 5
    binary_mask[small_noise_mask[labeled_components]] = 0
    temporal_vector = np.sum(binary_mask, axis=0) > 0
    temporal_vector = ndi.binary_dilation(temporal_vector.astype(int), structure=np.ones(5))

    sr_per_frame = HOP_LENGTH / sr
    indices = np.where(temporal_vector == 1)[0]
    raw_segments = []
    segment_bounds = []

    # Detect initial blobs
    if len(indices) > 0:
        start = indices[0]
        current_frames = [start]
        for i in range(1, len(indices)):
            if (indices[i] - indices[i - 1]) * sr_per_frame > max_segment_gap:
                segment_start_time = start * sr_per_frame
                segment_end_time = indices[i - 1] * sr_per_frame
                segment_duration = segment_end_time - segment_start_time
                if segment_duration >= min_segment_length:
                    raw_segments.append((segment_start_time, segment_end_time))
                    segment_frames = binary_mask[:, current_frames]
                    active_freqs = np.where(np.any(segment_frames, axis=1))[0]
                    min_freq = freqs[np.min(active_freqs)] if len(active_freqs) > 0 else 0
                    max_freq = freqs[np.max(active_freqs)] if len(active_freqs) > 0 else sr / 2
                    segment_bounds.append((min_freq, max_freq))
                start = indices[i]
                current_frames = [start]
            else:
                current_frames.append(indices[i])

        segment_start_time = start * sr_per_frame
        segment_end_time = indices[-1] * sr_per_frame
        segment_duration = segment_end_time - segment_start_time
        if segment_duration >= min_segment_length:
            raw_segments.append((segment_start_time, segment_end_time))
            segment_frames = binary_mask[:, current_frames]
            active_freqs = np.where(np.any(segment_frames, axis=1))[0]
            min_freq = freqs[np.min(active_freqs)] if len(active_freqs) > 0 else 0
            max_freq = freqs[np.max(active_freqs)] if len(active_freqs) > 0 else sr / 2
            segment_bounds.append((min_freq, max_freq))

        # Merge overlapping or close segments
        merged_segments = []
        merged_bounds = []
        for i, (seg, bounds) in enumerate(zip(raw_segments, segment_bounds)):
            if merged_segments and (seg[0] - merged_segments[-1][1]) <= max_segment_gap:
                new_min_freq = min(merged_bounds[-1][0], bounds[0])
                new_max_freq = max(merged_bounds[-1][1], bounds[1])
                merged_bounds[-1] = (new_min_freq, new_max_freq)
                merged_segments[-1] = (merged_segments[-1][0], seg[1])
            else:
                merged_segments.append(seg)
                merged_bounds.append(bounds)

        # Create fixed-length non-overlapping segments from merged blobs
        fixed_segments = []
        fixed_bounds = []
        for merge_idx, (blob_start, blob_end) in enumerate(merged_segments):
            blob_duration = blob_end - blob_start
            if blob_duration > MAX_BLOB_DURATION:
                blob_end = blob_start + MAX_BLOB_DURATION
                blob_duration = MAX_BLOB_DURATION

            blob_bounds = merged_bounds[merge_idx]

            if blob_duration < SEGMENT_DURATION:
                center_time = (blob_start + blob_end) / 2
                box_start = max(0, center_time - half_seg)
                box_end = box_start + SEGMENT_DURATION
                if box_end > audio_duration:
                    box_end = audio_duration
                    box_start = max(0, box_end - SEGMENT_DURATION)
                fixed_segments.append((box_start, box_end))
                fixed_bounds.append(blob_bounds)

            elif blob_duration <= 2 * SEGMENT_DURATION:
                box1_start = max(0, blob_start)
                box1_end = min(audio_duration, box1_start + SEGMENT_DURATION)
                if box1_end - box1_start == SEGMENT_DURATION:
                    fixed_segments.append((box1_start, box1_end))
                    fixed_bounds.append(blob_bounds)

                box2_start = box1_end
                box2_end = min(audio_duration, box2_start + SEGMENT_DURATION)
                if box2_end - box2_start == SEGMENT_DURATION and box2_start < blob_end:
                    fixed_segments.append((box2_start, box2_end))
                    fixed_bounds.append(blob_bounds)

            else:
                current_start = blob_start
                while current_start < blob_end and len(fixed_segments) < MAX_SEGMENTS:
                    box_end = min(blob_end, current_start + SEGMENT_DURATION)
                    if box_end - current_start == SEGMENT_DURATION and box_end <= audio_duration:
                        fixed_segments.append((current_start, box_end))
                        fixed_bounds.append(blob_bounds)
                    current_start += SEGMENT_DURATION

        # Ensure no overlaps and exact segment duration
        final_segments = []
        final_bounds = []
        fixed_segments.sort()
        last_end = -SEGMENT_DURATION
        for i, (start, end) in enumerate(fixed_segments):
            if end - start == SEGMENT_DURATION and start >= last_end:
                final_segments.append((start, end))
                final_bounds.append(fixed_bounds[i])
                last_end = end
            elif end - start != SEGMENT_DURATION:
                continue
            if len(final_segments) >= MAX_SEGMENTS:
                break

        # Verify frequency bounds for final segments
        for i, (start_time, end_time) in enumerate(final_segments):
            start_idx = int(start_time / sr_per_frame)
            end_idx = int(end_time / sr_per_frame) + 1
            if end_idx > binary_mask.shape[1]:
                end_idx = binary_mask.shape[1]
            chunk_mask = binary_mask[:, start_idx:end_idx]
            active_freqs = np.where(np.any(chunk_mask, axis=1))[0]
            if len(active_freqs) > 0:
                min_freq = freqs[np.min(active_freqs)]
                max_freq = freqs[np.max(active_freqs)]
            else:
                min_freq = 0
                max_freq = sr / 2
            final_bounds[i] = (min_freq, max_freq)

        return final_segments, final_bounds, S

    return [], [], S


def auto_tune_parameters(y, sr):
    """
    Automatically tune the median threshold based on signal characteristics.
    Returns the suggested median threshold value.
    """
    S = np.abs(librosa.stft(y, n_fft=STFT_FRAME_SIZE, hop_length=HOP_LENGTH))
    mean_to_median_ratio = np.mean(S) / np.median(S)
    if mean_to_median_ratio > 5:
        return 3.0
    elif mean_to_median_ratio > 2:
        return 4.0
    else:
        return 5.0


def interactive_segment_detector(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr
    if audio_duration < 3:
        print(f"File {audio_path} is too short ({audio_duration:.2f}s < 3s). Skipping.")
        return

    y_filtered = pre_process_audio(y, sr)
    S_filtered = np.abs(librosa.stft(y_filtered, n_fft=STFT_FRAME_SIZE, hop_length=HOP_LENGTH))
    initial_threshold = auto_tune_parameters(y, sr)
    current_segments, current_bounds, _ = detect_segments(y_filtered, sr, initial_threshold, INITIAL_MAX_SEGMENT_GAP)

    times_waveform = np.linspace(0, audio_duration, len(y))
    fig, (ax_spec, ax_wave) = plt.subplots(
        2, 1, figsize=(20, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True
    )
    fig.canvas.manager.set_window_title(os.path.basename(audio_path))
    plt.subplots_adjust(bottom=0.25)

    ax_wave.plot(times_waveform, y, color='black')
    ax_wave.set_title('Waveform with Detected 3s Segments (Drag to reposition)')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_xlim(0, audio_duration)

    wave_segment_patches = []
    spec_segment_rects = []

    S_db = librosa.amplitude_to_db(S_filtered, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                             y_axis='log', x_axis='time', ax=ax_spec, cmap='plasma')
    ax_spec.set_title('Spectrogram with Detected 3s Segments')
    ax_spec.set_xlabel("")
    ax_spec.set_xlim(0, audio_duration)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    ax_threshold = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_gap = plt.axes([0.2, 0.10, 0.6, 0.03])
    ax_save = plt.axes([0.8, 0.05, 0.1, 0.04])
    ax_quit = plt.axes([0.01, 0.05, 0.08, 0.04])
    ax_info = plt.axes([0.2, 0.05, 0.5, 0.04])
    ax_play_stop = plt.axes([0.1, 0.05, 0.08, 0.04])

    slider_threshold = widgets.Slider(ax_threshold, 'Median Threshold', 1, 10, valinit=initial_threshold, valstep=0.1)
    slider_gap = widgets.Slider(ax_gap, 'Max Segment Gap (s)', 0.1, 2.0, valinit=INITIAL_MAX_SEGMENT_GAP, valstep=0.1)
    save_button = widgets.Button(ax_save, 'Save')
    quit_button = widgets.Button(ax_quit, 'Quit')
    play_stop_button = widgets.Button(ax_play_stop, 'Play')
    ax_info.axis('off')

    snr_metrics = calculate_enhanced_snr(y_filtered, sr, current_segments, fade_buffer=0.1)
    info_text = ax_info.text(0.5, 0.5,
                             f"Average SNR: {snr_metrics['average_snr']:.2f} dB | "
                             f"Peak SNR: {snr_metrics['peak_snr']:.2f} dB | "
                             f"Num Segments: {len(current_segments)}",
                             ha='center', va='center', fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.7))

    def update_segments():
        nonlocal wave_segment_patches, spec_segment_rects
        for p in wave_segment_patches + spec_segment_rects:
            if p in ax_wave.patches or p in ax_spec.patches:
                p.remove()
        wave_segment_patches.clear()
        spec_segment_rects.clear()

        for i, (start, end) in enumerate(current_segments):
            span = ax_wave.axvspan(start, end, alpha=0.5, color='pink')
            span.segment_idx = i
            wave_segment_patches.append(span)

        for i, ((start, end), (min_freq, max_freq)) in enumerate(zip(current_segments, current_bounds)):
            rect = patches.Rectangle((start, min_freq), end - start, max_freq - min_freq,
                                     linewidth=2, edgecolor='pink', facecolor='none')
            spec_segment_rects.append(rect)
            ax_spec.add_patch(rect)

        snr_metrics = calculate_enhanced_snr(y_filtered, sr, current_segments, fade_buffer=0.1)
        info_text.set_text(
            f"Average SNR: {snr_metrics['average_snr']:.2f} dB | "
            f"Peak SNR: {snr_metrics['peak_snr']:.2f} dB | "
            f"Num Segments: {len(current_segments)}"
        )

        fig.canvas.draw_idle()

    # --- Drag-to-reposition / click-to-remove state ---
    drag_state = {'idx': None, 'start_x': None, 'original_start': None, 'axis': None}

    def _find_segment_at(x):
        """Return the index of the segment containing time *x*, or None."""
        for i, (start, end) in enumerate(current_segments):
            if start <= x <= end:
                return i
        return None

    def on_press(event):
        if event.inaxes not in (ax_wave, ax_spec) or event.button != 1:
            return
        if event.xdata is None:
            return
        idx = _find_segment_at(event.xdata)
        if idx is not None:
            drag_state['idx'] = idx
            drag_state['start_x'] = event.xdata
            drag_state['original_start'] = current_segments[idx][0]
            drag_state['axis'] = event.inaxes

    def on_motion(event):
        idx = drag_state['idx']
        if idx is None:
            return
        # Only allow dragging from the waveform axis
        if drag_state['axis'] != ax_wave:
            return
        if event.xdata is None:
            return

        dx = event.xdata - drag_state['start_x']
        new_start = drag_state['original_start'] + dx

        # Clamp: left boundary
        if idx > 0:
            left_limit = current_segments[idx - 1][1]
        else:
            left_limit = 0.0
        # Clamp: right boundary
        if idx < len(current_segments) - 1:
            right_limit = current_segments[idx + 1][0] - SEGMENT_DURATION
        else:
            right_limit = audio_duration - SEGMENT_DURATION

        new_start = max(left_limit, min(new_start, right_limit))
        new_end = new_start + SEGMENT_DURATION

        # Move waveform patch in-place (axvspan returns a Rectangle)
        if idx < len(wave_segment_patches):
            wave_segment_patches[idx].set_x(new_start)
            wave_segment_patches[idx].set_width(SEGMENT_DURATION)

        # Move spectrogram rectangle in-place
        if idx < len(spec_segment_rects):
            spec_segment_rects[idx].set_x(new_start)
            spec_segment_rects[idx].set_width(new_end - new_start)

        fig.canvas.draw_idle()

    def on_release(event):
        idx = drag_state['idx']
        if idx is None:
            return

        if drag_state['axis'] == ax_wave and event.xdata is not None:
            total_move = abs(event.xdata - drag_state['start_x'])
        else:
            total_move = 0.0

        if total_move < 0.05 and drag_state['axis'] == ax_spec:
            # Click on spectrogram – remove the segment
            print(f"Removing segment {idx} at time {current_segments[idx][0]:.3f}s")
            if idx < len(current_bounds):
                del current_segments[idx]
                del current_bounds[idx]
                update_segments()
        elif total_move >= 0.05 and drag_state['axis'] == ax_wave:
            # Drag – commit new position
            dx = event.xdata - drag_state['start_x'] if event.xdata is not None else 0.0
            new_start = drag_state['original_start'] + dx

            if idx > 0:
                left_limit = current_segments[idx - 1][1]
            else:
                left_limit = 0.0
            if idx < len(current_segments) - 1:
                right_limit = current_segments[idx + 1][0] - SEGMENT_DURATION
            else:
                right_limit = audio_duration - SEGMENT_DURATION

            new_start = max(left_limit, min(new_start, right_limit))
            new_end = new_start + SEGMENT_DURATION

            current_segments[idx] = (new_start, new_end)
            print(f"Moved segment {idx} to {new_start:.3f}s – {new_end:.3f}s")
            update_segments()

        drag_state['idx'] = None
        drag_state['start_x'] = None
        drag_state['original_start'] = None
        drag_state['axis'] = None

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

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
            print("Playing audio...")
            sd.play(y, sr)
            play_start_time = time.time()
            play_stop_button.label.set_text('Stop')
            is_playing = True
            progress_line_spec.set_visible(True)
            progress_line_wave.set_visible(True)
            progress_line_spec.set_xdata([0, 0])
            progress_line_wave.set_xdata([0, 0])
            progress_animation = FuncAnimation(
                fig, update_frame, interval=50, blit=True, cache_frame_data=False
            )
            fig.canvas.draw_idle()

    play_stop_button.on_clicked(toggle_play_stop)

    def update_plot(val):
        new_segments, new_bounds, _ = detect_segments(
            y_filtered, sr,
            slider_threshold.val,
            slider_gap.val
        )
        current_segments[:] = new_segments
        current_bounds[:] = new_bounds
        update_segments()

    slider_threshold.on_changed(update_plot)
    slider_gap.on_changed(update_plot)

    def save_segments(event):
        base_filename = os.path.splitext(audio_path)[0]
        segments_filename = base_filename + '.txt'
        final_snr = calculate_enhanced_snr(y_filtered, sr, current_segments, fade_buffer=0.1)
        with open(segments_filename, 'w') as f:
            for idx, (start, end) in enumerate(current_segments[:10]):
                f.write(f"{start:.3f}\t{end:.3f}\tsong\t{idx}\n")
        print(f"Segments saved to {segments_filename}")
        print(f"Final Average SNR: {final_snr['average_snr']:.2f} dB")
        print(f"Final Peak SNR: {final_snr['peak_snr']:.2f} dB")
        plt.close(fig)

    save_button.on_clicked(save_segments)

    def quit_without_saving(event):
        print("Cancelled without saving.")
        plt.close(fig)

    quit_button.on_clicked(quit_without_saving)
    update_segments()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive audio segment detector with spectrogram visualization."
    )
    parser.add_argument(
        "--sound-dir",
        default=DEFAULT_SOUND_DIR,
        help=(
            f"Path to sound files folder (default: '{DEFAULT_SOUND_DIR}'). "
            "Relative paths are resolved from the current working directory. "
            "The file dialog opens one level above this folder."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve sound directory (relative paths based on cwd)
    sound_dir = os.path.abspath(args.sound_dir)
    # Open file dialog inside the sound directory, fall back to its parent
    if os.path.isdir(sound_dir):
        initial_dir = sound_dir
    else:
        initial_dir = os.path.dirname(sound_dir)
        if not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()

    root = tk.Tk()
    root.withdraw()
    audio_file = filedialog.askopenfilename(
        title="Select audio file",
        initialdir=initial_dir,
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("M4A files", "*.m4a"),
            ("FLAC files", "*.flac"),
            ("OGG files", "*.ogg"),
            ("AAC files", "*.aac"),
            ("WMA files", "*.wma"),
        ],
    )
    root.destroy()

    if audio_file:
        print(f"Processing file: {audio_file}")
        interactive_segment_detector(audio_file)
    else:
        print("No file selected.")
