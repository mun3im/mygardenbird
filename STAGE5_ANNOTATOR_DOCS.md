# How to create annotations with Stage5_find_segments_interactive.py

This tool was used to create the segment annotations for the
[SEA-Bird dataset](https://github.com/mun3im/seabird) -- 6,000 three-second
clips across 10 Southeast Asian bird species, sourced from 1,074 Xeno-canto
recordings.  Each recording was processed with this tool to locate vocally
active regions and extract fixed-length segments for downstream classification.

## Prerequisites

Install the required Python packages:

```bash
pip install numpy scipy librosa matplotlib sounddevice
```

## Running the tool

```bash
python Stage3_find_segments_interactive.py
```

A file dialog opens. Select a WAV, MP3, M4A, FLAC, or OGG audio file.

To set the starting directory for the file dialog:

```bash
python Stage3_find_segments_interactive.py --sound-dir /path/to/audio/files
```

Files shorter than 3 seconds are skipped automatically.

## Interface overview

The window shows two plots and a control panel:

- **Top plot (spectrogram)**: Log-frequency spectrogram with pink rectangle outlines marking detected segments and their frequency bounds.
- **Bottom plot (waveform)**: Time-domain waveform with pink shaded bars marking detected 3-second segments.
- **Sliders**: Median Threshold and Max Segment Gap control the automatic detection algorithm.
- **Buttons**: Quit, Play/Stop, Save.
- **Info bar**: Displays average SNR, peak SNR, and segment count.

## Adjusting detection parameters

### Median Threshold (1 -- 10)

Controls sensitivity of the spectrogram blob detector. Lower values detect more (including noise); higher values detect less. The tool auto-tunes an initial value based on the signal's mean-to-median spectral ratio.

### Max Segment Gap (0.1 -- 2.0 s)

Gaps between detected blobs shorter than this value are merged into a single blob before segment placement. Increase this to join fragmented vocalisations; decrease it to keep them separate.

Moving either slider re-runs the detection algorithm and replaces all current segments.

## Editing segments

### Removing a segment

Click on a pink rectangle outline in the **spectrogram** (top plot). The segment is removed immediately.

### Repositioning a segment

Click and drag a pink bar in the **waveform** (bottom plot) left or right. The segment slides in real time. Constraints enforced during drag:

- Segment cannot overlap its neighbours.
- Segment cannot extend before time 0 or past the end of the audio.
- Segment length stays exactly 3 seconds.

Release the mouse to commit the new position. SNR values update automatically.

## Playback

Click **Play** to hear the full audio. A red vertical line tracks playback progress across both plots. Click **Stop** (same button) to halt playback.

## Saving

Click **Save**. The tool writes a tab-separated `.txt` file alongside the audio file with the same base name:

```
0.500000	3.500000	song
5.000000	8.000000	call
12.300000	15.300000	song
```

Columns: `start_time	end_time	type`

- Times are written to **6 decimal places**.
- The `type` field is read from the Xeno-canto metadata CSV for that recording
  (`song`, `call`, or `other`). Falls back to `birdsong` if the recording is
  not found in the metadata.
- **No sequence index is stored** — the WAV filename suffix is the onset in
  milliseconds, which is always recoverable from `start_time`.

Up to 10 segments are saved (hard limit). The window closes after saving. Final SNR metrics are printed to the terminal.

## Quitting without saving

Click **Quit** or close the window. No file is written.

## Output format

Each line in the output `.txt` file contains:

| Column | Description |
|--------|-------------|
| 1 | Start time in seconds (6 decimal places) |
| 2 | End time in seconds (6 decimal places) |
| 3 | Vocalisation type from XC metadata: `song`, `call`, `other`, or `birdsong` |

Fields are tab-separated. All segments are exactly 3.000000 seconds long.
No sequence index is stored; the WAV clip suffix is derived from column 1 (onset ms).
