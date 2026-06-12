# Stage4_annotate_segments.py — Annotator Documentation

Interactive GUI for identifying and annotating 3-second bird vocalization segments
in Xeno-canto FLAC recordings. Used to build the
[MyGardenBird dataset](https://github.com/mun3im/mygardenbird) — 7,200 three-second
clips across 12 Malaysian bird species, sourced from Xeno-canto recordings.
Each recording is processed with this tool to locate vocally active regions and
extract fixed-length segments for downstream classification.

## Prerequisites

```bash
pip install numpy scipy librosa matplotlib sounddevice
```

## Running the tool

```bash
python pipeline/Stage4_annotate_segments.py
```

A file dialog opens. Select a WAV, MP3, M4A, FLAC, or OGG audio file.

To set the starting directory for the file dialog:

```bash
python pipeline/Stage4_annotate_segments.py --sound-dir /path/to/audio/files
```

Files shorter than 3 seconds are skipped automatically.

---

## Blob Detection Algorithm

The automatic segment proposal implements the signal/noise separation method from
**Sprengel et al. (2016)** "Audio Based Bird Species Identification using Deep Learning
Techniques" (CLEF 2016), §2.1.

### Original Sprengel (2016) algorithm

The paper describes the following pipeline for separating signal from noise:

1. **STFT**: Hanning window, size 512, 75% overlap (hop = 128).
   Normalise the magnitude spectrogram to [0, 1] by dividing by the global maximum.
   **Do not take the logarithm** for this detection step.

2. **Binary pixel mask**: for each time–frequency cell *X(t, f)*,
   - Let *row_med(f)* = median of all values in frequency row *f* across time.
   - Let *col_med(t)* = median of all values in time column *t* across frequency.
   - Set *B(t, f) = 1* if *X(t, f) > 3 × row_med(f)* **and** *X(t, f) > 3 × col_med(t)*;
     otherwise set *B(t, f) = 0*.

3. **Morphological cleaning**:
   - Apply **binary erosion** with a 4×4 structuring element.
   - Apply **binary dilation** with a 4×4 structuring element.
   This removes isolated noise pixels while preserving contiguous vocalization blobs.

4. **Column indicator vector**: *c(t) = 1* if any row of *B(:, t) = 1*, else *c(t) = 0*.
   This collapses the 2-D mask to a 1-D time signal.

5. **Smoothing**: apply two successive **binary dilation** passes with a 4×1
   (time-only) structuring element. This fills short gaps between active columns.

6. **Scale to audio length**: map the frame-rate indicator vector back to sample
   indices to obtain the signal mask in the audio time domain.

7. **Noise extraction**: repeat the same procedure with threshold **2.5×**
   (instead of 3×) and **invert** the resulting mask.
   This yields a noise mask complementary to the signal mask.

8. **Concatenation**: join all signal intervals into one audio segment; join all
   noise intervals into one audio segment. These are used for spectrogram-level
   normalisation in the original paper's classifier, not in this GUI.

### Implementation in this script

The script implements the core of the Sprengel pipeline in `detect_sound_blobs()`:

- **STFT** via `librosa.stft` with `n_fft=512`, `hop_length=128` (75% overlap),
  magnitude normalised to [0, 1].
- **Row×column threshold**: each cell must exceed `3 × row_median` **and**
  `3 × col_median` (thresholds adjustable via sliders at runtime).
- **Erosion + dilation** with a 4×4 binary structuring element (both passes).
- **Column indicator** collapsed from the 2-D mask.
- **Two dilation smoothing passes** with a 4×1 element (time axis only).
- **Segment extraction**: contiguous runs of active columns are converted to
  `(start_s, end_s)` intervals; each interval is padded to exactly 3.0 seconds,
  centred on the midpoint of the detected event.

The user can adjust erosion/dilation aggressiveness and the threshold multiplier
via on-screen sliders without re-loading the audio.

---

## Interface overview

The window shows two plots and a control panel:

- **Top plot (spectrogram)**: Log-power dB spectrogram displayed for human review.
  Blue overlay: binary mask *B* from the Sprengel detector.
  Cyan dashed outlines: raw detected event bounds (before padding to 3 s).
  Yellow boxes: final 3-second segments accepted so far.
- **Bottom plot (waveform)**: Time-domain waveform with blue shaded bars marking
  accepted 3-second segments.
- **Sliders**: Row/column threshold multiplier, erosion cycles, dilation cycles,
  and max display frequency.
- **Buttons**: Quit (no save), Play/Stop, Save.
- **Info bar**: Displays average SNR, peak SNR, and current segment count.

## Adjusting detection parameters

### Energy Threshold (lower = more sensitive)

The unified threshold applied to the normalised Sprengel score before binarisation.
Lower values detect more (including noise); higher values retain only strong peaks.

### Erosion cycles (0–4)

Number of binary erosion passes with the 4×4 structuring element.
More erosion removes small noise speckles but can also clip weak vocalizations.

### Dilation cycles (0–4)

Number of binary dilation passes with the 4×4 structuring element.
More dilation reconnects fragmented blobs but can merge distinct vocalizations.

Moving any slider re-runs the full detection pipeline and replaces all proposed
segments (existing manual edits are lost).

## Editing segments

### Removing a segment

Click on a yellow segment box in the **spectrogram** (top plot). The segment is
removed immediately.

### Repositioning a segment

Click and drag a blue bar in the **waveform** (bottom plot) left or right.
Constraints enforced during drag:
- Segment cannot overlap its neighbours.
- Segment cannot extend before time 0 or past the end of the audio.
- Segment length stays exactly 3 seconds.

Release the mouse to commit the new position. SNR values update automatically.

**Adding segments**: not supported in the GUI. To add a segment that the detector
missed, use Audacity with the exported label file — drag to mark the region, then
accept in the next pipeline stage.

## Playback

Click **Play** to hear the full audio. A red vertical line tracks playback progress
across both plots. Click **Stop** (same button) to halt playback.

## Saving

Click **Save**. The tool writes a tab-separated `.txt` file alongside the audio file
with the same base name:

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

Up to 10 segments are saved (hard limit). The window closes after saving. Final SNR
metrics are printed to the terminal.

All accepted segments generate label files — including those verified as clean and
accepted directly from the detector without further review. Uncertain clips are
confirmed in Audacity using the exported label file before corpus inclusion.

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

## Reference

Sprengel, E., Jaggi, M., Kilcher, Y., & Hofmann, T. (2016).
Audio Based Bird Species Identification using Deep Learning Techniques.
*Working Notes of CLEF 2016*. CEUR Workshop Proceedings, Vol. 1609.
