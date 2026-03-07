#!/usr/bin/env python3
"""
plot_spectrograms.py

10-panel mel-spectrogram figure (5×2 grid) — one representative clip per species.
Clip selected as the one closest to the per-species median SNR.

Output: fig/example_spectrograms.pdf + fig/example_spectrograms.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

QC_CSV   = Path("/Volumes/Evo/MYGARDENBIRD/metadata16khz/qc_report.csv")
SEG_DIR  = Path("/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz")
OUT_DIR  = Path("/Users/mun3im/Dropbox/Paper Scientific Data/fig")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MELS   = 128
HOP      = 256
N_FFT    = 1024
F_MAX    = 8000   # Hz — clips are 16 kHz, so Nyquist = 8 kHz

# ---------------------------------------------------------------------------
# Load QC report, pick representative clip per species
# ---------------------------------------------------------------------------
df = pd.read_csv(QC_CSV)
df["snr_db"] = pd.to_numeric(df["snr_db"], errors="coerce")
df = df.dropna(subset=["snr_db"])
df = df[df["valid"] == True]

# Sort species alphabetically for consistent panel order
species_list = sorted(df["species"].unique())

def pick_clip(sp_df):
    """Return the filename whose SNR is closest to the species median."""
    median_snr = sp_df["snr_db"].median()
    idx = (sp_df["snr_db"] - median_snr).abs().idxmin()
    return sp_df.loc[idx, "file"]

representative = {}
for sp in species_list:
    sub = df[df["species"] == sp]
    fname = pick_clip(sub)
    wav = SEG_DIR / sp / fname
    if wav.exists():
        representative[sp] = wav
    else:
        # Try case-insensitive match
        for d in SEG_DIR.iterdir():
            if d.name.lower() == sp.lower():
                candidate = d / fname
                if candidate.exists():
                    representative[sp] = candidate
                    break

# ---------------------------------------------------------------------------
# Plot: 2×5 grid
# ---------------------------------------------------------------------------
n_rows = 2
n_cols = 5

fig = plt.figure(figsize=(15, 6))
gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.35, wspace=0.25)

for idx, sp in enumerate(species_list[:10]):  # First 10 species
    if sp not in representative:
        print(f"  Warning: no clip found for {sp}")
        continue

    wav_path = representative[sp]
    audio, sr = librosa.load(wav_path, sr=None, mono=True)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT,
        fmin=0, fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    row, col = divmod(idx, n_cols)
    ax = fig.add_subplot(gs[row, col])

    librosa.display.specshow(
        mel_db,
        sr=sr, hop_length=HOP,
        x_axis=None, y_axis=None,  # No axes initially
        fmax=F_MAX,
        ax=ax,
        cmap="magma",
    )

    # Species name as panel title
    snr_val = df.loc[df["file"] == wav_path.name, "snr_db"].values
    snr_str = f"SNR {snr_val[0]:.1f} dB" if len(snr_val) else ""
    ax.set_title(f"{sp}\n{snr_str}", fontsize=9, fontfamily="serif", pad=4)

    # Remove all axis labels (explained in caption)
    ax.set_xticks([])
    ax.set_yticks([])

for ext in ("pdf", "png"):
    out = OUT_DIR / f"example_spectrograms.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()
