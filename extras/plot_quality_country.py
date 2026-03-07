#!/usr/bin/env python3
"""
plot_quality_country.py

Two-panel figure:
  (a) Quality distribution — stacked bar per species (A / B / C)
  (b) Recording country distribution — horizontal bar of unique sources

Output: fig/quality_country.pdf + fig/quality_country.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

MANIFEST = Path("/Volumes/Evo/MYGARDENBIRD/project_csv/recordings.csv")
OUT_DIR  = Path("/Users/mun3im/Dropbox/Paper Scientific Data/fig")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"Common myna", "Zebra dove"}

# Okabe-Ito colours for quality A B C
QUALITY_COLOURS = {
    "A": "#009E73",
    "B": "#56B4E9",
    "C": "#E69F00",
}
QUALITY_LABELS = {"A": "Quality A", "B": "Quality B", "C": "Quality C"}

# ---------------------------------------------------------------------------
df = pd.read_csv(MANIFEST)
df = df[~df["species_common"].isin(EXCLUDE)]

# --- Panel (a): quality stacked bar ---
quality_ct = (df.groupby(["species_common", "quality_grade"])
              .size()
              .unstack(fill_value=0))
# Ensure consistent column order (only A, B, C)
for q in ["A", "B", "C"]:
    if q not in quality_ct.columns:
        quality_ct[q] = 0
quality_ct = quality_ct[["A", "B", "C"]]

# Sort species by Quality A count descending, then by Quality B count descending
quality_ct = quality_ct.sort_values(["A", "B"], ascending=False)
species_labels = [s.replace(" ", "\n") for s in quality_ct.index]

# --- Panel (b): country horizontal bar ---
uniq = df.drop_duplicates(subset=["source_id"])
country_ct = uniq["country"].value_counts().head(12)

# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5),
    gridspec_kw={"width_ratios": [1.4, 1]},
)

# --- (a) ---
bottom = pd.Series([0] * len(quality_ct), index=quality_ct.index)
for quality in ["A", "B", "C"]:
    vals = quality_ct[quality]
    ax1.bar(
        range(len(quality_ct)), vals,
        bottom=bottom,
        color=QUALITY_COLOURS[quality],
        label=QUALITY_LABELS[quality],
        edgecolor="white", linewidth=0.4,
        width=0.65,
    )
    bottom = bottom + vals

ax1.set_xticks(range(len(quality_ct)))
ax1.set_xticklabels(species_labels, fontsize=8, fontfamily="serif", rotation=45, ha="right")
ax1.set_ylabel("Number of clips", fontsize=9, fontfamily="serif")
ax1.set_title("(a) Xeno-canto quality per species",
              fontsize=10, fontfamily="serif", pad=6)
ax1.set_ylim(0, 160)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(40))
# Add dashed lines every 40 on y-axis
for y_val in range(40, 160, 40):
    ax1.axhline(y=y_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.4, zorder=1)
ax1.grid(axis="y", linewidth=0.3, alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(fontsize=8, prop={"family": "serif"},
           framealpha=0.9, edgecolor="#cccccc",
           loc="upper right")

# --- (b) ---
countries = country_ct.index.tolist()
counts    = country_ct.values
colours_b = ["#0072B2" if c == "Malaysia" else "#56B4E9" for c in countries]

bars = ax2.barh(range(len(countries)), counts,
                color=colours_b, edgecolor="white", linewidth=0.4,
                height=0.65)
ax2.set_yticks(range(len(countries)))
ax2.set_yticklabels(countries, fontsize=8, fontfamily="serif")
ax2.invert_yaxis()
ax2.set_xlabel("Unique source recordings", fontsize=9, fontfamily="serif")
ax2.set_title("(b) Country of recording (top 12)",
              fontsize=10, fontfamily="serif", pad=6)
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax2.grid(axis="x", linewidth=0.3, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)

# Annotate Malaysia bar
my_idx = countries.index("Malaysia") if "Malaysia" in countries else None
if my_idx is not None:
    ax2.get_yticklabels()[my_idx].set_fontweight("bold")

plt.tight_layout(pad=1.5)
for ext in ("pdf", "png"):
    out = OUT_DIR / f"quality_country.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()
