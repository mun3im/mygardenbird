#!/usr/bin/env python3
"""
Zero-shot BirdNET v2.4 evaluation on MyGardenBird test sets.

Uses BirdNET-Analyzer's internal modules directly — bypasses the high-level
analyze pipeline (which applies a species-list filter that silently drops all
predictions) and calls the model directly to get raw per-class scores.

For each 3-second clip:
  - Load audio at 48 kHz (BirdNET native rate)
  - Run model.predict() → shape (1, 6521)
  - Apply flat_sigmoid
  - Restrict scores to the 12 MyGardenBird classes
  - argmax → predicted species
  - Compare to ground truth

Run with:
    conda run -n birdnet python eval_birdnet_mygardenbird.py \
        --splits_csv metadata16khz/splits_mip_75_10_15.csv \
        --audio_dir /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz \
        --output_dir birdnet_results/split_75_10_15

Run all 3 splits:
    for split in 70_15_15 75_10_15 80_10_10; do
        conda run -n birdnet python eval_birdnet_mygardenbird.py \
            --splits_csv metadata16khz/splits_mip_${split}.csv \
            --audio_dir /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz \
            --output_dir birdnet_results/split_${split}
    done
"""

import argparse
import json
import os
import sys

BIRDNET_DIR = os.path.expanduser("~/BirdNET-Analyzer")
sys.path.insert(0, BIRDNET_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)

import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as birdnet_model
import birdnet_analyzer.audio as birdnet_audio

# ── Species mapping ───────────────────────────────────────────────────────────
# Maps MyGardenBird directory name → BirdNET label (scientific_common format)
SPECIES_MAP = {
    "Asian Koel":               "Eudynamys scolopaceus_Asian Koel",
    "Collared Kingfisher":      "Todiramphus chloris_Collared Kingfisher",
    "Common Iora":              "Aegithina tiphia_Common Iora",
    "Common Tailorbird":        "Orthotomus sutorius_Common Tailorbird",
    "Coppersmith Barbet":       "Psilopogon haemacephalus_Coppersmith Barbet",
    "Large-tailed Nightjar":    "Caprimulgus macrurus_Large-tailed Nightjar",
    "Olive-backed Sunbird":     "Cinnyris jugularis_Olive-backed Sunbird",
    "Pied Fantail":             "Rhipidura javanica_Malaysian Pied-Fantail",
    "Spotted Dove":             "Streptopelia chinensis_Spotted Dove",
    "White-breasted Waterhen":  "Amaurornis phoenicurus_White-breasted Waterhen",
    "White-throated Kingfisher":"Halcyon smyrnensis_White-throated Kingfisher",
    "Yellow-vented Bulbul":     "Pycnonotus goiavier_Yellow-vented Bulbul",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot BirdNET v2.4 evaluation on MyGardenBird"
    )
    parser.add_argument(
        "--splits_csv", default=None,
        help="Path to MIP splits CSV. If omitted, all clips in --audio_dir are evaluated.",
    )
    parser.add_argument(
        "--audio_dir", required=True,
        help="Path to mygardenbird16khz directory",
    )
    parser.add_argument(
        "--output_dir", default="birdnet_results",
        help="Directory to write results into",
    )
    parser.add_argument(
        "--lat", type=float, default=3.139,
        help="Latitude for metadata model (default: 3.139 Kuala Lumpur)",
    )
    parser.add_argument(
        "--lon", type=float, default=101.687,
        help="Longitude for metadata model (default: 101.687 Kuala Lumpur)",
    )
    return parser.parse_args()


def load_test_clips(splits_csv, audio_dir):
    """Return list of {filepath, species} dicts.

    If splits_csv is given, uses only clips in the test split.
    If splits_csv is None, uses every WAV in audio_dir (full dataset eval).
    """
    if splits_csv is not None:
        with open(splits_csv) as f:
            lines = [l for l in f if not l.startswith("#")]
        df = pd.read_csv(pd.io.common.StringIO("".join(lines)))
        test_ids = set(df.loc[df["split"] == "test", "file_id"].str.lower())
    else:
        test_ids = None  # all clips

    clips = []
    for species_dir in sorted(Path(audio_dir).iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue
        if species_dir.name not in SPECIES_MAP:
            continue
        for wav in species_dir.glob("*.wav"):
            if test_ids is None or wav.stem.lower() in test_ids:
                clips.append({"filepath": str(wav), "species": species_dir.name})

    label = "all" if test_ids is None else "test"
    print(f"Clips ({label}): {len(clips)} across {len(set(c['species'] for c in clips))} species")
    return clips


def setup_birdnet():
    """Configure and load BirdNET model; return label list."""
    _ckpt = os.path.join(BIRDNET_DIR, "checkpoints/V2.4")
    cfg.set_config({
        "MODEL_PATH":  os.path.join(_ckpt, "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"),
        "LABELS_FILE": os.path.join(_ckpt, "BirdNET_GLOBAL_6K_V2.4_Labels.txt"),
        "SAMPLE_RATE": cfg.SAMPLE_RATE,   # 48000
        "SIG_LENGTH":  cfg.SIG_LENGTH,    # 3.0
        "APPLY_SIGMOID":       True,
        "SIGMOID_SENSITIVITY": 1.0,
        "SIG_OVERLAP": 0.0,
        "SIG_MINLEN":  1.0,
        "CUSTOM_CLASSIFIER": None,
        "USE_PERCH":   False,
    })
    print("Loading BirdNET v2.4 model...")
    birdnet_model.load_model()
    labels_path = os.path.join(_ckpt, "BirdNET_GLOBAL_6K_V2.4_Labels.txt")
    with open(labels_path) as f:
        labels = [l.strip() for l in f if l.strip()]
    print(f"Model loaded. Label count: {len(labels)}")
    return labels


def get_scores(filepath):
    """Return raw sigmoid scores (n_classes,) for a single 3-second clip."""
    try:
        sig, rate = birdnet_audio.open_audio_file(
            filepath,
            sample_rate=cfg.SAMPLE_RATE,
            offset=0.0,
            duration=cfg.SIG_LENGTH,
        )
        chunks = birdnet_audio.split_signal(
            sig, rate,
            seconds=cfg.SIG_LENGTH,
            overlap=cfg.SIG_OVERLAP,
            minlen=cfg.SIG_MINLEN,
        )
        if not chunks:
            return None
        data = np.array(chunks, dtype="float32")
        pred = birdnet_model.predict(data)
        if cfg.APPLY_SIGMOID:
            pred = birdnet_model.flat_sigmoid(
                np.array(pred), sensitivity=-cfg.SIGMOID_SENSITIVITY
            )
        # chunks=1 for a 3-second clip; take first (only) chunk
        return pred[0]
    except Exception as e:
        print(f"\nWARNING: {filepath}: {e}")
        return None


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clips = load_test_clips(args.splits_csv, args.audio_dir)
    labels = setup_birdnet()
    label_index = {lbl: i for i, lbl in enumerate(labels)}

    # Indices for our 12 species in the 6521-class label space
    species_names = sorted(SPECIES_MAP.keys())
    birdnet_labels = [SPECIES_MAP[s] for s in species_names]
    class_indices = [label_index[lbl] for lbl in birdnet_labels]

    print(f"\nRunning inference on {len(clips)} clips...")
    rows = []
    for clip in tqdm(clips):
        scores = get_scores(clip["filepath"])
        if scores is None:
            # Treat as uniform — model will be wrong but clip is counted
            scores = np.zeros(len(labels), dtype="float32")

        # Restrict to our 12 species
        class_scores = scores[class_indices]           # (12,)
        pred_idx = int(np.argmax(class_scores))
        pred_species = species_names[pred_idx]
        max_conf = float(class_scores[pred_idx])

        rows.append({
            "filepath":   clip["filepath"],
            "ground_truth": clip["species"],
            "predicted":  pred_species,
            "correct":    int(pred_species == clip["species"]),
            "max_conf":   max_conf,
            **{f"score_{s.replace(' ', '_')}": float(class_scores[i])
               for i, s in enumerate(species_names)},
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_clip_results.csv", index=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true = df["ground_truth"].tolist()
    y_pred = df["predicted"].tolist()
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        labels=species_names,
        target_names=species_names,
        zero_division=0,
    )

    # Per-class AUC (one-vs-rest)
    auc_rows = []
    for i, sp in enumerate(species_names):
        true_bin = (df["ground_truth"] == sp).astype(int).values
        conf_col = f"score_{sp.replace(' ', '_')}"
        if conf_col in df.columns and true_bin.sum() > 0:
            auc = roc_auc_score(true_bin, df[conf_col].values)
            auc_rows.append({"species": sp, "auc": auc})
    macro_auc = np.mean([r["auc"] for r in auc_rows]) if auc_rows else float("nan")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=species_names)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=species_names, yticklabels=species_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Ground Truth", fontsize=10)
    ax.set_title(f"BirdNET v2.4 Zero-Shot — MyGardenBird\nAccuracy: {accuracy*100:.2f}%",
                 fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    # ── Save summary ─────────────────────────────────────────────────────────
    splits_label = args.splits_csv if args.splits_csv else "all clips (no split)"
    summary = {
        "splits_csv":       splits_label,
        "total_test_clips": len(df),
        "accuracy":         round(accuracy, 6),
        "macro_auc":        round(macro_auc, 6),
        "per_class_auc":    {r["species"]: round(r["auc"], 4) for r in auc_rows},
    }
    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    report_text = (
        f"BirdNET v2.4 Zero-Shot Evaluation — MyGardenBird\n"
        f"{'=' * 60}\n"
        f"Splits CSV : {splits_label}\n"
        f"Test clips : {len(df)}\n"
        f"Accuracy   : {accuracy*100:.2f}%\n"
        f"Macro AUC  : {macro_auc:.4f}\n"
        f"{'=' * 60}\n\n"
        f"{report}\n"
        f"{'=' * 60}\n"
        f"Per-class AUC (one-vs-rest):\n"
    )
    for r in auc_rows:
        report_text += f"  {r['species']:<28} {r['auc']:.4f}\n"
    report_text += f"  {'Macro avg':<28} {macro_auc:.4f}\n"

    with open(out / "classification_report.txt", "w") as f:
        f.write(report_text)

    print(f"\n{'=' * 60}")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"Results  : {out}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
