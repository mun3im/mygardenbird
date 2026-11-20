"""Stage 7a (optional; nothing downstream depends on it). Whole-recording BirdNET audit of the SOURCE FLACs behind a class.

Per-clip restricted scores cannot catch a mis-filed Xeno-canto upload (wrong
species or no bird): the 15-class argmax picks the target for anything that is
not clearly another class.  This runs BirdNET v2.4 over each contributing
source's full recording, with the location/week species filter, and reports
which species it detects and whether the target class is among them.

  python Stage7a_audit_sources_birdnet.py --species "Scarlet-backed Flowerpecker" \
      --out birdnet_results/source_audit_flowerpecker.csv
"""
import argparse, csv, json, math, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import Stage7_eval_birdnet as S7
import birdnet_analyzer.config as cfg
import birdnet_analyzer.model as M
import birdnet_analyzer.audio as A

ROOT = Path("/Volumes/Evo/MYGARDENBIRD")
SF_THRESH = 0.03      # BirdNET's default location-filter threshold
MIN_CONF = 0.25       # BirdNET's default detection threshold

def week_of(date):
    try:
        y, m, d = (int(x) for x in date.split("-")[:3])
        return max(1, min(48, (m - 1) * 4 + min(4, (d - 1) // 8 + 1)))
    except Exception:
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-location", action="store_true")
    a = ap.parse_args()
    labels = S7.setup_birdnet()
    target = S7.SPECIES_MAP[a.species]
    ti = labels.index(target)
    rec = {r["source_id"]: r for r in csv.DictReader(open(ROOT / "project_csv/recordings.csv"))
           if r["species_common"] == a.species}
    flacs = {p.stem[2:]: p for p in (ROOT / "per_species_flacs" / a.species).rglob("*.flac")}
    rows = []
    for n, sid in enumerate(sorted(rec), 1):
        if sid not in flacs:
            print("no flac for", sid); continue
        meta = json.load(open(ROOT / f"xc_metadata_20261113/by_id/{sid}.json"))["recordings"][0]
        lat, lon = float(meta["lat"] or "nan"), float(meta["lon"] or "nan")
        mask = np.ones(len(labels), bool); target_in_filter = ""
        if not a.no_location and not math.isnan(lat) and not math.isnan(lon):
            pf = M.predict_filter(lat, lon, week_of(meta.get("date", "")))
            mask = pf >= SF_THRESH; target_in_filter = int(mask[ti])
        sig, rate = A.open_audio_file(str(flacs[sid]), sample_rate=48000)
        chunks = A.split_signal(sig, rate, seconds=3.0, overlap=0.0, minlen=1.0)
        if not chunks: print("empty", sid); continue
        pred = M.flat_sigmoid(np.array(M.predict(np.array(chunks, dtype="float32"))), sensitivity=-1.0)
        filt = np.where(mask[None, :], pred, 0.0)
        top = filt.argmax(1); topc = filt.max(1)
        det = Counter(labels[i].split("_")[1] for i, c in zip(top, topc) if c >= MIN_CONF)
        tgt = pred[:, ti]
        rows.append({
            "source_id": sid, "recordist": rec[sid]["recordist"], "country": rec[sid]["country"],
            "grade": rec[sid]["quality_grade"], "xc_also": ";".join(meta.get("also", [])),
            "windows": len(chunks), "target_in_location_filter": target_in_filter,
            "target_max": round(float(tgt.max()), 3),
            "windows_target_ge_0.25": int((tgt >= MIN_CONF).sum()),
            "windows_target_is_top_filtered": int(((top == ti) & (topc >= MIN_CONF)).sum()),
            "windows_any_detection": int((topc >= MIN_CONF).sum()),
            "top_species": "; ".join(f"{k} {v}" for k, v in det.most_common(4)),
        })
        print(f"[{n}/{len(rec)}] {sid} win={len(chunks)} target>=.25={rows[-1]['windows_target_ge_0.25']} {rows[-1]['top_species'][:70]}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, len(rows))

if __name__ == "__main__":
    main()
