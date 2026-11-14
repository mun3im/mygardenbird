#!/usr/bin/env python3
"""Rank the 50 catalogue species by *usable* availability, splitting out ND.

Stage 1's own ranking reports regional and global totals but ignores licence,
and CC BY-NC-ND forbids derivative works -- 3-second segmentation is a
derivative, so an ND recording cannot enter the dataset at all. A species can
therefore look well-supplied on regional hours and still be unusable. The
2026-11-08 Zenodo finding (298 ND-derived clips, 25% of the Plus edition) came
from exactly that blind spot.

Columns per species, regional (lon 60-140 OR ASEAN) unless named global:
  reg_files / reg_hours            every regional recording >= 3 s
  nonnd_files / nonnd_hours        the subset that is actually usable
  nd_files / nd_pct                what the ND licence costs
  global_files / global_hours      from global_stats.json, for comparison
  A..E                             quality grades, non-ND only

Ranked by nonnd_hours, because that is the quantity that bounds the dataset.

ND detection is imported from xc_download so this file and the download
stages cannot disagree about what counts as ND.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xc_download import is_nd_licence          # noqa: E402

try:
    from config import ACTIVE_SPECIES
except ImportError:
    ACTIVE_SPECIES = None


def hours(rows):
    total = 0.0
    for r in rows:
        try:
            total += float(r.get("length_seconds") or 0)
        except (TypeError, ValueError):
            pass
    return total / 3600.0


def grade_counts(rows):
    out = {g: 0 for g in "ABCDE"}
    for r in rows:
        g = (r.get("q") or "").strip().upper()
        if g in out:
            out[g] += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", default="/Volumes/Evo/MYGARDENBIRD/per_species_csv")
    p.add_argument("--out-dir", default="/Volumes/Evo/MYGARDENBIRD/project_csv")
    a = p.parse_args()

    csv_dir, out_dir = Path(a.csv_dir), Path(a.out_dir)

    gstats = {}
    gpath = csv_dir / "global_stats.json"
    if gpath.exists():
        with open(gpath) as f:
            gstats = json.load(f)
    missing_global = [k for k, v in gstats.items() if not v.get("global_files")]

    rows = []
    for path in sorted(csv_dir.glob("*.csv")):
        species = path.stem.replace("_", " ")
        with open(path, newline="") as f:
            recs = list(csv.DictReader(f))
        if not recs:
            rows.append({"species": species, "reg_files": 0, "reg_hours": 0.0,
                         "nonnd_files": 0, "nonnd_hours": 0.0, "nd_files": 0,
                         "nd_pct": 0.0, "global_files": 0, "global_hours": 0.0,
                         **{g: 0 for g in "ABCDE"}})
            continue

        nonnd = [r for r in recs if not is_nd_licence(r.get("lic"))]
        nd = [r for r in recs if is_nd_licence(r.get("lic"))]
        g = gstats.get(species, {})

        rows.append({
            "species": species,
            "reg_files": len(recs),
            "reg_hours": round(hours(recs), 2),
            "nonnd_files": len(nonnd),
            "nonnd_hours": round(hours(nonnd), 2),
            "nd_files": len(nd),
            "nd_pct": round(100.0 * len(nd) / len(recs), 1),
            "global_files": g.get("global_files", 0),
            "global_hours": round(g.get("global_hours", 0.0), 2),
            **grade_counts(nonnd),
        })

    rows.sort(key=lambda r: r["nonnd_hours"], reverse=True)

    fields = ["rank", "species", "nonnd_hours", "nonnd_files", "reg_hours",
              "reg_files", "nd_files", "nd_pct", "global_hours", "global_files",
              "A", "B", "C", "D", "E"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "regional_ranking_nd.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"rank": i, **{k: r.get(k, "") for k in fields[1:]}})

    with open(out_dir / "regional_ranking_nd.md", "w") as f:
        f.write("# Species availability — non-ND aware\n\n")
        f.write("Regional filter: 60 < longitude < 140 OR ASEAN country "
                "(for missing longitude); recordings >= 3 s.\n\n")
        f.write("**Ranked by non-ND regional hours**, the quantity that actually "
                "bounds the dataset: CC BY-NC-ND forbids derivatives, and "
                "3-second segmentation is a derivative, so ND recordings cannot "
                "be used at all. `ND%` is the share of the regional pool lost "
                "to that licence.\n\n")
        if missing_global:
            f.write(f"> Global columns are 0 for {len(missing_global)} species "
                    f"whose entry in `global_stats.json` has no counts.\n\n")
        f.write("| Rank | Species | non-ND Hours | non-ND Files | Reg Hours | "
                "Reg Files | ND Files | ND% | Global Hours | Global Files | "
                "A | B | C | D | E |\n")
        f.write("|-----:|---------|-------------:|-------------:|----------:|"
                "----------:|---------:|----:|-------------:|-------------:|"
                "--:|--:|--:|--:|--:|\n")
        for i, r in enumerate(rows, 1):
            f.write(f"| {i} | {r['species']} | {r['nonnd_hours']:.2f} | "
                    f"{r['nonnd_files']} | {r['reg_hours']:.2f} | "
                    f"{r['reg_files']} | {r['nd_files']} | {r['nd_pct']:.1f} | "
                    f"{r['global_hours']:.2f} | {r['global_files']} | "
                    f"{r['A']} | {r['B']} | {r['C']} | {r['D']} | {r['E']} |\n")

        tot_reg = sum(r["reg_files"] for r in rows)
        tot_nd = sum(r["nd_files"] for r in rows)
        f.write(f"\n**Totals:** {tot_reg} regional recordings, "
                f"{tot_reg - tot_nd} usable, {tot_nd} ND "
                f"({100.0 * tot_nd / tot_reg:.1f}% unusable).\n")

    print(f"Wrote {out_dir / 'regional_ranking_nd.md'}")
    print(f"Wrote {out_dir / 'regional_ranking_nd.csv'}")
    if missing_global:
        print(f"NOTE: {len(missing_global)} species have no global counts in "
              f"global_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
