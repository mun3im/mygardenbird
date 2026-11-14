#!/usr/bin/env python3
"""Estimate the annotation work left to reach a per-class clip target.

Pulls together the three things that determine it:

  1. clips now      -- segments in non-empty sidecars under per_species_flacs/
  2. sources on disk -- FLACs downloaded, split into annotated (has .txt) and
                        not-yet-touched
  3. sources available -- non-ND regional recordings from the fresh Stage 1
                        catalogue, minus what is already downloaded

Yield is projected at the *recent* sparse-sampling rate (~3.5 segments per
accepted source), not each species' lifetime average: since early November the
project deliberately takes fewer segments per recording to spread clips across
more distinct sources, so the lifetime figure is inflated by early dense
annotation and understates the work left by roughly 40%.

Sources are also discounted by an observed accept rate, because a sidecar with
zero segments is a valid outcome (explicit rejection) and those files consume
annotation time without producing clips.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SEG_PER_SOURCE = 3.5     # recent sparse-sampling rate
FLAC_ROOT = Path("/Volumes/Evo/MYGARDENBIRD/per_species_flacs")
RANK_CSV = Path("/Volumes/Evo/MYGARDENBIRD/project_csv/regional_ranking_nd.csv")


def species_state(species_dir):
    """-> (clips, annotated_srcs, accepted_srcs, rejected_srcs, unannotated_srcs)"""
    flacs = list(species_dir.rglob("*.flac"))
    clips = accepted = rejected = 0
    annotated_stems = set()

    for txt in species_dir.rglob("*.txt"):
        if txt.name.endswith(".txt.bak") or ".bak" in txt.suffixes:
            continue
        annotated_stems.add(txt.stem)
        n = sum(1 for line in txt.read_text().splitlines() if line.strip())
        clips += n
        if n:
            accepted += 1
        else:
            rejected += 1

    unannotated = sum(1 for f in flacs if f.stem not in annotated_stems)
    return clips, len(annotated_stems), accepted, rejected, unannotated


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=600)
    p.add_argument("--seg-per-source", type=float, default=SEG_PER_SOURCE)
    p.add_argument("--min-per-file", type=float, default=2.0,
                   help="Minutes of annotator time per source file (default 2)")
    a = p.parse_args()

    avail = {}
    if RANK_CSV.exists():
        with open(RANK_CSV, newline="") as f:
            for r in csv.DictReader(f):
                avail[r["species"]] = int(r["nonnd_files"] or 0)
    else:
        print(f"WARNING: {RANK_CSV} missing; availability unknown", file=sys.stderr)

    rows = []
    for d in sorted(FLAC_ROOT.iterdir()):
        if not d.is_dir():
            continue
        clips, annotated, accepted, rejected, unannot = species_state(d)
        need = max(0, a.target - clips)

        # Accept rate from this species' own history; fall back to a neutral
        # 0.85 before there is enough evidence to estimate one.
        #
        # A species where every sidecar is zero-length (accepted == 0) is a
        # genuine 0% -- Olive-backed Sunbird's 86 sources were all rejected on
        # taxonomic grounds, the real species being Ornate Sunbird. Taking
        # that literally would make the yield zero and divide by it, so the
        # species is reported as unworkable rather than as needing no work.
        exhausted = annotated >= 10 and accepted == 0
        accept_rate = accepted / annotated if annotated >= 10 else 0.85

        pool = avail.get(d.name, 0)
        on_disk_total = annotated + unannot
        # Fresh catalogue count minus what we already hold. The catalogue
        # counts recordings, the disk counts files, so a species that holds
        # more than the pool reports 0 rather than a negative.
        to_download = max(0, pool - on_disk_total)
        reachable = unannot + to_download

        # Sources needed, discounted by the accept rate.
        eff = a.seg_per_source * accept_rate
        srcs_needed = int(-(-need // eff)) if need and eff > 0 else 0

        rows.append({
            "species": d.name, "clips": clips, "need": need,
            "annotated": annotated, "accepted": accepted, "rejected": rejected,
            "accept_rate": accept_rate, "unannot": unannot,
            "pool": pool, "to_download": to_download,
            "srcs_needed": srcs_needed, "reachable": reachable,
            "shortfall": max(0, srcs_needed - reachable),
            "minutes": srcs_needed * a.min_per_file,
            "exhausted": exhausted,
        })

    rows.sort(key=lambda r: (-r["need"], r["species"]))

    print(f"\nTarget {a.target} clips/species | "
          f"{a.seg_per_source} seg/source | {a.min_per_file} min/file\n")
    hdr = (f"{'Species':30s}{'Clips':>6}{'Need':>6}{'Acc%':>6}"
           f"{'OnDisk':>8}{'DLable':>8}{'SrcReq':>8}{'Short':>7}{'Hours':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "  (all rejected)" if r["exhausted"] else ""
        print(f"{r['species']:30s}{r['clips']:6d}{r['need']:6d}"
              f"{100 * r['accept_rate']:5.0f}%{r['unannot']:8d}"
              f"{r['to_download']:8d}{r['srcs_needed']:8d}"
              f"{r['shortfall']:7d}{r['minutes'] / 60:7.1f}{flag}")

    tot_need = sum(r["need"] for r in rows)
    tot_src = sum(r["srcs_needed"] for r in rows)
    tot_short = sum(r["shortfall"] for r in rows)
    tot_hours = sum(r["minutes"] for r in rows) / 60.0
    print("-" * len(hdr))
    print(f"{'TOTAL':30s}{sum(r['clips'] for r in rows):6d}{tot_need:6d}"
          f"{'':6s}{sum(r['unannot'] for r in rows):8d}"
          f"{sum(r['to_download'] for r in rows):8d}{tot_src:8d}"
          f"{tot_short:7d}{tot_hours:7.1f}")

    blocked = [r for r in rows if r["shortfall"] > 0]
    if blocked:
        print(f"\nCannot reach {a.target} even using every non-ND source:")
        for r in blocked:
            cap = r["clips"] + int(r["reachable"] * a.seg_per_source
                                   * r["accept_rate"])
            print(f"  {r['species']:30s} short {r['shortfall']:4d} sources "
                  f"-> ceiling ~{cap} clips")

    dead = [r for r in rows if r["exhausted"] and r["need"] > 0]
    if dead:
        print("\nEvery downloaded source rejected -- no yield to project:")
        for r in dead:
            print(f"  {r['species']:30s} {r['annotated']} sources, all "
                  f"rejected; not costed above")

    print(f"\nAt 2 h/day: ~{tot_hours / 2:.0f} days of annotation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
