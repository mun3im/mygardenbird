#!/usr/bin/env python3
"""
Stage 1: Fetch Xeno-Canto metadata for all target species and rank by
regional data availability.

Regional definition: 60 < longitude < 140 OR ASEAN countries (for missing longitude).

Uses the XC v3 API (API key required).
Reuses fetch logic from xc_get_metadata_all_seabirds_histo.py.
"""

import argparse
import csv
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import SPECIES, VALID_QUALITIES, resolve_species, PER_SPECIES_CSV

# ── constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_URL_V3 = "https://xeno-canto.org/api/3/recordings"
RATE_LIMIT_DELAY = 0.15  # seconds between requests
MAX_RETRIES = 4
REQUEST_TIMEOUT = 30  # seconds

# Regional filtering: 60 < longitude < 140 OR ASEAN countries (for missing lon)
LONGITUDE_MIN = 60.0
LONGITUDE_MAX = 140.0

ASEAN_COUNTRIES = [
    "Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia",
    "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam",
    "Timor-Leste",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xc_stage1")

# ── API helpers ──────────────────────────────────────────────────────────────


def _load_api_key(cli_key: Optional[str]) -> str:
    """Resolve API key from CLI arg, env var, or xc_key.txt (in that order)."""
    if cli_key:
        return cli_key
    env_key = os.environ.get("XENO_API_KEY")
    if env_key:
        return env_key
    key_file = os.path.join(SCRIPT_DIR, "xc_key.txt")
    if os.path.isfile(key_file):
        with open(key_file) as f:
            key = f.read().strip()
        if key:
            return key
    logger.error(
        "No API key found. Pass --api-key, set XENO_API_KEY env var, "
        "or place key in xc_key.txt."
    )
    sys.exit(2)


def _build_tag_query(scientific_name: str) -> str:
    """Build gen:"Genus" sp:"species" query from scientific name."""
    parts = scientific_name.strip().split()
    if len(parts) >= 2:
        return f'gen:"{parts[0]}" sp:"{parts[1]}"'
    return f'en:"{scientific_name}"'


def _request_with_retries(
    url: str, params: Dict[str, str], headers: Dict[str, str], retries: int = 0
) -> Optional[requests.Response]:
    """GET with exponential backoff on transient errors."""
    try:
        time.sleep(RATE_LIMIT_DELAY)
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp
        if 400 <= resp.status_code < 500:
            logger.warning(
                "Client error %d for %s — %s",
                resp.status_code, resp.url, resp.text[:200],
            )
            return None
        if 500 <= resp.status_code < 600:
            raise requests.exceptions.HTTPError(f"Server error {resp.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            backoff = 2 ** retries
            logger.warning("Request failed (attempt %d) — retrying in %ds: %s", retries + 1, backoff, e)
            time.sleep(backoff)
            return _request_with_retries(url, params, headers, retries + 1)
        logger.error("Request failed after %d retries: %s", MAX_RETRIES, e)
        return None


def _fetch_all_records(query_tag: str, api_key: str) -> Optional[List[dict]]:
    """Paginate through XC v3 API for one query. Returns list of record dicts or None."""
    headers = {
        "User-Agent": "seabird-fetcher/1.0",
        "Accept": "application/json",
    }
    params: Dict[str, object] = {"key": api_key, "query": query_tag, "page": 1}
    all_records: List[dict] = []

    while True:
        logger.info("  page %d  query: %s", params["page"], query_tag)
        resp = _request_with_retries(BASE_URL_V3, params=params, headers=headers)
        if resp is None:
            return None
        try:
            data = resp.json()
        except Exception as e:
            logger.error("JSON parse error: %s", e)
            return None

        recordings = data.get("recordings", [])
        if not recordings:
            break
        all_records.extend(recordings)

        # detect last page
        num_pages = None
        for k in ("numPages", "num_pages", "numPagesTotal", "num_pages_total"):
            if k in data:
                try:
                    num_pages = int(data[k])
                    break
                except Exception:
                    continue
        if num_pages and params["page"] >= num_pages:
            break
        params["page"] += 1

    return all_records


def _save_records_csv(
    records: List[dict],
    common_name: str,
    output_dir: str,
    lon_min: float = LONGITUDE_MIN,
    lon_max: float = LONGITUDE_MAX,
    no_region_filter: bool = False,
    min_duration: float = 3.0,
) -> str:
    """Save REGIONAL-FILTERED records to CSV.

    Applies regional filtering (lon_min–lon_max OR ASEAN) + duration >=min_duration
    before saving.  Pass no_region_filter=True to skip the geographic filter entirely.
    This eliminates duplicate filtering in Stage 2.

    An empty records list produces a header-only CSV so the species still
    appears in the ranking with 0 counts.

    Uses English common name for filename (e.g., "Javan_Myna.csv").
    """
    MIN_LENGTH_S = min_duration

    df = pd.DataFrame(records)

    # Drop unnecessary columns
    for col in ("sono", "osci"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    # Convert length to length_seconds
    if "length" in df.columns:
        df["length_seconds"] = pd.to_numeric(df["length"], errors="coerce")
        mask = df["length_seconds"].isna() & df["length"].notna()
        if mask.any():
            def _convert(ts):
                try:
                    s = str(ts).strip()
                    if ":" in s:
                        parts = [float(x) for x in s.split(":")]
                        if len(parts) == 2:
                            return int(parts[0]) * 60 + parts[1]
                        if len(parts) == 3:
                            return int(parts[0]) * 3600 + int(parts[1]) * 60 + parts[2]
                    return float(s)
                except Exception:
                    return None
            df.loc[mask, "length_seconds"] = df.loc[mask, "length"].apply(_convert)

    # APPLY REGIONAL FILTER BEFORE SAVING
    if not df.empty:
        # Keep only downloadable rows (non-empty file field)
        if "file" in df.columns:
            df = df[df["file"].notna() & (df["file"].astype(str).str.strip() != "")]

        # Regional subset: lon_min < longitude < lon_max OR ASEAN countries (for missing lon)
        if no_region_filter:
            pass  # keep all records regardless of geography
        else:
            df["_lon"] = pd.to_numeric(df.get("lng"), errors="coerce")

            # Condition 1: Valid longitude in range [lon_min, lon_max]
            lon_mask = df["_lon"].notna() & (df["_lon"] > lon_min) & (df["_lon"] < lon_max)

            # Condition 2: Missing longitude AND country in ASEAN list
            missing_lon_mask = df["_lon"].isna()
            if "cnt" in df.columns:
                asean_mask = df["cnt"].isin(ASEAN_COUNTRIES)
                fallback_mask = missing_lon_mask & asean_mask
            else:
                fallback_mask = pd.Series([False] * len(df), index=df.index)

            # Combine: longitude range OR (missing lon AND ASEAN country)
            df = df[lon_mask | fallback_mask].copy()
            df.drop(columns=["_lon"], inplace=True)

        # Filter by minimum duration
        if "length_seconds" in df.columns:
            df = df[df["length_seconds"].notna() & (df["length_seconds"] >= MIN_LENGTH_S)]

    os.makedirs(output_dir, exist_ok=True)
    safe = common_name.replace(" ", "_").replace("/", "-").replace(":", "-")
    path = os.path.join(output_dir, f"{safe}.csv")
    df.to_csv(path, index=False)
    logger.info("Saved %d regional records → %s", len(df), path)
    return path


# ── fetch command ────────────────────────────────────────────────────────────


def fetch_metadata(
    species_list,
    output_dir: str,
    api_key: str,
    dry_run: bool,
    lon_min: float = LONGITUDE_MIN,
    lon_max: float = LONGITUDE_MAX,
    no_region_filter: bool = False,
    min_duration: float = 3.0,
):
    """Fetch XC metadata for each species in *species_list*.

    Also saves global statistics (before regional filtering) to global_stats.json
    for use in ranking report.
    """
    import json
    global_stats = {}

    for idx, (common, scientific, code) in enumerate(species_list, 1):
        tag = f"[{idx}/{len(species_list)}]"
        if dry_run:
            print(f"{tag} [dry-run] Would fetch: {common} ({scientific})")
            continue

        logger.info("%s Fetching: %s (%s)", tag, common, scientific)
        query = _build_tag_query(scientific)
        records = _fetch_all_records(query, api_key)

        if records is None:
            logger.warning("%s  FAILED (network/API error) for %s — saving empty CSV", tag, common)
            records = []
        if not records:
            logger.info("%s  0 records for %s", tag, common)

        # Calculate global stats BEFORE filtering
        global_files = len(records)
        global_hours = 0.0
        if records and "length" in pd.DataFrame(records).columns:
            df_global = pd.DataFrame(records)
            # Convert length to seconds
            df_global["length_seconds"] = pd.to_numeric(df_global["length"], errors="coerce")
            mask = df_global["length_seconds"].isna() & df_global["length"].notna()
            if mask.any():
                def _convert(ts):
                    try:
                        s = str(ts).strip()
                        if ":" in s:
                            parts = [float(x) for x in s.split(":")]
                            if len(parts) == 2:
                                return int(parts[0]) * 60 + parts[1]
                            if len(parts) == 3:
                                return int(parts[0]) * 3600 + int(parts[1]) * 60 + parts[2]
                        return float(s)
                    except Exception:
                        return None
                df_global.loc[mask, "length_seconds"] = df_global.loc[mask, "length"].apply(_convert)
            global_hours = df_global["length_seconds"].sum() / 3600.0 if "length_seconds" in df_global.columns else 0.0

        global_stats[common] = {
            "global_files": global_files,
            "global_hours": global_hours
        }

        # Save regional-filtered CSV
        _save_records_csv(
            records, common, output_dir,
            lon_min=lon_min, lon_max=lon_max,
            no_region_filter=no_region_filter,
            min_duration=min_duration,
        )

    # Save global stats to JSON for ranking
    stats_path = os.path.join(output_dir, "global_stats.json")
    with open(stats_path, "w") as f:
        json.dump(global_stats, f, indent=2)
    logger.info("Saved global stats → %s", stats_path)


# ── rank command ─────────────────────────────────────────────────────────────


def rank_from_csvs(output_dir: str):
    """Read per-species CSVs and rank by total downloadable hours in target region.

    CSVs now contain only regional-filtered records (lon 60-140 OR ASEAN, >=3s).
    Global stats are read from global_stats.json for comparison.

    Sorted descending by total regional hours.
    """
    import json

    # Load global stats
    stats_path = os.path.join(output_dir, "global_stats.json")
    global_stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            global_stats = json.load(f)
    else:
        logger.warning("global_stats.json not found — global columns will be empty")

    rows = []
    zero_q = {q: 0 for q in VALID_QUALITIES}

    for common, scientific, code in SPECIES:
        safe = common.replace(" ", "_").replace("/", "-").replace(":", "-")
        csv_path = os.path.join(output_dir, f"{safe}.csv")

        # Get global stats from JSON
        species_global = global_stats.get(common, {})
        global_files = species_global.get("global_files", 0)
        global_hours = species_global.get("global_hours", 0.0)

        if not os.path.isfile(csv_path):
            rows.append({
                "common_name": common, "scientific_name": scientific,
                "ebird_code": code, "regional_hours": 0.0,
                "regional_files": 0, "global_files": global_files,
                "global_hours": global_hours,
                "q_counts": dict(zero_q),
            })
            continue

        try:
            regional = pd.read_csv(csv_path, low_memory=False)
        except pd.errors.EmptyDataError:
            rows.append({
                "common_name": common, "scientific_name": scientific,
                "ebird_code": code, "regional_hours": 0.0,
                "regional_files": 0, "global_files": global_files,
                "global_hours": global_hours,
                "q_counts": dict(zero_q),
            })
            continue

        # CSV now contains ONLY regional records (already filtered)
        regional_files = len(regional)
        regional_hours = (
            regional["length_seconds"].sum() / 3600.0
            if "length_seconds" in regional.columns and not regional.empty else 0.0
        )

        # Quality breakdown (from regional CSV)
        q_counts = {}
        if "q" in regional.columns:
            vc = regional["q"].value_counts()
            for q in VALID_QUALITIES:
                q_counts[q] = int(vc.get(q, 0))
        else:
            q_counts = dict(zero_q)

        rows.append({
            "common_name": common, "scientific_name": scientific,
            "ebird_code": code, "regional_hours": regional_hours,
            "regional_files": regional_files, "global_files": global_files,
            "global_hours": global_hours,
            "q_counts": q_counts,
        })

    if not rows:
        print("No metadata CSVs found in", output_dir)
        return

    # Sort descending by total regional hours
    rows.sort(key=lambda r: r["regional_hours"], reverse=True)

    # ── terminal table ───────────────────────────────────────────────────
    hdr = (
        f"{'Rank':>4}  {'Species':<35} {'Code':<10} "
        f"{'Hours':>7} {'Files':>6} {'Global':>7} "
        f"{'A':>5} {'B':>5} {'C':>5} {'D':>5} {'E':>5}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    csv_rows = []
    for rank, r in enumerate(rows, 1):
        hrs = r["regional_hours"]
        nf = r["regional_files"]
        glb = r["global_files"]
        qc = r["q_counts"]
        print(
            f"{rank:>4}  {r['common_name']:<35} {r['ebird_code']:<10} "
            f"{hrs:>7.1f} {nf:>6} {glb:>7} "
            f"{qc['A']:>5} {qc['B']:>5} {qc['C']:>5} {qc['D']:>5} {qc['E']:>5}"
        )
        csv_rows.append({
            "rank": rank,
            "common_name": r["common_name"],
            "scientific_name": r["scientific_name"],
            "ebird_code": r["ebird_code"],
            "regional_hours": f"{hrs:.2f}",
            "regional_files": nf,
            "global_hours": f"{r['global_hours']:.2f}",
            "global_files": glb,
            "q_A": qc["A"], "q_B": qc["B"], "q_C": qc["C"],
            "q_D": qc["D"], "q_E": qc["E"],
        })
    print(sep)

    # ── save ranking CSV ─────────────────────────────────────────────────
    # Save to project_csv/ instead of output_dir
    project_csv_dir = os.path.join(output_dir, "..", "project_csv")
    os.makedirs(project_csv_dir, exist_ok=True)
    ranking_path = os.path.join(project_csv_dir, "regional_ranking.csv")
    fieldnames = [
        "rank", "common_name", "scientific_name", "ebird_code",
        "regional_hours", "regional_files", "global_hours", "global_files",
        "q_A", "q_B", "q_C", "q_D", "q_E",
    ]
    with open(ranking_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nRanking saved to: {ranking_path}")

    # ── save ranking Markdown ────────────────────────────────────────────
    md_path = os.path.join(project_csv_dir, "regional_ranking.md")
    md_lines = [
        "# Regional Ranking — Longitude 60-140° OR ASEAN Countries",
        "",
        "Ranked by total downloadable hours (files >= 3s, non-null download URL).",
        "Regional filter: 60 < longitude < 140 OR ASEAN countries (for missing longitude).",
        "",
        "| Rank | Species | Scientific name | Code | Reg Hours | Reg Files | Global Hours | Global Files | A | B | C | D | E |",
        "|-----:|---------|-----------------|------|----------:|----------:|-------------:|-------------:|--:|--:|--:|--:|--:|",
    ]
    for r in csv_rows:
        md_lines.append(
            f"| {r['rank']} | {r['common_name']} | {r['scientific_name']} "
            f"| {r['ebird_code']} | {r['regional_hours']} | {r['regional_files']} "
            f"| {r['global_hours']} | {r['global_files']} | {r['q_A']} | {r['q_B']} | {r['q_C']} "
            f"| {r['q_D']} | {r['q_E']} |"
        )
    md_lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Ranking saved to: {md_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Fetch Xeno-Canto metadata for target species and rank by regional availability.",
    )
    parser.add_argument(
        "--species", nargs="+", default=["all"],
        help='Species to fetch (common name, scientific name, or eBird code). Default: all.',
    )
    parser.add_argument(
        "--output-dir", default=str(PER_SPECIES_CSV),
        help=f"Directory for per-species CSV files. Default: {PER_SPECIES_CSV}",
    )
    parser.add_argument(
        "--rank-only", action="store_true",
        help="Skip fetching; rank from existing CSVs in output-dir.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="XC API key. Default: XENO_API_KEY env var, then xc_key.txt.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without making API calls.",
    )
    parser.add_argument(
        "--lon-min", type=float, default=LONGITUDE_MIN,
        help=f"Minimum longitude for regional filter (default: {LONGITUDE_MIN}).",
    )
    parser.add_argument(
        "--lon-max", type=float, default=LONGITUDE_MAX,
        help=f"Maximum longitude for regional filter (default: {LONGITUDE_MAX}).",
    )
    parser.add_argument(
        "--no-region-filter", action="store_true",
        help="Disable geographic filtering entirely (keep all recordings worldwide).",
    )
    parser.add_argument(
        "--min-duration", type=float, default=3.0,
        help="Minimum recording duration in seconds (default: 3.0).",
    )
    args = parser.parse_args()

    # resolve species list
    if len(args.species) == 1 and args.species[0].lower() == "all":
        species_list = list(SPECIES)
    else:
        species_list = []
        tokens = args.species
        i = 0
        while i < len(tokens):
            matched = False
            for length in range(min(4, len(tokens) - i), 0, -1):
                candidate = " ".join(tokens[i:i + length])
                result = resolve_species(candidate)
                if result:
                    species_list.append(result)
                    i += length
                    matched = True
                    break
            if not matched:
                print(f"Error: Unknown species '{tokens[i]}'")
                sys.exit(1)

    output_dir = args.output_dir

    # Print startup information
    print("=" * 80)
    print("STAGE 1: FETCH XENO-CANTO METADATA")
    print("=" * 80)
    print("WHAT THIS DOES:")
    print("  - Fetches recording metadata from Xeno-Canto API for target species")
    if args.no_region_filter:
        print("  - Geographic filter: DISABLED (all worldwide recordings kept)")
    else:
        print(f"  - Filters recordings by region (longitude {args.lon_min}–{args.lon_max}° OR ASEAN countries)")
    print(f"  - Filters by duration (≥{args.min_duration}s) and downloadability")
    print("  - Saves ONLY regional-filtered records to per-species CSV files")
    print("  - Saves global stats (before filtering) to global_stats.json")
    print("  - Generates regional ranking with global comparison")
    print()
    print("NOTE:")
    print("  - Stage 1 fetches metadata for ALL species (ignores 'active' column)")
    print("  - Stage 2 uses 'active' column to decide which species to download")
    from pathlib import Path
    project_csv_dir = Path(output_dir).parent / "project_csv"
    print(f"  - To shortlist species: Edit {project_csv_dir}/target_species.csv")
    print(f"      Set active=yes for species you want to download in Stage 2")
    print()
    print("INPUT:")
    print(f"  - Species list: {len(species_list)} species")
    if len(species_list) <= 5:
        for common, sci, code in species_list:
            print(f"      • {common} ({sci})")
    else:
        print(f"      • {species_list[0][0]} ({species_list[0][1]})")
        print(f"      • {species_list[1][0]} ({species_list[1][1]})")
        print(f"      ... and {len(species_list) - 2} more")
    print()
    print("OUTPUT:")
    print(f"  - Per-species CSVs (regional-filtered only): {output_dir}/")
    print(f"      Example: {output_dir}/Javan_Myna.csv (uses English names)")
    print(f"  - Global stats (before filtering): {output_dir}/global_stats.json")
    print(f"  - Regional ranking: {project_csv_dir}/regional_ranking.csv")
    print(f"      (includes both regional and global hours/files for transparency)")
    print(f"  - Regional ranking: {project_csv_dir}/regional_ranking.md")
    print("=" * 80)
    print()

    if not args.rank_only:
        api_key = _load_api_key(args.api_key)
        fetch_metadata(
            species_list, output_dir, api_key, args.dry_run,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            no_region_filter=args.no_region_filter,
            min_duration=args.min_duration,
        )

    if not args.dry_run:
        rank_from_csvs(output_dir)


if __name__ == "__main__":
    main()
