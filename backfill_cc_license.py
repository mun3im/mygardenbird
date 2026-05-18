#!/usr/bin/env python3
"""
Backfill cc_license into an existing recordings.csv without re-running Stage 6.

Reads the per-species CSVs produced by Stage 1 (which contain the raw `lic`
field from the XC API), converts each URL to an SPDX identifier, and inserts
a new `cc_license` column after `quality_grade`.

Usage:
    python backfill_cc_license.py
    python backfill_cc_license.py --recordings /path/to/recordings.csv \
                                  --metadata-dir /path/to/per_species_csv
"""

import argparse
import csv
import io
from pathlib import Path

from config import RECORDINGS_CSV, PER_SPECIES_CSV


def _lic_to_spdx(url: str) -> str:
    if not url:
        return ""
    url = url.strip().rstrip("/").lower()
    marker = "/licenses/"
    idx = url.find(marker)
    if idx == -1:
        if "publicdomain" in url or "zero" in url:
            return "CC0-1.0"
        return url
    slug = url[idx + len(marker):]
    parts = [p for p in slug.split("/") if p]
    if not parts:
        return url
    code = parts[0].upper()
    version = parts[1] if len(parts) > 1 else ""
    spdx = f"CC-{code}"
    if version:
        spdx += f"-{version}"
    return spdx


def load_license_map(metadata_dir: Path) -> dict:
    """Return {xc_id: spdx_string} from all per-species CSVs.

    Also scans <metadata_dir>_old (if it exists) so that recordings that were
    later filtered out of the current Stage 1 CSVs still get a licence value.
    Current CSVs take precedence over the old backup.
    """
    dirs = []
    old_dir = metadata_dir.parent / (metadata_dir.name + "_old")
    if old_dir.is_dir():
        dirs.append(old_dir)   # old first; current overwrites
    dirs.append(metadata_dir)

    lic_map = {}
    for d in dirs:
        for csv_file in d.glob("*.csv"):
            try:
                with open(csv_file, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        xc_id = str(row.get("id", "")).strip()
                        if xc_id:
                            lic_map[xc_id] = _lic_to_spdx(row.get("lic", ""))
            except Exception as e:
                print(f"  Warning: could not read {csv_file.name}: {e}")
        print(f"  Loaded from {d}: {len(lic_map)} unique IDs so far")
    return lic_map


def backfill(recordings_path: Path, metadata_dir: Path) -> None:
    print(f"Loading licence map from {metadata_dir} ...")
    lic_map = load_license_map(metadata_dir)
    print(f"  {len(lic_map)} XC recordings found in per-species CSVs")

    with open(recordings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        rows = list(reader)

    if "cc_license" in old_fields:
        print("cc_license column already present — overwriting values.")
        new_fields = old_fields
    else:
        # Insert after quality_grade
        idx = old_fields.index("quality_grade") + 1 if "quality_grade" in old_fields else len(old_fields)
        new_fields = old_fields[:idx] + ["cc_license"] + old_fields[idx:]

    matched = 0
    for row in rows:
        xc_id = str(row.get("source_id", "")).strip()
        spdx = lic_map.get(xc_id, "")
        row["cc_license"] = spdx
        if spdx:
            matched += 1

    # Write back in-place
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=new_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    recordings_path.write_text(buf.getvalue(), encoding="utf-8")

    total = len(rows)
    print(f"  {matched}/{total} rows received a licence ({total - matched} blank).")
    print(f"Updated recordings.csv written to: {recordings_path}")

    # Show a sample
    print("\nSample (first 5 rows):")
    for row in rows[:5]:
        print(f"  {row['source_id']:>10}  {row.get('cc_license', ''):<22}  {row.get('species_common', '')}")


def main():
    parser = argparse.ArgumentParser(description="Backfill cc_license into recordings.csv")
    parser.add_argument("--recordings",    default=str(RECORDINGS_CSV),  help="Path to recordings.csv")
    parser.add_argument("--metadata-dir",  default=str(PER_SPECIES_CSV), help="Directory of per-species Stage 1 CSVs")
    args = parser.parse_args()

    recordings_path = Path(args.recordings)
    metadata_dir    = Path(args.metadata_dir)

    if not recordings_path.exists():
        raise SystemExit(f"recordings.csv not found: {recordings_path}")
    if not metadata_dir.is_dir():
        raise SystemExit(f"metadata-dir not found: {metadata_dir}")

    backfill(recordings_path, metadata_dir)


if __name__ == "__main__":
    main()
