import os
import csv
import math
import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional

from geolib_plus.gef_cpt import GefCpt
from utils import read_files

# ---------------------------
# Helpers
# ---------------------------


def normalize_coord(val: Optional[float]) -> Tuple[Optional[float], bool]:
    """
    If val < 1000 (e.g., 129.656), assume a misplaced decimal and multiply by 1000.
    Otherwise leave as-is. Returns (normalized_value, was_fixed).
    """
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return np.nan, False

    was_fixed = False
    if val < 1000:  # pattern like 129.656 -> 129656
        val = val * 1000.0
        was_fixed = True

    return val, was_fixed


def rd_range_warning(x: float, y: float) -> Optional[str]:
    """
    Optional sanity check for Dutch RD New coordinates.
    Typical rough ranges:
      X: ~0 to 300,000
      Y: ~300,000 to 630,000
    We won't reject out-of-range, just warn.
    """
    warnings = []
    if not (0 <= x <= 400_000):
        warnings.append(f"X out of typical RD range: {x:.2f}")
    if not (200_000 <= y <= 700_000):
        warnings.append(f"Y out of typical RD range: {y:.2f}")
    return "; ".join(warnings) if warnings else None


# ---------------------------
# Main processing
# ---------------------------


def collect_fixed_coords(
    gef_files: List[Path], check_rd_ranges: bool = True
) -> List[dict]:
    rows = []
    for p in gef_files:
        try:
            cpt = GefCpt()
            cpt.read(str(p))
            cpt.pre_process_data()
            x, y = cpt.coordinates  # tuple (x, y)

            x_norm, x_fixed = normalize_coord(x)
            y_norm, y_fixed = normalize_coord(y)

            was_fixed = x_fixed or y_fixed
            warn = rd_range_warning(x_norm, y_norm) if check_rd_ranges else None

            rows.append(
                {
                    "name": p.stem,
                    "x": x_norm,
                    "y": y_norm,
                    "was_fixed": was_fixed,
                    "warning": warn or "",
                }
            )

        except Exception as e:
            rows.append(
                {
                    "name": p.stem,
                    "x": np.nan,
                    "y": np.nan,
                    "was_fixed": False,
                    "warning": f"Failed to read: {e}",
                }
            )
    return rows


def save_coords_csv(rows: List[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "x", "y", "was_fixed", "warning"])
        for r in rows:
            w.writerow([r["name"], r["x"], r["y"], r["was_fixed"], r["warning"]])


def collect_simple_coords(
    gef_files: List[Path], output_csv: Path = Path("simple_coords.csv")
):
    rows = []
    for p in gef_files:
        cpt = GefCpt()
        cpt.read(str(p))
        cpt.pre_process_data()
        x, y = cpt.coordinates  # tuple (x, y)

        rows.append({"name": p.stem, "x": x, "y": y})

    # Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved coordinates to {output_csv}")


# ---------------------------
# Usage
# ---------------------------

if __name__ == "__main__":
    sondering_path = r"C:\VOW\data\Site_A\O\cpt_bro"
    out_csv = Path("CPT_siteA_O_bro_coord.csv")

    gef_files = read_files(sondering_path, ".gef")
    print(f"Found {len(gef_files)} GEF files.")

    collect_simple_coords(gef_files)

    rows = collect_fixed_coords(gef_files, check_rd_ranges=True)
    save_coords_csv(rows, out_csv)

    # Quick summary
    total = len(rows)
    fixed = sum(1 for r in rows if r["was_fixed"])
    failed = sum(1 for r in rows if "Failed to read" in r["warning"])
    print(f"Done. Wrote {out_csv}")
    print(f"Summary: total={total}, fixed={fixed}, failed={failed}")
    flagged = [r for r in rows if r["warning"] and "Failed to read" not in r["warning"]]
    if flagged:
        print("Range warnings on these rows (first 10 shown):")
        for r in flagged[:10]:
            print(f"  {r['name']}: {r['warning']}")
