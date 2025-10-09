# Some CPT files have coordinates that are clearly wrong (e.g. 123.456 or 123456789).
# This script checks all .GEF files in the success and failure folders, and tries to fix
# them to have a 6-digit integer part (in the Dutch RD coordinate system).
# If not possible, the coordinates are marked as not correct.

import os
import csv
import math
import numpy as np

from pathlib import Path
from typing import Optional, Tuple
from geolib_plus.gef_cpt import GefCpt
from utils import read_files

# ---------------------------
# Configure your folders
# ---------------------------
success_dir = r"C:\ark\data\sonderingen\SON\success"
failure_dir = r"C:\ark\data\sonderingen\SON\failure"
output_csv = Path("cpt_coordinates_status.csv")

# ---------------------------
# Helpers
# ---------------------------


def safe_coordinates(
    gef_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        cpt = GefCpt()
        cpt.read(str(gef_path))
        cpt.pre_process_data()
        x, y = cpt.coordinates
        return x, y, None
    except Exception as e:
        return None, None, str(e)


def int_part_is_six_digits(val: Optional[float]) -> bool:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return False
    try:
        ival = int(abs(float(val)))
        return 100000 <= ival <= 999999
    except Exception:
        return False


def try_fix_to_six_digits(val: Optional[float]) -> Tuple[Optional[float], bool]:
    """
    Try to ensure val has a 6-digit integer part.
    - If already OK, return as-is.
    - If <1000, multiply by 1000 and recheck.
    - Else leave as-is but mark not fixable.
    """
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None, False

    if int_part_is_six_digits(val):
        return float(val), True

    if float(val) < 1000:
        fixed = float(val) * 1000.0
        if int_part_is_six_digits(fixed):
            return fixed, True

    return val, False  # keep original if not fixable


def fmt_val(val: Optional[float]) -> str:
    """Format for CSV output (handles None/nan gracefully)."""
    if val is None:
        return "ND"
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return "ND"
        # avoid unnecessary .0
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(f)
    except Exception:
        return str(val)


# ---------------------------
# Main
# ---------------------------


def main():
    all_files = read_files(success_dir) + read_files(failure_dir)
    rows = []

    for f in all_files:
        x_raw, y_raw, err = safe_coordinates(f)

        if err is None:
            x_fixed, x_ok = try_fix_to_six_digits(x_raw)
            y_fixed, y_ok = try_fix_to_six_digits(y_raw)
            ok = x_ok and y_ok
            if ok:
                x_out = fmt_val(x_fixed)
                y_out = fmt_val(y_fixed)
            else:
                # keep whatever was in the file, even if “wrong”
                x_out = fmt_val(x_raw)
                y_out = fmt_val(y_raw)
        else:
            ok = False
            x_out = "ND"
            y_out = "ND"

        rows.append(
            {
                "name": f.stem,
                "x": x_out,
                "y": y_out,
                "correct_coordinate": "TRUE" if ok else "FALSE",
            }
        )

    # Write single CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "x", "y", "correct_coordinate"])
        for r in rows:
            w.writerow([r["name"], r["x"], r["y"], r["correct_coordinate"]])

    # Summary
    total = len(rows)
    correct = sum(1 for r in rows if r["correct_coordinate"] == "TRUE")
    print(
        f"Wrote {output_csv} | total={total}, correct={correct}, not-correct={total - correct}"
    )


if __name__ == "__main__":
    main()
