# In this code, we give a folder with .gef files and we generate a csv with coordinates from it.
# The output csv has columns: name, x, y, fixed.
# It does several steps in between.
# 1st: checks that all .gef files have coordinates. If not, move the file to a subfolder called "no_coords".
# 2nd: checks that the coordinates are in the right format (not 123.456 instead of 123456.000) and in the Netherlands RD coordinate system.
# 3rd: saves the coordinates in a csv file tagging those that were fixed.

import sys
from pathlib import Path

# Add your local GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

import csv
import numpy as np
import logging
from geolib_plus.gef_cpt import GefCpt
from utils import read_files

# --------------------------------------------------------------------------
# Disable noisy GeoLib logging
# --------------------------------------------------------------------------
initial_logging_level = logging.getLogger().getEffectiveLevel()
logging.disable(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


# --------------------------------------------------------------------------
# Check if file has coordinates
# --------------------------------------------------------------------------
def check_file_for_coords(
    cpt_object: GefCpt, file: Path, cpt_folder: Path, sucess_count: int, fail_count: int
):
    """
    Check if a GEF file has valid coordinates. If not, move the file to a subfolder "no_coords".
    Additional rule: if x or y equals 0.0, treat it as missing.
    """
    try:
        x, y = cpt_object.coordinates  # tuple (x, y)

        invalid = (
            x is None
            or y is None
            or np.isnan(x)
            or np.isnan(y)
            or (x == 0.0 and y == 0.0)  # both are zero
            or (x == 0.0 or y == 0.0)  # either one is zero (optional, up to you)
        )

        if invalid:
            no_coords_folder = cpt_folder / "no_coords"
            no_coords_folder.mkdir(exist_ok=True)
            file.rename(no_coords_folder / file.name)
            fail_count += 1
        else:
            sucess_count += 1

    except Exception:
        no_coords_folder = cpt_folder / "no_coords"
        no_coords_folder.mkdir(exist_ok=True)
        file.rename(no_coords_folder / file.name)
        fail_count += 1

    return sucess_count, fail_count


# --------------------------------------------------------------------------
# Fix incorrect coordinate scaling
# --------------------------------------------------------------------------
def fix_broken_coords(cpt_object: GefCpt):
    """
    Check and fix coordinates that are clearly scaled or formatted incorrectly.
    Coordinates in the Netherlands (RD system) should have a 6-digit integer part,
    e.g., 123456.000 (not 123.456 or 123456789).

    Rules:
      - If already 6 digits -> keep as is.
      - If <1000 -> multiply by 1000 (common scaling issue).
      - Otherwise -> keep unchanged.

    Returns:
        tuple: (x_fixed, y_fixed, was_fixed)
    """
    x, y = cpt_object.coordinates

    def int_part_is_six_digits(value):
        """Return True if the integer part has exactly 6 digits."""
        try:
            ival = int(abs(float(value)))
            return 100000 <= ival <= 999999
        except Exception:
            return False

    def fix_value(value):
        """Try to fix a single coordinate according to the rules above."""
        try:
            v = float(value)
        except Exception:
            return value, False

        # Already correct
        if int_part_is_six_digits(v):
            return v, False

        # Looks too small → probably missing scale factor
        if v < 1000:
            v_fixed = v * 1000.0
            if int_part_is_six_digits(v_fixed):
                return v_fixed, True

        # Otherwise leave unchanged
        return v, False

    x_fixed, x_fixed_flag = fix_value(x)
    y_fixed, y_fixed_flag = fix_value(y)
    was_fixed = x_fixed_flag or y_fixed_flag

    return x_fixed, y_fixed, was_fixed


# --------------------------------------------------------------------------
# Save CSV
# --------------------------------------------------------------------------
def save_coordinates_to_csv(rows, output_csv: Path):
    """
    Save extracted and (optionally) fixed coordinates to a CSV file.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "fixed"])
        writer.writerows(rows)

    print(f"✅ Coordinates saved to: {output_csv}")


# --------------------------------------------------------------------------
# Main process
# --------------------------------------------------------------------------
def process_cpt_coords(cpt_folder: Path, output_csv: Path) -> None:
    """
    Process .GEF files in a specified folder to extract and validate coordinates,
    then save the results to a CSV file.
    """
    all_files = read_files(str(cpt_folder), extension=".gef")
    print(f"Processing {len(all_files)} files for coordinates...")

    sucess_count = 0
    fail_count = 0
    rows = []

    no_coords_folder = cpt_folder / "no_coords"
    no_coords_folder.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------
    # Loop through each file
    # ----------------------------------------------------------------------
    for file in all_files:
        # Try to read the file (some might lack #XYID=)
        try:
            cpt = GefCpt()
            cpt.read(str(file))
        except Exception:
            # Move unreadable file to "no_coords"
            try:
                file.rename(no_coords_folder / file.name)
            except Exception:
                pass
            fail_count += 1
            continue

        # Step 1: Check if it has coordinates (may move file)
        sucess_count, fail_count = check_file_for_coords(
            cpt_object=cpt,
            file=file,
            cpt_folder=cpt_folder,
            sucess_count=sucess_count,
            fail_count=fail_count,
        )

        # If file was moved, skip fixing/CSV
        if not file.exists():
            continue

        # Step 2: Fix broken coordinates
        x_fixed, y_fixed, was_fixed = fix_broken_coords(cpt)

        # Step 3: Add to CSV rows
        rows.append([file.stem, x_fixed, y_fixed, "TRUE" if was_fixed else "FALSE"])

    # Step 4: Save all results to CSV
    save_coordinates_to_csv(rows, output_csv)

    # Step 5: Print summary
    print("Coordinate check completed.")
    print(f"Files with valid coordinates: {sucess_count}")
    print(f"Files moved to 'no_coords':   {fail_count}")
    print(f"Files written to CSV:         {len(rows)}")


# --------------------------------------------------------------------------
# Run script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ### PATHS AND SETTINGS ##################################################
    GEF_FOLDER = Path(r"C:\VOW\data\test_cpts")
    OUTPUT_CSV = Path(r"C:\VOW\data\test_outputs\coordinates_cpts_test_result.csv")
    #########################################################################
    process_cpt_coords(GEF_FOLDER, OUTPUT_CSV)
