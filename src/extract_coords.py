# In this code, we give a folder with .gef files and we generate a csv with coordinates from it.
# The output csv has columns: name, x, y.
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
from pathlib import Path
import logging

# sys.path.append("C:\schemaGAN")
# import schemaGAN

from utils import read_files
from geolib_plus.gef_cpt import GefCpt

# to avoid warnings coming from GeoLib
initial_logging_level = logging.getLogger().getEffectiveLevel()
logging.disable(logging.ERROR)
# remove warning messages
logging.getLogger().setLevel(logging.ERROR)


def check_file_for_coords(
    file: Path, cpt_folder: Path, sucess_count: int, fail_count: int
):
    """
    Check if a GEF file has valid coordinates. If not, move the file to a subfolder "no_coords".
    Args:
        file (Path): Path to the GEF file.
        cpt_folder (Path): Path to the folder containing GEF files.

    Returns:
        None
    """

    try:
        cpt = GefCpt()
        cpt.read(str(file))
        x, y = cpt.coordinates  # tuple (x, y)

        if x is None or y is None or np.isnan(x) or np.isnan(y):
            no_coords_folder = cpt_folder / "no_coords"
            no_coords_folder.mkdir(exist_ok=True)
            file.rename(no_coords_folder / file.name)
            fail_count += 1
        else:
            sucess_count += 1

    except Exception as e:
        no_coords_folder = cpt_folder / "no_coords"
        no_coords_folder.mkdir(exist_ok=True)
        file.rename(no_coords_folder / file.name)
        fail_count += 1


# def extract_coords(gef_folder: Path, output_csv: Path) -> None:
#     """
#     Extract coordinates from .GEF files in a specified folder and save them to a CSV file.
#     """
#     # Read .GEF files from the specified folder
#     gef_files = read_files(str(gef_folder), extension=".gef")

#     # Extract coordinates
#     coords = []
#     for file in gef_files:
#         cpt = GefCpt()
#         cpt.read(str(file))
#         coords.append(
#             {"name": file.stem, "x": cpt.coordinates[0], "y": cpt.coordinates[1]}
#         )

#     # Save to CSV
#     output_csv.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["name", "x", "y"])
#         for r in coords:
#             w.writerow([r["name"], r["x"], r["y"]])
#     print(f"Coordinates saved to {output_csv}")


def process_cpt_coords(cpt_folder: Path, output_csv: Path) -> None:
    """
    Process .GEF files in a specified folder to extract and validate coordinates,
    then save the results to a CSV file.
    """
    # 1. Read the .GEF files from the specified folder
    all_files = read_files(str(cpt_folder), extension=".gef")
    print(f"Processing {len(all_files)} files for coordinates...")

    sucess_count = 0
    fail_count = 0
    # 2. Loop through each file
    for file in all_files:
        # 3. Check file by file for coordinates and move those without to "no_coords"
        check_file_for_coords(file, cpt_folder, sucess_count, fail_count)

    print("Coordinate check completed.")
    print(f"Total files with valid coordinates: {sucess_count}")
    print(f"Total files moved to 'no_coords' folder: {fail_count}")


if __name__ == "__main__":

    ### PATHS AND SETTINGS ########################################################
    GEF_FOLDER = Path(r"C:\VOW\data\test_cpts")
    OUTPUT_CSV = Path(r"C:\VOW\data\test_outputs")
    CSV_NAME = "coordinates_cpts_test_result.csv"
    OUTPUT_CSV = OUTPUT_CSV / CSV_NAME
    ###########################################################################

    # Run the extraction
    process_cpt_coords(GEF_FOLDER, OUTPUT_CSV)
