import sys
import os
import math
from pathlib import Path


# Add your local GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import (
    UnitWeightMethod,
    ShearWaveVelocityMethod,
    OCRMethod,
    InterpretationMethod,
)


def read_files(path: str, extension: str = ".gef") -> list:
    """
    Read all files in a directory and filter by extension (case-insensitive).

    Args:
        path (str): Path to the directory.
        extension (str): File extension to filter by (default is ".gef").

    Returns:
        list: List of file paths.
    """
    extension = extension.lower()
    cpts = os.listdir(path)
    cpt_files = [
        Path(path, c)
        for c in cpts
        if Path(path, c).is_file() and c.lower().endswith(extension)
    ]
    return cpt_files


def process_cpts(cpts, save_plot=False, plot_dir=None):
    """
    Process all CPT files, save the data to CSV, and optionally plot and save the figures.
    """
    data = {"coordinates": [], "name": []}

    for cpt_path in cpts:
        print(f"Processing CPT: {cpt_path}")

        cpt_gef = GefCpt()
        cpt_gef.read(cpt_path)
        cpt_gef.pre_process_data()

        # Save filename (without extension) as the CPT name
        cpt_name = Path(cpt_path).stem
        data["name"].append(cpt_name)

        # Save coordinates (tuple (x, y))
        data["coordinates"].append(cpt_gef.coordinates)

        # # Interpretation (not strictly needed if you only want coordinates)
        # interpreter = RobertsonCptInterpretation()
        # interpreter.interpretation_method = InterpretationMethod.LENGKEEK_2022
        # interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK_2022
        # interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.ZANG
        # interpreter.ocrmethod = OCRMethod.MAYNE
        # interpreter.user_defined_water_level = True
        # cpt_gef.pwp = 0
        # cpt_gef.interpret_cpt(interpreter)

        if save_plot:
            if not plot_dir:
                raise ValueError("Plot directory must be provided if save_plot=True.")

            os.makedirs(plot_dir, exist_ok=True)
            cpt_gef.plot(Path(plot_dir))

    return data


def save_coords_csv(data, output_file, show_plot=False):
    """
    Save processed CPT data to a CSV file and optionally plot the coordinates.

    Args:
        data (dict): Processed CPT data.
        output_file (str): Path to the output CSV file.
        show_plot (bool): Whether to show the plot of coordinates (default: False).
    """
    coordinates = data["coordinates"]
    names = data["name"]

    # Save data to CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "x", "y"])  # Header
        for i, coord in enumerate(coordinates):
            writer.writerow([names[i], coord[0], coord[1]])

    print(f"Data saved to {output_file}")

    # Plot coordinates if requested
    if show_plot:
        coordinates_array = np.array(coordinates)

        plt.figure(figsize=(10, 8))
        plt.plot(
            coordinates_array[:, 0],
            coordinates_array[:, 1],
            marker="v",
            linestyle="",
            markersize=8,
            label="CPT Locations",
            color="darkblue",
        )

        for i, name in enumerate(names):
            plt.text(
                coordinates_array[i, 0],
                coordinates_array[i, 1],
                name,
                fontsize=8,
                ha="right",
                va="bottom",
            )

        plt.title("CPT Locations with Names")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.clf()


def euclid(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x2 - x1, y2 - y1))


def IC_normalization(data):
    """
    Normalize IC values in the data from [0 - MaxIC] to [-1 - 1].

    Parameters:
    data (list): List containing the source and target data.

    Returns:
    list: A list containing the normalized source and target data.
    """
    # print('Normalizing the IC data...')

    # Define the maximum and minimum values of IC in the source and target images
    max_IC_value = 4.3  # Maximum expected IC value
    min_IC_value = 0  # Minimum expected IC value, it's not really zero,
    # but when deleting data it will be zero

    # Unpack the data, where data[0] is source data and data[1] is target data
    src_data, tar_data = data

    # Calculate the range of the data
    data_range = max_IC_value - min_IC_value

    # Scale the source and target data to the range [-1, 1]
    # Formula used for normalization is:
    # normalized_data = 2 * (original_data / data_range) - 1
    src_normalized = 2 * (src_data / data_range) - 1
    tar_normalized = 2 * (tar_data / data_range) - 1

    return [src_normalized, tar_normalized]


def reverse_IC_normalization(data):
    """
    Reverse the normalization of IC values in the data from [-1 - 1] to [0 - MaxIC].

    Parameters:
    data (np.array): Array containing the normalized data.

    Returns:
    np.array: An array containing the rescaled data.
    """

    # Define the maximum and minimum values of IC in the source and target images
    max_IC_value = 4.3  # Maximum expected IC value
    min_IC_value = 0  # Minimum expected IC value, it's not really zero,
    # but when deleting data it will be zero

    # Calculate the range of the data
    data_range = max_IC_value - min_IC_value

    # Rescale the data to the original range [min_IC_value, max_IC_value]
    # Formula used for rescaling is:
    # rescaled_data = (normalized_data + 1) * (data_range / 2) + min_IC_value
    X = (data + 1) * (data_range / 2) + min_IC_value

    return X


def setup_experiment(
    base_dir: Path, region: str, exp_name: str, description: str = ""
) -> dict:
    """
    Create a standard folder structure for a SchemaGAN experiment.

    Example structure:
    res/
      north/
        exp_1/
          1_coords/
          2_compressed_cpt/
          3_sections/
          4_gan_images/
          5_mosaic/
          README.txt

    Args:
        base_dir (Path): Base directory for all results (e.g., Path('C:/VOW/res'))
        region (str): Region name ('north', 'south', etc.)
        exp_name (str): Experiment name ('exp_1', 'exp_2', etc.)
        description (str): Optional short text describing the experiment
    Returns:
        dict: Dictionary with all created folder paths
    """

    # Root for this experiment
    exp_root = base_dir / region / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    # Define subfolders
    subfolders = [
        "1_coords",
        "2_compressed_cpt",
        "3_sections",
        "4_gan_images",
        "5_mosaic",
    ]

    # Create subfolders
    paths = {}
    for name in subfolders:
        path = exp_root / name
        path.mkdir(exist_ok=True)
        paths[name] = path

    # Create or overwrite README.txt with experiment description
    readme_path = exp_root / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Region: {region}\n\n")
        if description:
            f.write("Description:\n")
            f.write(description.strip() + "\n")

    # Use print here since logger might not be configured yet
    # The main script will log the completion
    return {"root": exp_root, **paths}
