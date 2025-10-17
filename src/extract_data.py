# This code makes the 32 px CPT csv file. All CPTs together. This does NOT make the sections filled with zeros.

import sys
import os
import math
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import read_files
from pathlib import Path

# Add your local GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import (
    UnitWeightMethod,
    InterpretationMethod,
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Disable noisy GeoLib logging only for GeoLib, not for everything
geolib_logger = logging.getLogger("geolib_plus")
geolib_logger.setLevel(logging.ERROR)


def process_cpts(gef_list: list[Path]):
    """
    Process CPT files using geolib_plus library.

    Params:
        gef_list (list[Path]): List of paths to GEF files.

    Returns:
        data (list): List of dicts with CPT data.
        coords (list): List of dicts with CPT coordinates.
    """
    data = []
    coords = []
    for cpt in gef_list:
        cpt_gef = GefCpt()
        try:
            cpt_gef.read(cpt)
        except Exception as e:
            logger.warning(f"Error reading CPT file {cpt}: {e}")
            continue

        cpt_gef.pre_process_data()

        interpreter = RobertsonCptInterpretation()
        # interpreter.interpretation_method = InterpretationMethod.ROBERTSON
        interpreter.unitweightmethod = UnitWeightMethod.ROBERTSON
        interpreter.user_defined_water_level = True
        cpt_gef.pwp = 0

        cpt_gef.interpret_cpt(interpreter)

        # Extract relevant part of the file name
        file_name = cpt.stem  # Get the file name without extension
        cpt_id = file_name.split("_")[0]  # Extract up to the first underscore

        data.append(
            {
                "Name": cpt_id,
                "depth": cpt_gef.depth_to_reference,
                "depth_max": max(cpt_gef.depth_to_reference, default=0),
                "depth_min": min(cpt_gef.depth_to_reference, default=0),
                "IC": cpt_gef.IC,
                "coordinates": cpt_gef.coordinates,
            }
        )

        coords.append(
            {"name": cpt_id, "x": cpt_gef.coordinates[0], "y": cpt_gef.coordinates[1]}
        )
        # Log which file is processed
        logger.info(f"Processed CPT file: {cpt.name}")

    return data, coords


def save_coords_to_csv(coords: list, output_dir: str):
    """
    Save CPT coordinates to a CSV file with columns: Name, x, y

    Params:
        coords (list): List of dicts with CPT coordinates
        output_dir (str): Directory where the CSV file will be saved

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(coords)
    output_file = os.path.join(output_dir, "simple_coords.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"CPT coordinates saved to: {output_file}")


def equalize_top(data_cpts: list[dict]) -> list[dict]:
    """
    Equalize the starting depth of all CPTs by removing IC and depth data above the lowest maximum depth.

    Params:
        data_cpts (list): List of dictionaries containing CPT data.

    Returns:
        list: List of dictionaries with equalized CPT data.
    """
    # Make a copy of the original data to keep it unchanged
    equalized_cpts = []

    # Find the lowest maximum depth across all CPTs
    lowest_max_depth = min(cpt["depth_max"] for cpt in data_cpts)
    # Log a message for the depth used for equalization
    logger.info(f"Equalizing to the lowest maximum depth of {lowest_max_depth} m")

    # Equalize the depth and IC data for each CPT
    for cpt in data_cpts:
        # Create a new dictionary to store the equalized data
        equalized_cpt = cpt.copy()

        depth = equalized_cpt["depth"]
        IC = equalized_cpt["IC"]

        # Filter depth and IC values that are below the lowest maximum depth (strictly below)
        filtered_data = [
            (d, ic) for d, ic in zip(depth, IC) if d < lowest_max_depth
        ]  # Adjusted to < instead of <=

        # Separate filtered depth and IC values
        equalized_cpt["depth"], equalized_cpt["IC"] = (
            zip(*filtered_data) if filtered_data else ([], [])
        )

        # Update depth_min and depth_max for the equalized data
        equalized_cpt["depth_min"] = min(equalized_cpt["depth"], default=0)
        equalized_cpt["depth_max"] = max(equalized_cpt["depth"], default=0)

        # Append the equalized data to the new list
        equalized_cpts.append(equalized_cpt)

    return equalized_cpts


def equalize_depth(data_cpts, lowest_min_depth):
    """
    Equalize the depth of all CPTs by extending their depths to the lowest minimum depth.
    New depth values are added starting from the current min depth.

    Params:
        data_cpts (list): List of dictionaries containing CPT data.
        lowest_min_depth (float): The lowest minimum depth to equalize to.

    Returns:
        list: List of dictionaries with equalized CPT data.
    """
    equalized_depth_cpts = []

    for cpt in data_cpts:
        # Copy the original CPT data
        equalized_cpt = cpt.copy()

        # Get the depth and IC values
        depth = list(equalized_cpt["depth"])
        IC = list(equalized_cpt["IC"])

        # Calculate the depth interval (assuming uniform intervals for each CPT)
        if len(depth) > 1:
            depth_interval = depth[1] - depth[0]
        else:
            # If only one depth value exists, assume an arbitrary interval (set to 1 for now)
            depth_interval = 1

        # Get the current minimum depth
        current_min_depth = min(depth)

        # Calculate how many new depth values need to be added
        num_steps_to_add = int((lowest_min_depth - current_min_depth) / depth_interval)

        # Add new depth values starting from the current minimum depth
        new_depths = [
            current_min_depth + (i + 1) * depth_interval
            for i in range(num_steps_to_add)
        ]
        new_ics = [0] * num_steps_to_add  # Add zeros for IC values at the new depths

        # Append the new data to the CPT
        equalized_cpt["depth"] = depth + new_depths
        equalized_cpt["IC"] = IC + new_ics

        # Update the depth_min and depth_max for the equalized data
        equalized_cpt["depth_min"] = min(equalized_cpt["depth"], default=0)
        equalized_cpt["depth_max"] = max(equalized_cpt["depth"], default=0)

        # Append the equalized CPT data to the new list
        equalized_depth_cpts.append(equalized_cpt)

    return equalized_depth_cpts


def compress_to_32px(equalized_cpts, method="mean"):
    """
    Compress CPT data to 32 pixels by dividing the depth into 32 equal groups
    and aggregating IC values for each group.

    Params:
        equalized_cpts (list): List of dictionaries containing equalized CPT data.
        method (str): Aggregation method, either "mean" or "max".

    Returns:
        list: List of dictionaries with compressed CPT data (32 depth and IC values).
    """
    if method not in ["mean", "max"]:
        raise ValueError("Invalid method. Use 'mean' or 'max'.")

    compressed_cpts = []

    for cpt in equalized_cpts:
        # Copy the CPT data to avoid modifying the original
        compressed_cpt = {key: cpt[key] for key in cpt if key not in ["depth", "IC"]}

        depth = np.array(cpt["depth"])
        IC = np.array(cpt["IC"])

        # Ensure data is sorted by depth
        sort_indices = np.argsort(depth)
        depth = depth[sort_indices]
        IC = IC[sort_indices]

        # Define 32 equal depth intervals
        depth_bins = np.linspace(0, 31, 33)  # 33 edges for 32 bins

        # Aggregate IC values within each depth interval
        IC_compressed = []
        depth_compressed = []
        for i in range(len(depth_bins) - 1):
            # Find indices of original depths that map into the current bin range (scaled to 0-31)
            depth_min = depth[0]
            depth_max = depth[-1]
            scaled_bins_min = depth_min + (depth_bins[i] / 31) * (depth_max - depth_min)
            scaled_bins_max = depth_min + (depth_bins[i + 1] / 31) * (
                depth_max - depth_min
            )

            mask = (depth >= scaled_bins_min) & (depth < scaled_bins_max)

            # Compute aggregated IC for the current bin
            if mask.any():
                if method == "mean":
                    IC_compressed.append(IC[mask].mean())
                elif method == "max":
                    IC_compressed.append(IC[mask].max())
            else:
                IC_compressed.append(0)
            depth_compressed.append(depth_bins[i])  # Assign bin index as depth

        # Store the compressed data
        compressed_cpt["depth"] = depth_compressed
        compressed_cpt["IC"] = IC_compressed
        compressed_cpts.append(compressed_cpt)

    return compressed_cpts


def save_cpt_to_csv(data_cpts: list, output_folder: str, output_name: str):
    """
    Save the compressed CPT data (32 pixels) to a CSV file.

    The CSV will have 33 rows: one for depth indices (0 to 31) and 32 depth bins,
    and one for each CPT with IC values.

    Params:
        data_cpts (list): List of dictionaries containing compressed CPT data.
        output_folder (str): Directory where the CSV file will be saved.
        output_name (str): Name of the output CSV file.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a DataFrame for storing depth and IC data
    df = pd.DataFrame()

    # The first column is the depth (from 0 to 31)
    df["Depth_Index"] = range(32)

    # Add each CPT's compressed IC values as a column
    for cpt in data_cpts:
        cpt_name = cpt["Name"]  # Use the CPT name as the column header
        # Add the compressed IC values to the DataFrame in inverse order
        df[cpt_name] = cpt["IC"][::-1]

    # Define the output file path
    output_file = os.path.join(output_folder, output_name)

    # Save the DataFrame to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Compressed CPT data saved to: {output_file}")


def plot_equalized_depth_cpts(
    data_cpts_original,
    data_cpts_modified,
    data_cpts_32px,
    num_to_plot=10,
    lowest_min_depth=0,
    lowest_max_depth=0,
):
    """
    Plot individual CPTs before and after equalization in a 3-row, 10-column layout.
    Depth is plotted on the y-axis, IC on the x-axis, with a dotted line indicating the lowest min depth and max depth.

    Params:
        data_cpts_original (list): List of dictionaries containing original CPT data.
        data_cpts_modified (list): List of dictionaries containing equalized CPT data.
        data_cpts_32px (list): List of dictionaries containing compressed CPT data.
        num_to_plot (int): Number of CPTs to plot (default is 10).
        lowest_min_depth (float): The lowest minimum depth across all CPTs.
        lowest_max_depth (float): The lowest maximum depth across all CPTs.

    Returns:
        None
    """
    # Limit to the specified number of CPTs
    data_cpts_original = data_cpts_original[:num_to_plot]
    data_cpts_modified = data_cpts_modified[:num_to_plot]
    data_cpts_32px = data_cpts_32px[:num_to_plot]

    fig, axs = plt.subplots(3, num_to_plot, figsize=(16, 8), sharex=True, sharey=False)
    fig.suptitle("CPT Data Before and After Depth Equalization", fontsize=10)

    for i in range(num_to_plot):
        # Plot individual CPT in the top row (before equalization)
        axs[0, i].plot(
            data_cpts_original[i]["IC"],
            data_cpts_original[i]["depth"],
            label="Before Equalized Top",
        )
        axs[0, i].axhline(
            lowest_max_depth, color="r", linestyle="dotted", label="Lowest Max Depth"
        )
        axs[0, i].axhline(
            lowest_min_depth, color="r", linestyle="dotted", label="Lowest Max Depth"
        )
        # axs[0, i].invert_yaxis()  # Depth increases downward
        axs[0, i].set_title(f"CPT-{i + 1}")
        axs[0, i].tick_params(axis="x", labelsize=8)
        axs[0, i].tick_params(axis="y", labelsize=8)
        # add gridlines
        axs[0, i].grid(True, linewidth=0.5, alpha=0.7)

        # Plot individual CPT in the middle row (after equalized top)
        axs[1, i].plot(
            data_cpts_modified[i]["IC"],
            data_cpts_modified[i]["depth"],
            label="Equalized Top",
        )
        axs[1, i].axhline(
            lowest_max_depth, color="r", linestyle="dotted", label="Lowest Max Depth"
        )
        axs[1, i].axhline(
            lowest_min_depth, color="r", linestyle="dotted", label="Lowest Max Depth"
        )
        # axs[1, i].invert_yaxis()  # Depth increases downward
        axs[1, i].tick_params(axis="x", labelsize=8)
        axs[1, i].tick_params(axis="y", labelsize=8)
        # add gridlines
        axs[1, i].grid(True, linewidth=0.5, alpha=0.7)

        # Plot individual CPT in the bottom row (after depth equalization to lowest_min_depth)
        axs[2, i].plot(
            data_cpts_32px[i]["IC"],
            data_cpts_32px[i]["depth"],
            label="Equalized Bottom",
        )
        axs[2, i].invert_yaxis()  # Depth increases downward
        axs[2, i].tick_params(axis="x", labelsize=8)
        axs[2, i].tick_params(axis="y", labelsize=8)
        # add gridlines
        axs[2, i].grid(True, linewidth=0.5, alpha=0.7)

    # Add labels for rows
    axs[0, 0].set_ylabel("Depth (Before)", fontsize=10)
    axs[1, 0].set_ylabel("Depth (Equalized Top)", fontsize=10)
    axs[2, 0].set_ylabel("Depth (Equalized Bottom)", fontsize=10)

    # Set common x-axis label
    fig.supxlabel("IC", fontsize=10)

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def plot_compression_results(equalized_cpts, compressed_cpts, num_to_plot=10):
    """
    Plot the results of compression for comparison in a 3-row layout.

    Params:
        equalized_cpts (list): List of dictionaries containing equalized CPT data.
        compressed_cpts (list): List of dictionaries containing compressed CPT data.
        num_to_plot (int): Number of CPTs to plot (default is 10).

    Returns:
        None
    """
    # Calculate the number of columns and rows
    rows = 3
    cols = math.ceil(num_to_plot / rows)

    # Create the figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 3), sharey=True)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    for i in range(num_to_plot):
        eq_cpt = equalized_cpts[i]
        comp_cpt = compressed_cpts[i]

        ax = axs[i]

        # Plot equalized CPT
        ax.plot(eq_cpt["IC"], eq_cpt["depth"], label="Equalized", color="blue")

        # Plot compressed CPT with a secondary y-axis
        ax_twin = ax.twinx()
        ax_twin.plot(
            comp_cpt["IC"],
            comp_cpt["depth"],
            label="Compressed",
            color="red",
            linestyle="--",
        )

        # Formatting
        ax.invert_yaxis()  # Depth increases downward
        ax.set_title(f"CPT-{str(eq_cpt['Name'])[-5:]}", fontsize=10)
        if i % cols == 0:
            ax.set_ylabel("Depth (m)")

        ax.set_xlabel("IC (Equalized)")
        ax_twin.set_ylabel("Depth (Compressed)")
        # Add gridlines
        ax.grid(True, linewidth=0.5, alpha=0.7)

    # Turn off unused subplots
    for j in range(num_to_plot, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    #### USER INPUT ####
    CPT_FOLDER = Path(r"C:\VOW\data\cpts\betuwepand\dike_north_BRO")
    OUT_FOLDER = Path(r"C:\VOW\data\schgan_inputs\testtestest")
    OUT_NAME = "test_dike_north_input_new.csv"
    ####################

    # Directory containing the CPT files
    cpts_path = read_files(path=CPT_FOLDER, extension=".gef")

    # Process CPT files
    data_cpts, coords = process_cpts(cpts_path)

    # Save coordinates to CSV
    # output_dir = r"C:\VOW\gis"
    # save_coords_to_csv(coords, output_dir)

    # Create a copy of the original data for plotting
    original_data_cpts = [cpt.copy() for cpt in data_cpts]

    # Find the lowest maximum and minimum depth across all CPTs
    lowest_max_depth = min(cpt["depth_max"] for cpt in data_cpts)
    lowest_min_depth = min(cpt["depth_min"] for cpt in data_cpts)

    # Log the results
    logger.info(f"The lowest maximum depth is: {lowest_max_depth}")
    logger.info(f"The lowest minimum depth is: {lowest_min_depth}")

    # Equalize depths to match the lowest_max_depth (equalized top)
    equalized_top_cpts = equalize_top(original_data_cpts)

    # Now extend depths to match the lowest_min_depth (equalized bottom)
    equalized_depth_cpts = equalize_depth(equalized_top_cpts, lowest_min_depth)

    # Compress data to 32 points
    compressed_cpts = compress_to_32px(equalized_depth_cpts, method="mean")

    # Plot the original, equalized, and compressed data
    plot_equalized_depth_cpts(
        original_data_cpts,
        equalized_depth_cpts,
        compressed_cpts,
        num_to_plot=10,
        lowest_min_depth=lowest_min_depth,
        lowest_max_depth=lowest_max_depth,
    )

    # Plot the results of compression
    plot_compression_results(
        equalized_depth_cpts, compressed_cpts, num_to_plot=len(compressed_cpts)
    )

    # Save the compressed data to a CSV file
    save_cpt_to_csv(compressed_cpts, OUT_FOLDER, OUT_NAME)
