# This is a script that reads the coordiantes of the CPTs and organizes them from west to east (left to right).
# Then it calculates the euclidean distance in two ways: first the distance between the first cpt and all the rest
# and then the distance between each cpt and the next one.
# Once we know all the distances, we can create the input file for schemaGAN. This uses a csv file of size 512x32
# with 512 columns (representing the distance) and 32 rows (representing the depth. In each position a value is assigned
# to represent the soil type at that depth and distance. To do this we first create a 512x32 matrix filled with 0s
# and then we find the closest cpt to each column and assign the soil type of that cpt to all the rows of that column.
# Finally we save the matrix as a csv file. Key here is to keep track of the distance scale. It is not necesarry
# to have a 1:1 scale. First lets find the max distance, then we can divide that by 512 see how many sections of
# that size fit in the max distance. We want to fit around 6 CPTs per 512 columns, so we can adjust the scale accordingly.

import math
from pathlib import Path
import numpy as np
import pandas as pd


from typing import List
from utils import euclid


def validate_input_files(coords_df: pd.DataFrame, cpt_data_df: pd.DataFrame):
    """
    Validate the input DataFrames to ensure they have the expected structure.

    Params:
        coords_df (pd.DataFrame): DataFrame containing CPT coordinates.
        cpt_data_df (pd.DataFrame): DataFrame containing CPT data.

    Raises:
        AssertionError: If the DataFrames do not have the expected columns or number of rows.
    """

    assert {"name", "x", "y"}.issubset(
        coords_df.columns
    ), "coords CSV must have columns: name, x, y"
    assert (
        "Depth_Index" in cpt_data_df.columns
    ), "CPT data CSV must have a Depth_Index column"
    assert (
        len(cpt_data_df) == N_ROWS
    ), f"CPT data must have {N_ROWS} rows (Depth_Index=0..{N_ROWS-1})"


def sort_cpt_by_coordinates(
    coords_df: pd.DataFrame, from_where: str = "west", to_where: str = "east"
) -> pd.DataFrame:
    """
    Sort CPT DataFrame by coordinates depending on the specified direction.

    Params:
        coords_df (pd.DataFrame): DataFrame containing CPT coordinates with 'x' and 'y' columns.
        from_where (str): Direction to sort from ('west', 'east', 'north', 'south').
        to_where (str): Direction to sort to ('west', 'east', 'north', 'south').

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    if from_where == "west" and to_where == "east":
        sorted_df = coords_df.sort_values(by="x", ascending=True).reset_index(drop=True)
    elif from_where == "east" and to_where == "west":
        sorted_df = coords_df.sort_values(by="x", ascending=False).reset_index(
            drop=True
        )
    elif from_where == "north" and to_where == "south":
        sorted_df = coords_df.sort_values(by="y", ascending=False).reset_index(
            drop=True
        )
    elif from_where == "south" and to_where == "north":
        sorted_df = coords_df.sort_values(by="y", ascending=True).reset_index(drop=True)
    else:
        raise ValueError(
            "Invalid direction specified. Use 'west', 'east', 'north', or 'south'."
        )

    return sorted_df


def compute_distances(coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distances assuming `coords_df` is already sorted in the desired order.

    Adds:
      - dist_from_first_m : distance from the first CPT
      - dist_from_prev_m  : distance from the previous CPT
      - cum_along_m       : cumulative distance along the chain

    Params:
        coords_df (pd.DataFrame): DataFrame containing sorted CPT coordinates with 'x' and 'y' columns.

    Returns:
        pd.DataFrame: DataFrame with added distance columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = coords_df.reset_index(drop=True).copy()
    if df.empty:
        return df

    # Initialize lists to hold distance values
    dist_from_first, dist_from_prev, cum_along = [], [], []
    x0, y0 = df.loc[0, ["x", "y"]]  # first CPT reference point
    prev_x, prev_y = x0, y0
    total = 0.0

    # Loop through each CPT and calculate distances
    for _, row in df.iterrows():
        x, y = row["x"], row["y"]
        d_first = euclid(x0, y0, x, y)
        d_prev = euclid(prev_x, prev_y, x, y)
        total += d_prev
        # Append distances to respective lists
        dist_from_first.append(d_first)
        dist_from_prev.append(d_prev)
        cum_along.append(total)
        # Update previous point
        prev_x, prev_y = x, y

    # Add new distance columns to the DataFrame
    df["dist_from_first_m"] = dist_from_first
    df["dist_from_prev_m"] = dist_from_prev
    df["cum_along_m"] = cum_along
    return df


if __name__ == "__main__":
    ##### PATHS #########################################################################
    COORDS_CSV = Path(r"D:\codes\vow_schGAN\data\processed\cpt_coords_fixed.csv")
    CPT_DATA_CSV = Path(r"D:\codes\vow_schGAN\data\processed\all_cpt_32px.csv")
    OUT_DIR = Path(r"D:\codes\vow_schGAN\data\processed")
    ##### SECTIONING LOGIC ##############################################################
    N_COLS = 512  # Number of columns in the output matrix
    N_ROWS = 32  # Number of rows in the output matrix (depth levels)
    CPTS_PER_SECTION = 6  # Aim to have around 6 CPTs per 512 columns
    OVERLAP_CPTS = 2  # Number of CPTs to overlap between sections
    LEFT_PAD_FRACTION = 0.1  # Fraction of total distance to pad on the left
    RIGHT_PAD_FRACTION = 0.1  # Fraction of total distance to pad on the right
    DIR_1, DIR_2 = (
        "west",
        "east",
    )  # Direction to sort CPTs, choose from "west", "east", "north", "south"
    #####################################################################################

    # 1. If the OUT_DIR does not exist, create it
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load the data
    coords_df = pd.read_csv(COORDS_CSV)
    cpt_data_df = pd.read_csv(CPT_DATA_CSV)

    # 3. Validate the input files
    validate_input_files(coords_df=coords_df, cpt_data_df=cpt_data_df)

    # 4. Sort the coordinates DataFrame
    sorted_coords_df = sort_cpt_by_coordinates(
        coords_df=coords_df, from_where=DIR_1, to_where=DIR_2
    )

    # 5. Compute the distances
    coords_w_distances_df = compute_distances(sorted_coords_df)

    # 6.
