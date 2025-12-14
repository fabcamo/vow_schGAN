"""
Validation module for VOW SchemaGAN using leave-out cross-validation strategy.

This module validates the SchemaGAN model by:
1. Loading experiment configuration from config.py
2. Randomly removing N CPTs per run (configurable via VALIDATION_N_REMOVE)
3. Creating sections from remaining CPTs using the preprocessing pipeline
4. Generating SchemaGAN predictions for each section/window
5. Extracting predictions at left-out CPT locations
6. Comparing predictions against left-out CPTs
7. Computing MAE and MSE metrics
8. Saving results to 8_validation folder

Rules for CPT removal:
- Never remove the first or last CPT
- Never remove two consecutive CPTs
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging
import tempfile
import shutil

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add GEOLib-Plus path if needed
sys.path.append(r"D:\GEOLib-Plus")

# Import from project modules
from modules.visualization import create_custom_ic_colormap
from core.utils import IC_normalization, reverse_IC_normalization
from core.create_schGAN_input_file import (
    split_cpt_into_windows,
    process_sections,
    validate_input_files,
)

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.
    Only computes error for non-zero values in y_true (ignores zero-padding).

    Parameters:
        y_true: Array of true IC values from CPT
        y_pred: Array of predicted IC values from SchemaGAN

    Returns:
        float: The mean absolute error (only for non-zero true values)
    """
    # Mask to exclude zero-padded regions
    mask = y_true != 0
    if not np.any(mask):
        return 0.0  # If all zeros, return 0
    return np.abs(y_true[mask] - y_pred[mask]).mean()


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.
    Only computes error for non-zero values in y_true (ignores zero-padding).

    Parameters:
        y_true: Array of true IC values from CPT
        y_pred: Array of predicted IC values from SchemaGAN

    Returns:
        float: The mean squared error (only for non-zero true values)
    """
    # Mask to exclude zero-padded regions
    mask = y_true != 0
    if not np.any(mask):
        return 0.0  # If all zeros, return 0
    return ((y_true[mask] - y_pred[mask]) ** 2).mean()


def select_cpts_to_remove(
    cpt_names: List[str], n_remove: int = 12, random_state: int = None
) -> List[str]:
    """
    Select CPTs to remove following the validation rules:
    1. Never remove the first or last CPT
    2. Never remove two consecutive CPTs

    Parameters:
        cpt_names: List of all CPT names in order
        n_remove: Number of CPTs to remove (default: 12)
        random_state: Random seed for reproducibility

    Returns:
        List of CPT names to remove
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_total = len(cpt_names)
    if n_total < n_remove + 2:
        raise ValueError(
            f"Not enough CPTs to remove {n_remove}. Need at least {n_remove + 2} CPTs."
        )

    # Create list of valid indices (excluding first and last)
    valid_indices = list(range(1, n_total - 1))

    selected_indices = []

    while len(selected_indices) < n_remove:
        # Randomly select from remaining valid indices
        remaining = [idx for idx in valid_indices if idx not in selected_indices]
        if not remaining:
            raise ValueError(
                f"Cannot select {n_remove} non-consecutive CPTs. Try reducing n_remove."
            )

        idx = np.random.choice(remaining)
        selected_indices.append(idx)

        # Remove adjacent indices to prevent consecutive selection
        valid_indices = [i for i in valid_indices if i not in [idx - 1, idx, idx + 1]]

    # Sort indices and return corresponding CPT names
    selected_indices.sort()
    return [cpt_names[i] for i in selected_indices]


def run_validation(
    compressed_csv: Path,
    coords_csv: Path,
    model: tf.keras.Model,
    n_cols: int,
    n_rows: int,
    cpts_per_section: int,
    overlap_cpts: int,
    left_pad_frac: float,
    right_pad_frac: float,
    dir_from: str,
    dir_to: str,
    y_top_m: float,
    y_bottom_m: float,
    n_remove: int = 12,
    random_state: int = None,
    vertical_overlap_pct: float = 50.0,
    save_images: bool = True,
    images_folder: Path = None,
) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """
    Run a single validation iteration using proper sectioning pipeline.

    Parameters:
        compressed_csv: Path to compressed CPT data CSV
        coords_csv: Path to CPT coordinates CSV
        model: Loaded SchemaGAN model
        n_cols: Number of columns for SchemaGAN (512)
        n_rows: Number of rows for SchemaGAN (32)
        cpts_per_section: Number of CPTs per section (e.g., 6)
        overlap_cpts: Number of overlapping CPTs between sections
        left_pad_frac: Left padding fraction
        right_pad_frac: Right padding fraction
        dir_from: Sorting direction from
        dir_to: Sorting direction to
        n_remove: Number of CPTs to remove
        random_state: Random seed for reproducibility
        vertical_overlap_pct: Vertical overlap percentage for windows
        save_images: Whether to save PNG images of generated sections
        images_folder: Folder to save PNG images (if save_images=True)

    Returns:
        Tuple containing:
        - mae_results: Dict mapping CPT name to MAE (averaged across all occurrences)
        - mse_results: Dict mapping CPT name to MSE (averaged across all occurrences)
        - removed_cpts: List of removed CPT names
    """
    # Load compressed CPT data and coordinates
    cpt_df_full = pd.read_csv(compressed_csv)
    coords_df_full = pd.read_csv(coords_csv)

    # IMPORTANT: Sort coordinates spatially first to get correct CPT order
    # This ensures we don't remove consecutive CPTs along the transect
    from core.create_schGAN_input_file import sort_cpt_by_coordinates

    coords_df_full_sorted = sort_cpt_by_coordinates(
        coords_df=coords_df_full, from_where=dir_from, to_where=dir_to
    )

    # Get CPT names in spatial order (not CSV column order)
    all_cpt_names_spatial = coords_df_full_sorted["name"].tolist()

    # Select CPTs to remove based on spatial order
    removed_cpts = select_cpts_to_remove(all_cpt_names_spatial, n_remove, random_state)
    remaining_cpts = [cpt for cpt in all_cpt_names_spatial if cpt not in removed_cpts]

    print(
        f"  Removing {len(removed_cpts)} CPTs: {removed_cpts[:3]}... (showing first 3)"
    )
    print(f"  Remaining {len(remaining_cpts)} CPTs for sections")

    # Verify all CPTs exist in the dataframe
    cpt_df_columns = set(cpt_df_full.columns[1:])  # Exclude Depth_Index
    if not set(all_cpt_names_spatial).issubset(cpt_df_columns):
        missing = set(all_cpt_names_spatial) - cpt_df_columns
        raise ValueError(f"CPTs in coordinates not found in data: {missing}")

    # Create reduced CPT DataFrame (only remaining CPTs + Depth_Index)
    cpt_df_reduced = cpt_df_full[["Depth_Index"] + remaining_cpts].copy()

    # Create reduced coordinates DataFrame (only remaining CPTs)
    coords_df_reduced = (
        coords_df_full[coords_df_full["name"].isin(remaining_cpts)]
        .copy()
        .reset_index(drop=True)
    )

    # Create temporary directory for sections
    temp_dir = Path(tempfile.mkdtemp(prefix="validation_"))

    try:
        total_cpt_rows = len(cpt_df_reduced)

        # IMPORTANT: Create distance file for ALL CPTs (including removed ones)
        # This is needed to locate removed CPTs spatially in the sections
        from core.create_schGAN_input_file import (
            sort_cpt_by_coordinates,
            compute_distances,
        )

        coords_df_full_sorted = sort_cpt_by_coordinates(
            coords_df=coords_df_full, from_where=dir_from, to_where=dir_to
        )
        coords_df_full_sorted = compute_distances(coords_df_full_sorted)
        coords_df_full_sorted.to_csv(
            temp_dir / "cpt_coords_full_with_distances.csv", index=False
        )

        # Handle vertical windowing if needed
        if total_cpt_rows > n_rows:
            print(
                f"  Splitting {total_cpt_rows} rows into {n_rows}-row windows (overlap: {vertical_overlap_pct}%)"
            )
            depth_windows = split_cpt_into_windows(
                cpt_df=cpt_df_reduced,
                window_rows=n_rows,
                vertical_overlap_pct=vertical_overlap_pct,
            )
        else:
            # No windowing needed
            depth_windows = [(0, 0, total_cpt_rows, cpt_df_reduced)]

        # Store predictions at removed CPT locations for each depth window
        mae_by_cpt_by_window = {cpt: [] for cpt in removed_cpts}
        mse_by_cpt_by_window = {cpt: [] for cpt in removed_cpts}

        # Collect all manifests from all depth windows
        all_manifests = []

        # Process each depth window
        for w_idx, start_row, end_row, cpt_df_win in depth_windows:
            print(f"    Depth window {w_idx} (rows {start_row}-{end_row})...")

            # Validate this window
            validate_input_files(coords_df_reduced, cpt_df_win, n_rows)

            # Create sections for this window using the proper sectioning pipeline
            manifest = process_sections(
                coords_df=coords_df_reduced,
                cpt_df=cpt_df_win,
                out_dir=temp_dir,
                n_cols=n_cols,
                n_rows=n_rows,
                per=cpts_per_section,
                overlap=overlap_cpts,
                left_pad_frac=left_pad_frac,
                right_pad_frac=right_pad_frac,
                from_where=dir_from,
                to_where=dir_to,
                depth_window=w_idx,
                depth_start_row=start_row,
                depth_end_row=end_row,
                write_distances=(w_idx == 0),  # Only write once
            )

            print(f"      Created {len(manifest)} sections")

            # Add this window's sections to the overall manifest
            all_manifests.extend(manifest)

            # Load coordinates with distances for REMAINING CPTs (used by sections)
            if w_idx == 0:
                coords_with_dist_reduced = pd.read_csv(
                    temp_dir / "cpt_coords_with_distances.csv"
                )
                # Also load FULL coordinates (including removed CPTs) for locating them
                coords_with_dist_full = pd.read_csv(
                    temp_dir / "cpt_coords_full_with_distances.csv"
                )

            # For each section, generate SchemaGAN and extract predictions at removed CPT locations
            for section_info in manifest:
                section_csv = Path(section_info["csv_path"])

                # Load section data
                section_df = pd.read_csv(section_csv)
                if section_df.shape[1] == n_cols + 1:
                    section_data = section_df.iloc[:, 1:].values  # Skip index column
                else:
                    section_data = section_df.values

                # Reshape and normalize
                cs_input = section_data.reshape(1, n_rows, n_cols, 1).astype(float)
                cs_norm = IC_normalization([cs_input, cs_input])[0]

                # Generate prediction
                gan_result = model.predict(cs_norm, verbose=0)
                gan_result = reverse_IC_normalization(gan_result)
                gan_result = np.squeeze(gan_result)  # Shape: (n_rows, n_cols)

                # Save section CSV and PNG if requested (for mosaic creation)
                if save_images and images_folder is not None:
                    # Use the section csv_path stem to create matching GAN filename
                    # This ensures compatibility with create_mosaic's find_latest_gan_csv_for_row
                    section_csv_stem = Path(section_info["csv_path"]).stem
                    # Add a seed for consistency with main_processing (use a fixed seed for validation)
                    output_csv = images_folder / f"{section_csv_stem}_seed99999_gan.csv"
                    output_png = images_folder / f"{section_csv_stem}_seed99999_gan.png"

                    # Save CSV
                    pd.DataFrame(gan_result).to_csv(output_csv, index=False)

                # Find which removed CPTs fall within this section's spatial extent
                section_start_idx = section_info["start_idx"]
                section_end_idx = section_info["end_idx"]

                # Get the spatial extent (in meters along line) from REMAINING CPTs
                left_pad_m = section_info["left_pad_m"]
                right_pad_m = section_info["right_pad_m"]
                span_m = section_info["span_m"]
                total_span_m = left_pad_m + span_m + right_pad_m

                # Get start position in meters from REMAINING CPTs coordinates
                start_m = (
                    coords_with_dist_reduced.loc[section_start_idx, "cum_along_m"]
                    - left_pad_m
                )

                # Generate PNG visualization if requested
                if save_images and images_folder is not None:
                    # Get spatial extent for axes
                    x0 = start_m
                    x1 = start_m + total_span_m
                    dx = total_span_m / (n_cols - 1) if n_cols > 1 else total_span_m

                    # Use custom colormap
                    cmap, vmin, vmax = create_custom_ic_colormap(
                        config.IC_MIN,
                        config.IC_SAND_SANDMIX_BOUNDARY,
                        config.IC_SANDMIX_SILTMIX_BOUNDARY,
                        config.IC_SILTMIX_CLAY_BOUNDARY,
                        config.IC_CLAY_ORGANIC_BOUNDARY,
                        config.IC_MAX,
                    )

                    plt.figure(figsize=(10, 2.4))
                    plt.imshow(
                        gan_result,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                        extent=[x0, x1, n_rows - 1, 0],
                    )
                    cbar = plt.colorbar(label="IC Value", extend="both")
                    cbar.ax.tick_params(labelsize=config.PLOT_FONT_SIZE)
                    cbar.set_label("IC Value", fontsize=config.PLOT_FONT_SIZE)
                    cbar.set_ticks(
                        [
                            config.IC_MIN,
                            config.IC_SAND_SANDMIX_BOUNDARY,
                            config.IC_SANDMIX_SILTMIX_BOUNDARY,
                            config.IC_SILTMIX_CLAY_BOUNDARY,
                            config.IC_CLAY_ORGANIC_BOUNDARY,
                            config.IC_MAX,
                        ]
                    )
                    cbar.set_ticklabels(
                        [
                            f"{config.IC_MIN:g}",
                            f"{config.IC_SAND_SANDMIX_BOUNDARY:g}",
                            f"{config.IC_SANDMIX_SILTMIX_BOUNDARY:g}",
                            f"{config.IC_SILTMIX_CLAY_BOUNDARY:g}",
                            f"{config.IC_CLAY_ORGANIC_BOUNDARY:g}",
                            f"{config.IC_MAX:g}",
                        ],
                        fontsize=config.PLOT_FONT_SIZE,
                    )

                    ax = plt.gca()
                    ax.set_xlabel(
                        "Distance along line (m)", fontsize=config.PLOT_FONT_SIZE
                    )
                    ax.set_ylabel("Depth Index", fontsize=config.PLOT_FONT_SIZE)
                    ax.tick_params(axis="both", labelsize=config.PLOT_FONT_SIZE)
                    ax.set_xlim(x0, x1)

                    # Top x-axis: pixel index
                    def m_to_px(x):
                        return (x - x0) / dx if dx != 0 else 0

                    def px_to_m(p):
                        return x0 + p * dx

                    top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
                    top.set_xlabel(f"Pixel index (0…{n_cols - 1})")

                    # Right y-axis: real depth in meters
                    # Calculate depth for this window based on start_row and end_row
                    window_y_top_m = y_top_m + (start_row / (total_cpt_rows - 1)) * (
                        y_bottom_m - y_top_m
                    )
                    window_y_bottom_m = y_top_m + (
                        (end_row - 1) / (total_cpt_rows - 1)
                    ) * (y_bottom_m - y_top_m)

                    def idx_to_meters(y_idx):
                        return window_y_top_m + (y_idx / (n_rows - 1)) * (
                            window_y_bottom_m - window_y_top_m
                        )

                    def meters_to_idx(y_m):
                        denom = window_y_bottom_m - window_y_top_m
                        return (
                            0.0
                            if abs(denom) < 1e-12
                            else (y_m - window_y_top_m) * (n_rows - 1) / denom
                        )

                    right = ax.secondary_yaxis(
                        "right", functions=(idx_to_meters, meters_to_idx)
                    )
                    right.set_ylabel("Depth (m)")

                    # Add vertical lines at CPT positions
                    # Present CPTs: solid black lines
                    for idx in range(section_start_idx, section_end_idx + 1):
                        cpt_x = coords_with_dist_reduced.loc[idx, "cum_along_m"]
                        ax.axvline(
                            x=cpt_x,
                            color="black",
                            linewidth=1,
                            linestyle="-",
                            alpha=0.5,
                            zorder=10,
                        )

                    # Removed CPTs: dashed black lines (if they fall within this section)
                    # Need to access full coords with distances (created earlier)
                    coords_full_with_dist_path = (
                        temp_dir / "cpt_coords_full_with_distances.csv"
                    )
                    if coords_full_with_dist_path.exists():
                        coords_full_sorted = pd.read_csv(coords_full_with_dist_path)
                        coords_full_indexed = (
                            coords_full_sorted.set_index("name")
                            if "name" in coords_full_sorted.columns
                            else coords_full_sorted
                        )
                        for cpt in removed_cpts:
                            if cpt in coords_full_indexed.index:
                                cpt_x = coords_full_indexed.loc[cpt, "cum_along_m"]
                                # Check if this CPT falls within the section extent
                                if x0 <= cpt_x <= x1:
                                    ax.axvline(
                                        x=cpt_x,
                                        color="black",
                                        linewidth=1.5,
                                        linestyle="--",
                                        alpha=0.7,
                                        zorder=11,
                                    )

                    section_idx = section_info["section_index"]
                    plt.title(
                        f"Validation - Section {section_idx:03d}, Depth Window {w_idx}",
                        fontsize=config.PLOT_FONT_SIZE,
                    )
                    plt.tight_layout()
                    plt.savefig(output_png, dpi=220, bbox_inches="tight")
                    plt.close()

                # Check each removed CPT using FULL coordinates
                for cpt in removed_cpts:
                    # Find CPT's position in the FULL coordinates DataFrame
                    cpt_dist_row = coords_with_dist_full[
                        coords_with_dist_full["name"] == cpt
                    ]
                    if cpt_dist_row.empty:
                        continue

                    cpt_m = cpt_dist_row["cum_along_m"].values[0]

                    # Check if this CPT falls within the section's extent
                    if start_m <= cpt_m <= start_m + total_span_m:
                        # Calculate pixel position within section
                        relative_m = cpt_m - start_m
                        pixel_position = int((relative_m / total_span_m) * (n_cols - 1))
                        pixel_position = np.clip(pixel_position, 0, n_cols - 1)

                        # Extract prediction at this position
                        y_pred = gan_result[:, pixel_position]

                        # Get true values from original data (for this depth window)
                        y_true = cpt_df_full.iloc[
                            start_row:end_row, cpt_df_full.columns.get_loc(cpt)
                        ].values

                        # Calculate metrics
                        mae = mean_absolute_error(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)

                        mae_by_cpt_by_window[cpt].append(mae)
                        mse_by_cpt_by_window[cpt].append(mse)

        # Average metrics across all occurrences for each CPT
        mae_results = {}
        mse_results = {}

        for cpt in removed_cpts:
            if mae_by_cpt_by_window[cpt]:
                mae_results[cpt] = np.mean(mae_by_cpt_by_window[cpt])
                mse_results[cpt] = np.mean(mse_by_cpt_by_window[cpt])
            else:
                # CPT not found in any section (shouldn't happen but handle it)
                mae_results[cpt] = np.nan
                mse_results[cpt] = np.nan
                print(f"      WARNING: CPT {cpt} not found in any section!")

        # Save combined manifest for mosaic creation
        if all_manifests:
            manifest_df = pd.DataFrame(all_manifests)
            manifest_csv_path = temp_dir / "manifest_sections.csv"
            manifest_df.to_csv(manifest_csv_path, index=False)
            print(f"  Saved manifest with {len(all_manifests)} sections")

        # Create mosaic if images were saved
        if save_images and images_folder is not None:
            print("  Creating validation mosaic...")
            manifest_csv = temp_dir / "manifest_sections.csv"
            coords_csv_reduced = temp_dir / "cpt_coords_with_distances.csv"
            if manifest_csv.exists() and coords_csv_reduced.exists():
                try:
                    coords_csv_full = temp_dir / "cpt_coords_full_with_distances.csv"
                    create_validation_mosaic(
                        manifest_csv,
                        coords_csv_reduced,
                        coords_csv_full,
                        images_folder,
                        n_cols,
                        n_rows,
                        y_top_m,
                        y_bottom_m,
                        removed_cpts,
                    )
                    print("  ✓ Mosaic created successfully")
                except Exception as e:
                    print(f"      Warning: Failed to create mosaic: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                print(
                    f"      Warning: Missing manifest or coordinates for mosaic creation"
                )
                if not manifest_csv.exists():
                    print(f"        Missing: {manifest_csv}")
                if not coords_csv_reduced.exists():
                    print(f"        Missing: {coords_csv_reduced}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    return mae_results, mse_results, removed_cpts


def create_validation_mosaic(
    manifest_csv: Path,
    coords_csv: Path,
    coords_csv_full: Path,
    images_folder: Path,
    n_cols: int,
    n_rows: int,
    y_top_m: float,
    y_bottom_m: float,
    removed_cpts: List[str] = None,
) -> None:
    """
    Create a mosaic combining all validation sections into a single image.

    Parameters:
        manifest_csv: Path to manifest CSV with section information
        coords_csv: Path to coordinates CSV (reduced, after removing CPTs)
        coords_csv_full: Path to full coordinates CSV (including removed CPTs)
        images_folder: Folder containing section CSV/PNG files and where mosaic will be saved
        n_cols: Number of columns per section (512)
        n_rows: Number of rows per section (32)
        y_top_m: Top depth in meters
        y_bottom_m: Bottom depth in meters
        removed_cpts: List of removed CPT names (for dashed line markers)
    """
    from core import create_mosaic as mosaic_creation
    from core.create_mosaic import build_mosaic

    print("  Creating mosaic from all sections...")

    # Load manifest and coordinates
    try:
        manifest = pd.read_csv(manifest_csv)
        coords = pd.read_csv(coords_csv)
    except Exception as e:
        print(f"    Failed to load manifest or coordinates: {e}")
        return

    # Temporarily patch mosaic_creation module constants to match validation context
    # This is necessary because build_mosaic uses these module-level constants
    original_n_cols = mosaic_creation.N_COLS
    original_n_rows = mosaic_creation.N_ROWS_WINDOW
    original_y_top = mosaic_creation.Y_TOP_M
    original_y_bottom = mosaic_creation.Y_BOTTOM_M
    original_gan_dir = mosaic_creation.GAN_DIR

    mosaic_creation.N_COLS = n_cols
    mosaic_creation.N_ROWS_WINDOW = n_rows
    mosaic_creation.Y_TOP_M = y_top_m
    mosaic_creation.Y_BOTTOM_M = y_bottom_m
    mosaic_creation.GAN_DIR = images_folder  # Point to validation GAN files

    try:
        # Use the build_mosaic function with validation manifest
        mosaic, xmin, xmax, global_dx, n_rows_total = build_mosaic(manifest, coords)

        # Save mosaic CSV
        mosaic_csv = images_folder / "validation_mosaic.csv"
        pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False, header=False)
        print(f"    Mosaic CSV saved: {mosaic_csv.name}")

        # Save mosaic PNG with custom IC colormap
        mosaic_png = images_folder / "validation_mosaic.png"
        cmap, vmin_val, vmax_val = create_custom_ic_colormap(
            config.IC_MIN,
            config.IC_SAND_SANDMIX_BOUNDARY,
            config.IC_SANDMIX_SILTMIX_BOUNDARY,
            config.IC_SILTMIX_CLAY_BOUNDARY,
            config.IC_CLAY_ORGANIC_BOUNDARY,
            config.IC_MAX,
        )
        ic_boundaries = (
            config.IC_MIN,
            config.IC_SAND_SANDMIX_BOUNDARY,
            config.IC_SANDMIX_SILTMIX_BOUNDARY,
            config.IC_SILTMIX_CLAY_BOUNDARY,
            config.IC_CLAY_ORGANIC_BOUNDARY,
            config.IC_MAX,
        )

        # Plot mosaic using the same plot_mosaic function as the original mosaic
        # This ensures consistent axis labels and styling
        from core.create_mosaic import plot_mosaic

        # Use plot_mosaic to create base visualization with proper axes
        plot_mosaic(
            mosaic,
            xmin,
            xmax,
            global_dx,
            n_rows_total,
            mosaic_png,
            coords=coords,
            show_cpt_locations=True,
            vmin=vmin_val,
            vmax=vmax_val,
            cmap=cmap,
            colorbar_label="IC Value",
            ic_boundaries=ic_boundaries,
        )

        # Now add removed CPT markers by reopening the saved figure
        if removed_cpts and coords_csv_full.exists():
            from matplotlib.image import imread

            img = imread(mosaic_png)

            # Create figure with same settings as plot_mosaic
            base_width = 20
            height = base_width / 8
            fig, ax = plt.subplots(figsize=(base_width, height))

            # Display the mosaic with proper extent
            ax.imshow(
                mosaic,
                cmap=cmap,
                vmin=vmin_val,
                vmax=vmax_val,
                aspect="auto",
                extent=[
                    xmin - global_dx / 2,
                    xmax + global_dx / 2,
                    n_rows_total - 0.5,
                    -0.5,
                ],
            )

            # Add colorbar
            cbar = plt.colorbar(ax.images[0], label="IC Value", extend="both")
            cbar.ax.tick_params(labelsize=config.PLOT_FONT_SIZE)
            cbar.set_label("IC Value", fontsize=config.PLOT_FONT_SIZE)
            cbar.set_ticks(list(ic_boundaries))
            cbar.set_ticklabels(
                [f"{val:g}" for val in ic_boundaries], fontsize=config.PLOT_FONT_SIZE
            )

            # Add present CPT markers (solid)
            for cpt_x in coords["cum_along_m"]:
                if xmin <= cpt_x <= xmax:
                    ax.axvline(
                        x=cpt_x, color="black", linewidth=1, alpha=0.5, zorder=10
                    )

            # Add removed CPT markers (dashed)
            coords_full = pd.read_csv(coords_csv_full)
            coords_full_indexed = (
                coords_full.set_index("name")
                if "name" in coords_full.columns
                else coords_full
            )
            for cpt in removed_cpts:
                if cpt in coords_full_indexed.index:
                    cpt_x = coords_full_indexed.loc[cpt, "cum_along_m"]
                    if xmin <= cpt_x <= xmax:
                        ax.axvline(
                            x=cpt_x,
                            color="black",
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.7,
                            zorder=11,
                        )

            # Set labels and axes (matching plot_mosaic)
            ax.set_xlabel("Distance along line (m)", fontsize=config.PLOT_FONT_SIZE)
            ax.set_ylabel("Depth Index (global)", fontsize=config.PLOT_FONT_SIZE)
            ax.tick_params(axis="both", labelsize=config.PLOT_FONT_SIZE)

            # Add secondary axes (matching plot_mosaic)
            def m_to_px(x):
                return (x - xmin) / global_dx

            def px_to_m(p):
                return xmin + p * global_dx

            top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
            top.set_xlabel("Pixel index (0…W-1)", fontsize=config.PLOT_FONT_SIZE)
            top.tick_params(labelsize=config.PLOT_FONT_SIZE)

            def idx_to_m(y_idx):
                return y_top_m + (y_idx / (n_rows_total - 1)) * (y_bottom_m - y_top_m)

            def m_to_idx(y_m):
                denom = y_bottom_m - y_top_m
                return (
                    0
                    if abs(denom) < 1e-12
                    else (y_m - y_top_m) * (n_rows_total - 1) / denom
                )

            right = ax.secondary_yaxis("right", functions=(idx_to_m, m_to_idx))
            right.set_ylabel("Depth (m)", fontsize=config.PLOT_FONT_SIZE)
            right.tick_params(labelsize=config.PLOT_FONT_SIZE)

            plt.title(
                f"Validation Mosaic (with {len(removed_cpts)} CPTs removed)",
                fontsize=config.PLOT_FONT_SIZE,
            )
            plt.tight_layout()
            plt.savefig(mosaic_png, dpi=800, bbox_inches="tight")
            plt.close()

        # Create interactive HTML viewer with proper axes
        try:
            from core.utils import create_interactive_html

            mosaic_html = images_folder / "validation_mosaic.html"
            extent = (xmin, xmax, n_rows_total - 1, 0)
            create_interactive_html(
                mosaic_png,
                mosaic_html,
                title="Validation Mosaic (Interactive)",
                extent=extent,
                xlabel="Distance along line (m)",
                ylabel="Depth Index (global)",
            )
            print(f"    Interactive HTML created: {mosaic_html.name}")
        except Exception as html_err:
            print(f"    Warning: Could not create HTML: {html_err}")
        print(f"    Mosaic PNG saved: {mosaic_png.name}")

    finally:
        # Restore original module constants
        mosaic_creation.N_COLS = original_n_cols
        mosaic_creation.N_ROWS_WINDOW = original_n_rows
        mosaic_creation.Y_TOP_M = original_y_top
        mosaic_creation.Y_BOTTOM_M = original_y_bottom
        mosaic_creation.GAN_DIR = original_gan_dir


def save_validation_results(
    all_mae_results: List[Dict[str, float]],
    all_mse_results: List[Dict[str, float]],
    output_folder: Path,
):
    """
    Save validation results to CSV files with comprehensive statistics.

    Parameters:
        all_mae_results: List of MAE dictionaries (one per run)
        all_mse_results: List of MSE dictionaries (one per run)
        output_folder: Folder to save results
    """
    # Build MAE DataFrame with per-run results
    mae_rows = []
    for run_idx, mae_dict in enumerate(all_mae_results, 1):
        row = {"Run_no": run_idx}
        row.update(mae_dict)
        mae_rows.append(row)

    df_mae = pd.DataFrame(mae_rows)

    # Calculate mean and std for each run (across CPTs) and add as last columns
    cpt_columns = [col for col in df_mae.columns if col != "Run_no"]
    df_mae["Mean_MAE_of_run"] = df_mae[cpt_columns].mean(axis=1)
    df_mae["Std_MAE_of_run"] = df_mae[cpt_columns].std(axis=1, ddof=1)

    # Calculate per-CPT statistics across all runs
    mean_row_mae = {"Run_no": "MEAN"}
    std_row_mae = {"Run_no": "STD"}

    for cpt in cpt_columns:
        values = df_mae[cpt].dropna()
        if len(values) > 0:
            mean_row_mae[cpt] = np.mean(values)
            std_row_mae[cpt] = np.std(values, ddof=1)
        else:
            mean_row_mae[cpt] = np.nan
            std_row_mae[cpt] = np.nan

    # Overall statistics across all CPTs and runs
    all_mae_values = df_mae[cpt_columns].values.flatten()
    all_mae_values = all_mae_values[~np.isnan(all_mae_values)]
    mean_row_mae["Mean_MAE_of_run"] = np.mean(all_mae_values)
    mean_row_mae["Std_MAE_of_run"] = np.std(all_mae_values, ddof=1)

    # For STD row, show std of the per-run means and stds
    std_row_mae["Mean_MAE_of_run"] = np.std(df_mae["Mean_MAE_of_run"].dropna(), ddof=1)
    std_row_mae["Std_MAE_of_run"] = np.std(df_mae["Std_MAE_of_run"].dropna(), ddof=1)

    # Add statistics rows
    df_mae = pd.concat(
        [df_mae, pd.DataFrame([mean_row_mae, std_row_mae])], ignore_index=True
    )

    mae_csv = output_folder / "validation_mae_results.csv"
    df_mae.to_csv(mae_csv, index=False)
    print(f"\n✓ Saved MAE results to: {mae_csv}")

    # Build MSE DataFrame with per-run results
    mse_rows = []
    for run_idx, mse_dict in enumerate(all_mse_results, 1):
        row = {"Run_no": run_idx}
        row.update(mse_dict)
        mse_rows.append(row)

    df_mse = pd.DataFrame(mse_rows)

    # Calculate mean and std for each run (across CPTs) and add as last columns
    cpt_columns = [col for col in df_mse.columns if col != "Run_no"]
    df_mse["Mean_MSE_of_run"] = df_mse[cpt_columns].mean(axis=1)
    df_mse["Std_MSE_of_run"] = df_mse[cpt_columns].std(axis=1, ddof=1)

    # Calculate per-CPT statistics across all runs
    mean_row_mse = {"Run_no": "MEAN"}
    std_row_mse = {"Run_no": "STD"}

    for cpt in cpt_columns:
        values = df_mse[cpt].dropna()
        if len(values) > 0:
            mean_row_mse[cpt] = np.mean(values)
            std_row_mse[cpt] = np.std(values, ddof=1)
        else:
            mean_row_mse[cpt] = np.nan
            std_row_mse[cpt] = np.nan

    # Overall statistics across all CPTs and runs
    all_mse_values = df_mse[cpt_columns].values.flatten()
    all_mse_values = all_mse_values[~np.isnan(all_mse_values)]
    mean_row_mse["Mean_MSE_of_run"] = np.mean(all_mse_values)
    mean_row_mse["Std_MSE_of_run"] = np.std(all_mse_values, ddof=1)

    # For STD row, show std of the per-run means and stds
    std_row_mse["Mean_MSE_of_run"] = np.std(df_mse["Mean_MSE_of_run"].dropna(), ddof=1)
    std_row_mse["Std_MSE_of_run"] = np.std(df_mse["Std_MSE_of_run"].dropna(), ddof=1)

    # Add statistics rows
    df_mse = pd.concat(
        [df_mse, pd.DataFrame([mean_row_mse, std_row_mse])], ignore_index=True
    )

    mse_csv = output_folder / "validation_mse_results.csv"
    df_mse.to_csv(mse_csv, index=False)
    print(f"✓ Saved MSE results to: {mse_csv}")


def run_validation_pipeline(
    folders: dict,
    compressed_csv: Path,
    y_top_m: float,
    y_bottom_m: float,
    n_runs: int = 10,
    n_remove: int = 12,
    base_seed: int = None,
):
    """
    Run the validation pipeline with leave-out cross-validation.

    Args:
        folders: Dictionary of folder paths from main pipeline
        compressed_csv: Path to compressed CPT data CSV
        y_top_m: Top depth in meters
        y_bottom_m: Bottom depth in meters
        n_runs: Number of validation runs
        n_remove: Number of CPTs to remove per run
        base_seed: Random seed for reproducibility
    """
    import json

    logger.info(
        f"Starting validation with {n_runs} runs, removing {n_remove} CPTs per run"
    )

    VALIDATION_FOLDER = folders["8_validation"]
    VALIDATION_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load SchemaGAN model
    logger.info(f"Loading SchemaGAN model from: {config.SCHGAN_MODEL_PATH}")
    model = load_model(config.SCHGAN_MODEL_PATH, compile=False)
    logger.info("Model loaded successfully")

    # Path to coordinates
    coords_csv = folders["1_coords"] / "cpt_coordinates.csv"
    if not compressed_csv.exists():
        raise FileNotFoundError(
            f"Compressed CPT data not found: {compressed_csv}\n"
            f"Please run Step 2 (RUN_STEP_2_PREPARE_CPTS = True) to generate this file first."
        )
    if not coords_csv.exists():
        raise FileNotFoundError(
            f"CPT coordinates not found: {coords_csv}\n"
            f"Please run Step 1 (RUN_STEP_1_EXTRACT_COORDS = True) to generate this file first."
        )

    # Load CPT names
    df_temp = pd.read_csv(compressed_csv)
    cpt_names = df_temp.columns[1:].tolist()
    n_cpts = len(cpt_names)
    cpt_depth_pixels = len(df_temp)

    logger.info(f"Total CPTs available: {n_cpts}")
    logger.info(f"CPT depth resolution: {cpt_depth_pixels} pixels")

    if cpt_depth_pixels > config.N_ROWS:
        n_windows = int(
            np.ceil(
                (cpt_depth_pixels - config.N_ROWS)
                / (config.N_ROWS * (1 - config.VERTICAL_OVERLAP / 100))
            )
            + 1
        )
    else:
        n_windows = 1

    logger.info(f"Number of depth windows: {n_windows}")

    # Run validation iterations
    all_mae_results = []
    all_mse_results = []

    for run_idx in range(1, n_runs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Validation Run {run_idx}/{n_runs}")
        logger.info(f"{'='*60}")

        # Set seed for this run
        run_seed = base_seed + run_idx if base_seed is not None else None

        # Create subfolder for this run
        run_folder = VALIDATION_FOLDER / f"run_{run_idx:02d}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # Run validation (this internally filters CPTs, creates sections, and generates predictions)
        mae_dict, mse_dict, removed_cpts = run_validation(
            compressed_csv=compressed_csv,
            coords_csv=coords_csv,
            model=model,
            n_cols=config.N_COLS,
            n_rows=config.N_ROWS,
            cpts_per_section=config.CPTS_PER_SECTION,
            overlap_cpts=config.OVERLAP_CPTS,
            left_pad_frac=config.LEFT_PAD_FRACTION,
            right_pad_frac=config.RIGHT_PAD_FRACTION,
            dir_from=config.DIR_FROM,
            dir_to=config.DIR_TO,
            y_top_m=y_top_m,
            y_bottom_m=y_bottom_m,
            n_remove=n_remove,
            random_state=run_seed,
            vertical_overlap_pct=config.VERTICAL_OVERLAP,
            save_images=True,
            images_folder=run_folder,
        )

        logger.info(f"Removed CPTs ({len(removed_cpts)}): {', '.join(removed_cpts)}")

        all_mae_results.append(mae_dict)
        all_mse_results.append(mse_dict)

        # Display results
        mean_mae = np.mean(list(mae_dict.values()))
        mean_mse = np.mean(list(mse_dict.values()))
        logger.info(f"Run {run_idx} - Mean MAE: {mean_mae:.4f}")
        logger.info(f"Run {run_idx} - Mean MSE: {mean_mse:.4f}")

    # Save aggregated results
    logger.info(f"\n{'='*60}")
    logger.info("Saving aggregated results...")
    logger.info(f"{'='*60}")
    save_validation_results(all_mae_results, all_mse_results, VALIDATION_FOLDER)

    # Compute and display overall statistics
    all_mae_means = [np.mean(list(d.values())) for d in all_mae_results]
    all_mse_means = [np.mean(list(d.values())) for d in all_mse_results]

    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total runs: {n_runs}")
    logger.info(f"CPTs removed per run: {n_remove}")
    logger.info(
        f"\nOverall MAE: {np.mean(all_mae_means):.4f} ± {np.std(all_mae_means):.4f}"
    )
    logger.info(
        f"Overall MSE: {np.mean(all_mse_means):.4f} ± {np.std(all_mse_means):.4f}"
    )
    logger.info(
        f"\nMAE range: [{np.min(all_mae_means):.4f}, {np.max(all_mae_means):.4f}]"
    )
    logger.info(
        f"MSE range: [{np.min(all_mse_means):.4f}, {np.max(all_mse_means):.4f}]"
    )
    logger.info(f"{'='*60}")
    logger.info(f"\nValidation complete! Results saved to: {VALIDATION_FOLDER}")
