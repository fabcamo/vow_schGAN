"""
Main script for the VOW SchemaGAN pipeline.

This script orchestrates the complete workflow by calling individual script functions:
1. Setup experiment folder structure
2. Extract coordinates from CPT files (calls extract_coords.py)
3. Extract and compress CPT data (calls extract_data.py)
4. Create sections for SchemaGAN input (calls create_schGAN_input_file.py)
5. Generate schemas using trained SchemaGAN model (calls create_schema.py)
6. Create mosaic from generated schemas (calls create_mosaic.py)

Configure the paths and parameters in the CONFIG section below.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Configure logging
LOG_LEVEL = logging.INFO  # Set to logging.DEBUG for more detail
VERBOSE = (
    True  # Default True to show verbose internal progress (set False to reduce output)
)

# Logger will be configured in main() after experiment folder is created
logger = None


def trace(message: str, level: int = logging.INFO):
    """Emit a verbose message if VERBOSE mode is enabled.

    Args:
        message: The log message to emit.
        level: Logging level (default: logging.INFO). Can be INFO, DEBUG, WARNING, ERROR.

    Note:
        Messages are only emitted when the global VERBOSE flag is True.
    """
    if VERBOSE:
        logger.log(level, message)


# Add GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

from utils import setup_experiment

# =============================================================================
# CONFIG - Modify these paths and parameters for your experiment
# =============================================================================

# Base configuration
RES_DIR = Path(r"C:\VOW\res")
REGION = "north"
EXP_NAME = "exp_9"
DESCRIPTION = "CPT compression to 64, 3 CPT overlap, 50% vertical overlap. 10% padding"

# Input data paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts\betuwepand\dike_north_BRO")
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")

# Processing parameters
COMPRESSION_METHOD = "mean"  # "mean" or "max" for IC value compression

# CPT Data Compression - can be different from model input size
CPT_DEPTH_PIXELS = 64  # Number of depth levels to compress raw CPT data to
# Can be 32, 64, 128, etc. independent of N_ROWS
# Higher values preserve more detail from raw CPT data

# Model Input Dimensions - MUST match what SchemaGAN expects
N_COLS = 512  # Number of columns in sections (SchemaGAN expects 512)
N_ROWS = 32  # Number of rows in sections (SchemaGAN expects 32)
# If CPT_DEPTH_PIXELS != N_ROWS, resampling will occur in section creation

CPTS_PER_SECTION = 6  # Number of CPTs per section
OVERLAP_CPTS = 3  # Number of overlapping CPTs between sections (horizontal)

# Vertical windowing parameters
VERTICAL_OVERLAP = 50  # [%] Vertical overlap between depth windows (0.0 = no overlap, 50.0 = 50% overlap)
# Only used if CPT_DEPTH_PIXELS > N_ROWS
# Example: With 128px CPT data and 32px windows, 0% overlap = 4 windows, 50% overlap = 7 windows

LEFT_PAD_FRACTION = 0.1  # Left padding as fraction of section span
RIGHT_PAD_FRACTION = 0.1  # Right padding as fraction of section span
DIR_FROM, DIR_TO = "west", "east"  # Sorting direction

# Optional: Real depth range for visualization (will be computed if None)
Y_TOP_M: Optional[float] = None
Y_BOTTOM_M: Optional[float] = None

# =============================================================================


def run_coordinate_extraction(cpt_folder: Path, coords_csv: Path):
    """Extract and validate CPT coordinates from GEF files.

    This function processes all .gef files in the specified folder, validates their
    coordinates, and exports them to a CSV file. Invalid files are moved to a 'no_coords'
    subfolder.

    Args:
        cpt_folder: Path to folder containing .gef CPT files.
        coords_csv: Output path for the coordinates CSV file.

    Raises:
        FileNotFoundError: If the CPT folder does not exist.

    Note:
        Coordinates are validated for Netherlands RD system (6-digit format).
        Auto-corrects common scaling errors (e.g., 123.456 → 123456.000).
    """
    from extract_coords import process_cpt_coords

    if not cpt_folder.exists():
        raise FileNotFoundError(f"CPT folder does not exist: {cpt_folder}")

    process_cpt_coords(cpt_folder, coords_csv)
    logger.info(f"Coordinates extracted to: {coords_csv}")
    logger.info("Step 2 complete.")


def run_cpt_data_processing(
    cpt_folder: Path,
    output_folder: Path,
    compression_method: str = "mean",
    cpt_depth_pixels: int = 32,
):
    """Process, interpret, and compress CPT data to specified depth resolution.

    This function reads GEF files, performs Robertson CPT interpretation to calculate
    soil behavior index (IC), equalizes depth ranges across all CPTs, and compresses
    the data to a standardized format. The compressed data can later be resampled
    to match the model's expected input size if needed.

    Args:
        cpt_folder: Path to folder containing .gef CPT files.
        output_folder: Path where compressed CPT data will be saved.
        compression_method: Method for aggregating IC values ("mean" or "max").
            - "mean": Average IC value in each depth bin (smoother profiles)
            - "max": Maximum IC value in each depth bin (preserves extremes)
        cpt_depth_pixels: Number of depth levels (pixels) to compress CPT data to.
            This is independent of the model's expected input size (N_ROWS).
            For example, compress to 64 pixels, then later resample to 32 for the model.

    Returns:
        tuple: (output_path, lowest_max_depth, lowest_min_depth)
            - output_path: Path to saved compressed CSV file
            - lowest_max_depth: Shallowest starting depth across all CPTs (meters)
            - lowest_min_depth: Deepest ending depth across all CPTs (meters)

    Note:
        Output CSV has columns: Depth_Index (0 to cpt_depth_pixels-1), CPT1, CPT2, ...
        Each CPT column contains IC values at corresponding depth levels.
        The compression size (cpt_depth_pixels) does NOT need to match N_ROWS.
    """
    from extract_data import (
        process_cpts,
        equalize_top,
        equalize_depth,
        compress_cpt_data,
        save_cpt_to_csv,
    )
    from utils import read_files

    # Get CPT files
    cpt_files = read_files(str(cpt_folder), extension=".gef")

    # Process CPTs
    data_cpts, coords_simple = process_cpts(cpt_files)
    logger.info(f"Processed {len(data_cpts)} CPT files")

    # Find depth limits and equalize
    original_data_cpts = [cpt.copy() for cpt in data_cpts]
    lowest_max_depth = min(cpt["depth_max"] for cpt in data_cpts)
    lowest_min_depth = min(cpt["depth_min"] for cpt in data_cpts)

    logger.info(f"Depth range: {lowest_max_depth:.3f} to {lowest_min_depth:.3f} m")

    # Process data
    equalized_top_cpts = equalize_top(original_data_cpts)
    equalized_depth_cpts = equalize_depth(equalized_top_cpts, lowest_min_depth)
    compressed_cpts = compress_cpt_data(
        equalized_depth_cpts, method=compression_method, n_pixels=cpt_depth_pixels
    )

    # Save results
    output_filename = (
        f"compressed_cpt_data_{compression_method}_{cpt_depth_pixels}px.csv"
    )
    save_cpt_to_csv(compressed_cpts, str(output_folder), output_filename)

    output_path = output_folder / output_filename
    logger.info(f"Compressed CPT data saved to: {output_path}")
    logger.info("Step 3 complete.")

    return output_path, lowest_max_depth, lowest_min_depth


def run_section_creation(
    coords_csv: Path,
    cpt_csv: Path,
    output_folder: Path,
    n_cols: int,
    n_rows: int,
    cpts_per_section: int,
    overlap: int,
    left_pad: float,
    right_pad: float,
    dir_from: str,
    dir_to: str,
    vertical_overlap: float = 0.0,
):
    """Create spatial sections with overlapping CPTs for SchemaGAN input.

    This function sorts CPTs spatially, divides them into overlapping sections both
    horizontally (by CPT count) and vertically (by depth windows), and generates input
    matrices (n_rows x n_cols) where CPT data is positioned at correct spatial locations
    with padding. Areas without data are filled with zeros.

    If CPT data has more rows than n_rows, it will be split into vertical windows
    (e.g., 128px compressed data → 4 windows of 32px each).

    Args:
        coords_csv: Path to CSV file with CPT coordinates (columns: name, x, y).
        cpt_csv: Path to compressed CPT data CSV (columns: Depth_Index, CPT names...).
        output_folder: Directory where section files will be saved.
        n_cols: Number of columns (width) in each section matrix.
        n_rows: Number of rows (depth levels) in each section matrix (SchemaGAN expects 32).
        cpts_per_section: Number of CPTs to include in each section.
        overlap: Number of CPTs that overlap between consecutive sections.
        left_pad: Left padding as fraction of section span (e.g., 0.10 = 10%).
        right_pad: Right padding as fraction of section span (e.g., 0.10 = 10%).
        dir_from: Starting direction ("west", "east", "north", or "south").
        dir_to: Ending direction ("west", "east", "north", or "south").
        vertical_overlap: Vertical overlap percentage between depth windows (0.0 to 99.0).
                         Only applies if CPT data rows > n_rows.

    Returns:
        list: Combined manifest list of dictionaries for all depth windows, each containing:
            - section_index: Section number
            - start_idx, end_idx: CPT indices in this section
            - span_m: Real-world distance between first and last CPT
            - left_pad_m, right_pad_m: Padding distances in meters
            - depth_window: Depth window index (0, 1, 2, ...)
            - depth_start_row: Starting row in original CPT data
            - depth_end_row: Ending row in original CPT data
            - csv_path: Path to section CSV file

    """
    from create_schGAN_input_file import (
        process_sections,
        write_manifest,
        validate_input_files,
        split_cpt_into_windows,
    )
    import pandas as pd
    import numpy as np

    # Load data
    coords_df = pd.read_csv(coords_csv)
    cpt_df_full = pd.read_csv(cpt_csv)

    # Adjust padding strategy based on horizontal overlap
    # Strategy: With overlap, use modest padding for blending
    #           Without overlap, use extended padding to fill gaps between CPT groups
    actual_left_pad = left_pad
    actual_right_pad = right_pad

    if overlap == 0:
        # With no CPT overlap, sections cover different CPT groups
        # Use larger padding (50%) to extend sections and fill gaps between groups
        logger.info(
            "No horizontal overlap (overlap=0) detected. "
            "Using extended padding (50%) to fill gaps between CPT groups."
        )
        actual_left_pad = 0.50
        actual_right_pad = 0.50
    else:
        # With CPT overlap, use configured padding for smooth blending
        actual_left_pad = left_pad
        actual_right_pad = right_pad
        logger.info(
            f"Horizontal overlap detected (overlap={overlap}). "
            f"Using padding: left={actual_left_pad:.2%}, right={actual_right_pad:.2%}"
        )

    # Determine if we need vertical windowing
    total_cpt_rows = len(cpt_df_full)

    if total_cpt_rows == n_rows:
        # No vertical windowing needed - process as single depth level
        logger.info(
            f"CPT data has {total_cpt_rows} rows matching n_rows={n_rows}. Processing as single depth level."
        )

        validate_input_files(coords_df, cpt_df_full, n_rows)

        manifest = process_sections(
            coords_df=coords_df,
            cpt_df=cpt_df_full,
            out_dir=output_folder,
            n_cols=n_cols,
            n_rows=n_rows,
            per=cpts_per_section,
            overlap=overlap,
            left_pad_frac=actual_left_pad,
            right_pad_frac=actual_right_pad,
            from_where=dir_from,
            to_where=dir_to,
            depth_window=None,  # No depth windowing
            depth_start_row=None,
            depth_end_row=None,
            write_distances=True,
        )

    elif total_cpt_rows > n_rows:
        # Vertical windowing required
        logger.info(
            f"CPT data has {total_cpt_rows} rows > n_rows={n_rows}. "
            f"Splitting into vertical windows with {vertical_overlap:.1f}% overlap."
        )

        # Split CPT data into vertical windows
        depth_windows = split_cpt_into_windows(
            cpt_df=cpt_df_full,
            window_rows=n_rows,
            vertical_overlap_pct=vertical_overlap,
        )

        logger.info(f"Created {len(depth_windows)} depth windows")

        all_manifests = []

        for w_idx, start_row, end_row, cpt_df_win in depth_windows:
            logger.info(
                f"Processing depth window z_{w_idx:02d} "
                f"(rows {start_row}..{end_row-1} of original CPT data)..."
            )

            # Validate this window
            validate_input_files(coords_df, cpt_df_win, n_rows)

            # Only write distances file for first window (identical for all)
            write_dists = w_idx == 0

            manifest = process_sections(
                coords_df=coords_df,
                cpt_df=cpt_df_win,
                out_dir=output_folder,
                n_cols=n_cols,
                n_rows=n_rows,
                per=cpts_per_section,
                overlap=overlap,
                left_pad_frac=actual_left_pad,
                right_pad_frac=actual_right_pad,
                from_where=dir_from,
                to_where=dir_to,
                depth_window=w_idx,
                depth_start_row=start_row,
                depth_end_row=end_row,
                write_distances=write_dists,
            )

            all_manifests.extend(manifest)

        manifest = all_manifests

    else:
        raise ValueError(
            f"CPT data has {total_cpt_rows} rows but n_rows={n_rows}. "
            f"CPT data must have at least n_rows rows."
        )

    write_manifest(manifest, output_folder)
    logger.info(f"Created {len(manifest)} total sections in: {output_folder}")
    logger.info("Step 4 complete.")

    return manifest


def run_schema_generation(
    sections_folder: Path,
    gan_images_folder: Path,
    model_path: Path,
    y_top_m: float,
    y_bottom_m: float,
):
    """Generate detailed subsurface schemas using trained SchemaGAN model.

    This function loads section CSV files (sparse IC data with zeros), feeds them
    through the trained SchemaGAN generator network, and produces detailed subsurface
    schemas as both CSV data and PNG visualizations with proper coordinate axes.

    Args:
        sections_folder: Path to folder containing section CSV files and manifest.
        gan_images_folder: Output folder for generated schema images and data.
        model_path: Path to trained SchemaGAN model file (.h5).
        y_top_m: Top depth in meters (e.g., 6.8 for 6.8m below surface).
        y_bottom_m: Bottom depth in meters (e.g., -13.1 for 13.1m below surface).

    Returns:
        tuple: (success_count, fail_count)
            - success_count: Number of sections successfully processed
            - fail_count: Number of sections that failed processing

    Note:
        For each section, generates:
        - CSV file: Raw schema data (N_ROWS × N_COLS matrix of IC values)
        - PNG file: Visualization with 4 axes:
            * Bottom: Distance along line (meters)
            * Top: Pixel index (0 to N_COLS-1)
            * Left: Depth index (0 to N_ROWS-1)
            * Right: Real depth (meters)
    """

    # Ensure output directory exists
    gan_images_folder.mkdir(parents=True, exist_ok=True)

    # Get section files
    section_files = sorted(sections_folder.glob("section_*_cpts_*.csv"))
    if not section_files:
        raise FileNotFoundError(f"No section CSVs found in {sections_folder}")

    logger.info(f"Generating schemas for {len(section_files)} sections...")
    trace(f"Found {len(section_files)} section CSV files for schema generation")

    # Import required modules (avoiding create_schema module imports)
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt
    from utils import IC_normalization, reverse_IC_normalization
    import re

    # Load model
    logger.info("Loading SchemaGAN model...")
    model = load_model(model_path, compile=False)

    # Set random seed
    seed = np.random.randint(20220412, 20230412)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Using random seed: {seed}")

    # Load manifest and coordinates for proper axes
    manifest_csv = sections_folder / "manifest_sections.csv"
    coords_csv = sections_folder / "cpt_coords_with_distances.csv"

    manifest_df = pd.read_csv(manifest_csv)
    coords_df = pd.read_csv(coords_csv)

    # Use configured grid size
    SIZE_X, SIZE_Y = N_COLS, N_ROWS

    success_count = 0
    fail_count = 0

    for i, section_file in enumerate(section_files, 1):
        try:
            # Load and prepare section data
            df = pd.read_csv(section_file)

            # Remove index column if it exists
            if df.shape[1] == SIZE_X + 1:
                df_vals = df.iloc[:, 1:]  # Skip first column
            else:
                df_vals = df

            # Convert to GAN input format
            cs = df_vals.to_numpy(dtype=float).reshape(1, SIZE_Y, SIZE_X, 1)

            # Normalize and predict
            cs_norm = IC_normalization([cs, cs])[0]
            gan_result = model.predict(cs_norm, verbose=0)
            gan_result = reverse_IC_normalization(gan_result)
            gan_result = np.squeeze(gan_result)

            # Save CSV
            output_csv = gan_images_folder / f"{section_file.stem}_seed{seed}_gan.csv"
            pd.DataFrame(gan_result).to_csv(output_csv, index=False)
            logger.info(f"[INFO] Created GAN CSV: {output_csv.name}")
            trace(f"GAN CSV written: {output_csv}")

            # Parse section index for PNG
            m = re.search(r"section_(\d+)", section_file.stem)
            sec_index = int(m.group(1)) if m else i

            # Get section placement for proper coordinates
            r = manifest_df.loc[manifest_df["section_index"] == sec_index]
            if not r.empty:
                r = r.iloc[0]
                total_span = float(r["span_m"] + r["left_pad_m"] + r["right_pad_m"])
                start_idx = int(r["start_idx"])
                end_idx = int(r["end_idx"])
                m0 = float(coords_df.loc[start_idx, "cum_along_m"])
                x0 = m0 - float(r["left_pad_m"])
                dx = 1.0 if total_span <= 0 else total_span / (SIZE_X - 1)
                x1 = x0 + (SIZE_X - 1) * dx

                # Get CPT positions for this section (for markers)
                cpt_positions = coords_df.loc[start_idx:end_idx, "cum_along_m"].values
            else:
                x0, x1, dx = 0, SIZE_X - 1, 1
                cpt_positions = []

            # Create and save PNG
            output_png = gan_images_folder / f"{section_file.stem}_seed{seed}_gan.png"

            plt.figure(figsize=(10, 2.4))
            plt.imshow(
                gan_result,
                cmap="viridis",
                vmin=0,
                vmax=4.5,
                aspect="auto",
                extent=[x0, x1, SIZE_Y - 1, 0],
            )
            plt.colorbar(label="Value")

            ax = plt.gca()
            ax.set_xlabel("Distance along line (m)")
            ax.set_ylabel("Depth Index")

            # Top x-axis: pixel index
            def m_to_px(x):
                return (x - x0) / dx if dx != 0 else 0

            def px_to_m(p):
                return x0 + p * dx

            top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
            top.set_xlabel(f"Pixel index (0…{SIZE_X-1})")

            # Right y-axis: real depth
            def idx_to_meters(y_idx):
                return y_top_m + (y_idx / (SIZE_Y - 1)) * (y_bottom_m - y_top_m)

            def meters_to_idx(y_m):
                denom = y_bottom_m - y_top_m
                return (
                    0.0
                    if abs(denom) < 1e-12
                    else (y_m - y_top_m) * (SIZE_Y - 1) / denom
                )

            right = ax.secondary_yaxis(
                "right", functions=(idx_to_meters, meters_to_idx)
            )
            right.set_ylabel("Depth (m)")

            # Add vertical lines at CPT positions (for all sections/depth windows)
            for cpt_x in cpt_positions:
                ax.axvline(
                    x=cpt_x,
                    color="black",
                    linewidth=1,
                    linestyle="-",
                    alpha=0.5,
                    zorder=10,
                )

            # Explicitly set x-limits to prevent whitespace beyond image extent
            ax.set_xlim(x0, x1)

            plt.title(
                f"SchemaGAN Generated Image (Section {sec_index:03d}, Seed: {seed})"
            )
            plt.tight_layout()
            plt.savefig(output_png, dpi=220, bbox_inches="tight")
            plt.close()
            logger.info(f"[INFO] Created GAN PNG: {output_png.name}")
            trace(f"GAN PNG saved: {output_png}")

            success_count += 1
            logger.info(
                f"[{i:03d}/{len(section_files)}] Generated: {output_csv.name} & {output_png.name}"
            )

        except Exception as e:
            fail_count += 1
            logger.error(
                f"[{i:03d}/{len(section_files)}] Failed on {section_file.name}: {e}"
            )

    logger.info(
        f"Schema generation complete. Success: {success_count}, Failed: {fail_count}"
    )
    trace(f"Schema generation summary: success={success_count} failed={fail_count}")
    logger.info("Step 5 complete.")
    return success_count, fail_count


def run_mosaic_creation(
    sections_folder: Path,
    gan_images_folder: Path,
    mosaic_folder: Path,
    y_top_m: float,
    y_bottom_m: float,
):
    """Combine all generated schema sections into a seamless mosaic with vertical stacking.

    This function assembles individual schema sections into a complete subsurface
    visualization. It handles both horizontal overlaps (between adjacent sections) and
    vertical stacking (when depth windows were used), using bilinear interpolation
    for smooth blending.

    Args:
        sections_folder: Path to folder containing section manifest and coordinates.
        gan_images_folder: Path to folder with generated schema CSV files (*_gan.csv).
        mosaic_folder: Output folder for mosaic files.
        y_top_m: Top depth in meters for visualization axis.
        y_bottom_m: Bottom depth in meters for visualization axis.

    Raises:
        RuntimeError: If no valid GAN CSV files are found.

    Note:
        The mosaic uses bilinear interpolation to blend overlapping sections smoothly,
        both horizontally and vertically. If the manifest contains depth_window,
        depth_start_row, and depth_end_row columns, vertical stacking is applied
        based on original CPT data row indices. Otherwise, simple non-overlapping
        vertical stacking is used.

        Output files:
        - schemaGAN_mosaic.csv: Complete mosaic data matrix
        - schemaGAN_mosaic.png: Visualization with distance and depth axes
    """
    from create_mosaic import (
        load_inputs,
        find_latest_gan_csv_for_row,
        build_mosaic,
        plot_mosaic,
    )
    import pandas as pd

    # Ensure output directory exists
    mosaic_folder.mkdir(parents=True, exist_ok=True)
    trace(f"Mosaic output folder ready: {mosaic_folder}")

    # Check if we have generated schemas
    gan_files = list(gan_images_folder.glob("*_gan.csv"))
    if not gan_files:
        logger.warning("No generated schema files found. Skipping mosaic creation.")
        return

    logger.info(f"Creating mosaic from {len(gan_files)} generated schemas...")
    trace(f"GAN CSV count for mosaic: {len(gan_files)}")

    # Load manifest and coordinates
    manifest_csv = sections_folder / "manifest_sections.csv"
    coords_csv = sections_folder / "cpt_coords_with_distances.csv"

    logger.info(f"Loading manifest from: {manifest_csv}")
    logger.info(f"Loading coordinates from: {coords_csv}")

    try:
        manifest, coords = load_inputs(manifest_csv, coords_csv)
    except Exception as e:
        logger.error(f"Failed to load manifest or coordinates: {e}")
        raise

    # Debug: Check if vertical windowing columns are present
    logger.info(f"Manifest columns: {manifest.columns.tolist()}")
    if "depth_start_row" in manifest.columns and "depth_end_row" in manifest.columns:
        logger.info(
            f"Vertical windowing detected: depth_start_row range [{manifest['depth_start_row'].min()}..{manifest['depth_end_row'].max()-1}]"
        )
        logger.info(
            f"Expected mosaic height: {int(manifest['depth_end_row'].max()) - int(manifest['depth_start_row'].min())} rows"
        )
    else:
        logger.info("No vertical windowing columns found - will use simple stacking")

    # Temporarily update create_mosaic module constants BEFORE calling find_latest_gan_csv_for_row
    import create_mosaic

    original_gan_dir = create_mosaic.GAN_DIR
    original_n_cols = create_mosaic.N_COLS
    original_n_rows = create_mosaic.N_ROWS_WINDOW
    original_y_top = create_mosaic.Y_TOP_M
    original_y_bottom = create_mosaic.Y_BOTTOM_M

    create_mosaic.GAN_DIR = gan_images_folder
    create_mosaic.N_COLS = N_COLS
    create_mosaic.N_ROWS_WINDOW = N_ROWS
    create_mosaic.Y_TOP_M = y_top_m
    create_mosaic.Y_BOTTOM_M = y_bottom_m

    # Add GAN file paths to manifest using the external module's function
    logger.info("Matching GAN CSV files to sections...")
    manifest["gan_csv"] = manifest.apply(
        lambda row: find_latest_gan_csv_for_row(row), axis=1
    )

    # Check for missing GAN files
    missing = manifest[manifest["gan_csv"].isna()]
    if not missing.empty:
        logger.warning(
            f"Missing GAN CSV for sections: {missing['section_index'].tolist()}"
        )
        # Debug: Show some examples of what we're looking for
        for idx in missing["section_index"].head(3):
            row = manifest[manifest["section_index"] == idx].iloc[0]
            section_stem = Path(row["csv_path"]).stem
            pattern = f"{section_stem}_seed*_gan.csv"
            logger.warning(
                f"  Section {idx}: Looking for '{pattern}' in {gan_images_folder}"
            )

    manifest = manifest.dropna(subset=["gan_csv"]).reset_index(drop=True)
    if manifest.empty:
        logger.error("No sections with GAN CSVs found.")
        # Restore original values before raising
        create_mosaic.GAN_DIR = original_gan_dir
        create_mosaic.N_COLS = original_n_cols
        create_mosaic.N_ROWS_WINDOW = original_n_rows
        create_mosaic.Y_TOP_M = original_y_top
        create_mosaic.Y_BOTTOM_M = original_y_bottom
        raise RuntimeError("No sections with GAN CSVs found.")

    # Build mosaic (handles both horizontal overlaps and vertical stacking)
    logger.info("Building mosaic with vertical and horizontal blending...")
    logger.info(
        f"[DEBUG] Before build_mosaic: manifest shape={manifest.shape}, has depth_start_row={('depth_start_row' in manifest.columns)}"
    )
    try:
        mosaic, xmin, xmax, global_dx, n_rows_total = build_mosaic(manifest, coords)
        logger.info(
            f"[SUCCESS] Mosaic built: shape={mosaic.shape}, n_rows_total={n_rows_total}"
        )
    except Exception as e:
        logger.error(f"Failed to build mosaic: {e}")
        # Restore original values
        create_mosaic.GAN_DIR = original_gan_dir
        create_mosaic.N_COLS = original_n_cols
        create_mosaic.N_ROWS_WINDOW = original_n_rows
        create_mosaic.Y_TOP_M = original_y_top
        create_mosaic.Y_BOTTOM_M = original_y_bottom
        raise

    # Save mosaic CSV
    mosaic_csv = mosaic_folder / "schemaGAN_mosaic.csv"
    pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)
    logger.info(f"Mosaic CSV saved: {mosaic_csv}")
    trace(f"Mosaic shape: {mosaic.shape}, saved to {mosaic_csv}")

    # Create and save mosaic visualization
    mosaic_png = mosaic_folder / "schemaGAN_mosaic.png"
    logger.info(f"Creating mosaic visualization: {mosaic_png}")

    try:
        plot_mosaic(mosaic, xmin, xmax, global_dx, n_rows_total, mosaic_png)
    except Exception as e:
        logger.error(f"Failed to plot mosaic: {e}")
        # Restore original values
        create_mosaic.GAN_DIR = original_gan_dir
        create_mosaic.N_COLS = original_n_cols
        create_mosaic.N_ROWS_WINDOW = original_n_rows
        create_mosaic.Y_TOP_M = original_y_top
        create_mosaic.Y_BOTTOM_M = original_y_bottom
        raise

    # Restore original values
    create_mosaic.GAN_DIR = original_gan_dir
    create_mosaic.N_COLS = original_n_cols
    create_mosaic.N_ROWS_WINDOW = original_n_rows
    create_mosaic.Y_TOP_M = original_y_top
    create_mosaic.Y_BOTTOM_M = original_y_bottom

    logger.info(f"Mosaic complete: {mosaic_csv.name} & {mosaic_png.name}")
    logger.info("Step 6 complete.")
    trace(f"Mosaic creation finished: CSV and PNG saved")


def main():
    """Execute the complete VOW SchemaGAN pipeline.

    Orchestrates all six steps of the workflow:
    1. Creates experiment folder structure
    2. Extracts and validates CPT coordinates from GEF files
    3. Processes CPT data and compresses to 32-pixel depth profiles
    4. Creates spatial sections with overlapping CPTs
    5. Generates detailed schemas using SchemaGAN model
    6. Assembles all sections into a seamless mosaic

    All configuration is taken from module-level CONFIG constants.
    Results are saved in: {RES_DIR}/{REGION}/{EXP_NAME}/

    Logging:
        - Console: All INFO and higher messages
        - File: Complete log saved to {experiment_folder}/pipeline.log

    Returns:
        None. Exits early if any critical step fails.

    Note:
        Depth range (y_top_m, y_bottom_m) is auto-computed from CPT data
        if not explicitly set in CONFIG section.
    """

    global logger

    # First, create the experiment folder structure
    folders = setup_experiment(
        base_dir=RES_DIR, region=REGION, exp_name=EXP_NAME, description=DESCRIPTION
    )

    # Now configure logging to save in the experiment folder
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Set up file logging in the experiment folder
    try:
        LOG_FILE = folders["root"] / "pipeline.log"
        file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {LOG_FILE.resolve()}")
    except Exception as _e:
        print(f"Warning: Could not set up file logging: {_e}")

    logger.info("=" * 60)
    logger.info("Starting VOW SchemaGAN Pipeline")
    logger.info("=" * 60)

    # Debug: Check initial state
    logger.info(f"[DEBUG] Initial Y_TOP_M: {Y_TOP_M}")
    logger.info(f"[DEBUG] Initial Y_BOTTOM_M: {Y_BOTTOM_M}")
    logger.info(f"[DEBUG] CPT_FOLDER exists: {CPT_FOLDER.exists()}")
    logger.info(f"[DEBUG] SCHGAN_MODEL_PATH exists: {SCHGAN_MODEL_PATH.exists()}")

    # =============================================================================
    # 1. CREATE EXPERIMENT FOLDER STRUCTURE
    # =============================================================================
    logger.info("Step 1: Creating experiment folder structure...")
    logger.info(f"Experiment folders created at: {folders['root']}")

    # =============================================================================
    # 2. EXTRACT COORDINATES FROM CPT FILES
    # =============================================================================
    logger.info("Step 2: Extracting coordinates from CPT files...")

    coords_csv = folders["1_coords"] / "cpt_coordinates.csv"

    try:
        run_coordinate_extraction(CPT_FOLDER, coords_csv)
    except Exception as e:
        logger.error(f"Failed to extract coordinates: {e}")
        return

    # =============================================================================
    # 3. EXTRACT AND COMPRESS CPT DATA
    # =============================================================================
    logger.info("Step 3: Extracting and compressing CPT data...")

    # Initialize depth variables with module defaults
    y_top_final = Y_TOP_M
    y_bottom_final = Y_BOTTOM_M

    try:
        compressed_csv, y_top, y_bottom = run_cpt_data_processing(
            CPT_FOLDER,
            folders["2_compressed_cpt"],
            COMPRESSION_METHOD,
            CPT_DEPTH_PIXELS,
        )

        # Store depth range for later steps (use computed values if not set)
        if y_top_final is None:
            y_top_final = y_top
        if y_bottom_final is None:
            y_bottom_final = y_bottom

        logger.info(
            f"[DEBUG] CPT processing complete. y_top_final={y_top_final}, y_bottom_final={y_bottom_final}"
        )
        logger.info(f"[DEBUG] Compressed CSV: {compressed_csv}")

    except Exception as e:
        logger.error(f"Failed to process CPT data: {e}")
        logger.error(f"[DEBUG] Exception details: {e}")
        import traceback

        logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return

    # =============================================================================
    # 4. CREATE SECTIONS FOR SCHEMAGAN INPUT
    # =============================================================================
    logger.info("=" * 60)
    logger.info("Step 4: Creating sections for SchemaGAN input...")
    logger.info(f"[DEBUG] About to create sections with coords_csv: {coords_csv}")
    logger.info(f"[DEBUG] compressed_csv: {compressed_csv}")
    logger.info(f"[DEBUG] sections folder: {folders['3_sections']}")

    try:
        logger.info("[DEBUG] Calling run_section_creation function...")
        manifest = run_section_creation(
            coords_csv,
            compressed_csv,
            folders["3_sections"],
            N_COLS,
            N_ROWS,
            CPTS_PER_SECTION,
            OVERLAP_CPTS,
            LEFT_PAD_FRACTION,
            RIGHT_PAD_FRACTION,
            DIR_FROM,
            DIR_TO,
            VERTICAL_OVERLAP,
        )
        logger.info(f"[DEBUG] Section creation returned {len(manifest)} sections")
    except Exception as e:
        logger.error(f"Failed to create sections: {e}")
        logger.error(f"[DEBUG] Section creation exception: {e}")
        import traceback

        logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return

    # =============================================================================
    # 5. GENERATE SCHEMAS USING SCHEMAGAN MODEL
    # =============================================================================
    logger.info("=" * 60)
    logger.info("Step 5: Generating schemas using SchemaGAN model...")
    logger.info(f"[DEBUG] SchemaGAN model path: {SCHGAN_MODEL_PATH}")
    logger.info(f"[DEBUG] Model exists: {SCHGAN_MODEL_PATH.exists()}")

    if not SCHGAN_MODEL_PATH.exists():
        logger.warning(f"SchemaGAN model not found at: {SCHGAN_MODEL_PATH}")
        logger.warning("Skipping schema generation. Please provide valid model path.")
    else:
        try:
            logger.info("[DEBUG] Calling run_schema_generation function...")
            run_schema_generation(
                folders["3_sections"],
                folders["4_gan_images"],
                SCHGAN_MODEL_PATH,
                y_top_final,
                y_bottom_final,
            )
            logger.info("[DEBUG] Schema generation completed successfully")
        except Exception as e:
            logger.error(f"Failed to generate schemas: {e}")
            logger.error(f"[DEBUG] Schema generation exception: {e}")
            import traceback

            logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return

    # =============================================================================
    # 6. CREATE MOSAIC FROM GENERATED SCHEMAS
    # =============================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Creating mosaic from generated schemas...")
    logger.info("[DEBUG] About to call run_mosaic_creation function...")

    try:
        logger.info("[DEBUG] Calling run_mosaic_creation function...")
        run_mosaic_creation(
            folders["3_sections"],
            folders["4_gan_images"],
            folders["5_mosaic"],
            y_top_final,
            y_bottom_final,
        )
        logger.info("[DEBUG] Mosaic creation completed successfully")
    except Exception as e:
        logger.error(f"Failed to create mosaic: {e}")
        logger.error(f"[DEBUG] Mosaic creation exception: {e}")
        import traceback

        logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return

    # =============================================================================
    # COMPLETION
    # =============================================================================
    logger.info("=" * 60)
    logger.info("VOW SchemaGAN Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved in: {folders['root']}")
    logger.info("Folder structure:")
    for name, path in folders.items():
        if name != "root":
            logger.info(f"  {name}: {path}")

    # Summary statistics
    logger.info("\nPipeline Summary:")
    logger.info(
        f"  - Sections created: {len(manifest) if 'manifest' in locals() else 'N/A'}"
    )
    # Use computed depth range fallback if module-level values are None
    depth_top = (
        Y_TOP_M if Y_TOP_M is not None else (locals().get("y_top_final") or "N/A")
    )
    depth_bottom = (
        Y_BOTTOM_M
        if Y_BOTTOM_M is not None
        else (locals().get("y_bottom_final") or "N/A")
    )
    if isinstance(depth_top, float) and isinstance(depth_bottom, float):
        logger.info(f"  - Depth range: {depth_top:.3f} to {depth_bottom:.3f} m")
    else:
        logger.info(f"  - Depth range: {depth_top} to {depth_bottom} m")
    logger.info(f"  - Grid size: {N_ROWS} × {N_COLS} pixels")


if __name__ == "__main__":
    main()
