"""
Main script for the VOW SchemaGAN pipeline.

This script orchestrates the complete workflow by calling individual script functions:
1. Setup experiment folder structure (1_coords, 2_compressed_cpt, 3_sections, 4_gan_images, 5_enhance, 6_mosaic, 7_uncertainty)
2. Extract coordinates from CPT files (calls extract_coords.py)
3. Extract and compress CPT data (calls extract_data.py)
4. Create sections for SchemaGAN input (calls create_schGAN_input_file.py)
5. Generate schemas using trained SchemaGAN model (calls create_schema.py → 4_gan_images)
6. Enhance schemas with boundary sharpening (calls boundary_enhancement.py → 5_enhance)
7. Create mosaics from generated schemas (calls create_mosaic.py → 6_mosaic, creates both original and enhanced)
8. Compute prediction uncertainty using MC Dropout (calls uncertainty_quantification.py → 7_uncertainty)

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


# =============================================================================
# COLOR SCALE CONFIGURATION - Edit these values to change the color boundaries
# =============================================================================
# Three-category color scale for IC values:
# Category 1 (Yellow): IC_MIN to IC_YELLOW_ORANGE_BOUNDARY
# Category 2 (Orange): IC_YELLOW_ORANGE_BOUNDARY to IC_ORANGE_RED_BOUNDARY
# Category 3 (Red):    IC_ORANGE_RED_BOUNDARY to IC_MAX
# Everything outside IC_MIN to IC_MAX will be black

IC_MIN = 1.3  # Minimum IC value (start of yellow)
IC_YELLOW_ORANGE_BOUNDARY = 2.0  # Boundary between yellow and orange
IC_ORANGE_RED_BOUNDARY = 3.4  # Boundary between orange and red
IC_MAX = 4.2  # Maximum IC value (end of red)
# =============================================================================


def create_custom_ic_colormap():
    """Create custom segmented colormap for IC values.

    Uses the global configuration variables:
    - IC_MIN: Start of yellow range
    - IC_YELLOW_ORANGE_BOUNDARY: Yellow to orange transition
    - IC_ORANGE_RED_BOUNDARY: Orange to red transition
    - IC_MAX: End of red range

    Returns three-category colormap with sharp boundaries and black for out-of-range values.
    """
    from matplotlib.colors import ListedColormap
    import numpy as np

    # Use global configuration
    vmin = IC_MIN
    vmax = IC_MAX
    yellow_end = IC_YELLOW_ORANGE_BOUNDARY
    orange_end = IC_ORANGE_RED_BOUNDARY

    # Total number of discrete colors
    n_bins = 256

    # Proportion of each segment
    yellow_prop = (yellow_end - vmin) / (vmax - vmin)
    orange_prop = (orange_end - yellow_end) / (vmax - vmin)
    red_prop = (vmax - orange_end) / (vmax - vmin)

    # Number of bins for each segment
    n_yellow = int(n_bins * yellow_prop)
    n_orange = int(n_bins * orange_prop)
    n_red = n_bins - n_yellow - n_orange

    # Build color array with pure color gradients (no cross-segment blending)
    color_array = []

    # Yellow segment: dark yellow to bright yellow
    for i in range(n_yellow):
        intensity = 0.5 + 0.5 * (i / max(n_yellow - 1, 1))  # 0.5 to 1.0
        color_array.append([1.0, 1.0, 0.0, intensity])  # Pure yellow with varying alpha

    # Orange segment: dark orange to bright orange
    for i in range(n_orange):
        intensity = 0.5 + 0.5 * (i / max(n_orange - 1, 1))  # 0.5 to 1.0
        color_array.append(
            [1.0, 0.647, 0.0, intensity]
        )  # Pure orange with varying alpha

    # Red segment: dark red to bright red
    for i in range(n_red):
        intensity = 0.5 + 0.5 * (i / max(n_red - 1, 1))  # 0.5 to 1.0
        color_array.append([1.0, 0.0, 0.0, intensity])  # Pure red with varying alpha

    cmap = ListedColormap(color_array, name="custom_ic")
    cmap.set_under("black")  # Values < vmin
    cmap.set_over("black")  # Values > vmax
    cmap.set_bad("black")  # NaN values

    return cmap, vmin, vmax


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


base_path = Path(r"C:\VOW")  # Base path for experiments
# base_path = Path(r"N:\Projects\11211500\11211566\B. Measurements and calculations\007 - Remote sensing AI")

# Add GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")


from utils import setup_experiment

# =============================================================================
# CONFIG - Modify these paths and parameters for your experiment
# =============================================================================

# Base configuration
RES_DIR = Path(base_path / "res")  # Base results directory
REGION = "south"  # Region name for experiment folder and the data subfolder
EXP_NAME = "exp_12"
DESCRIPTION = (
    "Using the top fill with zeros method for equalizing CPT tops,"
    "new padding ideas,"
    "new color scale for IC visualization,"
    "CPT compression to 32,"
    "3 CPT overlap,"
    "50% vertical overlap,"
    "10% padding,"
    "Added boundary enhancement,"
    "Added uncertainty quantification with 10 samples,"
)

# Input data paths
CPT_FOLDER = Path(
    base_path / "data" / "cpts" / "betuwepand" / "dike_south_BRO"
)  # Folder with .gef CPT files
# CPT_FOLDER = Path(r"C:\VOW\data\cpts\waalbandijk")  # For quick testing with fewer CPTs
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")

# Processing parameters
COMPRESSION_METHOD = "mean"  # "mean" or "max" for IC value compression

# CPT Data Compression - can be different from model input size
CPT_DEPTH_PIXELS = 32  # Number of depth levels to compress raw CPT data to
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

# Visualization
SHOW_CPT_LOCATIONS = True  # Show vertical lines at CPT positions in plots (both individual sections and mosaic)

# Boundary Enhancement
ENHANCE_METHOD = "none"  # Enhancement method to sharpen layer boundaries
# Options:
#   "guided_filter": Edge-preserving guided filter (RECOMMENDED - best for GAN outputs, no halos)
#   "unsharp_mask": Classic sharpening via unsharp masking (can add artifacts)
#   "laplacian": Laplacian-based sharpening (aggressive, creates artifacts)
#   "dense_crf": Dense CRF edge-aware smoothing (experimental, ineffective)
#   "none" or None: No enhancement (use original GAN output)

# Uncertainty Quantification (Monte Carlo Dropout)
COMPUTE_UNCERTAINTY = True  # Compute prediction uncertainty using MC Dropout
N_MC_SAMPLES = (
    10  # Number of MC Dropout samples (20-100 typical, more = slower but more accurate)
)
# MC Dropout reveals where the GAN is uncertain in its predictions:
#   - High uncertainty: complex transitions, far from data, ambiguous interpolations
#   - Low uncertainty: near CPT locations, homogeneous layers, clear patterns

# Padding strategy: Use percentage of section span
LEFT_PAD_FRACTION = 0.1  # Left padding as fraction of section span (10%)
RIGHT_PAD_FRACTION = 0.1  # Right padding as fraction of section span (10%)
# Note: Padding is calculated per section based on its span, so:
#   - Small section (100m) → 10m padding on each side
#   - Large section (300m) → 30m padding on each side
# This keeps padding proportional to section size

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
        fill_top_with_zeros,
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
    # equalized_top_cpts = equalize_top(original_data_cpts)
    equalized_top_cpts = fill_top_with_zeros(original_data_cpts)
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
    uncertainty_folder: Optional[Path] = None,
    compute_uncertainty: bool = False,
    n_mc_samples: int = 50,
):
    """Generate detailed subsurface schemas using trained SchemaGAN model.

    This function loads section CSV files (sparse IC data with zeros), feeds them
    through the trained SchemaGAN generator network, and produces detailed subsurface
    schemas as both CSV data and PNG visualizations with proper coordinate axes.

    Optionally computes prediction uncertainty using Monte Carlo Dropout.

    Args:
        sections_folder: Path to folder containing section CSV files and manifest.
        gan_images_folder: Output folder for generated schema images and data.
        model_path: Path to trained SchemaGAN model file (.h5).
        y_top_m: Top depth in meters (e.g., 6.8 for 6.8m below surface).
        y_bottom_m: Bottom depth in meters (e.g., -13.1 for 13.1m below surface).
        uncertainty_folder: Output folder for uncertainty maps (if computing uncertainty).
        compute_uncertainty: Whether to compute MC Dropout uncertainty.
        n_mc_samples: Number of MC samples for uncertainty estimation.

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
    from uncertainty_quantification import (
        compute_mc_dropout_uncertainty,
        visualize_uncertainty,
        save_uncertainty_csv,
    )
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

            # Set up custom colormap
            cmap, vmin, vmax = create_custom_ic_colormap()

            plt.figure(figsize=(10, 2.4))
            plt.imshow(
                gan_result,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                extent=[x0, x1, SIZE_Y - 1, 0],
            )
            cbar = plt.colorbar(label="IC Value", extend="both")
            # Set custom ticks at color transition boundaries
            cbar.set_ticks(
                [IC_MIN, IC_YELLOW_ORANGE_BOUNDARY, IC_ORANGE_RED_BOUNDARY, IC_MAX]
            )
            cbar.set_ticklabels(
                [
                    f"{IC_MIN:g}",
                    f"{IC_YELLOW_ORANGE_BOUNDARY:g}",
                    f"{IC_ORANGE_RED_BOUNDARY:g}",
                    f"{IC_MAX:g}",
                ]
            )

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
            if SHOW_CPT_LOCATIONS:
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

            # Compute uncertainty using Monte Carlo Dropout if enabled
            if compute_uncertainty and uncertainty_folder is not None:
                try:
                    logger.info(
                        f"[{i:03d}/{len(section_files)}] Computing MC Dropout uncertainty ({n_mc_samples} samples)..."
                    )
                    trace(f"MC Dropout: Starting {n_mc_samples} forward passes")

                    logger.debug(f"Input shape for uncertainty: {cs_norm.shape}")

                    # Compute uncertainty (mean and std) using MC Dropout
                    pred_mean, pred_std = compute_mc_dropout_uncertainty(
                        model, cs_norm, n_samples=n_mc_samples
                    )

                    logger.debug(
                        f"MC Dropout output shapes - mean: {pred_mean.shape}, std: {pred_std.shape}"
                    )

                    # Denormalize mean prediction
                    pred_mean_denorm = reverse_IC_normalization(pred_mean)

                    # Ensure shapes are correct (32, 512)
                    if pred_std.shape != (N_ROWS, N_COLS):
                        logger.warning(
                            f"Uncertainty shape mismatch: got {pred_std.shape}, expected ({N_ROWS}, {N_COLS}). Reshaping..."
                        )
                        # If there's a channel dimension, remove it
                        if pred_std.ndim == 3:
                            pred_std = pred_std[:, :, 0]
                        if pred_mean_denorm.ndim == 3:
                            pred_mean_denorm = pred_mean_denorm[:, :, 0]

                    logger.debug(f"Final uncertainty map shape: {pred_std.shape}")

                    # Save uncertainty CSV (match GAN output naming)
                    uncertainty_csv = (
                        uncertainty_folder
                        / f"{section_file.stem}_seed{seed}_uncertainty.csv"
                    )
                    save_uncertainty_csv(pred_std, uncertainty_csv)
                    logger.info(f"[INFO] Saved uncertainty CSV: {uncertainty_csv.name}")
                    trace(f"Uncertainty CSV saved: {uncertainty_csv}")

                    # Save mean prediction CSV (for mean mosaic)
                    mean_csv = (
                        uncertainty_folder / f"{section_file.stem}_seed{seed}_mean.csv"
                    )
                    save_uncertainty_csv(pred_mean_denorm, mean_csv)
                    logger.info(f"[INFO] Saved mean prediction CSV: {mean_csv.name}")
                    trace(f"Mean prediction CSV saved: {mean_csv}")

                    # Create uncertainty visualization (match GAN output naming)
                    uncertainty_png = (
                        uncertainty_folder
                        / f"{section_file.stem}_seed{seed}_uncertainty.png"
                    )
                    visualize_uncertainty(
                        uncertainty_map=pred_std,
                        output_png=uncertainty_png,
                        x0=x0,
                        x1=x1,
                        y_top_m=y_top_m,
                        y_bottom_m=y_bottom_m,
                        cpt_positions=cpt_positions,
                        show_cpt_locations=SHOW_CPT_LOCATIONS,
                        mean_prediction=pred_mean_denorm,
                    )
                    logger.info(
                        f"[INFO] Created uncertainty PNG: {uncertainty_png.name}"
                    )
                    trace(f"Uncertainty PNG saved: {uncertainty_png}")

                except Exception as ue:
                    logger.warning(
                        f"[{i:03d}/{len(section_files)}] Failed to compute uncertainty: {ue}"
                    )
                    trace(f"Uncertainty computation failed: {ue}")

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
    mosaic_prefix: str = "schemaGAN",
    file_suffix: str = "gan",
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
        mosaic_prefix: Prefix for output files (e.g., \"original\", \"enhanced\").
        file_suffix: Suffix for input files (e.g., \"gan\", \"uncertainty\").
            Files will be matched as *_{file_suffix}.csv

    Raises:
        RuntimeError: If no valid GAN CSV files are found.

    Note:
        The mosaic uses bilinear interpolation to blend overlapping sections smoothly,
        both horizontally and vertically. If the manifest contains depth_window,
        depth_start_row, and depth_end_row columns, vertical stacking is applied
        based on original CPT data row indices. Otherwise, simple non-overlapping
        vertical stacking is used.

        Output files:
        - {mosaic_prefix}_mosaic.csv: Complete mosaic data matrix
        - {mosaic_prefix}_mosaic.png: Visualization with distance and depth axes
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
    gan_files = list(gan_images_folder.glob(f"*_{file_suffix}.csv"))
    if not gan_files:
        logger.warning(f"No {file_suffix} files found. Skipping mosaic creation.")
        return

    logger.info(f"Creating mosaic from {len(gan_files)} {file_suffix} files...")
    trace(f"{file_suffix} CSV count for mosaic: {len(gan_files)}")

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
    original_file_suffix = getattr(create_mosaic, "FILE_SUFFIX", "gan")

    create_mosaic.GAN_DIR = gan_images_folder
    create_mosaic.N_COLS = N_COLS
    create_mosaic.N_ROWS_WINDOW = N_ROWS
    create_mosaic.Y_TOP_M = y_top_m
    create_mosaic.Y_BOTTOM_M = y_bottom_m
    create_mosaic.FILE_SUFFIX = file_suffix

    # Add GAN file paths to manifest using the external module's function
    logger.info(f"Matching {file_suffix} CSV files to sections...")
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
    mosaic_csv = mosaic_folder / f"{mosaic_prefix}_mosaic.csv"
    pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)
    logger.info(f"Mosaic CSV saved: {mosaic_csv}")
    trace(f"Mosaic shape: {mosaic.shape}, saved to {mosaic_csv}")

    # Create and save mosaic visualization
    mosaic_png = mosaic_folder / f"{mosaic_prefix}_mosaic.png"
    logger.info(f"Creating mosaic visualization: {mosaic_png}")

    # Set visualization parameters based on data type
    if file_suffix == "uncertainty":
        # For uncertainty: auto-scale, use 'hot' colormap
        vmin_val, vmax_val = None, None
        cmap_val = "hot"
        colorbar_label = "Uncertainty (Std Dev)"
        ic_boundaries = None
    else:
        # For GAN/enhanced: custom IC scale with black for out-of-range
        cmap_val, vmin_val, vmax_val = create_custom_ic_colormap()
        colorbar_label = "IC Value"
        ic_boundaries = (
            IC_MIN,
            IC_YELLOW_ORANGE_BOUNDARY,
            IC_ORANGE_RED_BOUNDARY,
            IC_MAX,
        )

    try:
        plot_mosaic(
            mosaic,
            xmin,
            xmax,
            global_dx,
            n_rows_total,
            mosaic_png,
            coords=coords,
            show_cpt_locations=SHOW_CPT_LOCATIONS,
            vmin=vmin_val,
            vmax=vmax_val,
            cmap=cmap_val,
            colorbar_label=colorbar_label,
            ic_boundaries=ic_boundaries,
        )
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
    create_mosaic.FILE_SUFFIX = original_file_suffix

    logger.info(f"Mosaic complete: {mosaic_csv.name} & {mosaic_png.name}")
    logger.info("Step 6 complete.")
    trace(f"Mosaic creation finished: CSV and PNG saved")


def run_enhancement(
    gan_images_folder: Path,
    output_folder: Path,
    sections_folder: Path,
    y_top_m: float,
    y_bottom_m: float,
    method: str = "guided_filter",
):
    """Apply boundary enhancement to generated schemas and create visualizations.

    Processes all GAN-generated CSV files and applies the specified enhancement
    method to sharpen layer boundaries. Creates both CSV and PNG outputs.

    Args:
        gan_images_folder: Folder containing original GAN output CSVs
        output_folder: Folder where enhanced CSVs and PNGs will be saved
        sections_folder: Folder containing manifest and coordinates for PNG generation
        y_top_m: Top depth in meters for visualization
        y_bottom_m: Bottom depth in meters for visualization
        method: Enhancement method to use:
            - "guided_filter": Edge-preserving filter (recommended, best for GAN)
            - "unsharp_mask": Classic sharpening
            - "laplacian": Aggressive sharpening
            - "dense_crf": CRF-based (experimental)
            - "none": No enhancement

    Note:
        Only processes CSV files matching pattern: *_gan.csv
        Enhanced files are saved with same name in output_folder.
        PNGs are created with same visualization style as original GAN images.
    """
    from boundary_enhancement import enhance_schema_from_file, create_enhanced_png
    import pandas as pd

    logger.info(f"Enhancement method: {method}")
    logger.info(f"Input folder: {gan_images_folder}")
    logger.info(f"Output folder: {output_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Load manifest and coords for PNG generation
    manifest_csv = sections_folder / "manifest_sections.csv"
    coords_csv = sections_folder / "cpt_coords_with_distances.csv"

    if not manifest_csv.exists() or not coords_csv.exists():
        logger.warning(f"Manifest or coords CSV not found. PNGs will not be created.")
        create_pngs = False
    else:
        create_pngs = True

    # Find all GAN-generated CSV files
    gan_csv_files = sorted(gan_images_folder.glob("*_gan.csv"))

    if not gan_csv_files:
        logger.warning(f"No GAN CSV files found in {gan_images_folder}")
        return

    logger.info(f"Found {len(gan_csv_files)} GAN images to enhance")

    ok, fail = 0, 0

    for i, csv_file in enumerate(gan_csv_files, 1):
        try:
            # Output paths
            output_csv = output_folder / csv_file.name
            output_png = output_folder / csv_file.name.replace(".csv", ".png")

            # Apply enhancement to CSV
            enhanced_csv, method_used = enhance_schema_from_file(
                csv_file,
                output_csv,
                method=method,
            )

            # Create PNG visualization
            if create_pngs:
                try:
                    create_enhanced_png(
                        enhanced_csv,
                        output_png,
                        manifest_csv,
                        coords_csv,
                        y_top_m,
                        y_bottom_m,
                        show_cpt_locations=SHOW_CPT_LOCATIONS,
                    )
                except Exception as png_error:
                    logger.warning(
                        f"Failed to create PNG for {csv_file.name}: {png_error}"
                    )

            ok += 1
            if (i % 10 == 0) or (i == len(gan_csv_files)):
                logger.info(
                    f"[{i:03d}/{len(gan_csv_files)}] Enhanced: {csv_file.name} using {method_used}"
                )

        except Exception as e:
            fail += 1
            logger.error(f"[{i:03d}/{len(gan_csv_files)}] FAIL on {csv_file.name}: {e}")

    logger.info(f"Enhancement complete: {ok} succeeded, {fail} failed")
    logger.info(f"Enhanced schemas saved in: {output_folder}")
    logger.info("Step 6 complete.")


def main():
    """Execute the complete VOW SchemaGAN pipeline.

    Orchestrates all eight steps of the workflow:
    1. Creates experiment folder structure (1_coords, 2_compressed_cpt, 3_sections, 4_gan_images, 5_enhance, 6_mosaic, 7_uncertainty)
    2. Extracts and validates CPT coordinates from GEF files
    3. Processes CPT data and compresses to specified depth resolution
    4. Creates spatial sections with overlapping CPTs
    5. Generates schemas using SchemaGAN model (saved to 4_gan_images)
    6. Enhances schemas with boundary sharpening (if enabled, saved to 5_enhance)
    7. Assembles sections into mosaics (creates both original and enhanced mosaics)
    8. Computes prediction uncertainty using MC Dropout (if enabled, saved to 7_uncertainty)

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
    logger.info("Step 5: Generating schemas using SchemaGAN model (original output)...")
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
                uncertainty_folder=folders["7_uncertainty"],
                compute_uncertainty=COMPUTE_UNCERTAINTY,
                n_mc_samples=N_MC_SAMPLES,
            )
            logger.info("[DEBUG] Schema generation completed successfully")
        except Exception as e:
            logger.error(f"Failed to generate schemas: {e}")
            logger.error(f"[DEBUG] Schema generation exception: {e}")
            import traceback

            logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return

    # =============================================================================
    # 6. ENHANCE GENERATED SCHEMAS (BOUNDARY SHARPENING)
    # =============================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Enhancing generated schemas (boundary sharpening)...")
    logger.info(f"[DEBUG] Enhancement method: {ENHANCE_METHOD}")

    if ENHANCE_METHOD and ENHANCE_METHOD.lower() != "none":
        try:
            logger.info(f"[DEBUG] Applying {ENHANCE_METHOD} enhancement...")
            run_enhancement(
                folders["4_gan_images"],
                folders["5_enhance"],
                folders["3_sections"],
                y_top_final,
                y_bottom_final,
                ENHANCE_METHOD,
            )
            logger.info("[DEBUG] Enhancement completed successfully")
        except Exception as e:
            logger.error(f"Failed to enhance schemas: {e}")
            logger.error(f"[DEBUG] Enhancement exception: {e}")
            import traceback

            logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return
    else:
        logger.info("Enhancement disabled (ENHANCE_METHOD=None or 'none')")
        logger.info("Skipping enhancement step.")

    # =============================================================================
    # 7. CREATE MOSAICS FROM GENERATED SCHEMAS
    # =============================================================================
    logger.info("=" * 60)
    logger.info("Step 7: Creating mosaics from generated schemas...")
    logger.info("[DEBUG] About to call run_mosaic_creation function...")

    # Create mosaic from original GAN images
    try:
        logger.info(
            "[DEBUG] Creating mosaic from original GAN images (4_gan_images)..."
        )
        run_mosaic_creation(
            folders["3_sections"],
            folders["4_gan_images"],
            folders["6_mosaic"],
            y_top_final,
            y_bottom_final,
            mosaic_prefix="original",
        )
        logger.info("[DEBUG] Original mosaic creation completed successfully")
    except Exception as e:
        logger.error(f"Failed to create original mosaic: {e}")
        logger.error(f"[DEBUG] Mosaic creation exception: {e}")
        import traceback

        logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return

    # Create mosaic from enhanced images if enhancement was applied
    if ENHANCE_METHOD and ENHANCE_METHOD.lower() != "none":
        try:
            logger.info(f"[DEBUG] Creating mosaic from enhanced images (5_enhance)...")
            run_mosaic_creation(
                folders["3_sections"],
                folders["5_enhance"],
                folders["6_mosaic"],
                y_top_final,
                y_bottom_final,
                mosaic_prefix="enhanced",
            )
            logger.info("[DEBUG] Enhanced mosaic creation completed successfully")
        except Exception as e:
            logger.error(f"Failed to create enhanced mosaic: {e}")
            logger.error(f"[DEBUG] Enhanced mosaic creation exception: {e}")
            import traceback

            logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
            # Don't return here, original mosaic was already created
    else:
        logger.info("Skipping enhanced mosaic (no enhancement applied)")

    # =============================================================================
    # 8. CREATE UNCERTAINTY MOSAIC (IF UNCERTAINTY COMPUTED)
    # =============================================================================
    if COMPUTE_UNCERTAINTY:
        logger.info("=" * 60)
        logger.info("Step 8: Creating uncertainty mosaic from MC Dropout results...")
        logger.info("[DEBUG] About to create uncertainty mosaic...")

        try:
            # Collect all uncertainty CSV files
            uncertainty_files = sorted(
                folders["7_uncertainty"].glob("*_uncertainty.csv")
            )

            if len(uncertainty_files) == 0:
                logger.warning("No uncertainty files found. Skipping mosaic creation.")
            else:
                logger.info(f"[DEBUG] Found {len(uncertainty_files)} uncertainty files")

                # Create mosaic from uncertainty maps using same function as GAN/enhanced
                run_mosaic_creation(
                    folders["3_sections"],
                    folders["7_uncertainty"],
                    folders["7_uncertainty"],
                    y_top_final,
                    y_bottom_final,
                    mosaic_prefix="uncertainty",
                    file_suffix="uncertainty",  # Look for *_uncertainty.csv instead of *_gan.csv
                )
                logger.info(
                    "[DEBUG] Uncertainty mosaic creation completed successfully"
                )

                # Create mosaic from mean predictions
                logger.info("[DEBUG] Creating mean prediction mosaic...")
                run_mosaic_creation(
                    folders["3_sections"],
                    folders["7_uncertainty"],
                    folders["7_uncertainty"],
                    y_top_final,
                    y_bottom_final,
                    mosaic_prefix="mean",
                    file_suffix="mean",  # Look for *_mean.csv
                )
                logger.info("[DEBUG] Mean mosaic creation completed successfully")

                # Create combined visualization (mean + uncertainty in 2-row plot)
                logger.info(
                    "[DEBUG] Creating combined mean+uncertainty visualization..."
                )
                try:
                    from create_mosaic import load_inputs
                    import matplotlib.pyplot as plt
                    import pandas as pd
                    import numpy as np

                    # Load both mosaics
                    mean_mosaic_csv = folders["7_uncertainty"] / "mean_mosaic.csv"
                    uncertainty_mosaic_csv = (
                        folders["7_uncertainty"] / "uncertainty_mosaic.csv"
                    )

                    mean_mosaic = pd.read_csv(mean_mosaic_csv, header=None).values
                    uncertainty_mosaic = pd.read_csv(
                        uncertainty_mosaic_csv, header=None
                    ).values

                    # Load coords for CPT markers
                    coords_csv = folders["3_sections"] / "cpt_coords_with_distances.csv"
                    coords = pd.read_csv(coords_csv)

                    # Get mosaic extent
                    xmin = coords["cum_along_m"].min()
                    xmax = coords["cum_along_m"].max()
                    n_rows_total = mean_mosaic.shape[0]

                    # Create 2-row plot
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

                    # Top: Mean prediction
                    im1 = ax1.imshow(
                        mean_mosaic,
                        cmap="viridis",
                        vmin=0,
                        vmax=4.5,
                        aspect="auto",
                        extent=[xmin, xmax, n_rows_total - 1, 0],
                    )
                    plt.colorbar(im1, ax=ax1, label="IC (Mean Prediction)")
                    ax1.set_ylabel("Depth Index")
                    ax1.set_title("Mean MC Dropout Prediction Mosaic")

                    # Bottom: Uncertainty
                    im2 = ax2.imshow(
                        uncertainty_mosaic,
                        cmap="hot",
                        aspect="auto",
                        extent=[xmin, xmax, n_rows_total - 1, 0],
                    )
                    plt.colorbar(im2, ax=ax2, label="Uncertainty (Std Dev)")
                    ax2.set_ylabel("Depth Index")
                    ax2.set_xlabel("Distance along line (m)")
                    ax2.set_title("Prediction Uncertainty Mosaic")

                    # Add CPT markers to both plots
                    if SHOW_CPT_LOCATIONS:
                        for cpt_x in coords["cum_along_m"]:
                            ax1.axvline(
                                x=cpt_x,
                                color="black",
                                linewidth=1,
                                alpha=0.5,
                                zorder=10,
                            )
                            ax2.axvline(
                                x=cpt_x, color="cyan", linewidth=1, alpha=0.7, zorder=10
                            )

                    plt.tight_layout()
                    combined_png = (
                        folders["7_uncertainty"]
                        / "combined_mean_uncertainty_mosaic.png"
                    )
                    plt.savefig(combined_png, dpi=150, bbox_inches="tight")
                    plt.close()

                    logger.info(f"[DEBUG] Combined mosaic saved: {combined_png.name}")

                except Exception as ce:
                    logger.warning(f"Failed to create combined visualization: {ce}")

        except Exception as e:
            logger.error(f"Failed to create uncertainty mosaic: {e}")
            logger.error(f"[DEBUG] Uncertainty mosaic exception: {e}")
            import traceback

            logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
            # Don't return here, other results are still valid
    else:
        logger.info("Skipping uncertainty quantification (COMPUTE_UNCERTAINTY=False)")

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
    logger.info(f"  - Enhancement: {ENHANCE_METHOD if ENHANCE_METHOD else 'None'}")

    if ENHANCE_METHOD and ENHANCE_METHOD.lower() != "none":
        logger.info(
            f"  - Mosaics created: original (4_gan_images) + enhanced (5_enhance)"
        )
    else:
        logger.info(f"  - Mosaics created: original (4_gan_images) only")


if __name__ == "__main__":
    main()
