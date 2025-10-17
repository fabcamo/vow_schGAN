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
    """Emit a verbose message (defaults INFO) if VERBOSE is enabled."""
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
REGION = "south"
EXP_NAME = "exp_1"
DESCRIPTION = "Baseline with 6 CPTs and 2 overlapping"

# Input data paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts\betuwepand\dike_south_BRO")
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")

# Processing parameters
COMPRESSION_METHOD = "mean"  # "mean" or "max" for IC value compression
N_COLS = 512  # Number of columns in output sections
N_ROWS = 32  # Number of rows (depth levels) in output sections
CPTS_PER_SECTION = 6  # Number of CPTs per section
OVERLAP_CPTS = 2  # Number of overlapping CPTs between sections
LEFT_PAD_FRACTION = 0.10  # Left padding as fraction of section span
RIGHT_PAD_FRACTION = 0.10  # Right padding as fraction of section span
DIR_FROM, DIR_TO = "west", "east"  # Sorting direction

# Optional: Real depth range for visualization (will be computed if None)
Y_TOP_M: Optional[float] = None
Y_BOTTOM_M: Optional[float] = None

# =============================================================================


def run_coordinate_extraction(cpt_folder: Path, coords_csv: Path):
    """Wrapper for extract_coords.py functionality."""
    from extract_coords import process_cpt_coords

    if not cpt_folder.exists():
        raise FileNotFoundError(f"CPT folder does not exist: {cpt_folder}")

    process_cpt_coords(cpt_folder, coords_csv)
    logger.info(f"Coordinates extracted to: {coords_csv}")
    logger.info("Step 2 complete.")


def run_cpt_data_processing(
    cpt_folder: Path, output_folder: Path, compression_method: str = "mean"
):
    """Wrapper for extract_data.py functionality."""
    from extract_data import (
        process_cpts,
        equalize_top,
        equalize_depth,
        compress_to_32px,
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
    compressed_cpts = compress_to_32px(equalized_depth_cpts, method=compression_method)

    # Save results
    output_filename = f"compressed_cpt_data_{compression_method}.csv"
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
):
    """Wrapper for create_schGAN_input_file.py functionality."""
    from create_schGAN_input_file import (
        process_sections,
        write_manifest,
        validate_input_files,
    )
    import pandas as pd

    # Load data
    coords_df = pd.read_csv(coords_csv)
    cpt_df = pd.read_csv(cpt_csv)

    # Validate and process
    validate_input_files(coords_df, cpt_df, n_rows)

    manifest = process_sections(
        coords_df=coords_df,
        cpt_df=cpt_df,
        out_dir=output_folder,
        n_cols=n_cols,
        n_rows=n_rows,
        per=cpts_per_section,
        overlap=overlap,
        left_pad_frac=left_pad,
        right_pad_frac=right_pad,
        from_where=dir_from,
        to_where=dir_to,
    )

    write_manifest(manifest, output_folder)
    logger.info(f"Created {len(manifest)} sections in: {output_folder}")
    logger.info("Step 4 complete.")

    return manifest


def run_schema_generation(
    sections_folder: Path,
    gan_images_folder: Path,
    model_path: Path,
    y_top_m: float,
    y_bottom_m: float,
):
    """Generate schemas using SchemaGAN model - reimplemented to avoid import issues."""

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

    # Constants
    SIZE_X, SIZE_Y = 512, 32

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
                m0 = float(coords_df.loc[start_idx, "cum_along_m"])
                x0 = m0 - float(r["left_pad_m"])
                dx = 1.0 if total_span <= 0 else total_span / (SIZE_X - 1)
                x1 = x0 + (SIZE_X - 1) * dx
            else:
                x0, x1, dx = 0, SIZE_X - 1, 1

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

            plt.title(
                f"SchemaGAN Generated Image (Section {sec_index:03d}, Seed: {seed})"
            )
            plt.tight_layout()
            plt.savefig(output_png, dpi=220)
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
    """Create mosaic from generated schemas - reimplemented to avoid import issues."""

    # Ensure output directory exists
    mosaic_folder.mkdir(parents=True, exist_ok=True)
    trace(f"Mosaic output folder ready: {mosaic_folder}")

    # Check if we have generated schemas
    gan_files = list(gan_images_folder.glob("*_gan.csv"))
    if not gan_files:
        logger.warning("No generated schema files found. Skipping mosaic creation.")
        return

    logger.info(f"Creating mosaic from {len(gan_files)} generated schemas...")
    trace(f"Initial GAN CSV count for mosaic: {len(gan_files)}")

    # Import required modules for mosaic creation
    logger.info("=" * 60)
    logger.info("MOSAIC CREATION STARTING")
    logger.info("=" * 60)
    logger.info("[INFO] Ensuring mosaic output directory exists: %s", mosaic_folder)
    mosaic_folder.mkdir(parents=True, exist_ok=True)

    gan_files = list(gan_images_folder.glob("*_gan.csv"))
    logger.info(
        "[INFO] Found %d GAN CSV files in %s", len(gan_files), gan_images_folder
    )
    trace(f"Proceeding with {len(gan_files)} GAN CSV files for mosaic")

    # List all GAN files found for debugging
    for i, gan_file in enumerate(gan_files):
        logger.info(f"[INFO] GAN file {i+1}: {gan_file.name}")

    if not gan_files:
        logger.warning("No generated schema files found. Skipping mosaic creation.")
        logger.warning(f"Checked directory: {gan_images_folder}")
        logger.warning("Expected files matching pattern: *_gan.csv")
        return

    logger.info(f"[INFO] Creating mosaic from {len(gan_files)} generated schemas...")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import re

    N_COLS = 512
    N_ROWS = 32
    GLOBAL_DX = None
    TOP_AXIS_0_TO_32 = False

    manifest_csv = sections_folder / "manifest_sections.csv"
    coords_csv = sections_folder / "cpt_coords_with_distances.csv"
    logger.info("[INFO] Loading manifest from: %s", manifest_csv)
    logger.info("[INFO] Loading coordinates from: %s", coords_csv)
    manifest = pd.read_csv(manifest_csv)
    coords = pd.read_csv(coords_csv)

    def find_latest_gan_csv(section_index):
        # Correct pattern: filenames created as section_01_cpts_XXX_to_YYY_seedNNNN_gan.csv (02d padding)
        pattern = f"section_{section_index:02d}_cpts_*_gan.csv"
        matches = list(gan_images_folder.glob(pattern))
        trace(
            f"Section {section_index:02d} filename pattern '{pattern}' matches={len(matches)}"
        )
        if matches:
            logger.info(
                f"[INFO] Section {section_index:02d}: Using GAN file: {matches[-1].name}"
            )
            return matches[-1]
        else:
            logger.warning(
                f"[WARN] Section {section_index:02d}: No GAN CSV files found for pattern '{pattern}'"
            )
            return None

    def compute_section_placement(row, coords):
        total_span = float(row["span_m"] + row["left_pad_m"] + row["right_pad_m"])
        start_idx = int(row["start_idx"])
        m0 = float(coords.loc[start_idx, "cum_along_m"])
        x0 = m0 - float(row["left_pad_m"])
        dx = 1.0 if total_span <= 0 else total_span / (N_COLS - 1)
        x1 = x0 + (N_COLS - 1) * dx
        logger.info(
            f"[INFO] Section {row['section_index']}: x0={x0:.2f}, dx={dx:.4f}, x1={x1:.2f}"
        )
        return x0, dx, x1

    def choose_global_grid(manifest):
        xmin = float(manifest["x0"].min())
        xmax = float(manifest["x1"].max())
        dx = GLOBAL_DX if GLOBAL_DX is not None else float(np.median(manifest["dx"]))
        width = int(round((xmax - xmin) / dx)) + 1
        logger.info(
            f"[INFO] Global grid: xmin={xmin:.2f}, xmax={xmax:.2f}, dx={dx:.4f}, width={width}"
        )
        return xmin, xmax, dx, width

    def add_section_to_accumulator(acc, wts, section_csv, x0, dx, xmin, global_dx):
        logger.info(f"[INFO] Adding section to accumulator: {section_csv}")
        arr = pd.read_csv(section_csv).to_numpy(dtype=float)
        if arr.shape != (N_ROWS, N_COLS):
            logger.error(
                f"[ERROR] {section_csv.name}: expected shape (32,512), got {arr.shape}"
            )
            raise ValueError(
                f"{section_csv.name}: expected shape (32,512), got {arr.shape}"
            )

        xj = x0 + np.arange(N_COLS) * dx
        pos = (xj - xmin) / global_dx
        k0 = np.floor(pos).astype(int)
        frac = pos - k0
        k1 = k0 + 1

        valid = (k0 >= 0) & (k0 < wts.size)
        if not np.any(valid):
            logger.warning(
                f"[WARN] No valid columns in section {section_csv.name} for mosaic."
            )
            return

        k0v = k0[valid]
        f0 = 1.0 - frac[valid]
        k1v = k0v + 1
        f1 = frac[valid]

        acc[:, k0v] += arr[:, valid] * f0
        wts[k0v] += f0

        in_range = k1v < wts.size
        if np.any(in_range):
            acc[:, k1v[in_range]] += arr[:, valid][:, in_range] * f1[in_range]
            wts[k1v[in_range]] += f1[in_range]

    def build_mosaic(manifest, coords):
        logger.info("[INFO] Building mosaic from sections...")
        trace(f"Mosaic build: initial manifest size={len(manifest)}")
        manifest = manifest.copy()
        manifest["gan_csv"] = manifest["section_index"].apply(find_latest_gan_csv)
        trace(
            f"Mosaic build: gan_csv assignment nulls={manifest['gan_csv'].isna().sum()}"
        )

        missing = manifest[manifest["gan_csv"].isna()]
        if not missing.empty:
            logger.warning(
                f"[WARN] Missing GAN csv for sections: {missing['section_index'].tolist()}"
            )
            trace(
                f"Mosaic build: missing GAN CSV for sections {missing['section_index'].tolist()}"
            )
        manifest = manifest.dropna(subset=["gan_csv"]).reset_index(drop=True)
        trace(f"Mosaic build: manifest size after drop={len(manifest)}")
        if manifest.empty:
            logger.error("[ERROR] No sections with GAN CSVs found.")
            trace(
                "Mosaic build abort: empty manifest after dropping missing sections",
                level=logging.ERROR,
            )
            raise RuntimeError("No sections with GAN CSVs found.")

        x0_list, dx_list, x1_list = [], [], []
        for _, row in manifest.iterrows():
            x0, dx, x1 = compute_section_placement(row, coords)
            x0_list.append(x0)
            dx_list.append(dx)
            x1_list.append(x1)

        manifest["x0"] = x0_list
        manifest["dx"] = dx_list
        manifest["x1"] = x1_list

        xmin, xmax, global_dx, width = choose_global_grid(manifest)

        acc = np.zeros((N_ROWS, width))
        wts = np.zeros(width)

        for _, row in manifest.iterrows():
            logger.info(f"[INFO] Adding section {row['section_index']} to mosaic.")
            trace(
                f"Mosaic accumulation: section {row['section_index']} x0={row['x0']:.3f} dx={row['dx']:.4f}"
            )
            add_section_to_accumulator(
                acc,
                wts,
                row["gan_csv"],
                float(row["x0"]),
                float(row["dx"]),
                xmin,
                global_dx,
            )
            col_wt_nonzero = int((wts > 0).sum())
            trace(
                f"Mosaic accumulation: after section {row['section_index']} nonzero weighted cols={col_wt_nonzero}"
            )

        eps = 1e-12
        mosaic = acc / np.maximum(wts, eps)[None, :]
        logger.info("[INFO] Mosaic built successfully.")
        trace(
            f"Mosaic built: shape={mosaic.shape} xmin={xmin:.2f} xmax={xmax:.2f} dx={global_dx:.4f}"
        )
        return mosaic, xmin, xmax, global_dx

    def plot_mosaic(mosaic, xmin, xmax, global_dx, out_png):
        logger.info(f"[INFO] Plotting mosaic to PNG: {out_png}")
        horiz_m = xmax - xmin
        vert_m = abs(y_bottom_m - y_top_m)
        base_width = 16
        height = np.clip(base_width * (vert_m / max(horiz_m, 1e-12)), 2, 12)
        fig, ax = plt.subplots(figsize=(base_width, height))
        im = ax.imshow(
            mosaic,
            cmap="viridis",
            vmin=0,
            vmax=4.5,
            aspect="auto",
            extent=[xmin, xmax, N_ROWS - 1, 0],
        )
        plt.colorbar(im, label="Value")
        ax.set_xlabel("Distance along line (m)")
        ax.set_ylabel("Depth Index")

        def m_to_px(x):
            return (x - xmin) / global_dx

        def px_to_m(p):
            return xmin + p * global_dx

        top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
        top.set_xlabel("Pixel index")

        def idx_to_meters(y_idx):
            return y_top_m + (y_idx / (N_ROWS - 1)) * (y_bottom_m - y_top_m)

        def meters_to_idx(y_m):
            denom = y_bottom_m - y_top_m
            return 0.0 if abs(denom) < 1e-12 else (y_m - y_top_m) * (N_ROWS - 1) / denom

        right = ax.secondary_yaxis("right", functions=(idx_to_meters, meters_to_idx))
        right.set_ylabel("Depth (m)")
        plt.title("SchemaGAN Mosaic")
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()
        logger.info(f"[INFO] Mosaic PNG saved: {out_png}")

    try:
        logger.info("[INFO] Starting mosaic creation...")
        mosaic, xmin, xmax, global_dx = build_mosaic(manifest, coords)
        mosaic_csv = mosaic_folder / "schemaGAN_mosaic.csv"
        pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)
        logger.info(f"[INFO] Mosaic CSV saved: {mosaic_csv}")
        trace(f"Mosaic CSV saved at {mosaic_csv}")
        mosaic_png = mosaic_folder / "schemaGAN_mosaic.png"
        plot_mosaic(mosaic, xmin, xmax, global_dx, mosaic_png)
        logger.info(f"[INFO] Mosaic saved: {mosaic_csv.name} & {mosaic_png.name}")
        trace(f"Mosaic PNG saved at {mosaic_png}")
        logger.info("Step 6 complete.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to create mosaic: {e}")
        raise


def main():
    """Execute the complete VOW SchemaGAN pipeline."""

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

        # Also add console handler to ensure we see output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

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
            CPT_FOLDER, folders["2_compressed_cpt"], COMPRESSION_METHOD
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
