"""
Main script for the VOW SchemaGAN pipeline.

This script orchestrates the complete workflow:
1. Setup experiment folder structure
2. Extract coordinates from CPT files
3. Extract and compress CPT data to 32-pixel depth
4. Create sections for SchemaGAN input
5. Generate schemas using trained SchemaGAN model
6. Create mosaic from generated schemas

Configure the paths and parameters in the CONFIG section below.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

from utils import setup_experiment, read_files
from extract_coords import process_cpt_coords
from extract_data import (
    process_cpts,
    equalize_top,
    equalize_depth,
    compress_to_32px,
    save_cpt_to_csv,
)
from create_schGAN_input_file import process_sections, write_manifest
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Force reconfigure logging
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG - Modify these paths and parameters for your experiment
# =============================================================================

# Base configuration
RES_DIR = Path(r"C:\VOW\res")
REGION = "north"
EXP_NAME = "exp_1"
DESCRIPTION = (
    "Complete VOW SchemaGAN pipeline - coordinates extraction to mosaic generation"
)

# Input data paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts\betuwepand\dike_north_BRO")
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


def main():
    """Execute the complete VOW SchemaGAN pipeline."""

    print("DEBUG: Main function started")  # Debug print
    logger.info("=" * 60)
    logger.info("Starting VOW SchemaGAN Pipeline")
    logger.info("=" * 60)
    print("DEBUG: Logger initialized and working")  # Debug print

    # =============================================================================
    # 1. CREATE EXPERIMENT FOLDER STRUCTURE
    # =============================================================================
    logger.info("Step 1: Creating experiment folder structure...")

    folders = setup_experiment(
        base_dir=RES_DIR,
        region=REGION,
        exp_name=EXP_NAME,
        description=DESCRIPTION,
    )

    logger.info(f"Experiment folders created at: {folders['root']}")
    print(f"DEBUG: Folders created: {folders}")  # Debug print

    # =============================================================================
    # 2. EXTRACT COORDINATES FROM CPT FILES
    # =============================================================================
    logger.info("Step 2: Extracting coordinates from CPT files...")
    print("DEBUG: Starting coordinate extraction")  # Debug print

    coords_csv = folders["1_coords"] / "cpt_coordinates.csv"

    # Check if CPT folder exists
    if not CPT_FOLDER.exists():
        logger.error(f"CPT folder does not exist: {CPT_FOLDER}")
        print(f"ERROR: CPT folder not found at {CPT_FOLDER}")
        return

    print(f"DEBUG: CPT folder exists: {CPT_FOLDER}")
    print(f"DEBUG: Output coords CSV: {coords_csv}")

    try:
        logger.info(f"Processing CPT files from: {CPT_FOLDER}")
        process_cpt_coords(CPT_FOLDER, coords_csv)
        logger.info(f"Coordinates extracted to: {coords_csv}")
        print("DEBUG: Coordinate extraction completed successfully")
    except Exception as e:
        logger.error(f"Failed to extract coordinates: {e}")
        print(f"DEBUG ERROR in coordinate extraction: {e}")
        import traceback

        traceback.print_exc()
        return

    # =============================================================================
    # 3. EXTRACT AND COMPRESS CPT DATA
    # =============================================================================
    logger.info("Step 3: Extracting and compressing CPT data...")

    # Get valid CPT files (those not moved to no_coords)
    cpt_files = read_files(str(CPT_FOLDER), extension=".gef")

    try:
        # Process CPT files
        data_cpts, coords_simple = process_cpts(cpt_files)
        logger.info(f"Processed {len(data_cpts)} CPT files")

        # Create copy for processing
        original_data_cpts = [cpt.copy() for cpt in data_cpts]

        # Find depth limits
        lowest_max_depth = min(cpt["depth_max"] for cpt in data_cpts)
        lowest_min_depth = min(cpt["depth_min"] for cpt in data_cpts)

        logger.info(f"Depth range: {lowest_max_depth:.3f} to {lowest_min_depth:.3f} m")

        # Store depth range for later use
        global Y_TOP_M, Y_BOTTOM_M
        if Y_TOP_M is None:
            Y_TOP_M = lowest_max_depth
        if Y_BOTTOM_M is None:
            Y_BOTTOM_M = lowest_min_depth

        # Equalize depths
        equalized_top_cpts = equalize_top(original_data_cpts)
        equalized_depth_cpts = equalize_depth(equalized_top_cpts, lowest_min_depth)

        # Compress to 32 pixels
        compressed_cpts = compress_to_32px(
            equalized_depth_cpts, method=COMPRESSION_METHOD
        )

        # Save compressed data
        compressed_csv = (
            folders["2_compressed_cpt"]
            / f"compressed_cpt_data_{COMPRESSION_METHOD}.csv"
        )
        save_cpt_to_csv(
            compressed_cpts, str(folders["2_compressed_cpt"]), compressed_csv.name
        )

        logger.info(f"Compressed CPT data saved to: {compressed_csv}")

    except Exception as e:
        logger.error(f"Failed to process CPT data: {e}")
        return

    # =============================================================================
    # 4. CREATE SECTIONS FOR SCHEMAGAN INPUT
    # =============================================================================
    logger.info("Step 4: Creating sections for SchemaGAN input...")
    print("DEBUG: Starting section creation")  # Debug print

    try:
        # Load coordinates and CPT data
        print(f"DEBUG: Loading coords CSV: {coords_csv}")
        coords_df = pd.read_csv(coords_csv)
        print(f"DEBUG: Coords loaded, shape: {coords_df.shape}")

        print(f"DEBUG: Loading compressed CSV: {compressed_csv}")
        cpt_df = pd.read_csv(compressed_csv)
        print(f"DEBUG: CPT data loaded, shape: {cpt_df.shape}")

        # Validate inputs
        from create_schGAN_input_file import validate_input_files

        print("DEBUG: Validating input files...")
        validate_input_files(coords_df, cpt_df, N_ROWS)
        print("DEBUG: Input validation successful")

        # Process sections
        print("DEBUG: Starting process_sections...")
        manifest = process_sections(
            coords_df=coords_df,
            cpt_df=cpt_df,
            out_dir=folders["3_sections"],
            n_cols=N_COLS,
            n_rows=N_ROWS,
            per=CPTS_PER_SECTION,
            overlap=OVERLAP_CPTS,
            left_pad_frac=LEFT_PAD_FRACTION,
            right_pad_frac=RIGHT_PAD_FRACTION,
            from_where=DIR_FROM,
            to_where=DIR_TO,
        )
        print("DEBUG: process_sections completed")

        # Write manifest
        from create_schGAN_input_file import write_manifest

        print("DEBUG: Writing manifest...")
        write_manifest(manifest, folders["3_sections"])
        print("DEBUG: Manifest written successfully")

        logger.info(f"Created {len(manifest)} sections in: {folders['3_sections']}")

    except Exception as e:
        logger.error(f"Failed to create sections: {e}")
        print(f"DEBUG ERROR in section creation: {e}")
        import traceback

        traceback.print_exc()
        return

    # =============================================================================
    # 5. GENERATE SCHEMAS USING SCHEMAGAN MODEL
    # =============================================================================
    logger.info("Step 5: Generating schemas using SchemaGAN model...")
    print("DEBUG: Starting schema generation step")  # Debug print

    if not SCHGAN_MODEL_PATH.exists():
        logger.warning(f"SchemaGAN model not found at: {SCHGAN_MODEL_PATH}")
        logger.warning("Skipping schema generation. Please provide valid model path.")
        print(f"DEBUG: Model path does not exist: {SCHGAN_MODEL_PATH}")
        print("DEBUG: Skipping to step 6...")
    else:
        print(f"DEBUG: Model found at: {SCHGAN_MODEL_PATH}")
        try:
            # Import required modules for schema generation
            print("DEBUG: Setting up TensorFlow environment...")
            import os

            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from utils import IC_normalization, reverse_IC_normalization

            # Load model
            logger.info("Loading SchemaGAN model...")
            print("DEBUG: Loading TensorFlow model...")
            model = load_model(SCHGAN_MODEL_PATH, compile=False)
            print("DEBUG: Model loaded successfully")

            # Set random seed for reproducibility
            seed = np.random.randint(20220412, 20230412)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            logger.info(f"Using random seed: {seed}")

            # Find all section CSV files
            section_files = list(folders["3_sections"].glob("section_*_cpts_*.csv"))
            print(f"DEBUG: Found {len(section_files)} section files")

            if not section_files:
                logger.error("No section CSV files found")
                print("DEBUG: No section files found, returning")
                return

            logger.info(f"Generating schemas for {len(section_files)} sections...")

            success_count = 0
            fail_count = 0

            for i, section_file in enumerate(section_files, 1):
                try:
                    print(
                        f"DEBUG: Processing section {i}/{len(section_files)}: {section_file.name}"
                    )

                    # Read section data
                    df = pd.read_csv(section_file)
                    print(f"DEBUG: Section data shape: {df.shape}")

                    # Remove index column if it exists (first column is usually the index/depth)
                    if df.shape[1] == N_COLS + 1:  # 513 columns instead of 512
                        print("DEBUG: Removing first column (depth/index column)")
                        df_vals = df.iloc[:, 1:]  # Skip first column
                    else:
                        df_vals = df

                    print(f"DEBUG: Data shape after column adjustment: {df_vals.shape}")

                    # Convert to numpy array and reshape for GAN input
                    cs = df_vals.to_numpy(dtype=float).reshape(1, N_ROWS, N_COLS, 1)
                    print(f"DEBUG: Reshaped data for GAN: {cs.shape}")

                    # Normalize input
                    cs_norm = IC_normalization([cs, cs])[0]
                    print("DEBUG: Data normalized")

                    # Generate schema using GAN
                    print("DEBUG: Running GAN prediction...")
                    gan_result = model.predict(cs_norm, verbose=0)
                    print("DEBUG: GAN prediction completed")

                    # Reverse normalization
                    gan_result = reverse_IC_normalization(gan_result)
                    gan_result = np.squeeze(gan_result)  # Shape: (32, 512)
                    print(f"DEBUG: Final result shape: {gan_result.shape}")

                    # Save CSV - following create_schema.py approach
                    output_csv = (
                        folders["4_gan_images"]
                        / f"{section_file.stem}_seed{seed}_gan.csv"
                    )
                    pd.DataFrame(gan_result).to_csv(output_csv, index=False)
                    print(f"DEBUG: Saved CSV to: {output_csv}")

                    # Save PNG visualization - following create_schema.py approach exactly
                    import matplotlib.pyplot as plt

                    # Parse section index from filename
                    import re

                    m = re.search(r"section_(\d+)", section_file.stem)
                    sec_index = int(m.group(1)) if m else i

                    # Get section placement info from manifest for proper axes
                    manifest_df = pd.read_csv(
                        folders["3_sections"] / "manifest_sections.csv"
                    )
                    coords_df_dist = pd.read_csv(
                        folders["3_sections"] / "cpt_coords_with_distances.csv"
                    )

                    r = manifest_df.loc[manifest_df["section_index"] == sec_index]
                    if not r.empty:
                        r = r.iloc[0]
                        total_span = float(
                            r["span_m"] + r["left_pad_m"] + r["right_pad_m"]
                        )
                        start_idx = int(r["start_idx"])
                        m0 = float(coords_df_dist.loc[start_idx, "cum_along_m"])
                        x0 = m0 - float(r["left_pad_m"])
                        dx = 1.0 if total_span <= 0 else total_span / (N_COLS - 1)
                        x1 = x0 + (N_COLS - 1) * dx
                    else:
                        # Fallback if manifest lookup fails
                        x0, x1, dx = 0, N_COLS - 1, 1

                    output_png = (
                        folders["4_gan_images"]
                        / f"{section_file.stem}_seed{seed}_gan.png"
                    )
                    plt.figure(figsize=(10, 2.4))

                    # Bottom axis: meters (following create_schema.py exactly)
                    plt.imshow(
                        gan_result,
                        cmap="viridis",
                        vmin=0,
                        vmax=4.5,
                        aspect="auto",
                        extent=[x0, x1, N_ROWS - 1, 0],  # x in meters, y inverted
                    )
                    plt.colorbar(label="Value")

                    ax = plt.gca()
                    ax.set_xlabel("Distance along line (m)")
                    ax.set_ylabel("Depth Index")

                    # Top x-axis: pixel index (following create_schema.py)
                    def m_to_px(x):
                        return (x - x0) / dx if dx != 0 else 0

                    def px_to_m(p):
                        return x0 + p * dx

                    top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
                    top.set_xlabel(f"Pixel index (0…{N_COLS-1})")

                    # Right y-axis: real depth (following create_schema.py)
                    def idx_to_meters(y_idx):
                        return Y_TOP_M + (y_idx / (N_ROWS - 1)) * (Y_BOTTOM_M - Y_TOP_M)

                    def meters_to_idx(y_m):
                        denom = Y_BOTTOM_M - Y_TOP_M
                        return (
                            0.0
                            if abs(denom) < 1e-12
                            else (y_m - Y_TOP_M) * (N_ROWS - 1) / denom
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
                    print(f"DEBUG: Saved PNG to: {output_png}")

                    logger.info(
                        f"[{i:03d}/{len(section_files)}] Generated schema: {output_csv.name} & {output_png.name}"
                    )
                    success_count += 1

                except Exception as e:
                    logger.error(
                        f"[{i:03d}/{len(section_files)}] Failed on {section_file.name}: {e}"
                    )
                    print(f"DEBUG: Error processing {section_file.name}: {e}")
                    fail_count += 1

            logger.info(
                f"Schema generation complete. Success: {success_count}, Failed: {fail_count}"
            )
            print(
                f"DEBUG: Schema generation finished. Success: {success_count}, Failed: {fail_count}"
            )

        except Exception as e:
            logger.error(f"Failed to generate schemas: {e}")
            print(f"DEBUG: Exception in schema generation: {e}")
            import traceback

            traceback.print_exc()
            return

    # =============================================================================
    # 6. CREATE MOSAIC FROM GENERATED SCHEMAS
    # =============================================================================
    logger.info("Step 6: Creating mosaic from generated schemas...")
    print("DEBUG: Starting mosaic creation step")  # Debug print

    try:
        # Import mosaic creation functionality (following create_mosaic.py)
        import matplotlib.pyplot as plt
        import numpy as np

        # Check if we have generated schemas
        gan_files = list(folders["4_gan_images"].glob("*_gan.csv"))
        print(f"DEBUG: Found {len(gan_files)} GAN files in {folders['4_gan_images']}")

        if not gan_files:
            logger.warning("No generated schema files found. Skipping mosaic creation.")
            logger.info(f"Schema files should be in: {folders['4_gan_images']}")
            print("DEBUG: No GAN files found, creating summary anyway")
        else:
            logger.info(f"Creating mosaic from {len(gan_files)} generated schemas...")
            print(f"DEBUG: Processing {len(gan_files)} schema files for mosaic")

            # Load manifest and coordinates (following create_mosaic.py structure)
            manifest_csv = folders["3_sections"] / "manifest_sections.csv"
            coords_with_dist_csv = (
                folders["3_sections"] / "cpt_coords_with_distances.csv"
            )
            print(f"DEBUG: Checking for manifest: {manifest_csv.exists()}")
            print(f"DEBUG: Checking for coords: {coords_with_dist_csv.exists()}")

            if manifest_csv.exists() and coords_with_dist_csv.exists():
                # Load data following create_mosaic.py approach
                manifest_df = pd.read_csv(manifest_csv)
                coords_df = pd.read_csv(coords_with_dist_csv)

                print(f"DEBUG: Loaded manifest with {len(manifest_df)} sections")
                print(f"DEBUG: Loaded coordinates with {len(coords_df)} CPTs")

                # Helper functions from create_mosaic.py
                def find_latest_gan_csv(section_index):
                    pattern = f"section_{section_index:02d}_cpts_*_gan.csv"
                    candidates = list(folders["4_gan_images"].glob(pattern))
                    if not candidates:
                        return None
                    return candidates[0]  # Just take the first match

                def compute_section_placement(row, coords):
                    total_span = float(
                        row["span_m"] + row["left_pad_m"] + row["right_pad_m"]
                    )
                    if total_span <= 0:
                        raise ValueError(
                            f"Invalid total span for section {row['section_index']}"
                        )
                    start_idx = int(row["start_idx"])
                    m0 = float(coords.loc[start_idx, "cum_along_m"])
                    x0 = m0 - float(row["left_pad_m"])
                    dx = total_span / (N_COLS - 1)
                    x1 = x0 + (N_COLS - 1) * dx
                    return x0, dx, x1

                # Following create_mosaic.py build_mosaic function
                manifest_work = manifest_df.copy()
                manifest_work["gan_csv"] = manifest_work["section_index"].apply(
                    find_latest_gan_csv
                )

                # Drop sections without GAN output
                missing = manifest_work[manifest_work["gan_csv"].isna()]
                if not missing.empty:
                    print(
                        f"[WARN] Missing GAN csv for sections: {missing['section_index'].tolist()}"
                    )
                manifest_work = manifest_work.dropna(subset=["gan_csv"]).reset_index(
                    drop=True
                )

                if manifest_work.empty:
                    logger.error("No sections with GAN CSVs found.")
                    return

                # Compute placement for each section
                x0_list, dx_list, x1_list = [], [], []
                for _, row in manifest_work.iterrows():
                    x0, dx, x1 = compute_section_placement(row, coords_df)
                    x0_list.append(x0)
                    dx_list.append(dx)
                    x1_list.append(x1)

                manifest_work["x0"] = x0_list
                manifest_work["dx"] = dx_list
                manifest_work["x1"] = x1_list

                # Define global grid (following create_mosaic.py)
                xmin = float(manifest_work["x0"].min())
                xmax = float(manifest_work["x1"].max())
                global_dx = float(np.median(manifest_work["dx"]))  # Use median dx
                width = int(round((xmax - xmin) / global_dx)) + 1

                print(
                    f"[INFO] Global extent: {xmin:.2f}..{xmax:.2f} m ({xmax - xmin:.2f} m), dx={global_dx:.4f} m/px, width={width} px"
                )

                # Prepare accumulator arrays (following create_mosaic.py)
                acc = np.zeros((N_ROWS, width))
                wts = np.zeros(width)

                # Add each section to the global mosaic (following create_mosaic.py interpolation)
                for _, row in manifest_work.iterrows():
                    gan_csv_path = row["gan_csv"]
                    x0 = float(row["x0"])
                    dx = float(row["dx"])

                    # Load section data
                    arr = pd.read_csv(gan_csv_path).to_numpy(dtype=float)
                    if arr.shape != (N_ROWS, N_COLS):
                        print(
                            f"[WARN] {gan_csv_path.name}: expected shape ({N_ROWS},{N_COLS}), got {arr.shape}"
                        )
                        continue

                    # Compute global positions (following create_mosaic.py interpolation)
                    xj = x0 + np.arange(N_COLS) * dx
                    pos = (xj - xmin) / global_dx
                    k0 = np.floor(pos).astype(int)
                    frac = pos - k0
                    k1 = k0 + 1

                    # Keep columns that fall inside mosaic width
                    valid = (k0 >= 0) & (k0 < width)
                    if not np.any(valid):
                        continue

                    k0v = k0[valid]
                    f0 = 1.0 - frac[valid]
                    k1v = k0v + 1
                    f1 = frac[valid]

                    # Add weighted contributions
                    acc[:, k0v] += arr[:, valid] * f0
                    wts[k0v] += f0

                    in_range = k1v < width
                    if np.any(in_range):
                        acc[:, k1v[in_range]] += (
                            arr[:, valid][:, in_range] * f1[in_range]
                        )
                        wts[k1v[in_range]] += f1[in_range]

                # Weighted average (following create_mosaic.py)
                eps = 1e-12
                mosaic = acc / np.maximum(wts, eps)[None, :]

                # Save mosaic CSV (following create_mosaic.py)
                mosaic_csv = folders["5_mosaic"] / "schemaGAN_mosaic.csv"
                pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)
                print(f"DEBUG: Saved mosaic CSV: {mosaic_csv}")

                # Create mosaic visualization (following create_mosaic.py plot_mosaic exactly)
                mosaic_png = folders["5_mosaic"] / "schemaGAN_mosaic.png"

                horiz_m = xmax - xmin
                vert_m = abs(Y_BOTTOM_M - Y_TOP_M)
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

                # Top x-axis: pixel index (following create_mosaic.py)
                def m_to_px(x):
                    return (x - xmin) / global_dx

                def px_to_m(p):
                    return xmin + p * global_dx

                top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
                top.set_xlabel("Pixel index (0…W-1)")

                # Right y-axis: real depths (following create_mosaic.py)
                def idx_to_m(y_idx):
                    return Y_TOP_M + (y_idx / (N_ROWS - 1)) * (Y_BOTTOM_M - Y_TOP_M)

                def m_to_idx(y_m):
                    denom = Y_BOTTOM_M - Y_TOP_M
                    return (
                        0
                        if abs(denom) < 1e-12
                        else (y_m - Y_TOP_M) * (N_ROWS - 1) / denom
                    )

                right = ax.secondary_yaxis("right", functions=(idx_to_m, m_to_idx))
                right.set_ylabel("Depth (m)")

                plt.title("SchemaGAN Mosaic")
                plt.tight_layout()
                plt.savefig(mosaic_png, dpi=500)
                plt.close()
                print(f"DEBUG: Saved mosaic PNG: {mosaic_png}")

                logger.info(
                    f"[DONE] Mosaic saved:\n  CSV → {mosaic_csv}\n  PNG → {mosaic_png}"
                )

            else:
                logger.warning(
                    "Required manifest or coordinates files not found for mosaic creation"
                )
                print("DEBUG: Missing required files for mosaic")

        # Also create the summary file
        mosaic_summary = folders["5_mosaic"] / "mosaic_summary.txt"
        print(f"DEBUG: Creating summary at: {mosaic_summary}")

        with open(mosaic_summary, "w") as f:
            f.write(f"Mosaic Summary\n")
            f.write(f"==============\n")
            f.write(f"Generated schemas: {len(gan_files)}\n")
            f.write(f"Schema files location: {folders['4_gan_images']}\n")
            f.write(f"Mosaic files location: {folders['5_mosaic']}\n")
            if gan_files:
                f.write(
                    f"Processing used create_mosaic.py approach with interpolation\n"
                )
            f.write(f"\nGenerated schema files:\n")
            for gan_file in sorted(gan_files):
                f.write(f"  - {gan_file.name}\n")

        print("DEBUG: Mosaic summary created successfully")

    except Exception as e:
        logger.error(f"Failed to create mosaic: {e}")
        print(f"DEBUG: Exception in mosaic creation: {e}")
        import traceback

        traceback.print_exc()
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
        f"  - CPTs processed: {len(data_cpts) if 'data_cpts' in locals() else 'N/A'}"
    )
    logger.info(
        f"  - Sections created: {len(manifest) if 'manifest' in locals() else 'N/A'}"
    )
    logger.info(f"  - Depth range: {Y_TOP_M:.3f} to {Y_BOTTOM_M:.3f} m")
    logger.info(f"  - Grid size: {N_ROWS} × {N_COLS} pixels")


if __name__ == "__main__":
    main()
