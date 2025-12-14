"""
Main script for the VOW SchemaGAN pipeline.

This script orchestrates the complete workflow:
1. Setup experiment folder structure
2. Extract coordinates from CPT files
3. Extract and compress CPT data
4. Create sections for SchemaGAN input
5. Generate schemas using trained SchemaGAN model
6. Enhance schemas with boundary sharpening (optional)
7. Create mosaics from generated schemas
8. Compute prediction uncertainty using MC Dropout (optional)

All configuration is centralized in config.py.
"""

import sys
import json
import logging
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src directory to path for core and modules packages
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
import config

# Add GEOLib-Plus path
sys.path.append(config.GEOLIB_PLUS_PATH)

# Import module functions
from core.utils import setup_experiment
from modules.visualization import create_custom_ic_colormap, trace
from modules.preprocessing.coordinate_extraction import run_coordinate_extraction
from modules.preprocessing.cpt_processing import run_cpt_data_processing
from modules.preprocessing.section_creation import run_section_creation
from modules.generation.schema_generation import run_schema_generation
from modules.postprocessing.enhancement import run_enhancement
from modules.postprocessing.mosaic_creation import run_mosaic_creation

# Configure logging
logger = None


def setup_logging(experiment_folder: Path, log_level: str = "INFO"):
    """Configure logging for the pipeline.

    Args:
        experiment_folder: Path to experiment folder for log file.
        log_level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    global logger

    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Set up file logging
    try:
        log_file = experiment_folder / "pipeline.log"
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file.resolve()}")
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}")


def main():
    """Execute the complete VOW SchemaGAN pipeline."""
    global logger

    # =============================================================================
    # 1. CREATE EXPERIMENT FOLDER STRUCTURE
    # =============================================================================
    print("=" * 60)
    print("Starting VOW SchemaGAN Pipeline")
    print("=" * 60)

    folders = setup_experiment(
        base_dir=config.RES_DIR,
        region=config.REGION,
        exp_name=config.EXP_NAME,
        description=config.DESCRIPTION,
    )

    # Configure logging
    setup_logging(folders["root"], config.LOG_LEVEL)

    logger.info("=" * 60)
    logger.info("VOW SchemaGAN Pipeline - Starting")
    logger.info("=" * 60)
    logger.info(f"Experiment folders created at: {folders['root']}")

    # =============================================================================
    # 2. EXTRACT COORDINATES FROM CPT FILES
    # =============================================================================
    logger.info("Step 2: Extracting coordinates from CPT files...")

    coords_csv = folders["1_coords"] / "cpt_coordinates.csv"

    try:
        run_coordinate_extraction(config.CPT_FOLDER, coords_csv)
    except Exception as e:
        logger.error(f"Failed to extract coordinates: {e}")
        return

    # =============================================================================
    # 3. EXTRACT AND COMPRESS CPT DATA
    # =============================================================================
    # Initialize depth variables
    y_top_final = config.Y_TOP_M
    y_bottom_final = config.Y_BOTTOM_M

    if config.RUN_STEP_2_PREPARE_CPTS:
        logger.info("Step 2: Extracting and compressing CPT data...")

        try:
            compressed_csv, y_top, y_bottom = run_cpt_data_processing(
                config.CPT_FOLDER,
                folders["2_compressed_cpt"],
                config.COMPRESSION_METHOD,
                config.CPT_DEPTH_PIXELS,
            )

            # Use computed values if not set in config
            if y_top_final is None:
                y_top_final = y_top
            if y_bottom_final is None:
                y_bottom_final = y_bottom

            logger.info(f"Depth range: {y_top_final:.3f} to {y_bottom_final:.3f} m")

            # Save depth range
            depth_range_file = folders["root"] / "depth_range.json"
            with open(depth_range_file, "w") as f:
                json.dump(
                    {
                        "y_top_m": float(y_top_final),
                        "y_bottom_m": float(y_bottom_final),
                    },
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error(f"Failed to process CPT data: {e}")
            return
    else:
        logger.info("=" * 60)
        logger.info("Step 2: Skipped (RUN_STEP_2_PREPARE_CPTS = False)")
        compressed_csv = folders["2_compressed_cpt"] / "cpt_data_compressed.csv"
        # Load depth range from previous run if available
        depth_range_file = folders["root"] / "depth_range.json"
        if depth_range_file.exists():
            with open(depth_range_file, "r") as f:
                depth_data = json.load(f)
                y_top_final = depth_data.get("y_top_m", y_top_final)
                y_bottom_final = depth_data.get("y_bottom_m", y_bottom_final)

    # =============================================================================
    # 4. CREATE SECTIONS FOR SCHEMAGAN INPUT
    # =============================================================================
    if config.RUN_STEP_3_CREATE_SECTIONS:
        logger.info("=" * 60)
        logger.info("Step 3: Creating sections for SchemaGAN input...")

        try:
            manifest = run_section_creation(
                coords_csv,
                compressed_csv,
                folders["3_sections"],
                config.N_COLS,
                config.N_ROWS,
                config.CPTS_PER_SECTION,
                config.OVERLAP_CPTS,
                config.LEFT_PAD_FRACTION,
                config.RIGHT_PAD_FRACTION,
                config.DIR_FROM,
                config.DIR_TO,
                config.VERTICAL_OVERLAP,
            )
        except Exception as e:
            logger.error(f"Failed to create sections: {e}")
            return
    else:
        logger.info("=" * 60)
        logger.info("Step 3: Skipped (RUN_STEP_3_CREATE_SECTIONS = False)")

    # =============================================================================
    # 5. GENERATE SCHEMAS USING SCHEMAGAN MODEL
    # =============================================================================
    if config.RUN_STEP_4_CREATE_GAN_IMAGES:
        logger.info("=" * 60)
        logger.info("Step 4: Generating schemas using SchemaGAN model...")

        if not config.SCHGAN_MODEL_PATH.exists():
            logger.warning(f"SchemaGAN model not found at: {config.SCHGAN_MODEL_PATH}")
            logger.warning("Skipping schema generation.")
        else:
            try:
                # Prepare IC boundaries tuple
                ic_boundaries = (
                    config.IC_MIN,
                    config.IC_SAND_SANDMIX_BOUNDARY,
                    config.IC_SANDMIX_SILTMIX_BOUNDARY,
                    config.IC_SILTMIX_CLAY_BOUNDARY,
                    config.IC_CLAY_ORGANIC_BOUNDARY,
                    config.IC_MAX,
                )

                # Create trace function wrapper
                def trace_wrapper(msg, level=logging.INFO):
                    trace(msg, level, config.VERBOSE)

                run_schema_generation(
                    folders["3_sections"],
                    folders["4_gan_images"],
                    config.SCHGAN_MODEL_PATH,
                    y_top_final,
                    y_bottom_final,
                    config.N_ROWS,
                    config.N_COLS,
                    ic_boundaries,
                    show_cpt_locations=config.SHOW_CPT_LOCATIONS,
                    create_colormap_func=create_custom_ic_colormap,
                    trace_func=trace_wrapper,
                    uncertainty_folder=folders["7_model_uncert"],
                    compute_uncertainty=config.COMPUTE_UNCERTAINTY,
                    n_mc_samples=config.N_MC_SAMPLES,
                )
            except Exception as e:
                logger.error(f"Failed to generate schemas: {e}")
                return
    else:
        logger.info("=" * 60)
        logger.info("Step 4: Skipped (RUN_STEP_4_CREATE_GAN_IMAGES = False)")

    # =============================================================================
    # 6. ENHANCE GENERATED SCHEMAS (OPTIONAL)
    # =============================================================================
    if (
        config.RUN_STEP_5_ENHANCE
        and config.ENHANCE_METHOD
        and config.ENHANCE_METHOD.lower() != "none"
    ):
        logger.info("=" * 60)
        logger.info("Step 5: Enhancing generated schemas...")
        try:
            run_enhancement(
                folders["4_gan_images"],
                folders["5_enhance"],
                folders["3_sections"],
                y_top_final,
                y_bottom_final,
                config.SHOW_CPT_LOCATIONS,
                config.ENHANCE_METHOD,
            )
        except Exception as e:
            logger.error(f"Failed to enhance schemas: {e}")
            return
    else:
        logger.info("=" * 60)
        if not config.RUN_STEP_5_ENHANCE:
            logger.info("Step 5: Skipped (RUN_STEP_5_ENHANCE = False)")
        else:
            logger.info(
                "Step 5: Enhancement disabled in config (ENHANCE_METHOD = None)"
            )

    # =============================================================================
    # 7. CREATE MOSAICS FROM GENERATED SCHEMAS
    # =============================================================================
    if config.RUN_STEP_6_CREATE_MOSAIC:
        logger.info("=" * 60)
        logger.info("Step 6: Creating mosaics...")

    # Prepare IC boundaries
    ic_boundaries = (
        config.IC_MIN,
        config.IC_SAND_SANDMIX_BOUNDARY,
        config.IC_SANDMIX_SILTMIX_BOUNDARY,
        config.IC_SILTMIX_CLAY_BOUNDARY,
        config.IC_CLAY_ORGANIC_BOUNDARY,
        config.IC_MAX,
    )

    # Original mosaic
    try:
        run_mosaic_creation(
            folders["3_sections"],
            folders["4_gan_images"],
            folders["6_mosaic"],
            y_top_final,
            y_bottom_final,
            config.N_COLS,
            config.N_ROWS,
            config.SHOW_CPT_LOCATIONS,
            create_custom_ic_colormap,
            ic_boundaries,
            mosaic_prefix="original",
        )
    except Exception as e:
        logger.error(f"Failed to create original mosaic: {e}")
        return

    # Enhanced mosaic (if applicable)
    if config.ENHANCE_METHOD and config.ENHANCE_METHOD.lower() != "none":
        try:
            run_mosaic_creation(
                folders["3_sections"],
                folders["5_enhance"],
                folders["6_mosaic"],
                y_top_final,
                y_bottom_final,
                config.N_COLS,
                config.N_ROWS,
                config.SHOW_CPT_LOCATIONS,
                create_custom_ic_colormap,
                ic_boundaries,
                mosaic_prefix="enhanced",
            )
        except Exception as e:
            logger.error(f"Failed to create enhanced mosaic: {e}")
    else:
        logger.info("=" * 60)
        logger.info("Step 6: Skipped (RUN_STEP_6_CREATE_MOSAIC = False)")

    # =============================================================================
    # 8. CREATE UNCERTAINTY MOSAICS (OPTIONAL)
    # =============================================================================
    if config.RUN_STEP_7_MODEL_UNCERTAINTY and config.COMPUTE_UNCERTAINTY:
        logger.info("=" * 60)
        logger.info("Step 8: Creating uncertainty mosaics...")

        try:
            # Check if uncertainty files exist
            uncertainty_files = list(folders["7_model_uncert"].glob("*_uncertainty.csv"))

            if len(uncertainty_files) == 0:
                logger.warning("No uncertainty files found.")
            else:
                # Uncertainty mosaic
                run_mosaic_creation(
                    folders["3_sections"],
                    folders["7_model_uncert"],
                    folders["7_model_uncert"],
                    y_top_final,
                    y_bottom_final,
                    config.N_COLS,
                    config.N_ROWS,
                    config.SHOW_CPT_LOCATIONS,
                    None,  # No custom colormap for uncertainty
                    None,  # No IC boundaries for uncertainty
                    mosaic_prefix="uncertainty",
                    file_suffix="uncertainty",
                )

                # Mean prediction mosaic
                run_mosaic_creation(
                    folders["3_sections"],
                    folders["7_model_uncert"],
                    folders["7_model_uncert"],
                    y_top_final,
                    y_bottom_final,
                    config.N_COLS,
                    config.N_ROWS,
                    config.SHOW_CPT_LOCATIONS,
                    create_custom_ic_colormap,
                    ic_boundaries,
                    mosaic_prefix="mean",
                    file_suffix="mean",
                )

        except Exception as e:
            logger.error(f"Failed to create uncertainty mosaics: {e}")
    else:
        logger.info("=" * 60)
        if not config.RUN_STEP_7_MODEL_UNCERTAINTY:
            logger.info("Step 7: Skipped (RUN_STEP_7_MODEL_UNCERTAINTY = False)")
        else:
            logger.info(
                "Step 7: Uncertainty quantification disabled in config (COMPUTE_UNCERTAINTY = False)"
            )

    # =============================================================================
    # COMPLETION
    # =============================================================================
    logger.info("=" * 60)
    logger.info("VOW SchemaGAN Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved in: {folders['root']}")
    logger.info("\nFolder structure:")
    for name, path in folders.items():
        if name != "root":
            logger.info(f"  {name}: {path}")

    # Summary statistics
    logger.info("\nPipeline Summary:")
    logger.info(f"  - Sections created: {len(manifest)}")
    logger.info(f"  - Depth range: {y_top_final:.3f} to {y_bottom_final:.3f} m")
    logger.info(f"  - Grid size: {config.N_ROWS} × {config.N_COLS} pixels")
    logger.info(
        f"  - Enhancement: {config.ENHANCE_METHOD if config.ENHANCE_METHOD else 'None'}"
    )
    logger.info(f"  - Uncertainty quantification: {config.COMPUTE_UNCERTAINTY}")


if __name__ == "__main__":
    main()
