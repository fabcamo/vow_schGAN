"""
Configuration file for VOW SchemaGAN pipeline.

This file centralizes all configuration parameters for the pipeline.
Modify these values to customize your experiment without touching the main code.
"""

from pathlib import Path
from typing import Optional

# =============================================================================
# PIPELINE STEPS CONTROL
# =============================================================================
# Enable or disable each step of the pipeline
# Set to False to skip steps that have already been completed

RUN_STEP_1_GET_COORDS = True  # Extract coordinates from CPT files
RUN_STEP_2_PREPARE_CPTS = True  # Process and compress CPT data
RUN_STEP_3_CREATE_SECTIONS = True  # Create sections for GAN input
RUN_STEP_4_CREATE_GAN_IMAGES = True  # Generate schemas with GAN
RUN_STEP_5_ENHANCE = True  # Boundary enhancement (if method selected)
RUN_STEP_6_CREATE_MOSAIC = True  # Create mosaic from schemas
RUN_STEP_7_MODEL_UNCERTAINTY = True  # Compute uncertainty (if enabled)

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Base paths
BASE_PATH = Path(r"C:\VOW")  # Base path for experiments
GEOLIB_PLUS_PATH = r"D:\GEOLib-Plus"  # Path to GEOLib-Plus library

# Output directories
RES_DIR = Path(BASE_PATH / "res")  # Base results directory

# Input data paths
CPT_FOLDER = Path(
    BASE_PATH / "data" / "cpts" / "betuwepand" / "dike_south_BRO"
)  # Folder with .gef CPT files
# CPT_FOLDER = Path(r"C:\VOW\data\cpts\waalbandijk")  # For quick testing with fewer CPTs

SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")  # Trained SchemaGAN model


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

REGION = "south"  # Region name for experiment folder and the data subfolder
EXP_NAME = "exp_18"
DESCRIPTION = (
    "added interactive html plots"
    "new color scale for IC visualization,"
    "CPT compression to 64,"
    "2 CPT overlap,"
    "30% vertical overlap,"
    "10% padding,"
    "No boundary enhancement,"
    "Added uncertainty quantification with 10 samples,"
)


# =============================================================================
# CPT DATA PROCESSING PARAMETERS
# =============================================================================

COMPRESSION_METHOD = "mean"  # "mean" or "max" for IC value compression

# CPT Data Compression - can be different from model input size
CPT_DEPTH_PIXELS = 64  # Number of depth levels to compress raw CPT data to
# Can be 32, 64, 128, etc. independent of N_ROWS
# Higher values preserve more detail from raw CPT data


# =============================================================================
# MODEL INPUT DIMENSIONS
# =============================================================================

# Model Input Dimensions - MUST match what SchemaGAN expects
N_COLS = 512  # Number of columns in sections (SchemaGAN expects 512)
N_ROWS = 32  # Number of rows in sections (SchemaGAN expects 32)
# If CPT_DEPTH_PIXELS != N_ROWS, resampling will occur in section creation


# =============================================================================
# SECTION CREATION PARAMETERS
# =============================================================================

CPTS_PER_SECTION = 6  # Number of CPTs per section
OVERLAP_CPTS = 2  # Number of overlapping CPTs between sections (horizontal)

# Vertical windowing parameters
VERTICAL_OVERLAP = 50  # [%] Vertical overlap between depth windows (0.0 = no overlap, 50.0 = 50% overlap)

# Padding strategy: Use percentage of section span
LEFT_PAD_FRACTION = 0.1  # Left padding as fraction of section span (10%)
RIGHT_PAD_FRACTION = 0.1  # Right padding as fraction of section span (10%)
# Note: Padding is calculated per section based on its span, so:
#   - Small section (100m) → 10m padding on each side
#   - Large section (300m) → 30m padding on each side
# This keeps padding proportional to section size

DIR_FROM, DIR_TO = "west", "east"  # Sorting direction


# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

SHOW_CPT_LOCATIONS = True  # Show vertical lines at CPT positions in plots

# Font size for all plot elements (labels, titles, legends, tick labels)
PLOT_FONT_SIZE = 8

# Optional: Real depth range for visualization (will be computed if None)
Y_TOP_M: Optional[float] = None
Y_BOTTOM_M: Optional[float] = None


# =============================================================================
# COLOR SCALE CONFIGURATION
# =============================================================================
# Five-category color scale for IC values:
# Category 1 (Sand - Gold):       IC_MIN to IC_SAND_SANDMIX_BOUNDARY (Ic ≤ 2.05)
# Category 2 (Sand mix - Orange): IC_SAND_SANDMIX_BOUNDARY to IC_SANDMIX_SILTMIX_BOUNDARY (2.05 < Ic ≤ 2.6)
# Category 3 (Silt mix - Light Blue): IC_SANDMIX_SILTMIX_BOUNDARY to IC_SILTMIX_CLAY_BOUNDARY (2.6 < Ic ≤ 2.95)
# Category 4 (Clay - Green):      IC_SILTMIX_CLAY_BOUNDARY to IC_CLAY_ORGANIC_BOUNDARY (2.95 < Ic ≤ 3.6)
# Category 5 (Organic - Red):     IC_CLAY_ORGANIC_BOUNDARY to IC_MAX (Ic > 3.6)
# Everything outside IC_MIN to IC_MAX will be black

IC_MIN = 1.0  # Minimum IC value (start of sand)
IC_SAND_SANDMIX_BOUNDARY = 2.05  # Boundary between sand and sand mix
IC_SANDMIX_SILTMIX_BOUNDARY = 2.6  # Boundary between sand mix and silt mix
IC_SILTMIX_CLAY_BOUNDARY = 2.95  # Boundary between silt mix and clay
IC_CLAY_ORGANIC_BOUNDARY = 3.6  # Boundary between clay and organic
IC_MAX = 4.5  # Maximum IC value (end of organic)


# =============================================================================
# BOUNDARY ENHANCEMENT PARAMETERS
# =============================================================================

ENHANCE_METHOD = "none"  # Enhancement method to sharpen layer boundaries
# Options:
#   "guided_filter": Edge-preserving guided filter (RECOMMENDED - best for GAN outputs, no halos)
#   "unsharp_mask": Classic sharpening via unsharp masking (can add artifacts)
#   "laplacian": Laplacian-based sharpening (aggressive, creates artifacts)
#   "dense_crf": Dense CRF edge-aware smoothing (experimental, ineffective)
#   "none" or None: No enhancement (use original GAN output)


# =============================================================================
# UNCERTAINTY QUANTIFICATION PARAMETERS
# =============================================================================

COMPUTE_UNCERTAINTY = True  # Compute prediction uncertainty using MC Dropout
N_MC_SAMPLES = (
    5  # Number of MC Dropout samples (20-100 typical, more = slower but more accurate)
)
# MC Dropout reveals where the GAN is uncertain in its predictions:
#   - High uncertainty: complex transitions, far from data, ambiguous interpolations
#   - Low uncertainty: near CPT locations, homogeneous layers, clear patterns


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = "INFO"  # Set to "DEBUG" for more detail
VERBOSE = (
    True  # Default True to show verbose internal progress (set False to reduce output)
)
