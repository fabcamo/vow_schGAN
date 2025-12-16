"""
CPT data processing module.

This module handles processing, interpretation, and compression of CPT data.
"""

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def run_cpt_data_processing(
    cpt_folder: Path,
    output_folder: Path,
    compression_method: str = "mean",
    cpt_depth_pixels: int = 32,
) -> Tuple[Path, float, float]:
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

    Returns:
        Tuple of (output_path, lowest_max_depth, lowest_min_depth)
            - output_path: Path to saved compressed CSV file
            - lowest_max_depth: Shallowest starting depth across all CPTs (meters)
            - lowest_min_depth: Deepest ending depth across all CPTs (meters)

    Note:
        Output CSV has columns: Depth_Index (0 to cpt_depth_pixels-1), CPT1, CPT2, ...
        Each CPT column contains IC values at corresponding depth levels.
    """
    from core.extract_data import (
        process_cpts,
        equalize_top,
        fill_top_with_zeros,
        equalize_depth,
        compress_cpt_data,
        save_cpt_to_csv,
    )
    from core.utils import read_files

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
    #equalized_top_cpts = fill_top_with_zeros(original_data_cpts)
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
