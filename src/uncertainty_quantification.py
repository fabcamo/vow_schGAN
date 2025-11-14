"""
Uncertainty Quantification for SchemaGAN outputs using Monte Carlo Dropout.

This module computes prediction uncertainty by running the GAN multiple times
with dropout enabled (Monte Carlo Dropout). The variance in predictions
indicates model uncertainty - areas where the GAN is less confident in its
interpolation.

High uncertainty typically appears:
- Far from CPT data points
- In complex geological transitions
- Where the model struggles to interpolate

Low uncertainty appears:
- At CPT locations (grounded in real data via skip connections)
- In homogeneous layers
- Where patterns are clear and consistent
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_mc_dropout_uncertainty(
    model,
    section_input: np.ndarray,
    n_samples: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute uncertainty using Monte Carlo Dropout.

    Runs the GAN model multiple times with dropout enabled to estimate
    prediction uncertainty. The model must have dropout layers that are
    active during inference (training=True).

    Args:
        model: Loaded Keras/TensorFlow GAN model with dropout layers
        section_input: Input section array, shape (1, H, W, C) or (H, W, C)
        n_samples: Number of MC samples to generate (20-100 typical)
            - More samples = more accurate uncertainty estimate
            - More samples = slower computation
            - Recommended: 30-50

    Returns:
        (mean_prediction, uncertainty_map) tuple
        - mean_prediction: Average of all MC samples (best estimate)
        - uncertainty_map: Standard deviation across samples (uncertainty)

    Example:
        >>> model = load_model("schemaGAN.h5")
        >>> section = load_section("section_001.csv")
        >>> mean, uncertainty = compute_mc_dropout_uncertainty(model, section, n_samples=50)
    """
    # Ensure input has batch dimension
    if section_input.ndim == 3:
        section_input = np.expand_dims(section_input, axis=0)

    logger.info(f"Running MC Dropout with {n_samples} samples...")

    # Collect predictions
    predictions = []
    for i in range(n_samples):
        # Run model with training=True to keep dropout active
        # Note: model.predict() vs model() - both work if training=True is baked in
        pred = model.predict(section_input, verbose=0)
        predictions.append(pred)

        if (i + 1) % 10 == 0:
            logger.debug(f"  MC sample {i + 1}/{n_samples}")

    # Stack predictions: (n_samples, batch, H, W, C)
    predictions = np.array(predictions)

    # Compute statistics across MC samples
    mean_prediction = np.mean(predictions, axis=0)  # (batch, H, W, C)
    std_prediction = np.std(predictions, axis=0)  # (batch, H, W, C)

    # Remove batch dimension
    mean_prediction = np.squeeze(mean_prediction)  # (H, W) or (H, W, C)
    std_prediction = np.squeeze(std_prediction)  # (H, W) or (H, W, C)

    # If single channel, ensure 2D
    if mean_prediction.ndim == 3 and mean_prediction.shape[-1] == 1:
        mean_prediction = mean_prediction[:, :, 0]
        std_prediction = std_prediction[:, :, 0]

    logger.info(
        f"MC Dropout complete. Uncertainty range: [{std_prediction.min():.4f}, {std_prediction.max():.4f}]"
    )

    return mean_prediction, std_prediction


def visualize_uncertainty(
    uncertainty_map: np.ndarray,
    output_png: Path,
    x0: float,
    x1: float,
    y_top_m: float,
    y_bottom_m: float,
    cpt_positions: Optional[np.ndarray] = None,
    show_cpt_locations: bool = True,
    title: Optional[str] = None,
    mean_prediction: Optional[np.ndarray] = None,
):
    """
    Create visualization of uncertainty map, optionally with mean prediction.

    Generates a heatmap showing model uncertainty (standard deviation from
    MC Dropout). If mean_prediction is provided, creates a 2-row plot showing
    both the mean prediction and uncertainty.

    Args:
        uncertainty_map: 2D array (H, W) with uncertainty values (std deviation)
        output_png: Path for output PNG file
        x0: Left x-coordinate in meters
        x1: Right x-coordinate in meters
        y_top_m: Top depth in meters
        y_bottom_m: Bottom depth in meters
        cpt_positions: Array of CPT x-positions in meters (optional)
        show_cpt_locations: Whether to show CPT position markers
        title: Plot title (optional)
        mean_prediction: 2D array (H, W) with mean MC prediction (optional)
            If provided, creates 2-row plot with mean on top, uncertainty below
    """
    SIZE_Y, SIZE_X = uncertainty_map.shape

    # Create 2-row plot if mean_prediction provided, otherwise single plot
    if mean_prediction is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
        axes = [ax1, ax2]

        # Top plot: Mean prediction
        im1 = ax1.imshow(
            mean_prediction,
            cmap="viridis",
            vmin=0,
            vmax=4.5,
            aspect="auto",
            extent=[x0, x1, SIZE_Y - 1, 0],
            interpolation="bilinear",
        )
        cbar1 = plt.colorbar(im1, label="IC (Mean Prediction)", ax=ax1)
        ax1.set_ylabel("Depth Index")
        ax1.set_title("Mean Prediction (50 MC samples)")

        # Bottom plot: Uncertainty
        im2 = ax2.imshow(
            uncertainty_map,
            cmap="hot",
            aspect="auto",
            extent=[x0, x1, SIZE_Y - 1, 0],
            interpolation="bilinear",
        )
        cbar2 = plt.colorbar(im2, label="Uncertainty (Std Dev)", ax=ax2)
        ax2.set_ylabel("Depth Index")
        ax2.set_xlabel("Distance along line (m)")
        ax2.set_title("Prediction Uncertainty")

        # Use bottom axis for shared settings
        ax_main = ax2
    else:
        # Single plot: uncertainty only
        fig, ax_main = plt.subplots(figsize=(10, 2.4))
        axes = [ax_main]

        im = ax_main.imshow(
            uncertainty_map,
            cmap="hot",
            aspect="auto",
            extent=[x0, x1, SIZE_Y - 1, 0],
            interpolation="bilinear",
        )
        cbar = plt.colorbar(im, label="Uncertainty (Std Dev)", ax=ax_main)
        ax_main.set_xlabel("Distance along line (m)")
        ax_main.set_ylabel("Depth Index")

    # Top x-axis: pixel indices (on topmost axis)
    dx = (x1 - x0) / (SIZE_X - 1)

    def m_to_px(x):
        return (x - x0) / dx

    def px_to_m(p):
        return x0 + p * dx

    top = axes[0].secondary_xaxis("top", functions=(m_to_px, px_to_m))
    top.set_xlabel(f"Pixel index (0…{SIZE_X-1})")

    # Right y-axis: real depth (on all axes)
    def idx_to_meters(y_idx):
        return y_top_m + (y_idx / (SIZE_Y - 1)) * (y_bottom_m - y_top_m)

    def meters_to_idx(y_m):
        denom = y_bottom_m - y_top_m
        return 0.0 if abs(denom) < 1e-12 else (y_m - y_top_m) * (SIZE_Y - 1) / denom

    for ax in axes:
        right = ax.secondary_yaxis("right", functions=(idx_to_meters, meters_to_idx))
        right.set_ylabel("Depth (m)")

    # Add CPT position markers to all axes
    if show_cpt_locations and cpt_positions is not None:
        for ax in axes:
            for cpt_x in cpt_positions:
                ax.axvline(
                    x=cpt_x,
                    color="cyan",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                    zorder=10,
                )

    # Set x-limits on all axes
    for ax in axes:
        ax.set_xlim(x0, x1)

    # Overall title (only if single plot and title provided)
    if mean_prediction is None and title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved uncertainty visualization: {output_png.name}")


def save_uncertainty_csv(
    uncertainty_map: np.ndarray,
    output_csv: Path,
):
    """
    Save uncertainty map as CSV file with header row to match GAN output format.

    Args:
        uncertainty_map: 2D array (H, W) with uncertainty values
        output_csv: Path for output CSV file
    """
    # Save with header=True to include column numbers (0, 1, 2, ...)
    # This matches the GAN output format which has 33 rows (1 header + 32 data)
    pd.DataFrame(uncertainty_map).to_csv(output_csv, index=False, header=True)
    logger.debug(f"Saved uncertainty CSV: {output_csv.name}")


def create_uncertainty_mosaic(
    uncertainty_folder: Path,
    sections_folder: Path,
    output_folder: Path,
    y_top_m: float,
    y_bottom_m: float,
    coords_csv: Path,
    show_cpt_locations: bool = True,
):
    """
    Create mosaic from individual uncertainty maps.

    Assembles uncertainty maps using the same logic as schema mosaics,
    blending overlapping sections.

    Args:
        uncertainty_folder: Folder containing uncertainty CSV files
        sections_folder: Folder with manifest and coordinates
        output_folder: Folder for mosaic output
        y_top_m: Top depth in meters
        y_bottom_m: Bottom depth in meters
        coords_csv: Path to coordinates CSV
        show_cpt_locations: Whether to show CPT markers on mosaic
    """
    from create_mosaic import (
        load_inputs,
        build_mosaic,
        choose_global_grid,
    )

    logger.info("Creating uncertainty mosaic...")

    # Load manifest and coords
    manifest_csv = sections_folder / "manifest_sections.csv"
    manifest = pd.read_csv(manifest_csv)
    coords = pd.read_csv(coords_csv)

    # Temporarily update create_mosaic module constants
    import create_mosaic

    original_gan_dir = create_mosaic.GAN_DIR
    original_n_cols = create_mosaic.N_COLS
    original_n_rows = create_mosaic.N_ROWS_WINDOW
    original_y_top = create_mosaic.Y_TOP_M
    original_y_bottom = create_mosaic.Y_BOTTOM_M

    try:
        # Set module constants for uncertainty data
        create_mosaic.GAN_DIR = uncertainty_folder
        create_mosaic.N_COLS = 512
        create_mosaic.N_ROWS_WINDOW = 32
        create_mosaic.Y_TOP_M = y_top_m
        create_mosaic.Y_BOTTOM_M = y_bottom_m

        # Build mosaic using uncertainty CSVs (named *_uncertainty.csv)
        # Need to temporarily rename or handle file matching
        logger.info("Building uncertainty mosaic from individual maps...")

        # The mosaic builder expects *_gan.csv files
        # We need to match uncertainty files to sections
        mosaic, xmin, xmax, global_dx, n_rows_total = build_mosaic(manifest, coords)

        logger.info(f"Uncertainty mosaic built: shape={mosaic.shape}")

    finally:
        # Restore original values
        create_mosaic.GAN_DIR = original_gan_dir
        create_mosaic.N_COLS = original_n_cols
        create_mosaic.N_ROWS_WINDOW = original_n_rows
        create_mosaic.Y_TOP_M = original_y_top
        create_mosaic.Y_BOTTOM_M = original_y_bottom

    # Save mosaic CSV
    mosaic_csv = output_folder / "uncertainty_mosaic.csv"
    pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)
    logger.info(f"Uncertainty mosaic CSV saved: {mosaic_csv}")

    # Visualize mosaic
    mosaic_png = output_folder / "uncertainty_mosaic.png"

    SIZE_Y, SIZE_X = mosaic.shape
    fig, ax = plt.subplots(figsize=(16, 4))

    im = ax.imshow(
        mosaic,
        cmap="hot",
        aspect="auto",
        extent=[xmin, xmax, n_rows_total - 1, 0],
        interpolation="bilinear",
    )

    plt.colorbar(im, label="Uncertainty (Std Dev)", ax=ax)

    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Depth Index (global)")

    # Top x-axis
    def m_to_px(x):
        return (x - xmin) / global_dx

    def px_to_m(p):
        return xmin + p * global_dx

    top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
    top.set_xlabel("Pixel index (0…W-1)")

    # Right y-axis
    def idx_to_m(y_idx):
        return y_top_m + (y_idx / (n_rows_total - 1)) * (y_bottom_m - y_top_m)

    def m_to_idx(y_m):
        denom = y_bottom_m - y_top_m
        return 0 if abs(denom) < 1e-12 else (y_m - y_top_m) * (n_rows_total - 1) / denom

    right = ax.secondary_yaxis("right", functions=(idx_to_m, m_to_idx))
    right.set_ylabel("Depth (m)")

    # Add CPT markers
    if show_cpt_locations and "cum_along_m" in coords.columns:
        for cpt_x in coords["cum_along_m"]:
            ax.axvline(
                x=cpt_x, color="cyan", linewidth=1, linestyle="--", alpha=0.6, zorder=10
            )

    plt.title("Uncertainty Mosaic (MC Dropout)")
    plt.tight_layout()
    plt.savefig(mosaic_png, dpi=500)
    plt.close()

    logger.info(f"Uncertainty mosaic PNG saved: {mosaic_png}")

    return mosaic_csv, mosaic_png


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("Uncertainty Quantification Module")
    print("\nThis module provides Monte Carlo Dropout uncertainty estimation")
    print("for SchemaGAN predictions.")
    print("\nKey functions:")
    print("  - compute_mc_dropout_uncertainty(): Run MC Dropout on a section")
    print("  - visualize_uncertainty(): Create uncertainty heatmap")
    print("  - create_uncertainty_mosaic(): Assemble mosaic from uncertainty maps")
