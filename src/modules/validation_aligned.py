"""
Validation pipeline for SchemaGAN with spatially-aligned sections.

This version ensures that the validation mosaic matches the original mosaic in physical extent and column count, by dynamically generating sections with 6 CPTs and variable overlap to cover the full extent.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from modules.visualization import create_custom_ic_colormap
from core.utils import IC_normalization, reverse_IC_normalization
from tensorflow.keras.models import load_model
import config
import tempfile
import shutil
import matplotlib.pyplot as plt
from core.create_schGAN_input_file import (
    map_dist_to_cols,
    resolve_collisions,
    build_section_matrix,
)


# 1. Load original manifest and mosaic CSV to get extent and column count
def get_original_extent_and_columns(folders):
    manifest_path = Path(folders["3_sections"]) / "manifest_sections.csv"
    mosaic_csv_path = Path(folders["6_mosaic"]) / "original_mosaic.csv"
    manifest = pd.read_csv(manifest_path)
    coords_path = Path(folders["1_coords"]) / "cpt_coordinates.csv"
    coords = pd.read_csv(coords_path)
    if "cum_along_m" not in coords.columns:
        coords = compute_cum_along_m(coords)
    first_cpt = manifest.iloc[0]["first_name"]
    last_cpt = manifest.iloc[-1]["last_name"]
    first_pos = coords.loc[coords["name"] == first_cpt, "cum_along_m"].values[0]
    last_pos = coords.loc[coords["name"] == last_cpt, "cum_along_m"].values[0]
    total_extent = last_pos - first_pos
    mosaic = pd.read_csv(mosaic_csv_path)
    n_cols = mosaic.shape[1]
    return total_extent, n_cols, first_pos, last_pos


# 2. Generate validation sections with 6 CPTs and variable overlap to cover full extent
def generate_validation_sections(
    coords, removed_cpts, n_cpts_per_section, total_extent, n_cols, first_pos, last_pos
):
    available = coords[~coords["name"].isin(removed_cpts)].copy()
    available = available.sort_values("cum_along_m").reset_index(drop=True)
    meters_per_col = total_extent / (n_cols - 1)
    section_width = meters_per_col * 512  # Each section is 512 columns wide
    # Plan section start positions so that sections tile the mosaic exactly
    section_starts = [
        first_pos + i * (512 - 64) * meters_per_col
        for i in range((n_cols - 1) // (512 - 64) + 1)
    ]
    sections = []
    for start_m in section_starts:
        end_m = start_m + section_width
        # For this window, select the 6 CPTs closest to 6 evenly spaced positions in [start_m, end_m]
        ideal_positions = np.linspace(start_m, end_m, n_cpts_per_section)
        cpt_indices = []
        for pos in ideal_positions:
            idx = np.abs(available["cum_along_m"] - pos).argmin()
            cpt_indices.append(idx)
        # Ensure uniqueness and preserve order
        cpt_indices = sorted(set(cpt_indices), key=lambda x: x)
        # If not enough unique CPTs, fill from neighbors
        while len(cpt_indices) < n_cpts_per_section:
            for idx in range(len(available)):
                if idx not in cpt_indices:
                    cpt_indices.append(idx)
                if len(cpt_indices) == n_cpts_per_section:
                    break
        cpt_indices = sorted(cpt_indices)
        section_cpts = available.iloc[cpt_indices]["name"].tolist()
        sections.append(
            {
                "cpts": section_cpts,
                "start_m": start_m,
                "end_m": end_m,
            }
        )
    return sections


def select_cpts_to_remove(
    cpt_names: List[str], n_remove: int = 12, random_state: int = None
) -> List[str]:
    if random_state is not None:
        np.random.seed(random_state)
    n_total = len(cpt_names)
    if n_total < n_remove + 2:
        raise ValueError(
            f"Not enough CPTs to remove {n_remove}. Need at least {n_remove + 2} CPTs."
        )
    valid_indices = list(range(1, n_total - 1))
    selected_indices = []
    while len(selected_indices) < n_remove:
        remaining = [idx for idx in valid_indices if idx not in selected_indices]
        if not remaining:
            raise ValueError(
                f"Cannot select {n_remove} non-consecutive CPTs. Try reducing n_remove."
            )
        idx = np.random.choice(remaining)
        selected_indices.append(idx)
        valid_indices = [i for i in valid_indices if i not in [idx - 1, idx, idx + 1]]
    selected_indices.sort()
    return [cpt_names[i] for i in selected_indices]


def compute_cum_along_m(coords):
    # Compute cumulative distance along the CPT line (sorted by x or y)
    coords = coords.copy()
    coords = coords.sort_values(["x", "y"]).reset_index(drop=True)
    cum_dist = [0.0]
    for i in range(1, len(coords)):
        dx = coords.loc[i, "x"] - coords.loc[i - 1, "x"]
        dy = coords.loc[i, "y"] - coords.loc[i - 1, "y"]
        dist = np.sqrt(dx**2 + dy**2)
        cum_dist.append(cum_dist[-1] + dist)
    coords["cum_along_m"] = cum_dist
    return coords


def assemble_aligned_mosaic(
    sections, section_outputs, n_rows, n_cols, meters_per_col, first_pos
):
    # Create empty mosaic and count arrays
    mosaic = np.zeros((n_rows, n_cols), dtype=float)
    count = np.zeros((n_rows, n_cols), dtype=int)
    for section, output in zip(sections, section_outputs):
        # Determine start and end columns for this section
        start_col = int(round((section["start_m"] - first_pos) / meters_per_col))
        end_col = start_col + output.shape[1]
        # Clip to mosaic bounds
        start_col = max(0, start_col)
        end_col = min(n_cols, end_col)
        # Compute overlap region
        out_start = 0
        out_end = output.shape[1] - (end_col - start_col)
        if out_end < 0:
            out_end = 0
        # Add section output to mosaic
        mosaic[:, start_col:end_col] += output[
            :, out_start : out_start + (end_col - start_col)
        ]
        count[:, start_col:end_col] += 1
    # Avoid division by zero
    mask = count > 0
    mosaic[mask] /= count[mask]
    return mosaic


def build_validation_section_matrix(section, coords, cpt_df_full, n_rows):
    # Get CPTs and their positions
    cpt_names = section["cpts"]
    cpt_df_sorted = cpt_df_full[cpt_names]
    # Compute relative distances for this section
    cpt_coords = coords[coords["name"].isin(cpt_names)].sort_values("cum_along_m")
    rel_dists = cpt_coords["cum_along_m"].values - cpt_coords["cum_along_m"].values[0]
    # Map to 512 columns, no extra padding (since we want full coverage)
    cols, _, _, _ = map_dist_to_cols(rel_dists, 0.0, 0.0, 512)
    cols = resolve_collisions(cols, 512)
    grid, painted, skipped = build_section_matrix(
        cpt_names, cols, cpt_df_full, n_rows, 512
    )
    return grid


# Entry point
def run_validation_pipeline_aligned(
    folders, compressed_csv, y_top_m, y_bottom_m, n_runs, n_remove, base_seed
):
    # 1. Get original extent and columns
    total_extent, n_cols, first_pos, last_pos = get_original_extent_and_columns(folders)

    # 2. Load coordinates
    coords = pd.read_csv(Path(folders["1_coords"]) / "cpt_coordinates.csv")
    if "cum_along_m" not in coords.columns:
        coords = compute_cum_along_m(coords)

    # 3. For each run, select removed CPTs and generate sections
    VALIDATION_FOLDER = folders["8_validation"]
    VALIDATION_FOLDER.mkdir(parents=True, exist_ok=True)
    model = load_model(config.SCHGAN_MODEL_PATH, compile=False)
    cpt_df_full = pd.read_csv(compressed_csv)
    all_cpt_names_spatial = coords.sort_values("cum_along_m")["name"].tolist()
    meters_per_col = total_extent / (n_cols - 1)
    n_rows = config.N_ROWS
    for run_idx in range(1, n_runs + 1):
        run_seed = base_seed + run_idx if base_seed is not None else None
        run_folder = VALIDATION_FOLDER / f"run_{run_idx:02d}"
        run_folder.mkdir(parents=True, exist_ok=True)
        removed_cpts = select_cpts_to_remove(all_cpt_names_spatial, n_remove, run_seed)
        sections = generate_validation_sections(
            coords, removed_cpts, 6, total_extent, n_cols, first_pos, last_pos
        )
        section_outputs = []
        for i, section in enumerate(sections):
            section_matrix = build_validation_section_matrix(
                section, coords, cpt_df_full, n_rows
            )
            # Normalize and run through GAN
            cs_input = section_matrix.reshape(1, n_rows, 512, 1).astype(float)
            cs_norm = IC_normalization([cs_input, cs_input])[0]
            gan_result = model.predict(cs_norm, verbose=0)
            gan_result = reverse_IC_normalization(gan_result)
            gan_result = np.squeeze(gan_result)  # (n_rows, 512)
            section_outputs.append(gan_result)
        # Assemble mosaic
        mosaic = assemble_aligned_mosaic(
            sections, section_outputs, n_rows, n_cols, meters_per_col, first_pos
        )
        # Save mosaic as CSV
        mosaic_csv = run_folder / "validation_mosaic.csv"
        pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)
        print(
            f"Run {run_idx}: Saved aligned validation mosaic with shape {mosaic.shape} to {mosaic_csv}"
        )
        # ...rest of pipeline: GAN, mosaic, metrics, plots...
