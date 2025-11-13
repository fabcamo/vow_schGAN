import os

# Suppress TensorFlow logging before importing it
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import re
import sys
import numpy as np
import pandas as pd
import logging, warnings
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model
from absl import logging as absl_logging
from utils import IC_normalization, reverse_IC_normalization

# Silence TF logging
tf.get_logger().setLevel(logging.ERROR)
absl_logging.set_verbosity(absl_logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
tf.autograph.set_verbosity(0)

# --------------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------------
SECTIONS_DIR = Path(r"C:\VOW\res\test")  # where the section CSVs are
PATH_TO_MODEL = Path(r"D:\schemaGAN\h5\schemaGAN.h5")  # generator .h5
MANIFEST_CSV = SECTIONS_DIR / "manifest_sections.csv"
COORDS_WITH_DIST_CSV = SECTIONS_DIR / "cpt_coords_with_distances.csv"
OUT_DIR = Path(r"C:\VOW\res\test_images")  # where to save outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE_X = 512
SIZE_Y = 32

# Real-depth range used during equalization/compression (set these from previous scripts)
Y_TOP_M = 6.773  # depth at Depth_Index = 0
Y_BOTTOM_M = -13.128  # depth at Depth_Index = 31

# Top x-axis appearance: False -> 0..511 px, True -> 0..32 normalized
TOP_AXIS_0_TO_32 = False
# -------------------------------------------------------------------------------------------

# Choose either a fixed seed here or random seed
# seed = 123456  # fixed seed for reproducibility
seed = np.random.randint(20220412, 20230412)
np.random.seed(seed)
tf.random.set_seed(seed)
print(f"[INFO] Using seed: {seed}")

# -------------------
# LOAD MODEL + METADATA
# -------------------
print("[INFO] Loading model…")
model = load_model(PATH_TO_MODEL, compile=False)

# Load manifest and coords once
manifest = pd.read_csv(MANIFEST_CSV)
coords = pd.read_csv(COORDS_WITH_DIST_CSV)

required_manifest_cols = [
    "section_index",
    "span_m",
    "left_pad_m",
    "right_pad_m",
    "start_idx",
    "csv_path",
]
for col in required_manifest_cols:
    if col not in manifest.columns:
        raise ValueError(f"Manifest missing column: {col}")
if "cum_along_m" not in coords.columns:
    raise ValueError("cpt_coords_with_distances.csv must contain 'cum_along_m'")

manifest["section_index"] = manifest["section_index"].astype(int)
manifest["start_idx"] = manifest["start_idx"].astype(int)
if "depth_window" in manifest.columns:
    manifest["depth_window"] = manifest["depth_window"].astype(int)


def _get_manifest_row_for_file(section_path: Path) -> pd.Series:
    """
    Find the manifest row corresponding to a given section CSV.

    We match on the filename (basename) of csv_path, so manifest can store
    absolute or relative paths.
    """
    sec_name = section_path.name
    matches = manifest[manifest["csv_path"].apply(lambda p: Path(p).name) == sec_name]
    if matches.empty:
        raise ValueError(f"No manifest row found whose csv_path ends with '{sec_name}'")
    return matches.iloc[0]


def _sec_x0_dx_from_manifest_row(r: pd.Series) -> tuple[float, float]:
    """
    Bottom x axis spans meters via:
      x0 = cum_along(first CPT of section) - left_pad_m
      dx = (span + left_pad + right_pad) / (SIZE_X - 1)

    Args:
        r: manifest row for this section CSV

    Returns:
        (x0, dx)
    """
    total_span = float(r["span_m"] + r["left_pad_m"] + r["right_pad_m"])
    start_idx = int(r["start_idx"])
    m0 = float(coords.loc[start_idx, "cum_along_m"])
    x0 = m0 - float(r["left_pad_m"])
    dx = 1.0 if total_span <= 0 else total_span / (SIZE_X - 1)
    return x0, dx


# Y mapping funcs (primary y = Depth_Index, secondary y = meters)
def idx_to_meters(y_idx: float) -> float:
    """Convert Depth_Index (0..31) to meters using linear mapping."""
    return Y_TOP_M + (y_idx / (SIZE_Y - 1)) * (Y_BOTTOM_M - Y_TOP_M)


def meters_to_idx(y_m: float) -> float:
    """Convert meters to Depth_Index (0..31) using linear mapping."""
    denom = Y_BOTTOM_M - Y_TOP_M
    return 0.0 if abs(denom) < 1e-12 else (y_m - Y_TOP_M) * (SIZE_Y - 1) / denom


# -------------------
# CORE
# -------------------
def run_gan_on_section_csv(csv_path: Path) -> tuple[Path, Path]:
    """
    Run SchemaGAN on one section CSV and save CSV + PNG image.
    Returns (csv_out, png_out).

    Args:
        csv_path (Path): Path to section CSV file

    Returns:
        (csv_out, png_out)
    """
    # Look up manifest metadata for this specific file
    mrow = _get_manifest_row_for_file(csv_path)
    sec_index = int(mrow["section_index"])
    depth_win = int(mrow["depth_window"]) if "depth_window" in mrow.index else None

    # Load csv and strip Depth_Index column
    df = pd.read_csv(csv_path)
    if df.shape[0] != SIZE_Y:
        raise ValueError(f"{csv_path.name}: expected {SIZE_Y} rows, got {df.shape[0]}")
    df_vals = df.iloc[:, 1:]  # drop first column (Depth_Index)

    if df_vals.shape[1] != SIZE_X:
        raise ValueError(
            f"{csv_path.name}: expected {SIZE_X} columns (after dropping first), got {df_vals.shape[1]}"
        )

    # To numpy & reshape to (1, 32, 512, 1)
    cs = df_vals.to_numpy(dtype=float).reshape(1, SIZE_Y, SIZE_X, 1)

    # Normalization trick (your util returns a pair)
    cs_norm = IC_normalization([cs, cs])[0]

    # Predict
    gan_res = model.predict(cs_norm, verbose=0)

    # Reverse normalization (back to your plotting range)
    gan_res = reverse_IC_normalization(gan_res)
    gan_res = np.squeeze(gan_res)  # (32, 512)

    # Save CSV (the stem already includes z_XX, so it propagates)
    out_csv = OUT_DIR / f"{csv_path.stem}_seed{seed}_gan.csv"
    pd.DataFrame(gan_res).to_csv(out_csv, index=False)

    # --------- Dual axes plotting ---------
    x0, dx = _sec_x0_dx_from_manifest_row(mrow)
    x1 = x0 + (SIZE_X - 1) * dx

    out_png = OUT_DIR / f"{csv_path.stem}_seed{seed}_gan.png"
    plt.figure(figsize=(10, 2.4))

    # Bottom axis: meters (extent sets x in meters; y stays in Depth_Index 31..0)
    plt.imshow(
        gan_res,
        cmap="viridis",
        vmin=0,
        vmax=4.5,
        aspect="auto",
        extent=[x0, x1, SIZE_Y - 1, 0],  # x in meters, y inverted so 0 at top
    )
    plt.colorbar(label="Value")

    ax = plt.gca()
    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Depth Index")

    # Top x-axis (pixels or 0..32 normalized)
    if not TOP_AXIS_0_TO_32:

        def m_to_px(x):
            return (x - x0) / dx

        def px_to_m(p):
            return x0 + p * dx

        top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
        top.set_xlabel(f"Pixel index (0…{SIZE_X-1})")
    else:

        def m_to_u32(x):
            return 32.0 * (x - x0) / (x1 - x0 + 1e-12)

        def u32_to_m(u):
            return x0 + (u / 32.0) * (x1 - x0)

        top = ax.secondary_xaxis("top", functions=(m_to_u32, u32_to_m))
        top.set_xlabel("Normalized distance (0…32)")

    # Right y-axis: real depth (m)
    right = ax.secondary_yaxis("right", functions=(idx_to_meters, meters_to_idx))
    right.set_ylabel("Depth (m)")

    if depth_win is not None:
        title = f"SchemaGAN Generated Image (Section {sec_index:03d}, z_{depth_win:02d}, Seed: {seed})"
    else:
        title = f"SchemaGAN Generated Image (Section {sec_index:03d}, Seed: {seed})"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    # ----------------------------------------

    return out_csv, out_png


# -------------------
# MAIN LOOP
# -------------------
section_files = sorted(SECTIONS_DIR.glob("section_*_cpts_*.csv"))
if not section_files:
    raise FileNotFoundError(f"No section CSVs found in {SECTIONS_DIR}")

print(f"[INFO] Found {len(section_files)} section file(s) in {SECTIONS_DIR}")

ok, fail = 0, 0

for i, sec in enumerate(section_files, 1):
    try:
        csv_out, png_out = run_gan_on_section_csv(sec)
        ok += 1
        print(
            f"[{i:03d}/{len(section_files)}] OK → CSV: {csv_out.name} | PNG: {png_out.name}"
        )
    except Exception as e:
        fail += 1
        print(f"[{i:03d}/{len(section_files)}] FAIL on {sec.name}: {e}")

print(f"[DONE] Success: {ok}, Failed: {fail}. Outputs in: {OUT_DIR}")
