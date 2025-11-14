from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
MANIFEST_CSV = Path(r"C:\VOW\res\test_outputs\manifest_sections.csv")
COORDS_WITH_DIST_CSV = Path(r"C:\VOW\res\test_outputs\cpt_coords_with_distances.csv")

# Where the *_gan.csv files are (output of the GAN script)
GAN_DIR = Path(r"C:\VOW\res\test_images")

# File suffix to look for (e.g., "gan" or "uncertainty")
FILE_SUFFIX = "gan"

# Where to save mosaic csv/png
OUT_DIR = Path(r"C:\VOW\res\test_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Section & image constants
N_COLS = 512  # width of each section
N_ROWS_WINDOW = 32  # rows per depth window (SchemaGAN image height)

# Real-depth range for the FULL vertical (all compressed rows combined)
Y_TOP_M = 6.773
Y_BOTTOM_M = -13.128

# Optional: set global pixel size (m/px). If None, use median section dx
GLOBAL_DX = None

# Top axis appearance: False → show 0..(W-1) pixels; True → show 0..32 normalized
TOP_AXIS_0_TO_32 = False


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================
REQUIRED_MANIFEST_COLS = {
    "section_index",
    "span_m",
    "left_pad_m",
    "right_pad_m",
    "start_idx",
    "csv_path",
}
REQUIRED_COORDS_COLS = {"cum_along_m"}


# =============================================================================
# LOAD INPUT DATA
# =============================================================================
def load_inputs(manifest_csv: Path, coords_csv: Path):
    """Load and validate the manifest and coordinates tables."""
    manifest = pd.read_csv(manifest_csv)
    coords = pd.read_csv(coords_csv)

    # Check required columns
    missing = REQUIRED_MANIFEST_COLS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    missing = REQUIRED_COORDS_COLS - set(coords.columns)
    if missing:
        raise ValueError(f"Coords file missing columns: {missing}")

    # Convert numeric index columns
    manifest["section_index"] = manifest["section_index"].astype(int)
    manifest["start_idx"] = manifest["start_idx"].astype(int)

    # depth_window is optional (old pipeline) but strongly recommended now
    if "depth_window" in manifest.columns:
        manifest["depth_window"] = manifest["depth_window"].astype(int)

    if "depth_start_row" in manifest.columns:
        manifest["depth_start_row"] = manifest["depth_start_row"].astype(int)
    if "depth_end_row" in manifest.columns:
        manifest["depth_end_row"] = manifest["depth_end_row"].astype(int)

    return manifest, coords


def find_latest_gan_csv_for_row(row):
    """
    Find the newest file matching FILE_SUFFIX for a given manifest row.

    Uses the stem of csv_path so this works with depth windows:
      csv_path = ".../section_01_z_00_cpts_001_to_006.csv"
      -> look for files like:
         "section_01_z_00_cpts_001_to_006_seed*_{FILE_SUFFIX}.csv"
    """
    section_stem = Path(row["csv_path"]).stem
    pattern = f"{section_stem}_seed*_{FILE_SUFFIX}.csv"
    candidates = list(GAN_DIR.glob(pattern))
    if not candidates:
        return None
    # Sort by modification time (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# =============================================================================
# X-DIRECTION PLACEMENT
# =============================================================================
def compute_section_placement(row, coords):
    """
    Calculate the real-world x-range for this section.
    Includes left/right padding and converts pixel index → meters.

    Returns a tuple: (x0, dx, x1)
      x0: leftmost coordinate (meters) including left padding
      dx: meters per pixel in this section
      x1: rightmost coordinate (meters)
    """
    total_span = float(row["span_m"] + row["left_pad_m"] + row["right_pad_m"])
    if total_span <= 0:
        raise ValueError(f"Invalid total span for section {row['section_index']}")

    start_idx = int(row["start_idx"])
    m0 = float(coords.loc[start_idx, "cum_along_m"])
    x0 = m0 - float(row["left_pad_m"])  # shift to include left padding
    dx = total_span / (N_COLS - 1)  # meters per pixel in section
    x1 = x0 + (N_COLS - 1) * dx
    return x0, dx, x1


def choose_global_grid(manifest, coords):
    """
    Compute the global horizontal range (meters) and pixel size.
    The mosaic will use a uniform GLOBAL_DX in meters per pixel.

    Uses actual CPT positions (not padded section bounds) to determine extent,
    so the mosaic doesn't extend beyond real data with artificial padding.
    """
    # Use actual CPT positions to determine the true data extent
    xmin = float(coords["cum_along_m"].min())
    xmax = float(coords["cum_along_m"].max())

    # Use median section dx if not provided
    dx = GLOBAL_DX if GLOBAL_DX is not None else float(np.median(manifest["dx"]))
    width = int(round((xmax - xmin) / dx)) + 1

    print(
        f"[INFO] Global grid based on actual CPT positions: "
        f"{xmin:.2f}..{xmax:.2f} m (span: {xmax - xmin:.2f} m)"
    )

    return xmin, xmax, dx, width


# =============================================================================
# INTERPOLATION AND AVERAGING
# =============================================================================
def add_section_to_accumulator(
    acc: np.ndarray,
    wts: np.ndarray,
    section_csv: Path,
    x0: float,
    dx: float,
    xmin: float,
    global_dx: float,
    y_offset: int,
):
    """
    Map one section's (N_ROWS_WINDOW x N_COLS) array into the global mosaic grid.

    For each column j:
      - Compute its real-world position xj = x0 + j*dx
      - Find its fractional position in global pixels: pos = (xj - xmin)/global_dx
      - Split contribution between nearest global pixels using linear interpolation

    Vertical placement uses y_offset (global starting row index for this window).
    With overlapping depth windows, multiple windows may contribute to the same
    global rows; they are averaged via 2D weights.

    acc and wts are both 2D: shape (n_rows_total, width).
    """
    arr = pd.read_csv(section_csv).to_numpy(dtype=float)
    if arr.shape != (N_ROWS_WINDOW, N_COLS):
        raise ValueError(
            f"{section_csv.name}: expected shape ({N_ROWS_WINDOW},{N_COLS}), got {arr.shape}"
        )

    # Compute global positions (in meters)
    xj = x0 + np.arange(N_COLS) * dx
    pos = (xj - xmin) / global_dx
    k0 = np.floor(pos).astype(int)
    frac = pos - k0
    k1 = k0 + 1

    # Keep columns that fall inside mosaic width
    width = wts.shape[1]
    valid = (k0 >= 0) & (k0 < width)
    if not np.any(valid):
        return

    k0v = k0[valid]
    f0 = 1.0 - frac[valid]
    k1v = k0v + 1
    f1 = frac[valid]

    rows_slice = slice(y_offset, y_offset + N_ROWS_WINDOW)

    # Add weighted contributions to accumulator and weights (per pixel)
    acc[rows_slice, k0v] += arr[:, valid] * f0
    wts[rows_slice, k0v] += f0  # broadcast f0 over all rows in this window

    in_range = k1v < width
    if np.any(in_range):
        acc[rows_slice, k1v[in_range]] += arr[:, valid][:, in_range] * f1[in_range]
        wts[rows_slice, k1v[in_range]] += f1[in_range]


# =============================================================================
# MAIN MOSAIC BUILDER
# =============================================================================
def build_mosaic(manifest, coords):
    """Create the mosaic by combining all sections and depth windows."""
    manifest = manifest.copy()

    # Attach per-row GAN CSV path
    manifest["gan_csv"] = manifest.apply(find_latest_gan_csv_for_row, axis=1)

    # Drop rows without GAN output
    missing = manifest[manifest["gan_csv"].isna()]
    if not missing.empty:
        print("[WARN] Missing GAN csv for some rows (section_index, csv_path):")
        print(missing[["section_index", "csv_path"]])
    manifest = manifest.dropna(subset=["gan_csv"]).reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No rows with GAN CSVs found in manifest.")

    # Compute x0, dx, x1 for each row
    x0_list, dx_list, x1_list = [], [], []
    for _, row in manifest.iterrows():
        x0, dx, x1 = compute_section_placement(row, coords)
        x0_list.append(x0)
        dx_list.append(dx)
        x1_list.append(x1)

    manifest["x0"] = x0_list
    manifest["dx"] = dx_list
    manifest["x1"] = x1_list

    # Global horizontal grid (based on actual CPT positions, not padded sections)
    xmin, xmax, global_dx, width = choose_global_grid(manifest, coords)
    print(
        f"[INFO] Global extent: {xmin:.2f}..{xmax:.2f} m "
        f"({xmax - xmin:.2f} m), dx={global_dx:.4f} m/px, width={width} px"
    )

    # ---- Vertical grid: prefer depth_start_row/depth_end_row if present ----
    if "depth_start_row" in manifest.columns and "depth_end_row" in manifest.columns:
        row_min = int(manifest["depth_start_row"].min())
        row_max = int(manifest["depth_end_row"].max()) - 1
        n_rows_total = row_max - row_min + 1

        print(
            f"[INFO] Vertical grid from original compressed rows: "
            f"{row_min}..{row_max} → total {n_rows_total} rows"
        )

        def get_y_offset(row):
            # global starting row index for this window
            return int(row["depth_start_row"] - row_min)

    else:
        # Fallback: old behavior = non-overlapping stacking by depth_window
        if "depth_window" in manifest.columns:
            depth_levels = sorted(manifest["depth_window"].unique())
        else:
            depth_levels = [0]

        n_windows = len(depth_levels)
        n_rows_total = n_windows * N_ROWS_WINDOW
        depth_to_offset = {dw: i * N_ROWS_WINDOW for i, dw in enumerate(depth_levels)}

        print(
            f"[INFO] Vertical stacking (fallback): {n_windows} depth window(s), "
            f"{N_ROWS_WINDOW} rows each → total {n_rows_total} rows"
        )

        def get_y_offset(row):
            dw = int(row["depth_window"]) if "depth_window" in row.index else 0
            return depth_to_offset[dw]

    # Prepare accumulator arrays (2D weights!)
    acc = np.zeros((n_rows_total, width))
    wts = np.zeros((n_rows_total, width))

    # Add each section (and window) into the mosaic
    for _, row in manifest.iterrows():
        y_offset = get_y_offset(row)

        add_section_to_accumulator(
            acc=acc,
            wts=wts,
            section_csv=row["gan_csv"],
            x0=float(row["x0"]),
            dx=float(row["dx"]),
            xmin=xmin,
            global_dx=global_dx,
            y_offset=y_offset,
        )

    # Weighted average (per pixel)
    eps = 1e-12
    mosaic = acc / np.maximum(wts, eps)
    return mosaic, xmin, xmax, global_dx, n_rows_total


# =============================================================================
# PLOT & SAVE
# =============================================================================
def plot_mosaic(
    mosaic,
    xmin,
    xmax,
    global_dx,
    n_rows_total,
    out_png: Path,
    coords=None,
    show_cpt_locations=True,
    vmin=0,
    vmax=4.5,
    cmap="viridis",
    colorbar_label="Value",
):
    """Plot the mosaic with dual axes and save as PNG.

    Args:
        mosaic: 2D numpy array of the mosaic
        xmin: Minimum x-coordinate in meters
        xmax: Maximum x-coordinate in meters
        global_dx: Pixel size in meters
        n_rows_total: Total number of rows
        out_png: Output PNG path
        coords: Coordinates dataframe (optional)
        show_cpt_locations: Whether to show CPT markers
        vmin: Minimum value for colormap (None = auto)
        vmax: Maximum value for colormap (None = auto)
        cmap: Colormap name
        colorbar_label: Label for colorbar
    """
    horiz_m = xmax - xmin
    vert_m = abs(Y_BOTTOM_M - Y_TOP_M)

    # Adjust figure height to preserve approximate real ratio
    base_width = 16
    height = np.clip(base_width * (vert_m / max(horiz_m, 1e-12)), 2, 12)

    fig, ax = plt.subplots(figsize=(base_width, height))
    im = ax.imshow(
        mosaic,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=[xmin, xmax, n_rows_total - 1, 0],
    )
    plt.colorbar(im, label=colorbar_label)

    # Add vertical lines at CPT positions if enabled and coords provided
    if show_cpt_locations and coords is not None and "cum_along_m" in coords.columns:
        for cpt_x in coords["cum_along_m"]:
            ax.axvline(x=cpt_x, color="black", linewidth=1, alpha=0.5, zorder=10)

    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Depth Index (global)")

    # Top x-axis: pixels or normalized 0..32
    if not TOP_AXIS_0_TO_32:

        def m_to_px(x):
            return (x - xmin) / global_dx

        def px_to_m(p):
            return xmin + p * global_dx

        top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
        top.set_xlabel("Pixel index (0…W-1)")
    else:

        def m_to_u32(x):
            return 32.0 * (x - xmin) / (xmax - xmin + 1e-12)

        def u32_to_m(u):
            return xmin + (u / 32.0) * (xmax - xmin)

        top = ax.secondary_xaxis("top", functions=(m_to_u32, u32_to_m))
        top.set_xlabel("Normalized distance (0…32)")

    # Right y-axis: real depths across the full vertical domain
    def idx_to_m(y_idx):
        return Y_TOP_M + (y_idx / (n_rows_total - 1)) * (Y_BOTTOM_M - Y_TOP_M)

    def m_to_idx(y_m):
        denom = Y_BOTTOM_M - Y_TOP_M
        return 0 if abs(denom) < 1e-12 else (y_m - Y_TOP_M) * (n_rows_total - 1) / denom

    right = ax.secondary_yaxis("right", functions=(idx_to_m, m_to_idx))
    right.set_ylabel("Depth (m)")

    plt.title("SchemaGAN Mosaic (with vertical & horizontal blending)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=500)
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    manifest, coords = load_inputs(MANIFEST_CSV, COORDS_WITH_DIST_CSV)
    mosaic, xmin, xmax, global_dx, n_rows_total = build_mosaic(manifest, coords)

    # Save CSV
    mosaic_csv = OUT_DIR / "schemaGAN_mosaic.csv"
    pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)

    # Save PNG
    mosaic_png = OUT_DIR / "schemaGAN_mosaic.png"
    plot_mosaic(mosaic, xmin, xmax, global_dx, n_rows_total, mosaic_png)

    print(f"[DONE] Mosaic saved:\n  CSV → {mosaic_csv}\n  PNG → {mosaic_png}")


if __name__ == "__main__":
    main()
