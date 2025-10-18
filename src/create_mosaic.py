from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
MANIFEST_CSV = Path(r"C:\VOW\res\north\exp_1\3_sections\manifest_sections.csv")
COORDS_WITH_DIST_CSV = Path(
    r"C:\VOW\res\north\exp_1\3_sections\cpt_coords_with_distances.csv"
)
GAN_DIR = Path(r"C:\VOW\res\north\exp_1\4_gan_images")  # where the *_gan.csv files are
OUT_DIR = Path(r"C:\VOW\res\north\exp_1\5_mosaic")  # where to save mosaic csv/png
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Section & image constants
N_COLS = 512
N_ROWS = 32

# Real-depth range used during equalization/compression (for right-hand axis on the plot)
Y_TOP_M = 6.862
Y_BOTTOM_M = -13.041

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
    if missing := REQUIRED_MANIFEST_COLS - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {missing}")
    if missing := REQUIRED_COORDS_COLS - set(coords.columns):
        raise ValueError(f"Coords file missing columns: {missing}")

    # Convert index columns to integers
    manifest["section_index"] = manifest["section_index"].astype(int)
    manifest["start_idx"] = manifest["start_idx"].astype(int)
    return manifest, coords


def find_latest_gan_csv(section_index: int):
    """Find the newest *_gan.csv file for a given section index."""
    # Look for files like: section_01_cpts_*_gan.csv
    pattern = f"section_{section_index:02d}_cpts_*_gan.csv"
    candidates = list(GAN_DIR.glob(pattern))
    if not candidates:
        return None
    # Sort by modification time (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# =============================================================================
# COMPUTE SECTION PLACEMENTS (NO CLASSES)
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


def choose_global_grid(manifest):
    """
    Compute the global horizontal range (meters) and pixel size.
    The mosaic will use a uniform GLOBAL_DX in meters per pixel.
    """
    xmin = float(manifest["x0"].min())
    xmax = float(manifest["x1"].max())

    # Use median section dx if not provided
    dx = GLOBAL_DX if GLOBAL_DX is not None else float(np.median(manifest["dx"]))
    width = int(round((xmax - xmin) / dx)) + 1
    return xmin, xmax, dx, width


# =============================================================================
# INTERPOLATION AND AVERAGING
# =============================================================================
def add_section_to_accumulator(acc, wts, section_csv, x0, dx, xmin, global_dx):
    """
    Map one section's 512 columns into the global mosaic grid.

    For each column j:
      - Compute its real-world position xj = x0 + j*dx
      - Find its fractional position in global pixels: pos = (xj - xmin)/global_dx
      - Split contribution between nearest global pixels using linear interpolation
    """
    arr = pd.read_csv(section_csv).to_numpy(dtype=float)
    if arr.shape != (N_ROWS, N_COLS):
        raise ValueError(
            f"{Path(section_csv).name}: expected shape (32,512), got {arr.shape}"
        )

    # Compute global positions
    xj = x0 + np.arange(N_COLS) * dx
    pos = (xj - xmin) / global_dx
    k0 = np.floor(pos).astype(int)
    frac = pos - k0
    k1 = k0 + 1

    # Keep columns that fall inside mosaic width
    valid = (k0 >= 0) & (k0 < wts.size)
    if not np.any(valid):
        return

    k0v = k0[valid]
    f0 = 1.0 - frac[valid]
    k1v = k0v + 1
    f1 = frac[valid]

    # Add weighted contributions to accumulator and weights
    acc[:, k0v] += arr[:, valid] * f0
    wts[k0v] += f0

    in_range = k1v < wts.size
    if np.any(in_range):
        acc[:, k1v[in_range]] += arr[:, valid][:, in_range] * f1[in_range]
        wts[k1v[in_range]] += f1[in_range]


# =============================================================================
# MAIN MOSAIC BUILDER
# =============================================================================
def build_mosaic(manifest, coords):
    """Create the mosaic by combining all sections into one global grid."""
    manifest = manifest.copy()
    manifest["gan_csv"] = manifest["section_index"].apply(find_latest_gan_csv)

    # Drop sections without GAN output
    missing = manifest[manifest["gan_csv"].isna()]
    if not missing.empty:
        print("[WARN] Missing GAN csv for sections:", missing["section_index"].tolist())
    manifest = manifest.dropna(subset=["gan_csv"]).reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No sections with GAN CSVs found.")

    # Compute placement for each section (as tuples)
    x0_list, dx_list, x1_list = [], [], []
    for _, row in manifest.iterrows():
        x0, dx, x1 = compute_section_placement(row, coords)
        x0_list.append(x0)
        dx_list.append(dx)
        x1_list.append(x1)

    # Attach placement columns
    manifest["x0"] = x0_list
    manifest["dx"] = dx_list
    manifest["x1"] = x1_list

    # Define global grid
    xmin, xmax, global_dx, width = choose_global_grid(manifest)
    print(
        f"[INFO] Global extent: {xmin:.2f}..{xmax:.2f} m "
        f"({xmax - xmin:.2f} m), dx={global_dx:.4f} m/px, width={width} px"
    )

    # Prepare arrays
    acc = np.zeros((N_ROWS, width))
    wts = np.zeros(width)

    # Add each section to the global mosaic
    for _, row in manifest.iterrows():
        add_section_to_accumulator(
            acc,
            wts,
            row["gan_csv"],
            float(row["x0"]),
            float(row["dx"]),
            xmin,
            global_dx,
        )

    # Weighted average
    eps = 1e-12
    mosaic = acc / np.maximum(wts, eps)[None, :]
    return mosaic, xmin, xmax, global_dx


# =============================================================================
# PLOT & SAVE
# =============================================================================
def plot_mosaic(mosaic, xmin, xmax, global_dx, out_png):
    """Plot the mosaic with dual axes and save as PNG."""
    horiz_m = xmax - xmin
    vert_m = abs(Y_BOTTOM_M - Y_TOP_M)

    # Adjust figure height to preserve approximate real ratio
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

    # Right y-axis: real depths
    def idx_to_m(y_idx):
        return Y_TOP_M + (y_idx / (N_ROWS - 1)) * (Y_BOTTOM_M - Y_TOP_M)

    def m_to_idx(y_m):
        denom = Y_BOTTOM_M - Y_TOP_M
        return 0 if abs(denom) < 1e-12 else (y_m - Y_TOP_M) * (N_ROWS - 1) / denom

    right = ax.secondary_yaxis("right", functions=(idx_to_m, m_to_idx))
    right.set_ylabel("Depth (m)")

    plt.title("SchemaGAN Mosaic")
    plt.tight_layout()
    plt.savefig(out_png, dpi=500)
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    manifest, coords = load_inputs(MANIFEST_CSV, COORDS_WITH_DIST_CSV)
    mosaic, xmin, xmax, global_dx = build_mosaic(manifest, coords)

    # Save CSV
    mosaic_csv = OUT_DIR / "schemaGAN_mosaic.csv"
    pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)

    # Save PNG
    mosaic_png = OUT_DIR / "schemaGAN_mosaic.png"
    plot_mosaic(mosaic, xmin, xmax, global_dx, mosaic_png)

    print(f"[DONE] Mosaic saved:\n  CSV → {mosaic_csv}\n  PNG → {mosaic_png}")


if __name__ == "__main__":
    main()
