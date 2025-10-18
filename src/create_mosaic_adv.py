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

# ---- NEW: strategy/config switches ----
STRATEGY = "linear"  # one of: "linear", "voronoi", "mincut"
SEAM_SMOOTH_PX = 0  # 0 = perfectly sharp; 1 = tiny cleanup around seam
NORMALIZE_PER_SECTION = False  # keep False to avoid changing values
EXPORT_SEAMLINES = True  # export seamlines for diagnostics (only affects "mincut")

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
# REGRIDDING & HELPERS (NEW)
# =============================================================================
def _load_section_array(section_csv: Path, n_rows: int, n_cols: int) -> np.ndarray:
    arr = pd.read_csv(section_csv).to_numpy(dtype=float)
    if arr.shape != (n_rows, n_cols):
        raise ValueError(
            f"{Path(section_csv).name}: expected shape ({n_rows},{n_cols}), got {arr.shape}"
        )
    return arr


def _regrid_section_to_global_nearest(arr, x0, dx, xmin, global_dx, width):
    """
    Nearest-neighbor reprojection of a (N_ROWS, N_COLS) section onto the global grid.
    Returns (canvas, mask) where canvas is (N_ROWS, width) with NaN outside coverage,
    mask is boolean coverage.
    """
    n_rows, n_cols = arr.shape
    xj = x0 + np.arange(n_cols) * dx
    pos = np.round((xj - xmin) / global_dx).astype(int)

    canvas = np.full((n_rows, width), np.nan, dtype=float)
    mask = np.zeros((n_rows, width), dtype=bool)

    valid = (pos >= 0) & (pos < width)
    if not np.any(valid):
        return canvas, mask

    posv = pos[valid]
    canvas[:, posv] = arr[:, valid]
    mask[:, posv] = True
    return canvas, mask


def _regrid_section_to_global_linear(arr, x0, dx, xmin, global_dx, width):
    """
    Linear (1D) horizontal distribution into global grid (like your accumulator).
    Returns (canvas, weight) to allow compositing by summation then normalization.
    """
    n_rows, n_cols = arr.shape
    xj = x0 + np.arange(n_cols) * dx
    pos = (xj - xmin) / global_dx
    k0 = np.floor(pos).astype(int)
    frac = pos - k0
    k1 = k0 + 1

    canvas = np.zeros((n_rows, width), dtype=float)
    weight = np.zeros((width,), dtype=float)

    valid0 = (k0 >= 0) & (k0 < width)
    if np.any(valid0):
        canvas[:, k0[valid0]] += arr[:, valid0] * (1.0 - frac[valid0])
        weight[k0[valid0]] += 1.0 - frac[valid0]

    valid1 = (k1 >= 0) & (k1 < width)
    if np.any(valid1):
        canvas[:, k1[valid1]] += arr[:, valid1] * frac[valid1]
        weight[k1[valid1]] += frac[valid1]

    return canvas, weight


def _min_cost_vertical_seam(cost):
    """
    Dynamic programming minimal vertical path from top row to bottom row.
    cost: (H, W_overlap) array, non-negative.
    Returns seam_x_idx per row (list of ints) in [0..W_overlap-1].
    """
    H, W = cost.shape
    dp = cost.copy()
    back = np.zeros((H, W), dtype=np.int16)

    for i in range(1, H):
        for j in range(W):
            j0 = max(j - 1, 0)
            j1 = min(j + 1, W - 1)
            prev_slice = dp[i - 1, j0 : j1 + 1]
            k = np.argmin(prev_slice)
            dp[i, j] += prev_slice[k]
            back[i, j] = j0 + k

    # trace back from bottom
    seam = [int(np.argmin(dp[-1]))]
    for i in range(H - 1, 0, -1):
        seam.append(int(back[i, seam[-1]]))
    seam.reverse()
    return seam  # length H


# =============================================================================
# MOSAIC METHODS
# =============================================================================
def mosaic_linear_average(manifest, N_ROWS, N_COLS, xmin, xmax, global_dx, width):
    """
    Your current accumulator + linear interpolation, then weighted average.
    Smooth transitions (feathered). Kept for comparison.
    """
    acc = np.zeros((N_ROWS, width), dtype=float)
    wts = np.zeros((width,), dtype=float)

    for _, row in manifest.iterrows():
        arr = _load_section_array(row["gan_csv"], N_ROWS, N_COLS)
        if NORMALIZE_PER_SECTION:
            arr = normalize_section(arr)  # optional hook
        canvas, weight = _regrid_section_to_global_linear(
            arr, float(row["x0"]), float(row["dx"]), xmin, global_dx, width
        )
        acc += canvas
        wts += weight

    eps = 1e-12
    mosaic = acc / np.maximum(wts, eps)[None, :]
    return mosaic


def mosaic_voronoi_hard(manifest, N_ROWS, N_COLS, xmin, xmax, global_dx, width):
    """
    Sharp seams via nearest section center (Voronoi along X).
    """
    canvases = []
    masks = []
    centers = []

    for _, row in manifest.iterrows():
        arr = _load_section_array(row["gan_csv"], N_ROWS, N_COLS)
        if NORMALIZE_PER_SECTION:
            arr = normalize_section(arr)
        canvas, mask = _regrid_section_to_global_nearest(
            arr, float(row["x0"]), float(row["dx"]), xmin, global_dx, width
        )
        canvases.append(canvas)
        masks.append(mask)
        centers.append(0.5 * (float(row["x0"]) + float(row["x1"])))

    centers = np.array(centers, dtype=float)
    xs = xmin + np.arange(width) * global_dx

    # Owner section per column = nearest center
    owners = np.argmin(np.abs(centers[:, None] - xs[None, :]), axis=0)

    mosaic = np.full((N_ROWS, width), np.nan, dtype=float)
    for s_idx, (canvas, mask) in enumerate(zip(canvases, masks)):
        cols = np.where(owners == s_idx)[0]
        if cols.size == 0:
            continue
        valid_cols = cols[np.any(mask[:, cols], axis=0)]
        if valid_cols.size:
            mosaic[:, valid_cols] = canvas[:, valid_cols]

    # Fill isolated NaNs (should be rare with guaranteed continuity)
    if np.isnan(mosaic).any():
        for j in range(width):
            if np.isnan(mosaic[:, j]).all():
                continue
            col = mosaic[:, j]
            if np.isnan(col).any():
                for s in range(len(canvases)):
                    cand = canvases[s][:, j]
                    if not np.isnan(cand).all():
                        col[np.isnan(col)] = cand[np.isnan(col)]
                        mosaic[:, j] = col
                        break
    return mosaic


def mosaic_min_cut_seams(
    manifest,
    N_ROWS,
    N_COLS,
    xmin,
    xmax,
    global_dx,
    width,
    seam_smooth_px=0,
    export_dir: Path = None,
):
    """
    Sharp, content-aware seams. Left-to-right compositing; in each overlap,
    compute cost = |A - B| and find minimal vertical path. No feathering.
    """
    man = manifest.sort_values("x0").reset_index(drop=True)

    def load_arr(row):
        arr = _load_section_array(row["gan_csv"], N_ROWS, N_COLS)
        return normalize_section(arr) if NORMALIZE_PER_SECTION else arr

    arr0 = load_arr(man.loc[0])
    base, base_mask = _regrid_section_to_global_nearest(
        arr0, float(man.loc[0, "x0"]), float(man.loc[0, "dx"]), xmin, global_dx, width
    )

    seam_records = []  # for optional export

    for i in range(1, len(man)):
        arr = load_arr(man.loc[i])
        nxt, nxt_mask = _regrid_section_to_global_nearest(
            arr,
            float(man.loc[i, "x0"]),
            float(man.loc[i, "dx"]),
            xmin,
            global_dx,
            width,
        )

        overlap_cols = np.where(np.any(base_mask, axis=0) & np.any(nxt_mask, axis=0))[0]
        if overlap_cols.size == 0:
            put_cols = np.where(~np.any(base_mask, axis=0) & np.any(nxt_mask, axis=0))[
                0
            ]
            if put_cols.size:
                base[:, put_cols] = nxt[:, put_cols]
                base_mask[:, put_cols] = nxt_mask[:, put_cols]
            continue

        j0, j1 = overlap_cols[0], overlap_cols[-1] + 1
        A = base[:, j0:j1]
        B = nxt[:, j0:j1]

        A_nan = np.isnan(A)
        B_nan = np.isnan(B)
        both = ~(A_nan | B_nan)
        cost = np.full_like(A, np.nan, dtype=float)
        if np.any(both):
            cost[both] = np.abs(A[both] - B[both])
            # Avoid missing data in seam path
            cost[np.isnan(cost)] = np.nanmax(cost[both]) * 10.0
        else:
            cost[:, :] = 1.0

        seam_x = _min_cost_vertical_seam(cost)

        # Optional record for diagnostics/export
        seam_records.append(
            pd.DataFrame(
                {
                    "row": np.arange(A.shape[0], dtype=int),
                    "seam_x_in_overlap": seam_x,
                    "global_col": j0 + np.array(seam_x, dtype=int),
                }
            )
        )

        # Compose with sharp seam (optionally 1px band cleanup)
        H, W = A.shape
        for r in range(H):
            s = seam_x[r]
            if seam_smooth_px > 0:
                sL = max(s - seam_smooth_px, 0)
                sR = min(s + seam_smooth_px, W - 1)
            else:
                sL, sR = s, s

            # left of seam → A
            base[r, j0 : j0 + sL] = A[r, :sL]
            base_mask[r, j0 : j0 + sL] = ~np.isnan(A[r, :sL])

            # seam band → choose per-pixel lower absolute diff (still sharp)
            if sR >= sL:
                bandA = A[r, sL : sR + 1]
                bandB = B[r, sL : sR + 1]
                chooseA = np.abs(bandA - bandB) <= np.abs(bandB - bandA)
                band = np.where(chooseA, bandA, bandB)
                base[r, j0 + sL : j0 + sR + 1] = band
                base_mask[r, j0 + sL : j0 + sR + 1] = ~np.isnan(band)

            # right of seam → B
            base[r, j0 + sR + 1 : j1] = B[r, sR + 1 :]
            base_mask[r, j0 + sR + 1 : j1] = ~np.isnan(B[r, sR + 1 :])

        # Non-overlap parts of nxt
        left_cols = np.where(np.any(nxt_mask, axis=0) & (np.arange(width) < j0))[0]
        right_cols = np.where(np.any(nxt_mask, axis=0) & (np.arange(width) >= j1))[0]
        if left_cols.size:
            base[:, left_cols] = nxt[:, left_cols]
            base_mask[:, left_cols] = nxt_mask[:, left_cols]
        if right_cols.size:
            base[:, right_cols] = nxt[:, right_cols]
            base_mask[:, right_cols] = nxt_mask[:, right_cols]

    # Optional export of seamlines
    if export_dir is not None and EXPORT_SEAMLINES and seam_records:
        seams_df = pd.concat(
            seam_records, keys=range(1, len(man)), names=["pair_idx", "row_idx"]
        )
        seams_csv = export_dir / "mosaic_seamlines.csv"
        seams_df.to_csv(seams_csv, index=True)

    return base


# =============================================================================
# OPTIONAL NORMALIZATION HOOK (kept off by default)
# =============================================================================
def normalize_section(arr: np.ndarray) -> np.ndarray:
    """
    Placeholder: per-section normalization (DISABLED by default).
    To keep values as measured, we return arr unchanged unless you enable it.
    """
    # Example (if you ever enable):
    # p_lo, p_hi = np.nanpercentile(arr, [5, 95])
    # scale = max(p_hi - p_lo, 1e-9)
    # return (arr - p_lo) / scale
    return arr


# =============================================================================
# MOSAIC DISPATCHER
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

    # Choose compositing strategy
    if STRATEGY == "linear":
        mosaic = mosaic_linear_average(
            manifest, N_ROWS, N_COLS, xmin, xmax, global_dx, width
        )
    elif STRATEGY == "voronoi":
        mosaic = mosaic_voronoi_hard(
            manifest, N_ROWS, N_COLS, xmin, xmax, global_dx, width
        )
    elif STRATEGY == "mincut":
        mosaic = mosaic_min_cut_seams(
            manifest,
            N_ROWS,
            N_COLS,
            xmin,
            xmax,
            global_dx,
            width,
            seam_smooth_px=SEAM_SMOOTH_PX,
            export_dir=OUT_DIR,
        )
    else:
        raise ValueError(f"Unknown STRATEGY={STRATEGY}")

    return mosaic, xmin, xmax, global_dx


# =============================================================================
# PLOT & SAVE (unchanged)
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
    mosaic_csv = OUT_DIR / f"schemaGAN_mosaic_{STRATEGY}.csv"
    pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)

    # Save PNG
    mosaic_png = OUT_DIR / f"schemaGAN_mosaic_{STRATEGY}.png"
    plot_mosaic(mosaic, xmin, xmax, global_dx, mosaic_png)

    if STRATEGY == "mincut" and EXPORT_SEAMLINES:
        seam_csv = OUT_DIR / "mosaic_seamlines.csv"
        if seam_csv.exists():
            print(f"[INFO] Seamlines exported → {seam_csv}")

    print(f"[DONE] Mosaic saved:\n  CSV → {mosaic_csv}\n  PNG → {mosaic_png}")


if __name__ == "__main__":
    main()
