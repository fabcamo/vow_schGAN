from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# INPUTS (edit these)
# -----------------------------
sigma_csv = Path(r"C:\VOW\res\south\exp_23\9_leaveout_uncert\leaveout_sigma_map.csv")
coords_csv = Path(r"C:\VOW\res\south\exp_23\3_sections\cpt_coords_with_distances.csv")
manifest_csv = Path(
    r"C:\VOW\res\south\exp_23\3_sections\manifest_sections.csv"
)

out_png = Path(
    r"C:\VOW\res\south\exp_23\9_leaveout_uncert\leaveout_sigma_map_FIXED.svg"
)

y_top_m = 0.0  # <-- set to your values
y_bottom_m = 1.0  # <-- set to your values (same ones you used before)

# If you want a fixed x-range (optional). Otherwise inferred from coords.
xmin_m = None
xmax_m = None

# -----------------------------
# LOAD
# -----------------------------
arr = pd.read_csv(sigma_csv, header=None).to_numpy(float)  # (64, 1668)
coords = pd.read_csv(coords_csv)

n_rows, n_cols = arr.shape
assert "cum_along_m" in coords.columns, "coords file must contain 'cum_along_m'"

manifest = pd.read_csv(manifest_csv, sep=",")
manifest.columns = manifest.columns.str.strip()


print(manifest.columns.tolist())
print(manifest.head(1))


min_cpt = float(coords["cum_along_m"].min())
max_cpt = float(coords["cum_along_m"].max())

left_pad = float(manifest["left_pad_m"].max())
right_pad = float(manifest["right_pad_m"].max())

if xmin_m is None:
    xmin_m = min_cpt - left_pad
if xmax_m is None:
    xmax_m = max_cpt + right_pad


# -----------------------------
# PLOT IN METERS
# -----------------------------
dx = (xmax_m - xmin_m) / max(n_cols - 1, 1)

plt.figure(figsize=(15, 15 / 6))
im = plt.imshow(
    arr,
    cmap="hot",
    aspect="auto",
    interpolation="nearest",
    extent=[xmin_m - dx / 2, xmax_m + dx / 2, n_rows - 0.5, -0.5],
)

cbar = plt.colorbar(im)
cbar.set_label("Uncertainty (Std Dev)")

# CPT lines (now consistent units: meters)
for x in coords["cum_along_m"].values:
    if xmin_m <= x <= xmax_m:
        plt.axvline(x=x, color="black", linewidth=1, alpha=0.5)

plt.xlabel("Distance along line (m)")
plt.ylabel("Depth index (global)")
plt.title("West leave-out uncertainty (Std Dev)")

# Right y-axis in meters (optional)
ax = plt.gca()


# -----------------------------
# TOP X-AXIS: pixel indices
# -----------------------------
def px_to_m(x_px):
    return xmin_m + (x_px / (n_cols - 1)) * (xmax_m - xmin_m)


def m_to_px(x_m):
    denom = xmax_m - xmin_m
    return 0 if abs(denom) < 1e-12 else (x_m - xmin_m) * (n_cols - 1) / denom


top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
top.set_xlabel("X (pixel index)")
top.tick_params(labelsize=8)


def idx_to_m(y_idx):
    return y_top_m + (y_idx / (n_rows - 1)) * (y_bottom_m - y_top_m)


def m_to_idx(y_m):
    denom = y_bottom_m - y_top_m
    return 0 if abs(denom) < 1e-12 else (y_m - y_top_m) * (n_rows - 1) / denom


right = ax.secondary_yaxis("right", functions=(idx_to_m, m_to_idx))
right.set_ylabel("Depth (m)")

plt.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=800, bbox_inches="tight")
plt.show()

print("Saved:", out_png)
print("Sigma shape:", arr.shape)
print("X range (m):", xmin_m, "to", xmax_m, "| CPTs:", len(coords))
