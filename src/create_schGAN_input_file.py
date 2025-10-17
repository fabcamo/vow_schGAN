"""Build schemaGAN-ready section CSVs from CPT coordinates and CPT data.

This script:
  1) Sorts CPT coordinates in a chosen direction (e.g., west→east).
  2) Computes distances: from-first, from-previous, and cumulative-along-chain.
  3) Slides a window of CPTS_PER_SECTION across the sorted list with overlap.
  4) Maps relative distances (with left/right padding) to 0..N_COLS-1 columns.
  5) Resolves column collisions and paints a (N_ROWS × N_COLS) grid per section.
  6) Writes section CSVs, a distances CSV, and a manifest CSV.

Assumptions:
  - Coordinates CSV has columns: name, x, y
  - CPT data CSV has one column per CPT name, plus a Depth_Index column with N_ROWS rows
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import logging

from utils import (
    euclid,
)  # def euclid(x1: float, y1: float, x2: float, y2: float) -> float

# Get logger for this module
logger = logging.getLogger(__name__)


# -----------------------------
# Basic helpers
# -----------------------------
def ensure_outdir(path: Path) -> None:
    """Create output directory if it does not exist.

    Args:
        path: Target directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def validate_input_files(
    coords_df: pd.DataFrame, cpt_df: pd.DataFrame, n_rows: int
) -> None:
    """Validate required columns and expected row count for inputs.

    Args:
        coords_df: DataFrame of CPT coordinates with columns ['name', 'x', 'y'].
        cpt_df: DataFrame of CPT values with columns ['Depth_Index', <CPT names...>].
        n_rows: Expected number of rows (depth levels) in cpt_df.

    Raises:
        AssertionError: If any required column is missing or row count mismatches.
    """
    assert {"name", "x", "y"}.issubset(
        coords_df.columns
    ), "coords CSV must have columns: name, x, y"
    assert (
        "Depth_Index" in cpt_df.columns
    ), "CPT data CSV must have a Depth_Index column"
    assert (
        len(cpt_df) == n_rows
    ), f"CPT data must have {n_rows} rows (Depth_Index=0..{n_rows-1})"


def sort_cpt_by_coordinates(
    coords_df: pd.DataFrame, from_where: str, to_where: str
) -> pd.DataFrame:
    """Sort CPTs by direction: west/east uses X; south/north uses Y.

    Args:
        coords_df: Coordinates DataFrame with columns ['name', 'x', 'y'].
        from_where: One of {'west','east','north','south'}.
        to_where: One of {'west','east','north','south'}.

    Returns:
        Sorted DataFrame (index reset, original unchanged).

    Raises:
        ValueError: If an invalid direction pair is provided.
    """
    df = coords_df.copy()

    if from_where == "west" and to_where == "east":
        df = df.sort_values(by="x", ascending=True)
    elif from_where == "east" and to_where == "west":
        df = df.sort_values(by="x", ascending=False)
    elif from_where == "south" and to_where == "north":
        df = df.sort_values(by="y", ascending=True)
    elif from_where == "north" and to_where == "south":
        df = df.sort_values(by="y", ascending=False)
    else:
        raise ValueError("Invalid direction. Use 'west'/'east'/'north'/'south'.")

    return df.reset_index(drop=True)


def compute_distances(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Compute distances assuming `coords_df` is already sorted.

    Adds three columns:
      - dist_from_first_m : distance from the first CPT in the table
      - dist_from_prev_m  : distance from the previous CPT (0.0 for the first row)
      - cum_along_m       : cumulative distance along the ordered chain

    Args:
        coords_df: Sorted DataFrame with columns ['name','x','y'].

    Returns:
        A copy of the input DataFrame with distance columns added.
    """
    df = coords_df.reset_index(drop=True).copy()
    if df.empty:
        return df

    # Pre-allocate result lists for clarity and speed
    dist_from_first: List[float] = []
    dist_from_prev: List[float] = []
    cum_along: List[float] = []

    # First point acts as baseline (distance-from-first = 0 for itself)
    x0, y0 = float(df.loc[0, "x"]), float(df.loc[0, "y"])
    prev_x, prev_y = x0, y0
    total = 0.0

    # Loop over rows: compute distances against first and previous
    for i in range(len(df)):
        x, y = float(df.loc[i, "x"]), float(df.loc[i, "y"])
        d_first = euclid(x0, y0, x, y)
        d_prev = euclid(prev_x, prev_y, x, y) if i > 0 else 0.0
        total += d_prev

        dist_from_first.append(d_first)
        dist_from_prev.append(d_prev)
        cum_along.append(total)

        # Prepare for next iteration
        prev_x, prev_y = x, y

    df["dist_from_first_m"] = dist_from_first
    df["dist_from_prev_m"] = dist_from_prev
    df["cum_along_m"] = cum_along
    return df


def match_names(
    coords_df: pd.DataFrame, cpt_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Mark which coordinate names exist in CPT data columns.

    Adds a boolean column 'has_data' to the coordinates indicating whether each
    'name' appears as a column in the CPT data (excluding 'Depth_Index').
    Returns a list of unmatched names.

    Args:
        coords_df: Coordinates DataFrame with a 'name' column.
        cpt_df: CPT data DataFrame, columns: ['Depth_Index', <CPT names...>].

    Returns:
        Tuple:
            - DataFrame with added 'has_data' boolean column.
            - List of CPT names that are NOT present in the CPT data.
    """
    coords = coords_df.copy()
    cpt_names = [c for c in cpt_df.columns if c != "Depth_Index"]

    has_data: List[bool] = []
    for name in coords["name"]:
        has_data.append(name in cpt_names)

    coords["has_data"] = has_data
    unmatched = coords.loc[
        coords["has_data"] == False, "name"
    ].tolist()  # noqa: E712 (explicit comparison for clarity)
    return coords, unmatched


def section_starts(n: int, per: int, overlap: int) -> List[int]:
    """Compute start indices for overlapping sections.

    Uses a sliding window of width `per` advancing by `per - overlap`. Ensures
    the last section covers the last CPT.

    Args:
        n: Total number of CPTs.
        per: Number of CPTs per section.
        overlap: Number of CPTs that adjacent sections share.

    Returns:
        List of start indices.
    """
    step = max(1, per - overlap)
    starts = list(range(0, max(1, n - per + 1), step))

    # Ensure a final section includes the last CPT (tail coverage).
    if n > 0 and (n - per) > 0:
        last_start = n - per
        if not starts or last_start > starts[-1]:
            starts.append(last_start)
    return starts


def map_dist_to_cols(
    dists_rel: np.ndarray, left_pad_frac: float, right_pad_frac: float, n_cols: int
) -> Tuple[np.ndarray, float, float, float]:
    """Map relative distances (per section) to integer columns with padding.

    Args:
        dists_rel: 1D array of relative distances from the first CPT in the section.
        left_pad_frac: Fraction of span to pad on the left.
        right_pad_frac: Fraction of span to pad on the right.
        n_cols: Number of columns in the target grid.

    Returns:
        Tuple:
            - cols: int array of column indices in [0, n_cols-1].
            - span: total unpadded span of distances in the section.
            - left_pad: actual left padding applied (meters).
            - right_pad: actual right padding applied (meters).
    """
    # Avoid divide-by-zero: ensure minimum span
    span = float(max(1e-9, float(dists_rel.max())))
    left_pad = left_pad_frac * span
    right_pad = right_pad_frac * span
    total_span = span + left_pad + right_pad

    # Normalize each distance within [0, 1] across padded span, then scale to columns
    u = (dists_rel + left_pad) / total_span
    cols = np.rint(u * (n_cols - 1)).astype(int)
    cols = np.clip(cols, 0, n_cols - 1)
    return cols, span, left_pad, right_pad


def resolve_collisions(cols: np.ndarray, n_cols: int) -> np.ndarray:
    """Ensure each CPT maps to a unique column by nudging right, then left.

    Args:
        cols: Initial column indices (may have duplicates).
        n_cols: Total number of columns.

    Returns:
        Array of unique column indices after collision resolution.
    """
    used = set()
    resolved = np.empty_like(cols)

    for i, c in enumerate(cols.tolist()):
        cc = c
        # Prefer nudging to the right if the slot is taken
        while cc in used and cc < n_cols - 1:
            cc += 1
        # If still taken at the right edge, nudge back left
        if cc in used:
            cc = c
            while cc in used and cc > 0:
                cc -= 1
        used.add(cc)
        resolved[i] = cc

    return resolved


def build_section_matrix(
    sect_names: List[str],
    sect_cols: np.ndarray,
    cpt_df_sorted: pd.DataFrame,
    n_rows: int,
    n_cols: int,
) -> Tuple[np.ndarray, List[Dict[str, int]], List[str]]:
    """Create a (n_rows × n_cols) grid and paint CPT columns.

    Args:
        sect_names: List of CPT names in the current section (in order).
        sect_cols: Column indices in the output grid corresponding to each CPT.
        cpt_df_sorted: CPT data DataFrame sorted by Depth_Index.
        n_rows: Number of rows (depth levels).
        n_cols: Number of columns (distance bins).

    Returns:
        Tuple:
            - grid: Filled matrix (float) with CPT columns painted.
            - painted: List of dicts with {'name': str, 'col': int} for painted CPTs.
            - skipped: List of CPT names that were missing or had wrong length.
    """
    grid = np.zeros((n_rows, n_cols), dtype=float)
    painted: List[Dict[str, int]] = []
    skipped: List[str] = []

    for name, col in zip(sect_names, sect_cols):
        col = int(col)
        if name in cpt_df_sorted.columns:
            vals = cpt_df_sorted[name].to_numpy()
            if len(vals) != n_rows:
                skipped.append(name)
                continue
            # Paint this CPT data into the chosen column (0 = surface/top row)
            grid[:, col] = vals
            painted.append({"name": name, "col": col})
        else:
            skipped.append(name)

    return grid, painted, skipped


def write_section_csv(grid: np.ndarray, out_csv: Path, n_cols: int) -> None:
    """Write a single section matrix to CSV with a Depth_Index column.

    Args:
        grid: (n_rows × n_cols) array of floats (soil class or metric).
        out_csv: Path to the output CSV file.
        n_cols: Number of columns in the grid (used for header naming).
    """
    out_df = pd.DataFrame(grid, columns=[f"x{j:03d}" for j in range(n_cols)])
    out_df.insert(0, "Depth_Index", np.arange(grid.shape[0], dtype=int))
    out_df.to_csv(out_csv, index=False)


def write_manifest(manifest: List[Dict], out_dir: Path) -> None:
    """Persist a human-readable manifest of the generated sections.

    Args:
        manifest: List of section metadata dictionaries.
        out_dir: Directory where 'manifest_sections.csv' will be written.
    """
    rows = []
    for m in manifest:
        rows.append(
            {
                "section_index": m["section_index"],
                "start_idx": m["start_idx"],
                "end_idx": m["end_idx"],
                "first_name": m["first_name"],
                "last_name": m["last_name"],
                "span_m": m["span_m"],
                "left_pad_m": m["left_pad_m"],
                "right_pad_m": m["right_pad_m"],
                "painted_count": m["painted_count"],
                "skipped_count": m["skipped_count"],
                "csv_path": m["csv_path"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "manifest_sections.csv", index=False)


# -----------------------------
# Core pipeline
# -----------------------------
def process_sections(
    coords_df: pd.DataFrame,
    cpt_df: pd.DataFrame,
    out_dir: Path,
    n_cols: int,
    n_rows: int,
    per: int,
    overlap: int,
    left_pad_frac: float,
    right_pad_frac: float,
    from_where: str,
    to_where: str,
) -> List[Dict]:
    """Orchestrate the full pipeline and return a manifest of produced sections.

    Steps:
      1) Sort CPTs by direction and compute distance metrics.
      2) Mark which coordinates have corresponding CPT data columns.
      3) Create overlapping sections and map their distances to columns.
      4) Resolve column collisions and paint grids.
      5) Save each section to CSV and collect manifest info.
      6) Save a distances CSV for traceability.

    Args:
        coords_df: Coordinates DataFrame with ['name','x','y'].
        cpt_df: CPT data DataFrame with ['Depth_Index', <CPT names...>].
        out_dir: Output directory for all generated CSVs.
        n_cols: Number of columns (distance bins) in the output grid.
        n_rows: Number of rows (depth levels) in the output grid.
        per: Number of CPTs per section.
        overlap: Number of CPTs to overlap between adjacent sections.
        left_pad_frac: Left padding as a fraction of the section span.
        right_pad_frac: Right padding as a fraction of the section span.
        from_where: Sort start direction ('west','east','north','south').
        to_where: Sort end direction ('west','east','north','south').

    Returns:
        Manifest: A list of dictionaries describing each generated section.
    """
    ensure_outdir(out_dir)

    # (1) Sort + distances table (clear logging helps validate spacing)
    coords_sorted = sort_cpt_by_coordinates(coords_df, from_where, to_where)
    coords_dist = compute_distances(coords_sorted)
    logger.info("Distances between CPTs (m):")
    logger.info(
        f"\n{coords_dist[['name', 'dist_from_first_m', 'dist_from_prev_m', 'cum_along_m']].to_string()}"
    )

    # (2) Mark names that exist in CPT data (warn about missing ones)
    coords_marked, unmatched = match_names(coords_dist, cpt_df)
    if unmatched:
        logger.warning(
            f"{len(unmatched)} coordinate names have no matching CPT-data columns and will be skipped."
        )

    # (3) Section starts (sliding window)
    cpt_df_sorted = cpt_df.sort_values("Depth_Index")
    n = len(coords_marked)
    starts = section_starts(n, per, overlap)

    manifest: List[Dict] = []

    # (4) Build each section
    for i, start in enumerate(starts, 1):
        end = min(start + per, n)
        sect = coords_marked.iloc[start:end].copy()
        if sect.empty:
            continue

        # Relative distances to the first CPT in the section (explicit loop)
        base_x, base_y = float(sect.iloc[0]["x"]), float(sect.iloc[0]["y"])
        rel: List[float] = []
        for j in range(len(sect)):
            rel.append(
                euclid(
                    base_x, base_y, float(sect.iloc[j]["x"]), float(sect.iloc[j]["y"])
                )
            )
        sect["rel_dist_m"] = rel

        # Map distances to columns + resolve collisions
        cols, span, left_pad, right_pad = map_dist_to_cols(
            dists_rel=sect["rel_dist_m"].to_numpy(),
            left_pad_frac=left_pad_frac,
            right_pad_frac=right_pad_frac,
            n_cols=n_cols,
        )
        cols = resolve_collisions(cols, n_cols)

        # Build and paint the grid for this section
        names = sect["name"].tolist()
        grid, painted, skipped = build_section_matrix(
            sect_names=names,
            sect_cols=cols,
            cpt_df_sorted=cpt_df_sorted,
            n_rows=n_rows,
            n_cols=n_cols,
        )

        # Write section CSV (name includes index)
        out_csv = out_dir / f"section_{i:02d}_cpts_{start+1:03d}_to_{end:03d}.csv"
        write_section_csv(grid, out_csv, n_cols)

        # Collect manifest info
        manifest.append(
            {
                "section_index": i,
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "first_name": names[0],
                "last_name": names[-1],
                "span_m": float(span),
                "left_pad_m": float(left_pad),
                "right_pad_m": float(right_pad),
                "painted_count": len(painted),
                "skipped_count": len(skipped),
                "csv_path": str(out_csv),
                "painted": painted,
                "skipped": skipped,
            }
        )

    # (5) Save distances for traceability
    coords_dist.to_csv(out_dir / "cpt_coords_with_distances.csv", index=False)
    return manifest


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # ----- CONFIG -----
    COORDS_CSV = Path(r"C:\VOW\gis\coords\betuwepand_dike_north.csv")
    CPT_DATA_CSV = Path(
        r"C:\VOW\data\schgan_inputs\testtestest\test_dike_north_input_new.csv"
    )
    OUT_DIR = Path(r"C:\VOW\data\test_outputs")

    N_COLS = 512
    N_ROWS = 32
    CPTS_PER_SECTION = 6
    OVERLAP_CPTS = 2
    LEFT_PAD_FRACTION = 0.10  # Use 0.05 to match original script exactly
    RIGHT_PAD_FRACTION = 0.10  # Use 0.05 to match original script exactly
    DIR_FROM, DIR_TO = "west", "east"
    # ------------------

    ensure_outdir(OUT_DIR)

    coords_df = pd.read_csv(COORDS_CSV)
    cpt_df = pd.read_csv(CPT_DATA_CSV)

    validate_input_files(coords_df, cpt_df, N_ROWS)

    manifest = process_sections(
        coords_df=coords_df,
        cpt_df=cpt_df,
        out_dir=OUT_DIR,
        n_cols=N_COLS,
        n_rows=N_ROWS,
        per=CPTS_PER_SECTION,
        overlap=OVERLAP_CPTS,
        left_pad_frac=LEFT_PAD_FRACTION,
        right_pad_frac=RIGHT_PAD_FRACTION,
        from_where=DIR_FROM,
        to_where=DIR_TO,
    )

    write_manifest(manifest, OUT_DIR)

    logger.info(f"Written {len(manifest)} sections to: {OUT_DIR.resolve()}")
    if any(m["skipped_count"] for m in manifest):
        logger.info(
            "[NOTE] Some CPT names in coords had no matching data columns and were left as zero columns. "
            "Consider aligning names or adding a mapping step."
        )
