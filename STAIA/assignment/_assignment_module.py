# importing the packages
from __future__ import annotations
import matplotlib

# NOTE: the backend is intentionally left for the notebook to choose,
# rather than forced here. Vertical/horizontal/diagonal splits work fine
# under the default inline backend, but the freehand split tool needs a
# live, event-driven backend to receive mouse events. Before creating a
# SplitSession, run in a notebook cell:
#     %matplotlib widget
# (requires `pip install ipympl`). If you only ever use straight-line
# splits, `%matplotlib inline` continues to work as before.

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import closing, disk
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.morphology import erosion, disk as morph_disk
import scipy.ndimage as ndi
from anndata import AnnData


def _load_xenium(
        adata: AnnData,
        output_dir: str
) -> pd.DataFrame:
    """
    Read data from input.

    This function reads the given Xenium path and checks if the input file exists. Additionally, it
    checks the presence of x/y columns and returns a dataframe with this information.

    Parameters
    ----------
    adata : str
        Annotated data matrix containing x and y values.
    output_dir: str
        Path to the desired output directory.
    
    Returns
    -------
    df : pd.DataFrame
        Dataframe containing x/y information of the cells.

    Notes
    -----
    This function does not use an AnnData object, it looks at the cells from the Xenium output to
    obtain their coordinates for further processing.
    """
    print(f"Loaded {len(adata.obs):,} rows.")
    x_candidates = ["x_centroid", "x_location", "X", "x", "X_centroid", "centroid_x"]
    y_candidates = ["y_centroid", "y_location", "Y", "y", "Y_centroid", "centroid_y"]

    if not any(col in adata.obs.columns for col in x_candidates):
        adata.obs['x_centroid'] = adata.obsm['spatial'][:, 0]
        adata.obs['y_centroid'] = adata.obsm['spatial'][:, 1]


    os.makedirs(output_dir, exist_ok=True)
    df = adata.obs.copy()

    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if x_col is None or y_col is None:
        print("Could not auto-detect x/y columns. Available:", list(df.columns))

    df = df.rename(columns={x_col: "x", y_col: "y"})
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    return df


def plot_samples(
        df : pd.DataFrame, 
        cluster_ids, 
        highlight: int = None,
         conditions: dict = None, 
         ax: matplotlib.axes = None
):
    """
    Visualize clustered cells in a 2D scatter plot of the slide.

    This function plots cells from a dataframe containing spatial coordinates
    and cluster assignments. Each cluster is rendered with a unique color, 
    optionally highlighting one cluster and annotating clusters with labels.

    Cluster centroids are labeled on the plot, and a legend describing
    cluster size and condition metadata is displayed.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing x/y coordinates of each points, including the 
        cluster/sample identifier for each point.

    cluster_ids : iterable of int
        List or iterable of cluster(s) to plot. each ID corresponds to a 
        group of points.
    
    highlight : int, optional
        A cluster ID to visually emphasize. Highlighted clusters are plotted
        with larger markers, higher opacity, and a different background color.
        Default = None

    conditions : dict[int, str], optional
        If provided, the condition text is appended to the cluster label and 
        shown in both the annotation and legend.

    ax : Axes, optional
        Existing matplotlib axes to plot on.

    Notes
    -----
    - Each cluster in cluster_ids is plotted with a color from the global CMAP colormap
    - Clustered centroids are computed from the mean x and y coordinates and annotated 
      directly on the plot.
      - A legend is provided showing different colors per cluster and which cluster
        is
    """
    standalone = ax is None
    CMAP = plt.get_cmap("tab20")
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#13131f")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#6e6f88", linewidth=0.5, linestyle="--", alpha=0.8)


    # Minor gridlines (in between)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#6e6f88", linewidth=0.5, linestyle=":", alpha=0.8)
    
    noise = df[df["sample_id"] == -1]
    if len(noise):
        ax.scatter(noise["x"], noise["y"], s=0.3, c="#444455",
                   alpha=0.3, linewidths=0, rasterized=True)
    legend_handles = []
    for cid in cluster_ids:
        pts = df[df["sample_id"] == cid]
        col = CMAP(cid % 20)
        is_sel = (cid == highlight)
        ax.scatter(pts["x"], pts["y"],
                   s=1.5 if is_sel else 0.6,
                   c=[col], alpha=0.9 if is_sel else 0.45,
                   linewidths=0, rasterized=True)
        cx, cy = pts["x"].mean(), pts["y"].mean()
        cond_str = conditions.get(cid, "") if conditions else ""
        lbl = f"S{cid}" + (f"\n{cond_str}" if cond_str else "")
        fc = "#3a7bd5" if is_sel else "#2a2a3e"
        ax.annotate(lbl, (cx, cy), color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, alpha=0.85))
        legend_handles.append(
            mpatches.Patch(color=col,
                           label=f"S{cid}: {cond_str or 'unlabelled'}  ({len(pts):,} cells)"))
    ax.legend(handles=legend_handles,
              loc="upper left",
              bbox_to_anchor=(1.01, 1),
              borderaxespad=0,
              framealpha=0.4,
              fontsize=8,
              facecolor="#1e1e2e",
              labelcolor="white")
    ax.invert_yaxis()
    ax.tick_params(colors="grey")
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.set_xlabel("x (um)", color="grey")
    ax.set_ylabel("y (um)", color="grey")
    ax.set_title(f"{len(cluster_ids)} sample(s) detected", color="white", fontsize=11)
    if standalone:
        plt.tight_layout()
        plt.show(fig)


def detect_samples_watershed( # TODO: add these options to the functional API for testing purposes
        df: pd.DataFrame, 
        pixel_size_um: int = 20, 
        blur_sigma: int = 3,
        closing_radius: int = 5, 
        erosion_radius: int = 30, 
        min_cells: int = 500
):
    """ 
    Detect spatial samples using grid-based watershed segmentation.

    This function converts spatial cell coordinates into a 2D density grid,
    performing smoothing and morphological preprocessing, and then applies 
    watershed segmentation to identify spatially separated samples.
    Cells are assigned to segmented regions and small regions are filtered
    based on a minimum cell count threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing cell coordinates (x/y)
    
    pixel_size_um : int, default=20
        Pixel size of the intermediate spatial grid in micrometers.
        Larger values produce a coarser grid and faster processing but
        may reduce segmentation resolution.
    
    blur_sigma : int, default=3
        Standard deviation of the Gaussian filter applied to the density
        grid. Controls smoothing of the spatial density map.

    closing_radius : int, default=5
        Radius of the morphological closing operation used to fill small
        gaps in the binary mask.

    erosion_radius : int, default=30
        Radius of the morphological erosion used to generate seed regions
        for watershed segmentation. Larger values enforce stronger separation 
        between nearby samples.

    min_cells : int, default=500
        Minimum number of cells required for a segmented region to be considered
        a valid sample. Regions with fewer cells are laballed as noise.

    Returns
    -------
    final_labels : numpy.ndarray
        Array of length ``len(df)`` assigning a sample ID to each cell.
        Valid samples are labeled with consecutive integers starting
        from 0. Cells not assigned to a valid region are labeled ``-1``.

    labeled_image : numpy.ndarray
        2D watershed segmentation mask where each pixel contains the
        watershed component label.

    x_min : float
        Minimum x-coordinate of the spatial domain. Useful for mapping
        between grid coordinates and original spatial coordinates.

    y_min : float
        Minimum y-coordinate of the spatial domain.

    pixel_size_um : int
        Pixel size used for the spatial grid. Required for converting
        between grid indices and real-world coordinates.

    Notes
    -----
    - The algorithm assumes spatially separated samples with clear
      density gaps between them.
    - The erosion step is critical for separating adjacent samples
      and may require tuning depending on sample spacing.
    - Grid resolution (``pixel_size_um``) and smoothing (``blur_sigma``)
      strongly influence segmentation quality.

    The workflow follows these steps:
    1. Convert x/y cell coordinates into a coarse pixel grid
    2. Smooth the grid using a Gaussian filter to create a density map
    3. Threshold the density map to obtain a binary mask
    4. Apply morphological closing and hole filling to clean the mask.
    5. Erode the mask to create seed regions separating nearby samples
    6. Use distance transform + watershed to segment individual samples.
    7. Assign each cell to a segmented region.
    8. Remove regions containing fewer than the minimum amount of cells defined.
    """
    coords = df[["x", "y"]].values
    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

    nx = int(np.ceil((x_max - x_min) / pixel_size_um)) + 1
    ny = int(np.ceil((y_max - y_min) / pixel_size_um)) + 1
    print(f"Grid: {nx} x {ny} px  ({pixel_size_um} µm/px)")

    xi = ((coords[:, 0] - x_min) / pixel_size_um).astype(int)
    yi = ((coords[:, 1] - y_min) / pixel_size_um).astype(int)
    grid = np.zeros((ny, nx), dtype=np.float32)
    np.add.at(grid, (yi, xi), 1)

    blurred = gaussian_filter(grid, sigma=blur_sigma)
    threshold = blurred.max() * 0.05
    binary = blurred > threshold
    binary = closing(binary, disk(closing_radius))
    binary = binary_fill_holes(binary)

    # perform erosion to separate close samples
    eroded = erosion(binary, disk(erosion_radius))
    seeds  = label(eroded)
    print(f"Seeds after erosion: {seeds.max()}")

    # create watershed surface by distance transforming
    distance = ndi.distance_transform_edt(binary)
    labeled_image = watershed(-distance, seeds, mask=binary)
    print(f"Components after watershed: {labeled_image.max()}")

    # map cells to labels
    cell_labels = labeled_image[yi, xi] - 1
    unique, counts = np.unique(cell_labels[cell_labels >= 0], return_counts=True)
    keep  = unique[counts >= min_cells]
    mask  = np.isin(cell_labels, keep)
    cell_labels[~mask] = -1
    remap = {old: new for new, old in enumerate(sorted(keep))}
    final_labels = np.array([remap.get(l, -1) for l in cell_labels])

    print(f"\n✓ {len(remap)} sample(s)  |  {(final_labels==-1).sum():,} unassigned cells")
    return final_labels, labeled_image, x_min, y_min, pixel_size_um


SplitRecord = dict  # {"type", "sample_id", "single_value"?  "x_points"? "y_points"?}
SplitHistory = list[SplitRecord]


def _split_by_curve(
    cx: np.ndarray,
    cy: np.ndarray,
    x_points: list[float],
    y_points: list[float],
) -> np.ndarray:
    """
    Classify points relative to an open, possibly curved polyline.

    For each point, this finds the closest point on the polyline (checked
    across all of its segments) and uses the perpendicular signed side of
    that nearest segment to decide which side of the cut the point falls
    on. This generalizes the straight-line cross-product test used for
    vertical/horizontal/diagonal splits to an arbitrary hand-drawn curve,
    without requiring the curve to be closed into a polygon first — which
    matters here since a hand-drawn cut is naturally open (it starts and
    ends at the edge of a sample, it doesn't loop back on itself).

    Parameters
    ----------
    cx, cy : np.ndarray
        Coordinates of the cells belonging to the sample being split.
    x_points, y_points : list[float]
        Vertices of the cut polyline, in the order they were drawn.
        Must contain at least two points each.

    Returns
    -------
    np.ndarray of bool
        ``True`` for cells on the "positive" side of the curve, ``False``
        for the other side. Matches the ``cross >= 0`` convention used
        elsewhere in this module for vertical/horizontal/diagonal splits.

    Notes
    -----
    This assumes the drawn curve reasonably separates the sample into two
    sides (i.e. it spans across the sample rather than looping around a
    small region within it). For a curve that self-intersects or loops,
    the "nearest segment" side test is not guaranteed to match a
    conventional inside/outside polygon definition.
    """
    pts = np.column_stack([np.asarray(x_points, dtype=float),
                            np.asarray(y_points, dtype=float)])
    cells = np.column_stack([np.asarray(cx, dtype=float),
                              np.asarray(cy, dtype=float)])

    if len(pts) < 2:
        raise ValueError("A freehand cut needs at least two points.")

    n_cells = len(cells)
    best_dist2 = np.full(n_cells, np.inf)
    best_side  = np.zeros(n_cells)

    for p1, p2 in zip(pts[:-1], pts[1:]):
        seg = p2 - p1
        seg_len2 = seg @ seg
        if seg_len2 == 0:
            continue  # duplicate consecutive points (e.g. a mouse-jitter
                      # artifact) — skip the degenerate segment

        # Closest point on the segment to each cell (clamped to the segment)
        t = np.clip(((cells - p1) @ seg) / seg_len2, 0.0, 1.0)
        proj = p1 + t[:, None] * seg
        dist2 = np.sum((cells - proj) ** 2, axis=1)

        # Signed perpendicular side of *this* segment
        cross = seg[0] * (cells[:, 1] - p1[1]) - seg[1] * (cells[:, 0] - p1[0])
        side = np.sign(cross)

        closer = dist2 < best_dist2
        best_dist2[closer] = dist2[closer]
        best_side[closer]  = side[closer]

    return best_side >= 0


#  split helpers
def _do_one_split(
    df: pd.DataFrame,
    sample_id: int,
    split_type: str,
    single_value: float | None = None,
    x_points: list[float] | None = None,
    y_points: list[float] | None = None,
) -> pd.DataFrame:
    """
    Split a single sample into two new samples along a user-defined cut.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``x``, ``y``, ``sample_id``.
    sample_id : int
        The raw sample ID to split.
    split_type : {"vertical", "horizontal", "diagonal", "freehand"}
        Type of geometric split. "freehand" accepts an arbitrary polyline
        (two or more points, in drawing order) tracing a possibly curved
        cut, e.g. hand-drawn with the mouse in the notebook UI.
    single_value : float, optional
        Cut position for vertical (x = value) or horizontal (y = value) splits.
    x_points : list[float], optional
        x-coordinates defining the cut line. Exactly two values for a
        "diagonal" split; two or more (in drawing order) for "freehand".
    y_points : list[float], optional
        y-coordinates defining the cut line, paired index-for-index with
        ``x_points``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the target sample replaced by two new samples.
        New sample IDs are ``max(sample_id) + 1`` and ``max(sample_id) + 2``.
    """
    df = df.copy()
    is_target = df["sample_id"] == sample_id

    if is_target.sum() == 0:
        print(f"  WARNING: sample_id {sample_id} not found — skipping.")
        return df

    cx = df.loc[is_target, "x"]
    cy = df.loc[is_target, "y"]

    if split_type == "vertical":
        cross = cx - single_value
    elif split_type == "horizontal":
        cross = cy - single_value
    elif split_type == "diagonal":
        x1, x2 = x_points
        y1, y2 = y_points
        cross = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    elif split_type == "freehand":
        if x_points is None or y_points is None or len(x_points) < 2:
            raise ValueError("Freehand split requires at least two (x, y) points.")
        on_positive_side = _split_by_curve(cx.to_numpy(), cy.to_numpy(), x_points, y_points)
        cross = pd.Series(np.where(on_positive_side, 1.0, -1.0), index=cx.index)
    else:
        raise ValueError(f"Unknown split_type: {split_type!r}. "
                         "Expected 'vertical', 'horizontal', 'diagonal', or 'freehand'.")

    max_id = int(df["sample_id"].max())
    side_a = is_target & pd.Series(cross >= 0, index=df.index).fillna(False)
    side_b = is_target & pd.Series(cross <  0, index=df.index).fillna(False)

    df.loc[side_a, "sample_id"] = max_id + 1
    df.loc[side_b, "sample_id"] = max_id + 2
    return df


def _renumber(df: pd.DataFrame, row_tolerance: float = None) -> tuple[pd.DataFrame, list[int]]:
    """
    Renumber sample IDs sequentially, ordered row by row (top → bottom, left → right).

    Cells labelled ``-1`` (noise/unassigned) are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``x``, ``y``, ``sample_id``.
    row_tolerance : float, optional
        Max y-centroid difference to consider two samples in the same row.
        Defaults to 1/3 of the median sample height if not provided.

    Returns
    -------
    df : pd.DataFrame
        Copy with remapped ``sample_id`` values starting from 0.
    new_ids : list[int]
        Sorted list of valid sample IDs after renumbering.
    """
    valid_ids = [c for c in df["sample_id"].unique() if c != -1]
    if not valid_ids:
        return df.copy(), []

    valid_df = df[df["sample_id"] != -1]

    # Compute x and y centroids per sample
    centroids = valid_df.groupby("sample_id")[["x", "y"]].mean()

    # Auto-compute row_tolerance from median sample height if not given
    if row_tolerance is None:
        heights = valid_df.groupby("sample_id")["y"].apply(lambda s: s.max() - s.min())
        row_tolerance = heights.median() / 3

    # Sort by y first, then assign row groups using tolerance
    centroids = centroids.sort_values("y")
    rows = []
    current_row = [centroids.index[0]]
    current_y = centroids.loc[centroids.index[0], "y"]

    for sid in centroids.index[1:]:
        y = centroids.loc[sid, "y"]
        if abs(y - current_y) <= row_tolerance:
            current_row.append(sid)
        else:
            rows.append(current_row)
            current_row = [sid]
            current_y = y
    rows.append(current_row)

    # Within each row, sort left to right by x centroid
    sorted_ids = []
    for row in rows:
        row_sorted = sorted(row, key=lambda sid: centroids.loc[sid, "x"])
        sorted_ids.extend(row_sorted)

    remap = {old: new for new, old in enumerate(sorted_ids)}

    df = df.copy()
    df["sample_id"] = df["sample_id"].apply(lambda x: remap.get(x, -1))
    new_ids = sorted(c for c in df["sample_id"].unique() if c != -1)
    return df, new_ids


def replay_all_splits(base_df, history):
    df = base_df.copy()
    for entry in history:
        df = _do_one_split(
            df,
            sample_id    = entry["sample_id"],
            split_type   = entry["type"],
            single_value = entry.get("single_value"),
            x_points     = entry.get("x_points"),
            y_points     = entry.get("y_points"),
        )
        df, _ = _renumber(df)  # keep IDs as 0,1,2... after every split
    return _renumber(df)

# Split-record constructors (keep record creation consistent)
def _make_vertical_record(sample_id: int, x: float) -> SplitRecord:
    """Return a split record for a vertical cut at x."""
    return {"type": "vertical", "sample_id": sample_id, "single_value": x}


def _make_horizontal_record(sample_id: int, y: float) -> SplitRecord:
    """Return a split record for a horizontal cut at y."""
    return {"type": "horizontal", "sample_id": sample_id, "single_value": y}


def _make_diagonal_record(
    sample_id: int,
    x1: float, y1: float,
    x2: float, y2: float,
) -> SplitRecord:
    """Return a split record for a diagonal cut through two points."""
    return {
        "type": "diagonal",
        "sample_id": sample_id,
        "x_points": [x1, x2],
        "y_points": [y1, y2],
    }


def _make_freehand_record(
    sample_id: int,
    x_points: list[float],
    y_points: list[float],
) -> SplitRecord:
    """Return a split record for a freehand (hand-drawn, possibly curved) cut."""
    if len(x_points) < 2 or len(x_points) != len(y_points):
        raise ValueError(
            "Freehand record needs matching, non-empty x/y point lists (>= 2 points)."
        )
    return {
        "type": "freehand",
        "sample_id": sample_id,
        "x_points": list(x_points),
        "y_points": list(y_points),
    }


def run_watershed( 
        adata: AnnData,
        output_dir: str,
        pixel_size_um: int = 20,
        blur_sigma: int = 3,
        closing_radius: int = 5, 
        erosion_radius: int = 30, 
        min_cells: int = 500
        ):
    """
    Run the watershed segmentation.

    Parameters
    ----------
    adata : str
        Annotated data matrix containing x and y coordinates.
    output_dir : str
        Path to the desired output directory.
    pixel_size_um : int, default=20
        Pixel size of the intermediate spatial grid in micrometers.
        Larger values produce a coarser grid and faster processing but
        may reduce segmentation resolution.
    
    blur_sigma : int, default=3
        Standard deviation of the Gaussian filter applied to the density
        grid. Controls smoothing of the spatial density map.

    closing_radius : int, default=5
        Radius of the morphological closing operation used to fill small
        gaps in the binary mask.

    erosion_radius : int, default=30
        Radius of the morphological erosion used to generate seed regions
        for watershed segmentation. Larger values enforce stronger separation 
        between nearby samples.

    min_cells : int, default=500
        Minimum number of cells required for a segmented region to be considered
        a valid sample. Regions with fewer cells are laballed as noise.


    Returns
    -------
    df_test : pandas.DataFrame
        Dataframe containing new cell labels after watershed.

    Notes
    -----
    X and Y coordinates of the centroids should be present in adata.obs.
    This can be obtained by running:
    - `adata.obs['centroid_x'] = adata.obsm['spatial'][:, 0]`
    - `adata.obs['centroid_y'] = adata.obsm['spatial'][:, 1]`

    The algorithm will recognize these variations of the coordinate names in 
    adata.obs: centroid_x, x, X, x_centroid, x_location
    """
    df_raw = _load_xenium(adata=adata, output_dir=output_dir)

    test_labels, test_img, *_ = detect_samples_watershed(df_raw, pixel_size_um, blur_sigma, closing_radius, erosion_radius, min_cells)

    df_test = df_raw.copy()
    df_test["sample_id"] = test_labels

    df_test, test_ids = _renumber(df_test) 

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#1e1e2e")
    axes[0].imshow(test_img, origin="upper", cmap="tab20", aspect="auto")
    axes[0].set_title("Watershed segmentation mask", color="white")
    axes[0].tick_params(colors="grey")
    plot_samples(df_test, test_ids, ax=axes[1])
    plt.tight_layout()
    # fig.savefig(
    #     f"assigning_labels/watershed_segmentationv2.png",
    #     dpi=500,                # increase to 600 for very high resolution
    #     bbox_inches="tight",    # trims whitespace
    #     facecolor=fig.get_facecolor()  # preserves dark background
    # )
    plt.show(fig)
    return df_test


def manual_split_samples(df):
    RESET = True

    if RESET:
        split_history = []
        df_split_base = df.copy()
        for var in ["df", "test_ids"]:
            if var in globals():
                del globals()[var]