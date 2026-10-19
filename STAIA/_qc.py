import os
import spatialdata as sd
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from spatialdata import SpatialData
from matplotlib.colors import LinearSegmentedColormap 
from matplotlib.lines import Line2D
from typing import Optional, Sequence, Tuple 
from scipy.stats import median_abs_deviation
from sklearn.neighbors import NearestNeighbors

"""
Quality Control (QC) utilities for spatial transcriptomics data.

This module provides functions for:
- Plotting outliers
- Recommendation of threshold values
- Control probe counts and code words
- Create plots for decision on QC
- Perform filtering for QC

Designed for AnnData objects derived from SpatialData or similar pipelines.
"""

def plot_outliers(
    adata: AnnData,
    sample_id: str = "cell_id",
    sample: Optional[str] = None,
    metric: str = "detected",
    outliers: Optional[str] = None,
    point_size: float = 2,
    colors: Sequence[str] = ("white", "black"),
    stroke: float = 1.0,
    coord_key: str = "spatial",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
    legend: bool = False,  
    ring_overlay: bool = True, 
) -> plt.Figure:
    """
    Visualize spatial distribution of a QC metric and highlight outliers.

    Parameters
    ----------
    adata : AnnData
        Must contain spatial coordinates in adata.obsm[coord_key] and 
        the specified metric in adata.obs[metric]
    sample_id : str, default = "cell_id"
        Column in adata.obs used to identify samples. If `sample_id` is
        set to 'all', all samples on the slide are shown.
    sample : str or None
        Sample to plot. If None, the first sample is used.
    metric : str
        Observation column used for color mapping.
    outliers : str or None
        Boolean column indicating outlier cell/spots.
    point_size : float, default = 2
        Marker size.
    colors : sequence of str
        Colors of defining the gradient colormap.
    stroke : float, default = 1.0
        Outline width for outline markers.
    coord_key : str, default = "spatial"
        Key in adata.obsm containing spatial coordinates.
    legend : bool, default = False
        Whether to display outlier legend.
    ring_overlay : bool, default = True
        Overlay outliers as red rings

    Returns
    -------
    matplotlib.figure.Figure
    
    Notes
    -----
    Useful for identifying spatial artifacts or low-quality regions. Also after filter, good
    to check performance of filtering with this function.
    """
    # This code is from Spotsweeper_py (Github: https://github.com/danielchen05/spotsweeper_py/blob/master/src/spotsweeper/plot_QC.py)
    # subset adata to the specified sample
    if sample_id == "all" or sample == "all": 
        sample = adata.obs
        adata_sub = adata.copy() #TODO check if 'adata' or 'adata.obs' is needed
    elif sample is None:
        sample = adata.obs[sample_id].unique()[0]
        mask = np.array(adata.obs[sample_id] == sample)
        adata_sub = adata[mask].copy()  # copy to avoid issues
    else:
        mask = np.array(adata.obs[sample_id] == sample)
        adata_sub = adata[mask].copy()

    # extract relevant data to build the plot
    coords = np.array(adata_sub.obsm[coord_key])
    df = pd.DataFrame(data=coords, index=adata_sub.obs_names, columns=["x", "y"])
    df[metric] = adata_sub.obs[metric]

    # add outliers if they are present
    if outliers is not None:
        df["outlier"] = adata_sub.obs[outliers].astype(bool)
    else:
        df["outlier"] = False

    # build custom color scale
    if len(colors) >= 2:
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    else:
        raise ValueError("Color gradient must have at least 2 elements")

    # build plot
    plt.figure(figsize=figsize)  # custom control of figure size

    if ring_overlay:
        # 2-layer: base gradient + red rings (good for Visium)
        base = plt.scatter(
            df["x"],
            df["y"],
            c=df[metric],
            s=point_size**2,
            cmap=cmap,
            linewidths=0,
            rasterized=True,
        )

        # overlay: outliers as red rings
        highlighted = df[df["outlier"]]
        if outliers is not None and len(highlighted) > 0:
            plt.scatter(
                highlighted["x"],
                highlighted["y"],
                facecolors="none",
                edgecolors="red",
                linewidths=stroke * 1.5,
                s=(point_size * 1.4) ** 2,
            )
    else:
        # 1-layer: single scatter with red edges for outliers (better for Visium HD)
        base = plt.scatter(
            df["x"],
            df["y"],
            c=df[metric],
            s=point_size**2,
            cmap=cmap,
            edgecolors=["red" if i else "none" for i in df["outlier"]],
            linewidths=stroke,
        )

    plt.title(title if title is not None else f"Sample: {sample}")  # controlled title
    plt.axis("equal")
    plt.gca().invert_yaxis()  # to match tissue orientation
    plt.colorbar(base, label=metric)  # use base instead of scatter 
    plt.xlabel("x")
    plt.ylabel("y")

    # optional legend: outlier vs non-outlier
    if legend and outliers is not None:
        legend_elements = [
            Line2D([0], [0], marker="o", linestyle="None",
                markerfacecolor="white", markeredgecolor="black",
                markersize=6, label="Non-outlier"),
            Line2D([0], [0], marker="o", linestyle="None",
                markerfacecolor="white", markeredgecolor="red",
                markersize=6, label="Outlier"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", frameon=True)

    plt.tight_layout()
    # plt.savefig('assigning_labels/QCplotv2.png', dpi=500)
    return plt.gcf()
    

def perform_quality_control(
        adata: AnnData,
        thr: dict,
) -> AnnData:
    """
    Filter cells based on QC thresholds.

    Parameters
    ----------
    thr : dict
        Dictionary containing:
        - min_genes
        - max_genes
        - min_counts
        - max_counts
    adata : AnnData 
        Must contain:
        - total_counts
        - n_genes_by_counts (computed if missing)
    
    Returns
    -------
    AnnData
        Filtered and normalized dataset.

    Notes
    -----
    If cell_area is available, the density is computed and added to the AnnData object.

    Applies:
    - Library size normalization
    - Log1p transformation
    """
    # add n_genes_by_count attribute
    if 'n_genes_by_counts' not in adata.obs:
        sc.pp.calculate_qc_metrics(
            adata,
            inplace=True,
            percent_top=None,
            log1p=False
        )

    # add density for segmentation aware QC
    try:
        adata.obs["density"] = adata.obs["total_counts"] / adata.obs["cell_area"]
    except KeyError:
        print(
            "Density has not been added to the AnnData object, because either the total_counts "
            "or cell_area is not present in the AnnData object which is mandatory for computation."
            )
        
    # filtering based on thresholds
    vals_thr = ("min_genes", "max_genes", "min_counts", "max_counts")
    if all(name in thr for name in vals_thr):
        mask = (
            (adata.obs["n_genes_by_counts"] >= thr["min_genes"]) &
            (adata.obs["n_genes_by_counts"] <= thr["max_genes"]) &
            (adata.obs["total_counts"] >= thr["min_counts"]) &
            (adata.obs["total_counts"] <= thr["max_counts"])
        )
    else:
        print("A threshold value is missing. Expected threshold values: min_genes, max_genes, min_counts and max_counts.")

    print(f"Cells before filtering: {adata.n_obs}")
    print(f"Cells after filtering: {mask.sum()}")

    # # Normalize and transform
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    return adata[mask].copy()


def control_probes_codew(
        adata: AnnData
) -> tuple[float, float]:
    """ 
    Compute percentage of control probes and control codewords.

    Parameters
    ----------
    adata : Anndata
        Annotated data matrix used for computations.

    Returns
    -------
    tuple of float
        (control_probe_percentage, control_codeword_percentage)

    Notes
    -----
    Required columns:
    - control_probe_counts
    - control_codeword_counts
    - total_counts
    """
    if "control_probe_counts" in adata.obs:
        cprobes = (
        adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100
        )
    else:
        cprobes = ("Column `control_probe_counts` not present in adata.")

    if "control_codeword_counts" in adata.obs:
        cwords = (
            adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100
        )
    else:
        cwords = ("Column `control_codeword_counts` not present in adata.")

    return (cprobes, cwords)


def plot_qc_metrics(
        adata: AnnData,
        save: bool = True,
        output_path : str | None = None
):
    """
    Generate standard QC diagnostic plots.

    Plots include:
    - Total counts per cell
    - Genes per cell
    - Nucleus-to-cell are ratio (if available)
    - Log-transformed distributions
    - Counts vs genes scatter

    Parameters
    ----------
    adata : AnnData
        Must contain total_counts and optionally area metrics.
    save : bool, default = True
        Save figure to visuals/cell_statistics.png
    output_path : str, optional
        Output path, if not given, current working directory will be used.

    Returns
    -------
    matplotlib.figure.Figure
    """
    obs = adata.obs
    if 'n_genes_by_counts' not in adata.obs:
        print("The column 'n_genes_by_counts' is not present in the data. Computing this column...")
        sc.pp.calculate_qc_metrics(
            adata,
            inplace=True,
            percent_top=None,
            log1p=False
        )
        print("Column 'n_genes_by_counts' is computed. Generating visualizations...")

    # QC plots - cell-level statistics
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Cell-Level Quality Metrics', fontsize=16)

    # Total transcripts per cell
    sns.histplot(obs["total_counts"], kde=False, ax=axs[0, 0], bins=50)
    axs[0, 0].set_title("Total Transcripts per Cell")
    axs[0, 0].set_xlabel("Total counts")
    axs[0, 0].axvline(100, color='r', linestyle='--', label='Filter threshold (100)')
    axs[0, 0].legend()

    # Unique genes per cell
    sns.histplot(obs["n_genes_by_counts"], kde=False, ax=axs[0, 1], bins=50)
    axs[0, 1].set_title("Unique Genes per Cell")
    axs[0, 1].set_xlabel("Number of genes")

    # Nucleus ratio
    if 'nucleus_area' in obs.columns and 'cell_area' in obs.columns:
        nucleus_ratio = obs["nucleus_area"] / obs["cell_area"]
        sns.histplot(nucleus_ratio, kde=False, ax=axs[0, 2], bins=50)
        axs[0, 2].set_title("Nucleus Ratio")
        axs[0, 2].set_xlabel("Nucleus / Cell area")
    else:
        axs[0, 2].text(0.5, 0.5, 'Nucleus area not available', ha='center', va='center')
        axs[0, 2].set_title("Nucleus Ratio")

    n_genes = obs["n_genes_by_counts"].values
    total_counts = obs["total_counts"].values
    # complexity = n_genes / (total_counts + 1e-9)

    bins = 100
    n_genes_plot = np.log10(n_genes + 1)
    total_counts_plot = np.log10(total_counts + 1)
    xlabel_genes = "log10(n_genes + 1)"
    xlabel_counts = "log10(total_counts + 1)"


    # counts per cell
    sns.histplot(total_counts_plot, kde=False, ax=axs[1, 0], bins=bins)
    # axes[0, 1].hist(total_counts_plot, bins=bins)
    axs[1, 0].set_title("Counts per cell")
    axs[1, 0].set_xlabel(xlabel_counts)

    # genes per cell
    sns.histplot(n_genes_plot, kde=False, ax=axs[1, 1], bins=bins)
    # axes[0, 0].hist(n_genes_plot, bins=bins)
    axs[1, 1].set_title("Genes per cell")
    axs[1, 1].set_xlabel(xlabel_genes)

    # scatter
    axs[1, 2].scatter(total_counts_plot, n_genes_plot, s=5, alpha=0.3)
    axs[1, 2].set_xlabel(xlabel_counts)
    axs[1, 2].set_ylabel(xlabel_genes)
    axs[1, 2].set_title("Counts vs Genes")

    plt.show()

    if save:
        fig.tight_layout()
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "cell_statistics.png")
        fig.savefig(output_path, format='png', dpi=300, bbox_inches="tight")
        # plt.savefig('visuals/cell_statistics.png', format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"The distribution plots are saved in the current running directory: {output_path}")
    return fig
    

# def recommend_threshold(
#         adata: AnnData
# ) -> dict:
#     """
#     Suggest QC filtering thresholds based on percentile ranges.

#     Parameters
#     ----------
#     adata : AnnData
#         Must contain:
#         - total_counts
#         - n_genes_by_counts
    
#     Returns
#     -------
#     dict
#         Recommended thresholds:
#         {
#             "min_genes"
#             "max_genes"
#             "min_counts"
#             "max_counts"
#         }

#     Notes
#     -----
#     Thresholds are based on the 1st and 99th percentiles.
#     """
#     obs = adata.obs
#     if "n_genes_by_count" not in adata.obs:
#         sc.pp.calculate_qc_metrics(
#             adata,
#             inplace=True,
#             percent_top=None,
#             log1p=False
#         )
#     genes = obs["n_genes_by_counts"].values
#     counts = obs["total_counts"].values

#     thresholds = {
#         "min_genes": int(np.percentile(genes, 1)),
#         "max_genes": int(np.percentile(genes, 99)),
#         "min_counts": int(np.percentile(counts, 1)),
#         "max_counts": int(np.percentile(counts, 99))
#     }
#     return thresholds

import numpy as np
import scanpy as sc

def recommend_threshold(
    adata: AnnData, 
    nmads: int = 3, 
    min_genes_floor: int = 5, 
    min_counts_floor: int =10
    ):
    """
    Suggest QC filtering thresholds using MAD-based outlier detection on QC metrics.
    
    Computes thresholds per slide from the data itself, so it adapts
    across samples without manual tuning. Call SpatioFlow.perform_quality_control() to 
    perform the actual filtering.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object (pre QC metrics calculation)
    nmads : int, default = 3
        Number of MADs away from median to set threshold.
        3 is the standard; lower = stricter filtering.
    min_genes_floor : int, default = 5
        Hard minimum for n_genes lower bound (prevents
        negative threshold on very sparse data).
    min_counts_floor : int, default = 10 
        Hard minimum for total_counts lower bound.

    Returns
    -------
    thr : dict
        dict of the thresholds that need to be applied, containing:
            - min genes
            - max genes
            - min_counts
            - max_counts
    """
    # Ensure QC metrics exist
    sc.pp.calculate_qc_metrics(
        adata,
        percent_top=None,
        inplace=True,
    )

    def _mad_bounds(series, direction="both", floor=None):
        med = np.median(series)
        mad = np.median(np.abs(series - med))
        lower = med - nmads * mad
        upper = med + nmads * mad
        if floor is not None:
            lower = max(floor, lower)
        if direction == "upper":
            return None, upper
        if direction == "lower":
            return lower, None
        return lower, upper

    # Compute per-metric bounds
    min_genes, max_genes = _mad_bounds(
        adata.obs["n_genes_by_counts"], floor=min_genes_floor
    )
    min_counts, max_counts = _mad_bounds(
        adata.obs["total_counts"], floor=min_counts_floor
    )

    thr = {
        "min_genes":   min_genes,
        "max_genes":   max_genes,
        "min_counts":  min_counts,
        "max_counts":  max_counts
    }
    return thr


def is_outlier(
        vec, 
        nmads: int = 3, 
        log : bool = False, 
        direction: str = 'both'
    ) -> np.ndarray:
    """
    Check if column in adata.obs has outliers.

    This function detects extreme values in a numeric vector based on
    deviations from the median. The MAD-based approach is robust to
    skewed distributions and commonly used for quality control in
    single-cell and spatial transcriptomics.

    Parameters
    ----------
    vec : adata.obs.column
        Column of adata.obs containing counts to check for outliers.
    nmads : int, default = 3
        Number of median absolute deviations used to define the
        outlier threshold.
    log : bool, default=False
        Whether to apply log1p transformation before computing outliers.
        Recommended for count-based metrics.
    direction : {"lower", "higher", "both"}, default="both"
        Direction of outlier detection:
        - "lower": values below the lower threshold
        - "higher": values above the upper threshold
        - "both": values outside the threshold range

    Returns
    -------
    numpy.ndarray (bool)
        Boolean mask indicating outlier observations.

    Notes
    -----
    Thresholds are defined as:

    lower = median - nmads * MAD  
    upper = median + nmads * MAD

    This method is robust to heavy-tailed and non-normal distributions.

    Examples
    --------
    >>> adata.obs["low_counts_outlier"] = is_outlier(
    ...     adata.obs["total_counts"],
    ...     nmads=3,
    ...     log=True,
    ...     direction="lower"
    ... )
    """
    arr = np.log1p(vec) if log else np.asarray(vec)
    med = np.nanmedian(arr)
    mad = median_abs_deviation(arr, nan_policy='omit')
    lower = med - nmads * mad
    upper = med + nmads * mad
    if direction == 'lower':
        return arr < lower
    elif direction == 'higher':
        return arr > upper
    else:
        return (arr < lower) | (arr > upper)
    

def detect_outlier(
        adata: AnnData,
) -> AnnData:
    """
    Detect spatially local outliers based on total counts.

    Uses nearest neighbors in spatial coordinates and computes a
    robust z-score (median/MAD) within each neighborhood.

    Parameters
    ----------
    adata : AnnData
        Must contain:
        - ``adata.obs["total_counts"]``
        - spatial coordinates in ``adata.obsm["spatial"]``

    Returns
    -------
    AnnData
        Same object with added columns:
        - ``log_total_counts``
        - ``log_total_counts_z``
        - ``log_total_counts_outliers``

    Notes
    -----
    Useful for detecting local technical failures or tissue damage.
    """
    metric = "log_total_counts"
    n_neighbors = 36
    cutoff = 3.0
    direction = "lower"
    adata.obs["log_total_counts"] = np.log1p(adata.obs["total_counts"]).copy()

    # initialize output columns
    adata.obs[f"{metric}_z"] = 0.0
    adata.obs[f"{metric}_outliers"] = False

    coords = adata.obsm["spatial"]
    values = adata.obs[metric].to_numpy()

    # nearest neighbors in spatial space
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    neigh_idx = nn.kneighbors(return_distance=False)

    # robust z-score using median and MAD
    def robust_z(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return 0.0
        return 0.6745 * (x[0] - med) / mad  # x[0] = focal spot

    # compute local z for each spot
    z_scores = np.zeros(len(values))
    for i in range(len(values)):
        idx = np.concatenate(([i], neigh_idx[i]))  # self + neighbors
        z_scores[i] = robust_z(values[idx])

    # determine outliers
    if direction == "lower":
        outliers = z_scores < -cutoff
    elif direction == "higher":
        outliers = z_scores > cutoff
    else:
        outliers = (z_scores > cutoff) | (z_scores < -cutoff)

    # save results
    adata.obs[f"{metric}_z"] = z_scores
    adata.obs[f"{metric}_outliers"] = outliers 
    return adata

