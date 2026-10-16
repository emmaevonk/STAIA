from matplotlib import pyplot as plt
import squidpy as sq
import scanpy as sc
import spatialdata as sd
import spatialdata_plot
from spatialdata import SpatialData
from anndata import AnnData
from matplotlib.lines import Line2D

"""
Visualization and exploratory analysis utilities for spatial transcriptomics.

This module provides plotting functions for:
- Spatial visualization
- Quality control distributions
- Dimensionality reduction
- Marker gene ranking
- Image exploration and cropping

Designed for AnnData and SpatialData objects
"""

def plot_image(
        adata: AnnData,
        colors: list = ["leiden"]
):
    """
    Plot spatial scatter colored by gene expression or metadata
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates in 
        adata.obsm["spatial"]
    colors : list of str
        Features or observation columns used for coloring. 
        Examples: gene names, "leiden"

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the spatial scatter plot.

    Notes
    -----
    Wrapper around sq.pl.spatial_scatter
    """
    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        color=colors,
        shape=None,
        size=2,
        img=False,
        # dpi=500,
        # save="/exports/archive/hg-funcgenom-research/evonk/spatial_plot.png"
    )

def rank_genes_group(
        adata: AnnData,
        res: float = 1
) -> AnnData:
    """
    Identify marker genes per Leiden cluster.

    Parameters
    ----------
    adata : AnnData
        Must contain adata.obs["leiden"].
    res : float, default = 0.3
        Resolution used for Leiden clustering

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Results stored in adata.uns["rank_genes_groups"].
    Uses Wilcoxon rank-sum test.
    """
    if "leiden" not in adata.obs:
        print("Leiden clustering has not been performed yet. Adding 'Leiden' column...")
        sc.tl.leiden(adata, resolution=res, random_state=42)

    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden",
        method="wilcoxon"
    )

    # Plot ranking of genes. Plot top 10 genes per cluster
    sc.pl.rank_genes_groups(adata, n_genes=10)
    return adata


def plot_labels(
    adata: AnnData,
    label: str = "label"
):
    """
    This function plots the assigned labels to the samples on the same Xenium slide.

    It also shows the unassigned cells, visualizating the quality of the assignment of
    the samples.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain ``adata.obsm["spatial"]``
    label : str, default = "label"
        Column name where the label is assigned. ``label`` is default.

    Notes
    -----
    This is supposed to be used after performing the watershed segmentation and assigning
    labels to samples with the classes of SpatialAPI.
    """
    # Extract coordinates and condition 
    coords = adata.obsm["spatial"]           

    adata.obs[label] = adata.obs[label].astype("category")
    conditions = adata.obs[label]
    categories = conditions.cat.categories

    # Define colors
    palette = plt.cm.tab20.colors      
    color_map = {cat: palette[i] for i, cat in enumerate(categories)}
    colors = [color_map[c] for c in conditions]

    #  Compute figure size to match the data's aspect ratio
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    aspect_ratio = y_range / x_range

    # Scale the image
    fig_width = 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * aspect_ratio))

    # Scatter plot
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors,
        s=1.0,                              
        linewidths=0,
        rasterized=True,
    )

    # Flip axis to keep original coordinate system of Xenium
    ax.set_ylim(coords[:, 1].max(), coords[:, 1].min())
    # ax.invert_yaxis()
    # ax.set_ylim(22000, 0) TODO: I removed this, test if it's okay now.

    # Equal aspect ratio so coordinates are not distorted
    ax.set_aspect("equal", adjustable="box")

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
            markerfacecolor=color_map[cat], markersize=8, label=cat)
        for cat in categories
    ]
    ax.legend(handles=legend_handles, title="Label",
            bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    ax.set_title("Labels on Xenium grid")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")

    plt.tight_layout()
    plt.show()

def plot_count_distr(
    adata : AnnData,
    samples : list[str],
    vline: int | None = None,
    zoom: int | None = None,
    sample_dataset : str = "sample_dataset",
):
    """
    This function plots the distribution of counts between two datasets.
    It shows one dataset in orange and the other in blue, with on the 
    x-axis the total amount of counts.

    Parameters 
    ----------
    adata : AnnData
        Annotated data matrix containing two combined datasets.
    samples : list[str]
        List of samples names present in the AnnData object.
    vline : int, optional
        The vertical value line shown in the plot
    zoom: int, optional
        The value on which to zoom in. 
    sample_dataset : str, default = "sample_dataset"
        Column in `adata.obs` showing the samples of the datasets. It can be a combination of the sample ID and the dataset ID.
    """
    plt.figure(figsize=(8, 5))
    if sample_dataset not in adata.obs:
        print(f"{sample_dataset} is not present in the AnnData object. Try again.")
        return
    for s in samples:
        subset = adata.obs.loc[
            adata.obs[sample_dataset] == s, "total_counts"
        ]
        if zoom is not None:
            subset = subset[subset <= 10000]
        plt.hist(subset, bins=50, alpha=0.5, label=s)
        if vline is not None:
            plt.axvline(vline, color="red", linestyle="--", linewidth=2, label=f"{vline} counts")
        
        plt.xlabel("Total counts per cell")
        plt.ylabel("Frequency")
        if zoom is None:
            plt.title(f"Counts distribution")
        else:
            plt.title(f"Counts distribution (zoom ≤ {zoom} counts)")

