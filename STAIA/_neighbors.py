import squidpy as sq
import matplotlib.pyplot as plt
from anndata import AnnData
import os
import scanpy as sc

def nhood_enrichment(
        adata: AnnData,
        cluster_key: str = "leiden",
        show: bool = True,
        save: str | bool = False,
) -> plt.Figure:
    """
    Compute and visualize neighborhood enrichment between clusters.

    Neighborhood enrichment quantifies whether cells from specific clusters
    are found in spatial proximity more or less often than expected by chance.

    This function computes enrichment statistics using Squidpy and generates
    a heatmap visualization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial connectivity information.
        Spatial neighbors must be computed beforehand using 
        ``sq.gr.spatial_neighbors``
    cluster_key : str, default = "leiden"
        Column in adata.obs containing cluster labels.
    show : bool, default = True
        Whether to display the plot.
    save : bool, default=False
        A boolean deciding whether or not the neighborhood enrichment is being saved. 
        If the data is saved (so not False), provide the path to the 
        output directory.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the neighborhood enrichment heatmap.

    Raises
    ------
    KeyError
        If ``cluster_key`` is not found in adata.obs.
    ValueError
        If spatial neighbors are missing.

    Notes
    -----
    Results are stored in adata.uns["nhood_enrichment"]

    Positive values indicate clusters that are spatially enriched together,
    while negative values indicate spatial avoidance.

    Examples
    --------
    >>> import squidpy as sq
    >>> sq.gr.spatial_neighbors(adata)
    >>> nhood_enrichment(adata, cluster_key="leiden", save=output_dir)
    """
    if save != False:
        folder_path = os.path.dirname(save)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

    # Single figure — ax and fig stay linked
    fig, ax = plt.subplots(figsize=(8, 8))

    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)

    sq.pl.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        figsize=(8, 8),
        title="Neighborhood enrichment adata",
        ax=ax,
    )

    if save != False:
        fig.savefig(save, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def compute_spatial_neighbors(
        adata: AnnData,
        cluster_key: str = "leiden"
) -> AnnData:
    # Compute enrichment
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    sq.pl.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        figsize=(8, 8),
        title="Neighborhood enrichment adata",
    )
    fig.show()
    return fig