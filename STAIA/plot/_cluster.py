import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import pandas as pd
import os

def cluster(
    adata: AnnData,
    # colors: list = ["transcript_counts", "nucleus_count", "cell_area", "segmentation_method"],
    colors: list = ["leiden"],
    n_comps: int = 50,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    color_map: str = "viridis",
    palette: str | None = None,
    save: str | bool = False,
    random_state: int = 0,
    resolution: float = 0.3
):
    """
    Compute UMAP clustering for specified columns. 

    This function generates a UMAP clustering, computing PCA, neighbors and 
    Leiden clustering if necessary using Scanpy. Optionally it generates
    a UMAP containing specified columns to color by the user.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates.
    colors : list
        List of columns present in adata.obs to color the UMAP with.
    n_comps : int, default = 50
        Number of principal components to compute for the PCA.
    n_neighbors : int, default = 15
        Size of the local neighborhood used for manifold approximation.
    n_pcs : int, default = 30
        Amount of principal components to use when computing nearest
        neigbor.
    color_map : str, default = "viridis"
        Color map to use for continuous variables.
    palette : str
        Palette used for the UMAP.
    save : bool, default=False
        A boolean deciding whether or not the UMAP is being saved. 
        If the data is saved (so not False), provide the path to the 
        output directory.
    random_state : int, default = 0
        Random seed for reproducibility of the data.
    resolution : float, default = 0.3
        The Leiden resolution used for clustering. If Leiden clustering
        has already been performed on the data, this parameter can be ignored.

    Returns
    -------
    AnnData
        AnnData object containing Leiden clustering, UMAP clustering and 
        neighbors.

    Notes
    -----
    This function adds columns to adata.obs (leiden, pca, and umap).

    Examples
    --------
    >>> cluster(adata, colors=["n_counts"], n_neighbors=10)

    """     
    if type(colors) != list:
        colors = list(colors)
    # PCA
    if "pca" not in adata.uns:
        sc.tl.pca(adata, n_comps=n_comps, random_state=random_state)

    # Neighbor graph
    if "neighbors" not in adata.uns:
        sc.settings.dpi_save = 500 
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            random_state=random_state
        )

    if len(colors) == 1 and colors[0] == "leiden" and "leiden" not in adata.obs:
        print("Computing Leiden clusters...")
        sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
        print("Leiden clusters computed.")

    # UMAP
    sc.tl.umap(adata, random_state=random_state)

    if save != False:
        folder_path = os.path.dirname(save)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)
        sc.settings.figdir = save

    single_panel = len(colors) == 1
    ax = plt.subplots(figsize=(6, 5))[1] if single_panel else None

    plot_kwargs = dict(
        adata=adata,
        color=colors,
        color_map=color_map,
        ncols=3,
        show=False,
        return_fig=not single_panel,  # get a Figure back for multi-panel case
    )
    if palette is not None:
        plot_kwargs["palette"] = palette
    if single_panel:
        plot_kwargs["ax"] = ax

    result = sc.pl.umap(**plot_kwargs)
    fig = ax.figure if single_panel else result

    if save != False:
        fig.savefig(save, bbox_inches='tight', dpi=300)
        plt.close(fig)
    return adata
    