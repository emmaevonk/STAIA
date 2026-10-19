import squidpy as sq
import scanpy as sc
from anndata import AnnData

def morans_score(
        adata: AnnData,
        n_perms: int = 100,
        coord_type: str = "generic",
        delaunay: bool = True,
        top_n: int = 10
) -> AnnData:
    """
    Compute Moran's I spatial autocorrelation statistics.

    Moran's I measures global spatial autocorrelation, indicating whether 
    gene expression or features are spatially clustered, dispersed, or random.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates.
    n_perms : int, default = 100
        Number of permutations used to assess significance.
    coord_type : str, default = "generic"
        Coordinate system type used for spatial nieghbor computation.
    delaunay : bool, default = True
        Whether to compute neighbors using Delaunay triangulation.
    top_n : int, default = 10
        Number of top features (by Moran's I) to return.

    Returns
    -------
    pandas.DataFrame
        Top features ranked by Moran's I statistic.

    Notes
    -----
    Results are stored in adata.uns["moranI"] as a side effect.

    Examples
    --------
    >>> top_genes = morans_score(adata, n_perms=1000, top_n=20)
    """
    sq.gr.spatial_neighbors(adata, coord_type=coord_type, delaunay=delaunay)
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        n_perms=n_perms,
        n_jobs=1,
    )
    morans_score = adata.uns["moranI"].head(top_n)
    return morans_score