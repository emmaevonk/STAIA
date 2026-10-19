from anndata import AnnData
from spatialdata import SpatialData
from scipy.sparse import issparse
import scanpy as sc
import numpy as np
import pandas as pd


def batch_correct(adata, batch_key="sample"):
    """
    Perform batch correction on spatial transcriptomics data using Harmony.

    Parameters
    ----------
    adata : AnnData
        Spatial data object containing gene expression and metadata.
        Must have PCA computed in obsm["X_pca"] prior to calling this function.
    batch_key : str, optional
        Column in adata.obs that defines batch labels (e.g. sample, slide, condition).
        Default is "sample".

    Returns
    -------
    AnnData
        The input object with the Harmony-corrected PCA embedding added
        to obsm["X_pca_harmony"].

    Raises
    ------
    ValueError
        If batch_key is not found in adata.obs.
    KeyError
        If obsm["X_pca"] is missing.

    Notes
    -----
    Prerequisites:
    - PCA must be computed before calling this function (e.g. via sc.pp.pca()).
    - The batch_key column must be present in adata.obs.
    - The corrected embedding can be used downstream by setting
      use_rep="X_pca_harmony" in sc.pp.neighbors().

    Examples
    --------
    >>> sdata = STAIA.batch_correct(sdata, batch_key="sample")
    """
    import harmonypy as hm

    if batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs.")
    if "X_pca" not in adata.obsm:
        raise KeyError("PCA embedding missing. Run sc.pp.pca() before batch correction.")

    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key)
    adata.obsm["X_pca_harmony"] = ho.Z_corr

    return adata


def size_normalization(
    sdata: SpatialData,
    table_layer: str = "table",
    shape_layer: str = "cell_boundaries",
    cellsize_key: str = "shapeSize",
    cell_id_col: str = "cell_ID",
    target_sum: float = 100.0,
    log1p: bool = True
) -> AnnData:
    """
    Size-normalize a SpatialData object by cell area, following the SPArrOW 
    convention (GitHub: https://github.com/saeyslab/napari-sparrow/tree/main).

    Parameters
    ----------
    sdata: SpatialData
        SpatialData object containing spatial transcriptomic information.
    table_layer: str, default = "table"
        Table layer in the SpatialData object.
    shape_layer: str, default = "cell_boundaries"
        Shapes layer in the SpatialData object from where area is computed.
    cellsize_key: str, default = "shapeSize"
        Column name for the cell sizes (computed in this function).
    cell_id_col: str, default = "cell_ID"
        Column in object where cell IDs can be found.
    target_sum: float, default = 100.0
        Scaling factor after area normalization (SPArrOW uses 100).
    log1p: bool, default = True
        Whether to apply log1p transformation after normalization.

    Returns
    -------
    AnnData with:
        - adata.layers['raw_counts']: counts before normalization
        - adata.layers['size_normalized']: size-normalized counts (before log1p)
        - adata.X: log1p-transformed size-normalized counts (if log1p is True)
    """
    # check if input data is really sdata
    if isinstance(sdata, AnnData):
        print(f"Expected input is SpatialData (mandatory layer: shapes). You provided AnnData. Performing counts normalization instead.")
        adata = counts_normalized(sdata)
        print(f"Counts normalization finished.")
        return adata

    adata = sdata.tables[table_layer].copy()
    gdf = sdata.shapes[shape_layer]
    cell_areas = gdf.geometry.area
    if cellsize_key not in adata.obs:
        print(f"'{cellsize_key}' not in the data. Computing {cellsize_key}...")
        adata.obs[cellsize_key] = adata.obs[cell_id_col].map(cell_areas)

        # filter for invalid entries
        adata = adata[adata.obs[cellsize_key].notna()].copy()
        adata = adata[adata.obs[cellsize_key] > 0].copy()
    
    # perform normalization
    adata = _size_norm(adata, cellsize_key, target_sum, log1p)
    return adata


def _size_norm(
        adata: AnnData,
        cellsize_key: str,
        target_sum: int,
        log1p: bool
) -> AnnData:
    sizes = adata.obs[cellsize_key].values
    if np.any(np.isnan(sizes)):
        raise ValueError(f"NaN values found in '{cellsize_key}'. Filter them out before normalizing.")
    if np.any(sizes <= 0):
        raise ValueError(f"Non-positive values found in '{cellsize_key}'. All sizes must be > 0.")

    # save raw counts
    adata.layers["raw_counts"] = adata.X.copy()

    if issparse(adata.X):
        X_norm = (adata.X.T * target_sum / sizes).T.tocsr()
    else:
        X_norm = (adata.X.T * target_sum / sizes).T

    adata.X = X_norm
    adata.layers["size_normalized"] = adata.X.copy()
    if log1p:
        sc.pp.log1p(adata)
    return adata

def counts_normalized(
    adata: AnnData,
    target_sum: int = 1e4
):
    """
    This function normalizes the AnnData object by counts.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix which is not (yet) normalized
    target_sum: int, default=1e4
        The target sum for the normalization

    Returns
    -------
        Annotated data matrix including normalized counts.
    """
    # normal normalization (on counts)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def scaling(
    adata: AnnData
):
    """
    Scale data to unit variance and zero mean.
    """
    return sc.pp.scale(adata)