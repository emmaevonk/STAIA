from anndata import AnnData
from spatialdata import SpatialData
from scipy.sparse import issparse
import scanpy as sc
import numpy as np
import pandas as pd


# def size_normalization(
#     adata: AnnData,
#     cell_areas: pd.Series | None = None,
#     cellsize_key: str = "shapeSize",
#     cell_id_col: str = "cell_ID",
#     target_sum: float = 100.0,
#     log1p: bool = True
# ) -> AnnData:
#     """
#     Size-normalize an AnnData object by cell area.

#     Parameters
#     ----------
#     adata : AnnData
#         AnnData object containing spatial transcriptomic information.
#     cell_areas : pd.Series or None, default = None
#         Series mapping cell IDs to their areas. Required if `cellsize_key` is
#         not already present in `adata.obs`.
#     cellsize_key : str, default = "shapeSize"
#         Column name for the cell sizes in `adata.obs`.
#     cell_id_col : str, default = "cell_ID"
#         Column in `adata.obs` where cell IDs can be found.
#     target_sum : float, default = 100.0
#         Scaling factor after area normalization (SPArrOW uses 100).
#     log1p : bool, default = True
#         Whether to apply log1p transformation after normalization.

#     Returns
#     -------
#     AnnData with:
#         - adata.layers['raw_counts']: counts before normalization
#         - adata.layers['size_normalized']: size-normalized counts (before log1p)
#         - adata.X: log1p-transformed size-normalized counts (if log1p=True)
#     """
#     adata = adata.copy()

#     if cellsize_key not in adata.obs:
#         if cell_areas is None:
#             raise ValueError(
#                 f"'{cellsize_key}' not found in adata.obs and no `cell_areas` "
#                 "Series was provided. Please pass cell areas explicitly."
#             )
#         print(f"'{cellsize_key}' not in the data. Computing {cellsize_key}...")
#         adata.obs[cellsize_key] = adata.obs[cell_id_col].map(cell_areas)

#         # filter for invalid entries
#         adata = adata[adata.obs[cellsize_key].notna()].copy()
#         adata = adata[adata.obs[cellsize_key] > 0].copy()

#     adata = _size_norm(adata, cellsize_key, target_sum, log1p)
#     return adata

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