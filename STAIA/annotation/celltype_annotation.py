"""
celltype_annotation.py
======================
Wrappers for cell type annotation methods in spatial transcriptomics

Supported methods
-----------------
- CellTypist    : marker/classifier-based, no reference scRNA-seq needed
- Tangam        : optimal-transport mapping, required scRNA-seq reference
- Cell2location : Bayesian deconvolution, requires scRNA-seq reference
- TACCO         : compositional annotation framework, requires scRNA-seq reference
"""

from __future__ import annotations

import warnings
from typing import Optional

import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse as sp


# 1. CellTypist
def annotate_celltypist(
    adata: ad.AnnData,
    model: str = "Immune_All_Low.pkl",
    majority_voting: bool = True,
    over_clustering: Optional[str] = None,
    result_key: str = "celltypist_cell_type",
    inplace: bool = True
):
    """
    Annotate cells using CellTypist (marker/classifier-based).
 
    Does *not* require an external scRNA-seq reference — uses pre-trained
    models that ship with the package or can be downloaded automatically.
 
    Parameters
    ----------
    adata : AnnData
        Spatial or single-cell AnnData object.
        ``adata.X`` should contain normalised, log1p-transformed counts
        (CellTypist normalises to 10,000 counts/cell internally if needed).
    model : str
        CellTypist model name or path to a local ``.pkl`` model file.
        Run ``celltypist.models.get_all_models()`` to list available models.
        Commonly used models:
          - ``"Immune_All_Low.pkl"``   – immune cells, fine-grained
          - ``"Immune_All_High.pkl"``  – immune cells, coarse
          - ``"Adult_Mouse_Gut.pkl"``  – mouse gut atlas
    majority_voting : bool
        Whether to refine predictions by majority voting within clusters.
        Recommended for spatial data. Requires a clustering column to be set
        in ``adata.obs`` (see ``over_clustering``).
    over_clustering : str, optional
        Column in ``adata.obs`` with cluster labels used for majority voting.
        If None, CellTypist performs its own over-clustering automatically.
    result_key : str
        Column name written to ``adata.obs`` with the predicted cell types.

    Returns
    -------
    AnnData or None
 
    Examples
    --------
    >>> STAIA.annotate_celltypist(adata)
    >>> adata.obs["celltypist_cell_type"]
    """
    import celltypist
    from celltypist import models

    X = adata.X
    mat = X.toarray() if sp.issparse(X) else np.asarray(X)
 
    # check raw counts
    looks_raw = (mat >= 0).all() and np.allclose(mat, mat.astype(int))
    per_cell_sums = mat.sum(axis=1)
    looks_normalised_not_logged = (per_cell_sums.mean() > 100)
 
    # already log1p: max value is small (log1p(10000) ≈ 9.2)
    looks_log1p = (mat.max() < 20) and (per_cell_sums.mean() < 100)
 
    if looks_log1p:
        # Data already looks correctly preprocessed
        pass
    else:
        if looks_raw:
            warnings.warn(
                "adata.X appears to contain raw counts (non-negative integers). "
                "Automatically applying sc.pp.normalize_total(target_sum=1e4) "
                "and sc.pp.log1p(). The result is stored in "
                "adata.layers['log1p_norm'] and used for CellTypist; "
                "adata.X is left unchanged.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "adata.X does not look like log1p-normalised data "
                f"(mean per-cell sum = {per_cell_sums.mean():.1f}, "
                f"max value = {mat.max():.2f}). "
                "Automatically applying sc.pp.normalize_total(target_sum=1e4) "
                "and sc.pp.log1p(). The result is stored in "
                "adata.layers['log1p_norm'] and used for CellTypist; "
                "adata.X is left unchanged.",
                UserWarning,
                stacklevel=2,
            )
 
        # Normalise into a temporary copy so we never overwrite .X
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)
 
        # Cache the preprocessed matrix as a layer for transparency
        adata.layers["log1p_norm"] = adata_norm.X
 
        # Point .X at the normalised matrix for the CellTypist call,
        # then restore it afterwards
        _original_X = adata.X
        adata.X = adata_norm.X
 

    # download model if it is not already cached
    models.download_models(model=model, force_update=False)
    ct_model = models.Model.load(model=model)

    predictions = celltypist.annotate(
        adata,
        model=ct_model,
        majority_voting=majority_voting,
        over_clustering=over_clustering,
    )

    # adata.obs[result_key] = predictions.predicted_labels.predicted_labels.values
    if majority_voting:
        adata.obs[result_key] = predictions.predicted_labels.majority_voting.values
    else:
        adata.obs[result_key] = predictions.predicted_labels.predicted_labels.values
    adata.obs[f"{result_key}_score"] = predictions.probability_matrix.max(axis=1).values

    # store full probability matrix in obsm
    adata.obsm[f"{result_key}_prob_matrix"] = predictions.probability_matrix 
    return adata

# 2. Tangram
def annotate_tangram(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    annotation_key: str,
    result_key: str = "tangram_cell_type",
    mode: str = "clusters",
    num_epochs: int = 1000,
    device: str = "cpu",
) -> Optional[ad.AnnData]:
    """
    Annotate spatial data via Tangram optimal-transport cell-to-space mapping.
 
    Requires a scRNA-seq reference with existing cell type labels.
    Tangram maps single cells onto spatial coordinates and transfers
    cell-type labels, producing per-spot cell type proportion estimates.
 
    Parameters
    ----------
    adata_sp : AnnData
        Spatial AnnData object. Must have ``obsm["spatial"]`` with coordinates.
    adata_sc : AnnData
        scRNA-seq reference AnnData with cell type labels in ``.obs``.
    annotation_key : str
        Column in ``adata_sc.obs`` containing cell type labels.
    result_key : str
        Prefix for columns written to ``adata_sp.obsm`` with cell type
        fractions (``adata_sp.obsm[result_key]``) and to ``adata_sp.obs``
        with the most abundant cell type per spot
        (``adata_sp.obs[result_key + "_dominant"]``).
    mode : str
        Tangram mapping mode:
          - ``"clusters"`` – maps cell-type clusters (faster, recommended)
          - ``"cells"``    – maps individual cells (more granular, slower)
    num_epochs : int
        Number of optimisation epochs.
    device : str
        ``"cpu"`` or ``"cuda"`` (if a GPU is available).
 
    Returns
    -------
    AnnData or None
 
    Examples
    --------
    >>> STAIA.annotate_tangram(adata_sp, adata_sc, annotation_key="cell_type")
    >>> adata_sp.obs["tangram_cell_type_dominant"]
    """
    import tangram as tg

    # find shared markers between scRNAseq and spatial dataset
    tg.pp_adatas(adata_sc, adata_sp, genes=None)

    # learn cell-to-space mapping
    ad_map = tg.map_cells_to_space(
        adata_sc,
        adata_sp,
        mode=mode,
        cluster_label=annotation_key,
        num_epochs=num_epochs,
        device=device,
        verbose=False,
    )

    # project cell type annotations onto spatial spots
    tg.project_cell_annotations(
        ad_map,
        adata_sp,
        annotation=annotation_key,
    )

    # store fractions in obsm and dominant type in obs
    ct_df: pd.DataFrame = adata_sp.obsm["tangram_ct_pred"]
    adata_sp.obsm[result_key] = ct_df
    adata_sp.obs[f"{result_key}_dominant"] = ct_df.idxmax(axis=1).values
    return adata_sp

# 3. TACCO
def annotate_tacco(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    annotation_key: str,
    result_key: str = "tacco_cell_type",
    method: str = "OT",
    counts_layer: str = None,
) -> Optional[ad.AnnData]:
    """
    Annotate spatial spots with TACCO (Transfer of Annotations to Cells
    and their Combinations). 
    Requires a scRNA-seq reference with existing cell type labels and raw counts.

    Parameters
    ----------
    adata_sp : AnnData
        Spatial AnnData object. X (or `counts_layer`) must contain raw
        non-negative integer counts.
    adata_sc : AnnData
        scRNA-seq reference AnnData with cell type labels in ``.obs``.
    annotation_key : str
        Column in ``adata_sc.obs`` containing cell type labels.
    result_key : str
        Key written to ``adata_sp.obsm`` with the compositional fractions
        and to ``adata_sp.obs`` for the dominant cell type per spot.
    method : str
        Annotation method passed to ``tc.tl.annotate``.
    counts_layer : str, optional
        Layer in ``adata_sp.layers`` holding raw integer counts (e.g.
        ``"counts"`` or ``"raw_counts"``). If provided, a temporary view
        with that layer as ``.X`` is passed to TACCO, leaving the original
        object untouched. If None, ``.X`` is used directly.

    Returns
    -------
    AnnData
    """
    import tacco as tc
    adata_sp, adata_sc = _check_duplicates(adata_sp, adata_sc)

    # drop cells with missing annotation in reference
    mask = adata_sc.obs[annotation_key].notna()
    if not mask.all():
        n_dropped = (~mask).sum()
        print(f"Dropping {n_dropped} cells with NaN in '{annotation_key}'")
        adata_sc = adata_sc[mask].copy()

    if len(adata_sc) == 0:
        raise ValueError(
            f"No cells remaining in reference after dropping NaN in '{annotation_key}'. "
            f"Check that annotation_key='{annotation_key}' is the correct column."
        )

    adata_sp_raw = _resolve_counts_layer(adata_sp, counts_layer, "adata_sp")

    tc.tl.annotate(
        adata_sp_raw,
        adata_sc,
        annotation_key=annotation_key,
        result_key=result_key,
        method=method,
    )

    # Copy results back to the original object
    adata_sp.obsm[result_key] = adata_sp_raw.obsm[result_key]

    ct_df: pd.DataFrame = adata_sp.obsm[result_key]
    adata_sp.obs[f"{result_key}_dominant"] = ct_df.idxmax(axis=1).values
    return adata_sp
 
 
# PRIVATE FUNCTIONS
def _check_duplicates(adata_sp, adata_sc):
    """Check duplicate var names in `adata`"""
    # --- ensure unique gene names ---
    if adata_sp.var_names.duplicated().any():
        adata_sp = adata_sp.copy()
        adata_sp.var_names_make_unique()

    if adata_sc.var_names.duplicated().any():
        adata_sc = adata_sc.copy()
        adata_sc.var_names_make_unique()
    return adata_sp, adata_sc

def _resolve_counts_layer(adata: ad.AnnData, counts_layer: Optional[str], name: str) -> ad.AnnData:
    """
    Return a copy of ``adata`` with raw counts in ``.X``.

    If ``counts_layer`` is given, copies that layer into ``.X``.
    If ``counts_layer`` is None, ``.X`` is used as-is.
    """
    import scipy.sparse as sp
    import numpy as np

    adata = adata.copy()
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"Layer '{counts_layer}' not found in {name}. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        adata.X = adata.layers[counts_layer]

    X = adata.X
    vals = X.data if sp.issparse(X) else X
    if not np.all(vals == np.floor(vals)):
        src = f".layers['{counts_layer}']" if counts_layer else ".X"
        raise ValueError(
            f"{name}: {src} does not contain integer counts. "
            f"Pass the correct layer via counts_layer=."
        )
    return adata
