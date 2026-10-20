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
import numpy as np

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

    import scipy.sparse as sp
    import scanpy as sc
 
    X = adata.X
    mat = X.toarray() if sp.issparse(X) else np.asarray(X)
 
    # Heuristic 1 – raw counts: values are non-negative integers
    looks_raw = (mat >= 0).all() and np.allclose(mat, mat.astype(int))
 
    # Heuristic 2 – already normalised but not log1p: per-cell sums near 10 000
    #   (log1p data has per-cell sums far below 10 000)
    per_cell_sums = mat.sum(axis=1)
    looks_normalised_not_logged = (per_cell_sums.mean() > 100)
 
    # Heuristic 3 – already log1p: max value is small (log1p(10000) ≈ 9.2)
    looks_log1p = (mat.max() < 20) and (per_cell_sums.mean() < 100)
 
    if looks_log1p:
        # Data already looks correctly preprocessed — proceed as-is
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

    adata.obs[result_key] = predictions.predicted_labels.predicted_labels.values
    adata.obs[f"{result_key}_score"] = predictions.probability_matrix.max(axis=1).values

    # store full probability matrix in obsm
    adata.obsm[f"{result_key}_prob_matrix"] = predictions.probability_matrix #TODO: maybe discard later?
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

# 3. Cell2location
def annotate_cell2location(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    annotation_key: str,
    result_key: str = "cell2location",
    batch_key: Optional[str] = None,
    max_epochs_reference: int = 200,
    max_epochs_spatial: int = 30000,
    accelerator: str = "cpu",
    inplace: bool = True,
) -> Optional[ad.AnnData]:
    """
    Deconvolve spatial spots with Cell2location (Bayesian model).
 
    Fits a two-stage model: (1) estimate per-cell-type reference signatures
    from the scRNA-seq data, (2) deconvolve each spatial spot into cell type
    abundances using those signatures.
 
    Parameters
    ----------
    adata_sp : AnnData
        Spatial AnnData object. ``adata_sp.X`` must contain **raw counts**.
    adata_sc : AnnData
        scRNA-seq reference AnnData. ``adata_sc.X`` must contain **raw counts**.
    annotation_key : str
        Column in ``adata_sc.obs`` containing cell type labels.
    result_key : str
        Key prefix written to ``adata_sp.obsm`` for the cell type abundance
        matrix and to ``adata_sp.obs`` for the dominant cell type per spot.
    batch_key : str, optional
        Column in ``adata_sc.obs`` with batch/sample labels for the
        regression model. Pass None if there is only one batch.
    max_epochs_reference : int
        Training epochs for the reference signature model.
    max_epochs_spatial : int
        Training epochs for the spatial deconvolution model.
    accelerator : bool
        Whether to use GPU acceleration via PyTorch.
 
    Returns
    -------
    AnnData or None
 
    Examples
    --------
    >>> STAIA.annotate_cell2location(
    ...     adata_sp, adata_sc,
    ...     annotation_key="cell_type",
    ...     batch_key="sample",
    ... )
    >>> adata_sp.obs["cell2location_dominant"]
    """
    import cell2location
    from cell2location.models import RegressionModel
  
    # Stage 1: Learn reference cell type signatures from scRNA-seq data
    adata_ref = adata_sc.copy()
 
    # Filter genes present in both datasets
    shared_genes = adata_ref.var_names.intersection(adata_sp.var_names)
    adata_ref = adata_ref[:, shared_genes].copy()
 
    RegressionModel.setup_anndata(
        adata_ref,
        labels_key=annotation_key,
        batch_key=batch_key,
    )
 
    ref_model = RegressionModel(adata_ref)
    ref_model.train(max_epochs=max_epochs_reference, accelerator=accelerator)
 
    # Export estimated expression signatures per cell type
    inf_aver = ref_model.export_posterior(
        adata_ref,
        sample_kwargs={"num_samples": 1000, "batch_size": 2500},
    )
    inf_aver = inf_aver["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{ct}" for ct in adata_ref.obs[annotation_key].unique()]
    ]
 
    # Stage 2: Deconvolve spatial data
    adata_sp = adata_sp[:, shared_genes].copy()
 
    cell2location.models.Cell2location.setup_anndata(
        adata_sp,
        batch_key=None,
    )
 
    spatial_model = cell2location.models.Cell2location(
        adata_sp,
        cell_state_df=inf_aver,
        N_cells_per_location=5,    # expected average cells per spot; adjust for your data
        detection_alpha=20,
    )
    spatial_model.train(
        max_epochs=max_epochs_spatial,
        batch_size=None,
        train_size=1,
        accelerator=accelerator,
    )
 
    adata_sp = spatial_model.export_posterior(
        adata_sp,
        sample_kwargs={"num_samples": 1000, "batch_size": 2500},
    )
 
    # Extract mean cell type abundance per spot
    ct_df = adata_sp.obsm["q05_cell_abundance_w_sf"]
    # Normalise to fractions
    ct_fractions = ct_df.div(ct_df.sum(axis=1), axis=0)
 
    adata_sp.obsm[result_key] = ct_fractions
    adata_sp.obs[f"{result_key}_dominant"] = ct_fractions.idxmax(axis=1).values
 
    return adata_sp

# 4. TACCO
def annotate_tacco(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    annotation_key: str,
    result_key: str = "tacco_cell_type",
    method: str = "OT",
) -> Optional[ad.AnnData]:
    """
    Annotate spatial spots with TACCO (Transfer of Annotations to Cells
    and their COmbinations).
 
    TACCO transfers compositional annotations from a labelled scRNA-seq
    reference to spatial data, storing per-spot cell type fractions and
    a dominant cell type label.
 
    Parameters
    ----------
    adata_sp : AnnData
        Spatial AnnData object.
    adata_sc : AnnData
        scRNA-seq reference AnnData with cell type labels in ``.obs``.
    annotation_key : str
        Column in ``adata_sc.obs`` containing cell type labels.
    result_key : str
        Key written to ``adata_sp.obsm`` with the compositional fractions
        (DataFrame, spots × cell types) and to ``adata_sp.obs`` for the
        dominant cell type per spot.
    method : str
        Annotation method passed to ``tc.tl.annotate``:
          - ``"OT"``     – optimal transport (default, recommended)
          - ``"NMFreg"`` – NMF-based regression
          - ``"NNLS"``   – non-negative least squares (fastest)
          - ``"tangram"``– wraps Tangram inside TACCO
          - ``"RCTD"``   – wraps RCTD inside TACCO (requires R)
        See TACCO docs for the full list.
 
    Returns
    -------
    AnnData or None
 
    Examples
    --------
    >>> annotate_tacco(adata_sp, adata_sc, annotation_key="cell_type")
    >>> adata_sp.obs["tacco_cell_type_dominant"]
    """
    import tacco as tc
 
    tc.tl.annotate(
        adata_sp,
        adata_sc,
        annotation_key=annotation_key,
        result_key=result_key,
        method=method,
    )
 
    # adata_sp.obsm[result_key] is a DataFrame (spots × cell types)
    ct_df: pd.DataFrame = adata_sp.obsm[result_key]
    adata_sp.obs[f"{result_key}_dominant"] = ct_df.idxmax(axis=1).values
 
    return adata_sp


# TODO: add consensus of all functions
def consensus_annotations():
    pass
 