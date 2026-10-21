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
import scipy.sparse as sp

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


# PUBLIC FUNCTIONS
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

    # adata.obs[result_key] = predictions.predicted_labels.predicted_labels.values
    if majority_voting:
        adata.obs[result_key] = predictions.predicted_labels.majority_voting.values
    else:
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
# def annotate_cell2location(
#     adata_sp: ad.AnnData,
#     adata_sc: ad.AnnData,
#     annotation_key: str,
#     result_key: str = "cell2location",
#     batch_key: Optional[str] = None,
#     max_epochs_reference: int = 200,
#     max_epochs_spatial: int = 30000,
#     accelerator: str = "cpu",
#     inplace: bool = True,
#     counts_layer: Optional[str] = None
# ) -> Optional[ad.AnnData]:
#     """
#     Deconvolve spatial spots with Cell2location (Bayesian model).

#     Requires raw counts in the ref scRNA-seq data and the ST data. The raw counts can either be in `adata_sc.X` or in a counts layer in `adata`.

#     If the raw counts are in a separate layer (so not in `adata.X`): in both `adata_ref`, and `adata_sp` objects the raw counts layer should have the same name.

#     Fits a two-stage model: (1) estimate per-cell-type reference signatures
#     from the scRNA-seq data, (2) deconvolve each spatial spot into cell type
#     abundances using those signatures.

#     Parameters
#     ----------
#     adata_sp : AnnData
#         Spatial AnnData object. ``adata_sp.X`` must contain **raw counts**.
#     adata_sc : AnnData
#         scRNA-seq reference AnnData. ``adata_sc.X`` must contain **raw counts**.
#     annotation_key : str
#         Column in ``adata_sc.obs`` containing cell type labels.
#     result_key : str
#         Key prefix written to ``adata_sp.obsm`` for the cell type abundance
#         matrix and to ``adata_sp.obs`` for the dominant cell type per spot.
#     batch_key : str, optional
#         Column in ``adata_sc.obs`` with batch/sample labels for the
#         regression model. Pass None if there is only one batch.
#     max_epochs_reference : int
#         Training epochs for the reference signature model.
#     max_epochs_spatial : int
#         Training epochs for the spatial deconvolution model.
#     accelerator : str
#         Accelerator to use for PyTorch (e.g. "cpu", "gpu").
#     counts_layer : str, optional
#         Layer in both AnnData objects containing raw integer counts.
#         If None, ``.X`` is used and must already contain raw counts.

#     Returns
#     -------
#     AnnData or None
#         If ``inplace=True``, modifies ``adata_sp`` in place and returns None.
#         Otherwise returns a new AnnData with results written to ``.obsm`` and
#         ``.obs``.

#     Examples
#     --------
#     >>> STAIA.annotate_cell2location(
#     ...     adata_sp, adata_sc,
#     ...     annotation_key="cell_type",
#     ...     batch_key="sample",
#     ... )
#     >>> adata_sp.obs["cell2location_dominant"]
#     """
#     import cell2location
#     from cell2location.models import RegressionModel

#     # Force CPU: disable CUDA entirely for this session
#     if accelerator == "cpu":
#         os.environ["CUDA_VISIBLE_DEVICES"] = ""
#         scvi.settings.dl_num_workers = 0
#         torch.set_num_threads(1)  # optional: limit CPU threads

#     train_kwargs = {"accelerator": accelerator}

#     adata_sp, adata_sc = _check_duplicates(adata_sp, adata_sc)

#     adata_ref = _resolve_counts_layer(adata_sc, counts_layer, "adata_sc")
#     adata_sp_work = _resolve_counts_layer(adata_sp, counts_layer, "adata_sp")   

#     # ------------------------------------------------------------------ #
#     # Stage 1: Learn reference cell type signatures from scRNA-seq data   #
#     # ------------------------------------------------------------------ #

#     # Filter genes present in both datasets
#     shared_genes = adata_ref.var_names.intersection(adata_sp.var_names)
#     adata_ref = adata_ref[:, shared_genes].copy()

#     # Ensure cell type labels have a unique, clean index (avoids reindex errors)
#     adata_ref.obs[annotation_key] = (
#         adata_ref.obs[annotation_key].astype(str).astype("category")
#     )

#     RegressionModel.setup_anndata(
#         adata_ref,
#         labels_key=annotation_key,
#         batch_key=batch_key,
#     )

#     ref_model = RegressionModel(adata_ref)
#     ref_model.train(max_epochs=max_epochs_reference, accelerator=accelerator)

#     # Export estimated expression signatures per cell type.
#     # export_posterior stores results in adata_ref.varm; inf_aver is a
#     # (genes × cell_types) DataFrame — the correct input for Cell2location.
#     adata_ref = ref_model.export_posterior(
#         adata_ref,
#         sample_kwargs={"num_samples": 1000, "batch_size": 2500},

#     )

#     # The standard way to get inf_aver: use the uns-stored posterior mean,
#     # then drop any genes with unexpectedly high unexplained variance.
#     if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
#         inf_aver = adata_ref.varm["means_per_cluster_mu_fg"].copy()
#     else:
#         raise KeyError(
#             "'means_per_cluster_mu_fg' not found in adata_ref.varm after "
#             "export_posterior. Check your cell2location version."
#         )

#     # Drop genes with high unexplained variance (optional but recommended)
#     if "stds_per_cluster_mu_fg" in adata_ref.varm.keys():
#         inf_aver_sd = adata_ref.varm["stds_per_cluster_mu_fg"].copy()
#         inf_aver = inf_aver.loc[
#             (inf_aver_sd / (inf_aver + 1e-10)).max(axis=1) < 1.0
#         ]

#     # ------------------------------------------------------------------ #
#     # Stage 2: Deconvolve spatial data                                    #
#     # ------------------------------------------------------------------ #
#     adata_sp_model = adata_sp[:, inf_aver.index].copy()

#     cell2location.models.Cell2location.setup_anndata(
#         adata_sp_model,
#         batch_key=None,
#     )

#     spatial_model = cell2location.models.Cell2location(
#         adata_sp_model,
#         cell_state_df=inf_aver,
#         N_cells_per_location=5,
#         detection_alpha=20,
#     )
#     spatial_model.train(
#         max_epochs=max_epochs_spatial,
#         batch_size=None,
#         train_size=1,
#         accelerator=accelerator
#     )

#     adata_sp_model = spatial_model.export_posterior(
#         adata_sp_model,
#         sample_kwargs={"num_samples": 1000, "batch_size": 2500},
#     )

#     # Extract mean cell type abundance per spot (q05 = posterior 5th percentile)
#     ct_df = adata_sp_model.obsm["q05_cell_abundance_w_sf"].copy()
#     # Normalise to fractions
#     ct_fractions = ct_df.div(ct_df.sum(axis=1), axis=0)

#     # ------------------------------------------------------------------ #
#     # Write results back                                                   #
#     # ------------------------------------------------------------------ #
#     if inplace:
#         adata_sp.obsm[result_key] = ct_fractions
#         adata_sp.obs[f"{result_key}_dominant"] = ct_fractions.idxmax(axis=1).values
#         return None
#     else:
#         adata_out = adata_sp.copy()
#         adata_out.obsm[result_key] = ct_fractions
#         adata_out.obs[f"{result_key}_dominant"] = ct_fractions.idxmax(axis=1).values
#         return adata_out

# 4. TACCO
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

    # --- drop cells with missing annotation in reference ---
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

    # # --- resolve counts layer ---
    # if counts_layer is not None:
    #     if counts_layer not in adata_sp.layers:
    #         raise KeyError(
    #             f"Layer '{counts_layer}' not found in adata_sp. "
    #             f"Available layers: {list(adata_sp.layers.keys())}"
    #         )
    #     adata_sp_raw = adata_sp.copy()
    #     adata_sp_raw.X = adata_sp.layers[counts_layer]
    # else:
    #     adata_sp_raw = adata_sp

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

# TODO: add consensus of all functions
def consensus_annotations():
    pass
 