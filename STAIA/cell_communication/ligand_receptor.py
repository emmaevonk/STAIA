import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp

import liana as li

# PRIVATE FUNCTIONS

def _filter_small_types(
    adata: ad.AnnData,
    cell_type_key: str,
    min_cells: int,
    verbose: bool
) -> ad.AnnData:
    """Drop cell types with fewer than ``min_cells`` observations. ``cell_type_key`` can be anything the data can be grouped by, e.g. cell types or Leiden clusters."""
    counts = adata.obs[cell_type_key].value_counts()
    keep = counts[counts >= min_cells].index
    dropped = list(counts[counts < min_cells].index)
    if dropped and verbose:
        print(f"[STAIA] Dropping cell types with < {min_cells} cells: {dropped}")
    return adata[adata.obs[cell_type_key].isin(keep)].copy()

def _ensure_spatial_neighbors(
    adata: ad.AnnData,
    spatial_key: str = "spatial",
    n_rings: int = 1,
    n_neighs: int = 6,
    coord_type: str = "grid",
    verbose: bool = False,
) -> ad.AnnData:
    """Build a squipdy spatial neighbour graph if one does not already exist."""
    if "spatial_connectivities" not in adata.obsp:
        if verbose:
            print("[STAIA] Computing spatial neighbors...")
        sq.gr.spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            coord_type=coord_type,
            n_rings=n_rings,
            n_neighs=n_neighs,
            key_added="spatial",
        )
    return adata

# PUBLIC FUNCTIONS
def run_liana(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    spatial_key: str = "spatial",
    resource_name: str = "consensus",
    method: str = "rank_aggregate",
    spatial: bool = True,
    local_name: str = "cosine",
    n_perms: int = 500,
    expr_prop: float = 0.1,
    min_cells: int = 5,
    n_rings: int = 1,
    coord_type: str = "grid",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[ad.AnnData]]:
    """
    Run LIANA+ ligand-receptor inference on spatial or single-cell data.
 
    When ``spatial=True``, two complementary analyses are performed:
 
    1. **Global (cell-type level)** - scores every LR pair for every
       source → target cell-type combination using the spatial neighbourhood
       to restrict interactions to physically proximal cell types.
 
    2. **Local (spot level)** - ``li.mt.bivariate`` computes per-spot
       spatially-weighted similarity scores for each LR pair, letting you
       pinpoint *where* in the tissue a given interaction is most active.
 
    Parameters
    ----------
    adata
        Preprocessed AnnData. ``adata.X`` should contain log-normalised counts.
        For spatial mode, ``adata.obsm[spatial_key]`` must hold (x, y) coords.
    cell_type_key
        Column in ``adata.obs`` holding cell-type / cluster labels. E.g. Leiden clusters.
    spatial_key
        Key in ``adata.obsm`` for spatial coordinates.
    resource_name
        LR resource to use. Run ``li.rs.show_resources()`` to list options.
        Common choices: ``'consensus'``, ``'cellphonedb'``, ``'cellchat'``.
    method
        Scoring method for global cell-type-level inference.
        ``'rank_aggregate'`` (default) combines eight methods into a consensus.
        Other options: ``'cellphonedb'``, ``'cellchat'``, ``'natmi'``,
        ``'logfc'``, ``'singlecellsignalr'``, ``'connectome'``.
        Note: ``'connectome'`` is the liana-py re-implementation of Raredon
        et al. (2021) and requires log-normalised counts in ``adata.X``.
    spatial
        Build spatial neighbours and pass them to LIANA so that only
        spatially proximal cell-type pairs are tested.  Also runs the local
        bivariate scoring, which returns ``liana_local``.
    local_name
        Local similarity metric for ``li.mt.bivariate``.
        Options: ``'cosine'`` (best in LIANA+ benchmarks), ``'morans'``,
        ``'pearson'``, ``'spearman'``.
    n_perms
        Number of permutations for p-value calculation.
        Reduce to 100 for speed during development.
    expr_prop
        Minimum proportion of cells in a cell type that must express a
        gene for it to be considered expressed in that type.
    min_cells
        Cell types with fewer cells than this are dropped before analysis.
    n_rings
        Rings of grid neighbours for Visium data (``coord_type='grid'``).
        Increase to 2 to widen the neighbourhood.
    coord_type
        ``'grid'`` for Visium; ``'generic'`` for other ST platforms.
    verbose
        Print progress messages.
 
    Returns
    -------
    liana_global : pd.DataFrame
        Cell-type-level LR scores. Key columns:
            ``ligand``, ``receptor``, ``source``, ``target`` - the pair.
            ``aggregate_rank``- consensus rank (lower = more significant).
            ``cellphone_pvals`` - permutation p-value.
            ``lr_means`` - mean LR co-expression score.
    liana_local : ad.AnnData or None
        Per-spot local scores (one variable per LR pair, one obs per spot).
        ``None`` when ``spatial=False``.
        Pass to ``plot_spatial_lr_scores()`` for tissue-level visualisation.
 
    Examples
    --------
    >>> liana_global, liana_local = run_liana(
    ...     adata,
    ...     cell_type_key="cell_type",
    ...     spatial=True,
    ...     n_perms=100,
    ... )
    >>> liana_global.sort_values("aggregate_rank").head(10)
    """
    if cell_type_key not in adata.obs.columns:
        raise ValueError(
            f"cell_type_key '{cell_type_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    adata = adata.copy()
    if verbose:
        n_types = adata.obs[cell_type_key].nunique()
        print(
            f"[STAIA | LIANA+] {adata.n_obs} spots x "
            f"{adata.n_vars} genes, {n_types} cell types"
        )
    
    adata = _filter_small_types(adata, cell_type_key, min_cells, verbose)

    # spatial neighbors
    if spatial:
        adata = _ensure_spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            n_rings=n_rings,
            coord_type=coord_type,
            verbose=verbose,
        )

    # method dispatch
    _method_map = {
        "rank_aggregate":    li.mt.rank_aggregate,
        "cellphonedb":       li.mt.cellphonedb,
        "cellchat":          li.mt.cellchat,
        "natmi":             li.mt.natmi,
        "logfc":             li.mt.logfc,
        "singlecellsignalr": li.mt.singlecellsignalr,
        "connectome":        li.mt.connectome,
    }
    if method not in _method_map:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(_method_map)}"
        )

    # global cell type level scoring
    if verbose:
        print(f"[STAIA | LIANA+] Running global scoring (method='{method}') …")
 
    call_kwargs = dict(
        groupby=cell_type_key,
        resource_name=resource_name,
        expr_prop=expr_prop,
        n_perms=n_perms,
        use_raw=False,
        layer=None,
        verbose=verbose,
    )
 
    # `inplace` argument only exists in `rank_aggregate` method
    if method == "rank_aggregate":
        call_kwargs["inplace"] = True
        if spatial:
            call_kwargs["spatial_key"] = spatial_key
 
    _method_map[method](adata, **call_kwargs)

    liana_global = adata.uns["liana_res"].copy()
 
    if verbose:
        print(
            f"[STAIA | LIANA+] Global scoring done. "
            f"{len(liana_global)} LR pair x cell-type combinations."
        )

    # local spot level scoring
    liana_local = None
    if spatial:
        if verbose:
            print(
                f"[STAIA | LIANA+] Running local bivariate scoring "
                f"(local_name='{local_name}') …"
            )
        liana_local = li.mt.bivariate(
            adata,
            local_name=local_name,
            resource_name=resource_name,
            connectivity_key="spatial_connectivities",
            n_perms=n_perms,
            mask_negatives=False,
            add_categories=True,
            use_raw=False,
            layer=None,
            verbose=verbose,
        )
        if verbose:
            print("[STAIA | LIANA+] Local bivariate scoring done.")
 
    return liana_global, liana_local

# PLOTTING HELPERS
def plot_liana_dotplot(
    liana_global: pd.DataFrame,
    top_n: int = 20,
    source_groups: Optional[List[str]] = None,
    target_groups: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save: Optional[str] = None,
) -> None:
    """
    LIANA+ dot-plot of the top LR pairs.

    Parameters
    ----------
    liana_global
        DataFrame returned by ``run_liana()`` (first return value).
    top_n
        Number of top interactions to display.
    source_groups
        Show only these source cell types (``None`` = all).
    target_groups
        Show only these target cell types (``None`` = all).
    figsize
        Figure width and height in inches.
    save
        File path to save the figure (e.g. ``'dotplot.pdf'``).
    """
    df = liana_global.copy()

    if source_groups:
        df = df[df["source"].isin(source_groups)]
    if target_groups:
        df = df[df["target"].isin(target_groups)]

    df = df.sort_values("magnitude_rank").head(top_n)
    df["lr_pair"] = df["ligand_complex"] + " → " + df["receptor_complex"]
    df["ct_pair"] = df["source"] + " → " + df["target"]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        x=df["ct_pair"],
        y=df["lr_pair"],
        c=df["magnitude_rank"],
        s=(1 - df["cellphone_pvals"]) * 200,
        cmap="viridis_r",
        edgecolors="grey",
        linewidths=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="Magnitude rank")
    ax.set_xlabel("Cell-type pair (source → target)")
    ax.set_ylabel("Ligand–Receptor pair")
    ax.set_title("LIANA+ – Top LR interactions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[STAIA] Figure saved to {save}")
    plt.show()

def plot_spatial_lr_scores(
    adata: ad.AnnData,
    liana_local: ad.AnnData,
    lr_pair: Tuple[str, str],
    local_name: str = "cosine",
    spot_size: int = 1,
    cmap: str = "magma",
    figsize: Tuple[int, int] = (6, 5),
    save: Optional[str] = None,
) -> None:
    """
    Overlay per-spot LR interaction scores on the tissue section.
 
    Requires the ``liana_local`` AnnData returned by ``run_liana(spatial=True)``.
 
    Parameters
    ----------
    adata
        The original spatial AnnData (used for spatial coordinates).
    liana_local
        AnnData of per-spot local scores returned by ``run_liana()``.
    lr_pair
        ``(ligand, receptor)`` to visualise, e.g. ``("Vegfa", "Flt1")``.
        Gene names are case-sensitive.
    local_name
        Label for the colour bar title (should match what was used in
        ``run_liana()``).
    spot_size, cmap, figsize
        Passed to ``sc.pl.spatial``.
    save
        File path to save the figure.
    """
    ligand, receptor = lr_pair
    key = f"{ligand}^{receptor}"

    if key not in liana_local.var_names:
        sample = list(liana_local.var_names[:10])
        raise ValueError(
            f"LR pair '{key}' not found in liana_local. "
            f"First 10 available pairs: {sample}"
        )

    # Subset adata to the same spots as liana_local, then add scores
    adata_plot = adata[liana_local.obs_names].copy()
    X = liana_local[:, key].X
    if sp.issparse(X):
        scores = np.array(X.todense()).flatten()
    else:
        scores = np.array(X).flatten()
    adata_plot.obs[f"lr_{key}"] = scores

    sc.pl.spatial(
        adata_plot,
        color=f"lr_{key}",
        spot_size=spot_size,
        cmap=cmap,
        title=f"Local {local_name} score\n{ligand} → {receptor}",
        show=save is None,
    )

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[STAIA] Figure saved to {save}")

def plot_liana_circle(
    liana_global: pd.DataFrame,
    cell_type_key: str,
    adata: ad.AnnData,
    score_key: str = "magnitude_rank",
    inverse_score: bool = True,
    top_n: int = 30,
    source_labels: Optional[List[str]] = None,
    target_labels: Optional[List[str]] = None,
    pivot_mode: str = "mean",
    figure_size: Tuple[int, int] = (8, 8),
    save: Optional[str] = None,
) -> None:
    """
    Circle plot showing cell-type communication network from LIANA+ results.

    Parameters
    ----------
    liana_global
        DataFrame returned by ``run_liana()`` (first return value).
    cell_type_key
        The same groupby key used in ``run_liana()``, e.g. ``'cell_type'``.
        Must match a column in ``adata.obs`` so node colours can be assigned.
    adata
        The AnnData object passed to ``run_liana()``.
    score_key
        Column in ``liana_global`` to use as edge weight.
        ``'magnitude_rank'`` or ``'specificity_rank'``.
    inverse_score
        Whether to invert the score with -log10. Set to ``True`` for rank
        columns so that lower ranks (= stronger interactions) map to larger
        edge weights.
    top_n
        Number of top interactions to show as arrows.
    source_labels
        Show only these source cell types (``None`` = all).
    target_labels
        Show only these target cell types (``None`` = all).
    pivot_mode
        ``'counts'`` (number of interactions) or ``'mean'`` (mean score).
    figure_size
        Figure width and height in inches.
    save
        File path to save the figure (e.g. ``'circle_plot.pdf'``).
    """
    ax = li.pl.circle_plot(
        adata=adata,
        liana_res=liana_global,
        groupby=cell_type_key,
        score_key=score_key,
        inverse_score=inverse_score,
        top_n=top_n,
        orderby=score_key,
        orderby_ascending=True,
        source_labels=source_labels,
        target_labels=target_labels,
        pivot_mode=pivot_mode,
        figure_size=figure_size,
    )

    if save:
        ax.figure.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[STAIA] Figure saved to {save}")
    plt.show()
    