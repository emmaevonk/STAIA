import warnings
import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData

from typing import Literal, Optional

# this script is to detect spatially variable genes

def morans_score(
        adata: AnnData,
        n_perms: int = 100,
        coord_type: str = "generic",
        delaunay: bool = True,
        top_n: int = 10,
        n_top_svgs: Optional[int] = 500,
        pval_threshold: float = 0.05,
) -> pd.DataFrame:
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
    n_top_svgs : int or None, default = 500
        Number of top genes to flag as spatially variable in adata.var.
        Prioritises FDR-significant genes (q < pval_threshold) first.
        If None, no flagging is performed.
    pval_threshold : float, default = 0.05
        FDR-corrected p-value cutoff used for SVG flagging.

    Returns
    -------
    pandas.DataFrame
        Full Moran's I results table (all genes), sorted by I descending.

    Notes
    -----
    Results are stored in:
    - adata.uns["moranI"]                  — full results table
    - adata.var["spatially_variable"]      — boolean SVG flag
    - adata.var["morans_I"]                — Moran's I score per gene
    - adata.var["morans_pval"]             — FDR-corrected p-value per gene

    Examples
    --------
    >>> results = STAIA.morans_score(adata, n_perms=1000, top_n=20, n_top_svgs=300)
    >>> results.head(10)
    >>> adata.var[adata.var["spatially_variable"]]
    """
    sq.gr.spatial_neighbors(adata, coord_type=coord_type, delaunay=delaunay)
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        n_perms=n_perms,
        n_jobs=1,
    )
    results = adata.uns["moranI"].sort_values("I", ascending=False)

    # write per-gene scores back to adata.var
    adata.var["morans_I"] = results.reindex(adata.var_names)["I"]
    adata.var["morans_pval"] = results.reindex(adata.var_names)["pval_norm_fdr_bh"]

    # flag spatially variable genes
    adata.var["spatially_variable"] = False
    if n_top_svgs is not None:
        n_flag = min(n_top_svgs, len(results))
        sig = results["pval_norm_fdr_bh"] < pval_threshold
        if sig.sum() >= n_flag:
            flagged = results[sig].head(n_flag).index
        else:
            flagged = results[sig].index.tolist()
            remaining = n_flag - len(flagged)
            extra = results[~sig].head(remaining).index.tolist()
            flagged = flagged + extra
        adata.var.loc[flagged, "spatially_variable"] = True
    n_svg = adata.var["spatially_variable"].sum()
    print(f"[STAIA] Moran's I complete — {n_svg} SVGs flagged "
          f"(n_top_svgs={n_top_svgs}, pval_threshold={pval_threshold})")

    return results


def plot_svgs(
        adata: AnnData,
        n_genes: int = 6,
        coord_key: str = "spatial",
        layer: Optional[str] = None,
        ncols: int = 3,
        cmap: str = "Reds",
        spot_size: float = 40,
        save: Optional[str] = None,
) -> None:
    """
    Plot spatial expression maps for the top spatially variable genes.

    Requires morans_score() to have been run first (so that
    adata.var["spatially_variable"] and adata.var["morans_I"] exist).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial coordinates in adata.obsm[coord_key].
    n_genes : int, default = 6
        Number of top SVGs to plot.
    coord_key : str, default = "spatial"
        Key in adata.obsm where 2-D spatial coordinates are stored.
    layer : str or None
        Expression layer to use. None uses adata.X.
    ncols : int, default = 3
        Number of columns in the plot grid.
    cmap : str, default = "Reds"
        Matplotlib colormap for expression values.
    spot_size : float, default = 40
        Scatter plot marker size.
    save : str or None
        File path to save the figure (e.g. "svgs.png"). If None, only displays.

    Examples
    --------
    >>> STAIA.morans_score(adata, n_top_svgs=300)
    >>> STAIA.plot_spatially_variable_genes(adata, n_genes=9, ncols=3)
    """
    if "spatially_variable" not in adata.var.columns:
        raise ValueError(
            "adata.var['spatially_variable'] not found. "
            "Run STAIA.morans_score() with n_top_svgs > 0 first."
        )
    if coord_key not in adata.obsm:
        raise ValueError(
            f"Spatial coordinates not found in adata.obsm['{coord_key}']. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    # Top SVGs ranked by Moran's I
    top_genes = (
        adata.var[adata.var["spatially_variable"]]
        .sort_values("morans_I", ascending=False)
        .head(n_genes)
        .index.tolist()
    )
    if not top_genes:
        print("No SVGs flagged. Run morans_score() with n_top_svgs > 0.")
        return

    coords = np.array(adata.obsm[coord_key])

    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    nrows = int(np.ceil(len(top_genes) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
    )
    axes = np.array(axes).flatten()

    for ax, gene in zip(axes, top_genes):
        idx = adata.var_names.get_loc(gene)
        expr = np.asarray(X[:, idx]).flatten()
        i_score = adata.var.loc[gene, "morans_I"]

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=expr, cmap=cmap, s=spot_size, linewidths=0,
        )
        ax.set_title(f"{gene}\nMoran's I = {i_score:.3f}", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
        plt.colorbar(sc, ax=ax, shrink=0.7)

    for ax in axes[len(top_genes):]:
        ax.set_visible(False)

    fig.suptitle(f"Top {len(top_genes)} Spatially Variable Genes (Moran's I)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[STAIA] Figure saved to {save}")

    plt.show()


def svg_summary(
        adata: AnnData,
        top_n: int = 20,
) -> pd.DataFrame:
    """
    Return a summary table of the top spatially variable genes.

    Parameters
    ----------
    adata : AnnData
        Must have been processed by morans_score().
    top_n : int, default = 20
        Number of top SVGs to include in the summary.

    Returns
    -------
    pandas.DataFrame
        Table with columns: gene, morans_I, morans_pval, spatially_variable.

    Examples
    --------
    >>> STAIA.svg_summary(adata, top_n=10)
    """
    required = {"spatially_variable", "morans_I", "morans_pval"}
    missing = required - set(adata.var.columns)
    if missing:
        raise ValueError(
            f"Missing columns in adata.var: {missing}. "
            "Run STAIA.morans_score() first."
        )

    summary = (
        adata.var[adata.var["spatially_variable"]]
        [["morans_I", "morans_pval"]]
        .sort_values("morans_I", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "gene", "morans_pval": "morans_pval_fdr_bh"})
    )
    return summary