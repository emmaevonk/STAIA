import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from itertools import combinations

"""
This module performs pseudobulk DE analysis per condition
"""

import numpy as np

def _safe_filename(s: str) -> str:
    """Replace characters that are invalid in filenames with underscores."""
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

def plot_volcano(
    de_df: pd.DataFrame,
    celltype: str = "",
    contrast: str = "",
    lfc_thresh: float = 0.25,
    padj_thresh: float = 0.05,
    top_n_labels: int = 10,
    save: bool = False,
    output_path: str | None = None,
):
    df = de_df.dropna(subset=["log2FoldChange", "padj"]).copy()
    df["-log10padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    conditions = [
        (df["padj"] < padj_thresh) & (df["log2FoldChange"] > lfc_thresh),
        (df["padj"] < padj_thresh) & (df["log2FoldChange"] < -lfc_thresh),
    ]
    colors_map = ["#d62728", "#1f77b4"]
    df["color"] = "#aaaaaa"
    for cond, color in zip(conditions, colors_map):
        df.loc[cond, "color"] = color

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = {
        f"Up (LFC > {lfc_thresh}, padj < {padj_thresh})": "#d62728",
        f"Down (LFC < -{lfc_thresh}, padj < {padj_thresh})": "#1f77b4",
        "Not significant": "#aaaaaa",
    }
    for label, color in groups.items():
        mask = df["color"] == color
        ax.scatter(
            df.loc[mask, "log2FoldChange"],
            df.loc[mask, "-log10padj"],
            c=color,
            s=10,
            alpha=0.6,
            linewidths=0,
            label=f"{label} (n={mask.sum()})",
            rasterized=True,
        )

    ax.axhline(-np.log10(padj_thresh), color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(lfc_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(-lfc_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    sig = df[df["color"] != "#aaaaaa"].nsmallest(top_n_labels, "padj")
    for gene, row in sig.iterrows():
        ax.text(
            row["log2FoldChange"],
            row["-log10padj"],
            gene,
            fontsize=7,
            ha="left",
            va="bottom",
        )

    title = f"Volcano plot — {celltype}"
    if contrast:
        title += f"\n{contrast}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("log2 Fold Change", fontsize=11)
    ax.set_ylabel("-log10(padj)", fontsize=11)
    ax.legend(fontsize=8, markerscale=2, framealpha=0.5)
    plt.tight_layout()

    if save:
        if output_path is None:
            fname = f"volcano_{_safe_filename(celltype)}_{_safe_filename(contrast)}.png"
            output_path = os.path.join(os.getcwd(), fname)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Volcano plot saved to: {output_path}")

    plt.show()
    return fig, ax


def _make_pseudobulk(
    adata,
    celltype_col,
    sample_col="sample_id",
    condition_col="condition",
    layer="raw_counts",
):
    pseudobulk_dict = {}

    for ct in adata.obs[celltype_col].dropna().unique():
        print(f"Processing {ct}...")
        ad_ct = adata[adata.obs[celltype_col] == ct].copy()

        X = ad_ct.layers[layer]

        df = pd.DataFrame.sparse.from_spmatrix(
            X,
            index=ad_ct.obs[sample_col],
            columns=ad_ct.var_names,
        )

        pb_df = df.groupby(level=0).sum()
        pb_df.index = pb_df.index.astype(str)

        pb = sc.AnnData(pb_df)

        treatment_map = (
            ad_ct.obs[[sample_col, condition_col]]
            .drop_duplicates()
            .set_index(sample_col)
        )
        treatment_map.index = treatment_map.index.astype(str)
        pb.obs = treatment_map.loc[~treatment_map.index.duplicated(keep="first")].loc[pb_df.index, [condition_col]]

        pseudobulk_dict[ct] = pb

    return pseudobulk_dict


def pseudobulk(
    adata: AnnData,
    celltype: str = "Macrophages",
    cond: list = ["HC", "TEST"],
    save: bool = True,
    output_path: str | None = None,
    sample_col: str = "sample_id",
    condition_col="condition",
    pb_dict: dict | None = None,
    celltype_col: str = "cell_type",
    plot: bool = True,
    lfc_threshold: float = 0.25,
    padj_threshold: float = 0.05,
    layer: str = "raw_counts",
):
    """
    Perform pseudobulk diferential expression analysis for a specific cell type.

    This function:
    1. Generates pseudobulk counts per sample
    2. Selects a specified cell type
    3. Run DESeq2 using treatments as the design factor
    4. Identifies significantly differentially expressed genes

    This module is built on top of PyDESeq2. PyDESeq2 performs its own filtering automatically (documentation can be found here: https://pydeseq2.readthedocs.io/en/stable/). 

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing raw counts and metadata.
    celltype : str, default = "Macrophages"
        Cell type to analyze.
    cond : list[str], default = ["HC", "TEST"].
        Two treatment groups: [reference_condition, comparison_condition].
        LFC is computed as log2(comparison / reference), i.e. log2(TEST / HC).
    save : bool, default = True
        Whether to save the significant genes to a CSV file.
    output_path : str or None, default = None
        Path to save the CSV file.. If None, saves to the curent working directory
        with an automatically generated filename.
    sample_col : str, default = "sample_id"
        Column in annotated data matrix referring to the sample IDs in the data
    condition_col : str, default = "condition"
        Column in annotated data matrix referring to assigned labels of the samples.
    pb_dict : dict, optional
        Pre-built pseudobulk dict from outside. If provided, ``layer`` is ignored
        since the pseudobulk aggregation has already been done.
    plot : bool, default = True
        A boolean deciding whether or not to plot the volcano plot.
    lfc_threshold : float, default = 0.25
        Minimum absolute log2 fold change for a gene to be considered significant
    padj_threshold : float, default = 0.05
        Maximum adjusted p-value for a gene to be considered significant
    layer : str, default = "raw_counts"
        Name of the layer in ``adata`` containing raw integer counts. Ignored if
        ``pb_dict`` is provided.

    Returns
    -------
    pd.DataFrame
        Dataframe containing significant genes with:
        - ``padj`` < ``padj_threshold``
        - ``|log2FoldChange|`` > ``lfc_threshold``

    Notes
    -----
    - Counts are converted to integers (required for DESeq2)
    - The design formula is: ~ assigned_treatment
    - PyDESeq2 is used as a Python implementation of DESeq2
    - If save = True, results are written to disk
    """
    if pb_dict is None:
        adata_ct = adata[adata.obs[celltype_col] == celltype].copy()
        pb_dict = _make_pseudobulk(
            adata_ct,
            sample_col=sample_col,
            condition_col=condition_col,
            celltype_col=celltype_col,
            layer=layer,
        )

    try:
        pb = pb_dict[celltype].copy()
    except KeyError:
        print(f"`{celltype}` is not present in the data. Are you sure you used the correct name?")
        return None

    counts = pd.DataFrame(
        pb.X.toarray(),
        index=pb.obs_names,
        columns=pb.var_names
    )
    metadata = pb.obs[[condition_col]]

    counts = counts.astype(int)

    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors=condition_col,
        refit_cooks=True
    )

    dds.deseq2()

    contrast_list = [condition_col, cond[1], cond[0]]
    stat_res = DeseqStats(dds, contrast=contrast_list)
    stat_res.summary()

    de_df = stat_res.results_df

    if plot:
        plot_volcano(
            de_df,
            celltype=celltype,
            contrast=f"{cond[0]}_vs_{cond[1]}",
            save=save,
            output_path=output_path.replace(".csv", "_volcano.png") if output_path else None,
            padj_thresh=padj_threshold,
            lfc_thresh=lfc_threshold,
        )

    sig = de_df.query("padj < @padj_threshold & abs(log2FoldChange) > @lfc_threshold")

    if save:
        if output_path is None:
            output_path = os.path.join(os.getcwd(), f"sig_genes_{_safe_filename(celltype)}_{contrast_list[1]}_{contrast_list[2]}.csv")
        sig.to_csv(output_path)
        print(f"The significant genes are written to a CSV file in the current running directory: {output_path}")
    return sig


def pseudobulk_per_condition(
    adata: AnnData,
    all_conditions: list,
    condition_col: str = "label",
    sample_col: str = "sample_id",
    celltype_col: str = "cell_type",
    save: bool = True,
    output_path: str | None = None,
    lfc_threshold: float = 0.25,
    padj_threshold: float = 0.05,
    layer: str = "raw_counts",
):
    """
    Perform pseudobulk differential expression analysis across all condition
    pairs.

    Iterates over every pairwise combination of the provided conditions,
    subsets the data to the relevant samples, and runs pseudobulk DE analysis
    (via ``pseudobulk``) for every celltype present in each subset. Results are
    collected and optionally saved to disk.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing raw counts and metadata.
    all_conditions : list of str
        List of condition labels to compare. All combinations are tested.
    condition_col : str, default = "label"
        Column in ``adata.obs`` containing the condition/treatment group labels.
        It must match values provided in ``all_conditions``.
    sample_col : str, default = "sample_id"
        Column in ``adata.obs`` identifying biological samples.
    celltype_col : str, default = "cell_type"
        Column in ``adata.obs`` specifying cell type annotations.
    save : bool, default = True
        Whether to save the significant genes to a CSV file.
    output_path : str or None, default = None
        Path to save the CSV file. If None, saves to the current working directory
        with an automatically generated filename.
    lfc_threshold : float, default = 0.25
        Minimum absolute log2 fold change for a gene to be considered significant.
        Passed to each ``pseudobulk`` call.
    padj_threshold : float, default = 0.05
        Maximum adjusted p-value for a gene to be considered significant.
        Passed to each ``pseudobulk`` call.
    layer : str, default = "raw_counts"
        Name of the layer in ``adata`` containing raw integer counts.
        Passed to each ``pseudobulk`` call.

    Returns
    -------
    dict
        Nested dictionary with keys of the form ``(cell_type, "condA_vs_condB")``
        and values being DataFrames of significant DE genes (``padj`` <
        ``padj_threshold``, ``|log2FoldChange|`` > ``lfc_threshold``) for that
        cell type and comparison. Comparisons or cell types that fail or are
        skipped are excluded.

    Notes
    -----
    - Comparisons are skipped if either condition has fewer than 2 samples,
      as DESeq2 requires biological replication.
    - Results CSVs are saved to ``pseudobulk_results/{celltype}_{condA}_vs_{condB}.csv``
    - Any cell type that raises an exception during DE analysis is skipped with a warning,
      allowing the remaining comparisons to proceed.
    """
    condition_pairs = list(combinations(all_conditions, 2))
    results = {}

    if condition_col not in adata.obs:
        print(f"The provided label column ({condition_col}) is not present in the data.\n Available columns: {adata.obs.columns}")
        return

    if save:
        base_dir = output_path if output_path is not None else os.path.join(os.getcwd(), "pseudobulk_results")
        os.makedirs(base_dir, exist_ok=True)

    for cond_a, cond_b in condition_pairs:
        key = f"{cond_a}_vs_{cond_b}"
        mask = adata.obs[condition_col].isin([cond_a, cond_b])
        adata_sub = adata[mask].copy()

        samples_per_cond = adata.obs.groupby(condition_col)[sample_col].nunique()
        if samples_per_cond.min() < 2:
            print(f"Skipped {key}: not enough samples.")
            continue

        pb_dict = _make_pseudobulk(
            adata_sub,
            celltype_col=celltype_col,
            sample_col=sample_col,
            condition_col=condition_col,
            layer=layer,
        )

        for ct_focus in pb_dict.keys():
            print(f"Processing {ct_focus}...")
            try:
                ct_output_path = (
                    os.path.join(base_dir, f"{ct_focus}_{key}.csv")
                    if save else None
                )

                sig_genes = pseudobulk(
                    adata_sub,
                    celltype=ct_focus,
                    celltype_col=celltype_col,
                    sample_col=sample_col,
                    cond=[cond_a, cond_b],
                    condition_col=condition_col,
                    pb_dict=pb_dict,
                    save=save,
                    output_path=ct_output_path,
                    lfc_threshold=lfc_threshold,
                    padj_threshold=padj_threshold,
                    layer=layer,
                )

                results[(ct_focus, key)] = sig_genes
            except Exception as e:
                print(f"Skipped {ct_focus}: {e}")
    return results