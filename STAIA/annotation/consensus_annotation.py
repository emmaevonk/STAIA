"""
consensus_annotation.py
========================
Consensus cell type annotation.

This module aggregates per-spot cell type fraction matrices produced by any
combination of CellTypist, Tangram, Cell2location, and TACCO into a single
consensus call, using one of three strategies:

  - ``"weighted_average"``  : weighted mean of fraction matrices (default)
  - ``"majority_vote"``     : plurality vote on each method's dominant label
  - ``"rank_aggregation"``  : Borda-count rank aggregation of fraction matrices

All strategies write:
  - ``adata.obsm["consensus_fractions"]``  – (spots × cell types) DataFrame
  - ``adata.obs["consensus_cell_type"]``   – dominant cell type per spot
  - ``adata.obs["consensus_confidence"]``  – fraction assigned to winner [0-1]
  - ``adata.obs["consensus_agreement"]``   – fraction of methods that agree
                                             with the consensus call [0-1]
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default method registry
# Entries: obsm key that holds the (spots × cell_types) fraction DataFrame
# ---------------------------------------------------------------------------
_DEFAULT_OBSM_KEYS: Dict[str, str] = {
    "celltypist":    "celltypist_cell_type_prob_matrix",
    "tangram":       "tangram_cell_type",
    "cell2location": "cell2location",
    "tacco":         "tacco_cell_type",
}

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "celltypist":    1.0,
    "tangram":       1.0,
    "cell2location": 1.0,
    "tacco":         1.0,
}

ConsensusStrategy = Literal["weighted_average", "majority_vote", "rank_aggregation"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_consensus(
    adata: ad.AnnData,
    obsm_keys: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
    strategy: ConsensusStrategy = "weighted_average",
    result_key: str = "consensus",
    min_methods: int = 2,
) -> Optional[ad.AnnData]:
    """
    Build a consensus cell type annotation from multiple method outputs.

    Reads per-spot cell type fraction matrices (spots x cell types DataFrames)
    already stored in ``adata.obsm`` by the individual annotation wrappers,
    and combines them into a single consensus call.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData that has been annotated by one or more methods.
    obsm_keys : dict, optional
        Mapping of ``{method_name: obsm_key}`` pointing to the fraction
        DataFrame for each method. Defaults to the keys written by the
        wrappers in ``cell_type_annotation.py``:

        .. code-block:: python

            {
                "celltypist":    "celltypist_cell_type_prob_matrix",
                "tangram":       "tangram_cell_type",
                "cell2location": "cell2location",
                "tacco":         "tacco_cell_type",
            }

        You can pass a subset to use only specific methods, or add custom
        keys for any other method that stores fractions in ``adata.obsm``.
    weights : dict, optional
        Per-method weights for the ``"weighted_average"`` strategy.
        Keys must match ``obsm_keys``. Missing keys default to 1.0.
        Weights are normalised internally so they do not need to sum to 1.
        Ignored by ``"majority_vote"`` and ``"rank_aggregation"``.
    strategy : {"weighted_average", "majority_vote", "rank_aggregation"}
        How to combine method outputs:

        - ``"weighted_average"``  – weighted mean of fraction matrices.
          Best when methods produce calibrated probability/fraction outputs.
        - ``"majority_vote"``     – plurality vote on each method's dominant
          label (the argmax of its fraction matrix). Best when you trust
          individual method calls more than their continuous scores.
        - ``"rank_aggregation"``  – Borda-count: cell types are ranked per
          method per spot and ranks summed. Robust to scale differences
          between methods.
    result_key : str
        Prefix for output keys:
          - ``adata.obsm[f"{result_key}_fractions"]``
          - ``adata.obs[f"{result_key}_cell_type"]``
          - ``adata.obs[f"{result_key}_confidence"]``
          - ``adata.obs[f"{result_key}_agreement"]``
    min_methods : int
        Minimum number of methods that must be present in ``adata.obsm``
        to attempt consensus. Raises ``ValueError`` if fewer are found.

    Returns
    -------
    AnnData or None

    Raises
    ------
    ValueError
        If fewer than ``min_methods`` fraction matrices are found in
        ``adata.obsm``.
    """
    obsm_keys = obsm_keys or _DEFAULT_OBSM_KEYS
    weights   = weights   or _DEFAULT_WEIGHTS

    # 1. Collect whichever fraction matrices are present
    available: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    for method, key in obsm_keys.items():
        if key in adata.obsm:
            mat = adata.obsm[key]
            if not isinstance(mat, pd.DataFrame):
                mat = pd.DataFrame(mat, index=adata.obs_names)
            available[method] = mat
        else:
            missing.append(f"{method} ({key!r})")

    if missing:
        warnings.warn(
            f"The following methods were not found in adata.obsm and will be "
            f"skipped: {', '.join(missing)}. "
            f"Run the corresponding annotation wrappers first.",
            UserWarning,
            stacklevel=2,
        )

    if len(available) < min_methods:
        raise ValueError(
            f"Only {len(available)} method(s) found in adata.obsm, but "
            f"min_methods={min_methods}. Found: {list(available)}. "
            f"Run at least {min_methods} annotation methods before calling "
            f"annotate_consensus()."
        )

    # 2. Align cell type columns across methods
    all_cell_types = sorted(
        set().union(*[set(df.columns) for df in available.values()])
    )

    # Re-index every DataFrame to the full union of cell types (fill with 0)
    aligned: Dict[str, pd.DataFrame] = {
        method: df.reindex(columns=all_cell_types, fill_value=0.0)
        for method, df in available.items()
    }

    # 3. Build consensus fractions via chosen strategy
    if strategy == "weighted_average":
        consensus_fractions = _weighted_average(aligned, weights)

    elif strategy == "majority_vote":
        consensus_fractions = _majority_vote(aligned, all_cell_types)

    elif strategy == "rank_aggregation":
        consensus_fractions = _rank_aggregation(aligned)

    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Choose from 'weighted_average', 'majority_vote', 'rank_aggregation'."
        )

    # Row-normalise so fractions sum to 1 (guarded against zero-sum rows)
    row_sums = consensus_fractions.sum(axis=1).replace(0, 1)
    consensus_fractions = consensus_fractions.div(row_sums, axis=0)

    # 4. Derive dominant label, confidence, and agreement
    dominant_cell_type = consensus_fractions.idxmax(axis=1)
    confidence         = consensus_fractions.max(axis=1)

    # Agreement: fraction of methods whose dominant label matches consensus
    method_dominant = pd.DataFrame(
        {m: df.idxmax(axis=1) for m, df in aligned.items()},
        index=adata.obs_names,
    )
    agreement = method_dominant.eq(dominant_cell_type, axis=0).mean(axis=1)

    # 5. Write results
    adata.obsm[f"{result_key}_fractions"]  = consensus_fractions
    adata.obs[f"{result_key}_cell_type"]   = dominant_cell_type.values
    adata.obs[f"{result_key}_confidence"]  = confidence.values
    adata.obs[f"{result_key}_agreement"]   = agreement.values

    # Store metadata about which methods contributed
    adata.uns[f"{result_key}_methods"]  = list(available.keys())
    adata.uns[f"{result_key}_strategy"] = strategy

    return adata

# Strategy implementations

def _weighted_average(
    aligned: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
) -> pd.DataFrame:
    """
    Weighted mean of fraction matrices.

    Each method's matrix is multiplied by its weight, summed, then divided
    by the total weight. Methods not listed in ``weights`` get weight 1.0.
    """
    total_weight = 0.0
    accumulator: Optional[pd.DataFrame] = None

    for method, df in aligned.items():
        w = weights.get(method, 1.0)
        total_weight += w
        if accumulator is None:
            accumulator = df * w
        else:
            accumulator = accumulator + df * w

    return accumulator / total_weight  # type: ignore[operator]


def _majority_vote(
    aligned: Dict[str, pd.DataFrame],
    all_cell_types: List[str],
) -> pd.DataFrame:
    """
    Plurality vote on each method's dominant (argmax) label.

    Returns a binary-like matrix where each row sums to 1, with the
    fraction of votes received by each cell type.
    """
    n_spots   = next(iter(aligned.values())).shape[0]
    obs_names = next(iter(aligned.values())).index

    # Matrix of dominant labels: (spots × methods)
    dominant = pd.DataFrame(
        {m: df.idxmax(axis=1) for m, df in aligned.items()},
        index=obs_names,
    )

    # Count votes per cell type
    vote_matrix = pd.DataFrame(0.0, index=obs_names, columns=all_cell_types)
    for col in dominant.columns:
        for ct in all_cell_types:
            vote_matrix[ct] += (dominant[col] == ct).astype(float)

    return vote_matrix


def _rank_aggregation(
    aligned: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Borda-count rank aggregation.

    For each method, cell types are ranked per spot (highest fraction →
    highest rank = n_cell_types). Ranks are summed across methods. The
    resulting Borda scores serve as the consensus fractions (after
    row-normalisation by the caller).
    """
    obs_names      = next(iter(aligned.values())).index
    all_cell_types = next(iter(aligned.values())).columns.tolist()
    n_ct           = len(all_cell_types)

    borda = pd.DataFrame(0.0, index=obs_names, columns=all_cell_types)

    for df in aligned.values():
        # rank(axis=1): lowest value → rank 1; highest → rank n_ct
        ranks = df.rank(axis=1, method="average")
        borda = borda + ranks

    return borda

# Convenience helper: summarise per-spot method agreement

def summarise_agreement(
    adata: ad.AnnData,
    obsm_keys: Optional[Dict[str, str]] = None,
    result_key: str = "consensus",
) -> pd.DataFrame:
    """
    Return a per-spot summary DataFrame comparing all method calls.

    Useful for inspecting where methods agree or disagree before or after
    running ``annotate_consensus()``.

    Parameters
    ----------
    adata : AnnData
        Annotated spatial AnnData.
    obsm_keys : dict, optional
        Same mapping used in ``annotate_consensus()``.
    result_key : str
        If consensus has been run, includes its call in the summary.

    Returns
    -------
    pd.DataFrame
        Columns: one per method (dominant label), plus ``consensus_cell_type``,
        ``consensus_confidence``, and ``consensus_agreement`` if available.
    """
    obsm_keys = obsm_keys or _DEFAULT_OBSM_KEYS

    summary = pd.DataFrame(index=adata.obs_names)

    for method, key in obsm_keys.items():
        if key in adata.obsm:
            df = adata.obsm[key]
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df, index=adata.obs_names)
            summary[f"{method}_call"] = df.idxmax(axis=1).values

    for col in [
        f"{result_key}_cell_type",
        f"{result_key}_confidence",
        f"{result_key}_agreement",
    ]:
        if col in adata.obs.columns:
            summary[col] = adata.obs[col].values
    return summary