"""
consensus_annotation.py
========================
Consensus cell type annotation.

Aggregates per-spot cell type results from any combination of annotation
methods into a single consensus call.

Two input modes
---------------
Fraction matrices (obsm)
    Methods that produce a (spots × cell_types) probability/fraction
    DataFrame store it in ``adata.obsm``. Pass ``obsm_keys`` to name them,
    or leave it None for auto-detection.

Dominant-label columns (obs)
    Methods that only write a single predicted label per spot (e.g. a
    custom classifier stored in ``adata.obs["my_method"]``) are passed via
    ``obs_keys``. They are one-hot encoded internally and treated as a
    fraction matrix with weight 1/n_methods, so they participate in all
    three consensus strategies on equal footing.

Both modes can be combined freely.

Consensus strategies
--------------------
- ``"weighted_average"``  : weighted mean of fraction matrices (default)
- ``"majority_vote"``     : plurality vote on each method's dominant label
- ``"rank_aggregation"``  : Borda-count rank aggregation of fraction matrices

All strategies write:
  - ``adata.obsm[f"{result_key}_fractions"]``   – (spots × cell types) DataFrame
  - ``adata.obs[f"{result_key}_cell_type"]``    – dominant cell type per spot
  - ``adata.obs[f"{result_key}_confidence"]``   – fraction assigned to winner [0-1]
  - ``adata.obs[f"{result_key}_agreement"]``    – fraction of methods agreeing
                                                   with the consensus call [0-1]
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default registry: known obsm keys written by celltype_annotation.py
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
# Internal helpers
# ---------------------------------------------------------------------------

def _one_hot_from_obs_col(
    adata: ad.AnnData,
    obs_col: str,
) -> pd.DataFrame:
    """
    Convert a categorical obs column (dominant label per spot) into a
    binary fraction DataFrame (spots × cell_types).

    Each row has exactly one 1.0 — the predicted cell type — and 0.0
    everywhere else. This lets obs-based methods participate in all three
    consensus strategies without special-casing.
    """
    series = adata.obs[obs_col].astype(str)
    dummies = pd.get_dummies(series).astype(float)
    dummies.index = adata.obs_names
    return dummies


def _to_fraction_df(
    mat: Union[pd.DataFrame, np.ndarray],
    obs_names: pd.Index,
) -> pd.DataFrame:
    """Ensure mat is a row-normalised DataFrame indexed by obs_names."""
    if not isinstance(mat, pd.DataFrame):
        mat = pd.DataFrame(mat, index=obs_names)
    # Row-normalise (guard against zero-sum rows)
    row_sums = mat.sum(axis=1).replace(0, 1)
    return mat.div(row_sums, axis=0)


def _detect_obsm_keys(adata: ad.AnnData) -> Dict[str, str]:
    """
    Auto-detect annotation fraction matrices in adata.obsm.

    Checks all known default keys first, then picks up any additional
    obsm entry whose value is a DataFrame with string column names
    (heuristic for a cell-type fraction matrix).
    """
    found: Dict[str, str] = {}

    # 1. Check known keys from the default registry
    for method, key in _DEFAULT_OBSM_KEYS.items():
        if key in adata.obsm:
            found[method] = key

    # 2. Pick up any unknown obsm DataFrames that look like fraction matrices
    known_keys = set(_DEFAULT_OBSM_KEYS.values())
    for key, val in adata.obsm.items():
        if key in known_keys:
            continue
        if isinstance(val, pd.DataFrame) and val.columns.dtype == object:
            found[key] = key  # use the obsm key itself as the method name

    return found


def _detect_obs_keys(adata: ad.AnnData) -> Dict[str, str]:
    """
    Auto-detect dominant-label columns in adata.obs.

    Picks up any obs column ending in ``_cell_type`` or ``_dominant``
    that looks like a categorical cell-type annotation, excluding columns
    written by consensus itself.
    """
    found: Dict[str, str] = {}
    for col in adata.obs.columns:
        if col.startswith("consensus_"):
            continue
        if col.endswith("_cell_type") or col.endswith("_dominant"):
            if adata.obs[col].dtype == object or hasattr(adata.obs[col], "cat"):
                found[col] = col
    return found


def _collect_fraction_matrices(
    adata: ad.AnnData,
    obsm_keys: Optional[Dict[str, str]],
    obs_keys: Optional[Dict[str, str]],
    auto_detect: bool,
) -> tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Gather all fraction matrices from obsm and obs sources.

    Returns
    -------
    available : dict of method_name → fraction DataFrame
    skipped   : list of human-readable strings describing missing sources
    """
    resolved_obsm: Dict[str, str] = {}
    resolved_obs:  Dict[str, str] = {}

    if auto_detect and obsm_keys is None and obs_keys is None:
        resolved_obsm = _detect_obsm_keys(adata)
        resolved_obs  = _detect_obs_keys(adata)
    else:
        resolved_obsm = obsm_keys or {}
        resolved_obs  = obs_keys  or {}

    available: Dict[str, pd.DataFrame] = {}
    skipped:   List[str] = []

    # --- obsm sources ---
    for method, key in resolved_obsm.items():
        if key in adata.obsm:
            available[method] = _to_fraction_df(adata.obsm[key], adata.obs_names)
        else:
            skipped.append(f"{method} (obsm['{key}'] not found)")

    # --- obs sources ---
    for method, col in resolved_obs.items():
        if col in adata.obs.columns:
            available[method] = _one_hot_from_obs_col(adata, col)
        else:
            skipped.append(f"{method} (obs['{col}'] not found)")

    return available, skipped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_consensus(
    adata: ad.AnnData,
    obsm_keys: Optional[Dict[str, str]] = None,
    obs_keys: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
    strategy: ConsensusStrategy = "weighted_average",
    result_key: str = "consensus",
    min_methods: int = 2,
    auto_detect: bool = True,
) -> ad.AnnData:
    """
    Build a consensus cell type annotation from multiple method outputs.

    Reads per-spot cell type information already stored in ``adata.obsm``
    (fraction/probability matrices) and/or ``adata.obs`` (dominant-label
    columns), and combines them into a single consensus call.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData that has been annotated by one or more methods.
    obsm_keys : dict, optional
        Mapping of ``{method_name: obsm_key}`` pointing to a (spots ×
        cell_types) fraction DataFrame for each method. Example::

            obsm_keys = {
                "celltypist": "celltypist_cell_type_prob_matrix",
                "my_method":  "my_method_fractions",
            }

        If None and ``auto_detect=True``, all known and unknown fraction
        DataFrames in ``adata.obsm`` are used automatically.
    obs_keys : dict, optional
        Mapping of ``{method_name: obs_column}`` for methods that only
        stored a single dominant label per spot in ``adata.obs``. Example::

            obs_keys = {
                "sctype":  "sctype_cell_type",
                "manual":  "manual_annotation",
            }

        These columns are one-hot encoded internally so they participate
        in all three consensus strategies on equal footing with fraction
        matrices. If None and ``auto_detect=True``, obs columns ending
        in ``_cell_type`` or ``_dominant`` are picked up automatically.
    weights : dict, optional
        Per-method weights for the ``"weighted_average"`` strategy.
        Keys must match those in ``obsm_keys`` / ``obs_keys``. Missing
        keys default to 1.0. Weights are normalised internally.
        Ignored by ``"majority_vote"`` and ``"rank_aggregation"``.
    strategy : {"weighted_average", "majority_vote", "rank_aggregation"}
        How to combine method outputs:

        - ``"weighted_average"``  – weighted mean of fraction matrices.
          Best when methods produce calibrated probability outputs.
        - ``"majority_vote"``     – plurality vote on each method's dominant
          label. Best when you trust per-method calls over continuous scores.
        - ``"rank_aggregation"``  – Borda-count: ranks cell types per method
          per spot, then sums ranks. Robust to scale differences between
          methods.
    result_key : str
        Prefix for output keys written to ``adata``:

        - ``adata.obsm[f"{result_key}_fractions"]``
        - ``adata.obs[f"{result_key}_cell_type"]``
        - ``adata.obs[f"{result_key}_confidence"]``
        - ``adata.obs[f"{result_key}_agreement"]``
        - ``adata.uns[f"{result_key}_methods"]``
        - ``adata.uns[f"{result_key}_strategy"]``

    min_methods : int
        Minimum number of methods that must be found before attempting
        consensus. Raises ``ValueError`` if fewer are available.
    auto_detect : bool
        When ``obsm_keys`` and ``obs_keys`` are both None, automatically
        scan ``adata.obsm`` and ``adata.obs`` for annotation results.
        Set to False to require explicit keys.

    Returns
    -------
    AnnData
        The input ``adata`` with consensus results written in place.

    Raises
    ------
    ValueError
        If fewer than ``min_methods`` sources are found.

    Examples
    --------
    **Auto-detect everything** (after running celltypist + tangram + tacco):

    >>> STAIA.annotate_consensus(adata)
    >>> adata.obs["consensus_cell_type"]

    **Use only specific obsm fraction matrices:**

    >>> STAIA.annotate_consensus(
    ...     adata,
    ...     obsm_keys={
    ...         "celltypist": "celltypist_cell_type_prob_matrix",
    ...         "tangram":    "tangram_cell_type",
    ...     },
    ... )

    **Mix fraction matrices with a custom obs column:**

    >>> STAIA.annotate_consensus(
    ...     adata,
    ...     obsm_keys={"tacco": "tacco_cell_type"},
    ...     obs_keys={"sctype": "sctype_cell_type"},
    ... )

    **Weight celltypist more heavily:**

    >>> STAIA.annotate_consensus(
    ...     adata,
    ...     weights={"celltypist": 2.0, "tangram": 1.0, "tacco": 1.0},
    ... )

    **Use majority vote with a custom result key:**

    >>> STAIA.annotate_consensus(adata, strategy="majority_vote", result_key="vote")
    >>> adata.obs["vote_cell_type"]
    """
    weights = weights or {}

    # 1. Gather all fraction matrices
    available, skipped = _collect_fraction_matrices(
        adata, obsm_keys, obs_keys, auto_detect
    )

    if skipped:
        warnings.warn(
            "The following sources were not found and will be skipped:\n  "
            + "\n  ".join(skipped)
            + "\nRun the corresponding annotation methods first, or check "
            "your key names.",
            UserWarning,
            stacklevel=2,
        )

    if len(available) < min_methods:
        raise ValueError(
            f"Only {len(available)} annotation source(s) found "
            f"({list(available)}), but min_methods={min_methods}. "
            f"Run at least {min_methods} annotation methods before calling "
            f"annotate_consensus(), or lower min_methods."
        )

    print(
        f"Building consensus from {len(available)} method(s) "
        f"using strategy='{strategy}': {list(available)}"
    )

    # 2. Align cell type columns across all methods (union, fill missing with 0)
    all_cell_types: List[str] = sorted(
        set().union(*[set(df.columns) for df in available.values()])
    )
    aligned: Dict[str, pd.DataFrame] = {
        method: df.reindex(columns=all_cell_types, fill_value=0.0)
        for method, df in available.items()
    }

    # 3. Build consensus fractions
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

    # Row-normalise (guard against zero-sum rows)
    row_sums = consensus_fractions.sum(axis=1).replace(0, 1)
    consensus_fractions = consensus_fractions.div(row_sums, axis=0)

    # 4. Derive per-spot summary statistics
    dominant_cell_type = consensus_fractions.idxmax(axis=1)
    confidence         = consensus_fractions.max(axis=1)

    method_dominant = pd.DataFrame(
        {m: df.idxmax(axis=1) for m, df in aligned.items()},
        index=adata.obs_names,
    )
    agreement = method_dominant.eq(dominant_cell_type, axis=0).mean(axis=1)

    # 5. Write results back to adata
    adata.obsm[f"{result_key}_fractions"]  = consensus_fractions
    adata.obs[f"{result_key}_cell_type"]   = dominant_cell_type.values
    adata.obs[f"{result_key}_confidence"]  = confidence.astype(float).values
    adata.obs[f"{result_key}_agreement"]   = agreement.astype(float).values
    adata.uns[f"{result_key}_methods"]     = list(available.keys())
    adata.uns[f"{result_key}_strategy"]    = strategy

    return adata


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _weighted_average(
    aligned: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
) -> pd.DataFrame:
    """
    Weighted mean of fraction matrices.

    Methods absent from ``weights`` get weight 1.0. Weights are normalised
    internally so they do not need to sum to 1.
    """
    total_weight = 0.0
    accumulator: Optional[pd.DataFrame] = None

    for method, df in aligned.items():
        w = weights.get(method, 1.0)
        total_weight += w
        accumulator = df * w if accumulator is None else accumulator + df * w

    return accumulator / total_weight  # type: ignore[operator]


def _majority_vote(
    aligned: Dict[str, pd.DataFrame],
    all_cell_types: List[str],
) -> pd.DataFrame:
    """
    Plurality vote on each method's dominant (argmax) label.

    Returns a vote-count matrix where each row records the number of methods
    that called each cell type. Ties are broken at the argmax step downstream.
    """
    obs_names = next(iter(aligned.values())).index

    dominant = pd.DataFrame(
        {m: df.idxmax(axis=1) for m, df in aligned.items()},
        index=obs_names,
    )

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
    highest rank). Ranks are summed across methods. Robust to calibration
    differences between methods (e.g. mixing probabilities with one-hot obs
    columns).
    """
    obs_names      = next(iter(aligned.values())).index
    all_cell_types = next(iter(aligned.values())).columns.tolist()

    borda = pd.DataFrame(0.0, index=obs_names, columns=all_cell_types)
    for df in aligned.values():
        borda += df.rank(axis=1, method="average")

    return borda


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def summarise_agreement(
    adata: ad.AnnData,
    obsm_keys: Optional[Dict[str, str]] = None,
    obs_keys: Optional[Dict[str, str]] = None,
    result_key: str = "consensus",
    auto_detect: bool = True,
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
    obs_keys : dict, optional
        Same mapping used in ``annotate_consensus()``.
    result_key : str
        If consensus has been run, includes its call in the summary.
    auto_detect : bool
        Whether to auto-detect sources when no explicit keys are given.

    Returns
    -------
    pd.DataFrame
        One column per method (dominant label) plus consensus columns
        if they exist.
    """
    available, _ = _collect_fraction_matrices(
        adata, obsm_keys, obs_keys, auto_detect
    )

    summary = pd.DataFrame(index=adata.obs_names)

    for method, df in available.items():
        summary[f"{method}_call"] = df.idxmax(axis=1).values

    for col in [
        f"{result_key}_cell_type",
        f"{result_key}_confidence",
        f"{result_key}_agreement",
    ]:
        if col in adata.obs.columns:
            summary[col] = adata.obs[col].values

    return summary