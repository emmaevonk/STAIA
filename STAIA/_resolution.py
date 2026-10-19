"""
Generating metrics to see which Leiden resolution is best to use according to the Adjusted Rand Index (ARI).
"""

import numpy as np
import scanpy as sc
import igraph as ig
import leidenalg
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _build_graph_from_h5ad(
    adata,
    n_neighbors=15,
    n_pcs=30,
):
    """
    Build a weighted undirected igraph from an adata object.

    The function ensures PCA is performed, computes a k-nearest neighbor
    graph in PCA space via scanpy, then converts the sparse connectivity
    matrix into an igraph suitable for leidenalg.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must contain PCA embeddings or raw/
        normalized counts from which PCA can be computed.
    n_neighbors : int, optional
        Number of nearest neighbors sed when constructing the kNN graph.
        Default is 15.
    n_pcs : int, optional
        Number of PCs to use for the neighbor search. 
        Default is 30.

    Returns
    -------
    graph : igraph.Graph
        Undirected weighted graph where each node corresponds to a cell and
        edge weights reflect kNN connectivity strengths.
    adata : anndata.AnnData
        The annotated data matrix.

    Notes
    -----
    If PCA embeddings are absent, ``sc.pp.pca`` is called in-place, which
    modifies adata.
    """
    # If PCA not present, compute it
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=n_pcs)

    # Build kNN graph (reuses PCA automatically)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    adj = adata.obsp["connectivities"]

    sources, targets = adj.nonzero()
    weights = adj[sources, targets].A1

    graph = ig.Graph(
        n=adj.shape[0],
        edges=list(zip(sources, targets)),
        edge_attrs={"weight": weights},
        directed=False
    )
    return graph, adata


def _compute_stability_fast(partitions):
    """
    Estimate partition stability using ARI.

    Compares every partition against the first one (used as a 
    reference) and returns the mean ARI. A score near 1.0 means the algorithm
    consistently produces the same clustering across different random seeds.

    Parameters
    ----------
    partitions : list of array-like
        Each element is a sequence of integer cluster labels, one per cell,
        produced by a single Leiden run. Must contain at least two elements
        for a meaningful comparison.

    Returns
    -------
    float
        Mean ARI between the first partition and the rest. 
        Returns 1.0 when fewer than two partitions are provided.

    Notes
    -----
    ARI ranges from -1 to 1; random labellings score around 0 and perfect
    agreement scores 1.
    """
    base = partitions[0]
    scores = [
        adjusted_rand_score(base, p)
        for p in partitions[1:]
    ]
    return np.mean(scores) if scores else 1.0


def _leiden_auto_resolution(
    graph,
    n_cells,
    resolution_range=(0.2, 2.0),
    resolution_step=0.2,
    n_runs=3,
    random_state=42,
    fast_mode=False,
):
    """
    Search for the best Leiden resolution over a grid of candidate values.

    For each candidate resolution the function:
    1. Runs the leiden algorithm with different random seeds
    2. Records the number of clusters, ARI and mean modularity
    3. Filters out resolutions that produce too many clusters
    4. Ranks the remaining resolutions by a composite score.

    Parameters
    ----------
    graph : igraph.Graph
        Weighted kNN graph
    n_cells : int
        Total number of cells.
    resolution_range : tuple, optional
        min and max resolution for the grid search.
        Default is (0.2, 2.0)
    resolution_step : float, optional
        Step size between consecutive candidate resolutions.
        Default is 2.0
    n_runs : int, optional
        Number of independent Leiden runs per resolution.
    random_state : int, optional
        Random state provided by the user.
        Default is 42
    fast_mode : boolean, optional
        Boolean value showing whether fast mode of the algorithm
        should be used.
        Default is False.

    Returns
    -------
    best_resolution : float
        Resolution value that maximises the composite stability–modularity
        score while satisfying the over-fragmentation guard.
    df : pandas.DataFrame
        Summary table with columns ``resolution``, ``n_clusters``,
        ``stability``, ``modularity``, and ``score`` for every evaluated
        resolution that passed the fragmentation filter.

    Notes
    -----
    * The partition algorithm used is ``RBConfigurationVertexPartition``
      (Reichardt–Bornholdt with configuration-model null), which supports
      a resolution parameter directly.
    * Resolutions are evaluated in ascending order; the first element of each
      partition list becomes the ARI reference, so the reference itself is
      always included in the returned membership arrays.
    """
    rng = np.random.default_rng(random_state)
    resolutions = np.arange(
        resolution_range[0],
        resolution_range[1] + resolution_step,
        resolution_step
    )

    results = []
    for res in resolutions:
        partitions = []
        modularities = []
        runs = 1 if fast_mode else n_runs
        for _ in range(runs):
            seed = int(rng.integers(0, 1_000_000))

            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=res,
                weights="weight",
                seed=seed,
            )

            partitions.append(partition.membership)
            modularities.append(partition.modularity)

        n_clusters = len(set(partitions[0]))
        stability = 1.0 if fast_mode else _compute_stability_fast(partitions)
        results.append({
            "resolution": res,
            "n_clusters": n_clusters,
            "stability": stability,
            "modularity": np.mean(modularities),
        })
    df = pd.DataFrame(results)

    # removing resolutions with too many clusters
    df = df[df["n_clusters"] < np.sqrt(n_cells)]
    df["score"] = 0.7 * df["stability"] + 0.3 * df["modularity"]
    best_row = df.sort_values("score", ascending=False).iloc[0]
    return best_row["resolution"], df


def compute_leiden_resolution(
    adata,
    resolution="auto",
    fast_mode=False,
    n_neighbors=15,
    n_pcs=30,
    n_runs=3,
):
    """
    Cluster cells with the Leiden algorithm, optionally auto-selecting the
    resolution parameter.

    This is the primary public interface for the module.  It orchestrates
    graph construction, optional resolution search, and final clustering,
    then writes cluster labels into ``adata.obs["leiden"]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing cells × genes (or cells × features).
        A PCA embedding (``obsm["X_pca"]``) is computed automatically if
        absent.
    resolution : float or "auto", optional
        * ``"auto"`` (default): Run the grid search described in
          :func:`_leiden_auto_resolution` to select the best resolution
          automatically.
        * A positive float: Use this resolution directly, skipping the search.
    fast_mode : bool, optional
        Passed to :func:`_leiden_auto_resolution` when ``resolution="auto"``.
        Reduces computation time at the cost of skipping stability estimation.
        Has no effect when a fixed resolution is supplied.
        Default is ``False``.
    n_neighbors : int, optional
        Number of nearest neighbours for kNN graph construction.
        Default is ``15``.
    n_pcs : int, optional
        Number of principal components used during neighbour search.
        Default is ``30``.
    n_runs : int, optional
        Number of Leiden runs per resolution candidate when estimating
        stability (used only when ``resolution="auto"`` and
        ``fast_mode=False``).
        Default is ``3``.

    Returns
    -------
    adata : anndata.AnnData
        Input object with an additional column ``adata.obs["leiden"]``
        containing integer cluster labels from the final Leiden partition.
    best_resolution : float
        The resolution that was used for the final clustering (equals the
        input *resolution* when a fixed value is provided).
    table : pandas.DataFrame or None
        When ``resolution="auto"``, a DataFrame summarising all evaluated
        resolutions (columns: ``resolution``, ``n_clusters``, ``stability``,
        ``modularity``, ``score``).  ``None`` when a fixed resolution is used.
    """
    graph, adata = _build_graph_from_h5ad(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
    )
    if resolution == "auto":
        best_res, table = _leiden_auto_resolution(
            graph,
            n_cells=adata.n_obs,
            n_runs=n_runs,
            fast_mode=fast_mode,
        )
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=best_res,
            weights="weight",
        )
        adata.obs["leiden"] = partition.membership
        return adata, best_res, table
    else:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="weight",
        )
        adata.obs["leiden"] = partition.membership
        return adata, resolution, None