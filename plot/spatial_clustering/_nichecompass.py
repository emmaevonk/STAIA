"""
nichecompass_STAIA.py
=====================
NicheCompass spatial niche analysis.
Supports multi-sample training with per-sample spatial graphs and
batch correction via categorical covariates.

Three functions
-------------
niche_analysis_train(adatas, sample_names, gp_dict_path, ...)
    Full pipeline: build per-sample spatial graphs, stitch them into one
    disconnected graph, load GP dict from JSON, train model, cluster niches
    at multiple resolutions, and save everything to disk.
    Run this once (or once per new cohort).

niche_analysis_load(model_folder_path, ...)
    Load the saved adata and model weights from disk, then re-run clustering
    and visualization. No retraining, no GP rebuilding needed. The saved
    adata already has all GP masks baked in.

save_gp_dict()
    Generated a combined gene-program (GP) dictionary and saving
    it to a given path. This has to be run once (before training)

Example usage
-----
    import nichecompass_STAIA

    # Load your samples (raw counts in .X, coordinates in obsm["spatial"])
    sample1 = sc.read_h5ad("sample1.h5ad")
    sample2 = sc.read_h5ad("sample2.h5ad")

    # First time — train on all samples together
    model_dir, adata = nichecompass_STAIA.niche_analysis_train(
        adatas=[sample1, sample2],
        sample_names=["sample1", "sample2"],
        gp_dict_path="./combined_gp_dict.json",
        output_dir="./artifacts/",
    )

    # Every subsequent run — load from disk, no adata needed
    adata = nichecompass_STAIA.niche_analysis_load(
        model_folder_path=model_dir,
        output_dir="./artifacts/",
        cell_type_key="cell_type",
    )
"""

import json
import os
import warnings
from datetime import datetime

import urllib.request
import urllib.error
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import squidpy as sq
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from nichecompass.models import NicheCompass
from nichecompass.utils import (
    add_gps_from_gp_dict_to_adata,
    create_new_color_dict,
    extract_gp_dict_from_omnipath_lr_interactions,
    extract_gp_dict_from_nichenet_lrt_interactions,
    extract_gp_dict_from_mebocost_ms_interactions,
    filter_and_combine_gp_dict_gps_v2,
)

warnings.filterwarnings("ignore")


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _setup_output_dirs(output_dir, dataset_name):
    """Create timestamped output folders. Returns (model_dir, figure_dir)."""
    current_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    model_folder_path  = os.path.join(output_dir, dataset_name, current_timestamp, "model")
    figure_folder_path = os.path.join(output_dir, dataset_name, current_timestamp, "figures")
    for d in (model_folder_path, figure_folder_path):
        os.makedirs(d, exist_ok=True)
    return model_folder_path, figure_folder_path


def _make_figure_dir(output_dir, dataset_name):
    """Create a timestamped figures folder for load runs."""
    current_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    figure_folder_path = os.path.join(output_dir, dataset_name, current_timestamp, "figures")
    os.makedirs(figure_folder_path, exist_ok=True)
    return figure_folder_path


def _build_per_sample_spatial_graphs(adata, sample_key,
                                      n_neighbors, radius_around,
                                      min_cells_around, adj_key, spatial_key):
    """
    Build a spatial neighbor graph for each sample separately, removing
    cells that have fewer than min_cells_around neighbors within radius_around.

    This mirrors the per-tissue graph construction in the reference script:
    each sample gets its own graph, cells without enough neighbors are dropped,
    and graphs are later stitched as disconnected components.

    Parameters
    ----------
    adata : AnnData
        Combined AnnData with all samples. Must have sample labels in
        obs[sample_key] and spatial coordinates in obsm[spatial_key].
    sample_key : str
        Column in adata.obs that identifies each sample.
    n_neighbors : int
        Number of spatial neighbors per cell.
    radius_around : int or None
        If set, cells with fewer than min_cells_around neighbors within this
        radius (in coordinate units) are removed. Set None to skip filtering.
    min_cells_around : int
        Minimum number of neighbors required within radius_around.
    adj_key : str
        Key to store the adjacency matrix in obsp.
    spatial_key : str
        Key in obsm containing (x, y) coordinates.

    Returns
    -------
    list of AnnData
        Filtered AnnDatas, each with a symmetric spatial adjacency matrix.
    """
    samples = adata.obs[sample_key].unique().tolist()
    print(f"Found {len(samples)} samples: {samples}")

    processed = []
    for sample in tqdm(samples, desc="Building per-sample spatial graphs"):
        adata_s = adata[adata.obs[sample_key] == sample].copy()

        # remove cells without enough neighbors in radius
        if radius_around is not None:
            coords = adata_s.obsm[spatial_key]
            distances = cdist(coords, coords, metric="euclidean")
            distances_below_radius = (distances < radius_around) & (distances > 0)
            n_cells_within_radius  = np.sum(distances_below_radius, axis=1)
            keep = n_cells_within_radius > min_cells_around
            n_removed = adata_s.n_obs - keep.sum()
            if n_removed > 0:
                print(f"  [{name}] Removed {n_removed} cells with <{min_cells_around} "
                      f"neighbors within radius {radius_around}.")
            adata_s = adata_s[keep].copy()
        if adata_s.n_obs < n_neighbors + 1:
            print(f"[{sample}] WARNING: only {adata_s.n_obs} cells - skipping this sample.")
            continue

        # Build spatial neighbor graph
        sq.gr.spatial_neighbors(
            adata_s,
            coord_type="generic",
            spatial_key=spatial_key,
            n_neighs=n_neighbors,
        )
        # Symmetrize adjacency matrix
        adata_s.obsp[adj_key] = (
            adata_s.obsp[adj_key].maximum(adata_s.obsp[adj_key].T)
        )
        processed.append(adata_s)
    return processed


def _stitch_spatial_graphs(adatas_by_sample, adj_key):
    """
    Concatenate per-sample AnnDatas and stitch their spatial graphs as
    disconnected components in one large sparse adjacency matrix.

    Cells from different samples are never spatially connected — the model
    sees them as separate tissues sharing the same gene panel and GP space.

    Parameters
    ----------
    adatas_by_sample : list of AnnData
        Each has a per-sample adjacency matrix in obsp[adj_key].
    adj_key : str
        Key for the spatial adjacency matrix.

    Returns
    -------
    AnnData
        Concatenated AnnData with the stitched block-diagonal adjacency matrix.
    """
    print("Stitching per-sample spatial graphs into one disconnected graph...")
    combined = ad.concat(adatas_by_sample, join="inner")
    n_total  = combined.shape[0]
    tissue_connectivities = []
    len_before = 0

    for i, adata_s in enumerate(tqdm(adatas_by_sample, desc="Stitching graphs")):
        n_s = adata_s.shape[0]
        if i == 0:
            # First sample: pad zeros to the right
            right_pad = sp.csr_matrix((n_s, n_total - n_s))
            tissue_connectivities.append(
                sp.hstack((adata_s.obsp[adj_key], right_pad))
            )
        elif i == len(adatas_by_sample) - 1:
            # Last sample: pad zeros to the left
            left_pad = sp.csr_matrix((n_s, n_total - n_s))
            tissue_connectivities.append(
                sp.hstack((left_pad, adata_s.obsp[adj_key]))
            )
        else:
            # Middle samples: pad zeros on both sides
            left_pad  = sp.csr_matrix((n_s, len_before))
            right_pad = sp.csr_matrix((n_s, n_total - n_s - len_before))
            tissue_connectivities.append(
                sp.hstack((left_pad, adata_s.obsp[adj_key], right_pad))
            )

        len_before += n_s

    combined.obsp[adj_key] = sp.vstack(tissue_connectivities)
    print(f"  Combined: {combined.n_obs} cells | "
          f"{combined.obsp[adj_key].nnz} edges across {len(adatas_by_sample)} samples")
    return combined


def _load_and_add_gps(adata, gp_dict_path):
    """Load GP dict from JSON and add masks to adata.varm."""
    print(f"Loading GP dictionary from: {gp_dict_path}")
    with open(gp_dict_path) as f:
        gp_dict = json.load(f)
    print(f"  Loaded {len(gp_dict)} gene programs")

    add_gps_from_gp_dict_to_adata(
        gp_dict=gp_dict,
        adata=adata,
        gp_targets_mask_key="nichecompass_gp_targets",
        gp_targets_categories_mask_key="nichecompass_gp_targets_categories",
        gp_sources_mask_key="nichecompass_gp_sources",
        gp_sources_categories_mask_key="nichecompass_gp_sources_categories",
        gp_names_key="nichecompass_gp_names",
        min_genes_per_gp=2,
        min_source_genes_per_gp=1,
        min_target_genes_per_gp=1,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None,
    )
    return adata


def _cluster_and_visualize(model, latent_key, leiden_resolutions,
                            cell_type_key, sample_key,
                            differential_gp_test_results_key,
                            spot_size, log_bayes_factor_thresh,
                            figure_folder_path):
    """
    Cluster niches at multiple resolutions, visualize, run differential GP tests. Shared between train and load to avoid duplication.
    """
    adata       = model.adata
    gp_names_key = "nichecompass_gp_names"

    # Latent space: neighbors + UMAP 
    print("Computing neighbors and UMAP in latent space...")
    sc.pp.neighbors(adata, use_rep=latent_key, key_added=latent_key)
    sc.tl.umap(adata, neighbors_key=latent_key)

    # Leiden clustering at all requested resolutions 
    print(f"Running Leiden clustering at {len(leiden_resolutions)} resolutions...")
    for res in leiden_resolutions:
        key = f"latent_res_{res}"
        sc.tl.leiden(
            adata=adata,
            resolution=res,
            key_added=key,
            neighbors_key=latent_key,
            flavor="igraph",
        )
        n_niches = adata.obs[key].nunique()
        print(f"  Resolution {res}: {n_niches} niches")

    # Use the middle resolution as the default for plots
    default_res = leiden_resolutions[len(leiden_resolutions) // 2]
    default_niche_key = f"latent_res_{default_res}"
    print(f"\nUsing resolution {default_res} for plots (change via leiden_resolutions).")

    niche_colors = create_new_color_dict(adata=adata, cat_key=default_niche_key)

    # UMAP + spatial scatter 
    color_keys = [default_niche_key]
    if sample_key and sample_key in adata.obs.columns:
        color_keys.append(sample_key)
    if cell_type_key and cell_type_key in adata.obs.columns:
        color_keys.append(cell_type_key)

    for color_key in color_keys:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        palette = niche_colors if color_key == default_niche_key else None
        sc.pl.umap(
            adata,
            color=color_key,
            palette=palette,
            title=f"{color_key} — Latent Space",
            ax=axes[0],
            show=False,
        )
        sc.pl.spatial(
            adata,
            color=color_key,
            palette=palette,
            spot_size=spot_size,
            title=f"{color_key} — Physical Space",
            ax=axes[1],
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(figure_folder_path, f"{color_key}_umap_spatial.svg"),
            bbox_inches="tight",
        )
        plt.show()

    # Sample composition per niche
    if sample_key and sample_key in adata.obs.columns:
        df_sample = (
            adata.obs.groupby([default_niche_key, sample_key])
            .size()
            .unstack(fill_value=0)
        )
        df_sample.plot(kind="bar", stacked=True, figsize=(10, 6))
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Sample")
        plt.title(f"Sample Composition of Niches (res={default_res})")
        plt.xlabel("Niche")
        plt.ylabel("Cell Counts")
        plt.savefig(
            os.path.join(figure_folder_path, "niche_sample_composition.svg"),
            bbox_inches="tight",
        )
        plt.show()

    # Cell type composition per niche (optional)
    if cell_type_key and cell_type_key in adata.obs.columns:
        df_counts = (
            adata.obs.groupby([default_niche_key, cell_type_key])
            .size()
            .unstack(fill_value=0)
        )
        df_counts.plot(kind="bar", stacked=True, figsize=(10, 8))
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Cell Type")
        plt.title(f"Cell Type Composition of Niches (res={default_res})")
        plt.xlabel("Niche")
        plt.ylabel("Cell Counts")
        plt.savefig(
            os.path.join(figure_folder_path, "niche_composition.svg"),
            bbox_inches="tight",
        )
        plt.show()
    else:
        print("cell_type_key not set or not found in adata.obs — skipping composition plot.")

    # Active GP summary 
    active_gps = model.get_active_gps()
    print(f"Total GPs: {len(adata.uns[gp_names_key])} | Active GPs: {len(active_gps)}")

    gp_summary_df = model.get_gp_summary()
    gp_summary_df[gp_summary_df["gp_active"] == True].to_csv(
        os.path.join(figure_folder_path, "active_gp_summary.csv"), index=False
    )

    # Differential GP testing 
    print(f"Running differential GP tests (log_bayes_factor_thresh={log_bayes_factor_thresh})...")
    enriched_gps = model.run_differential_gp_tests(
        cat_key=default_niche_key,
        selected_cats=None,
        comparison_cats="rest",
        log_bayes_factor_thresh=log_bayes_factor_thresh,
    )
    print(f"Enriched GPs found: {len(enriched_gps)}")

    if differential_gp_test_results_key in adata.uns:
        adata.uns[differential_gp_test_results_key].to_csv(
            os.path.join(figure_folder_path, "differential_gps.csv"), index=False
        )

    # Enriched GP heatmap
    if enriched_gps:
        df = (
            adata.obs[[default_niche_key] + enriched_gps]
            .groupby(default_niche_key)
            .mean()
        )
        scaler = MinMaxScaler()
        normalized_df = pd.DataFrame(
            scaler.fit_transform(df), columns=df.columns, index=df.index
        )
        plt.figure(figsize=(16, 8))
        sns.heatmap(normalized_df, cmap="viridis", annot=False, linewidths=0)
        plt.xticks(rotation=45, fontsize=8, ha="right")
        plt.xlabel("Gene Programs", fontsize=14)
        plt.title(f"Enriched Gene Program Activity per Niche (res={default_res})")
        plt.savefig(
            os.path.join(figure_folder_path, "enriched_gps_heatmap.svg"),
            bbox_inches="tight",
        )
        plt.show()
    else:
        print("No enriched GPs above threshold — heatmap skipped.")

    return adata


# =============================================================================
# PUBLIC FUNCTION 1: TRAIN
# =============================================================================

def niche_analysis_train(
    adata,
    gp_dict_path,
    sample_key = "sample",
    output_dir="./artifacts",
    dataset_name="xenium_muscle",
    spatial_key="spatial",
    # spatial graph
    n_neighbors=4,
    radius_around=None,
    min_cells_around=0,
    # model architecture
    conv_layer_encoder="gatv2conv",
    active_gp_thresh_ratio=0.01,
    cat_covariates_embeds_nums=3,
    # training
    n_epochs=400,
    n_epochs_all_gps=25,
    lr=0.001,
    lambda_edge_recon=500000.0,
    lambda_gene_expr_recon=300.0,
    lambda_l1_masked=0.0,
    lambda_l1_addon=30.0,
    edge_batch_size=4096,
    n_sampled_neighbors=4,
    use_cuda_if_available=False,
    cell_type_key=None,
    leiden_resolutions=None,
    spot_size=100,
    log_bayes_factor_thresh=2.3,
):
    """
    Full NicheCompass pipeline for multiple Xenium samples.

    Builds per-sample spatial graphs (as disconnected components), loads a
    pre-saved GP dictionary, trains the model with sample-level batch
    correction, clusters niches at multiple resolutions, and saves everything.

    Parameters
    ----------
    adata : AnnData
        Combined AnnData with all samples. Raw counts in .X (not
        log-normalized). Spatial coordinates in obsm[spatial_key] as (x, y).
        Must have a sample identifier column in obs[sample_key].
    gp_dict_path : str
        Path to the pre-saved combined GP dictionary JSON file.
        Build this once with the GP building utilities and save with json.dump.
        See module docstring for how to create this file.
   sample_key : str
        Column in adata.obs that identifies each sample. The function splits
        on this column and uses it for batch correction. Default: "sample".
    output_dir : str
        Root folder for all outputs (model, figures, CSVs).
    dataset_name : str
        Used to name subfolders inside output_dir.
    spatial_key : str
        Key in obsm containing (x, y) spatial coordinates.
    n_neighbors : int
        Number of spatial neighbors per cell. Authors recommend 4.
    radius_around : int or None
        If set, cells with fewer than min_cells_around neighbors within this
        radius are removed before graph construction. Set None to skip.
    min_cells_around : int
        Minimum neighbors within radius_around to keep a cell.
    conv_layer_encoder : str
        'gatv2conv' (recommended, better quality) or 'gcnconv' (faster).
    cat_covariates_embeds_nums : int
        Embedding dimension for the sample/batch covariate. Default 3.
    n_epochs : int
        Training epochs. 400 is the paper default.
    edge_batch_size : int
        Edges per training batch. Reduce if running out of GPU memory.
    cell_type_key : str, optional
        adata.obs column with cell type labels for composition plots.
    leiden_resolutions : list of float, optional
        Leiden resolutions to run. Defaults to [0.1, 0.2, ..., 1.0].
    log_bayes_factor_thresh : float
        Threshold for differential GP testing. 2.3 = 'strong evidence'.

    Returns
    -------
    model_folder_path : str
        Path to the saved model. Pass this to niche_analysis_load().
    adata : AnnData
        Combined AnnData with latent embedding and niche cluster labels
        for all samples.
    """

    if leiden_resolutions is None:
        leiden_resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Fixed AnnData keys
    adj_key = "spatial_connectivities"
    latent_key = "nichecompass_latent"
    differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

    # Setup
    model_folder_path, figure_folder_path = _setup_output_dirs(output_dir, dataset_name)
    print(f"\n{'='*60}")
    print("NicheCompass — TRAINING")
    print(f"Model will be saved to: {model_folder_path}")
    print(f"{'='*60}\n")

    # Per-sample spatial graphs
    adatas_by_sample = _build_per_sample_spatial_graphs(
        adata=adata,
        sample_key=sample_key,
        n_neighbors=n_neighbors,
        radius_around=radius_around,
        min_cells_around=min_cells_around,
        adj_key=adj_key,
        spatial_key=spatial_key,
    )

    if len(adatas_by_sample) == 0:
        raise ValueError(
            "No samples remaining after filtering. Check your sample_key "
            "and radius_around / min_cells_around settings."
        )

    # Stitch into one disconnected graph
    adata_combined = _stitch_spatial_graphs(adatas_by_sample, adj_key)

    # Load GP dict from JSON and add masks
    adata_combined = _load_and_add_gps(adata_combined, gp_dict_path)

    # Initialize model with batch correction
    print("Initializing NicheCompass model...")
    model = NicheCompass(
        adata_combined,
        counts_key=None,
        adj_key=adj_key,
        cat_covariates_keys=[sample_key],
        cat_covariates_embeds_injection=["gene_expr_decoder"],
        cat_covariates_embeds_nums=[cat_covariates_embeds_nums],
        cat_covariates_no_edges=[True],
        gp_names_key="nichecompass_gp_names",
        active_gp_names_key="nichecompass_active_gp_names",
        gp_targets_mask_key="nichecompass_gp_targets",
        gp_targets_categories_mask_key="nichecompass_gp_targets_categories",
        gp_sources_mask_key="nichecompass_gp_sources",
        gp_sources_categories_mask_key="nichecompass_gp_sources_categories",
        latent_key=latent_key,
        conv_layer_encoder=conv_layer_encoder,
        active_gp_thresh_ratio=active_gp_thresh_ratio,
    )

    # Train
    model.train(
        n_epochs=n_epochs,
        n_epochs_all_gps=n_epochs_all_gps,
        lr=lr,
        lambda_edge_recon=lambda_edge_recon,
        lambda_gene_expr_recon=lambda_gene_expr_recon,
        lambda_l1_masked=lambda_l1_masked,
        lambda_l1_addon=lambda_l1_addon,
        edge_batch_size=edge_batch_size,
        n_sampled_neighbors=n_sampled_neighbors,
        use_cuda_if_available=use_cuda_if_available,
        verbose=True,
    )

    # Save model + adata (adata has all GP masks and latent embedding baked in)
    model.save(
        dir_path=model_folder_path,
        overwrite=True,
        save_adata=True,
        adata_file_name="nichecompass_adata.h5ad",
    )
    print(f"Model saved to: {model_folder_path}")

    # Cluster + visualize + differential GPs
    adata = _cluster_and_visualize(
        model=model,
        latent_key=latent_key,
        leiden_resolutions=leiden_resolutions,
        cell_type_key=cell_type_key,
        sample_key=sample_key,
        differential_gp_test_results_key=differential_gp_test_results_key,
        spot_size=spot_size,
        log_bayes_factor_thresh=log_bayes_factor_thresh,
        figure_folder_path=figure_folder_path,
    )

    print(f"\nDone. Figures saved to: {figure_folder_path}")
    return model_folder_path, adata


# =============================================================================
# PUBLIC FUNCTION 2: LOAD FROM DISK (no retraining)
# =============================================================================

def niche_analysis_load(
    model_folder_path,
    output_dir="./artifacts",
    dataset_name="xenium_tissue",
    sample_key="sample",
    # analysis — can differ freely from the training run
    cell_type_key=None,
    leiden_resolutions=None,
    spot_size=100,
    log_bayes_factor_thresh=2.3,
):
    """
    Load a previously trained NicheCompass model and its saved adata from
    disk, then re-run clustering and visualization. No retraining needed.

    The adata saved during training already contains all GP masks, the spatial
    graph, and the latent embedding, so the model architecture always matches
    the saved weights exactly. No new input data is required.

    You can freely change leiden_resolutions, cell_type_key, and
    log_bayes_factor_thresh between load runs without retraining.

    Parameters
    ----------
    model_folder_path : str
        Path to the saved model directory returned by niche_analysis_train().
        Must contain both the model weights and nichecompass_adata.h5ad.
    output_dir : str
        Root folder for figures and CSVs from this run.
    dataset_name : str
        Used to name figure subfolders inside output_dir.
   sample_key : str
        Column in adata.obs that identifies each sample. Must match what
        was used during training. Default: "sample".
    cell_type_key : str, optional
        adata.obs column with cell type labels. Can be added after training —
        just add the column to the saved adata before calling this function.
    leiden_resolutions : list of float, optional
        Leiden resolutions to run. Can differ from the training run.
        Defaults to [0.1, 0.2, ..., 1.0].
    spot_size : int
        Spot size for sc.pl.spatial().
    log_bayes_factor_thresh : float
        Threshold for differential GP testing.

    Returns
    -------
    adata : AnnData
        AnnData with latent embedding and niche cluster labels.

    Notes
    -----
    To add cell type labels to the saved adata after training:
        adata = sc.read_h5ad("<model_folder>/nichecompass_adata.h5ad")
        adata.obs["cell_type"] = <your labels>
        adata.write_h5ad("<model_folder>/nichecompass_adata.h5ad")
        # Then call niche_analysis_load(..., cell_type_key="cell_type")
    """

    if leiden_resolutions is None:
        leiden_resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Fixed keys — must match training
    latent_key                       = "nichecompass_latent"
    differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

    # Setup
    figure_folder_path = _make_figure_dir(output_dir, dataset_name)

    print(f"\n{'='*60}")
    print("NicheCompass — LOAD FROM DISK (no retraining)")
    print(f"Loading model from: {model_folder_path}")
    print(f"{'='*60}\n")

    # Load saved adata (GP masks already baked in)
    adata_path = os.path.join(model_folder_path, "nichecompass_adata.h5ad")
    print(f"Loading saved adata from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  Cells: {adata.n_obs} | Genes: {adata.n_vars}")
    if sample_key in adata.obs.columns:
        print(f"  Samples: {sorted(adata.obs[sample_key].unique().tolist())}")

    # Load model weights
    print("Loading trained model weights...")
    model = NicheCompass.load(
        dir_path=model_folder_path,
        adata=adata,
    )

    # Recompute latent embedding from loaded weights
    print("Computing latent embedding...")
    latent = model.get_latent_representation()
    model.adata.obsm[latent_key] = latent

    # Cluster + visualize + differential GPs
    adata = _cluster_and_visualize(
        model=model,
        latent_key=latent_key,
        leiden_resolutions=leiden_resolutions,
        cell_type_key=cell_type_key,
        sample_key=sample_key,
        differential_gp_test_results_key=differential_gp_test_results_key,
        spot_size=spot_size,
        log_bayes_factor_thresh=log_bayes_factor_thresh,
        figure_folder_path=figure_folder_path,
    )

    print(f"\nDone. Figures saved to: {figure_folder_path}")
    return adata


# =============================================================================
# PUBLIC FUNCTION 3: GENERATE COMBINED GP DICT
# =============================================================================

def save_gp_dict(
    species="human",
    save_path="gp_dict",
    include_mebocost=True,
    gene_orthologs_mapping_file_path=None,
    overwrite=False,
):
    """
    Build the combined gene-program (GP) dictionary NicheCompass needs for training, from OmniPath (ligand-receptor), NicheNet (ligand-receptor target), and, optionally, MEBOCOST (metabolite-enzyme-sensor) interactions, and cache it to disk.

    This is meant to be run once per species (it downloads and parses several fairly large reference databases), after which the resulting ``combined_gp_dict.json`` can just be loaded from disk every subsequent training run.

    Parameters
    ----------
    species : str, default "human"
        Species to build gene programs for. Passed through to all three
        extraction functions. NicheCompass / MEBOCOST support "human" and
        "mouse".
    save_path : str, default "gp_dict"
        Directory in which all intermediate reference files (OmniPath /
        NicheNet networks, MEBOCOST tables) and the final
        ``combined_gp_dict.json`` are stored. Created if it doesn't exist.
    include_mebocost : bool, default True
        Whether to additionally retrieve MEBOCOST metabolite-sensor gene
        programs. MEBOCOST doesn't ship its interaction tables with the
        `nichecompass` package, so this function downloads the four TSVs
        NicheCompass's own tutorials rely on directly from the
        NicheCompass GitHub repo (this is the manual `curl` step you'd
        otherwise have to run yourself, see ``_download_mebocost_files``
        below). If the download fails (e.g. no internet access in this
        environment) a warning is printed and MEBOCOST is skipped rather
        than raising, so OmniPath + NicheNet GPs are still produced.
    gene_orthologs_mapping_file_path : str or None, default None
        Forwarded to ``extract_gp_dict_from_omnipath_lr_interactions``.
        Only needed if you want to map OmniPath interactions (human by
        default) onto a different species without native OmniPath
        coverage; leave as None for human/mouse.
    overwrite : bool, default False
        If a ``combined_gp_dict.json`` already exists at ``save_path``,
        skip all extraction and just load and return it, unless
        ``overwrite=True``.
 
    Returns
    -------
    dict
        The combined gene-program dictionary (also written to
        ``{save_path}/combined_gp_dict.json``).
 
    Notes
    -----
    Requires the `nichecompass` package to be installed and importable.
    """
    os.makedirs(save_path, exist_ok=True)
    combined_gp_dict_path = os.path.join(save_path, "combined_gp_dict.json")

    if os.path.isfile(combined_gp_dict_path) and not overwrite:
        print(
            f"Found existing combined GP dict at `{combined_gp_dict_path}`, "
            f"loading it instead of recomputing (pass overwrite=True to "
            f"force recomputation)"
        )
        with open(combined_gp_dict_path, "r") as f:
            return json.load(f)

    print("Etracting OmniPath ligand-receptor programs...")
    omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=f"{save_path}/omnipath_lr_network.csv",
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=True,
    )

    print("Extracting NicheNet ligand-receptor-target gene programs...")
    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=species,
        version="v2",
        keep_target_genes_ratio=1.,
        max_n_target_genes_per_gp=250,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=f"{save_path}/nichenet_lr_network_v2_{species}.csv",
        ligand_target_matrix_file_path=f"{save_path}/nichenet_ligand_target_matrix_v2_{species}.csv",
        plot_gp_gene_count_distributions=True,
    )

    gp_dicts = [omnipath_gp_dict, nichenet_gp_dict]

    if include_mebocost:
        try:
            mebocost_dir = _download_mebocost_files(save_path, species)
            print("Extracting MEBOCOST metabolite-sensor gene programs...")
            mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
                dir_path=f"{save_path}/data/gene_programs/metabolite_enzyme_sensor_gps",
                species=species,
                plot_gp_gene_count_distributions=True,
            )
            gp_dicts.append(mebocost_gp_dict)
        except (urllib.error.URLError, OSError) as e:
            print(f"Warning: could not retrieve MEBOCOST gene programs ({e!r}). Contuining with OmniPath + NicheNet GPs only.")

    print("Filtering and combining gene programs...")
    combined_gp_dict = filter_and_combine_gp_dict_gps_v2(gp_dicts, verbose=True)

    print("Combining GP programs is finished.")

    import json
    print(json.__file__)
    print(json.dump)

    with open(combined_gp_dict_path, "w") as f:
        # json.dump(combined_gp_dict, f)
        f.write(json.dumps(combined_gp_dict))

    print(
        f"Saved combined gene program dictionary with {len(combined_gp_dict)} gene programs to '{combined_gp_dict_path}'."
    )
 

def _download_mebocost_files(save_path, species):
    """
    Download the MEBOCOST metabolite-enzyme and metabolite-sensor
    interaction tables that ``extract_gp_dict_from_mebocost_ms_interactions``
    expects to find on disk.
 
    NicheCompass doesn't bundle these with the package; its own tutorials
    fetch them from the NicheCompass GitHub repo. This replaces the
    equivalent shell commands:
 
        mkdir -p ./data/gene_programs/metabolite_enzyme_sensor_gps
        cd ./data/gene_programs/metabolite_enzyme_sensor_gps
        for f in human_metabolite_enzymes.tsv human_metabolite_sensors.tsv \\
                 mouse_metabolite_enzymes.tsv mouse_metabolite_sensors.tsv; do
          curl -sSL -O "https://raw.githubusercontent.com/Lotfollahi-lab/\\
nichecompass/main/data/gene_programs/metabolite_enzyme_sensor_gps/$f"
        done
 
    Files are downloaded once and reused on subsequent calls.
 
    Parameters
    ----------
    save_path : str
        Base directory passed to ``save_gp_dict``.
    species : str
        Only used to decide which pair of files is strictly required;
        both human and mouse files are downloaded regardless, since
        MEBOCOST's own loader expects both to be present.
 
    Returns
    -------
    str
        Path to the directory containing the downloaded TSVs, suitable to
        pass as ``dir_path`` to ``extract_gp_dict_from_mebocost_ms_interactions``.
    """
    mebocost_dir = os.path.join(save_path, "metabolite_enzyme_sensor_gps")
    os.makedirs(mebocost_dir, exist_ok=True)

    base_url = (
        "https://raw.githubusercontent.com/Lotfollahi-lab/nichecompass/"
        "main/data/gene_programs/metabolite_enzyme_sensor_gps/"
    )
    filenames = [
        "human_metabolite_enzymes.tsv",
        "human_metabolite_sensors.tsv",
        "mouse_metabolite_enzymes.tsv",
        "mouse_metabolite_sensors.tsv",
    ]
 
    for filename in filenames:
        file_path = os.path.join(mebocost_dir, filename)
        if os.path.isfile(file_path):
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(base_url + filename, file_path)
 
    return mebocost_dir