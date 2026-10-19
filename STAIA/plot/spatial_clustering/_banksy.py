import os
import warnings
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from spatialdata import SpatialData
from anndata import AnnData

from typing import List, Tuple
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"banksy(\..*)?",
)

from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy.main import concatenate_all
from banksy_utils.umap_pca import pca_umap
from banksy.cluster_methods import run_Leiden_partition
from banksy.plot_banksy import plot_results

def _preprocess(
        adata: AnnData,
):
    # for now, we assume that the data has already been normalized
    # check if x_centroid and y_centroid are present in the data
    if "x_centroid" not in adata.obs:
        adata.obs['x_centroid'] = adata.obsm['spatial'][:, 0]
    if "y_centroid" not in adata.obs:
        adata.obs['y_centroid'] = adata.obsm['spatial'][:, 1]

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
    # only keep highly variable genes in the data
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, zero_center=False)
    return adata


def run_banksy(
        adata: AnnData,
        coord_keys: Tuple[str] = ("x_centroid", "y_centroid", "spatial"),
        resolutions: List[float] = [0.5, 1.0],
        pca_dims: List[int] = [20],
        lambda_list: List[float] = [0.2, 0.8],
        max_m: int = 1,
        nonspatial: bool = True,
        output_path: str = "banksy_output"
):
    """
    Main pipeline.
    
    Assumptions:
        - data is already log normalized (either by size or counts)
        - data is already filtered

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix 
    coord_keys : Tuple[str], default = ("x_centroid", "y_centroid", "spatial")
        Coordinate keys present in adata.obsm
    resolutions : List[float], default = [0.5, 1.0]
        Leiden resolutions
    pca_dims : List[int], default = [20]
        Amount of dimensions used for the PCA.
    lambda_list : List[float], default = [0.2, 0.8]
        Lambda values used to generate BANKSY matrix
    max_m : int, default = 1
        Int value used to initialize BANKSY
    output_path : str, default = "banksy_output"
        File path to the BANKSY output.

    Returns
    -------
    results_df, banksy_matrix
        Results of BANKSY
    """
    print(f"Please take into account that running the BANKSY algorithm can take " \
          "a while before finishing. Have patience... ")
    adata = _preprocess(adata)

    # initialize banksy
    banksy_dict = initialize_banksy(
        adata, 
        coord_keys=coord_keys,
        num_neighbours=15,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=True,
        plt_nbr_weights=True,
        plt_agf_angles=False,
        plt_theta=True
    )

    banksy_dict, banksy_matrix = generate_banksy_matrix(adata, banksy_dict, lambda_list, max_m)

    if nonspatial:
        banksy_dict["nonspatial"] = {
            0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }
        }

    pca_umap(banksy_dict,
            pca_dims = pca_dims,
            add_umap = True,
            plt_remaining_var = False,
        )

    results_df, max_num_labels = run_Leiden_partition(
        banksy_dict,
        resolutions,
        num_nn = 50,
        num_iterations = -1,
        partition_seed = 1234,
        match_labels = True,
    )

    c_map =  'tab20' # specify color map
    weights_graph =  banksy_dict['scaled_gaussian']['weights'][0]

    plot_results(
        results_df,
        weights_graph,
        c_map,
        match_labels = True,
        coord_keys = coord_keys,
        max_num_labels  =  max_num_labels, 
        save_path = os.path.join(output_path, 'tmp_png'),
        save_fig = True, # save the spatial map of all clusters
        save_seperate_fig = True, # save the figure of all clusters plotted seperately
    )
    return results_df, banksy_matrix

