import warnings

# Silence the Dask legacy-dataframe FutureWarning at its source
import dask
dask.config.set({"dataframe.query-planning": True})

# Suppress noisy import-time warnings from transitive deps we don't control
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*rpy2.*")
    from ._analysis import (
        dotplot,
        celltype_composition
    )

    from ._cluster import (
        cluster
    )

    from ._plot import (
        plot_image,
        rank_genes_group,
        plot_labels,
        plot_count_distr
    )

    from ._segmentation import (
        multimodal_segmentation,
        multimodal_segmentation_slice
    )

    from .spatial_clustering import (
        run_banksy,
        run_cellcharter,
        stability_cellcharter,
        prepare_cellcharter,
        niche_analysis_train,
        niche_analysis_load,
        save_gp_dict
    )