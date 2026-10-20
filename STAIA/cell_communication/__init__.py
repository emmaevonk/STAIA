import warnings

# Silence the Dask legacy-dataframe FutureWarning at its source
import dask
dask.config.set({"dataframe.query-planning": True})

# Suppress noisy import-time warnings from transitive deps we don't control
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*rpy2.*")

    from .ligand_receptor import (
        run_liana,
        plot_liana_dotplot,
        plot_spatial_lr_scores,
        plot_liana_circle,
    )

    from .svg import (
        morans_score,
        svg_summary,
        plot_svgs,
    )