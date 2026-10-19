import warnings

# Silence the Dask legacy-dataframe FutureWarning at its source
import dask
dask.config.set({"dataframe.query-planning": True})

# Suppress noisy import-time warnings from transitive deps we don't control
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*rpy2.*")
    from ._banksy import (
        run_banksy
    )

    from ._cellcharter import (
        run_cellcharter,
        stability_cellcharter,
        prepare_cellcharter
    )

    from ._nichecompass import (
        niche_analysis_train,
        niche_analysis_load,
        save_gp_dict
    )