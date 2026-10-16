import warnings

# Silence the Dask legacy-dataframe FutureWarning at its source
import dask
dask.config.set({"dataframe.query-planning": True})

# Suppress noisy import-time warnings from transitive deps we don't control
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*rpy2.*")
    from ._split_ui import (
        SplitSession,
        LabelSession
    )

    from ._assignment_module import (
        plot_samples,
        detect_samples_watershed,
        # do_one_split,
        # _renumber,
        replay_all_splits,
        # make_vertical_record,
        # make_horizontal_record,
        # make_diagonal_record,
        run_watershed,
        manual_split_samples
    )