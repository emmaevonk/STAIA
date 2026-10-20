import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*legacy Dask DataFrame implementation is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*No rpy2 installed.*",
        category=UserWarning,
    )

    from .celltype_annotation import (
        annotate_celltypist,
        annotate_tangram
    )