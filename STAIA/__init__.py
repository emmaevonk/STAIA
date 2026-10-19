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

    from .plot import (
        dotplot,
        celltype_composition,
        cluster,
        plot_image,
        rank_genes_group,
        multimodal_segmentation,
        multimodal_segmentation_slice,
        plot_labels,
        run_banksy,
        run_cellcharter,
        stability_cellcharter,
        plot_count_distr,
        prepare_cellcharter,
        niche_analysis_train,
        niche_analysis_load,
        save_gp_dict
    )
    from .table import (
        add_nuclei,
        get_adata_nuclei
    )
    from .assignment import (
        SplitSession,
        LabelSession,
        plot_samples,
        detect_samples_watershed,
        replay_all_splits,
        run_watershed,
        manual_split_samples
    )
    from ._spatialobject import (
        read_data,
        convert_sdata_adata,
        roi
    )
    from ._metrics import (
        frac_transcripts,
        staining_positive,
        metrics
    )
    from ._qc import (
        plot_outliers,
        perform_quality_control,
        control_probes_codew,
        plot_qc_metrics,
        recommend_threshold,
        is_outlier,
        detect_outlier
    )
    from ._cellcom import (
        morans_score
    )
    from ._neighbors import (
        nhood_enrichment
    )
    from ._pseudobulk import (
        pseudobulk,
        _make_pseudobulk,
        pseudobulk_per_condition
    )
    from ._resolution import (
        compute_leiden_resolution
    )
    from ._preprocess import (
        size_normalization,
        counts_normalized,
        scaling
    )