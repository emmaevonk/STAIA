"""
Segmentation utilities for spatial transcriptomics data.

This module provides two functions for running cell segmentation:

- :func:`baysor` — transcript-based segmentation with Baysor.
- :func:`xeniumranger` — 10x Genomics re-segmentation with XeniumRanger.

Both functions default to submitting a SLURM job, which is strongly
recommended for large Xenium datasets. Set ``use_slurm=False`` to run
locally instead (useful for small datasets or workstations).

Typical usage
-------------
Submit a Baysor job to SLURM:

>>> staia.baysor(
...     transcripts_path="/data/transcripts.parquet",
...     output_path="/results/baysor/",
...     prior_segmentation=":cell_id",
...     slurm_args={"time": "06:00:00", "mem": "128G", "partition": "gpu"},
... )

Submit a XeniumRanger re-segmentation job:

>>> staia.xeniumranger(
...     xenium_bundle="/path/to/xenium/bundle",
...     run_id="sample_01",
...     output_path="/results/xeniumranger/",
...     boundary_stain="ATP1A1/CD45/E-Cadherin",
...     slurm_args={"time": "08:00:00", "mem": "128G", "cpus": 16},
... )

Run Baysor locally (no SLURM):

>>> staia.baysor(
...     transcripts_path="/data/transcripts.parquet",
...     output_path="/results/baysor/",
...     use_slurm=False,
... )
"""

# NOTE: not added to __init__.py yet — implementation/testing phase.

import json
import os
import shutil
import subprocess
import textwrap
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# XeniumRanger / XeniumAnalyzer compatibility helpers
# ---------------------------------------------------------------------------

def _get_xenium_analyzer_version(xenium_bundle: Path) -> tuple[int, int, int] | None:
    """
    Read the XeniumAnalyzer version from ``experiment.xenium`` inside the bundle.

    The analyzer version is stored in the ``"analysis_sw_version"`` field as a
    string, e.g. ``"xenium-4.0.2.2"`` or ``"4.0.2.2"``. Only the first three
    components (major, minor, patch) are returned; any fourth component is
    ignored.

    Returns a ``(major, minor, patch)`` tuple, or ``None`` if the file is
    missing or the version string cannot be parsed.
    """
    import re

    experiment_file = xenium_bundle / "experiment.xenium"
    if not experiment_file.exists():
        return None

    with open(experiment_file) as f:
        data = json.load(f)

    sw_version = data.get("analysis_sw_version")
    if not sw_version:
        return None

    # Strip any leading non-numeric prefix (e.g. "xenium-")
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(sw_version))
    if not match:
        return None

    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _get_xeniumranger_version(xeniumranger_executable: str) -> tuple[int, int, int] | None:
    """
    Run ``xeniumranger --version`` and parse the output.

    Returns a ``(major, minor, patch)`` tuple, or ``None`` if the version
    cannot be determined.
    """
    import re

    try:
        result = subprocess.run(
            [xeniumranger_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Output is typically "xeniumranger xenium4.0.1" or "xeniumranger 4.0.1"
        match = re.search(r"(\d+)\.(\d+)\.?(\d*)", result.stdout + result.stderr)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            patch = int(match.group(3)) if match.group(3) else 0
            return (major, minor, patch)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return None


def _check_xenium_version_compatibility(
    xenium_bundle: Path,
    xeniumranger_executable: str,
) -> bool:
    """
    Compare XeniumAnalyzer (bundle) and XeniumRanger versions.

    Prints a summary of the versions found. Returns ``True`` when the patch
    is required (XeniumAnalyzer minor version > XeniumRanger minor version
    within the same major version), ``False`` otherwise.
    """
    analyzer_ver = _get_xenium_analyzer_version(xenium_bundle)
    ranger_ver   = _get_xeniumranger_version(xeniumranger_executable)

    analyzer_str = ".".join(map(str, analyzer_ver)) if analyzer_ver else "unknown"
    ranger_str   = ".".join(map(str, ranger_ver))   if ranger_ver   else "unknown"

    # NOTE: Changed the code here for the version check if xeniumranger and xeniumanalyzer

    print(f"XeniumAnalyzer version (bundle) : {analyzer_str}")
    print(f"XeniumRanger version            : {ranger_str}")

    if analyzer_ver is None or ranger_ver is None:
        warnings.warn(
            "Could not determine one or both versions. "
            "Skipping compatibility check — proceeding without patch.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False

    # analyzer_major, analyzer_minor, analyzer_patch = analyzer_ver
    ranger_major,   ranger_minor,   ranger_patch   = ranger_ver

    # if analyzer_major != ranger_major:
    #     warnings.warn(
    #         f"Major version mismatch: XeniumAnalyzer {analyzer_str} vs "
    #         f"XeniumRanger {ranger_str}. Proceeding, but results may be "
    #         "unreliable.",
    #         RuntimeWarning,
    #         stacklevel=3,
    #     )
    #     return False

    print(f"Check if it works, analyzer version: {analyzer_ver}")
    print(f"Check if it works, ranger version: {ranger_ver}")

    if analyzer_ver == (4, 0, 2) and (ranger_major,   ranger_minor) == (4, 0):
        needs_patch = True
    else: 
        needs_patch = False

    # needs_patch = (analyzer_minor, analyzer_patch) > (ranger_minor, ranger_patch) #TODO: this is wrong! Need another method to check for version incompatibility
    if needs_patch:
        print(
            f"Version mismatch detected (XeniumAnalyzer {analyzer_str} > "
            f"XeniumRanger {ranger_str}). "
            "The experiment.xenium compatibility patch will be applied."
        )
    else:
        print("Versions are compatible — no patch needed.")

    return needs_patch


def _patch_experiment_xenium(xenium_bundle: Path, output_path: Path) -> Path:
    """
    Copy the Xenium bundle to ``<output_path>/xenium_bundle_patched/`` and
    patch ``experiment.xenium`` in the copy so that XeniumRanger 4.0.x can
    process a bundle produced by XeniumAnalyzer 4.0.2+.

    The original bundle is never modified.

    Two changes are made to the copy:
    - The ``"segmented_cell_boundary_large_frac"`` key is removed (new field
      unknown to older XeniumRanger builds).
    - All ``minor_version`` fields are reset to ``0`` so that XeniumRanger
      does not reject the bundle on a minor-version check.

    If the patched copy already exists, the copy step is skipped and the
    existing copy is returned directly.

    Parameters
    ----------
    xenium_bundle : Path
        Path to the original Xenium bundle directory.
    output_path : Path
        Directory under which the patched copy is written.

    Returns
    -------
    Path
        Path to the patched bundle copy (safe to pass to XeniumRanger).
    """
    patched_bundle = output_path / "xenium_bundle_patched"

    if patched_bundle.exists():
        print(f"Patched bundle already exists at {patched_bundle} — reusing it.")
        return patched_bundle

    print(f"Copying bundle to {patched_bundle} (original will not be modified) …")
    shutil.copytree(xenium_bundle, patched_bundle)
    print("Copy complete.")

    src = patched_bundle / "experiment.xenium"

    with open(src) as f:
        data = json.load(f)

    def find_and_remove(obj, target_key, path=""):
        if isinstance(obj, dict):
            if target_key in obj:
                print(f"Found and removing '{target_key}' at {path or '/'}")
                del obj[target_key]
            for k, v in obj.items():
                find_and_remove(v, target_key, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_and_remove(item, target_key, f"{path}[{i}]")

    def fix_minor_version(obj, path=""):
        if isinstance(obj, dict):
            if "major_version" in obj and "minor_version" in obj and "patch_version" in obj:
                old = obj["minor_version"]
                obj["minor_version"] = 0
                print(f"Set minor version at {path or '/'} from {old} to 0.")
            for k, v in obj.items():
                fix_minor_version(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                fix_minor_version(item, f"{path}[{i}]")

    find_and_remove(data, "segmented_cell_boundary_large_frac")
    fix_minor_version(data)

    with open(src, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Patch applied successfully → {src}")
    return patched_bundle



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def baysor(
    transcripts_path: str | Path,
    output_path: str | Path,
    prior_type: str | None = ":Xenium_cell_idx",
    config_path: str | Path | None = None,
    x_column: str = "x_location",
    y_column: str = "y_location",
    z_column: str = "z_location",
    gene_column: str = "feature_name",
    min_molecules_cell: int = 10,
    scale_um: float = 10,
    scale_std: str = "25%",
    n_clusters: int = 5,
    prior_confidence: float = 0.7,
    polygon_format: str = "FeatureCollection",
    count_matrix_format: str = "loom",
    plot: bool = False,
    use_slurm: bool = True,
    submit: bool = True,
    slurm_args: dict | None = None,
    script_path: str | Path | None = None,
    conda_env: str | None = None,
    baysor_path: str | Path | None = None,
) -> str | None:
    """
    Run Baysor cell segmentation on spatial transcriptomics data.

    Baysor performs transcript-based cell segmentation, optionally guided by
    a prior segmentation (e.g. nuclei staining). It works directly on
    transcript coordinates, making it suitable for Xenium and similar
    imaging-based platforms.

    By default a SLURM batch script is generated and submitted via ``sbatch``.
    Set ``use_slurm=False`` to run locally.

    Parameters
    ----------
    transcripts_path : str or Path
        Path to the transcript file. Accepts ``.csv`` or ``.parquet`` format.
        For Xenium data, use ``transcripts.parquet`` from the Xenium output
        directory.
    output_path : str or Path
        Directory where Baysor writes its output files (``segmentation.csv``,
        polygon files, count matrix, etc.).
    prior_type : str or None, default=":Xenium_cell_idx"
        Prior segmentation to guide Baysor. Two formats are supported:

        - Column name in ``transcripts_path``, preceded by ``':'``,
          e.g. ``":Xenium_cell_idx"`` to use Xenium's built-in cell ID column.
        - Path to an image or MAT file with a segmentation mask.

        If ``None``, Baysor runs without a prior (transcript-only mode).
    config_path : str, Path or None, default=None
        Path to a TOML configuration file. CLI flags in this function
        override values set in the config.
    x_column : str, default="x_location"
        Name of the x-coordinate column in ``transcripts_path``.
    y_column : str, default="y_location"
        Name of the y-coordinate column in ``transcripts_path``.
    z_column : str, default="z_location"
        Name of the z-coordinate column in ``transcripts_path``.
    gene_column : str, default="feature_name"
        Name of the gene/feature column in ``transcripts_path``.
    min_molecules_cell : int, default=10
        Minimum number of transcripts for a cell to be considered real.
        This is one of the most important parameters — tested with values
        of ``10`` (permissive) and ``30`` (stricter, fewer noise cells).
    scale_um : float, default=10
        Approximate expected cell radius in micrometres (must match the
        units of ``x`` / ``y`` coordinates). For Xenium data, ``10`` µm
        is a good starting point for most cell types.
    scale_std : str, default="25%"
        Allowed variation in cell radius. Either an absolute number
        (same units as coordinates) or a percentage of ``scale``
        (e.g. ``"25%"``).
    n_clusters : int, default=5
        Number of molecule clusters (broad cell types) for the NCV model.
        A value between 3 and 15 works well for most datasets.
    prior_confidence : float, default=0.7
        Confidence weight given to the prior segmentation. Range ``[0, 1]``.
        ``0.7`` allows some flexibility while staying close to the Xenium
        prior; ``1.0`` locks Baysor to the prior boundaries exactly.
    polygon_format : str, default="FeatureCollection"
        Format for the output polygon file. Options:

        - ``"FeatureCollection"`` — GeoJSON compatible with XeniumRanger.
        - ``"GeometryCollection"`` — Baysor v0.6 legacy format.
        - ``"none"`` — do not save polygons.
    count_matrix_format : str, default="loom"
        Storage format for the segmented cell count matrix.
        Either ``"loom"`` or ``"tsv"``.
    plot : bool, default=False
        If ``True``, save an HTML file with an interactive segmentation plot.
    use_slurm : bool, default=True
        If ``True``, write a SLURM batch script and optionally submit it.
        If ``False``, run Baysor directly in the current process.
    submit : bool, default=True
        Only relevant when ``use_slurm=True``. If ``True``, submit the
        script via ``sbatch``. If ``False``, only write the script
        (e.g. for manual inspection or non-SLURM schedulers).
    slurm_args : dict or None, default=None
        SLURM resource directives. Accepted keys:

        - ``"time"`` — wall-clock limit, e.g. ``"04:00:00"``
        - ``"mem"`` — memory per node, e.g. ``"64G"``
        - ``"cpus"`` — CPUs per task, e.g. ``8``
        - ``"partition"`` — queue name, e.g. ``"gpu"``
        - ``"job_name"`` — name shown in ``squeue``
        - ``"account"`` — billing account

        Values override the defaults (4 h, 64 GB, 8 CPUs).
    script_path : str, Path or None, default=None
        Where to write the generated SLURM script. Defaults to
        ``<output_path>/segment_baysor.sh``.
    conda_env : str or None, default=None
        Conda environment to activate inside the job script. If ``None``,
        no activation line is added.
    baysor_path : str, Path or None, default=None
        Full path to the ``baysor`` executable, e.g.
        ``"/opt/software/baysor/bin/baysor"``. If ``None``, ``baysor`` is
        assumed to be on ``PATH``.

    Returns
    -------
    str or None
        - SLURM + submit: the job ID string returned by ``sbatch``.
        - SLURM + no submit: path to the generated ``.sh`` script.
        - Local mode: ``None``.

    Raises
    ------
    FileNotFoundError
        If ``transcripts_path`` or ``baysor_path`` does not exist.
    RuntimeError
        If the Baysor process or ``sbatch`` exits with a non-zero return code.

    Examples
    --------
    SLURM submission with a prior segmentation column:

    >>> staia.baysor(
    ...     transcripts_path="/data/transcripts.parquet",
    ...     output_path="/results/baysor/",
    ...     prior_type=":Xenium_cell_idx",
    ...     min_molecules_cell=10,
    ...     prior_confidence=0.7,
    ...     slurm_args={"time": "06:00:00", "mem": "128G"},
    ... )

    Generate the script only, without submitting:

    >>> staia.baysor(
    ...     transcripts_path="/data/transcripts.parquet",
    ...     output_path="/results/baysor/",
    ...     submit=False,
    ... )

    Local run (no SLURM):

    >>> staia.baysor(
    ...     transcripts_path="/data/transcripts.parquet",
    ...     output_path="/results/baysor/",
    ...     use_slurm=False,
    ... )
    """
    transcripts_path = Path(transcripts_path)
    output_path = Path(output_path)

    if not transcripts_path.exists():
        raise FileNotFoundError(f"transcripts_path does not exist: '{transcripts_path}'")

    if baysor_path is not None and not Path(baysor_path).exists():
        raise FileNotFoundError(f"baysor_path does not exist: '{baysor_path}'")

    output_path.mkdir(parents=True, exist_ok=True)

    executable = str(Path(baysor_path) / "baysor") if baysor_path else "baysor"
    command = _build_baysor_command(
        executable=executable,
        transcripts_path=transcripts_path,
        output_path=output_path,
        prior_type=prior_type,
        config_path=config_path,
        x_column=x_column,
        y_column=y_column,
        z_column=z_column,
        gene_column=gene_column,
        min_molecules_cell=min_molecules_cell,
        scale_um=scale_um,
        scale_std=scale_std,
        n_clusters=n_clusters,
        prior_confidence=prior_confidence,
        polygon_format=polygon_format,
        count_matrix_format=count_matrix_format,
        plot=plot,
    )

    if use_slurm:
        return _run_slurm(
            method="baysor",
            command=command,
            output_path=output_path,
            submit=submit,
            slurm_args=slurm_args or {},
            script_path=script_path,
            conda_env=conda_env,
            meta={
                "Input (transcripts)": str(transcripts_path),
                "Output": str(output_path),
                "Prior type": str(prior_type) if prior_type else "none",
            },
        )
    else:
        return _run_local(method="baysor", command=command)


def xeniumranger(
    xenium_bundle: str | Path,
    run_id: str,
    output_path: str | Path,
    expansion_distance: int = 0,
    boundary_stain: str | None = None,
    interior_stain: str | None = None,
    segment_large_cells: bool = True,
    localcores: int | None = None,
    localmem: int | None = None,
    use_slurm: bool = True,
    submit: bool = True,
    slurm_args: dict | None = None,
    script_path: str | Path | None = None,
    conda_env: str | None = None,
    xeniumranger_path: str | Path | None = None,
    patch_version_mismatch: bool = True,
) -> str | None:
    """
    Re-segment Xenium data using XeniumRanger.

    XeniumRanger ``resegment`` re-runs cell segmentation on an existing
    Xenium bundle, allowing you to tune expansion distance, boundary stain,
    and large-cell handling.

    By default a SLURM batch script is generated and submitted via ``sbatch``.
    Set ``use_slurm=False`` to run locally (not recommended).

    Version compatibility
    ---------------------
    The XeniumAnalyzer version embedded in the bundle is compared against the
    installed XeniumRanger version before building the job. If XeniumAnalyzer
    produced the bundle with a newer minor version than XeniumRanger supports
    (e.g. XeniumAnalyzer 4.0.2 vs XeniumRanger 4.0.0), the
    ``experiment.xenium`` file inside the bundle is patched automatically:

    - The ``"segmented_cell_boundary_large_frac"`` key is removed.
    - All ``minor_version`` fields are reset to ``0``.

    A backup (``experiment.xenium.orig``) is always written before any
    changes. Set ``patch_version_mismatch=False`` to disable this behaviour.

    Parameters
    ----------
    xenium_bundle : str or Path
        Path to the Xenium bundle directory passed to ``--xenium-bundle``.
        This is the top-level output folder produced by the Xenium instrument
        (contains ``experiment.xenium``, ``transcripts.parquet``, etc.).
    run_id : str
        Unique identifier for this run, passed to ``--id``. XeniumRanger
        uses this as the name of its output subdirectory, so it must be
        unique per invocation (e.g. ``"sample_01_reseg"``).
    output_path : str or Path
        Parent directory where XeniumRanger creates its ``<run_id>/`` output
        folder. Results land in ``<output_path>/<run_id>/outs/``.
    expansion_distance : int, default=0
        Number of pixels to expand nuclei outward to approximate cell
        boundaries, passed to ``--expansion-distance``. A value of ``0``
        uses only the nucleus boundary. Larger values capture more cytoplasm.
    boundary_stain : str or None, default=None
        Name of the stain channel used to define cell boundaries, passed to
        ``--boundary-stain`` (e.g. ``"ATP1A1/CD45/E-Cadherin"``). If ``None``
        the flag is omitted and XeniumRanger uses its default boundary channel.
    interior_stain : str or None, default=None
        Name of the stain channel used to identify the cell interior, passed
        to ``--interior-stain`` (e.g. ``"DAPI"``). If ``None`` the flag is
        omitted.
    segment_large_cells : bool, default=True
        Whether to apply the large-cell segmentation step, passed to
        ``--segment-large-cells``. Disable this if your tissue does not
        contain large cells to speed up the run.
    localcores : int or None, default=None
        Maximum number of CPU cores XeniumRanger may use. If ``None``,
        XeniumRanger uses all cores available on the node.
        Tip: set this to match ``slurm_args["cpus"]`` to avoid
        over-subscribing the node.
    localmem : int or None, default=None
        Maximum memory in GB XeniumRanger may use. If ``None``,
        XeniumRanger auto-detects available memory.
        Tip: set this slightly below your SLURM ``mem`` request (e.g. ``120``
        when requesting ``128G``) to leave headroom for the OS.
    use_slurm : bool, default=True
        If ``True``, write a SLURM batch script and optionally submit it.
        If ``False``, run XeniumRanger directly in the current process.
    submit : bool, default=True
        Only relevant when ``use_slurm=True``. If ``True``, submit the
        script via ``sbatch``. If ``False``, only write the script (useful
        for inspection or non-SLURM schedulers).
    slurm_args : dict or None, default=None
        SLURM resource directives. Accepted keys:

        - ``"time"`` — wall-clock limit, e.g. ``"08:00:00"``
        - ``"mem"`` — memory per node, e.g. ``"256G"``
        - ``"cpus"`` — CPUs per task, e.g. ``16``
        - ``"partition"`` — queue name
        - ``"job_name"`` — name shown in ``squeue``
        - ``"account"`` — billing account

        Values override the defaults (8 h, 256 GB, 16 CPUs).
    script_path : str, Path or None, default=None
        Where to write the generated SLURM script. Defaults to
        ``<output_path>/segment_xeniumranger.sh``.
    conda_env : str or None, default=None
        Conda environment to activate inside the job script. If ``None``,
        no activation line is added.
    xeniumranger_path : str, Path or None, default=None
        Full path to the directory containing the ``xeniumranger`` executable,
        e.g. ``"/opt/software/xeniumranger-3.0.1"``. If ``None``,
        ``xeniumranger`` is assumed to be on ``PATH``.
    patch_version_mismatch : bool, default=True
        If ``True`` (the default), automatically check for XeniumAnalyzer /
        XeniumRanger version mismatches and patch ``experiment.xenium`` when
        needed. Set to ``False`` to skip the check and patch entirely.

    Returns
    -------
    str or None
        - SLURM + submit: the job ID string returned by ``sbatch``.
        - SLURM + no submit: path to the generated ``.sh`` script.
        - Local mode: ``None``.

    Raises
    ------
    FileNotFoundError
        If ``xenium_bundle`` or ``xeniumranger_path`` does not exist.
    RuntimeError
        If the XeniumRanger process or ``sbatch`` exits with a non-zero
        return code.

    Examples
    --------
    SLURM submission matching your team's typical usage:

    >>> staia.xeniumranger(
    ...     xenium_bundle="/path/to/xenium/bundle",
    ...     run_id="sample_01",
    ...     output_path="/results/xeniumranger/",
    ...     expansion_distance=0,
    ...     boundary_stain="ATP1A1/CD45/E-Cadherin",
    ...     segment_large_cells=True,
    ...     localcores=16,
    ...     localmem=120,
    ...     slurm_args={
    ...         "time": "08:00:00",
    ...         "mem": "128G",
    ...         "cpus": 16,
    ...         "partition": "gpu-long",
    ...     },
    ... )

    Generate the script only, without submitting:

    >>> staia.xeniumranger(
    ...     xenium_bundle="/path/to/xenium/bundle",
    ...     run_id="sample_01",
    ...     output_path="/results/xeniumranger/",
    ...     submit=False,
    ... )

    Local run (no SLURM):

    >>> staia.xeniumranger(
    ...     xenium_bundle="/path/to/xenium/bundle",
    ...     run_id="sample_01",
    ...     output_path="/results/xeniumranger/",
    ...     use_slurm=False,
    ... )

    Disable automatic version patching:

    >>> staia.xeniumranger(
    ...     xenium_bundle="/path/to/xenium/bundle",
    ...     run_id="sample_01",
    ...     output_path="/results/xeniumranger/",
    ...     patch_version_mismatch=False,
    ... )
    """
    xenium_bundle = Path(xenium_bundle)
    output_path   = Path(output_path)

    if not xenium_bundle.exists():
        raise FileNotFoundError(f"xenium_bundle does not exist: '{xenium_bundle}'")

    if xeniumranger_path is not None and not Path(xeniumranger_path).exists():
        raise FileNotFoundError(f"xeniumranger_path does not exist: '{xeniumranger_path}'")

    output_path.mkdir(parents=True, exist_ok=True)

    executable = (
        str(Path(xeniumranger_path) / "xeniumranger")
        if xeniumranger_path
        else "xeniumranger"
    )

    # ------------------------------------------------------------------
    # Version check + patch (runs before the job is submitted / started)
    # A COPY of the bundle is patched — the original is never modified.
    # ------------------------------------------------------------------
    bundle_to_use = xenium_bundle
    if patch_version_mismatch:
        needs_patch = _check_xenium_version_compatibility(xenium_bundle, executable)
        if needs_patch:
            bundle_to_use = _patch_experiment_xenium(xenium_bundle, output_path)

    command = _build_xeniumranger_command(
        executable=executable,
        xenium_bundle=bundle_to_use,
        run_id=run_id,
        output_path=output_path,
        expansion_distance=expansion_distance,
        boundary_stain=boundary_stain,
        interior_stain=interior_stain,
        segment_large_cells=segment_large_cells,
        localcores=localcores,
        localmem=localmem,
    )

    if use_slurm:
        return _run_slurm(
            method="xeniumranger",
            command=command,
            output_path=output_path,
            submit=submit,
            slurm_args=slurm_args or {},
            script_path=script_path,
            conda_env=conda_env,
            meta={
                "Input (xenium bundle)": str(xenium_bundle),
                "Output": str(output_path),
                "Run ID": run_id,
                "Expansion distance": expansion_distance,
                "Boundary stain": str(boundary_stain) if boundary_stain else "default",
                "Segment large cells": segment_large_cells,
            },
        )
    else:
        return _run_local(method="xeniumranger", command=command)


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _build_baysor_command(
    executable: str,
    transcripts_path: Path,
    output_path: Path,
    prior_type: str | None,
    config_path: Path | None,
    x_column: str,
    y_column: str,
    z_column: str,
    gene_column: str,
    min_molecules_cell: int,
    scale_um: float,
    scale_std: str,
    n_clusters: int,
    prior_confidence: float,
    polygon_format: str,
    count_matrix_format: str,
    plot: bool,
) -> str:
    """Return the full ``baysor run`` command string."""
    parts = [
        executable, "run",
        f"-x {x_column}",
        f"-y {y_column}",
        f"-z {z_column}",
        f"-g {gene_column}",
        f"-m {min_molecules_cell}",
        f"-s {scale_um}",
        f"--scale-std={scale_std}",
        f"--n-clusters={n_clusters}",
        f"--prior-segmentation-confidence={prior_confidence}",
        f"--polygon-format={polygon_format}",
        f"--count-matrix-format={count_matrix_format}",
        f"-o {output_path}",
    ]

    if config_path is not None:
        parts.append(f"-c {config_path}")

    if plot:
        parts.append("-p")

    # Positional args: transcripts file, then optional prior
    parts.append(str(transcripts_path))
    if prior_type is not None:
        parts.append(str(prior_type))

    return " \\\n    ".join(parts)


def _build_xeniumranger_command(
    executable: str,
    xenium_bundle: Path,
    run_id: str,
    output_path: Path,
    expansion_distance: int,
    boundary_stain: str | None,
    interior_stain: str | None,
    segment_large_cells: bool,
    localcores: int | None,
    localmem: int | None,
) -> str:
    """Return the full ``xeniumranger resegment`` command string."""
    parts = [
        executable, "resegment",
        f"--xenium-bundle={xenium_bundle}",
        f"--id={run_id}",
        f"--expansion-distance={expansion_distance}",
        f"--segment-large-cells={'true' if segment_large_cells else 'false'}",
    ]

    if boundary_stain is not None:
        parts.append(f"--boundary-stain={boundary_stain}")

    if interior_stain is not None:
        parts.append(f"--interior-stain={interior_stain}")

    if localcores is not None:
        parts.append(f"--localcores={localcores}")

    if localmem is not None:
        parts.append(f"--localmem={localmem}")

    return " \\\n    ".join(parts)


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

_SLURM_DEFAULTS = {
    "baysor": {
        "job_name": "baysor_segmentation",
        "time":     "04:00:00",
        "mem":      "256G",
        "cpus":     8,
    },
    "xeniumranger": {
        "job_name": "xeniumranger_segmentation",
        "time":     "72:00:00",
        "mem":      "256G",
        "cpus":     16,
        "partition": "gpu-long",
    },
}


def _build_slurm_header(method: str, log_dir: Path, slurm_args: dict) -> str:
    """Return the ``#SBATCH`` directive block for a job script."""
    cfg = {**_SLURM_DEFAULTS[method], **slurm_args}

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={cfg.get('job_name', method)}",
        f"#SBATCH --time={cfg.get('time', '04:00:00')}",
        f"#SBATCH --mem={cfg.get('mem', '64G')}",
        f"#SBATCH --cpus-per-task={cfg.get('cpus', 8)}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
    ]

    if "partition" in cfg:
        lines.append(f"#SBATCH --partition={cfg['partition']}")
    if "account" in cfg:
        lines.append(f"#SBATCH --account={cfg['account']}")

    return "\n".join(lines)


def _build_slurm_script(
    method: str,
    command: str,
    output_path: Path,
    slurm_args: dict,
    conda_env: str | None,
    meta: dict,
) -> str:
    """Assemble the full SLURM batch script as a string."""
    log_dir = output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    header = _build_slurm_header(method, log_dir, slurm_args)

    meta_block = "\n".join(f"# {k:<25}: {v}" for k, v in meta.items())

    env_block = ""
    if conda_env:
        env_block = textwrap.dedent(f"""
            # Activate conda environment
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate {conda_env}
        """).strip()

    return textwrap.dedent(f"""
        {header}
 
        # -----------------------------------------------------------------------
        # Auto-generated by STAIA _segmentation.py — do not edit while running
        # Method                   : {method}
        {meta_block}
        # -----------------------------------------------------------------------
 
        set -euo pipefail
 
        # Unset Jupyter-injected matplotlib backend — incompatible with XeniumRanger's bundled Python
        unset MPLBACKEND
        {("" + chr(10) + env_block + chr(10)) if env_block else ""}
        echo "[$(date)] Starting {method} segmentation"
 
        {command}
 
        echo "[$(date)] Finished {method} segmentation"
    """).strip()


def _run_slurm(
    method: str,
    command: str,
    output_path: Path,
    submit: bool,
    slurm_args: dict,
    script_path: str | Path | None,
    conda_env: str | None,
    meta: dict,
) -> str:
    """Write the SLURM script and optionally submit it via sbatch."""
    if script_path is None:
        script_path = output_path / f"segment_{method}.sh"

    script_path = Path(script_path)
    script = _build_slurm_script(
        method=method,
        command=command,
        output_path=output_path,
        slurm_args=slurm_args,
        conda_env=conda_env,
        meta=meta,
    )

    script_path.write_text(script)
    print(f"SLURM script written to: {script_path}")

    if not submit:
        print(
            f"submit=False — script was not submitted.\n"
            f"Inspect and run manually with:\n  sbatch {script_path}"
        )
        return str(script_path)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (return code {result.returncode}).\n"
            f"stderr: {result.stderr.strip()}"
        )

    job_id = result.stdout.strip()
    print(f"Job submitted: {job_id}")
    print(f"SLURM script saved to: {script_path}")
    print(f"To review the full command:  cat {script_path}")
    return job_id


# ---------------------------------------------------------------------------
# Local (non-SLURM) runner
# ---------------------------------------------------------------------------

def _run_local(method: str, command: str) -> None:
    """Run the segmentation command directly in the current process."""
    print(
        f"Running {method} locally — this may take a long time.\n"
        f"Consider use_slurm=True for large datasets.\n"
    )

    flat_command = " ".join(command.split())
    print(f"Command: {flat_command}\n")

    result = subprocess.run(flat_command, shell=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"{method} exited with return code {result.returncode}."
        )

    print(f"{method} segmentation complete.")