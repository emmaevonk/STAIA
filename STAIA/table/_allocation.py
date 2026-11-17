from spatialdata.transformations import get_transformation, set_transformation
from spatialdata.models import ShapesModel, TableModel
from shapely.validation import make_valid
from shapely.geometry import Polygon
from spatialdata import SpatialData
from anndata import AnnData
import scipy.sparse as sp
import geopandas as gpd
import spatialdata as sd
import pandas as pd
import scanpy as sc
import numpy as np
import shapely
import copy
import gc
import os


def add_nuclei(
    sdata : SpatialData,
    path_xenium : str,
    label_id : str = "label_id",
    cell_id : str = "cell_id",
    shapes_layer : str = "nucleus_boundaries_fixed"
):
    """
    Add nuclei inside the cells to a separate shape in the SpatialData object.

    Parameters
    ----------
    sdata: SpatialData
        SpatialData object containing spatial transcriptomic information.
    path_xenium: str
        Path to the Xenium output of the segmentation file. Must contain `nucleus_boundaries.parquet`
    label_id : str, default = "label_id"
        column in the Xenium parquet file shown the label IDs (unique per nuclei)
    cell_id : str, default = "cell_id"
        Column in the parquet file showing the unique cell IDs
    shapes_layer : str, default = "nucleus_boundaries_fixed"
        Shapes layer added to the sdata object containing the nucleus boundaries per nucleus in the cells

    Returns
    -------
    sdata : sdata where a new shape which includes the nuclei in the cells has been added in shape
    ``nucleus_boundaries_fixed``.
    """
    if os.path.isdir(path_xenium):
        sep = "" if path_xenium.endswith("/") else "/"
        path_xenium = path_xenium + sep + "nucleus_boundaries.parquet"

    if not os.path.isfile(path_xenium):
        print(
            "The path to the Xenium output (containing `nucleus_boundaries.parquet`) "
            "does not exist. Change the path and try again."
        )
        return

    df = pd.read_parquet(path_xenium)
    gdf = _assign_nuclei(df, label_id, cell_id)
    parsed = ShapesModel.parse(gdf)

    if "nucleus_boundaries" in sdata.shapes:
        transform = get_transformation(
            sdata.shapes["nucleus_boundaries"],
            to_coordinate_system="global"
        )
        set_transformation(parsed, transform, "global")
        print("Copied transform from 'nucleus_boundaries':", transform)
    else:
        # fallback
        transform = get_transformation(
            sdata.points["transcripts"],
            to_coordinate_system="global"
        )
        set_transformation(parsed, transform, "global")
        print("Warning: 'nucleus_boundaries' not found, copied transform "
              "from transcripts instead:", transform)

    sdata.shapes[shapes_layer] = parsed
    return sdata
    

def get_adata_per_nucleus(sdata: SpatialData, table_layer: str = "table", min_qv: float = 20.0):
    """
    Return per-nucleus transcript counts and a matching per-cell fiber AnnData.

    Uses the Xenium-provided ``overlaps_nucleus`` flag to identify nuclear
    transcripts (consistent with XeniumRanger), then uses a spatial join
    only to assign each nuclear transcript to a specific nucleus within its
    cell. Applies the same quality score filter (qv >= 20 by default) that
    XeniumRanger uses internally, so counts are consistent with the official
    Xenium output. Fiber counts are obtained by subtracting the per-nucleus
    counts (summed to cell level) from the original per-cell counts.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object. Must contain:
        - sdata.shapes["nucleus_boundaries_fixed"]  (run STAIA.add_nuclei() first)
        - sdata.shapes["cell_boundaries"]
        - sdata.points["transcripts"]  with ``overlaps_nucleus`` and ``qv`` columns
        - sdata.tables[table_layer]
    table_layer : str, default "table"
        Key of the per-cell table in ``sdata.tables``.
    min_qv : float, default 20.0
        Minimum quality score threshold. Transcripts with qv below this
        value are excluded, consistent with XeniumRanger's default filter.
        Set to 0.0 to disable filtering.

    Returns
    -------
    tuple[AnnData, AnnData]
        ``(adata_per_nucleus, adata_fibers)``

        - ``adata_per_nucleus``: one row per nucleus, with ``cell_id``,
          ``label_id``, ``area``, ``nuclear_counts`` and ``nuc_area_fraction``
          in ``.obs``.
        - ``adata_fibers``: one row per cell, counts outside all nuclei
          (total counts minus nuclear counts), with ``cell_area``,
          ``nucleus_area``, ``fiber_area``, ``cytoplasmic_counts``,
          ``total_counts``, ``nuc_area_fraction``, and
          ``fiber_area_fraction`` in ``.obs``.

        Both are stored in ``sdata.tables`` as ``"adata_per_nucleus"``
        and ``"adata_fibers_per_nucleus"``.
    """
    if "nucleus_boundaries_fixed" not in sdata.shapes:
        raise ValueError(
            "Run STAIA.add_nuclei() first — 'nucleus_boundaries_fixed' not found."
        )

    print("Building per-nucleus AnnData using overlaps_nucleus flag...")
    adata = sdata.tables[table_layer]

    # PREPARE NUCLEUS SHAPES                                               
    nuc_shapes = sdata.shapes["nucleus_boundaries_fixed"].copy()
    nuc_shapes["geometry"] = nuc_shapes.geometry.apply(make_valid).apply(_to_polygonal)
    nuc_shapes = nuc_shapes[nuc_shapes.geometry.notna()].copy()
    nuc_shapes["nucleus_id"] = nuc_shapes.index
    nuc_shapes = nuc_shapes.reset_index(drop=True)

    # Compute area before anything else so it is available later
    nuc_shapes["area"] = nuc_shapes.geometry.area
    nucleus_index = pd.Index(nuc_shapes["nucleus_id"])
    gene_index    = pd.Index(adata.var_names)

    # 1. Load transcripts, filter unassigned and by quality score         #
    print("Loading transcripts...")
    tx = sdata.points["transcripts"].compute()
    tx = tx[tx["cell_id"] != "UNASSIGNED"].copy()
    before = len(tx)
    tx = tx[tx["qv"] >= min_qv].copy()
    print(f"  Transcripts after qv >= {min_qv} filter: "
          f"{len(tx):,} (removed {before - len(tx):,})")

    original_total = int(adata.X.sum())
    if abs(len(tx) - original_total) / max(original_total, 1) > 0.05:
        print(
            f"  WARNING: filtered transcript count ({len(tx):,}) differs "
            f"from original table total counts ({original_total:,}) by more "
            f"than 5%. The original table may have been built with a different "
            f"qv threshold. Consider adjusting min_qv."
        )

    tx_nuc = tx[tx["overlaps_nucleus"] == 1].copy()
    del tx
    gc.collect()
    print(f"  Nuclear transcripts: {len(tx_nuc):,}")

    # 2. Spatial join: assign each nuclear transcript to a nucleus
    print(f"Spatial join: {len(tx_nuc):,} nuclear transcripts x "
          f"{len(nuc_shapes):,} nuclei...")

    tx_nuc_gdf = gpd.GeoDataFrame(
        tx_nuc.reset_index(drop=True),
        geometry=gpd.points_from_xy(tx_nuc["x"].values, tx_nuc["y"].values),
        crs=nuc_shapes.crs,
    )
    del tx_nuc
    gc.collect()

    joined = gpd.sjoin(
        tx_nuc_gdf,
        nuc_shapes[["nucleus_id", "cell_id", "label_id", "geometry"]],
        how="left",
        predicate="within",
    )
    del tx_nuc_gdf
    gc.collect()

    unmatched = joined["index_right"].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched:,} nuclear transcripts could not be "
              f"assigned to a specific nucleus (edge/boundary cases) — excluded.")

    joined = joined[joined["index_right"].notna()].copy()

    # Map integer index_right to nucleus_id string
    joined["nucleus_id"] = joined["index_right"].map(
        pd.Series(nuc_shapes["nucleus_id"].values, index=range(len(nuc_shapes)))
    )

    # 3. Build sparse count matrix using Categorical codes
    nuc_cat = pd.Categorical(
        joined["nucleus_id"].astype(str),
        categories=nucleus_index.astype(str),
    )
    gene_cat = pd.Categorical(
        joined["feature_name"],
        categories=gene_index,
    )

    valid = (nuc_cat.codes >= 0) & (gene_cat.codes >= 0)
    print(f"  Valid transcript-to-nucleus assignments: {valid.sum():,} "
          f"out of {len(joined):,}")

    if valid.sum() == 0:
        print("ERROR: No valid assignments after Categorical mapping.")
        print("  nucleus_id sample:", joined["nucleus_id"].head(5).tolist())
        print("  nucleus_index sample:", nucleus_index[:5].tolist())
        print("  feature_name sample:", joined["feature_name"].head(5).tolist())
        print("  gene_index sample:", gene_index[:5].tolist())
        raise ValueError(
            "No transcripts could be assigned to nuclei. "
            "Check the debug output above for index mismatches."
        )

    nuc_pos  = nuc_cat.codes[valid]
    gene_pos = gene_cat.codes[valid]

    nuc_X = sp.csr_matrix(
        (np.ones(len(nuc_pos), dtype=np.int32), (nuc_pos, gene_pos)),
        shape=(len(nucleus_index), len(gene_index)),
    )
    print(f"  nuc_X total counts: {nuc_X.sum():,}")

    del joined, nuc_cat, gene_cat, nuc_pos, gene_pos
    gc.collect()

    # 4. Build per-nucleus AnnData
    nuc_obs       = nuc_shapes[["nucleus_id", "cell_id", "label_id"]].copy()
    nuc_obs.index = nucleus_index

    adata_per_nucleus = AnnData(
        X=nuc_X,
        obs=nuc_obs,
        var=adata.var.copy(),
    )
    del nuc_X
    gc.collect()

    # 5. Build fiber AnnData by subtracting nuclear counts from original  

    # per-cell counts — fiber = cytoplasmic only
    # Map cell_id -> row position in adata via adata.obs["cell_id"]

    cell_id_to_pos = pd.Series(
        range(len(adata.obs)),
        index=adata.obs["cell_id"].astype(str),
    )

    cell_id_series  = adata_per_nucleus.obs["cell_id"].astype(str)
    nucleus_to_cell = cell_id_series.map(cell_id_to_pos)
    n_unmapped = nucleus_to_cell.isna().sum()
    if n_unmapped > 0:
        print(f"  WARNING: {n_unmapped} nuclei could not be mapped to a cell "
              f"— excluded from fiber subtraction (expected for nuclei "
              f"outside cell boundaries).")

    valid_mask = nucleus_to_cell.notna().values
    valid_positions = nucleus_to_cell[valid_mask].astype(int).values
    n_cells = len(adata.obs)
    n_valid = valid_mask.sum()

    print(f"  Nuclei mapped to cells: {n_valid:,} / {len(adata_per_nucleus):,}")

    summation_matrix = sp.csr_matrix(
        (
            np.ones(n_valid, dtype=np.float32),
            (valid_positions, np.arange(n_valid)),
        ),
        shape=(n_cells, n_valid),
    )

    nuc_X_valid = sp.csr_matrix(adata_per_nucleus.X[valid_mask])
    nuc_counts_per_cell = summation_matrix @ nuc_X_valid
    del summation_matrix, nuc_X_valid
    gc.collect()

    print(f"  nuc_counts_per_cell shape: {nuc_counts_per_cell.shape}")
    print(f"  nuc_counts_per_cell total: {int(nuc_counts_per_cell.sum()):,}")

    # Reindex columns to match adata.var_names exactly before subtraction
    if not pd.Index(adata_per_nucleus.var_names).equals(pd.Index(adata.var_names)):
        print("  WARNING: var_names differ — reindexing nuc_counts_per_cell columns...")
        col_pos = pd.Index(adata_per_nucleus.var_names).get_indexer(adata.var_names)
        present = col_pos >= 0
        reindexed = sp.lil_matrix((n_cells, len(adata.var_names)), dtype=np.float32)
        reindexed[:, np.where(present)[0]] = nuc_counts_per_cell[:, col_pos[present]]
        nuc_counts_per_cell = sp.csr_matrix(reindexed)
        del reindexed
        gc.collect()

    whole_X  = sp.csr_matrix(adata.X)
    fiber_X  = whole_X - nuc_counts_per_cell
    del whole_X, nuc_counts_per_cell
    gc.collect()

    fiber_X      = fiber_X.tocsr()
    fiber_X.data = np.clip(fiber_X.data, a_min=0, a_max=None)
    fiber_X.eliminate_zeros()

    # print(f"  whole_X total: {int(sp.csr_matrix(adata.X).sum()):,}")
    # print(f"  fiber_X total: {int(fiber_X.sum()):,}")
    # print(f"  nuc_X total:   {int(adata_per_nucleus.X.sum()):,}")
    # print(f"  fiber + nuc should equal whole: "
    #       f"{int(fiber_X.sum()) + int(adata_per_nucleus.X.sum()):,}")

    obsm = {"spatial": adata.obsm["spatial"].copy()} if "spatial" in adata.obsm else {}
    obs  = adata.obs.drop(columns=["region"], errors="ignore").copy()

    adata_fibers = AnnData(
        X=fiber_X,
        obs=obs,
        var=adata.var.copy(),
        obsm=obsm,
    )
    del fiber_X
    gc.collect()

    # 6. QC metrics, areas, fractions and count columns
    sc.pp.calculate_qc_metrics(
        adata_per_nucleus, inplace=True, percent_top=None, log1p=False
    )
    sc.pp.calculate_qc_metrics(
        adata_fibers, inplace=True, percent_top=None, log1p=False
    )

    # Explicit count columns
    adata_per_nucleus.obs["nuclear_counts"] = np.asarray(
        adata_per_nucleus.X.sum(axis=1)
    ).ravel()
    adata_fibers.obs["cytoplasmic_counts"] = adata_fibers.obs["total_counts"].copy()
    adata_fibers.obs["total_counts"] = adata.obs["total_counts"].values

    # Per-nucleus area from nuc_shapes
    adata_per_nucleus.obs["area"] = (
        adata_per_nucleus.obs["nucleus_id"]
        .map(nuc_shapes.set_index("nucleus_id")["area"])
    )

    # Sum nucleus areas per cell from adata_per_nucleus
    nucleus_area_per_cell = (
        adata_per_nucleus.obs.groupby("cell_id")["area"].sum()
    )
    nucleus_area_per_cell.index = nucleus_area_per_cell.index.astype(str)

    # Cell area from adata.obs — already in µm² from XeniumRanger
    cell_area = adata.obs.set_index("cell_id")["cell_area"]
    cell_area.index = cell_area.index.astype(str)

    adata_fibers.obs["cell_area"] = (
        adata_fibers.obs["cell_id"].astype(str).map(cell_area)
    )
    adata_fibers.obs["nucleus_area"] = (
        adata_fibers.obs["cell_id"].astype(str).map(nucleus_area_per_cell)
    )
    adata_fibers.obs["cytoplasm_area"] = (
        adata_fibers.obs["cell_area"] - adata_fibers.obs["nucleus_area"].fillna(0)
    ).clip(lower=0)
    adata_fibers.obs["nuc_area_fraction"] = (
        adata_fibers.obs["nucleus_area"] / adata_fibers.obs["cell_area"]
    ).clip(upper=1.0)
    adata_fibers.obs["cytoplasm_area_fraction"] = (
        1 - adata_fibers.obs["nuc_area_fraction"]
    ).clip(lower=0)

    # 7. Store into sdata via TableModel.parse
    region_key = adata.uns.get("spatialdata_attrs", {}).get("region_key", "region")

    adata_per_nucleus.obs[region_key] = pd.Categorical(
        ["nucleus_boundaries_fixed"] * len(adata_per_nucleus)
    )
    sdata.tables["adata_per_nucleus"] = TableModel.parse(
        adata_per_nucleus,
        region="nucleus_boundaries_fixed",
        region_key=region_key,
        instance_key="label_id",
    )

    adata_fibers.obs[region_key] = pd.Categorical(
        ["cell_boundaries"] * len(adata_fibers)
    )
    sdata.tables["adata_fibers_per_nucleus"] = TableModel.parse(
        adata_fibers,
        region="cell_boundaries",
        region_key=region_key,
        instance_key="cell_id",
    )

    # print(f"Done. adata_per_nucleus: {adata_per_nucleus.shape}, "
    #       f"adata_fibers: {adata_fibers.shape}")
    adata_fibers.obs = adata_fibers.obs.drop(columns=["total_counts"])
    return adata_per_nucleus, adata_fibers

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assign_nuclei(
        df : pd.DataFrame,
        label : str,
        cell_id : str
):
    """
    Build a GeoDataFrame of nucleus polygons from a Xenium vertex table.

    Parameters
    ----------
    df : pd.DataFrame
        Xenium ``nucleus_boundaries.parquet`` loaded as a DataFrame.
    label : str
        Column name for the per-nucleus label ID.
    cell_id : str
        Column name for the parent cell ID.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per nucleus with columns ``label_id``, ``cell_id``, and
        ``geometry``.
    """
    polygons, cell_ids, label_ids = [], [], []

    for label_id, group in df.groupby(label):
        coords = list(zip(group["vertex_x"], group["vertex_y"]))
        if len(coords) > 3:
            polygons.append(Polygon(coords))
            label_ids.append(label_id)
            cell_ids.append(group[cell_id].iloc[0])

    # create a geo dataframe
    gdf = gpd.GeoDataFrame({
        "label_id": label_ids,
        "cell_id": cell_ids,
        "geometry": polygons
    })

    # fix indexing
    gdf.index = [f"nucleus_{i}" for i in range(len(gdf))]
    return gdf


def _to_polygonal(geom):
    """
    Coerce a (possibly mixed-type) geometry into a Polygon or MultiPolygon.

    Geometry repair via `shapely.validation.make_valid` can occasionally
    return a `GeometryCollection` containing a mix of points, lines, and
    polygons. This function extracts only the polygonal parts and merges
    them, discarding stray points/lines that carry no area.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input geometry, typically the output of `make_valid()`.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None
        The polygonal part(s) of `geom`, merged into a single geometry.
        Returns `None` if `geom` contains no polygonal parts at all.
    """
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    polys = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.unary_union(polys) if polys else None
