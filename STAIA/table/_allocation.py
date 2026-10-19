from spatialdata.transformations import get_transformation
from shapely.validation import make_valid
from spatialdata.transformations import get_transformation, set_transformation
from spatialdata.models import ShapesModel
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata import read_zarr
from anndata import AnnData
import scipy.sparse as sp
import spatialdata as sd
import geopandas as gpd
import pandas as pd
import scanpy as sc
import shapely
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
        if path_xenium[-1] == "/":
            path_xenium = path_xenium + "nucleus_boundaries.parquet"
        else:
            path_xenium = path_xenium + "/nucleus_boundaries.parquet"
    # check if path to xenium output exists
    if os.path.isfile(path_xenium):
        df = pd.read_parquet(path_xenium)
        # returns geodataframe containg unique label IDs with cell IDs and polygon shapes
        gdf = _assign_nuclei(df, label_id, cell_id)

        parsed = ShapesModel.parse(gdf)
        transform = get_transformation(sdata.points["transcripts"], to_coordinate_system="global")
        set_transformation(parsed, transform, "global")

        sdata.shapes[shapes_layer] = parsed
        # sdata_fixed = sdata.shapes[shapes_layer]
        return sdata
    else:
        print(f"The path to the Xenium output (containing the `nucleus_boundaries.parquet` file) does not exist. The nuclei have not been added to sdata. Change the path to an existing path with Xenium output and try again.")
        return
    
def _assign_nuclei(
        df : pd.DataFrame,
        label : str,
        cell_id : str
):
    polygons, cell_ids, label_ids = [], [], []

    for label_id, group in df.groupby(label):
        coords = list(zip(group["vertex_x"], group["vertex_y"]))

        # check if the polygon is valid
        if len(coords) > 3:
            poly = Polygon(coords)
            polygons.append(poly)
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


#NOTE: original
# def get_adata_nuclei(sdata: SpatialData):
#     """
#     Split each cell's transcripts into nuclear and fiber (non-nuclear) compartments.

#     Uses the nucleus shapes layer added by `STAIA.add_nuclei()` (or if that hasn't been run, falls back to the raw `nucleus_boundaries` layer with a warning) to build two separate AnnData objects: one containing only transcripts whose coordinates fall within a cell's nucleus, and one containing the remainder (transcripts within th ecell boundary but outside the nucleus). Both objects are annotated with standard QC metrics (total counts, n_genes) and a per-cell `area` column via `_add_wc_col`.

#     Parameters
#     ----------
#     sdata : SpatialData
#         A SpatialData object containing a "table" element
    
#     Returns
#     -------
#     tuple[AnnData, AnnData]
#         `(adata_fibers, adata_nuclei)` — AnnData objects with the same
#         `obs`/`var` as the original table, but with `X` restricted to
#         nuclear or fiber transcript counts respectively, and QC columns
#         (`total_counts`, `n_genes_by_counts`, `area`) added to `.obs`.

#     Notes
#     -----
#     This can take a while on large samples, since it aggregates transcripts
#     over polygon geometries for every cell.
#     """

#     print("It takes a while to retrieve both the nuclei AnnData object, and the fiber AnnData object. Please wait...")
#     if type(sdata) != SpatialData:
#         print(f'The input provided should be SpatialData, not {type(sdata)}. Exiting code...')
#         return
#     # check if `STAIA.add_nuclei()` has been run
#     if 'nucleus_boundaries_fixed' in sdata.shapes:
#         sdata, nuc = _repair_nucleus(sdata)
#     else: # use nucleus_boundaries and show a warning 
#         print(f"STAIA.add_nuclei() function is not used before. Will use the 'nucleus_boundaries' column in sdata")
#         sdata, nuc = _repair_nucleus(sdata, shape="nucleus_boundaries")

#     agg_result = sd.aggregate(
#         values=sdata.points["transcripts"],
#         by=sdata.shapes["nucleus_boundaries_by_cell"],
#         value_key="feature_name",
#         agg_func="count",
#     )
#     adata_nuclei_raw = agg_result.tables["table"]
    
#     # raw_df = _to_df(adata_nuclei_raw)
#     # raw_df.index = raw_df.index.astype(str)

#     adata = sdata.tables["table"]  

#     # nuc_counts = raw_df.reindex(index=adata.obs_names, columns=adata.var_names, fill_value=0)
#     #NOTE: no counts come out of this function. Walk through all of this.
#     nuc_counts = _to_df(adata_nuclei_raw).reindex(index=adata.obs_names, columns=adata.var_names, fill_value=0) # NOTE: what does `fill_value` do? Is this causing the problem
#     # temporary check - can be removed later
#     if nuc_counts.values.sum() == 0:
#         print("WARNING: nuc_counts is all zeros — obs_names mismatch between "
#             "adata_nuclei_raw and adata. Check: \n"
#             f"  adata_nuclei_raw.obs_names[:5]: {adata_nuclei_raw.obs_names[:5].tolist()}\n"
#             f"  adata.obs_names[:5]:            {adata.obs_names[:5].tolist()}")
            
#     whole_counts = _to_df(adata)
#     fiber_counts = (whole_counts - nuc_counts).clip(lower=0)

#     adata_nuclei = AnnData(X=sp.csr_matrix(nuc_counts.values), obs=adata.obs.copy(), var=adata.var.copy())
#     adata_fibers = AnnData(X=sp.csr_matrix(fiber_counts.values), obs=adata.obs.copy(), var=adata.var.copy())
#     adata_fiber, adata_nuclei= _add_qc_col(adata_nuclei, adata_fibers, nuc, sdata)
#     return adata_fiber, adata_nuclei

def get_adata_nuclei(sdata: SpatialData):
    """
    Split each cell's transcripts into nuclear and fiber (non-nuclear) compartments.

    Uses the nucleus shapes layer added by `STAIA.add_nuclei()` (or if that hasn't been run, falls back to the raw `nucleus_boundaries` layer with a warning) to build two separate AnnData objects: one containing only transcripts whose coordinates fall within a cell's nucleus, and one containing the remainder (transcripts within the cell boundary but outside the nucleus). Both objects are annotated with standard QC metrics (total counts, n_genes) and a per-cell `area` column via `_add_qc_col`.

    Parameters
    ----------
    sdata : SpatialData
        A SpatialData object containing a "table" element
    
    Returns
    -------
    tuple[AnnData, AnnData]
        `(adata_fibers, adata_nuclei)` — AnnData objects with the same
        `obs`/`var`/`obsm`/`uns` as the original table, but with `X` restricted to
        fiber or nuclear transcript counts respectively, and QC columns
        (`total_counts`, `n_genes_by_counts`, `area`) added to `.obs`.
        Both tables are also stored in `sdata.tables` as `"adata_fibers"`
        and `"adata_nuclei"`.

    Notes
    -----
    This can take a while on large samples, since it aggregates transcripts
    over polygon geometries for every cell.
    """
    import copy

    print("It takes a while to retrieve both the nuclei AnnData object, and the fiber AnnData object. Please wait...")
    if type(sdata) != SpatialData:
        print(f'The input provided should be SpatialData, not {type(sdata)}. Exiting code...')
        return
    # check if `STAIA.add_nuclei()` has been run
    if 'nucleus_boundaries_fixed' in sdata.shapes:
        sdata, nuc = _repair_nucleus(sdata)
    else: # use nucleus_boundaries and show a warning 
        print(f"STAIA.add_nuclei() function is not used before. Will use the 'nucleus_boundaries' column in sdata")
        sdata, nuc = _repair_nucleus(sdata, shape="nucleus_boundaries")

    agg_result = sd.aggregate(
        values=sdata.points["transcripts"],
        by=sdata.shapes["nucleus_boundaries_by_cell"],
        value_key="feature_name",
        agg_func="count",
    )
    adata_nuclei_raw = agg_result.tables["table"]
    
    raw_df = _to_df(adata_nuclei_raw)
    raw_df.index = raw_df.index.astype(str)

    adata = sdata.tables["table"]  

    nuc_counts = raw_df.reindex(index=adata.obs_names, columns=adata.var_names, fill_value=0)
    # temporary check - can be removed later
    if nuc_counts.values.sum() == 0:
        print("WARNING: nuc_counts is all zeros — obs_names mismatch between "
            "adata_nuclei_raw and adata. Check: \n"
            f"  adata_nuclei_raw.obs_names[:5]: {adata_nuclei_raw.obs_names[:5].tolist()}\n"
            f"  adata.obs_names[:5]:            {adata.obs_names[:5].tolist()}")
            
    whole_counts = _to_df(adata)
    fiber_counts = (whole_counts - nuc_counts).clip(lower=0)

    # Copy obsm so centroid coordinates are preserved in both tables
    obsm = {"spatial": adata.obsm["spatial"].copy()} if "spatial" in adata.obsm else {}

    # Build nuclei AnnData with corrected region pointer
    nuc_attrs = copy.deepcopy(adata.uns.get("spatialdata_attrs", {}))
    nuc_attrs["region"] = "nucleus_boundaries_fixed"
    adata_nuclei = AnnData(
        X=sp.csr_matrix(nuc_counts.values),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm=obsm,
        uns={"spatialdata_attrs": nuc_attrs},
    )

    # Build fibers AnnData with corrected region pointer
    fib_attrs = copy.deepcopy(adata.uns.get("spatialdata_attrs", {}))
    fib_attrs["region"] = "cell_boundaries"
    adata_fibers = AnnData(
        X=sp.csr_matrix(fiber_counts.values),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm=obsm,
        uns={"spatialdata_attrs": fib_attrs},
    )

    adata_fibers, adata_nuclei = _add_qc_col(adata_nuclei, adata_fibers, nuc, sdata)

    # Store back into sdata so results are not lost if return values are not captured
    sdata.tables["adata_nuclei"] = adata_nuclei
    sdata.tables["adata_fibers"] = adata_fibers

    return adata_fibers, adata_nuclei


def _add_qc_col(adata_nuclei, adata_fibers, nuc, sdata):
    """
    Compute standard QC metrics and per-cell area for nuclear and fiber AnnData objects.

    Runs `scanpy.pp.calculate_qc_metrics` on both objects in place, then
    computes per-cell nucleus area (summed, in case a cell has more than
    one nucleus) and fiber area (cell area minus nucleus area, clipped at
    zero), and adds these as an `"area"` column to each object's `.obs`.

    Parameters
    ----------
    adata_nuclei : AnnData
        AnnData object containing nucleus-only transcript counts, as
        produced by `get_adata_nuclei`.
    adata_fibers : AnnData
        AnnData object containing fiber-only (non-nuclear) transcript
        counts, as produced by `get_adata_nuclei`.
    nuc : geopandas.GeoDataFrame
        Repaired, per-nucleus GeoDataFrame (with a `cell_id` column) as
        returned by `_repair_nucleus`, used to compute nucleus area.
    sdata : SpatialData
        The SpatialData object containing the `"cell_boundaries"` shapes
        layer, used to compute whole-cell area.

    Returns
    -------
    tuple[AnnData, AnnData]
        `(adata_fibers, adata_nuclei)`, each with `.obs["total_counts"]`,
        `.obs["n_genes_by_counts"]`, and `.obs["area"]` populated.
    """
    sc.pp.calculate_qc_metrics(adata_nuclei, inplace=True, percent_top=None, log1p=False)
    sc.pp.calculate_qc_metrics(adata_fibers, inplace=True, percent_top=None, log1p=False)

    nuc["area"] = nuc.geometry.area
    nucleus_area = nuc.groupby("cell_id")["area"].sum()
    gdf_cell = sdata.shapes["cell_boundaries"]
    cell_area = gdf_cell.geometry.area
    cell_area.index = gdf_cell["cell_id"] if "cell_id" in gdf_cell.columns else gdf_cell.index
    fiber_area = (cell_area - nucleus_area.reindex(cell_area.index).fillna(0)).clip(lower=0)

    adata_nuclei.obs["area"] = adata_nuclei.obs_names.map(nucleus_area)
    adata_fibers.obs["area"] = adata_fibers.obs_names.map(fiber_area)
    return adata_fibers, adata_nuclei


def _to_df(ad):
    """
    Convert an AnnData object's count matrix to a dense pandas DataFrame.

    Parameters
    ----------
    ad : AnnData
        AnnData object whose `.X` may be a sparse or dense matrix.

    Returns
    -------
    pandas.DataFrame
        Dense DataFrame of counts, indexed by `ad.obs_names` with columns
        `ad.var_names`.
    """
    X = ad.X.toarray() if sp.issparse(ad.X) else ad.X
    return pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)


def _repair_nucleus(sdata, shape="nucleus_boundaries_fixed"):
    """
    Repair invalid nucleus geometries and dissolve them into one polygon per cell.

    Fixes self-intersecting/invalid polygons (which otherwise raise a
    `GEOSException` on operations like `.dissolve()`), discards non-polygonal
    remnants, merges multiple nuclei per cell into a single (multi)polygon
    per `cell_id`, and registers the result as a new, valid SpatialData
    shapes element (`"nucleus_boundaries_by_cell"`), reusing the coordinate
    transformation from the source layer.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing the nucleus shapes layer to repair.
    shape : str, default "nucleus_boundaries_fixed"
        Name of the shapes layer in `sdata.shapes` to repair. Use
        `"nucleus_boundaries"` as a fallback if `STAIA.add_nuclei()` has
        not been run.

    Returns
    -------
    tuple[SpatialData, geopandas.GeoDataFrame]
        The updated `sdata` (now containing a `"nucleus_boundaries_by_cell"`
        shapes layer), and `nuc`, the repaired (but not yet dissolved)
        per-nucleus GeoDataFrame with a `cell_id` column.
    """
    nuc = sdata.shapes[shape].copy()
    nuc["geometry"] = nuc.geometry.apply(make_valid)
    nuc["geometry"] = nuc.geometry.apply(_to_polygonal)
    nuc = nuc[nuc.geometry.notna()].copy()
    nuc_by_cell = nuc.dissolve(by="cell_id")
    transformations = get_transformation(sdata.shapes[shape], get_all=True)
    sdata.shapes["nucleus_boundaries_by_cell"] = ShapesModel.parse(nuc_by_cell, transformations=transformations)
    return sdata, nuc


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