from spatialdata import SpatialData
import spatialdata as sd

def _crop0(
        x: SpatialData,
        min_coord: list | None = None,
        max_coord: list | None = None
) -> SpatialData:
    """
    Crop a SpatialData object using a bounding box.

    Parameters
    ----------
    sdata : SpatialData
        Input spatial dataset.
    min_coord : list of float or None
        Minimum [x, y] coordinates.
    max_coord : list of float or None
        Maximum [x, y] coordinates.

    Returns
    -------
    SpatialData
        Cropped SpatialData view.

    Notes
    -----
    Coordinates are interpreted in the global coordinate system.
    Default region corresponds to a predefined area of interest
    
    """
    if min_coord is None and max_coord is None:
        return sd.bounding_box_query(
            x,
            min_coordinate=[36000, 72000],
            max_coordinate=[48000, 88000],
            axes=("x", "y"),
            target_coordinate_system="global",
        )
    elif min_coord is not None and max_coord is None:
        return sd.bounding_box_query(
            x,
            min_coordinate=min_coord,
            max_coordinate=[48000, 88000],
            axes=("x", "y"),
            target_coordinate_system="global",
        )
    elif min_coord is None and max_coord is not None:
        return sd.bounding_box_query(
            x,
            min_coordinate=[36000, 72000],
            max_coordinate=max_coord,
            axes=("x", "y"),
            target_coordinate_system="global",
        )
    else:
        return sd.bounding_box_query(
            x,
            min_coordinate=min_coord,
            max_coordinate=max_coord,
            axes=("x", "y"),
            target_coordinate_system="global",
        )    


def multimodal_segmentation(
        sdata: SpatialData,
        channelnames: list | None = None,
        img_layer: str = "morphology_focus"
) -> None:
    """
    Visualize morphology channels for multimodal segmentation inspection.

    Parameters
    ----------
    sdata : SpatialData
        Must contain ``sdata["morphology_focus"]``.
    channelnames : list of str or None
        Channels to display. If None, all channels are shown.
    img_layer : str, default = "morphology_focus"
        The image layer of the SpatialData object

    Notes
    -----
    Useful for evaluating segmentation quality across imaging channels.
    """
    if channelnames is None:
        channelnames = sd.models.get_channel_names(sdata[img_layer])
    
    for i in channelnames:
        sdata.pl.render_images(img_layer, channel=i).pl.show(
        figsize=(6, 6)
    )
        
    print("If you want to zoom in on specific slices, run multimodal_segmentation_slice.")


def multimodal_segmentation_slice(
        sdata: SpatialData,
        channelnames: list | None = None,
        min_coord: list | None = None,
        max_coord: list | None = None,
        img_layer: str = "morphology_focus"
) -> None:
    """
    Visualize selected morphology channels within a spatial region.

    Parameters
    ----------
    sdata : SpatialData
    channelnames : list of str or None
        Channels to display.
    min_coord : list or None
        Minimum [x, y] crop coordinates.
    max_coord : list or None
        Maximum [x, y] crop coordinates.
    img_layer : str, default = "morphology_focus"
        The image layer of the SpatialData object

    Notes
    -----
    Wrapper around bounding-box cropping followed by image rendering.
    """
    # Some redundant code, need to improve this.
    if channelnames is None:
        channelnames = sd.models.get_channel_names(sdata[img_layer])

    for i in channelnames: 
        _crop0(sdata, min_coord, max_coord).pl.render_images(img_layer, channel=i).pl.show(
            title=i, figsize=(10, 3)
        )
