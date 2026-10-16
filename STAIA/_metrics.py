import spatialdata as sd
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData
import sys

def frac_transcripts(
    sdata: SpatialData,
    label_unassigned: str = "UNASSIGNED",
    tran : str = "transcripts"
) -> float:
    """
    Compute fraction fo transcripts allocated.

    This function calculates the percentage of transcripts allocated and 
    looks at the amount of transcripts which are unnasigned.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing Xenium output containing 
        sdata.points["transcripts"].
    label_unassigned : str, default = "UNASSIGNED"
        The label assigned to unassigned transcripts.
    tran : str, default = "transcripts"
        Column name showing the transcripts in sdata.
    
    Returns
    -------
    float
        Float containing information about the percentage of transcripts
        allocated.
    
    Notes
    -----
    This function computes the fraction of unassigned reads, while not 
    changing the original SpatialData object.

    Examples
    --------
    >>> frac_transcripts(sdata)
    92.34

    """
    if _errorhandler(sdata):
        return None, None
    # Get the percentage of transcripts allocated
    if tran in sdata.points:
        transcripts = sdata.points[tran]
    else:
        print(f"``{tran}`` is not found in sdata.points. Fractions \
              of transcripts can not be computed.")
        return

    # Get all assigned reads
    assigned = (transcripts.cell_id != label_unassigned)

    # Compute percentage
    perc_assigned_transcripts = assigned.sum().compute() / len(transcripts) * 100

    return "%.2f" % perc_assigned_transcripts


def staining_positive(
        sdata: SpatialData,
        channel: str = "DAPI",
        allocation: str = "cell"
 ) -> float:
    """
    Computes the percentage of staining positive pixels allocated.

    This function thresholds a morphology channel using Otsu's method and
    calculates the fraction of positive pixels that fall within a segmentation
    mask (e.g., cell boundaries).

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing:
        - sdata.images["morphology_focus"]
        - sdata.labels["cell_lables"]
    channel : str, default = "ATP1A1/CD45/E-Cadherin"
        Channel name in the morphology image used for staining detection.
    allocation : {"cell", "nucleus"}, default = "cell"
        Segmentation mask used for allocation:
        - "cell": use cell segmentation
        - "nucleus": use nuclear segmentation (if available)

    Returns
    -------
    float
        Percentage of staining-positive pixels allocated to the segmentation.

    Raises
    ------
    KeyError
        If required image or label layers are missing.
    ValueError
        If the specified channel is not present. 

    Notes
    -----
    Processing steps:
    1. Extract highest-resolution image (scale0)
    2. Apply Gaussian smoothing (sigma=1)
    3. Compute Otsu threshold
    4. Identify positive pixels
    5. Measure overlap with segmentation mask

    The SpatialData object is not modified.

    Examples
    --------
    >>> staining_positive(sdata, channel="CD45)
    87.12
    """
    if _errorhandler(sdata):
        return None, None
    img = sdata.images["morphology_focus"]
    cell_labels = sdata.labels["cell_labels"]

    # Get data at highest resolution
    img_level = img["scale0"].data_vars["image"]
    labels_level = cell_labels["scale0"].data_vars["image"]

    stain = img_level.sel(c=channel)
    stain_np = stain.values

    stain_smooth = gaussian_filter(stain_np, sigma=1)

    thres = threshold_otsu(stain_smooth)
    positive_mask = stain_smooth > thres

    # For cell allocation
    if allocation == "cell":
        seg_mask = labels_level.values > 0
    # For nuclear allocation
    elif allocation.startswith("nucl"):
        labels_level = cell_labels["scale0"].data_vars["image"]
        seg_mask = labels_level.values > 0
    else:
        print(f"'{allocation}' is not a valid input. Using default (cell allocation).")
        seg_mask = labels_level.values > 0

    # Get percentage
    positive_pixels = positive_mask.sum()
    allocated_positive_pixels = np.logical_and(positive_mask, seg_mask).sum()

    percentage_allocated = allocated_positive_pixels / positive_pixels * 100

    return "%.2f" % percentage_allocated


def _errorhandler(sdata):
    if type(sdata) != sd._core.spatialdata.SpatialData:
        print("ERROR: The input provided is not in SpatialData format, so not applicable for this function.")
        return True

def metrics(
        sdata: SpatialData
) -> float:
    """
    Compute basic spatial quality metrics.

    This function evaluates:
    1. Fraction of transcripts assigned to cells
    2. Fraction of staining-positive pixels allocated to segmentation

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing transcript points and morphology data

    Returns
    -------
    tuple of float
        (assigned_transcripts_percentage, allocated_staining_percentage)

    Notes
    -----
    This is a convenience wrapper around frac_transcripts() and 
    staining_positive()

    Examples
    --------
    >>> metrics(sdata)
    (92.3, 85.7)
    """
    if _errorhandler(sdata):
        return None, None
    perc_assigned_transcripts = frac_transcripts(sdata)
    staining_positives = staining_positive(sdata)
    return perc_assigned_transcripts, staining_positives
