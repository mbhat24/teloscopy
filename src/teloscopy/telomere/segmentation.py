"""Chromosome / nucleus segmentation from DAPI fluorescence channel.

Provides classical (Otsu + watershed) and deep-learning (Cellpose) based
segmentation of chromosomes or nuclei in DAPI-stained qFISH images, along
with morphometric property extraction including chromosome tip localisation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.spatial.distance import pdist, squareform
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

# ---------------------------------------------------------------------------
# Otsu + Watershed segmentation
# ---------------------------------------------------------------------------


def segment_otsu_watershed(
    dapi: np.ndarray,
    min_area: int = 500,
    min_distance: int = 20,
) -> np.ndarray:
    """Segment chromosomes / nuclei using Otsu thresholding and watershed.

    Algorithm
    ---------
    1. Compute a global Otsu threshold on the DAPI image.
    2. Binarise, fill holes, and remove objects smaller than *min_area*.
    3. Compute the Euclidean distance transform of the binary mask.
    4. Find peaks in the distance map (``peak_local_max``) separated by at
       least *min_distance* pixels.
    5. Marker-controlled watershed on the inverted distance transform.

    Parameters
    ----------
    dapi : np.ndarray
        2-D DAPI fluorescence image (any numeric dtype).
    min_area : int, optional
        Minimum object area in pixels; smaller objects are discarded.
    min_distance : int, optional
        Minimum distance between watershed seeds (pixels).

    Returns
    -------
    np.ndarray
        Integer-labelled image where 0 = background and 1, 2, 3, …
        correspond to individual chromosomes / nuclei.
    """
    img = dapi.astype(np.float64)

    # 1. Otsu threshold
    thresh = threshold_otsu(img)
    binary = img > thresh

    # 2. Fill holes and remove small objects
    binary = binary_fill_holes(binary)
    # remove_small_objects expects a labelled or boolean array
    binary = remove_small_objects(binary.astype(bool), min_size=min_area)

    # 3. Distance transform
    distance = distance_transform_edt(binary)

    # 4. Peak detection for watershed seeds
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary.astype(int),
    )

    # Build marker image from peak coordinates
    markers = np.zeros_like(binary, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    # 5. Watershed on inverted distance (ridges between objects)
    labels = watershed(-distance, markers, mask=binary)

    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Cellpose segmentation
# ---------------------------------------------------------------------------


def segment_cellpose(
    dapi: np.ndarray,
    model_type: str = "nuclei",
    diameter: int = 60,
) -> np.ndarray:
    """Segment chromosomes / nuclei using the Cellpose deep-learning model.

    If ``cellpose`` is not installed the function emits a warning and falls
    back to :func:`segment_otsu_watershed`.

    Parameters
    ----------
    dapi : np.ndarray
        2-D DAPI fluorescence image.
    model_type : str, optional
        Cellpose model type (``"nuclei"``, ``"cyto"``, …).
    diameter : int, optional
        Expected object diameter in pixels.

    Returns
    -------
    np.ndarray
        Integer-labelled segmentation image.
    """
    try:
        from cellpose import models as cp_models  # type: ignore[import-untyped]
    except ImportError:
        warnings.warn(
            "cellpose is not installed – falling back to Otsu + watershed "
            "segmentation.  Install cellpose for deep-learning segmentation: "
            "pip install cellpose",
            stacklevel=2,
        )
        return segment_otsu_watershed(dapi)

    model = cp_models.Cellpose(model_type=model_type, gpu=False)
    masks, _flows, _styles, _diams = model.eval(
        dapi.astype(np.float32),
        diameter=diameter,
        channels=[0, 0],  # grayscale
    )
    return np.asarray(masks, dtype=np.int32)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SEGMENTERS = {
    "otsu_watershed": segment_otsu_watershed,
    "cellpose": segment_cellpose,
}


def segment(
    dapi: np.ndarray,
    method: str = "otsu_watershed",
    **kwargs: Any,
) -> np.ndarray:
    """Segment chromosomes / nuclei from a DAPI channel image.

    Parameters
    ----------
    dapi : np.ndarray
        2-D DAPI fluorescence image.
    method : str, optional
        Segmentation algorithm name.  One of ``"otsu_watershed"`` or
        ``"cellpose"``.
    **kwargs
        Forwarded to the chosen segmentation function.

    Returns
    -------
    np.ndarray
        Integer-labelled segmentation image.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method not in _SEGMENTERS:
        raise ValueError(
            f"Unknown segmentation method '{method}'. Available: {sorted(_SEGMENTERS)}"
        )
    return _SEGMENTERS[method](dapi, **kwargs)


# ---------------------------------------------------------------------------
# Morphometric property extraction
# ---------------------------------------------------------------------------


def _find_tips(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two most distant points (chromosome tips) in a point set.

    For large regions we subsample via the convex hull to keep runtime
    manageable – full pairwise distance computation is O(N²) and can be
    prohibitive when a region contains tens of thousands of pixels.

    Parameters
    ----------
    coords : np.ndarray
        ``(N, 2)`` array of (row, col) pixel coordinates.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        Two (row, col) tuples representing the p-arm and q-arm tips.
    """
    if len(coords) <= 2:
        tip_p = tuple(int(v) for v in coords[0])
        tip_q = tuple(int(v) for v in coords[-1])
        return tip_p, tip_q  # type: ignore[return-value]

    # Use convex hull to reduce point count for distance computation
    try:
        from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices]
    except Exception:
        # Degenerate cases (e.g. collinear points) – use raw coords
        hull_pts = coords

    # Fall back to a subsample if hull is still very large
    if len(hull_pts) > 500:
        indices = np.linspace(0, len(hull_pts) - 1, 500, dtype=int)
        hull_pts = hull_pts[indices]

    dists = squareform(pdist(hull_pts))
    i, j = np.unravel_index(dists.argmax(), dists.shape)

    tip_p = (int(hull_pts[i, 0]), int(hull_pts[i, 1]))
    tip_q = (int(hull_pts[j, 0]), int(hull_pts[j, 1]))
    return tip_p, tip_q


def get_chromosome_properties(
    labels: np.ndarray,
    dapi: np.ndarray,
) -> list[dict[str, Any]]:
    """Extract morphometric properties for each segmented chromosome.

    Parameters
    ----------
    labels : np.ndarray
        Integer-labelled segmentation mask (0 = background).
    dapi : np.ndarray
        Original DAPI fluorescence image (used for intensity statistics).

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per labelled region with the following keys:

        * **label** (*int*) – region label.
        * **centroid** (*tuple[float, float]*) – ``(y, x)`` centroid.
        * **area** (*int*) – region area in pixels.
        * **bbox** (*tuple[int, int, int, int]*) –
          ``(min_row, min_col, max_row, max_col)``.
        * **major_axis_length** (*float*) – length of the fitted ellipse
          major axis.
        * **minor_axis_length** (*float*) – length of the fitted ellipse
          minor axis.
        * **orientation** (*float*) – orientation angle in radians.
        * **tip_p** (*tuple[int, int]*) – ``(y, x)`` of one chromosome tip.
        * **tip_q** (*tuple[int, int]*) – ``(y, x)`` of the other tip.
        * **mean_intensity** (*float*) – mean DAPI intensity within the
          region.
    """
    props = regionprops(labels, intensity_image=dapi.astype(np.float64))

    results: list[dict[str, Any]] = []
    for rp in props:
        coords = rp.coords  # (N, 2) array of (row, col) positions
        tip_p, tip_q = _find_tips(coords)

        results.append(
            {
                "label": int(rp.label),
                "centroid": (float(rp.centroid[0]), float(rp.centroid[1])),
                "area": int(rp.area),
                "bbox": tuple(int(v) for v in rp.bbox),
                "major_axis_length": float(rp.major_axis_length),
                "minor_axis_length": float(rp.minor_axis_length),
                "orientation": float(rp.orientation),
                "tip_p": tip_p,
                "tip_q": tip_q,
                "mean_intensity": float(rp.mean_intensity),
            }
        )

    return results
