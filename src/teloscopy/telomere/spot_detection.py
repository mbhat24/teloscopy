"""Telomere spot detection from fluorescence channel (Cy3 / FITC).

Detects sub-resolution fluorescent spots corresponding to telomeric FISH
signals using scale-space blob detectors (Laplacian of Gaussian, Difference
of Gaussians, Determinant of Hessian) and provides intensity / size-based
filtering utilities.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from skimage.feature import blob_dog, blob_doh, blob_log

# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------


def _normalise_to_float(image: np.ndarray) -> np.ndarray:
    """Normalise any numeric image to ``float64`` in ``[0, 1]``.

    Handles uint8, uint16, other integer types, and floating-point images.
    A constant image (max == min) is mapped to all zeros.
    """
    img = image.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo == 0:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Low-level blob detectors
# ---------------------------------------------------------------------------


def detect_spots_log(
    image: np.ndarray,
    min_sigma: float = 1.0,
    max_sigma: float = 5.0,
    num_sigma: int = 10,
    threshold: float = 0.05,
) -> np.ndarray:
    """Detect telomere spots using the Laplacian of Gaussian (LoG).

    This is the most accurate of the three scale-space detectors but also
    the slowest.

    Parameters
    ----------
    image : np.ndarray
        2-D single-channel fluorescence image (any numeric dtype).
    min_sigma, max_sigma : float
        Range of Gaussian standard deviations to search.
    num_sigma : int
        Number of intermediate sigma values.
    threshold : float
        Minimum LoG response to accept a blob.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` array where each row is ``[y, x, sigma]``.
    """
    normed = _normalise_to_float(image)
    blobs = blob_log(
        normed,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )
    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return blobs.astype(np.float64)


def detect_spots_dog(
    image: np.ndarray,
    min_sigma: float = 1.0,
    max_sigma: float = 5.0,
    threshold: float = 0.05,
) -> np.ndarray:
    """Detect telomere spots using the Difference of Gaussians (DoG).

    A faster approximation to the LoG detector.

    Parameters
    ----------
    image : np.ndarray
        2-D single-channel fluorescence image.
    min_sigma, max_sigma : float
        Range of Gaussian standard deviations.
    threshold : float
        Minimum DoG response to accept a blob.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` array: ``[y, x, sigma]`` per detected spot.
    """
    normed = _normalise_to_float(image)
    blobs = blob_dog(
        normed,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
    )
    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return blobs.astype(np.float64)


def detect_spots_doh(
    image: np.ndarray,
    min_sigma: float = 1.0,
    max_sigma: float = 5.0,
    threshold: float = 0.01,
) -> np.ndarray:
    """Detect telomere spots using the Determinant of Hessian (DoH).

    The fastest of the three detectors, but slightly less accurate for
    overlapping or elongated blobs.

    Parameters
    ----------
    image : np.ndarray
        2-D single-channel fluorescence image.
    min_sigma, max_sigma : float
        Range of Gaussian standard deviations.
    threshold : float
        Minimum DoH response to accept a blob.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` array: ``[y, x, sigma]`` per detected spot.
    """
    normed = _normalise_to_float(image)
    blobs = blob_doh(
        normed,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
    )
    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return blobs.astype(np.float64)


# ---------------------------------------------------------------------------
# Dispatcher / main entry point
# ---------------------------------------------------------------------------

_DETECTORS: dict[str, Any] = {
    "blob_log": detect_spots_log,
    "blob_dog": detect_spots_dog,
    "blob_doh": detect_spots_doh,
}

_SQRT2 = math.sqrt(2.0)


def _peak_intensity_at(
    image: np.ndarray,
    y: float,
    x: float,
    window: int = 1,
) -> float:
    """Return the maximum pixel value in a small window around ``(y, x)``.

    The window is a ``(2*window+1) × (2*window+1)`` square centred on the
    nearest integer coordinates.  Values outside image bounds are ignored.
    """
    iy, ix = int(round(y)), int(round(x))
    h, w = image.shape[:2]
    r0 = max(iy - window, 0)
    r1 = min(iy + window + 1, h)
    c0 = max(ix - window, 0)
    c1 = min(ix + window + 1, w)
    patch = image[r0:r1, c0:c1]
    if patch.size == 0:
        return 0.0
    return float(np.max(patch))


def detect_spots(
    image: np.ndarray,
    method: str = "blob_log",
    min_intensity: float = 0,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Detect telomere spots in a fluorescence image.

    This is the main entry point that dispatches to one of the low-level
    blob detectors, converts raw blob arrays to rich dictionaries, and
    optionally filters by minimum peak intensity.

    Parameters
    ----------
    image : np.ndarray
        2-D single-channel fluorescence image.
    method : str, optional
        Detection algorithm.  One of ``"blob_log"`` (default),
        ``"blob_dog"``, or ``"blob_doh"``.
    min_intensity : float, optional
        Discard spots whose peak pixel value (in the *original* image)
        is below this threshold.  Defaults to 0 (no filtering).
    **kwargs
        Additional keyword arguments forwarded to the chosen detector.

    Returns
    -------
    list[dict[str, float]]
        One dictionary per detected spot with keys:

        * **y** – row coordinate of the spot centre.
        * **x** – column coordinate of the spot centre.
        * **sigma** – fitted Gaussian sigma (scale parameter).
        * **radius** – approximate spot radius (``sigma × √2``).
        * **peak_intensity** – maximum pixel value at the spot centre
          (measured on the *original*, unnormalised image).

    Raises
    ------
    ValueError
        If *method* is not one of the supported detector names.
    """
    if method not in _DETECTORS:
        raise ValueError(f"Unknown detection method '{method}'. Available: {sorted(_DETECTORS)}")

    detector_fn = _DETECTORS[method]
    blobs = detector_fn(image, **kwargs)

    # Convert to float64 for intensity look-up on the original image
    img_f = image.astype(np.float64)

    spots: list[dict[str, float]] = []
    for row in blobs:
        y, x, sigma = float(row[0]), float(row[1]), float(row[2])
        peak = _peak_intensity_at(img_f, y, x)

        if peak < min_intensity:
            continue

        spots.append(
            {
                "y": y,
                "x": x,
                "sigma": sigma,
                "radius": sigma * _SQRT2,
                "peak_intensity": peak,
            }
        )

    return spots


# ---------------------------------------------------------------------------
# Post-detection filtering
# ---------------------------------------------------------------------------


def filter_spots(
    spots: list[dict[str, float]],
    image: np.ndarray,
    min_intensity: float = 0,
    max_radius: float = 20.0,
) -> list[dict[str, float]]:
    """Filter previously detected spots by intensity and size.

    Parameters
    ----------
    spots : list[dict[str, float]]
        Spot dictionaries as returned by :func:`detect_spots`.
    image : np.ndarray
        Original fluorescence image (used to re-measure peak intensity if
        the ``peak_intensity`` key is missing from a spot dict).
    min_intensity : float, optional
        Discard spots with peak intensity below this value.
    max_radius : float, optional
        Discard spots with radius exceeding this value (pixels).

    Returns
    -------
    list[dict[str, float]]
        Filtered list of spot dictionaries.
    """
    img_f = image.astype(np.float64)
    filtered: list[dict[str, float]] = []

    for spot in spots:
        # Re-measure intensity if not present
        peak = spot.get("peak_intensity")
        if peak is None:
            peak = _peak_intensity_at(img_f, spot["y"], spot["x"])

        radius = spot.get("radius", spot.get("sigma", 0.0) * _SQRT2)

        if peak < min_intensity:
            continue
        if radius > max_radius:
            continue

        # Ensure consistent keys in output
        filtered.append(
            {
                "y": spot["y"],
                "x": spot["x"],
                "sigma": spot["sigma"],
                "radius": radius,
                "peak_intensity": peak,
            }
        )

    return filtered
