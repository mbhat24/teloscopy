"""Image preprocessing for qFISH telomere microscopy images.

Provides functions for loading multi-channel fluorescence microscopy images,
background subtraction, denoising, and a combined preprocessing pipeline
suitable for quantitative FISH (qFISH) telomere length analysis.

Supported formats:
    - TIFF / multi-page TIFF (via ``tifffile``)
    - PNG, JPG, BMP and other common raster formats (via ``cv2``)

Channel conventions:
    The pipeline expects two fluorescence channels:
        * **DAPI** – nuclear / chromosome counterstain (blue)
        * **Cy3**  – telomere probe signal (red / orange)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, opening, white_tophat

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _normalise_to_uint16(arr: np.ndarray) -> np.ndarray:
    """Cast an array to uint16, rescaling if necessary."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return arr.astype(np.uint16) * np.uint16(257)  # 0-255 -> 0-65535
    if np.issubdtype(arr.dtype, np.floating):
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo == 0:
            return np.zeros_like(arr, dtype=np.uint16)
        scaled = (arr - lo) / (hi - lo) * 65535.0
        return scaled.astype(np.uint16)
    # Fallback for other integer types
    info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
    if info is not None:
        scaled = (arr.astype(np.float64) - info.min) / (info.max - info.min) * 65535.0
        return scaled.astype(np.uint16)
    return arr.astype(np.uint16)


def _load_tiff(path: str) -> np.ndarray:
    """Load a TIFF file using ``tifffile``."""
    import tifffile  # type: ignore[import-untyped]

    return tifffile.imread(path)


def _load_cv2(path: str) -> np.ndarray:
    """Load an image using OpenCV, preserving bit-depth."""
    import cv2  # type: ignore[import-untyped]

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read image: {path}")
    # OpenCV loads colour images as BGR; convert to RGB
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
    return img


def load_image(path: str) -> dict[str, np.ndarray]:
    """Load a multi-channel microscopy image.

    Supports TIFF (via ``tifffile``), PNG / JPG and other common formats
    (via ``cv2``).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``'dapi'`` and ``'cy3'``, each mapping to a
        2-D ``uint16`` array representing one fluorescence channel.

    Channel auto-detection
    ----------------------
    * Shape ``(C, Y, X)`` → channel 0 = DAPI, channel 1 = Cy3.
    * Shape ``(Y, X, C)`` → same mapping after transpose.
    * Shape ``(Y, X)`` (grayscale) → returned as both DAPI and Cy3.
    * RGB images (3 colour channels, last axis) → blue → DAPI, red → Cy3.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or could not be read.
    ValueError
        If the image shape cannot be interpreted.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = p.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        raw = _load_tiff(path)
    else:
        raw = _load_cv2(path)

    # --- 2-D grayscale -------------------------------------------------
    if raw.ndim == 2:
        ch = _normalise_to_uint16(raw)
        return {"dapi": ch.copy(), "cy3": ch.copy()}

    # --- 3-D array: determine axis layout ------------------------------
    if raw.ndim != 3:
        raise ValueError(f"Unexpected image dimensions ({raw.ndim}). Expected 2-D or 3-D.")

    # Heuristic: if last axis is small (≤4) → (Y, X, C); otherwise if
    # first axis is small → (C, Y, X).
    if raw.shape[2] <= 4:
        # (Y, X, C)
        c_axis = 2
    elif raw.shape[0] <= 4:
        # (C, Y, X) → move channel axis to the end for uniform handling
        raw = np.moveaxis(raw, 0, -1)
        c_axis = 2
    else:
        raise ValueError(f"Cannot determine channel axis for shape {raw.shape}.")

    n_channels = raw.shape[c_axis]

    if n_channels == 1:
        ch = _normalise_to_uint16(raw[..., 0])
        return {"dapi": ch.copy(), "cy3": ch.copy()}

    if n_channels == 2:
        dapi = _normalise_to_uint16(raw[..., 0])
        cy3 = _normalise_to_uint16(raw[..., 1])
        return {"dapi": dapi, "cy3": cy3}

    if n_channels in (3, 4):
        # Treat as RGB(A): blue → DAPI, red → Cy3
        dapi = _normalise_to_uint16(raw[..., 2])  # blue channel
        cy3 = _normalise_to_uint16(raw[..., 0])  # red channel
        return {"dapi": dapi, "cy3": cy3}

    raise ValueError(
        f"Unsupported number of channels ({n_channels}) in image of shape {raw.shape}."
    )


# ---------------------------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------------------------


def subtract_background(
    image: np.ndarray,
    method: str = "rolling_ball",
    radius: int = 50,
) -> np.ndarray:
    """Subtract background from a single-channel image.

    Parameters
    ----------
    image : np.ndarray
        2-D single-channel image (any numeric dtype).
    method : str, optional
        Background estimation method.

        ``"rolling_ball"``
            Uses :func:`skimage.restoration.rolling_ball` when available;
            otherwise falls back to morphological opening with ``disk(radius)``
            as a structuring element.
        ``"tophat"``
            White top-hat transform with ``disk(radius)``.
        ``"gaussian"``
            Subtracts a Gaussian-blurred copy of the image
            (``sigma = radius``).

    radius : int, optional
        Size parameter (pixels) controlling background estimation scale.

    Returns
    -------
    np.ndarray
        Background-subtracted image with the same dtype as *image*.
    """
    original_dtype = image.dtype

    if method == "rolling_ball":
        try:
            from skimage.restoration import (
                rolling_ball as _rolling_ball,  # type: ignore[import-untyped]
            )

            background = _rolling_ball(image.astype(np.float64), radius=radius)
            result = image.astype(np.float64) - background
        except (ImportError, AttributeError):
            # Fallback: morphological opening
            selem = disk(radius)
            img_f = image.astype(np.float64)
            background = opening(img_f, selem)
            result = img_f - background

    elif method == "tophat":
        selem = disk(radius)
        result = white_tophat(image.astype(np.float64), selem)

    elif method == "gaussian":
        img_f = image.astype(np.float64)
        background = gaussian_filter(img_f, sigma=float(radius))
        result = img_f - background

    else:
        raise ValueError(
            f"Unknown background subtraction method '{method}'. "
            "Choose from 'rolling_ball', 'tophat', 'gaussian'."
        )

    # Clip negatives and restore original dtype
    np.clip(result, 0, None, out=result)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        np.clip(result, info.min, info.max, out=result)

    return result.astype(original_dtype)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


def denoise(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian denoising to a single-channel image.

    Parameters
    ----------
    image : np.ndarray
        2-D single-channel image.
    sigma : float, optional
        Standard deviation of the Gaussian kernel (pixels).

    Returns
    -------
    np.ndarray
        Smoothed image as ``float64``.
    """
    return gaussian_filter(image.astype(np.float64), sigma=sigma)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def preprocess(
    path: str,
    bg_method: str = "rolling_ball",
    bg_radius: int = 50,
    denoise_sigma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Full preprocessing pipeline for a qFISH image.

    Sequentially performs:

    1. **Load** the image (multi-channel TIFF or standard raster format).
    2. **Background subtraction** on each channel independently.
    3. **Gaussian denoising** on each channel.

    Parameters
    ----------
    path : str
        Filesystem path to the microscopy image.
    bg_method : str, optional
        Background subtraction algorithm (see :func:`subtract_background`).
    bg_radius : int, optional
        Radius parameter for background subtraction.
    denoise_sigma : float, optional
        Gaussian sigma for denoising.

    Returns
    -------
    dict[str, np.ndarray]
        ``{'dapi': ..., 'cy3': ...}`` with preprocessed ``float64`` arrays.
    """
    channels = load_image(path)

    result: dict[str, np.ndarray] = {}
    for name, channel in channels.items():
        bg_sub = subtract_background(channel, method=bg_method, radius=bg_radius)
        smoothed = denoise(bg_sub, sigma=denoise_sigma)
        result[name] = smoothed

    return result
