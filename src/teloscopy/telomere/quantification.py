"""Telomere fluorescence intensity quantification and calibration.

Provides aperture-photometry-style intensity measurement for telomere spots
detected in qFISH Cy3 images, with local background subtraction using an
annular region around each spot.  Also includes a :class:`Calibration` class
for converting raw fluorescence units into telomere length (base pairs) via
linear or polynomial regression against reference standards.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Intensity measurement
# ---------------------------------------------------------------------------


def _make_circular_mask(radius: int, dtype: type = bool) -> np.ndarray:
    """Return a 2D boolean mask for a filled circle of the given *radius*.

    The mask is a square array of shape ``(2*radius + 1, 2*radius + 1)``
    centred at ``(radius, radius)``.
    """
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    dist2 = (yy - radius) ** 2 + (xx - radius) ** 2
    return (dist2 <= radius * radius).astype(dtype)


def _make_annular_mask(inner_radius: int, outer_radius: int, dtype: type = bool) -> np.ndarray:
    """Return a 2D boolean mask for an annulus (ring).

    The mask is a square array of shape
    ``(2*outer_radius + 1, 2*outer_radius + 1)`` centred at
    ``(outer_radius, outer_radius)``.
    """
    size = 2 * outer_radius + 1
    yy, xx = np.ogrid[:size, :size]
    dist2 = (yy - outer_radius) ** 2 + (xx - outer_radius) ** 2
    return ((dist2 >= inner_radius**2) & (dist2 <= outer_radius**2)).astype(dtype)


def measure_spot_intensity(
    image: np.ndarray,
    y: float,
    x: float,
    radius: int = 5,
    bg_inner: int = 7,
    bg_outer: int = 12,
) -> dict:
    """Measure telomere fluorescence with local background subtraction.

    Uses aperture photometry: a circular *signal* aperture and a surrounding
    annular region to estimate the local background level.

    Args:
        image: 2-D fluorescence image (typically the Cy3 channel).
            Can be any numeric dtype; internally promoted to ``float64``.
        y, x: Sub-pixel spot centre coordinates.
        radius: Radius (pixels) of the circular signal aperture.
        bg_inner: Inner radius of the background annulus.
        bg_outer: Outer radius of the background annulus.

    Returns:
        A dict containing:

        - ``raw_intensity`` (float): Sum of pixel values inside the signal
          aperture (*without* background correction).
        - ``corrected_intensity`` (float): Background-subtracted total flux
          (``signal - bg_per_pixel * signal_area``).  Clipped to ≥ 0.
        - ``background_level`` (float): Mean pixel value in the background
          annulus.
        - ``signal_area`` (int): Number of pixels inside the signal aperture
          (may be less than the full circle if near an edge).
        - ``snr`` (float): Signal-to-noise ratio, defined as
          ``corrected_intensity / (background_std * sqrt(signal_area))``.
          Set to 0.0 when background std is zero.
        - ``valid`` (bool): ``False`` when the signal aperture extends
          beyond the image boundary (the measurement is still returned but
          should be treated with caution).

    Raises:
        ValueError: If *bg_inner* < *radius* or *bg_outer* ≤ *bg_inner*.
    """
    if bg_inner < radius:
        raise ValueError(f"bg_inner ({bg_inner}) must be >= radius ({radius})")
    if bg_outer <= bg_inner:
        raise ValueError(f"bg_outer ({bg_outer}) must be > bg_inner ({bg_inner})")

    img = image.astype(np.float64)
    h, w = img.shape[:2]

    # Integer centre (nearest pixel)
    cy, cx = int(round(y)), int(round(x))

    # Determine the full patch extent required (centred on bg_outer)
    patch_half = bg_outer
    y0 = cy - patch_half
    y1 = cy + patch_half + 1
    x0 = cx - patch_half
    x1 = cx + patch_half + 1

    # Check whether the signal aperture stays inside the image
    valid = cy - radius >= 0 and cy + radius < h and cx - radius >= 0 and cx + radius < w

    # Clip to image bounds for extraction
    ey0 = max(y0, 0)
    ey1 = min(y1, h)
    ex0 = max(x0, 0)
    ex1 = min(x1, w)

    patch = img[ey0:ey1, ex0:ex1]

    # Build masks at full patch size and then crop to match
    full_size = 2 * patch_half + 1
    signal_mask_full = _make_circular_mask(radius, dtype=bool)
    bg_mask_full = _make_annular_mask(bg_inner, bg_outer, dtype=bool)

    # Embed the small signal mask in the big frame
    signal_embed = np.zeros((full_size, full_size), dtype=bool)
    offset = patch_half - radius
    signal_embed[
        offset : offset + signal_mask_full.shape[0],
        offset : offset + signal_mask_full.shape[1],
    ] = signal_mask_full

    # Crop both masks to the extracted patch region
    mask_y0 = ey0 - y0
    mask_y1 = mask_y0 + (ey1 - ey0)
    mask_x0 = ex0 - x0
    mask_x1 = mask_x0 + (ex1 - ex0)

    sig_mask = signal_embed[mask_y0:mask_y1, mask_x0:mask_x1]
    bg_mask = bg_mask_full[mask_y0:mask_y1, mask_x0:mask_x1]

    # -- Signal measurement ------------------------------------------------
    signal_pixels = patch[sig_mask]
    signal_area = int(sig_mask.sum())
    raw_intensity = float(signal_pixels.sum()) if signal_area > 0 else 0.0

    # -- Background measurement --------------------------------------------
    bg_pixels = patch[bg_mask]
    if bg_pixels.size > 0:
        background_level = float(bg_pixels.mean())
        background_std = float(bg_pixels.std(ddof=1)) if bg_pixels.size > 1 else 0.0
    else:
        background_level = 0.0
        background_std = 0.0

    # -- Corrected intensity -----------------------------------------------
    corrected_intensity = max(0.0, raw_intensity - background_level * signal_area)

    # -- SNR ---------------------------------------------------------------
    noise = background_std * np.sqrt(signal_area) if signal_area > 0 else 0.0
    snr = corrected_intensity / noise if noise > 0 else 0.0

    return {
        "raw_intensity": raw_intensity,
        "corrected_intensity": corrected_intensity,
        "background_level": background_level,
        "signal_area": signal_area,
        "snr": snr,
        "valid": valid,
    }


def quantify_all_spots(
    spots: list[dict],
    image: np.ndarray,
    radius: int = 5,
    bg_inner: int = 7,
    bg_outer: int = 12,
) -> list[dict]:
    """Measure intensity for every telomere spot in a list.

    Calls :func:`measure_spot_intensity` for each spot and merges the
    resulting keys (``raw_intensity``, ``corrected_intensity``, etc.)
    into the spot dict **in-place**.

    Args:
        spots: List of spot dicts, each containing at least ``y`` and ``x``.
        image: 2-D fluorescence image (Cy3 channel).
        radius: Signal aperture radius.
        bg_inner: Background annulus inner radius.
        bg_outer: Background annulus outer radius.

    Returns:
        The same *spots* list (modified in-place) for convenience.
    """
    for spot in spots:
        measurement = measure_spot_intensity(
            image,
            y=spot["y"],
            x=spot["x"],
            radius=radius,
            bg_inner=bg_inner,
            bg_outer=bg_outer,
        )
        spot.update(measurement)
    return spots


# ---------------------------------------------------------------------------
# Calibration: fluorescence → base pairs
# ---------------------------------------------------------------------------


class Calibration:
    """Convert fluorescence intensity to telomere length in base pairs.

    Fits a calibration curve (linear or polynomial) from reference data
    with known telomere lengths.  Once fitted, the :meth:`predict` and
    :meth:`predict_batch` methods convert arbitrary intensity values to
    estimated base-pair lengths.

    Args:
        method: Fitting method.  ``"linear"`` performs ordinary least-squares
            linear regression (degree 1).  ``"poly2"`` / ``"poly3"`` fit a
            2nd / 3rd degree polynomial respectively.  The string
            ``"polyN"`` for any integer *N* is also accepted.

    Example::

        cal = Calibration(method="linear")
        cal.fit(
            intensities=[1000, 5000, 10000, 20000],
            lengths_bp=[2000, 6000, 11000, 21000],
        )
        print(cal.predict(8000))   # estimated bp length
    """

    _VALID_METHODS = {"linear", "poly2", "poly3"}

    def __init__(self, method: str = "linear") -> None:
        self.method = method
        self.coefficients: np.ndarray | None = None
        self.fitted: bool = False
        self._degree: int = self._parse_degree(method)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _parse_degree(method: str) -> int:
        if method == "linear":
            return 1
        if method.startswith("poly"):
            try:
                return int(method[4:])
            except ValueError:
                pass
        raise ValueError(
            f"Unknown calibration method '{method}'. Use 'linear' or 'polyN' (e.g. 'poly2')."
        )

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        intensities: Sequence[float],
        lengths_bp: Sequence[float],
    ) -> Calibration:
        """Fit the calibration curve from reference data.

        Args:
            intensities: Measured fluorescence intensities for the
                reference standards.
            lengths_bp: Known telomere lengths (base pairs) corresponding
                to each intensity value.

        Returns:
            ``self`` (for method-chaining).

        Raises:
            ValueError: If fewer data points than the polynomial degree + 1
                are provided.
        """
        x = np.asarray(intensities, dtype=np.float64)
        y_bp = np.asarray(lengths_bp, dtype=np.float64)

        if len(x) < self._degree + 1:
            raise ValueError(
                f"Need at least {self._degree + 1} data points for a "
                f"degree-{self._degree} fit; got {len(x)}."
            )

        self.coefficients = np.polyfit(x, y_bp, self._degree)
        self.fitted = True
        return self

    # -- prediction --------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.fitted or self.coefficients is None:
            raise RuntimeError(
                "Calibration has not been fitted yet.  "
                "Call .fit() or use a class constructor first."
            )

    def predict(self, intensity: float) -> float:
        """Convert a single intensity value to estimated base-pair length.

        Args:
            intensity: Background-corrected fluorescence intensity.

        Returns:
            Estimated telomere length in base pairs.  The result is clipped
            to a minimum of 0.
        """
        self._check_fitted()
        value = float(np.polyval(self.coefficients, intensity))
        return max(0.0, value)

    def predict_batch(self, intensities: Sequence[float]) -> list[float]:
        """Convert a sequence of intensity values to base-pair lengths.

        Args:
            intensities: Iterable of fluorescence intensity values.

        Returns:
            List of estimated telomere lengths (clipped to ≥ 0).
        """
        self._check_fitted()
        arr = np.asarray(intensities, dtype=np.float64)
        values = np.polyval(self.coefficients, arr)
        return list(np.clip(values, 0.0, None).astype(np.float64))

    # -- alternative constructors ------------------------------------------

    @classmethod
    def from_references(
        cls,
        references: list[dict],
        method: str = "linear",
    ) -> Calibration:
        """Create and fit a calibration from a list of reference dicts.

        Args:
            references: Each dict must contain ``'intensity'`` (float) and
                ``'length_bp'`` (float).
            method: Calibration method (see :class:`Calibration`).

        Returns:
            A fitted :class:`Calibration` instance.
        """
        intensities = [r["intensity"] for r in references]
        lengths_bp = [r["length_bp"] for r in references]
        return cls(method=method).fit(intensities, lengths_bp)

    @classmethod
    def identity(cls) -> Calibration:
        """Create a no-op calibration that returns raw intensity as 'length'.

        Useful as a placeholder when no calibration data is available.
        The mapping is simply ``length_bp = intensity`` (slope=1, intercept=0).
        """
        cal = cls(method="linear")
        cal.coefficients = np.array([1.0, 0.0])
        cal.fitted = True
        return cal

    # -- serialisation helpers ---------------------------------------------

    def to_dict(self) -> dict:
        """Serialise calibration state to a plain dict."""
        return {
            "method": self.method,
            "coefficients": self.coefficients.tolist() if self.coefficients is not None else None,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        """Restore a :class:`Calibration` from a dict produced by :meth:`to_dict`."""
        cal = cls(method=d["method"])
        if d.get("coefficients") is not None:
            cal.coefficients = np.array(d["coefficients"], dtype=np.float64)
            cal.fitted = d.get("fitted", True)
        return cal

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        status = "fitted" if self.fitted else "unfitted"
        return f"<Calibration method={self.method!r} {status}>"
