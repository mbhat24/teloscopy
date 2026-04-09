"""Generate synthetic qFISH microscopy images for testing.

Creates realistic-looking metaphase spread images with:

* **DAPI channel** – elongated chromosome shapes with slight curvature,
  random orientations, Poisson-noise background.
* **Cy3 channel** – 2-D Gaussian telomere spots placed at each chromosome
  tip with variable intensity.
* **Ground truth** – labelled chromosome masks, spot coordinates, true
  intensities, and assigned base-pair lengths.

This module is designed to produce test data for the full qFISH telomere
length analysis pipeline without requiring real microscopy acquisitions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate as ndi_rotate

# ---------------------------------------------------------------------------
# Single-object generators
# ---------------------------------------------------------------------------


def generate_chromosome(
    shape: tuple[int, int] = (100, 30),
    intensity: float = 5000.0,
    noise_std: float = 200.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single synthetic chromosome image (DAPI-like).

    Creates an elongated bright shape on a dark background that mimics
    the appearance of a condensed metaphase chromosome stained with DAPI.
    Slight curvature and tapering are applied for realism.

    Args:
        shape: ``(height, width)`` of the output array.
        intensity: Peak fluorescence intensity of the chromosome body.
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        2-D ``float64`` array of the given *shape*.
    """
    h, w = shape
    img = np.zeros((h, w), dtype=np.float64)

    # Parametric chromosome body -------------------------------------------
    # x-profile: smooth elliptical cross-section with tapering at ends
    yy = np.linspace(-1, 1, h)
    xx = np.linspace(-1, 1, w)
    yg, xg = np.meshgrid(yy, xx, indexing="ij")

    # Longitudinal envelope (tapers at both ends)
    long_env = np.clip(1.0 - yg**6, 0, 1)  # super-Gaussian taper

    # Lateral envelope (elliptical cross-section that narrows at tips)
    width_factor = np.clip(1.0 - 0.3 * yg**2, 0.3, 1.0)  # narrower at tips
    lat_env = np.clip(1.0 - (xg / (0.35 * width_factor)) ** 4, 0, 1)

    body = long_env * lat_env

    # Add slight curvature: shift columns by a small sinusoidal offset
    curvature_amp = 0.08 * w
    for row_idx in range(h):
        shift = curvature_amp * np.sin(np.pi * row_idx / h)
        shift_int = int(round(shift))
        if shift_int != 0:
            img[row_idx, :] = np.roll(body[row_idx, :] * intensity, shift_int)
        else:
            img[row_idx, :] = body[row_idx, :] * intensity

    # Smooth and add noise
    img = gaussian_filter(img, sigma=1.5)
    if noise_std > 0:
        img += (rng or np.random.default_rng()).normal(0, noise_std, size=img.shape)
    img = np.clip(img, 0, None)
    return img


def generate_telomere_spot(
    sigma: float = 2.0,
    intensity: float = 10000.0,
    size: int = 20,
) -> np.ndarray:
    """Generate a single 2-D Gaussian telomere spot.

    Args:
        sigma: Standard deviation of the Gaussian (pixels).
        intensity: Peak (centre) intensity value.
        size: Side length of the square output array.

    Returns:
        2-D ``float64`` array of shape ``(size, size)``.
    """
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[:size, :size]
    r2 = (yy - centre) ** 2 + (xx - centre) ** 2
    spot = intensity * np.exp(-r2 / (2.0 * sigma**2))
    return spot


# ---------------------------------------------------------------------------
# Metaphase spread generator
# ---------------------------------------------------------------------------


def _place_stamp(
    canvas: np.ndarray,
    stamp: np.ndarray,
    cy: int,
    cx: int,
) -> None:
    """Additively place a *stamp* onto *canvas* centred at ``(cy, cx)``."""
    sh, sw = stamp.shape[:2]
    ch, cw = canvas.shape[:2]

    # Source (stamp) and destination (canvas) slicing
    sy0 = max(0, sh // 2 - cy)
    sy1 = min(sh, sh // 2 + (ch - cy))
    sx0 = max(0, sw // 2 - cx)
    sx1 = min(sw, sw // 2 + (cw - cx))

    dy0 = max(0, cy - sh // 2)
    dy1 = dy0 + (sy1 - sy0)
    dx0 = max(0, cx - sw // 2)
    dx1 = dx0 + (sx1 - sx0)

    if dy1 > dy0 and dx1 > dx0 and sy1 > sy0 and sx1 > sx0:
        canvas[dy0:dy1, dx0:dx1] += stamp[sy0:sy1, sx0:sx1]


def _place_stamp_max(
    canvas: np.ndarray,
    stamp: np.ndarray,
    cy: int,
    cx: int,
) -> None:
    """Place *stamp* onto *canvas* centred at ``(cy, cx)`` using max blending."""
    sh, sw = stamp.shape[:2]
    ch, cw = canvas.shape[:2]

    sy0 = max(0, sh // 2 - cy)
    sy1 = min(sh, sh // 2 + (ch - cy))
    sx0 = max(0, sw // 2 - cx)
    sx1 = min(sw, sw // 2 + (cw - cx))

    dy0 = max(0, cy - sh // 2)
    dy1 = dy0 + (sy1 - sy0)
    dx0 = max(0, cx - sw // 2)
    dx1 = dx0 + (sx1 - sx0)

    if dy1 > dy0 and dx1 > dx0 and sy1 > sy0 and sx1 > sx0:
        np.maximum(
            canvas[dy0:dy1, dx0:dx1],
            stamp[sy0:sy1, sx0:sx1],
            out=canvas[dy0:dy1, dx0:dx1],
        )


def _rotation_safe(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by *angle* degrees with zero-padded borders."""
    return ndi_rotate(image, angle, reshape=True, order=1, mode="constant", cval=0.0)


def _check_overlap(
    occupied: np.ndarray,
    label_mask: np.ndarray,
    cy: int,
    cx: int,
    max_overlap_fraction: float = 0.15,
) -> bool:
    """Return True if placing *label_mask* at ``(cy, cx)`` would overlap
    existing chromosomes by more than *max_overlap_fraction*."""
    sh, sw = label_mask.shape[:2]
    ch, cw = occupied.shape[:2]

    sy0 = max(0, sh // 2 - cy)
    sy1 = min(sh, sh // 2 + (ch - cy))
    sx0 = max(0, sw // 2 - cx)
    sx1 = min(sw, sw // 2 + (cw - cx))

    dy0 = max(0, cy - sh // 2)
    dy1 = dy0 + (sy1 - sy0)
    dx0 = max(0, cx - sw // 2)
    dx1 = dx0 + (sx1 - sx0)

    if dy1 <= dy0 or dx1 <= dx0 or sy1 <= sy0 or sx1 <= sx0:
        return True  # entirely outside -> treat as overlap

    stamp_region = label_mask[sy0:sy1, sx0:sx1] > 0
    canvas_region = occupied[dy0:dy1, dx0:dx1] > 0

    overlap_pixels = int(np.sum(stamp_region & canvas_region))
    stamp_pixels = int(np.sum(stamp_region))
    if stamp_pixels == 0:
        return True
    return (overlap_pixels / stamp_pixels) > max_overlap_fraction


def _paint_label(
    label_canvas: np.ndarray,
    label_mask: np.ndarray,
    cy: int,
    cx: int,
    label: int,
) -> None:
    """Write *label* into *label_canvas* where *label_mask* > 0."""
    sh, sw = label_mask.shape[:2]
    ch, cw = label_canvas.shape[:2]

    sy0 = max(0, sh // 2 - cy)
    sy1 = min(sh, sh // 2 + (ch - cy))
    sx0 = max(0, sw // 2 - cx)
    sx1 = min(sw, sw // 2 + (cw - cx))

    dy0 = max(0, cy - sh // 2)
    dy1 = dy0 + (sy1 - sy0)
    dx0 = max(0, cx - sw // 2)
    dx1 = dx0 + (sx1 - sx0)

    if dy1 > dy0 and dx1 > dx0 and sy1 > sy0 and sx1 > sx0:
        region = label_mask[sy0:sy1, sx0:sx1] > 0
        label_canvas[dy0:dy1, dx0:dx1][region] = label


def generate_metaphase_spread(
    image_size: tuple[int, int] = (1024, 1024),
    n_chromosomes: int = 46,
    telomere_intensity_range: tuple[float, float] = (3000.0, 30000.0),
    background_noise: float = 100.0,
    chromosome_intensity: float = 8000.0,
    telomere_sigma: float = 2.5,
    seed: int | None = None,
) -> dict:
    """Generate a complete synthetic metaphase spread with telomere spots.

    Produces two image channels (DAPI + Cy3) and full ground-truth
    annotations suitable for end-to-end pipeline testing.

    Args:
        image_size: ``(height, width)`` of the output images.
        n_chromosomes: Number of chromosomes to place.
        telomere_intensity_range: ``(min, max)`` peak intensity for telomere
            spots in the Cy3 channel.
        background_noise: Standard deviation of Poisson-like background
            noise added to both channels.
        chromosome_intensity: Mean peak intensity of chromosome bodies in
            the DAPI channel.
        telomere_sigma: Gaussian sigma for telomere spots.
        seed: Random seed for reproducibility.

    Returns:
        A dict with the following keys:

        - ``dapi`` -- ``uint16`` DAPI channel image.
        - ``cy3`` -- ``uint16`` Cy3 channel image.
        - ``ground_truth`` -- dict containing:

          - ``chromosomes`` -- list of dicts (``label``, ``center_y``,
            ``center_x``, ``angle``, ``length``, ``width``, ``tip_p``,
            ``tip_q``).
          - ``telomeres`` -- list of dicts (``y``, ``x``,
            ``chromosome_label``, ``arm``, ``intensity``, ``length_bp``).
          - ``labels`` -- ``int32`` labelled mask where each chromosome
            has a unique integer label.
    """
    rng = np.random.default_rng(seed)
    img_h, img_w = image_size

    dapi = np.zeros((img_h, img_w), dtype=np.float64)
    cy3 = np.zeros((img_h, img_w), dtype=np.float64)
    labels = np.zeros((img_h, img_w), dtype=np.int32)

    chrom_records: list[dict] = []
    telo_records: list[dict] = []

    margin = 80  # keep chromosomes away from image edge

    for idx in range(1, n_chromosomes + 1):
        # Randomise chromosome geometry
        chrom_length = int(rng.uniform(50, 110))
        chrom_width = int(rng.uniform(18, 32))
        angle = rng.uniform(0, 360)
        chrom_int = chromosome_intensity * rng.uniform(0.7, 1.3)

        # Generate and rotate the chromosome body
        body = generate_chromosome(
            shape=(chrom_length, chrom_width),
            intensity=chrom_int,
            noise_std=150.0,
            rng=rng,
        )
        body_rot = _rotation_safe(body, angle)

        # Binary mask for overlap checking and labelling
        body_mask = (body_rot > chrom_int * 0.1).astype(np.float64)

        # Try to place without excessive overlap (up to 40 attempts)
        placed = False
        for _attempt in range(40):
            cy = int(rng.uniform(margin, img_h - margin))
            cx = int(rng.uniform(margin, img_w - margin))
            if not _check_overlap(labels, body_mask, cy, cx, max_overlap_fraction=0.15):
                placed = True
                break

        if not placed:
            # Fall back: place anyway at last attempted position
            pass

        # Paint onto canvases
        _place_stamp_max(dapi, body_rot, cy, cx)
        _paint_label(labels, body_mask, cy, cx, idx)

        # Compute tip positions in image coordinates -----------------------
        # Before rotation the tips are top-centre and bottom-centre of the
        # chromosome body.  We rotate these relative vectors.
        half_len = chrom_length / 2.0
        angle_rad = np.deg2rad(angle)
        # scipy.ndimage.rotate rotates counter-clockwise in image coords.
        # The chromosome body has its long axis along y, so the tips are at
        # dy = +/- half_len, dx = 0 before rotation.
        dy_p = -half_len * np.cos(angle_rad)
        dx_p = half_len * np.sin(angle_rad)
        dy_q = half_len * np.cos(angle_rad)
        dx_q = -half_len * np.sin(angle_rad)

        tip_p = (cy + dy_p, cx + dx_p)
        tip_q = (cy + dy_q, cx + dx_q)

        chrom_records.append(
            {
                "label": idx,
                "center_y": cy,
                "center_x": cx,
                "angle": float(angle),
                "length": chrom_length,
                "width": chrom_width,
                "tip_p": (float(tip_p[0]), float(tip_p[1])),
                "tip_q": (float(tip_q[0]), float(tip_q[1])),
            }
        )

        # Place telomere spots at each tip ---------------------------------
        lo, hi = telomere_intensity_range
        for arm, tip in (("p", tip_p), ("q", tip_q)):
            telo_int = float(rng.uniform(lo, hi))
            # Simulate a correlation: brighter <-> longer telomere
            length_bp = max(
                200.0,
                500.0 + (telo_int - lo) / (hi - lo) * 20000.0 + rng.normal(0, 800),
            )

            spot = generate_telomere_spot(
                sigma=telomere_sigma,
                intensity=telo_int,
                size=int(6 * telomere_sigma + 1) | 1,  # ensure odd
            )
            ty, tx = int(round(tip[0])), int(round(tip[1]))
            _place_stamp(cy3, spot, ty, tx)

            telo_records.append(
                {
                    "y": float(tip[0]),
                    "x": float(tip[1]),
                    "chromosome_label": idx,
                    "arm": arm,
                    "intensity": telo_int,
                    "length_bp": float(length_bp),
                }
            )

    # Add background noise to both channels --------------------------------
    if background_noise > 0:
        dapi += rng.normal(0, background_noise, size=dapi.shape)
        cy3 += rng.normal(0, background_noise, size=cy3.shape)

    # Clip and convert to uint16
    dapi = np.clip(dapi, 0, 65535).astype(np.uint16)
    cy3 = np.clip(cy3, 0, 65535).astype(np.uint16)

    return {
        "dapi": dapi,
        "cy3": cy3,
        "ground_truth": {
            "chromosomes": chrom_records,
            "telomeres": telo_records,
            "labels": labels,
        },
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_synthetic_image(data: dict, path: str) -> None:
    """Save a synthetic image dict as a multi-channel TIFF.

    Requires ``tifffile`` to be installed.  Falls back to saving the two
    channels as separate ``.npy`` files if tifffile is unavailable.

    Args:
        data: Dict returned by :func:`generate_metaphase_spread`.
        path: Output file path (e.g. ``"metaphase_001.tif"``).
    """
    dapi = data["dapi"]
    cy3 = data["cy3"]

    # Stack as (C, H, W) for multi-channel TIFF
    multichannel = np.stack([dapi, cy3], axis=0)

    try:
        import tifffile

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            path,
            multichannel,
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "Cy3"]}},
        )
    except ImportError:
        warnings.warn(
            "tifffile not installed -- saving channels as .npy files instead.",
            stacklevel=2,
        )
        base = str(Path(path).with_suffix(""))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{base}_dapi.npy", dapi)
        np.save(f"{base}_cy3.npy", cy3)
        # Also save ground truth coordinates as npz
        gt = data.get("ground_truth", {})
        if "labels" in gt:
            np.save(f"{base}_labels.npy", gt["labels"])


def generate_test_dataset(
    output_dir: str,
    n_images: int = 5,
    seed: int = 42,
    **kwargs,
) -> list[str]:
    """Generate multiple synthetic metaphase spread images.

    Args:
        output_dir: Directory to write images into (created if needed).
        n_images: Number of images to generate.
        seed: Base random seed.  Image *i* uses ``seed + i``.
        **kwargs: Extra keyword arguments forwarded to
            :func:`generate_metaphase_spread`.

    Returns:
        List of file paths for the saved images.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for i in range(n_images):
        data = generate_metaphase_spread(seed=seed + i, **kwargs)
        fname = f"metaphase_{i:03d}.tif"
        fpath = str(out / fname)
        save_synthetic_image(data, fpath)
        saved_paths.append(fpath)

    return saved_paths
