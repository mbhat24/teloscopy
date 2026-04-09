"""End-to-end telomere analysis pipeline.

Orchestrates every stage of the qFISH analysis — from raw image loading
through to final statistics — and exposes a batch-processing helper that
iterates over a directory of images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    # --- image loading ---
    "dapi_channel": 0,
    "cy3_channel": 1,
    # --- preprocessing ---
    "background_method": "rolling_ball",
    "rolling_ball_radius": 50,
    "denoise_sigma": 1.0,
    # --- segmentation ---
    "segmentation_method": "otsu_watershed",
    "min_chromosome_area": 80,
    "max_chromosome_area": 8000,
    # --- spot detection ---
    "spot_sigma_min": 1.5,
    "spot_sigma_max": 4.0,
    "spot_threshold": 0.02,
    "spot_min_snr": 3.0,
    # --- association ---
    "max_tip_distance": 15.0,
    # --- quantification ---
    "annulus_inner": 5,
    "annulus_outer": 8,
    # --- calibration ---
    "calibration_slope": None,
    "calibration_intercept": None,
}


def get_default_config() -> dict[str, Any]:
    """Return a copy of the default pipeline configuration.

    Returns
    -------
    dict
        Mutable copy so callers can override individual keys.
    """
    return dict(_DEFAULT_CONFIG)


def load_config(path: str) -> dict[str, Any]:
    """Load pipeline configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    RuntimeError
        If ``pyyaml`` is not installed or the file cannot be parsed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load config files. Install with: pip install pyyaml"
        ) from exc

    with open(path) as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")

    return data


# ---------------------------------------------------------------------------
# Internal stage helpers
# ---------------------------------------------------------------------------


def _load_image(image_path: str, cfg: dict) -> dict[str, np.ndarray]:
    """Load a multi-channel TIFF and return individual channel arrays.

    Tries ``tifffile`` first, then falls back to ``skimage.io.imread``.
    """
    try:
        import tifffile

        img = tifffile.imread(image_path)
    except ImportError:
        from skimage.io import imread

        img = imread(image_path)

    if img.ndim == 2:
        # Single-channel: treat as DAPI, synthesise empty Cy3
        logger.warning("Image is single-channel; using as DAPI with empty Cy3.")
        return {"dapi": img.astype(np.float64), "cy3": np.zeros_like(img, dtype=np.float64)}

    if img.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        if img.shape[0] <= 5:
            channels_first = True
        elif img.shape[2] <= 5:
            channels_first = False
        else:
            channels_first = True  # assume (C, H, W)

        if channels_first:
            dapi = img[cfg["dapi_channel"]].astype(np.float64)
            cy3 = (
                img[cfg["cy3_channel"]].astype(np.float64)
                if img.shape[0] > cfg["cy3_channel"]
                else np.zeros_like(dapi)
            )
        else:
            dapi = img[:, :, cfg["dapi_channel"]].astype(np.float64)
            cy3 = (
                img[:, :, cfg["cy3_channel"]].astype(np.float64)
                if img.shape[2] > cfg["cy3_channel"]
                else np.zeros_like(dapi)
            )

        return {"dapi": dapi, "cy3": cy3}

    raise ValueError(f"Unsupported image shape {img.shape} for {image_path}")


def _preprocess_channel(channel: np.ndarray, cfg: dict) -> np.ndarray:
    """Background-subtract and denoise a single channel."""
    from scipy.ndimage import gaussian_filter, uniform_filter

    processed = channel.copy()

    # Background subtraction
    method = cfg.get("background_method", "rolling_ball")
    if method == "rolling_ball":
        radius = cfg.get("rolling_ball_radius", 50)
        background = uniform_filter(processed, size=radius)
        processed = np.clip(processed - background, 0, None)
    elif method == "median":
        from scipy.ndimage import median_filter

        radius = cfg.get("rolling_ball_radius", 50)
        background = median_filter(processed, size=radius)
        processed = np.clip(processed - background, 0, None)
    else:
        logger.warning("Unknown background method '%s'; skipping.", method)

    # Denoising
    sigma = cfg.get("denoise_sigma", 1.0)
    if sigma > 0:
        processed = gaussian_filter(processed, sigma=sigma)

    return processed


def _segment_chromosomes(dapi: np.ndarray, cfg: dict) -> np.ndarray:
    """Segment chromosomes from the DAPI channel and return a label mask."""
    from scipy.ndimage import binary_fill_holes, distance_transform_edt
    from scipy.ndimage import label as ndi_label
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops
    from skimage.morphology import disk, opening, remove_small_objects
    from skimage.segmentation import watershed

    method = cfg.get("segmentation_method", "otsu_watershed")

    if method == "otsu_watershed":
        thresh = threshold_otsu(dapi)
        binary = dapi > thresh
        binary = opening(binary, disk(2))
        binary = binary_fill_holes(binary)

        min_area = cfg.get("min_chromosome_area", 80)
        binary = remove_small_objects(binary, min_size=min_area)

        # Distance-transform watershed to split touching chromosomes
        distance = distance_transform_edt(binary)
        from skimage.feature import peak_local_max

        coords = peak_local_max(distance, min_distance=10, labels=binary)
        markers = np.zeros_like(binary, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        markers, _ = ndi_label(markers)
        labels = watershed(-distance, markers, mask=binary)
    elif method == "cellpose":
        # Cellpose integration (optional dependency)
        try:
            from cellpose import models

            model = models.Cellpose(model_type="nuclei", gpu=False)
            labels, _, _, _ = model.eval([dapi], channels=[0, 0], diameter=None)
            labels = labels[0]
        except ImportError:
            logger.error("Cellpose is not installed. Falling back to otsu_watershed.")
            return _segment_chromosomes(dapi, {**cfg, "segmentation_method": "otsu_watershed"})
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    # Filter by area
    min_area = cfg.get("min_chromosome_area", 80)
    max_area = cfg.get("max_chromosome_area", 8000)
    props = regionprops(labels)
    for prop in props:
        if prop.area < min_area or prop.area > max_area:
            labels[labels == prop.label] = 0

    # Relabel contiguously
    unique_labels = np.unique(labels[labels > 0])
    relabeled = np.zeros_like(labels)
    for new_id, old_id in enumerate(unique_labels, start=1):
        relabeled[labels == old_id] = new_id

    return relabeled


def _extract_chromosome_properties(labels: np.ndarray) -> list[dict[str, Any]]:
    """Extract centroid, bounding box, orientation, and tip positions."""
    from skimage.measure import regionprops

    chromosomes: list[dict[str, Any]] = []
    for prop in regionprops(labels):
        # Tips estimated as the two extremes along the major axis
        cy, cx = prop.centroid
        orientation = prop.orientation  # radians, counter-clockwise from horizontal
        major_len = prop.major_axis_length / 2.0

        dy = major_len * np.sin(orientation)
        dx = major_len * np.cos(orientation)

        tip_p = (cy - dy, cx + dx)  # p-arm tip
        tip_q = (cy + dy, cx - dx)  # q-arm tip

        chromosomes.append(
            {
                "label": int(prop.label),
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "area": int(prop.area),
                "major_axis_length": float(prop.major_axis_length),
                "minor_axis_length": float(prop.minor_axis_length),
                "orientation": float(orientation),
                "bbox": prop.bbox,
                "tips": [tip_p, tip_q],
                "chromosome_label": str(prop.label),
                "arms": ["p", "q"],
            }
        )

    return chromosomes


def _detect_spots(cy3: np.ndarray, cfg: dict) -> list[dict[str, Any]]:
    """Detect bright telomere spots in the Cy3 channel using LoG."""
    from skimage.feature import blob_log

    sigma_min = cfg.get("spot_sigma_min", 1.5)
    sigma_max = cfg.get("spot_sigma_max", 4.0)
    threshold = cfg.get("spot_threshold", 0.02)

    # Normalise image to [0, 1] for blob detection
    img = cy3.copy()
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)

    blobs = blob_log(
        img,
        min_sigma=sigma_min,
        max_sigma=sigma_max,
        threshold=threshold,
        overlap=0.5,
    )

    spots: list[dict[str, Any]] = []
    for blob in blobs:
        y, x, sigma = blob
        radius = sigma * np.sqrt(2)
        iy, ix = int(round(y)), int(round(x))

        # Peak intensity (from original, un-normalised channel)
        peak = float(cy3[iy, ix]) if 0 <= iy < cy3.shape[0] and 0 <= ix < cy3.shape[1] else 0.0

        spots.append(
            {
                "y": float(y),
                "x": float(x),
                "sigma": float(sigma),
                "radius": float(radius),
                "peak_intensity": peak,
                "corrected_intensity": 0.0,  # placeholder – computed in quantification
                "background_level": 0.0,
                "snr": 0.0,
                "associated": False,
                "valid": True,
                "chromosome_label": "",
                "arm": "",
            }
        )

    return spots


def _associate_spots(
    spots: list[dict],
    chromosomes: list[dict],
    cfg: dict,
) -> list[dict]:
    """Associate each spot with the nearest chromosome tip."""
    max_dist = cfg.get("max_tip_distance", 15.0)

    for spot in spots:
        sy, sx = spot["y"], spot["x"]
        best_dist = float("inf")
        best_chrom: dict | None = None
        best_arm = ""
        best_tip: tuple[float, float] = (0.0, 0.0)

        for chrom in chromosomes:
            for arm_idx, (ty, tx) in enumerate(chrom["tips"]):
                d = np.sqrt((sy - ty) ** 2 + (sx - tx) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_chrom = chrom
                    best_arm = chrom["arms"][arm_idx]
                    best_tip = (ty, tx)

        if best_dist <= max_dist and best_chrom is not None:
            spot["associated"] = True
            spot["chromosome_label"] = best_chrom["chromosome_label"]
            spot["arm"] = best_arm
            spot["tip_y"] = best_tip[0]
            spot["tip_x"] = best_tip[1]
            spot["tip_distance"] = float(best_dist)
        else:
            spot["associated"] = False

    return spots


def _quantify_spots(
    spots: list[dict],
    cy3: np.ndarray,
    cfg: dict,
) -> list[dict]:
    """Compute background-corrected intensity and SNR for each spot."""
    annulus_inner = cfg.get("annulus_inner", 5)
    annulus_outer = cfg.get("annulus_outer", 8)
    min_snr = cfg.get("spot_min_snr", 3.0)

    h, w = cy3.shape[:2]

    for spot in spots:
        iy, ix = int(round(spot["y"])), int(round(spot["x"]))
        r_inner = annulus_inner
        r_outer = annulus_outer

        # Signal: sum in inner circle
        signal_values = []
        for dy in range(-r_inner, r_inner + 1):
            for dx in range(-r_inner, r_inner + 1):
                yy_c, xx_c = iy + dy, ix + dx
                if 0 <= yy_c < h and 0 <= xx_c < w and (dy**2 + dx**2) <= r_inner**2:
                    signal_values.append(cy3[yy_c, xx_c])

        # Background: annulus between inner and outer
        bg_values = []
        for dy in range(-r_outer, r_outer + 1):
            for dx in range(-r_outer, r_outer + 1):
                yy_c, xx_c = iy + dy, ix + dx
                dist_sq = dy**2 + dx**2
                if (
                    0 <= yy_c < h
                    and 0 <= xx_c < w
                    and dist_sq > r_inner**2
                    and dist_sq <= r_outer**2
                ):
                    bg_values.append(cy3[yy_c, xx_c])

        signal = float(np.mean(signal_values)) if signal_values else 0.0
        bg = float(np.median(bg_values)) if bg_values else 0.0
        bg_std = float(np.std(bg_values)) if len(bg_values) > 1 else 1.0

        corrected = max(signal - bg, 0.0)
        snr = corrected / bg_std if bg_std > 0 else 0.0

        spot["corrected_intensity"] = corrected
        spot["background_level"] = bg
        spot["snr"] = snr

        # Mark spots with low SNR as invalid
        if snr < min_snr:
            spot["valid"] = False

    return spots


def _apply_calibration(
    spots: list[dict],
    cfg: dict,
    calibration: Any = None,
) -> list[dict]:
    """Convert corrected intensities to base-pair lengths using calibration.

    Uses a linear model: ``length_bp = slope * corrected_intensity + intercept``.
    The calibration can come from a dedicated *Calibration* object or from
    ``calibration_slope`` / ``calibration_intercept`` keys in *cfg*.
    """
    slope: float | None = None
    intercept: float | None = None

    if calibration is not None:
        # Duck-typed Calibration object
        slope = getattr(calibration, "slope", None)
        intercept = getattr(calibration, "intercept", None)

    if slope is None:
        slope = cfg.get("calibration_slope")
    if intercept is None:
        intercept = cfg.get("calibration_intercept")

    if slope is None or intercept is None:
        return spots  # no calibration available

    slope = float(slope)
    intercept = float(intercept)

    for spot in spots:
        spot["length_bp"] = slope * spot["corrected_intensity"] + intercept

    return spots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_image(
    image_path: str,
    config: dict[str, Any] | None = None,
    calibration: Any = None,
) -> dict[str, Any]:
    """Run the full telomere analysis pipeline on a single image.

    Processing steps
    ----------------
    1. Load & split channels (DAPI + Cy3)
    2. Pre-process (background subtraction, denoising)
    3. Segment chromosomes from the DAPI channel
    4. Extract chromosome properties (centroids, tips)
    5. Detect telomere spots in the Cy3 channel (LoG blob detection)
    6. Associate spots with nearest chromosome tips
    7. Quantify spot intensities (background-corrected + SNR)
    8. Apply calibration if available
    9. Compute summary statistics

    Parameters
    ----------
    image_path : str
        Path to a multi-channel TIFF image.
    config : dict | None
        Pipeline parameters (see :func:`get_default_config`).
    calibration
        Optional calibration object with ``slope`` and ``intercept`` attrs.

    Returns
    -------
    dict
        ``image_path``, ``channels``, ``labels``, ``chromosomes``,
        ``spots``, ``statistics``, ``association_summary``.
    """
    from ..analysis.statistics import compute_cell_statistics

    cfg = get_default_config()
    if config is not None:
        cfg.update(config)

    logger.info("Analyzing %s", image_path)

    # 1. Load image -------------------------------------------------------
    channels = _load_image(image_path, cfg)

    # 2. Pre-process ------------------------------------------------------
    dapi_proc = _preprocess_channel(channels["dapi"], cfg)
    cy3_proc = _preprocess_channel(channels["cy3"], cfg)

    # 3. Segment chromosomes ----------------------------------------------
    labels = _segment_chromosomes(dapi_proc, cfg)
    n_seg = int(labels.max())
    logger.info("Segmented %d chromosomes", n_seg)

    # 4. Chromosome properties --------------------------------------------
    chromosomes = _extract_chromosome_properties(labels)

    # 5. Detect spots -----------------------------------------------------
    spots = _detect_spots(cy3_proc, cfg)
    logger.info("Detected %d candidate spots", len(spots))

    # 6. Associate spots with tips ----------------------------------------
    spots = _associate_spots(spots, chromosomes, cfg)

    # 7. Quantify ---------------------------------------------------------
    spots = _quantify_spots(spots, channels["cy3"], cfg)

    # 8. Calibrate --------------------------------------------------------
    spots = _apply_calibration(spots, cfg, calibration)

    # 9. Statistics -------------------------------------------------------
    stats = compute_cell_statistics(spots)

    # Association summary
    n_associated = sum(1 for s in spots if s.get("associated", False))
    n_valid = sum(1 for s in spots if s.get("valid", True))
    association_summary = {
        "total_spots": len(spots),
        "associated": n_associated,
        "unassociated": len(spots) - n_associated,
        "valid": n_valid,
        "invalid": len(spots) - n_valid,
        "association_rate": n_associated / len(spots) if spots else 0.0,
    }

    return {
        "image_path": image_path,
        "channels": {"dapi": channels["dapi"], "cy3": channels["cy3"]},
        "labels": labels,
        "chromosomes": chromosomes,
        "spots": spots,
        "statistics": stats,
        "association_summary": association_summary,
    }


def analyze_batch(
    input_dir: str,
    output_dir: str,
    config: dict[str, Any] | None = None,
    pattern: str = "*.tif",
    calibration: Any = None,
) -> pd.DataFrame:
    """Batch-process a directory of images.

    For each matching file the full pipeline is executed, per-image CSV
    results and overlay images are saved, and a combined CSV is written
    at the end.

    Parameters
    ----------
    input_dir : str
        Directory containing source images.
    output_dir : str
        Directory where outputs are written.
    config : dict | None
        Pipeline configuration.
    pattern : str
        Glob pattern for matching image files (default ``"*.tif"``).
    calibration
        Optional calibration object.

    Returns
    -------
    pd.DataFrame
        Combined results across all images.
    """
    from ..analysis.statistics import compute_sample_statistics, create_results_dataframe
    from ..visualisation.plots import plot_telomere_overlay

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob(pattern))
    if not files:
        logger.warning("No files matching '%s' in %s", pattern, input_dir)
        return pd.DataFrame()

    all_dfs: list[pd.DataFrame] = []
    all_cell_stats: list[dict] = []

    for fpath in files:
        logger.info("Processing %s", fpath.name)
        try:
            result = analyze_image(str(fpath), config=config, calibration=calibration)
        except Exception:
            logger.exception("Failed to process %s; skipping.", fpath.name)
            continue

        spots = result["spots"]
        image_name = fpath.stem

        # Per-image CSV
        df = create_results_dataframe(spots, image_name=image_name)
        csv_out = out_path / f"{image_name}_telomeres.csv"
        df.to_csv(csv_out, index=False)
        all_dfs.append(df)
        all_cell_stats.append(result["statistics"])

        # Overlay
        try:
            overlay_out = out_path / f"{image_name}_overlay.png"
            plot_telomere_overlay(
                dapi=result["channels"]["dapi"],
                cy3=result["channels"]["cy3"],
                spots=spots,
                chromosomes=result.get("chromosomes"),
                labels=result.get("labels"),
                save_path=str(overlay_out),
            )
        except Exception:
            logger.exception("Failed to save overlay for %s.", fpath.name)

    # Combined CSV
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out_path / "combined_results.csv", index=False)
    else:
        combined = pd.DataFrame()

    # Sample-level summary
    if all_cell_stats:
        sample_stats = compute_sample_statistics(all_cell_stats)
        summary_df = pd.DataFrame([sample_stats])
        summary_df.to_csv(out_path / "sample_summary.csv", index=False)

    return combined
