"""Statistical analysis and reporting for telomere measurements.

Provides functions for computing per-cell summary statistics from
quantified telomere spots and for assembling results into tidy
pandas DataFrames suitable for downstream analysis and export.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_cell_statistics(spots: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for a single cell's telomere measurements.

    Only spots that are both ``associated`` (linked to a chromosome) and
    ``valid`` (aperture fully inside the image) are counted as true
    telomere measurements.

    Parameters
    ----------
    spots : list[dict]
        Spot dictionaries after association and quantification.  Each dict
        should contain at least ``associated`` (bool), ``valid`` (bool),
        and ``corrected_intensity`` (float).

    Returns
    -------
    dict[str, Any]
        Summary statistics with the following keys:

        - ``n_telomeres`` (int): Number of associated + valid spots.
        - ``total_spots`` (int): Total input spots.
        - ``associated_spots`` (int): Spots with ``associated=True``.
        - ``mean_intensity`` (float): Mean corrected intensity of telomeres.
        - ``median_intensity`` (float): Median corrected intensity.
        - ``std_intensity`` (float): Standard deviation (ddof=1).
        - ``min_intensity`` (float): Minimum corrected intensity.
        - ``max_intensity`` (float): Maximum corrected intensity.
        - ``cv`` (float): Coefficient of variation (std / mean), 0 if mean=0.
    """
    telomere_spots = [s for s in spots if s.get("associated", False) and s.get("valid", False)]

    n_telomeres = len(telomere_spots)
    total_spots = len(spots)
    associated_spots = sum(1 for s in spots if s.get("associated", False))

    if n_telomeres > 0:
        intensities = np.array([s["corrected_intensity"] for s in telomere_spots], dtype=np.float64)
        mean_intensity = float(np.mean(intensities))
        median_intensity = float(np.median(intensities))
        std_intensity = float(np.std(intensities, ddof=1)) if n_telomeres > 1 else 0.0
        min_intensity = float(np.min(intensities))
        max_intensity = float(np.max(intensities))
        cv = std_intensity / mean_intensity if mean_intensity != 0 else 0.0
    else:
        mean_intensity = 0.0
        median_intensity = 0.0
        std_intensity = 0.0
        min_intensity = 0.0
        max_intensity = 0.0
        cv = 0.0

    return {
        "n_telomeres": n_telomeres,
        "total_spots": total_spots,
        "associated_spots": associated_spots,
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "cv": cv,
    }


def create_results_dataframe(
    spots: list[dict[str, Any]],
    image_name: str = "",
) -> pd.DataFrame:
    """Create a tidy pandas DataFrame from spot measurement results.

    Each row corresponds to one detected telomere spot.  An ``image``
    column is prepended so that results from multiple images can be
    concatenated into a single DataFrame.

    Parameters
    ----------
    spots : list[dict]
        Spot dictionaries with arbitrary keys (typically including
        ``y``, ``x``, ``corrected_intensity``, ``chromosome_label``,
        ``arm``, ``associated``, ``valid``, etc.).
    image_name : str
        Identifier for the source image; stored in the ``image`` column.

    Returns
    -------
    pd.DataFrame
        One row per spot.  Column order is ``image`` first, followed by
        all keys present in the spot dicts.
    """
    if not spots:
        return pd.DataFrame(columns=["image"])

    df = pd.DataFrame(spots)
    df.insert(0, "image", image_name)
    return df
