"""Visualization for telomere analysis results.

All plotting functions use the non-interactive ``Agg`` backend so they work
in headless / CI environments.  Every function returns the
:class:`~matplotlib.figure.Figure` and optionally saves to disk and/or
displays the plot interactively.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    _HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _finish(
    fig: plt.Figure,
    save_path: str | None,
    show: bool,
    tight: bool = True,
) -> plt.Figure:
    """Shared logic to optionally save / show a figure."""
    if tight:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:  # pragma: no cover
        plt.show()
    return fig


def _rescale(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    """Contrast-stretch a 2-D image to [0, 1] using percentile clipping."""
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)


def _valid_spots(spots: list[dict]) -> list[dict]:
    """Return only spots that are associated *and* valid."""
    return [s for s in spots if s.get("associated", False) and s.get("valid", True)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_telomere_overlay(
    dapi: np.ndarray,
    cy3: np.ndarray,
    spots: list[dict],
    chromosomes: list[dict] | None = None,
    labels: np.ndarray | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot merged microscopy image with detected telomere spots overlaid.

    The composite shows DAPI in blue and Cy3 in red.  Detected spots are
    drawn as yellow circles, chromosome tips as green dots, and association
    lines as white dotted connectors.

    Parameters
    ----------
    dapi : np.ndarray
        2-D DAPI (blue) channel image.
    cy3 : np.ndarray
        2-D Cy3 (red) channel image, same shape as *dapi*.
    spots : list[dict]
        Spot dictionaries with at least ``y``, ``x``, ``radius`` keys and
        optionally ``tip_y``, ``tip_x`` for association lines.
    chromosomes : list[dict] | None
        Optional chromosome dicts with ``tips`` key (list of ``(y, x)``).
    labels : np.ndarray | None
        Optional segmentation label mask (same shape as *dapi*).  If provided,
        contours are drawn around each segmented region.
    save_path, show : str | None, bool
        Standard save/display flags.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Build RGB composite ------------------------------------------------
    blue = _rescale(dapi)
    red = _rescale(cy3)
    rgb = np.stack([red, np.zeros_like(blue), blue], axis=-1)

    # Overlay segmentation contours ---------------------------------------
    if labels is not None:
        from scipy.ndimage import binary_dilation, find_objects

        boundary = np.zeros(labels.shape, dtype=bool)
        for sl in find_objects(labels):
            if sl is None:
                continue
            patch = labels[sl] > 0
            dilated = binary_dilation(patch, iterations=1)
            boundary[sl] |= dilated ^ patch
        # Cyan contour
        rgb[boundary] = [0.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper")

    # Chromosome tips (green) ---------------------------------------------
    if chromosomes is not None:
        for chrom in chromosomes:
            for ty, tx in chrom.get("tips", []):
                ax.plot(tx, ty, "o", color="lime", markersize=3, alpha=0.8)

    # Detected spots (yellow circles) and association lines ---------------
    for s in spots:
        y, x = s["y"], s["x"]
        r = s.get("radius", 3)
        colour = "yellow" if s.get("valid", True) else "gray"
        circle = plt.Circle((x, y), r, fill=False, edgecolor=colour, linewidth=0.8)
        ax.add_patch(circle)

        # Association connector
        if "tip_y" in s and "tip_x" in s:
            ax.plot(
                [x, s["tip_x"]],
                [y, s["tip_y"]],
                ":",
                color="white",
                linewidth=0.5,
                alpha=0.6,
            )

    ax.set_title("Telomere overlay", fontsize=13)
    ax.axis("off")

    # Legend
    handles = [
        mpatches.Patch(color="blue", label="DAPI"),
        mpatches.Patch(color="red", label="Cy3 (telomere)"),
        mpatches.Patch(facecolor="none", edgecolor="yellow", label="Detected spot"),
    ]
    if chromosomes is not None:
        handles.append(mpatches.Patch(color="lime", label="Chromosome tip"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)

    return _finish(fig, save_path, show)


def plot_intensity_histogram(
    spots: list[dict],
    calibrated: bool = False,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Histogram of telomere intensities (or calibrated lengths).

    Parameters
    ----------
    spots : list[dict]
        Spot dictionaries.
    calibrated : bool
        If ``True``, use the ``length_bp`` key instead of
        ``corrected_intensity``.
    save_path, show : str | None, bool
        Standard save/display flags.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid = _valid_spots(spots)
    if calibrated:
        values = np.array([s["length_bp"] for s in valid if "length_bp" in s], dtype=np.float64)
        xlabel = "Telomere length (bp)"
    else:
        values = np.array([s["corrected_intensity"] for s in valid], dtype=np.float64)
        xlabel = "Corrected intensity (a.u.)"

    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_bins = max(10, min(80, int(np.sqrt(len(values)))))
    if _HAS_SEABORN:
        sns.histplot(values, bins=n_bins, kde=True, color="steelblue", ax=ax)
    else:
        ax.hist(values, bins=n_bins, color="steelblue", edgecolor="white", alpha=0.85)

    # Summary statistics lines
    mean_val = float(np.mean(values)) if values.size else 0
    median_val = float(np.median(values)) if values.size else 0
    ax.axvline(
        mean_val,
        color="firebrick",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean {mean_val:.1f}",
    )
    ax.axvline(
        median_val,
        color="darkorange",
        linestyle="-.",
        linewidth=1.2,
        label=f"Median {median_val:.1f}",
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Telomere intensity distribution", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)

    return _finish(fig, save_path, show)


def plot_chromosome_heatmap(
    spots: list[dict],
    n_chromosomes: int = 23,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Heatmap of telomere length per chromosome arm.

    Rows represent chromosome numbers (1-22, X, Y) and columns the p-arm
    and q-arm.  Cell colour encodes mean intensity or calibrated length.

    Parameters
    ----------
    spots : list[dict]
        Spot dicts with ``chromosome_label`` (int or str like ``'X'``) and
        ``arm`` (``'p'`` or ``'q'``) keys.
    n_chromosomes : int
        Total chromosomes to display (default 23 → 1-22 + X).
    save_path, show : str | None, bool
        Standard save/display flags.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid = _valid_spots(spots)

    # Determine value key
    use_bp = all("length_bp" in s for s in valid) if valid else False
    val_key = "length_bp" if use_bp else "corrected_intensity"
    cbar_label = "Telomere length (bp)" if use_bp else "Corrected intensity (a.u.)"

    # Chromosome labels
    chrom_labels: list[str] = [str(i) for i in range(1, n_chromosomes)]
    if n_chromosomes >= 23:
        chrom_labels.append("X")
    if n_chromosomes >= 24:
        chrom_labels.append("Y")
    arms = ["p", "q"]

    # Build matrix (rows=chromosomes, cols=arms)
    matrix = np.full((len(chrom_labels), 2), np.nan, dtype=np.float64)
    for s in valid:
        cl = str(s.get("chromosome_label", ""))
        arm = s.get("arm", "")
        if cl in chrom_labels and arm in arms:
            ri = chrom_labels.index(cl)
            ci = arms.index(arm)
            current = matrix[ri, ci]
            val = s.get(val_key, np.nan)
            if np.isnan(current):
                matrix[ri, ci] = val
            else:
                # Running mean (simple accumulation)
                matrix[ri, ci] = (current + val) / 2.0

    fig, ax = plt.subplots(figsize=(4, max(6, len(chrom_labels) * 0.35)))

    cmap = plt.cm.YlOrRd  # type: ignore[attr-defined]
    cmap.set_bad(color="lightgray")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["p-arm", "q-arm"], fontsize=10)
    ax.set_yticks(range(len(chrom_labels)))
    ax.set_yticklabels([f"Chr {c}" for c in chrom_labels], fontsize=9)
    ax.set_title("Telomere length per chromosome arm", fontsize=12, pad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label=cbar_label)

    # Annotate cells with values
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            val = matrix[ri, ci]
            if np.isfinite(val):
                txt = f"{val:.0f}" if use_bp else f"{val:.1f}"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=7, color="black")

    return _finish(fig, save_path, show)


def plot_cell_comparison(
    cells: list[dict],
    labels: list[str] | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Box plot comparing telomere length distributions across cells.

    Parameters
    ----------
    cells : list[dict]
        Each dict should contain an ``intensities`` key mapping to a list
        or array of per-spot intensity values.  Alternatively, a
        ``lengths_bp`` key for calibrated data.
    labels : list[str] | None
        Labels for the x-axis (one per cell).  Defaults to ``Cell 1``, etc.
    save_path, show : str | None, bool
        Standard save/display flags.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if labels is None:
        labels = [f"Cell {i + 1}" for i in range(len(cells))]

    # Determine which value key to use
    use_bp = all("lengths_bp" in c for c in cells) if cells else False
    val_key = "lengths_bp" if use_bp else "intensities"
    ylabel = "Telomere length (bp)" if use_bp else "Corrected intensity (a.u.)"

    data: list[np.ndarray] = []
    for c in cells:
        arr = np.asarray(c.get(val_key, []), dtype=np.float64)
        data.append(arr)

    fig, ax = plt.subplots(figsize=(max(5, len(cells) * 0.9), 5))

    if _HAS_SEABORN:
        import pandas as pd

        long_rows: list[dict[str, Any]] = []
        for lbl, arr in zip(labels, data):
            for v in arr:
                long_rows.append({"Cell": lbl, "Value": float(v)})
        if long_rows:
            df = pd.DataFrame(long_rows)
            sns.boxplot(data=df, x="Cell", y="Value", ax=ax, palette="Set2", width=0.5)
            sns.stripplot(
                data=df,
                x="Cell",
                y="Value",
                ax=ax,
                color="0.3",
                size=2,
                alpha=0.4,
                jitter=True,
            )
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
    else:
        ax.boxplot(data, labels=labels, patch_artist=True)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Telomere length comparison across cells", fontsize=13)
    plt.xticks(rotation=45, ha="right")

    return _finish(fig, save_path, show)


def plot_spot_gallery(
    image: np.ndarray,
    spots: list[dict],
    n_spots: int = 20,
    patch_size: int = 30,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Gallery of cropped telomere-spot images sorted by descending intensity.

    Parameters
    ----------
    image : np.ndarray
        2-D image (typically the Cy3 channel) from which patches are
        cropped around each spot centre.
    spots : list[dict]
        Spot dictionaries with ``y``, ``x``, and ``corrected_intensity``.
    n_spots : int
        Maximum number of spots to show.
    patch_size : int
        Side length (pixels) of each cropped patch.
    save_path, show : str | None, bool
        Standard save/display flags.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid = _valid_spots(spots)
    # Sort descending by intensity
    valid = sorted(valid, key=lambda s: s.get("corrected_intensity", 0), reverse=True)
    valid = valid[:n_spots]

    h, w = image.shape[:2]
    half = patch_size // 2

    n = len(valid)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(
            0.5,
            0.5,
            "No valid spots to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return _finish(fig, save_path, show, tight=False)

    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes_flat: Sequence[plt.Axes] = np.asarray(axes).ravel() if n > 1 else [axes]  # type: ignore[assignment]

    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.axis("off")
            continue
        s = valid[idx]
        cy, cx = int(round(s["y"])), int(round(s["x"]))

        y0 = max(cy - half, 0)
        y1 = min(cy + half, h)
        x0 = max(cx - half, 0)
        x1 = min(cx + half, w)
        patch = image[y0:y1, x0:x1]

        ax.imshow(patch, cmap="hot", interpolation="nearest")
        intensity = s.get("corrected_intensity", 0)
        ax.set_title(f"I={intensity:.1f}", fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.suptitle("Top telomere spots by intensity", fontsize=13, y=1.02)
    return _finish(fig, save_path, show)
