"""Associate detected telomere spots with chromosome endpoints.

This module provides functions for matching telomere fluorescence spots
(detected in the Cy3 channel) to the nearest chromosome tip identified
from DAPI segmentation. Uses spatial indexing via KDTree for efficient
nearest-neighbor queries, and resolves conflicts when multiple spots
compete for the same chromosome tip.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def associate_spots_to_chromosomes(
    spots: list[dict],
    chromosomes: list[dict],
    max_distance: float = 15.0,
) -> list[dict]:
    """Match telomere spots to nearest chromosome tip.

    Args:
        spots: List of dicts, each with at least:
            - ``y`` (float): vertical coordinate of spot centre
            - ``x`` (float): horizontal coordinate of spot centre
            Additional keys (``sigma``, ``radius``, ``peak_intensity``, etc.)
            are preserved unchanged.
        chromosomes: List of dicts, each with at least:
            - ``label`` (int): unique chromosome identifier
            - ``tip_p`` (tuple[float, float]): (y, x) of the p-arm tip
            - ``tip_q`` (tuple[float, float]): (y, x) of the q-arm tip
        max_distance: Maximum pixel distance for a valid association.
            Spots farther than this from every tip are left unassociated.

    Algorithm:
        1. Build a flat list of all chromosome tips with metadata
           ``[(y, x), ...]`` paired with ``[(chr_label, arm), ...]``.
        2. Construct a :class:`scipy.spatial.cKDTree` from tip coordinates.
        3. For each spot, query the nearest tip.
        4. Accept the association only if distance < *max_distance*.
        5. Resolve conflicts: if two spots claim the same tip, keep the
           one with the shorter distance and leave the other unassociated.

    Returns:
        The *same* ``spots`` list, with each dict augmented by:
            - ``chromosome_label`` (int | None): matched chromosome label
            - ``arm`` (str | None): ``'p'`` or ``'q'``
            - ``tip_distance`` (float): distance to assigned tip (``inf``
              if unassociated)
            - ``associated`` (bool): whether a valid match was found

    Note:
        The input list is modified **in-place** and also returned for
        convenience.
    """
    # -- Edge cases --------------------------------------------------------
    if not spots:
        return spots
    if not chromosomes:
        for spot in spots:
            spot.update(
                chromosome_label=None,
                arm=None,
                tip_distance=float("inf"),
                associated=False,
            )
        return spots

    # -- 1. Collect all tips -----------------------------------------------
    tip_coords: list[tuple[float, float]] = []
    tip_meta: list[tuple[int, str]] = []  # (chromosome_label, arm)

    for chrom in chromosomes:
        label = chrom["label"]
        for arm in ("p", "q"):
            tip_key = f"tip_{arm}"
            tip = chrom[tip_key]
            tip_coords.append((float(tip[0]), float(tip[1])))
            tip_meta.append((label, arm))

    tip_array = np.asarray(tip_coords, dtype=np.float64)  # (N_tips, 2)

    # -- 2. Build KDTree ---------------------------------------------------
    tree = cKDTree(tip_array)

    # -- 3. Query nearest tip for every spot -------------------------------
    spot_coords = np.array([[s["y"], s["x"]] for s in spots], dtype=np.float64)  # (N_spots, 2)

    distances, indices = tree.query(spot_coords, k=1)

    # -- 4 & 5. Assign, respecting max_distance, then resolve conflicts ----
    # First pass: provisional assignment
    # tip_key -> (spot_index, distance)  – best candidate per tip
    best_for_tip: dict[tuple[int, str], tuple[int, float]] = {}

    provisional: list[tuple[int | None, str | None, float]] = []
    for i, (dist, tip_idx) in enumerate(zip(distances, indices)):
        if dist > max_distance:
            provisional.append((None, None, float("inf")))
            continue
        chrom_label, arm = tip_meta[tip_idx]
        tip_key_val = (chrom_label, arm)
        provisional.append((chrom_label, arm, float(dist)))

        # Track the closest spot for this tip
        if tip_key_val not in best_for_tip or dist < best_for_tip[tip_key_val][1]:
            best_for_tip[tip_key_val] = (i, float(dist))

    # Build set of winning (spot_index, tip_key) pairs
    winners: dict[int, tuple[int, str]] = {}
    for (chrom_label, arm), (spot_idx, _dist) in best_for_tip.items():
        winners[spot_idx] = (chrom_label, arm)

    # Second pass: write results, evicting losers
    for i, spot in enumerate(spots):
        chrom_label, arm, dist = provisional[i]
        if chrom_label is None:
            # Beyond max_distance
            spot.update(
                chromosome_label=None,
                arm=None,
                tip_distance=float("inf"),
                associated=False,
            )
        elif i in winners and winners[i] == (chrom_label, arm):
            # This spot is the closest claimant for this tip
            spot.update(
                chromosome_label=chrom_label,
                arm=arm,
                tip_distance=dist,
                associated=True,
            )
        else:
            # Lost the conflict to a closer spot
            spot.update(
                chromosome_label=None,
                arm=None,
                tip_distance=float("inf"),
                associated=False,
            )

    return spots


def summarize_associations(
    spots: list[dict],
    chromosomes: list[dict],
) -> dict:
    """Summarize association results after :func:`associate_spots_to_chromosomes`.

    Args:
        spots: Spot list **after** association (must contain ``associated``
            and ``chromosome_label`` / ``arm`` keys).
        chromosomes: The same chromosome list used during association.

    Returns:
        A summary dict with the following keys:

        - ``total_spots`` (int): Total number of input spots.
        - ``associated_spots`` (int): Spots successfully linked to a tip.
        - ``unassociated_spots`` (int): Spots without a chromosome match.
        - ``chromosomes_with_both_telomeres`` (int): Chromosomes that have
          associated spots on *both* the p-arm and q-arm tips.
        - ``chromosomes_with_one_telomere`` (int): Chromosomes with exactly
          one associated telomere.
        - ``chromosomes_with_no_telomere`` (int): Chromosomes with no
          associated telomeres at all.
        - ``association_rate`` (float): Fraction of spots that were
          associated (0.0--1.0).  Returns 0.0 when there are no spots.
    """
    total = len(spots)
    associated = sum(1 for s in spots if s.get("associated", False))
    unassociated = total - associated

    # Track which arms each chromosome has covered
    # chrom_label -> set of arms ('p', 'q')
    chrom_arms: dict[int, set[str]] = {}
    for spot in spots:
        if spot.get("associated"):
            label = spot["chromosome_label"]
            arm = spot["arm"]
            chrom_arms.setdefault(label, set()).add(arm)

    both = 0
    one = 0
    none_ = 0
    for chrom in chromosomes:
        label = chrom["label"]
        arms = chrom_arms.get(label, set())
        n_arms = len(arms)
        if n_arms == 2:
            both += 1
        elif n_arms == 1:
            one += 1
        else:
            none_ += 1

    return {
        "total_spots": total,
        "associated_spots": associated,
        "unassociated_spots": unassociated,
        "chromosomes_with_both_telomeres": both,
        "chromosomes_with_one_telomere": one,
        "chromosomes_with_no_telomere": none_,
        "association_rate": associated / total if total > 0 else 0.0,
    }
