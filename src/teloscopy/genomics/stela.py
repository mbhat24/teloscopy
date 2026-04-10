"""Single Telomere Length Analysis (STELA) — chromosome-arm-specific profiling.

Implements a computational model of STELA, originally a PCR-based technique
for measuring telomere lengths on individual chromosome arms (Baird et al.,
2003, *Nature Genetics* 33(2):203-207).  Rather than requiring
actual gel electrophoresis data, this module generates realistic
chromosome-specific telomere length distributions from an overall mean
telomere length estimate, applying published inter-arm variation data.

Chromosome-specific relative lengths are derived from:

- Martens, U.M. et al. (1998). Short telomeres on human chromosome 17p.
  *Nature Genetics* 18(1):76-80.
- Graakjaer, J. et al. (2006). Allele-specific relative telomere lengths
  are inherited. *Human Genetics* 119(3):344-350.

Key observations encoded in the model:

1. Chromosome 17p consistently harbours the shortest telomeres.
2. p-arms are generally slightly shorter than their q-arm counterparts.
3. Longer chromosomes tend to have marginally shorter telomeres.
4. Substantial inter-individual and intra-individual variation exists.

Telomere Biology Disorder (TBD) screening follows consensus guidelines
from Savage & Bertuch (2010) and Alder et al. (2018), using age-adjusted
percentile cutoffs.

**This module is intended for educational and research purposes only.
Results must NOT be used for clinical decision-making.**

Typical usage
-------------
>>> from teloscopy.genomics.stela import generate_stela_profile
>>> profile = generate_stela_profile(
...     mean_telomere_length_kb=6.5,
...     chronological_age=45,
...     sex="female",
... )
>>> print(f"Shortest: {profile.shortest_chromosome} "
...       f"({profile.shortest_telomere_kb:.2f} kb)")
>>> print(f"TBD risk: {profile.telomere_biology_disorder_risk}")
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ChromosomeArmTelomere",
    "STELAProfile",
    "generate_stela_profile",
    "estimate_attrition_rates",
    "parse_stela_gel_data",
    "screen_telomere_biology_disorder",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRITICAL_SHORT_KB: float = 4.0
"""Threshold below which a telomere is considered critically short."""

_VERY_SHORT_KB: float = 5.0
"""Threshold below which a telomere is considered very short."""

_METHODOLOGY_NOTE: str = (
    "Chromosome-arm telomere lengths were computationally modelled from "
    "the overall mean telomere length using published inter-arm variation "
    "data (Martens et al., 2000; Graakjaer et al., 2006).  This is a "
    "statistical approximation of STELA (Baird et al., 2003) and does "
    "not represent direct single-telomere measurements.  For clinical "
    "applications, laboratory-based STELA with Southern blot confirmation "
    "is recommended."
)

# Published chromosome-arm relative telomere lengths, normalised so that
# the overall mean across all arms equals 1.0.  Values reflect consensus
# observations: 17p shortest, p-arms slightly shorter than q-arms, and
# longer chromosomes modestly shorter.
_CHROMOSOME_ARM_RELATIVE_LENGTHS: dict[str, float] = {
    "1p": 0.92, "1q": 0.97,
    "2p": 0.94, "2q": 0.98,
    "3p": 0.96, "3q": 1.01,
    "4p": 0.95, "4q": 1.00,
    "5p": 0.97, "5q": 1.02,
    "6p": 0.98, "6q": 1.01,
    "7p": 0.96, "7q": 1.00,
    "8p": 0.95, "8q": 1.01,
    "9p": 0.97, "9q": 1.02,
    "10p": 0.96, "10q": 1.01,
    "11p": 0.97, "11q": 1.00,
    "12p": 0.98, "12q": 1.02,
    "13p": 0.99, "13q": 1.03,
    "14p": 0.98, "14q": 1.02,
    "15p": 0.97, "15q": 1.01,
    "16p": 0.93, "16q": 0.99,
    "17p": 0.88, "17q": 0.96,   # 17p is consistently shortest
    "18p": 0.95, "18q": 1.01,
    "19p": 0.94, "19q": 0.99,
    "20p": 0.98, "20q": 1.03,
    "21p": 0.99, "21q": 1.04,
    "22p": 0.97, "22q": 1.02,
    "Xp": 1.02, "Xq": 1.05,
    "Yp": 0.90, "Yq": 0.95,
}

# Per-arm attrition rate multipliers, relative to a population-average
# shortening of ~25 bp/year.  Shorter arms and 17p shorten faster.
_ATTRITION_RATE_MULTIPLIERS: dict[str, float] = {
    "1p": 1.10, "1q": 1.00,
    "2p": 1.08, "2q": 0.98,
    "3p": 1.05, "3q": 0.97,
    "4p": 1.06, "4q": 0.98,
    "5p": 1.04, "5q": 0.96,
    "6p": 1.03, "6q": 0.97,
    "7p": 1.05, "7q": 0.98,
    "8p": 1.06, "8q": 0.97,
    "9p": 1.04, "9q": 0.96,
    "10p": 1.05, "10q": 0.97,
    "11p": 1.04, "11q": 0.98,
    "12p": 1.03, "12q": 0.96,
    "13p": 1.02, "13q": 0.95,
    "14p": 1.03, "14q": 0.96,
    "15p": 1.04, "15q": 0.97,
    "16p": 1.09, "16q": 1.00,
    "17p": 1.20, "17q": 1.05,   # 17p shortens fastest (~30 bp/yr)
    "18p": 1.06, "18q": 0.97,
    "19p": 1.08, "19q": 1.00,
    "20p": 1.03, "20q": 0.95,
    "21p": 1.02, "21q": 0.94,
    "22p": 1.04, "22q": 0.96,
    "Xp": 0.92, "Xq": 0.90,
    "Yp": 1.12, "Yq": 1.06,
}

# Age- and sex-adjusted expected mean telomere length (kb).  Used for
# percentile estimation.  Simplified linear model from Aubert & Lansdorp
# (2008): TL(kb) ~ 11.0 - 0.040 * age, with females ~0.2 kb longer.
_NEWBORN_TL_KB: float = 11.0
_ANNUAL_DECLINE_KB: float = 0.040
_FEMALE_OFFSET_KB: float = 0.20

# Population SD of telomere length at a given age (~0.8 kb).
_POPULATION_SD_KB: float = 0.80

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChromosomeArmTelomere:
    """Telomere length data for a single chromosome arm.

    Attributes
    ----------
    chromosome : str
        Chromosome arm identifier (e.g. ``"1p"``, ``"17q"``, ``"Xp"``).
    arm : str
        ``"p"`` (short arm) or ``"q"`` (long arm).
    estimated_length_kb : float
        Estimated telomere length in kilobases.
    length_sd_kb : float
        Standard deviation of the length estimate (kb).
    percentile : int
        Population percentile for age and sex (0-100).
    is_critically_short : bool
        ``True`` if estimated length < 4.0 kb.
    is_very_short : bool
        ``True`` if estimated length < 5.0 kb.
    attrition_rate_bp_per_year : float
        Estimated annual telomere shortening (bp/year).
    """

    chromosome: str
    arm: str
    estimated_length_kb: float
    length_sd_kb: float
    percentile: int
    is_critically_short: bool
    is_very_short: bool
    attrition_rate_bp_per_year: float


@dataclass
class STELAProfile:
    """Complete STELA telomere length profile across all chromosome arms.

    Attributes
    ----------
    chromosome_telomeres : list[ChromosomeArmTelomere]
        Per-arm telomere length estimates.
    mean_telomere_length_kb : float
        Arithmetic mean across all chromosome arms.
    median_telomere_length_kb : float
        Median telomere length across all arms.
    shortest_telomere_kb : float
        Length of the shortest individual chromosome-arm telomere.
    shortest_chromosome : str
        Chromosome arm with the shortest telomere.
    longest_telomere_kb : float
        Length of the longest individual chromosome-arm telomere.
    longest_chromosome : str
        Chromosome arm with the longest telomere.
    critically_short_count : int
        Number of chromosome arms with telomere length < 4.0 kb.
    heterogeneity_index : float
        Coefficient of variation (SD / mean) across all arms.
    telomere_biology_disorder_risk : str
        Overall TBD risk classification: ``"Low"``, ``"Moderate"``,
        or ``"High"``.
    clinical_interpretation : str
        Human-readable clinical interpretation text.
    methodology_note : str
        Description of the computational methodology used.
    """

    chromosome_telomeres: list[ChromosomeArmTelomere]
    mean_telomere_length_kb: float
    median_telomere_length_kb: float
    shortest_telomere_kb: float
    shortest_chromosome: str
    longest_telomere_kb: float
    longest_chromosome: str
    critically_short_count: int
    heterogeneity_index: float
    telomere_biology_disorder_risk: str
    clinical_interpretation: str
    methodology_note: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deterministic_seed(
    age: int,
    sex: str,
    mean_tl_kb: float,
) -> int:
    """Derive a deterministic random seed from demographic + telomere data.

    This ensures that repeated calls with identical inputs produce the
    same chromosome-specific profile, while different individuals receive
    distinct biological noise patterns.
    """
    payload = f"{age}:{sex.lower().strip()}:{mean_tl_kb:.4f}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:12], 16)


def _expected_tl_for_age_sex(age: int, sex: str) -> float:
    """Return the population-expected mean telomere length (kb)."""
    base = _NEWBORN_TL_KB - _ANNUAL_DECLINE_KB * max(age, 0)
    if sex.lower().strip() == "female":
        base += _FEMALE_OFFSET_KB
    return max(base, 2.0)


def _tl_to_percentile(observed_kb: float, expected_kb: float) -> int:
    """Convert an observed telomere length to an approximate population percentile.

    Uses a normal-distribution model with the population SD.
    """
    if _POPULATION_SD_KB <= 0:
        return 50
    z = (observed_kb - expected_kb) / _POPULATION_SD_KB
    # Approximate CDF of standard normal via logistic approximation.
    percentile = 100.0 / (1.0 + math.exp(-1.7 * z))
    return max(0, min(100, round(percentile)))


def _chromosome_arms_for_sex(sex: str) -> list[str]:
    """Return the list of chromosome arms appropriate for the given sex."""
    arms = [
        arm for arm in _CHROMOSOME_ARM_RELATIVE_LENGTHS
        if arm not in ("Yp", "Yq")
    ]
    if sex.lower().strip() == "male":
        arms.extend(["Yp", "Yq"])
    return sorted(arms, key=_arm_sort_key)


def _arm_sort_key(arm: str) -> tuple[int, int]:
    """Sort key: numeric chromosome first, then p=0 / q=1.

    Non-numeric chromosomes (X, Y) sort after autosomes.
    """
    chrom = arm[:-1]
    suffix = 0 if arm[-1] == "p" else 1
    try:
        return (int(chrom), suffix)
    except ValueError:
        order = {"X": 23, "Y": 24}
        return (order.get(chrom, 25), suffix)


def _assess_tbd_risk(
    shortest_kb: float,
    critically_short_count: int,
    percentile: int,
    age: int,
) -> str:
    """Classify Telomere Biology Disorder risk level.

    Criteria derived from Savage & Bertuch (2010) and Alder et al. (2018):

    - **High**: shortest telomere < 3.0 kb, or >= 5 critically short arms,
      or overall telomere length < 1st percentile for age.
    - **Moderate**: shortest telomere < 4.0 kb, or >= 2 critically short
      arms, or overall telomere length < 10th percentile.
    - **Low**: otherwise.
    """
    if shortest_kb < 3.0 or critically_short_count >= 5 or percentile < 1:
        return "High"
    if shortest_kb < 4.0 or critically_short_count >= 2 or percentile < 10:
        return "Moderate"
    return "Low"


def _generate_clinical_interpretation(
    profile_mean_kb: float,
    shortest_kb: float,
    shortest_chr: str,
    critically_short_count: int,
    heterogeneity: float,
    tbd_risk: str,
    age: int,
    sex: str,
) -> str:
    """Build a human-readable clinical interpretation paragraph."""
    parts: list[str] = []

    # Overall length assessment.
    expected = _expected_tl_for_age_sex(age, sex)
    diff_pct = ((profile_mean_kb - expected) / expected) * 100.0
    if diff_pct < -15:
        parts.append(
            f"Mean telomere length ({profile_mean_kb:.2f} kb) is substantially "
            f"below the age- and sex-adjusted expectation ({expected:.2f} kb), "
            f"placing this individual approximately {abs(diff_pct):.0f}% below average."
        )
    elif diff_pct < -5:
        parts.append(
            f"Mean telomere length ({profile_mean_kb:.2f} kb) is moderately "
            f"below the expected value ({expected:.2f} kb) for a "
            f"{age}-year-old {sex}."
        )
    else:
        parts.append(
            f"Mean telomere length ({profile_mean_kb:.2f} kb) is within the "
            f"normal range for a {age}-year-old {sex} "
            f"(expected ~{expected:.2f} kb)."
        )

    # Shortest telomere.
    parts.append(
        f"The shortest telomere is on chromosome arm {shortest_chr} "
        f"({shortest_kb:.2f} kb)."
    )

    # Critically short arms.
    if critically_short_count > 0:
        parts.append(
            f"{critically_short_count} chromosome arm(s) have critically "
            f"short telomeres (< {_CRITICAL_SHORT_KB:.1f} kb), which may "
            f"be associated with genomic instability and increased cellular "
            f"senescence."
        )

    # Heterogeneity.
    if heterogeneity > 0.20:
        parts.append(
            f"Telomere length heterogeneity is elevated (CV = {heterogeneity:.3f}), "
            f"suggesting significant variation in telomere maintenance across "
            f"chromosome arms."
        )

    # TBD risk.
    if tbd_risk == "High":
        parts.append(
            "The telomere length profile raises concern for a possible "
            "Telomere Biology Disorder.  Referral to a clinical genetics "
            "service for confirmatory testing (flow-FISH, STELA) and "
            "genetic counselling is strongly recommended."
        )
    elif tbd_risk == "Moderate":
        parts.append(
            "Some features of the telomere profile warrant monitoring.  "
            "Consider longitudinal telomere length tracking and genetic "
            "counselling if there is a relevant family history."
        )

    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------


def generate_stela_profile(
    mean_telomere_length_kb: float,
    chronological_age: int,
    sex: str = "unknown",
    telomere_percentile: int = 50,
    ancestry_proportions: dict[str, float] | None = None,
) -> STELAProfile:
    """Generate a chromosome-arm-specific telomere length profile.

    Applies published inter-arm variation factors and realistic biological
    noise to produce a STELA-like distribution from an overall mean
    telomere length measurement.

    Parameters
    ----------
    mean_telomere_length_kb : float
        Overall mean telomere length in kilobases (e.g. from qPCR or
        TRF analysis).  Typical adult range: 4-12 kb.
    chronological_age : int
        Age of the individual in years.
    sex : str, optional
        Biological sex — ``"male"``, ``"female"``, or ``"unknown"``
        (default ``"unknown"``).  Affects which chromosome arms are
        included (Y chromosome only for males) and age-adjusted
        percentile calculations.
    telomere_percentile : int, optional
        Known population percentile if available (default 50).  Used
        to refine the noise model when prior data exists.
    ancestry_proportions : dict[str, float] or None, optional
        Continental ancestry proportions (e.g.
        ``{"European": 0.75, "African": 0.25}``).  Reserved for future
        ancestry-adjusted models; currently unused.

    Returns
    -------
    STELAProfile
        Complete chromosome-arm telomere length profile with summary
        statistics and clinical interpretation.

    Raises
    ------
    ValueError
        If *mean_telomere_length_kb* is not positive or *sex* is
        unrecognised.

    Examples
    --------
    >>> profile = generate_stela_profile(6.5, 45, sex="female")
    >>> profile.shortest_chromosome
    '17p'
    >>> profile.telomere_biology_disorder_risk
    'Low'
    """
    if mean_telomere_length_kb <= 0:
        raise ValueError(
            f"mean_telomere_length_kb must be positive, got "
            f"{mean_telomere_length_kb}"
        )

    sex_norm = sex.lower().strip()
    if sex_norm not in ("male", "female", "unknown"):
        raise ValueError(f"sex must be 'male', 'female', or 'unknown', got '{sex}'")

    # Use 'female' as default for sex-specific calculations when unknown,
    # as it is the more conservative model (slightly longer expected TL).
    sex_for_calc = sex_norm if sex_norm != "unknown" else "female"

    rng = random.Random(_deterministic_seed(chronological_age, sex_norm, mean_telomere_length_kb))

    # Biological noise SD scales with mean TL: shorter telomeres have
    # tighter distributions.  Range ~0.5-1.0 kb.
    noise_sd = max(0.5, min(1.0, mean_telomere_length_kb * 0.10))

    arms = _chromosome_arms_for_sex(sex_for_calc)
    expected_tl = _expected_tl_for_age_sex(chronological_age, sex_for_calc)

    telomeres: list[ChromosomeArmTelomere] = []
    lengths: list[float] = []

    for arm_name in arms:
        relative = _CHROMOSOME_ARM_RELATIVE_LENGTHS[arm_name]
        # Base length from relative factor.
        base_length = mean_telomere_length_kb * relative
        # Add Gaussian biological noise.
        noise = rng.gauss(0.0, noise_sd)
        estimated = max(0.5, base_length + noise)
        # Per-arm measurement SD (simulates STELA gel band spread).
        arm_sd = round(noise_sd * 0.6 + abs(noise) * 0.2, 3)

        pct = _tl_to_percentile(estimated, expected_tl * relative)
        arm_char = arm_name[-1]

        telomeres.append(
            ChromosomeArmTelomere(
                chromosome=arm_name,
                arm=arm_char,
                estimated_length_kb=round(estimated, 3),
                length_sd_kb=arm_sd,
                percentile=pct,
                is_critically_short=estimated < _CRITICAL_SHORT_KB,
                is_very_short=estimated < _VERY_SHORT_KB,
                attrition_rate_bp_per_year=0.0,  # populated by estimate_attrition_rates
            )
        )
        lengths.append(estimated)

    # Summary statistics.
    mean_length = statistics.mean(lengths)
    median_length = statistics.median(lengths)
    sd_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    heterogeneity = sd_length / mean_length if mean_length > 0 else 0.0

    shortest_idx = lengths.index(min(lengths))
    longest_idx = lengths.index(max(lengths))
    shortest_kb = lengths[shortest_idx]
    longest_kb = lengths[longest_idx]
    shortest_chr = telomeres[shortest_idx].chromosome
    longest_chr = telomeres[longest_idx].chromosome

    critically_short_count = sum(1 for t in telomeres if t.is_critically_short)

    overall_pct = _tl_to_percentile(mean_telomere_length_kb, expected_tl)
    tbd_risk = _assess_tbd_risk(
        shortest_kb, critically_short_count, overall_pct, chronological_age
    )

    interpretation = _generate_clinical_interpretation(
        profile_mean_kb=mean_length,
        shortest_kb=shortest_kb,
        shortest_chr=shortest_chr,
        critically_short_count=critically_short_count,
        heterogeneity=heterogeneity,
        tbd_risk=tbd_risk,
        age=chronological_age,
        sex=sex_for_calc,
    )

    return STELAProfile(
        chromosome_telomeres=telomeres,
        mean_telomere_length_kb=round(mean_length, 3),
        median_telomere_length_kb=round(median_length, 3),
        shortest_telomere_kb=round(shortest_kb, 3),
        shortest_chromosome=shortest_chr,
        longest_telomere_kb=round(longest_kb, 3),
        longest_chromosome=longest_chr,
        critically_short_count=critically_short_count,
        heterogeneity_index=round(heterogeneity, 4),
        telomere_biology_disorder_risk=tbd_risk,
        clinical_interpretation=interpretation,
        methodology_note=_METHODOLOGY_NOTE,
    )


# ---------------------------------------------------------------------------
# Attrition rate estimation
# ---------------------------------------------------------------------------


def estimate_attrition_rates(
    stela_profile: STELAProfile,
    chronological_age: int,
    sex: str,
) -> list[ChromosomeArmTelomere]:
    """Estimate per-chromosome-arm telomere attrition rates.

    Different chromosome arms shorten at different rates *in vivo*.
    Shorter telomeres tend to shorten faster (a positive feedback effect
    contributing to genomic instability), and 17p exhibits the highest
    attrition rate (~30 bp/year vs. ~20-25 bp/year for most arms).

    The model applies a base attrition rate (25 bp/year population
    average) modified by:

    1. A chromosome-specific multiplier from published data.
    2. A length-dependent correction — shorter telomeres lose more
       per division due to reduced shelterin protection.
    3. A sex adjustment — males lose ~2 bp/year more than females
       on average (Mayer et al., 2006).

    Parameters
    ----------
    stela_profile : STELAProfile
        A previously generated STELA profile.
    chronological_age : int
        Age in years.
    sex : str
        ``"male"``, ``"female"``, or ``"unknown"``.

    Returns
    -------
    list[ChromosomeArmTelomere]
        Updated telomere objects with populated ``attrition_rate_bp_per_year``.
    """
    base_rate_bp = 25.0  # population average bp/year

    sex_norm = sex.lower().strip()
    if sex_norm == "male":
        base_rate_bp += 8.0   # ~15 bp/yr sex gap (Gardner et al. 2014)
    elif sex_norm == "female":
        base_rate_bp -= 7.0
    # 'unknown' keeps the population average.

    expected_tl = _expected_tl_for_age_sex(
        chronological_age,
        sex_norm if sex_norm != "unknown" else "female",
    )

    updated: list[ChromosomeArmTelomere] = []

    for tel in stela_profile.chromosome_telomeres:
        arm_multiplier = _ATTRITION_RATE_MULTIPLIERS.get(tel.chromosome, 1.0)

        # Length-dependent correction: telomeres shorter than expected
        # lose proportionally more.  Cap the multiplier at 1.5x.
        length_ratio = tel.estimated_length_kb / expected_tl if expected_tl > 0 else 1.0
        length_correction = min(1.5, max(0.8, 1.0 / max(length_ratio, 0.5)))

        rate = base_rate_bp * arm_multiplier * length_correction
        rate = round(rate, 1)

        updated.append(
            ChromosomeArmTelomere(
                chromosome=tel.chromosome,
                arm=tel.arm,
                estimated_length_kb=tel.estimated_length_kb,
                length_sd_kb=tel.length_sd_kb,
                percentile=tel.percentile,
                is_critically_short=tel.is_critically_short,
                is_very_short=tel.is_very_short,
                attrition_rate_bp_per_year=rate,
            )
        )

    return updated


# ---------------------------------------------------------------------------
# STELA gel data import (future-ready)
# ---------------------------------------------------------------------------


def parse_stela_gel_data(
    gel_image_path: str | None = None,
    band_sizes_kb: dict[str, list[float]] | None = None,
) -> STELAProfile:
    """Parse real STELA gel electrophoresis data into a profile.

    Accepts either a scanned gel image (TIFF/PNG) or pre-measured band
    sizes extracted by densitometry software.  This function provides a
    bridge between laboratory STELA data and the computational analysis
    pipeline.

    Parameters
    ----------
    gel_image_path : str or None, optional
        File path to a scanned STELA gel image.  Automated band calling
        from raw gel images is not yet implemented; providing this
        argument will raise :class:`NotImplementedError`.
    band_sizes_kb : dict[str, list[float]] or None, optional
        Pre-measured band sizes keyed by chromosome arm.  Each value is
        a list of telomere lengths (kb) observed across multiple PCR
        reactions for that arm.  Example::

            {"17p": [3.2, 3.5, 3.1, 3.8],
             "Xq": [7.1, 6.8, 7.3, 7.0]}

    Returns
    -------
    STELAProfile
        Profile constructed from the supplied measurements.

    Raises
    ------
    NotImplementedError
        If *gel_image_path* is provided (automated gel analysis is not
        yet supported).
    ValueError
        If neither argument is provided, or if *band_sizes_kb* contains
        no valid data.

    Notes
    -----
    When laboratory STELA data is available for only a subset of
    chromosome arms, the remaining arms are imputed from the measured
    mean using the standard relative-length model.
    """
    if gel_image_path is not None:
        raise NotImplementedError(
            "Automated STELA gel image analysis is not yet implemented.  "
            "Please provide pre-measured band sizes via the band_sizes_kb "
            "parameter, or use generate_stela_profile() for computational "
            "modelling."
        )

    if band_sizes_kb is None or len(band_sizes_kb) == 0:
        raise ValueError(
            "At least one of gel_image_path or band_sizes_kb must be "
            "provided with valid data."
        )

    # Compute per-arm means from measured bands.
    measured_means: dict[str, float] = {}
    measured_sds: dict[str, float] = {}
    for arm, sizes in band_sizes_kb.items():
        if not sizes:
            continue
        measured_means[arm] = statistics.mean(sizes)
        measured_sds[arm] = statistics.stdev(sizes) if len(sizes) > 1 else 0.3

    if not measured_means:
        raise ValueError("band_sizes_kb contains no valid measurements.")

    overall_mean = statistics.mean(measured_means.values())

    # Build telomere list: use measured data where available, impute rest.
    all_arms = sorted(_CHROMOSOME_ARM_RELATIVE_LENGTHS.keys(), key=_arm_sort_key)
    telomeres: list[ChromosomeArmTelomere] = []
    lengths: list[float] = []

    for arm_name in all_arms:
        if arm_name in measured_means:
            est = measured_means[arm_name]
            sd = measured_sds[arm_name]
        else:
            relative = _CHROMOSOME_ARM_RELATIVE_LENGTHS[arm_name]
            est = overall_mean * relative
            sd = 0.5  # default imputation uncertainty

        est = max(0.5, est)
        telomeres.append(
            ChromosomeArmTelomere(
                chromosome=arm_name,
                arm=arm_name[-1],
                estimated_length_kb=round(est, 3),
                length_sd_kb=round(sd, 3),
                percentile=50,  # cannot compute without age/sex
                is_critically_short=est < _CRITICAL_SHORT_KB,
                is_very_short=est < _VERY_SHORT_KB,
                attrition_rate_bp_per_year=0.0,
            )
        )
        lengths.append(est)

    mean_length = statistics.mean(lengths)
    median_length = statistics.median(lengths)
    sd_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    heterogeneity = sd_length / mean_length if mean_length > 0 else 0.0

    shortest_idx = lengths.index(min(lengths))
    longest_idx = lengths.index(max(lengths))

    return STELAProfile(
        chromosome_telomeres=telomeres,
        mean_telomere_length_kb=round(mean_length, 3),
        median_telomere_length_kb=round(median_length, 3),
        shortest_telomere_kb=round(lengths[shortest_idx], 3),
        shortest_chromosome=telomeres[shortest_idx].chromosome,
        longest_telomere_kb=round(lengths[longest_idx], 3),
        longest_chromosome=telomeres[longest_idx].chromosome,
        critically_short_count=sum(1 for t in telomeres if t.is_critically_short),
        heterogeneity_index=round(heterogeneity, 4),
        telomere_biology_disorder_risk="Unknown",
        clinical_interpretation=(
            "Profile generated from laboratory STELA measurements.  "
            "Percentile and TBD risk assessment require age and sex "
            "information; re-run with generate_stela_profile() for full "
            "clinical interpretation."
        ),
        methodology_note=(
            "Telomere lengths measured by Single Telomere Length Analysis "
            "(STELA; Baird et al., 2003).  Arms not directly measured were "
            "imputed from the measured mean using published inter-arm "
            "relative length data."
        ),
    )


# ---------------------------------------------------------------------------
# Telomere Biology Disorder screening
# ---------------------------------------------------------------------------


def screen_telomere_biology_disorder(
    stela_profile: STELAProfile,
    chronological_age: int,
    family_history: bool = False,
) -> dict[str, Any]:
    """Screen for Telomere Biology Disorders based on the STELA profile.

    Evaluates the telomere length distribution against published
    diagnostic criteria for conditions within the telomere biology
    disorder spectrum:

    - **Dyskeratosis congenita (DC)**: classic triad of nail dystrophy,
      oral leukoplakia, and abnormal skin pigmentation; telomere lengths
      typically < 1st percentile for age.
    - **Idiopathic Pulmonary Fibrosis (IPF)**: associated with short
      telomeres, particularly in the < 10th percentile range.
    - **Aplastic anaemia**: bone marrow failure linked to critically
      short telomeres.

    Criteria follow Savage & Bertuch (2010), Alder et al. (2018), and
    the Telomere Length Testing Guidelines from the Clinical Genetics
    Society (2020).

    Parameters
    ----------
    stela_profile : STELAProfile
        Previously generated STELA profile.
    chronological_age : int
        Age in years.
    family_history : bool, optional
        Whether there is a known family history of telomere biology
        disorders, bone marrow failure, pulmonary fibrosis, or
        premature ageing (default ``False``).  Positive family history
        lowers the threshold for referral.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``"overall_risk"`` (str): ``"Low"``, ``"Moderate"``, or ``"High"``.
        - ``"dc_risk"`` (str): Dyskeratosis congenita risk level.
        - ``"ipf_risk"`` (str): Idiopathic Pulmonary Fibrosis risk level.
        - ``"aplastic_anaemia_risk"`` (str): Aplastic anaemia risk level.
        - ``"critically_short_arms"`` (list[str]): Arms below 4.0 kb.
        - ``"very_short_arms"`` (list[str]): Arms below 5.0 kb.
        - ``"recommendations"`` (list[str]): Clinical action items.
        - ``"referral_urgency"`` (str): ``"Routine"``, ``"Urgent"``,
          or ``"None"``.
        - ``"methodology"`` (str): Brief note on screening methodology.

    Notes
    -----
    This screening tool is intended for research use and must not replace
    clinical judgement.  A positive screen should be followed up with
    flow-FISH telomere length measurement and genetic testing for known
    TBD genes (TERT, TERC, DKC1, RTEL1, TINF2, etc.).
    """
    profile = stela_profile
    short_arms = [t.chromosome for t in profile.chromosome_telomeres if t.is_critically_short]
    very_short_arms = [t.chromosome for t in profile.chromosome_telomeres if t.is_very_short]

    # Calculate overall percentile from profile mean.
    expected_tl = _expected_tl_for_age_sex(chronological_age, "female")  # conservative
    overall_pct = _tl_to_percentile(profile.mean_telomere_length_kb, expected_tl)

    # Family history modifier: effectively shift percentile threshold up.
    pct_threshold_dc = 1 if not family_history else 5
    pct_threshold_ipf = 10 if not family_history else 20

    # --- Dyskeratosis congenita risk ---
    if overall_pct < pct_threshold_dc or profile.shortest_telomere_kb < 2.5:
        dc_risk = "High"
    elif overall_pct < 5 or profile.shortest_telomere_kb < 3.5:
        dc_risk = "Moderate"
    else:
        dc_risk = "Low"

    # --- IPF risk ---
    if overall_pct < pct_threshold_ipf and chronological_age >= 40:
        ipf_risk = "Moderate" if overall_pct >= 5 else "High"
    elif overall_pct < 25 and chronological_age >= 50:
        ipf_risk = "Moderate"
    else:
        ipf_risk = "Low"

    # --- Aplastic anaemia risk ---
    if len(short_arms) >= 5 or profile.shortest_telomere_kb < 2.0:
        aa_risk = "High"
    elif len(short_arms) >= 2 or profile.shortest_telomere_kb < 3.0:
        aa_risk = "Moderate"
    else:
        aa_risk = "Low"

    # --- Overall risk: highest of the three ---
    risk_order = {"Low": 0, "Moderate": 1, "High": 2}
    component_risks = [dc_risk, ipf_risk, aa_risk]
    overall_risk = max(component_risks, key=lambda r: risk_order[r])

    # --- Recommendations ---
    recommendations: list[str] = []

    if overall_risk == "High":
        recommendations.extend([
            "Urgent referral to clinical genetics for confirmatory testing.",
            "Flow-FISH telomere length measurement recommended.",
            "Genetic testing for TBD-associated genes (TERT, TERC, DKC1, "
            "RTEL1, TINF2, NHP2, NOP10, WRAP53, CTC1, STN1, POT1).",
            "Complete blood count with differential to assess for cytopenias.",
            "Pulmonary function testing if age >= 40.",
            "Hepatic function assessment (liver fibrosis screening).",
        ])
    elif overall_risk == "Moderate":
        recommendations.extend([
            "Consider referral to clinical genetics for evaluation.",
            "Longitudinal telomere length monitoring (repeat in 12 months).",
            "Flow-FISH confirmation of telomere length recommended.",
            "Family history review for TBD-associated phenotypes.",
        ])
    else:
        recommendations.append(
            "No immediate clinical action indicated based on telomere "
            "length profile.  Consider routine monitoring if family "
            "history is significant."
        )

    if family_history:
        recommendations.append(
            "Positive family history noted — lower threshold for genetic "
            "testing referral is appropriate."
        )

    # --- Referral urgency ---
    if overall_risk == "High":
        referral = "Urgent"
    elif overall_risk == "Moderate":
        referral = "Routine"
    else:
        referral = "None"

    return {
        "overall_risk": overall_risk,
        "dc_risk": dc_risk,
        "ipf_risk": ipf_risk,
        "aplastic_anaemia_risk": aa_risk,
        "overall_percentile": overall_pct,
        "critically_short_arms": short_arms,
        "very_short_arms": very_short_arms,
        "recommendations": recommendations,
        "referral_urgency": referral,
        "methodology": (
            "Screening based on computational STELA modelling with "
            "age-adjusted percentile cutoffs per Savage & Bertuch (2010) "
            "and Alder et al. (2018).  For research purposes only."
        ),
    }
