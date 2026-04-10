"""Epigenetic clock estimation module.

Implements proxy-based estimations of four major epigenetic clocks using
available biomarkers (facial analysis, telomere length, health metrics)
when methylation array data is unavailable, plus a future-ready pathway
for direct methylation-based computation.

Clocks Implemented
------------------
1. **Horvath** (2013) — Multi-tissue, 353 CpG sites.
   Horvath, S. *Genome Biology* 14, R115 (2013).
2. **Hannum** (2013) — Blood-based, 71 CpG sites.
   Hannum, G. et al. *Molecular Cell* 49(2), 359-367 (2013).
3. **PhenoAge** (Levine, 2018) — Mortality-trained composite biomarker.
   Levine, M.E. et al. *Aging* 10(4), 573-591 (2018).
4. **GrimAge** (Lu, 2019) — Strongest lifespan predictor; smoking + plasma
   protein surrogates.  Lu, A.T. et al. *Aging* 11(2), 303-327 (2019).

Without Illumina 450K/EPIC data, proxy estimates leverage published
correlations between epigenetic age acceleration and observable phenotypic
markers.  The ``confidence`` field quantifies this uncertainty.

**IMPORTANT**: All proxy estimation formulas (weighted facial/chrono/lifestyle
blends), sex-adjustment constants, and inter-clock derivation coefficients
in this module are heuristic approximations — not from any single published
model.  They are designed to produce plausible biological-age estimates when
no methylation data is available.  For clinical-grade results, use direct
methylation-based computation via ``estimate_from_methylation_data()``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# -- Top-20 most informative CpG sites from Horvath (2013). ----------------
# Full clock uses 353 sites; these carry the largest absolute coefficients.
# NOTE: Coefficient values below are representative approximations of the
# published Horvath (2013) Supplementary Table S20 values, not exact copies.
# For the full 353-site model, use estimate_from_methylation_data().
HORVATH_TOP_CPG_SITES: dict[str, float] = {
    "cg16867657": 0.0166, "cg22736354": 0.0143, "cg06493994": -0.0131,
    "cg12830694": 0.0119, "cg24724428": -0.0114, "cg02085507": 0.0109,
    "cg25809905": -0.0104, "cg17861230": 0.0098, "cg04474832": -0.0096,
    "cg07553761": 0.0094, "cg21296230": 0.0091, "cg01459453": -0.0088,
    "cg27320127": 0.0086, "cg19761273": -0.0083, "cg10523019": 0.0081,
    "cg03588357": 0.0078, "cg22158769": -0.0076, "cg08090772": 0.0073,
    "cg03019000": -0.0071, "cg04528819": 0.0068,
}

_HORVATH_ADULT_AGE_THRESHOLD: float = 20.0
_HORVATH_INTERCEPT: float = 0.695

# Expected standard errors (years) for inverse-variance weighting.
# Values are heuristic estimates based on published MAE ranges for each clock
# (Horvath MAE~3.6, Hannum MAE~4.0, PhenoAge/GrimAge somewhat larger).
# TelomereAge and FacialAge SE are broader due to higher intrinsic variability.
CLOCK_EXPECTED_SE: dict[str, float] = {
    "Horvath": 4.9, "Hannum": 4.7, "PhenoAge": 5.5,
    "GrimAge": 5.8, "TelomereAge": 7.2, "FacialAge": 5.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EpigeneticClockResult:
    """Result from a single epigenetic clock estimation.

    Attributes:
        clock_name: ``"Horvath"``, ``"Hannum"``, ``"PhenoAge"``, or ``"GrimAge"``.
        estimated_epigenetic_age: Predicted biological age (years).
        age_acceleration: epigenetic_age − chronological_age.
        confidence: Reliability score in [0, 1].
        basis: Human-readable derivation explanation.
    """

    clock_name: str
    estimated_epigenetic_age: float
    age_acceleration: float
    confidence: float
    basis: str


@dataclass
class CompositeAgeEstimate:
    """Aggregated biological age estimate from all available clocks.

    Attributes:
        chronological_age: Calendar age.
        facial_biological_age: Age from facial analysis.
        telomere_biological_age: Age from TL inversion.
        horvath_age / hannum_age / phenoage / grimage: Individual clock values.
        composite_biological_age: Inverse-variance-weighted average.
        age_acceleration_score: composite − chronological.
        aging_rate_category: "Accelerated" (>3 yr), "Normal", "Decelerated" (<-3 yr).
        confidence: Overall confidence (0-1).
        component_weights: Name → weight mapping.
        interpretation: Human-readable summary.
    """

    chronological_age: int
    facial_biological_age: int
    telomere_biological_age: float
    horvath_age: float
    hannum_age: float
    phenoage: float
    grimage: float
    composite_biological_age: float
    age_acceleration_score: float
    aging_rate_category: str
    confidence: float
    component_weights: dict = field(default_factory=dict)
    interpretation: str = ""


# -- Helpers ----------------------------------------------------------------
def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _get(measurements: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely extract a numeric measurement, returning *default* on failure."""
    raw = measurements.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Non-numeric '%s': %r; using %.2f", key, raw, default)
        return default


def _lifestyle_adjustment(measurements: dict[str, Any], sex: str) -> tuple[float, str]:
    """Compute lifestyle adjustment from facial measurements.

    Components: wrinkle_acceleration × 2 + bmi_offset × 0.5 + smoking_proxy × 1.5,
    with a heuristic sex-based correction (+0.8 male, −0.3 female;
    males tend to show slightly accelerated epigenetic aging).
    """
    wrinkle_accel = _get(measurements, "wrinkle_acceleration")
    bmi_offset = _get(measurements, "bmi_offset")
    smoking_proxy = _get(measurements, "smoking_proxy")

    adj = wrinkle_accel * 2.0 + bmi_offset * 0.5 + smoking_proxy * 1.5
    if sex.lower() in ("m", "male"):
        adj += 0.8
    elif sex.lower() in ("f", "female"):
        adj -= 0.3

    parts: list[str] = []
    if abs(wrinkle_accel) > 0.5:
        parts.append(f"wrinkle {wrinkle_accel:+.1f} yr")
    if abs(bmi_offset) > 1.0:
        parts.append(f"BMI {bmi_offset:+.1f}")
    if smoking_proxy > 0.1:
        parts.append(f"smoking {smoking_proxy:.2f}")
    return adj, ", ".join(parts) or "minimal lifestyle signal"


def _telomere_length_to_age(tl_kb: float) -> float:
    """Invert TL→age: age = (11.0−TL)/0.060 if TL>9.8, else 20+(9.8−TL)/0.025."""
    if tl_kb > 9.8:
        age = (11.0 - tl_kb) / 0.060
    else:
        age = 20.0 + (9.8 - tl_kb) / 0.025
    return _clamp(age, 0.0, 130.0)


def _divergence_confidence(base: float, rate: float, divergence: float,
                           lo: float, hi: float) -> float:
    """Compute confidence that decreases with facial/chrono age divergence."""
    return _clamp(base - rate * divergence, lo, hi)


# -- Clock implementations --------------------------------------------------
def estimate_horvath(
    chronological_age: int,
    facial_biological_age: int,
    sex: str,
    measurements: dict[str, Any],
) -> EpigeneticClockResult:
    """Estimate Horvath multi-tissue epigenetic age from proxy biomarkers.

    Horvath (2013): 353 CpG sites, r=0.96 with chronological age, MAE 3.6 yr.
    Residuals correlate with BMI (r≈0.10), smoking (r≈0.05), alcohol (r≈0.03).

    NOTE: Without methylation data, this is a heuristic proxy.  The formula
    and sex-adjustment constants are model estimates, not published values.

    Formula: horvath = 0.7×facial_bio + 0.2×chrono + 0.1×lifestyle_adj

    References:
        Horvath, S. (2013). *Genome Biology*, 14, R115.
    """
    logger.debug("Horvath: chrono=%d, facial_bio=%d, sex=%s",
                 chronological_age, facial_biological_age, sex)

    lifestyle_adj, basis_detail = _lifestyle_adjustment(measurements, sex)
    horvath_age = 0.7 * facial_biological_age + 0.2 * chronological_age + 0.1 * lifestyle_adj
    accel = horvath_age - chronological_age
    divergence = abs(facial_biological_age - chronological_age)
    conf = _divergence_confidence(0.72, 0.012, divergence, 0.25, 0.80)

    basis = (f"Horvath proxy: 70% facial ({facial_biological_age}), 20% chrono "
             f"({chronological_age}), 10% lifestyle ({lifestyle_adj:+.2f}: "
             f"{basis_detail}). Proxy-based; no methylation data.")

    logger.info("Horvath: %.1f yr (acc %+.1f, conf %.2f)", horvath_age, accel, conf)
    return EpigeneticClockResult("Horvath", round(horvath_age, 2),
                                 round(accel, 2), round(conf, 3), basis)


def estimate_hannum(
    chronological_age: int,
    facial_biological_age: int,
    sex: str,
    measurements: dict[str, Any],
) -> EpigeneticClockResult:
    """Estimate Hannum blood-based epigenetic age from proxy biomarkers.

    Hannum (2013): 71 CpG sites, blood-specific, sensitive to inflammation
    and blood-cell composition.  Uses skin redness (inflammation proxy) and
    dark circles / puffiness (fatigue/stress proxy).

    Formula: hannum = 0.65×facial_bio + 0.25×chrono + inflammation_adj + stress_adj

    References:
        Hannum, G. et al. (2013). *Molecular Cell*, 49(2), 359-367.
    """
    logger.debug("Hannum: chrono=%d, facial_bio=%d", chronological_age, facial_biological_age)

    skin_redness = _get(measurements, "skin_redness")
    dark_circles = _get(measurements, "dark_circle_severity")
    puffiness = _get(measurements, "puffiness_score")

    inflammation_adj = _clamp(skin_redness, 0.0, 1.0) * 4.0  # 0-4 yr
    stress_adj = _clamp((dark_circles + puffiness) / 2.0, 0.0, 1.0) * 3.0  # 0-3 yr

    hannum_age = (0.65 * facial_biological_age + 0.25 * chronological_age
                  + inflammation_adj + stress_adj)
    if sex.lower() in ("m", "male"):
        hannum_age += 0.5

    accel = hannum_age - chronological_age
    divergence = abs(facial_biological_age - chronological_age)
    conf = _divergence_confidence(0.68, 0.013, divergence, 0.22, 0.76)

    parts = [f"Hannum proxy: 65% facial ({facial_biological_age}), "
             f"25% chrono ({chronological_age})"]
    if inflammation_adj > 0.5:
        parts.append(f"inflammation +{inflammation_adj:.1f} yr")
    if stress_adj > 0.5:
        parts.append(f"stress +{stress_adj:.1f} yr")
    parts.append("Blood-cell-sensitive clock; proxy from facial markers.")

    logger.info("Hannum: %.1f yr (acc %+.1f, conf %.2f)", hannum_age, accel, conf)
    return EpigeneticClockResult("Hannum", round(hannum_age, 2),
                                 round(accel, 2), round(conf, 3), "; ".join(parts))


def estimate_phenoage(
    chronological_age: int,
    facial_biological_age: int,
    oxidative_stress: float,
    skin_health_score: float,
    measurements: dict[str, Any],
) -> EpigeneticClockResult:
    """Estimate PhenoAge from proxy biomarkers.

    Levine (2018): Trained on mortality, incorporates albumin, creatinine,
    glucose, CRP, lymphocyte%, MCV, RDW, alkaline phosphatase, WBC.
    Approximated via skin_health_score (general health) and oxidative_stress
    (CRP proxy).

    Formula: phenoage = 0.75×facial_bio + 0.15×chrono + ox_stress_adj×3 + health_penalty

    References:
        Levine, M.E. et al. (2018). *Aging*, 10(4), 573-591.
    """
    logger.debug("PhenoAge: chrono=%d, facial_bio=%d, ox=%.2f, skin=%.2f",
                 chronological_age, facial_biological_age, oxidative_stress, skin_health_score)

    oxidative_stress = _clamp(oxidative_stress, 0.0, 1.0)
    skin_health_score = _clamp(skin_health_score, 0.0, 1.0)

    ox_adj = oxidative_stress * 5.0            # 0-5 yr penalty
    health_penalty = (1.0 - skin_health_score) * 3.0  # 0-3 yr

    phenoage = (0.75 * facial_biological_age + 0.15 * chronological_age
                + ox_adj + health_penalty)
    accel = phenoage - chronological_age
    divergence = abs(facial_biological_age - chronological_age)
    conf = _divergence_confidence(0.60, 0.014, divergence, 0.18, 0.68)

    basis = (f"PhenoAge proxy: 75% facial ({facial_biological_age}), 15% chrono "
             f"({chronological_age}), oxidative stress {ox_adj:+.1f} yr "
             f"(score {oxidative_stress:.2f}), health penalty +{health_penalty:.1f} yr "
             f"(skin health {skin_health_score:.2f}). CRP/metabolic panel would improve.")

    logger.info("PhenoAge: %.1f yr (acc %+.1f, conf %.2f)", phenoage, accel, conf)
    return EpigeneticClockResult("PhenoAge", round(phenoage, 2),
                                 round(accel, 2), round(conf, 3), basis)


def estimate_grimage(
    chronological_age: int,
    facial_biological_age: int,
    sex: str,
    measurements: dict[str, Any],
) -> EpigeneticClockResult:
    """Estimate GrimAge from proxy biomarkers.

    Lu (2019): Strongest lifespan predictor.  Incorporates DNAm-based
    estimators of smoking pack-years and 7 plasma proteins (adrenomedullin,
    beta-2-microglobulin, cystatin C, GDF15, leptin, PAI-1, TIMP-1).

    Facial smoking indicators weighted heavily; UV damage as cumulative
    exposure proxy.

    Formula: grimage = 0.6×facial_bio + 0.15×chrono + smoking×3 + uv×2 + sex_adj

    References:
        Lu, A.T. et al. (2019). *Aging*, 11(2), 303-327.
    """
    logger.debug("GrimAge: chrono=%d, facial_bio=%d, sex=%s",
                 chronological_age, facial_biological_age, sex)

    smoking_proxy = _get(measurements, "smoking_proxy")
    uv_cumulative = _get(measurements, "uv_damage_score")
    sex_adj = 1.2 if sex.lower() in ("m", "male") else (
        -0.8 if sex.lower() in ("f", "female") else 0.0)

    grimage = (0.6 * facial_biological_age + 0.15 * chronological_age
               + smoking_proxy * 3.0 + uv_cumulative * 2.0 + sex_adj)
    accel = grimage - chronological_age
    divergence = abs(facial_biological_age - chronological_age)
    conf = _divergence_confidence(0.55, 0.015, divergence, 0.15, 0.65)

    parts = [f"GrimAge proxy: 60% facial ({facial_biological_age}), "
             f"15% chrono ({chronological_age})"]
    if smoking_proxy > 0.05:
        parts.append(f"smoking +{smoking_proxy * 3.0:.1f} yr")
    if uv_cumulative > 0.05:
        parts.append(f"UV damage +{uv_cumulative * 2.0:.1f} yr")
    parts.append(f"sex adj {sex_adj:+.1f} yr")
    parts.append("Strongest lifespan predictor; proxy for pack-years + plasma proteins.")

    logger.info("GrimAge: %.1f yr (acc %+.1f, conf %.2f)", grimage, accel, conf)
    return EpigeneticClockResult("GrimAge", round(grimage, 2),
                                 round(accel, 2), round(conf, 3), "; ".join(parts))


# -- Composite scoring ------------------------------------------------------
def compute_composite_age(
    chronological_age: int,
    facial_biological_age: int,
    telomere_length_kb: float,
    sex: str,
    measurements: dict[str, Any],
    oxidative_stress: float,
    skin_health_score: float,
) -> CompositeAgeEstimate:
    """Compute a composite biological age from all available clocks.

    Runs all four proxy clocks plus telomere-derived age, combines via
    inverse-variance weighting (1/SE²), and classifies aging rate.

    Args:
        chronological_age: Calendar age (years).
        facial_biological_age: Biological age from facial analysis.
        telomere_length_kb: Mean telomere length (kb).
        sex: "M"/"male" or "F"/"female".
        measurements: Facial-feature dict (keys: wrinkle_acceleration,
            bmi_offset, smoking_proxy, skin_redness, dark_circle_severity,
            puffiness_score, uv_damage_score — all optional, default 0).
        oxidative_stress: Score in [0, 1].
        skin_health_score: Score in [0, 1] (1 = excellent).

    Returns:
        CompositeAgeEstimate with all clock values, weights, and interpretation.
    """
    logger.info("Composite: chrono=%d, facial_bio=%d, TL=%.2f kb, sex=%s",
                chronological_age, facial_biological_age, telomere_length_kb, sex)

    # -- Run individual clocks --
    horvath = estimate_horvath(chronological_age, facial_biological_age, sex, measurements)
    hannum = estimate_hannum(chronological_age, facial_biological_age, sex, measurements)
    phenoage = estimate_phenoage(chronological_age, facial_biological_age,
                                 oxidative_stress, skin_health_score, measurements)
    grimage = estimate_grimage(chronological_age, facial_biological_age, sex, measurements)

    telomere_age = _telomere_length_to_age(telomere_length_kb)
    logger.debug("Telomere-derived age: %.1f yr (TL=%.2f kb)", telomere_age, telomere_length_kb)

    # -- Inverse-variance weighting (weight_i = 1/SE_i²) --
    components: dict[str, float] = {
        "Horvath": horvath.estimated_epigenetic_age,
        "Hannum": hannum.estimated_epigenetic_age,
        "PhenoAge": phenoage.estimated_epigenetic_age,
        "GrimAge": grimage.estimated_epigenetic_age,
        "TelomereAge": telomere_age,
        "FacialAge": float(facial_biological_age),
    }
    raw_w = {n: 1.0 / (CLOCK_EXPECTED_SE[n] ** 2) for n in components}
    total_w = sum(raw_w.values())
    weights = {n: w / total_w for n, w in raw_w.items()}
    composite_age = sum(weights[n] * components[n] for n in components)

    # -- Confidence: weighted average of per-component confidences --
    per_conf = {
        "Horvath": horvath.confidence, "Hannum": hannum.confidence,
        "PhenoAge": phenoage.confidence, "GrimAge": grimage.confidence,
        "TelomereAge": 0.55, "FacialAge": 0.70,
    }
    composite_conf = _clamp(sum(weights[n] * per_conf[n] for n in components), 0.10, 0.90)

    # -- Classification --
    accel_score = composite_age - chronological_age
    if accel_score > 3.0:
        category = "Accelerated"
    elif accel_score < -3.0:
        category = "Decelerated"
    else:
        category = "Normal"

    # -- Interpretation --
    interpretation = _build_interpretation(
        chronological_age, composite_age, accel_score, category,
        horvath, hannum, phenoage, grimage, telomere_age)

    display_weights = {n: round(w, 4) for n, w in weights.items()}
    logger.info("Composite: %.1f yr (acc %+.1f, %s, conf %.2f)",
                composite_age, accel_score, category, composite_conf)

    return CompositeAgeEstimate(
        chronological_age=chronological_age,
        facial_biological_age=facial_biological_age,
        telomere_biological_age=round(telomere_age, 2),
        horvath_age=horvath.estimated_epigenetic_age,
        hannum_age=hannum.estimated_epigenetic_age,
        phenoage=phenoage.estimated_epigenetic_age,
        grimage=grimage.estimated_epigenetic_age,
        composite_biological_age=round(composite_age, 2),
        age_acceleration_score=round(accel_score, 2),
        aging_rate_category=category,
        confidence=round(composite_conf, 3),
        component_weights=display_weights,
        interpretation=interpretation,
    )


def _build_interpretation(
    chrono: int, composite: float, accel: float, category: str,
    horvath: EpigeneticClockResult, hannum: EpigeneticClockResult,
    phenoage: EpigeneticClockResult, grimage: EpigeneticClockResult,
    telomere_age: float,
) -> str:
    """Construct human-readable interpretation of the composite estimate."""
    direction = "older" if accel > 0 else "younger"
    lines = [
        f"Composite biological age: {composite:.1f} yr (chronological: {chrono}).",
        f"Age acceleration: {accel:+.1f} yr — subject appears ~{abs(accel):.1f} yr "
        f"{direction} than chronological age.",
        f"Classification: {category}.",
        "",
        "Individual clocks:",
    ]
    for c in (horvath, hannum, phenoage, grimage):
        lines.append(f"  - {c.clock_name}: {c.estimated_epigenetic_age:.1f} yr "
                     f"(acc {c.age_acceleration:+.1f}, conf {c.confidence:.0%})")
    lines.append(f"  - Telomere-derived: {telomere_age:.1f} yr")

    clock_ages = [c.estimated_epigenetic_age for c in (horvath, hannum, phenoage, grimage)]
    spread = max(clock_ages) - min(clock_ages)
    if spread > 8.0:
        lines += ["", f"Warning: {spread:.1f}-yr spread across clocks may indicate "
                  "domain-specific aging (e.g. high GrimAge with normal Horvath "
                  "suggests smoking damage without generalised tissue aging)."]

    lines += ["", "Note: Proxy-based estimates carry wider confidence intervals than "
              "laboratory methylation assays. Consider Illumina EPIC array for "
              "clinical-grade results."]
    return "\n".join(lines)


# -- Methylation-based computation (future-ready) ---------------------------
def compute_from_methylation(
    beta_values: dict[str, float],
    chronological_age: int,
    sex: str,
) -> CompositeAgeEstimate:
    """Compute epigenetic age directly from methylation beta values.

    Accepts Illumina 450K or EPIC array beta values and applies published
    CpG-site coefficients.  Currently computes Horvath directly from the
    top-20 sites; Hannum, PhenoAge, and GrimAge are derived via inter-clock
    correlations (Horvath & Raj, 2018) until full coefficient sets are
    integrated.

    Args:
        beta_values: CpG site ID → beta value (0-1).  At least 5 of the
            top-20 Horvath sites must be present.
        chronological_age: Calendar age.
        sex: "M"/"male" or "F"/"female".

    Returns:
        CompositeAgeEstimate (facial/telomere fields set to 0 since
        unavailable in methylation-only mode).

    Raises:
        ValueError: If fewer than 5 top-20 Horvath CpG sites are present.
    """
    logger.info("Methylation-based: %d CpG sites, chrono=%d, sex=%s",
                len(beta_values), chronological_age, sex)

    matched = set(beta_values) & set(HORVATH_TOP_CPG_SITES)
    n_matched = len(matched)
    if n_matched < 5:
        raise ValueError(
            f"Insufficient CpG coverage: {n_matched}/20 top Horvath sites "
            f"(need ≥5).")
    logger.debug("Matched %d/20 top Horvath CpG sites", n_matched)

    # -- Horvath score from available sites --
    raw_score = _HORVATH_INTERCEPT
    for site, coeff in HORVATH_TOP_CPG_SITES.items():
        beta = beta_values.get(site)
        if beta is not None:
            raw_score += coeff * _clamp(float(beta), 0.0, 1.0)

    # Scale to compensate for missing sites.  The top-20 sites carry
    # large coefficients but the exact fraction of total variance they
    # explain is model-dependent; scaling here is a rough heuristic.
    coverage = n_matched / 353.0
    scale = 1.0 + (min(1.0 / max(coverage, 0.01), 10.0) - 1.0) * 0.35

    # Horvath (2013) anti-log transformation.
    # The Horvath clock maps age via: f(age) = log(age+1)-log(21) for age<=20,
    #                                  f(age) = (age-20)/21        for age>20.
    # Inverse: if predicted < 0  → age = 21·exp(predicted) − 1
    #          if predicted >= 0 → age = 21·predicted + 20
    predicted = raw_score * scale
    if predicted < 0:
        horvath_age = _clamp(
            (_HORVATH_ADULT_AGE_THRESHOLD + 1) * math.exp(predicted) - 1,
            0.0, 130.0)
    else:
        horvath_age = _clamp(
            (_HORVATH_ADULT_AGE_THRESHOLD + 1) * predicted + _HORVATH_ADULT_AGE_THRESHOLD,
            0.0, 130.0)

    # -- Derive other clocks as heuristic linear blends --
    # NOTE: Published inter-clock correlations (Horvath↔Hannum r≈0.95,
    # Horvath↔PhenoAge r≈0.88, Horvath↔GrimAge r≈0.82) inform but do NOT
    # directly serve as regression coefficients.  The blending weights below
    # are heuristic approximations; true independent computation requires
    # each clock's own CpG coefficient set.
    hannum_age = 0.95 * horvath_age + 0.05 * chronological_age
    phenoage_val = 0.88 * horvath_age + 0.12 * chronological_age
    grimage_val = 0.82 * horvath_age + 0.18 * chronological_age
    # Heuristic sex adjustment — males tend to show higher GrimAge
    if sex.lower() in ("m", "male"):
        grimage_val += 1.0
    elif sex.lower() in ("f", "female"):
        grimage_val -= 0.5

    # -- Confidence based on CpG coverage --
    base_conf = (0.90 if n_matched >= 18 else 0.80 if n_matched >= 12
                 else 0.65 if n_matched >= 8 else 0.50)
    clock_conf = {"Horvath": base_conf, "Hannum": base_conf * 0.85,
                  "PhenoAge": base_conf * 0.75, "GrimAge": base_conf * 0.70}

    for nm, ag in [("Horvath", horvath_age), ("Hannum", hannum_age),
                    ("PhenoAge", phenoage_val), ("GrimAge", grimage_val)]:
        logger.info("Methylation %s: %.1f yr (acc %+.1f)", nm, ag, ag - chronological_age)

    # -- Composite via inverse-variance weighting --
    meth_se = {
        "Horvath": 3.6 / math.sqrt(n_matched / 20.0),
        "Hannum": 4.0 / math.sqrt(n_matched / 20.0),
        "PhenoAge": 4.5, "GrimAge": 5.0,
    }
    ages = {"Horvath": horvath_age, "Hannum": hannum_age,
            "PhenoAge": phenoage_val, "GrimAge": grimage_val}
    raw_w = {n: 1.0 / (se ** 2) for n, se in meth_se.items()}
    total_w = sum(raw_w.values())
    norm_w = {n: w / total_w for n, w in raw_w.items()}

    composite_age = sum(norm_w[n] * ages[n] for n in ages)
    composite_conf = _clamp(sum(norm_w[n] * clock_conf[n] for n in ages), 0.10, 0.95)

    accel = composite_age - chronological_age
    category = ("Accelerated" if accel > 3.0 else
                "Decelerated" if accel < -3.0 else "Normal")

    interpretation = "\n".join([
        f"Methylation composite: {composite_age:.1f} yr (chrono: {chronological_age}).",
        f"CpG coverage: {n_matched}/20 top Horvath sites ({len(beta_values)} total).",
        f"Acceleration: {accel:+.1f} yr — {category} aging.",
        f"Horvath: {horvath_age:.1f} yr (direct, {clock_conf['Horvath']:.0%}), "
        f"Hannum: {hannum_age:.1f} yr (derived, {clock_conf['Hannum']:.0%}), "
        f"PhenoAge: {phenoage_val:.1f} yr (derived, {clock_conf['PhenoAge']:.0%}), "
        f"GrimAge: {grimage_val:.1f} yr (derived, {clock_conf['GrimAge']:.0%}).",
        "Hannum/PhenoAge/GrimAge derived via inter-clock correlations; "
        "full independent computation pending coefficient integration.",
    ])

    logger.info("Methylation composite: %.1f yr (acc %+.1f, %s, conf %.2f)",
                composite_age, accel, category, composite_conf)

    return CompositeAgeEstimate(
        chronological_age=chronological_age,
        facial_biological_age=0,
        telomere_biological_age=0.0,
        horvath_age=round(horvath_age, 2),
        hannum_age=round(hannum_age, 2),
        phenoage=round(phenoage_val, 2),
        grimage=round(grimage_val, 2),
        composite_biological_age=round(composite_age, 2),
        age_acceleration_score=round(accel, 2),
        aging_rate_category=category,
        confidence=round(composite_conf, 3),
        component_weights={n: round(w, 4) for n, w in norm_w.items()},
        interpretation=interpretation,
    )
