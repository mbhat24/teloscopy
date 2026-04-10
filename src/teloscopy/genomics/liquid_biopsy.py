"""Cell-free DNA (cfDNA) telomere content estimation for liquid biopsy.

Minimally invasive telomere length analysis from peripheral blood draws.
cfDNA — short nucleosomal fragments (~167 bp) shed into plasma by apoptotic
and necrotic cells — carries telomeric sequences whose abundance reflects the
weighted telomere lengths of contributing tissues.

Capabilities:
    - Tissue-of-origin deconvolution of cfDNA telomere signal
    - Tumor-fraction-aware telomere length estimation
    - Serial monitoring with trend detection for treatment response
    - WGS-based cfDNA telomere length calculation (TelomereHunter/TelSeq)

References:
    Nersisyan, S. et al. (2023). *Scientific Reports*, 13, 4212.
    Moss, J. et al. (2018). *Nature Communications*, 9, 5068.
    Wan, J.C.M. et al. (2017). *Nature Reviews Cancer*, 17(4), 223-238.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Constants — tissue-of-origin model
# ---------------------------------------------------------------------------

# Normal cfDNA tissue contributions from genome-wide methylation deconvolution
# (Moss et al., 2018, Nature Communications 9:5068).
_NORMAL_TISSUE_FRACTIONS: dict[str, float] = {
    "hematopoietic": 0.76,          # ~76% from white blood cells (leukocyte turnover)
    "vascular_endothelial": 0.09,
    "hepatocyte": 0.04,
    "lung_epithelial": 0.02,
    "colon_epithelial": 0.02,
    "pancreas": 0.01,
    "breast_epithelial": 0.01,
    "kidney": 0.01,
    "brain": 0.005,
    "skeletal_muscle": 0.005,
    "adipose": 0.005,
    "other": 0.025,
}

# Tissue-specific telomere length offsets relative to leukocyte TL (kb).
# Compiled from Southern blot / qPCR surveys (Demanelis et al., 2020).
_TISSUE_TL_OFFSETS_KB: dict[str, float] = {
    "hematopoietic": 0.0,           # reference tissue
    "vascular_endothelial": -0.5,    # shorter due to replicative stress
    "hepatocyte": +0.8,             # longer, low turnover
    "lung_epithelial": -0.3,
    "colon_epithelial": -0.8,        # short, high turnover
    "pancreas": +0.5,
    "breast_epithelial": -0.2,
    "kidney": +0.3,
    "brain": +1.2,                   # longest, very low turnover
    "skeletal_muscle": +0.6,
    "adipose": +0.2,
    "other": 0.0,
}

# Tumor cfDNA fractions by stage (Wan et al., 2017; Bettegowda et al., 2014).
_TUMOR_FRACTION_RANGES: dict[tuple[str | None, str], tuple[float, float]] = {
    (None, "I"):   (0.0001, 0.001),  # 0.01-0.1%
    (None, "II"):  (0.001,  0.01),
    (None, "III"): (0.01,   0.10),
    (None, "IV"):  (0.05,   0.40),
    ("lung",   "I"): (0.0001, 0.005), ("lung",   "IV"): (0.10, 0.50),
    ("breast", "I"): (0.0001, 0.002), ("breast", "IV"): (0.05, 0.30),
    ("colon",  "I"): (0.0005, 0.005), ("colon",  "IV"): (0.10, 0.45),
    ("liver",  "I"): (0.001,  0.01),  ("liver",  "IV"): (0.15, 0.50),
}

# Tumor TL offsets relative to matched normal tissue (kb).
_TUMOR_TL_OFFSETS: dict[str, float] = {
    "lung": -1.5, "breast": -1.0, "colon": -2.0,  # Nersisyan et al., 2023
    "liver": -1.2, "prostate": -0.8, "pancreas": -1.8,
    "brain": +0.5,  # ALT-pathway gliomas can be longer
    "melanoma": -0.5, "kidney": -0.6, "ovary": -1.0,
}

_TREATMENT_FRACTION_MULTIPLIERS: dict[str, float] = {
    "pre_treatment": 1.0, "on_treatment": 0.3, "post_treatment": 0.1,
}

_NORMAL_ATTRITION_BP_PER_YEAR: float = 24.7  # Muezzinler et al., 2013 meta
_CFDNA_FRAGMENT_SIZE: int = 167               # nucleosomal fragment size (bp)
_CHROMOSOME_ENDS: int = 92                    # 46 chromosomes x 2 ends


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TissueContribution:
    """Single tissue's contribution to cfDNA telomere signal."""
    tissue: str                        # e.g. "hematopoietic", "liver", "lung"
    fraction: float                    # proportion of total cfDNA
    estimated_telomere_length_kb: float
    confidence: float                  # 0-1, lower when tumor-confounded


@dataclass
class CfDNATelomereResult:
    """Complete cfDNA-based telomere content estimation result."""
    total_cfdna_telomere_content: float       # T/S ratio or normalized score
    estimated_mean_telomere_length_kb: float
    tissue_contributions: list[TissueContribution]
    tumor_fraction_estimate: float            # 0-1, fraction of cfDNA from tumor
    is_tumor_confounded: bool                 # True if tumor fraction > 5%
    serial_change: float | None               # change from previous measurement
    serial_trend: str                         # "Stable", "Shortening", "Lengthening", "Unknown"
    clinical_interpretation: str
    methodology_note: str


@dataclass
class SerialMonitoringResult:
    """Longitudinal cfDNA telomere monitoring result."""
    timepoints: list[dict[str, Any]]          # [{date, tl_kb, tumor_fraction, treatment_phase}]
    attrition_rate_kb_per_year: float
    trend: str                                # "Accelerated shortening", "Normal", "Stable", "Lengthening"
    treatment_response_indicator: str
    visualization_data: dict[str, Any]        # data for plotting


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_tumor_fraction(
    cancer_type: str | None, tumor_stage: str | None, treatment_phase: str,
) -> float:
    """Geometric-mean tumor fraction for cancer/stage/phase combination."""
    if cancer_type is None or tumor_stage is None:
        return 0.0
    stage = {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(
        tumor_stage.upper().strip(), tumor_stage.upper().strip(),
    )
    key: tuple[str | None, str] = (cancer_type.lower(), stage)
    if key not in _TUMOR_FRACTION_RANGES:
        key = (None, stage)
    if key not in _TUMOR_FRACTION_RANGES:
        return 0.0
    lo, hi = _TUMOR_FRACTION_RANGES[key]
    multiplier = _TREATMENT_FRACTION_MULTIPLIERS.get(treatment_phase, 1.0)
    return min(math.sqrt(lo * hi) * multiplier, 1.0)


def _resolve_tumor_tl_offset(cancer_type: str | None) -> float:
    """Expected tumour TL offset (kb) relative to normal tissue."""
    if cancer_type is None:
        return 0.0
    return _TUMOR_TL_OFFSETS.get(cancer_type.lower(), -1.0)


def _age_sex_adjustment(base_tl_kb: float, age: int, sex: str) -> float:
    """Adjust baseline TL for sex (Gardner et al., 2014). Floor at 1.0 kb."""
    tl = base_tl_kb
    sex_lower = (sex or "unknown").lower()
    if sex_lower in ("f", "female"):
        tl += 0.09   # oestrogen-mediated telomerase upregulation
    elif sex_lower in ("m", "male"):
        tl -= 0.04
    return max(tl, 1.0)


def _compute_tissue_contributions(
    leukocyte_tl_kb: float, tissue_fractions: dict[str, float],
    tl_offsets: dict[str, float] | None = None,
) -> list[TissueContribution]:
    """Build per-tissue contribution list from fractions and TL offsets."""
    offsets = tl_offsets if tl_offsets is not None else _TISSUE_TL_OFFSETS_KB
    contributions: list[TissueContribution] = []
    for tissue, fraction in tissue_fractions.items():
        offset = offsets.get(tissue, 0.0)
        tl = max(leukocyte_tl_kb + offset, 0.5)
        confidence = min(1.0, 0.5 + fraction * 2.0)
        contributions.append(TissueContribution(
            tissue=tissue,
            fraction=round(fraction, 6),
            estimated_telomere_length_kb=round(tl, 3),
            confidence=round(confidence, 3),
        ))
    return contributions


def _weighted_telomere_length(contributions: list[TissueContribution]) -> float:
    """Fraction-weighted mean telomere length across tissues."""
    weight_sum = sum(c.fraction for c in contributions)
    if weight_sum == 0:
        return 0.0
    return sum(c.fraction * c.estimated_telomere_length_kb for c in contributions) / weight_sum


def _simple_linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS slope and intercept. Returns (0, mean_y) for degenerate inputs."""
    if len(xs) < 2:
        return 0.0, (statistics.mean(ys) if ys else 0.0)
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        return 0.0, mean_y
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx
    return slope, mean_y - slope * mean_x


def _classify_attrition(rate_kb_per_year: float) -> str:
    """Map attrition rate to qualitative trend (normal ~24.7 bp/yr)."""
    normal = _NORMAL_ATTRITION_BP_PER_YEAR / 1000.0
    if rate_kb_per_year < -2 * normal:
        return "Accelerated shortening"
    if rate_kb_per_year < -0.5 * normal:
        return "Normal"
    if abs(rate_kb_per_year) <= 0.5 * normal:
        return "Stable"
    return "Lengthening"


def _generate_interpretation(
    mean_tl_kb: float, age: int, tumor_fraction: float, cancer_type: str | None,
) -> str:
    """Produce a concise clinical interpretation string."""
    expected_tl = max(11.0 - (age * _NORMAL_ATTRITION_BP_PER_YEAR / 1000.0), 3.0)
    delta = mean_tl_kb - expected_tl
    if delta > 0.8:
        tl_part = (f"cfDNA-estimated mean TL ({mean_tl_kb:.2f} kb) is above the "
                   f"age-expected range ({expected_tl:.2f} kb), suggesting longer "
                   "telomeres in predominant contributing tissues.")
    elif delta < -0.8:
        tl_part = (f"cfDNA-estimated mean TL ({mean_tl_kb:.2f} kb) is below the "
                   f"age-expected range ({expected_tl:.2f} kb), which may reflect "
                   "accelerated biological ageing or high-turnover contributions.")
    else:
        tl_part = (f"cfDNA-estimated mean TL ({mean_tl_kb:.2f} kb) is within the "
                   f"age-expected range ({expected_tl:.2f} kb).")
    if tumor_fraction > 0.05:
        tl_part += (f"  Caution: tumour fraction ~{tumor_fraction:.1%} may "
                    "substantially confound the composite telomere signal.")
    elif tumor_fraction > 0.001 and cancer_type:
        tl_part += (f"  Low-level tumour cfDNA (~{tumor_fraction:.2%}) detected; "
                    "minor impact but relevant for serial monitoring.")
    return tl_part


# ---------------------------------------------------------------------------
# Public API — core estimation
# ---------------------------------------------------------------------------

def estimate_cfdna_telomere(
    leukocyte_telomere_length_kb: float,
    chronological_age: int,
    sex: str = "unknown",
    cancer_type: str | None = None,
    tumor_stage: str | None = None,
    treatment_phase: str = "pre_treatment",
    previous_tl_kb: float | None = None,
) -> CfDNATelomereResult:
    """Estimate composite cfDNA telomere content from a leukocyte TL anchor.

    Models the cfDNA pool as a tissue-specific fragment mixture (Moss et al.,
    2018).  When a malignancy is specified, ctDNA is folded in with stage-
    and treatment-dependent fraction estimates (Nersisyan et al., 2023).

    Args:
        leukocyte_telomere_length_kb: Measured leukocyte TRF / qPCR TL (kb).
        chronological_age: Patient age in years.
        sex: ``"male"``, ``"female"``, or ``"unknown"``.
        cancer_type: Cancer site (e.g. ``"lung"``), or ``None``.
        tumor_stage: Roman-numeral stage (``"I"``-``"IV"``).
        treatment_phase: ``"pre_treatment"`` | ``"on_treatment"`` | ``"post_treatment"``.
        previous_tl_kb: Prior cfDNA TL for serial delta computation.

    Returns:
        Fully populated :class:`CfDNATelomereResult`.
    """
    adjusted_ltl = _age_sex_adjustment(leukocyte_telomere_length_kb, chronological_age, sex)

    # Build tissue fractions (start from healthy baseline)
    tissue_fractions = dict(_NORMAL_TISSUE_FRACTIONS)
    tumor_fraction = _resolve_tumor_fraction(cancer_type, tumor_stage, treatment_phase)
    tumor_tl_offset = _resolve_tumor_tl_offset(cancer_type)

    # Local copy of offsets to avoid mutating module-level constant
    tl_offsets = dict(_TISSUE_TL_OFFSETS_KB)
    if tumor_fraction > 0 and cancer_type:
        scale = 1.0 - tumor_fraction
        tissue_fractions = {t: f * scale for t, f in tissue_fractions.items()}
        tissue_fractions[f"tumor_{cancer_type.lower()}"] = tumor_fraction
        tl_offsets.setdefault(f"tumor_{cancer_type.lower()}", tumor_tl_offset)

    contributions = _compute_tissue_contributions(adjusted_ltl, tissue_fractions, tl_offsets)
    mean_tl = round(_weighted_telomere_length(contributions), 3)
    ts_ratio = round(mean_tl / 5.0, 4)  # normalised T/S-ratio-like score

    # Serial comparison
    serial_change: float | None = None
    serial_trend = "Unknown"
    if previous_tl_kb is not None:
        serial_change = round(mean_tl - previous_tl_kb, 3)
        serial_trend = ("Stable" if abs(serial_change) < 0.1
                        else "Shortening" if serial_change < 0 else "Lengthening")

    is_confounded = tumor_fraction > 0.05
    interpretation = _generate_interpretation(mean_tl, chronological_age, tumor_fraction, cancer_type)
    methodology_note = (
        "cfDNA TL estimated via tissue-of-origin deconvolution (Moss et al., "
        "2018) anchored to leukocyte TL.  Tumour fraction from ctDNA shedding "
        "rates (Wan et al., 2017).  Computational estimate; direct cfDNA TL "
        "measurement (TelomereHunter on shallow WGS) recommended for clinical use.")

    return CfDNATelomereResult(
        total_cfdna_telomere_content=ts_ratio,
        estimated_mean_telomere_length_kb=mean_tl,
        tissue_contributions=contributions,
        tumor_fraction_estimate=round(tumor_fraction, 6),
        is_tumor_confounded=is_confounded,
        serial_change=serial_change,
        serial_trend=serial_trend,
        clinical_interpretation=interpretation,
        methodology_note=methodology_note,
    )


# ---------------------------------------------------------------------------
# Public API — tumour fraction modelling
# ---------------------------------------------------------------------------

def model_tumor_cfdna(
    cancer_type: str,
    tumor_stage: str,
    treatment_phase: str = "pre_treatment",
) -> tuple[float, float]:
    """Model expected tumour-derived cfDNA fraction and TL offset.

    Stage I tumours shed ~0.01-0.1% ctDNA; stage IV can reach 1-40%
    (Bettegowda et al., 2014; Wan et al., 2017).  Tumour TL offset depends
    on immortalisation: telomerase-positive (~85%) = shortened; ALT (~15%) =
    heterogeneous (Nersisyan et al., 2023).

    Returns:
        ``(tumor_fraction, tumor_tl_offset_kb)`` tuple.
    """
    fraction = _resolve_tumor_fraction(cancer_type, tumor_stage, treatment_phase)
    offset = _resolve_tumor_tl_offset(cancer_type)
    return round(fraction, 6), round(offset, 3)


# ---------------------------------------------------------------------------
# Public API — serial monitoring
# ---------------------------------------------------------------------------

def analyze_serial_cfdna(
    measurements: list[dict[str, Any]],
    treatment_start_date: str | None = None,
) -> SerialMonitoringResult:
    """Analyse longitudinal cfDNA telomere measurements.

    Computes linear attrition rate, classifies trend, and produces
    visualisation-ready data.  Each dict in *measurements* requires ``date``
    (ISO-8601), ``tl_kb``, and ``tumor_fraction``; optional ``treatment_phase``
    enriches interpretation.

    Args:
        measurements: Chronological measurement records (minimum 2).
        treatment_start_date: ISO-8601 date for pre/post-treatment partitioning.

    Raises:
        ValueError: If fewer than two measurements are provided.
    """
    if len(measurements) < 2:
        raise ValueError("At least two timepoints are required for serial analysis.")

    # Parse and sort chronologically
    parsed: list[tuple[datetime, float, float, str]] = []
    for m in measurements:
        parsed.append((
            datetime.fromisoformat(m["date"]),
            float(m["tl_kb"]),
            float(m.get("tumor_fraction", 0.0)),
            m.get("treatment_phase", "unknown"),
        ))
    parsed.sort(key=lambda t: t[0])

    baseline_dt = parsed[0][0]
    years = [(p[0] - baseline_dt).days / 365.25 for p in parsed]
    tls = [p[1] for p in parsed]

    slope, intercept = _simple_linear_regression(years, tls)
    attrition_rate = round(slope, 4)
    trend = _classify_attrition(attrition_rate)

    # Treatment-response heuristic
    response_indicator = "No treatment context provided"
    if treatment_start_date is not None:
        tx_dt = datetime.fromisoformat(treatment_start_date)
        pre_tls = [tl for (dt, tl, _, _) in parsed if dt < tx_dt]
        post_tls = [tl for (dt, tl, _, _) in parsed if dt >= tx_dt]
        if pre_tls and post_tls:
            delta = statistics.mean(post_tls) - statistics.mean(pre_tls)
            if delta > 0.3:
                response_indicator = (
                    "Post-treatment TL increase — may indicate reduced tumour "
                    "burden or treatment-induced telomerase modulation.")
            elif delta < -0.3:
                response_indicator = (
                    "Post-treatment TL decrease — may reflect ongoing tumour "
                    "shedding or therapy-induced senescence.")
            else:
                response_indicator = (
                    "No significant TL shift around treatment start — telomere "
                    "dynamics appear stable across treatment phases.")

    timepoints: list[dict[str, Any]] = [
        {"date": dt.isoformat()[:10], "tl_kb": round(tl, 3),
         "tumor_fraction": round(tf, 6), "treatment_phase": phase}
        for dt, tl, tf, phase in parsed
    ]

    regression_ys = [round(intercept + slope * y, 3) for y in years]
    visualization_data: dict[str, Any] = {
        "x_dates": [tp["date"] for tp in timepoints],
        "x_years_from_baseline": [round(y, 3) for y in years],
        "y_tl_kb": [tp["tl_kb"] for tp in timepoints],
        "y_tumor_fraction": [tp["tumor_fraction"] for tp in timepoints],
        "regression_line_y": regression_ys,
        "regression_slope_kb_per_year": attrition_rate,
        "regression_intercept_kb": round(intercept, 3),
        "treatment_start_date": treatment_start_date,
        "annotations": [],
    }

    # Annotate inflection points (sign changes in consecutive TL deltas)
    deltas = [tls[i + 1] - tls[i] for i in range(len(tls) - 1)]
    for i in range(len(deltas) - 1):
        if deltas[i] * deltas[i + 1] < 0:
            visualization_data["annotations"].append({
                "date": timepoints[i + 1]["date"],
                "tl_kb": timepoints[i + 1]["tl_kb"],
                "label": "Trend inflection",
            })

    return SerialMonitoringResult(
        timepoints=timepoints,
        attrition_rate_kb_per_year=attrition_rate,
        trend=trend,
        treatment_response_indicator=response_indicator,
        visualization_data=visualization_data,
    )


# ---------------------------------------------------------------------------
# Public API — WGS-based cfDNA telomere estimation
# ---------------------------------------------------------------------------

def estimate_from_wgs_cfdna(
    telomeric_read_count: int,
    total_read_count: int,
    mean_read_length: int = 150,
    genome_size: int = 3_200_000_000,
) -> float:
    """Estimate telomere length from whole-genome sequencing of cfDNA.

    Implements TelomereHunter / TelSeq adapted for nucleosomal cfDNA (~167 bp)::

        TL = (telomeric_reads / total_reads) x (genome_size / 92) x correction

    Correction factor (300/167) compensates for reduced k-mer capture in short
    cfDNA fragments vs. standard gDNA libraries (~300 bp inserts).

    Args:
        telomeric_read_count: Reads with >= 7 consecutive TTAGGG/CCCTAA hexamers.
        total_read_count: Total mapped + unmapped reads in the library.
        mean_read_length: Average read length in bp (default 150).
        genome_size: Reference genome size in bp (default 3.2 Gb, GRCh38).

    Returns:
        Estimated mean TL in kb, clamped to [0.5, 25.0].

    Raises:
        ValueError: If ``total_read_count`` <= 0 or counts are negative.

    References:
        Ding et al. (2014). *Nucleic Acids Res.*, 42(9), e75.
        Feuerbach et al. (2019). *BMC Bioinformatics*, 20, 272.
    """
    if total_read_count <= 0:
        raise ValueError("total_read_count must be a positive integer.")
    if telomeric_read_count < 0:
        raise ValueError("telomeric_read_count cannot be negative.")

    telomere_fraction = telomeric_read_count / total_read_count
    bases_per_end = genome_size / _CHROMOSOME_ENDS
    # cfDNA correction: standard library (~300 bp) vs nucleosomal cfDNA (~167 bp)
    raw_tl_bp = telomere_fraction * bases_per_end * (300.0 / _CFDNA_FRAGMENT_SIZE)
    tl_kb = raw_tl_bp / 1000.0
    return round(max(0.5, min(tl_kb, 25.0)), 3)
