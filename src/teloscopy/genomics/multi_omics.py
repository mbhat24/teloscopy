"""Multi-omics integration for telomere biology research.

Combines transcriptomic, proteomic, metabolomic, and microbiome data to
build an integrated view of telomere health and biological aging.  Each
omics layer contributes independent evidence that is fused via
inverse-variance weighted averaging to produce a single biological-age
estimate and a composite telomere-health score.

**This module is intended for research use only.**

References
----------
.. [1] Lopez-Otin C, Blasco MA, Partridge L, Serrano M, Kroemer G (2013).
       The hallmarks of aging. Cell 153(6):1194-1217. PMID:23746838
.. [2] Azzalin CM, Reichenbach P, Khoriauli L, Giulotto E, Lingner J
       (2007). Telomeric repeat containing RNA and RNA surveillance
       factors at mammalian chromosome ends. Science 318(5851):798-801.
.. [3] Armanios M, Blackburn EH (2012). The telomere syndromes.
       Nature Reviews Genetics 13(10):693-704.
.. [4] Franceschi C, Garagnani P, Parini P, Giuliani C, Santoro A
       (2018). Inflammaging: a new immune-metabolic viewpoint for
       age-related diseases. Nature Reviews Endocrinology 14(10):576-590.
.. [5] Coppe JP, Desprez PY, Krtolica A, Campisi J (2010). The
       senescence-associated secretory phenotype: the dark side of tumor
       suppression. Annual Review of Pathology 5:99-118.
.. [6] Claesson MJ, Jeffery IB, Conde S, et al. (2012). Gut microbiota
       composition correlates with diet and health in the elderly.
       Nature 488(7410):178-184.
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "RESEARCH_ONLY_DISCLAIMER",
    "TranscriptomicProfile",
    "ProteomicProfile",
    "MetabolomicProfile",
    "MicrobiomeProfile",
    "MultiOmicsResult",
    "integrate_multi_omics",
    "estimate_from_transcriptomics",
    "estimate_from_proteomics",
    "estimate_from_metabolomics",
    "estimate_from_microbiome",
    "compute_pathway_enrichment",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

RESEARCH_ONLY_DISCLAIMER: str = (
    "This module is provided for academic and exploratory research purposes "
    "only.  It has NOT been validated for clinical or diagnostic use.  Do "
    "not use these results to make medical decisions.  All biological-age "
    "estimates and telomere-health scores are approximations derived from "
    "published population-level associations and carry substantial "
    "uncertainty at the individual level."
)

# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------

_TERRA_REFERENCE_LEVELS: Dict[str, float] = {
    # Expected TERRA expression (arbitrary normalised units).
    # 17p is the highest producer (Azzalin et al., 2007).
    "1p": 1.0, "1q": 0.8, "2p": 0.7, "2q": 0.6, "3p": 0.5, "4p": 0.6,
    "5p": 0.5, "6p": 0.7, "7p": 0.8, "7q": 0.6, "9p": 0.9, "10q": 0.7,
    "11p": 0.5, "12q": 0.6, "13q": 0.4, "15q": 0.5, "16p": 0.8,
    "17p": 2.5,  # highest TERRA-producing locus
    "17q": 0.7, "18p": 0.6, "20q": 0.9, "Xp": 1.1, "Xq": 1.0, "Yq": 0.3,
}

_SHELTERIN_REFERENCE_EXPRESSION: Dict[str, Tuple[float, float]] = {
    # Normal TPM ranges for shelterin genes in PBMCs.
    "TRF1": (8.0, 25.0), "TRF2": (10.0, 35.0), "POT1": (5.0, 18.0),
    "TIN2": (12.0, 40.0), "TPP1": (6.0, 22.0), "RAP1": (4.0, 15.0),
}

_SASP_MARKERS: Dict[str, Tuple[float, float]] = {
    # Reference concentration ranges (pg/mL) for key SASP factors
    # (Coppe et al., 2010).
    "IL6": (0.5, 5.0), "IL8": (2.0, 15.0), "IL1beta": (0.1, 1.5),
    "TNFalpha": (0.5, 3.0), "MMP3": (5.0, 30.0), "MMP9": (20.0, 100.0),
    "CCL2": (50.0, 300.0), "PAI1": (5.0, 40.0),
}

_OXIDATIVE_STRESS_MARKERS: Dict[str, Tuple[float, float]] = {
    # Reference ranges – units vary by analyte.
    "8-OHdG": (2.0, 15.0),     # ng/mL – DNA oxidation
    "MDA": (0.5, 3.5),         # nmol/mL – lipid peroxidation
    "SOD": (120.0, 240.0),     # U/mL – superoxide dismutase
    "GPx": (30.0, 70.0),       # U/L – glutathione peroxidase
    "catalase": (10.0, 50.0),  # kU/L
}

_INFLAMMATION_MARKERS: Dict[str, Tuple[float, float]] = {
    # Systemic inflammation – normal ranges (Franceschi et al., 2018).
    "CRP": (0.0, 3.0),            # mg/L
    "ESR": (0.0, 20.0),           # mm/hr
    "ferritin": (20.0, 250.0),    # ng/mL
    "fibrinogen": (200.0, 400.0), # mg/dL
}

_MICROBIOME_AGING_ASSOCIATIONS: Dict[str, float] = {
    # Standardised beta for gut microbiome → telomere length.
    # Positive = longer telomeres (Claesson et al., 2012).
    "diversity_index": 0.15, "firmicutes_bacteroidetes_ratio": -0.10,
    "butyrate_producers": 0.20, "inflammation_score": -0.25,
    "bifidobacterium_abundance": 0.12, "akkermansia_abundance": 0.08,
    "prevotella_abundance": 0.05, "bacteroides_abundance": -0.03,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptomicProfile:
    """RNA-seq derived telomere-biology features.

    Attributes
    ----------
    gene_expression : dict[str, float]
        Genome-wide TPM values.
    terra_levels : dict[str, float]
        TERRA transcript levels per chromosome arm.
    telomerase_activity_score : float
        Composite score (0-1) from hTERT/hTR expression.
    shelterin_expression : dict[str, float]
        TPM for TRF1, TRF2, POT1, TIN2, TPP1, RAP1.
    ddr_pathway_score : float
        DNA-damage-response activation score (0-1).
    """
    gene_expression: Dict[str, float] = field(default_factory=dict)
    terra_levels: Dict[str, float] = field(default_factory=dict)
    telomerase_activity_score: float = 0.0
    shelterin_expression: Dict[str, float] = field(default_factory=dict)
    ddr_pathway_score: float = 0.0

@dataclass
class ProteomicProfile:
    """Proteomics-derived telomere-biology features.

    Attributes
    ----------
    shelterin_protein_levels : dict
        Protein abundances for shelterin complex members.
    post_translational_mods : dict
        PTMs relevant to telomere maintenance.
    sasp_markers : dict[str, float]
        Concentrations of key SASP factors.
    telomerase_protein_complex : dict
        Telomerase holoenzyme component levels.
    """
    shelterin_protein_levels: Dict[str, float] = field(default_factory=dict)
    post_translational_mods: Dict[str, float] = field(default_factory=dict)
    sasp_markers: Dict[str, float] = field(default_factory=dict)
    telomerase_protein_complex: Dict[str, float] = field(default_factory=dict)

@dataclass
class MetabolomicProfile:
    """Metabolomics-derived telomere-biology features.

    Attributes
    ----------
    oxidative_stress_markers : dict
        Oxidative stress biomarker concentrations.
    inflammation_markers : dict
        Systemic inflammation biomarkers.
    nad_pathway : dict
        NAD+ pathway metabolites (NAD+, NADH, NMN, NR, NAM).
    one_carbon_metabolites : dict
        One-carbon metabolism intermediates (folate, methionine,
        homocysteine, SAM, SAH).
    """
    oxidative_stress_markers: Dict[str, float] = field(default_factory=dict)
    inflammation_markers: Dict[str, float] = field(default_factory=dict)
    nad_pathway: Dict[str, float] = field(default_factory=dict)
    one_carbon_metabolites: Dict[str, float] = field(default_factory=dict)

@dataclass
class MicrobiomeProfile:
    """Gut-microbiome features relevant to telomere biology.

    Attributes
    ----------
    diversity_index : float
        Shannon diversity index.
    firmicutes_bacteroidetes_ratio : float
        F/B ratio — elevated in older adults.
    inflammation_score : float
        Microbiome-derived inflammation score (0-1).
    butyrate_producers_fraction : float
        Relative abundance of butyrate-producing taxa (0-1).
    """
    diversity_index: float = 0.0
    firmicutes_bacteroidetes_ratio: float = 1.0
    inflammation_score: float = 0.0
    butyrate_producers_fraction: float = 0.0

@dataclass
class MultiOmicsResult:
    """Integrated multi-omics telomere-health assessment.

    Attributes
    ----------
    transcriptomic, proteomic, metabolomic, microbiome
        Individual omics layers (``None`` if not provided).
    biological_age_estimate : float
        Estimated biological age in years.
    telomere_health_score : float
        Composite score (0-100).
    aging_acceleration : float
        Biological age minus chronological age; positive = accelerated.
    pathway_enrichment : dict
        Pathway enrichment p-values.
    cross_omics_correlations : dict
        Pairwise cross-layer consistency scores.
    recommendations : list[str]
        Research or follow-up recommendations.
    confidence : float
        Overall confidence (0-1).
    """
    transcriptomic: Optional[TranscriptomicProfile] = None
    proteomic: Optional[ProteomicProfile] = None
    metabolomic: Optional[MetabolomicProfile] = None
    microbiome: Optional[MicrobiomeProfile] = None
    biological_age_estimate: float = 0.0
    telomere_health_score: float = 0.0
    aging_acceleration: float = 0.0
    pathway_enrichment: Dict[str, float] = field(default_factory=dict)
    cross_omics_correlations: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))

def _z_score(value: float, ref_low: float, ref_high: float) -> float:
    """Z-like score: 0 at midpoint, magnitude > 1 outside range."""
    mid = (ref_low + ref_high) / 2.0
    half = (ref_high - ref_low) / 2.0
    return (value - mid) / half if half else 0.0

def _shannon_diversity(abundances: Sequence[float]) -> float:
    """Shannon diversity H' for a vector of (possibly unnormalised) abundances."""
    total = sum(abundances)
    if total <= 0:
        return 0.0
    return -sum((a / total) * math.log(a / total) for a in abundances if a > 0)

def _inverse_variance_weighted_mean(
    estimates: Sequence[float], variances: Sequence[float],
) -> Tuple[float, float]:
    """Inverse-variance weighted average.  Returns (mean, variance)."""
    if not estimates:
        return 0.0, float("inf")
    weights = [1.0 / v if v > 0 else 0.0 for v in variances]
    tw = sum(weights)
    if tw == 0:
        return statistics.mean(estimates), float("inf")
    return sum(e * w for e, w in zip(estimates, weights)) / tw, 1.0 / tw

def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return cov / denom if denom else 0.0

def _pseudo_correlation(a: float, b: float) -> float:
    """Heuristic consistency score in [-1, 1] for two [0, 1] scalars."""
    return _clamp((a - 0.5) * (b - 0.5) * 4.0, -1.0, 1.0)

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_shelterin_expression(shelterin: Dict[str, float]) -> float:
    """Score shelterin expression vs reference; returns 0-1 (1 = normal)."""
    if not shelterin:
        return 0.5
    devs = [abs(_z_score(shelterin[g], lo, hi))
            for g, (lo, hi) in _SHELTERIN_REFERENCE_EXPRESSION.items()
            if g in shelterin]
    return 1.0 / (1.0 + statistics.mean(devs)) if devs else 0.5


def _score_terra_levels(terra: Dict[str, float]) -> float:
    """Compare observed TERRA to reference; returns 0-1."""
    if not terra:
        return 0.5
    ratios = [terra[arm] / ref for arm, ref in _TERRA_REFERENCE_LEVELS.items()
              if arm in terra and ref > 0]
    if not ratios:
        return 0.5
    return _clamp(1.0 / (1.0 + abs(statistics.mean(ratios) - 1.0)))


def _score_sasp(sasp: Dict[str, float]) -> float:
    """SASP activation score in [0, 1]; higher = worse."""
    if not sasp:
        return 0.0
    zs = [max(0.0, _z_score(sasp[m], lo, hi))
          for m, (lo, hi) in _SASP_MARKERS.items() if m in sasp]
    return _clamp(statistics.mean(zs) / 3.0) if zs else 0.0


def _score_oxidative_stress(markers: Dict[str, float]) -> float:
    """Oxidative-stress burden in [0, 1]; higher = worse."""
    if not markers:
        return 0.5
    damage = {"8-OHdG", "MDA"}
    antioxidant = {"SOD", "GPx", "catalase"}
    d_z = [_z_score(markers[n], lo, hi) for n, (lo, hi)
           in _OXIDATIVE_STRESS_MARKERS.items() if n in markers and n in damage]
    a_z = [_z_score(markers[n], lo, hi) for n, (lo, hi)
           in _OXIDATIVE_STRESS_MARKERS.items() if n in markers and n in antioxidant]
    d = statistics.mean(d_z) if d_z else 0.0
    a = statistics.mean(a_z) if a_z else 0.0
    return _clamp(((d - a) / 2.0 + 1.0) / 2.0)


def _score_inflammation(markers: Dict[str, float]) -> float:
    """Systemic inflammation score in [0, 1]."""
    if not markers:
        return 0.5
    zs = [max(0.0, _z_score(markers[n], lo, hi))
          for n, (lo, hi) in _INFLAMMATION_MARKERS.items() if n in markers]
    return _clamp(statistics.mean(zs) / 3.0) if zs else 0.5

# ---------------------------------------------------------------------------
# Per-layer biological age estimators
# ---------------------------------------------------------------------------

def _bio_age_from_tl(tl_kb: float, age: float, sex: str) -> Tuple[float, float]:
    """Biological age from telomere length (Armanios & Blackburn, 2012)."""
    intercept, slope = (11.0, 0.040) if sex.lower().startswith("f") else (10.5, 0.042)
    return max(0.0, (intercept - tl_kb) / slope) if slope else age, 64.0


def _bio_age_from_transcriptomics(
    p: TranscriptomicProfile, age: float,
) -> Tuple[float, float]:
    """Biological age from transcriptomic features."""
    health = (
        _score_shelterin_expression(p.shelterin_expression)
        + _score_terra_levels(p.terra_levels)
        + (1.0 - p.ddr_pathway_score)
    ) / 3.0
    return max(0.0, age + 15.0 - 20.0 * health), 49.0


def _bio_age_from_proteomics(
    p: ProteomicProfile, age: float,
) -> Tuple[float, float]:
    """Biological age from proteomic features."""
    health = (1.0 - _score_sasp(p.sasp_markers)
              + min(len(p.shelterin_protein_levels) / 6.0, 1.0)) / 2.0
    return max(0.0, age + 15.0 - 20.0 * health), 56.25


def _bio_age_from_metabolomics(
    p: MetabolomicProfile, age: float,
) -> Tuple[float, float]:
    """Biological age from metabolomic features."""
    nad_score = 0.5
    if "NAD+" in p.nad_pathway and "NADH" in p.nad_pathway:
        ratio = p.nad_pathway["NAD+"] / max(p.nad_pathway["NADH"], 0.01)
        nad_score = _clamp((ratio - 1.0) / 9.0)
    health = ((1.0 - _score_oxidative_stress(p.oxidative_stress_markers))
              + (1.0 - _score_inflammation(p.inflammation_markers))
              + nad_score) / 3.0
    return max(0.0, age + 15.0 - 20.0 * health), 81.0


def _bio_age_from_microbiome(
    p: MicrobiomeProfile, age: float,
) -> Tuple[float, float]:
    """Biological age from microbiome features (Claesson et al., 2012)."""
    assoc = _MICROBIOME_AGING_ASSOCIATIONS
    effect = (
        assoc["diversity_index"] * _clamp(p.diversity_index / 5.0)
        + assoc["firmicutes_bacteroidetes_ratio"]
          * _clamp(p.firmicutes_bacteroidetes_ratio / 5.0)
        + assoc["butyrate_producers"] * p.butyrate_producers_fraction
        + assoc["inflammation_score"] * p.inflammation_score
    )
    return max(0.0, age - effect * 10.0), 100.0


# ---------------------------------------------------------------------------
# Cross-omics correlations
# ---------------------------------------------------------------------------

def _compute_cross_omics_correlations(
    tr: Optional[TranscriptomicProfile], pr: Optional[ProteomicProfile],
    me: Optional[MetabolomicProfile], mb: Optional[MicrobiomeProfile],
) -> Dict[str, float]:
    """Pairwise consistency scores between omics layers."""
    c: Dict[str, float] = {}
    if tr and pr:
        shared = [(tr.shelterin_expression[g], pr.shelterin_protein_levels[g])
                  for g in _SHELTERIN_REFERENCE_EXPRESSION
                  if g in tr.shelterin_expression and g in pr.shelterin_protein_levels]
        c["transcriptomic_proteomic"] = (
            _pearson([s[0] for s in shared], [s[1] for s in shared])
            if len(shared) >= 2 else float("nan"))
    if tr and me:
        c["transcriptomic_metabolomic"] = _pseudo_correlation(
            tr.ddr_pathway_score, _score_oxidative_stress(me.oxidative_stress_markers))
    if pr and me:
        c["proteomic_metabolomic"] = _pseudo_correlation(
            _score_sasp(pr.sasp_markers), _score_inflammation(me.inflammation_markers))
    if me and mb:
        c["metabolomic_microbiome"] = _pseudo_correlation(
            _score_inflammation(me.inflammation_markers), mb.inflammation_score)
    if tr and mb:
        c["transcriptomic_microbiome"] = _pseudo_correlation(
            tr.ddr_pathway_score, mb.inflammation_score)
    if pr and mb:
        c["proteomic_microbiome"] = _pseudo_correlation(
            _score_sasp(pr.sasp_markers), mb.inflammation_score)
    return c


# ---------------------------------------------------------------------------
# Telomere health score
# ---------------------------------------------------------------------------

def _compute_telomere_health_score(
    tl_kb: float, age: float, sex: str,
    tr: Optional[TranscriptomicProfile], pr: Optional[ProteomicProfile],
    me: Optional[MetabolomicProfile], mb: Optional[MicrobiomeProfile],
) -> float:
    """Composite telomere-health score (0-100).

    Weights: TL 30 %, shelterin 20 %, TERRA 10 %, inflammation 20 %,
    oxidative stress 10 %, microbiome 10 %.
    """
    bio, _ = _bio_age_from_tl(tl_kb, age, sex)
    tl_s = _clamp(1.0 - (bio - age) / 30.0)

    sh_s = 0.5
    if tr and tr.shelterin_expression:
        sh_s = _score_shelterin_expression(tr.shelterin_expression)
    if pr and pr.shelterin_protein_levels:
        sh_s = (sh_s + min(len(pr.shelterin_protein_levels) / 6.0, 1.0)) / 2.0

    terra_s = (_score_terra_levels(tr.terra_levels)
               if tr and tr.terra_levels else 0.5)

    inf_s, n_inf = 0.5, 0
    if pr:
        inf_s, n_inf = 1.0 - _score_sasp(pr.sasp_markers), 1
    if me:
        v = 1.0 - _score_inflammation(me.inflammation_markers)
        inf_s = (inf_s + v) / 2.0 if n_inf else v

    ox_s = (1.0 - _score_oxidative_stress(me.oxidative_stress_markers)
            if me else 0.5)

    mb_s = 0.5
    if mb:
        mb_s = (_clamp(mb.diversity_index / 5.0) * 0.4
                + _clamp(1.0 - mb.firmicutes_bacteroidetes_ratio / 5.0) * 0.2
                + mb.butyrate_producers_fraction * 0.2
                + (1.0 - mb.inflammation_score) * 0.2)

    total = (tl_s * 0.30 + sh_s * 0.20 + terra_s * 0.10
             + inf_s * 0.20 + ox_s * 0.10 + mb_s * 0.10)
    return round(_clamp(total) * 100.0, 1)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def _generate_recommendations(
    health: float, accel: float, tr: Optional[TranscriptomicProfile],
    pr: Optional[ProteomicProfile], me: Optional[MetabolomicProfile],
    mb: Optional[MicrobiomeProfile],
) -> List[str]:
    """Generate evidence-based research recommendations."""
    r: List[str] = []
    if accel > 5.0:
        r.append(f"Significant aging acceleration (>{accel:.1f} yr).  "
                 "Consider longitudinal multi-omics profiling at 6-12 month intervals.")
    if health < 30.0:
        r.append(f"Low telomere-health score ({health:.1f}/100).  Investigate "
                 "telomere biology disorders per Armanios & Blackburn (2012).")
    if tr is not None:
        if tr.ddr_pathway_score > 0.7:
            r.append("Elevated DDR signature.  Evaluate for replicative "
                     "stress or genotoxic exposures.")
        if tr.telomerase_activity_score < 0.2:
            r.append("Low telomerase activity.  Screen for hTERT promoter "
                     "mutations or epigenetic silencing.")
    if pr is not None:
        sasp = _score_sasp(pr.sasp_markers)
        if sasp > 0.6:
            r.append(f"High SASP activation ({sasp:.2f}).  Consider "
                     "senolytic research protocols (Coppe et al., 2010).")
    if me is not None:
        if _score_oxidative_stress(me.oxidative_stress_markers) > 0.7:
            r.append("Elevated oxidative-stress burden.  Assess "
                     "mitochondrial function and antioxidant capacity.")
        if me.nad_pathway.get("NAD+", 999.0) < 100.0:
            r.append("Low NAD+ levels.  NAD+ precursor supplementation "
                     "may be relevant (research context).")
    if mb is not None:
        if mb.diversity_index < 2.0:
            r.append(f"Low microbiome diversity (H'={mb.diversity_index:.2f}).  "
                     "Dietary diversity may help (Claesson et al., 2012).")
        if mb.firmicutes_bacteroidetes_ratio > 3.0:
            r.append(f"Elevated F/B ratio ({mb.firmicutes_bacteroidetes_ratio:.2f}).  "
                     "Associated with inflammaging in elderly cohorts.")
    if not r:
        r.append("Multi-omics profile within expected ranges.  "
                 "Continue standard longitudinal monitoring.")
    return r


# ---------------------------------------------------------------------------
# Fisher's exact test helpers
# ---------------------------------------------------------------------------

def _log_factorial(n: int) -> float:
    """log(n!) via summation."""
    return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0


def _hypergeom_pmf(k: int, K: int, n: int, N: int) -> float:
    """Hypergeometric PMF: P(X=k) drawing n from N with K successes."""
    if k < max(0, n + K - N) or k > min(n, K):
        return 0.0
    lp = (_log_factorial(K) + _log_factorial(N - K)
          + _log_factorial(n) + _log_factorial(N - n)
          - _log_factorial(N) - _log_factorial(k)
          - _log_factorial(K - k) - _log_factorial(n - k)
          - _log_factorial(N - n - K + k))
    return math.exp(lp)


def _fishers_exact_p(a: int, b: int, c: int, d: int) -> float:
    """One-tailed Fisher's exact test p-value (enrichment direction)."""
    n, K, N = a + b, a + c, a + b + c + d
    return min(sum(_hypergeom_pmf(k, K, n, N) for k in range(a, min(n, K) + 1)), 1.0)


# ---------------------------------------------------------------------------
# Public functions — omics layer estimators
# ---------------------------------------------------------------------------

def estimate_from_transcriptomics(
    gene_expression_tpm: Dict[str, float],
    terra_counts: Optional[Dict[str, float]] = None,
) -> TranscriptomicProfile:
    """Extract telomere-relevant features from RNA-seq TPM data.

    Parameters
    ----------
    gene_expression_tpm : dict[str, float]
        Gene-level expression in transcripts per million.
    terra_counts : dict[str, float], optional
        TERRA transcript counts per chromosome arm.

    Returns
    -------
    TranscriptomicProfile
    """
    # Shelterin expression (handle official HGNC and common aliases)
    aliases = {
        "TERF1": "TRF1", "TRF1": "TRF1", "TERF2": "TRF2", "TRF2": "TRF2",
        "POT1": "POT1", "TINF2": "TIN2", "TIN2": "TIN2",
        "ACD": "TPP1", "TPP1": "TPP1", "TERF2IP": "RAP1", "RAP1": "RAP1",
    }
    shelterin: Dict[str, float] = {}
    for gene, canonical in aliases.items():
        if gene in gene_expression_tpm:
            shelterin[canonical] = gene_expression_tpm[gene]

    htert = gene_expression_tpm.get("TERT", gene_expression_tpm.get("hTERT", 0.0))
    htr = gene_expression_tpm.get("TERC", gene_expression_tpm.get("hTR", 0.0))
    telomerase_score = _clamp((min(htert / 5.0, 1.0) + min(htr / 50.0, 1.0)) / 2.0)

    ddr_genes = ["ATM", "ATR", "CHEK1", "CHEK2", "TP53", "BRCA1", "H2AFX"]
    ddr_vals = [gene_expression_tpm[g] for g in ddr_genes if g in gene_expression_tpm]
    ddr_score = _clamp((statistics.median(ddr_vals) - 10.0) / 40.0) if ddr_vals else 0.0

    return TranscriptomicProfile(
        gene_expression=gene_expression_tpm,
        terra_levels=terra_counts or {},
        telomerase_activity_score=telomerase_score,
        shelterin_expression=shelterin,
        ddr_pathway_score=ddr_score,
    )


def estimate_from_proteomics(
    protein_levels: Dict[str, float],
) -> ProteomicProfile:
    """Process proteomics data for shelterin/SASP markers.

    Parameters
    ----------
    protein_levels : dict[str, float]
        Protein abundances keyed by protein/gene symbol.

    Returns
    -------
    ProteomicProfile
    """
    shelterin = {n: protein_levels[n] for n in
                 ("TRF1", "TRF2", "POT1", "TIN2", "TPP1", "RAP1")
                 if n in protein_levels}
    sasp = {m: protein_levels[m] for m in _SASP_MARKERS if m in protein_levels}
    ptm_prefixes = ("phospho_", "ubiq_", "sumo_", "acetyl_")
    ptms = {k: v for k, v in protein_levels.items()
            if any(k.lower().startswith(p) for p in ptm_prefixes)}
    tel_comps = {"TERT", "TERC", "DKC1", "NHP2", "NOP10", "GAR1"}
    telomerase = {c: protein_levels[c] for c in tel_comps if c in protein_levels}
    return ProteomicProfile(
        shelterin_protein_levels=shelterin,
        post_translational_mods=ptms,
        sasp_markers=sasp,
        telomerase_protein_complex=telomerase,
    )


def estimate_from_metabolomics(
    metabolite_levels: Dict[str, float],
) -> MetabolomicProfile:
    """Assess oxidative stress and inflammation from metabolomics.

    Parameters
    ----------
    metabolite_levels : dict[str, float]
        Metabolite concentrations keyed by analyte name.

    Returns
    -------
    MetabolomicProfile
    """
    ox = {n: metabolite_levels[n] for n in _OXIDATIVE_STRESS_MARKERS
          if n in metabolite_levels}
    inf = {n: metabolite_levels[n] for n in _INFLAMMATION_MARKERS
           if n in metabolite_levels}
    nad = {n: metabolite_levels[n] for n in ("NAD+", "NADH", "NMN", "NR", "NAM")
           if n in metabolite_levels}
    oc = {n: metabolite_levels[n]
          for n in ("folate", "methionine", "homocysteine", "SAM", "SAH")
          if n in metabolite_levels}
    return MetabolomicProfile(
        oxidative_stress_markers=ox, inflammation_markers=inf,
        nad_pathway=nad, one_carbon_metabolites=oc,
    )


def estimate_from_microbiome(
    taxa_abundances: Dict[str, float],
) -> MicrobiomeProfile:
    """Compute diversity index and inflammation score from microbiome.

    Parameters
    ----------
    taxa_abundances : dict[str, float]
        Relative abundances of microbial taxa (should sum to ~1).

    Returns
    -------
    MicrobiomeProfile
    """
    diversity = _shannon_diversity([v for v in taxa_abundances.values() if v > 0])

    firm = sum(v for k, v in taxa_abundances.items() if "firmicute" in k.lower())
    bact = sum(v for k, v in taxa_abundances.items() if "bacteroidete" in k.lower())
    fb_ratio = firm / bact if bact > 0 else 1.0

    butyrate_genera = {"faecalibacterium", "roseburia", "eubacterium",
                       "coprococcus", "butyrivibrio", "anaerostipes"}
    butyrate = sum(v for k, v in taxa_abundances.items()
                   if any(g in k.lower() for g in butyrate_genera))

    pro_inf_genera = {"escherichia", "klebsiella", "enterococcus",
                      "clostridium_difficile"}
    anti_inf_genera = {"bifidobacterium", "lactobacillus", "akkermansia"}
    pro = sum(v for k, v in taxa_abundances.items()
              if any(g in k.lower() for g in pro_inf_genera))
    anti = sum(v for k, v in taxa_abundances.items()
               if any(g in k.lower() for g in anti_inf_genera))
    inflammation = _clamp(pro - anti * 0.5)

    return MicrobiomeProfile(
        diversity_index=round(diversity, 4),
        firmicutes_bacteroidetes_ratio=round(fb_ratio, 4),
        inflammation_score=round(inflammation, 4),
        butyrate_producers_fraction=round(_clamp(butyrate), 4),
    )


# ---------------------------------------------------------------------------
# Public functions — integration and enrichment
# ---------------------------------------------------------------------------

def integrate_multi_omics(
    telomere_length_kb: float,
    age: float,
    sex: str,
    transcriptomic: Optional[TranscriptomicProfile] = None,
    proteomic: Optional[ProteomicProfile] = None,
    metabolomic: Optional[MetabolomicProfile] = None,
    microbiome: Optional[MicrobiomeProfile] = None,
) -> MultiOmicsResult:
    """Integrate available omics layers into a unified assessment.

    Parameters
    ----------
    telomere_length_kb : float
        Mean telomere length in kilobases.
    age : float
        Chronological age in years.
    sex : str
        Biological sex (``'male'``, ``'female'``, or ``'other'``).
    transcriptomic : TranscriptomicProfile, optional
    proteomic : ProteomicProfile, optional
    metabolomic : MetabolomicProfile, optional
    microbiome : MicrobiomeProfile, optional

    Returns
    -------
    MultiOmicsResult
        Integrated result with biological-age estimate, telomere-health
        score, pathway enrichment, cross-omics correlations, and
        recommendations.  Uses inverse-variance weighted averaging.
    """
    n_layers = sum(x is not None for x in
                   (transcriptomic, proteomic, metabolomic, microbiome)) + 1
    logger.info("Integrating multi-omics: TL=%.2f kb, age=%.1f, sex=%s, layers=%d",
                telomere_length_kb, age, sex, n_layers)

    # Per-layer biological age estimates
    ests: List[float] = []
    vrs: List[float] = []
    ba, v = _bio_age_from_tl(telomere_length_kb, age, sex)
    ests.append(ba); vrs.append(v)
    if transcriptomic is not None:
        ba, v = _bio_age_from_transcriptomics(transcriptomic, age)
        ests.append(ba); vrs.append(v)
    if proteomic is not None:
        ba, v = _bio_age_from_proteomics(proteomic, age)
        ests.append(ba); vrs.append(v)
    if metabolomic is not None:
        ba, v = _bio_age_from_metabolomics(metabolomic, age)
        ests.append(ba); vrs.append(v)
    if microbiome is not None:
        ba, v = _bio_age_from_microbiome(microbiome, age)
        ests.append(ba); vrs.append(v)

    bio_age = round(_inverse_variance_weighted_mean(ests, vrs)[0], 2)
    aging_accel = round(bio_age - age, 2)

    health = _compute_telomere_health_score(
        telomere_length_kb, age, sex,
        transcriptomic, proteomic, metabolomic, microbiome)

    correlations = {k: round(v, 4) for k, v in
                    _compute_cross_omics_correlations(
                        transcriptomic, proteomic, metabolomic, microbiome
                    ).items()}

    # Confidence from layer count + cross-omics consistency
    base_conf = min(n_layers / 5.0, 1.0)
    valid_c = [v for v in correlations.values() if not math.isnan(v)]
    bonus = _clamp(statistics.mean(valid_c) * 0.1, -0.1, 0.1) if valid_c else 0.0
    confidence = round(_clamp(base_conf + bonus), 3)

    result = MultiOmicsResult(
        transcriptomic=transcriptomic, proteomic=proteomic,
        metabolomic=metabolomic, microbiome=microbiome,
        biological_age_estimate=bio_age, telomere_health_score=health,
        aging_acceleration=aging_accel, pathway_enrichment={},
        cross_omics_correlations=correlations,
        recommendations=[], confidence=confidence)

    result.pathway_enrichment = compute_pathway_enrichment(result)
    result.recommendations = _generate_recommendations(
        health, aging_accel, transcriptomic, proteomic, metabolomic, microbiome)
    return result


def compute_pathway_enrichment(
    multi_omics_result: MultiOmicsResult,
) -> Dict[str, float]:
    """Compute pathway enrichment p-values via Fisher's exact test.

    Evaluates telomere maintenance, DNA-damage response, SASP/senescence,
    and oxidative stress pathways.

    Parameters
    ----------
    multi_omics_result : MultiOmicsResult
        An integrated multi-omics result.

    Returns
    -------
    dict
        Pathway name to one-tailed Fisher's exact p-value.
    """
    pathways: Dict[str, List[str]] = {
        "telomere_maintenance": [
            "TRF1", "TRF2", "POT1", "TIN2", "TPP1", "RAP1",
            "TERT", "TERC", "DKC1"],
        "dna_damage_response": [
            "ATM", "ATR", "CHEK1", "CHEK2", "TP53", "BRCA1",
            "H2AFX", "MDC1", "RNF8"],
        "sasp_senescence": list(_SASP_MARKERS.keys()),
        "oxidative_stress": list(_OXIDATIVE_STRESS_MARKERS.keys()),
    }

    measured: set[str] = set()
    elevated: set[str] = set()

    tr = multi_omics_result.transcriptomic
    if tr is not None:
        measured.update(tr.gene_expression.keys())
        measured.update(tr.shelterin_expression.keys())
        for gene, (lo, hi) in _SHELTERIN_REFERENCE_EXPRESSION.items():
            val = tr.shelterin_expression.get(gene)
            if val is not None:
                measured.add(gene)
                if val > hi or val < lo:
                    elevated.add(gene)
        for g in ("ATM", "ATR", "CHEK1", "CHEK2", "TP53", "BRCA1", "H2AFX"):
            if g in tr.gene_expression:
                measured.add(g)
                if tr.gene_expression[g] > 30.0:
                    elevated.add(g)

    pr = multi_omics_result.proteomic
    if pr is not None:
        measured.update(pr.sasp_markers.keys())
        measured.update(pr.shelterin_protein_levels.keys())
        for marker, (_, hi) in _SASP_MARKERS.items():
            val = pr.sasp_markers.get(marker)
            if val is not None:
                measured.add(marker)
                if val > hi:
                    elevated.add(marker)

    me = multi_omics_result.metabolomic
    if me is not None:
        measured.update(me.oxidative_stress_markers.keys())
        measured.update(me.inflammation_markers.keys())
        for marker, (lo, hi) in _OXIDATIVE_STRESS_MARKERS.items():
            val = me.oxidative_stress_markers.get(marker)
            if val is not None:
                measured.add(marker)
                if marker in {"8-OHdG", "MDA"}:
                    if val > hi:
                        elevated.add(marker)
                elif val < lo:
                    elevated.add(marker)

    N = max(len(measured), 1)
    n_elev = len(elevated)

    results: Dict[str, float] = {}
    for pw_name, pw_genes in pathways.items():
        K = sum(1 for g in pw_genes if g in measured)
        k = sum(1 for g in pw_genes if g in elevated)
        if K == 0:
            results[pw_name] = 1.0
            continue
        a, b, c = k, n_elev - k, K - k
        d = max((N - n_elev) - (K - k), 0)
        results[pw_name] = round(_fishers_exact_p(a, b, c, d), 6)

    return results
