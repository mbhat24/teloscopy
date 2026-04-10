"""Genetic disease risk prediction from SNP genotypes and telomere data.

Provides a polygenic risk scoring engine that combines known SNP–disease
associations with telomere-length measurements and optional image-analysis
outputs to produce per-condition lifetime risk estimates.  A built-in
database of ≥ 500 curated SNP–disease associations across 26 categories
covers cardiovascular disease, cancer predisposition, diabetes,
Alzheimer's, neurological disorders, autoimmune conditions, metabolic
disorders, eye diseases, bone health, blood disorders, respiratory,
kidney, liver, mental health, reproductive, dermatological,
gastrointestinal, endocrine, infectious susceptibility, dental, sleep,
aging, hearing, pain, pharmacogenomics, and cardiac arrhythmia.

**This module is intended for educational and research purposes only.
Predictions must NOT be used for clinical decision-making.**

Typical usage
-------------
>>> predictor = DiseasePredictor()
>>> profile = predictor.predict_from_variants(
...     variants={"rs429358": "CT", "rs7412": "CC"},
...     age=55,
...     sex="female",
... )
>>> for risk in profile.top_risks(n=5):
...     print(f"{risk.condition}: {risk.lifetime_risk_pct:.1f}%")
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DISCLAIMER",
    "DISCLAIMER_SHORT",
    "GeneticVariant",
    "DiseaseRisk",
    "RiskProfile",
    "DiseasePredictor",
    "BUILTIN_VARIANT_DB",
    "BASELINE_INCIDENCE",
]

# ---------------------------------------------------------------------------
# Disclaimer constants
# ---------------------------------------------------------------------------

DISCLAIMER: str = (
    "IMPORTANT DISCLAIMER: This genetic risk prediction module is provided "
    "strictly for educational and research purposes.  The results are NOT "
    "validated for clinical use and must NOT be used to make medical "
    "decisions.  Genetic risk assessment requires interpretation by a "
    "qualified genetic counsellor or clinical geneticist in conjunction "
    "with family history, lifestyle factors, and confirmatory laboratory "
    "testing.  The authors assume no liability for any consequences arising "
    "from the use or misuse of these predictions."
)

DISCLAIMER_SHORT: str = "For research/educational use only — not clinical advice."

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneticVariant:
    """A single SNP–disease association record.

    Attributes
    ----------
    rsid : str
        dbSNP reference identifier (e.g. ``"rs429358"``).
    gene : str
        HGNC gene symbol closest to or containing the variant.
    chromosome : str
        Chromosome on which the variant resides (e.g. ``"19"``).
    position : int
        GRCh38 genomic coordinate.
    risk_allele : str
        The allele associated with *increased* disease risk.
    protective_allele : str
        The alternative (reference / protective) allele.
    effect_size : float
        Per-allele odds ratio (OR).  Values > 1 indicate elevated risk.
    condition : str
        Human-readable disease or trait name.
    category : str
        Broad disease category (e.g. ``"cardiovascular"``).
    population_frequency : float
        Risk-allele frequency in the general population (0–1).
    evidence_level : str
        Strength of the association — one of ``"strong"``,
        ``"moderate"``, or ``"suggestive"``.
    """

    rsid: str
    gene: str
    chromosome: str
    position: int
    risk_allele: str
    protective_allele: str
    effect_size: float
    condition: str
    category: str
    population_frequency: float
    evidence_level: str  # "strong", "moderate", "suggestive"

    # -- helpers ----------------------------------------------------------

    def allele_count(self, genotype: str) -> int:
        """Return the number of risk alleles in a diploid *genotype* string.

        Parameters
        ----------
        genotype : str
            Two-character genotype (e.g. ``"CT"``).

        Returns
        -------
        int
            0, 1, or 2.
        """
        if len(genotype) != 2:
            return 0
        return sum(1 for a in genotype if a == self.risk_allele)


@dataclass()
class DiseaseRisk:
    """Predicted risk for a single disease or condition.

    Attributes
    ----------
    condition : str
        Name of the disease or condition.
    category : str
        Broad disease category.
    lifetime_risk_pct : float
        Estimated lifetime risk expressed as a percentage (0–100).
    relative_risk : float
        Risk relative to the population average (1.0 = average).
    confidence : float
        Confidence in the estimate (0–1), based on evidence quality
        and variant coverage.
    contributing_variants : list[str]
        ``rsid`` identifiers of variants that informed this estimate.
    age_of_onset_range : tuple[int, int]
        Typical (earliest, latest) age-of-onset window.
    preventability_score : float
        Rough score (0–1) indicating how modifiable the risk is through
        lifestyle or medical intervention.
    """

    condition: str
    category: str
    lifetime_risk_pct: float
    relative_risk: float
    confidence: float
    contributing_variants: list[str] = field(default_factory=list)
    age_of_onset_range: tuple[int, int] = (0, 100)
    preventability_score: float = 0.5

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.condition} ({self.category}): "
            f"lifetime risk {self.lifetime_risk_pct:.1f}%, "
            f"RR {self.relative_risk:.2f}, "
            f"confidence {self.confidence:.2f}"
        )


class RiskProfile:
    """Container for a complete set of per-condition :class:`DiseaseRisk` estimates.

    Parameters
    ----------
    risks : list[DiseaseRisk]
        Individual disease risk estimates.
    metadata : dict[str, Any]
        Arbitrary metadata (age, sex, variant count, …).
    """

    def __init__(
        self,
        risks: list[DiseaseRisk] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.risks: list[DiseaseRisk] = risks or []
        self.metadata: dict[str, Any] = metadata or {}

    # -- query helpers ----------------------------------------------------

    def top_risks(self, n: int = 10) -> list[DiseaseRisk]:
        """Return the *n* highest lifetime-risk conditions, sorted descending."""
        return sorted(self.risks, key=lambda r: r.lifetime_risk_pct, reverse=True)[:n]

    def filter_by_category(self, category: str) -> list[DiseaseRisk]:
        """Return risks belonging to a given *category* (case-insensitive)."""
        cat = category.lower()
        return [r for r in self.risks if r.category.lower() == cat]

    def filter_by_confidence(self, min_confidence: float = 0.5) -> list[DiseaseRisk]:
        """Return risks with confidence ≥ *min_confidence*."""
        return [r for r in self.risks if r.confidence >= min_confidence]

    def summary(self) -> pd.DataFrame:
        """Return a tidy :class:`~pandas.DataFrame` summarising all risks.

        Columns: ``condition``, ``category``, ``lifetime_risk_pct``,
        ``relative_risk``, ``confidence``, ``n_variants``,
        ``preventability_score``.
        """
        rows = [
            {
                "condition": r.condition,
                "category": r.category,
                "lifetime_risk_pct": round(r.lifetime_risk_pct, 2),
                "relative_risk": round(r.relative_risk, 3),
                "confidence": round(r.confidence, 3),
                "n_variants": len(r.contributing_variants),
                "preventability_score": round(r.preventability_score, 2),
                "onset_min_age": r.age_of_onset_range[0],
                "onset_max_age": r.age_of_onset_range[1],
            }
            for r in self.risks
        ]
        return pd.DataFrame(rows)

    @property
    def categories(self) -> list[str]:
        """Unique disease categories present in the profile."""
        return sorted({r.category for r in self.risks})

    def __len__(self) -> int:
        return len(self.risks)

    def __repr__(self) -> str:  # pragma: no cover
        return f"RiskProfile({len(self.risks)} conditions, categories={self.categories})"


# ---------------------------------------------------------------------------
# JSON data loading helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"


def _load_json(name: str) -> Any:
    """Read and return parsed JSON from *_DATA_DIR / name*."""
    with open(_DATA_DIR / name) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Built-in baseline incidence rates (per 100 000 person-years)
# Used for absolute-risk calibration.  Sources: GBD, SEER, literature.
# Keys are (condition, sex) with sex in {"male", "female", "all"}.
# ---------------------------------------------------------------------------

BASELINE_INCIDENCE: dict[tuple[str, str], float] = {
    (condition, sex): rate
    for condition, sexes in _load_json("baseline_incidence.json").items()
    for sex, rate in sexes.items()
}

# ---------------------------------------------------------------------------
# Built-in SNP–disease database  (≥ 500 entries)
# ---------------------------------------------------------------------------

BUILTIN_VARIANT_DB: list[GeneticVariant] = [
    GeneticVariant(**entry) for entry in _load_json("builtin_variant_db.json")
]

# Quick count assertion (fails at import time if DB is under-populated)
assert len(BUILTIN_VARIANT_DB) >= 500, (
    f"Built-in variant DB has only {len(BUILTIN_VARIANT_DB)} entries; expected ≥ 500."
)

# ---------------------------------------------------------------------------
# Internal constants used by the predictor
# ---------------------------------------------------------------------------

_EVIDENCE_WEIGHTS: dict[str, float] = {
    "strong": 1.0,
    "moderate": 0.6,
    "suggestive": 0.3,
}

# Age-of-onset windows (condition -> (min_age, max_age)).
_ONSET_RANGES: dict[str, tuple[int, int]] = {
    condition: (d["min"], d["max"])
    for condition, d in _load_json("onset_ranges.json").items()
}

# Preventability / modifiability scores by condition (0-1).
_PREVENTABILITY: dict[str, float] = _load_json("preventability_scores.json")

# Telomere-length risk modifiers: per-kb-shortening odds ratio multiplier
# (relative to age-adjusted expected length).
_TELOMERE_RISK_MODIFIERS: dict[str, float] = _load_json("telomere_risk_modifiers.json")

# Screening / prevention recommendations keyed by category.
_SCREENING_RECS: dict[str, list[dict[str, str]]] = _load_json("screening_recommendations.json")

class DiseasePredictor:
    """Polygenic disease risk prediction engine.

    Combines a built-in SNP–disease database with optional custom variant
    files, telomere-length data, and image-analysis results to compute
    per-condition lifetime risk estimates.

    Parameters
    ----------
    custom_db_path : str or Path or None
        Optional path to a JSON file containing additional
        :class:`GeneticVariant` records (list of dicts).  The records
        are merged with the built-in database.

    Examples
    --------
    >>> pred = DiseasePredictor()
    >>> profile = pred.predict_from_variants(
    ...     {"rs429358": "CC", "rs7903146": "TT"}, age=60, sex="male"
    ... )
    >>> len(profile) > 0
    True
    """

    def __init__(self, custom_db_path: str | Path | None = None) -> None:
        self._db: list[GeneticVariant] = list(BUILTIN_VARIANT_DB)

        if custom_db_path is not None:
            self._merge_custom_db(Path(custom_db_path))

        # Pre-index: condition → list[GeneticVariant]
        self._condition_index: dict[str, list[GeneticVariant]] = {}
        for v in self._db:
            self._condition_index.setdefault(v.condition, []).append(v)

        # Pre-index: rsid → list[GeneticVariant]
        self._rsid_index: dict[str, list[GeneticVariant]] = {}
        for v in self._db:
            self._rsid_index.setdefault(v.rsid, []).append(v)

    # -- database helpers -------------------------------------------------

    def _merge_custom_db(self, path: Path) -> None:
        """Load and merge a custom JSON variant database.

        Parameters
        ----------
        path : Path
            Path to a JSON file whose top-level value is a list of
            objects with the same keys as :class:`GeneticVariant`.
        """
        if not path.exists():
            warnings.warn(
                f"Custom database file not found: {path}",
                UserWarning,
                stacklevel=3,
            )
            return

        with open(path) as fh:
            raw: list[dict[str, Any]] = json.load(fh)

        for entry in raw:
            try:
                self._db.append(GeneticVariant(**entry))
            except TypeError as exc:
                warnings.warn(
                    f"Skipping malformed custom variant entry: {exc}",
                    UserWarning,
                    stacklevel=3,
                )

    @property
    def conditions(self) -> list[str]:
        """Sorted list of all conditions covered by the database."""
        return sorted(self._condition_index)

    @property
    def variant_count(self) -> int:
        """Total number of variant–condition associations in the database."""
        return len(self._db)

    # -- core predictions -------------------------------------------------

    def predict_from_variants(
        self,
        variants: dict[str, str],
        age: int,
        sex: str,
    ) -> RiskProfile:
        """Predict disease risks from a genotype map.

        Parameters
        ----------
        variants : dict[str, str]
            Mapping of ``rsid`` → diploid genotype string (e.g.
            ``{"rs429358": "CT", "rs7903146": "TT"}``).
        age : int
            Current age of the individual (years).
        sex : str
            Biological sex — ``"male"`` or ``"female"``.

        Returns
        -------
        RiskProfile
            A profile containing :class:`DiseaseRisk` entries for all
            conditions with ≥ 1 matching variant.
        """
        sex = sex.lower().strip()
        if sex not in ("male", "female"):
            raise ValueError(f"sex must be 'male' or 'female', got '{sex}'")

        # Collect per-condition variant hits.
        condition_hits: dict[str, list[tuple[GeneticVariant, int]]] = {}
        for rsid, genotype in variants.items():
            for var in self._rsid_index.get(rsid, []):
                n = var.allele_count(genotype)
                if n > 0:
                    condition_hits.setdefault(var.condition, []).append((var, n))

        risks: list[DiseaseRisk] = []
        for condition, hits in condition_hits.items():
            category = hits[0][0].category
            rr = self._combine_odds_ratios(hits)
            baseline = self._get_baseline_incidence(condition, sex)
            lifetime_risk = self._compute_lifetime_risk(baseline, rr, age, condition)
            confidence = self._compute_confidence(hits)

            risks.append(
                DiseaseRisk(
                    condition=condition,
                    category=category,
                    lifetime_risk_pct=round(lifetime_risk, 4),
                    relative_risk=round(rr, 4),
                    confidence=round(confidence, 4),
                    contributing_variants=[h[0].rsid for h in hits],
                    age_of_onset_range=_ONSET_RANGES.get(condition, (0, 100)),
                    preventability_score=_PREVENTABILITY.get(condition, 0.5),
                )
            )

        metadata = {
            "age": age,
            "sex": sex,
            "input_variants": len(variants),
            "matched_variants": sum(len(h) for h in condition_hits.values()),
            "disclaimer": DISCLAIMER_SHORT,
        }
        return RiskProfile(risks=risks, metadata=metadata)

    def predict_from_telomere_data(
        self,
        mean_length_bp: float,
        age: int,
        sex: str,
    ) -> list[DiseaseRisk]:
        """Predict disease risks based on telomere length.

        Short telomeres (below the age-adjusted expected length) are
        associated with increased cardiovascular and cancer risk.  The
        expected length is approximated as:

            ``expected = 11000 − 40 * age``  (bp)

        using a consensus linear model (Müezzinler et al. 2013;
        Aubert & Lansdorp 2008).  A per-kb shortening multiplier is
        applied to conditions listed in ``_TELOMERE_RISK_MODIFIERS``.

        Parameters
        ----------
        mean_length_bp : float
            Mean telomere length in base pairs.
        age : int
            Current age (years).
        sex : str
            ``"male"`` or ``"female"``.

        Returns
        -------
        list[DiseaseRisk]
            One :class:`DiseaseRisk` per condition modified by telomere
            length.
        """
        sex = sex.lower().strip()
        expected_bp = 11_000.0 - 40.0 * age
        shortening_kb = max(0.0, (expected_bp - mean_length_bp) / 1000.0)

        risks: list[DiseaseRisk] = []
        for condition, per_kb_or in _TELOMERE_RISK_MODIFIERS.items():
            rr = per_kb_or**shortening_kb
            if rr <= 1.0:
                continue

            category = self._category_for_condition(condition)
            baseline = self._get_baseline_incidence(condition, sex)
            lifetime = self._compute_lifetime_risk(baseline, rr, age, condition)

            # Confidence is lower for telomere-only prediction.
            confidence = min(0.45, 0.15 * shortening_kb)

            risks.append(
                DiseaseRisk(
                    condition=condition,
                    category=category,
                    lifetime_risk_pct=round(lifetime, 4),
                    relative_risk=round(rr, 4),
                    confidence=round(confidence, 4),
                    contributing_variants=["telomere_length"],
                    age_of_onset_range=_ONSET_RANGES.get(condition, (0, 100)),
                    preventability_score=_PREVENTABILITY.get(condition, 0.5),
                )
            )
        return risks

    def predict_from_image_analysis(
        self,
        analysis_results: dict[str, Any],
    ) -> list[DiseaseRisk]:
        """Derive risk signals from the existing Teloscopy image-analysis pipeline.

        Integrates with outputs from :mod:`teloscopy.telomere.pipeline` and
        :mod:`teloscopy.analysis.statistics`.  The following keys are
        inspected (all optional):

        - ``"mean_intensity"`` → proxy for mean telomere length.
        - ``"cv"`` → high coefficient of variation may indicate genomic
          instability.
        - ``"n_telomeres"`` → very low counts can be a QC flag.
        - ``"age"`` and ``"sex"`` → demographics for risk calibration.

        Parameters
        ----------
        analysis_results : dict[str, Any]
            Dictionary produced by upstream analysis modules.

        Returns
        -------
        list[DiseaseRisk]
            Risk entries derived from image-based telomere measurements.
        """
        mean_intensity = analysis_results.get("mean_intensity")
        cv = analysis_results.get("cv", 0.0)
        age = analysis_results.get("age", 50)
        sex = analysis_results.get("sex", "female")

        risks: list[DiseaseRisk] = []

        # Use mean intensity as a telomere-length proxy.  This is an
        # *approximate* conversion that depends heavily on microscope
        # settings, probe efficiency, and exposure time.  Properly
        # calibrated conversion should use quantification.Calibration.
        if mean_intensity is not None and mean_intensity > 0:
            warnings.warn(
                "Using uncalibrated intensity→bp proxy (×1.5); "
                "results are approximate.  Use quantification.Calibration "
                "for instrument-specific conversion.",
                stacklevel=2,
            )
            proxy_bp = float(mean_intensity) * 1.5
            risks.extend(self.predict_from_telomere_data(proxy_bp, age, sex))

        # High CV → genomic instability signal → elevated cancer risk.
        if cv > 0.5:
            instability_rr = 1.0 + (cv - 0.5) * 0.4  # mild linear model
            for condition in ("Breast cancer", "Colorectal cancer", "Lung cancer"):
                category = self._category_for_condition(condition)
                baseline = self._get_baseline_incidence(condition, sex)
                lifetime = self._compute_lifetime_risk(baseline, instability_rr, age, condition)
                risks.append(
                    DiseaseRisk(
                        condition=condition,
                        category=category,
                        lifetime_risk_pct=round(lifetime, 4),
                        relative_risk=round(instability_rr, 4),
                        confidence=round(min(0.30, (cv - 0.5) * 0.3), 4),
                        contributing_variants=["image_cv_instability"],
                        age_of_onset_range=_ONSET_RANGES.get(condition, (0, 100)),
                        preventability_score=_PREVENTABILITY.get(condition, 0.5),
                    )
                )

        return risks

    def calculate_polygenic_risk(
        self,
        variants: dict[str, str],
        condition: str,
    ) -> float:
        """Compute a polygenic risk score (PRS) for a single condition.

        The score is a weighted sum of risk-allele dosages across all
        variants in the database for the given condition:

        .. math::

            PRS = \\sum_i \\bigl(\\text{dosage}_i \\times \\ln(OR_i)
                  \\times w_i\\bigr)

        where *w* is the evidence-level weight (1.0 / 0.6 / 0.3).

        Parameters
        ----------
        variants : dict[str, str]
            ``rsid`` → genotype mapping.
        condition : str
            Exact condition name (case-sensitive) as it appears in the
            variant database.

        Returns
        -------
        float
            Raw (un-normalised) polygenic risk score.  Higher values
            indicate greater genetic predisposition.
        """
        db_variants = self._condition_index.get(condition, [])
        if not db_variants:
            return 0.0

        score = 0.0
        for var in db_variants:
            genotype = variants.get(var.rsid)
            if genotype is None:
                continue
            dosage = var.allele_count(genotype)
            if dosage == 0:
                continue
            weight = _EVIDENCE_WEIGHTS.get(var.evidence_level, 0.3)
            # Use natural log of OR so that protective (OR<1) variants
            # contribute negatively and risk (OR>1) contribute positively.
            score += dosage * math.log(var.effect_size) * weight
        return round(score, 6)

    def project_risk_over_time(
        self,
        risk_profile: RiskProfile,
        current_age: int,
        years: int = 30,
    ) -> dict[str, list[dict[str, float]]]:
        """Project cumulative risk year-by-year for each condition.

        Uses a simplified proportional-hazards model where the annual
        hazard ``h(t)`` at age *t* is the baseline incidence (per
        person-year) scaled by the individual's relative risk and
        concentrated within the condition's age-of-onset window.

        Parameters
        ----------
        risk_profile : RiskProfile
            Previously computed risk profile.
        current_age : int
            The individual's current age.
        years : int
            Number of future years to project (default 30).

        Returns
        -------
        dict[str, list[dict[str, float]]]
            Mapping of ``condition`` → list of dicts, one per year, each
            containing ``"age"`` and ``"cumulative_risk_pct"``.
        """
        sex = risk_profile.metadata.get("sex", "all")
        projections: dict[str, list[dict[str, float]]] = {}

        for risk in risk_profile.risks:
            condition = risk.condition
            baseline = self._get_baseline_incidence(condition, sex)
            onset_min, onset_max = risk.age_of_onset_range

            cumulative = 0.0
            yearly: list[dict[str, float]] = []

            for y in range(years + 1):
                age = current_age + y
                if onset_min <= age <= onset_max:
                    # Annual hazard rate (per-person) scaled by RR.
                    annual_hazard = (baseline / 100_000) * risk.relative_risk
                    # Apply age-weighting: peak risk in the middle of the
                    # onset window.
                    window_mid = (onset_min + onset_max) / 2.0
                    window_half = max((onset_max - onset_min) / 2.0, 1.0)
                    age_weight = math.exp(-0.5 * ((age - window_mid) / window_half) ** 2)
                    annual_hazard *= age_weight
                    # Convert hazard to probability and accumulate.
                    survival = 1.0 - cumulative / 100.0
                    increment = annual_hazard * survival * 100.0
                    cumulative += increment

                cumulative = min(cumulative, 100.0)
                yearly.append(
                    {
                        "age": age,
                        "cumulative_risk_pct": round(cumulative, 4),
                    }
                )

            projections[condition] = yearly

        return projections

    def get_actionable_insights(
        self,
        risk_profile: RiskProfile,
    ) -> list[dict[str, Any]]:
        """Return prevention and screening recommendations.

        Recommendations are derived from the risk profile's categories
        and individual conditions, prioritised by relative risk and
        preventability.

        Parameters
        ----------
        risk_profile : RiskProfile
            Previously computed risk profile.

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains:

            - ``"condition"`` (str)
            - ``"category"`` (str)
            - ``"risk_level"`` (str): ``"high"`` / ``"moderate"`` / ``"low"``
            - ``"relative_risk"`` (float)
            - ``"recommendations"`` (list[dict]): specific screening / lifestyle
              actions.
            - ``"disclaimer"`` (str)
        """
        insights: list[dict[str, Any]] = []

        # Sort by relative risk descending so highest-risk comes first.
        ordered = sorted(
            risk_profile.risks,
            key=lambda r: r.relative_risk,
            reverse=True,
        )

        for risk in ordered:
            if risk.relative_risk < 1.0:
                level = "low"
            elif risk.relative_risk < 1.5:
                level = "moderate"
            else:
                level = "high"

            cat_recs = _SCREENING_RECS.get(risk.category, [])

            # Condition-specific supplementary recommendations.
            condition_recs = self._condition_specific_recs(risk)

            insights.append(
                {
                    "condition": risk.condition,
                    "category": risk.category,
                    "risk_level": level,
                    "relative_risk": risk.relative_risk,
                    "recommendations": cat_recs + condition_recs,
                    "disclaimer": DISCLAIMER_SHORT,
                }
            )

        return insights

    # -- internal calculation helpers -------------------------------------

    @staticmethod
    def _combine_odds_ratios(
        hits: list[tuple[GeneticVariant, int]],
    ) -> float:
        """Combine per-variant ORs using a multiplicative model.

        Each variant's OR is raised to the power of the risk-allele
        dosage (0/1/2) and weighted by evidence level.  The combined
        relative risk is the product of all weighted ORs.

        Parameters
        ----------
        hits : list[tuple[GeneticVariant, int]]
            (variant, allele_count) pairs for a single condition.

        Returns
        -------
        float
            Combined relative risk (≥ 0).
        """
        log_rr = 0.0
        for var, dosage in hits:
            w = _EVIDENCE_WEIGHTS.get(var.evidence_level, 0.3)
            log_rr += dosage * math.log(var.effect_size) * w
        return math.exp(log_rr)

    @staticmethod
    def _get_baseline_incidence(condition: str, sex: str) -> float:
        """Look up the baseline incidence rate (per 100 000 person-years).

        Falls back from sex-specific to ``"all"`` if not found, and
        ultimately returns a conservative default of 50.0.
        """
        key_sex = (condition, sex)
        key_all = (condition, "all")
        return BASELINE_INCIDENCE.get(key_sex, BASELINE_INCIDENCE.get(key_all, 50.0))

    @staticmethod
    def _compute_lifetime_risk(
        baseline_per_100k: float,
        relative_risk: float,
        current_age: int,
        condition: str,
    ) -> float:
        """Convert baseline incidence + RR into a remaining-lifetime risk %.

        Uses a simplified competing-risks survival model:

        .. math::

            P = 1 - \\exp\\bigl(-\\lambda \\cdot RR \\cdot T_{\\text{eff}}\\bigr)

        where *T_eff* is the number of remaining at-risk years within the
        onset window.

        Parameters
        ----------
        baseline_per_100k : float
            Baseline incidence per 100 000 person-years.
        relative_risk : float
            Individual's relative risk vs. population.
        current_age : int
            Current age.
        condition : str
            Condition name (used to look up onset window).

        Returns
        -------
        float
            Lifetime risk percentage (0–100).
        """
        onset_min, onset_max = _ONSET_RANGES.get(condition, (0, 100))
        effective_start = max(current_age, onset_min)
        at_risk_years = max(0, onset_max - effective_start)

        if at_risk_years <= 0:
            return 0.0

        annual_rate = baseline_per_100k / 100_000
        cumulative_hazard = annual_rate * relative_risk * at_risk_years
        lifetime_prob = 1.0 - math.exp(-cumulative_hazard)
        return lifetime_prob * 100.0

    @staticmethod
    def _compute_confidence(
        hits: list[tuple[GeneticVariant, int]],
    ) -> float:
        """Estimate confidence in the risk prediction.

        Confidence grows with the number of variants and their evidence
        levels, saturating at 0.95.

        Parameters
        ----------
        hits : list[tuple[GeneticVariant, int]]
            (variant, dosage) pairs.

        Returns
        -------
        float
            Confidence score in [0, 0.95].
        """
        if not hits:
            return 0.0

        evidence_sum = sum(_EVIDENCE_WEIGHTS.get(v.evidence_level, 0.3) for v, _ in hits)
        # Logistic-style saturation: more evidence → higher confidence.
        raw = 1.0 - math.exp(-0.5 * evidence_sum)
        return min(raw, 0.95)

    def _category_for_condition(self, condition: str) -> str:
        """Return the category string for a condition, or ``"unknown"``."""
        variants = self._condition_index.get(condition, [])
        if variants:
            return variants[0].category
        return "unknown"

    @staticmethod
    def _condition_specific_recs(risk: DiseaseRisk) -> list[dict[str, str]]:
        """Generate condition-specific recommendations beyond category defaults.

        Parameters
        ----------
        risk : DiseaseRisk
            The individual disease risk entry.

        Returns
        -------
        list[dict[str, str]]
            Additional recommendation dicts (may be empty).
        """
        recs: list[dict[str, str]] = []
        cond = risk.condition

        if cond == "Breast cancer" and risk.relative_risk >= 2.0:
            recs.append(
                {
                    "action": "Discuss enhanced breast-cancer screening",
                    "frequency": "Annual MRI + mammogram from age 30",
                    "detail": "Consider risk-reducing strategies (e.g. "
                    "chemoprevention, prophylactic surgery) with "
                    "oncologist.",
                }
            )
        elif cond == "Coronary artery disease" and risk.relative_risk >= 1.5:
            recs.append(
                {
                    "action": "Coronary artery calcium (CAC) score",
                    "frequency": "Once at age 40–50; repeat per cardiologist",
                    "detail": "Non-invasive CT scan to quantify coronary "
                    "calcification and refine risk.",
                }
            )
        elif cond == "Type 2 diabetes" and risk.relative_risk >= 1.3:
            recs.append(
                {
                    "action": "Oral glucose tolerance test (OGTT)",
                    "frequency": "Annually",
                    "detail": "More sensitive than fasting glucose for "
                    "detecting impaired glucose tolerance.",
                }
            )
        elif cond == "Alzheimer's disease" and risk.relative_risk >= 2.0:
            recs.append(
                {
                    "action": "Discuss APOE-informed prevention trial enrolment",
                    "frequency": "Once",
                    "detail": "Several clinical trials target APOE-ε4 carriers; "
                    "genetic counsellor can advise on eligibility.",
                }
            )
        elif cond == "Sickle cell disease" and risk.relative_risk >= 2.0:
            recs.append(
                {
                    "action": "Haemoglobin electrophoresis",
                    "frequency": "Once (confirmatory)",
                    "detail": "Confirm carrier vs. disease status; partner screening recommended.",
                }
            )
        elif cond == "Hereditary haemochromatosis" and risk.relative_risk >= 2.0:
            recs.append(
                {
                    "action": "Serum ferritin + transferrin saturation",
                    "frequency": "Every 6–12 months",
                    "detail": "Therapeutic phlebotomy if iron overload confirmed.",
                }
            )

        return recs
