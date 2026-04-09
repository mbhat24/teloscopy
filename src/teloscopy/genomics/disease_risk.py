"""Genetic disease risk prediction from SNP genotypes and telomere data.

Provides a polygenic risk scoring engine that combines known SNP–disease
associations with telomere-length measurements and optional image-analysis
outputs to produce per-condition lifetime risk estimates.  A built-in
database of ≥ 50 curated SNP–disease associations covers cardiovascular
disease, cancer predisposition, diabetes, Alzheimer's, autoimmune
conditions, metabolic disorders, eye diseases, neurological conditions,
bone health, and blood disorders.

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
# Built-in baseline incidence rates (per 100 000 person-years)
# Used for absolute-risk calibration.  Sources: GBD, SEER, literature.
# Keys are (condition, sex) with sex in {"male", "female", "all"}.
# ---------------------------------------------------------------------------

BASELINE_INCIDENCE: dict[tuple[str, str], float] = {
    # Cardiovascular
    ("Coronary artery disease", "male"): 580.0,
    ("Coronary artery disease", "female"): 340.0,
    ("Hypercholesterolaemia", "all"): 1200.0,
    ("Atrial fibrillation", "male"): 210.0,
    ("Atrial fibrillation", "female"): 130.0,
    ("Aortic stenosis", "all"): 45.0,
    ("Hypertension", "all"): 1500.0,
    # Cancer
    ("Breast cancer", "female"): 130.0,
    ("Breast cancer", "male"): 1.3,
    ("Ovarian cancer", "female"): 11.0,
    ("Colorectal cancer", "male"): 45.0,
    ("Colorectal cancer", "female"): 34.0,
    ("Li-Fraumeni syndrome cancers", "all"): 2.0,
    ("Endometrial cancer", "female"): 28.0,
    ("Lynch syndrome cancers", "all"): 5.0,
    ("Prostate cancer", "male"): 110.0,
    ("Lung cancer", "male"): 60.0,
    ("Lung cancer", "female"): 45.0,
    ("Melanoma", "all"): 22.0,
    # Diabetes
    ("Type 2 diabetes", "all"): 600.0,
    # Alzheimer's / neuro
    ("Alzheimer's disease", "male"): 120.0,
    ("Alzheimer's disease", "female"): 170.0,
    ("Parkinson's disease", "male"): 20.0,
    ("Parkinson's disease", "female"): 12.0,
    ("Huntington's disease", "all"): 0.5,
    # Autoimmune
    ("Rheumatoid arthritis", "male"): 25.0,
    ("Rheumatoid arthritis", "female"): 50.0,
    ("Type 1 diabetes", "all"): 15.0,
    ("Systemic lupus erythematosus", "female"): 5.0,
    ("Systemic lupus erythematosus", "male"): 0.5,
    # Metabolic
    ("Obesity", "all"): 900.0,
    ("Hyperhomocysteinaemia", "all"): 200.0,
    ("Neural tube defects (offspring)", "female"): 10.0,
    # Eye
    ("Age-related macular degeneration", "all"): 80.0,
    # Bone
    ("Osteoporosis", "female"): 300.0,
    ("Osteoporosis", "male"): 80.0,
    # Blood
    ("Sickle cell disease", "all"): 3.0,
    ("Hereditary haemochromatosis", "all"): 5.0,
    ("Beta-thalassaemia", "all"): 2.0,
}

# ---------------------------------------------------------------------------
# Built-in SNP–disease database  (≥ 50 entries)
# ---------------------------------------------------------------------------

BUILTIN_VARIANT_DB: list[GeneticVariant] = [
    # ===================================================================
    # CARDIOVASCULAR  (genes: APOE, PCSK9, LPA, LDLR, and others)
    # ===================================================================
    GeneticVariant(
        "rs429358",
        "APOE",
        "19",
        44908684,
        "C",
        "T",
        1.45,
        "Coronary artery disease",
        "cardiovascular",
        0.15,
        "strong",
    ),
    GeneticVariant(
        "rs7412",
        "APOE",
        "19",
        44908822,
        "C",
        "T",
        0.80,
        "Coronary artery disease",
        "cardiovascular",
        0.08,
        "strong",
    ),
    GeneticVariant(
        "rs11591147",
        "PCSK9",
        "1",
        55505647,
        "T",
        "G",
        0.50,
        "Hypercholesterolaemia",
        "cardiovascular",
        0.01,
        "strong",
    ),
    GeneticVariant(
        "rs11206510",
        "PCSK9",
        "1",
        55496039,
        "T",
        "C",
        1.08,
        "Coronary artery disease",
        "cardiovascular",
        0.82,
        "strong",
    ),
    GeneticVariant(
        "rs10455872",
        "LPA",
        "6",
        160589086,
        "G",
        "A",
        1.51,
        "Coronary artery disease",
        "cardiovascular",
        0.07,
        "strong",
    ),
    GeneticVariant(
        "rs3798220",
        "LPA",
        "6",
        160540105,
        "C",
        "T",
        1.47,
        "Coronary artery disease",
        "cardiovascular",
        0.02,
        "strong",
    ),
    GeneticVariant(
        "rs688",
        "LDLR",
        "19",
        11113592,
        "T",
        "C",
        1.10,
        "Hypercholesterolaemia",
        "cardiovascular",
        0.43,
        "moderate",
    ),
    GeneticVariant(
        "rs5925",
        "LDLR",
        "19",
        11120511,
        "T",
        "C",
        1.08,
        "Hypercholesterolaemia",
        "cardiovascular",
        0.40,
        "moderate",
    ),
    # --- Additional cardiovascular ---
    GeneticVariant(
        "rs1333049",
        "CDKN2B-AS1",
        "9",
        22125504,
        "C",
        "G",
        1.29,
        "Coronary artery disease",
        "cardiovascular",
        0.47,
        "strong",
    ),
    GeneticVariant(
        "rs10757274",
        "CDKN2B-AS1",
        "9",
        22096055,
        "G",
        "A",
        1.25,
        "Coronary artery disease",
        "cardiovascular",
        0.49,
        "strong",
    ),
    # ===================================================================
    # CANCER  (genes: BRCA1, BRCA2, TP53, APC, MLH1, MSH2)
    # ===================================================================
    GeneticVariant(
        "rs80357906",
        "BRCA1",
        "17",
        43093449,
        "A",
        "G",
        11.0,
        "Breast cancer",
        "cancer",
        0.001,
        "strong",
    ),
    GeneticVariant(
        "rs80357906",
        "BRCA1",
        "17",
        43093449,
        "A",
        "G",
        8.0,
        "Ovarian cancer",
        "cancer",
        0.001,
        "strong",
    ),
    GeneticVariant(
        "rs766173",
        "BRCA2",
        "13",
        32906729,
        "A",
        "C",
        3.0,
        "Breast cancer",
        "cancer",
        0.01,
        "strong",
    ),
    GeneticVariant(
        "rs11571833",
        "BRCA2",
        "13",
        32972626,
        "T",
        "A",
        2.70,
        "Breast cancer",
        "cancer",
        0.01,
        "strong",
    ),
    GeneticVariant(
        "rs28897743",
        "TP53",
        "17",
        7674220,
        "A",
        "G",
        5.0,
        "Li-Fraumeni syndrome cancers",
        "cancer",
        0.0005,
        "strong",
    ),
    GeneticVariant(
        "rs121913332",
        "APC",
        "5",
        112175211,
        "A",
        "G",
        10.0,
        "Colorectal cancer",
        "cancer",
        0.0001,
        "strong",
    ),
    GeneticVariant(
        "rs63750447",
        "MLH1",
        "3",
        37042337,
        "A",
        "G",
        4.0,
        "Lynch syndrome cancers",
        "cancer",
        0.001,
        "strong",
    ),
    GeneticVariant(
        "rs267607908",
        "MSH2",
        "2",
        47630553,
        "A",
        "G",
        4.2,
        "Lynch syndrome cancers",
        "cancer",
        0.001,
        "strong",
    ),
    # --- Additional cancer susceptibility ---
    GeneticVariant(
        "rs1042522",
        "TP53",
        "17",
        7676154,
        "C",
        "G",
        1.15,
        "Lung cancer",
        "cancer",
        0.63,
        "moderate",
    ),
    GeneticVariant(
        "rs6983267",
        "MYC",
        "8",
        128413305,
        "G",
        "T",
        1.27,
        "Colorectal cancer",
        "cancer",
        0.50,
        "strong",
    ),
    # ===================================================================
    # DIABETES  (genes: TCF7L2, PPARG, KCNJ11, SLC30A8)
    # ===================================================================
    GeneticVariant(
        "rs7903146",
        "TCF7L2",
        "10",
        112998590,
        "T",
        "C",
        1.37,
        "Type 2 diabetes",
        "diabetes",
        0.30,
        "strong",
    ),
    GeneticVariant(
        "rs12255372",
        "TCF7L2",
        "10",
        113049143,
        "T",
        "G",
        1.33,
        "Type 2 diabetes",
        "diabetes",
        0.29,
        "strong",
    ),
    GeneticVariant(
        "rs1801282",
        "PPARG",
        "3",
        12351626,
        "G",
        "C",
        0.86,
        "Type 2 diabetes",
        "diabetes",
        0.12,
        "strong",
    ),
    GeneticVariant(
        "rs5219",
        "KCNJ11",
        "11",
        17388025,
        "T",
        "C",
        1.15,
        "Type 2 diabetes",
        "diabetes",
        0.35,
        "strong",
    ),
    GeneticVariant(
        "rs13266634",
        "SLC30A8",
        "8",
        117172544,
        "C",
        "T",
        1.12,
        "Type 2 diabetes",
        "diabetes",
        0.70,
        "strong",
    ),
    GeneticVariant(
        "rs10811661",
        "CDKN2A/B",
        "9",
        22134094,
        "T",
        "C",
        1.20,
        "Type 2 diabetes",
        "diabetes",
        0.83,
        "moderate",
    ),
    # ===================================================================
    # ALZHEIMER'S  (genes: APOE, CLU, PICALM, BIN1)
    # ===================================================================
    GeneticVariant(
        "rs429358",
        "APOE",
        "19",
        44908684,
        "C",
        "T",
        3.68,
        "Alzheimer's disease",
        "alzheimers",
        0.15,
        "strong",
    ),
    GeneticVariant(
        "rs7412",
        "APOE",
        "19",
        44908822,
        "C",
        "T",
        0.62,
        "Alzheimer's disease",
        "alzheimers",
        0.08,
        "strong",
    ),
    GeneticVariant(
        "rs11136000",
        "CLU",
        "8",
        27464519,
        "T",
        "C",
        0.86,
        "Alzheimer's disease",
        "alzheimers",
        0.38,
        "strong",
    ),
    GeneticVariant(
        "rs3851179",
        "PICALM",
        "11",
        85867875,
        "T",
        "C",
        0.87,
        "Alzheimer's disease",
        "alzheimers",
        0.36,
        "strong",
    ),
    GeneticVariant(
        "rs744373",
        "BIN1",
        "2",
        127137039,
        "G",
        "A",
        1.17,
        "Alzheimer's disease",
        "alzheimers",
        0.29,
        "strong",
    ),
    GeneticVariant(
        "rs6656401",
        "CR1",
        "1",
        207518704,
        "A",
        "G",
        1.18,
        "Alzheimer's disease",
        "alzheimers",
        0.20,
        "moderate",
    ),
    # ===================================================================
    # AUTOIMMUNE  (genes: HLA-DRB1, CTLA4, PTPN22)
    # ===================================================================
    GeneticVariant(
        "rs6897932",
        "HLA-DRB1",
        "6",
        32578775,
        "C",
        "T",
        2.50,
        "Rheumatoid arthritis",
        "autoimmune",
        0.10,
        "strong",
    ),
    GeneticVariant(
        "rs3087243",
        "CTLA4",
        "2",
        203867991,
        "G",
        "A",
        1.20,
        "Type 1 diabetes",
        "autoimmune",
        0.42,
        "strong",
    ),
    GeneticVariant(
        "rs2476601",
        "PTPN22",
        "1",
        113834946,
        "A",
        "G",
        1.75,
        "Rheumatoid arthritis",
        "autoimmune",
        0.10,
        "strong",
    ),
    GeneticVariant(
        "rs2476601",
        "PTPN22",
        "1",
        113834946,
        "A",
        "G",
        1.90,
        "Type 1 diabetes",
        "autoimmune",
        0.10,
        "strong",
    ),
    GeneticVariant(
        "rs2476601",
        "PTPN22",
        "1",
        113834946,
        "A",
        "G",
        1.40,
        "Systemic lupus erythematosus",
        "autoimmune",
        0.10,
        "moderate",
    ),
    # ===================================================================
    # METABOLIC DISORDERS  (genes: MTHFR, FTO, MC4R)
    # ===================================================================
    GeneticVariant(
        "rs1801133",
        "MTHFR",
        "1",
        11796321,
        "T",
        "C",
        1.20,
        "Hyperhomocysteinaemia",
        "metabolic",
        0.35,
        "strong",
    ),
    GeneticVariant(
        "rs1801131",
        "MTHFR",
        "1",
        11794419,
        "C",
        "A",
        1.10,
        "Neural tube defects (offspring)",
        "metabolic",
        0.30,
        "moderate",
    ),
    GeneticVariant(
        "rs9939609", "FTO", "16", 53786615, "A", "T", 1.30, "Obesity", "metabolic", 0.42, "strong"
    ),
    GeneticVariant(
        "rs1558902", "FTO", "16", 53803574, "A", "T", 1.32, "Obesity", "metabolic", 0.42, "strong"
    ),
    GeneticVariant(
        "rs17782313", "MC4R", "18", 60183864, "C", "T", 1.18, "Obesity", "metabolic", 0.24, "strong"
    ),
    # ===================================================================
    # EYE DISEASES  (genes: CFH, ARMS2 — macular degeneration)
    # ===================================================================
    GeneticVariant(
        "rs1061170",
        "CFH",
        "1",
        196659237,
        "C",
        "T",
        2.45,
        "Age-related macular degeneration",
        "eye",
        0.34,
        "strong",
    ),
    GeneticVariant(
        "rs10490924",
        "ARMS2",
        "10",
        122454932,
        "T",
        "G",
        2.69,
        "Age-related macular degeneration",
        "eye",
        0.22,
        "strong",
    ),
    GeneticVariant(
        "rs2230199",
        "C3",
        "19",
        6677897,
        "G",
        "C",
        1.42,
        "Age-related macular degeneration",
        "eye",
        0.20,
        "moderate",
    ),
    # ===================================================================
    # NEUROLOGICAL  (genes: LRRK2, PARK2, HTT)
    # ===================================================================
    GeneticVariant(
        "rs34637584",
        "LRRK2",
        "12",
        40340400,
        "A",
        "G",
        9.50,
        "Parkinson's disease",
        "neurological",
        0.001,
        "strong",
    ),
    GeneticVariant(
        "rs33939927",
        "LRRK2",
        "12",
        40346052,
        "A",
        "G",
        3.00,
        "Parkinson's disease",
        "neurological",
        0.005,
        "strong",
    ),
    GeneticVariant(
        "rs1801582",
        "PARK2",
        "6",
        161768590,
        "A",
        "G",
        1.30,
        "Parkinson's disease",
        "neurological",
        0.15,
        "moderate",
    ),
    GeneticVariant(
        "rs3758549",
        "HTT",
        "4",
        3076408,
        "A",
        "G",
        1.10,
        "Huntington's disease",
        "neurological",
        0.05,
        "suggestive",
    ),
    # ===================================================================
    # BONE HEALTH  (genes: ESR1, VDR, COL1A1)
    # ===================================================================
    GeneticVariant(
        "rs2234693",
        "ESR1",
        "6",
        151842246,
        "T",
        "C",
        1.20,
        "Osteoporosis",
        "bone",
        0.45,
        "moderate",
    ),
    GeneticVariant(
        "rs1544410", "VDR", "12", 47846052, "A", "G", 1.15, "Osteoporosis", "bone", 0.40, "moderate"
    ),
    GeneticVariant(
        "rs1800012",
        "COL1A1",
        "17",
        50200388,
        "T",
        "G",
        1.30,
        "Osteoporosis",
        "bone",
        0.18,
        "moderate",
    ),
    # ===================================================================
    # BLOOD DISORDERS  (genes: HBB, HFE)
    # ===================================================================
    GeneticVariant(
        "rs334",
        "HBB",
        "11",
        5227002,
        "T",
        "A",
        15.0,
        "Sickle cell disease",
        "blood",
        0.06,
        "strong",
    ),
    GeneticVariant(
        "rs33930165",
        "HBB",
        "11",
        5226773,
        "A",
        "G",
        8.0,
        "Beta-thalassaemia",
        "blood",
        0.02,
        "strong",
    ),
    GeneticVariant(
        "rs1800562",
        "HFE",
        "6",
        26092913,
        "A",
        "G",
        5.00,
        "Hereditary haemochromatosis",
        "blood",
        0.06,
        "strong",
    ),
    GeneticVariant(
        "rs1799945",
        "HFE",
        "6",
        26091179,
        "G",
        "C",
        1.60,
        "Hereditary haemochromatosis",
        "blood",
        0.14,
        "strong",
    ),
    # ===================================================================
    # Additional variants to reach ≥ 50 entries
    # ===================================================================
    # Prostate cancer
    GeneticVariant(
        "rs1447295",
        "MSMB",
        "8",
        128194098,
        "A",
        "C",
        1.62,
        "Prostate cancer",
        "cancer",
        0.11,
        "strong",
    ),
    # Melanoma
    GeneticVariant(
        "rs910873", "CDKN2A", "9", 21975862, "C", "T", 1.75, "Melanoma", "cancer", 0.01, "moderate"
    ),
    # Endometrial cancer
    GeneticVariant(
        "rs4430796",
        "HNF1B",
        "17",
        37738049,
        "G",
        "A",
        1.20,
        "Endometrial cancer",
        "cancer",
        0.47,
        "moderate",
    ),
    # Additional diabetes
    GeneticVariant(
        "rs7756992",
        "CDKAL1",
        "6",
        20661019,
        "G",
        "A",
        1.12,
        "Type 2 diabetes",
        "diabetes",
        0.28,
        "strong",
    ),
    # Atrial fibrillation
    GeneticVariant(
        "rs2200733",
        "PITX2",
        "4",
        111710169,
        "T",
        "C",
        1.72,
        "Atrial fibrillation",
        "cardiovascular",
        0.11,
        "strong",
    ),
    # Aortic stenosis
    GeneticVariant(
        "rs10455872",
        "LPA",
        "6",
        160589086,
        "G",
        "A",
        1.68,
        "Aortic stenosis",
        "cardiovascular",
        0.07,
        "strong",
    ),
    # Hypertension
    GeneticVariant(
        "rs699",
        "AGT",
        "1",
        230710048,
        "G",
        "A",
        1.15,
        "Hypertension",
        "cardiovascular",
        0.40,
        "moderate",
    ),
]

# Quick count assertion (fails at import time if DB is under-populated)
assert len(BUILTIN_VARIANT_DB) >= 50, (
    f"Built-in variant DB has only {len(BUILTIN_VARIANT_DB)} entries; expected ≥ 50."
)

# ---------------------------------------------------------------------------
# Internal constants used by the predictor
# ---------------------------------------------------------------------------

_EVIDENCE_WEIGHTS: dict[str, float] = {
    "strong": 1.0,
    "moderate": 0.6,
    "suggestive": 0.3,
}

# Age-of-onset windows (condition → (min_age, max_age)).
_ONSET_RANGES: dict[str, tuple[int, int]] = {
    "Coronary artery disease": (40, 85),
    "Hypercholesterolaemia": (20, 70),
    "Atrial fibrillation": (50, 90),
    "Aortic stenosis": (55, 90),
    "Hypertension": (30, 80),
    "Breast cancer": (30, 80),
    "Ovarian cancer": (40, 75),
    "Colorectal cancer": (40, 80),
    "Li-Fraumeni syndrome cancers": (5, 60),
    "Lynch syndrome cancers": (25, 70),
    "Endometrial cancer": (40, 75),
    "Prostate cancer": (50, 85),
    "Lung cancer": (45, 80),
    "Melanoma": (25, 80),
    "Type 2 diabetes": (30, 75),
    "Alzheimer's disease": (60, 95),
    "Parkinson's disease": (50, 85),
    "Huntington's disease": (30, 60),
    "Rheumatoid arthritis": (25, 70),
    "Type 1 diabetes": (1, 30),
    "Systemic lupus erythematosus": (15, 50),
    "Hyperhomocysteinaemia": (20, 70),
    "Neural tube defects (offspring)": (18, 45),
    "Obesity": (5, 60),
    "Age-related macular degeneration": (50, 90),
    "Osteoporosis": (45, 90),
    "Sickle cell disease": (0, 5),
    "Beta-thalassaemia": (0, 5),
    "Hereditary haemochromatosis": (30, 60),
}

# Preventability / modifiability scores by condition (0–1).
_PREVENTABILITY: dict[str, float] = {
    "Coronary artery disease": 0.70,
    "Hypercholesterolaemia": 0.75,
    "Atrial fibrillation": 0.45,
    "Aortic stenosis": 0.25,
    "Hypertension": 0.70,
    "Breast cancer": 0.40,
    "Ovarian cancer": 0.25,
    "Colorectal cancer": 0.55,
    "Li-Fraumeni syndrome cancers": 0.15,
    "Lynch syndrome cancers": 0.40,
    "Endometrial cancer": 0.35,
    "Prostate cancer": 0.30,
    "Lung cancer": 0.65,
    "Melanoma": 0.60,
    "Type 2 diabetes": 0.75,
    "Alzheimer's disease": 0.30,
    "Parkinson's disease": 0.20,
    "Huntington's disease": 0.05,
    "Rheumatoid arthritis": 0.25,
    "Type 1 diabetes": 0.10,
    "Systemic lupus erythematosus": 0.15,
    "Hyperhomocysteinaemia": 0.60,
    "Neural tube defects (offspring)": 0.70,
    "Obesity": 0.80,
    "Age-related macular degeneration": 0.35,
    "Osteoporosis": 0.55,
    "Sickle cell disease": 0.05,
    "Beta-thalassaemia": 0.05,
    "Hereditary haemochromatosis": 0.50,
}

# Telomere-length risk modifiers: per-kb-shortening odds ratio multiplier
# (relative to age-adjusted expected length).
_TELOMERE_RISK_MODIFIERS: dict[str, float] = {
    "Coronary artery disease": 1.15,
    "Type 2 diabetes": 1.10,
    "Alzheimer's disease": 1.08,
    "Colorectal cancer": 1.12,
    "Breast cancer": 1.09,
    "Lung cancer": 1.14,
    "Osteoporosis": 1.06,
}

# Screening / prevention recommendations keyed by category.
_SCREENING_RECS: dict[str, list[dict[str, str]]] = {
    "cardiovascular": [
        {
            "action": "Regular lipid panel screening",
            "frequency": "Every 1–2 years after age 40",
            "detail": "Monitor LDL-C, HDL-C, triglycerides, and Lp(a).",
        },
        {
            "action": "Blood pressure monitoring",
            "frequency": "At every clinical visit",
            "detail": "Target < 130/80 mmHg for high-risk individuals.",
        },
        {
            "action": "Lifestyle modification",
            "frequency": "Ongoing",
            "detail": "Mediterranean diet, ≥ 150 min/week moderate exercise, smoking cessation.",
        },
    ],
    "cancer": [
        {
            "action": "Genetic counselling referral",
            "frequency": "Once (with updates as guidelines change)",
            "detail": "Discuss BRCA, Lynch, and TP53 implications.",
        },
        {
            "action": "Cancer-specific screening programme",
            "frequency": "Per national guidelines",
            "detail": "Mammography, colonoscopy, PSA, or skin checks as "
            "indicated by variant profile.",
        },
    ],
    "diabetes": [
        {
            "action": "Fasting glucose / HbA1c screening",
            "frequency": "Annually from age 35 or earlier if high-risk",
            "detail": "Pre-diabetes detection allows lifestyle intervention.",
        },
        {
            "action": "Weight management programme",
            "frequency": "Ongoing",
            "detail": "Maintain BMI 18.5–24.9; consider dietitian referral.",
        },
    ],
    "alzheimers": [
        {
            "action": "Cognitive screening",
            "frequency": "Every 1–2 years after age 55",
            "detail": "MoCA or MMSE; discuss emerging amyloid biomarkers.",
        },
        {
            "action": "Modifiable risk reduction",
            "frequency": "Ongoing",
            "detail": "Cardiovascular risk management, social engagement, "
            "physical activity, sleep hygiene.",
        },
    ],
    "autoimmune": [
        {
            "action": "Autoantibody panel",
            "frequency": "If symptomatic",
            "detail": "ANA, RF, anti-CCP, HLA typing as directed by clinician.",
        },
        {
            "action": "Monitor inflammatory markers",
            "frequency": "As directed",
            "detail": "CRP, ESR for early detection of flares.",
        },
    ],
    "metabolic": [
        {
            "action": "Homocysteine level testing",
            "frequency": "Baseline + follow-up if elevated",
            "detail": "B-vitamin supplementation can reduce levels.",
        },
        {
            "action": "Nutritional counselling",
            "frequency": "Ongoing",
            "detail": "Folate-rich diet; prenatal folic acid supplementation for MTHFR carriers.",
        },
    ],
    "eye": [
        {
            "action": "Dilated eye examination",
            "frequency": "Every 1–2 years after age 50",
            "detail": "Early detection of drusen and RPE changes.",
        },
        {
            "action": "AREDS2 supplementation discussion",
            "frequency": "Once (if intermediate AMD detected)",
            "detail": "Lutein + zeaxanthin, zinc, vitamins C/E.",
        },
    ],
    "neurological": [
        {
            "action": "Neurological evaluation",
            "frequency": "If symptomatic or family history positive",
            "detail": "DaTscan, genetic counselling for LRRK2 carriers.",
        },
    ],
    "bone": [
        {
            "action": "DEXA bone density scan",
            "frequency": "Every 2 years after age 50 (women) / 65 (men)",
            "detail": "Assess fracture risk; consider FRAX calculation.",
        },
        {
            "action": "Calcium + Vitamin D optimisation",
            "frequency": "Ongoing",
            "detail": "1 000–1 200 mg Ca/day; 800–1 000 IU Vitamin D/day.",
        },
    ],
    "blood": [
        {
            "action": "Complete blood count + iron studies",
            "frequency": "Annually if carrier status confirmed",
            "detail": "Monitor ferritin and transferrin saturation for HFE; "
            "Hb electrophoresis for HBB.",
        },
        {
            "action": "Genetic counselling for family planning",
            "frequency": "Once",
            "detail": "Carrier screening for partner; discuss reproductive options.",
        },
    ],
}


# ---------------------------------------------------------------------------
# DiseasePredictor — main engine
# ---------------------------------------------------------------------------


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

            ``expected = 11000 − 30 * age``  (bp)

        A per-kb shortening multiplier is applied to conditions listed in
        ``_TELOMERE_RISK_MODIFIERS``.

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
        expected_bp = 11_000.0 - 30.0 * age
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

        # Use mean intensity as a telomere-length proxy (calibrated via
        # an approximate linear model: bp ≈ intensity * 1.5).
        if mean_intensity is not None and mean_intensity > 0:
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
