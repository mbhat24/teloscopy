"""Facial-genomic prediction engine.

Extracts facial features from photographs and maps them to estimated
genomic profiles using published phenotype-genotype correlations.

Scientific basis:
- **Biological age estimation**: Facial texture, wrinkle depth, and skin
  quality correlate with biological age (Gunn et al., 2009; Christensen
  et al., 2009).  Biological age in turn correlates with telomere length
  via the formula: TL ≈ 11.0 − 0.040 × age (Müezzinler et al., 2013;
  Aubert & Lansdorp, 2008).
- **Ancestry estimation**: Facial morphology varies systematically across
  populations (Claes et al., 2014).  Population ancestry correlates with
  allele frequencies for many health-relevant SNPs.
- **Pigmentation genes**: Skin tone, eye colour, and hair colour are
  strongly determined by MC1R, SLC24A5, SLC45A2, HERC2/OCA2, TYR, and
  KITLG variants (Sturm, 2009; Liu et al., 2015).
- **Facial morphology genes**: IRF6, PAX3, DCHS2, RUNX2, and EDAR
  influence specific craniofacial measurements (Adhikari et al., 2016;
  Claes et al., 2018).
- **Oxidative stress markers**: Skin evenness, dark circles, and UV
  damage correlate with systemic oxidative stress and DNA damage
  (Kowalska et al., 2020).

References
----------
.. [1] Christensen K. et al. (2009) BMJ 339:b5262 — Perceived age as
       clinically useful biomarker of ageing.
.. [2] Aubert G. & Lansdorp P. (2008) Physiol Rev 88(2):557-579 —
       Telomeres and aging.
.. [3] Claes P. et al. (2014) PLoS Genetics 10(3):e1004224 — Modeling
       3D facial shape from DNA.
.. [4] Liu F. et al. (2015) Forensic Sci Int Genet 17:76-83 — HIrisPlex-S.
.. [5] Adhikari K. et al. (2016) Nature Comms 7:11616 — Facial feature
       genetics in Latin Americans.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class FacialMeasurements:
    """Extracted facial feature measurements."""

    face_width: float = 0.0
    face_height: float = 0.0
    face_ratio: float = 0.0  # width / height
    skin_brightness: float = 0.0  # 0-255 mean skin luminance
    skin_uniformity: float = 0.0  # lower = more even (std dev of skin patch)
    skin_redness: float = 0.0  # mean red channel - mean of others
    skin_yellowness: float = 0.0  # proxy for melanin / carotenoid
    wrinkle_score: float = 0.0  # edge density in forehead/eye regions (0-1)
    symmetry_score: float = 0.0  # left-right symmetry (0-1, 1=perfect)
    dark_circle_score: float = 0.0  # darkness under eyes (0-1)
    texture_roughness: float = 0.0  # Laplacian variance of skin texture
    uv_damage_score: float = 0.0  # estimated UV damage from pigmentation irregularity


@dataclass
class AncestryEstimate:
    """Estimated ancestral composition from facial features."""

    european: float = 0.0
    east_asian: float = 0.0
    south_asian: float = 0.0
    african: float = 0.0
    middle_eastern: float = 0.0
    latin_american: float = 0.0
    confidence: float = 0.0


@dataclass
class PredictedVariant:
    """A predicted genetic variant with confidence."""

    rsid: str
    gene: str
    predicted_genotype: str
    confidence: float  # 0-1
    basis: str  # explanation of why this was predicted
    risk_allele: str = ""  # e.g. "T", "A", etc.
    ref_allele: str = ""  # reference allele


@dataclass
class ReconstructedSequence:
    """A reconstructed DNA sequence fragment around a predicted SNP."""

    rsid: str
    gene: str
    chromosome: str
    position: int  # GRCh38 coordinate
    ref_allele: str
    predicted_allele_1: str
    predicted_allele_2: str
    flanking_5prime: str  # 25 bp upstream (reference)
    flanking_3prime: str  # 25 bp downstream (reference)
    confidence: float

    @property
    def fasta_record(self) -> str:
        """Render as a FASTA record with predicted alleles inserted."""
        seq = f"{self.flanking_5prime}[{self.predicted_allele_1}/{self.predicted_allele_2}]{self.flanking_3prime}"
        header = (
            f">{self.rsid}|{self.gene}|chr{self.chromosome}:{self.position}"
            f"|predicted={self.predicted_allele_1}/{self.predicted_allele_2}"
            f"|confidence={self.confidence:.2f}"
        )
        return f"{header}\n{seq}"


@dataclass
class ReconstructedDNA:
    """Complete reconstructed partial genome from predicted variants."""

    sequences: list[ReconstructedSequence] = field(default_factory=list)
    total_variants: int = 0
    genome_build: str = "GRCh38/hg38"
    disclaimer: str = (
        "RECONSTRUCTED SEQUENCE — This is a statistical reconstruction based "
        "on facial-genomic predictions, NOT actual DNA sequencing. Predicted "
        "genotypes are derived from population-level allele frequencies and "
        "phenotypic correlations. Do not use for clinical decisions."
    )

    @property
    def fasta(self) -> str:
        """Full FASTA output of all reconstructed sequence fragments."""
        header = (
            f"# Teloscopy Reconstructed Partial Genome\n"
            f"# Build: {self.genome_build}\n"
            f"# Variants: {self.total_variants}\n"
            f"# {self.disclaimer}\n"
        )
        return header + "\n".join(s.fasta_record for s in self.sequences)


@dataclass
class PharmacogenomicPrediction:
    """Predicted drug metabolism and response profile for a single gene."""

    gene: str  # e.g. "CYP2D6"
    rsid: str  # key SNP tested
    predicted_phenotype: str  # e.g. "Poor Metabolizer", "Intermediate", "Normal", "Ultra-rapid"
    confidence: float  # 0-1
    affected_drugs: list[str] = field(default_factory=list)
    clinical_recommendation: str = ""
    basis: str = ""


@dataclass
class FacialHealthScreening:
    """Health indicators derived from facial analysis."""

    estimated_bmi_category: str = "Unknown"  # Underweight/Normal/Overweight/Obese
    bmi_confidence: float = 0.0
    anemia_risk_score: float = 0.0  # 0-1
    cardiovascular_risk_indicators: list[str] = field(default_factory=list)
    thyroid_indicators: list[str] = field(default_factory=list)
    fatigue_stress_score: float = 0.0  # 0-1 (higher = more fatigue/stress visible)
    hydration_score: float = 50.0  # 0-100 (100 = well hydrated)


@dataclass
class DermatologicalAnalysis:
    """Detailed skin analysis from facial imaging."""

    rosacea_risk_score: float = 0.0  # 0-1
    melasma_risk_score: float = 0.0  # 0-1
    photo_aging_gap: int = 0  # years above/below expected skin age for chronological age
    acne_severity_score: float = 0.0  # 0-1
    skin_cancer_risk_factors: list[str] = field(default_factory=list)
    pigmentation_disorder_risk: float = 0.0  # 0-1
    moisture_barrier_score: float = 50.0  # 0-100


@dataclass
class ConditionScreening:
    """Facial-feature-based screening result for a medical condition."""

    condition: str  # e.g. "Acromegaly", "Cushing syndrome"
    risk_score: float = 0.0  # 0-1
    facial_markers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    recommendation: str = ""


@dataclass
class AncestryDerivedPredictions:
    """Ancestry-based genetic predictions beyond direct facial features."""

    predicted_mtdna_haplogroup: str = "Unknown"
    haplogroup_confidence: float = 0.0
    lactose_tolerance_probability: float = 0.5
    alcohol_flush_probability: float = 0.0
    caffeine_sensitivity: str = "Unknown"  # "Fast" / "Slow"
    bitter_taste_sensitivity: str = "Unknown"  # "Taster" / "Non-taster"
    population_specific_risks: list[str] = field(default_factory=list)


@dataclass
class FacialGenomicProfile:
    """Complete facial-genomic prediction result."""

    estimated_biological_age: int
    estimated_telomere_length_kb: float
    telomere_percentile: int  # percentile for chronological age
    measurements: FacialMeasurements
    ancestry: AncestryEstimate
    predicted_variants: list[PredictedVariant] = field(default_factory=list)
    reconstructed_dna: ReconstructedDNA = field(default_factory=ReconstructedDNA)
    skin_health_score: float = 0.0  # 0-100
    oxidative_stress_score: float = 0.0  # 0-1 (higher = more stress)
    predicted_eye_colour: str = "unknown"
    predicted_hair_colour: str = "unknown"
    predicted_skin_type: str = "unknown"  # Fitzpatrick scale
    analysis_warnings: list[str] = field(default_factory=list)
    # --- v2.0 expansion ---
    pharmacogenomic_predictions: list[PharmacogenomicPrediction] = field(default_factory=list)
    health_screening: FacialHealthScreening = field(default_factory=FacialHealthScreening)
    dermatological_analysis: DermatologicalAnalysis = field(default_factory=DermatologicalAnalysis)
    condition_screenings: list[ConditionScreening] = field(default_factory=list)
    ancestry_derived: AncestryDerivedPredictions = field(default_factory=AncestryDerivedPredictions)


# ---------------------------------------------------------------------------
# Population SNP frequency tables (research-derived)
# ---------------------------------------------------------------------------

# Allele frequencies for key health-relevant SNPs by population.
# Sources: 1000 Genomes, gnomAD, published GWAS meta-analyses.
# Format: {rsid: {population: (risk_allele_freq, gene, condition)}}
_POPULATION_SNP_FREQUENCIES: dict[str, dict[str, tuple[float, str, str]]] = {
    "rs429358": {  # APOE e4
        "european": (0.15, "APOE", "Alzheimer's / cardiovascular"),
        "east_asian": (0.09, "APOE", "Alzheimer's / cardiovascular"),
        "african": (0.27, "APOE", "Alzheimer's / cardiovascular"),
        "south_asian": (0.12, "APOE", "Alzheimer's / cardiovascular"),
        "latin_american": (0.11, "APOE", "Alzheimer's / cardiovascular"),
        "middle_eastern": (0.14, "APOE", "Alzheimer's / cardiovascular"),
    },
    "rs7903146": {  # TCF7L2
        "european": (0.30, "TCF7L2", "Type 2 diabetes"),
        "east_asian": (0.03, "TCF7L2", "Type 2 diabetes"),
        "african": (0.28, "TCF7L2", "Type 2 diabetes"),
        "south_asian": (0.32, "TCF7L2", "Type 2 diabetes"),
        "latin_american": (0.24, "TCF7L2", "Type 2 diabetes"),
        "middle_eastern": (0.29, "TCF7L2", "Type 2 diabetes"),
    },
    "rs1801133": {  # MTHFR C677T
        "european": (0.33, "MTHFR", "Folate metabolism"),
        "east_asian": (0.44, "MTHFR", "Folate metabolism"),
        "african": (0.10, "MTHFR", "Folate metabolism"),
        "south_asian": (0.15, "MTHFR", "Folate metabolism"),
        "latin_american": (0.42, "MTHFR", "Folate metabolism"),
        "middle_eastern": (0.28, "MTHFR", "Folate metabolism"),
    },
    "rs4988235": {  # LCT — lactase persistence
        "european": (0.75, "LCT", "Lactose tolerance"),
        "east_asian": (0.01, "LCT", "Lactose tolerance"),
        "african": (0.15, "LCT", "Lactose tolerance"),
        "south_asian": (0.30, "LCT", "Lactose tolerance"),
        "latin_american": (0.50, "LCT", "Lactose tolerance"),
        "middle_eastern": (0.55, "LCT", "Lactose tolerance"),
    },
    "rs1229984": {  # ADH1B — alcohol metabolism
        "european": (0.05, "ADH1B", "Alcohol metabolism"),
        "east_asian": (0.70, "ADH1B", "Alcohol metabolism"),
        "african": (0.03, "ADH1B", "Alcohol metabolism"),
        "south_asian": (0.10, "ADH1B", "Alcohol metabolism"),
        "latin_american": (0.15, "ADH1B", "Alcohol metabolism"),
        "middle_eastern": (0.08, "ADH1B", "Alcohol metabolism"),
    },
    "rs671": {  # ALDH2 — alcohol flush
        "european": (0.00, "ALDH2", "Alcohol flush / acetaldehyde"),
        "east_asian": (0.28, "ALDH2", "Alcohol flush / acetaldehyde"),
        "african": (0.00, "ALDH2", "Alcohol flush / acetaldehyde"),
        "south_asian": (0.02, "ALDH2", "Alcohol flush / acetaldehyde"),
        "latin_american": (0.02, "ALDH2", "Alcohol flush / acetaldehyde"),
        "middle_eastern": (0.00, "ALDH2", "Alcohol flush / acetaldehyde"),
    },
    "rs762551": {  # CYP1A2 — caffeine
        "european": (0.33, "CYP1A2", "Slow caffeine metabolism"),
        "east_asian": (0.40, "CYP1A2", "Slow caffeine metabolism"),
        "african": (0.24, "CYP1A2", "Slow caffeine metabolism"),
        "south_asian": (0.30, "CYP1A2", "Slow caffeine metabolism"),
        "latin_american": (0.36, "CYP1A2", "Slow caffeine metabolism"),
        "middle_eastern": (0.30, "CYP1A2", "Slow caffeine metabolism"),
    },
    "rs1800562": {  # HFE C282Y — haemochromatosis
        "european": (0.06, "HFE", "Iron overload risk"),
        "east_asian": (0.00, "HFE", "Iron overload risk"),
        "african": (0.00, "HFE", "Iron overload risk"),
        "south_asian": (0.01, "HFE", "Iron overload risk"),
        "latin_american": (0.02, "HFE", "Iron overload risk"),
        "middle_eastern": (0.02, "HFE", "Iron overload risk"),
    },
    "rs12913832": {  # HERC2/OCA2 — eye colour
        "european": (0.72, "HERC2", "Blue/green eye colour"),
        "east_asian": (0.01, "HERC2", "Blue/green eye colour"),
        "african": (0.01, "HERC2", "Blue/green eye colour"),
        "south_asian": (0.05, "HERC2", "Blue/green eye colour"),
        "latin_american": (0.25, "HERC2", "Blue/green eye colour"),
        "middle_eastern": (0.20, "HERC2", "Blue/green eye colour"),
    },
    "rs1426654": {  # SLC24A5 — skin pigmentation
        "european": (0.98, "SLC24A5", "Light skin pigmentation"),
        "east_asian": (0.02, "SLC24A5", "Light skin pigmentation"),
        "african": (0.01, "SLC24A5", "Light skin pigmentation"),
        "south_asian": (0.50, "SLC24A5", "Light skin pigmentation"),
        "latin_american": (0.55, "SLC24A5", "Light skin pigmentation"),
        "middle_eastern": (0.80, "SLC24A5", "Light skin pigmentation"),
    },
    "rs16891982": {  # SLC45A2 — pigmentation
        "european": (0.87, "SLC45A2", "Light pigmentation"),
        "east_asian": (0.02, "SLC45A2", "Light pigmentation"),
        "african": (0.01, "SLC45A2", "Light pigmentation"),
        "south_asian": (0.20, "SLC45A2", "Light pigmentation"),
        "latin_american": (0.45, "SLC45A2", "Light pigmentation"),
        "middle_eastern": (0.60, "SLC45A2", "Light pigmentation"),
    },
    "rs1805007": {  # MC1R — red hair / fair skin
        "european": (0.10, "MC1R", "Red hair / fair skin / freckling"),
        "east_asian": (0.00, "MC1R", "Red hair / fair skin / freckling"),
        "african": (0.00, "MC1R", "Red hair / fair skin / freckling"),
        "south_asian": (0.01, "MC1R", "Red hair / fair skin / freckling"),
        "latin_american": (0.03, "MC1R", "Red hair / fair skin / freckling"),
        "middle_eastern": (0.03, "MC1R", "Red hair / fair skin / freckling"),
    },
    "rs2476601": {  # PTPN22 — autoimmune
        "european": (0.10, "PTPN22", "Autoimmune disease susceptibility"),
        "east_asian": (0.00, "PTPN22", "Autoimmune disease susceptibility"),
        "african": (0.01, "PTPN22", "Autoimmune disease susceptibility"),
        "south_asian": (0.04, "PTPN22", "Autoimmune disease susceptibility"),
        "latin_american": (0.04, "PTPN22", "Autoimmune disease susceptibility"),
        "middle_eastern": (0.06, "PTPN22", "Autoimmune disease susceptibility"),
    },
    "rs4880": {  # SOD2 — oxidative stress
        "european": (0.47, "SOD2", "Reduced antioxidant defence"),
        "east_asian": (0.14, "SOD2", "Reduced antioxidant defence"),
        "african": (0.37, "SOD2", "Reduced antioxidant defence"),
        "south_asian": (0.40, "SOD2", "Reduced antioxidant defence"),
        "latin_american": (0.40, "SOD2", "Reduced antioxidant defence"),
        "middle_eastern": (0.42, "SOD2", "Reduced antioxidant defence"),
    },
    "rs9939609": {  # FTO — obesity
        "european": (0.42, "FTO", "Obesity / appetite regulation"),
        "east_asian": (0.14, "FTO", "Obesity / appetite regulation"),
        "african": (0.45, "FTO", "Obesity / appetite regulation"),
        "south_asian": (0.32, "FTO", "Obesity / appetite regulation"),
        "latin_american": (0.34, "FTO", "Obesity / appetite regulation"),
        "middle_eastern": (0.38, "FTO", "Obesity / appetite regulation"),
    },
    "rs10811661": {  # CDKN2A/B — diabetes
        "european": (0.83, "CDKN2A/B", "Type 2 diabetes"),
        "east_asian": (0.60, "CDKN2A/B", "Type 2 diabetes"),
        "african": (0.90, "CDKN2A/B", "Type 2 diabetes"),
        "south_asian": (0.85, "CDKN2A/B", "Type 2 diabetes"),
        "latin_american": (0.78, "CDKN2A/B", "Type 2 diabetes"),
        "middle_eastern": (0.82, "CDKN2A/B", "Type 2 diabetes"),
    },
    "rs1333049": {  # 9p21 — cardiovascular
        "european": (0.47, "CDKN2B-AS1", "Coronary artery disease"),
        "east_asian": (0.52, "CDKN2B-AS1", "Coronary artery disease"),
        "african": (0.26, "CDKN2B-AS1", "Coronary artery disease"),
        "south_asian": (0.55, "CDKN2B-AS1", "Coronary artery disease"),
        "latin_american": (0.40, "CDKN2B-AS1", "Coronary artery disease"),
        "middle_eastern": (0.50, "CDKN2B-AS1", "Coronary artery disease"),
    },
    "rs4646994": {  # ACE I/D — hypertension
        "european": (0.45, "ACE", "Salt-sensitive hypertension"),
        "east_asian": (0.38, "ACE", "Salt-sensitive hypertension"),
        "african": (0.60, "ACE", "Salt-sensitive hypertension"),
        "south_asian": (0.52, "ACE", "Salt-sensitive hypertension"),
        "latin_american": (0.48, "ACE", "Salt-sensitive hypertension"),
        "middle_eastern": (0.48, "ACE", "Salt-sensitive hypertension"),
    },
    "rs174546": {  # FADS1 — omega-3 metabolism
        "european": (0.35, "FADS1", "Reduced omega-3 conversion"),
        "east_asian": (0.55, "FADS1", "Reduced omega-3 conversion"),
        "african": (0.90, "FADS1", "Reduced omega-3 conversion"),
        "south_asian": (0.45, "FADS1", "Reduced omega-3 conversion"),
        "latin_american": (0.50, "FADS1", "Reduced omega-3 conversion"),
        "middle_eastern": (0.40, "FADS1", "Reduced omega-3 conversion"),
    },
    # ---- v2.0 expansion: HIrisPlex-S pigmentation SNPs ----
    "rs1129038": {  # HERC2 — blue vs brown eye colour (LD with rs12913832)
        "european": (0.25, "HERC2", "Blue eye colour"),
        "east_asian": (0.01, "HERC2", "Blue eye colour"),
        "african": (0.01, "HERC2", "Blue eye colour"),
        "south_asian": (0.04, "HERC2", "Blue eye colour"),
        "latin_american": (0.15, "HERC2", "Blue eye colour"),
        "middle_eastern": (0.12, "HERC2", "Blue eye colour"),
    },
    "rs12896399": {  # SLC24A4 — eye/hair colour modifier
        "european": (0.45, "SLC24A4", "Eye/hair colour"),
        "east_asian": (0.05, "SLC24A4", "Eye/hair colour"),
        "african": (0.10, "SLC24A4", "Eye/hair colour"),
        "south_asian": (0.20, "SLC24A4", "Eye/hair colour"),
        "latin_american": (0.30, "SLC24A4", "Eye/hair colour"),
        "middle_eastern": (0.25, "SLC24A4", "Eye/hair colour"),
    },
    "rs1393350": {  # TYR — eye colour (green/hazel)
        "european": (0.22, "TYR", "Green/hazel eye colour"),
        "east_asian": (0.02, "TYR", "Green/hazel eye colour"),
        "african": (0.01, "TYR", "Green/hazel eye colour"),
        "south_asian": (0.08, "TYR", "Green/hazel eye colour"),
        "latin_american": (0.12, "TYR", "Green/hazel eye colour"),
        "middle_eastern": (0.10, "TYR", "Green/hazel eye colour"),
    },
    "rs1800407": {  # OCA2 — eye colour (R419Q)
        "european": (0.06, "OCA2", "Eye colour modifier"),
        "east_asian": (0.00, "OCA2", "Eye colour modifier"),
        "african": (0.00, "OCA2", "Eye colour modifier"),
        "south_asian": (0.02, "OCA2", "Eye colour modifier"),
        "latin_american": (0.03, "OCA2", "Eye colour modifier"),
        "middle_eastern": (0.03, "OCA2", "Eye colour modifier"),
    },
    "rs2402130": {  # SLC24A4 — eye and hair colour
        "european": (0.48, "SLC24A4", "Eye and hair colour"),
        "east_asian": (0.06, "SLC24A4", "Eye and hair colour"),
        "african": (0.12, "SLC24A4", "Eye and hair colour"),
        "south_asian": (0.22, "SLC24A4", "Eye and hair colour"),
        "latin_american": (0.32, "SLC24A4", "Eye and hair colour"),
        "middle_eastern": (0.28, "SLC24A4", "Eye and hair colour"),
    },
    # ---- v2.0: Additional MC1R / hair colour SNPs ----
    "rs1805008": {  # MC1R R160W — red hair (strong effect)
        "european": (0.09, "MC1R", "Red hair / fair skin (R160W)"),
        "east_asian": (0.00, "MC1R", "Red hair / fair skin (R160W)"),
        "african": (0.00, "MC1R", "Red hair / fair skin (R160W)"),
        "south_asian": (0.01, "MC1R", "Red hair / fair skin (R160W)"),
        "latin_american": (0.03, "MC1R", "Red hair / fair skin (R160W)"),
        "middle_eastern": (0.02, "MC1R", "Red hair / fair skin (R160W)"),
    },
    "rs1805009": {  # MC1R D294H — red hair (very strong penetrance)
        "european": (0.02, "MC1R", "Red hair (D294H)"),
        "east_asian": (0.00, "MC1R", "Red hair (D294H)"),
        "african": (0.00, "MC1R", "Red hair (D294H)"),
        "south_asian": (0.00, "MC1R", "Red hair (D294H)"),
        "latin_american": (0.01, "MC1R", "Red hair (D294H)"),
        "middle_eastern": (0.01, "MC1R", "Red hair (D294H)"),
    },
    "rs2228479": {  # MC1R V92M — hair/eye colour; perceived age
        "european": (0.10, "MC1R", "Hair/eye colour modifier (V92M)"),
        "east_asian": (0.25, "MC1R", "Hair/eye colour modifier (V92M)"),
        "african": (0.01, "MC1R", "Hair/eye colour modifier (V92M)"),
        "south_asian": (0.08, "MC1R", "Hair/eye colour modifier (V92M)"),
        "latin_american": (0.12, "MC1R", "Hair/eye colour modifier (V92M)"),
        "middle_eastern": (0.06, "MC1R", "Hair/eye colour modifier (V92M)"),
    },
    "rs885479": {  # MC1R R163Q — hair colour modifier
        "european": (0.06, "MC1R", "Hair colour modifier (R163Q)"),
        "east_asian": (0.70, "MC1R", "Hair colour modifier (R163Q)"),
        "african": (0.01, "MC1R", "Hair colour modifier (R163Q)"),
        "south_asian": (0.15, "MC1R", "Hair colour modifier (R163Q)"),
        "latin_american": (0.20, "MC1R", "Hair colour modifier (R163Q)"),
        "middle_eastern": (0.05, "MC1R", "Hair colour modifier (R163Q)"),
    },
    "rs35264875": {  # TPCN2 — blond vs brown hair (M484L)
        "european": (0.10, "TPCN2", "Blond hair colour"),
        "east_asian": (0.01, "TPCN2", "Blond hair colour"),
        "african": (0.01, "TPCN2", "Blond hair colour"),
        "south_asian": (0.03, "TPCN2", "Blond hair colour"),
        "latin_american": (0.06, "TPCN2", "Blond hair colour"),
        "middle_eastern": (0.04, "TPCN2", "Blond hair colour"),
    },
    # ---- v2.0: Skin pigmentation / freckling ----
    "rs1800414": {  # OCA2 H615R — skin lightening in East Asians
        "european": (0.01, "OCA2", "East Asian skin lightening"),
        "east_asian": (0.55, "OCA2", "East Asian skin lightening"),
        "african": (0.00, "OCA2", "East Asian skin lightening"),
        "south_asian": (0.05, "OCA2", "East Asian skin lightening"),
        "latin_american": (0.08, "OCA2", "East Asian skin lightening"),
        "middle_eastern": (0.01, "OCA2", "East Asian skin lightening"),
    },
    "rs10756819": {  # BNC2 — skin pigmentation, freckling
        "european": (0.68, "BNC2", "Skin pigmentation / freckling"),
        "east_asian": (0.25, "BNC2", "Skin pigmentation / freckling"),
        "african": (0.80, "BNC2", "Skin pigmentation / freckling"),
        "south_asian": (0.50, "BNC2", "Skin pigmentation / freckling"),
        "latin_american": (0.55, "BNC2", "Skin pigmentation / freckling"),
        "middle_eastern": (0.45, "BNC2", "Skin pigmentation / freckling"),
    },
    "rs6059655": {  # ASIP upstream — skin pigmentation
        "european": (0.10, "ASIP", "Skin pigmentation"),
        "east_asian": (0.02, "ASIP", "Skin pigmentation"),
        "african": (0.05, "ASIP", "Skin pigmentation"),
        "south_asian": (0.06, "ASIP", "Skin pigmentation"),
        "latin_american": (0.08, "ASIP", "Skin pigmentation"),
        "middle_eastern": (0.07, "ASIP", "Skin pigmentation"),
    },
    "rs1015362": {  # ASIP — skin/hair pigmentation, tanning
        "european": (0.17, "ASIP", "Skin/hair pigmentation, tanning"),
        "east_asian": (0.05, "ASIP", "Skin/hair pigmentation, tanning"),
        "african": (0.08, "ASIP", "Skin/hair pigmentation, tanning"),
        "south_asian": (0.10, "ASIP", "Skin/hair pigmentation, tanning"),
        "latin_american": (0.12, "ASIP", "Skin/hair pigmentation, tanning"),
        "middle_eastern": (0.10, "ASIP", "Skin/hair pigmentation, tanning"),
    },
    # ---- v2.0: Facial morphology SNPs ----
    "rs7559271": {  # PAX3 — nasion position / nose bridge
        "european": (0.48, "PAX3", "Nose bridge width / nasion position"),
        "east_asian": (0.35, "PAX3", "Nose bridge width / nasion position"),
        "african": (0.55, "PAX3", "Nose bridge width / nasion position"),
        "south_asian": (0.42, "PAX3", "Nose bridge width / nasion position"),
        "latin_american": (0.50, "PAX3", "Nose bridge width / nasion position"),
        "middle_eastern": (0.45, "PAX3", "Nose bridge width / nasion position"),
    },
    "rs927833": {  # DCHS2 — nose shape / protrusion
        "european": (0.38, "DCHS2", "Nose protrusion / wing breadth"),
        "east_asian": (0.22, "DCHS2", "Nose protrusion / wing breadth"),
        "african": (0.50, "DCHS2", "Nose protrusion / wing breadth"),
        "south_asian": (0.40, "DCHS2", "Nose protrusion / wing breadth"),
        "latin_american": (0.45, "DCHS2", "Nose protrusion / wing breadth"),
        "middle_eastern": (0.35, "DCHS2", "Nose protrusion / wing breadth"),
    },
    "rs2045323": {  # RUNX2 — nose bridge width
        "european": (0.41, "RUNX2", "Nose bridge width"),
        "east_asian": (0.30, "RUNX2", "Nose bridge width"),
        "african": (0.60, "RUNX2", "Nose bridge width"),
        "south_asian": (0.45, "RUNX2", "Nose bridge width"),
        "latin_american": (0.48, "RUNX2", "Nose bridge width"),
        "middle_eastern": (0.42, "RUNX2", "Nose bridge width"),
    },
    "rs4648379": {  # SUPT3H — nose tip shape
        "european": (0.32, "SUPT3H", "Nose tip shape / columella inclination"),
        "east_asian": (0.20, "SUPT3H", "Nose tip shape / columella inclination"),
        "african": (0.45, "SUPT3H", "Nose tip shape / columella inclination"),
        "south_asian": (0.35, "SUPT3H", "Nose tip shape / columella inclination"),
        "latin_american": (0.38, "SUPT3H", "Nose tip shape / columella inclination"),
        "middle_eastern": (0.30, "SUPT3H", "Nose tip shape / columella inclination"),
    },
    # ---- v2.0: Chin, jaw, face shape ----
    "rs3827760": {  # EDAR 370A — hair thickness, chin, ear morphology
        "european": (0.01, "EDAR", "Hair thickness / chin / ear morphology"),
        "east_asian": (0.90, "EDAR", "Hair thickness / chin / ear morphology"),
        "african": (0.00, "EDAR", "Hair thickness / chin / ear morphology"),
        "south_asian": (0.05, "EDAR", "Hair thickness / chin / ear morphology"),
        "latin_american": (0.40, "EDAR", "Hair thickness / chin / ear morphology"),
        "middle_eastern": (0.01, "EDAR", "Hair thickness / chin / ear morphology"),
    },
    "rs642961": {  # IRF6 — lip shape / thickness
        "european": (0.19, "IRF6", "Lip shape / thickness"),
        "east_asian": (0.12, "IRF6", "Lip shape / thickness"),
        "african": (0.15, "IRF6", "Lip shape / thickness"),
        "south_asian": (0.16, "IRF6", "Lip shape / thickness"),
        "latin_american": (0.18, "IRF6", "Lip shape / thickness"),
        "middle_eastern": (0.17, "IRF6", "Lip shape / thickness"),
    },
    "rs11684042": {  # TFAP2B region — chin cleft
        "european": (0.43, "TFAP2B", "Chin cleft / dimple"),
        "east_asian": (0.30, "TFAP2B", "Chin cleft / dimple"),
        "african": (0.35, "TFAP2B", "Chin cleft / dimple"),
        "south_asian": (0.38, "TFAP2B", "Chin cleft / dimple"),
        "latin_american": (0.40, "TFAP2B", "Chin cleft / dimple"),
        "middle_eastern": (0.37, "TFAP2B", "Chin cleft / dimple"),
    },
    "rs6740960": {  # TMEM163 — facial width (bizygomatic breadth)
        "european": (0.25, "TMEM163", "Facial width"),
        "east_asian": (0.40, "TMEM163", "Facial width"),
        "african": (0.30, "TMEM163", "Facial width"),
        "south_asian": (0.28, "TMEM163", "Facial width"),
        "latin_american": (0.32, "TMEM163", "Facial width"),
        "middle_eastern": (0.27, "TMEM163", "Facial width"),
    },
    # ---- v2.0: Hair traits (texture, baldness, graying) ----
    "rs11803731": {  # TCHH — hair curliness in Europeans (L790M)
        "european": (0.40, "TCHH", "Hair curliness (L790M)"),
        "east_asian": (0.01, "TCHH", "Hair curliness (L790M)"),
        "african": (0.05, "TCHH", "Hair curliness (L790M)"),
        "south_asian": (0.10, "TCHH", "Hair curliness (L790M)"),
        "latin_american": (0.20, "TCHH", "Hair curliness (L790M)"),
        "middle_eastern": (0.15, "TCHH", "Hair curliness (L790M)"),
    },
    "rs2180439": {  # AR/EDA2R — male pattern baldness
        "european": (0.56, "AR/EDA2R", "Male pattern baldness"),
        "east_asian": (0.45, "AR/EDA2R", "Male pattern baldness"),
        "african": (0.40, "AR/EDA2R", "Male pattern baldness"),
        "south_asian": (0.50, "AR/EDA2R", "Male pattern baldness"),
        "latin_american": (0.48, "AR/EDA2R", "Male pattern baldness"),
        "middle_eastern": (0.52, "AR/EDA2R", "Male pattern baldness"),
    },
    "rs929626": {  # EBF1 — male pattern baldness (vertex)
        "european": (0.43, "EBF1", "Male pattern baldness (vertex)"),
        "east_asian": (0.35, "EBF1", "Male pattern baldness (vertex)"),
        "african": (0.50, "EBF1", "Male pattern baldness (vertex)"),
        "south_asian": (0.40, "EBF1", "Male pattern baldness (vertex)"),
        "latin_american": (0.42, "EBF1", "Male pattern baldness (vertex)"),
        "middle_eastern": (0.44, "EBF1", "Male pattern baldness (vertex)"),
    },
    # ---- v2.0: Aging / perceived age ----
    "rs258322": {  # STXBP5L — perceived facial age
        "european": (0.28, "STXBP5L", "Perceived facial age"),
        "east_asian": (0.20, "STXBP5L", "Perceived facial age"),
        "african": (0.35, "STXBP5L", "Perceived facial age"),
        "south_asian": (0.25, "STXBP5L", "Perceived facial age"),
        "latin_american": (0.30, "STXBP5L", "Perceived facial age"),
        "middle_eastern": (0.26, "STXBP5L", "Perceived facial age"),
    },
    "rs2228145": {  # IL6R — inflammation / skin aging (Asp358Ala)
        "european": (0.39, "IL6R", "Inflammation / skin aging"),
        "east_asian": (0.25, "IL6R", "Inflammation / skin aging"),
        "african": (0.05, "IL6R", "Inflammation / skin aging"),
        "south_asian": (0.30, "IL6R", "Inflammation / skin aging"),
        "latin_american": (0.28, "IL6R", "Inflammation / skin aging"),
        "middle_eastern": (0.35, "IL6R", "Inflammation / skin aging"),
    },
    "rs1801260": {  # CLOCK — circadian rhythm / photoaging
        "european": (0.22, "CLOCK", "Circadian rhythm / photoaging"),
        "east_asian": (0.15, "CLOCK", "Circadian rhythm / photoaging"),
        "african": (0.18, "CLOCK", "Circadian rhythm / photoaging"),
        "south_asian": (0.20, "CLOCK", "Circadian rhythm / photoaging"),
        "latin_american": (0.19, "CLOCK", "Circadian rhythm / photoaging"),
        "middle_eastern": (0.21, "CLOCK", "Circadian rhythm / photoaging"),
    },
    # ---- v2.0: Pharmacogenomic SNPs ----
    "rs4244285": {  # CYP2C19*2 — clopidogrel, PPIs, antidepressants
        "european": (0.15, "CYP2C19", "Drug metabolism (clopidogrel, PPIs)"),
        "east_asian": (0.30, "CYP2C19", "Drug metabolism (clopidogrel, PPIs)"),
        "african": (0.15, "CYP2C19", "Drug metabolism (clopidogrel, PPIs)"),
        "south_asian": (0.35, "CYP2C19", "Drug metabolism (clopidogrel, PPIs)"),
        "latin_american": (0.12, "CYP2C19", "Drug metabolism (clopidogrel, PPIs)"),
        "middle_eastern": (0.12, "CYP2C19", "Drug metabolism (clopidogrel, PPIs)"),
    },
    "rs12248560": {  # CYP2C19*17 — ultra-rapid metabolizer
        "european": (0.21, "CYP2C19", "Ultra-rapid drug metabolism"),
        "east_asian": (0.04, "CYP2C19", "Ultra-rapid drug metabolism"),
        "african": (0.18, "CYP2C19", "Ultra-rapid drug metabolism"),
        "south_asian": (0.15, "CYP2C19", "Ultra-rapid drug metabolism"),
        "latin_american": (0.16, "CYP2C19", "Ultra-rapid drug metabolism"),
        "middle_eastern": (0.20, "CYP2C19", "Ultra-rapid drug metabolism"),
    },
    "rs1065852": {  # CYP2D6*4 — codeine, tamoxifen, SSRIs
        "european": (0.20, "CYP2D6", "Drug metabolism (codeine, tamoxifen)"),
        "east_asian": (0.01, "CYP2D6", "Drug metabolism (codeine, tamoxifen)"),
        "african": (0.02, "CYP2D6", "Drug metabolism (codeine, tamoxifen)"),
        "south_asian": (0.08, "CYP2D6", "Drug metabolism (codeine, tamoxifen)"),
        "latin_american": (0.10, "CYP2D6", "Drug metabolism (codeine, tamoxifen)"),
        "middle_eastern": (0.10, "CYP2D6", "Drug metabolism (codeine, tamoxifen)"),
    },
    "rs776746": {  # CYP3A5*3 — tacrolimus, immunosuppressants
        "european": (0.94, "CYP3A5", "Drug metabolism (tacrolimus)"),
        "east_asian": (0.75, "CYP3A5", "Drug metabolism (tacrolimus)"),
        "african": (0.30, "CYP3A5", "Drug metabolism (tacrolimus)"),
        "south_asian": (0.65, "CYP3A5", "Drug metabolism (tacrolimus)"),
        "latin_american": (0.75, "CYP3A5", "Drug metabolism (tacrolimus)"),
        "middle_eastern": (0.85, "CYP3A5", "Drug metabolism (tacrolimus)"),
    },
    "rs9923231": {  # VKORC1 — warfarin dose sensitivity
        "european": (0.40, "VKORC1", "Warfarin dose sensitivity"),
        "east_asian": (0.92, "VKORC1", "Warfarin dose sensitivity"),
        "african": (0.10, "VKORC1", "Warfarin dose sensitivity"),
        "south_asian": (0.35, "VKORC1", "Warfarin dose sensitivity"),
        "latin_american": (0.45, "VKORC1", "Warfarin dose sensitivity"),
        "middle_eastern": (0.42, "VKORC1", "Warfarin dose sensitivity"),
    },
    "rs4149056": {  # SLCO1B1 — statin myopathy risk
        "european": (0.15, "SLCO1B1", "Statin myopathy risk"),
        "east_asian": (0.12, "SLCO1B1", "Statin myopathy risk"),
        "african": (0.02, "SLCO1B1", "Statin myopathy risk"),
        "south_asian": (0.05, "SLCO1B1", "Statin myopathy risk"),
        "latin_american": (0.08, "SLCO1B1", "Statin myopathy risk"),
        "middle_eastern": (0.10, "SLCO1B1", "Statin myopathy risk"),
    },
    "rs1045642": {  # ABCB1 (MDR1) — drug efflux / bioavailability
        "european": (0.52, "ABCB1", "Drug efflux / bioavailability"),
        "east_asian": (0.40, "ABCB1", "Drug efflux / bioavailability"),
        "african": (0.10, "ABCB1", "Drug efflux / bioavailability"),
        "south_asian": (0.45, "ABCB1", "Drug efflux / bioavailability"),
        "latin_american": (0.42, "ABCB1", "Drug efflux / bioavailability"),
        "middle_eastern": (0.50, "ABCB1", "Drug efflux / bioavailability"),
    },
    "rs3745274": {  # CYP2B6 — efavirenz, methadone, cyclophosphamide
        "european": (0.28, "CYP2B6", "Drug metabolism (efavirenz, methadone)"),
        "east_asian": (0.18, "CYP2B6", "Drug metabolism (efavirenz, methadone)"),
        "african": (0.35, "CYP2B6", "Drug metabolism (efavirenz, methadone)"),
        "south_asian": (0.25, "CYP2B6", "Drug metabolism (efavirenz, methadone)"),
        "latin_american": (0.30, "CYP2B6", "Drug metabolism (efavirenz, methadone)"),
        "middle_eastern": (0.25, "CYP2B6", "Drug metabolism (efavirenz, methadone)"),
    },
    # ---- v2.0: Telomere / longevity SNPs ----
    "rs2736100": {  # TERT — telomerase reverse transcriptase
        "european": (0.49, "TERT", "Telomere length / longevity"),
        "east_asian": (0.60, "TERT", "Telomere length / longevity"),
        "african": (0.30, "TERT", "Telomere length / longevity"),
        "south_asian": (0.45, "TERT", "Telomere length / longevity"),
        "latin_american": (0.50, "TERT", "Telomere length / longevity"),
        "middle_eastern": (0.47, "TERT", "Telomere length / longevity"),
    },
    "rs10936599": {  # TERC — telomerase RNA component
        "european": (0.25, "TERC", "Telomere length"),
        "east_asian": (0.10, "TERC", "Telomere length"),
        "african": (0.30, "TERC", "Telomere length"),
        "south_asian": (0.20, "TERC", "Telomere length"),
        "latin_american": (0.22, "TERC", "Telomere length"),
        "middle_eastern": (0.23, "TERC", "Telomere length"),
    },
    # ---- v2.0: Additional health-relevant SNPs ----
    "rs1801282": {  # PPARG Pro12Ala — insulin sensitivity
        "european": (0.12, "PPARG", "Insulin sensitivity"),
        "east_asian": (0.04, "PPARG", "Insulin sensitivity"),
        "african": (0.01, "PPARG", "Insulin sensitivity"),
        "south_asian": (0.08, "PPARG", "Insulin sensitivity"),
        "latin_american": (0.10, "PPARG", "Insulin sensitivity"),
        "middle_eastern": (0.09, "PPARG", "Insulin sensitivity"),
    },
    "rs1800795": {  # IL6 — inflammation marker
        "european": (0.42, "IL6", "Systemic inflammation"),
        "east_asian": (0.01, "IL6", "Systemic inflammation"),
        "african": (0.05, "IL6", "Systemic inflammation"),
        "south_asian": (0.20, "IL6", "Systemic inflammation"),
        "latin_american": (0.25, "IL6", "Systemic inflammation"),
        "middle_eastern": (0.30, "IL6", "Systemic inflammation"),
    },
}

# Risk allele and reference allele for each SNP.
# Sources: ClinVar, dbSNP, published GWAS.
_SNP_ALLELES: dict[str, tuple[str, str]] = {
    # rsid: (risk_allele, ref_allele)
    "rs429358": ("C", "T"),      # APOE e4
    "rs7903146": ("T", "C"),     # TCF7L2
    "rs1801133": ("T", "C"),     # MTHFR C677T
    "rs4988235": ("T", "C"),     # LCT
    "rs1229984": ("A", "G"),     # ADH1B
    "rs671": ("A", "G"),         # ALDH2
    "rs762551": ("C", "A"),      # CYP1A2
    "rs1800562": ("A", "G"),     # HFE C282Y
    "rs12913832": ("G", "A"),    # HERC2 (G = blue eye)
    "rs1426654": ("A", "G"),     # SLC24A5
    "rs16891982": ("G", "C"),    # SLC45A2
    "rs1805007": ("T", "C"),     # MC1R
    "rs2476601": ("T", "C"),     # PTPN22
    "rs4880": ("T", "C"),        # SOD2
    "rs9939609": ("A", "T"),     # FTO
    "rs10811661": ("T", "C"),    # CDKN2A/B
    "rs1333049": ("C", "G"),     # CDKN2B-AS1
    "rs4646994": ("D", "I"),     # ACE I/D
    "rs174546": ("T", "C"),      # FADS1
    # ---- v2.0 expansion ----
    "rs1129038": ("A", "G"),     # HERC2 (A = blue eye)
    "rs12896399": ("T", "G"),    # SLC24A4 (T = lighter)
    "rs1393350": ("A", "G"),     # TYR (A = lighter)
    "rs1800407": ("T", "C"),     # OCA2 R419Q
    "rs2402130": ("A", "G"),     # SLC24A4
    "rs1805008": ("T", "C"),     # MC1R R160W
    "rs1805009": ("C", "G"),     # MC1R D294H
    "rs2228479": ("A", "G"),     # MC1R V92M
    "rs885479": ("A", "G"),      # MC1R R163Q
    "rs35264875": ("T", "C"),    # TPCN2
    "rs1800414": ("A", "G"),     # OCA2 H615R
    "rs10756819": ("G", "A"),    # BNC2
    "rs6059655": ("A", "G"),     # ASIP
    "rs1015362": ("G", "A"),     # ASIP
    "rs7559271": ("T", "C"),     # PAX3
    "rs927833": ("C", "T"),      # DCHS2
    "rs2045323": ("G", "A"),     # RUNX2
    "rs4648379": ("A", "G"),     # SUPT3H
    "rs3827760": ("A", "G"),     # EDAR 370A
    "rs642961": ("A", "G"),      # IRF6
    "rs11684042": ("T", "C"),    # TFAP2B
    "rs6740960": ("G", "A"),     # TMEM163
    "rs11803731": ("A", "T"),    # TCHH L790M
    "rs2180439": ("C", "T"),     # AR/EDA2R
    "rs929626": ("A", "G"),      # EBF1
    "rs258322": ("A", "G"),      # STXBP5L
    "rs2228145": ("C", "A"),     # IL6R Asp358Ala
    "rs1801260": ("G", "A"),     # CLOCK
    "rs4244285": ("A", "G"),     # CYP2C19*2
    "rs12248560": ("T", "C"),    # CYP2C19*17
    "rs1065852": ("A", "G"),     # CYP2D6*4
    "rs776746": ("G", "A"),      # CYP3A5*3
    "rs9923231": ("A", "G"),     # VKORC1
    "rs4149056": ("C", "T"),     # SLCO1B1
    "rs1045642": ("T", "C"),     # ABCB1
    "rs3745274": ("T", "G"),     # CYP2B6
    "rs2736100": ("C", "A"),     # TERT
    "rs10936599": ("C", "T"),    # TERC
    "rs1801282": ("G", "C"),     # PPARG
    "rs1800795": ("C", "G"),     # IL6
}

# Genomic coordinates (GRCh38) and 25 bp flanking reference sequences
# for each predicted SNP.  Flanking sequences sourced from NCBI dbSNP
# and the GRCh38 reference assembly (Ensembl release 110).
# Format: rsid → (chromosome, position, 5' flank 25 bp, 3' flank 25 bp)
_SNP_GENOMIC_CONTEXT: dict[str, tuple[str, int, str, str]] = {
    # APOE e4 — chr19:44908684
    "rs429358": (
        "19", 44908684,
        "GCGGACATGGAGGACGTGTGCGGCC",
        "GCCTGCGCAAGCTGCGTAAGCGGCT",
    ),
    # TCF7L2 — chr10:112998590
    "rs7903146": (
        "10", 112998590,
        "AAGATAATTTAATTGCCGTATGAGG",
        "CATACACATACATACACCTATATAAA",
    ),
    # MTHFR C677T — chr1:11796321
    "rs1801133": (
        "1", 11796321,
        "GGAAGAATGTGTCAGCCTCAAAGAA",
        "AAGATCCCGGGGACGATGGGCTCAC",
    ),
    # LCT lactase persistence — chr2:135851076
    "rs4988235": (
        "2", 135851076,
        "CCAATCCTCGGCTAATACTCCAGCC",
        "TGCTATGGGTACTTAGAGTCATTCC",
    ),
    # ADH1B — chr4:99318162
    "rs1229984": (
        "4", 99318162,
        "TACACTCACAGCAATCCTGAATTCT",
        "CCGAAATGCAAGGAACATAATTAAC",
    ),
    # ALDH2 — chr12:111803962
    "rs671": (
        "12", 111803962,
        "GCATGACTGAAGGAGTACAAGCTGC",
        "AGCAGAGATCAACAAGATTTTTGGC",
    ),
    # CYP1A2 — chr15:74749576
    "rs762551": (
        "15", 74749576,
        "GGCAGAAATGCAGGTGTAGGATGCG",
        "TTATCATCTGCAGCAGCTCATCATG",
    ),
    # HFE C282Y — chr6:26091179
    "rs1800562": (
        "6", 26091179,
        "GGTTCTATGATCATGAGAGTCGCCG",
        "TGTCGATCATGGAGCAGTTGAGCCC",
    ),
    # HERC2/OCA2 eye colour — chr15:28120472
    "rs12913832": (
        "15", 28120472,
        "GGATGATACAAGCACAGGGCTGATT",
        "GTCGGCACAAAGGATCTGTGTCAGA",
    ),
    # SLC24A5 skin pigmentation — chr15:48134287
    "rs1426654": (
        "15", 48134287,
        "CTTCAACATCACCAGCCTCATCTTC",
        "TATTTGCCAGTTAAGAGAAAAGACT",
    ),
    # SLC45A2 pigmentation — chr5:33951693
    "rs16891982": (
        "5", 33951693,
        "GGGAATCAGATGATGCAAGTTCACC",
        "CTCATCCTCTGCACTGTGGTTTGTG",
    ),
    # MC1R red hair — chr16:89919736
    "rs1805007": (
        "16", 89919736,
        "CTGCTGGAGAACATCATCGACGCCA",
        "CAGACATCCTGAGCCAAGCAGGTGC",
    ),
    # PTPN22 autoimmune — chr1:113834946
    "rs2476601": (
        "1", 113834946,
        "ACTGATAATATTCTGATGATGACAC",
        "GGCTTCCAAACATCAGGAAAGCTGA",
    ),
    # SOD2 oxidative stress — chr6:159692840
    "rs4880": (
        "6", 159692840,
        "CCTCCCTCAGCTTCAGCACAGCACA",
        "CTCCCCTGCTCCAGACGCGACCCTC",
    ),
    # FTO obesity — chr16:53786615
    "rs9939609": (
        "16", 53786615,
        "GATGGTGATAAAATATCTTGTGTTT",
        "TTTAGTAAGCTTTGATGCTTACTGT",
    ),
    # CDKN2A/B diabetes — chr9:22134094
    "rs10811661": (
        "9", 22134094,
        "GCCTCAGCAGTCCCTTCATTGTTAT",
        "AATGTAAACTTACCAACAGTCAGGG",
    ),
    # CDKN2B-AS1 cardiovascular — chr9:22125503
    "rs1333049": (
        "9", 22125503,
        "GACCACACTGATGATGTATGCTTTA",
        "GGTTTAGGTCTCTAGTCATTTTCTG",
    ),
    # ACE I/D hypertension — chr17:63488529
    "rs4646994": (
        "17", 63488529,
        "GCCACTACGCTGGAGACCACTCCCA",
        "CTGGAGACCACTCCCATCCTTTCTC",
    ),
    # FADS1 omega-3 — chr11:61797212
    "rs174546": (
        "11", 61797212,
        "CTTCAGGTCCCTCCTACCCCTAAGA",
        "AGGAATGCCATAACATCACCTGCCC",
    ),
    # ---- v2.0 expansion: HIrisPlex-S pigmentation ----
    # HERC2 blue eye — chr15:28344238
    "rs1129038": (
        "15", 28344238,
        "ATACTGGTCATTATCTAAGCCTCAG",
        "TGTCCAAGGACCAAGTGGACAGATA",
    ),
    # SLC24A4 eye/hair — chr14:92801203
    "rs12896399": (
        "14", 92801203,
        "AGGGCAAAGCCTCAACTTGGTCCTT",
        "GTACAGCAAGTCTAGTGCAACTCAG",
    ),
    # TYR green/hazel eye — chr11:89178528
    "rs1393350": (
        "11", 89178528,
        "CTCTGGGAATGCTGTTTCATGACGG",
        "ACACATTTCAAGATATGGCCTTTGC",
    ),
    # OCA2 R419Q — chr15:28009534
    "rs1800407": (
        "15", 28009534,
        "TGGTTCATCCAAGAAACAAATGATG",
        "GTAGATGAAGCACAGAGGTATTCAC",
    ),
    # SLC24A4 eye/hair — chr14:92773663
    "rs2402130": (
        "14", 92773663,
        "GCTCTGTAAGTGACTCAAGTCACTG",
        "ACATGTACTCAGACATGTCCATCCC",
    ),
    # MC1R R160W — chr16:89919709
    "rs1805008": (
        "16", 89919709,
        "CGACGCCATCGTGAACATCATCGAC",
        "GGCTGCCTGGCCGTCTGGATGGCCA",
    ),
    # MC1R D294H — chr16:89920138
    "rs1805009": (
        "16", 89920138,
        "GCCATCACCAAGAACCGCAACCTGC",
        "CCACGAGCGTGCGCATCCTGGTGGA",
    ),
    # MC1R V92M — chr16:89919532
    "rs2228479": (
        "16", 89919532,
        "GCCAGCAGCCCCTTCCTGGCCATCG",
        "GGTCGTGCTGGAGACGGCCGTCATC",
    ),
    # MC1R R163Q — chr16:89919746
    "rs885479": (
        "16", 89919746,
        "ATCGACGCCACCAGACGCCCCCTCA",
        "CATCCTGAGCCAAGCAGGTGCCCAG",
    ),
    # TPCN2 blond hair — chr11:69025765
    "rs35264875": (
        "11", 69025765,
        "CCTCCAGGAGGTGATCCAGAACTAC",
        "TGGCCTCAGCAGCCTCTTCCTGGTC",
    ),
    # OCA2 H615R East Asian skin — chr15:28197037
    "rs1800414": (
        "15", 28197037,
        "CTGGTGGTGGGCTTTGCCATCTTCA",
        "TCACCAGCACCTCCATGATGGCCAG",
    ),
    # BNC2 freckling — chr9:16858084
    "rs10756819": (
        "9", 16858084,
        "TGAGAGCAGCATTCCAGGGCTCTAG",
        "AGCTCTCATTAGCACCAAAGCCATC",
    ),
    # ASIP skin pigment — chr20:34658752
    "rs6059655": (
        "20", 34658752,
        "CAGTTCTGGAAGTCCAGCAGATCTG",
        "TCTGAGGAGACCTTTCAGCTCAGCC",
    ),
    # ASIP tanning — chr20:34660002
    "rs1015362": (
        "20", 34660002,
        "ATCTGCCTCAAGGGTGTCCTCTCCC",
        "AGTCCAAGAGCTTGGAAATGCCTGG",
    ),
    # ---- v2.0: Facial morphology ----
    # PAX3 nose bridge — chr2:222772455
    "rs7559271": (
        "2", 222772455,
        "CTTTGTCAAATGTCTAGGCTCCCAG",
        "CCATCTCCATCTGTCCTCACTTTGC",
    ),
    # DCHS2 nose shape — chr4:155508021
    "rs927833": (
        "4", 155508021,
        "GAATCAGATTCTAAAGAGTCCAGAG",
        "TCACTCCCAGCAATGTAGCTACAGG",
    ),
    # RUNX2 nose bridge — chr6:45411022
    "rs2045323": (
        "6", 45411022,
        "TCTCTAGGACTCAGAAGCCACTGCA",
        "CAGCAGTGGCTTTGATGACATTCCG",
    ),
    # SUPT3H nose tip — chr6:45178920
    "rs4648379": (
        "6", 45178920,
        "AACAGCATTGCCTCAGCTCAGCCTG",
        "GAGCCAGCCAGGATCTCCTCATAGG",
    ),
    # EDAR 370A — chr2:108962124
    "rs3827760": (
        "2", 108962124,
        "AGCCCTCAGCCCCGATGGGCACCAC",
        "GGCTTCAACCCCATCTTCAACTCCG",
    ),
    # IRF6 lip shape — chr1:209989270
    "rs642961": (
        "1", 209989270,
        "CTTGCCTCCTAGTCCACTCCTTGAG",
        "ACTCAGGTCCAGTCCTCCTGCCTTC",
    ),
    # TFAP2B chin cleft — chr6:50863080
    "rs11684042": (
        "6", 50863080,
        "GCTCTGAAGCTGGCAATGCCACTGG",
        "TCTCCACAGCAGGCAGGTTCAGTCC",
    ),
    # TMEM163 facial width — chr2:134981759
    "rs6740960": (
        "2", 134981759,
        "AGGCAGATCACCTGAGGTCAGGAGT",
        "TGAGCCAAGATTGCACCACTGCACT",
    ),
    # ---- v2.0: Hair traits ----
    # TCHH curliness — chr1:152196195
    "rs11803731": (
        "1", 152196195,
        "GAGCACAGCCCACAGCCTCCTCCAG",
        "CAGCCCCAGCAGCACCATCTTCCTG",
    ),
    # AR/EDA2R baldness — chrX:67527747
    "rs2180439": (
        "X", 67527747,
        "TTCTAGTCATGCTCACCAGGCTGCC",
        "AGTGATCCACCCACCTTGGCCTCCC",
    ),
    # EBF1 baldness — chr5:158467199
    "rs929626": (
        "5", 158467199,
        "GTGATCCTCCCACCTCAGCCTCCCA",
        "AAGTGCTGGGATTAGAGGTGTGAGC",
    ),
    # ---- v2.0: Aging / perceived age ----
    # STXBP5L perceived age — chr3:121316831
    "rs258322": (
        "3", 121316831,
        "CTAACAGCTAGCCTCAGCCCCTTGC",
        "AGCAAGGAGCCTGAGGCACTCATGT",
    ),
    # IL6R inflammation — chr1:154453788
    "rs2228145": (
        "1", 154453788,
        "CCCATCCAGGCAGCCCACTGCTCAT",
        "CAGAATCCAGGTGCCCTGGAGTTAG",
    ),
    # CLOCK circadian — chr4:56300440
    "rs1801260": (
        "4", 56300440,
        "CAGGTCCCATCTTCAAACTGAGAAG",
        "GCTTGCAACCAAATCTTCCAGTAAG",
    ),
    # ---- v2.0: Pharmacogenomic ----
    # CYP2C19*2 — chr10:94781859
    "rs4244285": (
        "10", 94781859,
        "TTCCCACTATCATTGATTATTTCCC",
        "GGAACCCATAACAAATTACTTAAAA",
    ),
    # CYP2C19*17 — chr10:94761900
    "rs12248560": (
        "10", 94761900,
        "AATATCTTAATAAATAATCAATAGG",
        "TTGGATCCAGGGAAAGTATTTTTGT",
    ),
    # CYP2D6*4 — chr22:42130692
    "rs1065852": (
        "22", 42130692,
        "CTCGTCCCCCAGCCATGTTCCCTGA",
        "TCCACCAGGCCCCCTGCCACTGCCC",
    ),
    # CYP3A5*3 — chr7:99672916
    "rs776746": (
        "7", 99672916,
        "GATGAACTTTGGCCTCAGTTTGTGG",
        "AATACATCTCCTTCCAAAAGTAAAC",
    ),
    # VKORC1 warfarin — chr16:31096368
    "rs9923231": (
        "16", 31096368,
        "CAGCAGCCTCCCAGAGGATGGCAGC",
        "CACAGTCCAGGGGTCAGATACTCTG",
    ),
    # SLCO1B1 statin — chr12:21176804
    "rs4149056": (
        "12", 21176804,
        "AACAGATGATAGACTACAGTGATGG",
        "CATGAATCAAGCCATTTTCTTCTGC",
    ),
    # ABCB1 drug efflux — chr7:87509329
    "rs1045642": (
        "7", 87509329,
        "ATCTGTGAACTCTTGTTTTCAGCAA",
        "TTAATATTTTGAATGTTCTTTATAG",
    ),
    # CYP2B6 efavirenz — chr19:41006936
    "rs3745274": (
        "19", 41006936,
        "TTCCTCCTTCCCATCCCCCAGGCTC",
        "TGCAGATGATGTTGGCAAGCCATGC",
    ),
    # ---- v2.0: Telomere / longevity ----
    # TERT — chr5:1279790
    "rs2736100": (
        "5", 1279790,
        "CATCGTCAACCCCAAGCAGTGCCAG",
        "AGAGCTGTCGCCGTGGAAGACATTG",
    ),
    # TERC — chr3:169764480
    "rs10936599": (
        "3", 169764480,
        "TGGCTCAAACTCCTGGCCTCAAGTG",
        "TCCCAAAGTGCTGGGATTACAGGCG",
    ),
    # ---- v2.0: Additional health ----
    # PPARG — chr3:12351626
    "rs1801282": (
        "3", 12351626,
        "TTCTGGAAGATATTCAGTACATGTT",
        "CCAATTCAAGCCCAGTCCTTTCTGT",
    ),
    # IL6 inflammation — chr7:22726627
    "rs1800795": (
        "7", 22726627,
        "CCAAGCTGCACTTTTCCCCCTAGTT",
        "TAAACCTCATTCAAAGAGAGTTCTG",
    ),
}

# Fitzpatrick skin type classification by brightness
_FITZPATRICK_THRESHOLDS = [
    (200, "Type I (very fair, always burns)"),
    (175, "Type II (fair, usually burns)"),
    (145, "Type III (medium, sometimes burns)"),
    (115, "Type IV (olive, rarely burns)"),
    (85, "Type V (brown, very rarely burns)"),
    (0, "Type VI (dark brown/black, never burns)"),
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _safe_float(v: float, default: float = 0.0) -> float:
    """Return *default* if *v* is NaN or ±Inf."""
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _extract_facial_measurements(
    img: np.ndarray, face_box: tuple[int, int, int, int]
) -> FacialMeasurements:
    """Extract facial measurements from a detected face region.

    Skin-texture and pigmentation metrics are computed on blurred
    cheek patches (avoiding eyes, mouth, eyebrows, and hair) so that
    structural face features, camera noise, and JPEG compression
    artefacts do not inflate age-related scores.
    """
    x, y, w, h = face_box

    # Ensure we don't go out of bounds
    rows, cols = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, cols - x)
    h = min(h, rows - y)

    face_roi = img[y : y + h, x : x + w]
    if face_roi.size == 0:
        return FacialMeasurements()

    # Convert to various colour spaces
    if face_roi.ndim == 2:
        gray = face_roi
        lab = None
    else:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        _ = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)  # reserved for future use

    # Basic dimensions
    face_ratio = w / max(h, 1)

    # ----- Skin-only cheek patches for texture/pigment metrics -----
    # Left cheek:  x ∈ [10%-35%], y ∈ [50%-70%]
    # Right cheek: x ∈ [65%-90%], y ∈ [50%-70%]
    lc = gray[h * 50 // 100 : h * 70 // 100, w * 10 // 100 : w * 35 // 100]
    rc = gray[h * 50 // 100 : h * 70 // 100, w * 65 // 100 : w * 90 // 100]
    if lc.size > 0 and rc.size > 0:
        skin_patch = np.concatenate([lc.ravel(), rc.ravel()])
    elif lc.size > 0:
        skin_patch = lc.ravel()
    elif rc.size > 0:
        skin_patch = rc.ravel()
    else:
        # Fallback: central patch
        skin_patch = gray[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4].ravel()

    # L-channel cheek patches for pigmentation
    if lab is not None:
        l_lc = lab[h * 50 // 100 : h * 70 // 100, w * 10 // 100 : w * 35 // 100, 0]
        l_rc = lab[h * 50 // 100 : h * 70 // 100, w * 65 // 100 : w * 90 // 100, 0]
        if l_lc.size > 0 and l_rc.size > 0:
            l_skin_patch = np.concatenate([l_lc.ravel(), l_rc.ravel()])
        elif l_lc.size > 0:
            l_skin_patch = l_lc.ravel()
        else:
            l_skin_patch = l_rc.ravel() if l_rc.size > 0 else lab[:, :, 0].ravel()
    else:
        l_skin_patch = skin_patch

    # Skin brightness (from L channel of LAB, or grayscale)
    if lab is not None:
        l_channel = lab[:, :, 0]
        skin_brightness = _safe_float(float(np.mean(l_channel)))
    else:
        skin_brightness = _safe_float(float(np.mean(gray)))

    # Skin uniformity (std of cheek skin patch)
    skin_uniformity = _safe_float(float(np.std(skin_patch))) if skin_patch.size > 0 else 0.0

    # Skin redness (inflammation proxy)
    skin_redness = 0.0
    skin_yellowness = 0.0
    if face_roi.ndim == 3:
        b, g, r = (
            float(np.mean(face_roi[:, :, 0])),
            float(np.mean(face_roi[:, :, 1])),
            float(np.mean(face_roi[:, :, 2])),
        )
        skin_redness = _safe_float(r - (b + g) / 2)
        if lab is not None:
            skin_yellowness = _safe_float(float(np.mean(lab[:, :, 2])))

    # Wrinkle score (edge density in forehead region)
    # Skip the top 10% of the face box to avoid the hairline, which
    # creates strong Canny edges that are not wrinkles.  Analyse
    # rows 10%–25% (the actual forehead skin below the hairline).
    forehead = gray[h * 10 // 100 : h * 25 // 100, w // 4 : 3 * w // 4]
    if forehead.size > 0:
        forehead_blur = cv2.GaussianBlur(forehead, (7, 7), 0)
        edges = cv2.Canny(forehead_blur, 80, 180)
        wrinkle_score = float(np.sum(edges > 0) / max(forehead.size, 1))
    else:
        wrinkle_score = 0.0

    # Symmetry score (compare left and right halves)
    left_half = gray[:, : w // 2]
    right_half = cv2.flip(gray[:, w // 2 :], 1)
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w > 0:
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        diff = np.abs(left_half.astype(float) - right_half.astype(float))
        symmetry_score = _safe_float(1.0 - float(np.mean(diff) / 255.0), 0.5)
    else:
        symmetry_score = 0.5

    # Dark circle score (under-eye region darkness)
    # Subtract a small baseline (0.05) to account for natural shadowing
    # from the brow ridge, so normal lighting doesn't inflate the score.
    eye_y = h // 3
    under_eye = gray[eye_y : eye_y + h // 8, w // 4 : 3 * w // 4]
    cheek = gray[h // 2 : h // 2 + h // 8, w // 4 : 3 * w // 4]
    if under_eye.size > 0 and cheek.size > 0:
        raw_dc = (float(np.mean(cheek)) - float(np.mean(under_eye))) / 50.0
        dark_circle_score = max(0.0, min(_safe_float(raw_dc) - 0.05, 1.0))
    else:
        dark_circle_score = 0.0

    # Texture roughness — Laplacian variance of *blurred* cheek patches.
    # Pre-blur with GaussianBlur to suppress camera sensor noise, JPEG
    # compression artefacts, and skin micro-texture (pores, peach fuzz)
    # that are resolution-dependent, not age-related.  The /3000
    # normaliser is calibrated for smartphone-resolution images where
    # raw Laplacian variance on cheek skin is typically 200–1000.
    _lap_vars = []
    for _patch_2d in (lc, rc):
        if _patch_2d.size > 100:
            _blurred = cv2.GaussianBlur(_patch_2d, (7, 7), 0)
            _lp = cv2.Laplacian(_blurred, cv2.CV_64F)
            _lap_vars.append(float(np.var(_lp)))
    if _lap_vars:
        texture_roughness = min(sum(_lap_vars) / len(_lap_vars) / 3000.0, 1.0)
    else:
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        lap = cv2.Laplacian(blurred_gray, cv2.CV_64F)
        texture_roughness = min(float(np.var(lap)) / 10000.0, 1.0)

    # UV damage score — pigmentation irregularity in *blurred* cheek
    # patches.  A GaussianBlur removes high-frequency noise while
    # preserving real pigmentation variation.  The /60 normaliser
    # accounts for the fact that even uniform young skin has L-channel
    # std of 5–15 from 3D face curvature and lighting gradients.
    # Compute on each 2-D L-channel cheek patch, then average.
    _l_stds = []
    if lab is not None:
        for _lp in (l_lc, l_rc):
            if _lp.size > 50 and min(_lp.shape) >= 3:
                _lb = cv2.GaussianBlur(_lp.astype(np.float32), (5, 5), 0)
                _l_stds.append(float(np.std(_lb)))
    if _l_stds:
        uv_damage_score = min(sum(_l_stds) / len(_l_stds) / 60.0, 1.0)
    elif l_skin_patch.size > 2:
        uv_damage_score = min(_safe_float(float(np.std(l_skin_patch.astype(float)))) / 60.0, 1.0)
    else:
        uv_damage_score = 0.0

    return FacialMeasurements(
        face_width=float(w),
        face_height=float(h),
        face_ratio=face_ratio,
        skin_brightness=skin_brightness,
        skin_uniformity=skin_uniformity,
        skin_redness=skin_redness,
        skin_yellowness=skin_yellowness,
        wrinkle_score=wrinkle_score,
        symmetry_score=symmetry_score,
        dark_circle_score=dark_circle_score,
        texture_roughness=texture_roughness,
        uv_damage_score=uv_damage_score,
    )


# ---------------------------------------------------------------------------
# Age estimation
# ---------------------------------------------------------------------------


def _estimate_biological_age(
    measurements: FacialMeasurements,
    chronological_age: int,
    sex: str = "unknown",
) -> int:
    """Estimate biological age from facial measurements.

    Uses a weighted combination of wrinkle score, skin uniformity,
    dark circles, texture roughness, UV damage, and symmetry —
    calibrated against published perceived-age studies (Christensen
    et al., 2009; Matts et al., 2007).

    Key improvements over naive linear baselines:
    - Non-linear aging curves: baselines accelerate after age 50
      (menopause, cumulative UV threshold, elastin collapse).
    - Pigmentation uniformity (UV damage) given higher weight per
      Matts et al. (2007) — strongest single cue for perceived age.
    - Sex adjustment: women appear ~2 years younger on average
      (higher subcutaneous fat, estrogen-mediated collagen retention).
    - Fitzpatrick skin-type proxy: darker skin ages visibly slower
      due to melanin photoprotection.
    - Offset clamped to ±10 years (credible range for photo-based
      estimation per Horvath clock population SD ~5.5 years).

    References
    ----------
    .. [1] Matts P.J. et al. (2007) — Skin color homogeneity is the
           strongest predictor of perceived age.
    .. [2] Christensen K. et al. (2009) BMJ — Perceived age predicts
           survival in Danish twins.
    .. [3] Glogau R.G. (1996) — Wrinkle classification scale.
    """
    age_offset = 0.0

    # ── Non-linear "expected" baselines ──────────────────────────
    # Aging accelerates after ~50 (Glogau type II→III transition ~50,
    # post-menopausal collagen loss, cumulative photodamage threshold).
    age_over_25 = max(chronological_age - 25, 0)
    age_over_50 = max(chronological_age - 50, 0)

    # 1. WRINKLES (weight: 80)
    # Baseline: slow increase 25–50, then accelerates
    expected_wrinkle = 0.005 + age_over_25 * 0.0004 + age_over_50 * 0.0003
    wrinkle_diff = measurements.wrinkle_score - max(expected_wrinkle, 0.003)
    age_offset += wrinkle_diff * 80

    # 2. TEXTURE ROUGHNESS (weight: 30)
    expected_texture = 0.02 + max(chronological_age - 20, 0) * 0.0015 + age_over_50 * 0.002
    texture_diff = measurements.texture_roughness - expected_texture
    age_offset += texture_diff * 30

    # 3. DARK CIRCLES (weight: 5, now with age-expected baseline)
    # Dark circles increase naturally with age (skin thinning,
    # orbital fat descent — Matsui et al., 2012).
    expected_dc = min(max(chronological_age - 30, 0) * 0.01, 0.3)
    dc_excess = measurements.dark_circle_score - expected_dc
    age_offset += dc_excess * 5

    # 4. UV DAMAGE / PIGMENTATION IRREGULARITY (weight: 25)
    # Matts et al. (2007): skin color homogeneity is the strongest
    # single predictor of perceived age — even stronger than wrinkles.
    expected_uv = 0.05 + age_over_25 * 0.0015 + age_over_50 * 0.002
    uv_diff = measurements.uv_damage_score - expected_uv
    age_offset += uv_diff * 25

    # 5. SYMMETRY (weight: -8, good symmetry reduces perceived age)
    age_offset -= (measurements.symmetry_score - 0.7) * 8

    # 6. SKIN UNIFORMITY (high std dev → more aging)
    if measurements.skin_uniformity > 20:
        age_offset += (measurements.skin_uniformity - 20) * 0.2

    # ── Sex adjustment ───────────────────────────────────────────
    # Women appear ~2 years younger at same chronological age due
    # to higher subcutaneous fat and estrogen-mediated collagen
    # retention (though post-menopausal acceleration exists).
    if sex.lower() == "female":
        age_offset -= 2.0
    elif sex.lower() == "male":
        age_offset += 0.5

    # ── Fitzpatrick / skin-type adjustment ───────────────────────
    # Darker skin ages visibly slower due to melanin photoprotection.
    # Fitzpatrick V–VI: ~5 years slower visible aging vs III.
    # Use skin brightness as proxy (lower brightness → darker skin).
    brightness = measurements.skin_brightness
    if brightness < 100:
        # Fitzpatrick V–VI: strongly slower visible aging
        age_offset -= 3.0
    elif brightness < 130:
        # Fitzpatrick IV: moderately slower
        age_offset -= 1.5

    # ── Clamp ────────────────────────────────────────────────────
    # ±10 years is the credible range for photo-based estimation.
    # Horvath clock population SD is ~5.5 years; ±10 covers ~95%.
    age_offset = max(-10.0, min(10.0, age_offset))

    bio_age = chronological_age + age_offset
    bio_age = max(15, min(110, bio_age))
    return int(round(bio_age))


def _telomere_from_age(biological_age: int, sex: str = "unknown") -> float:
    """Estimate telomere length from biological age.

    Uses a two-phase piecewise model calibrated against population
    studies (Müezzinler et al., 2013; Aubert & Lansdorp, 2008):

    - Birth–20: faster attrition (~60 bp/year → 0.060 kb/year)
    - 20+: slower adult attrition (~25 bp/year → 0.025 kb/year)

    Females have ~0.2 kb longer telomeres on average (estrogen
    activates telomerase via hTERT promoter; Barrett & Richardson,
    2011; Gardner et al., 2014).
    """
    if biological_age <= 20:
        # Childhood/adolescent phase: ~60 bp/year attrition
        tl = 11.0 - 0.060 * biological_age
    else:
        # Adult phase: starts at TL(20) ≈ 9.8, then ~25 bp/year
        tl = 9.80 - 0.025 * (biological_age - 20)

    # Sex adjustment: females ~0.2 kb longer
    if sex.lower() == "female":
        tl += 0.2
    elif sex.lower() == "male":
        tl -= 0.05

    # Floor at 4.0 kb — cells with TL < 4 kb are in crisis/senescence;
    # 2.0 kb floor was unrealistically low for living adults.
    return max(round(tl, 2), 4.0)


def _telomere_percentile(tl_kb: float, chronological_age: int, sex: str = "unknown") -> int:
    """Compute telomere length percentile for chronological age.

    Uses age-adjusted reference ranges from population studies,
    with the two-phase attrition model and sex correction.
    """
    # Expected TL for this age (using the same two-phase model)
    if chronological_age <= 20:
        expected = 11.0 - 0.060 * chronological_age
    else:
        expected = 9.80 - 0.025 * (chronological_age - 20)

    # Sex adjustment to expected
    if sex.lower() == "female":
        expected += 0.2
    elif sex.lower() == "male":
        expected -= 0.05

    sd = 1.2  # population SD ≈ 1.2 kb
    z = (tl_kb - expected) / sd
    # Convert z-score to percentile using sigmoid approximation
    percentile = int(round(100 / (1 + math.exp(-1.7 * z))))
    return max(1, min(99, percentile))


# ---------------------------------------------------------------------------
# Ancestry estimation
# ---------------------------------------------------------------------------


def _estimate_ancestry(measurements: FacialMeasurements) -> AncestryEstimate:
    """Estimate ancestral composition from facial features.

    Uses skin brightness (LAB L* channel) as primary proxy with face
    ratio and skin yellowness as secondary features.  This is a
    simplified model — real ancestry prediction requires 3D facial
    morphology or genomic data.
    """
    brightness = measurements.skin_brightness
    yellowness = measurements.skin_yellowness
    face_ratio = measurements.face_ratio

    # Initialise scores
    scores = {
        "european": 0.0,
        "east_asian": 0.0,
        "south_asian": 0.0,
        "african": 0.0,
        "middle_eastern": 0.0,
        "latin_american": 0.0,
    }

    # Skin brightness signals
    if brightness > 170:
        scores["european"] += 0.5
        scores["east_asian"] += 0.2
    elif brightness > 140:
        scores["european"] += 0.2
        scores["east_asian"] += 0.3
        scores["south_asian"] += 0.2
        scores["middle_eastern"] += 0.2
        scores["latin_american"] += 0.2
    elif brightness > 110:
        scores["south_asian"] += 0.3
        scores["middle_eastern"] += 0.3
        scores["latin_american"] += 0.3
        scores["east_asian"] += 0.1
    else:
        scores["african"] += 0.5
        scores["south_asian"] += 0.2

    # Yellowness (b* channel) — higher in East Asian populations
    if yellowness > 140:
        scores["east_asian"] += 0.3
        scores["south_asian"] += 0.1
    elif yellowness > 130:
        scores["east_asian"] += 0.1
        scores["south_asian"] += 0.15
        scores["latin_american"] += 0.1

    # Face width-to-height ratio
    if face_ratio > 0.85:
        scores["east_asian"] += 0.1
    elif face_ratio < 0.70:
        scores["european"] += 0.1
        scores["african"] += 0.05

    # Normalise to sum to 1.0
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] = round(scores[k] / total, 3)
    else:
        # Uniform if no signal
        for k in scores:
            scores[k] = round(1.0 / len(scores), 3)

    return AncestryEstimate(
        european=scores["european"],
        east_asian=scores["east_asian"],
        south_asian=scores["south_asian"],
        african=scores["african"],
        middle_eastern=scores["middle_eastern"],
        latin_american=scores["latin_american"],
        confidence=min(max(scores.values()) * 1.5, 0.7),  # Cap at 0.7
    )


# ---------------------------------------------------------------------------
# Predicted phenotype traits
# ---------------------------------------------------------------------------


def _predict_eye_colour(brightness: float, ancestry: AncestryEstimate) -> str:
    """Predict eye colour from skin brightness and ancestry."""
    if ancestry.european > 0.5 and brightness > 170:
        return "blue/green (likely)"
    elif ancestry.european > 0.3 and brightness > 150:
        return "hazel/green (possible)"
    elif ancestry.east_asian > 0.3:
        return "dark brown"
    elif ancestry.african > 0.3:
        return "dark brown"
    else:
        return "brown"


def _predict_hair_colour(brightness: float, ancestry: AncestryEstimate) -> str:
    """Predict hair colour from features."""
    if brightness > 180 and ancestry.european > 0.4:
        return "light brown / blonde (likely)"
    elif brightness > 160 and ancestry.european > 0.3:
        return "brown"
    elif ancestry.east_asian > 0.3:
        return "black"
    elif ancestry.african > 0.3:
        return "black"
    else:
        return "dark brown / black"


def _classify_skin_type(brightness: float) -> str:
    """Classify Fitzpatrick skin type from skin brightness."""
    for threshold, label in _FITZPATRICK_THRESHOLDS:
        if brightness >= threshold:
            return label
    return "Type VI (dark brown/black, never burns)"


# ---------------------------------------------------------------------------
# SNP prediction from ancestry
# ---------------------------------------------------------------------------


def _predict_variants_from_ancestry(
    ancestry: AncestryEstimate,
    measurements: FacialMeasurements,
) -> list[PredictedVariant]:
    """Predict likely genetic variants based on estimated ancestry.

    Uses population-specific allele frequencies to estimate the most
    probable genotypes.  For each SNP, the predicted genotype probability
    is calculated as a weighted average of population frequencies based
    on the ancestry estimate.
    """
    variants: list[PredictedVariant] = []

    # Convert ancestry to weight vector
    pop_weights = {
        "european": ancestry.european,
        "east_asian": ancestry.east_asian,
        "african": ancestry.african,
        "south_asian": ancestry.south_asian,
        "latin_american": ancestry.latin_american,
        "middle_eastern": ancestry.middle_eastern,
    }

    for rsid, pop_data in _POPULATION_SNP_FREQUENCIES.items():
        # Weighted allele frequency
        weighted_freq = 0.0
        total_weight = 0.0
        gene = ""
        condition = ""

        for pop, (freq, g, cond) in pop_data.items():
            weight = pop_weights.get(pop, 0.0)
            weighted_freq += freq * weight
            total_weight += weight
            gene = g
            condition = cond

        if total_weight > 0:
            p = weighted_freq / total_weight
        else:
            p = 0.2  # default

        # Hardy-Weinberg: P(AA) = (1-p)^2, P(Aa) = 2p(1-p), P(aa) = p^2
        p_homo_ref = (1 - p) ** 2
        p_het = 2 * p * (1 - p)
        p_homo_alt = p**2

        # Look up actual risk/ref alleles for this SNP.
        risk_al, ref_al = _SNP_ALLELES.get(rsid, ("T", "C"))

        if p_homo_ref > p_het and p_homo_ref > p_homo_alt:
            genotype = "homozygous reference (low risk)"
            conf = p_homo_ref
        elif p_het > p_homo_alt:
            genotype = "heterozygous (moderate risk)"
            conf = p_het
        else:
            genotype = "homozygous variant (elevated risk)"
            conf = p_homo_alt

        variants.append(
            PredictedVariant(
                rsid=rsid,
                gene=gene,
                predicted_genotype=genotype,
                confidence=round(conf, 3),
                basis=f"Population frequency ({condition}); estimated risk allele freq={p:.2f}",
                risk_allele=risk_al,
                ref_allele=ref_al,
            )
        )

    # Add phenotype-specific variants based on measurements
    # MC1R from skin brightness + redness
    if measurements.skin_brightness > 180 and measurements.skin_redness > 10:
        variants.append(
            PredictedVariant(
                rsid="rs1805007",
                gene="MC1R",
                predicted_genotype="heterozygous (likely carrier)",
                confidence=0.4,
                basis="Very fair skin with redness suggests MC1R variant",
                risk_allele="T",
                ref_allele="C",
            )
        )

    # SOD2 from oxidative stress markers
    if measurements.uv_damage_score > 0.5:
        variants.append(
            PredictedVariant(
                rsid="rs4880",
                gene="SOD2",
                predicted_genotype="heterozygous (possible reduced antioxidant)",
                confidence=0.35,
                basis="Elevated UV damage markers suggest possible SOD2 variant",
                risk_allele="T",
                ref_allele="C",
            )
        )

    return variants


def _reconstruct_dna(variants: list[PredictedVariant]) -> ReconstructedDNA:
    """Build reconstructed DNA sequence fragments from predicted variants.

    For each predicted variant that has known genomic context (flanking
    reference sequences), constructs a 51-bp sequence fragment showing the
    predicted alleles in their genomic context.
    """
    sequences: list[ReconstructedSequence] = []

    # De-duplicate variants by rsid (phenotype-specific additions may overlap)
    seen: set[str] = set()
    for v in variants:
        if v.rsid in seen:
            continue
        seen.add(v.rsid)

        ctx = _SNP_GENOMIC_CONTEXT.get(v.rsid)
        if ctx is None:
            continue

        chrom, pos, flank5, flank3 = ctx
        risk_al, ref_al = _SNP_ALLELES.get(v.rsid, (v.risk_allele, v.ref_allele))

        # Determine diploid alleles from predicted genotype
        geno = v.predicted_genotype.lower()
        if "homozygous reference" in geno or "low risk" in geno:
            al1, al2 = ref_al, ref_al
        elif "heterozygous" in geno:
            al1, al2 = ref_al, risk_al
        else:
            # homozygous variant / elevated risk
            al1, al2 = risk_al, risk_al

        sequences.append(
            ReconstructedSequence(
                rsid=v.rsid,
                gene=v.gene,
                chromosome=chrom,
                position=pos,
                ref_allele=ref_al,
                predicted_allele_1=al1,
                predicted_allele_2=al2,
                flanking_5prime=flank5,
                flanking_3prime=flank3,
                confidence=v.confidence,
            )
        )

    # Sort by chromosome and position for a clean genomic ordering
    sequences.sort(key=lambda s: (s.chromosome.zfill(2), s.position))

    return ReconstructedDNA(
        sequences=sequences,
        total_variants=len(sequences),
    )


# ---------------------------------------------------------------------------
# Pharmacogenomic prediction (v2.0)
# ---------------------------------------------------------------------------

# Pharmacogenomic variant → drug / phenotype mapping
_PHARMACOGENOMIC_MAP: dict[str, dict] = {
    "rs4244285": {
        "gene": "CYP2C19",
        "star": "*2",
        "function": "no_function",
        "drugs": ["clopidogrel", "omeprazole", "escitalopram", "voriconazole"],
        "recommendation_het": "Intermediate metabolizer — consider alternative to clopidogrel; standard PPI dosing likely adequate.",
        "recommendation_hom": "Poor metabolizer — clopidogrel contraindicated (use prasugrel/ticagrelor); consider PPI dose reduction.",
    },
    "rs12248560": {
        "gene": "CYP2C19",
        "star": "*17",
        "function": "increased_function",
        "drugs": ["clopidogrel", "escitalopram", "sertraline", "voriconazole"],
        "recommendation_het": "Rapid metabolizer — enhanced clopidogrel activation; may need higher SSRI doses.",
        "recommendation_hom": "Ultra-rapid metabolizer — increased clopidogrel effect; reduced voriconazole efficacy.",
    },
    "rs1065852": {
        "gene": "CYP2D6",
        "star": "*4",
        "function": "no_function",
        "drugs": ["codeine", "tramadol", "tamoxifen", "metoprolol", "fluoxetine"],
        "recommendation_het": "Intermediate metabolizer — reduced codeine→morphine conversion; standard tamoxifen may suffice.",
        "recommendation_hom": "Poor metabolizer — codeine/tramadol ineffective; tamoxifen activation impaired; use alternatives.",
    },
    "rs776746": {
        "gene": "CYP3A5",
        "star": "*3",
        "function": "non_expressor",
        "drugs": ["tacrolimus", "cyclosporine", "midazolam"],
        "recommendation_het": "Intermediate expressor — may need standard tacrolimus dosing.",
        "recommendation_hom": "CYP3A5 non-expressor — standard tacrolimus dosing; expressors need ~50% higher doses.",
    },
    "rs9923231": {
        "gene": "VKORC1",
        "star": "-1639G>A",
        "function": "increased_sensitivity",
        "drugs": ["warfarin", "acenocoumarol", "phenprocoumon"],
        "recommendation_het": "Intermediate warfarin sensitivity — consider ~25% dose reduction from standard.",
        "recommendation_hom": "High warfarin sensitivity — consider ~50% dose reduction; start low, monitor INR closely.",
    },
    "rs4149056": {
        "gene": "SLCO1B1",
        "star": "*5 (Val174Ala)",
        "function": "decreased_transport",
        "drugs": ["simvastatin", "atorvastatin", "rosuvastatin", "pravastatin"],
        "recommendation_het": "Increased statin myopathy risk — avoid simvastatin >20mg; consider rosuvastatin.",
        "recommendation_hom": "High statin myopathy risk — avoid simvastatin entirely; use low-dose rosuvastatin or pravastatin.",
    },
    "rs1045642": {
        "gene": "ABCB1",
        "star": "3435C>T",
        "function": "reduced_efflux",
        "drugs": ["digoxin", "cyclosporine", "tacrolimus", "fexofenadine"],
        "recommendation_het": "Slightly altered drug bioavailability — monitor digoxin levels if prescribed.",
        "recommendation_hom": "Reduced P-glycoprotein function — increased bioavailability of substrate drugs; dose adjustments may be needed.",
    },
    "rs3745274": {
        "gene": "CYP2B6",
        "star": "*6",
        "function": "decreased_function",
        "drugs": ["efavirenz", "methadone", "cyclophosphamide", "bupropion"],
        "recommendation_het": "Intermediate metabolizer — consider efavirenz dose reduction to 400mg.",
        "recommendation_hom": "Poor metabolizer — efavirenz 200-400mg recommended; higher methadone levels expected.",
    },
}


def _predict_pharmacogenomics(
    variants: list[PredictedVariant],
    ancestry: AncestryEstimate,
) -> list[PharmacogenomicPrediction]:
    """Predict pharmacogenomic metabolizer phenotypes from predicted variants.

    Uses predicted genotypes at pharmacogene loci combined with ancestry-based
    population frequency priors to estimate drug metabolism phenotypes.
    """
    predictions: list[PharmacogenomicPrediction] = []
    variant_map = {v.rsid: v for v in variants}

    for rsid, pgx in _PHARMACOGENOMIC_MAP.items():
        v = variant_map.get(rsid)
        if v is None:
            continue

        geno = v.predicted_genotype.lower()
        if "homozygous variant" in geno or "elevated risk" in geno:
            phenotype = "Poor Metabolizer" if pgx["function"] == "no_function" else (
                "Ultra-rapid Metabolizer" if pgx["function"] == "increased_function" else
                "High Sensitivity" if pgx["function"] == "increased_sensitivity" else
                "Non-expressor" if pgx["function"] == "non_expressor" else
                "Reduced Function"
            )
            recommendation = pgx["recommendation_hom"]
            conf = v.confidence
        elif "heterozygous" in geno:
            phenotype = "Intermediate Metabolizer" if pgx["function"] in ("no_function", "decreased_function") else (
                "Rapid Metabolizer" if pgx["function"] == "increased_function" else
                "Intermediate Sensitivity" if pgx["function"] == "increased_sensitivity" else
                "Intermediate Expressor" if pgx["function"] == "non_expressor" else
                "Carrier"
            )
            recommendation = pgx["recommendation_het"]
            conf = v.confidence * 0.85
        else:
            phenotype = "Normal Metabolizer"
            recommendation = f"Standard {pgx['gene']} function predicted — no pharmacogenomic dose adjustments anticipated."
            conf = v.confidence * 0.7

        predictions.append(
            PharmacogenomicPrediction(
                gene=pgx["gene"],
                rsid=rsid,
                predicted_phenotype=phenotype,
                confidence=round(conf, 3),
                affected_drugs=pgx["drugs"],
                clinical_recommendation=recommendation,
                basis=f"Ancestry-weighted allele frequency ({pgx['star']}); genotype: {v.predicted_genotype}",
            )
        )

    return predictions


# ---------------------------------------------------------------------------
# Facial health screening (v2.0)
# ---------------------------------------------------------------------------


def _screen_facial_health(
    measurements: FacialMeasurements,
    bio_age: int,
    chrono_age: int,
    sex: str,
    ancestry: AncestryEstimate,
) -> FacialHealthScreening:
    """Derive health screening indicators from facial measurements.

    Based on published correlations between facial features and health
    biomarkers (Coetzee et al., 2009; Whitehead et al., 2012; Sundelin
    et al., 2013; Mannino et al., 2018).
    """
    # --- BMI estimation from facial adiposity ---
    # Face ratio (width/height) correlates with BMI: wider = higher BMI
    # Coetzee et al. (2009): facial adiposity is a valid cue to BMI
    fr = measurements.face_ratio
    if fr > 0.95:
        bmi_cat, bmi_conf = "Obese (estimated)", 0.35
    elif fr > 0.85:
        bmi_cat, bmi_conf = "Overweight (estimated)", 0.40
    elif fr > 0.70:
        bmi_cat, bmi_conf = "Normal (estimated)", 0.45
    else:
        bmi_cat, bmi_conf = "Underweight (estimated)", 0.35

    # --- Anemia risk from skin pallor ---
    # Mannino et al. (2018): facial color analysis correlates with hemoglobin
    # Low skin redness + low brightness (in lighter skin) suggests pallor
    anemia_score = 0.0
    brightness = measurements.skin_brightness
    redness = measurements.skin_redness
    if brightness > 140:
        # Lighter skin — redness is a stronger pallor signal
        if redness < 5:
            anemia_score = 0.6
        elif redness < 10:
            anemia_score = 0.3
        elif redness < 15:
            anemia_score = 0.1
    else:
        # Darker skin — harder to detect pallor from photos
        if redness < 3:
            anemia_score = 0.4
        elif redness < 8:
            anemia_score = 0.2
    # Dark circles also contribute to anemia screening
    anemia_score += measurements.dark_circle_score * 0.15
    anemia_score = min(1.0, anemia_score)

    # --- Cardiovascular risk indicators ---
    cv_indicators: list[str] = []
    # Facial aging gap > 5 years: Christensen et al. (2009), Esquirol et al. (2018)
    age_gap = bio_age - chrono_age
    if age_gap > 5:
        cv_indicators.append(f"Accelerated facial aging (+{age_gap}y) — associated with CV risk")
    # High skin redness: persistent erythema correlates with hypertension
    if redness > 25:
        cv_indicators.append("Elevated facial redness — hypertension screening recommended")
    # Skin yellowness: may indicate dyslipidemia
    if measurements.skin_yellowness > 145:
        cv_indicators.append("Elevated skin yellowness — lipid panel recommended")
    # High UV damage: Esquirol (2018) deep wrinkles correlate with CV mortality
    if measurements.wrinkle_score > 0.04:
        cv_indicators.append("Deep wrinkle pattern — correlates with cardiovascular mortality (Esquirol 2018)")

    # --- Thyroid indicators ---
    thyroid_indicators: list[str] = []
    # Skin dryness (high texture roughness) + puffiness + yellowish tint
    if measurements.texture_roughness > 0.15 and measurements.skin_yellowness > 135:
        thyroid_indicators.append("Dry skin with yellowish tinge — consider thyroid screening (carotenemia)")
    # Very smooth/moist skin could indicate hyperthyroidism
    if measurements.texture_roughness < 0.02 and redness > 20:
        thyroid_indicators.append("Smooth, flushed skin — hyperthyroid features possible")

    # --- Fatigue/stress score ---
    # Sundelin et al. (2013): dark circles, skin pallor, texture changes
    fatigue = (
        measurements.dark_circle_score * 0.40
        + measurements.wrinkle_score * 30  # wrinkles as stress indicator
        + min(measurements.skin_uniformity / 30.0, 1.0) * 0.15
        + max(measurements.texture_roughness - 0.05, 0) * 2.0
    )
    fatigue = max(0.0, min(1.0, fatigue))

    # --- Hydration score ---
    # Inverse of skin roughness + uniformity deviation
    hydration = 80.0
    hydration -= measurements.texture_roughness * 100
    hydration -= max(measurements.skin_uniformity - 15, 0) * 1.0
    hydration += (measurements.symmetry_score - 0.5) * 10
    hydration = max(10.0, min(100.0, hydration))

    return FacialHealthScreening(
        estimated_bmi_category=bmi_cat,
        bmi_confidence=bmi_conf,
        anemia_risk_score=round(anemia_score, 3),
        cardiovascular_risk_indicators=cv_indicators,
        thyroid_indicators=thyroid_indicators,
        fatigue_stress_score=round(fatigue, 3),
        hydration_score=round(hydration, 1),
    )


# ---------------------------------------------------------------------------
# Dermatological analysis (v2.0)
# ---------------------------------------------------------------------------


def _analyze_dermatology(
    measurements: FacialMeasurements,
    bio_age: int,
    chrono_age: int,
    sex: str,
    skin_type: str,
) -> DermatologicalAnalysis:
    """Perform dermatological analysis from facial measurements.

    Based on clinical dermatology correlations and computational skin
    analysis methods (Brinker et al., 2019; Kottner et al., 2013).
    """
    # --- Rosacea risk ---
    # High redness + visible texture + lighter skin
    rosacea = 0.0
    if measurements.skin_redness > 20:
        rosacea += 0.3
    if measurements.skin_redness > 30:
        rosacea += 0.2
    if measurements.skin_brightness > 150:
        rosacea += 0.15  # More common in fair skin
    if measurements.uv_damage_score > 0.3:
        rosacea += 0.1
    rosacea = min(1.0, rosacea)

    # --- Melasma risk ---
    # High pigmentation irregularity + moderate skin tone
    melasma = 0.0
    if measurements.skin_uniformity > 20:
        melasma += 0.25
    if measurements.skin_uniformity > 30:
        melasma += 0.15
    if 110 < measurements.skin_brightness < 170:
        melasma += 0.2  # More common in Fitzpatrick III-IV
    if sex.lower() == "female":
        melasma += 0.15
    melasma = min(1.0, melasma)

    # --- Photo-aging gap ---
    # How much skin appears aged relative to chronological age
    photo_aging_gap = bio_age - chrono_age

    # --- Acne severity ---
    # Proxy: texture roughness + redness + uniformity deviation in younger people
    acne = 0.0
    if chrono_age < 35:
        if measurements.texture_roughness > 0.08:
            acne += 0.3
        if measurements.skin_redness > 15 and measurements.skin_uniformity > 20:
            acne += 0.25
        if measurements.skin_redness > 25:
            acne += 0.2
    else:
        # Adult acne is less common but possible
        if measurements.texture_roughness > 0.12 and measurements.skin_redness > 20:
            acne += 0.2
    acne = min(1.0, acne)

    # --- Skin cancer risk factors ---
    cancer_risks: list[str] = []
    if "Type I" in skin_type or "Type II" in skin_type:
        cancer_risks.append("Fair skin type (Fitzpatrick I-II) — elevated melanoma/NMSC risk")
    if measurements.uv_damage_score > 0.4:
        cancer_risks.append("Significant UV damage detected — dermatological screening recommended")
    if measurements.uv_damage_score > 0.6:
        cancer_risks.append("Severe photo-damage — high risk for actinic keratoses/NMSC")
    if measurements.skin_redness > 25 and "Type I" in skin_type:
        cancer_risks.append("Fair skin with chronic erythema — increased BCC risk")

    # --- Pigmentation disorder risk ---
    pigment_risk = 0.0
    if measurements.skin_uniformity > 25:
        pigment_risk += 0.3
    if measurements.uv_damage_score > 0.4:
        pigment_risk += 0.25
    pigment_risk = min(1.0, pigment_risk)

    # --- Moisture barrier score ---
    moisture = 70.0
    moisture -= measurements.texture_roughness * 80
    moisture -= max(measurements.skin_uniformity - 15, 0) * 0.8
    if measurements.skin_redness > 20:
        moisture -= 10  # Barrier disruption often accompanies redness
    moisture = max(10.0, min(100.0, moisture))

    return DermatologicalAnalysis(
        rosacea_risk_score=round(rosacea, 3),
        melasma_risk_score=round(melasma, 3),
        photo_aging_gap=photo_aging_gap,
        acne_severity_score=round(acne, 3),
        skin_cancer_risk_factors=cancer_risks,
        pigmentation_disorder_risk=round(pigment_risk, 3),
        moisture_barrier_score=round(moisture, 1),
    )


# ---------------------------------------------------------------------------
# Condition screening (v2.0)
# ---------------------------------------------------------------------------


def _screen_conditions(
    measurements: FacialMeasurements,
    bio_age: int,
    chrono_age: int,
    sex: str,
) -> list[ConditionScreening]:
    """Screen for medical conditions based on facial feature patterns.

    Based on clinical literature for facial manifestations of endocrine
    and other conditions (Kong et al., 2018; Kosilek et al., 2017;
    Schneider et al., 2020; Boelaert et al., 2010).
    """
    screenings: list[ConditionScreening] = []

    # --- Acromegaly screening ---
    # Coarsened features: wide face, large nose, prominent jaw
    # Kong et al. (2018): DL model achieved AUC 0.96
    acro_score = 0.0
    acro_markers: list[str] = []
    if measurements.face_ratio > 0.90:
        acro_score += 0.15
        acro_markers.append("Wide facial proportions")
    if measurements.texture_roughness > 0.15:
        acro_score += 0.2
        acro_markers.append("Coarsened skin texture")
    if measurements.skin_redness > 20:
        acro_score += 0.1
        acro_markers.append("Facial plethora")
    if acro_score > 0.1:
        screenings.append(ConditionScreening(
            condition="Acromegaly",
            risk_score=round(min(acro_score, 1.0), 3),
            facial_markers=acro_markers,
            confidence=0.25,
            recommendation="Consider IGF-1 level testing if clinical suspicion warrants.",
        ))

    # --- Cushing syndrome screening ---
    # Moon face: high face ratio + plethora + skin changes
    # Kosilek et al. (2017): 90% sensitivity, 90% specificity
    cushing_score = 0.0
    cushing_markers: list[str] = []
    if measurements.face_ratio > 0.92:
        cushing_score += 0.2
        cushing_markers.append("Round face (moon facies)")
    if measurements.skin_redness > 22:
        cushing_score += 0.15
        cushing_markers.append("Facial plethora/redness")
    if measurements.texture_roughness < 0.03 and measurements.skin_brightness > 140:
        cushing_score += 0.1
        cushing_markers.append("Thin, atrophic skin appearance")
    if cushing_score > 0.1:
        screenings.append(ConditionScreening(
            condition="Cushing syndrome",
            risk_score=round(min(cushing_score, 1.0), 3),
            facial_markers=cushing_markers,
            confidence=0.20,
            recommendation="Consider 24-hour urinary cortisol or overnight dexamethasone suppression test.",
        ))

    # --- Hypothyroidism screening ---
    # Puffy face, dry skin, lateral eyebrow thinning
    hypo_score = 0.0
    hypo_markers: list[str] = []
    if measurements.texture_roughness > 0.12:
        hypo_score += 0.15
        hypo_markers.append("Coarse/dry skin texture")
    if measurements.skin_yellowness > 140 and measurements.skin_brightness > 130:
        hypo_score += 0.15
        hypo_markers.append("Yellowish skin tinge (possible carotenemia)")
    if measurements.dark_circle_score > 0.3:
        hypo_score += 0.1
        hypo_markers.append("Periorbital puffiness/darkening")
    if hypo_score > 0.1:
        screenings.append(ConditionScreening(
            condition="Hypothyroidism",
            risk_score=round(min(hypo_score, 1.0), 3),
            facial_markers=hypo_markers,
            confidence=0.20,
            recommendation="Consider TSH screening if symptoms present (fatigue, weight gain, cold intolerance).",
        ))

    # --- Hyperthyroidism / Graves' screening ---
    hyper_score = 0.0
    hyper_markers: list[str] = []
    if measurements.texture_roughness < 0.02:
        hyper_score += 0.1
        hyper_markers.append("Very smooth skin texture (warm, moist skin)")
    if measurements.skin_redness > 20:
        hyper_score += 0.1
        hyper_markers.append("Facial flushing")
    if measurements.symmetry_score < 0.6:
        hyper_score += 0.05
        hyper_markers.append("Periorbital asymmetry")
    if hyper_score > 0.1:
        screenings.append(ConditionScreening(
            condition="Hyperthyroidism/Graves' disease",
            risk_score=round(min(hyper_score, 1.0), 3),
            facial_markers=hyper_markers,
            confidence=0.15,
            recommendation="Consider TSH, free T4 screening if symptoms present (weight loss, palpitations, heat intolerance).",
        ))

    # --- Liver dysfunction indicators ---
    liver_score = 0.0
    liver_markers: list[str] = []
    if measurements.skin_yellowness > 150:
        liver_score += 0.25
        liver_markers.append("Elevated skin yellowness (possible jaundice)")
    if measurements.skin_redness > 30 and measurements.uv_damage_score > 0.3:
        liver_score += 0.1
        liver_markers.append("Telangiectasia pattern (spider angiomata)")
    if liver_score > 0.1:
        screenings.append(ConditionScreening(
            condition="Hepatic dysfunction",
            risk_score=round(min(liver_score, 1.0), 3),
            facial_markers=liver_markers,
            confidence=0.20,
            recommendation="Consider liver function panel (bilirubin, ALT, AST, ALP) if jaundice suspected.",
        ))

    # --- Chronic fatigue / sleep disorder ---
    fatigue_score = 0.0
    fatigue_markers: list[str] = []
    if measurements.dark_circle_score > 0.4:
        fatigue_score += 0.25
        fatigue_markers.append("Significant periorbital darkening")
    if measurements.symmetry_score < 0.6:
        fatigue_score += 0.1
        fatigue_markers.append("Reduced facial symmetry (fatigue indicator)")
    if measurements.wrinkle_score > 0.03 and chrono_age < 35:
        fatigue_score += 0.15
        fatigue_markers.append("Premature aging signs in young adult")
    if fatigue_score > 0.1:
        screenings.append(ConditionScreening(
            condition="Chronic fatigue / sleep disorder",
            risk_score=round(min(fatigue_score, 1.0), 3),
            facial_markers=fatigue_markers,
            confidence=0.25,
            recommendation="Consider sleep quality assessment; rule out iron deficiency, thyroid dysfunction.",
        ))

    return screenings


# ---------------------------------------------------------------------------
# Ancestry-derived predictions (v2.0)
# ---------------------------------------------------------------------------

# mtDNA haplogroup priors by ancestry
_MTDNA_HAPLOGROUP_PRIORS: dict[str, dict[str, float]] = {
    "european": {"H": 0.44, "U": 0.12, "K": 0.07, "J": 0.09, "T": 0.08,
                 "V": 0.04, "I": 0.03, "W": 0.02, "X": 0.02, "Other": 0.09},
    "east_asian": {"D": 0.20, "B": 0.15, "A": 0.05, "F": 0.12, "M7": 0.10,
                   "C": 0.08, "G": 0.06, "N9": 0.05, "Other": 0.19},
    "african": {"L2": 0.25, "L3": 0.20, "L1": 0.15, "L0": 0.10, "L4": 0.03,
                "Other_L": 0.15, "Other": 0.12},
    "south_asian": {"M": 0.25, "U": 0.15, "R": 0.12, "W": 0.05, "N": 0.08,
                    "H": 0.05, "Other": 0.30},
    "latin_american": {"A": 0.20, "B": 0.18, "C": 0.15, "D": 0.15, "H": 0.10,
                       "L": 0.08, "Other": 0.14},
    "middle_eastern": {"H": 0.20, "J": 0.15, "T": 0.12, "U": 0.10, "K": 0.08,
                       "R0": 0.08, "L": 0.05, "Other": 0.22},
}


def _predict_ancestry_derived(
    ancestry: AncestryEstimate,
    variants: list[PredictedVariant],
) -> AncestryDerivedPredictions:
    """Compute ancestry-based genetic predictions.

    Uses estimated ancestry proportions to infer population-level genetic
    traits including mtDNA haplogroup, lactose tolerance, alcohol flush
    reaction, caffeine sensitivity, and population-specific disease risks.
    """
    # --- mtDNA haplogroup prediction ---
    hg_probs: dict[str, float] = {}
    pop_weights = {
        "european": ancestry.european,
        "east_asian": ancestry.east_asian,
        "african": ancestry.african,
        "south_asian": ancestry.south_asian,
        "latin_american": ancestry.latin_american,
        "middle_eastern": ancestry.middle_eastern,
    }
    for pop, weight in pop_weights.items():
        for hg, freq in _MTDNA_HAPLOGROUP_PRIORS.get(pop, {}).items():
            hg_probs[hg] = hg_probs.get(hg, 0.0) + weight * freq
    total_hg = sum(hg_probs.values())
    if total_hg > 0:
        for k in hg_probs:
            hg_probs[k] /= total_hg
    top_hg = max(hg_probs, key=hg_probs.get) if hg_probs else "Unknown"
    top_hg_conf = hg_probs.get(top_hg, 0.0)

    # --- Lactose tolerance from variant data ---
    variant_map = {v.rsid: v for v in variants}
    lct_v = variant_map.get("rs4988235")
    if lct_v:
        geno = lct_v.predicted_genotype.lower()
        if "homozygous variant" in geno or "elevated" in geno:
            lactose_prob = 0.95  # Likely tolerant (T/T)
        elif "heterozygous" in geno:
            lactose_prob = 0.85  # Likely tolerant (C/T)
        else:
            lactose_prob = 0.20  # Likely intolerant (C/C)
    else:
        # Ancestry-based estimate
        lactose_prob = (
            ancestry.european * 0.85 + ancestry.middle_eastern * 0.60 +
            ancestry.latin_american * 0.55 + ancestry.south_asian * 0.35 +
            ancestry.african * 0.20 + ancestry.east_asian * 0.05
        )

    # --- Alcohol flush probability ---
    aldh2_v = variant_map.get("rs671")
    if aldh2_v:
        geno = aldh2_v.predicted_genotype.lower()
        if "homozygous variant" in geno or "elevated" in geno:
            flush_prob = 0.95
        elif "heterozygous" in geno:
            flush_prob = 0.70
        else:
            flush_prob = 0.02
    else:
        flush_prob = ancestry.east_asian * 0.30

    # --- Caffeine sensitivity ---
    cyp1a2_v = variant_map.get("rs762551")
    if cyp1a2_v:
        geno = cyp1a2_v.predicted_genotype.lower()
        caffeine = "Slow" if ("heterozygous" in geno or "homozygous variant" in geno) else "Fast"
    else:
        caffeine = "Unknown"

    # --- Bitter taste sensitivity ---
    # TAS2R38 not directly in our SNP panel, so use ancestry proxy
    bitter = "Unknown"
    if ancestry.african > 0.5:
        bitter = "Taster (likely)"
    elif ancestry.european > 0.5:
        bitter = "Non-taster (likely)"

    # --- Population-specific disease risks ---
    pop_risks: list[str] = []
    if ancestry.south_asian > 0.3:
        pop_risks.append("South Asian ancestry — elevated Type 2 diabetes risk (2-4x baseline)")
        pop_risks.append("South Asian ancestry — elevated coronary artery disease risk")
    if ancestry.african > 0.3:
        pop_risks.append("African ancestry — sickle cell trait screening recommended")
        pop_risks.append("African ancestry — elevated prostate cancer risk (1.7x)")
        pop_risks.append("African ancestry — G6PD deficiency screening recommended before oxidant drugs")
    if ancestry.east_asian > 0.3:
        pop_risks.append("East Asian ancestry — HLA-B*15:02 screening before carbamazepine (Stevens-Johnson risk)")
        pop_risks.append("East Asian ancestry — ALDH2 deficiency common; alcohol cancer risk elevated")
    if ancestry.european > 0.4:
        pop_risks.append("European ancestry — HFE hemochromatosis screening may be warranted")
        pop_risks.append("European ancestry — celiac disease risk (HLA-DQ2/DQ8 prevalence ~35%)")
    if ancestry.middle_eastern > 0.3:
        pop_risks.append("Middle Eastern ancestry — beta-thalassemia carrier screening recommended")

    return AncestryDerivedPredictions(
        predicted_mtdna_haplogroup=f"Haplogroup {top_hg}",
        haplogroup_confidence=round(top_hg_conf, 3),
        lactose_tolerance_probability=round(lactose_prob, 3),
        alcohol_flush_probability=round(flush_prob, 3),
        caffeine_sensitivity=caffeine,
        bitter_taste_sensitivity=bitter,
        population_specific_risks=pop_risks,
    )


def analyze_face(
    image_path: str,
    chronological_age: int,
    sex: str = "unknown",
) -> FacialGenomicProfile:
    """Perform facial-genomic analysis on a photograph.

    Parameters
    ----------
    image_path : str
        Path to the face photograph.
    chronological_age : int
        The subject's chronological age in years.
    sex : str
        Biological sex (``"male"``, ``"female"``, or ``"unknown"``).

    Returns
    -------
    FacialGenomicProfile
        Comprehensive facial-genomic prediction including estimated
        biological age, telomere length, ancestry, and predicted variants.
    """
    warnings: list[str] = [
        "DISCLAIMER: Facial-genomic predictions are statistical estimates "
        "based on population-level correlations. They are NOT equivalent to "
        "actual DNA sequencing or genotyping. Results are indicative only.",
    ]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return FacialGenomicProfile(
            estimated_biological_age=chronological_age,
            estimated_telomere_length_kb=_telomere_from_age(chronological_age),
            telomere_percentile=50,
            measurements=FacialMeasurements(),
            ancestry=AncestryEstimate(),
            analysis_warnings=["Could not read image file."],
        )

    # Detect face — try multiple cascades with progressively relaxed
    # parameters for maximum recall on real-world photos.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = np.empty((0, 4), dtype=int)
    _cascades = [
        (cv2.data.haarcascades + "haarcascade_frontalface_default.xml", 1.1, 5, 60),
        (cv2.data.haarcascades + "haarcascade_frontalface_default.xml", 1.05, 3, 40),
        (cv2.data.haarcascades + "haarcascade_frontalface_alt.xml", 1.1, 3, 40),
        (cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml", 1.1, 3, 40),
        (cv2.data.haarcascades + "haarcascade_profileface.xml", 1.1, 3, 40),
    ]
    for cpath, sf, mn, ms in _cascades:
        cc = cv2.CascadeClassifier(cpath)
        faces = cc.detectMultiScale(gray_eq, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms))
        if len(faces) > 0:
            break

    if len(faces) == 0:
        warnings.append(
            "No face detected in image. Using image centre region for analysis. "
            "Results will be less accurate."
        )
        # Use central region as fallback
        h, w = img.shape[:2]
        cx, cy = w // 4, h // 4
        face_box = (cx, cy, w // 2, h // 2)
    else:
        # Use largest face
        face_box = max(faces, key=lambda f: f[2] * f[3])
        face_box = (int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3]))
        if len(faces) > 1:
            warnings.append(f"Multiple faces detected ({len(faces)}). Analysing the largest face.")

    # Extract measurements
    measurements = _extract_facial_measurements(img, face_box)

    # Estimate biological age
    bio_age = _estimate_biological_age(measurements, chronological_age, sex)

    # Estimate telomere length
    tl_kb = _telomere_from_age(bio_age, sex)
    tl_percentile = _telomere_percentile(tl_kb, chronological_age, sex)

    # Estimate ancestry
    ancestry = _estimate_ancestry(measurements)

    # Predict variants
    predicted_variants = _predict_variants_from_ancestry(ancestry, measurements)

    # Reconstruct DNA sequences around predicted variant loci
    reconstructed_dna = _reconstruct_dna(predicted_variants)

    # Predict phenotype traits
    eye_colour = _predict_eye_colour(measurements.skin_brightness, ancestry)
    hair_colour = _predict_hair_colour(measurements.skin_brightness, ancestry)
    skin_type = _classify_skin_type(measurements.skin_brightness)

    # Skin health score (0-100, higher = healthier)
    skin_health = 100.0
    skin_health -= measurements.wrinkle_score * 200
    skin_health -= measurements.dark_circle_score * 15
    skin_health -= measurements.uv_damage_score * 20
    skin_health -= max(measurements.skin_uniformity - 15, 0) * 0.5
    skin_health += (measurements.symmetry_score - 0.5) * 20
    skin_health = max(10.0, min(100.0, skin_health))

    # Oxidative stress score
    ox_stress = (
        measurements.uv_damage_score * 0.4
        + measurements.dark_circle_score * 0.2
        + measurements.texture_roughness * 0.2
        + min(measurements.skin_uniformity / 40.0, 1.0) * 0.2
    )
    ox_stress = min(1.0, max(0.0, ox_stress))

    # --- v2.0 expansion modules ---
    pharmacogenomics = _predict_pharmacogenomics(predicted_variants, ancestry)
    health_screening = _screen_facial_health(
        measurements, bio_age, chronological_age, sex, ancestry
    )
    dermatology = _analyze_dermatology(
        measurements, bio_age, chronological_age, sex, skin_type
    )
    condition_screenings = _screen_conditions(
        measurements, bio_age, chronological_age, sex
    )
    ancestry_derived = _predict_ancestry_derived(ancestry, predicted_variants)

    # Deterministic hash for reproducibility
    with open(image_path, "rb") as f:
        img_hash = hashlib.md5(f.read()[:4096]).hexdigest()  # noqa: S324

    logger.info(
        "Facial analysis complete for image %s: bio_age=%d, tl=%.2f kb, ancestry_top=%s (%.1f%%), "
        "pgx_predictions=%d, condition_screenings=%d",
        img_hash[:8],
        bio_age,
        tl_kb,
        max(ancestry.__dict__, key=lambda k: getattr(ancestry, k) if k != "confidence" else 0),
        max(
            ancestry.european,
            ancestry.east_asian,
            ancestry.south_asian,
            ancestry.african,
            ancestry.middle_eastern,
            ancestry.latin_american,
        )
        * 100,
        len(pharmacogenomics),
        len(condition_screenings),
    )

    return FacialGenomicProfile(
        estimated_biological_age=bio_age,
        estimated_telomere_length_kb=tl_kb,
        telomere_percentile=tl_percentile,
        measurements=measurements,
        ancestry=ancestry,
        predicted_variants=predicted_variants,
        reconstructed_dna=reconstructed_dna,
        skin_health_score=round(skin_health, 1),
        oxidative_stress_score=round(ox_stress, 3),
        predicted_eye_colour=eye_colour,
        predicted_hair_colour=hair_colour,
        predicted_skin_type=skin_type,
        analysis_warnings=warnings,
        pharmacogenomic_predictions=pharmacogenomics,
        health_screening=health_screening,
        dermatological_analysis=dermatology,
        condition_screenings=condition_screenings,
        ancestry_derived=ancestry_derived,
    )
