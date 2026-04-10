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
# Main analysis function
# ---------------------------------------------------------------------------


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

    # Deterministic hash for reproducibility
    with open(image_path, "rb") as f:
        img_hash = hashlib.md5(f.read()[:4096]).hexdigest()  # noqa: S324

    logger.info(
        "Facial analysis complete for image %s: bio_age=%d, tl=%.2f kb, ancestry_top=%s (%.1f%%)",
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
    )
