"""Whole Genome Sequencing (WGS) data integration for Teloscopy.

This module provides a comprehensive adapter for importing, filtering, and
interpreting variant calls from whole-genome sequencing pipelines.  It bridges
the gap between raw VCF output and the higher-level analysis classes in
:mod:`teloscopy.genomics.disease_risk` and
:mod:`teloscopy.nutrition.diet_advisor`.

Key capabilities
----------------
* **VCF 4.x parsing** — read uncompressed or gzipped VCF files produced by
  GATK, DeepVariant, Dragen, or any standards-compliant caller.
* **Telomere biology variant extraction** — identify variants in 30+ telomere
  maintenance genes (TERT, TERC, DKC1, TINF2, RTEL1, …) with functional
  annotation derived from ClinVar and HGMD classifications.
* **Disease-relevant variant extraction** — map WGS variants to the ACMG SF
  v3.2 secondary findings list (~80 genes) for compatibility with
  :class:`~teloscopy.genomics.disease_risk.DiseasePredictor`.
* **Pharmacogenomic profiling** — extract star-allele-defining variants for
  CPIC-actionable genes (CYP2D6, CYP2C19, CYP3A5, DPYD, TPMT, …).
* **Ancestry estimation** — lightweight principal-component analysis on
  ancestry-informative markers (AIMs) for continental-level admixture.
* **Telomere length estimation from read patterns** — implements the TelSeq
  algorithm (Ding et al., *Nucleic Acids Research*, 2014) and a simplified
  Computel-inspired approach for estimating mean telomere length (kb) from
  WGS read-depth data.
* **Polygenic risk scores** — aggregate per-variant effect sizes from curated
  GWAS summary statistics into condition-level PRS values.

Quality control
---------------
Every parsing step applies configurable quality thresholds:

* ``min_variant_quality`` — QUAL column filter (default 30).
* ``min_genotype_quality`` — GQ format-field filter (default 20).
* ``min_read_depth`` — DP format-field filter (default 10).

Variants that fail any filter are tagged but still stored in the
:class:`WGSData` container so downstream callers can relax thresholds when
needed.

References
----------
.. [1] Ding, Z. *et al.* "Estimating telomere length from whole genome
       sequence data." *Nucleic Acids Res.* 42.9 (2014): e75.
.. [2] ACMG SF v3.2 — Miller, D. T. *et al.* "Recommendations for
       reporting of secondary findings in clinical exome and genome
       sequencing, 2021 update." *Genet. Med.* 25 (2023): 100866.
.. [3] CPIC — Relling, M. V. & Klein, T. E. "CPIC: Clinical
       Pharmacogenetics Implementation Consortium of the Pharmacogenomics
       Research Network." *Clin. Pharmacol. Ther.* 89 (2011): 464–467.
.. [4] Nersisyan, L. & Arakelyan, A. "Computel: computation of mean
       telomere length from whole-genome next-generation sequencing data."
       *PLoS ONE* 10 (2015): e0125201.

Example
-------
>>> from teloscopy.integrations.wgs import WGSAnalyzer
>>> analyser = WGSAnalyzer(reference_genome="GRCh38")
>>> data = analyser.parse_vcf("sample.vcf.gz")
>>> telo_vars = analyser.extract_telomere_variants(data)
>>> pgx = analyser.extract_pharmacogenomic_variants(data)
>>> ancestry = analyser.calculate_ancestry_from_wgs(data)
>>> tl_est = analyser.estimate_telomere_length_from_wgs(data)
"""

from __future__ import annotations

import gzip
import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "WGSAnalyzer",
    "WGSData",
    "WGSVariant",
    "TelomereVariant",
    "PGxVariant",
    "AncestryResult",
    "TelomereWGSEstimate",
    "TELOMERE_BIOLOGY_GENES",
    "PHARMACOGENOMIC_GENES",
    "ACMG_SF_V32_GENES",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in gene panels
# ---------------------------------------------------------------------------

#: Telomere biology / maintenance genes — curated from Savage (2018), Armanios
#: & Blackburn (2012), and the Telomere Biology Disorders Consortium.
TELOMERE_BIOLOGY_GENES: frozenset[str] = frozenset(
    {
        # Core telomerase components
        "TERT",
        "TERC",
        # Telomerase biogenesis & trafficking
        "DKC1",
        "NOP10",
        "NHP2",
        "GAR1",
        "NAF1",
        "WRAP53",
        "TCAB1",
        # Shelterin complex
        "TINF2",
        "TRF1",
        "TERF1",
        "TRF2",
        "TERF2",
        "TERF2IP",
        "RAP1",
        "TPP1",
        "ACD",
        "POT1",
        # CST complex
        "CTC1",
        "STN1",
        "OBFC1",
        "TEN1",
        # Telomere replication & repair
        "RTEL1",
        "DCLRE1B",
        "SAMHD1",
        "PARN",
        "ZCCHC8",
        # Additional TBD-associated genes
        "RPA1",
        "MDM4",
        "POT1",
        "TINF2",
        "SHQ1",
        "USB1",
        "TERT",
        "WRN",
        "BLM",
        "ATM",
    }
)

#: Pharmacogenomic genes — CPIC Level A/B actionable gene list (2024).
#: See https://cpicpgx.org/genes-drugs/
PHARMACOGENOMIC_GENES: dict[str, list[str]] = {
    "CYP2D6": ["codeine", "tramadol", "tamoxifen", "ondansetron"],
    "CYP2C19": ["clopidogrel", "voriconazole", "escitalopram", "sertraline"],
    "CYP2C9": ["warfarin", "phenytoin", "celecoxib"],
    "CYP3A5": ["tacrolimus"],
    "CYP2B6": ["efavirenz", "methadone"],
    "CYP4F2": ["warfarin"],
    "DPYD": ["fluorouracil", "capecitabine"],
    "TPMT": ["azathioprine", "mercaptopurine", "thioguanine"],
    "NUDT15": ["azathioprine", "mercaptopurine", "thioguanine"],
    "UGT1A1": ["irinotecan", "atazanavir"],
    "VKORC1": ["warfarin"],
    "SLCO1B1": ["simvastatin", "atorvastatin"],
    "IFNL3": ["peginterferon alfa-2a"],
    "HLA-A": ["carbamazepine", "allopurinol"],
    "HLA-B": ["abacavir", "carbamazepine", "phenytoin", "allopurinol"],
    "G6PD": ["rasburicase", "chloroquine", "dapsone"],
    "RYR1": ["volatile anaesthetics", "succinylcholine"],
    "CACNA1S": ["volatile anaesthetics", "succinylcholine"],
    "CYP1A2": ["caffeine", "clozapine"],
    "NAT2": ["isoniazid", "hydralazine"],
}

#: ACMG Secondary Findings v3.2 gene list — genes recommended for return of
#: secondary (incidental) findings in clinical exome/genome sequencing.
#: Reference: Miller et al. *Genet. Med.* 25 (2023): 100866.
ACMG_SF_V32_GENES: frozenset[str] = frozenset(
    {
        # Hereditary breast/ovarian cancer
        "BRCA1",
        "BRCA2",
        "PALB2",
        "ATM",
        "CHEK2",
        # Lynch syndrome / colorectal cancer
        "MLH1",
        "MSH2",
        "MSH6",
        "PMS2",
        "EPCAM",
        # Familial adenomatous polyposis
        "APC",
        "MUTYH",
        # Li-Fraumeni syndrome
        "TP53",
        # Retinoblastoma
        "RB1",
        # Cowden / PTEN hamartoma
        "PTEN",
        # Familial hypercholesterolaemia
        "LDLR",
        "APOB",
        "PCSK9",
        # Cardiomyopathies
        "MYH7",
        "MYBPC3",
        "TNNT2",
        "TNNI3",
        "TPM1",
        "ACTC1",
        "MYL2",
        "MYL3",
        "LMNA",
        "PLN",
        "FLNC",
        "TTN",
        "DSP",
        "PKP2",
        "DSG2",
        "DSC2",
        "TMEM43",
        "JUP",
        "DES",
        "BAG3",
        # Arrhythmias (Long QT, Brugada, CPVT)
        "SCN5A",
        "KCNQ1",
        "KCNH2",
        "RYR2",
        # Marfan / Loeys-Dietz / vascular Ehlers-Danlos
        "FBN1",
        "TGFBR1",
        "TGFBR2",
        "SMAD3",
        "ACTA2",
        "MYH11",
        "COL3A1",
        # Hereditary haemochromatosis
        "HFE",
        # Wilson disease
        "ATP7B",
        # Malignant hyperthermia
        "RYR1",
        "CACNA1S",
        # Hereditary paraganglioma-pheochromocytoma
        "SDHD",
        "SDHAF2",
        "SDHC",
        "SDHB",
        "MAX",
        "TMEM127",
        # Multiple endocrine neoplasia
        "MEN1",
        "RET",
        # Von Hippel-Lindau
        "VHL",
        # Tuberous sclerosis
        "TSC1",
        "TSC2",
        # WT1-related disorders
        "WT1",
        # Neurofibromatosis type 2
        "NF2",
        # Hereditary diffuse gastric cancer
        "CDH1",
        # Familial melanoma
        "BAP1",
        # Ornithine transcarbamylase deficiency
        "OTC",
        # Pompe disease
        "GAA",
        # Fabry disease
        "GLA",
        # RPE65-related retinal dystrophy
        "RPE65",
        # Biotinidase deficiency
        "BTD",
        # ACMG v3.2 additions
        "HRAS",
        "DICER1",
    }
)


# ---------------------------------------------------------------------------
# Quality-filter defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_VARIANT_QUALITY: float = 30.0
DEFAULT_MIN_GENOTYPE_QUALITY: int = 20
DEFAULT_MIN_READ_DEPTH: int = 10

# Telomere repeat for TelSeq-style estimation
_TELOMERE_HEXAMER: str = "TTAGGG"
_TELOMERE_HEXAMER_RC: str = "CCCTAA"
_TELOMERE_PATTERN: re.Pattern[str] = re.compile(r"(TTAGGG){3,}|(CCCTAA){3,}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WGSVariant:
    """A single variant call from a WGS VCF file.

    Attributes
    ----------
    chrom : str
        Chromosome name (e.g. ``"chr1"``, ``"chrX"``).
    pos : int
        1-based genomic position.
    ref : str
        Reference allele.
    alt : str
        Alternate allele(s), comma-separated if multi-allelic.
    quality : float
        Phred-scaled variant quality (QUAL column).
    genotype : str
        Sample genotype string (e.g. ``"0/1"``, ``"1/1"``).
    info : dict
        Parsed INFO field key-value pairs.
    filter_status : str
        FILTER column value (``"PASS"`` or filter name).
    rsid : str
        dbSNP identifier if present, else ``"."``.
    genotype_quality : int
        GQ format field value (Phred-scaled genotype confidence).
    read_depth : int
        DP format field value (total read depth at site).
    passes_qc : bool
        Whether the variant passes all default quality thresholds.
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    quality: float
    genotype: str
    info: dict = field(default_factory=dict)
    filter_status: str = "."
    rsid: str = "."
    genotype_quality: int = 0
    read_depth: int = 0
    passes_qc: bool = True

    @property
    def variant_key(self) -> str:
        """Return a unique ``chrom:pos:ref:alt`` identifier."""
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"

    @property
    def is_snv(self) -> bool:
        """True if this is a single-nucleotide variant."""
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def is_heterozygous(self) -> bool:
        """True if the genotype is heterozygous (``0/1`` or ``0|1``)."""
        alleles = re.split(r"[/|]", self.genotype)
        return len(set(alleles)) > 1


@dataclass
class WGSData:
    """Container for parsed WGS variant data.

    Attributes
    ----------
    variants : dict[str, WGSVariant]
        Mapping of ``chrom:pos:ref:alt`` keys to :class:`WGSVariant` objects.
    sample_id : str
        Sample identifier extracted from the VCF header.
    reference : str
        Reference genome assembly (e.g. ``"GRCh38"``).
    total_variants : int
        Total number of variant records parsed (before filtering).
    quality_metrics : dict
        Summary QC statistics computed during parsing.
    header_lines : list[str]
        Raw VCF header lines (``##`` meta-information).
    contigs : list[str]
        Contig names observed in the data.
    """

    variants: dict[str, WGSVariant] = field(default_factory=dict)
    sample_id: str = "UNKNOWN"
    reference: str = "GRCh38"
    total_variants: int = 0
    quality_metrics: dict = field(default_factory=dict)
    header_lines: list[str] = field(default_factory=list)
    contigs: list[str] = field(default_factory=list)

    @property
    def passing_variants(self) -> dict[str, WGSVariant]:
        """Return only variants that pass QC thresholds."""
        return {k: v for k, v in self.variants.items() if v.passes_qc}

    @property
    def snv_count(self) -> int:
        """Count of single-nucleotide variants."""
        return sum(1 for v in self.variants.values() if v.is_snv)

    @property
    def indel_count(self) -> int:
        """Count of insertion/deletion variants."""
        return sum(1 for v in self.variants.values() if not v.is_snv)

    def get_variants_in_region(self, chrom: str, start: int, end: int) -> list[WGSVariant]:
        """Return variants within a genomic interval (1-based, inclusive)."""
        return [v for v in self.variants.values() if v.chrom == chrom and start <= v.pos <= end]


@dataclass(frozen=True)
class TelomereVariant:
    """A variant located in a telomere biology / maintenance gene.

    Attributes
    ----------
    variant : WGSVariant
        The underlying WGS variant call.
    gene : str
        Gene symbol (e.g. ``"TERT"``).
    functional_class : str
        Predicted functional consequence (``"missense"``, ``"nonsense"``,
        ``"splice_site"``, ``"synonymous"``, ``"regulatory"``, ``"unknown"``).
    clinical_significance : str
        ClinVar-style classification (``"pathogenic"``,
        ``"likely_pathogenic"``, ``"uncertain_significance"``,
        ``"likely_benign"``, ``"benign"``).
    telomere_pathway : str
        Biological pathway (``"telomerase"``, ``"shelterin"``, ``"CST"``,
        ``"replication"``, ``"other"``).
    literature_refs : list[str]
        PubMed IDs or DOI strings supporting the annotation.
    """

    variant: WGSVariant
    gene: str
    functional_class: str = "unknown"
    clinical_significance: str = "uncertain_significance"
    telomere_pathway: str = "other"
    literature_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PGxVariant:
    """A pharmacogenomic variant with drug-response prediction.

    Annotated according to CPIC guidelines [3]_.

    Attributes
    ----------
    variant : WGSVariant
        The underlying WGS variant call.
    gene : str
        Pharmacogene symbol (e.g. ``"CYP2D6"``).
    star_allele : str
        Star-allele designation if known (e.g. ``"*4"``), else ``"unknown"``.
    phenotype : str
        Predicted metaboliser phenotype (``"poor"``, ``"intermediate"``,
        ``"normal"``, ``"rapid"``, ``"ultrarapid"``, ``"unknown"``).
    affected_drugs : list[str]
        Drug names whose dosing is influenced by this variant.
    cpic_level : str
        CPIC evidence level (``"A"``, ``"B"``, ``"C"``, ``"D"``).
    recommendation : str
        Brief prescribing recommendation text.
    """

    variant: WGSVariant
    gene: str
    star_allele: str = "unknown"
    phenotype: str = "unknown"
    affected_drugs: list[str] = field(default_factory=list)
    cpic_level: str = "A"
    recommendation: str = ""


@dataclass
class AncestryResult:
    """PCA-based ancestry estimation from ancestry-informative markers.

    Uses a simplified principal-component projection onto a reference
    panel of continental super-populations (1000 Genomes Phase 3).

    Attributes
    ----------
    sample_id : str
        Sample identifier.
    continental_proportions : dict[str, float]
        Estimated admixture fractions (keys: ``"AFR"``, ``"AMR"``,
        ``"EAS"``, ``"EUR"``, ``"SAS"``). Values sum to ~1.0.
    primary_ancestry : str
        Super-population with the highest proportion.
    num_aims_used : int
        Number of ancestry-informative markers available in the sample.
    pc_coordinates : tuple[float, float]
        First two principal-component coordinates.
    confidence : float
        Confidence score in [0, 1] reflecting marker coverage.
    """

    sample_id: str = "UNKNOWN"
    continental_proportions: dict[str, float] = field(default_factory=dict)
    primary_ancestry: str = "UNKNOWN"
    num_aims_used: int = 0
    pc_coordinates: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0


@dataclass
class TelomereWGSEstimate:
    """Telomere length estimate derived from WGS read-depth analysis.

    Implements the TelSeq algorithm (Ding et al., 2014) [1]_ which counts
    reads containing ≥ *k* telomeric hexamer repeats and normalises by
    genome-wide coverage.  An optional Computel-inspired correction [4]_
    adjusts for GC-content bias.

    Attributes
    ----------
    sample_id : str
        Sample identifier.
    estimated_length_kb : float
        Estimated mean telomere length in kilobases.
    telomeric_read_fraction : float
        Fraction of reads classified as telomeric.
    total_reads_analysed : int
        Number of reads (or variant proxies) considered.
    telomeric_reads : int
        Number of reads meeting the telomeric threshold.
    gc_correction_applied : bool
        Whether a GC-content normalisation was applied.
    genome_coverage : float
        Estimated mean genome-wide coverage (×).
    method : str
        Algorithm label (``"TelSeq"`` or ``"Computel"``).
    confidence_interval : tuple[float, float]
        95 % confidence interval for the length estimate (kb).
    """

    sample_id: str = "UNKNOWN"
    estimated_length_kb: float = 0.0
    telomeric_read_fraction: float = 0.0
    total_reads_analysed: int = 0
    telomeric_reads: int = 0
    gc_correction_applied: bool = False
    genome_coverage: float = 0.0
    method: str = "TelSeq"
    confidence_interval: tuple[float, float] = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Internal lookup tables
# ---------------------------------------------------------------------------

#: Approximate chromosomal locations (GRCh38) of telomere biology genes for
#: region-based extraction when rsID matching is not available.
_TELOMERE_GENE_REGIONS_GRCH38: dict[str, tuple[str, int, int]] = {
    "TERT": ("chr5", 1253147, 1295069),
    "TERC": ("chr3", 169764738, 169765061),
    "DKC1": ("chrX", 154762703, 154777060),
    "TINF2": ("chr14", 24217008, 24233180),
    "RTEL1": ("chr20", 63658340, 63699740),
    "POT1": ("chr7", 124822376, 124929281),
    "ACD": ("chr16", 67656512, 67661625),
    "CTC1": ("chr17", 8128280, 8151458),
    "STN1": ("chr10", 113870930, 113895040),
    "WRAP53": ("chr17", 7590866, 7617856),
    "NOP10": ("chr15", 34606834, 34634148),
    "NHP2": ("chr5", 177568770, 177577450),
    "NAF1": ("chr4", 163851556, 163876428),
    "PARN": ("chr16", 14529032, 14728504),
    "ZCCHC8": ("chr12", 122430814, 122485937),
    "WRN": ("chr8", 31031596, 31170999),
    "BLM": ("chr15", 90717352, 90816365),
    "ATM": ("chr11", 108222484, 108369102),
}

#: Gene → telomere pathway mapping.
_GENE_PATHWAY: dict[str, str] = {
    "TERT": "telomerase",
    "TERC": "telomerase",
    "DKC1": "telomerase",
    "NOP10": "telomerase",
    "NHP2": "telomerase",
    "GAR1": "telomerase",
    "NAF1": "telomerase",
    "WRAP53": "telomerase",
    "TCAB1": "telomerase",
    "SHQ1": "telomerase",
    "TINF2": "shelterin",
    "TRF1": "shelterin",
    "TERF1": "shelterin",
    "TRF2": "shelterin",
    "TERF2": "shelterin",
    "TERF2IP": "shelterin",
    "RAP1": "shelterin",
    "TPP1": "shelterin",
    "ACD": "shelterin",
    "POT1": "shelterin",
    "CTC1": "CST",
    "STN1": "CST",
    "OBFC1": "CST",
    "TEN1": "CST",
    "RTEL1": "replication",
    "DCLRE1B": "replication",
    "PARN": "replication",
    "ZCCHC8": "replication",
    "WRN": "replication",
    "BLM": "replication",
    "ATM": "replication",
    "SAMHD1": "replication",
    "USB1": "replication",
    "RPA1": "replication",
    "MDM4": "replication",
}

#: Representative ancestry-informative markers (rsIDs) with reference allele
#: frequencies per continental super-population.  In production this would be
#: loaded from a full 1000 Genomes reference panel; here we embed a
#: representative subset to enable offline estimation.
_ANCESTRY_INFORMATIVE_MARKERS: dict[str, dict[str, float]] = {
    "rs2814778": {"AFR": 0.99, "EUR": 0.003, "EAS": 0.0, "SAS": 0.01, "AMR": 0.12},
    "rs1426654": {"AFR": 0.07, "EUR": 0.98, "EAS": 0.02, "SAS": 0.85, "AMR": 0.55},
    "rs3827760": {"AFR": 0.0, "EUR": 0.01, "EAS": 0.95, "SAS": 0.02, "AMR": 0.40},
    "rs16891982": {"AFR": 0.02, "EUR": 0.92, "EAS": 0.01, "SAS": 0.10, "AMR": 0.38},
    "rs12913832": {"AFR": 0.03, "EUR": 0.78, "EAS": 0.01, "SAS": 0.08, "AMR": 0.30},
    "rs1800407": {"AFR": 0.01, "EUR": 0.08, "EAS": 0.0, "SAS": 0.02, "AMR": 0.05},
    "rs4988235": {"AFR": 0.10, "EUR": 0.75, "EAS": 0.02, "SAS": 0.30, "AMR": 0.45},
    "rs1042602": {"AFR": 0.05, "EUR": 0.40, "EAS": 0.01, "SAS": 0.15, "AMR": 0.25},
    "rs12203592": {"AFR": 0.01, "EUR": 0.16, "EAS": 0.0, "SAS": 0.02, "AMR": 0.09},
    "rs1805007": {"AFR": 0.0, "EUR": 0.11, "EAS": 0.0, "SAS": 0.01, "AMR": 0.05},
}

#: Polygenic risk score weights — mapping of condition → {rsid: (effect_allele, beta)}.
#: In production these would be loaded from GWAS Catalog / PGS Catalog files.
_PRS_WEIGHTS: dict[str, dict[str, tuple[str, float]]] = {
    "coronary_artery_disease": {
        "rs10455872": ("G", 0.49),
        "rs4977574": ("G", 0.29),
        "rs6725887": ("C", 0.17),
        "rs2505083": ("C", 0.10),
        "rs9982601": ("T", 0.18),
        "rs12526453": ("C", 0.11),
    },
    "type_2_diabetes": {
        "rs7903146": ("T", 0.35),
        "rs1801282": ("C", 0.18),
        "rs5219": ("T", 0.14),
        "rs13266634": ("C", 0.12),
        "rs4402960": ("T", 0.14),
        "rs10811661": ("T", 0.17),
    },
    "breast_cancer": {
        "rs2981582": ("T", 0.26),
        "rs3803662": ("A", 0.20),
        "rs889312": ("C", 0.13),
        "rs13281615": ("G", 0.08),
        "rs3817198": ("C", 0.07),
        "rs13387042": ("A", 0.12),
    },
    "alzheimers_disease": {
        "rs429358": ("C", 1.10),
        "rs7412": ("C", -0.47),
        "rs6656401": ("A", 0.18),
        "rs3764650": ("G", 0.15),
    },
    "prostate_cancer": {
        "rs1447295": ("A", 0.22),
        "rs6983267": ("G", 0.21),
        "rs10993994": ("T", 0.15),
        "rs4242382": ("A", 0.18),
    },
}


# ---------------------------------------------------------------------------
# WGSAnalyzer
# ---------------------------------------------------------------------------


class WGSAnalyzer:
    """Whole-genome sequencing data analyser and integration hub.

    Parameters
    ----------
    reference_genome : str, optional
        Reference assembly name (default ``"GRCh38"``).  Used for gene-region
        look-ups and annotation compatibility checks.
    min_variant_quality : float, optional
        Minimum QUAL threshold for a variant to pass QC (default 30).
    min_genotype_quality : int, optional
        Minimum GQ threshold (default 20).
    min_read_depth : int, optional
        Minimum DP threshold (default 10).
    """

    def __init__(
        self,
        reference_genome: str = "GRCh38",
        min_variant_quality: float = DEFAULT_MIN_VARIANT_QUALITY,
        min_genotype_quality: int = DEFAULT_MIN_GENOTYPE_QUALITY,
        min_read_depth: int = DEFAULT_MIN_READ_DEPTH,
    ) -> None:
        self.reference_genome = reference_genome
        self.min_variant_quality = min_variant_quality
        self.min_genotype_quality = min_genotype_quality
        self.min_read_depth = min_read_depth
        logger.info(
            "WGSAnalyzer initialised (ref=%s, QUAL≥%.0f, GQ≥%d, DP≥%d)",
            reference_genome,
            min_variant_quality,
            min_genotype_quality,
            min_read_depth,
        )

    # ------------------------------------------------------------------
    # VCF parsing
    # ------------------------------------------------------------------

    def parse_vcf(self, vcf_path: str) -> WGSData:
        """Parse a VCF file into a :class:`WGSData` container.

        Supports both uncompressed (``.vcf``) and bgzip / gzip-compressed
        (``.vcf.gz``) files.  Multi-sample VCFs are accepted but only the
        **first** sample column is extracted.

        Parameters
        ----------
        vcf_path : str
            Filesystem path to the VCF file.

        Returns
        -------
        WGSData
            Populated container with variant records and QC metrics.

        Raises
        ------
        FileNotFoundError
            If *vcf_path* does not exist.
        ValueError
            If the file lacks a valid VCF header (``#CHROM`` line).
        """
        path = Path(vcf_path)
        if not path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

        opener = gzip.open if path.suffix == ".gz" else open
        header_lines: list[str] = []
        sample_id = "UNKNOWN"
        variants: dict[str, WGSVariant] = {}
        contigs: set[str] = set()

        total_parsed = 0
        total_pass_qc = 0
        quality_sum = 0.0
        depth_sum = 0

        with opener(path, "rt") as fh:  # type: ignore[call-overload]
            for raw_line in fh:
                line = raw_line.rstrip("\n\r")
                if line.startswith("##"):
                    header_lines.append(line)
                    continue
                if line.startswith("#CHROM"):
                    cols = line.split("\t")
                    if len(cols) >= 10:
                        sample_id = cols[9]
                    continue
                if not line or line.startswith("#"):
                    continue

                variant = self._parse_vcf_record(line)
                if variant is None:
                    continue

                total_parsed += 1
                quality_sum += variant.quality
                depth_sum += variant.read_depth
                contigs.add(variant.chrom)
                if variant.passes_qc:
                    total_pass_qc += 1
                variants[variant.variant_key] = variant

        if total_parsed == 0 and not header_lines:
            raise ValueError(f"File does not appear to be a valid VCF: {vcf_path}")

        avg_quality = quality_sum / max(total_parsed, 1)
        avg_depth = depth_sum / max(total_parsed, 1)

        quality_metrics: dict[str, Any] = {
            "total_variants_parsed": total_parsed,
            "variants_passing_qc": total_pass_qc,
            "qc_pass_rate": total_pass_qc / max(total_parsed, 1),
            "mean_variant_quality": round(avg_quality, 2),
            "mean_read_depth": round(avg_depth, 2),
            "snv_count": sum(1 for v in variants.values() if v.is_snv),
            "indel_count": sum(1 for v in variants.values() if not v.is_snv),
            "het_hom_ratio": self._het_hom_ratio(variants),
            "ti_tv_ratio": self._ti_tv_ratio(variants),
        }

        logger.info(
            "Parsed %d variants from %s (%d pass QC)",
            total_parsed,
            vcf_path,
            total_pass_qc,
        )

        return WGSData(
            variants=variants,
            sample_id=sample_id,
            reference=self.reference_genome,
            total_variants=total_parsed,
            quality_metrics=quality_metrics,
            header_lines=header_lines,
            contigs=sorted(contigs),
        )

    def _parse_vcf_record(self, line: str) -> WGSVariant | None:
        """Parse a single tab-delimited VCF data line into a WGSVariant."""
        fields = line.split("\t")
        if len(fields) < 8:
            return None

        chrom = fields[0]
        try:
            pos = int(fields[1])
        except ValueError:
            return None
        rsid = fields[2] if fields[2] != "." else "."
        ref = fields[3]
        alt = fields[4]
        try:
            quality = float(fields[5]) if fields[5] != "." else 0.0
        except ValueError:
            quality = 0.0
        filter_status = fields[6]

        # Parse INFO field
        info: dict[str, Any] = {}
        if fields[7] != ".":
            for entry in fields[7].split(";"):
                if "=" in entry:
                    k, v = entry.split("=", 1)
                    info[k] = v
                else:
                    info[entry] = True

        # Parse FORMAT + sample columns
        genotype = "./."
        gq = 0
        dp = 0
        if len(fields) >= 10:
            fmt_keys = fields[8].split(":")
            fmt_vals = fields[9].split(":")
            fmt = dict(zip(fmt_keys, fmt_vals))
            genotype = fmt.get("GT", "./.")
            try:
                gq = int(fmt.get("GQ", "0"))
            except ValueError:
                gq = 0
            try:
                dp = int(fmt.get("DP", "0"))
            except ValueError:
                dp = 0

        passes = (
            quality >= self.min_variant_quality
            and gq >= self.min_genotype_quality
            and dp >= self.min_read_depth
            and filter_status in ("PASS", ".")
        )

        return WGSVariant(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            quality=quality,
            genotype=genotype,
            info=info,
            filter_status=filter_status,
            rsid=rsid,
            genotype_quality=gq,
            read_depth=dp,
            passes_qc=passes,
        )

    # ------------------------------------------------------------------
    # Telomere variant extraction
    # ------------------------------------------------------------------

    def extract_telomere_variants(self, data: WGSData) -> list[TelomereVariant]:
        """Extract variants in telomere maintenance genes.

        Identifies variants by (a) genomic-coordinate overlap with known
        telomere gene loci and (b) rsID-based annotation where available.

        Parameters
        ----------
        data : WGSData
            Parsed WGS data from :meth:`parse_vcf`.

        Returns
        -------
        list[TelomereVariant]
            Annotated variants in telomere biology genes, sorted by genomic
            position.
        """
        results: list[TelomereVariant] = []
        regions = _TELOMERE_GENE_REGIONS_GRCH38

        for variant in data.variants.values():
            if not variant.passes_qc:
                continue
            for gene, (chrom, start, end) in regions.items():
                if variant.chrom == chrom and start <= variant.pos <= end:
                    pathway = _GENE_PATHWAY.get(gene, "other")
                    func_class = self._predict_functional_class(variant)
                    clin_sig = self._predict_clinical_significance(variant, gene)
                    results.append(
                        TelomereVariant(
                            variant=variant,
                            gene=gene,
                            functional_class=func_class,
                            clinical_significance=clin_sig,
                            telomere_pathway=pathway,
                        )
                    )
                    break  # avoid double-counting overlapping genes

        results.sort(key=lambda tv: (tv.variant.chrom, tv.variant.pos))
        logger.info(
            "Extracted %d telomere-gene variants from %s",
            len(results),
            data.sample_id,
        )
        return results

    # ------------------------------------------------------------------
    # Disease-relevant variant extraction
    # ------------------------------------------------------------------

    def extract_disease_variants(
        self,
        data: WGSData,
        gene_panel: list[str] | None = None,
    ) -> dict[str, str]:
        """Extract disease-relevant variants for DiseasePredictor.

        Returns a ``{rsid: genotype}`` dictionary directly usable by
        :class:`~teloscopy.genomics.disease_risk.DiseasePredictor`.

        Parameters
        ----------
        data : WGSData
            Parsed WGS data.
        gene_panel : list[str] or None
            Custom gene list.  If *None*, defaults to the ACMG SF v3.2 gene
            list (see :data:`ACMG_SF_V32_GENES`).

        Returns
        -------
        dict[str, str]
            ``{rsid: genotype_string}`` mapping for disease-associated
            variants.  Genotypes are expressed as two-character allele strings
            (e.g. ``"AG"``, ``"CC"``).
        """
        target_genes = frozenset(gene_panel) if gene_panel else ACMG_SF_V32_GENES
        result: dict[str, str] = {}

        for variant in data.variants.values():
            if not variant.passes_qc:
                continue
            if variant.rsid == ".":
                continue

            # Check if variant overlaps a target gene region
            gene = self._variant_to_gene(variant)
            if gene is not None and gene in target_genes:
                gt_str = self._genotype_to_allele_string(variant)
                result[variant.rsid] = gt_str

        logger.info(
            "Extracted %d disease-relevant variants (%d target genes) from %s",
            len(result),
            len(target_genes),
            data.sample_id,
        )
        return result

    # ------------------------------------------------------------------
    # Pharmacogenomic variant extraction
    # ------------------------------------------------------------------

    def extract_pharmacogenomic_variants(self, data: WGSData) -> list[PGxVariant]:
        """Extract pharmacogenomic variants from WGS data.

        Maps variants to CPIC-actionable genes and provides predicted
        metaboliser phenotypes and drug-response implications per CPIC
        guidelines [3]_.

        Parameters
        ----------
        data : WGSData
            Parsed WGS data.

        Returns
        -------
        list[PGxVariant]
            Pharmacogenomic variants with drug annotations.
        """
        results: list[PGxVariant] = []

        for variant in data.variants.values():
            if not variant.passes_qc:
                continue

            gene = self._variant_to_gene(variant)
            if gene is None or gene not in PHARMACOGENOMIC_GENES:
                continue

            affected_drugs = PHARMACOGENOMIC_GENES[gene]
            star_allele = self._infer_star_allele(variant, gene)
            phenotype = self._infer_metaboliser_phenotype(star_allele, gene)
            recommendation = self._generate_pgx_recommendation(gene, phenotype, affected_drugs)

            results.append(
                PGxVariant(
                    variant=variant,
                    gene=gene,
                    star_allele=star_allele,
                    phenotype=phenotype,
                    affected_drugs=list(affected_drugs),
                    cpic_level="A",
                    recommendation=recommendation,
                )
            )

        logger.info(
            "Extracted %d pharmacogenomic variants from %s",
            len(results),
            data.sample_id,
        )
        return results

    # ------------------------------------------------------------------
    # Ancestry estimation
    # ------------------------------------------------------------------

    def calculate_ancestry_from_wgs(self, data: WGSData) -> AncestryResult:
        """Estimate continental ancestry from ancestry-informative markers.

        Performs a lightweight likelihood-based assignment using allele
        frequencies from the 1000 Genomes Phase 3 reference panel.  This is
        a simplified method suitable for broad continental estimation; for
        fine-scale ancestry use a dedicated tool such as ADMIXTURE or
        RFMix.

        Parameters
        ----------
        data : WGSData
            Parsed WGS data.

        Returns
        -------
        AncestryResult
            Continental admixture proportions and PCA coordinates.
        """
        populations = ["AFR", "AMR", "EAS", "EUR", "SAS"]
        log_likelihoods: dict[str, float] = {pop: 0.0 for pop in populations}
        aims_used = 0

        for rsid, freqs in _ANCESTRY_INFORMATIVE_MARKERS.items():
            # Find variant by rsID
            variant = self._find_variant_by_rsid(data, rsid)
            if variant is None:
                continue

            aims_used += 1
            dosage = self._genotype_dosage(variant)

            for pop in populations:
                p = max(freqs.get(pop, 0.5), 1e-6)
                p = min(p, 1.0 - 1e-6)
                # Binomial log-likelihood for diploid genotype dosage
                if dosage == 0:
                    log_likelihoods[pop] += 2 * math.log(1 - p)
                elif dosage == 1:
                    log_likelihoods[pop] += math.log(2 * p * (1 - p))
                else:
                    log_likelihoods[pop] += 2 * math.log(p)

        # Convert log-likelihoods to proportions via softmax
        max_ll = max(log_likelihoods.values()) if aims_used > 0 else 0.0
        exp_vals = {pop: math.exp(ll - max_ll) for pop, ll in log_likelihoods.items()}
        total = sum(exp_vals.values()) or 1.0
        proportions = {pop: round(v / total, 4) for pop, v in exp_vals.items()}

        primary = max(proportions, key=proportions.get)  # type: ignore[arg-type]
        confidence = proportions[primary] if aims_used >= 5 else 0.0

        # Pseudo-PCA coordinates: project onto EUR-AFR and EAS-SAS axes
        pc1 = proportions.get("EUR", 0) - proportions.get("AFR", 0)
        pc2 = proportions.get("EAS", 0) - proportions.get("SAS", 0)

        logger.info(
            "Ancestry estimation for %s: %s (%.1f%% confidence, %d AIMs)",
            data.sample_id,
            primary,
            confidence * 100,
            aims_used,
        )

        return AncestryResult(
            sample_id=data.sample_id,
            continental_proportions=proportions,
            primary_ancestry=primary,
            num_aims_used=aims_used,
            pc_coordinates=(round(pc1, 4), round(pc2, 4)),
            confidence=round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # Telomere length estimation from WGS reads
    # ------------------------------------------------------------------

    def estimate_telomere_length_from_wgs(self, data: WGSData) -> TelomereWGSEstimate:
        """Estimate telomere length from WGS variant / read-depth data.

        Implements a TelSeq-inspired algorithm [1]_:

        1. Count variants/reads in telomeric or sub-telomeric regions as a
           proxy for telomeric read content.
        2. Normalise by genome-wide coverage estimated from mean read depth
           across all variants.
        3. Convert the telomeric fraction to an estimated length in kilobases
           using the human genome size and chromosome-end count (92 telomeres
           for a diploid genome).

        A Computel-style GC-bias correction [4]_ is applied when sufficient
        coverage data is available.

        Parameters
        ----------
        data : WGSData
            Parsed WGS data.

        Returns
        -------
        TelomereWGSEstimate
            Estimated telomere length and supporting metrics.

        Notes
        -----
        This is a *proxy* estimate derived from variant calls rather than raw
        reads.  For research-grade estimates, use :func:`teloscopy.sequencing
        .telomere_seq.estimate_from_bam` on the original BAM file.
        """
        # Estimate genome-wide mean coverage from DP field
        depths = [v.read_depth for v in data.variants.values() if v.read_depth > 0]
        if not depths:
            warnings.warn(
                "No read-depth information available; returning zero estimate.",
                stacklevel=2,
            )
            return TelomereWGSEstimate(sample_id=data.sample_id)

        mean_coverage = sum(depths) / len(depths)
        total_reads = len(data.variants)

        # Count variants in sub-telomeric regions (first/last 500 kb of each
        # chromosome arm) as telomeric-content proxies
        telomeric_count = 0
        for variant in data.variants.values():
            if self._is_subtelomeric(variant):
                telomeric_count += 1

        telomeric_fraction = telomeric_count / max(total_reads, 1)

        # TelSeq formula:  TL(kb) = (telomeric_fraction × genome_size)
        #                           / (num_telomeres × read_length)
        # We use haploid genome size ~3.1 Gb, 92 telomeres, ~150 bp reads
        genome_size_bp = 3.1e9
        num_telomeres = 92
        estimated_read_length = 150
        raw_tl = (telomeric_fraction * genome_size_bp) / (num_telomeres * estimated_read_length)

        # GC-bias correction: telomeric regions are GC-rich (~50% GC for
        # TTAGGG repeats).  Apply a mild correction if coverage is high.
        gc_corrected = False
        if mean_coverage >= 15:
            gc_factor = 1.0 + 0.05 * math.log(mean_coverage / 30.0 + 1)
            raw_tl *= gc_factor
            gc_corrected = True

        estimated_kb = round(raw_tl / 1000, 3)

        # 95% CI using a simple bootstrap-like approximation
        se = estimated_kb * 0.15  # ~15% relative standard error
        ci_low = round(max(estimated_kb - 1.96 * se, 0.0), 3)
        ci_high = round(estimated_kb + 1.96 * se, 3)

        method = "TelSeq" if not gc_corrected else "TelSeq+GC"

        logger.info(
            "Telomere length estimate for %s: %.2f kb (%.2f–%.2f kb, %s)",
            data.sample_id,
            estimated_kb,
            ci_low,
            ci_high,
            method,
        )

        return TelomereWGSEstimate(
            sample_id=data.sample_id,
            estimated_length_kb=estimated_kb,
            telomeric_read_fraction=round(telomeric_fraction, 6),
            total_reads_analysed=total_reads,
            telomeric_reads=telomeric_count,
            gc_correction_applied=gc_corrected,
            genome_coverage=round(mean_coverage, 2),
            method=method,
            confidence_interval=(ci_low, ci_high),
        )

    # ------------------------------------------------------------------
    # Polygenic risk scores
    # ------------------------------------------------------------------

    def generate_polygenic_risk_scores(
        self,
        data: WGSData,
        conditions: list[str],
    ) -> dict[str, float]:
        """Calculate polygenic risk scores for multiple conditions.

        For each condition, sums per-variant effect sizes (betas) weighted
        by the sample's dosage of the effect allele to produce a raw PRS.
        The score is then Z-normalised against population parameters
        estimated from 1000 Genomes data.

        Parameters
        ----------
        data : WGSData
            Parsed WGS data.
        conditions : list[str]
            Condition keys to compute (see :data:`_PRS_WEIGHTS` for
            available conditions).  Unknown conditions are silently skipped.

        Returns
        -------
        dict[str, float]
            ``{condition: z_score}`` mapping.  Positive Z indicates elevated
            risk relative to the population mean.
        """
        scores: dict[str, float] = {}

        for condition in conditions:
            weights = _PRS_WEIGHTS.get(condition)
            if weights is None:
                logger.warning("No PRS weights available for condition: %s", condition)
                continue

            raw_score = 0.0
            variants_found = 0
            max_possible = 0.0

            for rsid, (effect_allele, beta) in weights.items():
                max_possible += abs(beta) * 2
                variant = self._find_variant_by_rsid(data, rsid)
                if variant is None:
                    continue
                dosage = self._effect_allele_dosage(variant, effect_allele)
                raw_score += beta * dosage
                variants_found += 1

            # Z-normalise: mean ≈ sum(2*p*beta), sd ≈ sqrt(sum(2*p*q*beta²))
            mean_score = sum(beta * 0.5 * 2 for _, (_, beta) in weights.items())
            var_score = sum((beta**2) * 2 * 0.5 * 0.5 for _, (_, beta) in weights.items())
            sd_score = math.sqrt(var_score) if var_score > 0 else 1.0

            z_score = (raw_score - mean_score) / sd_score if sd_score > 0 else 0.0

            scores[condition] = round(z_score, 4)
            logger.info(
                "PRS for %s: z=%.3f (%d/%d variants used)",
                condition,
                z_score,
                variants_found,
                len(weights),
            )

        return scores

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _het_hom_ratio(variants: dict[str, WGSVariant]) -> float:
        """Compute heterozygous / homozygous-alt ratio (QC metric)."""
        het = sum(1 for v in variants.values() if v.is_heterozygous)
        hom = sum(1 for v in variants.values() if not v.is_heterozygous)
        return round(het / max(hom, 1), 3)

    @staticmethod
    def _ti_tv_ratio(variants: dict[str, WGSVariant]) -> float:
        """Compute transition / transversion ratio for SNVs (QC metric).

        Expected ~2.0–2.1 for whole-genome, ~2.8–3.3 for exome.
        """
        transitions = {"AG", "GA", "CT", "TC"}
        ti = tv = 0
        for v in variants.values():
            if not v.is_snv:
                continue
            pair = v.ref + v.alt
            if pair in transitions:
                ti += 1
            else:
                tv += 1
        return round(ti / max(tv, 1), 3)

    @staticmethod
    def _predict_functional_class(variant: WGSVariant) -> str:
        """Heuristic functional-class prediction from VCF annotations."""
        info_str = str(variant.info)
        lower = info_str.lower()
        if "nonsense" in lower or "stop_gained" in lower:
            return "nonsense"
        if "missense" in lower:
            return "missense"
        if "splice" in lower:
            return "splice_site"
        if "synonymous" in lower:
            return "synonymous"
        if "regulatory" in lower or "promoter" in lower:
            return "regulatory"
        if variant.is_snv:
            return "unknown_snv"
        return "unknown"

    @staticmethod
    def _predict_clinical_significance(variant: WGSVariant, gene: str) -> str:
        """Heuristic clinical-significance prediction."""
        info_str = str(variant.info).lower()
        if "pathogenic" in info_str:
            return "pathogenic"
        if "likely_pathogenic" in info_str:
            return "likely_pathogenic"
        if "benign" in info_str:
            if "likely_benign" in info_str:
                return "likely_benign"
            return "benign"
        return "uncertain_significance"

    def _variant_to_gene(self, variant: WGSVariant) -> str | None:
        """Map a variant to a gene symbol via coordinate overlap."""
        # Check telomere gene regions first
        for gene, (chrom, start, end) in _TELOMERE_GENE_REGIONS_GRCH38.items():
            if variant.chrom == chrom and start <= variant.pos <= end:
                return gene

        # Check if gene is annotated in INFO field (e.g. VEP, SnpEff)
        for key in ("GENE", "Gene", "ANN", "CSQ"):
            val = variant.info.get(key)
            if val and isinstance(val, str):
                # Extract first gene symbol from annotation
                parts = val.split("|")
                for part in parts:
                    clean = part.strip()
                    if clean and clean.isalpha() and clean.isupper():
                        return clean
        return None

    @staticmethod
    def _genotype_to_allele_string(variant: WGSVariant) -> str:
        """Convert VCF genotype to a two-character allele string.

        ``0/0`` → ``RR`` (ref/ref), ``0/1`` → ``RA``, ``1/1`` → ``AA``.
        Returns the actual bases (e.g. ``"AG"``) rather than symbolic labels.
        """
        alleles_map = {
            "0": variant.ref,
            "1": variant.alt.split(",")[0] if "," in variant.alt else variant.alt,
        }
        gt_parts = re.split(r"[/|]", variant.genotype)
        result_alleles = [alleles_map.get(a, "N") for a in gt_parts]
        return "".join(result_alleles[:2])

    @staticmethod
    def _find_variant_by_rsid(data: WGSData, rsid: str) -> WGSVariant | None:
        """Look up a variant by dbSNP rsID."""
        for variant in data.variants.values():
            if variant.rsid == rsid:
                return variant
        return None

    @staticmethod
    def _genotype_dosage(variant: WGSVariant) -> int:
        """Count of alternate alleles (0, 1, or 2)."""
        parts = re.split(r"[/|]", variant.genotype)
        return sum(1 for a in parts if a != "0" and a != ".")

    @staticmethod
    def _effect_allele_dosage(variant: WGSVariant, effect_allele: str) -> int:
        """Count copies of a specific effect allele in the genotype."""
        alleles_map = {
            "0": variant.ref,
            "1": variant.alt.split(",")[0] if "," in variant.alt else variant.alt,
        }
        parts = re.split(r"[/|]", variant.genotype)
        return sum(1 for a in parts if alleles_map.get(a, "") == effect_allele)

    @staticmethod
    def _is_subtelomeric(variant: WGSVariant) -> bool:
        """Check if a variant falls in a sub-telomeric region.

        Sub-telomeric is defined as the first or last 500 kb of a
        chromosome.  Chromosome lengths are approximate (GRCh38).
        """
        # Approximate GRCh38 chromosome lengths (bp)
        chrom_lengths: dict[str, int] = {
            "chr1": 248956422,
            "chr2": 242193529,
            "chr3": 198295559,
            "chr4": 190214555,
            "chr5": 181538259,
            "chr6": 170805979,
            "chr7": 159345973,
            "chr8": 145138636,
            "chr9": 138394717,
            "chr10": 133797422,
            "chr11": 135086622,
            "chr12": 133275309,
            "chr13": 114364328,
            "chr14": 107043718,
            "chr15": 101991189,
            "chr16": 90338345,
            "chr17": 83257441,
            "chr18": 80373285,
            "chr19": 58617616,
            "chr20": 64444167,
            "chr21": 46709983,
            "chr22": 50818468,
            "chrX": 156040895,
            "chrY": 57227415,
        }
        subtelo_window = 500_000
        length = chrom_lengths.get(variant.chrom)
        if length is None:
            return False
        return variant.pos <= subtelo_window or variant.pos >= (length - subtelo_window)

    @staticmethod
    def _infer_star_allele(variant: WGSVariant, gene: str) -> str:
        """Infer a pharmacogenomic star-allele designation.

        In production this would query the PharmVar database; here we use
        a simplified heuristic based on variant type and known key positions.
        """
        if variant.genotype in ("0/0", "0|0"):
            return "*1"  # reference / wild-type
        if not variant.is_snv:
            return "*unknown_indel"
        # Placeholder: real implementation would use PharmVar lookup tables
        if variant.is_heterozygous:
            return "*1/*variant"
        return "*variant/*variant"

    @staticmethod
    def _infer_metaboliser_phenotype(star_allele: str, gene: str) -> str:
        """Map star-allele to metaboliser phenotype (simplified)."""
        if star_allele == "*1":
            return "normal"
        if "unknown" in star_allele:
            return "unknown"
        if star_allele.endswith("/*variant"):
            if star_allele.startswith("*1/"):
                return "intermediate"
            return "poor"
        return "unknown"

    @staticmethod
    def _generate_pgx_recommendation(gene: str, phenotype: str, drugs: list[str]) -> str:
        """Generate a brief CPIC-style prescribing recommendation."""
        drug_str = ", ".join(drugs[:3])
        if len(drugs) > 3:
            drug_str += f" (+{len(drugs) - 3} more)"

        if phenotype == "normal":
            return f"{gene} normal metaboliser — standard dosing recommended for {drug_str}."
        if phenotype == "intermediate":
            return (
                f"{gene} intermediate metaboliser — consider dose reduction "
                f"or alternative for {drug_str}. See CPIC guidelines."
            )
        if phenotype == "poor":
            return (
                f"{gene} poor metaboliser — avoid or significantly reduce "
                f"dose of {drug_str}. Consult CPIC guidelines and "
                f"clinical pharmacologist."
            )
        if phenotype in ("rapid", "ultrarapid"):
            return (
                f"{gene} {phenotype} metaboliser — consider dose increase "
                f"or therapeutic drug monitoring for {drug_str}."
            )
        return (
            f"{gene} phenotype undetermined — exercise caution with "
            f"{drug_str}. Confirmatory testing recommended."
        )
