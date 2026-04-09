"""Import raw genotype data from direct-to-consumer genomics services.

Provides a unified interface for parsing genotype files produced by 23andMe,
AncestryDNA, and other DTC genomics providers, as well as standard VCF
(Variant Call Format) files from clinical or research pipelines.  Parsed data
is returned as :class:`GenotypeData` containers that can be validated against
a built-in set of known rsIDs and converted to the ``{rsid: genotype}``
dictionary format consumed by
:class:`~teloscopy.genomics.disease_risk.DiseasePredictor` and
:class:`~teloscopy.nutrition.diet_advisor.DietAdvisor`.

File format references
----------------------
* **23andMe raw data format** — TSV with ``#``-comment headers.  Four columns:
  ``rsid``, ``chromosome``, ``position``, ``genotype``.  Missing calls ``--``.
  Ref: https://customercare.23andme.com/hc/en-us/articles/212196868

* **AncestryDNA raw data format** — TSV with ``#``-comment header block +
  column-header row.  Five columns: ``rsid``, ``chromosome``, ``position``,
  ``allele1``, ``allele2``.  Missing alleles ``0``.
  Ref: https://support.ancestry.com/s/article/Downloading-AncestryDNA-Raw-Data

* **VCF 4.3 specification** — GA4GH / SAMtools standard.  Extracts ``GT``
  from single-sample (or first sample in multi-sample) VCFs.  Biallelic SNVs
  with ``rs``-prefixed IDs only.
  Ref: https://samtools.github.io/hts-specs/VCFv4.3.pdf

**This module is intended for educational and research purposes only.**

Typical usage
-------------
>>> importer = GenotypeImporter()
>>> data = importer.parse_auto("my_23andme_raw.txt")
>>> report = importer.validate_genotypes(data)
>>> print(f"{report.valid}/{report.total} variants validated")
>>> variant_dict = importer.convert_to_variant_dict(data)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "GenotypeRecord",
    "GenotypeData",
    "ValidationReport",
    "FileFormat",
    "GenotypeImporter",
    "KNOWN_RSIDS",
    "FormatDetectionError",
    "GenotypeParseError",
]

logger = logging.getLogger(__name__)


class GenotypeParseError(Exception):
    """Raised when a genotype file cannot be parsed.

    Attributes
    ----------
    file_path : str       Path to the file that failed.
    line_number : int     Line where the error was encountered.
    reason : str          Human-readable failure description.
    """

    def __init__(self, reason: str, file_path: str = "", line_number: int | None = None) -> None:
        self.file_path, self.line_number, self.reason = file_path, line_number, reason
        loc = f" at line {line_number}" if line_number else ""
        ctx = f" in {file_path!r}" if file_path else ""
        super().__init__(f"Genotype parse error{ctx}{loc}: {reason}")


class FormatDetectionError(GenotypeParseError):
    """Raised when the file format cannot be automatically determined."""

    def __init__(self, file_path: str = "", reason: str = "") -> None:
        super().__init__(
            reason=reason or "Unable to detect genotype file format", file_path=file_path
        )


class FileFormat(Enum):
    """Recognised genotype file formats."""

    TWENTYTHREE_AND_ME = "23andme"
    ANCESTRY_DNA = "ancestry_dna"
    VCF = "vcf"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GenotypeRecord:
    """A single genotyped variant.

    Attributes
    ----------
    rsid : str          dbSNP reference identifier (e.g. ``"rs429358"``).
    chromosome : str    Chromosome label (``"1"``–``"22"``, ``"X"``, ``"Y"``, ``"MT"``).
    position : int      GRCh37 (23andMe, AncestryDNA) or GRCh38 (VCF) coordinate.
    genotype : str      Diploid genotype (e.g. ``"AG"``), ``"--"`` for no-call.
    allele1 : str       First allele (forward strand).
    allele2 : str       Second allele (forward strand).
    """

    rsid: str
    chromosome: str
    position: int
    genotype: str
    allele1: str
    allele2: str

    def is_missing(self) -> bool:
        """Return ``True`` if this call is a no-call / missing."""
        return self.genotype in ("--", "", "00", "NN", "..")

    def is_heterozygous(self) -> bool:
        """Return ``True`` if heterozygous."""
        return not self.is_missing() and len(self.genotype) >= 2 and self.allele1 != self.allele2

    def is_homozygous(self) -> bool:
        """Return ``True`` if homozygous (non-missing)."""
        return not self.is_missing() and len(self.genotype) >= 2 and self.allele1 == self.allele2


@dataclass()
class GenotypeData:
    """Container for parsed genotype data from a single file.

    Attributes
    ----------
    variants : dict[str, GenotypeRecord]  Mapping of rsID -> record.
    source : str            Source identifier (``"23andMe"``, ``"AncestryDNA"``, ``"VCF"``).
    chip_version : str      Chip / platform version if detected, else ``"unknown"``.
    sample_id : str         Sample identifier from file header, else ``"unknown"``.
    total_snps : int        Total variant records (including no-calls).
    build : str             Genome build (``"GRCh37"``, ``"GRCh38"``).
    file_path : str         Path to the original parsed file.
    """

    variants: dict[str, GenotypeRecord] = field(default_factory=dict)
    source: str = "unknown"
    chip_version: str = "unknown"
    sample_id: str = "unknown"
    total_snps: int = 0
    build: str = "GRCh37"
    file_path: str = ""

    def get_genotype(self, rsid: str) -> str | None:
        """Return the diploid genotype for *rsid*, or ``None``."""
        rec = self.variants.get(rsid)
        return rec.genotype if rec is not None else None

    def rsids(self) -> set[str]:
        """Return the set of all rsIDs in this dataset."""
        return set(self.variants.keys())

    def non_missing_count(self) -> int:
        """Count variants with successful genotype calls."""
        return sum(1 for r in self.variants.values() if not r.is_missing())

    def summary(self) -> str:
        """Return a brief human-readable summary."""
        n_miss = self.total_snps - self.non_missing_count()
        return (
            f"GenotypeData(source={self.source!r}, chip={self.chip_version!r}, "
            f"sample={self.sample_id!r}, build={self.build!r}, "
            f"total={self.total_snps}, missing={n_miss})"
        )


@dataclass()
class ValidationReport:
    """Results from validating genotype data against a known rsID database.

    Attributes
    ----------
    total : int          Total variants checked.
    valid : int          Variants whose rsID is in the known database.
    invalid : int        Variants with malformed rsIDs.
    unrecognized : int   Well-formed rsIDs not in our curated set.
    warnings : list[str] Human-readable warning messages.
    missing_calls : int  No-call genotypes.
    het_count : int      Heterozygous calls.
    hom_count : int      Homozygous calls.
    """

    total: int = 0
    valid: int = 0
    invalid: int = 0
    unrecognized: int = 0
    warnings: list[str] = field(default_factory=list)
    missing_calls: int = 0
    het_count: int = 0
    hom_count: int = 0

    @property
    def valid_pct(self) -> float:
        """Percentage of variants matching the known database."""
        return (self.valid / self.total * 100.0) if self.total > 0 else 0.0

    def is_acceptable(self, min_valid_pct: float = 50.0) -> bool:
        """Return ``True`` if validation passes a quality threshold."""
        return self.valid_pct >= min_valid_pct

    def summary(self) -> str:
        """Concise one-line summary."""
        return (
            f"Validation: {self.valid}/{self.total} known ({self.valid_pct:.1f}%), "
            f"{self.invalid} invalid, {self.unrecognized} unrecognized, "
            f"{self.missing_calls} missing, {len(self.warnings)} warnings"
        )


# ~200 key rsIDs aggregated from disease_risk and diet_advisor modules.
# Built as a space-separated string to keep compact, then split into a frozenset.
_KNOWN_RSID_STR = (
    # Cardiovascular / lipid metabolism (disease_risk)
    "rs429358 rs7412 rs11591147 rs11206510 rs10455872 rs3798220 rs688 rs5925 "
    "rs1333049 rs10757274 rs2200733 rs699 rs4961 rs1801253 rs5186 rs4340 "
    "rs1799998 rs10033464 rs2106261 rs6843082 rs13376333 rs12425791 rs11833579 "
    "rs2383207 rs1842896 rs12124533 rs10519210 rs1739843 rs2234962 rs2118181 "
    "rs10757278 rs7025486 rs10757269 rs10927875 rs9262636 rs12143842 rs10494366 "
    # Cancer predisposition (disease_risk)
    "rs80357906 rs766173 rs11571833 rs2981582 rs3803662 rs13387042 rs889312 "
    "rs13281615 rs3814113 rs2072590 rs1447295 rs16901979 rs6983267 rs10993994 "
    "rs4430796 rs1859962 rs121913332 rs4939827 rs6691170 rs4779584 rs1042522 "
    "rs1051730 rs8034191 rs2736100 rs4488809 rs910873 rs1805007 rs1805008 "
    "rs1801516 rs401681 rs505922 rs3790844 rs9543325 rs9642880 rs710521 "
    "rs2294008 rs965513 rs944289 rs2439302 rs2976392 rs2274223 rs17401966 "
    "rs9275319 rs7579899 rs11894252 rs7579014 rs4295627 rs498872 rs6457327 "
    "rs10484561 rs872071 rs17483466 rs3916765 "
    # Diabetes / metabolic (disease_risk)
    "rs7903146 rs12255372 rs1801282 rs5219 rs13266634 rs10811661 rs7756992 "
    "rs4402960 rs1111875 rs10830963 rs7578597 rs864745 rs689 rs1801133 "
    "rs1801131 rs9939609 rs1558902 rs17782313 rs571312 rs6548238 rs738409 "
    "rs58542926 rs2231142 rs16890979 rs1183201 rs646776 rs12740374 "
    # Autoimmune / neurological / Alzheimer's / Parkinson's (disease_risk)
    "rs3087243 rs2476601 rs2292239 rs3184504 rs12722495 rs1387153 rs1800574 "
    "rs137852689 rs3761847 rs1270942 rs7574865 rs2187668 rs7454108 rs17810546 "
    "rs1800629 rs179247 rs6127099 rs11136000 rs3851179 rs744373 rs6656401 "
    "rs9331896 rs10792832 rs1532278 rs3764650 rs28834970 rs11218343 rs75932628 "
    "rs190982 rs2718058 rs63750066 rs63749824 rs63750231 rs2043085 rs28897743 "
    "rs63750447 rs267607908 rs34637584 rs33939927 rs1801582 rs356219 rs11931074 "
    "rs199347 rs6532194 rs3758549 rs12608932 rs3849942 rs10260404 rs3135388 "
    "rs6897932 rs12722489 rs2104286 rs2947349 rs6732655 rs2651899 rs10166942 "
    "rs11172113 rs9652490 rs3794087 rs12469063 rs2300478 rs1026732 "
    # IBD / GI / eye / bone / blood / respiratory / kidney / liver (disease_risk)
    "rs2066847 rs10210302 rs11209026 rs2241880 rs6927022 rs2395185 rs12191877 "
    "rs20541 rs27524 rs4349859 rs30187 rs1061170 rs10490924 rs2230199 "
    "rs10033900 rs429608 rs10483727 rs4236601 rs11824032 rs2165241 rs524952 "
    "rs634990 rs8027411 rs1048315 rs7524776 rs3735520 rs2956540 rs1800553 "
    "rs61750900 rs2234693 rs1544410 rs1800012 rs3736228 rs4988235 rs2282679 "
    "rs2458413 rs10498635 rs143383 rs4730250 rs11177 rs334 rs33930165 "
    "rs1800562 rs1799945 rs6025 rs1799963 rs8176719 rs855791 rs4820268 "
    "rs1050828 rs1050829 rs6048 rs8076131 rs2305480 rs7216389 rs1342326 "
    "rs3894194 rs7671167 rs13141641 rs28929474 rs1980057 rs35705950 rs2076295 "
    "rs1891385 rs2076530 rs4293393 rs12917707 rs11959928 rs2856717 rs3803800 "
    "rs2738048 rs219780 rs4148166 rs4821480 rs73885319 rs7726159 rs2728127 "
    "rs641738 rs12531711 rs10488631 rs1061472 rs732774 rs17580 "
    # Mental health / dermatological / reproductive / atopy / infectious
    "rs1545843 rs7044150 rs2422321 rs4543289 rs4765913 rs10994359 rs1006737 "
    "rs12576775 rs1625579 rs2021722 rs6704768 rs1344706 rs7523273 rs1800497 "
    "rs6265 rs27048 rs1801260 rs8042149 rs406001 rs3780413 rs4622308 "
    "rs6589488 rs13405728 rs2479106 rs1351592 rs705702 rs12700667 rs7521902 "
    "rs10859871 rs5911500 rs10841686 rs10183486 rs2303369 rs4769613 rs2228145 "
    "rs7927894 rs2897442 rs2143950 rs12913832 rs4963169 rs6502867 rs9275572 "
    "rs3093023 rs4263839 rs2710102 rs10891491 rs2687201 rs9257809 rs3072 "
    "rs1801725 rs28931614 rs11575542 rs9264942 rs2395029 rs333 "
    # Nutrigenomics (diet_advisor) - omega-3, caffeine, folate, vitamins, detox
    "rs174546 rs1535 rs762551 rs2228570 rs33972313 rs7501331 rs12934922 "
    "rs1801280 rs71748309 rs1229984 rs4680 rs1800795 rs4880 rs4410790 "
    "rs1601993660 rs366631 rs1001179 rs1050450 rs1205 rs1800566 rs7946 "
    "rs4588 rs10741657 rs7385804 rs662 rs708272 rs1800588 rs5082 rs5400 "
    "rs713598 rs72921001 rs671 rs1801394 rs1805087 rs234706 rs3733890 "
    "rs4654748 rs602662 rs6543836 rs4646994 rs1799983 rs1042713 rs1800592 "
    "rs659366 rs1800849 rs2304672 rs73598374 rs11942223 rs9594738 rs3877899 "
    "rs780094 rs1501299 rs7799039 rs1137101 rs6277 rs25531 rs6323 rs6721961 "
    "rs1051740 rs1799930 rs1056836 rs9282861 rs4148323 rs1051266 rs70991108 "
    "rs558660 rs1801198 rs2274924 rs7914558 rs4072037 rs2066844 rs2740574 "
    "rs4149056 rs662799 rs328 rs17238484 "
    # Pharmacogenomics / misc
    "rs9896052 rs1617640 rs7583877 rs1426654 rs16891982"
)
KNOWN_RSIDS: frozenset[str] = frozenset(_KNOWN_RSID_STR.split())

_RSID_PATTERN: re.Pattern[str] = re.compile(r"^rs\d+$", re.IGNORECASE)
_VALID_CHROMOSOMES: frozenset[str] = frozenset(
    [str(c) for c in range(1, 23)] + ["X", "Y", "MT", "M", "XY"]
)
_VALID_ALLELES: frozenset[str] = frozenset("ACGTDINacgtdin-0.")


def _peek_lines(file_path: str, n: int = 50) -> list[str]:
    """Read the first *n* lines of a text file for format sniffing."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Genotype file not found: {file_path}")
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(n):
            line = fh.readline()
            if not line:
                break
            lines.append(line.rstrip("\n\r"))
    return lines


def detect_format(file_path: str) -> FileFormat:
    """Detect genotype file format via header / content heuristics.

    Strategy: (1) ``##fileformat=VCF`` -> VCF; (2) ``"AncestryDNA"`` keyword
    or 5-column TSV -> AncestryDNA; (3) 4-column TSV with rsID in col 0 or
    ``"23andMe"`` keyword -> 23andMe; (4) otherwise UNKNOWN.
    """
    lines = _peek_lines(file_path, n=60)
    if not lines:
        return FileFormat.UNKNOWN
    for line in lines:
        if line.startswith("##fileformat=VCF"):
            logger.debug("Detected VCF format via ##fileformat header.")
            return FileFormat.VCF
    has_ancestry = any(line.startswith("#") and "ancestrydna" in line.lower() for line in lines)
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cols = stripped.split("\t")
        if len(cols) == 5 and (has_ancestry or cols[0].lower() == "rsid"):
            logger.debug("Detected AncestryDNA format (5 tab columns).")
            return FileFormat.ANCESTRY_DNA
        if len(cols) == 4 and _RSID_PATTERN.match(cols[0]):
            logger.debug("Detected 23andMe format (4 tab columns, rsID col 0).")
            return FileFormat.TWENTYTHREE_AND_ME
        break
    if has_ancestry:
        return FileFormat.ANCESTRY_DNA
    if any(line.startswith("#") and "23andme" in line.lower() for line in lines):
        return FileFormat.TWENTYTHREE_AND_ME
    logger.warning("Unable to detect genotype file format for %r.", file_path)
    return FileFormat.UNKNOWN


def _normalise_chromosome(raw: str) -> str:
    """Normalise chromosome label: strip ``chr`` prefix, map 23->X, M->MT."""
    chrom = raw.strip().upper()
    if chrom.startswith("CHR"):
        chrom = chrom[3:]
    return {"23": "X", "24": "Y", "25": "XY", "26": "MT", "M": "MT"}.get(chrom, chrom)


def _normalise_allele(raw: str) -> str:
    """Upper-case allele; treat ``0`` / ``.`` / empty as ``"-"``."""
    a = raw.strip().upper()
    return "-" if a in ("0", ".", "") else a


def _build_genotype(allele1: str, allele2: str) -> str:
    """Construct diploid genotype from two alleles.  ``"-"`` = missing."""
    a1, a2 = _normalise_allele(allele1), _normalise_allele(allele2)
    if a1 == "-" and a2 == "-":
        return "--"
    if a1 == "-" or a2 == "-":
        return a1 if a2 == "-" else a2
    return a1 + a2


def _extract_header_field(headers: list[str], pattern: str, default: str = "unknown") -> str:
    """Search *headers* for regex *pattern*, return group 1 or *default*."""
    for line in headers:
        m = re.search(pattern, line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return default


def _extract_build(headers: list[str]) -> str:
    """Determine genome build from header comments (default GRCh37 for DTC)."""
    for line in headers:
        low = line.lower()
        if "grch38" in low or "hg38" in low:
            return "GRCh38"
        if "grch37" in low or "hg19" in low or "build 37" in low:
            return "GRCh37"
    return "GRCh37"


def _parse_vcf_gt(gt_field: str, ref: str, alt_list: list[str]) -> tuple[str, str]:
    """Convert VCF ``GT`` (e.g. ``"0/1"``) to an allele pair.

    See VCF 4.3 spec section 1.4.2 for GT encoding rules.
    """
    pool = [ref] + alt_list
    resolved: list[str] = []
    for p in re.split(r"[/|]", gt_field):
        p = p.strip()
        if p in (".", ""):
            resolved.append("-")
        else:
            try:
                idx = int(p)
                resolved.append(pool[idx] if idx < len(pool) else "-")
            except (ValueError, IndexError):
                resolved.append("-")
    if len(resolved) == 0:
        return ("-", "-")
    if len(resolved) == 1:
        return (resolved[0], resolved[0])
    return (resolved[0], resolved[1])


class GenotypeImporter:
    """Unified importer for DTC genotype files and VCF data.

    Parameters
    ----------
    strict : bool
        If ``True`` (default), raise :class:`GenotypeParseError` on
        malformed lines.  If ``False``, skip them with a warning.
    include_no_calls : bool
        If ``True``, include no-call variants in output.  Default ``True``.

    Examples
    --------
    >>> importer = GenotypeImporter()
    >>> data = importer.parse_23andme("genome_raw.txt")
    >>> variant_dict = importer.convert_to_variant_dict(data)
    """

    def __init__(self, strict: bool = True, include_no_calls: bool = True) -> None:
        self.strict = strict
        self.include_no_calls = include_no_calls

    def _line_error(self, msg: str, fp: str, ln: int) -> None:
        """Raise or warn depending on ``self.strict``."""
        if self.strict:
            raise GenotypeParseError(reason=msg, file_path=fp, line_number=ln)
        logger.warning("Line %d: %s -- skipping.", ln, msg)

    def _make_record(self, rsid: str, chrom: str, pos: int, a1: str, a2: str) -> GenotypeRecord:
        """Build a GenotypeRecord with normalised alleles/genotype."""
        return GenotypeRecord(
            rsid=rsid,
            chromosome=chrom,
            position=pos,
            genotype=_build_genotype(a1, a2),
            allele1=_normalise_allele(a1),
            allele2=_normalise_allele(a2),
        )

    # -- 23andMe ----------------------------------------------------------

    def parse_23andme(self, file_path: str) -> GenotypeData:
        """Parse a 23andMe raw data file.

        Format: ``rsid <tab> chromosome <tab> position <tab> genotype``
        with ``#``-comment headers.

        Ref: https://customercare.23andme.com/hc/en-us/articles/212196868
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"23andMe file not found: {file_path}")
        logger.info("Parsing 23andMe file: %s", file_path)
        hdrs: list[str] = []
        variants: dict[str, GenotypeRecord] = {}
        ln = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                ln += 1
                line = raw.rstrip("\n\r")
                if line.startswith("#"):
                    hdrs.append(line)
                    continue
                if not line.strip():
                    continue
                cols = line.split("\t")
                if len(cols) < 4:
                    self._line_error(f"Expected 4 columns, got {len(cols)}", file_path, ln)
                    continue
                try:
                    pos = int(cols[2].strip())
                except ValueError:
                    self._line_error(f"Non-integer position: {cols[2]!r}", file_path, ln)
                    continue
                g = cols[3].strip().upper()
                if len(g) == 2:
                    a1, a2 = g[0], g[1]
                elif len(g) == 1:
                    a1, a2 = g, "-"
                else:
                    a1, a2 = "-", "-"
                rec = self._make_record(
                    cols[0].strip(), _normalise_chromosome(cols[1]), pos, a1, a2
                )
                if rec.is_missing() and not self.include_no_calls:
                    continue
                variants[rec.rsid] = rec

        chip = _extract_header_field(hdrs, r"(?:platform|chip)[^:]*:\s*v?(\d+)")
        if chip != "unknown":
            chip = f"v{chip}"
        else:
            chip = _extract_header_field(hdrs, r"\bv(\d+)\b")
            chip = f"v{chip}" if chip != "unknown" else "unknown"
        sid = _extract_header_field(hdrs, r"(?:sample|customer|user)\s*(?:id|name)?\s*[:=]\s*(.+)")
        build = _extract_build(hdrs)
        data = GenotypeData(
            variants=variants,
            source="23andMe",
            chip_version=chip,
            sample_id=sid,
            total_snps=len(variants),
            build=build,
            file_path=file_path,
        )
        logger.info(
            "Parsed %d variants from 23andMe (chip=%s, build=%s).", data.total_snps, chip, build
        )
        return data

    # -- AncestryDNA ------------------------------------------------------

    def parse_ancestry_dna(self, file_path: str) -> GenotypeData:
        """Parse an AncestryDNA raw data file.

        Format: ``rsid <tab> chromosome <tab> position <tab> allele1 <tab> allele2``
        with ``#``-comment header block and column-header row.

        Ref: https://support.ancestry.com/s/article/Downloading-AncestryDNA-Raw-Data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"AncestryDNA file not found: {file_path}")
        logger.info("Parsing AncestryDNA file: %s", file_path)
        hdrs: list[str] = []
        variants: dict[str, GenotypeRecord] = {}
        ln = 0
        seen_col_hdr = False
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                ln += 1
                line = raw.rstrip("\n\r")
                if line.startswith("#"):
                    hdrs.append(line)
                    continue
                if not line.strip():
                    continue
                if not seen_col_hdr:
                    seen_col_hdr = True
                    continue  # skip column header row
                cols = line.split("\t")
                if len(cols) < 5:
                    self._line_error(f"Expected 5 columns, got {len(cols)}", file_path, ln)
                    continue
                try:
                    pos = int(cols[2].strip())
                except ValueError:
                    self._line_error(f"Non-integer position: {cols[2]!r}", file_path, ln)
                    continue
                rec = self._make_record(
                    cols[0].strip(),
                    _normalise_chromosome(cols[1]),
                    pos,
                    cols[3].strip(),
                    cols[4].strip(),
                )
                if rec.is_missing() and not self.include_no_calls:
                    continue
                variants[rec.rsid] = rec

        sid = _extract_header_field(hdrs, r"(?:sample|customer|user)\s*(?:id|name)?\s*[:=]\s*(.+)")
        build = _extract_build(hdrs)
        chip = "v2" if len(variants) > 700_000 else "v1"
        for h in hdrs:
            if "v2" in h.lower():
                chip = "v2"
                break
            if "v1" in h.lower():
                chip = "v1"
                break
        data = GenotypeData(
            variants=variants,
            source="AncestryDNA",
            chip_version=chip,
            sample_id=sid,
            total_snps=len(variants),
            build=build,
            file_path=file_path,
        )
        logger.info(
            "Parsed %d variants from AncestryDNA (chip=%s, build=%s).", data.total_snps, chip, build
        )
        return data

    # -- VCF --------------------------------------------------------------

    def parse_vcf(self, file_path: str, sample_index: int = 0) -> GenotypeData:
        """Parse a VCF v4.x file (single- or multi-sample).

        Only biallelic SNVs with ``rs``-prefixed IDs are included.  The
        *sample_index* selects which sample column to extract (default 0).

        Ref: https://samtools.github.io/hts-specs/VCFv4.3.pdf
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"VCF file not found: {file_path}")
        logger.info("Parsing VCF file: %s (sample_index=%d)", file_path, sample_index)
        hdrs: list[str] = []
        variants: dict[str, GenotypeRecord] = {}
        ln = 0
        col_names: list[str] = []
        sample_col: int | None = None
        fmt_col: int | None = None
        build = "GRCh38"
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                ln += 1
                line = raw.rstrip("\n\r")
                if line.startswith("##"):
                    hdrs.append(line)
                    if "grch37" in line.lower() or "hg19" in line.lower():
                        build = "GRCh37"
                    continue
                if line.startswith("#CHROM") or line.startswith("#chrom"):
                    col_names = line.split("\t")
                    try:
                        fmt_col = col_names.index("FORMAT")
                    except ValueError:
                        fmt_col = None
                    if fmt_col is not None:
                        sample_col = fmt_col + 1 + sample_index
                        if sample_col >= len(col_names):
                            raise GenotypeParseError(
                                reason=f"sample_index={sample_index} out of range; "
                                f"VCF has {len(col_names) - fmt_col - 1} sample(s)",
                                file_path=file_path,
                                line_number=ln,
                            )
                    continue
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 8:
                    self._line_error(f"VCF line has {len(cols)} cols (need >=8)", file_path, ln)
                    continue
                chrom = _normalise_chromosome(cols[0])
                try:
                    pos = int(cols[1])
                except ValueError:
                    self._line_error(f"Non-integer POS: {cols[1]!r}", file_path, ln)
                    continue
                rsid: str | None = None
                for rid in cols[2].split(";"):
                    rid = rid.strip()
                    if _RSID_PATTERN.match(rid):
                        rsid = rid
                        break
                if rsid is None:
                    continue
                ref = cols[3].strip().upper()
                alt_list = [a.strip() for a in cols[4].strip().upper().split(",")]
                if len(ref) != 1 or any(len(a) != 1 for a in alt_list if a != "."):
                    continue
                gt_raw = "."
                if sample_col is not None and fmt_col is not None:
                    fmt_fields = cols[fmt_col].split(":")
                    try:
                        gt_idx = fmt_fields.index("GT")
                    except ValueError:
                        continue
                    if sample_col < len(cols):
                        sf = cols[sample_col].split(":")
                        gt_raw = sf[gt_idx] if gt_idx < len(sf) else "."
                a1, a2 = _parse_vcf_gt(gt_raw, ref, alt_list)
                rec = self._make_record(rsid, chrom, pos, a1, a2)
                if rec.is_missing() and not self.include_no_calls:
                    continue
                variants[rsid] = rec

        sname = (
            col_names[sample_col]
            if sample_col is not None and sample_col < len(col_names)
            else "unknown"
        )
        data = GenotypeData(
            variants=variants,
            source="VCF",
            chip_version="unknown",
            sample_id=sname,
            total_snps=len(variants),
            build=build,
            file_path=file_path,
        )
        logger.info(
            "Parsed %d variants from VCF (sample=%s, build=%s).", data.total_snps, sname, build
        )
        return data

    # -- Auto-detect ------------------------------------------------------

    def parse_auto(self, file_path: str) -> GenotypeData:
        """Auto-detect file format and parse accordingly.

        Raises :class:`FormatDetectionError` if the format cannot be determined.
        """
        fmt = detect_format(file_path)
        logger.info("Auto-detected format %s for %s.", fmt.value, file_path)
        if fmt is FileFormat.TWENTYTHREE_AND_ME:
            return self.parse_23andme(file_path)
        if fmt is FileFormat.ANCESTRY_DNA:
            return self.parse_ancestry_dna(file_path)
        if fmt is FileFormat.VCF:
            return self.parse_vcf(file_path)
        raise FormatDetectionError(
            file_path=file_path,
            reason="Could not determine file format.  Ensure the file is a "
            "valid 23andMe, AncestryDNA, or VCF export.",
        )

    # -- Validation -------------------------------------------------------

    def validate_genotypes(
        self,
        data: GenotypeData,
        known_rsids: frozenset[str] | set[str] | None = None,
    ) -> ValidationReport:
        """Validate parsed genotype data against a known rsID set.

        Checks: (1) rsID syntax ``/^rs\\d+$/``, (2) membership in
        *known_rsids* (defaults to :data:`KNOWN_RSIDS`), (3) chromosome
        plausibility, (4) allele character validity.
        """
        if known_rsids is None:
            known_rsids = KNOWN_RSIDS
        report = ValidationReport(total=len(data.variants))
        warn_set: set[str] = set()
        for rsid, rec in data.variants.items():
            if rec.is_missing():
                report.missing_calls += 1
            if not _RSID_PATTERN.match(rsid):
                report.invalid += 1
                if len(warn_set) < 100:
                    warn_set.add(f"Malformed rsID: {rsid!r}")
                continue
            if rsid in known_rsids:
                report.valid += 1
            else:
                report.unrecognized += 1
            if rec.chromosome not in _VALID_CHROMOSOMES and len(warn_set) < 100:
                warn_set.add(f"{rsid}: unexpected chromosome {rec.chromosome!r}")
            for label, val in (("allele1", rec.allele1), ("allele2", rec.allele2)):
                if val and not all(c in _VALID_ALLELES for c in val) and len(warn_set) < 100:
                    warn_set.add(f"{rsid}: unexpected {label} char(s) in {val!r}")
            if not rec.is_missing():
                if rec.is_heterozygous():
                    report.het_count += 1
                elif rec.is_homozygous():
                    report.hom_count += 1
        report.warnings = sorted(warn_set)
        if report.missing_calls > data.total_snps * 0.1:
            report.warnings.append(
                f"High missing-call rate: {report.missing_calls}/{data.total_snps} "
                f"({report.missing_calls / max(data.total_snps, 1) * 100:.1f}%)"
            )
        if report.invalid > 0:
            report.warnings.append(f"{report.invalid} variant(s) have malformed rsIDs")
        logger.info("Validation complete: %s", report.summary())
        return report

    # -- Conversion -------------------------------------------------------

    def convert_to_variant_dict(
        self,
        data: GenotypeData,
        *,
        rsid_filter: set[str] | frozenset[str] | None = None,
        exclude_missing: bool = True,
    ) -> dict[str, str]:
        """Convert parsed genotype data to ``{rsid: genotype}`` dictionary.

        This is the interchange format expected by
        :class:`~teloscopy.genomics.disease_risk.DiseasePredictor` and
        :class:`~teloscopy.nutrition.diet_advisor.DietAdvisor`.

        Parameters
        ----------
        data : GenotypeData       Parsed genotype data.
        rsid_filter : set | None  If given, restrict to these rsIDs only.
        exclude_missing : bool    Omit no-call variants (default ``True``).

        Returns
        -------
        dict[str, str]  e.g. ``{"rs429358": "CT", "rs7412": "CC"}``
        """
        result: dict[str, str] = {}
        for rsid, rec in data.variants.items():
            if exclude_missing and rec.is_missing():
                continue
            if rsid_filter is not None and rsid not in rsid_filter:
                continue
            result[rsid] = rec.genotype
        logger.debug("Converted %d variants to variant dict.", len(result))
        return result
