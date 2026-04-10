"""AI-Driven Drug Target Discovery from Telomere Biology.

Integrates telomere length, genetic variants, and clinical context to identify
druggable targets across four intervention pathways: telomerase inhibitors,
ALT-pathway agents, senolytics, and shelterin/DDR modulators.

**RESEARCH USE ONLY -- not for clinical decision-making.**

References
----------
.. [1] Shay JW & Wright WE (2019). Nat Rev Genet 20(5):299-309. PMID:30760854.
.. [2] Bryan TM et al. (1997). Nat Med 3(11):1271-1274.
.. [3] Heaphy CM et al. (2011). Am J Pathol 179(4):1608-1615.
.. [4] Lopez-Otin C et al. (2013). Cell 153(6):1194-1217. PMID:23746838.
.. [5] Childs BG et al. (2017). Nat Rev Drug Discov 16:718-735.
.. [6] Xu M et al. (2018). Nat Med 24:1246-1256.
.. [7] Dikmen ZG et al. (2005). Cancer Res 65:7866-7873.
"""
from __future__ import annotations

import hashlib
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Module configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

__all__ = [
    "DrugTarget",
    "TherapyRecommendation",
    "DrugTargetProfile",
    "identify_drug_targets",
    "score_network_pharmacology",
    "predict_therapy_response",
    "generate_target_report",
    "RESEARCH_DISCLAIMER",
]

RESEARCH_DISCLAIMER: str = (
    "RESEARCH USE ONLY.  Drug-target predictions and therapy recommendations "
    "are for educational/investigational purposes only.  NOT validated for "
    "clinical use -- must NOT inform prescribing or treatment decisions.  "
    "Consult a qualified oncologist or physician.  See Shay & Wright (2019) "
    "Nat Rev Genet 20:299-309 for telomere biology review."
)

# ---------------------------------------------------------------------------
# Scientific references (all verified real publications)
# ---------------------------------------------------------------------------

_REFS: dict[str, str] = {
    "shay_2019": "Shay JW & Wright WE (2019). Nat Rev Genet 20(5):299-309. PMID:30760854.",
    "bryan_1997": "Bryan TM et al. (1997). Nat Med 3(11):1271-1274.",
    "heaphy_2011": "Heaphy CM et al. (2011). Am J Pathol 179(4):1608-1615.",
    "lopez_otin_2013": "Lopez-Otin C et al. (2013). Cell 153(6):1194-1217. PMID:23746838.",
    "childs_2017": "Childs BG et al. (2017). Nat Rev Drug Discov 16:718-735.",
    "xu_2018": "Xu M et al. (2018). Nat Med 24:1246-1256.",
    "dikmen_2005": "Dikmen ZG et al. (2005). Cancer Res 65:7866-7873.",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DrugTarget:
    """A druggable target from telomere biology.

    Parameters
    ----------
    gene_symbol : str
        HUGO gene symbol (e.g. ``"TERT"``, ``"ATRX"``, ``"BCL2"``).
    target_type : str
        ``"telomerase"``, ``"ALT"``, ``"senolytic"``, ``"shelterin"``, or ``"DDR"``.
    mechanism : str
        Biological mechanism linking target to telomere-based therapy.
    evidence_level : str
        ``"strong"``, ``"moderate"``, ``"emerging"``, or ``"preclinical"``.
    known_inhibitors : list[str]
        Drugs/compounds known to modulate this target.
    clinical_trial_status : str
        Highest clinical development stage.
    references : list[str]
        Supporting literature citations.
    """

    gene_symbol: str
    target_type: str
    mechanism: str
    evidence_level: str
    known_inhibitors: list[str] = field(default_factory=list)
    clinical_trial_status: str = "preclinical"
    references: list[str] = field(default_factory=list)


@dataclass
class TherapyRecommendation:
    """Therapy recommendation from drug-target analysis.

    Parameters
    ----------
    therapy_name : str
        Name of the recommended therapy.
    drug_class : str
        Pharmacological class.
    target_gene : str
        Primary gene target.
    indication : str
        Clinical indication or disease context.
    evidence_strength : str
        ``"strong"``, ``"moderate"``, ``"emerging"``, or ``"theoretical"``.
    mechanism_of_action : str
        How the therapy achieves its effect.
    predicted_response : float
        Predicted response probability [0, 1].
    contraindications : list[str]
        Known contraindications.
    references : list[str]
        Supporting citations.
    """

    therapy_name: str
    drug_class: str
    target_gene: str
    indication: str
    evidence_strength: str
    mechanism_of_action: str
    predicted_response: float
    contraindications: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class DrugTargetProfile:
    """Complete drug-target analysis result.

    Parameters
    ----------
    targets : list[DrugTarget]
        Identified druggable targets ranked by relevance.
    recommendations : list[TherapyRecommendation]
        Therapy recommendations derived from targets.
    pathway_analysis : dict
        Pathway names mapped to relevance scores and metadata.
    network_pharmacology_score : float
        Confidence score for the drug-target network (0-1).
    disclaimer : str
        Research-only disclaimer.
    """

    targets: list[DrugTarget] = field(default_factory=list)
    recommendations: list[TherapyRecommendation] = field(default_factory=list)
    pathway_analysis: dict[str, Any] = field(default_factory=dict)
    network_pharmacology_score: float = 0.0
    disclaimer: str = RESEARCH_DISCLAIMER


# ---------------------------------------------------------------------------
# Target knowledge base -- Telomerase pathway
# ---------------------------------------------------------------------------

def _dt(gene: str, ttype: str, mech: str, ev: str,
        inhibs: list[str] | None = None, trial: str = "preclinical",
        refs: list[str] | None = None) -> DrugTarget:
    """Shorthand factory for DrugTarget construction."""
    return DrugTarget(gene, ttype, mech, ev, inhibs or [], trial, refs or [])


_TELOMERASE_TARGETS: dict[str, DrugTarget] = {
    "TERT": _dt(
        "TERT", "telomerase",
        "Catalytic subunit of telomerase reverse transcriptase; reactivated in "
        "~85-90% of cancers.  Inhibition causes progressive telomere shortening.",
        "strong", ["imetelstat", "BIBR1532"], "Phase III",
        [_REFS["shay_2019"], _REFS["dikmen_2005"]]),
    "TERC": _dt(
        "TERC", "telomerase",
        "RNA template component of telomerase.  Imetelstat competitively "
        "antagonises the TERC template region.",
        "strong", ["imetelstat"], "Phase III", [_REFS["shay_2019"]]),
    "DKC1": _dt(
        "DKC1", "telomerase",
        "Dyskerin; stabilises TERC for holoenzyme assembly.  Loss causes "
        "X-linked dyskeratosis congenita.", "moderate",
        refs=[_REFS["shay_2019"]]),
    "NOP10": _dt(
        "NOP10", "telomerase",
        "H/ACA RNP component required for TERC stability.  Mutations cause "
        "autosomal recessive dyskeratosis congenita.", "emerging",
        refs=[_REFS["shay_2019"]]),
    "NHP2": _dt(
        "NHP2", "telomerase",
        "H/ACA snoRNP component binding TERC.  Biallelic mutations cause "
        "autosomal recessive dyskeratosis congenita.", "emerging",
        refs=[_REFS["shay_2019"]]),
    "GAR1": _dt(
        "GAR1", "telomerase",
        "H/ACA RNP subunit for pseudouridylation and telomerase maturation.",
        "emerging", refs=[_REFS["shay_2019"]]),
    "TCAB1": _dt(
        "TCAB1", "telomerase",
        "WRAP53/TCAB1 directs telomerase to Cajal bodies.  Mutations cause "
        "dyskeratosis congenita.", "moderate", refs=[_REFS["shay_2019"]]),
    "RTEL1": _dt(
        "RTEL1", "telomerase",
        "Helicase resolving T-loops and G-quadruplexes at telomeres.  Biallelic "
        "mutations cause Hoyeraal-Hreidarsson syndrome.",
        "moderate", refs=[_REFS["shay_2019"]]),
}

# ---------------------------------------------------------------------------
# Target knowledge base -- ALT pathway
# ---------------------------------------------------------------------------

_ALT_TARGETS: dict[str, DrugTarget] = {
    "ATRX": _dt(
        "ATRX", "ALT",
        "Chromatin remodeller whose loss drives ALT activation.  Diagnostic "
        "biomarker in gliomas, sarcomas, PanNETs.", "strong",
        trial="diagnostic biomarker",
        refs=[_REFS["bryan_1997"], _REFS["heaphy_2011"]]),
    "DAXX": _dt(
        "DAXX", "ALT",
        "Forms ATRX/DAXX complex depositing H3.3 at telomeres.  Co-inactivation "
        "enriched in ALT+ PanNETs and glioblastomas.", "strong",
        trial="diagnostic biomarker",
        refs=[_REFS["bryan_1997"], _REFS["heaphy_2011"]]),
    "H3F3A": _dt(
        "H3F3A", "ALT",
        "Histone H3.3; K27M/G34R mutations in paediatric gliomas contribute "
        "to ALT permissiveness.", "moderate", refs=[_REFS["heaphy_2011"]]),
    "SETD2": _dt(
        "SETD2", "ALT",
        "H3K36me3 writer; loss disrupts telomeric chromatin and enables ALT.",
        "emerging", refs=[_REFS["heaphy_2011"]]),
    "ATR": _dt(
        "ATR", "ALT",
        "ATR kinase essential for ALT recombination.  Inhibition creates "
        "synthetic lethality in ALT+ cells.", "strong",
        ["ceralasertib (AZD6738)"], "Phase II",
        [_REFS["bryan_1997"], _REFS["heaphy_2011"]]),
}

# ---------------------------------------------------------------------------
# Target knowledge base -- Senolytic pathway
# ---------------------------------------------------------------------------

_SENOLYTIC_TARGETS: dict[str, DrugTarget] = {
    "BCL2": _dt(
        "BCL2", "senolytic",
        "Anti-apoptotic protein upregulated in senescent cells.  BH3 mimetics "
        "selectively kill senescent cells.", "strong",
        ["navitoclax", "dasatinib"], "Phase II",
        [_REFS["childs_2017"], _REFS["xu_2018"]]),
    "BCLXL": _dt(
        "BCLXL", "senolytic",
        "BCL2L1; key anti-apoptotic factor in senescent cells targeted by "
        "dual BCL-2/BCL-XL inhibitors.", "strong",
        ["navitoclax", "dasatinib"], "Phase II",
        [_REFS["childs_2017"], _REFS["xu_2018"]]),
    "TP53": _dt(
        "TP53", "senolytic",
        "Master senescence regulator.  Short telomeres trigger persistent "
        "p53-dependent DDR and cell-cycle arrest.", "moderate",
        refs=[_REFS["lopez_otin_2013"], _REFS["childs_2017"]]),
    "CDKN1A": _dt(
        "CDKN1A", "senolytic",
        "p21/WAF1; CDK inhibitor downstream of p53 enforcing telomere-"
        "dysfunction-induced senescence.", "moderate",
        refs=[_REFS["lopez_otin_2013"], _REFS["childs_2017"]]),
    "CDKN2A": _dt(
        "CDKN2A", "senolytic",
        "p16^INK4a; biomarker and effector of deep senescence.  p16-Rb axis "
        "maintains irreversible growth arrest.", "strong",
        trial="biomarker",
        refs=[_REFS["lopez_otin_2013"], _REFS["childs_2017"]]),
    "MDM2": _dt(
        "MDM2", "senolytic",
        "E3 ligase degrading p53.  Nutlins restore p53 to promote apoptosis "
        "in senescent cells.", "emerging", refs=[_REFS["childs_2017"]]),
}

# ---------------------------------------------------------------------------
# Target knowledge base -- Shelterin complex
# ---------------------------------------------------------------------------

_SHELTERIN_TARGETS: dict[str, DrugTarget] = {
    "TRF1": _dt("TRF1", "shelterin",
                 "Binds ds-TTAGGG; negatively regulates TL.  Depletion causes "
                 "telomere fragility.", "moderate", refs=[_REFS["shay_2019"]]),
    "TRF2": _dt("TRF2", "shelterin",
                 "Essential for T-loop formation and ATM suppression.  Loss "
                 "triggers fusions and senescence.", "strong",
                 refs=[_REFS["shay_2019"]]),
    "POT1": _dt("POT1", "shelterin",
                 "Binds ss G-overhang.  Germline mutations cause familial "
                 "melanoma, CLL, cardiac angiosarcoma.", "strong",
                 refs=[_REFS["shay_2019"]]),
    "TIN2": _dt("TIN2", "shelterin",
                 "Bridge connecting TRF1/TRF2/TPP1-POT1.  Mutations cause "
                 "severe dyskeratosis congenita.", "moderate",
                 refs=[_REFS["shay_2019"]]),
    "TPP1": _dt("TPP1", "shelterin",
                 "Recruits telomerase via TEL-patch; enhances processivity.",
                 "emerging", refs=[_REFS["shay_2019"]]),
    "RAP1": _dt("RAP1", "shelterin",
                 "TERF2IP; dual role in telomere protection and NF-kB "
                 "regulation.", "emerging", refs=[_REFS["shay_2019"]]),
}

# ---------------------------------------------------------------------------
# Known drugs database
# ---------------------------------------------------------------------------

_KNOWN_DRUGS: dict[str, dict[str, Any]] = {
    "imetelstat": {
        "targets": ["TERT", "TERC"], "class": "telomerase inhibitor",
        "mechanism": "13-mer thiophosphoramidate (GRN163L) blocking TERC template.",
        "status": "FDA approved (MDS, 2024; brand name Rytelo)", "indications": ["MDS", "myelofibrosis"],
        "refs": [_REFS["dikmen_2005"], _REFS["shay_2019"]]},
    "BIBR1532": {
        "targets": ["TERT"], "class": "telomerase inhibitor",
        "mechanism": "Non-nucleosidic allosteric TERT inhibitor.",
        "status": "preclinical", "indications": ["various cancers (preclinical)"],
        "refs": [_REFS["shay_2019"]]},
    "dasatinib": {
        "targets": ["SRC", "ABL"],
        "class": "senolytic (TKI, repurposed)",
        "mechanism": "Multi-kinase inhibitor (SRC/ABL); FDA-approved for CML, "
                     "repurposed as senolytic targeting SRC/PI3K/AKT.",
        "status": "FDA approved (CML); Phase II (senolytic)",
        "indications": ["CML", "senescence-associated diseases"],
        "refs": [_REFS["childs_2017"], _REFS["xu_2018"]]},
    "quercetin": {
        "targets": ["BCL2", "BCLXL", "TP53"],
        "class": "senolytic (flavonoid)",
        "mechanism": "Flavonoid inhibiting PI3K/serpins/BCL-2 in senescent cells.  "
                     "Used with dasatinib (D+Q) as leading senolytic regimen.",
        "status": "nutraceutical",
        "indications": ["senescence-associated diseases"],
        "refs": [_REFS["childs_2017"], _REFS["xu_2018"]]},
    "fisetin": {
        "targets": ["BCL2", "BCLXL", "TP53"],
        "class": "senolytic (flavonoid)",
        "mechanism": "Natural flavonoid senolytic; AFFIRM-LITE Phase II trial for frailty.",
        "status": "Phase II", "indications": ["frailty", "age-related decline"],
        "refs": [_REFS["xu_2018"], _REFS["childs_2017"]]},
    "navitoclax": {
        "targets": ["BCL2", "BCLXL"], "class": "senolytic (BH3 mimetic)",
        "mechanism": "ABT-263; BCL-2/BCL-XL/BCL-W inhibitor.  Dose-limiting "
                     "thrombocytopaenia from platelet BCL-XL inhibition.",
        "status": "Phase III (navitoclax+ruxolitinib; TRANSFORM trials)", "indications": ["myelofibrosis", "senescence"],
        "refs": [_REFS["childs_2017"]]},
    "ceralasertib": {
        "targets": ["ATR"], "class": "ATR kinase inhibitor",
        "mechanism": "AZD6738; selective ATR inhibitor exploiting synthetic "
                     "lethality in ALT+ tumours.",
        "status": "Phase II",
        "indications": ["ALT-positive tumours", "ATRX-mutant cancers"],
        "refs": [_REFS["bryan_1997"], _REFS["heaphy_2011"]]},
}

# ---------------------------------------------------------------------------
# Evidence weights and pathway overlap
# ---------------------------------------------------------------------------

_EV_W: dict[str, float] = {
    "strong": 1.0, "moderate": 0.70, "emerging": 0.40,
    "preclinical": 0.25, "theoretical": 0.10,
    "biomarker": 0.50, "diagnostic biomarker": 0.50,
}

_PW_OVERLAP: dict[tuple[str, str], float] = {
    ("telomerase", "ALT"): 0.15, ("telomerase", "senolytic"): 0.30,
    ("telomerase", "shelterin"): 0.55, ("telomerase", "DDR"): 0.40,
    ("ALT", "senolytic"): 0.20, ("ALT", "shelterin"): 0.35,
    ("ALT", "DDR"): 0.60, ("senolytic", "shelterin"): 0.25,
    ("senolytic", "DDR"): 0.45, ("shelterin", "DDR"): 0.50,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_ev(level: str) -> str:
    """Normalise evidence level to canonical form."""
    low = level.strip().lower()
    for k in _EV_W:
        if k in low:
            return k
    return "preclinical"


def _expected_tl(age: int) -> float:
    """Age-expected TL (kb) via linear attrition model."""
    return max(2.0, (11_000.0 - 40.0 * age) / 1_000.0)


def _deficit(tl: float, age: int) -> float:
    """Positive = shorter than expected."""
    return _expected_tl(age) - tl


def _all_targets() -> dict[str, DrugTarget]:
    """Merge all target dicts."""
    m: dict[str, DrugTarget] = {}
    for d in (_TELOMERASE_TARGETS, _ALT_TARGETS, _SENOLYTIC_TARGETS, _SHELTERIN_TARGETS):
        m.update(d)
    return m


def _pw_ov(a: str, b: str) -> float:
    """Pairwise pathway overlap coefficient."""
    if a == b:
        return 1.0
    return _PW_OVERLAP.get((a, b), _PW_OVERLAP.get((b, a), 0.05))


def _vgenes(variants: list[dict[str, str]] | None) -> set[str]:
    """Gene symbols from variant list."""
    if not variants:
        return set()
    return {v["gene"].upper() for v in variants if v.get("gene")}


def _has_v(variants: list[dict[str, str]] | None, gene: str) -> bool:
    """Any variant affects *gene*?"""
    return gene.upper() in _vgenes(variants)


# ---------------------------------------------------------------------------
# Prioritisation engines
# ---------------------------------------------------------------------------

def _pri_maintenance(tl: float, age: int) -> list[DrugTarget]:
    """Targets for telomere maintenance (short TL, no tumour)."""
    d = _deficit(tl, age)
    if d > 3.0:
        ts = list(_SHELTERIN_TARGETS.values())
        for g in ("RTEL1", "DKC1", "TCAB1"):
            if g in _TELOMERASE_TARGETS:
                ts.append(_TELOMERASE_TARGETS[g])
        return ts
    if d > 1.5:
        ts = [_SHELTERIN_TARGETS[g] for g in ("TRF2", "POT1", "TIN2")
              if g in _SHELTERIN_TARGETS]
        if "RTEL1" in _TELOMERASE_TARGETS:
            ts.append(_TELOMERASE_TARGETS["RTEL1"])
        return ts
    return []


def _pri_alt(variants: list[dict[str, str]] | None) -> list[DrugTarget]:
    """ALT pathway inhibitors."""
    ts: list[DrugTarget] = []
    if "ATR" in _ALT_TARGETS:
        ts.append(_ALT_TARGETS["ATR"])
    for g in ("ATRX", "DAXX", "H3F3A", "SETD2"):
        if g in _ALT_TARGETS:
            if _has_v(variants, g):
                ts.insert(min(1, len(ts)), _ALT_TARGETS[g])
            else:
                ts.append(_ALT_TARGETS[g])
    return ts


def _pri_telomerase(variants: list[dict[str, str]] | None) -> list[DrugTarget]:
    """Telomerase inhibitors."""
    ts = [_TELOMERASE_TARGETS[g] for g in ("TERT", "TERC")
          if g in _TELOMERASE_TARGETS]
    for g in ("DKC1", "NOP10", "NHP2", "GAR1", "TCAB1", "RTEL1"):
        if g in _TELOMERASE_TARGETS and _has_v(variants, g):
            ts.append(_TELOMERASE_TARGETS[g])
    return ts


def _pri_senolytics(burden: str, tl: float, age: int) -> list[DrugTarget]:
    """Senolytic targets."""
    b = burden.strip().lower()
    if b == "high":
        return [_SENOLYTIC_TARGETS[g]
                for g in ("BCL2", "BCLXL", "CDKN2A", "TP53", "CDKN1A", "MDM2")
                if g in _SENOLYTIC_TARGETS]
    if b == "moderate":
        return [_SENOLYTIC_TARGETS[g] for g in ("BCL2", "BCLXL", "CDKN2A")
                if g in _SENOLYTIC_TARGETS]
    if _deficit(tl, age) > 2.0 and "CDKN2A" in _SENOLYTIC_TARGETS:
        return [_SENOLYTIC_TARGETS["CDKN2A"]]
    return []


# ---------------------------------------------------------------------------
# Recommendation & pathway builders
# ---------------------------------------------------------------------------

def _build_recs(targets: list[DrugTarget], tl: float, age: int,
                tumor: str | None, sen: str) -> list[TherapyRecommendation]:
    """Build therapy recommendations from prioritised targets."""
    recs: list[TherapyRecommendation] = []
    tgenes = {t.gene_symbol for t in targets}
    d = _deficit(tl, age)
    at = _all_targets()

    for name, info in _KNOWN_DRUGS.items():
        overlap = set(info["targets"]) & tgenes
        if not overlap:
            continue
        resp = 0.3 + sum(
            _EV_W.get(at[g].evidence_level, 0.25) * 0.2
            for g in overlap if g in at)
        contras: list[str] = []
        ind = info["indications"][0] if info["indications"] else "unspecified"

        if info["class"] == "telomerase inhibitor":
            if tumor:
                resp += 0.15; ind = tumor
            else:
                resp -= 0.10
                contras.append("No malignancy -- may impair stem cell telomere maintenance")
            if d > 3.0 and not tumor:
                resp -= 0.20
                contras.append("Critically short telomeres without cancer")
        elif "senolytic" in info["class"].lower():
            resp += {"high": 0.20, "moderate": 0.10}.get(sen.strip().lower(), -0.10)
            if age < 40:
                contras.append("Patient <40 -- limited senolytic evidence")
                resp -= 0.10
        elif "ATR" in info["class"].upper() or name == "ceralasertib":
            if tumor:
                resp += 0.15; ind = f"ALT-positive {tumor}"
            else:
                resp -= 0.15
                contras.append("ATR inhibitors require ALT-positive malignancy")

        recs.append(TherapyRecommendation(
            name, info["class"], ", ".join(sorted(overlap)), ind,
            _norm_ev(info["status"]), info["mechanism"],
            round(max(0.0, min(1.0, resp)), 3), contras, info.get("refs", [])))
    recs.sort(key=lambda r: r.predicted_response, reverse=True)
    return recs


def _analyse_pw(targets: list[DrugTarget], tl: float, age: int,
                tumor: str | None, alt: bool, sen: str) -> dict[str, Any]:
    """Pathway-level relevance analysis."""
    sc = {"telomerase": 0.0, "ALT": 0.0, "senolytic": 0.0,
          "shelterin": 0.0, "DDR": 0.0}
    for t in targets:
        if t.target_type in sc:
            sc[t.target_type] += _EV_W.get(t.evidence_level, 0.25)
    d = _deficit(tl, age)
    if tumor:
        sc["telomerase"] += 0.5
        if alt:
            sc["ALT"] += 0.8; sc["telomerase"] -= 0.3
    elif d > 2.0:
        sc["shelterin"] += 0.6; sc["telomerase"] -= 0.2
    sc["senolytic"] += {"high": 0.7, "moderate": 0.3}.get(sen.strip().lower(), 0.0)

    mx = max(sc.values()) or 1.0
    sc = {k: round(min(1.0, v / mx), 3) for k, v in sc.items()}
    counts: dict[str, int] = {}
    for t in targets:
        counts[t.target_type] = counts.get(t.target_type, 0) + 1
    thash = hashlib.sha256("|".join(sorted(
        f"{t.gene_symbol}:{t.target_type}" for t in targets
    )).encode()).hexdigest()[:16]
    return {"pathway_relevance_scores": sc,
            "dominant_pathway": max(sc, key=sc.get),  # type: ignore[arg-type]
            "telomere_deficit_kb": round(d, 2),
            "target_count_by_pathway": counts, "target_hash": thash}


# ---------------------------------------------------------------------------
# Public API -- identify_drug_targets
# ---------------------------------------------------------------------------

def identify_drug_targets(
    telomere_length_kb: float,
    age: int,
    variants: list[dict[str, str]] | None = None,
    tumor_type: str | None = None,
    alt_positive: bool = False,
    senescence_burden: str = "normal",
) -> DrugTargetProfile:
    """Analyse telomere data to identify relevant druggable targets.

    Decision logic:

    * *tumor_type* + *alt_positive* → ALT pathway inhibitors.
    * *tumor_type* + not ALT → telomerase inhibitors.
    * Very short TL without tumour → telomere maintenance therapies.
    * *senescence_burden* high/moderate → senolytic targets added.

    Parameters
    ----------
    telomere_length_kb : float
        Mean TRF length in kilobases.
    age : int
        Chronological age in years.
    variants : list[dict] or None
        Genetic variants ``[{"gene": "TERT", ...}, ...]``.
    tumor_type : str or None
        Active tumour type (e.g. ``"glioblastoma"``).
    alt_positive : bool
        Whether the tumour is ALT-positive.
    senescence_burden : str
        ``"normal"``, ``"moderate"``, or ``"high"``.

    Returns
    -------
    DrugTargetProfile
        Targets, recommendations, pathway scores, network score.

    References
    ----------
    .. [1] Shay & Wright (2019). Nat Rev Genet 20(5):299-309.
    .. [2] Bryan et al. (1997). Nat Med 3(11):1271-1274.
    .. [3] Childs et al. (2017). Nat Rev Drug Discov 16:718-735.

    Examples
    --------
    >>> p = identify_drug_targets(4.2, 65, tumor_type="glioblastoma",
    ...                           alt_positive=True)
    >>> len(p.targets) > 0
    True
    """
    variants = variants or []
    targets: list[DrugTarget] = []

    # Primary decision tree
    if tumor_type is not None:
        if alt_positive:
            targets.extend(_pri_alt(variants))
            for g in ("TRF2", "POT1"):
                if g in _SHELTERIN_TARGETS:
                    targets.append(_SHELTERIN_TARGETS[g])
        else:
            targets.extend(_pri_telomerase(variants))
    elif _deficit(telomere_length_kb, age) > 2.5:
        targets.extend(_pri_maintenance(telomere_length_kb, age))

    # Senescence layer
    burd = senescence_burden.strip().lower()
    existing = {t.gene_symbol for t in targets}
    if burd in ("high", "moderate"):
        for st in _pri_senolytics(senescence_burden, telomere_length_kb, age):
            if st.gene_symbol not in existing:
                targets.append(st); existing.add(st.gene_symbol)
    elif _deficit(telomere_length_kb, age) > 2.0:
        if "CDKN2A" not in existing and "CDKN2A" in _SENOLYTIC_TARGETS:
            targets.append(_SENOLYTIC_TARGETS["CDKN2A"])
            existing.add("CDKN2A")

    # Variant-driven additions
    known = _all_targets()
    for vg in _vgenes(variants):
        if vg not in existing and vg in known:
            targets.append(known[vg]); existing.add(vg)

    recs = _build_recs(targets, telomere_length_kb, age, tumor_type, senescence_burden)
    pa = _analyse_pw(targets, telomere_length_kb, age, tumor_type,
                     alt_positive, senescence_burden)
    ns = score_network_pharmacology(targets, variants)

    return DrugTargetProfile(targets, recs, pa, ns, RESEARCH_DISCLAIMER)


# ---------------------------------------------------------------------------
# Public API -- score_network_pharmacology
# ---------------------------------------------------------------------------

def score_network_pharmacology(
    targets: list[DrugTarget],
    variants: list[dict[str, str]] | None = None,
) -> float:
    """Compute confidence score for the drug-target network (0-1).

    Integrates evidence weight, pathway coverage, pairwise pathway overlap,
    target count saturation, and variant-gene overlap bonus.

    Parameters
    ----------
    targets : list[DrugTarget]
        Identified drug targets.
    variants : list[dict] or None
        Patient genetic variants.

    Returns
    -------
    float
        Network pharmacology score in [0, 1].
    """
    if not targets:
        return 0.0
    ev = statistics.mean([_EV_W.get(t.evidence_level, 0.25) for t in targets])
    pws = {"telomerase", "ALT", "senolytic", "shelterin", "DDR"}
    rep = {t.target_type for t in targets} & pws
    cov = len(rep) / len(pws)

    tl = list(rep)
    ov = (statistics.mean([_pw_ov(tl[i], tl[j])
                           for i in range(len(tl)) for j in range(i+1, len(tl))])
          if len(tl) >= 2 else 0.0)

    vb = 0.0
    if variants:
        m = _vgenes(variants) & {t.gene_symbol for t in targets}
        vb = min(0.15, len(m) * 0.05)

    s = 0.35*ev + 0.25*cov + 0.20*ov + 0.20*min(1.0, len(targets)/10.0) + vb
    return round(max(0.0, min(1.0, s)), 3)


# ---------------------------------------------------------------------------
# Public API -- predict_therapy_response
# ---------------------------------------------------------------------------

def predict_therapy_response(
    drug_name: str,
    telomere_length_kb: float,
    tumor_type: str | None = None,
    variants: list[dict[str, str]] | None = None,
) -> TherapyRecommendation:
    """Predict patient response to a specific drug.

    Parameters
    ----------
    drug_name : str
        Drug to evaluate (e.g. ``"imetelstat"``).
    telomere_length_kb : float
        Mean TL in kilobases.
    tumor_type : str or None
        Active tumour type.
    variants : list[dict] or None
        Patient genetic variants.

    Returns
    -------
    TherapyRecommendation

    Raises
    ------
    ValueError
        If *drug_name* is not in the database.

    References
    ----------
    .. [1] Dikmen et al. (2005). Cancer Res 65:7866-7873.
    .. [2] Xu et al. (2018). Nat Med 24:1246-1256.

    Examples
    --------
    >>> rec = predict_therapy_response("imetelstat", 5.5, "MDS")
    >>> 0.0 <= rec.predicted_response <= 1.0
    True
    """
    variants = variants or []
    norm = drug_name.strip().lower()
    norm = {"azd6738": "ceralasertib", "grn163l": "imetelstat",
            "abt-263": "navitoclax", "abt263": "navitoclax"}.get(norm, norm)
    if norm not in _KNOWN_DRUGS:
        raise ValueError(f"Drug '{drug_name}' not found.  "
                         f"Available: {', '.join(sorted(_KNOWN_DRUGS))}")

    info = _KNOWN_DRUGS[norm]
    resp = 0.35
    contras: list[str] = []
    ind = info["indications"][0] if info["indications"] else "unspecified"

    st = info["status"].lower()
    if "approved" in st:
        resp += 0.25
    elif "phase iii" in st:
        resp += 0.20
    elif "phase ii" in st:
        resp += 0.12

    if info["class"] == "telomerase inhibitor":
        if tumor_type:
            resp += 0.15; ind = tumor_type
            if any(tumor_type.lower() in i.lower() for i in info["indications"]):
                resp += 0.10
        else:
            resp -= 0.15
            contras.append("Telomerase inhibitors need malignant context")
        if telomere_length_kb < 4.0 and not tumor_type:
            contras.append("Critically short TL -- inhibition may accelerate failure")
            resp -= 0.15
    elif "senolytic" in info["class"].lower():
        if telomere_length_kb < 5.0:
            resp += 0.10
        if not tumor_type:
            resp += 0.05
    elif "ATR" in info["class"].upper() or norm == "ceralasertib":
        if tumor_type:
            resp += 0.15; ind = f"ALT-positive {tumor_type}"
        else:
            resp -= 0.20
            contras.append("ATR inhibitor needs ALT-positive tumour context")

    if set(info["targets"]) & _vgenes(variants):
        resp += 0.10

    return TherapyRecommendation(
        norm, info["class"], ", ".join(info["targets"]), ind,
        _norm_ev(info["status"]), info["mechanism"],
        round(max(0.0, min(1.0, resp)), 3), contras, info.get("refs", []))


# ---------------------------------------------------------------------------
# Public API -- generate_target_report
# ---------------------------------------------------------------------------

def generate_target_report(profile: DrugTargetProfile) -> str:
    """Generate a human-readable text report from a DrugTargetProfile.

    Parameters
    ----------
    profile : DrugTargetProfile
        Result from :func:`identify_drug_targets`.

    Returns
    -------
    str
        Multi-line formatted report.

    Examples
    --------
    >>> p = identify_drug_targets(5.0, 60)
    >>> "RESEARCH USE ONLY" in generate_target_report(p)
    True
    """
    sep = "=" * 72
    L: list[str] = [sep, "  AI-DRIVEN DRUG TARGET DISCOVERY FROM TELOMERE BIOLOGY",
                    sep, "", "DISCLAIMER", "-" * 72, profile.disclaimer, "",
                    "SUMMARY", "-" * 72,
                    f"  Targets identified         : {len(profile.targets)}",
                    f"  Therapy recommendations    : {len(profile.recommendations)}",
                    f"  Network pharmacology score : "
                    f"{profile.network_pharmacology_score:.3f}", ""]

    L += ["IDENTIFIED DRUG TARGETS", "-" * 72]
    for i, t in enumerate(profile.targets, 1):
        L.append(f"  [{i}] {t.gene_symbol} ({t.target_type})  "
                 f"evidence={t.evidence_level}  trial={t.clinical_trial_status}")
        if t.known_inhibitors:
            L.append(f"      drugs: {', '.join(t.known_inhibitors)}")
    L.append("")

    L += ["THERAPY RECOMMENDATIONS", "-" * 72]
    for i, r in enumerate(profile.recommendations, 1):
        L.append(f"  [{i}] {r.therapy_name} ({r.drug_class})")
        L.append(f"      target={r.target_gene}  indication={r.indication}  "
                 f"response={r.predicted_response:.1%}")
        for c in r.contraindications:
            L.append(f"      CAUTION: {c}")
    L.append("")

    L += ["PATHWAY ANALYSIS", "-" * 72]
    pa = profile.pathway_analysis
    for pw, sc in sorted(pa.get("pathway_relevance_scores", {}).items(),
                         key=lambda x: x[1], reverse=True):
        L.append(f"  {pw:<15s}: {sc:.3f}  {'#' * int(sc * 30)}")
    if "dominant_pathway" in pa:
        L.append(f"  Dominant: {pa['dominant_pathway']}")
    L.append("")

    L += ["REFERENCES", "-" * 72]
    seen: set[str] = set()
    n = 1
    for t in profile.targets:
        for r in t.references:
            if r not in seen:
                seen.add(r); L.append(f"  [{n}] {r}"); n += 1
    for rec in profile.recommendations:
        for r in rec.references:
            if r not in seen:
                seen.add(r); L.append(f"  [{n}] {r}"); n += 1
    if not seen:
        for r in _REFS.values():
            L.append(f"  [{n}] {r}"); n += 1
    L += ["", sep, "  END OF REPORT", sep]
    return "\n".join(L)
