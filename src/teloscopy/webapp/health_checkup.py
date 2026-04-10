"""Health Checkup Analysis Module.

Interprets annual health checkup lab results (blood tests, urine tests,
abdomen scan notes) to detect health conditions, compute an overall health
score, and generate a personalised diet plan that accounts for both
regional preferences and medical findings.

This module is strictly for **research / educational** purposes.
It is **not** a substitute for clinical diagnosis or medical advice.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..nutrition.diet_advisor import DietAdvisor, DietaryRecommendation
from ..nutrition.regional_diets import resolve_region
from .models import (
    AbdomenFindingResponse,
    BloodTestPanel,
    DietRecommendation,
    HealthCheckupRequest,
    HealthCheckupResponse,
    HealthFindingResponse,
    LabResultResponse,
    MealPlan,
    UrineTestPanel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON data directory and loader
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"


def _load_json(name: str) -> Any:
    """Load and return parsed JSON from *_DATA_DIR / name*."""
    with open(_DATA_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Reference ranges — keyed by (sex, age_group)
# age_group: "child" (<18), "adult" (18-65), "senior" (>65)
# ---------------------------------------------------------------------------

def _age_group(age: int) -> str:
    if age < 18:
        return "child"
    if age <= 65:
        return "adult"
    return "senior"


@dataclass(frozen=True)
class RefRange:
    """Reference range for a single lab parameter."""
    low: float
    high: float
    unit: str
    display_name: str
    category: str
    critical_low: float | None = None
    critical_high: float | None = None


# Default adult reference ranges.  Sex-specific overrides follow.
_BASE_BLOOD_RANGES: dict[str, RefRange] = {
    k: RefRange(
        low=v["low"],
        high=v["high"],
        unit=v["unit"],
        display_name=v["display_name"],
        category=v["category"],
        critical_low=v.get("critical_low"),
        critical_high=v.get("critical_high"),
    )
    for k, v in _load_json("blood_reference_ranges.json").items()
}

# Sex-specific overrides for certain parameters.
_SEX_OVERRIDES: dict[str, dict[str, tuple[float, float]]] = {
    param: {sex: (vals["low"], vals["high"]) for sex, vals in sexes.items()}
    for param, sexes in _load_json("sex_specific_overrides.json").items()
}

# Urine reference ranges.
_URINE_RANGES: dict[str, RefRange] = {
    k: RefRange(
        low=v["low"],
        high=v["high"],
        unit=v["unit"],
        display_name=v["display_name"],
        category=v["category"],
        critical_low=v.get("critical_low"),
        critical_high=v.get("critical_high"),
    )
    for k, v in _load_json("urine_reference_ranges.json").items()
}


def _get_ref(param: str, sex: str) -> RefRange:
    """Get reference range with sex-specific adjustments."""
    base = _BASE_BLOOD_RANGES.get(param) or _URINE_RANGES.get(param)
    if base is None:
        raise KeyError(param)
    overrides = _SEX_OVERRIDES.get(param, {})
    if sex in overrides:
        lo, hi = overrides[sex]
        return RefRange(lo, hi, base.unit, base.display_name, base.category,
                        base.critical_low, base.critical_high)
    return base


def _classify(value: float, ref: RefRange) -> str:
    """Classify a lab value as low/normal/high/critical."""
    if ref.critical_low is not None and value < ref.critical_low:
        return "critical_low"
    if ref.critical_high is not None and value > ref.critical_high:
        return "critical_high"
    if value < ref.low:
        return "low"
    if value > ref.high:
        return "high"
    return "normal"


# ---------------------------------------------------------------------------
# Condition detection rules
# ---------------------------------------------------------------------------

@dataclass
class _ConditionRule:
    """A rule that detects a health condition from lab values."""
    condition: str
    display_name: str
    check: str  # callable name on HealthCheckupAnalyzer
    dietary_impact: str
    nutrients_to_increase: list[str] = field(default_factory=list)
    nutrients_to_decrease: list[str] = field(default_factory=list)
    foods_to_increase: list[str] = field(default_factory=list)
    foods_to_avoid: list[str] = field(default_factory=list)


_CONDITION_RULES: list[_ConditionRule] = [
    _ConditionRule(**entry) for entry in _load_json("condition_rules.json")
]


# ---------------------------------------------------------------------------
# Abdomen scan keyword parser
# ---------------------------------------------------------------------------

_ABDOMEN_PATTERNS: list[tuple[str, dict[str, Any]]] = [
    (entry["pattern"], {k: v for k, v in entry.items() if k != "pattern"})
    for entry in _load_json("abdomen_patterns.json")
]


# ---------------------------------------------------------------------------
# Health score weights by category
# ---------------------------------------------------------------------------

_CATEGORY_WEIGHTS: dict[str, float] = _load_json("category_weights.json")


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class HealthCheckupAnalyzer:
    """Analyze health checkup data and generate diet recommendations.

    Parameters
    ----------
    diet_advisor : DietAdvisor
        Existing DietAdvisor instance for meal plan generation.
    """

    def __init__(self, diet_advisor: DietAdvisor) -> None:
        self._advisor = diet_advisor

    # -- public API ----------------------------------------------------------

    def analyze(self, request: HealthCheckupRequest) -> HealthCheckupResponse:
        """Run full analysis pipeline and return a response."""
        sex = request.sex.value if hasattr(request.sex, "value") else str(request.sex)
        age = request.age

        # 1. Interpret lab values
        lab_results = self._interpret_labs(request.blood_tests, request.urine_tests, sex, age)

        # 2. Detect conditions
        findings = self._detect_conditions(request.blood_tests, request.urine_tests, sex, age)

        # 3. Parse abdomen scan
        abdomen_findings = self._parse_abdomen(request.abdomen_scan_notes)

        # 4. Compute health score
        abnormal = [r for r in lab_results if r.status != "normal"]
        score, breakdown = self._compute_health_score(lab_results)

        # 5. Collect all detected conditions for diet integration
        detected_conditions = [f.condition for f in findings]
        for af in abdomen_findings:
            detected_conditions.append(af.finding)
        # Merge user-reported conditions
        all_conditions = list(set(detected_conditions + request.health_conditions))

        # 6. Generate diet plan using DietAdvisor
        region_id = resolve_region(
            request.region,
            country=request.country,
            state=request.state,
        )

        diet_rec, modifications, calorie_adj = self._generate_diet(
            all_conditions=all_conditions,
            findings=findings,
            abdomen_findings=abdomen_findings,
            region=region_id,
            age=age,
            sex=sex,
            dietary_restrictions=request.dietary_restrictions,
            known_variants=request.known_variants,
            calorie_target=request.calorie_target,
            meal_plan_days=request.meal_plan_days,
        )

        return HealthCheckupResponse(
            lab_results=lab_results,
            abnormal_count=len(abnormal),
            total_tested=len(lab_results),
            findings=findings,
            abdomen_findings=abdomen_findings,
            detected_conditions=detected_conditions,
            overall_health_score=score,
            health_score_breakdown=breakdown,
            diet_recommendation=diet_rec,
            dietary_modifications=modifications,
            calorie_adjustment=calorie_adj,
        )

    # -- lab interpretation --------------------------------------------------

    def _interpret_labs(
        self,
        blood: BloodTestPanel | None,
        urine: UrineTestPanel | None,
        sex: str,
        age: int,
    ) -> list[LabResultResponse]:
        results: list[LabResultResponse] = []
        if blood is not None:
            blood_dict = blood.model_dump(exclude_none=True)
            for param, value in blood_dict.items():
                try:
                    ref = _get_ref(param, sex)
                except KeyError:
                    continue
                status = _classify(value, ref)
                results.append(LabResultResponse(
                    parameter=param,
                    display_name=ref.display_name,
                    value=value,
                    unit=ref.unit,
                    status=status,
                    reference_low=ref.low,
                    reference_high=ref.high,
                    category=ref.category,
                ))
        if urine is not None:
            urine_dict = urine.model_dump(exclude_none=True)
            for param, value in urine_dict.items():
                try:
                    ref = _get_ref(param, sex)
                except KeyError:
                    continue
                status = _classify(value, ref)
                results.append(LabResultResponse(
                    parameter=param,
                    display_name=ref.display_name,
                    value=value,
                    unit=ref.unit,
                    status=status,
                    reference_low=ref.low,
                    reference_high=ref.high,
                    category=ref.category,
                ))
        return results

    # -- condition detection -------------------------------------------------

    def _detect_conditions(
        self,
        blood: BloodTestPanel | None,
        urine: UrineTestPanel | None,
        sex: str,
        age: int,
    ) -> list[HealthFindingResponse]:
        findings: list[HealthFindingResponse] = []
        for rule in _CONDITION_RULES:
            check_method = getattr(self, rule.check, None)
            if check_method is None:
                logger.warning("Unknown check method: %s", rule.check)
                continue
            result = check_method(blood, urine, sex, age)
            if result is not None:
                severity, evidence = result
                findings.append(HealthFindingResponse(
                    condition=rule.condition,
                    display_name=rule.display_name,
                    severity=severity,
                    evidence=evidence,
                    dietary_impact=rule.dietary_impact,
                    nutrients_to_increase=rule.nutrients_to_increase,
                    nutrients_to_decrease=rule.nutrients_to_decrease,
                    foods_to_increase=rule.foods_to_increase,
                    foods_to_avoid=rule.foods_to_avoid,
                ))
        return findings

    # -- individual condition checks (return (severity, evidence) or None) ---

    def _check_prediabetes(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.fasting_glucose is not None and 100 <= blood.fasting_glucose < 126:
            evidence.append(f"Fasting glucose {blood.fasting_glucose} mg/dL (100-125 = pre-diabetic)")
        if blood.hba1c is not None and 5.7 <= blood.hba1c < 6.5:
            evidence.append(f"HbA1c {blood.hba1c}% (5.7-6.4 = pre-diabetic)")
        if blood.postprandial_glucose is not None and 140 <= blood.postprandial_glucose < 200:
            evidence.append(f"PP glucose {blood.postprandial_glucose} mg/dL (140-199 = IGT)")
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_diabetes(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.fasting_glucose is not None and blood.fasting_glucose >= 126:
            evidence.append(f"Fasting glucose {blood.fasting_glucose} mg/dL (≥126 = diabetic)")
        if blood.hba1c is not None and blood.hba1c >= 6.5:
            evidence.append(f"HbA1c {blood.hba1c}% (≥6.5 = diabetic)")
        if blood.postprandial_glucose is not None and blood.postprandial_glucose >= 200:
            evidence.append(f"PP glucose {blood.postprandial_glucose} mg/dL (≥200 = diabetic)")
        if not evidence:
            return None
        severity = "severe" if len(evidence) >= 2 else "moderate"
        return severity, evidence

    def _check_dyslipidemia(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.total_cholesterol is not None and blood.total_cholesterol > 200:
            evidence.append(f"Total cholesterol {blood.total_cholesterol} mg/dL (>200)")
        if blood.ldl_cholesterol is not None and blood.ldl_cholesterol > 100:
            lvl = "borderline" if blood.ldl_cholesterol < 160 else "high"
            evidence.append(f"LDL {blood.ldl_cholesterol} mg/dL ({lvl})")
        if blood.hdl_cholesterol is not None and blood.hdl_cholesterol < 40:
            evidence.append(f"HDL {blood.hdl_cholesterol} mg/dL (low <40)")
        if blood.triglycerides is not None and blood.triglycerides > 150:
            evidence.append(f"Triglycerides {blood.triglycerides} mg/dL (>150)")
        if not evidence:
            return None
        severity = "severe" if len(evidence) >= 3 else ("moderate" if len(evidence) >= 2 else "mild")
        return severity, evidence

    def _check_liver_stress(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.sgpt_alt is not None and blood.sgpt_alt > 56:
            evidence.append(f"ALT {blood.sgpt_alt} U/L (>56)")
        if blood.sgot_ast is not None and blood.sgot_ast > 40:
            evidence.append(f"AST {blood.sgot_ast} U/L (>40)")
        if blood.ggt is not None and blood.ggt > 45:
            evidence.append(f"GGT {blood.ggt} U/L (>45)")
        if blood.alkaline_phosphatase is not None and blood.alkaline_phosphatase > 147:
            evidence.append(f"ALP {blood.alkaline_phosphatase} U/L (>147)")
        if blood.total_bilirubin is not None and blood.total_bilirubin > 1.2:
            evidence.append(f"Total bilirubin {blood.total_bilirubin} mg/dL (>1.2)")
        if not evidence:
            return None
        severity = "severe" if len(evidence) >= 3 else ("moderate" if len(evidence) >= 2 else "mild")
        return severity, evidence

    def _check_fatty_liver(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        # Fatty liver indicated by elevated liver enzymes + high triglycerides
        evidence = []
        alt_high = blood.sgpt_alt is not None and blood.sgpt_alt > 40
        ast_high = blood.sgot_ast is not None and blood.sgot_ast > 35
        tg_high = blood.triglycerides is not None and blood.triglycerides > 150
        ggt_high = blood.ggt is not None and blood.ggt > 45
        if alt_high:
            evidence.append(f"ALT {blood.sgpt_alt} U/L (elevated)")
        if ast_high:
            evidence.append(f"AST {blood.sgot_ast} U/L (elevated)")
        if tg_high:
            evidence.append(f"Triglycerides {blood.triglycerides} mg/dL (elevated)")
        if ggt_high:
            evidence.append(f"GGT {blood.ggt} U/L (elevated)")
        # Need at least liver enzyme + one other marker
        liver_marker = alt_high or ast_high
        metabolic_marker = tg_high or ggt_high
        if not (liver_marker and metabolic_marker):
            return None
        severity = "moderate" if len(evidence) >= 3 else "mild"
        return severity, evidence

    def _check_kidney_impairment(
        self, blood: BloodTestPanel | None, urine: UrineTestPanel | None, sex: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        evidence = []
        if blood is not None:
            creat_limit = 1.3 if sex == "male" else 1.1
            if blood.serum_creatinine is not None and blood.serum_creatinine > creat_limit:
                evidence.append(f"Creatinine {blood.serum_creatinine} mg/dL (>{creat_limit})")
            if blood.egfr is not None and blood.egfr < 90:
                evidence.append(f"eGFR {blood.egfr} mL/min (< 90)")
            if blood.bun is not None and blood.bun > 20:
                evidence.append(f"BUN {blood.bun} mg/dL (>20)")
            if blood.blood_urea is not None and blood.blood_urea > 20:
                evidence.append(f"Blood urea {blood.blood_urea} mg/dL (>20)")
        if urine is not None:
            if urine.protein is not None and urine.protein > 14:
                evidence.append(f"Urine protein {urine.protein} mg/dL (positive)")
        if not evidence:
            return None
        severity = "severe" if len(evidence) >= 3 else ("moderate" if len(evidence) >= 2 else "mild")
        return severity, evidence

    def _check_hyperuricemia(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, sex: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None or blood.uric_acid is None:
            return None
        limit = 7.2 if sex == "male" else 6.0
        if blood.uric_acid > limit:
            severity = "moderate" if blood.uric_acid > limit + 1.5 else "mild"
            return severity, [f"Uric acid {blood.uric_acid} mg/dL (>{limit})"]
        return None

    def _check_hypothyroidism(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.tsh is not None and blood.tsh > 4.0:
            evidence.append(f"TSH {blood.tsh} µIU/mL (>4.0)")
        if blood.free_t4 is not None and blood.free_t4 < 0.8:
            evidence.append(f"Free T4 {blood.free_t4} ng/dL (<0.8)")
        if blood.free_t3 is not None and blood.free_t3 < 2.0:
            evidence.append(f"Free T3 {blood.free_t3} pg/mL (<2.0)")
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_hyperthyroidism(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.tsh is not None and blood.tsh < 0.4:
            evidence.append(f"TSH {blood.tsh} µIU/mL (<0.4)")
        if blood.free_t4 is not None and blood.free_t4 > 1.8:
            evidence.append(f"Free T4 {blood.free_t4} ng/dL (>1.8)")
        if blood.free_t3 is not None and blood.free_t3 > 4.4:
            evidence.append(f"Free T3 {blood.free_t3} pg/mL (>4.4)")
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_anemia(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, sex: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        hb_limit = 13.0 if sex == "male" else 12.0
        if blood.hemoglobin is not None and blood.hemoglobin < hb_limit:
            evidence.append(f"Hemoglobin {blood.hemoglobin} g/dL (<{hb_limit})")
        if blood.hematocrit is not None:
            hct_limit = 38.0 if sex == "male" else 36.0
            if blood.hematocrit < hct_limit:
                evidence.append(f"Hematocrit {blood.hematocrit}% (<{hct_limit})")
        if blood.mcv is not None and blood.mcv < 80:
            evidence.append(f"MCV {blood.mcv} fL (<80, microcytic)")
        if blood.ferritin is not None:
            ferr_limit = 20.0 if sex == "male" else 12.0
            if blood.ferritin < ferr_limit:
                evidence.append(f"Ferritin {blood.ferritin} ng/mL (<{ferr_limit})")
        if not evidence:
            return None
        severity = "severe" if len(evidence) >= 3 else ("moderate" if len(evidence) >= 2 else "mild")
        return severity, evidence

    def _check_vitamin_d_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None or blood.vitamin_d is None:
            return None
        if blood.vitamin_d < 20:
            return "severe" if blood.vitamin_d < 10 else "moderate", [
                f"Vitamin D {blood.vitamin_d} ng/mL (<20 = deficient)"
            ]
        if blood.vitamin_d < 30:
            return "mild", [f"Vitamin D {blood.vitamin_d} ng/mL (20-29 = insufficient)"]
        return None

    def _check_b12_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None or blood.vitamin_b12 is None:
            return None
        if blood.vitamin_b12 < 200:
            severity = "severe" if blood.vitamin_b12 < 150 else "moderate"
            return severity, [f"Vitamin B12 {blood.vitamin_b12} pg/mL (<200 = deficient)"]
        if blood.vitamin_b12 < 300:
            return "mild", [f"Vitamin B12 {blood.vitamin_b12} pg/mL (200-300 = borderline)"]
        return None

    def _check_iron_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, sex: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.iron is not None:
            limit = 65.0 if sex == "male" else 50.0
            if blood.iron < limit:
                evidence.append(f"Serum iron {blood.iron} µg/dL (<{limit})")
        if blood.ferritin is not None:
            limit = 20.0 if sex == "male" else 12.0
            if blood.ferritin < limit:
                evidence.append(f"Ferritin {blood.ferritin} ng/mL (<{limit})")
        if blood.transferrin_saturation is not None and blood.transferrin_saturation < 20:
            evidence.append(f"Transferrin sat {blood.transferrin_saturation}% (<20)")
        if blood.tibc is not None and blood.tibc > 400:
            evidence.append(f"TIBC {blood.tibc} µg/dL (>400 = iron deficiency)")
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_inflammation(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.crp is not None and blood.crp > 3.0:
            evidence.append(f"CRP {blood.crp} mg/L (>3.0)")
        if blood.esr is not None and blood.esr > 20:
            evidence.append(f"ESR {blood.esr} mm/hr (>20)")
        if blood.homocysteine is not None and blood.homocysteine > 15:
            evidence.append(f"Homocysteine {blood.homocysteine} µmol/L (>15)")
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_electrolyte_imbalance(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.sodium is not None and (blood.sodium < 136 or blood.sodium > 145):
            status = "low" if blood.sodium < 136 else "high"
            evidence.append(f"Sodium {blood.sodium} mEq/L ({status})")
        if blood.potassium is not None and (blood.potassium < 3.5 or blood.potassium > 5.0):
            status = "low" if blood.potassium < 3.5 else "high"
            evidence.append(f"Potassium {blood.potassium} mEq/L ({status})")
        if blood.chloride is not None and (blood.chloride < 98 or blood.chloride > 106):
            status = "low" if blood.chloride < 98 else "high"
            evidence.append(f"Chloride {blood.chloride} mEq/L ({status})")
        if blood.calcium is not None and (blood.calcium < 8.5 or blood.calcium > 10.5):
            status = "low" if blood.calcium < 8.5 else "high"
            evidence.append(f"Calcium {blood.calcium} mg/dL ({status})")
        if blood.magnesium is not None and (blood.magnesium < 1.7 or blood.magnesium > 2.2):
            status = "low" if blood.magnesium < 1.7 else "high"
            evidence.append(f"Magnesium {blood.magnesium} mg/dL ({status})")
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_proteinuria(
        self, _blood: BloodTestPanel | None, urine: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if urine is None:
            return None
        evidence = []
        if urine.protein is not None and urine.protein > 14:
            evidence.append(f"Urine protein {urine.protein} mg/dL (positive)")
        if not evidence:
            return None
        severity = "moderate" if urine.protein is not None and urine.protein > 30 else "mild"
        return severity, evidence

    def _check_insulin_resistance(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        # Compute HOMA-IR if both fasting glucose and fasting insulin available
        homa_ir = None
        if blood.fasting_glucose is not None and blood.fasting_insulin is not None:
            homa_ir = (blood.fasting_glucose * blood.fasting_insulin) / 405.0
        # Use provided homa_ir if available and not computed
        if homa_ir is None and blood.homa_ir is not None:
            homa_ir = blood.homa_ir
        if blood.fasting_insulin is not None and blood.fasting_insulin > 25:
            evidence.append(f"Fasting insulin {blood.fasting_insulin} µIU/mL (>25)")
        if homa_ir is not None and homa_ir > 2.5:
            evidence.append(f"HOMA-IR {homa_ir:.2f} (>2.5)")
        # Secondary pattern: prediabetic glucose + high triglycerides
        if (blood.fasting_glucose is not None and 100 <= blood.fasting_glucose <= 125
                and blood.triglycerides is not None and blood.triglycerides > 150):
            evidence.append(
                f"Fasting glucose {blood.fasting_glucose} mg/dL (100-125) with "
                f"triglycerides {blood.triglycerides} mg/dL (>150)"
            )
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_folate_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.folate is not None and blood.folate < 3.0:
            severity_level = "severe" if blood.folate < 2.0 else "moderate"
            evidence.append(f"Folate {blood.folate} ng/mL (<3.0 = deficient)")
        # Elevated homocysteine with low-normal folate
        if (blood.homocysteine is not None and blood.homocysteine > 15
                and blood.folate is not None and blood.folate < 5.0):
            evidence.append(
                f"Homocysteine {blood.homocysteine} µmol/L (>15) with "
                f"folate {blood.folate} ng/mL (<5.0)"
            )
        if not evidence:
            return None
        severity = "severe" if (blood.folate is not None and blood.folate < 2.0) else (
            "moderate" if len(evidence) >= 2 else "mild"
        )
        return severity, evidence

    def _check_prehypertension(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.sodium is not None and blood.sodium > 145:
            evidence.append(f"Sodium {blood.sodium} mEq/L (>145)")
        if blood.potassium is not None and blood.potassium < 3.5:
            evidence.append(f"Potassium {blood.potassium} mEq/L (<3.5)")
        if blood.homocysteine is not None and blood.homocysteine > 12:
            evidence.append(f"Homocysteine {blood.homocysteine} µmol/L (>12)")
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_cardiac_risk(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        crp_high = blood.crp is not None and blood.crp > 3.0
        if crp_high:
            evidence.append(f"CRP {blood.crp} mg/L (>3.0)")
        # Lipid risk markers (only flag if CRP is elevated)
        lipid_risk = False
        if blood.total_cholesterol_hdl_ratio is not None and blood.total_cholesterol_hdl_ratio > 5.0:
            lipid_risk = True
            evidence.append(f"TC/HDL ratio {blood.total_cholesterol_hdl_ratio} (>5.0)")
        if blood.ldl_cholesterol is not None and blood.ldl_cholesterol > 160:
            lipid_risk = True
            evidence.append(f"LDL {blood.ldl_cholesterol} mg/dL (>160)")
        if blood.triglycerides is not None and blood.triglycerides > 200:
            lipid_risk = True
            evidence.append(f"Triglycerides {blood.triglycerides} mg/dL (>200)")
        # Elevated homocysteine as independent cardiac risk
        if blood.homocysteine is not None and blood.homocysteine > 15:
            evidence.append(f"Homocysteine {blood.homocysteine} µmol/L (>15)")
        # Require CRP + lipid risk, OR homocysteine alone
        has_crp_lipid = crp_high and lipid_risk
        has_homocysteine = blood.homocysteine is not None and blood.homocysteine > 15
        if not (has_crp_lipid or has_homocysteine):
            return None
        severity = "severe" if (has_crp_lipid and len(evidence) >= 4) else (
            "moderate" if len(evidence) >= 3 else "mild"
        )
        return severity, evidence

    def _check_vitamin_a_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        # Direct marker if available
        if blood.vitamin_a is not None and blood.vitamin_a < 20:
            severity_tag = "severe" if blood.vitamin_a < 10 else "moderate"
            evidence.append(f"Vitamin A {blood.vitamin_a} µg/dL (<20 = deficient)")
        # Co-occurrence pattern: low iron + low ferritin often co-occur with vitamin A deficiency
        if (blood.iron is not None and blood.iron < 60
                and blood.ferritin is not None and blood.ferritin < 30):
            evidence.append(
                f"Iron {blood.iron} µg/dL (<60) with ferritin {blood.ferritin} ng/mL (<30) "
                f"(often co-occurs with vitamin A deficiency)"
            )
        if not evidence:
            return None
        severity = "severe" if (blood.vitamin_a is not None and blood.vitamin_a < 10) else (
            "moderate" if len(evidence) >= 2 else "mild"
        )
        return severity, evidence

    def _check_vitamin_e_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        # Direct marker if available
        if blood.vitamin_e is not None and blood.vitamin_e < 5.0:
            evidence.append(f"Vitamin E {blood.vitamin_e} mg/L (<5.0 = deficient)")
        # Vitamin E is lipid-soluble; very low cholesterol impairs transport
        if blood.total_cholesterol is not None and blood.total_cholesterol < 150:
            evidence.append(
                f"Total cholesterol {blood.total_cholesterol} mg/dL (<150, "
                f"may impair fat-soluble vitamin transport)"
            )
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_zinc_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        # Direct marker if available
        if blood.zinc is not None and blood.zinc < 70:
            evidence.append(f"Zinc {blood.zinc} µg/dL (<70 = deficient)")
        # Co-occurrence pattern: low ALP + low albumin suggest zinc deficiency
        if (blood.alkaline_phosphatase is not None and blood.alkaline_phosphatase < 44
                and blood.albumin is not None and blood.albumin < 3.5):
            evidence.append(
                f"ALP {blood.alkaline_phosphatase} U/L (<44) with albumin "
                f"{blood.albumin} g/dL (<3.5) (suggestive of zinc deficiency)"
            )
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    def _check_calcium_magnesium_deficiency(
        self, blood: BloodTestPanel | None, _u: UrineTestPanel | None, _s: str, _a: int,
    ) -> tuple[str, list[str]] | None:
        if blood is None:
            return None
        evidence = []
        if blood.calcium is not None and blood.calcium < 8.5:
            evidence.append(f"Calcium {blood.calcium} mg/dL (<8.5 = low)")
        if blood.magnesium is not None and blood.magnesium < 1.7:
            evidence.append(f"Magnesium {blood.magnesium} mg/dL (<1.7 = low)")
        # Elevated phosphorus with low calcium suggests imbalance
        if (blood.phosphorus is not None and blood.phosphorus > 4.5
                and blood.calcium is not None and blood.calcium < 8.5):
            evidence.append(
                f"Phosphorus {blood.phosphorus} mg/dL (>4.5) with low calcium "
                f"(calcium-phosphorus imbalance)"
            )
        if not evidence:
            return None
        severity = "moderate" if len(evidence) >= 2 else "mild"
        return severity, evidence

    # -- abdomen scan parsing ------------------------------------------------

    def _parse_abdomen(self, notes: str | None) -> list[AbdomenFindingResponse]:
        if not notes:
            return []
        findings: list[AbdomenFindingResponse] = []
        text = notes.lower()
        seen_organs: set[str] = set()
        for pattern, info in _ABDOMEN_PATTERNS:
            if re.search(pattern, text):
                organ = info["organ"]
                # Avoid duplicate findings for the same organ
                if organ in seen_organs:
                    continue
                seen_organs.add(organ)
                findings.append(AbdomenFindingResponse(
                    organ=info["organ"],
                    finding=info["finding"],
                    severity=info["severity"],
                    dietary_impact=info["dietary_impact"],
                    foods_to_avoid=info.get("foods_to_avoid", []),
                    foods_to_increase=info.get("foods_to_increase", []),
                ))
        return findings

    # -- health score --------------------------------------------------------

    def _compute_health_score(
        self, lab_results: list[LabResultResponse],
    ) -> tuple[float, dict[str, float]]:
        """Compute overall health score (0–100) from lab results.

        Each tested category starts at 100 and is penalised for abnormal values.
        Categories are then weighted to produce the final score.
        """
        if not lab_results:
            return 0.0, {}

        # Group results by category
        by_cat: dict[str, list[LabResultResponse]] = {}
        for r in lab_results:
            by_cat.setdefault(r.category, []).append(r)

        breakdown: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for cat, results in by_cat.items():
            cat_score = 100.0
            for r in results:
                if r.status == "critical_low" or r.status == "critical_high":
                    cat_score -= 25.0
                elif r.status == "high" or r.status == "low":
                    # Penalise proportionally to how far out of range
                    if r.status == "high" and r.reference_high > 0:
                        deviation = (r.value - r.reference_high) / r.reference_high
                    elif r.status == "low" and r.reference_low > 0:
                        deviation = (r.reference_low - r.value) / r.reference_low
                    else:
                        deviation = 0.1
                    penalty = min(20.0, max(5.0, deviation * 50.0))
                    cat_score -= penalty

            cat_score = max(0.0, min(100.0, cat_score))
            breakdown[cat] = round(cat_score, 1)
            weight = _CATEGORY_WEIGHTS.get(cat, 0.05)
            total_weight += weight
            weighted_sum += weight * cat_score

        if total_weight == 0:
            return 0.0, breakdown

        overall = round(weighted_sum / total_weight, 1)
        return overall, breakdown

    # -- diet generation -----------------------------------------------------

    def _generate_diet(
        self,
        all_conditions: list[str],
        findings: list[HealthFindingResponse],
        abdomen_findings: list[AbdomenFindingResponse],
        region: str,
        age: int,
        sex: str,
        dietary_restrictions: list[str],
        known_variants: list[str],
        calorie_target: int,
        meal_plan_days: int,
    ) -> tuple[DietRecommendation | None, list[str], int]:
        """Generate diet plan integrating health findings with DietAdvisor."""
        # Map health conditions to genetic_risks for DietAdvisor
        genetic_risks = self._conditions_to_risks(all_conditions)

        # Build variant dict from known_variants (list of rsids)
        variants: dict[str, str] = {}
        for v in known_variants:
            # Accept "rs12345:CT" or just "rs12345"
            if ":" in v:
                rsid, geno = v.split(":", 1)
                variants[rsid.strip()] = geno.strip()
            else:
                variants[v.strip()] = "unknown"

        try:
            recommendations = self._advisor.generate_recommendations(
                genetic_risks=genetic_risks,
                variants=variants,
                region=region,
                age=age,
                sex=sex,
                dietary_restrictions=dietary_restrictions,
            )

            meal_plans = self._advisor.create_meal_plan(
                recommendations=recommendations,
                region=region,
                calories=calorie_target,
                days=meal_plan_days,
                dietary_restrictions=dietary_restrictions,
            )

            if dietary_restrictions:
                meal_plans = self._advisor.adapt_to_restrictions(meal_plans, dietary_restrictions)

        except Exception:
            logger.exception("DietAdvisor failed; falling back to summary-only")
            recommendations = []
            meal_plans = []

        # Build modifications list from findings
        modifications: list[str] = []
        all_increase: list[str] = []
        all_avoid: list[str] = []

        for f in findings:
            modifications.append(f"{f.display_name}: {f.dietary_impact}")
            all_increase.extend(f.foods_to_increase)
            all_avoid.extend(f.foods_to_avoid)

        for af in abdomen_findings:
            modifications.append(f"{af.finding}: {af.dietary_impact}")
            all_increase.extend(af.foods_to_increase)
            all_avoid.extend(af.foods_to_avoid)

        # Calorie adjustment based on conditions
        calorie_adj = 0
        fatty_liver = any(f.condition == "fatty_liver" for f in findings)
        diabetes = any(f.condition == "diabetes" for f in findings)
        if fatty_liver or diabetes:
            calorie_adj = -200  # Reduce by 200 kcal

        # Build the summary
        summary_parts = []
        if findings:
            summary_parts.append(
                f"Based on {len(findings)} detected health finding(s), "
                f"your diet plan has been personalised."
            )
        if recommendations:
            top_nutrients = [r.nutrient for r in recommendations[:5]]
            summary_parts.append(f"Key nutrients: {', '.join(top_nutrients)}.")
        if not summary_parts:
            summary_parts.append("Standard balanced diet recommended for your region and profile.")

        # Deduplicate food lists
        unique_increase = list(dict.fromkeys(all_increase))
        unique_avoid = list(dict.fromkeys(all_avoid))

        # Build key nutrients from recommendations
        key_nutrients = [r.nutrient for r in recommendations[:10]]

        # Convert DietAdvisor MealPlan dataclasses to Pydantic MealPlan models.
        # Pydantic MealPlan: day(str), breakfast(str), lunch(str), dinner(str), snacks(list[str])
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pydantic_meals: list[MealPlan] = []
        for mp in meal_plans:
            def _meal_str(items: list) -> str:
                return ", ".join(f"{fi.name} ({g:.0f}g)" for fi, g in items[:4]) or "Seasonal selection"

            if isinstance(mp.day, int):
                week_num = (mp.day - 1) // 7 + 1
                day_label = day_names[(mp.day - 1) % 7]
                label = f"Week {week_num} — {day_label}" if len(meal_plans) > 7 else day_label
            else:
                label = str(mp.day)

            pydantic_meals.append(MealPlan(
                day=label,
                breakfast=_meal_str(mp.breakfast),
                lunch=_meal_str(mp.lunch),
                dinner=_meal_str(mp.dinner),
                snacks=[_meal_str(mp.snacks)] if mp.snacks else [],
            ))

        diet_rec = DietRecommendation(
            summary=" ".join(summary_parts),
            key_nutrients=key_nutrients,
            foods_to_increase=unique_increase[:20],
            foods_to_avoid=unique_avoid[:20],
            meal_plans=pydantic_meals,
            calorie_target=calorie_target + calorie_adj,
        )

        return diet_rec, modifications, calorie_adj

    def _conditions_to_risks(self, conditions: list[str]) -> list[str]:
        """Map detected conditions to risk names the DietAdvisor understands."""
        mapping: dict[str, str] = _load_json("condition_risk_mapping.json")
        risks = []
        for cond in conditions:
            risk = mapping.get(cond)
            if risk and risk not in risks:
                risks.append(risk)
            elif cond not in mapping:
                # Pass through user-reported conditions as-is
                if cond not in risks:
                    risks.append(cond)
        return risks
