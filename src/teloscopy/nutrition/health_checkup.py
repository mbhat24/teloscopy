"""Health checkup data processing for diet plan integration.

Parses blood test, urine test, and abdomen scan findings to generate
personalised dietary adjustments. All reference ranges are age/sex-stratified
per standard clinical laboratory guidelines.

Typical usage::

    result = process_health_checkup(
        blood_data={"hemoglobin": 11.2, "fasting_glucose": 130, ...},
        urine_data={"ph": 6.0, "protein": 0.0, ...},
        abdomen_text="Mild fatty liver grade I. No gallstones.",
        age=45,
        sex="male",
    )
    overrides = get_diet_advisor_overrides(result)
    # Pass *overrides* straight into DietAdvisor.plan(...)
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` for Python < 3.11."""

        def __new__(cls, value: str) -> "StrEnum":
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return self.value


# ---------------------------------------------------------------------------
# Path to bundled JSON data files
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"


# ---------------------------------------------------------------------------
# Core enumerations & data-classes
# ---------------------------------------------------------------------------

class LabStatus(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"


@dataclass(frozen=True)
class ReferenceRange:
    """Age/sex-specific reference range for a lab parameter."""
    low: float
    high: float
    unit: str
    critical_low: float | None = None
    critical_high: float | None = None


@dataclass
class LabResult:
    """A single lab test result with interpretation."""
    parameter: str
    value: float
    unit: str
    status: LabStatus
    reference_range: ReferenceRange
    display_name: str = ""
    category: str = ""


@dataclass
class HealthFinding:
    """A health condition or concern identified from lab data."""
    condition: str
    severity: str
    evidence: list[str]
    dietary_impact: str
    nutrients_to_increase: list[str] = field(default_factory=list)
    nutrients_to_decrease: list[str] = field(default_factory=list)
    foods_to_increase: list[str] = field(default_factory=list)
    foods_to_avoid: list[str] = field(default_factory=list)
    calorie_adjustment: int = 0


@dataclass
class AbdomenFinding:
    """Findings from abdomen scan that affect diet."""
    organ: str
    finding: str
    severity: str
    dietary_impact: str
    foods_to_avoid: list[str] = field(default_factory=list)
    foods_to_increase: list[str] = field(default_factory=list)


@dataclass
class HealthCheckupResult:
    """Complete processed health checkup data."""
    lab_results: list[LabResult]
    findings: list[HealthFinding]
    abdomen_findings: list[AbdomenFinding]
    overall_health_score: float
    detected_conditions: list[str]
    nutrient_adjustments: dict[str, float]
    dietary_modifications: list[str]
    calorie_adjustment: int


# ---------------------------------------------------------------------------
# Parameter metadata  (display_name, category)  – loaded from JSON
# ---------------------------------------------------------------------------

def _load_parameter_meta() -> dict[str, tuple[str, str]]:
    """Load parameter metadata from ``parameter_metadata.json``."""
    with open(_DATA_DIR / "parameter_metadata.json", encoding="utf-8") as fh:
        raw: dict[str, dict[str, str]] = json.load(fh)
    return {key: (val["display_name"], val["category"]) for key, val in raw.items()}


_PARAMETER_META: dict[str, tuple[str, str]] = _load_parameter_meta()


# ---------------------------------------------------------------------------
# Reference-range builders  (age/sex stratified)
# ---------------------------------------------------------------------------
# Each entry in the following lookup is a callable
#     (age: int, sex: str) -> ReferenceRange
# This lets us represent age- and sex-dependent ranges compactly.
# ---------------------------------------------------------------------------

def _static(
    low: float,
    high: float,
    unit: str,
    critical_low: float | None = None,
    critical_high: float | None = None,
) -> Callable[[int, str], ReferenceRange]:
    """Return a factory for a parameter whose range is independent of age/sex."""
    _rr = ReferenceRange(low, high, unit, critical_low, critical_high)

    def _factory(_age: int, _sex: str) -> ReferenceRange:
        return _rr

    return _factory


def _sex_based(
    male: tuple[float, float],
    female: tuple[float, float],
    unit: str,
    critical_low: float | None = None,
    critical_high: float | None = None,
    male_critical: tuple[float | None, float | None] | None = None,
    female_critical: tuple[float | None, float | None] | None = None,
) -> Callable[[int, str], ReferenceRange]:
    """Return a factory for a parameter whose range differs by sex only."""

    def _factory(_age: int, sex: str) -> ReferenceRange:
        if sex == "female":
            cl = female_critical[0] if female_critical else critical_low
            ch = female_critical[1] if female_critical else critical_high
            return ReferenceRange(female[0], female[1], unit, cl, ch)
        cl = male_critical[0] if male_critical else critical_low
        ch = male_critical[1] if male_critical else critical_high
        return ReferenceRange(male[0], male[1], unit, cl, ch)

    return _factory


def _age_sex_based(
    rules: list[tuple[int, int, str | None, float, float]],
    unit: str,
    default_low: float,
    default_high: float,
    critical_low: float | None = None,
    critical_high: float | None = None,
) -> Callable[[int, str], ReferenceRange]:
    """Return a factory that picks a range from a list of (min_age, max_age, sex|None, low, high) rules.

    The first matching rule wins.  *sex=None* means "any sex".
    """

    def _factory(age: int, sex: str) -> ReferenceRange:
        for min_age, max_age, rule_sex, low, high in rules:
            if min_age <= age <= max_age and (rule_sex is None or rule_sex == sex):
                return ReferenceRange(low, high, unit, critical_low, critical_high)
        return ReferenceRange(default_low, default_high, unit, critical_low, critical_high)

    return _factory


# ---------------------------------------------------------------------------
# BLOOD_TEST_REFERENCE_RANGES
# ---------------------------------------------------------------------------

BLOOD_TEST_REFERENCE_RANGES: dict[str, Callable[[int, str], ReferenceRange]] = {
    # ── CBC ────────────────────────────────────────────────────────────────
    "hemoglobin": _age_sex_based(
        [
            (0, 1, None, 9.5, 14.0),
            (1, 6, None, 10.5, 14.0),
            (6, 12, None, 11.5, 15.5),
            (12, 18, "male", 13.0, 16.0),
            (12, 18, "female", 12.0, 16.0),
            (18, 50, "male", 13.5, 17.5),
            (18, 50, "female", 12.0, 16.0),
            (50, 65, "male", 13.0, 17.0),
            (50, 65, "female", 11.5, 15.5),
            (65, 120, "male", 12.5, 16.5),
            (65, 120, "female", 11.0, 15.0),
        ],
        unit="g/dL",
        default_low=12.0,
        default_high=17.5,
        critical_low=7.0,
        critical_high=20.0,
    ),
    "rbc_count": _age_sex_based(
        [
            (0, 1, None, 3.8, 5.5),
            (1, 12, None, 4.0, 5.5),
            (12, 18, "male", 4.5, 5.5),
            (12, 18, "female", 4.0, 5.0),
            (18, 65, "male", 4.5, 5.9),
            (18, 65, "female", 4.0, 5.2),
            (65, 120, "male", 4.2, 5.7),
            (65, 120, "female", 3.8, 5.0),
        ],
        unit="million/uL",
        default_low=4.0,
        default_high=5.5,
        critical_low=2.5,
        critical_high=7.5,
    ),
    "wbc_count": _age_sex_based(
        [
            (0, 1, None, 6.0, 17.5),
            (1, 6, None, 5.0, 15.5),
            (6, 12, None, 4.5, 13.5),
            (12, 18, None, 4.5, 11.0),
            (18, 120, None, 4.0, 11.0),
        ],
        unit="x10^3/uL",
        default_low=4.0,
        default_high=11.0,
        critical_low=2.0,
        critical_high=30.0,
    ),
    "platelet_count": _age_sex_based(
        [
            (0, 1, None, 150.0, 400.0),
            (1, 18, None, 150.0, 400.0),
            (18, 120, None, 150.0, 400.0),
        ],
        unit="x10^3/uL",
        default_low=150.0,
        default_high=400.0,
        critical_low=50.0,
        critical_high=1000.0,
    ),
    "hematocrit": _age_sex_based(
        [
            (0, 1, None, 28.0, 42.0),
            (1, 12, None, 33.0, 43.0),
            (12, 18, "male", 37.0, 49.0),
            (12, 18, "female", 36.0, 46.0),
            (18, 50, "male", 40.0, 54.0),
            (18, 50, "female", 36.0, 48.0),
            (50, 65, "male", 38.0, 52.0),
            (50, 65, "female", 35.0, 47.0),
            (65, 120, "male", 37.0, 51.0),
            (65, 120, "female", 34.0, 46.0),
        ],
        unit="%",
        default_low=36.0,
        default_high=54.0,
        critical_low=20.0,
        critical_high=60.0,
    ),
    "mcv": _age_sex_based(
        [
            (0, 1, None, 70.0, 100.0),
            (1, 6, None, 72.0, 88.0),
            (6, 12, None, 76.0, 90.0),
            (12, 18, None, 78.0, 98.0),
            (18, 120, None, 80.0, 100.0),
        ],
        unit="fL",
        default_low=80.0,
        default_high=100.0,
        critical_low=60.0,
        critical_high=120.0,
    ),
    "mch": _static(27.0, 33.0, "pg", 20.0, 40.0),
    "mchc": _static(32.0, 36.0, "g/dL", 28.0, 40.0),
    "rdw": _static(11.5, 14.5, "%", None, 20.0),
    "neutrophils": _age_sex_based(
        [
            (0, 1, None, 20.0, 50.0),
            (1, 12, None, 30.0, 60.0),
            (12, 120, None, 40.0, 70.0),
        ],
        unit="%",
        default_low=40.0,
        default_high=70.0,
        critical_low=10.0,
        critical_high=90.0,
    ),
    "lymphocytes": _age_sex_based(
        [
            (0, 1, None, 40.0, 70.0),
            (1, 12, None, 30.0, 50.0),
            (12, 120, None, 20.0, 40.0),
        ],
        unit="%",
        default_low=20.0,
        default_high=40.0,
        critical_low=5.0,
        critical_high=80.0,
    ),
    "monocytes": _static(2.0, 8.0, "%"),
    "eosinophils": _static(1.0, 4.0, "%", None, 15.0),
    "basophils": _static(0.0, 1.0, "%", None, 5.0),

    # ── Lipid Panel ───────────────────────────────────────────────────────
    "total_cholesterol": _age_sex_based(
        [
            (0, 18, None, 120.0, 200.0),
            (18, 40, None, 125.0, 200.0),
            (40, 65, None, 130.0, 200.0),
            (65, 120, None, 130.0, 220.0),
        ],
        unit="mg/dL",
        default_low=125.0,
        default_high=200.0,
        critical_high=300.0,
    ),
    "ldl_cholesterol": _age_sex_based(
        [
            (0, 18, None, 50.0, 110.0),
            (18, 40, None, 50.0, 100.0),
            (40, 65, None, 50.0, 100.0),
            (65, 120, None, 50.0, 100.0),
        ],
        unit="mg/dL",
        default_low=50.0,
        default_high=100.0,
        critical_high=190.0,
    ),
    "hdl_cholesterol": _sex_based(
        male=(40.0, 60.0),
        female=(50.0, 70.0),
        unit="mg/dL",
        critical_low=20.0,
    ),
    "triglycerides": _age_sex_based(
        [
            (0, 18, None, 30.0, 130.0),
            (18, 40, None, 40.0, 150.0),
            (40, 120, None, 40.0, 150.0),
        ],
        unit="mg/dL",
        default_low=40.0,
        default_high=150.0,
        critical_high=500.0,
    ),
    "vldl": _static(5.0, 40.0, "mg/dL", None, 80.0),
    "total_cholesterol_hdl_ratio": _sex_based(
        male=(3.0, 5.0),
        female=(2.5, 4.5),
        unit="ratio",
        critical_high=7.0,
    ),

    # ── Liver Function ────────────────────────────────────────────────────
    "sgot_ast": _age_sex_based(
        [
            (0, 1, None, 15.0, 60.0),
            (1, 18, None, 10.0, 40.0),
            (18, 60, "male", 10.0, 40.0),
            (18, 60, "female", 9.0, 32.0),
            (60, 120, "male", 10.0, 45.0),
            (60, 120, "female", 10.0, 35.0),
        ],
        unit="U/L",
        default_low=10.0,
        default_high=40.0,
        critical_high=1000.0,
    ),
    "sgpt_alt": _age_sex_based(
        [
            (0, 1, None, 10.0, 55.0),
            (1, 18, None, 7.0, 35.0),
            (18, 60, "male", 7.0, 56.0),
            (18, 60, "female", 7.0, 45.0),
            (60, 120, "male", 7.0, 50.0),
            (60, 120, "female", 7.0, 40.0),
        ],
        unit="U/L",
        default_low=7.0,
        default_high=56.0,
        critical_high=1000.0,
    ),
    "alkaline_phosphatase": _age_sex_based(
        [
            (0, 1, None, 150.0, 420.0),
            (1, 10, None, 100.0, 350.0),
            (10, 18, "male", 100.0, 390.0),
            (10, 18, "female", 50.0, 300.0),
            (18, 50, None, 44.0, 147.0),
            (50, 65, None, 50.0, 160.0),
            (65, 120, None, 55.0, 170.0),
        ],
        unit="U/L",
        default_low=44.0,
        default_high=147.0,
        critical_high=800.0,
    ),
    "total_bilirubin": _age_sex_based(
        [
            (0, 1, None, 0.1, 12.0),
            (1, 120, None, 0.1, 1.2),
        ],
        unit="mg/dL",
        default_low=0.1,
        default_high=1.2,
        critical_high=15.0,
    ),
    "direct_bilirubin": _static(0.0, 0.3, "mg/dL", None, 5.0),
    "ggt": _sex_based(
        male=(8.0, 61.0),
        female=(5.0, 36.0),
        unit="U/L",
        critical_high=500.0,
    ),
    "total_protein": _age_sex_based(
        [
            (0, 1, None, 4.8, 7.6),
            (1, 18, None, 5.7, 8.0),
            (18, 120, None, 6.0, 8.3),
        ],
        unit="g/dL",
        default_low=6.0,
        default_high=8.3,
        critical_low=4.0,
        critical_high=12.0,
    ),
    "albumin": _age_sex_based(
        [
            (0, 1, None, 2.5, 4.5),
            (1, 18, None, 3.5, 5.0),
            (18, 60, None, 3.5, 5.5),
            (60, 120, None, 3.2, 5.0),
        ],
        unit="g/dL",
        default_low=3.5,
        default_high=5.5,
        critical_low=2.0,
    ),
    "globulin": _static(2.0, 3.5, "g/dL", 1.0, 5.5),
    "ag_ratio": _static(1.0, 2.2, "ratio", 0.5, None),

    # ── Kidney Function ───────────────────────────────────────────────────
    "blood_urea": _age_sex_based(
        [
            (0, 12, None, 10.0, 36.0),
            (12, 18, None, 12.0, 40.0),
            (18, 60, None, 13.0, 43.0),
            (60, 120, None, 15.0, 50.0),
        ],
        unit="mg/dL",
        default_low=13.0,
        default_high=43.0,
        critical_high=100.0,
    ),
    "serum_creatinine": _age_sex_based(
        [
            (0, 12, None, 0.2, 0.6),
            (12, 18, "male", 0.5, 1.0),
            (12, 18, "female", 0.4, 0.9),
            (18, 60, "male", 0.7, 1.3),
            (18, 60, "female", 0.6, 1.1),
            (60, 120, "male", 0.8, 1.4),
            (60, 120, "female", 0.7, 1.2),
        ],
        unit="mg/dL",
        default_low=0.6,
        default_high=1.3,
        critical_high=10.0,
    ),
    "uric_acid": _sex_based(
        male=(3.5, 7.2),
        female=(2.6, 6.0),
        unit="mg/dL",
        critical_high=12.0,
    ),
    "bun": _age_sex_based(
        [
            (0, 12, None, 5.0, 18.0),
            (12, 18, None, 6.0, 20.0),
            (18, 60, None, 7.0, 20.0),
            (60, 120, None, 8.0, 23.0),
        ],
        unit="mg/dL",
        default_low=7.0,
        default_high=20.0,
        critical_high=60.0,
    ),
    "egfr": _age_sex_based(
        [
            (0, 18, None, 90.0, 150.0),
            (18, 40, None, 90.0, 130.0),
            (40, 60, None, 80.0, 120.0),
            (60, 80, None, 60.0, 110.0),
            (80, 120, None, 50.0, 100.0),
        ],
        unit="mL/min/1.73m2",
        default_low=60.0,
        default_high=120.0,
        critical_low=15.0,
    ),
    "bun_creatinine_ratio": _static(10.0, 20.0, "ratio", 5.0, 35.0),

    # ── Diabetes Panel ────────────────────────────────────────────────────
    "fasting_glucose": _age_sex_based(
        [
            (0, 12, None, 60.0, 100.0),
            (12, 18, None, 65.0, 100.0),
            (18, 60, None, 70.0, 100.0),
            (60, 120, None, 70.0, 110.0),
        ],
        unit="mg/dL",
        default_low=70.0,
        default_high=100.0,
        critical_low=40.0,
        critical_high=400.0,
    ),
    "hba1c": _static(4.0, 5.6, "%", None, 14.0),
    "postprandial_glucose": _age_sex_based(
        [
            (0, 18, None, 70.0, 120.0),
            (18, 60, None, 70.0, 140.0),
            (60, 120, None, 70.0, 150.0),
        ],
        unit="mg/dL",
        default_low=70.0,
        default_high=140.0,
        critical_low=40.0,
        critical_high=500.0,
    ),
    "fasting_insulin": _static(2.6, 24.9, "uIU/mL", None, 100.0),
    "homa_ir": _static(0.5, 2.5, "index", None, 10.0),

    # ── Thyroid ───────────────────────────────────────────────────────────
    "tsh": _age_sex_based(
        [
            (0, 1, None, 0.7, 11.0),
            (1, 6, None, 0.7, 6.0),
            (6, 18, None, 0.6, 5.0),
            (18, 50, None, 0.4, 4.0),
            (50, 80, None, 0.4, 4.5),
            (80, 120, None, 0.4, 5.0),
        ],
        unit="mIU/L",
        default_low=0.4,
        default_high=4.0,
        critical_low=0.01,
        critical_high=50.0,
    ),
    "t3": _static(80.0, 200.0, "ng/dL", 40.0, 400.0),
    "t4": _static(5.0, 12.0, "ug/dL", 2.0, 20.0),
    "free_t3": _static(2.3, 4.2, "pg/mL", 1.0, 8.0),
    "free_t4": _static(0.8, 1.8, "ng/dL", 0.4, 5.0),

    # ── Vitamins ──────────────────────────────────────────────────────────
    "vitamin_d": _age_sex_based(
        [
            (0, 18, None, 20.0, 70.0),
            (18, 65, None, 30.0, 100.0),
            (65, 120, None, 30.0, 80.0),
        ],
        unit="ng/mL",
        default_low=30.0,
        default_high=100.0,
        critical_low=10.0,
        critical_high=150.0,
    ),
    "vitamin_b12": _age_sex_based(
        [
            (0, 18, None, 200.0, 900.0),
            (18, 60, None, 200.0, 900.0),
            (60, 120, None, 250.0, 900.0),
        ],
        unit="pg/mL",
        default_low=200.0,
        default_high=900.0,
        critical_low=100.0,
    ),
    "folate": _static(3.0, 17.0, "ng/mL", 1.5, None),
    "vitamin_a": _sex_based(
        male=(20.0, 80.0),
        female=(20.0, 80.0),
        unit="ug/dL",
        critical_low=10.0,
        critical_high=120.0,
    ),
    "vitamin_e": _static(5.0, 18.0, "mg/L", 3.0, 40.0),
    "vitamin_k": _static(0.1, 2.2, "ng/mL"),

    # ── Minerals / Electrolytes ───────────────────────────────────────────
    "iron": _sex_based(
        male=(65.0, 175.0),
        female=(50.0, 170.0),
        unit="ug/dL",
        critical_low=20.0,
        critical_high=300.0,
    ),
    "ferritin": _age_sex_based(
        [
            (0, 5, None, 7.0, 140.0),
            (5, 12, None, 10.0, 120.0),
            (12, 18, "male", 12.0, 150.0),
            (12, 18, "female", 10.0, 120.0),
            (18, 50, "male", 20.0, 300.0),
            (18, 50, "female", 12.0, 150.0),
            (50, 120, "male", 20.0, 350.0),
            (50, 120, "female", 20.0, 200.0),
        ],
        unit="ng/mL",
        default_low=12.0,
        default_high=300.0,
        critical_low=5.0,
        critical_high=1000.0,
    ),
    "tibc": _static(250.0, 400.0, "ug/dL", 150.0, 600.0),
    "transferrin_saturation": _sex_based(
        male=(20.0, 50.0),
        female=(15.0, 50.0),
        unit="%",
        critical_low=10.0,
        critical_high=70.0,
    ),
    "calcium": _age_sex_based(
        [
            (0, 1, None, 8.8, 11.0),
            (1, 18, None, 8.8, 10.8),
            (18, 60, None, 8.5, 10.5),
            (60, 120, None, 8.4, 10.2),
        ],
        unit="mg/dL",
        default_low=8.5,
        default_high=10.5,
        critical_low=6.0,
        critical_high=13.0,
    ),
    "phosphorus": _age_sex_based(
        [
            (0, 1, None, 4.5, 6.7),
            (1, 12, None, 3.7, 5.6),
            (12, 18, None, 2.9, 5.4),
            (18, 120, None, 2.5, 4.5),
        ],
        unit="mg/dL",
        default_low=2.5,
        default_high=4.5,
        critical_low=1.0,
        critical_high=8.0,
    ),
    "magnesium": _static(1.7, 2.2, "mg/dL", 1.0, 4.0),
    "sodium": _age_sex_based(
        [
            (0, 1, None, 133.0, 146.0),
            (1, 18, None, 135.0, 145.0),
            (18, 120, None, 136.0, 145.0),
        ],
        unit="mEq/L",
        default_low=136.0,
        default_high=145.0,
        critical_low=120.0,
        critical_high=160.0,
    ),
    "potassium": _age_sex_based(
        [
            (0, 1, None, 3.7, 5.9),
            (1, 18, None, 3.4, 4.7),
            (18, 120, None, 3.5, 5.0),
        ],
        unit="mEq/L",
        default_low=3.5,
        default_high=5.0,
        critical_low=2.5,
        critical_high=6.5,
    ),
    "chloride": _static(98.0, 106.0, "mEq/L", 80.0, 120.0),
    "zinc": _sex_based(
        male=(75.0, 120.0),
        female=(65.0, 115.0),
        unit="ug/dL",
        critical_low=40.0,
    ),
    "copper": _sex_based(
        male=(70.0, 140.0),
        female=(80.0, 155.0),
        unit="ug/dL",
        critical_low=30.0,
        critical_high=250.0,
    ),

    # ── Inflammation ──────────────────────────────────────────────────────
    "crp": _static(0.0, 3.0, "mg/L", None, 100.0),
    "esr": _age_sex_based(
        [
            (0, 18, "male", 0.0, 10.0),
            (0, 18, "female", 0.0, 15.0),
            (18, 50, "male", 0.0, 15.0),
            (18, 50, "female", 0.0, 20.0),
            (50, 120, "male", 0.0, 20.0),
            (50, 120, "female", 0.0, 30.0),
        ],
        unit="mm/hr",
        default_low=0.0,
        default_high=20.0,
        critical_high=100.0,
    ),
    "homocysteine": _age_sex_based(
        [
            (0, 18, None, 4.0, 12.0),
            (18, 60, None, 5.0, 15.0),
            (60, 120, None, 5.0, 20.0),
        ],
        unit="umol/L",
        default_low=5.0,
        default_high=15.0,
        critical_high=50.0,
    ),

    # ── Cardiac markers ───────────────────────────────────────────────────
    "troponin": _static(0.0, 0.04, "ng/mL", None, 0.4),
    "bnp": _age_sex_based(
        [
            (0, 50, None, 0.0, 100.0),
            (50, 75, None, 0.0, 200.0),
            (75, 120, None, 0.0, 400.0),
        ],
        unit="pg/mL",
        default_low=0.0,
        default_high=100.0,
        critical_high=1000.0,
    ),
    "ldh": _age_sex_based(
        [
            (0, 1, None, 160.0, 450.0),
            (1, 12, None, 120.0, 300.0),
            (12, 18, None, 100.0, 250.0),
            (18, 120, None, 120.0, 246.0),
        ],
        unit="U/L",
        default_low=120.0,
        default_high=246.0,
        critical_high=1000.0,
    ),
}


# ---------------------------------------------------------------------------
# URINE_TEST_REFERENCE_RANGES  – loaded from JSON
# ---------------------------------------------------------------------------

def _load_urine_ranges() -> dict[str, Callable[[int, str], ReferenceRange]]:
    """Load urine reference ranges from ``urine_ranges_nutrition.json``.

    All urine parameters are age/sex-independent so each entry is
    reconstructed as a ``_static`` factory.
    """
    with open(_DATA_DIR / "urine_ranges_nutrition.json", encoding="utf-8") as fh:
        raw: dict[str, dict[str, Any]] = json.load(fh)
    return {
        param: _static(
            data["low"],
            data["high"],
            data["unit"],
            data.get("critical_low"),
            data.get("critical_high"),
        )
        for param, data in raw.items()
    }


URINE_TEST_REFERENCE_RANGES: dict[str, Callable[[int, str], ReferenceRange]] = _load_urine_ranges()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_reference_range(parameter: str, age: int, sex: str) -> ReferenceRange:
    """Look up the reference range for *parameter* given patient *age* and *sex*.

    Parameters
    ----------
    parameter : str
        Lab parameter key (e.g., ``"hemoglobin"``).
    age : int
        Age in years.
    sex : str
        ``"male"`` or ``"female"``.

    Returns
    -------
    ReferenceRange
        The age/sex-stratified reference range.

    Raises
    ------
    KeyError
        If the parameter is not recognised.
    """
    sex = sex.lower().strip()
    if parameter in BLOOD_TEST_REFERENCE_RANGES:
        return BLOOD_TEST_REFERENCE_RANGES[parameter](age, sex)
    if parameter in URINE_TEST_REFERENCE_RANGES:
        return URINE_TEST_REFERENCE_RANGES[parameter](age, sex)
    raise KeyError(f"Unknown lab parameter: {parameter!r}")


# ---------------------------------------------------------------------------
# 3. interpret_lab_value
# ---------------------------------------------------------------------------

def interpret_lab_value(
    parameter: str,
    value: float,
    age: int,
    sex: str,
) -> LabResult:
    """Return a :class:`LabResult` for a single test value.

    Parameters
    ----------
    parameter : str
        Lab parameter key (e.g., ``"hemoglobin"``).
    value : float
        Measured value.
    age : int
        Patient age in years.
    sex : str
        ``"male"`` or ``"female"``.

    Returns
    -------
    LabResult
    """
    rr = get_reference_range(parameter, age, sex)
    status = _classify_value(value, rr)
    display_name, category = _PARAMETER_META.get(parameter, (parameter, "other"))
    return LabResult(
        parameter=parameter,
        value=value,
        unit=rr.unit,
        status=status,
        reference_range=rr,
        display_name=display_name,
        category=category,
    )


def _classify_value(value: float, rr: ReferenceRange) -> LabStatus:
    """Classify *value* relative to *rr*."""
    if rr.critical_low is not None and value < rr.critical_low:
        return LabStatus.CRITICAL_LOW
    if rr.critical_high is not None and value > rr.critical_high:
        return LabStatus.CRITICAL_HIGH
    if value < rr.low:
        return LabStatus.LOW
    if value > rr.high:
        return LabStatus.HIGH
    return LabStatus.NORMAL


# ---------------------------------------------------------------------------
# Helpers for detect_health_findings
# ---------------------------------------------------------------------------

def _lab_map(results: list[LabResult]) -> dict[str, LabResult]:
    """Build a *parameter -> LabResult* mapping for O(1) lookups."""
    return {r.parameter: r for r in results}


def _is(lr: LabResult | None, *statuses: LabStatus) -> bool:
    """Return True if the lab result has one of the given statuses."""
    return lr is not None and lr.status in statuses


def _val(lr: LabResult | None) -> float:
    """Return the value of a lab result or NaN if missing."""
    return lr.value if lr is not None else float("nan")


# ---------------------------------------------------------------------------
# 4. detect_health_findings
# ---------------------------------------------------------------------------

def detect_health_findings(lab_results: list[LabResult]) -> list[HealthFinding]:
    """Analyse a collection of :class:`LabResult` objects and identify
    actionable health findings.

    The function is purely CPU-bound and allocates no I/O.

    Parameters
    ----------
    lab_results : list[LabResult]
        Interpreted lab results (output of :func:`interpret_lab_value`).

    Returns
    -------
    list[HealthFinding]
    """
    lm = _lab_map(lab_results)
    findings: list[HealthFinding] = []

    # -- Iron-deficiency anaemia ----------------------------------------
    _detect_iron_deficiency_anemia(lm, findings)
    # -- Macrocytic (B12/folate) anaemia --------------------------------
    _detect_macrocytic_anemia(lm, findings)
    # -- Dyslipidemia ---------------------------------------------------
    _detect_dyslipidemia(lm, findings)
    # -- Prediabetes / Diabetes -----------------------------------------
    _detect_diabetes(lm, findings)
    # -- Hypothyroidism -------------------------------------------------
    _detect_hypothyroidism(lm, findings)
    # -- Hyperthyroidism ------------------------------------------------
    _detect_hyperthyroidism(lm, findings)
    # -- Vitamin D deficiency -------------------------------------------
    _detect_vitamin_d_deficiency(lm, findings)
    # -- B12 deficiency -------------------------------------------------
    _detect_b12_deficiency(lm, findings)
    # -- Folate deficiency ----------------------------------------------
    _detect_folate_deficiency(lm, findings)
    # -- Liver stress ---------------------------------------------------
    _detect_liver_stress(lm, findings)
    # -- Kidney impairment ----------------------------------------------
    _detect_kidney_impairment(lm, findings)
    # -- Hyperuricemia / gout risk --------------------------------------
    _detect_hyperuricemia(lm, findings)
    # -- Inflammation ---------------------------------------------------
    _detect_inflammation(lm, findings)
    # -- Fatty liver indicators -----------------------------------------
    _detect_fatty_liver_indicators(lm, findings)
    # -- Electrolyte imbalance ------------------------------------------
    _detect_electrolyte_imbalance(lm, findings)
    # -- Thyroid-related metabolic changes ------------------------------
    _detect_thyroid_metabolic(lm, findings)
    # -- Pre-hypertension indicators ------------------------------------
    _detect_prehypertension_indicators(lm, findings)
    # -- Insulin resistance ---------------------------------------------
    _detect_insulin_resistance(lm, findings)
    # -- Vitamin A deficiency -------------------------------------------
    _detect_vitamin_a_deficiency(lm, findings)
    # -- Vitamin E deficiency -------------------------------------------
    _detect_vitamin_e_deficiency(lm, findings)
    # -- Zinc deficiency ------------------------------------------------
    _detect_zinc_deficiency(lm, findings)
    # -- Calcium deficiency ---------------------------------------------
    _detect_calcium_deficiency(lm, findings)
    # -- Magnesium deficiency -------------------------------------------
    _detect_magnesium_deficiency(lm, findings)
    # -- Elevated cardiac markers ---------------------------------------
    _detect_cardiac_risk(lm, findings)

    return findings


# -- Individual finding detectors ---------------------------------------

def _detect_iron_deficiency_anemia(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    hb = lm.get("hemoglobin")
    ferritin = lm.get("ferritin")
    iron = lm.get("iron")
    tibc = lm.get("tibc")
    tsat = lm.get("transferrin_saturation")
    mcv = lm.get("mcv")
    mch = lm.get("mch")

    evidence: list[str] = []
    low_count = 0

    if _is(hb, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("hemoglobin")
        low_count += 1
    if _is(ferritin, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("ferritin")
        low_count += 1
    if _is(iron, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("iron")
        low_count += 1
    if _is(tibc, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("tibc")
        low_count += 1
    if _is(tsat, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("transferrin_saturation")
        low_count += 1
    if _is(mcv, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("mcv")
        low_count += 1
    if _is(mch, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("mch")
        low_count += 1

    if low_count < 2:
        return

    # severity
    if _is(hb, LabStatus.CRITICAL_LOW) or low_count >= 5:
        severity = "severe"
    elif low_count >= 3:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="iron_deficiency_anemia",
        severity=severity,
        evidence=evidence,
        dietary_impact="Iron stores are depleted; diet must be enriched with bioavailable iron and vitamin C for absorption.",
        nutrients_to_increase=["iron", "vitamin_c", "folate", "vitamin_b12", "copper"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "spinach", "lentils", "chickpeas", "red meat", "liver",
            "tofu", "quinoa", "fortified cereals", "pumpkin seeds",
            "dark chocolate", "broccoli", "kale", "dried apricots",
            "beetroot", "pomegranate",
        ],
        foods_to_avoid=[
            "tea with meals", "coffee with meals", "calcium supplements with meals",
            "antacids near meals",
        ],
        calorie_adjustment=100,
    ))


def _detect_macrocytic_anemia(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    hb = lm.get("hemoglobin")
    mcv = lm.get("mcv")
    b12 = lm.get("vitamin_b12")
    folate = lm.get("folate")

    if not (_is(mcv, LabStatus.HIGH, LabStatus.CRITICAL_HIGH) and
            _is(hb, LabStatus.LOW, LabStatus.CRITICAL_LOW)):
        return

    evidence = ["hemoglobin", "mcv"]
    if _is(b12, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("vitamin_b12")
    if _is(folate, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("folate")

    severity = "severe" if _is(hb, LabStatus.CRITICAL_LOW) else "moderate"

    findings.append(HealthFinding(
        condition="macrocytic_anemia",
        severity=severity,
        evidence=evidence,
        dietary_impact="Large red blood cells indicate B12 or folate shortfall; diet must include rich sources.",
        nutrients_to_increase=["vitamin_b12", "folate", "iron"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "eggs", "dairy", "fortified cereals", "nutritional yeast",
            "leafy greens", "asparagus", "avocado", "legumes", "liver",
            "clams", "sardines",
        ],
        foods_to_avoid=["alcohol", "excessive processed foods"],
        calorie_adjustment=50,
    ))


def _detect_dyslipidemia(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    tc = lm.get("total_cholesterol")
    ldl = lm.get("ldl_cholesterol")
    hdl = lm.get("hdl_cholesterol")
    tg = lm.get("triglycerides")
    vldl = lm.get("vldl")
    ratio = lm.get("total_cholesterol_hdl_ratio")

    evidence: list[str] = []
    risk_count = 0

    if _is(tc, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("total_cholesterol")
        risk_count += 1
    if _is(ldl, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("ldl_cholesterol")
        risk_count += 1
    if _is(hdl, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("hdl_cholesterol")
        risk_count += 1
    if _is(tg, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("triglycerides")
        risk_count += 1
    if _is(vldl, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("vldl")
        risk_count += 1
    if _is(ratio, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("total_cholesterol_hdl_ratio")
        risk_count += 1

    if risk_count < 1:
        return

    if risk_count >= 4 or _is(ldl, LabStatus.CRITICAL_HIGH):
        severity = "severe"
    elif risk_count >= 2:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="dyslipidemia",
        severity=severity,
        evidence=evidence,
        dietary_impact="Abnormal lipid profile increases cardiovascular risk; diet must limit saturated and trans fats.",
        nutrients_to_increase=["omega_3", "fiber", "plant_sterols", "monounsaturated_fat"],
        nutrients_to_decrease=["saturated_fat", "trans_fat", "cholesterol", "refined_carbs"],
        foods_to_increase=[
            "oats", "barley", "walnuts", "almonds", "flaxseeds",
            "salmon", "mackerel", "sardines", "olive oil", "avocado",
            "beans", "lentils", "berries", "garlic", "soy products",
            "psyllium husk", "chia seeds",
        ],
        foods_to_avoid=[
            "fried foods", "butter", "ghee", "cream", "full-fat cheese",
            "red meat", "processed meats", "coconut oil", "palm oil",
            "baked goods with trans fat", "fast food", "sugary drinks",
            "white bread", "pastries",
        ],
        calorie_adjustment=-150,
    ))


def _detect_diabetes(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    fg = lm.get("fasting_glucose")
    hba1c = lm.get("hba1c")
    ppg = lm.get("postprandial_glucose")
    fi = lm.get("fasting_insulin")
    homa = lm.get("homa_ir")

    evidence: list[str] = []
    is_diabetic = False
    is_prediabetic = False

    # Fasting glucose thresholds
    if fg:
        if fg.value >= 126:
            evidence.append("fasting_glucose")
            is_diabetic = True
        elif fg.value >= 100:
            evidence.append("fasting_glucose")
            is_prediabetic = True

    # HbA1c thresholds
    if hba1c:
        if hba1c.value >= 6.5:
            evidence.append("hba1c")
            is_diabetic = True
        elif hba1c.value >= 5.7:
            evidence.append("hba1c")
            is_prediabetic = True

    # Post-prandial glucose
    if ppg:
        if ppg.value >= 200:
            evidence.append("postprandial_glucose")
            is_diabetic = True
        elif ppg.value >= 140:
            evidence.append("postprandial_glucose")
            is_prediabetic = True

    if _is(fi, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("fasting_insulin")
    if _is(homa, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("homa_ir")

    if not evidence:
        return

    if is_diabetic:
        condition = "diabetes"
        severity = "severe" if (hba1c and hba1c.value >= 9.0) else "moderate"
        calorie_adj = -300
    elif is_prediabetic:
        condition = "prediabetes"
        severity = "moderate" if len(evidence) >= 2 else "mild"
        calorie_adj = -200
    else:
        # Insulin or HOMA-IR elevation only
        condition = "insulin_resistance"
        severity = "mild"
        calorie_adj = -150

    findings.append(HealthFinding(
        condition=condition,
        severity=severity,
        evidence=evidence,
        dietary_impact="Blood sugar regulation is impaired; diet must focus on low glycaemic load and controlled carbohydrate intake.",
        nutrients_to_increase=["fiber", "chromium", "magnesium", "alpha_lipoic_acid", "omega_3"],
        nutrients_to_decrease=["refined_sugar", "refined_carbs", "saturated_fat"],
        foods_to_increase=[
            "leafy greens", "bitter gourd", "fenugreek", "cinnamon",
            "whole grains", "legumes", "nuts", "seeds", "berries",
            "non-starchy vegetables", "greek yogurt", "fish",
            "sweet potatoes", "quinoa", "barley", "oats",
        ],
        foods_to_avoid=[
            "white rice (large portions)", "white bread", "sugary drinks",
            "fruit juices", "candy", "pastries", "processed snacks",
            "sweetened cereals", "honey (excess)", "jaggery (excess)",
            "deep-fried foods", "potatoes (excess)", "white pasta",
        ],
        calorie_adjustment=calorie_adj,
    ))


def _detect_hypothyroidism(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    tsh = lm.get("tsh")
    t3 = lm.get("t3")
    t4 = lm.get("t4")
    ft3 = lm.get("free_t3")
    ft4 = lm.get("free_t4")

    if not _is(tsh, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        return

    evidence = ["tsh"]
    if _is(t3, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("t3")
    if _is(t4, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("t4")
    if _is(ft3, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("free_t3")
    if _is(ft4, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("free_t4")

    if _is(tsh, LabStatus.CRITICAL_HIGH) or len(evidence) >= 3:
        severity = "severe"
    elif len(evidence) >= 2:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="hypothyroidism",
        severity=severity,
        evidence=evidence,
        dietary_impact="Under-active thyroid slows metabolism; diet must support thyroid hormone synthesis and manage weight.",
        nutrients_to_increase=["iodine", "selenium", "zinc", "vitamin_d", "iron", "vitamin_b12"],
        nutrients_to_decrease=["goitrogens_raw"],
        foods_to_increase=[
            "iodised salt", "sea fish", "shrimp", "eggs",
            "brazil nuts", "sunflower seeds", "dairy",
            "seaweed (moderate)", "chicken", "whole grains",
        ],
        foods_to_avoid=[
            "raw cruciferous vegetables (large amounts)",
            "soy products (excess)", "millet",
            "highly processed foods", "gluten (if Hashimoto's)",
        ],
        calorie_adjustment=-100,
    ))


def _detect_hyperthyroidism(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    tsh = lm.get("tsh")
    t3 = lm.get("t3")
    t4 = lm.get("t4")
    ft3 = lm.get("free_t3")
    ft4 = lm.get("free_t4")

    if not _is(tsh, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    evidence = ["tsh"]
    if _is(t3, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("t3")
    if _is(t4, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("t4")
    if _is(ft3, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("free_t3")
    if _is(ft4, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("free_t4")

    if _is(tsh, LabStatus.CRITICAL_LOW) or len(evidence) >= 3:
        severity = "severe"
    elif len(evidence) >= 2:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="hyperthyroidism",
        severity=severity,
        evidence=evidence,
        dietary_impact="Over-active thyroid accelerates metabolism; diet must provide adequate calories, calcium, and vitamin D.",
        nutrients_to_increase=["calcium", "vitamin_d", "protein", "calories"],
        nutrients_to_decrease=["iodine", "caffeine"],
        foods_to_increase=[
            "dairy products", "leafy greens", "cruciferous vegetables",
            "berries", "lean meats", "eggs", "whole grains",
            "calcium-fortified foods", "sweet potatoes", "beans",
        ],
        foods_to_avoid=[
            "iodine-rich foods (excess seaweed)", "caffeine (excess)",
            "alcohol", "sugary foods", "processed foods",
        ],
        calorie_adjustment=200,
    ))


def _detect_vitamin_d_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    vd = lm.get("vitamin_d")
    if not _is(vd, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(vd, LabStatus.CRITICAL_LOW) or _val(vd) < 15 else (
        "moderate" if _val(vd) < 20 else "mild"
    )

    findings.append(HealthFinding(
        condition="vitamin_d_deficiency",
        severity=severity,
        evidence=["vitamin_d"],
        dietary_impact="Low vitamin D impairs calcium absorption and bone health; dietary and supplemental sources needed.",
        nutrients_to_increase=["vitamin_d", "calcium", "magnesium"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "fatty fish (salmon, mackerel, sardines)", "egg yolks",
            "fortified milk", "fortified orange juice", "mushrooms (UV-exposed)",
            "cod liver oil", "fortified cereals", "cheese",
        ],
        foods_to_avoid=[],
        calorie_adjustment=0,
    ))


def _detect_b12_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    b12 = lm.get("vitamin_b12")
    if not _is(b12, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(b12, LabStatus.CRITICAL_LOW) or _val(b12) < 150 else (
        "moderate" if _val(b12) < 200 else "mild"
    )

    findings.append(HealthFinding(
        condition="vitamin_b12_deficiency",
        severity=severity,
        evidence=["vitamin_b12"],
        dietary_impact="B12 is essential for nerve function and RBC formation; deficiency must be addressed through diet/supplementation.",
        nutrients_to_increase=["vitamin_b12", "folate"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "eggs", "dairy products", "fortified cereals",
            "nutritional yeast", "clams", "liver", "sardines",
            "tuna", "beef", "fortified plant milks",
        ],
        foods_to_avoid=["alcohol (impairs absorption)"],
        calorie_adjustment=0,
    ))


def _detect_folate_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    fol = lm.get("folate")
    if not _is(fol, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(fol, LabStatus.CRITICAL_LOW) else "mild"

    findings.append(HealthFinding(
        condition="folate_deficiency",
        severity=severity,
        evidence=["folate"],
        dietary_impact="Low folate can cause megaloblastic anaemia and elevated homocysteine.",
        nutrients_to_increase=["folate", "vitamin_b12", "vitamin_c"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "leafy greens", "asparagus", "broccoli", "Brussels sprouts",
            "avocado", "legumes", "fortified cereals", "citrus fruits",
            "beets", "papaya",
        ],
        foods_to_avoid=["alcohol"],
        calorie_adjustment=0,
    ))


def _detect_liver_stress(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    ast = lm.get("sgot_ast")
    alt = lm.get("sgpt_alt")
    ggt = lm.get("ggt")
    alp = lm.get("alkaline_phosphatase")
    tb = lm.get("total_bilirubin")
    db = lm.get("direct_bilirubin")
    alb = lm.get("albumin")

    evidence: list[str] = []
    if _is(ast, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("sgot_ast")
    if _is(alt, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("sgpt_alt")
    if _is(ggt, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("ggt")
    if _is(alp, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("alkaline_phosphatase")
    if _is(tb, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("total_bilirubin")
    if _is(db, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("direct_bilirubin")
    if _is(alb, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("albumin")

    if not evidence:
        return

    if len(evidence) >= 4 or any(_is(lm.get(e), LabStatus.CRITICAL_HIGH) for e in evidence):
        severity = "severe"
    elif len(evidence) >= 2:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="liver_stress",
        severity=severity,
        evidence=evidence,
        dietary_impact="Liver enzymes are elevated indicating hepatic stress; diet must reduce liver burden.",
        nutrients_to_increase=["vitamin_e", "vitamin_c", "selenium", "milk_thistle", "glutathione_precursors"],
        nutrients_to_decrease=["alcohol", "saturated_fat", "refined_sugar", "fructose"],
        foods_to_increase=[
            "cruciferous vegetables", "leafy greens", "garlic", "turmeric",
            "green tea", "walnuts", "berries", "beetroot", "artichoke",
            "lemon water", "grapefruit", "olive oil",
        ],
        foods_to_avoid=[
            "alcohol", "fried foods", "processed meats", "high-fructose corn syrup",
            "refined carbohydrates", "sugary drinks", "excess red meat",
            "packaged snacks", "fast food",
        ],
        calorie_adjustment=-100,
    ))


def _detect_kidney_impairment(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    creat = lm.get("serum_creatinine")
    egfr = lm.get("egfr")
    bun = lm.get("bun")
    urea = lm.get("blood_urea")
    ua = lm.get("uric_acid")
    bcr = lm.get("bun_creatinine_ratio")

    evidence: list[str] = []
    if _is(creat, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("serum_creatinine")
    if _is(egfr, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("egfr")
    if _is(bun, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("bun")
    if _is(urea, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("blood_urea")
    if _is(bcr, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("bun_creatinine_ratio")

    if not evidence:
        return

    if _is(egfr, LabStatus.CRITICAL_LOW) or _is(creat, LabStatus.CRITICAL_HIGH) or len(evidence) >= 4:
        severity = "severe"
    elif len(evidence) >= 2:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="kidney_impairment",
        severity=severity,
        evidence=evidence,
        dietary_impact="Kidney filtration is reduced; diet must limit protein load, sodium, potassium and phosphorus.",
        nutrients_to_increase=["omega_3", "vitamin_d", "iron"],
        nutrients_to_decrease=["sodium", "potassium", "phosphorus", "protein_excess"],
        foods_to_increase=[
            "cauliflower", "blueberries", "red grapes", "egg whites",
            "garlic", "olive oil", "cabbage", "bell peppers",
            "onions", "apples", "cranberries",
        ],
        foods_to_avoid=[
            "high-sodium foods", "processed meats", "canned soups",
            "bananas (excess)", "oranges (excess)", "potatoes (excess)",
            "tomatoes (excess)", "dairy (excess)", "nuts (excess)",
            "whole grains (excess if phosphorus-restricted)",
            "dark colas", "packaged foods",
        ],
        calorie_adjustment=-50,
    ))


def _detect_hyperuricemia(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    ua = lm.get("uric_acid")
    if not _is(ua, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        return

    severity = "severe" if _is(ua, LabStatus.CRITICAL_HIGH) else (
        "moderate" if _val(ua) > 9.0 else "mild"
    )

    findings.append(HealthFinding(
        condition="hyperuricemia",
        severity=severity,
        evidence=["uric_acid"],
        dietary_impact="Elevated uric acid increases gout and kidney stone risk; purine and fructose intake must be limited.",
        nutrients_to_increase=["vitamin_c", "water"],
        nutrients_to_decrease=["purines", "fructose", "alcohol"],
        foods_to_increase=[
            "cherries", "low-fat dairy", "vegetables", "whole grains",
            "water (2-3 litres daily)", "coffee (moderate)",
            "citrus fruits", "berries",
        ],
        foods_to_avoid=[
            "organ meats", "red meat", "shellfish", "anchovies",
            "sardines", "beer", "liquor", "high-fructose corn syrup",
            "sugary drinks", "gravy", "yeast extracts",
        ],
        calorie_adjustment=-100,
    ))


def _detect_inflammation(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    crp = lm.get("crp")
    esr = lm.get("esr")
    hcy = lm.get("homocysteine")

    evidence: list[str] = []
    if _is(crp, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("crp")
    if _is(esr, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("esr")
    if _is(hcy, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("homocysteine")

    if not evidence:
        return

    if len(evidence) >= 3 or _is(crp, LabStatus.CRITICAL_HIGH):
        severity = "severe"
    elif len(evidence) >= 2:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="chronic_inflammation",
        severity=severity,
        evidence=evidence,
        dietary_impact="Elevated inflammatory markers warrant an anti-inflammatory dietary pattern.",
        nutrients_to_increase=["omega_3", "vitamin_e", "vitamin_c", "curcumin", "folate", "vitamin_b6", "vitamin_b12"],
        nutrients_to_decrease=["omega_6_excess", "refined_sugar", "trans_fat", "saturated_fat"],
        foods_to_increase=[
            "turmeric", "ginger", "fatty fish", "walnuts", "flaxseeds",
            "berries", "leafy greens", "olive oil", "tomatoes",
            "green tea", "dark chocolate (>70%)", "cherries",
            "bell peppers", "mushrooms",
        ],
        foods_to_avoid=[
            "fried foods", "processed meats", "refined carbohydrates",
            "sugary drinks", "margarine", "seed oils (excess)",
            "white bread", "pastries", "alcohol",
        ],
        calorie_adjustment=-100,
    ))


def _detect_fatty_liver_indicators(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    alt = lm.get("sgpt_alt")
    ggt = lm.get("ggt")
    tg = lm.get("triglycerides")
    fg = lm.get("fasting_glucose")
    homa = lm.get("homa_ir")
    ast = lm.get("sgot_ast")

    evidence: list[str] = []
    score = 0

    if _is(alt, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("sgpt_alt")
        score += 1
    if _is(ggt, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("ggt")
        score += 1
    if _is(tg, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("triglycerides")
        score += 1
    if _is(fg, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("fasting_glucose")
        score += 1
    if _is(homa, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("homa_ir")
        score += 1
    if _is(ast, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("sgot_ast")
        score += 1

    # Need at least ALT elevated plus one more marker
    if "sgpt_alt" not in evidence or score < 2:
        return

    if score >= 4:
        severity = "severe"
    elif score >= 3:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="fatty_liver_indicators",
        severity=severity,
        evidence=evidence,
        dietary_impact="Lab pattern suggests non-alcoholic fatty liver disease; diet must reduce hepatic fat accumulation.",
        nutrients_to_increase=["omega_3", "vitamin_e", "fiber", "choline"],
        nutrients_to_decrease=["fructose", "refined_carbs", "saturated_fat", "alcohol"],
        foods_to_increase=[
            "fatty fish", "walnuts", "flaxseeds", "olive oil",
            "leafy greens", "coffee (moderate)", "oats",
            "cruciferous vegetables", "berries", "garlic",
            "avocado", "green tea",
        ],
        foods_to_avoid=[
            "alcohol", "sugary drinks", "fruit juices", "white bread",
            "fried foods", "fast food", "high-fructose corn syrup",
            "processed snacks", "sweetened cereals",
        ],
        calorie_adjustment=-200,
    ))


def _detect_electrolyte_imbalance(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    sodium = lm.get("sodium")
    potassium = lm.get("potassium")
    chloride = lm.get("chloride")
    calcium = lm.get("calcium")
    magnesium = lm.get("magnesium")
    phosphorus = lm.get("phosphorus")

    evidence: list[str] = []
    nutrients_up: list[str] = []
    nutrients_down: list[str] = []
    foods_up: list[str] = []
    foods_avoid: list[str] = []

    # Sodium
    if _is(sodium, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("sodium")
        nutrients_up.append("sodium")
        foods_up.extend(["broth", "salted nuts", "cheese"])
    elif _is(sodium, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("sodium")
        nutrients_down.append("sodium")
        foods_avoid.extend(["processed foods", "canned soups", "pickles", "soy sauce"])
        foods_up.extend(["fresh fruits", "vegetables", "water"])

    # Potassium
    if _is(potassium, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("potassium")
        nutrients_up.append("potassium")
        foods_up.extend(["bananas", "potatoes", "spinach", "avocado", "coconut water"])
    elif _is(potassium, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("potassium")
        nutrients_down.append("potassium")
        foods_avoid.extend(["bananas", "oranges", "potatoes", "tomatoes", "coconut water"])

    # Chloride
    if _is(chloride, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("chloride")
        nutrients_up.append("chloride")
    elif _is(chloride, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("chloride")
        nutrients_down.append("chloride")

    # Calcium
    if _is(calcium, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("calcium")
        nutrients_up.append("calcium")
        foods_up.extend(["dairy products", "fortified plant milk", "leafy greens", "sesame seeds"])
    elif _is(calcium, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("calcium")
        nutrients_down.append("calcium")
        foods_avoid.extend(["excess dairy", "calcium supplements"])
        foods_up.extend(["water (adequate hydration)"])

    # Magnesium
    if _is(magnesium, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("magnesium")
        nutrients_up.append("magnesium")
        foods_up.extend(["dark chocolate", "pumpkin seeds", "almonds", "spinach", "black beans"])
    elif _is(magnesium, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("magnesium")
        nutrients_down.append("magnesium")

    # Phosphorus
    if _is(phosphorus, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("phosphorus")
        nutrients_up.append("phosphorus")
        foods_up.extend(["dairy", "meat", "nuts", "seeds"])
    elif _is(phosphorus, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("phosphorus")
        nutrients_down.append("phosphorus")
        foods_avoid.extend(["processed foods", "dark colas", "organ meats"])

    if not evidence:
        return

    if any(_is(lm.get(e), LabStatus.CRITICAL_LOW, LabStatus.CRITICAL_HIGH) for e in evidence):
        severity = "severe"
    elif len(evidence) >= 3:
        severity = "moderate"
    else:
        severity = "mild"

    # Deduplicate food lists
    foods_up = list(dict.fromkeys(foods_up))
    foods_avoid = list(dict.fromkeys(foods_avoid))

    findings.append(HealthFinding(
        condition="electrolyte_imbalance",
        severity=severity,
        evidence=evidence,
        dietary_impact="Electrolyte levels are outside normal range; dietary mineral intake must be adjusted.",
        nutrients_to_increase=nutrients_up,
        nutrients_to_decrease=nutrients_down,
        foods_to_increase=foods_up,
        foods_to_avoid=foods_avoid,
        calorie_adjustment=0,
    ))


def _detect_thyroid_metabolic(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    """Detect thyroid-related metabolic changes (subclinical)."""
    tsh = lm.get("tsh")
    tc = lm.get("total_cholesterol")
    ldl = lm.get("ldl_cholesterol")
    tg = lm.get("triglycerides")

    if not tsh:
        return

    # Subclinical hypothyroidism → lipid elevation
    if (_is(tsh, LabStatus.HIGH) and
            (_is(tc, LabStatus.HIGH) or _is(ldl, LabStatus.HIGH) or _is(tg, LabStatus.HIGH))):
        evidence = ["tsh"]
        if _is(tc, LabStatus.HIGH):
            evidence.append("total_cholesterol")
        if _is(ldl, LabStatus.HIGH):
            evidence.append("ldl_cholesterol")
        if _is(tg, LabStatus.HIGH):
            evidence.append("triglycerides")

        findings.append(HealthFinding(
            condition="thyroid_metabolic_dysfunction",
            severity="moderate" if len(evidence) >= 3 else "mild",
            evidence=evidence,
            dietary_impact="Subclinical thyroid dysfunction is contributing to metabolic disturbance; thyroid-supportive and lipid-lowering diet advised.",
            nutrients_to_increase=["iodine", "selenium", "zinc", "omega_3", "fiber"],
            nutrients_to_decrease=["saturated_fat", "refined_carbs"],
            foods_to_increase=[
                "sea fish", "brazil nuts", "eggs", "seaweed (moderate)",
                "oats", "flaxseeds", "walnuts", "olive oil",
            ],
            foods_to_avoid=[
                "fried foods", "processed meats", "excess raw cruciferous",
                "sugary drinks",
            ],
            calorie_adjustment=-100,
        ))


def _detect_prehypertension_indicators(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    """Lab markers that indirectly suggest cardiovascular / BP risk."""
    sodium = lm.get("sodium")
    potassium = lm.get("potassium")
    crp = lm.get("crp")
    hcy = lm.get("homocysteine")
    ua = lm.get("uric_acid")
    tg = lm.get("triglycerides")
    fg = lm.get("fasting_glucose")
    homa = lm.get("homa_ir")

    evidence: list[str] = []
    if _is(sodium, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("sodium")
    if _is(potassium, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        evidence.append("potassium")
    if _is(crp, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("crp")
    if _is(hcy, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("homocysteine")
    if _is(ua, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("uric_acid")
    if _is(tg, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("triglycerides")
    if _is(fg, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("fasting_glucose")
    if _is(homa, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("homa_ir")

    if len(evidence) < 3:
        return

    severity = "severe" if len(evidence) >= 6 else ("moderate" if len(evidence) >= 4 else "mild")

    findings.append(HealthFinding(
        condition="prehypertension_indicators",
        severity=severity,
        evidence=evidence,
        dietary_impact="Multiple markers suggest elevated cardiovascular/blood pressure risk; a DASH-style diet is recommended.",
        nutrients_to_increase=["potassium", "magnesium", "calcium", "fiber", "omega_3"],
        nutrients_to_decrease=["sodium", "saturated_fat", "refined_sugar", "alcohol"],
        foods_to_increase=[
            "bananas", "sweet potatoes", "spinach", "avocado",
            "low-fat dairy", "berries", "beets", "oats",
            "fatty fish", "garlic", "dark chocolate (moderate)",
        ],
        foods_to_avoid=[
            "table salt (excess)", "pickles", "canned foods",
            "processed meats", "cheese (excess)", "alcohol",
            "fried foods", "fast food",
        ],
        calorie_adjustment=-150,
    ))


def _detect_insulin_resistance(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    homa = lm.get("homa_ir")
    fi = lm.get("fasting_insulin")
    fg = lm.get("fasting_glucose")
    tg = lm.get("triglycerides")

    if not (_is(homa, LabStatus.HIGH, LabStatus.CRITICAL_HIGH) or
            _is(fi, LabStatus.HIGH, LabStatus.CRITICAL_HIGH)):
        return

    # Skip if we already flagged diabetes / prediabetes (those subsume this)
    # We check by looking at glucose status
    if fg and fg.value >= 100:
        return

    evidence: list[str] = []
    if _is(homa, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("homa_ir")
    if _is(fi, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("fasting_insulin")
    if _is(tg, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("triglycerides")

    severity = "moderate" if len(evidence) >= 2 else "mild"

    findings.append(HealthFinding(
        condition="insulin_resistance",
        severity=severity,
        evidence=evidence,
        dietary_impact="Insulin resistance without frank hyperglycaemia; low-glycaemic, high-fibre diet recommended.",
        nutrients_to_increase=["fiber", "chromium", "magnesium", "omega_3"],
        nutrients_to_decrease=["refined_sugar", "refined_carbs"],
        foods_to_increase=[
            "legumes", "non-starchy vegetables", "whole grains",
            "nuts", "seeds", "cinnamon", "vinegar (apple cider)",
            "berries", "leafy greens",
        ],
        foods_to_avoid=[
            "white bread", "white rice (large portions)", "sugary drinks",
            "candy", "processed snacks", "sweetened cereals",
        ],
        calorie_adjustment=-100,
    ))


def _detect_vitamin_a_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    va = lm.get("vitamin_a")
    if not _is(va, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(va, LabStatus.CRITICAL_LOW) else "mild"

    findings.append(HealthFinding(
        condition="vitamin_a_deficiency",
        severity=severity,
        evidence=["vitamin_a"],
        dietary_impact="Vitamin A is essential for vision, immunity, and skin health.",
        nutrients_to_increase=["vitamin_a", "beta_carotene"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "sweet potatoes", "carrots", "spinach", "kale",
            "liver", "eggs", "mango", "cantaloupe", "red bell peppers",
        ],
        foods_to_avoid=[],
        calorie_adjustment=0,
    ))


def _detect_vitamin_e_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    ve = lm.get("vitamin_e")
    if not _is(ve, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(ve, LabStatus.CRITICAL_LOW) else "mild"

    findings.append(HealthFinding(
        condition="vitamin_e_deficiency",
        severity=severity,
        evidence=["vitamin_e"],
        dietary_impact="Vitamin E is a key antioxidant; deficiency increases oxidative stress.",
        nutrients_to_increase=["vitamin_e"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "sunflower seeds", "almonds", "hazelnuts", "spinach",
            "avocado", "olive oil", "wheat germ", "butternut squash",
        ],
        foods_to_avoid=[],
        calorie_adjustment=0,
    ))


def _detect_zinc_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    zn = lm.get("zinc")
    if not _is(zn, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(zn, LabStatus.CRITICAL_LOW) else "mild"

    findings.append(HealthFinding(
        condition="zinc_deficiency",
        severity=severity,
        evidence=["zinc"],
        dietary_impact="Zinc deficiency impairs immunity and wound healing.",
        nutrients_to_increase=["zinc"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "oysters", "red meat", "pumpkin seeds", "chickpeas",
            "lentils", "cashews", "fortified cereals", "yogurt",
        ],
        foods_to_avoid=[],
        calorie_adjustment=0,
    ))


def _detect_calcium_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    ca = lm.get("calcium")
    if not _is(ca, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(ca, LabStatus.CRITICAL_LOW) else "mild"

    findings.append(HealthFinding(
        condition="calcium_deficiency",
        severity=severity,
        evidence=["calcium"],
        dietary_impact="Low calcium increases risk of osteoporosis and muscle cramps.",
        nutrients_to_increase=["calcium", "vitamin_d", "magnesium"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "dairy products", "fortified plant milks", "leafy greens",
            "sardines (with bones)", "tofu", "sesame seeds", "almonds",
            "broccoli", "figs",
        ],
        foods_to_avoid=["excess caffeine", "excess sodium (increases calcium loss)"],
        calorie_adjustment=0,
    ))


def _detect_magnesium_deficiency(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    mg = lm.get("magnesium")
    if not _is(mg, LabStatus.LOW, LabStatus.CRITICAL_LOW):
        return

    severity = "severe" if _is(mg, LabStatus.CRITICAL_LOW) else "mild"

    findings.append(HealthFinding(
        condition="magnesium_deficiency",
        severity=severity,
        evidence=["magnesium"],
        dietary_impact="Low magnesium affects muscle, nerve, and heart function.",
        nutrients_to_increase=["magnesium"],
        nutrients_to_decrease=[],
        foods_to_increase=[
            "dark chocolate", "pumpkin seeds", "almonds", "spinach",
            "black beans", "quinoa", "avocado", "cashews", "whole grains",
        ],
        foods_to_avoid=["excess alcohol", "excess caffeine"],
        calorie_adjustment=0,
    ))


def _detect_cardiac_risk(
    lm: dict[str, LabResult],
    findings: list[HealthFinding],
) -> None:
    trop = lm.get("troponin")
    bnp_ = lm.get("bnp")
    ldh_ = lm.get("ldh")
    crp = lm.get("crp")
    hcy = lm.get("homocysteine")

    evidence: list[str] = []
    if _is(trop, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("troponin")
    if _is(bnp_, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("bnp")
    if _is(ldh_, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("ldh")
    if _is(crp, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("crp")
    if _is(hcy, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        evidence.append("homocysteine")

    if not evidence:
        return

    # Troponin elevation is always serious
    if _is(trop, LabStatus.HIGH, LabStatus.CRITICAL_HIGH):
        severity = "severe"
    elif len(evidence) >= 3:
        severity = "moderate"
    else:
        severity = "mild"

    findings.append(HealthFinding(
        condition="cardiac_risk_elevation",
        severity=severity,
        evidence=evidence,
        dietary_impact="Cardiac markers are elevated; a heart-healthy diet is essential.",
        nutrients_to_increase=["omega_3", "fiber", "potassium", "magnesium", "coenzyme_q10", "folate"],
        nutrients_to_decrease=["sodium", "saturated_fat", "trans_fat", "cholesterol"],
        foods_to_increase=[
            "fatty fish", "walnuts", "flaxseeds", "oats", "berries",
            "leafy greens", "olive oil", "avocado", "garlic",
            "dark chocolate (>70%, moderate)", "legumes", "tomatoes",
        ],
        foods_to_avoid=[
            "processed meats", "fried foods", "butter (excess)",
            "full-fat dairy (excess)", "red meat (excess)",
            "salt (excess)", "sugary drinks", "fast food",
        ],
        calorie_adjustment=-150,
    ))


# ---------------------------------------------------------------------------
# 5. process_abdomen_findings
# ---------------------------------------------------------------------------

# Pattern → (organ, finding_key, severity_hint, dietary_impact, avoid, increase)
# Loaded from JSON; each entry is converted to the tuple format expected by
# ``process_abdomen_findings``.

def _load_abdomen_patterns() -> list[tuple[str, str, str, str, str, list[str], list[str]]]:
    """Load abdomen scan patterns from ``abdomen_patterns_nutrition.json``."""
    with open(_DATA_DIR / "abdomen_patterns_nutrition.json", encoding="utf-8") as fh:
        raw: list[dict[str, Any]] = json.load(fh)
    return [
        (
            entry["pattern"],
            entry["organ"],
            entry["finding_key"],
            entry["severity_hint"],
            entry["dietary_impact"],
            entry["foods_to_avoid"],
            entry["foods_to_increase"],
        )
        for entry in raw
    ]


_ABDOMEN_PATTERNS: list[tuple[str, str, str, str, str, list[str], list[str]]] = _load_abdomen_patterns()


# ---------------------------------------------------------------------------
# Dietary advice mappings  – loaded from JSON
# ---------------------------------------------------------------------------

def _load_condition_advice() -> dict[str, list[str]]:
    """Load condition → dietary-advice mapping from ``condition_advice.json``."""
    with open(_DATA_DIR / "condition_advice.json", encoding="utf-8") as fh:
        return json.load(fh)


def _load_abdomen_advice() -> dict[str, list[str]]:
    """Load abdomen-finding → dietary-advice mapping from ``abdomen_advice.json``."""
    with open(_DATA_DIR / "abdomen_advice.json", encoding="utf-8") as fh:
        return json.load(fh)


_CONDITION_ADVICE: dict[str, list[str]] = _load_condition_advice()
_ABDOMEN_ADVICE: dict[str, list[str]] = _load_abdomen_advice()


def process_abdomen_findings(findings_text: str) -> list[AbdomenFinding]:
    """Parse free-text abdomen scan findings and return structured data.

    Parameters
    ----------
    findings_text : str
        Free-text radiology / ultrasound report.

    Returns
    -------
    list[AbdomenFinding]
    """
    if not findings_text or not findings_text.strip():
        return []

    text_lower = findings_text.lower()
    results: list[AbdomenFinding] = []
    seen: set[str] = set()

    _NEG_RE = re.compile(r"\b(no|not|without|absent|negative|nil|unremarkable|ruled\s*out)\b")

    for pattern, organ, finding_key, sev_hint, impact, avoid, increase in _ABDOMEN_PATTERNS:
        match = re.search(pattern, text_lower)
        if match and finding_key not in seen:
            # Check for negation in the 30 chars preceding the match
            start = max(0, match.start() - 30)
            prefix = text_lower[start:match.start()]
            if _NEG_RE.search(prefix):
                continue
            seen.add(finding_key)
            severity = _resolve_abdomen_severity(sev_hint, match)
            results.append(AbdomenFinding(
                organ=organ,
                finding=finding_key,
                severity=severity,
                dietary_impact=impact,
                foods_to_avoid=list(avoid),
                foods_to_increase=list(increase),
            ))

    return results


def _resolve_abdomen_severity(hint: str, match: re.Match[str]) -> str:
    """Determine severity either from the hint or from the regex capture group."""
    if hint != "auto":
        return hint

    captured = (match.group(1) or "").lower().strip() if match.lastindex else ""
    if "severe" in captured or "iii" in captured or "3" in captured:
        return "severe"
    if "moderate" in captured or "ii" in captured or "2" in captured:
        return "moderate"
    return "mild"


# ---------------------------------------------------------------------------
# 6. process_health_checkup  (main entry point)
# ---------------------------------------------------------------------------

def process_health_checkup(
    blood_data: dict[str, float],
    urine_data: dict[str, float],
    abdomen_text: str,
    age: int,
    sex: str,
) -> HealthCheckupResult:
    """Process a full health checkup and return a comprehensive result.

    This is the **main entry point** for the module.

    Parameters
    ----------
    blood_data : dict[str, float]
        Mapping of blood-test parameter names to measured values.
    urine_data : dict[str, float]
        Mapping of urine-test parameter names to measured values.
    abdomen_text : str
        Free-text report of abdominal ultrasound / scan.
    age : int
        Patient age in years.
    sex : str
        ``"male"`` or ``"female"``.

    Returns
    -------
    HealthCheckupResult
    """
    sex = sex.lower().strip()

    # 1. Interpret all lab values ------------------------------------------
    lab_results: list[LabResult] = []

    for param, value in blood_data.items():
        if param in BLOOD_TEST_REFERENCE_RANGES:
            lab_results.append(interpret_lab_value(param, value, age, sex))

    for param, value in urine_data.items():
        if param in URINE_TEST_REFERENCE_RANGES:
            lab_results.append(interpret_lab_value(param, value, age, sex))

    # 2. Detect health findings --------------------------------------------
    findings = detect_health_findings(lab_results)

    # 3. Process abdomen findings ------------------------------------------
    abdomen = process_abdomen_findings(abdomen_text)

    # 4. Calculate overall health score ------------------------------------
    health_score = _calculate_health_score(lab_results, findings, abdomen)

    # 5. Compute net nutrient adjustments ----------------------------------
    nutrient_adj = _compute_nutrient_adjustments(findings, abdomen)

    # 6. Build human-readable dietary modifications ------------------------
    dietary_mods = _build_dietary_modifications(findings, abdomen)

    # 7. Compute net calorie adjustment ------------------------------------
    calorie_adj = _compute_calorie_adjustment(findings, abdomen)

    # 8. Detected conditions list ------------------------------------------
    detected_conditions = _extract_condition_names(findings, abdomen)

    return HealthCheckupResult(
        lab_results=lab_results,
        findings=findings,
        abdomen_findings=abdomen,
        overall_health_score=round(health_score, 1),
        detected_conditions=detected_conditions,
        nutrient_adjustments=nutrient_adj,
        dietary_modifications=dietary_mods,
        calorie_adjustment=calorie_adj,
    )


# -- Scoring helpers -------------------------------------------------------

def _calculate_health_score(
    lab_results: list[LabResult],
    findings: list[HealthFinding],
    abdomen: list[AbdomenFinding],
) -> float:
    """Compute a 0-100 health score.

    Methodology
    -----------
    - Start at 100.
    - For each lab result: deduct points based on how far from normal.
    - For each finding: deduct points based on severity.
    - For each abdomen finding: deduct points based on severity.
    - Clamp to [0, 100].
    """
    score = 100.0
    total_params = max(len(lab_results), 1)

    # --- Lab deductions ---
    for lr in lab_results:
        if lr.status == LabStatus.NORMAL:
            continue
        rr = lr.reference_range
        if lr.status == LabStatus.CRITICAL_LOW:
            score -= min(4.0, 4.0 * _deviation_ratio(lr.value, rr.low, rr.critical_low or rr.low))
        elif lr.status == LabStatus.CRITICAL_HIGH:
            score -= min(4.0, 4.0 * _deviation_ratio(lr.value, rr.high, rr.critical_high or rr.high))
        elif lr.status == LabStatus.LOW:
            score -= min(2.0, 2.0 * _deviation_ratio_normal(lr.value, rr.low, rr.high))
        elif lr.status == LabStatus.HIGH:
            score -= min(2.0, 2.0 * _deviation_ratio_normal(lr.value, rr.high, rr.low))

    # Scale lab deductions so a perfect score isn't unfairly penalised by many params
    if total_params > 30:
        # Normalise: the maximum deduction from labs is ~40 points
        lab_deduction = 100.0 - score
        score = 100.0 - min(lab_deduction, 40.0)

    # --- Finding deductions ---
    _severity_penalty = {"mild": 3.0, "moderate": 6.0, "severe": 10.0}
    for f in findings:
        score -= _severity_penalty.get(f.severity, 3.0)

    # --- Abdomen deductions ---
    _abd_severity_penalty = {"mild": 2.0, "moderate": 5.0, "severe": 8.0}
    for a in abdomen:
        score -= _abd_severity_penalty.get(a.severity, 2.0)

    return max(0.0, min(100.0, score))


def _deviation_ratio(value: float, boundary: float, critical: float) -> float:
    """How far between *boundary* and *critical* the *value* has deviated (0-1+)."""
    span = abs(critical - boundary) or 1.0
    return abs(value - boundary) / span


def _deviation_ratio_normal(value: float, near_bound: float, far_bound: float) -> float:
    """Fraction of range width that *value* deviates past *near_bound*."""
    span = abs(far_bound - near_bound) or 1.0
    return abs(value - near_bound) / span


# -- Nutrient adjustment helpers -------------------------------------------

# Default multiplier bump per mention direction
_NUTRIENT_INCREASE_STEP = 0.25
_NUTRIENT_DECREASE_STEP = 0.15
_SEVERITY_MULTIPLIER = {"mild": 1.0, "moderate": 1.5, "severe": 2.0}


def _compute_nutrient_adjustments(
    findings: list[HealthFinding],
    abdomen: list[AbdomenFinding],
) -> dict[str, float]:
    """Compute nutrient multiplier map.

    A value of ``1.0`` means no change, ``1.5`` means +50%, ``0.5`` means -50%.
    """
    adj: dict[str, float] = {}

    for f in findings:
        sev = _SEVERITY_MULTIPLIER.get(f.severity, 1.0)
        for n in f.nutrients_to_increase:
            adj[n] = adj.get(n, 1.0) + _NUTRIENT_INCREASE_STEP * sev
        for n in f.nutrients_to_decrease:
            adj[n] = adj.get(n, 1.0) - _NUTRIENT_DECREASE_STEP * sev

    # Abdomen findings don't carry explicit nutrient lists, but some common
    # mappings are applied.
    _abdomen_nutrient_map: dict[str, tuple[list[str], list[str]]] = {
        "fatty_liver": (["omega_3", "vitamin_e", "fiber", "choline"], ["saturated_fat", "fructose", "alcohol"]),
        "hepatomegaly": (["vitamin_c", "selenium"], ["alcohol", "saturated_fat"]),
        "kidney_stones": (["citrate", "water"], ["oxalate", "sodium", "animal_protein"]),
        "gallstones": (["fiber"], ["saturated_fat", "cholesterol"]),
        "pancreatitis": (["lean_protein"], ["fat", "alcohol"]),
        "cirrhosis": (["vitamin_d", "zinc"], ["sodium", "alcohol"]),
        "ascites": ([], ["sodium"]),
    }

    for af in abdomen:
        mapping = _abdomen_nutrient_map.get(af.finding, ([], []))
        sev = _SEVERITY_MULTIPLIER.get(af.severity, 1.0)
        for n in mapping[0]:
            adj[n] = adj.get(n, 1.0) + _NUTRIENT_INCREASE_STEP * sev
        for n in mapping[1]:
            adj[n] = adj.get(n, 1.0) - _NUTRIENT_DECREASE_STEP * sev

    # Clamp to [0.1, 3.0]
    for k in adj:
        adj[k] = round(max(0.1, min(3.0, adj[k])), 2)

    return adj


def _build_dietary_modifications(
    findings: list[HealthFinding],
    abdomen: list[AbdomenFinding],
) -> list[str]:
    """Build a deduplicated list of human-readable dietary modification strings."""
    mods: list[str] = []
    seen: set[str] = set()

    for f in findings:
        for advice in _CONDITION_ADVICE.get(f.condition, []):
            if advice not in seen:
                seen.add(advice)
                mods.append(advice)

    for a in abdomen:
        for advice in _ABDOMEN_ADVICE.get(a.finding, []):
            if advice not in seen:
                seen.add(advice)
                mods.append(advice)

    return mods


def _compute_calorie_adjustment(
    findings: list[HealthFinding],
    abdomen: list[AbdomenFinding],
) -> int:
    """Compute net calorie adjustment from all findings.

    The adjustment is the *sum* of individual finding adjustments, clamped
    to [-500, +500].
    """
    total = sum(f.calorie_adjustment for f in findings)
    # Abdomen findings that strongly affect calories
    _abdomen_calorie: dict[str, int] = {
        "fatty_liver": -100,
        "cirrhosis": -100,
        "pancreatitis": -150,
        "ascites": -100,
    }
    for a in abdomen:
        total += _abdomen_calorie.get(a.finding, 0)

    return max(-500, min(500, total))


def _extract_condition_names(
    findings: list[HealthFinding],
    abdomen: list[AbdomenFinding],
) -> list[str]:
    """Return a deduplicated list of detected condition names."""
    conditions: list[str] = []
    seen: set[str] = set()

    for f in findings:
        if f.condition not in seen:
            seen.add(f.condition)
            conditions.append(f.condition)

    for a in abdomen:
        if a.finding not in seen:
            seen.add(a.finding)
            conditions.append(a.finding)

    return conditions


# ---------------------------------------------------------------------------
# 7. get_diet_advisor_overrides
# ---------------------------------------------------------------------------

def get_diet_advisor_overrides(result: HealthCheckupResult) -> dict[str, Any]:
    """Convert :class:`HealthCheckupResult` into a dict consumable by the
    DietAdvisor.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``additional_conditions`` – list of condition strings
        - ``nutrient_multipliers`` – ``{nutrient: float}``
        - ``calorie_adjustment`` – int (kcal)
        - ``foods_to_prioritize`` – deduplicated list
        - ``foods_to_avoid`` – deduplicated list
        - ``additional_restrictions`` – e.g. ``["low-sodium"]``
    """
    # Collect food lists (deduplicated, preserving order)
    prio_set: dict[str, None] = {}
    avoid_set: dict[str, None] = {}

    for f in result.findings:
        for food in f.foods_to_increase:
            prio_set.setdefault(food, None)
        for food in f.foods_to_avoid:
            avoid_set.setdefault(food, None)

    for a in result.abdomen_findings:
        for food in a.foods_to_increase:
            prio_set.setdefault(food, None)
        for food in a.foods_to_avoid:
            avoid_set.setdefault(food, None)

    # Derive additional restrictions from nutrient adjustments
    restrictions: list[str] = []
    na = result.nutrient_adjustments

    if na.get("sodium", 1.0) < 0.8:
        restrictions.append("low-sodium")
    if na.get("saturated_fat", 1.0) < 0.8:
        restrictions.append("low-saturated-fat")
    if na.get("refined_sugar", 1.0) < 0.8 or na.get("fructose", 1.0) < 0.8:
        restrictions.append("low-sugar")
    if na.get("refined_carbs", 1.0) < 0.8:
        restrictions.append("low-refined-carbs")
    if na.get("alcohol", 1.0) < 0.8:
        restrictions.append("no-alcohol")
    if na.get("purines", 1.0) < 0.8:
        restrictions.append("low-purine")
    if na.get("fat", 1.0) < 0.8:
        restrictions.append("low-fat")
    if na.get("potassium", 1.0) < 0.8:
        restrictions.append("low-potassium")
    if na.get("phosphorus", 1.0) < 0.8:
        restrictions.append("low-phosphorus")
    if na.get("oxalate", 1.0) < 0.8:
        restrictions.append("low-oxalate")
    if na.get("cholesterol", 1.0) < 0.8:
        restrictions.append("low-cholesterol")
    if na.get("protein_excess", 1.0) < 0.8:
        restrictions.append("moderate-protein")
    if na.get("iodine", 1.0) < 0.8:
        restrictions.append("low-iodine")
    if na.get("trans_fat", 1.0) < 0.8:
        restrictions.append("no-trans-fat")
    if na.get("goitrogens_raw", 1.0) < 0.8:
        restrictions.append("limit-raw-goitrogens")

    # Map internal condition names to DietAdvisor-friendly labels
    _label_map: dict[str, str] = {
        "iron_deficiency_anemia": "anemia",
        "macrocytic_anemia": "anemia",
        "dyslipidemia": "dyslipidemia",
        "diabetes": "diabetes",
        "prediabetes": "prediabetes",
        "insulin_resistance": "insulin_resistance",
        "hypothyroidism": "hypothyroidism",
        "hyperthyroidism": "hyperthyroidism",
        "vitamin_d_deficiency": "vitamin_d_deficiency",
        "vitamin_b12_deficiency": "vitamin_b12_deficiency",
        "folate_deficiency": "folate_deficiency",
        "liver_stress": "liver_stress",
        "fatty_liver_indicators": "fatty_liver",
        "kidney_impairment": "kidney_impairment",
        "hyperuricemia": "hyperuricemia",
        "chronic_inflammation": "chronic_inflammation",
        "electrolyte_imbalance": "electrolyte_imbalance",
        "thyroid_metabolic_dysfunction": "thyroid_metabolic_dysfunction",
        "prehypertension_indicators": "prehypertension",
        "cardiac_risk_elevation": "cardiac_risk",
        "vitamin_a_deficiency": "vitamin_a_deficiency",
        "vitamin_e_deficiency": "vitamin_e_deficiency",
        "zinc_deficiency": "zinc_deficiency",
        "calcium_deficiency": "calcium_deficiency",
        "magnesium_deficiency": "magnesium_deficiency",
        # abdomen
        "fatty_liver": "fatty_liver",
        "hepatomegaly": "hepatomegaly",
        "kidney_stones": "kidney_stones",
        "gallstones": "gallstones",
        "splenomegaly": "splenomegaly",
        "pancreatic_cyst": "pancreatic_cyst",
        "pancreatitis": "pancreatitis",
        "renal_cyst": "renal_cyst",
        "cirrhosis": "cirrhosis",
        "ascites": "ascites",
    }

    conditions: list[str] = []
    seen_labels: set[str] = set()
    for c in result.detected_conditions:
        label = _label_map.get(c, c)
        if label not in seen_labels:
            seen_labels.add(label)
            conditions.append(label)

    return {
        "additional_conditions": conditions,
        "nutrient_multipliers": dict(result.nutrient_adjustments),
        "calorie_adjustment": result.calorie_adjustment,
        "foods_to_prioritize": list(prio_set),
        "foods_to_avoid": list(avoid_set),
        "additional_restrictions": restrictions,
    }


# ---------------------------------------------------------------------------
# Convenience: quick summary for debugging / logging
# ---------------------------------------------------------------------------

def summarize_checkup(result: HealthCheckupResult) -> str:
    """Return a compact human-readable summary of a checkup result."""
    lines: list[str] = []
    lines.append(f"Health Score: {result.overall_health_score}/100")
    lines.append(f"Conditions detected: {len(result.detected_conditions)}")
    for c in result.detected_conditions:
        lines.append(f"  - {c}")
    lines.append(f"Calorie adjustment: {result.calorie_adjustment:+d} kcal")
    lines.append(f"Nutrient adjustments ({len(result.nutrient_adjustments)}):")
    for n, m in sorted(result.nutrient_adjustments.items()):
        direction = "increase" if m > 1.0 else ("decrease" if m < 1.0 else "unchanged")
        lines.append(f"  {n}: x{m:.2f} ({direction})")
    lines.append(f"Dietary modifications ({len(result.dietary_modifications)}):")
    for dm in result.dietary_modifications:
        lines.append(f"  * {dm}")
    return "\n".join(lines)
