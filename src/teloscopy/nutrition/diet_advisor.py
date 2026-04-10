"""Nutrigenomics-informed dietary recommendation engine.

Provides geography-aware, genetically-informed dietary recommendations by
integrating SNP-level nutrigenomic data with regional food availability to
produce personalised meal plans and nutritional guidance.  The module ships
with curated databases covering 30 world regions, 120+ gene-nutrient
associations, and 500+ food items with full macronutrient and
micronutrient profiles.

**This module is intended for educational and research purposes only.
Dietary recommendations must NOT replace professional nutritional or
medical advice.**

Typical usage
-------------
>>> advisor = DietAdvisor()
>>> recs = advisor.generate_recommendations(
...     genetic_risks=["Type 2 diabetes", "Coronary artery disease"],
...     variants={"rs1801133": "CT", "rs4988235": "CC"},
...     region="south_asia_north",
...     age=45,
...     sex="male",
... )
>>> for r in recs[:3]:
...     print(f"{r.nutrient}: {r.recommendation}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# JSON data directory & loader
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"


def _load_json(name: str) -> Any:
    """Read and return parsed JSON from *_DATA_DIR / name*."""
    with open(_DATA_DIR / name) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "NutrientNeed",
    "FoodItem",
    "MealPlan",
    "DietaryRecommendation",
    "GeographicProfile",
    "DietAdvisor",
    "GEOGRAPHIC_FOOD_DB",
    "NUTRIGENOMICS_DB",
    "FOOD_DATABASE",
]

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

_DISCLAIMER: str = (
    "These dietary recommendations are generated for educational and "
    "research purposes only.  They do not constitute medical or "
    "nutritional advice.  Consult a registered dietitian or physician "
    "before making dietary changes based on genetic information."
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NutrientNeed:
    """A single nutrient requirement derived from genetic and demographic data.

    Attributes
    ----------
    nutrient : str
        Name of the nutrient (e.g. ``"folate"``, ``"omega_3"``).
    daily_target_mg : float
        Recommended daily intake in milligrams.
    priority : str
        One of ``"critical"``, ``"high"``, ``"moderate"``, ``"low"``.
    source_gene : str
        Gene symbol whose variant drives this requirement.
    source_variant : str
        dbSNP rsid or variant identifier.
    rationale : str
        Human-readable explanation of why this nutrient is needed.
    upper_limit_mg : float
        Tolerable upper intake level in milligrams (0.0 if not set).
    """

    nutrient: str
    daily_target_mg: float
    priority: str
    source_gene: str
    source_variant: str
    rationale: str
    upper_limit_mg: float = 0.0


@dataclass(frozen=True)
class FoodItem:
    """Nutritional profile for a single food.

    Attributes
    ----------
    name : str
        Common English name.
    calories_per_100g : float
        Energy density (kcal per 100 g edible portion).
    protein : float
        Protein content (g per 100 g).
    carbs : float
        Total carbohydrate content (g per 100 g).
    fat : float
        Total fat content (g per 100 g).
    fiber : float
        Dietary fibre (g per 100 g).
    key_micronutrients : dict[str, float]
        Mapping of micronutrient name → amount per 100 g (mg unless
        otherwise noted in the key, e.g. ``"vitamin_d_iu"``).
    food_group : str
        Category such as ``"grain"``, ``"legume"``, ``"vegetable"``, etc.
    regions : list[str]
        Region identifiers where this food is commonly available.
    """

    name: str
    calories_per_100g: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    key_micronutrients: dict[str, float]
    food_group: str
    regions: list[str]


@dataclass()
class MealPlan:
    """A single day's meal plan.

    Attributes
    ----------
    day : int
        Day number (1-based) within a multi-day plan.
    breakfast : list[tuple[FoodItem, float]]
        (food, portion_grams) pairs for breakfast.
    lunch : list[tuple[FoodItem, float]]
        (food, portion_grams) pairs for lunch.
    dinner : list[tuple[FoodItem, float]]
        (food, portion_grams) pairs for dinner.
    snacks : list[tuple[FoodItem, float]]
        (food, portion_grams) pairs for snacks.
    total_calories : float
        Sum of calories across all meals.
    total_macros : dict[str, float]
        Aggregated macronutrient totals (``"protein"``, ``"carbs"``,
        ``"fat"``, ``"fiber"`` in grams).
    notes : list[str]
        Advisory notes specific to this day's plan.
    """

    day: int
    breakfast: list[tuple[FoodItem, float]] = field(default_factory=list)
    lunch: list[tuple[FoodItem, float]] = field(default_factory=list)
    dinner: list[tuple[FoodItem, float]] = field(default_factory=list)
    snacks: list[tuple[FoodItem, float]] = field(default_factory=list)
    total_calories: float = 0.0
    total_macros: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all items across meals."""
        rows: list[dict[str, Any]] = []
        for meal_name, items in [
            ("breakfast", self.breakfast),
            ("lunch", self.lunch),
            ("dinner", self.dinner),
            ("snacks", self.snacks),
        ]:
            for food, grams in items:
                scale = grams / 100.0
                rows.append(
                    {
                        "meal": meal_name,
                        "food": food.name,
                        "grams": grams,
                        "calories": round(food.calories_per_100g * scale, 1),
                        "protein_g": round(food.protein * scale, 1),
                        "carbs_g": round(food.carbs * scale, 1),
                        "fat_g": round(food.fat * scale, 1),
                        "fiber_g": round(food.fiber * scale, 1),
                    }
                )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class DietaryRecommendation:
    """A single actionable dietary recommendation.

    Attributes
    ----------
    nutrient : str
        Target nutrient or dietary factor.
    recommendation : str
        Human-readable recommendation text.
    priority : str
        ``"critical"``, ``"high"``, ``"moderate"``, or ``"low"``.
    target_foods : list[str]
        Names of recommended foods.
    avoid_foods : list[str]
        Names of foods to limit or avoid.
    genetic_basis : str
        Brief explanation of the genetic rationale.
    daily_target : str
        Suggested daily intake string (e.g. ``"400 µg DFE"``).
    confidence : float
        Confidence in the recommendation (0–1).
    """

    nutrient: str
    recommendation: str
    priority: str
    target_foods: list[str]
    avoid_foods: list[str]
    genetic_basis: str
    daily_target: str
    confidence: float = 0.7


@dataclass(frozen=True)
class GeographicProfile:
    """Dietary profile for a world region.

    Attributes
    ----------
    region_id : str
        Machine-readable identifier (e.g. ``"south_asia_north"``).
    name : str
        Human-readable region name.
    common_foods : list[str]
        Widely consumed foods in this region.
    staple_grains : list[str]
        Primary grain staples.
    protein_sources : list[str]
        Common protein sources (animal and plant).
    vegetables : list[str]
        Commonly consumed vegetables.
    fruits : list[str]
        Commonly consumed fruits.
    spices : list[str]
        Characteristic spices and seasonings.
    traditional_dishes : list[str]
        Representative traditional dishes.
    """

    region_id: str
    name: str
    common_foods: list[str]
    staple_grains: list[str]
    protein_sources: list[str]
    vegetables: list[str]
    fruits: list[str]
    spices: list[str]
    traditional_dishes: list[str]


# ---------------------------------------------------------------------------
# Geographic Food Database  (30 regions)
# ---------------------------------------------------------------------------

GEOGRAPHIC_FOOD_DB: list[GeographicProfile] = [
    GeographicProfile(**entry) for entry in _load_json("geographic_profiles.json")
]

# Build a quick-lookup dict by region_id.
_REGION_INDEX: dict[str, GeographicProfile] = {gp.region_id: gp for gp in GEOGRAPHIC_FOOD_DB}

# ---------------------------------------------------------------------------
# Nutrigenomics Database  (25+ gene–nutrient associations)
# ---------------------------------------------------------------------------

# Each entry: (rsid, gene, nutrient, effect, recommendation,
#              priority, daily_target_mg, upper_limit_mg, confidence)
NUTRIGENOMICS_DB: list[dict[str, Any]] = _load_json("nutrigenomics_database.json")

# Quick count assertion
assert len(NUTRIGENOMICS_DB) >= 100, (
    f"Nutrigenomics DB has only {len(NUTRIGENOMICS_DB)} entries; expected ≥ 25."
)

# Index by rsid for fast lookup.
_NUTRI_RSID_INDEX: dict[str, list[dict[str, Any]]] = {}
for _entry in NUTRIGENOMICS_DB:
    _NUTRI_RSID_INDEX.setdefault(_entry["rsid"], []).append(_entry)


# ---------------------------------------------------------------------------
# Food Database  (100+ items)
# ---------------------------------------------------------------------------



FOOD_DATABASE: list[FoodItem] = [
    FoodItem(**entry) for entry in _load_json("food_database.json")
]

# Quick count assertion.
assert len(FOOD_DATABASE) >= 450, f"Food DB has only {len(FOOD_DATABASE)} entries; expected ≥ 450."

# Index foods by name (lower-case) for quick lookup.
_FOOD_NAME_INDEX: dict[str, FoodItem] = {fi.name.lower(): fi for fi in FOOD_DATABASE}

# Index foods by region.
_FOOD_REGION_INDEX: dict[str, list[FoodItem]] = {}
for _food_item in FOOD_DATABASE:
    for _region in _food_item.regions:
        _FOOD_REGION_INDEX.setdefault(_region, []).append(_food_item)

# Index foods by food group.
_FOOD_GROUP_INDEX: dict[str, list[FoodItem]] = {}
for _food_item in FOOD_DATABASE:
    _FOOD_GROUP_INDEX.setdefault(_food_item.food_group, []).append(_food_item)


# ---------------------------------------------------------------------------
# Nutrient → food lookup table (maps nutrient names to micronutrient keys
# used in FoodItem.key_micronutrients)
# ---------------------------------------------------------------------------

_NUTRIENT_MICRO_KEYS: dict[str, list[str]] = _load_json("nutrient_micro_keys.json")

# ---------------------------------------------------------------------------
# Dietary restriction filters
# ---------------------------------------------------------------------------

_RESTRICTION_EXCLUDED_GROUPS: dict[str, set[str]] = {
    k: set(v) for k, v in _load_json("restriction_excluded_groups.json").items()
}

_dietary_kw = _load_json("dietary_filter_keywords.json")
_VEG_SAFE_GROUPS: set[str] = set(_dietary_kw["veg_safe_groups"])
_VEGAN_SAFE_GROUPS: set[str] = set(_dietary_kw["vegan_safe_groups"])
_NONVEG_KEYWORDS: set[str] = set(_dietary_kw["nonveg_keywords"])
_NONVEGAN_EXTRA_KEYWORDS: set[str] = set(_dietary_kw["nonvegan_extra_keywords"])
_NONVEGAN_KEYWORDS: set[str] = _NONVEG_KEYWORDS | _NONVEGAN_EXTRA_KEYWORDS
_GLUTEN_KEYWORDS: set[str] = set(_dietary_kw["gluten_keywords"])
_HALAL_EXCLUDED: set[str] = set(_dietary_kw["halal_excluded"])
_KOSHER_EXCLUDED: set[str] = set(_dietary_kw["kosher_excluded"])
_DAIRY_KEYWORDS: set[str] = set(_dietary_kw["dairy_keywords"])
_DAIRY_SAFE_NAMES: set[str] = set(_dietary_kw["dairy_safe_names"])
_NUT_KEYWORDS: set[str] = set(_dietary_kw["nut_keywords"])
_NUT_SAFE_NAMES: set[str] = set(_dietary_kw["nut_safe_names"])
_PESCATARIAN_MEAT_KEYWORDS: set[str] = set(_dietary_kw["pescatarian_meat_keywords"])
_PESCATARIAN_SAFE_GROUPS: set[str] = set(_dietary_kw["pescatarian_safe_groups"])
_VEGAN_DAIRY_KEYWORDS: set[str] = set(_dietary_kw["vegan_dairy_keywords"])
del _dietary_kw

# ---------------------------------------------------------------------------
# Telomere-protective nutrients (evidence-based associations)
# ---------------------------------------------------------------------------

_TELOMERE_PROTECTIVE_NUTRIENTS: list[dict[str, Any]] = _load_json(
    "telomere_protective_nutrients.json"
)


# ---------------------------------------------------------------------------
# DietAdvisor — main engine
# ---------------------------------------------------------------------------


class DietAdvisor:
    """Nutrigenomics-informed dietary recommendation engine.

    Generates personalised dietary guidance by combining genetic variant
    data, disease risk profiles, telomere-length measurements, regional
    food availability, and demographic factors.

    Parameters
    ----------
    food_db : list[FoodItem] or None
        Override the built-in food database.  When ``None`` (default),
        :data:`FOOD_DATABASE` is used.
    nutrigenomics_db : list[dict] or None
        Override the built-in nutrigenomics database.

    Examples
    --------
    >>> advisor = DietAdvisor()
    >>> needs = advisor.calculate_nutrient_needs(
    ...     variants={"rs1801133": "CT"},
    ...     age=40,
    ...     sex="female",
    ... )
    >>> needs[0].nutrient
    'folate'
    """

    def __init__(
        self,
        food_db: list[FoodItem] | None = None,
        nutrigenomics_db: list[dict[str, Any]] | None = None,
    ) -> None:
        self._food_db: list[FoodItem] = food_db or list(FOOD_DATABASE)
        self._nutri_db: list[dict[str, Any]] = nutrigenomics_db or list(NUTRIGENOMICS_DB)

        # Build internal indices.
        self._region_index: dict[str, list[FoodItem]] = {}
        for fi in self._food_db:
            for r in fi.regions:
                self._region_index.setdefault(r, []).append(fi)

        self._group_index: dict[str, list[FoodItem]] = {}
        for fi in self._food_db:
            self._group_index.setdefault(fi.food_group, []).append(fi)

        self._rsid_nutri: dict[str, list[dict[str, Any]]] = {}
        for entry in self._nutri_db:
            self._rsid_nutri.setdefault(entry["rsid"], []).append(entry)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def calculate_nutrient_needs(
        self,
        variants: dict[str, str],
        age: int,
        sex: str,
    ) -> list[NutrientNeed]:
        """Determine nutrient requirements based on genetic variants and demographics.

        For each variant present in the input that matches the
        nutrigenomics database, a :class:`NutrientNeed` is generated.
        The daily target is adjusted for age and sex using standard
        DRI scaling factors.

        Parameters
        ----------
        variants : dict[str, str]
            ``rsid`` → diploid genotype (e.g. ``{"rs1801133": "CT"}``).
        age : int
            Age of the individual in years.
        sex : str
            ``"male"`` or ``"female"``.

        Returns
        -------
        list[NutrientNeed]
            Prioritised nutrient requirements, sorted by priority
            (critical → high → moderate → low).
        """
        sex = sex.lower().strip()
        needs: list[NutrientNeed] = []
        seen_nutrients: dict[str, NutrientNeed] = {}

        for rsid, genotype in variants.items():
            entries = self._rsid_nutri.get(rsid, [])
            for entry in entries:
                # Determine allele dosage heuristic: heterozygous (1 risk
                # allele) → moderate effect; homozygous (2) → full effect.
                dosage = self._estimate_dosage(genotype)
                if dosage == 0:
                    continue

                nutrient = entry["nutrient"]
                base_target = entry["daily_target_mg"]
                # Scale target by dosage and demographic modifiers.
                adjusted_target = self._adjust_target(base_target, dosage, age, sex, nutrient)
                upper = entry["upper_limit_mg"]
                priority = entry["priority"]
                # Upgrade priority for homozygous carriers.
                if dosage == 2 and priority == "moderate":
                    priority = "high"

                need = NutrientNeed(
                    nutrient=nutrient,
                    daily_target_mg=round(adjusted_target, 3),
                    priority=priority,
                    source_gene=entry["gene"],
                    source_variant=rsid,
                    rationale=entry["effect"],
                    upper_limit_mg=upper,
                )

                # Keep the higher-priority entry if duplicates exist.
                existing = seen_nutrients.get(nutrient)
                if existing is None or _PRIORITY_RANK.get(need.priority, 0) > _PRIORITY_RANK.get(
                    existing.priority, 0
                ):
                    seen_nutrients[nutrient] = need

        needs = list(seen_nutrients.values())
        needs.sort(key=lambda n: _PRIORITY_RANK.get(n.priority, 0), reverse=True)
        return needs

    def generate_recommendations(
        self,
        genetic_risks: list[str],
        variants: dict[str, str],
        region: str,
        age: int,
        sex: str,
        dietary_restrictions: list[str] | None = None,
    ) -> list[DietaryRecommendation]:
        """Generate comprehensive dietary recommendations.

        Combines nutrigenomic analysis of *variants* with disease-risk
        context from *genetic_risks* and the regional food profile
        identified by *region* to produce actionable dietary guidance.

        Parameters
        ----------
        genetic_risks : list[str]
            Disease/condition names the individual is at elevated risk
            for (e.g. ``["Type 2 diabetes", "Coronary artery disease"]``).
        variants : dict[str, str]
            ``rsid`` → diploid genotype mapping.
        region : str
            Region identifier (must match a key in :data:`GEOGRAPHIC_FOOD_DB`).
        age : int
            Age in years.
        sex : str
            ``"male"`` or ``"female"``.
        dietary_restrictions : list[str] or None
            Optional restrictions such as ``["vegetarian", "gluten_free"]``.

        Returns
        -------
        list[DietaryRecommendation]
            Recommendations sorted by priority (critical first).
        """
        restrictions = dietary_restrictions or []
        nutrient_needs = self.calculate_nutrient_needs(variants, age, sex)

        # Also incorporate disease-risk–driven dietary guidance.
        risk_nutrients = self._nutrients_for_risks(genetic_risks)

        # Merge risk-driven nutrients with variant-driven needs.
        combined: dict[str, dict[str, Any]] = {}
        for nn in nutrient_needs:
            combined[nn.nutrient] = {
                "priority": nn.priority,
                "rationale": nn.rationale,
                "gene": nn.source_gene,
                "daily_target": nn.daily_target_mg,
            }
        for rn in risk_nutrients:
            if rn["nutrient"] not in combined:
                combined[rn["nutrient"]] = rn
            else:
                # Upgrade priority if risk-driven is higher.
                existing = combined[rn["nutrient"]]
                if _PRIORITY_RANK.get(rn.get("priority", "low"), 0) > _PRIORITY_RANK.get(
                    existing.get("priority", "low"), 0
                ):
                    existing["priority"] = rn["priority"]

        recommendations: list[DietaryRecommendation] = []
        for nutrient, info in combined.items():
            target_foods = self._find_target_foods(nutrient, region, restrictions)
            avoid_foods = self._find_avoid_foods(nutrient, genetic_risks)
            daily_str = self._format_daily_target(nutrient, info.get("daily_target", 0.0))
            # Find the matching nutrigenomics entry for confidence.
            confidence = self._lookup_confidence(nutrient)

            rec = DietaryRecommendation(
                nutrient=nutrient,
                recommendation=self._build_recommendation_text(nutrient, info, region),
                priority=info.get("priority", "moderate"),
                target_foods=[f.name for f in target_foods[:8]],
                avoid_foods=avoid_foods,
                genetic_basis=info.get(
                    "rationale",
                    f"Elevated risk associated with {nutrient} metabolism",
                ),
                daily_target=daily_str,
                confidence=confidence,
            )
            recommendations.append(rec)

        # Sort by priority.
        recommendations.sort(key=lambda r: _PRIORITY_RANK.get(r.priority, 0), reverse=True)
        return recommendations

    def create_meal_plan(
        self,
        recommendations: list[DietaryRecommendation],
        region: str,
        calories: int = 2000,
        days: int = 7,
        dietary_restrictions: list[str] | None = None,
    ) -> list[MealPlan]:
        """Create a multi-day meal plan aligned to recommendations and region.

        The planner selects foods from the regional database that
        maximise coverage of recommended nutrients while staying within
        the caloric target (±10%).

        Parameters
        ----------
        recommendations : list[DietaryRecommendation]
            Dietary recommendations (from :meth:`generate_recommendations`).
        region : str
            Region identifier.
        calories : int
            Daily calorie target (default 2 000).
        days : int
            Number of days to plan (default 7).
        dietary_restrictions : list[str] | None
            E.g. ``["vegetarian", "gluten_free"]``.  Foods violating these
            restrictions are excluded from the candidate pools up-front.

        Returns
        -------
        list[MealPlan]
            One :class:`MealPlan` per day.
        """
        regional_foods = self._region_index.get(region, self._food_db)
        if not regional_foods:
            regional_foods = self._food_db

        # Pre-filter foods by dietary restrictions so non-veg items
        # never enter the candidate pools for vegetarian plans, etc.
        restrictions = dietary_restrictions or []
        if restrictions:
            regional_foods = [
                fi for fi in regional_foods if self._food_passes_restrictions(fi, restrictions)
            ]

        # Score foods by how well they cover recommended nutrients.
        target_nutrients = {r.nutrient for r in recommendations}
        scored = self._score_foods(regional_foods, target_nutrients)

        # Split foods into meal-appropriate groups.
        breakfast_candidates = [
            fi
            for fi in scored
            if fi.food_group
            in ("grain", "fruit", "dairy", "eggs", "nut", "seed", "prepared", "beverage")
        ]
        main_candidates = [
            fi
            for fi in scored
            if fi.food_group
            in ("grain", "legume", "vegetable", "fish", "meat", "poultry", "prepared", "oil")
        ]
        snack_candidates = [
            fi
            for fi in scored
            if fi.food_group in ("fruit", "nut", "seed", "snack", "dairy", "beverage")
        ]

        rng = np.random.default_rng(seed=42)
        plans: list[MealPlan] = []

        # Track recently used food names to avoid repetition in long plans.
        _recent_breakfasts: list[set[str]] = []
        _recent_lunches: list[set[str]] = []
        _recent_dinners: list[set[str]] = []

        for day in range(1, days + 1):
            # Build a penalty set from the last 3 days to discourage repetition.
            avoid_breakfast: set[str] = set()
            avoid_lunch: set[str] = set()
            avoid_dinner: set[str] = set()
            lookback = min(3, len(_recent_breakfasts))
            for i in range(1, lookback + 1):
                avoid_breakfast |= _recent_breakfasts[-i]
                avoid_lunch |= _recent_lunches[-i]
                avoid_dinner |= _recent_dinners[-i]

            plan = self._build_day_plan(
                day=day,
                calories_target=calories,
                breakfast_pool=breakfast_candidates,
                main_pool=main_candidates,
                snack_pool=snack_candidates,
                rng=rng,
                avoid_breakfast=avoid_breakfast,
                avoid_lunch=avoid_lunch,
                avoid_dinner=avoid_dinner,
            )
            plans.append(plan)

            # Record what was used today.
            _recent_breakfasts.append({fi.name for fi, _ in plan.breakfast})
            _recent_lunches.append({fi.name for fi, _ in plan.lunch})
            _recent_dinners.append({fi.name for fi, _ in plan.dinner})

        return plans

    def get_region_specific_foods(
        self,
        region: str,
        nutrient: str,
    ) -> list[FoodItem]:
        """Find foods rich in a nutrient that are available in a region.

        Parameters
        ----------
        region : str
            Region identifier.
        nutrient : str
            Nutrient name (must be a key in the internal mapping).

        Returns
        -------
        list[FoodItem]
            Foods sorted by descending nutrient density.
        """
        regional = self._region_index.get(region, [])
        micro_keys = _NUTRIENT_MICRO_KEYS.get(nutrient, [nutrient])

        def nutrient_score(fi: FoodItem) -> float:
            return sum(fi.key_micronutrients.get(k, 0.0) for k in micro_keys)

        enriched = [(fi, nutrient_score(fi)) for fi in regional]
        enriched = [(fi, s) for fi, s in enriched if s > 0]
        enriched.sort(key=lambda t: t[1], reverse=True)
        return [fi for fi, _ in enriched]

    def get_telomere_protective_diet(
        self,
        telomere_data: dict[str, Any],
        region: str,
    ) -> list[DietaryRecommendation]:
        """Generate diet recommendations to support telomere maintenance.

        Uses the individual's telomere-length data (mean length, age)
        to prioritise nutrients with evidence for telomere protection.
        Shorter-than-expected telomeres trigger higher urgency.

        Parameters
        ----------
        telomere_data : dict[str, Any]
            Must contain ``"mean_length_bp"`` (float) and ``"age"`` (int).
            Optionally ``"sex"`` (str, default ``"female"``).
        region : str
            Region identifier for food selection.

        Returns
        -------
        list[DietaryRecommendation]
            Telomere-specific dietary recommendations.
        """
        mean_bp = float(telomere_data.get("mean_length_bp", 7000.0))
        age = int(telomere_data.get("age", 50))
        str(telomere_data.get("sex", "female")).lower()

        # Expected telomere length — consensus linear model
        # (Müezzinler et al. 2013; Aubert & Lansdorp 2008).
        expected_bp = 11_000.0 - 40.0 * age
        shortening_pct = max(0.0, (expected_bp - mean_bp) / expected_bp) * 100

        # Determine urgency multiplier.
        if shortening_pct > 20:
            urgency = "critical"
            urgency_mult = 1.5
        elif shortening_pct > 10:
            urgency = "high"
            urgency_mult = 1.2
        else:
            urgency = "moderate"
            urgency_mult = 1.0

        recommendations: list[DietaryRecommendation] = []
        for entry in _TELOMERE_PROTECTIVE_NUTRIENTS:
            nutrient = entry["nutrient"]
            target_foods = self.get_region_specific_foods(region, nutrient)
            priority = entry["priority"]
            # Escalate priority if telomeres are shorter than expected.
            if urgency == "critical" and priority != "critical":
                priority = "high" if priority == "moderate" else "critical"
            elif urgency == "high" and priority == "low":
                priority = "moderate"

            rec_text = (
                f"Increase {nutrient.replace('_', ' ')} intake to support "
                f"telomere maintenance. {entry['effect']}. "
                f"Telomere shortening is {shortening_pct:.1f}% beyond "
                f"age-expected length."
            )
            rec = DietaryRecommendation(
                nutrient=nutrient,
                recommendation=rec_text,
                priority=priority,
                target_foods=[f.name for f in target_foods[:6]],
                avoid_foods=self._telomere_damaging_foods(),
                genetic_basis=entry["effect"],
                daily_target=self._telomere_daily_target(nutrient, urgency_mult),
                confidence=entry["confidence"],
            )
            recommendations.append(rec)

        recommendations.sort(key=lambda r: _PRIORITY_RANK.get(r.priority, 0), reverse=True)
        return recommendations

    def adapt_to_restrictions(
        self,
        meal_plan: list[MealPlan],
        restrictions: list[str],
    ) -> list[MealPlan]:
        """Adapt an existing meal plan to dietary restrictions.

        Replaces foods that violate the given restrictions with
        nutritionally similar alternatives from the same region where
        possible.

        Parameters
        ----------
        meal_plan : list[MealPlan]
            Existing multi-day meal plan.
        restrictions : list[str]
            E.g. ``["vegetarian", "gluten_free", "halal"]``.

        Returns
        -------
        list[MealPlan]
            Adjusted meal plans with violating foods replaced.
        """
        adapted: list[MealPlan] = []
        for day_plan in meal_plan:
            new_plan = MealPlan(
                day=day_plan.day,
                breakfast=self._filter_meal(day_plan.breakfast, restrictions),
                lunch=self._filter_meal(day_plan.lunch, restrictions),
                dinner=self._filter_meal(day_plan.dinner, restrictions),
                snacks=self._filter_meal(day_plan.snacks, restrictions),
                notes=list(day_plan.notes),
            )
            # Re-calculate totals.
            new_plan.total_calories = self._sum_calories(new_plan)
            new_plan.total_macros = self._sum_macros(new_plan)
            new_plan.notes.append(f"Adapted for: {', '.join(restrictions)}")
            adapted.append(new_plan)
        return adapted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_dosage(genotype: str) -> int:
        """Heuristic dosage from a genotype string.

        Two identical non-reference alleles → 2 (homozygous);
        mixed → 1 (heterozygous); two reference → 0.
        """
        if len(genotype) != 2:
            return 0
        a, b = genotype[0], genotype[1]
        if a != b:
            return 1
        # Homozygous — assume risk if non-reference.
        return 2

    @staticmethod
    def _adjust_target(
        base_mg: float,
        dosage: int,
        age: int,
        sex: str,
        nutrient: str,
    ) -> float:
        """Scale a daily target by dosage, age, and sex.

        Uses simple multiplicative modifiers:
        - Dosage 2 → ×1.25
        - Age > 65 → ×1.15 (older adults need more of most nutrients)
        - Female during reproductive years → ×1.10 for folate/iron
        """
        target = base_mg
        if dosage == 2:
            target *= 1.25
        if age > 65:
            target *= 1.15
        if sex == "female" and 18 <= age <= 50:
            if nutrient in ("folate", "iron"):
                target *= 1.10
        return target

    def _nutrients_for_risks(
        self,
        genetic_risks: list[str],
    ) -> list[dict[str, Any]]:
        """Map disease risks to dietary nutrient priorities."""
        risk_map: dict[str, dict[str, Any]] = {
            "Type 2 diabetes": {
                "nutrient": "glycemic_control",
                "priority": "critical",
                "rationale": "Genetic predisposition to impaired glucose homeostasis",
                "daily_target": 0.0,
            },
            "Coronary artery disease": {
                "nutrient": "omega_3",
                "priority": "critical",
                "rationale": "Cardioprotective effect of omega-3 fatty acids",
                "daily_target": 1000.0,
            },
            "Obesity": {
                "nutrient": "calories",
                "priority": "high",
                "rationale": "Genetic predisposition to weight gain",
                "daily_target": 0.0,
            },
            "Alzheimer's disease": {
                "nutrient": "polyphenols",
                "priority": "high",
                "rationale": "Neuroprotective polyphenols may slow cognitive decline",
                "daily_target": 0.0,
            },
            "Osteoporosis": {
                "nutrient": "calcium",
                "priority": "critical",
                "rationale": "Genetic variants affecting bone mineral density",
                "daily_target": 1200.0,
            },
            "Age-related macular degeneration": {
                "nutrient": "vitamin_a",
                "priority": "high",
                "rationale": "Lutein/zeaxanthin and vitamin A for retinal health",
                "daily_target": 0.9,
            },
            "Hyperhomocysteinaemia": {
                "nutrient": "folate",
                "priority": "critical",
                "rationale": "Folate required to metabolise homocysteine",
                "daily_target": 0.8,
            },
            "Hereditary haemochromatosis": {
                "nutrient": "iron",
                "priority": "critical",
                "rationale": "Iron overload risk; restrict dietary iron",
                "daily_target": 8.0,
            },
            "Breast cancer": {
                "nutrient": "cruciferous_vegetables",
                "priority": "high",
                "rationale": "Sulforaphane and indole-3-carbinol may be protective",
                "daily_target": 0.0,
            },
            "Colorectal cancer": {
                "nutrient": "fiber",
                "priority": "high",
                "rationale": "High-fibre diet associated with reduced colorectal cancer risk",
                "daily_target": 0.0,
            },
        }
        results: list[dict[str, Any]] = []
        for risk in genetic_risks:
            if risk in risk_map:
                results.append(risk_map[risk])
        return results

    def _find_target_foods(
        self,
        nutrient: str,
        region: str,
        restrictions: list[str],
    ) -> list[FoodItem]:
        """Find region-specific foods rich in a nutrient, respecting restrictions."""
        candidates = self.get_region_specific_foods(region, nutrient)
        if not candidates:
            # Fall back to global database.
            candidates = self.get_region_specific_foods("north_america", nutrient)

        filtered: list[FoodItem] = []
        for fi in candidates:
            if self._food_passes_restrictions(fi, restrictions):
                filtered.append(fi)
        return filtered

    @staticmethod
    def _find_avoid_foods(
        nutrient: str,
        genetic_risks: list[str],
    ) -> list[str]:
        """Determine foods to avoid given nutrient context and risks."""
        avoid: list[str] = []
        if nutrient == "saturated_fat" or "Coronary artery disease" in genetic_risks:
            avoid.extend(["processed meat", "deep-fried foods", "butter", "palm oil", "margarine"])
        if nutrient == "glycemic_control" or "Type 2 diabetes" in genetic_risks:
            avoid.extend(
                ["white bread", "sugary drinks", "candy", "white rice (excess)", "pastries"]
            )
        if nutrient == "lactose":
            avoid.extend(["whole milk", "ice cream", "soft cheese"])
        if nutrient == "caffeine":
            avoid.extend(["energy drinks", "excessive coffee", "high-caffeine supplements"])
        if nutrient == "iron" and "Hereditary haemochromatosis" in genetic_risks:
            avoid.extend(["iron supplements", "liver", "excessive red meat"])
        if nutrient == "alcohol":
            avoid.extend(["beer", "wine", "spirits"])
        return avoid

    @staticmethod
    def _format_daily_target(nutrient: str, target_mg: float) -> str:
        """Format a daily target into a human-readable string."""
        if target_mg <= 0:
            unit_map: dict[str, str] = {
                "calories": "Reduce total intake by 10–15%",
                "saturated_fat": "< 7% of total calories",
                "glycemic_control": "GI < 55 for most carb sources",
                "lactose": "Minimise or eliminate dairy lactose",
                "caffeine": "≤ 200 mg/day",
                "cruciferous_vegetables": "≥ 3 servings/week",
                "anti_inflammatory": "Anti-inflammatory diet pattern",
                "polyphenols": "≥ 500 mg total polyphenols/day",
                "monounsaturated_fat": "25–35% of fat from MUFA",
                "alcohol": "Avoid or strictly minimise",
                "antioxidants": "≥ 5 servings fruits & vegetables/day",
            }
            return unit_map.get(nutrient, "See specific guidance")

        if target_mg < 1.0:
            return f"{target_mg * 1000:.0f} µg/day"
        return f"{target_mg:.0f} mg/day"

    def _lookup_confidence(self, nutrient: str) -> float:
        """Find the highest confidence score for a nutrient in the DB."""
        best = 0.5
        for entry in self._nutri_db:
            if entry["nutrient"] == nutrient:
                best = max(best, entry.get("confidence", 0.5))
        return best

    def _build_recommendation_text(
        self,
        nutrient: str,
        info: dict[str, Any],
        region: str,
    ) -> str:
        """Compose a detailed recommendation string."""
        region_profile = _REGION_INDEX.get(region)
        region_name = region_profile.name if region_profile else region

        base = info.get("rationale", "")
        gene = info.get("gene", "")
        parts = [
            f"Based on {'your ' + gene + ' variant' if gene else 'risk profile'}: {base}.",
            f"Focus on {nutrient.replace('_', ' ')}-rich foods available in {region_name}.",
        ]

        # Add region-specific advice.
        top_foods = self.get_region_specific_foods(region, nutrient)[:3]
        if top_foods:
            names = ", ".join(f.name for f in top_foods)
            parts.append(f"Recommended regional sources: {names}.")

        return " ".join(parts)

    def _score_foods(
        self,
        foods: list[FoodItem],
        target_nutrients: set[str],
    ) -> list[FoodItem]:
        """Score and sort foods by nutrient coverage, returning sorted list."""
        micro_keys_flat: set[str] = set()
        for nut in target_nutrients:
            micro_keys_flat.update(_NUTRIENT_MICRO_KEYS.get(nut, [nut]))

        def coverage(fi: FoodItem) -> float:
            s = 0.0
            for k in micro_keys_flat:
                if fi.key_micronutrients.get(k, 0.0) > 0:
                    s += 1.0
            # Bonus for high fibre and protein density.
            s += min(fi.fiber / 10.0, 0.5)
            s += min(fi.protein / 30.0, 0.5)
            return s

        return sorted(foods, key=coverage, reverse=True)

    def _build_day_plan(
        self,
        day: int,
        calories_target: int,
        breakfast_pool: list[FoodItem],
        main_pool: list[FoodItem],
        snack_pool: list[FoodItem],
        rng: np.random.Generator,
        avoid_breakfast: set[str] | None = None,
        avoid_lunch: set[str] | None = None,
        avoid_dinner: set[str] | None = None,
    ) -> MealPlan:
        """Construct a single day's meal plan using greedy selection."""
        # Calorie budgets: breakfast 25%, lunch 35%, dinner 30%, snacks 10%.
        budget_bkf = calories_target * 0.25
        budget_lun = calories_target * 0.35
        budget_din = calories_target * 0.30
        budget_snk = calories_target * 0.10

        breakfast = self._select_meal(breakfast_pool, budget_bkf, rng, day, avoid_breakfast)
        lunch = self._select_meal(main_pool, budget_lun, rng, day + 100, avoid_lunch)
        dinner = self._select_meal(main_pool, budget_din, rng, day + 200, avoid_dinner)
        snacks = self._select_meal(snack_pool, budget_snk, rng, day + 300)

        plan = MealPlan(
            day=day,
            breakfast=breakfast,
            lunch=lunch,
            dinner=dinner,
            snacks=snacks,
        )
        plan.total_calories = self._sum_calories(plan)
        plan.total_macros = self._sum_macros(plan)
        return plan

    @staticmethod
    def _select_meal(
        pool: list[FoodItem],
        calorie_budget: float,
        rng: np.random.Generator,
        seed_offset: int = 0,
        avoid_names: set[str] | None = None,
    ) -> list[tuple[FoodItem, float]]:
        """Greedily select foods to approximately fill a calorie budget.

        Introduces controlled randomness via shuffling to ensure variety
        across days.  When *avoid_names* is provided, those foods are
        pushed to the end of the candidate order so they are only used
        as a last resort.
        """
        if not pool:
            return []

        # Shuffle with offset for day-level variety.
        indices = np.arange(len(pool))
        rng_local = np.random.default_rng(seed=rng.integers(0, 2**31) + seed_offset)
        rng_local.shuffle(indices)

        # Push recently-used foods to the back of the order.
        if avoid_names:
            preferred = [i for i in indices if pool[int(i)].name not in avoid_names]
            avoided = [i for i in indices if pool[int(i)].name in avoid_names]
            indices = np.array(preferred + avoided)

        selected: list[tuple[FoodItem, float]] = []
        remaining = calorie_budget
        items_added = 0
        max_items = 5  # cap per meal

        for idx in indices:
            if remaining <= 20 or items_added >= max_items:
                break
            fi = pool[int(idx)]
            if fi.calories_per_100g <= 0:
                continue
            # Determine portion: aim for ~1/3 of remaining budget per item,
            # clamped to [50, 300] grams.
            target_cal = remaining / max(1, max_items - items_added)
            portion_g = (target_cal / fi.calories_per_100g) * 100.0
            portion_g = float(np.clip(portion_g, 50.0, 300.0))
            actual_cal = fi.calories_per_100g * (portion_g / 100.0)
            selected.append((fi, round(portion_g, 0)))
            remaining -= actual_cal
            items_added += 1

        return selected

    def _filter_meal(
        self,
        items: list[tuple[FoodItem, float]],
        restrictions: list[str],
    ) -> list[tuple[FoodItem, float]]:
        """Filter meal items against restrictions, substituting where possible."""
        result: list[tuple[FoodItem, float]] = []
        for fi, grams in items:
            if self._food_passes_restrictions(fi, restrictions):
                result.append((fi, grams))
            else:
                # Attempt substitution with same food group.
                sub = self._find_substitute(fi, restrictions)
                if sub is not None:
                    result.append((sub, grams))
        return result

    def _food_passes_restrictions(
        self,
        fi: FoodItem,
        restrictions: list[str],
    ) -> bool:
        """Check whether a food item satisfies all dietary restrictions."""
        name_lower = fi.name.lower()

        for restriction in restrictions:
            restriction = restriction.lower().replace("-", "_")

            # Group-based restrictions.
            excluded_groups = _RESTRICTION_EXCLUDED_GROUPS.get(restriction)
            if excluded_groups and fi.food_group in excluded_groups:
                return False

            # Name-based keyword check for vegetarian/vegan to catch
            # non-veg "prepared" dishes (e.g. "butter chicken").
            # Only applied to food groups NOT in the safe set to avoid
            # false positives like "goat cheese" (dairy).
            if restriction == "vegetarian":
                if fi.food_group not in _VEG_SAFE_GROUPS:
                    for kw in _NONVEG_KEYWORDS:
                        if kw in name_lower:
                            return False
            elif restriction == "vegan":
                if fi.food_group not in _VEGAN_SAFE_GROUPS:
                    for kw in _NONVEGAN_KEYWORDS:
                        if kw in name_lower:
                            return False
                # Also check for dairy in snack/prepared items.
                for kw in _VEGAN_DAIRY_KEYWORDS:
                    if kw in name_lower:
                        if not any(s in name_lower for s in _DAIRY_SAFE_NAMES):
                            return False

            # Gluten-free: keyword check.
            if restriction == "gluten_free":
                for kw in _GLUTEN_KEYWORDS:
                    if kw in name_lower:
                        return False

            # Halal.
            if restriction == "halal":
                for kw in _HALAL_EXCLUDED:
                    if kw in name_lower:
                        return False

            # Kosher.
            if restriction == "kosher":
                for kw in _KOSHER_EXCLUDED:
                    if kw in name_lower:
                        return False

            # Dairy-free / lactose-free: keyword check for prepared items.
            if restriction in ("dairy_free", "lactose_free"):
                for kw in _DAIRY_KEYWORDS:
                    if kw in name_lower:
                        if not any(s in name_lower for s in _DAIRY_SAFE_NAMES):
                            return False

            # Nut-free: keyword check for prepared items.
            if restriction == "nut_free":
                for kw in _NUT_KEYWORDS:
                    if kw in name_lower:
                        if not any(s in name_lower for s in _NUT_SAFE_NAMES):
                            return False

            # Pescatarian: keyword check for meat in prepared items.
            if restriction == "pescatarian":
                if fi.food_group not in _PESCATARIAN_SAFE_GROUPS:
                    for kw in _PESCATARIAN_MEAT_KEYWORDS:
                        if kw in name_lower:
                            return False

        return True

    def _find_substitute(
        self,
        original: FoodItem,
        restrictions: list[str],
    ) -> FoodItem | None:
        """Find a nutritionally similar substitute that passes restrictions."""
        candidates = self._group_index.get(original.food_group, [])
        # Also check adjacent groups.
        fallback_groups = {
            "meat": ["legume", "prepared"],
            "poultry": ["legume", "prepared"],
            "fish": ["legume"],
            "seafood": ["legume"],
            "dairy": ["legume", "nut"],
            "prepared": ["legume", "grain", "vegetable"],
        }
        all_candidates = list(candidates)
        for fg in fallback_groups.get(original.food_group, []):
            all_candidates.extend(self._group_index.get(fg, []))

        best: FoodItem | None = None
        best_diff = float("inf")

        for c in all_candidates:
            if c.name == original.name:
                continue
            if not self._food_passes_restrictions(c, restrictions):
                continue
            # Score similarity by calorie and macro profile.
            diff = (
                abs(c.calories_per_100g - original.calories_per_100g)
                + abs(c.protein - original.protein) * 5
                + abs(c.fat - original.fat) * 3
            )
            if diff < best_diff:
                best_diff = diff
                best = c

        return best

    @staticmethod
    def _sum_calories(plan: MealPlan) -> float:
        """Sum all calories in a meal plan."""
        total = 0.0
        for items in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
            for fi, grams in items:
                total += fi.calories_per_100g * (grams / 100.0)
        return round(total, 1)

    @staticmethod
    def _sum_macros(plan: MealPlan) -> dict[str, float]:
        """Sum macronutrients across all meals."""
        macros = {"protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
        for items in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
            for fi, grams in items:
                scale = grams / 100.0
                macros["protein"] += fi.protein * scale
                macros["carbs"] += fi.carbs * scale
                macros["fat"] += fi.fat * scale
                macros["fiber"] += fi.fiber * scale
        return {k: round(v, 1) for k, v in macros.items()}

    @staticmethod
    def _telomere_damaging_foods() -> list[str]:
        """Foods associated with accelerated telomere shortening."""
        return [
            "processed meat",
            "sugar-sweetened beverages",
            "trans fats",
            "excessive alcohol",
            "ultra-processed snacks",
            "refined carbohydrates",
            "deep-fried foods",
        ]

    @staticmethod
    def _telomere_daily_target(nutrient: str, urgency_mult: float) -> str:
        """Format daily target for telomere-protective nutrients."""
        base_targets: dict[str, str] = {
            "omega_3": "1000–2000 mg EPA+DHA/day",
            "folate": "400–800 µg DFE/day",
            "vitamin_d": "1000–2000 IU/day",
            "vitamin_c": "200–500 mg/day",
            "vitamin_e": "15–30 mg/day",
            "polyphenols": "500–1000 mg/day",
            "zinc": "8–15 mg/day",
            "magnesium": "320–420 mg/day",
            "anti_inflammatory": "Anti-inflammatory diet pattern",
        }
        target = base_targets.get(nutrient, "See specific guidance")
        if urgency_mult > 1.0:
            target += f" (elevated need ×{urgency_mult:.1f})"
        return target


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PRIORITY_RANK: dict[str, int] = _load_json("priority_rank.json")
