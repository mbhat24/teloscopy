"""Tests for dietary restriction filtering across all restriction types.

Validates that DietAdvisor.create_meal_plan correctly filters foods for:
vegetarian, vegan, gluten_free, halal, kosher, dairy_free, lactose_free,
nut_free, and pescatarian restrictions.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from teloscopy.nutrition.diet_advisor import DietAdvisor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def advisor():
    return DietAdvisor()


def _gen_plans(advisor, region, restriction, days=7):
    """Helper: generate meal plans with a single dietary restriction."""
    recs = advisor.generate_recommendations(
        genetic_risks=["Type 2 diabetes"],
        variants={"rs1801133": "CT"},
        region=region,
        age=35,
        sex="male",
        dietary_restrictions=[restriction],
    )
    return advisor.create_meal_plan(
        recs,
        region=region,
        calories=2000,
        days=days,
        dietary_restrictions=[restriction],
    )


def _collect_all_foods(plans):
    """Extract all (FoodItem, grams) from a list of MealPlans."""
    items = []
    for plan in plans:
        for meal in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
            for food, grams in meal:
                items.append(food)
    return items


# ---------------------------------------------------------------------------
# Keyword / group sets for auditing
# ---------------------------------------------------------------------------

_NONVEG_GROUPS = {"meat", "poultry", "fish", "seafood", "eggs"}
_VEG_SAFE_GROUPS = {
    "grain", "fruit", "vegetable", "legume", "nut", "seed",
    "oil", "spice", "beverage", "dairy",
}
_MEAT_KEYWORDS = {
    "chicken", "beef", "pork", "lamb", "mutton", "goat", "duck", "turkey",
    "bacon", "ham", "sausage", "steak", "salami", "pepperoni", "prosciutto",
    "kebab", "shawarma", "bolognese", "rendang",
}
_VEGAN_BANNED_GROUPS = {"meat", "poultry", "fish", "seafood", "dairy", "eggs"}
_VEGAN_SAFE_GROUPS = {
    "grain", "fruit", "vegetable", "legume", "nut", "seed",
    "oil", "spice", "beverage",
}
_GLUTEN_KEYWORDS = {
    "wheat", "roti", "bread", "pasta", "rye", "bulgur", "couscous",
    "naan", "pita", "tortilla", "barley", "seitan",
    "muffin", "pancake", "waffle", "croissant", "bagel", "pizza", "udon",
}
_HALAL_KEYWORDS = {"pork", "ham", "bacon", "wine", "beer", "lard"}
_KOSHER_KEYWORDS = {"pork", "ham", "bacon", "shellfish", "shrimp", "crab", "lobster", "lard"}
_DAIRY_KEYWORDS = {
    "cheese", "yogurt", "yoghurt", "cream", "butter", "ghee", "paneer",
    "ricotta", "mozzarella", "cheddar", "parmesan", "feta",
    "custard", "ice cream", "lassi", "kheer", "raita", "buttermilk",
    "milk chocolate", "condensed milk", "cottage cheese",
    "cream cheese", "sour cream", "milk",
}
_DAIRY_SAFE_NAMES = {
    "coconut milk", "almond milk", "soy milk", "oat milk", "rice milk",
    "cashew milk", "coconut cream", "coconut yogurt", "soy yogurt",
    "coconut butter", "peanut butter", "almond butter", "cocoa butter",
    "butternut", "buttercup", "butterscotch", "butterbean",
    "custard apple", "cream of tartar",
}
_NUT_KEYWORDS = {
    "almond", "walnut", "cashew", "pistachio", "pecan", "hazelnut",
    "macadamia", "brazil nut", "pine nut", "chestnut", "peanut", "groundnut",
}
_NUT_SAFE_NAMES = {"nutmeg", "coconut", "butternut", "water chestnut", "doughnut", "donut"}
_NONVEG_SAFE_NAMES = {"eggplant", "garden egg"}
_PESCATARIAN_SAFE_GROUPS = {
    "grain", "fruit", "vegetable", "legume", "nut", "seed",
    "oil", "spice", "beverage", "dairy", "eggs", "fish", "seafood",
}

# Representative regions to test (covering major food traditions).
_TEST_REGIONS = [
    "south_asia_north",
    "south_asia_south",
    "east_asia",
    "southeast_asia",
    "mediterranean",
    "northern_europe",
    "latin_america",
    "middle_east",
    "sub_saharan_africa",
    "north_america",
]


def _kw_in_name(name_lower, keywords, safe_names=None):
    """Return True if name contains any keyword and is not in safe set."""
    for kw in keywords:
        if kw in name_lower:
            if safe_names and any(s in name_lower for s in safe_names):
                continue
            return True
    return False


# ---------------------------------------------------------------------------
# Vegetarian
# ---------------------------------------------------------------------------

class TestVegetarianRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "vegetarian")

    def test_no_nonveg_food_groups(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [f.name for f in foods if f.food_group in _NONVEG_GROUPS]
        assert violations == [], f"[{region}] Non-veg groups: {violations}"

    def test_no_meat_keywords_outside_safe_groups(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if f.food_group not in _VEG_SAFE_GROUPS
            and _kw_in_name(f.name.lower(), _MEAT_KEYWORDS)
        ]
        assert violations == [], f"[{region}] Meat keywords: {violations}"

    def test_no_egg_items(self, plans):
        """Vegetarian plans must exclude egg-based foods."""
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if f.food_group == "eggs"
            or (
                "egg" in f.name.lower()
                and not any(s in f.name.lower() for s in _NONVEG_SAFE_NAMES)
            )
        ]
        assert violations == [], f"[{region}] Egg items in vegetarian plan: {violations}"


# ---------------------------------------------------------------------------
# Vegan
# ---------------------------------------------------------------------------

class TestVeganRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "vegan")

    def test_no_animal_food_groups(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [f.name for f in foods if f.food_group in _VEGAN_BANNED_GROUPS]
        assert violations == [], f"[{region}] Animal groups: {violations}"

    def test_no_animal_keywords_outside_safe_groups(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        animal_kw = _MEAT_KEYWORDS | {"egg", "cheese", "yogurt", "cream", "butter",
                                        "ghee", "paneer", "honey", "milk", "lassi"}
        violations = [
            f.name for f in foods
            if f.food_group not in _VEGAN_SAFE_GROUPS
            and _kw_in_name(f.name.lower(), animal_kw, _DAIRY_SAFE_NAMES)
        ]
        assert violations == [], f"[{region}] Animal keywords: {violations}"


# ---------------------------------------------------------------------------
# Gluten-free
# ---------------------------------------------------------------------------

class TestGlutenFreeRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "gluten_free")

    def test_no_gluten_keywords(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if _kw_in_name(f.name.lower(), _GLUTEN_KEYWORDS)
        ]
        assert violations == [], f"[{region}] Gluten items: {violations}"


# ---------------------------------------------------------------------------
# Halal
# ---------------------------------------------------------------------------

class TestHalalRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "halal")

    def test_no_halal_violations(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if _kw_in_name(f.name.lower(), _HALAL_KEYWORDS)
        ]
        assert violations == [], f"[{region}] Halal violations: {violations}"


# ---------------------------------------------------------------------------
# Kosher
# ---------------------------------------------------------------------------

class TestKosherRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "kosher")

    def test_no_kosher_violations(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if _kw_in_name(f.name.lower(), _KOSHER_KEYWORDS)
        ]
        assert violations == [], f"[{region}] Kosher violations: {violations}"


# ---------------------------------------------------------------------------
# Dairy-free
# ---------------------------------------------------------------------------

class TestDairyFreeRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "dairy_free")

    def test_no_dairy_food_group(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [f.name for f in foods if f.food_group == "dairy"]
        assert violations == [], f"[{region}] Dairy group items: {violations}"

    def test_no_dairy_keywords(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if _kw_in_name(f.name.lower(), _DAIRY_KEYWORDS, _DAIRY_SAFE_NAMES)
        ]
        assert violations == [], f"[{region}] Dairy keywords: {violations}"


# ---------------------------------------------------------------------------
# Lactose-free
# ---------------------------------------------------------------------------

class TestLactoseFreeRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "lactose_free")

    def test_no_dairy_food_group(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [f.name for f in foods if f.food_group == "dairy"]
        assert violations == [], f"[{region}] Dairy group items: {violations}"

    def test_no_dairy_keywords(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if _kw_in_name(f.name.lower(), _DAIRY_KEYWORDS, _DAIRY_SAFE_NAMES)
        ]
        assert violations == [], f"[{region}] Dairy keywords: {violations}"


# ---------------------------------------------------------------------------
# Nut-free
# ---------------------------------------------------------------------------

class TestNutFreeRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "nut_free")

    def test_no_nut_food_group(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [f.name for f in foods if f.food_group == "nut"]
        assert violations == [], f"[{region}] Nut group items: {violations}"

    def test_no_nut_keywords(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if _kw_in_name(f.name.lower(), _NUT_KEYWORDS, _NUT_SAFE_NAMES)
        ]
        assert violations == [], f"[{region}] Nut keywords: {violations}"


# ---------------------------------------------------------------------------
# Pescatarian
# ---------------------------------------------------------------------------

class TestPescatarianRestriction:

    @pytest.fixture(scope="class", params=_TEST_REGIONS)
    def plans(self, request, advisor):
        return request.param, _gen_plans(advisor, request.param, "pescatarian")

    def test_no_meat_food_groups(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods if f.food_group in ("meat", "poultry")
        ]
        assert violations == [], f"[{region}] Meat groups: {violations}"

    def test_no_meat_keywords_outside_safe_groups(self, plans):
        region, meal_plans = plans
        foods = _collect_all_foods(meal_plans)
        violations = [
            f.name for f in foods
            if f.food_group not in _PESCATARIAN_SAFE_GROUPS
            and _kw_in_name(f.name.lower(), _MEAT_KEYWORDS)
        ]
        assert violations == [], f"[{region}] Meat keywords: {violations}"
