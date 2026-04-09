"""Focused variety tests for the DietAdvisor meal-plan generator.

Validates that multi-day meal plans exhibit sufficient food diversity,
avoid excessive repetition, and include regionally appropriate items.
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


@pytest.fixture(scope="module")
def recs_north_india(advisor):
    return advisor.generate_recommendations(
        genetic_risks=["Type 2 diabetes"],
        variants={"rs1801133": "CT"},
        region="south_asia_north",
        age=30,
        sex="male",
    )


@pytest.fixture(scope="module")
def thirty_day_north_india(advisor, recs_north_india):
    return advisor.create_meal_plan(
        recs_north_india, region="south_asia_north", calories=2000, days=30
    )


@pytest.fixture(scope="module")
def recs_south_india(advisor):
    return advisor.generate_recommendations(
        genetic_risks=["Type 2 diabetes"],
        variants={"rs1801133": "CT"},
        region="south_asia_south",
        age=30,
        sex="male",
    )


@pytest.fixture(scope="module")
def thirty_day_south_india(advisor, recs_south_india):
    return advisor.create_meal_plan(
        recs_south_india, region="south_asia_south", calories=2000, days=30
    )


# ---------------------------------------------------------------------------
# Unique food count tests
# ---------------------------------------------------------------------------


class TestUniqueFoodCounts:
    """A 30-day plan should draw from a wide variety of foods."""

    def test_unique_breakfast_count(self, thirty_day_north_india):
        names = set()
        for plan in thirty_day_north_india:
            for food, _ in plan.breakfast:
                names.add(food.name)
        assert len(names) > 15, f"Only {len(names)} unique breakfast items"

    def test_unique_lunch_count(self, thirty_day_north_india):
        names = set()
        for plan in thirty_day_north_india:
            for food, _ in plan.lunch:
                names.add(food.name)
        assert len(names) > 10, f"Only {len(names)} unique lunch items"

    def test_unique_dinner_count(self, thirty_day_north_india):
        names = set()
        for plan in thirty_day_north_india:
            for food, _ in plan.dinner:
                names.add(food.name)
        assert len(names) > 15, f"Only {len(names)} unique dinner items"

    def test_unique_snack_count(self, thirty_day_north_india):
        names = set()
        for plan in thirty_day_north_india:
            for food, _ in plan.snacks:
                names.add(food.name)
        assert len(names) > 5, f"Only {len(names)} unique snack items"

    def test_total_unique_foods(self, thirty_day_north_india):
        """Across all meals, there should be broad variety."""
        names = set()
        for plan in thirty_day_north_india:
            for meal in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
                for food, _ in meal:
                    names.add(food.name)
        assert len(names) > 30, f"Only {len(names)} unique foods across 30-day plan"


# ---------------------------------------------------------------------------
# Consecutive day repetition tests
# ---------------------------------------------------------------------------


class TestConsecutiveDayDifferences:
    """Adjacent days should not repeat the same main items too often."""

    def _get_main_item_per_day(self, plans, meal_attr):
        """Extract the first food name from a meal for each day."""
        items = []
        for plan in plans:
            meal = getattr(plan, meal_attr)
            if meal:
                items.append(meal[0][0].name)
            else:
                items.append(None)
        return items

    def _max_consecutive_same(self, items):
        """Return the maximum streak of identical consecutive items."""
        if not items:
            return 0
        max_streak = 1
        streak = 1
        for i in range(1, len(items)):
            if items[i] == items[i - 1] and items[i] is not None:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        return max_streak

    def test_dinner_no_long_repeats(self, thirty_day_north_india):
        mains = self._get_main_item_per_day(thirty_day_north_india, "dinner")
        assert self._max_consecutive_same(mains) <= 2, (
            "Dinner main dish repeated >2 consecutive days"
        )

    def test_breakfast_no_long_repeats(self, thirty_day_north_india):
        mains = self._get_main_item_per_day(thirty_day_north_india, "breakfast")
        assert self._max_consecutive_same(mains) <= 2, (
            "Breakfast main item repeated >2 consecutive days"
        )

    def test_lunch_no_long_repeats(self, thirty_day_north_india):
        mains = self._get_main_item_per_day(thirty_day_north_india, "lunch")
        assert self._max_consecutive_same(mains) <= 2, (
            "Lunch main item repeated >2 consecutive days"
        )

    def test_adjacent_days_differ_in_at_least_one_meal(self, thirty_day_north_india):
        """Each pair of adjacent days should differ in at least one meal."""
        for i in range(len(thirty_day_north_india) - 1):
            day_a = thirty_day_north_india[i]
            day_b = thirty_day_north_india[i + 1]
            a_foods = {
                f.name for meal in (day_a.breakfast, day_a.lunch, day_a.dinner) for f, _ in meal
            }
            b_foods = {
                f.name for meal in (day_b.breakfast, day_b.lunch, day_b.dinner) for f, _ in meal
            }
            # They should not be identical
            assert a_foods != b_foods, f"Days {day_a.day} and {day_b.day} are identical"


# ---------------------------------------------------------------------------
# Regional food appropriateness
# ---------------------------------------------------------------------------


class TestRegionalFoodAppropriateness:
    """Foods in region-specific plans should reflect that cuisine."""

    def _all_food_names(self, plans):
        names = set()
        for plan in plans:
            for meal in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
                for food, _ in meal:
                    names.add(food.name.lower())
        return names

    def test_south_india_has_indian_foods(self, thirty_day_south_india):
        """South Indian plan should contain recognisable South Indian foods."""
        names = self._all_food_names(thirty_day_south_india)
        # At least some of these should appear
        south_indian_markers = {
            "idli",
            "dosa",
            "sambar",
            "coconut chutney",
            "rasam",
            "rice",
            "ragi",
            "brown rice",
            "white rice",
            "tamarind",
            "coconut",
            "banana",
            "drumstick",
            "curry leaves",
            "urad dal",
        }
        found = names & south_indian_markers
        assert len(found) >= 1, (
            f"South India plan lacks regional foods. Found: {names & south_indian_markers}"
        )

    def test_north_india_has_indian_foods(self, thirty_day_north_india):
        names = self._all_food_names(thirty_day_north_india)
        north_indian_markers = {
            "roti",
            "whole wheat roti",
            "dal",
            "paneer",
            "ghee",
            "basmati rice",
            "brown rice",
            "white rice",
            "chickpeas",
            "spinach",
            "potato",
            "lentils",
            "moong dal",
            "rajma",
            "turmeric",
        }
        found = names & north_indian_markers
        assert len(found) >= 1, (
            f"North India plan lacks regional foods. Names: {sorted(names)[:20]}"
        )

    def test_mediterranean_has_relevant_foods(self, advisor):
        recs = advisor.generate_recommendations(
            genetic_risks=["Coronary artery disease"],
            variants={"rs1801133": "CT"},
            region="mediterranean",
            age=40,
            sex="male",
        )
        plans = advisor.create_meal_plan(recs, region="mediterranean", calories=2000, days=30)
        names = self._all_food_names(plans)
        med_markers = {
            "olive oil",
            "feta",
            "pasta",
            "tomato",
            "lentils",
            "chickpeas",
            "sardine",
            "salmon",
            "brown rice",
            "greek yoghurt",
            "hummus",
            "tabbouleh",
            "grilled fish",
        }
        found = names & med_markers
        assert len(found) >= 1, (
            f"Mediterranean plan lacks regional foods. Names: {sorted(names)[:20]}"
        )

    def test_east_asian_has_relevant_foods(self, advisor):
        recs = advisor.generate_recommendations(
            genetic_risks=[],
            variants={"rs1801133": "CT"},
            region="east_asia",
            age=35,
            sex="female",
        )
        plans = advisor.create_meal_plan(recs, region="east_asia", calories=2000, days=30)
        names = self._all_food_names(plans)
        ea_markers = {
            "tofu",
            "rice",
            "brown rice",
            "white rice",
            "soy sauce",
            "miso",
            "edamame",
            "bok choy",
            "noodles",
            "tempeh",
            "seaweed",
            "shiitake",
            "green tea",
        }
        found = names & ea_markers
        assert len(found) >= 1, f"East Asian plan lacks regional foods. Names: {sorted(names)[:20]}"

    def test_different_regions_produce_different_plans(self, advisor):
        """Plans for two very different regions should not be identical."""
        recs_a = advisor.generate_recommendations(
            genetic_risks=["Type 2 diabetes"],
            variants={"rs1801133": "CT"},
            region="south_asia_south",
            age=30,
            sex="male",
        )
        recs_b = advisor.generate_recommendations(
            genetic_risks=["Type 2 diabetes"],
            variants={"rs1801133": "CT"},
            region="northern_europe",
            age=30,
            sex="male",
        )
        plans_a = advisor.create_meal_plan(recs_a, region="south_asia_south", calories=2000, days=7)
        plans_b = advisor.create_meal_plan(recs_b, region="northern_europe", calories=2000, days=7)

        names_a = set()
        names_b = set()
        for plan in plans_a:
            for meal in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
                for food, _ in meal:
                    names_a.add(food.name)
        for plan in plans_b:
            for meal in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
                for food, _ in meal:
                    names_b.add(food.name)

        # They should differ meaningfully (at least some items unique to each)
        only_a = names_a - names_b
        only_b = names_b - names_a
        assert len(only_a) > 0 or len(only_b) > 0, (
            "South India and Northern Europe plans contain identical food sets"
        )


# ---------------------------------------------------------------------------
# Dietary restriction compliance tests
# ---------------------------------------------------------------------------

# Keywords that indicate non-vegetarian food items.
# NOTE: The authoritative check is DietAdvisor._food_passes_restrictions().
# These keywords are a secondary sanity check only.
_MEAT_FISH_KEYWORDS = {
    "chicken",
    "beef",
    "pork",
    "lamb",
    "mutton",
    "goat",
    "duck",
    "turkey",
    "bacon",
    "ham",
    "sausage",
    "fish",
    "tuna",
    "salmon",
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "anchov",
    "sardine",
    "mackerel",
    "herring",
    "squid",
    "octopus",
    "mussel",
    "oyster",
    "clam",
    "scallop",
    "ceviche",
    "rendang",
    "shawarma",
    "kebab",
    "gyoza",
    "larb",
}

_NONVEG_FOOD_GROUPS = {"meat", "poultry", "fish", "seafood"}


def _collect_all_foods(plans):
    """Extract all (name, food_group) pairs from a list of MealPlans."""
    items = []
    for plan in plans:
        for meal in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks):
            for food, _ in meal:
                items.append((food.name, food.food_group))
    return items


class TestVegetarianCompliance:
    """Vegetarian meal plans must contain zero animal-derived foods."""

    @pytest.fixture(scope="class")
    def veg_plans_north_india(self, advisor):
        recs = advisor.generate_recommendations(
            genetic_risks=["Type 2 diabetes"],
            variants={"rs1801133": "CT"},
            region="south_asia_north",
            age=30,
            sex="male",
            dietary_restrictions=["vegetarian"],
        )
        return advisor.create_meal_plan(
            recs,
            region="south_asia_north",
            calories=2000,
            days=30,
            dietary_restrictions=["vegetarian"],
        )

    @pytest.fixture(scope="class")
    def veg_plans_south_india(self, advisor):
        recs = advisor.generate_recommendations(
            genetic_risks=["Type 2 diabetes"],
            variants={},
            region="south_asia_south",
            age=25,
            sex="female",
            dietary_restrictions=["vegetarian"],
        )
        return advisor.create_meal_plan(
            recs,
            region="south_asia_south",
            calories=1800,
            days=30,
            dietary_restrictions=["vegetarian"],
        )

    def test_no_nonveg_food_groups_north(self, veg_plans_north_india):
        """No meat/poultry/fish/seafood food groups in vegetarian plan."""
        items = _collect_all_foods(veg_plans_north_india)
        violations = [(n, g) for n, g in items if g in _NONVEG_FOOD_GROUPS]
        assert violations == [], f"Non-veg food groups found: {violations}"

    def test_no_nonveg_food_groups_south(self, veg_plans_south_india):
        items = _collect_all_foods(veg_plans_south_india)
        violations = [(n, g) for n, g in items if g in _NONVEG_FOOD_GROUPS]
        assert violations == [], f"Non-veg food groups found: {violations}"

    def test_no_nonveg_keywords_in_names_north(self, veg_plans_north_india):
        """No prepared dishes with meat/fish keywords (e.g. 'butter chicken')."""
        items = _collect_all_foods(veg_plans_north_india)
        violations = []
        for name, group in items:
            name_lower = name.lower()
            for kw in _MEAT_FISH_KEYWORDS:
                if kw in name_lower:
                    violations.append(name)
                    break
        assert violations == [], f"Non-veg keywords in food names: {violations}"

    def test_no_nonveg_keywords_in_names_south(self, veg_plans_south_india):
        items = _collect_all_foods(veg_plans_south_india)
        violations = []
        for name, group in items:
            name_lower = name.lower()
            for kw in _MEAT_FISH_KEYWORDS:
                if kw in name_lower:
                    violations.append(name)
                    break
        assert violations == [], f"Non-veg keywords in food names: {violations}"

    def test_butter_chicken_not_in_veg_plan(self, veg_plans_north_india):
        """Specific regression test: butter chicken must never appear."""
        items = _collect_all_foods(veg_plans_north_india)
        names = {n.lower() for n, _ in items}
        assert "butter chicken" not in names, "Butter chicken appeared in vegetarian plan!"

    def test_veg_plan_still_has_variety(self, veg_plans_north_india):
        """Vegetarian plan should still have reasonable food variety."""
        items = _collect_all_foods(veg_plans_north_india)
        unique_names = {n for n, _ in items}
        assert len(unique_names) > 20, (
            f"Vegetarian plan lacks variety: only {len(unique_names)} unique foods"
        )

    def test_veg_plan_reasonable_calories(self, veg_plans_north_india):
        """Each day should still hit ~1600-2400 calorie range."""
        for plan in veg_plans_north_india:
            total = sum(
                f.calories_per_100g * (g / 100.0)
                for meal in (plan.breakfast, plan.lunch, plan.dinner, plan.snacks)
                for f, g in meal
            )
            assert 1200 <= total <= 2800, (
                f"Day {plan.day}: {total:.0f} cal outside acceptable range"
            )


class TestVeganCompliance:
    """Vegan plans must also exclude dairy and eggs."""

    @pytest.fixture(scope="class")
    def vegan_plans(self, advisor):
        recs = advisor.generate_recommendations(
            genetic_risks=["Coronary heart disease"],
            variants={},
            region="northern_europe",
            age=35,
            sex="female",
            dietary_restrictions=["vegan"],
        )
        return advisor.create_meal_plan(
            recs,
            region="northern_europe",
            calories=2000,
            days=14,
            dietary_restrictions=["vegan"],
        )

    def test_no_animal_food_groups(self, vegan_plans):
        items = _collect_all_foods(vegan_plans)
        banned = _NONVEG_FOOD_GROUPS | {"dairy", "eggs"}
        violations = [(n, g) for n, g in items if g in banned]
        assert violations == [], f"Animal food groups in vegan plan: {violations}"

    def test_no_animal_keywords(self, vegan_plans):
        _vegan_kw = _MEAT_FISH_KEYWORDS | {
            "cheese",
            "cream",
            "butter",
            "yogurt",
            "paneer",
            "egg",
            "ghee",
            "whey",
        }
        items = _collect_all_foods(vegan_plans)
        violations = []
        for name, _ in items:
            name_lower = name.lower()
            for kw in _vegan_kw:
                if kw in name_lower:
                    violations.append(name)
                    break
        assert violations == [], f"Animal keywords in vegan plan: {violations}"


class TestAdaptToRestrictions:
    """Test the post-hoc adapt_to_restrictions safety net."""

    def test_adapt_removes_chicken_from_existing_plan(self, advisor):
        """If a plan was generated without restrictions, adapt should fix it."""
        recs = advisor.generate_recommendations(
            genetic_risks=["Type 2 diabetes"],
            variants={},
            region="south_asia_north",
            age=30,
            sex="male",
        )
        # Generate without restrictions.
        plans = advisor.create_meal_plan(
            recs,
            region="south_asia_north",
            calories=2000,
            days=7,
        )
        # Now adapt to vegetarian.
        adapted = advisor.adapt_to_restrictions(plans, ["vegetarian"])
        items = _collect_all_foods(adapted)
        violations = [(n, g) for n, g in items if g in _NONVEG_FOOD_GROUPS]
        kw_violations = []
        for name, _ in items:
            name_lower = name.lower()
            for kw in _MEAT_FISH_KEYWORDS:
                if kw in name_lower:
                    kw_violations.append(name)
                    break
        assert violations == [], f"Non-veg groups after adapt: {violations}"
        assert kw_violations == [], f"Non-veg keywords after adapt: {kw_violations}"


class TestMultipleRegionsVegetarian:
    """Vegetarian compliance should hold across ALL 30 regions."""

    REGIONS = [
        "south_asia_north",
        "south_asia_south",
        "south_asia_east",
        "south_asia_west",
        "east_asia",
        "southeast_asia",
        "middle_east",
        "mediterranean",
        "northern_europe",
        "sub_saharan_africa",
        "latin_america",
        "north_america",
        "central_asia",
        "west_africa",
        "east_africa",
        "southern_africa",
        "north_africa",
        "caribbean",
        "south_america",
        "pacific_islands",
        "eastern_europe",
        "western_europe",
        "central_america",
        "australian",
        "nordic",
        "balkans",
        "caucasus",
        "central_europe",
        "andean",
        "iberian",
    ]

    @pytest.fixture(scope="class", params=REGIONS)
    def veg_plans_by_region(self, request, advisor):
        region = request.param
        recs = advisor.generate_recommendations(
            genetic_risks=["Type 2 diabetes"],
            variants={},
            region=region,
            age=30,
            sex="male",
            dietary_restrictions=["vegetarian"],
        )
        plans = advisor.create_meal_plan(
            recs,
            region=region,
            calories=2000,
            days=7,
            dietary_restrictions=["vegetarian"],
        )
        return region, plans

    def test_no_nonveg_items_any_region(self, veg_plans_by_region):
        """Every food item must pass the actual vegetarian filter."""
        region, plans = veg_plans_by_region
        _advisor = DietAdvisor()
        violations = []
        for plan in plans:
            for meal_name, meal in [
                ("breakfast", plan.breakfast),
                ("lunch", plan.lunch),
                ("dinner", plan.dinner),
                ("snacks", plan.snacks),
            ]:
                for food, _ in meal:
                    if food.food_group in _NONVEG_FOOD_GROUPS:
                        violations.append(
                            f"Day {plan.day} {meal_name}: {food.name} (group={food.food_group})"
                        )
                    if not _advisor._food_passes_restrictions(food, ["vegetarian"]):
                        violations.append(
                            f"Day {plan.day} {meal_name}: {food.name} (filter_rejected)"
                        )
        assert violations == [], f"Region {region}: non-veg items found: {violations}"
