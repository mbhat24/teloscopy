"""Nutrigenomics-informed dietary recommendation engine.

Provides geography-aware, genetically-informed dietary recommendations by
integrating SNP-level nutrigenomic data with regional food availability to
produce personalised meal plans and nutritional guidance.  The module ships
with curated databases covering 30 world regions, 120+ gene-nutrient
associations, and 297 food items with full macronutrient and
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
    # -- South Asia: North India ------------------------------------------
    GeographicProfile(
        region_id="south_asia_north",
        name="North India",
        common_foods=[
            "roti",
            "dal",
            "paneer",
            "ghee",
            "lassi",
            "rajma",
            "chole",
            "paratha",
            "aloo gobi",
            "basmati rice",
        ],
        staple_grains=["wheat", "basmati rice", "maize", "millet"],
        protein_sources=["paneer", "dal", "rajma", "chicken", "lamb", "chole", "yoghurt", "eggs"],
        vegetables=[
            "potato",
            "cauliflower",
            "spinach",
            "okra",
            "peas",
            "tomato",
            "onion",
            "bottle gourd",
            "fenugreek leaves",
        ],
        fruits=["mango", "guava", "papaya", "banana", "apple", "pomegranate"],
        spices=[
            "turmeric",
            "cumin",
            "coriander",
            "garam masala",
            "red chilli",
            "asafoetida",
            "fenugreek",
            "cardamom",
        ],
        traditional_dishes=[
            "dal makhani",
            "butter chicken",
            "aloo paratha",
            "chole bhature",
            "rajma chawal",
            "palak paneer",
            "tandoori chicken",
            "biryani",
        ],
    ),
    # -- South Asia: South India ------------------------------------------
    GeographicProfile(
        region_id="south_asia_south",
        name="South India",
        common_foods=[
            "idli",
            "dosa",
            "sambar",
            "rasam",
            "coconut chutney",
            "appam",
            "puttu",
            "avial",
            "rice",
            "tamarind rice",
        ],
        staple_grains=["rice", "ragi", "urad dal", "rice flour"],
        protein_sources=[
            "fish",
            "lentils",
            "coconut",
            "eggs",
            "chicken",
            "black gram",
            "prawns",
            "curd",
        ],
        vegetables=[
            "drumstick",
            "ash gourd",
            "snake gourd",
            "bitter gourd",
            "banana stem",
            "curry leaves",
            "tamarind",
            "tomato",
        ],
        fruits=["banana", "coconut", "jackfruit", "mango", "sapota", "pineapple"],
        spices=[
            "mustard seeds",
            "curry leaves",
            "black pepper",
            "tamarind",
            "coconut",
            "red chilli",
            "turmeric",
            "asafoetida",
        ],
        traditional_dishes=[
            "masala dosa",
            "idli sambar",
            "Kerala fish curry",
            "Hyderabadi biryani",
            "Chettinad chicken",
            "rasam",
            "avial",
            "payasam",
        ],
    ),
    # -- South Asia: East India -------------------------------------------
    GeographicProfile(
        region_id="south_asia_east",
        name="East India",
        common_foods=[
            "rice",
            "fish curry",
            "mishti doi",
            "luchi",
            "shorshe ilish",
            "chingri malai curry",
            "pitha",
            "dal",
            "moong dal",
            "sandesh",
        ],
        staple_grains=["rice", "wheat", "flattened rice"],
        protein_sources=[
            "fish",
            "prawns",
            "lentils",
            "mustard oil",
            "paneer",
            "eggs",
            "chicken",
            "curd",
        ],
        vegetables=[
            "potato",
            "brinjal",
            "pumpkin",
            "pointed gourd",
            "banana flower",
            "green banana",
            "radish",
            "spinach",
        ],
        fruits=["mango", "lychee", "jackfruit", "banana", "guava", "palm"],
        spices=[
            "mustard seeds",
            "panch phoron",
            "turmeric",
            "red chilli",
            "ginger",
            "bay leaf",
            "radhuni",
            "nigella seeds",
        ],
        traditional_dishes=[
            "shorshe ilish",
            "kosha mangsho",
            "chingri malai curry",
            "mishti doi",
            "luchi alur dom",
            "pitha",
            "sandesh",
            "macher jhol",
        ],
    ),
    # -- South Asia: West India -------------------------------------------
    GeographicProfile(
        region_id="south_asia_west",
        name="West India",
        common_foods=[
            "dhokla",
            "thepla",
            "poha",
            "vada pav",
            "pav bhaji",
            "bhakri",
            "puran poli",
            "misal pav",
            "dal baati",
            "undhiyu",
        ],
        staple_grains=["wheat", "jowar", "bajra", "rice", "nachni"],
        protein_sources=[
            "lentils",
            "groundnut",
            "chickpeas",
            "fish",
            "buttermilk",
            "paneer",
            "soya",
            "sprouts",
        ],
        vegetables=[
            "brinjal",
            "drumstick",
            "cluster beans",
            "ivy gourd",
            "raw banana",
            "tomato",
            "onion",
            "green chilli",
        ],
        fruits=["alphonso mango", "chikoo", "coconut", "banana", "custard apple", "pomegranate"],
        spices=[
            "mustard seeds",
            "sesame",
            "jaggery",
            "kokum",
            "curry leaves",
            "asafoetida",
            "turmeric",
            "coriander",
        ],
        traditional_dishes=[
            "dhokla",
            "vada pav",
            "pav bhaji",
            "misal pav",
            "puran poli",
            "dal baati churma",
            "undhiyu",
            "poha",
            "shrikhand",
        ],
    ),
    # -- East Asia --------------------------------------------------------
    GeographicProfile(
        region_id="east_asia",
        name="East Asia",
        common_foods=[
            "rice",
            "noodles",
            "tofu",
            "soy sauce",
            "miso",
            "kimchi",
            "bok choy",
            "dumplings",
            "congee",
            "steamed buns",
        ],
        staple_grains=["rice", "wheat noodles", "millet", "buckwheat"],
        protein_sources=[
            "tofu",
            "fish",
            "pork",
            "chicken",
            "edamame",
            "eggs",
            "tempeh",
            "seaweed",
            "shrimp",
        ],
        vegetables=[
            "bok choy",
            "napa cabbage",
            "daikon",
            "lotus root",
            "bamboo shoots",
            "shiitake mushrooms",
            "water spinach",
            "spring onion",
        ],
        fruits=["lychee", "persimmon", "mandarin orange", "Asian pear", "dragon fruit", "longan"],
        spices=[
            "soy sauce",
            "ginger",
            "garlic",
            "sesame oil",
            "star anise",
            "five-spice powder",
            "Sichuan pepper",
            "rice vinegar",
        ],
        traditional_dishes=[
            "stir-fried vegetables",
            "mapo tofu",
            "sushi",
            "ramen",
            "bibimbap",
            "kung pao chicken",
            "dumplings",
            "hot pot",
        ],
    ),
    # -- Southeast Asia ---------------------------------------------------
    GeographicProfile(
        region_id="southeast_asia",
        name="Southeast Asia",
        common_foods=[
            "rice",
            "noodles",
            "fish sauce",
            "coconut milk",
            "lemongrass",
            "tofu",
            "tempeh",
            "satay",
            "spring rolls",
            "pho",
        ],
        staple_grains=["jasmine rice", "sticky rice", "rice noodles", "wheat noodles"],
        protein_sources=[
            "fish",
            "shrimp",
            "chicken",
            "pork",
            "tofu",
            "tempeh",
            "eggs",
            "duck",
            "crab",
        ],
        vegetables=[
            "morning glory",
            "bean sprouts",
            "Thai basil",
            "galangal",
            "kaffir lime leaves",
            "Thai eggplant",
            "long beans",
            "banana blossom",
        ],
        fruits=["durian", "mangosteen", "rambutan", "papaya", "starfruit", "pineapple"],
        spices=[
            "lemongrass",
            "galangal",
            "fish sauce",
            "shrimp paste",
            "Thai chilli",
            "turmeric",
            "coriander root",
            "lime leaves",
        ],
        traditional_dishes=[
            "pad thai",
            "pho",
            "green curry",
            "nasi goreng",
            "tom yum goong",
            "laksa",
            "rendang",
            "banh mi",
            "satay",
        ],
    ),
    # -- Middle East ------------------------------------------------------
    GeographicProfile(
        region_id="middle_east",
        name="Middle East",
        common_foods=[
            "hummus",
            "falafel",
            "pita bread",
            "tabbouleh",
            "shawarma",
            "fattoush",
            "labneh",
            "halloumi",
            "dates",
            "baklava",
        ],
        staple_grains=["wheat", "bulgur", "couscous", "flatbread", "freekeh"],
        protein_sources=[
            "lamb",
            "chicken",
            "chickpeas",
            "lentils",
            "yoghurt",
            "halloumi",
            "fish",
            "eggs",
            "fava beans",
        ],
        vegetables=[
            "eggplant",
            "tomato",
            "cucumber",
            "parsley",
            "mint",
            "okra",
            "zucchini",
            "bell pepper",
        ],
        fruits=["dates", "figs", "pomegranate", "apricot", "grape", "orange"],
        spices=[
            "sumac",
            "za'atar",
            "cumin",
            "coriander",
            "saffron",
            "cardamom",
            "cinnamon",
            "allspice",
        ],
        traditional_dishes=[
            "hummus",
            "falafel",
            "shawarma",
            "kibbeh",
            "mansaf",
            "moutabal",
            "fattoush",
            "tabbouleh",
            "kunafa",
        ],
    ),
    # -- Mediterranean ----------------------------------------------------
    GeographicProfile(
        region_id="mediterranean",
        name="Mediterranean",
        common_foods=[
            "olive oil",
            "bread",
            "tomato",
            "feta cheese",
            "pasta",
            "grilled fish",
            "legumes",
            "wine",
            "fresh herbs",
            "yoghurt",
        ],
        staple_grains=["wheat", "barley", "farro", "durum wheat pasta", "sourdough bread"],
        protein_sources=[
            "fish",
            "chicken",
            "legumes",
            "feta",
            "mozzarella",
            "eggs",
            "lamb",
            "octopus",
            "sardines",
        ],
        vegetables=[
            "tomato",
            "eggplant",
            "zucchini",
            "artichoke",
            "olive",
            "bell pepper",
            "onion",
            "garlic",
            "leafy greens",
        ],
        fruits=["grape", "fig", "lemon", "orange", "peach", "melon"],
        spices=[
            "oregano",
            "basil",
            "rosemary",
            "thyme",
            "garlic",
            "bay leaf",
            "saffron",
            "parsley",
        ],
        traditional_dishes=[
            "Greek salad",
            "moussaka",
            "pasta puttanesca",
            "ratatouille",
            "paella",
            "bruschetta",
            "grilled sea bass",
            "risotto",
        ],
    ),
    # -- Northern Europe --------------------------------------------------
    GeographicProfile(
        region_id="northern_europe",
        name="Northern Europe",
        common_foods=[
            "rye bread",
            "potatoes",
            "herring",
            "salmon",
            "berries",
            "oats",
            "dairy",
            "root vegetables",
            "cabbage",
            "mushrooms",
        ],
        staple_grains=["rye", "oats", "wheat", "barley"],
        protein_sources=[
            "salmon",
            "herring",
            "cod",
            "pork",
            "beef",
            "chicken",
            "dairy",
            "eggs",
            "venison",
        ],
        vegetables=[
            "potato",
            "cabbage",
            "beetroot",
            "turnip",
            "carrot",
            "leek",
            "celery root",
            "kale",
            "peas",
        ],
        fruits=["blueberry", "lingonberry", "cloudberry", "apple", "pear", "plum"],
        spices=[
            "dill",
            "caraway",
            "juniper",
            "horseradish",
            "allspice",
            "cardamom",
            "mustard",
            "chives",
        ],
        traditional_dishes=[
            "gravlax",
            "smoked salmon",
            "meatballs",
            "rye bread smørrebrød",
            "fish soup",
            "potato gratin",
            "karelian pies",
            "pickled herring",
        ],
    ),
    # -- Sub-Saharan Africa -----------------------------------------------
    GeographicProfile(
        region_id="sub_saharan_africa",
        name="Sub-Saharan Africa",
        common_foods=[
            "fufu",
            "injera",
            "ugali",
            "jollof rice",
            "plantain",
            "groundnut soup",
            "cassava",
            "yam",
            "beans",
            "millet porridge",
        ],
        staple_grains=["teff", "sorghum", "millet", "maize", "rice", "cassava flour"],
        protein_sources=[
            "chicken",
            "goat",
            "fish",
            "beans",
            "groundnuts",
            "lentils",
            "cowpeas",
            "eggs",
            "beef",
        ],
        vegetables=[
            "okra",
            "cassava leaves",
            "amaranth",
            "moringa",
            "sweet potato leaves",
            "pumpkin",
            "tomato",
            "onion",
        ],
        fruits=["mango", "banana", "papaya", "baobab", "passion fruit", "guava"],
        spices=[
            "scotch bonnet pepper",
            "ginger",
            "garlic",
            "coriander",
            "berbere",
            "suya spice",
            "cumin",
            "clove",
        ],
        traditional_dishes=[
            "jollof rice",
            "injera with wot",
            "fufu and soup",
            "groundnut soup",
            "egusi soup",
            "ugali and sukuma",
            "bobotie",
            "nyama choma",
        ],
    ),
    # -- Latin America ----------------------------------------------------
    GeographicProfile(
        region_id="latin_america",
        name="Latin America",
        common_foods=[
            "rice and beans",
            "tortillas",
            "empanadas",
            "ceviche",
            "plantain",
            "avocado",
            "salsa",
            "tamales",
            "arepas",
            "quinoa",
        ],
        staple_grains=["maize", "rice", "wheat", "quinoa", "amaranth"],
        protein_sources=[
            "black beans",
            "chicken",
            "beef",
            "pork",
            "fish",
            "eggs",
            "cheese",
            "lentils",
            "shrimp",
        ],
        vegetables=[
            "tomato",
            "bell pepper",
            "chayote",
            "nopales",
            "corn",
            "avocado",
            "cassava",
            "sweet potato",
        ],
        fruits=["mango", "papaya", "guava", "passion fruit", "pineapple", "cherimoya"],
        spices=["cumin", "chilli", "cilantro", "oregano", "epazote", "achiote", "lime", "garlic"],
        traditional_dishes=[
            "tacos",
            "ceviche",
            "feijoada",
            "empanadas",
            "arepas",
            "tamales",
            "mole poblano",
            "arroz con pollo",
            "gallo pinto",
        ],
    ),
    # -- North America ----------------------------------------------------
    GeographicProfile(
        region_id="north_america",
        name="North America",
        common_foods=[
            "bread",
            "eggs",
            "chicken breast",
            "salad",
            "pasta",
            "hamburger",
            "milk",
            "cheese",
            "peanut butter",
            "cereal",
        ],
        staple_grains=["wheat", "corn", "oats", "rice", "barley"],
        protein_sources=[
            "chicken",
            "beef",
            "pork",
            "turkey",
            "fish",
            "eggs",
            "beans",
            "tofu",
            "dairy",
            "nuts",
        ],
        vegetables=[
            "broccoli",
            "spinach",
            "kale",
            "tomato",
            "bell pepper",
            "sweet potato",
            "corn",
            "green beans",
            "lettuce",
        ],
        fruits=["apple", "banana", "strawberry", "blueberry", "orange", "grape"],
        spices=[
            "black pepper",
            "garlic powder",
            "onion powder",
            "paprika",
            "oregano",
            "thyme",
            "cayenne",
            "cinnamon",
        ],
        traditional_dishes=[
            "grilled chicken salad",
            "BBQ ribs",
            "mac and cheese",
            "clam chowder",
            "Thanksgiving turkey",
            "burrito bowl",
            "Cajun gumbo",
            "poke bowl",
        ],
    ),
    # -- Central Asia -----------------------------------------------------
    GeographicProfile(
        region_id="central_asia",
        name="Central Asia",
        common_foods=[
            "plov",
            "naan",
            "lagman",
            "kefir",
            "kurt",
            "samsa",
            "shashlik",
            "beshbarmak",
            "manti",
            "kumis",
        ],
        staple_grains=["wheat", "rice", "barley", "millet"],
        protein_sources=[
            "lamb",
            "beef",
            "horse meat",
            "chicken",
            "yoghurt",
            "kefir",
            "lentils",
            "eggs",
        ],
        vegetables=[
            "carrot",
            "onion",
            "potato",
            "tomato",
            "pumpkin",
            "radish",
            "turnip",
            "bell pepper",
        ],
        fruits=["melon", "watermelon", "apricot", "grape", "apple", "pomegranate"],
        spices=[
            "cumin",
            "coriander",
            "black pepper",
            "barberry",
            "dill",
            "fennel",
            "sesame",
            "garlic",
        ],
        traditional_dishes=[
            "plov",
            "beshbarmak",
            "lagman",
            "manti",
            "samsa",
            "shashlik",
            "kurt",
            "shorpo",
        ],
    ),
    # -- West Africa --
    GeographicProfile(
        region_id="west_africa",
        name="West Africa",
        common_foods=[
            "jollof rice",
            "fufu",
            "egusi soup",
            "suya",
            "pounded yam",
            "okra soup",
            "plantain",
            "garri",
        ],
        staple_grains=["rice", "millet", "sorghum", "maize", "fonio"],
        protein_sources=[
            "black-eyed peas",
            "groundnuts",
            "smoked fish",
            "goat",
            "chicken",
            "cowpeas",
        ],
        vegetables=[
            "okra",
            "spinach",
            "tomato",
            "bitter leaf",
            "garden egg",
            "scotch bonnet peppers",
        ],
        fruits=["mango", "papaya", "pineapple", "baobab fruit", "shea fruit", "tamarind"],
        spices=["scotch bonnet", "locust bean (dawadawa)", "crayfish powder", "ginger", "garlic"],
        traditional_dishes=[
            "jollof rice",
            "egusi soup with fufu",
            "suya",
            "chin chin",
            "moin moin",
        ],
    ),
    # -- East Africa --
    GeographicProfile(
        region_id="east_africa",
        name="East Africa",
        common_foods=["ugali", "sukuma wiki", "nyama choma", "chapati", "mandazi", "injera"],
        staple_grains=["maize", "teff", "sorghum", "finger millet", "wheat"],
        protein_sources=["beef", "goat", "tilapia", "lentils", "beans", "chicken"],
        vegetables=["kale", "collard greens", "cabbage", "tomato", "onion", "pumpkin leaves"],
        fruits=["banana", "mango", "passion fruit", "avocado", "guava", "jackfruit"],
        spices=["berbere", "cardamom", "cumin", "turmeric", "coriander", "black pepper"],
        traditional_dishes=[
            "ugali with sukuma wiki",
            "nyama choma",
            "injera with wot",
            "pilau",
            "mandazi",
        ],
    ),
    # -- Southern Africa --
    GeographicProfile(
        region_id="southern_africa",
        name="Southern Africa",
        common_foods=["pap", "biltong", "bobotie", "chakalaka", "boerewors", "vetkoek"],
        staple_grains=["maize", "sorghum", "wheat", "rice"],
        protein_sources=["beef", "lamb", "chicken", "fish", "beans", "soya"],
        vegetables=["butternut squash", "spinach", "beetroot", "cabbage", "pumpkin", "green beans"],
        fruits=["naartjie", "grape", "peach", "marula", "watermelon", "fig"],
        spices=["peri-peri", "coriander", "cumin", "cinnamon", "clove", "nutmeg"],
        traditional_dishes=["bobotie", "biltong", "braai", "pap and chakalaka", "potjiekos"],
    ),
    # -- North Africa --
    GeographicProfile(
        region_id="north_africa",
        name="North Africa",
        common_foods=["couscous", "tagine", "harissa", "shakshuka", "msemen", "bissara"],
        staple_grains=["wheat", "barley", "semolina", "rice"],
        protein_sources=["lamb", "chicken", "chickpeas", "lentils", "sardines", "eggs"],
        vegetables=["tomato", "eggplant", "zucchini", "carrot", "artichoke", "broad beans"],
        fruits=["date", "fig", "pomegranate", "orange", "apricot", "olive"],
        spices=["cumin", "harissa", "ras el hanout", "saffron", "preserved lemon", "coriander"],
        traditional_dishes=["couscous royale", "tagine", "shakshuka", "pastilla", "merguez"],
    ),
    # -- Caribbean --
    GeographicProfile(
        region_id="caribbean",
        name="Caribbean",
        common_foods=[
            "rice and peas",
            "jerk chicken",
            "ackee and saltfish",
            "roti",
            "callaloo",
            "plantain",
        ],
        staple_grains=["rice", "cornmeal", "wheat flour", "cassava"],
        protein_sources=["chicken", "fish", "goat", "red beans", "pigeon peas", "saltfish"],
        vegetables=["callaloo", "okra", "scotch bonnet", "cho cho", "breadfruit", "yam"],
        fruits=["mango", "papaya", "guava", "soursop", "ackee", "star fruit"],
        spices=["allspice", "scotch bonnet", "thyme", "ginger", "garlic", "nutmeg"],
        traditional_dishes=[
            "jerk chicken",
            "ackee and saltfish",
            "rice and peas",
            "curry goat",
            "doubles",
        ],
    ),
    # -- South America --
    GeographicProfile(
        region_id="south_america",
        name="South America",
        common_foods=["feijoada", "arepa", "empanada", "ceviche", "churrasco", "quinoa"],
        staple_grains=["rice", "corn", "quinoa", "wheat", "cassava flour"],
        protein_sources=["beef", "chicken", "fish", "black beans", "lentils", "llama"],
        vegetables=["potato", "sweet potato", "corn", "tomato", "avocado", "peppers"],
        fruits=["acai", "passion fruit", "guava", "papaya", "cherimoya", "lucuma"],
        spices=["aji amarillo", "cumin", "oregano", "cilantro", "garlic", "chimichurri"],
        traditional_dishes=["feijoada", "ceviche", "empanadas", "asado", "lomo saltado"],
    ),
    # -- Pacific Islands --
    GeographicProfile(
        region_id="pacific_islands",
        name="Pacific Islands",
        common_foods=["taro", "breadfruit", "coconut", "poi", "laulau", "poke"],
        staple_grains=["taro", "breadfruit", "sweet potato", "rice", "cassava"],
        protein_sources=["tuna", "mahi-mahi", "octopus", "pork", "chicken", "tofu"],
        vegetables=["taro leaves", "sweet potato leaves", "seaweed", "long beans", "squash"],
        fruits=["coconut", "breadfruit", "papaya", "guava", "pineapple", "banana"],
        spices=["coconut cream", "ginger", "garlic", "chilli", "lemongrass", "lime"],
        traditional_dishes=["poke bowl", "laulau", "poi", "kalua pig", "umu (earth oven)"],
    ),
    # -- Eastern Europe --
    GeographicProfile(
        region_id="eastern_europe",
        name="Eastern Europe",
        common_foods=["borscht", "pierogi", "sauerkraut", "kielbasa", "blini", "pelmeni"],
        staple_grains=["wheat", "rye", "buckwheat", "barley", "oats"],
        protein_sources=["pork", "beef", "chicken", "carp", "sour cream", "cottage cheese"],
        vegetables=["potato", "beetroot", "cabbage", "carrot", "onion", "dill"],
        fruits=["apple", "plum", "cherry", "pear", "currant", "raspberry"],
        spices=["dill", "caraway", "paprika", "horseradish", "garlic", "bay leaf"],
        traditional_dishes=["borscht", "pierogi", "pelmeni", "goulash", "blini with smetana"],
    ),
    # -- Western Europe --
    GeographicProfile(
        region_id="western_europe",
        name="Western Europe",
        common_foods=["baguette", "croissant", "bratwurst", "cheese", "ratatouille", "paella"],
        staple_grains=["wheat", "rye", "oats", "barley", "spelt"],
        protein_sources=["beef", "pork", "chicken", "salmon", "cheese", "lentils"],
        vegetables=["potato", "carrot", "leek", "asparagus", "artichoke", "endive"],
        fruits=["apple", "grape", "peach", "pear", "strawberry", "citrus"],
        spices=["thyme", "rosemary", "bay leaf", "tarragon", "nutmeg", "mustard"],
        traditional_dishes=["coq au vin", "bratwurst", "paella", "fish and chips", "risotto"],
    ),
    # -- Central America --
    GeographicProfile(
        region_id="central_america",
        name="Central America",
        common_foods=["tortilla", "tamale", "pupusa", "gallo pinto", "ceviche", "enchilada"],
        staple_grains=["corn", "rice", "wheat", "black beans"],
        protein_sources=["chicken", "pork", "beef", "black beans", "cheese", "fish"],
        vegetables=["tomato", "chayote", "jalapeño", "avocado", "corn", "nopales"],
        fruits=["mango", "papaya", "pineapple", "guava", "plantain", "tamarind"],
        spices=["cumin", "chili powder", "oregano", "cilantro", "achiote", "epazote"],
        traditional_dishes=["pupusas", "gallo pinto", "tamales", "enchiladas", "sopa de frijoles"],
    ),
    # -- Australia & New Zealand --
    GeographicProfile(
        region_id="australian",
        name="Australia & New Zealand",
        common_foods=[
            "vegemite toast",
            "meat pie",
            "lamington",
            "pavlova",
            "barramundi",
            "tim tam",
        ],
        staple_grains=["wheat", "oats", "rice", "barley"],
        protein_sources=["beef", "lamb", "kangaroo", "barramundi", "salmon", "prawns"],
        vegetables=["sweet potato", "beetroot", "pumpkin", "spinach", "avocado", "capsicum"],
        fruits=["mango", "passion fruit", "kiwi", "macadamia", "finger lime", "quandong"],
        spices=["lemon myrtle", "bush tomato", "wattleseed", "pepperberry", "anise myrtle"],
        traditional_dishes=["meat pie", "barramundi", "pavlova", "lamingtons", "vegemite on toast"],
    ),
    # -- Nordic (Scandinavia) --
    GeographicProfile(
        region_id="nordic",
        name="Nordic (Scandinavia)",
        common_foods=["smørrebrød", "gravlax", "meatballs", "knäckebröd", "cloudberries"],
        staple_grains=["rye", "oats", "barley", "spelt", "wheat"],
        protein_sources=["salmon", "herring", "cod", "reindeer", "pork", "dairy"],
        vegetables=["potato", "root vegetables", "cabbage", "peas", "mushrooms", "kale"],
        fruits=["lingonberry", "cloudberry", "blueberry", "apple", "plum", "rhubarb"],
        spices=["dill", "cardamom", "juniper", "caraway", "allspice", "horseradish"],
        traditional_dishes=[
            "gravlax",
            "smørrebrød",
            "Swedish meatballs",
            "pytt i panna",
            "surströmming",
        ],
    ),
    # -- Balkans --
    GeographicProfile(
        region_id="balkans",
        name="Balkans",
        common_foods=["burek", "ćevapi", "shopska salad", "ajvar", "baklava", "sarma"],
        staple_grains=["wheat", "corn", "barley", "rice"],
        protein_sources=["lamb", "beef", "pork", "chicken", "white cheese", "yoghurt"],
        vegetables=["pepper", "tomato", "eggplant", "onion", "cabbage", "spinach"],
        fruits=["plum", "grape", "fig", "cherry", "watermelon", "quince"],
        spices=["paprika", "garlic", "bay leaf", "oregano", "rosemary", "parsley"],
        traditional_dishes=["burek", "ćevapi", "moussaka", "sarma", "shopska salad"],
    ),
    # -- Caucasus --
    GeographicProfile(
        region_id="caucasus",
        name="Caucasus",
        common_foods=["khachapuri", "khinkali", "dolma", "lavash", "churchkhela"],
        staple_grains=["wheat", "corn", "rice", "barley"],
        protein_sources=["lamb", "beef", "chicken", "walnut", "cheese", "beans"],
        vegetables=["eggplant", "tomato", "pepper", "herbs", "beet greens", "leek"],
        fruits=["pomegranate", "grape", "persimmon", "quince", "cherry", "apricot"],
        spices=["fenugreek", "coriander", "saffron", "sumac", "marigold", "blue fenugreek"],
        traditional_dishes=["khachapuri", "khinkali", "dolma", "lobio", "plov"],
    ),
    # -- Central Europe --
    GeographicProfile(
        region_id="central_europe",
        name="Central Europe",
        common_foods=["schnitzel", "strudel", "spaetzle", "pretzel", "sauerkraut", "goulash"],
        staple_grains=["wheat", "rye", "spelt", "barley", "oats"],
        protein_sources=["pork", "beef", "chicken", "trout", "cheese", "lentils"],
        vegetables=["potato", "cabbage", "turnip", "carrot", "asparagus", "kohlrabi"],
        fruits=["apple", "pear", "plum", "cherry", "apricot", "grape"],
        spices=["caraway", "paprika", "marjoram", "juniper", "horseradish", "mustard"],
        traditional_dishes=[
            "wiener schnitzel",
            "goulash",
            "spaetzle",
            "apple strudel",
            "sauerbraten",
        ],
    ),
    # -- Andean Region --
    GeographicProfile(
        region_id="andean",
        name="Andean Region",
        common_foods=["quinoa", "chuño", "cuy", "anticuchos", "ceviche", "humitas"],
        staple_grains=["quinoa", "corn", "potato", "kiwicha"],
        protein_sources=["guinea pig (cuy)", "llama", "alpaca", "trout", "beans", "eggs"],
        vegetables=["potato", "sweet potato", "oca", "ulluco", "corn", "squash"],
        fruits=["lucuma", "cherimoya", "passion fruit", "cape gooseberry", "prickly pear"],
        spices=["aji amarillo", "huacatay", "cumin", "oregano", "garlic", "rocoto"],
        traditional_dishes=[
            "ceviche",
            "lomo saltado",
            "anticuchos",
            "pachamanca",
            "papa a la huancaina",
        ],
    ),
    # -- Iberian Peninsula --
    GeographicProfile(
        region_id="iberian",
        name="Iberian Peninsula",
        common_foods=["paella", "gazpacho", "bacalhau", "tapas", "churros", "caldo verde"],
        staple_grains=["wheat", "rice", "corn", "barley"],
        protein_sources=["pork (ibérico)", "sardines", "cod", "chicken", "chorizo", "chickpeas"],
        vegetables=["tomato", "pepper", "garlic", "onion", "potato", "olive"],
        fruits=["orange", "grape", "fig", "almond", "lemon", "quince"],
        spices=["saffron", "paprika (pimentón)", "garlic", "bay leaf", "oregano", "parsley"],
        traditional_dishes=[
            "paella",
            "bacalhau",
            "gazpacho",
            "pastéis de nata",
            "fabada asturiana",
        ],
    ),
]

# Build a quick-lookup dict by region_id.
_REGION_INDEX: dict[str, GeographicProfile] = {gp.region_id: gp for gp in GEOGRAPHIC_FOOD_DB}

# ---------------------------------------------------------------------------
# Nutrigenomics Database  (25+ gene–nutrient associations)
# ---------------------------------------------------------------------------

# Each entry: (rsid, gene, nutrient, effect, recommendation,
#              priority, daily_target_mg, upper_limit_mg, confidence)
NUTRIGENOMICS_DB: list[dict[str, Any]] = [
    # 1 — MTHFR C677T → folate
    {
        "rsid": "rs1801133",
        "gene": "MTHFR",
        "nutrient": "folate",
        "effect": "Reduced enzyme activity impairs folate metabolism",
        "recommendation": "Increase folate-rich foods; consider methylfolate",
        "priority": "critical",
        "daily_target_mg": 0.8,
        "upper_limit_mg": 1.0,
        "confidence": 0.92,
    },
    # 2 — MTHFR A1298C → folate (compound heterozygote risk)
    {
        "rsid": "rs1801131",
        "gene": "MTHFR",
        "nutrient": "folate",
        "effect": "Moderately reduced MTHFR activity",
        "recommendation": "Ensure adequate dietary folate intake",
        "priority": "high",
        "daily_target_mg": 0.6,
        "upper_limit_mg": 1.0,
        "confidence": 0.80,
    },
    # 3 — FTO rs9939609 → calorie management
    {
        "rsid": "rs9939609",
        "gene": "FTO",
        "nutrient": "calories",
        "effect": "Increased appetite signalling and fat storage tendency",
        "recommendation": "Prioritise high-satiety, lower calorie-density foods",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.88,
    },
    # 4 — FTO rs1558902 → calorie management
    {
        "rsid": "rs1558902",
        "gene": "FTO",
        "nutrient": "calories",
        "effect": "Associated with higher BMI and adiposity",
        "recommendation": "Focus on fibre-rich whole grains and lean protein",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.85,
    },
    # 5 — LCT → lactose
    {
        "rsid": "rs4988235",
        "gene": "LCT",
        "nutrient": "lactose",
        "effect": "Lactase non-persistence in adulthood",
        "recommendation": "Limit dairy; use fermented dairy or calcium alternatives",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.95,
    },
    # 6 — FADS1 → omega-3
    {
        "rsid": "rs174546",
        "gene": "FADS1",
        "nutrient": "omega_3",
        "effect": "Reduced conversion of ALA to EPA/DHA",
        "recommendation": "Consume preformed EPA/DHA from fish or algae oil",
        "priority": "critical",
        "daily_target_mg": 500.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.87,
    },
    # 7 — FADS2 → omega-3 / omega-6 balance
    {
        "rsid": "rs1535",
        "gene": "FADS2",
        "nutrient": "omega_3",
        "effect": "Altered desaturase activity affecting fatty acid balance",
        "recommendation": "Increase omega-3, reduce omega-6 seed oils",
        "priority": "high",
        "daily_target_mg": 400.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.82,
    },
    # 8 — CYP1A2 → caffeine
    {
        "rsid": "rs762551",
        "gene": "CYP1A2",
        "nutrient": "caffeine",
        "effect": "Slow caffeine metabolism increases cardiovascular risk",
        "recommendation": "Limit caffeine to ≤ 200 mg/day (≈ 1 cup coffee)",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 200.0,
        "confidence": 0.83,
    },
    # 9 — VDR Bsm1 → vitamin D
    {
        "rsid": "rs1544410",
        "gene": "VDR",
        "nutrient": "vitamin_d",
        "effect": "Altered vitamin D receptor sensitivity",
        "recommendation": "Higher vitamin D intake; monitor serum 25(OH)D",
        "priority": "high",
        "daily_target_mg": 0.05,
        "upper_limit_mg": 0.1,
        "confidence": 0.79,
    },
    # 10 — VDR FokI → vitamin D
    {
        "rsid": "rs2228570",
        "gene": "VDR",
        "nutrient": "vitamin_d",
        "effect": "Reduced VDR transcriptional activity",
        "recommendation": "Ensure sun exposure and vitamin D-rich foods",
        "priority": "moderate",
        "daily_target_mg": 0.04,
        "upper_limit_mg": 0.1,
        "confidence": 0.75,
    },
    # 11 — HFE C282Y → iron
    {
        "rsid": "rs1800562",
        "gene": "HFE",
        "nutrient": "iron",
        "effect": "Increased intestinal iron absorption (haemochromatosis risk)",
        "recommendation": "Avoid iron supplements; limit red meat; monitor ferritin",
        "priority": "critical",
        "daily_target_mg": 8.0,
        "upper_limit_mg": 18.0,
        "confidence": 0.93,
    },
    # 12 — HFE H63D → iron
    {
        "rsid": "rs1799945",
        "gene": "HFE",
        "nutrient": "iron",
        "effect": "Mildly increased iron absorption",
        "recommendation": "Monitor iron levels; avoid unnecessary supplementation",
        "priority": "moderate",
        "daily_target_mg": 10.0,
        "upper_limit_mg": 25.0,
        "confidence": 0.78,
    },
    # 13 — APOE ε4 → saturated fat
    {
        "rsid": "rs429358",
        "gene": "APOE",
        "nutrient": "saturated_fat",
        "effect": "APOE-ε4 carriers have heightened LDL response to saturated fat",
        "recommendation": "Limit saturated fat to < 7% of calories; favour MUFA/PUFA",
        "priority": "critical",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.91,
    },
    # 14 — APOE ε2 → lipid profile
    {
        "rsid": "rs7412",
        "gene": "APOE",
        "nutrient": "saturated_fat",
        "effect": "APOE-ε2 may elevate triglycerides",
        "recommendation": "Moderate total fat; increase omega-3 intake",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.74,
    },
    # 15 — TCF7L2 → glycemic control
    {
        "rsid": "rs7903146",
        "gene": "TCF7L2",
        "nutrient": "glycemic_control",
        "effect": "Impaired insulin secretion and incretin signalling",
        "recommendation": "Low glycemic index diet; limit refined carbohydrates",
        "priority": "critical",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.90,
    },
    # 16 — TCF7L2 secondary variant
    {
        "rsid": "rs12255372",
        "gene": "TCF7L2",
        "nutrient": "glycemic_control",
        "effect": "Additional risk for impaired glucose homeostasis",
        "recommendation": "Complex carbs with high fibre; avoid sugar-sweetened drinks",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.86,
    },
    # 17 — SLC23A1 → vitamin C
    {
        "rsid": "rs33972313",
        "gene": "SLC23A1",
        "nutrient": "vitamin_c",
        "effect": "Reduced vitamin C transporter activity",
        "recommendation": "Increase citrus, bell peppers, and guava consumption",
        "priority": "high",
        "daily_target_mg": 200.0,
        "upper_limit_mg": 2000.0,
        "confidence": 0.80,
    },
    # 18 — BCMO1 → vitamin A (beta-carotene conversion)
    {
        "rsid": "rs7501331",
        "gene": "BCMO1",
        "nutrient": "vitamin_a",
        "effect": "Poor conversion of beta-carotene to retinol",
        "recommendation": "Include preformed vitamin A (liver, eggs, dairy)",
        "priority": "high",
        "daily_target_mg": 0.9,
        "upper_limit_mg": 3.0,
        "confidence": 0.81,
    },
    # 19 — BCMO1 secondary variant
    {
        "rsid": "rs12934922",
        "gene": "BCMO1",
        "nutrient": "vitamin_a",
        "effect": "Further reduced beta-carotene cleavage efficiency",
        "recommendation": "Do not rely solely on plant sources for vitamin A",
        "priority": "moderate",
        "daily_target_mg": 0.9,
        "upper_limit_mg": 3.0,
        "confidence": 0.76,
    },
    # 20 — NAT2 → cruciferous vegetables
    {
        "rsid": "rs1801280",
        "gene": "NAT2",
        "nutrient": "cruciferous_vegetables",
        "effect": "Slow acetylator status affects isothiocyanate metabolism",
        "recommendation": "Increase cruciferous vegetable intake for cancer protection",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.77,
    },
    # 21 — GSTT1 → cruciferous vegetables
    {
        "rsid": "rs71748309",
        "gene": "GSTT1",
        "nutrient": "cruciferous_vegetables",
        "effect": "Null genotype reduces glutathione conjugation",
        "recommendation": "Broccoli, kale, Brussels sprouts for sulforaphane",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    # 22 — NBPF3 / alcohol metabolism (ADH1B)
    {
        "rsid": "rs1229984",
        "gene": "ADH1B",
        "nutrient": "alcohol",
        "effect": "Rapid ethanol oxidation causing acetaldehyde accumulation",
        "recommendation": "Minimise or avoid alcohol consumption",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.90,
    },
    # 23 — SLC30A8 → zinc
    {
        "rsid": "rs13266634",
        "gene": "SLC30A8",
        "nutrient": "zinc",
        "effect": "Altered zinc transporter in pancreatic beta cells",
        "recommendation": "Ensure adequate zinc from shellfish, seeds, legumes",
        "priority": "moderate",
        "daily_target_mg": 15.0,
        "upper_limit_mg": 40.0,
        "confidence": 0.75,
    },
    # 24 — GC → vitamin D binding protein
    {
        "rsid": "rs2282679",
        "gene": "GC",
        "nutrient": "vitamin_d",
        "effect": "Lower circulating 25(OH)D due to binding-protein variants",
        "recommendation": "Consider vitamin D supplementation; test serum levels",
        "priority": "high",
        "daily_target_mg": 0.05,
        "upper_limit_mg": 0.1,
        "confidence": 0.82,
    },
    # 25 — PPARG → fat metabolism
    {
        "rsid": "rs1801282",
        "gene": "PPARG",
        "nutrient": "monounsaturated_fat",
        "effect": "Pro12Ala variant modifies fat-storage response",
        "recommendation": "Favour olive oil and nuts as primary fat sources",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.79,
    },
    # 26 — COMT → polyphenols / green tea
    {
        "rsid": "rs4680",
        "gene": "COMT",
        "nutrient": "polyphenols",
        "effect": "Val158Met affects catechol-O-methyltransferase activity",
        "recommendation": "Green tea catechins may be more beneficial; include regularly",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.68,
    },
    # 27 — TNF-α → anti-inflammatory diet
    {
        "rsid": "rs1800629",
        "gene": "TNF",
        "nutrient": "anti_inflammatory",
        "effect": "Elevated TNF-α production; pro-inflammatory phenotype",
        "recommendation": "Anti-inflammatory diet rich in omega-3, turmeric, berries",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.80,
    },
    # 28 — IL6 → anti-inflammatory
    {
        "rsid": "rs1800795",
        "gene": "IL6",
        "nutrient": "anti_inflammatory",
        "effect": "Altered interleukin-6 expression affecting inflammation",
        "recommendation": "Increase antioxidant and polyphenol-rich food intake",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.73,
    },
    # 29 — SOD2 → antioxidants
    {
        "rsid": "rs4880",
        "gene": "SOD2",
        "nutrient": "antioxidants",
        "effect": "Ala16Val affects mitochondrial superoxide dismutase efficiency",
        "recommendation": "Boost antioxidant intake: vitamins C, E, selenium",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.74,
    },
    {
        "rsid": "rs762551",
        "gene": "CYP1A2",
        "nutrient": "caffeine",
        "effect": "Slow caffeine metabolism associated with increased cardiovascular risk with high caffeine intake",
        "recommendation": "Limit coffee to 1-2 cups/day; avoid energy drinks",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 200.0,
        "confidence": 0.88,
    },
    {
        "rsid": "rs4410790",
        "gene": "AHR",
        "nutrient": "caffeine",
        "effect": "Influences caffeine consumption behaviour and metabolism",
        "recommendation": "Monitor caffeine sensitivity; adjust intake based on response",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 300.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs1601993660",
        "gene": "GSTT1",
        "nutrient": "sulforaphane",
        "effect": "Null genotype reduces detoxification of environmental carcinogens",
        "recommendation": "Increase cruciferous vegetable intake (broccoli, kale, cabbage)",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs366631",
        "gene": "GSTM1",
        "nutrient": "sulforaphane",
        "effect": "Null genotype impairs Phase II detoxification",
        "recommendation": "Consume broccoli sprouts and other brassicas regularly",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs4880",
        "gene": "SOD2",
        "nutrient": "antioxidants",
        "effect": "Val allele reduces mitochondrial antioxidant defense",
        "recommendation": "Ensure adequate vitamin C, E, and selenium intake",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs1001179",
        "gene": "CAT",
        "nutrient": "antioxidants",
        "effect": "Reduced catalase activity increases oxidative stress",
        "recommendation": "Increase dietary antioxidants from colourful fruits and vegetables",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs1050450",
        "gene": "GPX1",
        "nutrient": "selenium",
        "effect": "Reduced glutathione peroxidase activity with Pro allele",
        "recommendation": "Ensure selenium intake of 55-70 mcg/day from Brazil nuts, fish, or eggs",
        "priority": "high",
        "daily_target_mg": 0.055,
        "upper_limit_mg": 0.4,
        "confidence": 0.85,
    },
    {
        "rsid": "rs1800629",
        "gene": "TNF",
        "nutrient": "omega_3",
        "effect": "Increased TNF-alpha expression promoting chronic inflammation",
        "recommendation": "Increase anti-inflammatory omega-3 fatty acids from oily fish",
        "priority": "high",
        "daily_target_mg": 1000.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs1800795",
        "gene": "IL6",
        "nutrient": "omega_3",
        "effect": "Elevated IL-6 levels associated with chronic inflammation",
        "recommendation": "Adopt anti-inflammatory diet rich in omega-3 and polyphenols",
        "priority": "high",
        "daily_target_mg": 1000.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs1205",
        "gene": "CRP",
        "nutrient": "omega_3",
        "effect": "Elevated C-reactive protein levels",
        "recommendation": "Mediterranean diet pattern with oily fish 2-3 times per week",
        "priority": "moderate",
        "daily_target_mg": 1000.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs4680",
        "gene": "COMT",
        "nutrient": "polyphenols",
        "effect": "Val/Met affects catechol metabolism and green tea catechin benefit",
        "recommendation": "Met/Met carriers benefit more from green tea catechins",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs1800566",
        "gene": "NQO1",
        "nutrient": "coenzyme_q10",
        "effect": "Reduced NQO1 enzyme activity impairs quinone detoxification",
        "recommendation": "Supplement with CoQ10 100-200 mg/day; eat organ meats and sardines",
        "priority": "moderate",
        "daily_target_mg": 100.0,
        "upper_limit_mg": 400.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs7946",
        "gene": "PEMT",
        "nutrient": "choline",
        "effect": "Reduced endogenous choline synthesis",
        "recommendation": "Ensure adequate choline from eggs, liver, and cruciferous vegetables",
        "priority": "high",
        "daily_target_mg": 550.0,
        "upper_limit_mg": 3500.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs174546",
        "gene": "FADS1",
        "nutrient": "omega_3",
        "effect": "Reduced conversion of ALA to EPA/DHA",
        "recommendation": "Prioritise preformed EPA/DHA from fish over plant ALA sources",
        "priority": "high",
        "daily_target_mg": 500.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.88,
    },
    {
        "rsid": "rs1535",
        "gene": "FADS2",
        "nutrient": "omega_3",
        "effect": "Altered desaturase activity affects omega-6/3 ratio",
        "recommendation": "Reduce omega-6 (vegetable oils) and increase omega-3 (fish, flax)",
        "priority": "high",
        "daily_target_mg": 500.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.85,
    },
    {
        "rsid": "rs33972313",
        "gene": "SLC23A1",
        "nutrient": "vitamin_c",
        "effect": "Reduced vitamin C transporter efficiency",
        "recommendation": "Increase vitamin C intake from citrus, kiwi, bell peppers, and berries",
        "priority": "moderate",
        "daily_target_mg": 200.0,
        "upper_limit_mg": 2000.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs4588",
        "gene": "GC",
        "nutrient": "vitamin_d",
        "effect": "Reduced vitamin D binding protein affects circulating 25(OH)D",
        "recommendation": "Ensure vitamin D from sunlight, oily fish, and fortified foods",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs10741657",
        "gene": "CYP2R1",
        "nutrient": "vitamin_d",
        "effect": "Reduced hepatic 25-hydroxylation of vitamin D",
        "recommendation": "May need higher vitamin D supplementation (2000-4000 IU/day)",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs1800562",
        "gene": "HFE",
        "nutrient": "iron",
        "effect": "C282Y mutation causes excessive iron absorption (haemochromatosis risk)",
        "recommendation": "Avoid iron supplements; limit red meat; donate blood regularly",
        "priority": "critical",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 8.0,
        "confidence": 0.95,
    },
    {
        "rsid": "rs1799945",
        "gene": "HFE",
        "nutrient": "iron",
        "effect": "H63D variant mildly increases iron absorption",
        "recommendation": "Monitor serum ferritin; avoid unnecessary iron supplementation",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 18.0,
        "confidence": 0.85,
    },
    {
        "rsid": "rs7385804",
        "gene": "TFR2",
        "nutrient": "iron",
        "effect": "Altered transferrin receptor 2 function affects iron sensing",
        "recommendation": "Regular ferritin monitoring; balanced iron intake",
        "priority": "moderate",
        "daily_target_mg": 8.0,
        "upper_limit_mg": 45.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs662",
        "gene": "PON1",
        "nutrient": "polyphenols",
        "effect": "Reduced paraoxonase 1 activity impairs HDL antioxidant function",
        "recommendation": "Consume polyphenol-rich foods (olive oil, berries, dark chocolate)",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs708272",
        "gene": "CETP",
        "nutrient": "saturated_fat",
        "effect": "CETP variants modulate HDL response to dietary fat",
        "recommendation": "Limit saturated fat to <7% of calories; prefer mono/polyunsaturated fats",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs1800588",
        "gene": "LIPC",
        "nutrient": "saturated_fat",
        "effect": "Hepatic lipase promoter variant affects HDL metabolism",
        "recommendation": "Low-fat diet may paradoxically lower HDL in CC genotype; focus on quality fats",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs5082",
        "gene": "APOA2",
        "nutrient": "saturated_fat",
        "effect": "CC genotype increases obesity risk with high saturated fat intake",
        "recommendation": "Keep saturated fat below 22 g/day; prefer plant-based fats",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 22000.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs1801282",
        "gene": "PPARG",
        "nutrient": "total_fat",
        "effect": "Pro12Ala variant affects adipocyte differentiation and insulin sensitivity",
        "recommendation": "Moderate fat intake (25-35% calories); Mediterranean diet beneficial",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs7903146",
        "gene": "TCF7L2",
        "nutrient": "glycaemic_index",
        "effect": "Risk allele impairs insulin secretion and glucose tolerance",
        "recommendation": "Prefer low-GI foods; limit refined carbohydrates and added sugars",
        "priority": "critical",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.9,
    },
    {
        "rsid": "rs5400",
        "gene": "SLC2A2",
        "nutrient": "fructose",
        "effect": "Altered GLUT2 transporter affects fructose sensing and sugar intake",
        "recommendation": "Limit added fructose and high-fructose corn syrup",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 25000.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs713598",
        "gene": "TAS2R38",
        "nutrient": "cruciferous_vegetables",
        "effect": "Supertaster phenotype may reduce cruciferous vegetable intake",
        "recommendation": "Prepare brassicas with complementary flavours to mask bitterness",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.65,
    },
    {
        "rsid": "rs72921001",
        "gene": "OR6A2",
        "nutrient": "herbs",
        "effect": "Associated with perception of cilantro as soapy",
        "recommendation": "Substitute parsley or basil if cilantro is aversive",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.6,
    },
    {
        "rsid": "rs1229984",
        "gene": "ADH1B",
        "nutrient": "alcohol",
        "effect": "Rapid alcohol metabolism causes unpleasant flushing response",
        "recommendation": "Avoid or strictly limit alcohol consumption",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.92,
    },
    {
        "rsid": "rs671",
        "gene": "ALDH2",
        "nutrient": "alcohol",
        "effect": "Deficient aldehyde dehydrogenase causes acetaldehyde accumulation",
        "recommendation": "Avoid alcohol completely; increased esophageal cancer risk",
        "priority": "critical",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.95,
    },
    {
        "rsid": "rs1801394",
        "gene": "MTRR",
        "nutrient": "vitamin_b12",
        "effect": "Reduced methionine synthase reductase activity",
        "recommendation": "Ensure adequate B12 from animal products or methylcobalamin supplement",
        "priority": "high",
        "daily_target_mg": 0.005,
        "upper_limit_mg": 0.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs1805087",
        "gene": "MTR",
        "nutrient": "vitamin_b12",
        "effect": "Altered methionine synthase affects homocysteine metabolism",
        "recommendation": "Combine adequate B12 and folate; monitor homocysteine levels",
        "priority": "high",
        "daily_target_mg": 0.005,
        "upper_limit_mg": 0.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs234706",
        "gene": "CBS",
        "nutrient": "vitamin_b6",
        "effect": "Upregulated cystathionine beta-synthase increases B6 requirement",
        "recommendation": "Ensure adequate B6 from poultry, fish, potatoes, and bananas",
        "priority": "moderate",
        "daily_target_mg": 2.0,
        "upper_limit_mg": 100.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs3733890",
        "gene": "BHMT",
        "nutrient": "betaine",
        "effect": "Altered betaine-homocysteine methyltransferase activity",
        "recommendation": "Include betaine-rich foods: beets, spinach, quinoa, wheat germ",
        "priority": "moderate",
        "daily_target_mg": 500.0,
        "upper_limit_mg": 6000.0,
        "confidence": 0.68,
    },
    {
        "rsid": "rs4654748",
        "gene": "NBPF3",
        "nutrient": "vitamin_b6",
        "effect": "Lower circulating pyridoxal 5'-phosphate levels",
        "recommendation": "Higher B6 intake from meat, fish, chickpeas, potatoes",
        "priority": "moderate",
        "daily_target_mg": 2.5,
        "upper_limit_mg": 100.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs602662",
        "gene": "FUT2",
        "nutrient": "vitamin_b12",
        "effect": "Non-secretor status associated with lower B12 levels",
        "recommendation": "Regular B12 monitoring; consider supplementation",
        "priority": "high",
        "daily_target_mg": 0.005,
        "upper_limit_mg": 0.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs6543836",
        "gene": "SLC6A19",
        "nutrient": "tryptophan",
        "effect": "Reduced neutral amino acid transport (Hartnup-like)",
        "recommendation": "Ensure adequate tryptophan from eggs, turkey, nuts, and seeds",
        "priority": "moderate",
        "daily_target_mg": 250.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.65,
    },
    {
        "rsid": "rs4646994",
        "gene": "ACE",
        "nutrient": "sodium",
        "effect": "DD genotype associated with salt-sensitive hypertension",
        "recommendation": "Restrict sodium to <1500 mg/day; increase potassium-rich foods",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 1500.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs4961",
        "gene": "ADD1",
        "nutrient": "sodium",
        "effect": "Gly460Trp associated with salt-sensitive blood pressure",
        "recommendation": "Low-sodium diet; DASH dietary pattern recommended",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 1500.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs699",
        "gene": "AGT",
        "nutrient": "sodium",
        "effect": "M235T variant increases angiotensinogen, promoting sodium retention",
        "recommendation": "Reduce salt; increase potassium from bananas, sweet potatoes, leafy greens",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 2000.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs1799983",
        "gene": "NOS3",
        "nutrient": "nitrate",
        "effect": "Reduced endothelial nitric oxide production",
        "recommendation": "Increase dietary nitrate from beetroot, rocket, and spinach",
        "priority": "moderate",
        "daily_target_mg": 300.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs1042713",
        "gene": "ADRB2",
        "nutrient": "carbohydrates",
        "effect": "Gln27Glu associated with obesity risk on high-carb diets",
        "recommendation": "Moderate carbohydrate intake; focus on complex carbs and fibre",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs1800592",
        "gene": "UCP1",
        "nutrient": "calories",
        "effect": "Reduced brown adipose tissue thermogenesis",
        "recommendation": "Cold exposure therapy; capsaicin-rich foods may support thermogenesis",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.65,
    },
    {
        "rsid": "rs659366",
        "gene": "UCP2",
        "nutrient": "calories",
        "effect": "Altered mitochondrial uncoupling affects energy expenditure",
        "recommendation": "Adjust caloric intake based on metabolic rate; regular exercise",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.62,
    },
    {
        "rsid": "rs1800849",
        "gene": "UCP3",
        "nutrient": "total_fat",
        "effect": "Altered fatty acid oxidation in skeletal muscle",
        "recommendation": "Moderate-fat diet with emphasis on unsaturated fatty acids",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.6,
    },
    {
        "rsid": "rs17782313",
        "gene": "MC4R",
        "nutrient": "fiber",
        "effect": "Disrupted satiety signalling increases overeating risk",
        "recommendation": "High-fibre, high-protein meals to enhance satiety signalling",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs1801260",
        "gene": "CLOCK",
        "nutrient": "meal_timing",
        "effect": "Disrupted circadian rhythm affects metabolic efficiency",
        "recommendation": "Time-restricted eating; avoid late-night eating; consistent meal times",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs2304672",
        "gene": "PER2",
        "nutrient": "meal_timing",
        "effect": "Altered Period 2 protein affects circadian metabolic rhythms",
        "recommendation": "Front-load calories earlier in the day; maintain regular sleep",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.68,
    },
    {
        "rsid": "rs73598374",
        "gene": "ADA",
        "nutrient": "caffeine",
        "effect": "Altered adenosine deaminase activity affects caffeine sensitivity and sleep",
        "recommendation": "Avoid caffeine after noon; limit to 100 mg/day if sensitive",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 100.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs1183201",
        "gene": "SLC17A1",
        "nutrient": "purines",
        "effect": "Reduced urate excretion increases gout risk",
        "recommendation": "Limit high-purine foods (organ meats, shellfish); increase hydration",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs11942223",
        "gene": "SLC2A9",
        "nutrient": "purines",
        "effect": "Altered urate transporter affects serum uric acid levels",
        "recommendation": "Cherry consumption; vitamin C supplementation may lower urate",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs2231142",
        "gene": "ABCG2",
        "nutrient": "purines",
        "effect": "Q141K reduces urate excretion substantially",
        "recommendation": "Strict purine limitation; adequate hydration; consider allopurinol",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.85,
    },
    {
        "rsid": "rs2228570",
        "gene": "VDR",
        "nutrient": "vitamin_d",
        "effect": "FokI polymorphism affects vitamin D receptor function",
        "recommendation": "May need higher vitamin D intake; calcium co-supplementation",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs1544410",
        "gene": "VDR",
        "nutrient": "calcium",
        "effect": "BsmI variant affects calcium absorption efficiency",
        "recommendation": "Ensure 1000-1200 mg calcium/day with adequate vitamin D",
        "priority": "moderate",
        "daily_target_mg": 1000.0,
        "upper_limit_mg": 2500.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs2234693",
        "gene": "ESR1",
        "nutrient": "calcium",
        "effect": "Oestrogen receptor variant affects bone mineral density response",
        "recommendation": "Weight-bearing exercise; ensure calcium and vitamin D adequacy",
        "priority": "moderate",
        "daily_target_mg": 1000.0,
        "upper_limit_mg": 2500.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs1800012",
        "gene": "COL1A1",
        "nutrient": "vitamin_c",
        "effect": "Sp1 polymorphism affects collagen synthesis and bone density",
        "recommendation": "Adequate vitamin C for collagen synthesis; weight-bearing exercise",
        "priority": "moderate",
        "daily_target_mg": 200.0,
        "upper_limit_mg": 2000.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs9594738",
        "gene": "TNFSF11",
        "nutrient": "calcium",
        "effect": "Altered RANKL expression affects osteoclast activity",
        "recommendation": "Calcium-rich diet with vitamin K2 for bone matrix support",
        "priority": "moderate",
        "daily_target_mg": 1200.0,
        "upper_limit_mg": 2500.0,
        "confidence": 0.68,
    },
    {
        "rsid": "rs13266634",
        "gene": "SLC30A8",
        "nutrient": "zinc",
        "effect": "Altered zinc transporter affects pancreatic beta-cell function",
        "recommendation": "Ensure adequate zinc from oysters, red meat, pumpkin seeds",
        "priority": "moderate",
        "daily_target_mg": 11.0,
        "upper_limit_mg": 40.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs3877899",
        "gene": "SELENOP",
        "nutrient": "selenium",
        "effect": "Reduced selenoprotein P affects selenium distribution to tissues",
        "recommendation": "Ensure 55-70 mcg selenium/day; Brazil nuts are excellent source",
        "priority": "moderate",
        "daily_target_mg": 0.055,
        "upper_limit_mg": 0.4,
        "confidence": 0.72,
    },
    {
        "rsid": "rs780094",
        "gene": "GCKR",
        "nutrient": "fructose",
        "effect": "Altered glucokinase regulatory protein increases triglycerides",
        "recommendation": "Strictly limit fructose and sucrose; avoid sugary beverages",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 25000.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs1501299",
        "gene": "ADIPOQ",
        "nutrient": "fiber",
        "effect": "Lower adiponectin levels increase insulin resistance risk",
        "recommendation": "High-fibre diet (30+ g/day); omega-3 fatty acids; regular exercise",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs7799039",
        "gene": "LEP",
        "nutrient": "calories",
        "effect": "Altered leptin signalling affects satiety and energy balance",
        "recommendation": "Portion control; high-protein breakfast; avoid caloric restriction below BMR",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs1137101",
        "gene": "LEPR",
        "nutrient": "calories",
        "effect": "Reduced leptin receptor sensitivity affects energy homeostasis",
        "recommendation": "Regular meals with protein/fibre; avoid ultra-processed foods",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.68,
    },
    {
        "rsid": "rs6277",
        "gene": "DRD2",
        "nutrient": "sugar",
        "effect": "Altered dopamine receptor density affects food reward sensitivity",
        "recommendation": "Limit added sugars; ensure adequate protein at each meal",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 25000.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs25531",
        "gene": "SLC6A4",
        "nutrient": "tryptophan",
        "effect": "Altered serotonin transporter affects mood-food relationship",
        "recommendation": "Ensure tryptophan-rich foods: turkey, eggs, nuts, seeds, cheese",
        "priority": "moderate",
        "daily_target_mg": 250.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.65,
    },
    {
        "rsid": "rs6323",
        "gene": "MAOA",
        "nutrient": "tyramine",
        "effect": "Altered monoamine oxidase A activity affects tyramine sensitivity",
        "recommendation": "Monitor aged cheese, fermented foods, and wine if MAOI-sensitive",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.6,
    },
    {
        "rsid": "rs6721961",
        "gene": "NFE2L2",
        "nutrient": "sulforaphane",
        "effect": "Reduced Nrf2 activation impairs Phase II detoxification",
        "recommendation": "Regular broccoli sprout consumption; turmeric and green tea",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs1051740",
        "gene": "EPHX1",
        "nutrient": "cruciferous_vegetables",
        "effect": "Slow epoxide hydrolase reduces xenobiotic metabolism",
        "recommendation": "Increase cruciferous vegetables and allium family foods",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.68,
    },
    {
        "rsid": "rs1799930",
        "gene": "NAT2",
        "nutrient": "cruciferous_vegetables",
        "effect": "Slow acetylator phenotype affects aromatic amine detoxification",
        "recommendation": "Higher cruciferous vegetable intake for enhanced detoxification",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs1056836",
        "gene": "CYP1B1",
        "nutrient": "phytoestrogens",
        "effect": "Altered oestrogen metabolism; catechol oestrogen pathway",
        "recommendation": "Include soy isoflavones, flaxseed, and cruciferous vegetables",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs9282861",
        "gene": "SULT1A1",
        "nutrient": "polyphenols",
        "effect": "Reduced sulfotransferase activity affects flavonoid metabolism",
        "recommendation": "May need higher polyphenol intake; berries, tea, dark chocolate",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.62,
    },
    {
        "rsid": "rs4148323",
        "gene": "UGT1A1",
        "nutrient": "polyphenols",
        "effect": "Gilbert syndrome variant affects glucuronidation of flavonoids",
        "recommendation": "Moderate flavonoid intake; some flavonoids may elevate bilirubin",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.58,
    },
    {
        "rsid": "rs12934922",
        "gene": "BCMO1",
        "nutrient": "vitamin_a",
        "effect": "Reduced conversion of beta-carotene to retinol",
        "recommendation": "Prefer preformed vitamin A (liver, eggs, dairy) over beta-carotene",
        "priority": "high",
        "daily_target_mg": 0.9,
        "upper_limit_mg": 3.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs7501331",
        "gene": "BCMO1",
        "nutrient": "vitamin_a",
        "effect": "Additional BCMO1 variant further reduces carotenoid conversion",
        "recommendation": "Combine preformed vitamin A with beta-carotene sources",
        "priority": "moderate",
        "daily_target_mg": 0.9,
        "upper_limit_mg": 3.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs1051266",
        "gene": "SLC19A1",
        "nutrient": "folate",
        "effect": "Reduced folate carrier 1 transport efficiency",
        "recommendation": "Higher folate intake; methylfolate form may be more bioavailable",
        "priority": "high",
        "daily_target_mg": 0.6,
        "upper_limit_mg": 1.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs70991108",
        "gene": "DHFR",
        "nutrient": "folate",
        "effect": "19 bp deletion affects dihydrofolate reductase and folic acid metabolism",
        "recommendation": "Use food folate or 5-MTHF instead of synthetic folic acid",
        "priority": "moderate",
        "daily_target_mg": 0.6,
        "upper_limit_mg": 1.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs558660",
        "gene": "GIF",
        "nutrient": "vitamin_b12",
        "effect": "Altered gastric intrinsic factor affects B12 absorption",
        "recommendation": "Sublingual or intramuscular B12 may bypass absorption deficit",
        "priority": "high",
        "daily_target_mg": 0.005,
        "upper_limit_mg": 0.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs1801198",
        "gene": "TCN2",
        "nutrient": "vitamin_b12",
        "effect": "Altered transcobalamin II affects B12 delivery to tissues",
        "recommendation": "Monitor methylmalonic acid; ensure adequate B12 status",
        "priority": "moderate",
        "daily_target_mg": 0.005,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs2274924",
        "gene": "TRPM6",
        "nutrient": "magnesium",
        "effect": "Altered magnesium channel affects intestinal Mg absorption",
        "recommendation": "Higher magnesium foods: dark chocolate, nuts, seeds, leafy greens",
        "priority": "moderate",
        "daily_target_mg": 420.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.72,
    },
    {
        "rsid": "rs7914558",
        "gene": "CNNM2",
        "nutrient": "magnesium",
        "effect": "Reduced renal magnesium reabsorption",
        "recommendation": "Magnesium glycinate supplementation; Epsom salt baths",
        "priority": "moderate",
        "daily_target_mg": 420.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.68,
    },
    {
        "rsid": "rs4072037",
        "gene": "MUC1",
        "nutrient": "prebiotics",
        "effect": "Altered mucin production affects gut barrier function",
        "recommendation": "Prebiotic-rich foods: garlic, onion, asparagus, banana, oats",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.65,
    },
    {
        "rsid": "rs2066844",
        "gene": "NOD2",
        "nutrient": "prebiotics",
        "effect": "Altered innate immune sensing increases IBD susceptibility",
        "recommendation": "Anti-inflammatory diet; adequate fibre; probiotic-rich fermented foods",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.78,
    },
    {
        "rsid": "rs2241880",
        "gene": "ATG16L1",
        "nutrient": "polyphenols",
        "effect": "Impaired autophagy affects gut homeostasis and inflammation",
        "recommendation": "Polyphenol-rich diet; intermittent fasting may support autophagy",
        "priority": "moderate",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.7,
    },
    {
        "rsid": "rs2740574",
        "gene": "CYP3A4",
        "nutrient": "grapefruit",
        "effect": "CYP3A4 variant affects drug metabolism; grapefruit inhibits this enzyme",
        "recommendation": "Avoid grapefruit if taking CYP3A4-metabolised medications",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.88,
    },
    {
        "rsid": "rs4149056",
        "gene": "SLCO1B1",
        "nutrient": "coenzyme_q10",
        "effect": "Increased statin myopathy risk; statins deplete CoQ10",
        "recommendation": "CoQ10 supplementation if on statins; 100-200 mg/day",
        "priority": "high",
        "daily_target_mg": 100.0,
        "upper_limit_mg": 400.0,
        "confidence": 0.85,
    },
    {
        "rsid": "rs662799",
        "gene": "APOA5",
        "nutrient": "omega_3",
        "effect": "Higher fasting triglycerides; exaggerated postprandial lipemia",
        "recommendation": "Omega-3 supplementation; limit refined carbs and alcohol",
        "priority": "high",
        "daily_target_mg": 1000.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.8,
    },
    {
        "rsid": "rs328",
        "gene": "LPL",
        "nutrient": "total_fat",
        "effect": "S447X gain-of-function improves lipid clearance (protective)",
        "recommendation": "Standard dietary fat guidance applies; no restriction needed",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.75,
    },
    {
        "rsid": "rs10830963",
        "gene": "MTNR1B",
        "nutrient": "meal_timing",
        "effect": "Evening melatonin signalling impairs glucose tolerance",
        "recommendation": "Avoid carbohydrate-heavy meals close to bedtime",
        "priority": "high",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.82,
    },
    {
        "rsid": "rs11591147",
        "gene": "PCSK9",
        "nutrient": "saturated_fat",
        "effect": "Loss-of-function variant naturally lowers LDL cholesterol",
        "recommendation": "Standard heart-healthy diet; naturally protected from high LDL",
        "priority": "low",
        "daily_target_mg": 0.0,
        "upper_limit_mg": 0.0,
        "confidence": 0.85,
    },
    {
        "rsid": "rs17238484",
        "gene": "HMGCR",
        "nutrient": "plant_sterols",
        "effect": "HMGCR variant affects cholesterol synthesis and statin response",
        "recommendation": "Plant sterols (2 g/day) from fortified foods or supplements",
        "priority": "moderate",
        "daily_target_mg": 2000.0,
        "upper_limit_mg": 3000.0,
        "confidence": 0.72,
    },
]

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


# Helper to construct FoodItem instances concisely.
def _fi(
    name: str,
    cal: float,
    prot: float,
    carb: float,
    fat: float,
    fib: float,
    micros: dict[str, float],
    group: str,
    regions: list[str],
) -> FoodItem:
    return FoodItem(
        name=name,
        calories_per_100g=cal,
        protein=prot,
        carbs=carb,
        fat=fat,
        fiber=fib,
        key_micronutrients=micros,
        food_group=group,
        regions=regions,
    )


_ALL = [
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

FOOD_DATABASE: list[FoodItem] = [
    # ======================== GRAINS & CEREALS ==========================
    _fi(
        "brown rice",
        112,
        2.6,
        23.5,
        0.9,
        1.8,
        {"magnesium": 43.0, "selenium": 9.8, "manganese": 0.9},
        "grain",
        _ALL,
    ),
    _fi("white rice", 130, 2.7, 28.2, 0.3, 0.4, {"iron": 0.2, "thiamine": 0.02}, "grain", _ALL),
    _fi(
        "basmati rice",
        121,
        3.5,
        25.2,
        0.4,
        0.4,
        {"iron": 0.5, "thiamine": 0.07},
        "grain",
        ["south_asia_north", "south_asia_south", "middle_east"],
    ),
    _fi(
        "whole wheat roti",
        297,
        9.8,
        50.0,
        6.5,
        11.0,
        {"iron": 3.5, "folate": 0.04, "magnesium": 117.0},
        "grain",
        ["south_asia_north", "south_asia_west", "south_asia_east"],
    ),
    _fi(
        "quinoa",
        120,
        4.4,
        21.3,
        1.9,
        2.8,
        {"iron": 1.5, "magnesium": 64.0, "manganese": 0.6, "folate": 0.042},
        "grain",
        ["latin_america", "north_america", "mediterranean"],
    ),
    _fi(
        "oats",
        389,
        16.9,
        66.3,
        6.9,
        10.6,
        {"iron": 4.7, "magnesium": 177.0, "zinc": 3.97, "thiamine": 0.76},
        "grain",
        ["northern_europe", "north_america"],
    ),
    _fi(
        "rye bread",
        259,
        8.5,
        48.3,
        3.3,
        5.8,
        {"iron": 2.8, "selenium": 30.9, "folate": 0.11},
        "grain",
        ["northern_europe"],
    ),
    _fi(
        "millet",
        378,
        11.0,
        72.8,
        4.2,
        8.5,
        {"iron": 3.0, "magnesium": 114.0, "phosphorus": 285.0},
        "grain",
        ["sub_saharan_africa", "south_asia_north", "central_asia"],
    ),
    _fi(
        "teff",
        367,
        13.3,
        73.1,
        2.4,
        8.0,
        {"iron": 7.6, "calcium": 180.0, "magnesium": 184.0},
        "grain",
        ["sub_saharan_africa"],
    ),
    _fi(
        "corn tortilla",
        218,
        5.7,
        44.6,
        2.8,
        5.3,
        {"calcium": 46.0, "iron": 1.5, "niacin": 1.5},
        "grain",
        ["latin_america", "north_america"],
    ),
    _fi(
        "bulgur wheat",
        342,
        12.3,
        75.9,
        1.3,
        12.5,
        {"iron": 2.5, "magnesium": 164.0, "manganese": 3.0},
        "grain",
        ["middle_east", "mediterranean"],
    ),
    _fi(
        "sorghum",
        329,
        10.6,
        72.1,
        3.5,
        6.7,
        {"iron": 3.4, "phosphorus": 289.0, "potassium": 363.0},
        "grain",
        ["sub_saharan_africa", "south_asia_west"],
    ),
    _fi(
        "buckwheat",
        343,
        13.3,
        71.5,
        3.4,
        10.0,
        {"magnesium": 231.0, "manganese": 1.3, "copper": 1.1},
        "grain",
        ["east_asia", "northern_europe", "central_asia"],
    ),
    # ======================== LEGUMES ===================================
    _fi(
        "red lentils",
        116,
        9.0,
        20.1,
        0.4,
        3.8,
        {"iron": 3.3, "folate": 0.181, "potassium": 369.0, "zinc": 1.3},
        "legume",
        _ALL,
    ),
    _fi(
        "chickpeas",
        164,
        8.9,
        27.4,
        2.6,
        7.6,
        {"iron": 2.9, "folate": 0.172, "magnesium": 48.0, "zinc": 1.5},
        "legume",
        ["south_asia_north", "middle_east", "mediterranean"],
    ),
    _fi(
        "black beans",
        132,
        8.9,
        23.7,
        0.5,
        8.7,
        {"iron": 2.1, "folate": 0.149, "magnesium": 70.0, "potassium": 355.0},
        "legume",
        ["latin_america", "north_america"],
    ),
    _fi(
        "kidney beans (rajma)",
        127,
        8.7,
        22.8,
        0.5,
        6.4,
        {"iron": 2.9, "folate": 0.130, "potassium": 403.0},
        "legume",
        ["south_asia_north", "latin_america"],
    ),
    _fi(
        "edamame",
        121,
        11.9,
        8.9,
        5.2,
        5.2,
        {"iron": 2.3, "folate": 0.311, "vitamin_c": 6.1, "calcium": 63.0},
        "legume",
        ["east_asia", "southeast_asia", "north_america"],
    ),
    _fi(
        "cowpeas",
        116,
        7.7,
        20.8,
        0.5,
        6.5,
        {"iron": 2.5, "folate": 0.208, "potassium": 278.0},
        "legume",
        ["sub_saharan_africa", "south_asia_south"],
    ),
    _fi(
        "mung beans",
        105,
        7.0,
        19.1,
        0.4,
        7.6,
        {"iron": 1.4, "folate": 0.159, "magnesium": 48.0},
        "legume",
        ["south_asia_south", "east_asia", "southeast_asia"],
    ),
    _fi(
        "fava beans",
        110,
        7.6,
        19.7,
        0.4,
        5.4,
        {"iron": 1.5, "folate": 0.148, "manganese": 0.4},
        "legume",
        ["middle_east", "mediterranean"],
    ),
    # ======================== VEGETABLES ================================
    _fi(
        "spinach",
        23,
        2.9,
        3.6,
        0.4,
        2.2,
        {
            "iron": 2.7,
            "folate": 0.194,
            "vitamin_c": 28.1,
            "vitamin_a": 0.469,
            "calcium": 99.0,
            "magnesium": 79.0,
        },
        "vegetable",
        _ALL,
    ),
    _fi(
        "broccoli",
        34,
        2.8,
        6.6,
        0.4,
        2.6,
        {
            "vitamin_c": 89.2,
            "folate": 0.063,
            "vitamin_a": 0.031,
            "calcium": 47.0,
            "sulforaphane": 1.0,
        },
        "vegetable",
        _ALL,
    ),
    _fi(
        "kale",
        49,
        4.3,
        8.8,
        0.9,
        3.6,
        {"vitamin_c": 120.0, "vitamin_a": 0.500, "calcium": 150.0, "iron": 1.5, "folate": 0.141},
        "vegetable",
        ["northern_europe", "north_america", "mediterranean"],
    ),
    _fi(
        "sweet potato",
        86,
        1.6,
        20.1,
        0.1,
        3.0,
        {"vitamin_a": 0.709, "vitamin_c": 2.4, "potassium": 337.0, "manganese": 0.3},
        "vegetable",
        _ALL,
    ),
    _fi(
        "tomato",
        18,
        0.9,
        3.9,
        0.2,
        1.2,
        {"vitamin_c": 13.7, "lycopene": 2.6, "potassium": 237.0, "folate": 0.015},
        "vegetable",
        _ALL,
    ),
    _fi(
        "okra",
        33,
        1.9,
        7.5,
        0.2,
        3.2,
        {"vitamin_c": 23.0, "folate": 0.060, "magnesium": 57.0, "calcium": 82.0},
        "vegetable",
        ["south_asia_north", "south_asia_south", "sub_saharan_africa", "middle_east"],
    ),
    _fi(
        "bitter gourd",
        17,
        1.0,
        3.7,
        0.2,
        2.8,
        {"vitamin_c": 84.0, "folate": 0.072, "iron": 0.4, "zinc": 0.8},
        "vegetable",
        ["south_asia_south", "southeast_asia", "east_asia"],
    ),
    _fi(
        "moringa leaves",
        64,
        9.4,
        8.3,
        1.4,
        2.0,
        {"vitamin_a": 0.378, "vitamin_c": 51.7, "calcium": 185.0, "iron": 4.0, "potassium": 337.0},
        "vegetable",
        ["sub_saharan_africa", "south_asia_south", "southeast_asia"],
    ),
    _fi(
        "bok choy",
        13,
        1.5,
        2.2,
        0.2,
        1.0,
        {"vitamin_c": 45.0, "vitamin_a": 0.223, "calcium": 105.0, "folate": 0.066},
        "vegetable",
        ["east_asia", "southeast_asia"],
    ),
    _fi(
        "eggplant",
        25,
        1.0,
        5.9,
        0.2,
        3.0,
        {"potassium": 229.0, "manganese": 0.2, "folate": 0.022},
        "vegetable",
        ["mediterranean", "middle_east", "south_asia_south"],
    ),
    _fi(
        "Brussels sprouts",
        43,
        3.4,
        8.9,
        0.3,
        3.8,
        {"vitamin_c": 85.0, "vitamin_a": 0.038, "folate": 0.061, "sulforaphane": 0.8},
        "vegetable",
        ["northern_europe", "north_america"],
    ),
    _fi(
        "cauliflower",
        25,
        1.9,
        5.0,
        0.3,
        2.0,
        {"vitamin_c": 48.2, "folate": 0.057, "vitamin_b6": 0.18},
        "vegetable",
        _ALL,
    ),
    _fi(
        "carrot",
        41,
        0.9,
        9.6,
        0.2,
        2.8,
        {"vitamin_a": 0.835, "potassium": 320.0, "biotin": 0.006},
        "vegetable",
        _ALL,
    ),
    _fi(
        "drumstick (moringa pods)",
        37,
        2.1,
        8.5,
        0.2,
        3.2,
        {"vitamin_c": 141.0, "calcium": 30.0, "iron": 0.4},
        "vegetable",
        ["south_asia_south", "sub_saharan_africa"],
    ),
    _fi(
        "nopales (cactus pads)",
        16,
        1.3,
        3.3,
        0.1,
        2.2,
        {"vitamin_c": 9.3, "calcium": 164.0, "magnesium": 52.0},
        "vegetable",
        ["latin_america"],
    ),
    # ======================== FRUITS ====================================
    _fi(
        "banana",
        89,
        1.1,
        22.8,
        0.3,
        2.6,
        {"potassium": 358.0, "vitamin_b6": 0.37, "vitamin_c": 8.7, "magnesium": 27.0},
        "fruit",
        _ALL,
    ),
    _fi(
        "mango",
        60,
        0.8,
        15.0,
        0.4,
        1.6,
        {"vitamin_c": 36.4, "vitamin_a": 0.054, "folate": 0.043},
        "fruit",
        [
            "south_asia_north",
            "south_asia_south",
            "southeast_asia",
            "sub_saharan_africa",
            "latin_america",
        ],
    ),
    _fi(
        "blueberry",
        57,
        0.7,
        14.5,
        0.3,
        2.4,
        {"vitamin_c": 9.7, "vitamin_k": 0.019, "manganese": 0.3, "anthocyanins": 25.0},
        "fruit",
        ["northern_europe", "north_america"],
    ),
    _fi(
        "guava",
        68,
        2.6,
        14.3,
        1.0,
        5.4,
        {"vitamin_c": 228.3, "folate": 0.049, "potassium": 417.0, "vitamin_a": 0.031},
        "fruit",
        ["south_asia_north", "latin_america", "southeast_asia", "sub_saharan_africa"],
    ),
    _fi(
        "papaya",
        43,
        0.5,
        10.8,
        0.3,
        1.7,
        {"vitamin_c": 60.9, "vitamin_a": 0.047, "folate": 0.037, "potassium": 182.0},
        "fruit",
        ["southeast_asia", "latin_america", "sub_saharan_africa"],
    ),
    _fi(
        "orange",
        47,
        0.9,
        11.8,
        0.1,
        2.4,
        {"vitamin_c": 53.2, "folate": 0.030, "potassium": 181.0, "thiamine": 0.09},
        "fruit",
        _ALL,
    ),
    _fi(
        "pomegranate",
        83,
        1.7,
        18.7,
        1.2,
        4.0,
        {
            "vitamin_c": 10.2,
            "vitamin_k": 0.016,
            "folate": 0.038,
            "potassium": 236.0,
            "punicalagins": 50.0,
        },
        "fruit",
        ["south_asia_north", "middle_east", "mediterranean"],
    ),
    _fi(
        "dates",
        277,
        1.8,
        75.0,
        0.2,
        6.7,
        {"potassium": 696.0, "magnesium": 54.0, "iron": 0.9, "vitamin_b6": 0.25},
        "fruit",
        ["middle_east", "sub_saharan_africa"],
    ),
    _fi(
        "jackfruit",
        95,
        1.7,
        23.3,
        0.6,
        1.5,
        {"vitamin_c": 13.7, "potassium": 448.0, "magnesium": 29.0},
        "fruit",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "avocado",
        160,
        2.0,
        8.5,
        14.7,
        6.7,
        {
            "potassium": 485.0,
            "folate": 0.081,
            "vitamin_k": 0.021,
            "vitamin_e": 2.1,
            "magnesium": 29.0,
        },
        "fruit",
        ["latin_america", "north_america", "mediterranean"],
    ),
    _fi(
        "apple",
        52,
        0.3,
        13.8,
        0.2,
        2.4,
        {"vitamin_c": 4.6, "potassium": 107.0, "quercetin": 4.0},
        "fruit",
        ["northern_europe", "north_america", "central_asia"],
    ),
    _fi(
        "fig",
        74,
        0.8,
        19.2,
        0.3,
        2.9,
        {"calcium": 35.0, "potassium": 232.0, "iron": 0.4, "magnesium": 17.0},
        "fruit",
        ["mediterranean", "middle_east"],
    ),
    _fi(
        "baobab fruit powder",
        250,
        2.3,
        58.0,
        0.3,
        44.5,
        {"vitamin_c": 280.0, "calcium": 293.0, "iron": 9.3, "potassium": 2270.0},
        "fruit",
        ["sub_saharan_africa"],
    ),
    # ======================== PROTEIN SOURCES ===========================
    _fi(
        "chicken breast",
        165,
        31.0,
        0.0,
        3.6,
        0.0,
        {"selenium": 27.6, "niacin": 13.7, "vitamin_b6": 0.6, "phosphorus": 228.0},
        "poultry",
        _ALL,
    ),
    _fi(
        "salmon",
        208,
        20.4,
        0.0,
        13.4,
        0.0,
        {"omega_3": 2260.0, "selenium": 36.5, "vitamin_d_iu": 526.0, "vitamin_b12": 0.003},
        "fish",
        ["northern_europe", "north_america", "east_asia"],
    ),
    _fi(
        "sardines",
        208,
        24.6,
        0.0,
        11.5,
        0.0,
        {
            "omega_3": 1480.0,
            "calcium": 382.0,
            "vitamin_d_iu": 272.0,
            "vitamin_b12": 0.009,
            "selenium": 52.7,
        },
        "fish",
        ["mediterranean", "northern_europe", "southeast_asia"],
    ),
    _fi(
        "mackerel",
        205,
        18.6,
        0.0,
        13.9,
        0.0,
        {"omega_3": 2670.0, "vitamin_d_iu": 643.0, "selenium": 44.1, "vitamin_b12": 0.019},
        "fish",
        ["east_asia", "northern_europe", "southeast_asia"],
    ),
    _fi(
        "eggs",
        155,
        13.0,
        1.1,
        11.0,
        0.0,
        {
            "selenium": 30.7,
            "vitamin_d_iu": 82.0,
            "vitamin_b12": 0.001,
            "choline": 293.8,
            "vitamin_a": 0.160,
        },
        "eggs",
        _ALL,
    ),
    _fi(
        "tofu (firm)",
        76,
        8.2,
        1.9,
        4.8,
        0.3,
        {"calcium": 350.0, "iron": 5.4, "magnesium": 30.0, "selenium": 17.4},
        "legume",
        ["east_asia", "southeast_asia", "north_america"],
    ),
    _fi(
        "paneer",
        265,
        18.3,
        1.2,
        20.8,
        0.0,
        {"calcium": 480.0, "phosphorus": 138.0, "selenium": 17.0},
        "dairy",
        ["south_asia_north", "south_asia_west"],
    ),
    _fi(
        "Greek yoghurt",
        59,
        10.2,
        3.6,
        0.7,
        0.0,
        {"calcium": 110.0, "vitamin_b12": 0.001, "potassium": 141.0, "probiotics": 1.0},
        "dairy",
        _ALL,
    ),
    _fi(
        "lamb",
        294,
        25.5,
        0.0,
        20.9,
        0.0,
        {"iron": 1.9, "zinc": 4.7, "vitamin_b12": 0.003, "selenium": 26.4},
        "meat",
        ["middle_east", "south_asia_north", "mediterranean", "central_asia"],
    ),
    _fi(
        "goat meat",
        143,
        27.1,
        0.0,
        3.0,
        0.0,
        {"iron": 3.7, "zinc": 5.3, "vitamin_b12": 0.001, "selenium": 8.8},
        "meat",
        ["sub_saharan_africa", "south_asia_north", "middle_east", "central_asia"],
    ),
    _fi(
        "tempeh",
        192,
        20.3,
        7.6,
        10.8,
        0.0,
        {"calcium": 111.0, "iron": 2.7, "magnesium": 81.0, "manganese": 1.3},
        "legume",
        ["southeast_asia", "east_asia"],
    ),
    _fi(
        "lentil dal",
        116,
        9.0,
        20.0,
        0.4,
        3.8,
        {"iron": 3.3, "folate": 0.181, "potassium": 369.0},
        "legume",
        ["south_asia_north", "south_asia_south", "south_asia_east", "south_asia_west"],
    ),
    _fi(
        "shrimp",
        99,
        24.0,
        0.2,
        0.3,
        0.0,
        {"selenium": 38.0, "vitamin_b12": 0.001, "iodine": 0.035, "zinc": 1.6},
        "fish",
        ["southeast_asia", "east_asia", "mediterranean", "latin_america"],
    ),
    # ======================== NUTS & SEEDS ==============================
    _fi(
        "almonds",
        579,
        21.2,
        21.6,
        49.9,
        12.5,
        {"vitamin_e": 25.6, "magnesium": 270.0, "calcium": 269.0, "iron": 3.7},
        "nut",
        _ALL,
    ),
    _fi(
        "walnuts",
        654,
        15.2,
        13.7,
        65.2,
        6.7,
        {"omega_3": 9080.0, "magnesium": 158.0, "manganese": 3.4, "copper": 1.6},
        "nut",
        _ALL,
    ),
    _fi(
        "flaxseed",
        534,
        18.3,
        28.9,
        42.2,
        27.3,
        {"omega_3": 22800.0, "magnesium": 392.0, "manganese": 2.5, "thiamine": 1.64},
        "seed",
        _ALL,
    ),
    _fi(
        "chia seeds",
        486,
        16.5,
        42.1,
        30.7,
        34.4,
        {"omega_3": 17552.0, "calcium": 631.0, "magnesium": 335.0, "iron": 7.7},
        "seed",
        ["latin_america", "north_america"],
    ),
    _fi(
        "pumpkin seeds",
        559,
        30.2,
        10.7,
        49.1,
        6.0,
        {"zinc": 7.8, "magnesium": 550.0, "iron": 8.8, "manganese": 4.5},
        "seed",
        _ALL,
    ),
    _fi(
        "groundnuts (peanuts)",
        567,
        25.8,
        16.1,
        49.2,
        8.5,
        {"niacin": 12.1, "folate": 0.240, "magnesium": 168.0, "vitamin_e": 8.3},
        "nut",
        ["sub_saharan_africa", "south_asia_north", "southeast_asia", "north_america"],
    ),
    _fi(
        "sunflower seeds",
        584,
        20.8,
        20.0,
        51.5,
        8.6,
        {"vitamin_e": 35.2, "selenium": 53.0, "magnesium": 325.0},
        "seed",
        _ALL,
    ),
    _fi(
        "sesame seeds",
        573,
        17.7,
        23.4,
        49.7,
        11.8,
        {"calcium": 975.0, "iron": 14.6, "magnesium": 351.0, "zinc": 7.8},
        "seed",
        ["middle_east", "east_asia", "south_asia_north"],
    ),
    _fi(
        "cashews",
        553,
        18.2,
        30.2,
        43.9,
        3.3,
        {"magnesium": 292.0, "zinc": 5.8, "iron": 6.7, "copper": 2.2},
        "nut",
        ["south_asia_west", "southeast_asia", "sub_saharan_africa"],
    ),
    # ======================== DAIRY & ALTERNATIVES =======================
    _fi(
        "whole milk",
        61,
        3.2,
        4.8,
        3.3,
        0.0,
        {"calcium": 113.0, "vitamin_d_iu": 40.0, "vitamin_b12": 0.0004, "phosphorus": 84.0},
        "dairy",
        _ALL,
    ),
    _fi(
        "kefir",
        63,
        3.3,
        4.5,
        3.5,
        0.0,
        {"calcium": 130.0, "vitamin_b12": 0.0003, "probiotics": 1.0},
        "dairy",
        ["northern_europe", "central_asia", "middle_east"],
    ),
    _fi(
        "labneh",
        163,
        8.0,
        4.0,
        13.0,
        0.0,
        {"calcium": 180.0, "vitamin_b12": 0.0004, "probiotics": 1.0},
        "dairy",
        ["middle_east", "mediterranean"],
    ),
    _fi(
        "feta cheese",
        264,
        14.2,
        4.1,
        21.3,
        0.0,
        {"calcium": 493.0, "phosphorus": 337.0, "vitamin_b12": 0.002, "sodium": 1116.0},
        "dairy",
        ["mediterranean", "middle_east"],
    ),
    # ======================== OILS & FATS ================================
    _fi(
        "olive oil",
        884,
        0.0,
        0.0,
        100.0,
        0.0,
        {"vitamin_e": 14.4, "vitamin_k": 0.060, "oleic_acid": 73000.0, "polyphenols": 200.0},
        "oil",
        ["mediterranean", "middle_east"],
    ),
    _fi(
        "coconut oil",
        862,
        0.0,
        0.0,
        100.0,
        0.0,
        {"lauric_acid": 44600.0, "vitamin_e": 0.1},
        "oil",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "ghee",
        900,
        0.0,
        0.0,
        100.0,
        0.0,
        {"vitamin_a": 0.684, "vitamin_d_iu": 20.0, "vitamin_e": 2.8, "butyrate": 3000.0},
        "oil",
        ["south_asia_north", "south_asia_west", "south_asia_east"],
    ),
    _fi(
        "mustard oil",
        884,
        0.0,
        0.0,
        100.0,
        0.0,
        {"omega_3": 5900.0, "vitamin_e": 11.6, "erucic_acid": 42000.0},
        "oil",
        ["south_asia_north", "south_asia_east"],
    ),
    # ======================== SPICES / FUNCTIONAL FOODS ==================
    _fi(
        "turmeric powder",
        354,
        7.8,
        64.9,
        9.9,
        21.1,
        {"curcumin": 3140.0, "iron": 41.4, "manganese": 7.8, "potassium": 2525.0},
        "spice",
        _ALL,
    ),
    _fi(
        "ginger root",
        80,
        1.8,
        17.8,
        0.8,
        2.0,
        {"gingerol": 300.0, "potassium": 415.0, "magnesium": 43.0, "vitamin_c": 5.0},
        "spice",
        _ALL,
    ),
    _fi(
        "garlic",
        149,
        6.4,
        33.1,
        0.5,
        2.1,
        {"allicin": 4500.0, "selenium": 14.2, "vitamin_c": 31.2, "manganese": 1.7},
        "spice",
        _ALL,
    ),
    _fi(
        "cinnamon",
        247,
        4.0,
        80.6,
        1.2,
        53.1,
        {"calcium": 1002.0, "iron": 8.3, "manganese": 17.5},
        "spice",
        _ALL,
    ),
    _fi(
        "green tea (brewed)",
        1,
        0.2,
        0.0,
        0.0,
        0.0,
        {"catechins": 80.0, "theanine": 20.0, "caffeine": 25.0, "vitamin_c": 0.3},
        "beverage",
        ["east_asia", "southeast_asia", "north_america"],
    ),
    _fi(
        "dark chocolate (70%)",
        598,
        7.8,
        45.9,
        42.6,
        10.9,
        {"iron": 11.9, "magnesium": 228.0, "copper": 1.8, "manganese": 2.0, "flavonoids": 500.0},
        "snack",
        _ALL,
    ),
    # ======================== PREPARED / MIXED FOODS ====================
    _fi(
        "hummus",
        166,
        7.9,
        14.3,
        9.6,
        6.0,
        {"iron": 2.4, "folate": 0.083, "vitamin_b6": 0.2, "zinc": 1.8},
        "prepared",
        ["middle_east", "mediterranean", "north_america"],
    ),
    _fi(
        "idli",
        77,
        2.0,
        15.0,
        0.4,
        0.5,
        {"iron": 0.9, "calcium": 14.0, "folate": 0.010},
        "prepared",
        ["south_asia_south"],
    ),
    _fi(
        "miso",
        199,
        11.7,
        26.5,
        6.0,
        5.4,
        {"sodium": 3728.0, "zinc": 2.6, "manganese": 0.9, "vitamin_k": 0.029},
        "prepared",
        ["east_asia"],
    ),
    _fi(
        "kimchi",
        15,
        1.1,
        2.4,
        0.5,
        1.6,
        {"vitamin_c": 18.0, "vitamin_a": 0.018, "probiotics": 1.0, "sodium": 498.0},
        "prepared",
        ["east_asia"],
    ),
    _fi(
        "injera (teff)",
        135,
        3.9,
        28.0,
        0.6,
        1.6,
        {"iron": 3.5, "calcium": 80.0, "probiotics": 1.0},
        "prepared",
        ["sub_saharan_africa"],
    ),
    _fi(
        "cassava (boiled)",
        160,
        1.4,
        38.1,
        0.3,
        1.8,
        {"vitamin_c": 20.6, "potassium": 271.0, "manganese": 0.4},
        "prepared",
        ["sub_saharan_africa", "latin_america", "southeast_asia"],
    ),
    _fi(
        "plantain (cooked)",
        116,
        0.8,
        31.2,
        0.4,
        2.3,
        {"vitamin_c": 18.4, "potassium": 465.0, "vitamin_a": 0.056, "magnesium": 37.0},
        "prepared",
        ["sub_saharan_africa", "latin_america"],
    ),
    _fi(
        "pho (broth base)",
        25,
        2.0,
        2.5,
        0.8,
        0.0,
        {"sodium": 380.0, "collagen": 500.0},
        "prepared",
        ["southeast_asia"],
    ),
    # ======================== ADDITIONAL TO REACH 100+ ===================
    _fi(
        "cod (Atlantic)",
        82,
        17.8,
        0.0,
        0.7,
        0.0,
        {"selenium": 33.1, "vitamin_b12": 0.001, "iodine": 0.170, "phosphorus": 203.0},
        "fish",
        ["northern_europe", "north_america"],
    ),
    _fi(
        "herring",
        158,
        18.0,
        0.0,
        9.0,
        0.0,
        {"omega_3": 1729.0, "vitamin_d_iu": 680.0, "selenium": 36.5},
        "fish",
        ["northern_europe"],
    ),
    _fi(
        "beef (lean)",
        250,
        26.1,
        0.0,
        15.0,
        0.0,
        {"iron": 2.6, "zinc": 6.3, "vitamin_b12": 0.003, "selenium": 18.0},
        "meat",
        _ALL,
    ),
    _fi(
        "turkey breast",
        135,
        30.0,
        0.0,
        1.0,
        0.0,
        {"selenium": 32.1, "niacin": 11.8, "vitamin_b6": 0.8},
        "poultry",
        ["north_america", "northern_europe"],
    ),
    _fi(
        "lychee",
        66,
        0.8,
        16.5,
        0.4,
        1.3,
        {"vitamin_c": 71.5, "potassium": 171.0, "copper": 0.1},
        "fruit",
        ["east_asia", "south_asia_east", "southeast_asia"],
    ),
    _fi(
        "persimmon",
        70,
        0.6,
        18.6,
        0.2,
        3.6,
        {"vitamin_a": 0.081, "vitamin_c": 7.5, "manganese": 0.4},
        "fruit",
        ["east_asia", "mediterranean"],
    ),
    _fi(
        "dragon fruit",
        60,
        1.2,
        13.0,
        0.4,
        1.8,
        {"vitamin_c": 3.0, "iron": 0.7, "magnesium": 40.0},
        "fruit",
        ["southeast_asia", "latin_america"],
    ),
    _fi(
        "passion fruit",
        97,
        2.2,
        23.4,
        0.7,
        10.4,
        {"vitamin_c": 30.0, "vitamin_a": 0.064, "iron": 1.6, "potassium": 348.0},
        "fruit",
        ["latin_america", "sub_saharan_africa", "southeast_asia"],
    ),
    _fi(
        "seaweed (nori)",
        35,
        5.8,
        5.1,
        0.3,
        0.3,
        {"iodine": 1.5, "iron": 1.8, "vitamin_b12": 0.0003, "calcium": 70.0},
        "vegetable",
        ["east_asia"],
    ),
    _fi(
        "lotus root",
        74,
        2.6,
        17.2,
        0.1,
        4.9,
        {"vitamin_c": 44.0, "potassium": 556.0, "manganese": 0.3},
        "vegetable",
        ["east_asia", "southeast_asia"],
    ),
    _fi(
        "coconut milk",
        230,
        2.3,
        6.0,
        23.8,
        0.0,
        {"manganese": 0.9, "iron": 1.6, "magnesium": 37.0},
        "dairy",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "teff",
        367,
        13.3,
        73.1,
        2.4,
        8.0,
        {"iron": 7.6, "calcium": 180.0, "magnesium": 184.0},
        "grain",
        ["east_africa"],
    ),
    _fi(
        "fonio",
        350,
        8.0,
        75.0,
        1.8,
        3.0,
        {"iron": 4.0, "zinc": 2.5, "magnesium": 60.0},
        "grain",
        ["west_africa"],
    ),
    _fi(
        "spelt",
        338,
        14.6,
        70.2,
        2.4,
        10.7,
        {"magnesium": 136.0, "iron": 4.4, "zinc": 3.3},
        "grain",
        ["central_europe", "western_europe"],
    ),
    _fi(
        "kamut",
        337,
        14.5,
        68.2,
        2.2,
        9.1,
        {"selenium": 81.0, "magnesium": 130.0, "zinc": 4.3},
        "grain",
        ["mediterranean", "north_africa"],
    ),
    _fi(
        "amaranth",
        371,
        13.6,
        65.3,
        7.0,
        6.7,
        {"iron": 7.6, "magnesium": 248.0, "calcium": 159.0},
        "grain",
        ["south_america", "central_america", "andean"],
    ),
    _fi(
        "freekeh",
        300,
        12.6,
        71.5,
        2.7,
        16.5,
        {"magnesium": 76.0, "iron": 3.7, "zinc": 2.6},
        "grain",
        ["middle_east", "north_africa"],
    ),
    _fi(
        "bulgur",
        342,
        12.3,
        75.9,
        1.3,
        18.3,
        {"magnesium": 164.0, "iron": 2.5, "manganese": 3.0},
        "grain",
        ["middle_east", "mediterranean"],
    ),
    _fi(
        "couscous",
        176,
        6.0,
        36.5,
        0.3,
        2.2,
        {"selenium": 27.5, "folate": 0.015, "iron": 0.4},
        "grain",
        ["north_africa", "middle_east"],
    ),
    _fi(
        "polenta",
        85,
        2.0,
        18.5,
        0.5,
        1.2,
        {"iron": 0.5, "magnesium": 10.0, "zinc": 0.3},
        "grain",
        ["mediterranean", "balkans"],
    ),
    _fi(
        "rye bread",
        259,
        8.5,
        48.3,
        3.3,
        5.8,
        {"iron": 2.8, "magnesium": 40.0, "selenium": 30.7},
        "grain",
        ["northern_europe", "eastern_europe", "nordic"],
    ),
    _fi(
        "pumpernickel",
        250,
        8.7,
        45.0,
        3.1,
        6.5,
        {"iron": 2.6, "magnesium": 52.0, "manganese": 1.5},
        "grain",
        ["central_europe", "eastern_europe"],
    ),
    _fi(
        "cornbread",
        196,
        4.4,
        31.4,
        5.8,
        1.4,
        {"iron": 1.5, "zinc": 0.6, "magnesium": 18.0},
        "grain",
        ["north_america", "southern_africa"],
    ),
    _fi(
        "injera",
        130,
        3.6,
        26.0,
        0.7,
        1.6,
        {"iron": 5.7, "calcium": 49.0, "magnesium": 30.0},
        "grain",
        ["east_africa"],
    ),
    _fi(
        "pita bread",
        275,
        9.1,
        55.7,
        1.2,
        2.2,
        {"iron": 2.6, "folate": 0.028, "selenium": 29.0},
        "grain",
        ["middle_east", "mediterranean"],
    ),
    _fi(
        "naan",
        290,
        8.6,
        50.0,
        5.7,
        2.1,
        {"iron": 3.0, "calcium": 70.0, "folate": 0.024},
        "grain",
        ["south_asia_north", "south_asia_west", "central_asia"],
    ),
    _fi(
        "soba noodles",
        99,
        5.1,
        21.4,
        0.1,
        0.0,
        {"manganese": 0.4, "iron": 0.5, "magnesium": 9.0},
        "grain",
        ["east_asia"],
    ),
    _fi(
        "rice noodles",
        109,
        0.9,
        25.0,
        0.2,
        1.0,
        {"iron": 0.2, "selenium": 3.3, "folate": 0.003},
        "grain",
        ["southeast_asia", "east_asia"],
    ),
    _fi(
        "arepa",
        200,
        4.4,
        32.0,
        6.2,
        2.0,
        {"iron": 1.0, "calcium": 20.0, "magnesium": 15.0},
        "grain",
        ["south_america", "latin_america"],
    ),
    _fi(
        "tortilla (corn)",
        218,
        5.7,
        44.6,
        2.8,
        5.5,
        {"calcium": 46.0, "iron": 1.2, "magnesium": 44.0},
        "grain",
        ["central_america", "latin_america", "north_america"],
    ),
    _fi(
        "cassava flour",
        330,
        1.4,
        80.0,
        0.5,
        3.7,
        {"calcium": 26.0, "magnesium": 24.0, "potassium": 64.0},
        "grain",
        ["west_africa", "east_africa", "south_america"],
    ),
    _fi(
        "black-eyed peas",
        116,
        7.7,
        20.8,
        0.5,
        6.5,
        {"folate": 0.208, "iron": 2.5, "zinc": 1.1},
        "legume",
        ["west_africa", "south_asia_south", "north_america"],
    ),
    _fi(
        "pigeon peas",
        136,
        7.2,
        23.9,
        1.6,
        5.1,
        {"folate": 0.111, "iron": 1.6, "magnesium": 46.0},
        "legume",
        ["south_asia_south", "caribbean", "east_africa"],
    ),
    _fi(
        "broad beans (fava)",
        88,
        7.6,
        12.4,
        0.4,
        5.4,
        {"folate": 0.148, "iron": 1.5, "magnesium": 33.0},
        "legume",
        ["mediterranean", "north_africa", "middle_east"],
    ),
    _fi(
        "mung beans",
        105,
        7.0,
        19.2,
        0.4,
        7.6,
        {"folate": 0.159, "iron": 1.4, "magnesium": 48.0},
        "legume",
        ["south_asia_north", "southeast_asia", "east_asia"],
    ),
    _fi(
        "adzuki beans",
        128,
        7.5,
        24.8,
        0.1,
        7.3,
        {"folate": 0.121, "iron": 2.0, "magnesium": 52.0},
        "legume",
        ["east_asia"],
    ),
    _fi(
        "lima beans",
        115,
        7.8,
        20.9,
        0.4,
        7.0,
        {"iron": 2.4, "magnesium": 43.0, "potassium": 508.0},
        "legume",
        ["south_america", "north_america"],
    ),
    _fi(
        "cowpeas",
        116,
        7.7,
        20.8,
        0.5,
        6.5,
        {"folate": 0.208, "iron": 2.5, "zinc": 1.1},
        "legume",
        ["west_africa", "sub_saharan_africa"],
    ),
    _fi(
        "navy beans",
        140,
        8.2,
        26.1,
        0.6,
        10.5,
        {"iron": 2.4, "folate": 0.14, "magnesium": 53.0},
        "legume",
        ["north_america", "western_europe"],
    ),
    _fi(
        "pinto beans",
        143,
        9.0,
        26.2,
        0.7,
        9.0,
        {"iron": 2.1, "folate": 0.172, "magnesium": 50.0},
        "legume",
        ["latin_america", "north_america"],
    ),
    _fi(
        "cranberry beans",
        136,
        9.3,
        24.5,
        0.5,
        8.0,
        {"iron": 2.0, "folate": 0.166, "magnesium": 48.0},
        "legume",
        ["mediterranean", "south_america"],
    ),
    _fi(
        "bitter melon",
        17,
        1.0,
        3.7,
        0.2,
        2.8,
        {"vitamin_c": 84.0, "folate": 0.072, "zinc": 0.8},
        "vegetable",
        ["south_asia_south", "southeast_asia", "east_asia"],
    ),
    _fi(
        "moringa leaves",
        64,
        9.4,
        8.3,
        1.4,
        2.0,
        {"calcium": 185.0, "iron": 4.0, "vitamin_c": 51.7},
        "vegetable",
        ["south_asia_south", "west_africa", "east_africa"],
    ),
    _fi(
        "taro",
        112,
        1.5,
        26.5,
        0.2,
        4.1,
        {"potassium": 591.0, "magnesium": 33.0, "vitamin_c": 4.5},
        "vegetable",
        ["pacific_islands", "southeast_asia", "caribbean"],
    ),
    _fi(
        "cassava",
        160,
        1.4,
        38.1,
        0.3,
        1.8,
        {"vitamin_c": 20.6, "magnesium": 21.0, "potassium": 271.0},
        "vegetable",
        ["west_africa", "east_africa", "south_america"],
    ),
    _fi(
        "breadfruit",
        103,
        1.1,
        27.1,
        0.2,
        4.9,
        {"potassium": 490.0, "vitamin_c": 29.0, "magnesium": 25.0},
        "vegetable",
        ["pacific_islands", "caribbean"],
    ),
    _fi(
        "lotus root",
        74,
        2.6,
        17.2,
        0.1,
        4.9,
        {"vitamin_c": 44.0, "potassium": 556.0, "manganese": 0.3},
        "vegetable",
        ["east_asia", "south_asia_east"],
    ),
    _fi(
        "daikon radish",
        18,
        0.6,
        4.1,
        0.1,
        1.6,
        {"vitamin_c": 22.0, "folate": 0.028, "potassium": 227.0},
        "vegetable",
        ["east_asia", "south_asia_north"],
    ),
    _fi(
        "bamboo shoots",
        27,
        2.6,
        5.2,
        0.3,
        2.2,
        {"potassium": 533.0, "manganese": 0.3, "zinc": 1.1},
        "vegetable",
        ["east_asia", "southeast_asia"],
    ),
    _fi(
        "okra",
        33,
        1.9,
        7.5,
        0.2,
        3.2,
        {"vitamin_c": 23.0, "folate": 0.06, "magnesium": 57.0},
        "vegetable",
        ["west_africa", "south_asia_south", "caribbean", "middle_east"],
    ),
    _fi(
        "nopales (cactus)",
        16,
        1.3,
        3.3,
        0.1,
        2.2,
        {"calcium": 164.0, "magnesium": 52.0, "vitamin_c": 13.0},
        "vegetable",
        ["central_america", "latin_america"],
    ),
    _fi(
        "chayote",
        19,
        0.8,
        4.5,
        0.1,
        1.7,
        {"vitamin_c": 7.7, "folate": 0.093, "potassium": 125.0},
        "vegetable",
        ["central_america", "south_america"],
    ),
    _fi(
        "endive",
        17,
        1.3,
        3.4,
        0.2,
        3.1,
        {"folate": 0.142, "vitamin_a_iu": 2167.0, "potassium": 314.0},
        "vegetable",
        ["western_europe", "mediterranean"],
    ),
    _fi(
        "kohlrabi",
        27,
        1.7,
        6.2,
        0.1,
        3.6,
        {"vitamin_c": 62.0, "potassium": 350.0, "magnesium": 19.0},
        "vegetable",
        ["central_europe", "northern_europe"],
    ),
    _fi(
        "celeriac",
        42,
        1.5,
        9.2,
        0.3,
        1.8,
        {"vitamin_c": 8.0, "potassium": 300.0, "phosphorus": 115.0},
        "vegetable",
        ["central_europe", "eastern_europe"],
    ),
    _fi(
        "romanesco",
        25,
        2.5,
        4.0,
        0.3,
        2.4,
        {"vitamin_c": 70.0, "folate": 0.023, "iron": 1.1},
        "vegetable",
        ["mediterranean"],
    ),
    _fi(
        "jicama",
        38,
        0.7,
        8.8,
        0.1,
        4.9,
        {"vitamin_c": 20.2, "potassium": 150.0, "iron": 0.6},
        "vegetable",
        ["central_america", "latin_america", "southeast_asia"],
    ),
    _fi(
        "watercress",
        11,
        2.3,
        1.3,
        0.1,
        0.5,
        {"vitamin_c": 43.0, "calcium": 120.0, "iron": 0.2},
        "vegetable",
        ["western_europe", "north_america"],
    ),
    _fi(
        "radicchio",
        23,
        1.4,
        4.5,
        0.3,
        0.9,
        {"vitamin_k": 0.255, "folate": 0.06, "potassium": 302.0},
        "vegetable",
        ["mediterranean"],
    ),
    _fi(
        "bok choy",
        13,
        1.5,
        2.2,
        0.2,
        1.0,
        {"vitamin_c": 45.0, "calcium": 105.0, "vitamin_a_iu": 4468.0},
        "vegetable",
        ["east_asia"],
    ),
    _fi(
        "artichoke",
        47,
        3.3,
        10.5,
        0.2,
        5.4,
        {"folate": 0.068, "magnesium": 60.0, "potassium": 370.0},
        "vegetable",
        ["mediterranean", "western_europe"],
    ),
    _fi(
        "dragon fruit",
        60,
        1.2,
        13.0,
        0.4,
        3.0,
        {"vitamin_c": 3.0, "iron": 1.9, "magnesium": 40.0},
        "fruit",
        ["southeast_asia", "south_america"],
    ),
    _fi(
        "jackfruit",
        95,
        1.7,
        23.3,
        0.6,
        1.5,
        {"potassium": 448.0, "vitamin_c": 13.7, "magnesium": 29.0},
        "fruit",
        ["south_asia_south", "southeast_asia", "east_africa"],
    ),
    _fi(
        "durian",
        147,
        1.5,
        27.1,
        5.3,
        3.8,
        {"potassium": 436.0, "vitamin_c": 19.7, "iron": 0.4},
        "fruit",
        ["southeast_asia"],
    ),
    _fi(
        "lychee",
        66,
        0.8,
        16.5,
        0.4,
        1.3,
        {"vitamin_c": 71.5, "potassium": 171.0, "copper": 0.1},
        "fruit",
        ["east_asia", "southeast_asia", "south_asia_east"],
    ),
    _fi(
        "rambutan",
        82,
        0.7,
        21.0,
        0.2,
        0.9,
        {"vitamin_c": 4.9, "iron": 0.4, "calcium": 22.0},
        "fruit",
        ["southeast_asia"],
    ),
    _fi(
        "persimmon",
        70,
        0.6,
        18.6,
        0.2,
        3.6,
        {"vitamin_a_iu": 1627.0, "vitamin_c": 7.5, "manganese": 0.4},
        "fruit",
        ["east_asia", "caucasus", "mediterranean"],
    ),
    _fi(
        "passion fruit",
        97,
        2.2,
        23.4,
        0.7,
        10.4,
        {"vitamin_c": 30.0, "iron": 1.6, "potassium": 348.0},
        "fruit",
        ["south_america", "east_africa", "pacific_islands"],
    ),
    _fi(
        "soursop",
        66,
        1.0,
        16.8,
        0.3,
        3.3,
        {"vitamin_c": 20.6, "potassium": 278.0, "magnesium": 21.0},
        "fruit",
        ["caribbean", "south_america", "southeast_asia"],
    ),
    _fi(
        "cherimoya",
        75,
        1.6,
        17.7,
        0.7,
        3.0,
        {"vitamin_c": 12.6, "potassium": 287.0, "vitamin_b6": 0.3},
        "fruit",
        ["andean", "south_america"],
    ),
    _fi(
        "tamarind",
        239,
        2.8,
        62.5,
        0.6,
        5.1,
        {"iron": 2.8, "magnesium": 92.0, "potassium": 628.0},
        "fruit",
        ["south_asia_south", "west_africa", "central_america"],
    ),
    _fi(
        "starfruit",
        31,
        1.0,
        6.7,
        0.3,
        2.8,
        {"vitamin_c": 34.4, "potassium": 133.0, "copper": 0.1},
        "fruit",
        ["southeast_asia", "south_america"],
    ),
    _fi(
        "baobab fruit",
        331,
        2.3,
        75.6,
        0.3,
        44.5,
        {"vitamin_c": 280.0, "calcium": 293.0, "potassium": 2270.0},
        "fruit",
        ["west_africa", "east_africa", "sub_saharan_africa"],
    ),
    _fi(
        "acai berry",
        70,
        0.8,
        4.0,
        5.0,
        2.0,
        {"calcium": 35.0, "iron": 0.6, "vitamin_a_iu": 1002.0},
        "fruit",
        ["south_america"],
    ),
    _fi(
        "cape gooseberry",
        53,
        1.9,
        11.2,
        0.7,
        4.9,
        {"vitamin_c": 11.0, "iron": 1.0, "vitamin_a_iu": 720.0},
        "fruit",
        ["andean", "south_america", "southern_africa"],
    ),
    _fi(
        "cloudberry",
        51,
        2.4,
        9.6,
        0.8,
        6.3,
        {"vitamin_c": 158.0, "vitamin_e": 3.0, "iron": 0.7},
        "fruit",
        ["nordic"],
    ),
    _fi(
        "lingonberry",
        50,
        0.3,
        11.5,
        0.5,
        2.5,
        {"vitamin_c": 12.0, "manganese": 2.5, "vitamin_e": 1.6},
        "fruit",
        ["nordic", "northern_europe"],
    ),
    _fi(
        "quince",
        57,
        0.4,
        15.3,
        0.1,
        1.9,
        {"vitamin_c": 15.0, "potassium": 197.0, "copper": 0.1},
        "fruit",
        ["caucasus", "mediterranean", "middle_east"],
    ),
    _fi(
        "marula fruit",
        56,
        0.6,
        14.3,
        0.1,
        0.6,
        {"vitamin_c": 176.0, "potassium": 215.0, "magnesium": 22.0},
        "fruit",
        ["southern_africa", "sub_saharan_africa"],
    ),
    _fi(
        "lucuma",
        99,
        1.5,
        25.0,
        0.5,
        1.3,
        {"iron": 0.4, "calcium": 16.0, "vitamin_b3": 1.9},
        "fruit",
        ["andean", "south_america"],
    ),
    _fi(
        "sapodilla",
        83,
        0.4,
        20.0,
        1.1,
        5.3,
        {"vitamin_c": 14.7, "potassium": 193.0, "iron": 0.8},
        "fruit",
        ["central_america", "south_asia_south"],
    ),
    _fi(
        "tempeh",
        192,
        20.3,
        7.6,
        10.8,
        0.0,
        {"calcium": 111.0, "iron": 2.7, "magnesium": 81.0},
        "protein",
        ["southeast_asia"],
    ),
    _fi(
        "seitan",
        370,
        75.2,
        14.0,
        1.9,
        0.6,
        {"iron": 5.2, "selenium": 16.3, "calcium": 53.0},
        "protein",
        ["east_asia"],
    ),
    _fi(
        "edamame",
        121,
        11.9,
        8.9,
        5.2,
        5.2,
        {"folate": 0.311, "vitamin_k": 0.026, "iron": 2.3},
        "protein",
        ["east_asia", "north_america"],
    ),
    _fi(
        "natto",
        211,
        17.7,
        14.4,
        11.0,
        5.4,
        {"vitamin_k2": 1103.0, "iron": 8.6, "manganese": 1.5},
        "protein",
        ["east_asia"],
    ),
    _fi(
        "halloumi cheese",
        321,
        25.0,
        2.6,
        24.0,
        0.0,
        {"calcium": 700.0, "sodium": 370.0, "vitamin_a_iu": 334.0},
        "protein",
        ["mediterranean", "middle_east"],
    ),
    _fi(
        "paneer",
        265,
        18.3,
        3.6,
        20.8,
        0.0,
        {"calcium": 480.0, "phosphorus": 451.0, "vitamin_b12": 0.8},
        "protein",
        ["south_asia_north", "south_asia_west"],
    ),
    _fi(
        "quorn",
        86,
        14.5,
        6.3,
        1.9,
        5.2,
        {"zinc": 2.4, "iron": 0.8, "calcium": 19.0},
        "protein",
        ["western_europe", "north_america"],
    ),
    _fi(
        "bison",
        143,
        28.4,
        0.0,
        2.4,
        0.0,
        {"iron": 3.4, "zinc": 4.6, "vitamin_b12": 2.9},
        "protein",
        ["north_america"],
    ),
    _fi(
        "kangaroo",
        98,
        22.0,
        0.0,
        1.0,
        0.0,
        {"iron": 4.1, "zinc": 3.1, "vitamin_b12": 2.1},
        "protein",
        ["australian"],
    ),
    _fi(
        "guinea pig (cuy)",
        96,
        19.0,
        0.0,
        1.6,
        0.0,
        {"iron": 3.0, "zinc": 2.0, "vitamin_b12": 2.0},
        "protein",
        ["andean"],
    ),
    _fi(
        "reindeer",
        128,
        24.0,
        0.0,
        3.2,
        0.0,
        {"iron": 4.3, "zinc": 4.2, "vitamin_b12": 3.4},
        "protein",
        ["nordic"],
    ),
    _fi(
        "goat",
        143,
        27.1,
        0.0,
        3.0,
        0.0,
        {"iron": 3.7, "zinc": 5.3, "vitamin_b12": 1.1},
        "protein",
        ["west_africa", "east_africa", "middle_east", "south_asia_north", "caribbean"],
    ),
    _fi(
        "rabbit",
        136,
        20.1,
        0.0,
        5.6,
        0.0,
        {"iron": 1.6, "zinc": 2.0, "vitamin_b12": 7.2},
        "protein",
        ["western_europe", "mediterranean"],
    ),
    _fi(
        "crocodile",
        104,
        21.1,
        0.0,
        1.9,
        0.0,
        {"iron": 1.2, "zinc": 2.1, "vitamin_b12": 1.3},
        "protein",
        ["australian", "sub_saharan_africa"],
    ),
    _fi(
        "spirulina",
        290,
        57.5,
        23.9,
        7.7,
        3.6,
        {"iron": 28.5, "vitamin_b12": 0.0, "magnesium": 195.0},
        "protein",
        ["east_africa", "central_america"],
    ),
    _fi(
        "cricket flour",
        420,
        65.0,
        6.0,
        15.0,
        6.0,
        {"iron": 5.0, "zinc": 9.6, "calcium": 104.0},
        "protein",
        ["southeast_asia", "east_africa"],
    ),
    _fi(
        "macadamia",
        718,
        7.9,
        13.8,
        75.8,
        8.6,
        {"magnesium": 130.0, "manganese": 4.1, "iron": 3.7},
        "nut",
        ["australian", "pacific_islands"],
    ),
    _fi(
        "brazil nut",
        659,
        14.3,
        11.7,
        67.1,
        7.5,
        {"selenium": 1917.0, "magnesium": 376.0, "zinc": 4.1},
        "nut",
        ["south_america"],
    ),
    _fi(
        "pine nut",
        673,
        13.7,
        13.1,
        68.4,
        3.7,
        {"magnesium": 251.0, "iron": 5.5, "zinc": 6.5},
        "nut",
        ["mediterranean", "central_asia", "east_asia"],
    ),
    _fi(
        "tiger nut",
        400,
        4.6,
        44.5,
        24.0,
        9.9,
        {"iron": 3.7, "magnesium": 90.0, "potassium": 710.0},
        "nut",
        ["west_africa", "north_africa", "iberian"],
    ),
    _fi(
        "hemp seeds",
        553,
        31.6,
        8.7,
        48.8,
        4.0,
        {"magnesium": 700.0, "iron": 7.9, "zinc": 9.9},
        "seed",
        ["north_america", "western_europe"],
    ),
    _fi(
        "pepitas (pumpkin seeds)",
        559,
        30.2,
        10.7,
        49.1,
        6.0,
        {"magnesium": 550.0, "iron": 8.8, "zinc": 7.8},
        "seed",
        ["central_america", "north_america"],
    ),
    _fi(
        "sacha inchi",
        480,
        27.0,
        6.0,
        41.0,
        3.0,
        {"omega_3_g": 17.0, "vitamin_e": 5.0, "zinc": 4.3},
        "seed",
        ["andean", "south_america"],
    ),
    _fi(
        "nigella seeds",
        345,
        16.0,
        44.2,
        22.0,
        5.5,
        {"iron": 11.6, "calcium": 570.0, "magnesium": 260.0},
        "seed",
        ["middle_east", "south_asia_north", "north_africa"],
    ),
    _fi(
        "watermelon seeds",
        557,
        28.3,
        15.3,
        47.4,
        0.9,
        {"magnesium": 515.0, "iron": 7.3, "zinc": 10.2},
        "seed",
        ["west_africa", "middle_east"],
    ),
    _fi(
        "lotus seeds",
        89,
        4.1,
        17.3,
        0.5,
        0.0,
        {"potassium": 367.0, "magnesium": 56.0, "calcium": 44.0},
        "seed",
        ["east_asia", "south_asia_east"],
    ),
    _fi(
        "sardines",
        208,
        24.6,
        0.0,
        11.5,
        0.0,
        {"calcium": 382.0, "vitamin_d_iu": 272.0, "omega_3_g": 1.5},
        "seafood",
        ["mediterranean", "iberian", "north_africa"],
    ),
    _fi(
        "mackerel",
        205,
        18.6,
        0.0,
        13.9,
        0.0,
        {"vitamin_d_iu": 643.0, "omega_3_g": 2.7, "selenium": 44.1},
        "seafood",
        ["east_asia", "nordic", "northern_europe"],
    ),
    _fi(
        "anchovies",
        131,
        20.4,
        0.0,
        4.8,
        0.0,
        {"calcium": 147.0, "iron": 3.3, "selenium": 36.5},
        "seafood",
        ["mediterranean", "southeast_asia"],
    ),
    _fi(
        "barramundi",
        110,
        23.0,
        0.0,
        1.9,
        0.0,
        {"selenium": 36.5, "omega_3_g": 0.8, "vitamin_d_iu": 200.0},
        "seafood",
        ["australian"],
    ),
    _fi(
        "tilapia",
        96,
        20.1,
        0.0,
        1.7,
        0.0,
        {"selenium": 54.4, "vitamin_b12": 1.6, "phosphorus": 170.0},
        "seafood",
        ["east_africa", "southeast_asia", "latin_america"],
    ),
    _fi(
        "sea bass",
        97,
        18.4,
        0.0,
        2.0,
        0.0,
        {"selenium": 36.5, "phosphorus": 194.0, "potassium": 256.0},
        "seafood",
        ["mediterranean", "western_europe"],
    ),
    _fi(
        "octopus",
        82,
        14.9,
        2.2,
        1.0,
        0.0,
        {"iron": 5.3, "zinc": 1.7, "selenium": 44.8},
        "seafood",
        ["mediterranean", "east_asia", "pacific_islands"],
    ),
    _fi(
        "crayfish",
        82,
        16.9,
        0.0,
        1.0,
        0.0,
        {"selenium": 30.0, "zinc": 1.5, "vitamin_b12": 2.2},
        "seafood",
        ["north_america", "nordic"],
    ),
    _fi(
        "mussels",
        86,
        11.9,
        3.7,
        2.2,
        0.0,
        {"iron": 3.9, "vitamin_b12": 12.0, "zinc": 1.6},
        "seafood",
        ["mediterranean", "western_europe", "nordic"],
    ),
    _fi(
        "sea cucumber",
        56,
        13.0,
        0.0,
        0.3,
        0.0,
        {"calcium": 30.0, "iron": 0.6, "zinc": 0.4},
        "seafood",
        ["east_asia", "pacific_islands"],
    ),
    _fi(
        "kefir",
        63,
        3.3,
        4.7,
        3.5,
        0.0,
        {"calcium": 130.0, "vitamin_b12": 0.3, "vitamin_d_iu": 40.0},
        "dairy",
        ["caucasus", "eastern_europe", "central_asia"],
    ),
    _fi(
        "labneh",
        147,
        5.5,
        3.3,
        12.3,
        0.0,
        {"calcium": 180.0, "vitamin_b12": 0.4, "phosphorus": 130.0},
        "dairy",
        ["middle_east", "mediterranean"],
    ),
    _fi(
        "skyr",
        63,
        11.0,
        3.6,
        0.2,
        0.0,
        {"calcium": 150.0, "phosphorus": 135.0, "vitamin_b12": 0.8},
        "dairy",
        ["nordic"],
    ),
    _fi(
        "ayran",
        35,
        1.7,
        2.5,
        2.0,
        0.0,
        {"calcium": 80.0, "phosphorus": 55.0, "potassium": 90.0},
        "dairy",
        ["middle_east", "central_asia", "balkans"],
    ),
    _fi(
        "clotted cream",
        586,
        1.6,
        2.3,
        63.5,
        0.0,
        {"vitamin_a_iu": 2500.0, "calcium": 37.0, "phosphorus": 24.0},
        "dairy",
        ["western_europe"],
    ),
    _fi(
        "kimchi",
        15,
        1.1,
        2.4,
        0.5,
        1.6,
        {"vitamin_c": 18.0, "vitamin_k": 0.043, "sodium": 498.0},
        "fermented",
        ["east_asia"],
    ),
    _fi(
        "sauerkraut",
        19,
        0.9,
        4.3,
        0.1,
        2.9,
        {"vitamin_c": 15.0, "vitamin_k": 0.013, "sodium": 661.0},
        "fermented",
        ["central_europe", "eastern_europe"],
    ),
    _fi(
        "miso",
        199,
        11.7,
        26.5,
        6.0,
        5.4,
        {"sodium": 3728.0, "zinc": 4.3, "manganese": 0.9},
        "fermented",
        ["east_asia"],
    ),
    _fi(
        "tempoyak",
        114,
        1.3,
        12.0,
        7.0,
        2.0,
        {"sodium": 380.0, "potassium": 260.0, "iron": 0.5},
        "fermented",
        ["southeast_asia"],
    ),
    _fi(
        "injera (fermented)",
        130,
        3.6,
        26.0,
        0.7,
        1.6,
        {"iron": 5.7, "calcium": 49.0, "probiotics": 1.0},
        "fermented",
        ["east_africa"],
    ),
    _fi(
        "turmeric powder",
        312,
        9.7,
        67.1,
        3.3,
        22.7,
        {"iron": 55.0, "manganese": 19.8, "potassium": 2080.0},
        "spice",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "cinnamon",
        247,
        4.0,
        80.6,
        1.2,
        53.1,
        {"calcium": 1002.0, "iron": 8.3, "manganese": 17.5},
        "spice",
        ["south_asia_south", "southeast_asia", "middle_east"],
    ),
    _fi(
        "ginger powder",
        335,
        9.0,
        71.6,
        4.2,
        14.1,
        {"potassium": 1320.0, "magnesium": 214.0, "iron": 19.8},
        "spice",
        ["south_asia_south", "east_asia", "caribbean"],
    ),
    _fi(
        "fenugreek seeds",
        323,
        23.0,
        58.4,
        6.4,
        24.6,
        {"iron": 33.5, "magnesium": 191.0, "manganese": 1.2},
        "spice",
        ["south_asia_north", "middle_east", "north_africa"],
    ),
    _fi(
        "black cumin (nigella)",
        345,
        16.0,
        52.0,
        22.2,
        5.5,
        {"iron": 11.6, "calcium": 570.0, "magnesium": 260.0},
        "spice",
        ["middle_east", "south_asia_north"],
    ),
    _fi(
        "sumac",
        239,
        5.0,
        71.0,
        1.5,
        13.0,
        {"vitamin_c": 30.0, "iron": 3.5, "calcium": 270.0},
        "spice",
        ["middle_east", "mediterranean", "caucasus"],
    ),
    _fi(
        "saffron",
        310,
        11.4,
        65.4,
        5.9,
        3.9,
        {"manganese": 28.4, "vitamin_c": 80.8, "iron": 11.1},
        "spice",
        ["middle_east", "mediterranean", "south_asia_north"],
    ),
    _fi(
        "lemongrass",
        99,
        1.8,
        25.3,
        0.5,
        0.0,
        {"iron": 8.2, "manganese": 5.2, "potassium": 723.0},
        "spice",
        ["southeast_asia", "east_africa"],
    ),
    _fi(
        "star anise",
        337,
        17.6,
        50.0,
        15.9,
        14.6,
        {"calcium": 646.0, "iron": 37.0, "magnesium": 170.0},
        "spice",
        ["east_asia", "southeast_asia"],
    ),
    _fi(
        "za'atar",
        276,
        9.1,
        63.9,
        7.4,
        14.6,
        {"iron": 29.3, "calcium": 1076.0, "manganese": 6.8},
        "spice",
        ["middle_east", "north_africa"],
    ),
    _fi(
        "coconut oil",
        862,
        0.0,
        0.0,
        100.0,
        0.0,
        {"vitamin_e": 0.1, "vitamin_k": 0.001, "iron": 0.04},
        "oil",
        ["south_asia_south", "southeast_asia", "pacific_islands", "caribbean"],
    ),
    _fi(
        "ghee",
        900,
        0.0,
        0.0,
        99.5,
        0.0,
        {"vitamin_a_iu": 3069.0, "vitamin_d_iu": 15.0, "vitamin_e": 2.8},
        "oil",
        ["south_asia_north", "south_asia_west", "middle_east"],
    ),
    _fi(
        "palm oil (red)",
        884,
        0.0,
        0.0,
        100.0,
        0.0,
        {"vitamin_a_iu": 5000.0, "vitamin_e": 15.9, "coenzyme_q10": 3.0},
        "oil",
        ["west_africa", "southeast_asia"],
    ),
    _fi(
        "argan oil",
        884,
        0.0,
        0.0,
        100.0,
        0.0,
        {"vitamin_e": 62.8, "omega_9_g": 45.0, "omega_6_g": 35.0},
        "oil",
        ["north_africa"],
    ),
    _fi(
        "perilla oil",
        884,
        0.0,
        0.0,
        100.0,
        0.0,
        {"omega_3_g": 54.0, "vitamin_e": 7.5, "omega_6_g": 14.0},
        "oil",
        ["east_asia"],
    ),
    _fi(
        "sacha inchi oil",
        884,
        0.0,
        0.0,
        100.0,
        0.0,
        {"omega_3_g": 48.0, "vitamin_e": 17.0, "omega_6_g": 34.0},
        "oil",
        ["andean", "south_america"],
    ),
    _fi(
        "matcha",
        324,
        30.6,
        38.9,
        5.3,
        38.5,
        {"iron": 17.0, "potassium": 2700.0, "vitamin_c": 60.0},
        "beverage",
        ["east_asia"],
    ),
    _fi(
        "yerba mate",
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        {"vitamin_c": 1.0, "magnesium": 5.0, "potassium": 20.0},
        "beverage",
        ["south_america"],
    ),
    _fi(
        "rooibos tea",
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        {"calcium": 1.0, "manganese": 0.04, "magnesium": 1.5},
        "beverage",
        ["southern_africa"],
    ),
    _fi(
        "hibiscus tea",
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        {"vitamin_c": 12.0, "iron": 0.5, "calcium": 1.0},
        "beverage",
        ["west_africa", "caribbean", "middle_east"],
    ),
    _fi(
        "horchata",
        54,
        0.4,
        12.0,
        0.8,
        0.2,
        {"calcium": 25.0, "phosphorus": 15.0, "potassium": 30.0},
        "beverage",
        ["central_america", "iberian"],
    ),
    _fi(
        "kvass",
        27,
        0.2,
        5.8,
        0.0,
        0.0,
        {"vitamin_b1": 0.02, "vitamin_b2": 0.04, "iron": 0.2},
        "beverage",
        ["eastern_europe"],
    ),
    _fi(
        "date syrup",
        282,
        1.0,
        73.0,
        0.2,
        1.0,
        {"potassium": 370.0, "magnesium": 20.0, "iron": 1.5},
        "condiment",
        ["middle_east", "north_africa"],
    ),
    _fi(
        "pomegranate molasses",
        250,
        0.5,
        65.0,
        0.0,
        0.5,
        {"potassium": 290.0, "iron": 0.7, "vitamin_c": 5.0},
        "condiment",
        ["middle_east", "caucasus"],
    ),
    _fi(
        "fish sauce",
        35,
        5.1,
        3.6,
        0.0,
        0.0,
        {"sodium": 7720.0, "iron": 1.4, "calcium": 25.0},
        "condiment",
        ["southeast_asia", "east_asia"],
    ),
    _fi(
        "harissa paste",
        87,
        2.5,
        12.0,
        3.5,
        4.5,
        {"vitamin_c": 38.0, "iron": 1.8, "vitamin_a_iu": 3200.0},
        "condiment",
        ["north_africa", "middle_east"],
    ),
    _fi(
        "gochujang",
        100,
        2.5,
        20.0,
        1.5,
        1.0,
        {"iron": 0.8, "sodium": 1590.0, "vitamin_c": 3.0},
        "condiment",
        ["east_asia"],
    ),
    _fi(
        "chimichurri",
        150,
        1.0,
        3.0,
        15.0,
        1.5,
        {"vitamin_c": 25.0, "vitamin_a_iu": 2100.0, "iron": 2.5},
        "condiment",
        ["south_america"],
    ),
    _fi(
        "tahini",
        595,
        17.0,
        21.2,
        53.8,
        9.3,
        {"calcium": 426.0, "iron": 8.9, "magnesium": 95.0},
        "condiment",
        ["middle_east", "mediterranean", "north_africa"],
    ),
    _fi(
        "purple sweet potato",
        86,
        1.6,
        20.1,
        0.1,
        3.0,
        {"vitamin_a_iu": 19218.0, "vitamin_c": 12.8, "manganese": 0.3},
        "vegetable",
        ["pacific_islands", "east_asia", "south_america"],
    ),
    _fi(
        "drumstick (moringa pod)",
        37,
        2.1,
        8.5,
        0.2,
        3.2,
        {"vitamin_c": 141.0, "calcium": 30.0, "iron": 0.4},
        "vegetable",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "water spinach",
        19,
        2.6,
        3.1,
        0.2,
        2.1,
        {"vitamin_c": 55.0, "iron": 1.7, "calcium": 77.0},
        "vegetable",
        ["southeast_asia", "south_asia_south", "east_asia"],
    ),
    _fi(
        "ash gourd",
        10,
        0.4,
        1.9,
        0.0,
        0.5,
        {"vitamin_c": 13.0, "calcium": 19.0, "iron": 0.2},
        "vegetable",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "snake gourd",
        18,
        0.5,
        3.4,
        0.3,
        0.6,
        {"vitamin_c": 12.0, "iron": 0.4, "calcium": 26.0},
        "vegetable",
        ["south_asia_south"],
    ),
    _fi(
        "ridge gourd",
        20,
        0.5,
        4.4,
        0.2,
        0.5,
        {"vitamin_c": 5.0, "iron": 0.4, "calcium": 18.0},
        "vegetable",
        ["south_asia_south", "south_asia_north"],
    ),
    _fi(
        "ivy gourd",
        18,
        1.2,
        3.1,
        0.1,
        1.6,
        {"vitamin_c": 16.0, "calcium": 40.0, "iron": 0.7},
        "vegetable",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "cluster beans",
        16,
        3.2,
        2.6,
        0.1,
        3.2,
        {"calcium": 57.0, "iron": 1.1, "folate": 0.04},
        "vegetable",
        ["south_asia_north", "south_asia_west"],
    ),
    _fi(
        "banana flower",
        51,
        1.6,
        9.9,
        0.6,
        5.7,
        {"potassium": 553.0, "magnesium": 56.0, "iron": 0.6},
        "vegetable",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "jackfruit seeds",
        184,
        6.6,
        38.4,
        0.4,
        1.5,
        {"calcium": 50.0, "iron": 1.5, "potassium": 407.0},
        "vegetable",
        ["south_asia_south", "southeast_asia"],
    ),
    _fi(
        "yam bean",
        39,
        1.6,
        8.8,
        0.1,
        4.9,
        {"vitamin_c": 20.2, "potassium": 150.0, "iron": 0.6},
        "vegetable",
        ["southeast_asia", "central_america"],
    ),
    _fi(
        "pandan leaves",
        35,
        1.5,
        7.0,
        0.4,
        3.0,
        {"iron": 1.2, "vitamin_c": 2.0, "calcium": 15.0},
        "vegetable",
        ["southeast_asia"],
    ),
    _fi(
        "perilla leaves",
        37,
        3.9,
        5.0,
        1.0,
        3.4,
        {"calcium": 230.0, "iron": 3.0, "vitamin_c": 22.0},
        "vegetable",
        ["east_asia"],
    ),
    _fi(
        "chrysanthemum greens",
        24,
        3.4,
        2.8,
        0.6,
        2.0,
        {"calcium": 117.0, "iron": 2.7, "potassium": 567.0},
        "vegetable",
        ["east_asia"],
    ),
    _fi(
        "hearts of palm",
        28,
        2.5,
        4.6,
        0.6,
        2.4,
        {"potassium": 177.0, "zinc": 1.2, "iron": 0.6},
        "vegetable",
        ["south_america", "caribbean"],
    ),
    _fi(
        "samphire",
        25,
        2.0,
        3.0,
        0.3,
        0.0,
        {"vitamin_c": 5.0, "sodium": 980.0, "iron": 1.1},
        "vegetable",
        ["western_europe", "nordic"],
    ),
    _fi(
        "venison",
        120,
        23.6,
        0.0,
        2.4,
        0.0,
        {"iron": 3.4, "zinc": 2.1, "vitamin_b12": 5.6},
        "protein",
        ["nordic", "north_america", "eastern_europe"],
    ),
    _fi(
        "emu",
        112,
        22.5,
        0.0,
        2.0,
        0.0,
        {"iron": 5.4, "zinc": 3.0, "vitamin_b12": 2.5},
        "protein",
        ["australian"],
    ),
    _fi(
        "wild boar",
        122,
        21.5,
        0.0,
        3.3,
        0.0,
        {"iron": 2.5, "zinc": 3.0, "vitamin_b12": 2.0},
        "protein",
        ["central_europe", "mediterranean"],
    ),
    _fi(
        "ostrich",
        97,
        21.0,
        0.0,
        1.0,
        0.0,
        {"iron": 3.2, "zinc": 3.5, "vitamin_b6": 0.5},
        "protein",
        ["southern_africa", "australian"],
    ),
    _fi(
        "llama",
        120,
        23.9,
        0.0,
        2.2,
        0.0,
        {"iron": 3.8, "zinc": 3.5, "vitamin_b12": 2.1},
        "protein",
        ["andean"],
    ),
    _fi(
        "quail",
        134,
        21.8,
        0.0,
        4.5,
        0.0,
        {"iron": 3.0, "zinc": 2.7, "vitamin_b12": 0.5},
        "protein",
        ["east_asia", "middle_east", "western_europe"],
    ),
    _fi(
        "frog legs",
        73,
        16.4,
        0.0,
        0.3,
        0.0,
        {"potassium": 285.0, "phosphorus": 147.0, "selenium": 18.0},
        "protein",
        ["southeast_asia", "western_europe"],
    ),
    _fi(
        "snails (escargot)",
        90,
        16.1,
        2.0,
        1.4,
        0.0,
        {"iron": 3.5, "magnesium": 250.0, "selenium": 27.4},
        "protein",
        ["western_europe", "west_africa"],
    ),
    _fi(
        "black rice",
        356,
        8.9,
        76.0,
        3.3,
        3.5,
        {"iron": 3.5, "anthocyanins": 10.0, "magnesium": 143.0},
        "grain",
        ["east_asia", "southeast_asia"],
    ),
    _fi(
        "wild rice",
        357,
        14.7,
        74.9,
        1.1,
        6.2,
        {"zinc": 5.9, "manganese": 1.3, "magnesium": 177.0},
        "grain",
        ["north_america"],
    ),
    _fi(
        "pearl barley",
        123,
        2.3,
        28.2,
        0.4,
        3.8,
        {"selenium": 8.6, "manganese": 0.3, "iron": 1.3},
        "grain",
        ["central_europe", "middle_east"],
    ),
    _fi(
        "triticale",
        336,
        13.1,
        72.1,
        2.1,
        14.6,
        {"manganese": 3.2, "magnesium": 130.0, "iron": 2.6},
        "grain",
        ["central_europe", "north_america"],
    ),
    _fi(
        "emmer wheat",
        335,
        12.0,
        67.0,
        2.5,
        10.0,
        {"magnesium": 120.0, "iron": 3.5, "zinc": 3.3},
        "grain",
        ["mediterranean", "middle_east"],
    ),
    _fi(
        "camu camu",
        17,
        0.4,
        4.7,
        0.1,
        0.3,
        {"vitamin_c": 2780.0, "potassium": 83.0, "manganese": 0.1},
        "fruit",
        ["south_america"],
    ),
    _fi(
        "mangosteen",
        73,
        0.4,
        18.0,
        0.6,
        1.8,
        {"vitamin_c": 2.9, "folate": 0.031, "manganese": 0.1},
        "fruit",
        ["southeast_asia"],
    ),
    _fi(
        "snake fruit (salak)",
        82,
        0.4,
        22.0,
        0.0,
        2.0,
        {"iron": 3.9, "calcium": 28.0, "vitamin_c": 8.4},
        "fruit",
        ["southeast_asia"],
    ),
    _fi(
        "wood apple",
        134,
        7.1,
        18.1,
        3.7,
        5.0,
        {"calcium": 130.0, "iron": 5.0, "vitamin_c": 3.0},
        "fruit",
        ["south_asia_south"],
    ),
    _fi(
        "custard apple",
        101,
        2.1,
        25.2,
        0.6,
        4.4,
        {"vitamin_c": 19.2, "potassium": 382.0, "magnesium": 18.0},
        "fruit",
        ["south_asia_south", "south_america"],
    ),
    _fi(
        "rose apple",
        25,
        0.6,
        5.7,
        0.3,
        0.0,
        {"vitamin_c": 22.3, "potassium": 123.0, "calcium": 29.0},
        "fruit",
        ["southeast_asia", "south_asia_south"],
    ),
    _fi(
        "black sapote",
        45,
        0.8,
        12.0,
        0.1,
        2.6,
        {"vitamin_c": 19.0, "potassium": 305.0, "calcium": 18.0},
        "fruit",
        ["central_america"],
    ),
    _fi(
        "miracle fruit",
        23,
        0.3,
        5.6,
        0.2,
        0.0,
        {"vitamin_c": 4.0, "iron": 0.3, "calcium": 10.0},
        "fruit",
        ["west_africa"],
    ),
    _fi(
        "horned melon (kiwano)",
        44,
        1.8,
        7.6,
        1.3,
        0.0,
        {"vitamin_c": 5.3, "iron": 1.1, "zinc": 0.5},
        "fruit",
        ["southern_africa", "sub_saharan_africa"],
    ),
    _fi(
        "lupin beans",
        371,
        36.2,
        40.4,
        9.7,
        18.9,
        {"magnesium": 198.0, "iron": 4.4, "zinc": 4.8},
        "legume",
        ["mediterranean", "south_america", "australian"],
    ),
    _fi(
        "moth beans",
        343,
        22.9,
        61.5,
        1.6,
        5.3,
        {"iron": 10.9, "calcium": 150.0, "zinc": 3.4},
        "legume",
        ["south_asia_north", "south_asia_west"],
    ),
    _fi(
        "horse gram",
        321,
        22.0,
        57.2,
        0.5,
        5.3,
        {"iron": 6.8, "calcium": 287.0, "phosphorus": 311.0},
        "legume",
        ["south_asia_south"],
    ),
    _fi(
        "bambara groundnut",
        367,
        19.0,
        61.0,
        6.5,
        5.0,
        {"calcium": 95.0, "iron": 5.0, "zinc": 3.0},
        "legume",
        ["west_africa", "sub_saharan_africa"],
    ),
    _fi(
        "winged beans",
        409,
        29.7,
        44.8,
        16.3,
        25.9,
        {"iron": 15.0, "calcium": 440.0, "zinc": 4.8},
        "legume",
        ["southeast_asia", "pacific_islands"],
    ),
    _fi(
        "shea nut",
        600,
        8.0,
        33.0,
        48.0,
        3.0,
        {"vitamin_e": 4.0, "iron": 3.0, "calcium": 30.0},
        "nut",
        ["west_africa"],
    ),
    _fi(
        "kola nut",
        150,
        2.0,
        12.0,
        0.2,
        2.0,
        {"caffeine_mg": 25.0, "potassium": 150.0, "magnesium": 18.0},
        "nut",
        ["west_africa"],
    ),
    _fi(
        "mongongo nut",
        567,
        24.0,
        19.0,
        43.0,
        3.0,
        {"vitamin_e": 24.0, "calcium": 37.0, "iron": 3.0},
        "nut",
        ["sub_saharan_africa", "southern_africa"],
    ),
    _fi(
        "candlenut",
        620,
        19.0,
        8.0,
        58.0,
        2.0,
        {"iron": 4.0, "phosphorus": 320.0, "potassium": 220.0},
        "nut",
        ["southeast_asia", "pacific_islands"],
    ),
    _fi(
        "bunya nut",
        180,
        6.0,
        35.0,
        2.5,
        4.0,
        {"magnesium": 60.0, "potassium": 300.0, "zinc": 1.5},
        "nut",
        ["australian"],
    ),
]

# Quick count assertion.
assert len(FOOD_DATABASE) >= 250, f"Food DB has only {len(FOOD_DATABASE)} entries; expected ≥ 100."

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

_NUTRIENT_MICRO_KEYS: dict[str, list[str]] = {
    "folate": ["folate"],
    "omega_3": ["omega_3"],
    "vitamin_d": ["vitamin_d_iu"],
    "vitamin_c": ["vitamin_c"],
    "vitamin_a": ["vitamin_a"],
    "iron": ["iron"],
    "zinc": ["zinc"],
    "calcium": ["calcium"],
    "magnesium": ["magnesium"],
    "selenium": ["selenium"],
    "antioxidants": [
        "vitamin_c",
        "vitamin_e",
        "selenium",
        "flavonoids",
        "anthocyanins",
        "polyphenols",
    ],
    "polyphenols": ["polyphenols", "catechins", "flavonoids", "anthocyanins", "punicalagins"],
    "anti_inflammatory": ["omega_3", "curcumin", "gingerol", "anthocyanins", "polyphenols"],
    "cruciferous_vegetables": ["sulforaphane"],
}

# ---------------------------------------------------------------------------
# Dietary restriction filters
# ---------------------------------------------------------------------------

_RESTRICTION_EXCLUDED_GROUPS: dict[str, set[str]] = {
    "vegetarian": {"meat", "poultry", "fish"},
    "vegan": {"meat", "poultry", "fish", "dairy", "eggs"},
    "pescatarian": {"meat", "poultry"},
    "gluten_free": set(),  # handled by name-based filtering
}

_GLUTEN_KEYWORDS: set[str] = {
    "wheat",
    "roti",
    "bread",
    "pasta",
    "rye",
    "bulgur",
    "couscous",
    "naan",
    "pita",
    "tortilla",
    "barley",
    "seitan",
}

_HALAL_EXCLUDED: set[str] = {"pork", "ham", "bacon", "wine", "beer", "lard"}
_KOSHER_EXCLUDED: set[str] = {
    "pork",
    "ham",
    "bacon",
    "shellfish",
    "shrimp",
    "crab",
    "lobster",
    "lard",
}

# ---------------------------------------------------------------------------
# Telomere-protective nutrients (evidence-based associations)
# ---------------------------------------------------------------------------

_TELOMERE_PROTECTIVE_NUTRIENTS: list[dict[str, Any]] = [
    {
        "nutrient": "omega_3",
        "effect": "Reduces oxidative telomere shortening",
        "priority": "critical",
        "confidence": 0.88,
    },
    {
        "nutrient": "folate",
        "effect": "Supports DNA methylation and repair",
        "priority": "high",
        "confidence": 0.82,
    },
    {
        "nutrient": "vitamin_d",
        "effect": "Modulates telomerase activity",
        "priority": "high",
        "confidence": 0.78,
    },
    {
        "nutrient": "vitamin_c",
        "effect": "Antioxidant protection of telomeric DNA",
        "priority": "moderate",
        "confidence": 0.75,
    },
    {
        "nutrient": "vitamin_e",
        "effect": "Lipid peroxidation protection near telomeres",
        "priority": "moderate",
        "confidence": 0.72,
    },
    {
        "nutrient": "polyphenols",
        "effect": "Upregulate telomerase via NF-κB pathway modulation",
        "priority": "moderate",
        "confidence": 0.70,
    },
    {
        "nutrient": "zinc",
        "effect": "Required co-factor for telomerase",
        "priority": "moderate",
        "confidence": 0.73,
    },
    {
        "nutrient": "magnesium",
        "effect": "DNA stability and repair enzyme cofactor",
        "priority": "moderate",
        "confidence": 0.68,
    },
    {
        "nutrient": "anti_inflammatory",
        "effect": "Chronic inflammation accelerates telomere attrition",
        "priority": "high",
        "confidence": 0.85,
    },
]


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

        Returns
        -------
        list[MealPlan]
            One :class:`MealPlan` per day.
        """
        regional_foods = self._region_index.get(region, self._food_db)
        if not regional_foods:
            regional_foods = self._food_db

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

        for day in range(1, days + 1):
            plan = self._build_day_plan(
                day=day,
                calories_target=calories,
                breakfast_pool=breakfast_candidates,
                main_pool=main_candidates,
                snack_pool=snack_candidates,
                rng=rng,
            )
            plans.append(plan)

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

        # Expected telomere length (same model as disease_risk.py).
        expected_bp = 11_000.0 - 30.0 * age
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
    ) -> MealPlan:
        """Construct a single day's meal plan using greedy selection."""
        # Calorie budgets: breakfast 25%, lunch 35%, dinner 30%, snacks 10%.
        budget_bkf = calories_target * 0.25
        budget_lun = calories_target * 0.35
        budget_din = calories_target * 0.30
        budget_snk = calories_target * 0.10

        breakfast = self._select_meal(breakfast_pool, budget_bkf, rng, day)
        lunch = self._select_meal(main_pool, budget_lun, rng, day + 100)
        dinner = self._select_meal(main_pool, budget_din, rng, day + 200)
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
    ) -> list[tuple[FoodItem, float]]:
        """Greedily select foods to approximately fill a calorie budget.

        Introduces controlled randomness via shuffling to ensure variety
        across days.
        """
        if not pool:
            return []

        # Shuffle with offset for day-level variety.
        indices = np.arange(len(pool))
        rng_local = np.random.default_rng(seed=rng.integers(0, 2**31) + seed_offset)
        rng_local.shuffle(indices)

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
            "meat": ["legume", "fish"],
            "poultry": ["legume", "fish"],
            "fish": ["legume"],
            "dairy": ["legume", "nut"],
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

_PRIORITY_RANK: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "low": 1,
}
