"""Nutrition and diet recommendation module for genetically-informed dietary guidance.

Provides geography-aware, nutrigenomics-based dietary recommendations
that integrate genetic variant data with regional food availability
to produce personalised meal plans and nutritional guidance.

Includes country-wise and state-wise dietary profiles for localised
meal planning across 20+ countries and 80+ states/provinces.
"""

from .diet_advisor import (
    FOOD_DATABASE,
    GEOGRAPHIC_FOOD_DB,
    NUTRIGENOMICS_DB,
    DietAdvisor,
    DietaryRecommendation,
    FoodItem,
    GeographicProfile,
    MealPlan,
    NutrientNeed,
)
from .regional_diets import (
    COUNTRY_STATES,
    FRONTEND_REGION_MAP,
    REGION_COUNTRIES,
    CountryProfile,
    StateProfile,
    get_country_profile,
    get_state_profile,
    list_countries_for_region,
    list_states_for_country,
    resolve_region,
)

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
    "CountryProfile",
    "StateProfile",
    "REGION_COUNTRIES",
    "COUNTRY_STATES",
    "FRONTEND_REGION_MAP",
    "resolve_region",
    "get_country_profile",
    "get_state_profile",
    "list_countries_for_region",
    "list_states_for_country",
]
