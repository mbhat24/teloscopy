"""Nutrition and diet recommendation module for genetically-informed dietary guidance.

Provides geography-aware, nutrigenomics-based dietary recommendations
that integrate genetic variant data with regional food availability
to produce personalised meal plans and nutritional guidance.
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
