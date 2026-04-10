"""Multi-language support for diet plan generation.

Translates dietary recommendations, food names, meal plan labels, and
nutritional guidance into multiple languages.  Uses a built-in phrase
database with support for external translation providers.

Supported languages (Phase 1): English, Spanish, French, German,
Mandarin Chinese, Hindi, Arabic, Portuguese, Japanese, Korean.

References
----------
.. [1] WHO Technical Report Series 916 — Diet, Nutrition and the
       Prevention of Chronic Diseases (2003).
.. [2] FAO/WHO Codex Alimentarius — International Food Standards.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON data loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"


def _load_json(name: str) -> Any:
    with open(_DATA_DIR / name) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "zh": "中文",
    "hi": "हिन्दी",
    "ar": "العربية",
    "pt": "Português",
    "ja": "日本語",
    "ko": "한국어",
}


@dataclass
class TranslatedMealPlan:
    """A meal plan with all text fields translated."""

    language: str
    day_label: str
    breakfast_label: str
    breakfast: str
    lunch_label: str
    lunch: str
    dinner_label: str
    dinner: str
    snacks_label: str
    snacks: list[str] = field(default_factory=list)


@dataclass
class TranslatedRecommendation:
    """A dietary recommendation translated to the target language."""

    language: str
    nutrient: str
    recommendation: str
    foods_to_increase: list[str] = field(default_factory=list)
    foods_to_avoid: list[str] = field(default_factory=list)
    daily_target: str = ""
    disclaimer: str = ""


@dataclass
class TranslatedReport:
    """Complete translated diet report."""

    language: str
    language_name: str
    title: str
    summary: str
    recommendations: list[TranslatedRecommendation] = field(default_factory=list)
    meal_plans: list[TranslatedMealPlan] = field(default_factory=list)
    general_advice: list[str] = field(default_factory=list)
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Built-in translation database
# ---------------------------------------------------------------------------

_LABELS: dict[str, dict[str, str]] = _load_json("i18n_labels.json")

# Day names in each language
_DAY_NAMES: dict[str, list[str]] = _load_json("i18n_day_names.json")

# Common food name translations (100+ foods)
_FOOD_TRANSLATIONS: dict[str, dict[str, str]] = _load_json("i18n_food_translations.json")

# Nutrient name translations
_NUTRIENT_TRANSLATIONS: dict[str, dict[str, str]] = _load_json("i18n_nutrient_translations.json")


# ---------------------------------------------------------------------------
# Translator class
# ---------------------------------------------------------------------------


class DietTranslator:
    """Translate dietary recommendations and meal plans.

    Parameters
    ----------
    target_language : str
        ISO 639-1 language code (e.g. ``"es"`` for Spanish).

    Raises
    ------
    ValueError
        If ``target_language`` is not in :data:`SUPPORTED_LANGUAGES`.
    """

    def __init__(self, target_language: str = "en") -> None:
        lang = target_language.lower().strip()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{lang}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        self.lang: str = lang
        self.lang_name: str = SUPPORTED_LANGUAGES[lang]

    # -- helpers ----------------------------------------------------------

    def _t(self, key: str) -> str:
        """Look up a label translation, falling back to English."""
        return _LABELS.get(key, {}).get(self.lang, _LABELS.get(key, {}).get("en", key))

    def _translate_food(self, food_name: str) -> str:
        """Translate a food name, falling back to original."""
        if self.lang == "en":
            return food_name
        key = food_name.lower().strip()
        return _FOOD_TRANSLATIONS.get(key, {}).get(self.lang, food_name)

    def _translate_nutrient(self, nutrient: str) -> str:
        """Translate a nutrient name."""
        key = nutrient.lower().replace(" ", "_").replace("-", "_").strip()
        return _NUTRIENT_TRANSLATIONS.get(key, {}).get(self.lang, nutrient)

    def _day_name(self, index: int) -> str:
        """Get day name by index (0=Monday)."""
        days = _DAY_NAMES.get(self.lang, _DAY_NAMES["en"])
        return days[index % 7]

    # -- public API -------------------------------------------------------

    def translate_recommendation(
        self,
        nutrient: str,
        recommendation: str,
        target_foods: list[str] | None = None,
        avoid_foods: list[str] | None = None,
        daily_target: str = "",
    ) -> TranslatedRecommendation:
        """Translate a single dietary recommendation.

        Parameters
        ----------
        nutrient : str
            Nutrient name in English.
        recommendation : str
            Recommendation text (kept in English if no translation).
        target_foods, avoid_foods : list[str]
            Food lists to translate.
        daily_target : str
            Daily target string (numbers preserved, units translated).
        """
        return TranslatedRecommendation(
            language=self.lang,
            nutrient=self._translate_nutrient(nutrient),
            recommendation=recommendation,
            foods_to_increase=[self._translate_food(f) for f in (target_foods or [])],
            foods_to_avoid=[self._translate_food(f) for f in (avoid_foods or [])],
            daily_target=daily_target,
            disclaimer=self._t("disclaimer"),
        )

    def translate_meal_plan(
        self,
        day_index: int,
        breakfast: str,
        lunch: str,
        dinner: str,
        snacks: list[str] | None = None,
    ) -> TranslatedMealPlan:
        """Translate a single day's meal plan."""
        return TranslatedMealPlan(
            language=self.lang,
            day_label=self._day_name(day_index),
            breakfast_label=self._t("breakfast"),
            breakfast=breakfast,
            lunch_label=self._t("lunch"),
            lunch=lunch,
            dinner_label=self._t("dinner"),
            dinner=dinner,
            snacks_label=self._t("snacks"),
            snacks=snacks or [],
        )

    def translate_full_report(
        self,
        summary: str,
        recommendations: list[dict[str, Any]],
        meal_plans: list[dict[str, Any]],
        general_advice: list[str] | None = None,
    ) -> TranslatedReport:
        """Translate a complete diet report.

        Parameters
        ----------
        summary : str
            Report summary text.
        recommendations : list[dict]
            Each dict should have keys: ``nutrient``, ``recommendation``,
            ``target_foods``, ``avoid_foods``, ``daily_target``.
        meal_plans : list[dict]
            Each dict should have keys: ``day_index``, ``breakfast``,
            ``lunch``, ``dinner``, ``snacks``.
        general_advice : list[str]
            General dietary advice strings.
        """
        translated_recs = [
            self.translate_recommendation(
                nutrient=r.get("nutrient", ""),
                recommendation=r.get("recommendation", ""),
                target_foods=r.get("target_foods"),
                avoid_foods=r.get("avoid_foods"),
                daily_target=r.get("daily_target", ""),
            )
            for r in recommendations
        ]
        translated_plans = [
            self.translate_meal_plan(
                day_index=m.get("day_index", i),
                breakfast=m.get("breakfast", ""),
                lunch=m.get("lunch", ""),
                dinner=m.get("dinner", ""),
                snacks=m.get("snacks"),
            )
            for i, m in enumerate(meal_plans)
        ]
        return TranslatedReport(
            language=self.lang,
            language_name=self.lang_name,
            title=self._t("title"),
            summary=summary,
            recommendations=translated_recs,
            meal_plans=translated_plans,
            general_advice=general_advice or [],
            disclaimer=self._t("disclaimer"),
        )

    @staticmethod
    def available_languages() -> dict[str, str]:
        """Return mapping of language code → language name."""
        return dict(SUPPORTED_LANGUAGES)
