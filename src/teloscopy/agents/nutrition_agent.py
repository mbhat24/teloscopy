"""Nutrition agent for dietary recommendations based on genetic profiles.

The :class:`NutritionAgent` translates genetic risk profiles and telomere
health data into actionable, region-aware dietary plans.  It maintains an
internal database of telomere-protective foods mapped to geographic
regions and generates personalised meal plans that account for individual
dietary restrictions and nutritional gaps.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentMessage, AgentState, BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regional food databases (simplified reference data)
# ---------------------------------------------------------------------------

_TELOMERE_PROTECTIVE_FOODS: dict[str, list[dict[str, Any]]] = {
    "global": [
        {"name": "Blueberries", "nutrients": ["anthocyanins", "vitamin_c"], "category": "fruit"},
        {"name": "Spinach", "nutrients": ["folate", "iron", "vitamin_k"], "category": "vegetable"},
        {
            "name": "Salmon",
            "nutrients": ["omega_3", "vitamin_d", "selenium"],
            "category": "protein",
        },
        {
            "name": "Walnuts",
            "nutrients": ["omega_3", "vitamin_e", "polyphenols"],
            "category": "nuts",
        },
        {"name": "Green tea", "nutrients": ["EGCG", "catechins"], "category": "beverage"},
        {"name": "Turmeric", "nutrients": ["curcumin"], "category": "spice"},
        {"name": "Lentils", "nutrients": ["folate", "iron", "fiber"], "category": "legume"},
        {
            "name": "Dark chocolate (70%+)",
            "nutrients": ["flavonoids", "magnesium"],
            "category": "treat",
        },
    ],
    "mediterranean": [
        {
            "name": "Extra virgin olive oil",
            "nutrients": ["oleic_acid", "polyphenols"],
            "category": "fat",
        },
        {
            "name": "Sardines",
            "nutrients": ["omega_3", "calcium", "vitamin_d"],
            "category": "protein",
        },
        {"name": "Tomatoes", "nutrients": ["lycopene", "vitamin_c"], "category": "vegetable"},
        {"name": "Red grapes", "nutrients": ["resveratrol", "anthocyanins"], "category": "fruit"},
        {"name": "Chickpeas", "nutrients": ["folate", "fiber", "zinc"], "category": "legume"},
        {"name": "Figs", "nutrients": ["calcium", "potassium", "fiber"], "category": "fruit"},
    ],
    "east_asian": [
        {
            "name": "Edamame",
            "nutrients": ["isoflavones", "folate", "protein"],
            "category": "legume",
        },
        {
            "name": "Matcha",
            "nutrients": ["EGCG", "L-theanine", "catechins"],
            "category": "beverage",
        },
        {
            "name": "Seaweed (nori)",
            "nutrients": ["iodine", "selenium", "omega_3"],
            "category": "vegetable",
        },
        {"name": "Miso", "nutrients": ["probiotics", "isoflavones"], "category": "fermented"},
        {
            "name": "Sweet potato",
            "nutrients": ["beta_carotene", "vitamin_c", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Shiitake mushrooms",
            "nutrients": ["vitamin_d", "selenium", "beta_glucans"],
            "category": "vegetable",
        },
    ],
    "south_asian": [
        {"name": "Turmeric (haldi)", "nutrients": ["curcumin"], "category": "spice"},
        {
            "name": "Moringa leaves",
            "nutrients": ["vitamin_c", "iron", "calcium"],
            "category": "vegetable",
        },
        {
            "name": "Amla (Indian gooseberry)",
            "nutrients": ["vitamin_c", "polyphenols"],
            "category": "fruit",
        },
        {"name": "Mung dal", "nutrients": ["folate", "protein", "fiber"], "category": "legume"},
        {
            "name": "Fenugreek seeds",
            "nutrients": ["fiber", "iron", "galactomannan"],
            "category": "spice",
        },
        {
            "name": "Curd (yoghurt)",
            "nutrients": ["probiotics", "calcium", "B12"],
            "category": "dairy",
        },
    ],
    "americas": [
        {
            "name": "Avocado",
            "nutrients": ["monounsaturated_fat", "folate", "potassium"],
            "category": "fruit",
        },
        {
            "name": "Black beans",
            "nutrients": ["anthocyanins", "folate", "fiber"],
            "category": "legume",
        },
        {"name": "Quinoa", "nutrients": ["protein", "magnesium", "folate"], "category": "grain"},
        {"name": "Chia seeds", "nutrients": ["omega_3", "fiber", "calcium"], "category": "seed"},
        {"name": "Acai berries", "nutrients": ["anthocyanins", "omega_3"], "category": "fruit"},
        {
            "name": "Wild salmon",
            "nutrients": ["omega_3", "astaxanthin", "vitamin_d"],
            "category": "protein",
        },
    ],
    "african": [
        {
            "name": "Baobab fruit",
            "nutrients": ["vitamin_c", "fiber", "calcium"],
            "category": "fruit",
        },
        {"name": "Moringa", "nutrients": ["vitamin_c", "iron", "calcium"], "category": "vegetable"},
        {"name": "Teff", "nutrients": ["iron", "calcium", "fiber"], "category": "grain"},
        {
            "name": "Cowpeas (black-eyed peas)",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {"name": "Okra", "nutrients": ["folate", "vitamin_c", "fiber"], "category": "vegetable"},
        {
            "name": "Hibiscus tea",
            "nutrients": ["anthocyanins", "vitamin_c"],
            "category": "beverage",
        },
    ],
}

_RISK_NUTRIENT_MAP: dict[str, list[str]] = {
    "cardiovascular": ["omega_3", "fiber", "magnesium", "potassium", "polyphenols"],
    "cancer": ["anthocyanins", "vitamin_c", "selenium", "curcumin", "sulforaphane"],
    "neurodegenerative": ["omega_3", "vitamin_d", "vitamin_e", "flavonoids", "B12"],
    "immune_dysfunction": ["vitamin_d", "zinc", "selenium", "probiotics", "vitamin_c"],
    "premature_aging": ["folate", "B12", "vitamin_c", "zinc", "EGCG"],
}


class NutritionAgent(BaseAgent):
    """Handles dietary recommendations based on genetic profile and geography.

    Listens for request messages with the following actions:

    * ``generate_diet_plan`` — build a personalised dietary plan.
    * ``get_protective_foods`` — list telomere-protective foods for a region.
    * ``adapt_to_preferences`` — adjust a plan for dietary restrictions.
    * ``calculate_gaps`` — identify nutritional gaps.

    Parameters
    ----------
    name : str
        Agent name (default ``"nutrition"``).
    """

    def __init__(self, name: str = "nutrition") -> None:
        super().__init__(
            name=name,
            capabilities=[
                "diet_planning",
                "nutritional_analysis",
                "regional_adaptation",
                "gap_analysis",
            ],
        )

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, msg: AgentMessage) -> None:
        """Route an incoming request to the appropriate handler.

        Parameters
        ----------
        msg : AgentMessage
            Incoming message.
        """
        action = msg.content.get("action", "")
        handlers: dict[str, Any] = {
            "generate_diet_plan": self._handle_diet_plan,
            "get_protective_foods": self._handle_protective_foods,
            "adapt_to_preferences": self._handle_adapt,
            "calculate_gaps": self._handle_gaps,
        }

        handler = handlers.get(action)
        if handler is None:
            await self.send_message(
                recipient=msg.sender,
                content={"error": f"Unknown action '{action}'."},
                msg_type="error",
                correlation_id=msg.correlation_id,
            )
            return

        self.state = AgentState.RUNNING
        try:
            result = handler(msg.content)
            await self.send_message(
                recipient=msg.sender,
                content=result,
                msg_type="response",
                correlation_id=msg.correlation_id,
            )
        except Exception as exc:
            logger.exception("NutritionAgent action '%s' failed.", action)
            await self.send_message(
                recipient=msg.sender,
                content={"error": str(exc), "action": action},
                msg_type="error",
                correlation_id=msg.correlation_id,
            )
        finally:
            self.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_diet_plan(
        self,
        genetic_risks: list[dict[str, Any]],
        region: str = "global",
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a personalised dietary plan based on genetic risks and geography.

        The plan prioritises nutrients that address the user's highest-risk
        conditions, sourced from foods available in the specified region.

        Parameters
        ----------
        genetic_risks : list[dict]
            Per-disease risk dicts (from :class:`GenomicsAgent`).
        region : str
            Geographic region for food sourcing (e.g. ``"mediterranean"``,
            ``"east_asian"``).  Falls back to ``"global"`` if unknown.
        profile : dict | None
            User metadata (``age``, ``sex``, ``dietary_restrictions``).

        Returns
        -------
        dict
            ``priority_nutrients``, ``recommended_foods``, ``meal_plan``,
            and ``notes``.
        """
        profile = profile or {}

        # Determine priority nutrients from top risks
        priority_nutrients = self._get_priority_nutrients(genetic_risks)

        # Get region-appropriate protective foods
        foods = self.get_telomere_protective_foods(region)

        # Score and rank foods by how many priority nutrients they supply
        scored_foods = self._score_foods(foods, priority_nutrients)

        # Build a simple daily meal plan
        meal_plan = self._build_meal_plan(scored_foods, profile)

        # Apply dietary restrictions
        restrictions = profile.get("dietary_restrictions", [])
        if restrictions:
            meal_plan = self._apply_restrictions(meal_plan, restrictions)

        return {
            "region": region,
            "priority_nutrients": priority_nutrients,
            "recommended_foods": scored_foods[:12],
            "meal_plan": meal_plan,
            "notes": self._generate_notes(genetic_risks, profile),
        }

    def get_telomere_protective_foods(self, region: str = "global") -> list[dict[str, Any]]:
        """Return telomere-protective foods available in the specified region.

        Parameters
        ----------
        region : str
            Geographic region identifier.

        Returns
        -------
        list[dict]
            Food entries with ``name``, ``nutrients``, and ``category``.
        """
        # Merge global foods with region-specific entries
        foods = list(_TELOMERE_PROTECTIVE_FOODS.get("global", []))
        regional = _TELOMERE_PROTECTIVE_FOODS.get(region, [])
        # Avoid duplicates by name
        existing_names = {f["name"] for f in foods}
        for item in regional:
            if item["name"] not in existing_names:
                foods.append(item)
                existing_names.add(item["name"])
        return foods

    def adapt_to_preferences(
        self,
        plan: dict[str, Any],
        restrictions: list[str],
    ) -> dict[str, Any]:
        """Adapt an existing diet plan to dietary restrictions.

        Supported restriction keywords: ``"vegetarian"``, ``"vegan"``,
        ``"gluten_free"``, ``"dairy_free"``, ``"nut_free"``,
        ``"shellfish_free"``.

        Parameters
        ----------
        plan : dict
            A diet plan as returned by :meth:`generate_diet_plan`.
        restrictions : list[str]
            List of restriction keywords.

        Returns
        -------
        dict
            Adjusted plan with unsuitable foods removed or substituted.
        """
        if not restrictions:
            return plan

        adapted = dict(plan)
        adapted["meal_plan"] = self._apply_restrictions(plan.get("meal_plan", {}), restrictions)

        # Filter recommended foods
        adapted["recommended_foods"] = [
            f for f in plan.get("recommended_foods", []) if self._food_allowed(f, restrictions)
        ]

        adapted["dietary_restrictions_applied"] = restrictions
        return adapted

    def calculate_nutritional_gaps(
        self,
        current_diet: dict[str, Any],
        genetic_needs: list[str],
    ) -> dict[str, Any]:
        """Identify nutritional gaps between current diet and genetic needs.

        Parameters
        ----------
        current_diet : dict
            Mapping of nutrient names to approximate daily intake levels
            (e.g. ``{"omega_3": "low", "folate": "adequate"}``).
        genetic_needs : list[str]
            Priority nutrients derived from genetic risk analysis.

        Returns
        -------
        dict
            ``gaps`` (list of under-consumed nutrients), ``adequate``
            (list of met nutrients), ``recommendations`` (targeted
            suggestions).
        """
        gaps: list[str] = []
        adequate: list[str] = []

        for nutrient in genetic_needs:
            level = current_diet.get(nutrient, "unknown")
            if level in ("low", "deficient", "unknown"):
                gaps.append(nutrient)
            else:
                adequate.append(nutrient)

        recommendations: list[str] = []
        for gap in gaps:
            recommendations.append(
                f"Increase intake of {gap.replace('_', ' ')} through diet or supplementation."
            )

        return {
            "gaps": gaps,
            "adequate": adequate,
            "total_needs": len(genetic_needs),
            "gap_count": len(gaps),
            "gap_percentage": round(len(gaps) / max(len(genetic_needs), 1) * 100, 1),
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_priority_nutrients(risks: list[dict[str, Any]]) -> list[str]:
        """Determine priority nutrients from genetic risk scores."""
        nutrient_scores: dict[str, float] = {}
        for risk in risks:
            disease = risk.get("disease", "")
            score = risk.get("risk_score", 0.0)
            for nutrient in _RISK_NUTRIENT_MAP.get(disease, []):
                nutrient_scores[nutrient] = nutrient_scores.get(nutrient, 0.0) + score

        # Sort by cumulative relevance
        sorted_nutrients = sorted(nutrient_scores, key=nutrient_scores.get, reverse=True)  # type: ignore[arg-type]
        return sorted_nutrients

    @staticmethod
    def _score_foods(
        foods: list[dict[str, Any]],
        priority_nutrients: list[str],
    ) -> list[dict[str, Any]]:
        """Score foods by how many priority nutrients they provide."""
        priority_set = set(priority_nutrients[:10])  # top 10 nutrients
        scored: list[dict[str, Any]] = []

        for food in foods:
            food_nutrients = set(food.get("nutrients", []))
            overlap = food_nutrients & priority_set
            scored.append(
                {
                    **food,
                    "relevance_score": len(overlap),
                    "matching_nutrients": sorted(overlap),
                }
            )

        scored.sort(key=lambda f: f["relevance_score"], reverse=True)
        return scored

    @staticmethod
    def _build_meal_plan(
        scored_foods: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a simple daily meal plan from scored foods."""
        meals: dict[str, list[str]] = {
            "breakfast": [],
            "lunch": [],
            "dinner": [],
            "snacks": [],
        }

        category_meal_map: dict[str, str] = {
            "grain": "breakfast",
            "fruit": "breakfast",
            "beverage": "breakfast",
            "vegetable": "lunch",
            "legume": "lunch",
            "protein": "dinner",
            "fat": "dinner",
            "fermented": "lunch",
            "nuts": "snacks",
            "seed": "snacks",
            "spice": "dinner",
            "treat": "snacks",
            "dairy": "breakfast",
        }

        for food in scored_foods[:16]:
            category = food.get("category", "snacks")
            meal = category_meal_map.get(category, "snacks")
            if len(meals[meal]) < 4:
                meals[meal].append(food["name"])

        return {
            "daily_plan": meals,
            "servings_note": "Aim for 5+ servings of fruits/vegetables daily.",
            "hydration": "8+ glasses of water; consider green tea for EGCG.",
        }

    @staticmethod
    def _apply_restrictions(
        meal_plan: dict[str, Any],
        restrictions: list[str],
    ) -> dict[str, Any]:
        """Remove foods that violate dietary restrictions."""
        restriction_filters: dict[str, set[str]] = {
            "vegetarian": {"Salmon", "Sardines", "Wild salmon"},
            "vegan": {
                "Salmon",
                "Sardines",
                "Wild salmon",
                "Curd (yoghurt)",
                "Dark chocolate (70%+)",
                "Miso",
            },
            "gluten_free": {"Teff"},  # teff is actually gluten-free; placeholder
            "dairy_free": {"Curd (yoghurt)"},
            "nut_free": {"Walnuts"},
            "shellfish_free": set(),
        }

        excluded: set[str] = set()
        for restriction in restrictions:
            excluded |= restriction_filters.get(restriction, set())

        if not excluded:
            return meal_plan

        adapted = dict(meal_plan)
        daily = adapted.get("daily_plan", {})
        for meal_name, items in daily.items():
            if isinstance(items, list):
                daily[meal_name] = [i for i in items if i not in excluded]
        adapted["daily_plan"] = daily
        adapted["excluded_foods"] = sorted(excluded)
        return adapted

    @staticmethod
    def _food_allowed(food: dict[str, Any], restrictions: list[str]) -> bool:
        """Check whether a food item is compatible with restrictions."""
        category_restrictions: dict[str, set[str]] = {
            "vegan": {"protein", "dairy"},
            "vegetarian": {"protein"},
            "dairy_free": {"dairy"},
        }
        category = food.get("category", "")
        for restriction in restrictions:
            disallowed = category_restrictions.get(restriction, set())
            if category in disallowed:
                return False
        return True

    @staticmethod
    def _generate_notes(
        risks: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> list[str]:
        """Generate contextual notes for the diet plan."""
        notes: list[str] = [
            "This plan is a guideline — consult a registered dietitian for personalisation.",
        ]

        high_risks = [r for r in risks if r.get("risk_score", 0) >= 0.5]
        if high_risks:
            diseases = ", ".join(r["disease"].replace("_", " ") for r in high_risks)
            notes.append(
                f"Elevated risk detected for: {diseases}. Dietary focus adjusted accordingly."
            )

        age = profile.get("age", 0)
        if age > 60:
            notes.append("Over 60: consider B12 and vitamin D supplementation.")
        elif age < 25:
            notes.append("Under 25: focus on growth-supportive nutrients (iron, calcium, protein).")

        return notes

    # ------------------------------------------------------------------
    # Message handler bridges
    # ------------------------------------------------------------------

    def _handle_diet_plan(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.generate_diet_plan(
            genetic_risks=content.get("genetic_risks", []),
            region=content.get("region", "global"),
            profile=content.get("profile"),
        )

    def _handle_protective_foods(self, content: dict[str, Any]) -> dict[str, Any]:
        foods = self.get_telomere_protective_foods(content.get("region", "global"))
        return {"foods": foods, "region": content.get("region", "global")}

    def _handle_adapt(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.adapt_to_preferences(
            plan=content.get("plan", {}),
            restrictions=content.get("restrictions", []),
        )

    def _handle_gaps(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.calculate_nutritional_gaps(
            current_diet=content.get("current_diet", {}),
            genetic_needs=content.get("genetic_needs", []),
        )
