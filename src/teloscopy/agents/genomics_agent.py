"""Genomics agent for disease-risk prediction and genetic analysis.

The :class:`GenomicsAgent` integrates telomere-length measurements with
single-nucleotide polymorphism (SNP) data and user-profile metadata to
compute multi-dimensional disease-risk scores, project health trajectories,
and generate prevention recommendations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .base import AgentMessage, AgentState, BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference risk models (simplified look-up tables)
# ---------------------------------------------------------------------------

_TELOMERE_RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    "cardiovascular": {"short": 6000.0, "very_short": 4000.0, "weight": 0.3},
    "cancer": {"short": 5500.0, "very_short": 3500.0, "weight": 0.25},
    "neurodegenerative": {"short": 5000.0, "very_short": 3000.0, "weight": 0.2},
    "immune_dysfunction": {"short": 5500.0, "very_short": 3500.0, "weight": 0.15},
    "premature_aging": {"short": 6500.0, "very_short": 4500.0, "weight": 0.1},
}

_SNP_RISK_MODIFIERS: dict[str, dict[str, float]] = {
    "TERT": {"cardiovascular": 0.15, "cancer": 0.20},
    "TERC": {"cardiovascular": 0.10, "cancer": 0.15},
    "OBFC1": {"cardiovascular": 0.08},
    "CTC1": {"premature_aging": 0.12},
    "RTEL1": {"cancer": 0.10, "neurodegenerative": 0.08},
    "APOE": {"neurodegenerative": 0.25, "cardiovascular": 0.10},
}


class GenomicsAgent(BaseAgent):
    """Handles disease-risk prediction and genetic analysis.

    Listens for request messages with the following actions:

    * ``assess_risk`` — compute disease-risk scores from telomere and SNP data.
    * ``project_timeline`` — project a multi-year health trajectory.
    * ``get_recommendations`` — generate prevention recommendations.
    * ``integrate`` — merge telomere results with SNP-level data.

    Parameters
    ----------
    name : str
        Agent name (default ``"genomics"``).
    """

    def __init__(self, name: str = "genomics") -> None:
        super().__init__(
            name=name,
            capabilities=[
                "risk_assessment",
                "health_projection",
                "prevention_recommendations",
                "snp_integration",
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
            "assess_risk": self._handle_assess_risk,
            "project_timeline": self._handle_project_timeline,
            "get_recommendations": self._handle_recommendations,
            "integrate": self._handle_integrate,
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
            logger.exception("GenomicsAgent action '%s' failed.", action)
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

    def assess_risk(
        self,
        telomere_data: dict[str, Any],
        variants: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute disease-risk scores from telomere measurements and genetic data.

        The risk model combines:
        * **Telomere-length component** — shorter telomeres increase risk.
        * **SNP modifier component** — known risk alleles adjust scores.
        * **Age/sex adjustment** — demographic factors modulate baseline.

        Parameters
        ----------
        telomere_data : dict
            Must contain ``mean_intensity`` (or ``mean_length_bp`` if
            calibrated).  Typically the ``statistics`` sub-dict from an
            image-analysis result.
        variants : dict | None
            Gene → allele mapping (e.g. ``{"TERT": "risk", "APOE": "e4"}``).
        profile : dict | None
            User metadata with optional ``age`` (int) and ``sex`` (str).

        Returns
        -------
        dict
            Top-level keys: ``risks`` (list of per-disease dicts),
            ``overall_risk_score`` (float 0–1), ``risk_category`` (str),
            ``telomere_percentile`` (float).
        """
        variants = variants or {}
        profile = profile or {}

        # Extract telomere length proxy
        mean_length = telomere_data.get(
            "mean_length_bp",
            telomere_data.get("mean_intensity", 0.0),
        )

        # Estimate percentile (very simplified model)
        telomere_percentile = self._estimate_percentile(mean_length, profile.get("age", 40))

        risks: list[dict[str, Any]] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for disease, thresholds in _TELOMERE_RISK_THRESHOLDS.items():
            base_score = 0.0
            if mean_length < thresholds["very_short"]:
                base_score = 0.8
            elif mean_length < thresholds["short"]:
                base_score = 0.4
            else:
                base_score = 0.1

            # SNP modifiers
            snp_modifier = 0.0
            for gene, allele in variants.items():
                gene_risks = _SNP_RISK_MODIFIERS.get(gene, {})
                if disease in gene_risks and allele in ("risk", "e4", "heterozygous"):
                    snp_modifier += gene_risks[disease]

            # Age adjustment (risk increases with age)
            age = profile.get("age", 40)
            age_factor = 1.0 + max(0, (age - 30)) * 0.005

            final_score = min(1.0, (base_score + snp_modifier) * age_factor)
            weight = thresholds["weight"]

            risks.append(
                {
                    "disease": disease,
                    "risk_score": round(final_score, 3),
                    "base_score": round(base_score, 3),
                    "snp_modifier": round(snp_modifier, 3),
                    "age_factor": round(age_factor, 3),
                    "category": self._categorise_risk(final_score),
                }
            )

            weighted_sum += final_score * weight
            total_weight += weight

        overall = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0

        return {
            "risks": risks,
            "overall_risk_score": overall,
            "risk_category": self._categorise_risk(overall),
            "telomere_percentile": round(telomere_percentile, 1),
            "mean_telomere_length": mean_length,
        }

    def project_health_timeline(
        self,
        risk_profile: dict[str, Any],
        years: int = 10,
    ) -> dict[str, Any]:
        """Project disease-risk trajectory over future years.

        Applies an exponential-decay model to the current risk scores,
        accounting for natural telomere attrition (~50–100 bp per year).

        Parameters
        ----------
        risk_profile : dict
            Output from :meth:`assess_risk`.
        years : int
            Projection horizon in years.

        Returns
        -------
        dict
            ``timeline`` (list of dicts per year) and ``summary``.
        """
        annual_attrition = 75.0  # bp/year average
        base_length = risk_profile.get("mean_telomere_length", 5000.0)

        timeline: list[dict[str, Any]] = []
        for year in range(years + 1):
            projected_length = base_length - (annual_attrition * year)
            # Re-assess risk at projected length
            projected_risk = self.assess_risk(
                telomere_data={"mean_length_bp": projected_length},
            )
            timeline.append(
                {
                    "year": year,
                    "projected_telomere_length": round(projected_length, 1),
                    "overall_risk_score": projected_risk["overall_risk_score"],
                    "risk_category": projected_risk["risk_category"],
                }
            )

        return {
            "timeline": timeline,
            "years_projected": years,
            "annual_attrition_bp": annual_attrition,
            "summary": (
                f"Over {years} years, projected telomere length decreases from "
                f"{base_length:.0f} bp to {base_length - annual_attrition * years:.0f} bp."
            ),
        }

    def get_prevention_recommendations(
        self,
        risks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate prevention recommendations for identified risks.

        Parameters
        ----------
        risks : list[dict]
            Per-disease risk dicts from :meth:`assess_risk`.

        Returns
        -------
        list[dict]
            Recommendations with ``disease``, ``priority``, and
            ``actions`` (list of strings).
        """
        recommendations_map: dict[str, list[str]] = {
            "cardiovascular": [
                "Regular aerobic exercise (150+ min/week moderate intensity).",
                "Mediterranean-style diet rich in omega-3 fatty acids.",
                "Monitor blood pressure and cholesterol annually.",
                "Stress management through meditation or yoga.",
            ],
            "cancer": [
                "Annual cancer screenings appropriate for age and risk.",
                "Anti-oxidant rich diet (berries, leafy greens, cruciferous vegetables).",
                "Avoid tobacco and limit alcohol consumption.",
                "Maintain healthy BMI (18.5–24.9).",
            ],
            "neurodegenerative": [
                "Cognitive engagement (puzzles, learning, social interaction).",
                "Regular physical exercise to maintain brain perfusion.",
                "Adequate sleep (7–9 hours) for amyloid clearance.",
                "Consider DHA/EPA supplementation.",
            ],
            "immune_dysfunction": [
                "Ensure adequate vitamin D levels (40–60 ng/mL).",
                "Probiotic-rich foods for gut-immune axis support.",
                "Vaccination schedule adherence.",
                "Chronic stress reduction.",
            ],
            "premature_aging": [
                "Telomere-supportive nutrients (folate, B12, vitamin C, zinc).",
                "UV protection and skin care.",
                "Caloric balance — avoid chronic caloric restriction or excess.",
                "Regular health monitoring and biomarker tracking.",
            ],
        }

        recommendations: list[dict[str, Any]] = []
        for risk in sorted(risks, key=lambda r: r.get("risk_score", 0), reverse=True):
            disease = risk.get("disease", "unknown")
            score = risk.get("risk_score", 0.0)

            if score >= 0.6:
                priority = "high"
            elif score >= 0.3:
                priority = "moderate"
            else:
                priority = "low"

            recommendations.append(
                {
                    "disease": disease,
                    "risk_score": score,
                    "priority": priority,
                    "actions": recommendations_map.get(disease, ["Consult a healthcare provider."]),
                }
            )

        return recommendations

    def integrate_telomere_with_snp(
        self,
        telomere_results: dict[str, Any],
        snp_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine telomere measurements with SNP-level variant data.

        Produces a unified profile that can be fed into risk assessment.

        Parameters
        ----------
        telomere_results : dict
            Image-analysis results (must include ``statistics``).
        snp_data : dict
            Variant information keyed by gene name.

        Returns
        -------
        dict
            Merged profile with ``telomere_summary``, ``variant_summary``,
            and ``integrated_features``.
        """
        stats = telomere_results.get("statistics", {})

        # Summarise telomere metrics
        telomere_summary = {
            "mean_intensity": stats.get("mean_intensity", 0.0),
            "median_intensity": stats.get("median_intensity", 0.0),
            "cv": stats.get("cv", 0.0),
            "n_telomeres": stats.get("n_telomeres", 0),
        }

        # Summarise variant data
        known_genes = set(_SNP_RISK_MODIFIERS.keys())
        relevant_variants = {
            gene: allele for gene, allele in snp_data.items() if gene in known_genes
        }

        risk_allele_count = sum(
            1 for allele in relevant_variants.values() if allele in ("risk", "e4", "heterozygous")
        )

        variant_summary = {
            "total_variants_checked": len(snp_data),
            "relevant_variants": len(relevant_variants),
            "risk_allele_count": risk_allele_count,
            "variants": relevant_variants,
        }

        return {
            "telomere_summary": telomere_summary,
            "variant_summary": variant_summary,
            "integrated_features": {
                **telomere_summary,
                "risk_allele_count": risk_allele_count,
                "relevant_variants": relevant_variants,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_percentile(length_bp: float, age: int) -> float:
        """Estimate the population percentile for a telomere length at a given age.

        Uses a simplified logistic model.  Real implementations would
        reference age- and sex-stratified normative databases.
        """
        # Approximate population median by age (very simplified)
        median_at_age = 7000.0 - (age * 30.0)
        std_at_age = 1500.0

        if std_at_age == 0:
            return 50.0

        z = (length_bp - median_at_age) / std_at_age
        # Convert z-score to percentile via logistic approximation
        percentile = 100.0 / (1.0 + math.exp(-1.7 * z))
        return max(0.1, min(99.9, percentile))

    @staticmethod
    def _categorise_risk(score: float) -> str:
        """Map a 0–1 risk score to a human-readable category."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "moderate"
        elif score >= 0.2:
            return "low"
        return "minimal"

    # ------------------------------------------------------------------
    # Message handler bridges
    # ------------------------------------------------------------------

    def _handle_assess_risk(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.assess_risk(
            telomere_data=content.get("telomere_data", {}),
            variants=content.get("variants"),
            profile=content.get("profile"),
        )

    def _handle_project_timeline(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.project_health_timeline(
            risk_profile=content.get("risk_profile", {}),
            years=content.get("years", 10),
        )

    def _handle_recommendations(self, content: dict[str, Any]) -> dict[str, Any]:
        recs = self.get_prevention_recommendations(content.get("risks", []))
        return {"recommendations": recs}

    def _handle_integrate(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.integrate_telomere_with_snp(
            telomere_results=content.get("telomere_results", {}),
            snp_data=content.get("snp_data", {}),
        )
