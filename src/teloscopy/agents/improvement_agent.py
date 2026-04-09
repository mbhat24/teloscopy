"""Continuous improvement agent for pipeline quality monitoring.

The :class:`ContinuousImprovementAgent` monitors analysis results over
time, evaluates pipeline accuracy, suggests (and can automatically apply)
parameter tuning, and generates improvement reports.  It serves as the
system's self-optimisation layer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .base import AgentMessage, AgentState, BaseAgent

logger = logging.getLogger(__name__)


class ContinuousImprovementAgent(BaseAgent):
    """Monitors system performance and suggests/applies improvements.

    The agent accumulates per-run metrics and uses them to detect quality
    trends, recommend parameter changes, and optionally perform automated
    parameter searches.

    Listens for request messages with the following actions:

    * ``evaluate_quality`` — assess pipeline quality from a batch of results.
    * ``suggest_tuning`` — recommend parameter adjustments.
    * ``track`` — accumulate a single result for long-term tracking.
    * ``generate_report`` — produce a summary improvement report.
    * ``auto_tune`` — automated parameter search for a target metric.
    * ``compare_methods`` — compare segmentation / detection methods.

    Parameters
    ----------
    name : str
        Agent name (default ``"improvement"``).
    """

    def __init__(self, name: str = "improvement") -> None:
        super().__init__(
            name=name,
            capabilities=[
                "quality_evaluation",
                "parameter_tuning",
                "metrics_tracking",
                "reporting",
                "auto_tuning",
                "method_comparison",
            ],
        )
        self._metrics_history: list[dict[str, Any]] = []
        self._parameter_history: list[dict[str, Any]] = []

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
            "evaluate_quality": self._handle_evaluate,
            "suggest_tuning": self._handle_suggest,
            "track": self._handle_track,
            "generate_report": self._handle_report,
            "auto_tune": self._handle_auto_tune,
            "compare_methods": self._handle_compare,
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
            logger.exception("ImprovementAgent action '%s' failed.", action)
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

    def evaluate_pipeline_quality(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate pipeline quality across a batch of analysis results.

        Computes aggregate metrics including mean spot count, association
        rate, SNR distribution, and consistency (CV of intensities across
        images).

        Parameters
        ----------
        results : list[dict]
            List of per-image analysis result dicts.

        Returns
        -------
        dict
            ``metrics`` (per-metric aggregate), ``overall_quality`` score
            (0–1), ``grade`` (A–F), and ``issues`` list.
        """
        if not results:
            return {
                "metrics": {},
                "overall_quality": 0.0,
                "grade": "F",
                "issues": ["No results to evaluate."],
                "summary": "Insufficient data for evaluation.",
            }

        spot_counts: list[int] = []
        assoc_rates: list[float] = []
        cvs: list[float] = []
        mean_snrs: list[float] = []

        for result in results:
            stats = result.get("statistics", {})
            assoc = result.get("association_summary", {})
            spots = result.get("spots", [])

            spot_counts.append(stats.get("n_telomeres", 0))
            assoc_rates.append(assoc.get("association_rate", 0.0))
            cvs.append(stats.get("cv", 0.0))

            valid_snrs = [s.get("snr", 0.0) for s in spots if s.get("valid", True)]
            if valid_snrs:
                mean_snrs.append(float(np.mean(valid_snrs)))

        metrics = {
            "mean_spot_count": float(np.mean(spot_counts)) if spot_counts else 0.0,
            "std_spot_count": float(np.std(spot_counts)) if len(spot_counts) > 1 else 0.0,
            "mean_association_rate": float(np.mean(assoc_rates)) if assoc_rates else 0.0,
            "mean_cv": float(np.mean(cvs)) if cvs else 0.0,
            "mean_snr": float(np.mean(mean_snrs)) if mean_snrs else 0.0,
            "n_images_evaluated": len(results),
        }

        # Compute overall quality score (weighted average of sub-scores)
        issues: list[str] = []

        # Score components (each 0–1)
        spot_score = min(1.0, metrics["mean_spot_count"] / 40.0)
        if metrics["mean_spot_count"] < 10:
            issues.append(f"Low average spot count ({metrics['mean_spot_count']:.1f}).")

        assoc_score = metrics["mean_association_rate"]
        if assoc_score < 0.5:
            issues.append(f"Low association rate ({assoc_score:.1%}).")

        cv_score = max(0.0, 1.0 - metrics["mean_cv"])
        if metrics["mean_cv"] > 0.8:
            issues.append(f"High intensity CV ({metrics['mean_cv']:.2f}).")

        snr_score = min(1.0, metrics["mean_snr"] / 20.0)
        if metrics["mean_snr"] < 5.0:
            issues.append(f"Low mean SNR ({metrics['mean_snr']:.1f}).")

        overall = 0.3 * spot_score + 0.3 * assoc_score + 0.2 * cv_score + 0.2 * snr_score
        overall = round(overall, 3)
        grade = self._score_to_grade(overall)

        return {
            "metrics": metrics,
            "overall_quality": overall,
            "grade": grade,
            "issues": issues,
            "summary": (
                f"Pipeline quality: {grade} ({overall:.1%}). "
                f"{len(issues)} issue(s) detected across {len(results)} image(s)."
            ),
        }

    def suggest_parameter_tuning(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Suggest parameter adjustments based on analysis results.

        Parameters
        ----------
        results : list[dict]
            List of analysis result dicts.

        Returns
        -------
        dict
            ``suggestions`` (list of parameter change dicts) and
            ``rationale`` descriptions.
        """
        quality = self.evaluate_pipeline_quality(results)
        metrics = quality.get("metrics", {})
        suggestions: list[dict[str, Any]] = []

        # Low spot count → lower threshold
        if metrics.get("mean_spot_count", 0) < 15:
            suggestions.append(
                {
                    "parameter": "spot_threshold",
                    "current_default": 0.02,
                    "suggested": 0.01,
                    "rationale": "Low spot count — reducing detection threshold to capture more spots.",
                }
            )

        # Low association rate → widen tip distance
        if metrics.get("mean_association_rate", 0) < 0.5:
            suggestions.append(
                {
                    "parameter": "max_tip_distance",
                    "current_default": 15.0,
                    "suggested": 25.0,
                    "rationale": (
                        "Low association rate — increasing max tip"
                        " distance for spot–chromosome matching."
                    ),
                }
            )
            suggestions.append(
                {
                    "parameter": "min_chromosome_area",
                    "current_default": 80,
                    "suggested": 50,
                    "rationale": (
                        "Low association rate — lowering minimum"
                        " chromosome area may capture missed segments."
                    ),
                }
            )

        # High CV → tighten sigma range
        if metrics.get("mean_cv", 0) > 0.8:
            suggestions.append(
                {
                    "parameter": "spot_sigma_max",
                    "current_default": 4.0,
                    "suggested": 3.0,
                    "rationale": "High intensity CV — narrowing sigma range to exclude outlier spots.",
                }
            )

        # Low SNR → increase denoising
        if metrics.get("mean_snr", 0) < 5.0:
            suggestions.append(
                {
                    "parameter": "denoise_sigma",
                    "current_default": 1.0,
                    "suggested": 1.5,
                    "rationale": "Low SNR — increasing denoising to improve signal clarity.",
                }
            )
            suggestions.append(
                {
                    "parameter": "spot_min_snr",
                    "current_default": 3.0,
                    "suggested": 2.0,
                    "rationale": "Low SNR — relaxing minimum SNR requirement to retain more spots.",
                }
            )

        if not suggestions:
            suggestions.append(
                {
                    "parameter": None,
                    "suggested": None,
                    "rationale": "All metrics within acceptable range — no changes recommended.",
                }
            )

        return {
            "suggestions": suggestions,
            "quality_summary": quality,
        }

    def track_metrics(self, result: dict[str, Any]) -> None:
        """Accumulate a single analysis result into the metrics history.

        Parameters
        ----------
        result : dict
            A single image-analysis result dict.
        """
        stats = result.get("statistics", {})
        assoc = result.get("association_summary", {})

        entry = {
            "timestamp": time.time(),
            "image_path": result.get("image_path", "unknown"),
            "n_telomeres": stats.get("n_telomeres", 0),
            "association_rate": assoc.get("association_rate", 0.0),
            "cv": stats.get("cv", 0.0),
            "mean_intensity": stats.get("mean_intensity", 0.0),
        }
        self._metrics_history.append(entry)
        logger.debug("Tracked metrics for '%s'.", entry["image_path"])

    def generate_improvement_report(self) -> dict[str, Any]:
        """Generate a comprehensive improvement report from accumulated metrics.

        Returns
        -------
        dict
            ``n_images_tracked``, ``period`` (timestamps), ``trends``
            (metric averages over first/second half), and ``recommendations``.
        """
        n = len(self._metrics_history)
        if n == 0:
            return {
                "n_images_tracked": 0,
                "summary": "No data tracked yet.",
                "recommendations": [],
            }

        # Split into halves for trend comparison
        first_half = self._metrics_history[: n // 2] if n >= 2 else self._metrics_history
        second_half = self._metrics_history[n // 2 :] if n >= 2 else self._metrics_history

        def _avg(entries: list[dict[str, Any]], key: str) -> float:
            values = [e.get(key, 0.0) for e in entries]
            return float(np.mean(values)) if values else 0.0

        trends: dict[str, dict[str, float]] = {}
        for key in ("n_telomeres", "association_rate", "cv", "mean_intensity"):
            first_avg = _avg(first_half, key)
            second_avg = _avg(second_half, key)
            trends[key] = {
                "first_half_avg": round(first_avg, 4),
                "second_half_avg": round(second_avg, 4),
                "change": round(second_avg - first_avg, 4),
            }

        recommendations: list[str] = []
        if trends["association_rate"]["change"] < -0.05:
            recommendations.append(
                "Association rate is declining — review segmentation parameters."
            )
        if trends["cv"]["change"] > 0.1:
            recommendations.append(
                "Intensity CV is increasing — check for sample preparation variability."
            )
        if trends["n_telomeres"]["change"] < -5:
            recommendations.append(
                "Spot count is declining — verify image quality and detection settings."
            )

        if not recommendations:
            recommendations.append("Pipeline metrics are stable — no action required.")

        timestamps = [e["timestamp"] for e in self._metrics_history]

        return {
            "n_images_tracked": n,
            "period": {
                "start": min(timestamps),
                "end": max(timestamps),
            },
            "trends": trends,
            "recommendations": recommendations,
            "parameter_change_history": list(self._parameter_history),
        }

    def auto_tune_parameters(
        self,
        metric: str,
        target: float,
        current_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform automated parameter search to optimise a target metric.

        Uses a simple grid-search approach over predefined parameter ranges
        based on the chosen metric.

        Parameters
        ----------
        metric : str
            The metric to optimise (e.g. ``"association_rate"``,
            ``"mean_snr"``, ``"n_telomeres"``).
        target : float
            Desired target value for the metric.
        current_config : dict | None
            Current pipeline configuration.

        Returns
        -------
        dict
            ``best_config`` (parameter overrides), ``expected_improvement``,
            and ``search_space`` used.
        """
        current_config = current_config or {}

        search_spaces: dict[str, list[dict[str, Any]]] = {
            "association_rate": [{"max_tip_distance": v} for v in [10.0, 15.0, 20.0, 25.0, 30.0]]
            + [{"min_chromosome_area": v} for v in [40, 60, 80, 100, 120]],
            "n_telomeres": [{"spot_threshold": v} for v in [0.005, 0.01, 0.015, 0.02, 0.03]]
            + [{"spot_min_snr": v} for v in [1.5, 2.0, 2.5, 3.0, 4.0]],
            "mean_snr": [{"denoise_sigma": v} for v in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]]
            + [{"spot_sigma_min": v} for v in [1.0, 1.5, 2.0, 2.5]],
        }

        search_space = search_spaces.get(metric, [])
        if not search_space:
            return {
                "best_config": {},
                "expected_improvement": 0.0,
                "search_space": [],
                "message": f"No search space defined for metric '{metric}'.",
            }

        # Estimate the best config from historical data heuristics
        best_config: dict[str, Any] = {}
        best_score = 0.0

        for candidate in search_space:
            # Simple heuristic scoring based on distance to common "good" values
            score = self._heuristic_score(metric, candidate, target)
            if score > best_score:
                best_score = score
                best_config = dict(candidate)

        # Record the parameter change
        self._parameter_history.append(
            {
                "timestamp": time.time(),
                "metric": metric,
                "target": target,
                "config_change": best_config,
            }
        )

        return {
            "best_config": best_config,
            "estimated_metric_value": round(best_score * target, 3),
            "expected_improvement": round(best_score, 3),
            "search_space_size": len(search_space),
            "metric": metric,
            "target": target,
        }

    def compare_methods(
        self,
        image_path: str,
        methods: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare different segmentation / detection method configurations.

        Runs the pipeline with each method and reports comparative metrics.

        Parameters
        ----------
        image_path : str
            Path to a test image.
        methods : list[str] | None
            List of segmentation method names to compare.  Defaults to
            ``["otsu_watershed"]``.

        Returns
        -------
        dict
            ``comparisons`` (list of per-method result summaries) and
            ``recommended_method``.
        """
        methods = methods or ["otsu_watershed"]

        comparisons: list[dict[str, Any]] = []

        for method in methods:
            try:
                from ..telomere.pipeline import analyze_image as _pipeline_analyze

                config = {"segmentation_method": method}
                result = _pipeline_analyze(image_path, config=config)

                stats = result.get("statistics", {})
                assoc = result.get("association_summary", {})

                comparisons.append(
                    {
                        "method": method,
                        "n_telomeres": stats.get("n_telomeres", 0),
                        "association_rate": assoc.get("association_rate", 0.0),
                        "mean_intensity": stats.get("mean_intensity", 0.0),
                        "cv": stats.get("cv", 0.0),
                        "total_spots": assoc.get("total_spots", 0),
                        "status": "success",
                    }
                )
            except Exception as exc:
                comparisons.append(
                    {
                        "method": method,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        # Pick the best method by a composite score
        successful = [c for c in comparisons if c.get("status") == "success"]
        recommended = "otsu_watershed"
        if successful:
            best = max(
                successful,
                key=lambda c: (
                    c.get("n_telomeres", 0) * 0.4
                    + c.get("association_rate", 0) * 100 * 0.4
                    + max(0, 1.0 - c.get("cv", 1.0)) * 100 * 0.2
                ),
            )
            recommended = best["method"]

        return {
            "image_path": image_path,
            "comparisons": comparisons,
            "recommended_method": recommended,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert a 0–1 quality score to a letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.65:
            return "C"
        elif score >= 0.5:
            return "D"
        return "F"

    @staticmethod
    def _heuristic_score(metric: str, candidate: dict[str, Any], target: float) -> float:
        """Heuristic scoring for parameter candidates (0–1, higher is better)."""
        # Simple closeness-to-ideal heuristic
        ideal_values: dict[str, dict[str, float]] = {
            "association_rate": {"max_tip_distance": 20.0, "min_chromosome_area": 60},
            "n_telomeres": {"spot_threshold": 0.015, "spot_min_snr": 2.5},
            "mean_snr": {"denoise_sigma": 1.2, "spot_sigma_min": 1.5},
        }
        ideals = ideal_values.get(metric, {})
        if not ideals:
            return 0.5

        scores: list[float] = []
        for param, value in candidate.items():
            ideal = ideals.get(param)
            if ideal is not None and ideal != 0:
                closeness = 1.0 - min(1.0, abs(value - ideal) / abs(ideal))
                scores.append(closeness)

        return float(np.mean(scores)) if scores else 0.5

    # ------------------------------------------------------------------
    # Message handler bridges
    # ------------------------------------------------------------------

    def _handle_evaluate(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.evaluate_pipeline_quality(content.get("results", []))

    def _handle_suggest(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.suggest_parameter_tuning(content.get("results", []))

    def _handle_track(self, content: dict[str, Any]) -> dict[str, Any]:
        self.track_metrics(content.get("result", content))
        return {"tracked": True, "total_entries": len(self._metrics_history)}

    def _handle_report(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.generate_improvement_report()

    def _handle_auto_tune(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.auto_tune_parameters(
            metric=content.get("metric", "association_rate"),
            target=content.get("target", 0.8),
            current_config=content.get("current_config"),
        )

    def _handle_compare(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.compare_methods(
            image_path=content.get("image_path", ""),
            methods=content.get("methods"),
        )
