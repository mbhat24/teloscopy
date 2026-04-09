"""Report-generation agent for comprehensive analysis reports.

The :class:`ReportAgent` assembles results from image analysis, genomics
risk assessment, and nutrition planning into structured reports that can
be rendered as JSON, HTML, or include auto-generated visualisations.
"""

from __future__ import annotations

import html
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import AgentMessage, AgentState, BaseAgent

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """Generates comprehensive analysis reports.

    Listens for request messages with the following actions:

    * ``generate_full_report`` — compile analysis, risk, and diet data.
    * ``format_html`` — render a report dict as HTML.
    * ``format_json`` — normalise a report dict for JSON export.
    * ``create_visualizations`` — generate matplotlib plots and return paths.

    Parameters
    ----------
    name : str
        Agent name (default ``"report"``).
    output_dir : str | None
        Directory for persisted outputs (plots, HTML files).  If ``None``
        outputs are written to a temporary directory.
    """

    def __init__(
        self,
        name: str = "report",
        output_dir: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            capabilities=[
                "report_generation",
                "html_formatting",
                "json_formatting",
                "visualization",
            ],
        )
        self._output_dir = Path(output_dir) if output_dir else None

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
            "generate_full_report": self._handle_full_report,
            "format_html": self._handle_format_html,
            "format_json": self._handle_format_json,
            "create_visualizations": self._handle_visualizations,
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
            logger.exception("ReportAgent action '%s' failed.", action)
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

    def generate_full_report(
        self,
        analysis: dict[str, Any],
        risks: dict[str, Any],
        diet: dict[str, Any],
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile a comprehensive report from all analysis components.

        Parameters
        ----------
        analysis : dict
            Image-analysis results (spots, statistics, association summary).
        risks : dict
            Genomics risk-assessment results.
        diet : dict
            Nutrition plan results.
        profile : dict | None
            User / sample metadata.

        Returns
        -------
        dict
            Structured report with sections: ``summary``, ``telomere_analysis``,
            ``risk_assessment``, ``nutrition_plan``, ``recommendations``,
            and ``metadata``.
        """
        profile = profile or {}
        generated_at = time.time()

        # --- Telomere analysis section -----------------------------------
        stats = analysis.get("statistics", {})
        assoc = analysis.get("association_summary", {})

        telomere_section = {
            "n_telomeres": stats.get("n_telomeres", 0),
            "total_spots_detected": assoc.get("total_spots", 0),
            "association_rate": assoc.get("association_rate", 0.0),
            "mean_intensity": stats.get("mean_intensity", 0.0),
            "median_intensity": stats.get("median_intensity", 0.0),
            "std_intensity": stats.get("std_intensity", 0.0),
            "cv": stats.get("cv", 0.0),
            "quality_passed": analysis.get("validation", {}).get("passed", False),
            "quality_warnings": analysis.get("validation", {}).get("warnings", []),
        }

        # --- Risk assessment section -------------------------------------
        risk_section = {
            "overall_risk_score": risks.get("overall_risk_score", 0.0),
            "risk_category": risks.get("risk_category", "unknown"),
            "telomere_percentile": risks.get("telomere_percentile", 0.0),
            "per_disease_risks": risks.get("risks", []),
        }

        # --- Nutrition plan section --------------------------------------
        nutrition_section = {
            "region": diet.get("region", "global"),
            "priority_nutrients": diet.get("priority_nutrients", []),
            "recommended_foods": [f.get("name", "") for f in diet.get("recommended_foods", [])[:8]],
            "meal_plan": diet.get("meal_plan", {}),
            "notes": diet.get("notes", []),
        }

        # --- Recommendations (aggregate from all sources) ----------------
        recommendations: list[str] = []
        recommendations.extend(analysis.get("suggestions", []))
        for risk in risks.get("risks", []):
            if risk.get("risk_score", 0) >= 0.5:
                disease = risk["disease"].replace("_", " ")
                recommendations.append(
                    f"Elevated {disease} risk ({risk['risk_score']:.0%}) — "
                    "see risk section for recommended actions."
                )
        recommendations.extend(diet.get("notes", []))

        # --- Summary -----------------------------------------------------
        summary = self._generate_summary(telomere_section, risk_section, nutrition_section, profile)

        return {
            "report_version": "1.0",
            "generated_at": generated_at,
            "summary": summary,
            "telomere_analysis": telomere_section,
            "risk_assessment": risk_section,
            "nutrition_plan": nutrition_section,
            "recommendations": recommendations,
            "metadata": {
                "profile": profile,
                "image_path": analysis.get("image_path", ""),
            },
        }

    def format_as_html(self, report: dict[str, Any]) -> str:
        """Render a report dict as a self-contained HTML document.

        Parameters
        ----------
        report : dict
            Report as returned by :meth:`generate_full_report`.

        Returns
        -------
        str
            Complete HTML document string.
        """
        summary = html.escape(report.get("summary", ""))
        tel = report.get("telomere_analysis", {})
        risk = report.get("risk_assessment", {})
        nutrition = report.get("nutrition_plan", {})
        recs = report.get("recommendations", [])

        # Build disease-risk rows
        risk_rows = ""
        for r in risk.get("per_disease_risks", []):
            disease = html.escape(r.get("disease", "").replace("_", " ").title())
            score = r.get("risk_score", 0)
            category = html.escape(r.get("category", ""))
            colour = self._risk_colour(score)
            risk_rows += (
                f"<tr><td>{disease}</td>"
                f"<td style='color:{colour};font-weight:bold'>{score:.0%}</td>"
                f"<td>{category}</td></tr>\n"
            )

        # Build food list
        food_items = "".join(
            f"<li>{html.escape(str(f))}</li>" for f in nutrition.get("recommended_foods", [])
        )

        # Build recommendations list
        rec_items = "".join(f"<li>{html.escape(str(r))}</li>" for r in recs)

        report_ver = html.escape(str(report.get("report_version", "1.0")))
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Teloscopy Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 2em; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }}
  h2 {{ color: #2980b9; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background-color: #3498db; color: white; }}
  tr:nth-child(even) {{ background-color: #f2f2f2; }}
  .summary {{ background: #eaf2f8; padding: 1em; border-radius: 6px; margin: 1em 0; }}
  .metric {{ display: inline-block; margin: 0.5em 1em; text-align: center; }}
  .metric .value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
  .metric .label {{ font-size: 0.85em; color: #777; }}
  ul {{ line-height: 1.8; }}
  footer {{ margin-top: 2em; font-size: 0.8em; color: #999; }}
</style>
</head>
<body>
<h1>Teloscopy Analysis Report</h1>

<div class="summary"><p>{summary}</p></div>

<h2>Telomere Analysis</h2>
<div>
  <span class="metric"><span class="value">{tel.get("n_telomeres", 0)}</span>
  <br/><span class="label">Telomeres</span></span>
  <span class="metric"><span class="value">{tel.get("association_rate", 0):.1%}</span>
  <br/><span class="label">Association Rate</span></span>
  <span class="metric"><span class="value">{tel.get("mean_intensity", 0):.1f}</span>
  <br/><span class="label">Mean Intensity</span></span>
  <span class="metric"><span class="value">{tel.get("cv", 0):.2f}</span>
  <br/><span class="label">CV</span></span>
</div>

<h2>Risk Assessment</h2>
<p>Overall risk: <strong>{risk.get("risk_category", "unknown")}</strong>
({risk.get("overall_risk_score", 0):.0%})</p>
<table>
  <tr><th>Disease</th><th>Risk Score</th><th>Category</th></tr>
  {risk_rows}
</table>

<h2>Nutrition Plan</h2>
<p>Region: <strong>{html.escape(nutrition.get("region", "global"))}</strong></p>
<h3>Recommended Foods</h3>
<ul>{food_items}</ul>

<h2>Recommendations</h2>
<ul>{rec_items}</ul>

<footer>
  Generated by Teloscopy v1.0 | Report version {report_ver}
</footer>
</body>
</html>"""
        return html_doc

    def format_as_json(self, report: dict[str, Any]) -> dict[str, Any]:
        """Normalise a report dict for clean JSON serialisation.

        Converts any non-serialisable values (numpy types, etc.) to native
        Python types.

        Parameters
        ----------
        report : dict
            Report dict.

        Returns
        -------
        dict
            JSON-safe copy of the report.
        """
        return self._make_json_safe(report)

    def create_visualizations(
        self,
        report: dict[str, Any],
        output_dir: str | None = None,
    ) -> list[str]:
        """Generate visualisation plots from a report and save to disk.

        Creates:
        1. A bar chart of per-disease risk scores.
        2. A histogram of telomere intensity distribution (if spots available).
        3. A nutrient-priority radar / bar chart.

        Parameters
        ----------
        report : dict
            Report as returned by :meth:`generate_full_report`.
        output_dir : str | None
            Directory to save plots.  Falls back to :attr:`_output_dir`.

        Returns
        -------
        list[str]
            File paths to the generated plot images.
        """
        out = Path(output_dir) if output_dir else self._output_dir
        if out is None:
            import tempfile

            out = Path(tempfile.mkdtemp(prefix="teloscopy_plots_"))

        out.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed; skipping visualisations.")
            return paths

        # 1. Risk score bar chart -----------------------------------------
        try:
            risk_data = report.get("risk_assessment", {}).get("per_disease_risks", [])
            if risk_data:
                diseases = [r["disease"].replace("_", " ").title() for r in risk_data]
                scores = [r.get("risk_score", 0) for r in risk_data]
                colours = [self._risk_colour(s) for s in scores]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(diseases, scores, color=colours)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Risk Score")
                ax.set_title("Disease Risk Assessment")
                ax.invert_yaxis()
                fig.tight_layout()

                path = str(out / "risk_scores.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)
                paths.append(path)
        except Exception:
            logger.exception("Failed to create risk-score chart.")

        # 2. Nutrient priority chart --------------------------------------
        try:
            nutrients = report.get("nutrition_plan", {}).get("priority_nutrients", [])[:8]
            if nutrients:
                fig, ax = plt.subplots(figsize=(8, 4))
                y_pos = range(len(nutrients))
                # Use descending relevance as bar length
                values = list(range(len(nutrients), 0, -1))
                ax.barh(list(y_pos), values, color="#27ae60")
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels([n.replace("_", " ").title() for n in nutrients])
                ax.set_xlabel("Relative Priority")
                ax.set_title("Nutrient Priorities")
                ax.invert_yaxis()
                fig.tight_layout()

                path = str(out / "nutrient_priorities.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)
                paths.append(path)
        except Exception:
            logger.exception("Failed to create nutrient-priority chart.")

        # 3. Telomere intensity summary -----------------------------------
        try:
            tel = report.get("telomere_analysis", {})
            mean_i = tel.get("mean_intensity", 0)
            median_i = tel.get("median_intensity", 0)
            std_i = tel.get("std_intensity", 0)

            if mean_i > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ["Mean", "Median"]
                values = [mean_i, median_i]
                ax.bar(labels, values, color=["#3498db", "#2ecc71"], width=0.5)
                # Add error bar on mean
                ax.errorbar(0, mean_i, yerr=std_i, fmt="none", ecolor="black", capsize=5)
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title("Telomere Intensity Summary")
                fig.tight_layout()

                path = str(out / "intensity_summary.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)
                paths.append(path)
        except Exception:
            logger.exception("Failed to create intensity summary chart.")

        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary(
        telomere: dict[str, Any],
        risk: dict[str, Any],
        nutrition: dict[str, Any],
        profile: dict[str, Any],
    ) -> str:
        """Generate a human-readable summary paragraph."""
        parts: list[str] = []

        n_tel = telomere.get("n_telomeres", 0)
        parts.append(f"Analysed {n_tel} telomere(s)")

        assoc = telomere.get("association_rate", 0)
        parts.append(f"with {assoc:.0%} association rate")

        category = risk.get("risk_category", "unknown")
        overall = risk.get("overall_risk_score", 0)
        parts.append(f"Overall health risk: {category} ({overall:.0%})")

        percentile = risk.get("telomere_percentile", 0)
        if percentile > 0:
            parts.append(f"Telomere length at {percentile:.0f}th percentile for age")

        region = nutrition.get("region", "global")
        parts.append(f"Dietary plan tailored to {region} region")

        age = profile.get("age")
        if age:
            parts.append(f"Patient age: {age}")

        return ". ".join(parts) + "."

    @staticmethod
    def _risk_colour(score: float) -> str:
        """Return a CSS colour string for a risk score."""
        if score >= 0.7:
            return "#e74c3c"  # red
        elif score >= 0.4:
            return "#f39c12"  # orange
        elif score >= 0.2:
            return "#f1c40f"  # yellow
        return "#27ae60"  # green

    @classmethod
    def _make_json_safe(cls, obj: Any) -> Any:
        """Recursively convert non-serialisable types to JSON-friendly equivalents."""
        if isinstance(obj, dict):
            return {k: cls._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._make_json_safe(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    # ------------------------------------------------------------------
    # Message handler bridges
    # ------------------------------------------------------------------

    def _handle_full_report(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.generate_full_report(
            analysis=content.get("analysis", {}),
            risks=content.get("risks", {}),
            diet=content.get("diet", {}),
            profile=content.get("profile"),
        )

    def _handle_format_html(self, content: dict[str, Any]) -> dict[str, Any]:
        report = content.get("report", content)
        html_str = self.format_as_html(report)
        return {"html": html_str}

    def _handle_format_json(self, content: dict[str, Any]) -> dict[str, Any]:
        report = content.get("report", content)
        return self.format_as_json(report)

    def _handle_visualizations(self, content: dict[str, Any]) -> dict[str, Any]:
        report = content.get("report", content)
        output_dir = content.get("output_dir")
        paths = self.create_visualizations(report, output_dir=output_dir)
        return {"visualization_paths": paths}
