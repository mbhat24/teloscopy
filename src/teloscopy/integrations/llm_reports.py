"""
LLM-powered report generation for telomere and genomic analysis.

Supports local Ollama models and OpenAI-compatible APIs for generating
narrative clinical reports.  Falls back to template-based formatting when
no LLM backend is reachable.

Usage::

    from teloscopy.integrations.llm_reports import ReportGenerator
    gen = ReportGenerator(backend="ollama", model="llama3")
    report = gen.generate_telomere_report(
        telomere_data={"mean_length_kb": 6.8, "percentile": 42},
        patient_age=55, sex="male",
    )
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates — scientific context + formatting instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = textwrap.dedent("""\
    You are a board-certified clinical geneticist writing a professional
    medical report.  Use precise scientific language while remaining
    accessible to an educated lay audience.  Cite underlying biology
    (e.g. shelterin complex, TRF1/TRF2, hTERT) so the reader understands
    *why* results matter.
    Rules: Use Markdown (## headings, bullet lists).  End with a disclaimer.
    Do NOT fabricate citations — say "studies suggest" instead of inventing DOIs.
""")

TELOMERE_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + (
    "Context: Telomeres are TTAGGG repeats capping chromosome ends.  Mean TL "
    "shortens ~20-50 bp/year.  Critically short telomeres trigger senescence "
    "via p53/p21.  Percentiles compare to age/sex-matched reference cohorts.\n"
)

TELOMERE_USER_PROMPT = textwrap.dedent("""\
    Generate a **Telomere Analysis Report**.
    Patient: {patient_age} y/o {sex}
    Measurements: ```json\n{telomere_json}\n```
    Include: 1) plain-language summary 2) age/sex-matched comparison
    3) biological implications (cellular aging, oxidative stress)
    4) lifestyle factors (exercise, sleep, stress, smoking, diet)
    5) actionable recommendations.""")

RISK_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + (
    "Context: Polygenic risk scores (PRS) aggregate many common variants. "
    "Report absolute risk where available; otherwise use odds ratios. "
    "Always contextualise genetic risk with modifiable environmental factors.\n"
)

RISK_USER_PROMPT = textwrap.dedent("""\
    Generate a **Disease Risk Report**.
    Patient: ```json\n{patient_json}\n```
    Risks: ```json\n{risks_json}\n```
    Per disease: 1) explain condition 2) interpret risk score 3) list
    modifiable factors 4) suggest screening.  End with top-3 actionable risks.""")

NUTRITION_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + (
    "Context: Nutrigenomic recommendations map variants (MTHFR C677T, "
    "APOE e4, LCT -13910C>T) to dietary guidance.  Respect regional food "
    "availability and cultural patterns.\n"
)

NUTRITION_USER_PROMPT = textwrap.dedent("""\
    Generate a **Personalised Nutrition Report**.
    Region: {region}
    Recommendations: ```json\n{diet_json}\n```
    Group by category (macronutrients, micronutrients, foods to emphasise,
    foods to limit).  Explain genetic basis.  Provide region-appropriate
    meal suggestions and a sample one-day plan if data permits.""")

FACIAL_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + (
    "Context: GWAS loci influence facial morphology (PAX3, DCHS2, RUNX2). "
    "Predictions are probabilistic and environmentally influenced.\n"
)

FACIAL_USER_PROMPT = textwrap.dedent("""\
    Generate a **Facial-Genomic Prediction Report**.
    Data: ```json\n{facial_json}\n```
    Include: 1) predicted traits + confidence 2) key loci 3) limitations
    (epigenetics, environment, aging) 4) forensic/ancestry implications.""")

EXECUTIVE_SUMMARY_PROMPT = textwrap.dedent("""\
    Write an **Executive Summary** (under 250 words, 2-3 paragraphs) for
    a multi-section genomic report.  Highlight the most clinically
    significant finding and top recommendation.
    Sections:\n{sections_text}""")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FullReport:
    """Complete multi-section genomic analysis report."""

    title: str
    executive_summary: str
    telomere_section: str
    disease_risk_section: str
    nutrition_section: str
    facial_analysis_section: str | None = None
    recommendations: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(
        default_factory=lambda: [
            "This report is AI-generated and informational only — it does not "
            "constitute medical advice, diagnosis, or treatment.",
            "Genetic risk scores reflect statistical associations and do not "
            "determine individual outcomes.  Consult a healthcare provider.",
            "Telomere measurements may vary between assay methods (qPCR, TRF, "
            "STELA, Flow-FISH).  Cross-method comparisons require caution.",
        ]
    )
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    model_used: str = "unknown"

    def to_markdown(self) -> str:
        """Render the full report as a single Markdown document."""
        parts = [
            f"# {self.title}",
            f"*Generated: {self.generated_at:%Y-%m-%d %H:%M UTC} | Model: {self.model_used}*\n",
            "## Executive Summary\n",
            self.executive_summary,
            "\n## Telomere Analysis\n",
            self.telomere_section,
            "\n## Disease Risk Assessment\n",
            self.disease_risk_section,
            "\n## Nutrition Guidance\n",
            self.nutrition_section,
        ]
        if self.facial_analysis_section:
            parts += ["\n## Facial-Genomic Predictions\n", self.facial_analysis_section]
        if self.recommendations:
            parts.append("\n## Recommendations\n")
            parts += [f"{i}. {r}" for i, r in enumerate(self.recommendations, 1)]
        if self.disclaimers:
            parts.append("\n---\n### Disclaimers\n")
            parts += [f"- {d}" for d in self.disclaimers]
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            f.name: (
                getattr(self, f.name).isoformat()
                if f.name == "generated_at"
                else getattr(self, f.name)
            )
            for f in self.__dataclass_fields__.values()
        }


# ---------------------------------------------------------------------------
# HTTP retry helper (shared by both clients)
# ---------------------------------------------------------------------------


def _retry_request(
    url: str,
    data: bytes | None,
    method: str,
    headers: dict[str, str],
    timeout: int,
    max_retries: int,
    retry_delay: float,
    label: str,
) -> dict[str, Any]:
    """HTTP request with exponential-backoff retry on 429/5xx and network errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"{label} API error {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
                ) from exc
            last_exc = exc
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
        except Exception as exc:
            raise ConnectionError(f"{label} unexpected error: {exc}") from exc
        delay = retry_delay * (2 ** (attempt - 1))
        logger.warning(
            "%s %s failed (%d/%d) — retry in %.1fs", label, url, attempt, max_retries, delay
        )
        time.sleep(delay)
    raise ConnectionError(
        f"Failed to reach {label} at {url} after {max_retries} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# LLM client: Ollama
# ---------------------------------------------------------------------------


class OllamaClient:
    """HTTP client for a locally-running Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model, self.timeout = model, timeout
        self.max_retries, self.retry_delay = max_retries, retry_delay

    def _call(self, path: str, payload: dict | None = None, method: str = "GET") -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload else None
        hdrs = {"Content-Type": "application/json"} if data else {}
        return _retry_request(
            f"{self.base_url}{path}",
            data,
            method or ("POST" if data else "GET"),
            hdrs,
            self.timeout,
            self.max_retries,
            self.retry_delay,
            "Ollama",
        )

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(self.base_url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return model tags available on the server."""
        return [m.get("name", "?") for m in self._call("/api/tags").get("models", [])]

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generate a completion.  Raises ConnectionError / RuntimeError."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        logger.info("Ollama generate: model=%s len=%d", self.model, len(prompt))
        text = self._call("/api/generate", payload, "POST").get("response", "")
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return text.strip()


# ---------------------------------------------------------------------------
# LLM client: OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAIClient:
    """Client for OpenAI-compatible /v1/chat/completions endpoints.

    Works with OpenAI, Azure OpenAI, vLLM, LM Studio, etc.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key = model, api_key
        self.timeout, self.max_retries, self.retry_delay = timeout, max_retries, retry_delay

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _call(self, path: str, payload: dict | None = None, method: str = "GET") -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload else None
        return _retry_request(
            f"{self.base_url}{path}",
            data,
            method or ("POST" if data else "GET"),
            self._headers(),
            self.timeout,
            self.max_retries,
            self.retry_delay,
            "OpenAI",
        )

    def is_available(self) -> bool:
        """Check whether the API endpoint responds."""
        try:
            self._call("/models")
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return model identifiers from the server."""
        return [m.get("id", "?") for m in self._call("/models").get("data", [])]

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generate a chat completion.  Raises ConnectionError / RuntimeError."""
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        logger.info("OpenAI generate: model=%s len=%d", self.model, len(prompt))
        choices = self._call(
            "/chat/completions",
            {"model": self.model, "messages": msgs, "temperature": temperature},
            "POST",
        ).get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI returned no choices")
        text = choices[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("OpenAI returned empty message")
        return text.strip()


# ---------------------------------------------------------------------------
# Template-based fallback (no LLM required)
# ---------------------------------------------------------------------------


class _TemplateFallback:
    """Generates structured reports via string templates when no LLM is available."""

    @staticmethod
    def telomere_report(data: dict[str, Any], age: int, sex: str) -> str:
        pct = data.get("percentile", "N/A")
        if isinstance(pct, (int, float)):
            interp = (
                "falls in the **lower quartile** — accelerated attrition may "
                "reflect oxidative stress or lifestyle factors."
                if pct < 25
                else "is **within the normal range** for the demographic cohort."
                if pct < 75
                else "is **above the 75th percentile**, suggesting robust chromosomal maintenance."
            )
        else:
            interp = "could not be compared (percentile unavailable)."
        return "\n".join(
            [
                "## Telomere Analysis Report",
                f"\n**Patient:** {age}-year-old {sex}",
                "\n### Measurements",
                f"- Mean telomere length: **{data.get('mean_length_kb', 'N/A')} kb**",
                f"- Age/sex-matched percentile: **{pct}th**",
                f"- Shortest telomere cluster: **{data.get('shortest_length_kb', 'N/A')} kb**",
                f"- Estimated attrition rate: **{data.get('attrition_rate_bp_year', 'N/A')} bp/year**",
                f"\n### Interpretation\nThe patient's telomere length {interp}",
                "\n### Recommendations",
                "1. Aerobic exercise >=150 min/week.  2. Sleep 7-9 h/night.",
                "3. Stress management.  4. Mediterranean-style diet.",
                "5. Avoid tobacco; limit alcohol.",
                "\n---\n*Disclaimer: Informational only — not medical advice.*",
            ]
        )

    @staticmethod
    def risk_report(risks: list[dict[str, Any]], profile: dict[str, Any]) -> str:
        sorted_r = sorted(risks, key=lambda r: r.get("risk_score", 0), reverse=True)
        lines = [
            "## Disease Risk Report",
            f"\n**Patient:** {profile.get('age', '?')}-year-old "
            f"{profile.get('sex', '?')}, ethnicity: {profile.get('ethnicity', '?')}\n",
        ]
        for e in sorted_r:
            v = e.get("key_variants", [])
            m = e.get("modifiable_factors", [])
            lines += [
                f"#### {e.get('disease', '?')}",
                f"- Risk score: **{e.get('risk_score', '?')}** ({e.get('category', '?')})",
                *([f"- Key variants: {', '.join(str(x) for x in v)}"] if v else []),
                *([f"- Modifiable: {', '.join(str(x) for x in m)}"] if m else []),
                "",
            ]
        lines.append("### Top-3 Actionable\n")
        for i, e in enumerate(sorted_r[:3], 1):
            lines.append(f"{i}. **{e.get('disease', '?')}** — targeted screening recommended.")
        lines += ["\n---\n*Disclaimer: Risk scores are statistical, not deterministic.*"]
        return "\n".join(lines)

    @staticmethod
    def nutrition_report(recs: list[dict[str, Any]], region: str) -> str:
        cats: dict[str, list[dict]] = {}
        for r in recs:
            cats.setdefault(r.get("category", "General"), []).append(r)
        lines = ["## Personalised Nutrition Report", f"\n**Region:** {region}\n"]
        for cat, items in cats.items():
            lines.append(f"#### {cat}")
            for r in items:
                g = r.get("gene", "")
                gn = f" (*{g}*)" if g else ""
                lines.append(
                    f"- **{r.get('nutrient', r.get('name', '?'))}**: "
                    f"{r.get('action', 'review')}{gn}"
                )
            lines.append("")
        lines += [
            "### General\n- Whole foods, vegetables, lean protein, healthy fats.",
            "- >=2 L water/day.  Limit processed foods.",
            "\n---\n*Disclaimer: Discuss with a registered dietitian.*",
        ]
        return "\n".join(lines)

    @staticmethod
    def facial_report(data: dict[str, Any]) -> str:
        lines = ["## Facial-Genomic Prediction Report\n"]
        traits = data.get("predicted_traits", {})
        if isinstance(traits, dict):
            for t, d in traits.items():
                if isinstance(d, dict):
                    loci = d.get("key_loci", [])
                    lines.append(
                        f"- **{t}**: {d.get('prediction', '?')} (conf: {d.get('confidence', '?')})"
                    )
                    if loci:
                        lines.append(f"  - Loci: {', '.join(str(loc) for loc in loci)}")
                else:
                    lines.append(f"- **{t}**: {d}")
        elif isinstance(traits, list):
            for item in traits:
                lines.append(f"- **{item.get('trait', '?')}**: {item.get('prediction', '?')}")
        lines += [
            "\n### Limitations",
            "- Highly polygenic; environment/nutrition/aging influence.",
            "- Probabilistic, not deterministic.",
            "\n---\n*Disclaimer: Not for identification without corroborating evidence.*",
        ]
        return "\n".join(lines)

    @staticmethod
    def executive_summary(sections: dict[str, str]) -> str:
        mapping = {
            "telomere": "Telomere analysis completed.",
            "risk": "Disease risk profiling identified elevated predispositions.",
            "nutrition": "Nutrition guidance tailored to genetic variants.",
            "facial": "Facial-genomic predictions provided.",
        }
        body = "  ".join(v for k, v in mapping.items() if sections.get(k))
        return (
            f"## Executive Summary\n\nThis report integrates telomere biology, "
            f"polygenic risk, nutrigenomics, and facial-genomic predictions.  {body}\n\n"
            f"See sections below for details and recommendations."
        )


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Unified facade for LLM-powered (or template-based) report generation.

    Args:
        backend: ``"ollama"`` or ``"openai"``.
        model: Model identifier for the chosen backend.
        base_url: API root URL.
        api_key: API key (required for OpenAI, ignored for Ollama).
    """

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        fallback_only: bool = False,
    ) -> None:
        self.backend_name = backend.lower()
        self.model = model
        self._fallback = _TemplateFallback()
        self._client: OllamaClient | OpenAIClient | None = None
        self._using_fallback: bool = fallback_only

        if fallback_only:
            logger.info("ReportGenerator: fallback-only mode.")
            return
        if self.backend_name == "ollama":
            self._client = OllamaClient(base_url, model, timeout, max_retries, retry_delay)
        elif self.backend_name == "openai":
            self._client = OpenAIClient(base_url, model, api_key, timeout, max_retries, retry_delay)
        else:
            raise ValueError(f"Unsupported backend {backend!r}; use 'ollama' or 'openai'.")

        if not self._client.is_available():
            logger.warning("Backend '%s' at %s unreachable — template fallback.", backend, base_url)
            self._using_fallback = True
        else:
            logger.info("Backend '%s' available. Model: %s", backend, model)

    def _generate(self, prompt: str, system: str, temperature: float = 0.3) -> str:
        """Call the LLM; return empty string on failure (signals fallback)."""
        if self._using_fallback or self._client is None:
            return ""
        try:
            return self._client.generate(prompt=prompt, system=system, temperature=temperature)
        except (ConnectionError, RuntimeError, OSError) as exc:
            logger.error("LLM failed — fallback: %s", exc)
            self._using_fallback = True
            return ""

    def generate_telomere_report(
        self, telomere_data: dict[str, Any], patient_age: int, sex: str
    ) -> str:
        """Generate a narrative telomere analysis report.

        Args:
            telomere_data: Keys ``mean_length_kb``, ``percentile``,
                ``shortest_length_kb``, ``attrition_rate_bp_year``.
            patient_age: Age in years.
            sex: ``"male"`` or ``"female"``.
        Returns: Markdown string.
        """
        prompt = TELOMERE_USER_PROMPT.format(
            patient_age=patient_age, sex=sex, telomere_json=json.dumps(telomere_data, indent=2)
        )
        return self._generate(prompt, TELOMERE_SYSTEM_PROMPT) or self._fallback.telomere_report(
            telomere_data, patient_age, sex
        )

    def generate_risk_report(
        self, disease_risks: list[dict[str, Any]], patient_profile: dict[str, Any]
    ) -> str:
        """Generate a disease-risk narrative.

        Args:
            disease_risks: Dicts with ``disease``, ``risk_score``, ``category``,
                ``key_variants``, ``modifiable_factors``.
            patient_profile: ``age``, ``sex``, ``ethnicity``.
        Returns: Markdown string.
        """
        prompt = RISK_USER_PROMPT.format(
            patient_json=json.dumps(patient_profile, indent=2),
            risks_json=json.dumps(disease_risks, indent=2),
        )
        return self._generate(prompt, RISK_SYSTEM_PROMPT) or self._fallback.risk_report(
            disease_risks, patient_profile
        )

    def generate_nutrition_report(
        self, diet_recommendations: list[dict[str, Any]], region: str
    ) -> str:
        """Generate personalised nutrition guidance.

        Args:
            diet_recommendations: Dicts with ``nutrient``, ``action``, ``gene``, ``category``.
            region: Geographic/cultural context.
        Returns: Markdown string.
        """
        prompt = NUTRITION_USER_PROMPT.format(
            region=region, diet_json=json.dumps(diet_recommendations, indent=2)
        )
        return self._generate(prompt, NUTRITION_SYSTEM_PROMPT) or self._fallback.nutrition_report(
            diet_recommendations, region
        )

    def generate_facial_analysis_report(self, facial_data: dict[str, Any]) -> str:
        """Generate facial-genomic prediction narrative.

        Args:
            facial_data: ``predicted_traits`` mapping trait names to detail dicts.
        Returns: Markdown string.
        """
        prompt = FACIAL_USER_PROMPT.format(facial_json=json.dumps(facial_data, indent=2))
        return self._generate(prompt, FACIAL_SYSTEM_PROMPT) or self._fallback.facial_report(
            facial_data
        )

    def generate_full_report(self, analysis_results: dict[str, Any]) -> FullReport:
        """Generate a complete multi-section report combining all analyses.

        Args:
            analysis_results: Dict with optional keys ``telomere_data``,
                ``patient_age``, ``sex``, ``disease_risks``, ``patient_profile``,
                ``diet_recommendations``, ``region``, ``facial_data``.
        Returns: Populated :class:`FullReport`.
        """
        age = analysis_results.get("patient_age", 0)
        sex = analysis_results.get("sex", "unknown")
        region = analysis_results.get("region", "Global")
        profile = analysis_results.get("patient_profile", {"age": age, "sex": sex})

        telo = analysis_results.get("telomere_data")
        telo_s = self.generate_telomere_report(telo, age, sex) if telo else ""
        risks = analysis_results.get("disease_risks")
        risk_s = self.generate_risk_report(risks, profile) if risks else ""
        diet = analysis_results.get("diet_recommendations")
        nutr_s = self.generate_nutrition_report(diet, region) if diet else ""
        face = analysis_results.get("facial_data")
        face_s = self.generate_facial_analysis_report(face) if face else None

        sec_map = {"telomere": telo_s, "risk": risk_s, "nutrition": nutr_s, "facial": face_s or ""}
        combined = "\n---\n".join(f"### {k.title()}\n{v}" for k, v in sec_map.items() if v)
        exec_sum = self._generate(
            EXECUTIVE_SUMMARY_PROMPT.format(sections_text=combined), SYSTEM_PROMPT_BASE
        )
        if not exec_sum:
            exec_sum = self._fallback.executive_summary(sec_map)

        return FullReport(
            title=f"Integrated Genomic Health Report — {age}-year-old {sex.title()}",
            executive_summary=exec_sum,
            telomere_section=telo_s,
            disease_risk_section=risk_s,
            nutrition_section=nutr_s,
            facial_analysis_section=face_s,
            recommendations=[
                "Follow-up with a clinical geneticist (family history context).",
                "Lifestyle modifications for top modifiable risk factors.",
                "Adopt personalised dietary guidance; consult a dietitian.",
                "Repeat telomere assessment in 12-24 months.",
            ],
            model_used=self.model if not self._using_fallback else "template-fallback",
            generated_at=datetime.now(UTC),
        )

    @property
    def is_using_fallback(self) -> bool:
        """``True`` if using template-based output (no LLM)."""
        return self._using_fallback

    @property
    def available_models(self) -> list[str]:
        """Model tags on the current backend, or ``[]``."""
        if self._client is None or self._using_fallback:
            return []
        try:
            return self._client.list_models()
        except Exception:
            return []

    def __repr__(self) -> str:
        mode = "fallback" if self._using_fallback else self.backend_name
        return (
            f"ReportGenerator(backend={self.backend_name!r}, model={self.model!r}, mode={mode!r})"
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def quick_telomere_report(
    telomere_data: dict,
    patient_age: int,
    sex: str,
    *,
    backend: str = "ollama",
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
) -> str:
    """One-shot telomere report without managing a ReportGenerator."""
    return ReportGenerator(
        backend=backend, model=model, base_url=base_url, api_key=api_key
    ).generate_telomere_report(telomere_data, patient_age, sex)


def quick_full_report(
    analysis_results: dict,
    *,
    backend: str = "ollama",
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
) -> FullReport:
    """One-shot full report without managing a ReportGenerator."""
    return ReportGenerator(
        backend=backend, model=model, base_url=base_url, api_key=api_key
    ).generate_full_report(analysis_results)


# ---------------------------------------------------------------------------
# CLI demo (python -m teloscopy.integrations.llm_reports)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    gen = ReportGenerator(fallback_only=True)
    sample = {
        "telomere_data": {
            "mean_length_kb": 6.2,
            "percentile": 35,
            "shortest_length_kb": 3.8,
            "attrition_rate_bp_year": 45,
        },
        "patient_age": 55,
        "sex": "male",
        "disease_risks": [
            {
                "disease": "Type 2 Diabetes",
                "risk_score": 0.78,
                "category": "elevated",
                "key_variants": ["TCF7L2 rs7903146"],
                "modifiable_factors": ["diet", "exercise"],
            },
            {
                "disease": "Coronary Artery Disease",
                "risk_score": 0.62,
                "category": "moderate",
                "key_variants": ["9p21.3 rs1333049"],
                "modifiable_factors": ["cholesterol"],
            },
        ],
        "patient_profile": {"age": 55, "sex": "male", "ethnicity": "European"},
        "diet_recommendations": [
            {
                "nutrient": "Folate",
                "action": "increase",
                "gene": "MTHFR",
                "category": "Micronutrients",
            },
            {
                "nutrient": "Omega-3",
                "action": "supplement",
                "gene": "FADS1",
                "category": "Macronutrients",
            },
        ],
        "region": "Northern Europe",
        "facial_data": {
            "predicted_traits": {
                "Eye colour": {
                    "prediction": "Brown",
                    "confidence": 0.87,
                    "key_loci": ["OCA2", "HERC2"],
                },
            }
        },
    }
    full = gen.generate_full_report(sample)
    print(full.to_markdown())
    print(f"\nModel: {full.model_used} | {gen!r}")
