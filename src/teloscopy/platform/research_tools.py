"""Research collaboration and data export toolkit for academic researchers.

Exports telomere analysis data for R/Python, builds research cohorts with
inclusion/exclusion criteria, performs case-control matching, computes
descriptive/inferential statistics, and generates standardized citations.

Statistical methods: Pearson r (product-moment), p-values via t-transform
t = r·√((n-2)/(1-r²)), power ≈ Φ(|d|·√(n/2) − z_{1−α/2}), greedy
nearest-neighbour matching on standardised Euclidean distance.
Refs: Cohen (1988) Statistical Power Analysis; Rothman (2012) Epidemiology.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StatsSummary:
    """Descriptive statistics per numeric variable.

    Uses arithmetic mean (Σxi/n), sample std (Bessel-corrected, n-1 denominator),
    median (interpolated for even n), and range endpoints.

    Attributes:
        n: Number of observations.
        variables: Names of numeric variables summarised.
        means/stds/medians/mins/maxs: Per-variable statistics as {name: value}.
    """

    n: int
    variables: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    medians: dict[str, float]
    mins: dict[str, float]
    maxs: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            k: getattr(self, k)
            for k in ("n", "variables", "means", "stds", "medians", "mins", "maxs")
        }


@dataclass
class CorrelationMatrix:
    """Pairwise Pearson correlation matrix with two-tailed p-values.

    Correlation coefficient r computed via product-moment formula.  P-values
    derived from t = r·√((n-2)/(1-r²)) with normal CDF approximation.

    Attributes:
        variables: Ordered variable names (row/column labels).
        matrix: Square matrix of Pearson r values (list of lists).
        p_values: Square matrix of two-tailed p-values.
        n: Number of complete observations used.
    """

    variables: list[str]
    matrix: list[list[float]]
    p_values: list[list[float]]
    n: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "variables": self.variables,
            "matrix": self.matrix,
            "p_values": self.p_values,
            "n": self.n,
        }


@dataclass
class CohortDefinition:
    """Research cohort definition with inclusion/exclusion criteria.

    Criteria map variable names to constraint dicts. Supported forms:
    ``{"age": {"min": 18, "max": 65}}``, ``{"status": {"equals": "active"}}``,
    ``{"dx": {"in": ["A","B"]}}``, ``{"score": {"not_equals": 0}}``.

    Attributes:
        name: Human-readable cohort label.
        inclusion_criteria: Criteria a record *must* satisfy.
        exclusion_criteria: Criteria that *exclude* a record.
        description: Free-text purpose description.
        created_at: ISO-8601 creation timestamp.
    """

    name: str
    inclusion_criteria: dict[str, dict] = field(default_factory=dict)
    exclusion_criteria: dict[str, dict] = field(default_factory=dict)
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            k: getattr(self, k)
            for k in (
                "name",
                "inclusion_criteria",
                "exclusion_criteria",
                "description",
                "created_at",
            )
        }


@dataclass
class PowerAnalysis:
    """Statistical power analysis results for a two-sample t-test.

    Power estimated via normal approximation: power ≈ Φ(|d|·√(n/2) − z_{1−α/2})
    where d is Cohen's d and n is per-group sample size.

    Attributes:
        effect_size: Cohen's d used in the calculation.
        n: Per-group sample size.
        alpha: Significance level (two-tailed).
        power: Estimated statistical power (P(reject H₀)).
        required_n_for_80_pct: Min per-group n for 80% power.
    """

    effect_size: float
    n: int
    alpha: float
    power: float
    required_n_for_80_pct: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            k: getattr(self, k)
            for k in ("effect_size", "n", "alpha", "power", "required_n_for_80_pct")
        }


# ---------------------------------------------------------------------------
# Internal statistical helpers
# ---------------------------------------------------------------------------


def _numeric_values(analyses: list[dict], key: str) -> list[float]:
    """Extract valid numeric values for *key*, skipping missing/non-numeric."""
    vals: list[float] = []
    for rec in analyses:
        v = rec.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return vals


def _mean(values: list[float]) -> float:
    """Arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected, n-1 denominator)."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _median(values: list[float]) -> float:
    """Median with linear interpolation for even-length samples."""
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson product-moment correlation. Returns 0.0 for constant/short inputs."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx, my = _mean(xs[:n]), _mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx and dy else 0.0


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 26.2.17, max error 7.5e-8)."""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    a = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
    p_c, sign = 0.3275911, (-1 if x < 0 else 1)
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p_c * x)
    y = 1.0 - (((((a[4] * t + a[3]) * t) + a[2]) * t + a[1]) * t + a[0]) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _inverse_normal(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro, ~1e-6 accuracy)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    t = math.sqrt(-2.0 * math.log(min(p, 1 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    return -result if p < 0.5 else result


def _p_value_from_r(r: float, n: int) -> float:
    """Two-tailed p-value for Pearson r via t-distribution normal approximation."""
    if n < 3 or abs(r) >= 1.0:
        return 1.0 if abs(r) < 1.0 else 0.0
    t_stat = r * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    z = t_stat * (1 - 1 / (4 * df)) / math.sqrt(1 + t_stat * t_stat / (2 * df))
    return min(2 * _normal_cdf(-abs(z)), 1.0)


def _random_id(length: int = 8) -> str:
    """Generate a random alphanumeric identifier."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _matches_criteria(record: dict, criteria: dict[str, dict]) -> bool:
    """Check if *record* satisfies all constraints (min/max/equals/in/contains)."""
    for var, cons in criteria.items():
        val = record.get(var)
        if "min" in cons:
            try:
                if val is None or float(val) < float(cons["min"]):
                    return False
            except (TypeError, ValueError):
                return False
        if "max" in cons:
            try:
                if val is None or float(val) > float(cons["max"]):
                    return False
            except (TypeError, ValueError):
                return False
        if "equals" in cons and val != cons["equals"]:
            return False
        if "not_equals" in cons and val == cons["not_equals"]:
            return False
        if "in" in cons and val not in cons["in"]:
            return False
        if "not_in" in cons and val in cons["not_in"]:
            return False
        if "contains" in cons:
            if val is None or str(cons["contains"]) not in str(val):
                return False
    return True


def _infer_types(
    analyses: list[dict], keys: list[str], type_map: dict[str, dict[type, str]], default: str
) -> dict[str, str]:
    """Generic column type inference from sample values."""
    result: dict[str, str] = {}
    for k in keys:
        sv = [r.get(k) for r in analyses[:100] if r.get(k) is not None]
        if not sv:
            result[k] = default
        elif all(isinstance(v, bool) for v in sv):
            result[k] = type_map["bool"]
        elif all(isinstance(v, int) for v in sv):
            result[k] = type_map["int"]
        elif all(isinstance(v, (int, float)) for v in sv):
            result[k] = type_map["float"]
        else:
            result[k] = default
    return result


_R_TYPES = {"bool": "logical", "int": "integer", "float": "numeric"}
_PD_TYPES = {"bool": "bool", "int": "int64", "float": "float64"}

# ---------------------------------------------------------------------------
# ResearchExporter
# ---------------------------------------------------------------------------


class ResearchExporter:
    """Export telomere analysis data in research-ready formats.

    Supports CSV, TSV, JSON, and Parquet-like columnar JSON.  Convenience
    methods produce files ready for R (``export_for_r``) or Python/pandas
    (``export_for_python``).

    Args:
        output_dir: Base directory for exports (created if absent).
    """

    SUPPORTED_FORMATS = {"csv", "tsv", "json", "parquet"}

    def __init__(self, output_dir: str = "./research_exports") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_dataset(self, analyses: list[dict], format: str = "csv") -> str:
        """Export *analyses* to CSV/TSV/JSON/Parquet-like JSON. Returns abs path."""
        fmt = format.lower().strip()
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Choose from: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        dispatch = {
            "csv": self._write_csv,
            "tsv": self._write_tsv,
            "json": self._write_json,
            "parquet": self._write_parquet_json,
        }
        return dispatch[fmt](analyses, ts)

    def export_for_r(self, analyses: list[dict], output_path: str) -> str:
        """Export R-compatible CSV (NA for missing, TRUE/FALSE for bools).

        Writes sidecar ``.meta.json`` with colClasses and R import hint.
        """
        if not analyses:
            self._write_empty(output_path)
            return os.path.abspath(output_path)
        keys = self._all_keys(analyses)
        col_types = self._infer_r_types(analyses, keys)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for rec in analyses:
                w.writerow(
                    [
                        "NA"
                        if rec.get(k) is None
                        else (
                            "TRUE"
                            if rec[k] is True
                            else "FALSE"
                            if rec[k] is False
                            else str(rec[k])
                        )
                        for k in keys
                    ]
                )
        meta = {
            "generated_by": "Teloscopy ResearchExporter",
            "generated_at": datetime.now(UTC).isoformat(),
            "n_records": len(analyses),
            "columns": keys,
            "col_classes": col_types,
            "na_representation": "NA",
            "r_import_hint": (
                f'df <- read.csv("{os.path.basename(output_path)}", '
                f'na.strings="NA", stringsAsFactors=FALSE)'
            ),
        }
        with open(output_path + ".meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        return os.path.abspath(output_path)

    def export_for_python(self, analyses: list[dict], output_path: str) -> str:
        """Export JSON with pandas dtype annotations.

        Load with: ``pd.DataFrame(blob['records']).astype(blob['dtypes'])``.
        """
        keys = self._all_keys(analyses) if analyses else []
        dtypes = self._infer_pandas_dtypes(analyses, keys)
        payload = {
            "generated_by": "Teloscopy ResearchExporter",
            "generated_at": datetime.now(UTC).isoformat(),
            "n_records": len(analyses),
            "dtypes": dtypes,
            "records": analyses,
            "python_import_hint": (
                f"import json, pandas as pd; "
                f'blob = json.load(open("{os.path.basename(output_path)}")); '
                f'df = pd.DataFrame(blob["records"]).astype(blob["dtypes"])'
            ),
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return os.path.abspath(output_path)

    def generate_summary_statistics(self, analyses: list[dict]) -> StatsSummary:
        """Compute mean, std (n-1), median, min, max for all numeric variables."""
        if not analyses:
            return StatsSummary(n=0, variables=[], means={}, stds={}, medians={}, mins={}, maxs={})
        keys = self._all_keys(analyses)
        stats: dict[str, dict[str, float]] = {}
        numeric_keys: list[str] = []
        for key in keys:
            vals = _numeric_values(analyses, key)
            if not vals:
                continue
            numeric_keys.append(key)
            stats[key] = {
                "mean": round(_mean(vals), 6),
                "std": round(_std(vals), 6),
                "median": round(_median(vals), 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
            }
        return StatsSummary(
            n=len(analyses),
            variables=numeric_keys,
            means={k: stats[k]["mean"] for k in numeric_keys},
            stds={k: stats[k]["std"] for k in numeric_keys},
            medians={k: stats[k]["median"] for k in numeric_keys},
            mins={k: stats[k]["min"] for k in numeric_keys},
            maxs={k: stats[k]["max"] for k in numeric_keys},
        )

    def create_correlation_matrix(
        self, analyses: list[dict], variables: list[str]
    ) -> CorrelationMatrix:
        """Pearson correlation matrix for *variables* (complete-case analysis)."""
        complete: dict[str, list[float]] = {v: [] for v in variables}
        for rec in analyses:
            row: dict[str, float] = {}
            ok = True
            for v in variables:
                try:
                    row[v] = float(rec[v])
                except (KeyError, TypeError, ValueError):
                    ok = False
                    break
            if ok:
                for v in variables:
                    complete[v].append(row[v])
        n = len(complete[variables[0]]) if variables else 0
        k = len(variables)
        r_mat = [[0.0] * k for _ in range(k)]
        p_mat = [[0.0] * k for _ in range(k)]
        for i in range(k):
            r_mat[i][i] = 1.0
            for j in range(i + 1, k):
                r = _pearson_r(complete[variables[i]], complete[variables[j]])
                p = _p_value_from_r(r, n)
                r_mat[i][j] = r_mat[j][i] = round(r, 6)
                p_mat[i][j] = p_mat[j][i] = round(p, 6)
        return CorrelationMatrix(variables=variables, matrix=r_mat, p_values=p_mat, n=n)

    def anonymize_for_publication(self, analyses: list[dict]) -> list[dict]:
        """Remove PII fields, assign random IDs, truncate dates to year only.

        PII heuristic strips fields containing name/email/phone/address/ssn/
        dob/patient_id/mrn/passport/driver_license tokens.
        """
        pii_tokens = {
            "name",
            "email",
            "phone",
            "address",
            "ssn",
            "dob",
            "date_of_birth",
            "patient_id",
            "mrn",
            "social_security",
            "passport",
            "driver_license",
        }
        anonymized: list[dict] = []
        for rec in analyses:
            clean: dict[str, Any] = {"anonymous_id": _random_id()}
            for key, value in rec.items():
                kl = key.lower()
                if any(tok in kl for tok in pii_tokens):
                    continue
                if "date" in kl and isinstance(value, str) and len(value) >= 10:
                    clean[key] = value[:4]  # year only
                else:
                    clean[key] = copy.deepcopy(value)
            anonymized.append(clean)
        return anonymized

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _all_keys(analyses: list[dict]) -> list[str]:
        """Unique keys across records in first-seen order."""
        seen: dict[str, None] = {}
        for rec in analyses:
            for k in rec:
                seen.setdefault(k, None)
        return list(seen)

    def _write_csv(self, a: list[dict], ts: str) -> str:
        return self._write_delimited(a, os.path.join(self.output_dir, f"export_{ts}.csv"), ",")

    def _write_tsv(self, a: list[dict], ts: str) -> str:
        return self._write_delimited(a, os.path.join(self.output_dir, f"export_{ts}.tsv"), "\t")

    def _write_delimited(self, analyses: list[dict], path: str, delim: str) -> str:
        keys = self._all_keys(analyses) if analyses else []
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=delim)
            w.writerow(keys)
            for rec in analyses:
                w.writerow([rec.get(k, "") for k in keys])
        return os.path.abspath(path)

    def _write_json(self, analyses: list[dict], ts: str) -> str:
        path = os.path.join(self.output_dir, f"export_{ts}.json")
        payload = {
            "exported_at": datetime.now(UTC).isoformat(),
            "n_records": len(analyses),
            "records": analyses,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return os.path.abspath(path)

    def _write_parquet_json(self, analyses: list[dict], ts: str) -> str:
        """Columnar (Parquet-like) JSON: data stored column-by-column."""
        path = os.path.join(self.output_dir, f"export_{ts}.parquet.json")
        keys = self._all_keys(analyses) if analyses else []
        columns = {k: [rec.get(k) for rec in analyses] for k in keys}
        payload = {
            "format": "columnar_json",
            "exported_at": datetime.now(UTC).isoformat(),
            "n_records": len(analyses),
            "n_columns": len(keys),
            "columns": columns,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return os.path.abspath(path)

    @staticmethod
    def _write_empty(path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")

    @staticmethod
    def _infer_r_types(analyses: list[dict], keys: list[str]) -> dict[str, str]:
        """Map columns to R colClasses (logical/integer/numeric/character)."""
        return _infer_types(analyses, keys, _R_TYPES, "character")

    @staticmethod
    def _infer_pandas_dtypes(analyses: list[dict], keys: list[str]) -> dict[str, str]:
        """Map columns to pandas dtypes (bool/int64/float64/object)."""
        return _infer_types(analyses, keys, _PD_TYPES, "object")


# ---------------------------------------------------------------------------
# CohortBuilder
# ---------------------------------------------------------------------------


class CohortBuilder:
    """Build research cohorts from telomere analysis data.

    Define inclusion/exclusion criteria, filter records, perform greedy
    nearest-neighbour case-control matching, and estimate statistical power.
    Example: ``builder.define_cohort({"name": "elderly", "inclusion": {"age": {"min": 65}}})``
    """

    def __init__(self) -> None:
        self._cohorts: list[CohortDefinition] = []

    def define_cohort(self, criteria: dict) -> CohortDefinition:
        """Create a CohortDefinition from a criteria dict.

        Expects keys: ``name`` (str), optional ``inclusion``/``exclusion`` dicts,
        optional ``description``.  Returns and stores the new definition.
        """
        cohort = CohortDefinition(
            name=criteria.get("name", "unnamed_cohort"),
            inclusion_criteria=criteria.get("inclusion", {}),
            exclusion_criteria=criteria.get("exclusion", {}),
            description=criteria.get("description", ""),
        )
        self._cohorts.append(cohort)
        return cohort

    def apply_criteria(self, analyses: list[dict], cohort: CohortDefinition) -> list[dict]:
        """Filter *analyses* to records satisfying all inclusion and no exclusion criteria."""
        result: list[dict] = []
        for rec in analyses:
            if cohort.inclusion_criteria and not _matches_criteria(rec, cohort.inclusion_criteria):
                continue
            if cohort.exclusion_criteria and _matches_criteria(rec, cohort.exclusion_criteria):
                continue
            result.append(rec)
        return result

    def match_controls(
        self, cases: list[dict], pool: list[dict], match_vars: list[str], ratio: int = 1
    ) -> list[dict]:
        """Greedy nearest-neighbour case-control matching without replacement.

        For each case, selects *ratio* closest controls from *pool* based on
        Euclidean distance of standardised *match_vars*. Randomise case order
        beforehand to reduce greedy bias.
        """
        if not cases or not pool or not match_vars:
            return []
        all_recs = cases + pool
        stats: dict[str, tuple[float, float]] = {}
        for var in match_vars:
            vals = _numeric_values(all_recs, var)
            m, s = _mean(vals), _std(vals)
            stats[var] = (m, s if s > 0 else 1.0)

        def _stdz(rec: dict) -> list[float]:
            out: list[float] = []
            for var in match_vars:
                try:
                    v = float(rec.get(var, 0))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    v = 0.0
                m, s = stats[var]
                out.append((v - m) / s)
            return out

        available = list(range(len(pool)))
        pool_vecs = [_stdz(r) for r in pool]
        selected: list[dict] = []
        for case in cases:
            cvec = _stdz(case)
            dists = sorted(
                (
                    (i, math.sqrt(sum((a - b) ** 2 for a, b in zip(cvec, pool_vecs[i]))))
                    for i in available
                ),
                key=lambda t: t[1],
            )
            taken: list[int] = []
            for idx, _ in dists:
                if len(taken) >= ratio:
                    break
                selected.append(pool[idx])
                taken.append(idx)
            for idx in taken:
                available.remove(idx)
        return selected

    def calculate_power(self, effect_size: float, n: int, alpha: float = 0.05) -> float:
        """Estimate power for a two-sample t-test (normal approximation).

        power ≈ Φ(|d|·√(n/2) − z_{1−α/2}) where d = Cohen's d, n = per-group size.
        Returns a float between 0 and 1.
        """
        if n < 2 or effect_size == 0.0:
            return 0.0
        z_crit = _inverse_normal(1 - alpha / 2)
        ncp = abs(effect_size) * math.sqrt(n / 2)
        return round(min(max(_normal_cdf(ncp - z_crit), 0.0), 1.0), 6)

    def power_analysis(self, effect_size: float, n: int, alpha: float = 0.05) -> PowerAnalysis:
        """Full power analysis with required n for 80% power."""
        power = self.calculate_power(effect_size, n, alpha)
        req_n = self._required_n_for_power(effect_size, alpha, 0.80)
        return PowerAnalysis(
            effect_size=effect_size, n=n, alpha=alpha, power=power, required_n_for_80_pct=req_n
        )

    @staticmethod
    def _required_n_for_power(effect_size: float, alpha: float, target: float = 0.80) -> int:
        """Find minimum per-group n achieving *target* power analytically."""
        if effect_size == 0.0:
            return 0
        z_crit = _inverse_normal(1 - alpha / 2)
        z_beta = _inverse_normal(target)
        approx = math.ceil(2 * ((z_crit + z_beta) / abs(effect_size)) ** 2)
        lo, hi = max(2, approx - 5), approx + 10
        for cand in range(lo, hi + 1):
            ncp = abs(effect_size) * math.sqrt(cand / 2)
            if _normal_cdf(ncp - z_crit) >= target:
                return cand
        return hi


# ---------------------------------------------------------------------------
# CitationGenerator
# ---------------------------------------------------------------------------


class CitationGenerator:
    """Generate standardised citations and methods text for Teloscopy analyses.

    Automates BibTeX entries, methods paragraphs, and data-availability
    statements for academic publications.
    """

    def __init__(self, version: str = "2.0") -> None:
        self.version = version
        self._year = datetime.now(UTC).year

    def generate_methods_section(self) -> str:
        """Return a standard methods paragraph for publications.

        Covers software version, read processing, telomere quantification
        (T/S ratio), QC thresholds (CV < 10%, depth >= 30x), and
        statistical methods (Pearson r, power analysis, cohort matching).
        """
        return (
            f"Telomere length analysis was performed using Teloscopy "
            f"v{self.version} (https://github.com/teloscopy/teloscopy). "
            f"Raw sequencing reads were quality-filtered and aligned to the "
            f"reference genome. Telomere repeat content was quantified using "
            f"the Teloscopy measurement pipeline, which identifies canonical "
            f"(TTAGGG)n repeats and common variant motifs. Relative telomere "
            f"length (T/S ratio) was computed as the ratio of telomere repeat "
            f"copy number to a single-copy reference gene. Quality control "
            f"metrics including coefficient of variation (CV < 10%) and "
            f"minimum read depth (>= 30x) were applied. Statistical analyses "
            f"of telomere measurements were conducted using built-in "
            f"Pearson correlation, descriptive statistics, and power "
            f"analysis modules. Cohort selection employed greedy nearest-"
            f"neighbour matching on standardised covariates."
        )

    def generate_bibtex(self) -> str:
        """Return a BibTeX @software entry for citing Teloscopy."""
        return (
            f"@software{{teloscopy{self._year},\n"
            f"  author    = {{Teloscopy Development Team}},\n"
            f"  title     = {{Teloscopy: Telomere Analysis Toolkit}},\n"
            f"  version   = {{{self.version}}},\n"
            f"  year      = {{{self._year}}},\n"
            f"  url       = {{https://github.com/teloscopy/teloscopy}},\n"
            f"  note      = {{Software for telomere length measurement "
            f"and analysis from sequencing data}}\n"
            f"}}"
        )

    def generate_data_availability_statement(self, export_path: str) -> str:
        """Return a data-availability statement referencing *export_path*.

        Mentions anonymisation, software version, and metadata reproducibility.
        """
        return (
            f"The telomere analysis data supporting the findings of this "
            f"study are available at: {export_path}. Data were exported using "
            f"Teloscopy v{self.version} ResearchExporter and have been "
            f"anonymised to remove personally identifiable information in "
            f"compliance with applicable data protection regulations. "
            f"The analysis pipeline configuration and software version are "
            f"recorded in the accompanying metadata files to ensure "
            f"reproducibility."
        )

    def generate_acknowledgement(self) -> str:
        """Return a suggested acknowledgement sentence for Teloscopy."""
        return (
            f"We acknowledge the use of Teloscopy v{self.version} for "
            f"telomere length analysis and research data export."
        )
