"""Case vs control statistical comparison for telomere length studies.

Automated group comparison with normality testing, non-parametric
fall-back (Mann-Whitney U), effect-size estimators (Cohen's d, Hedges' g,
Glass's delta), bootstrap confidence intervals, post-hoc power analysis,
and stratified comparisons by age and sex.

References
----------
Cohen J (1988). *Statistical Power Analysis for the Behavioral Sciences*.
Hedges LV & Olkin I (1985). *Statistical Methods for Meta-Analysis*.
Müezzinler A et al. (2013). Ageing Research Reviews 12(2):509-519.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ComparisonResult",
    "GroupSummary",
    "SubgroupAnalysis",
    "ComprehensiveComparison",
    "compare_groups",
    "compare_age_matched",
    "compare_sex_stratified",
    "compute_effect_size",
    "power_analysis",
    "generate_comparison_report",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of a two-group statistical test."""

    test_name: str                            # e.g. "Welch's t-test"
    statistic: float                          # test statistic value
    p_value: float                            # two-tailed p-value
    effect_size: float                        # standardised effect size
    effect_size_type: str                     # e.g. "Cohen's d", "Hedges' g"
    confidence_interval_95: tuple[float, float]  # 95 % CI for effect size
    significant: bool                         # p_value < alpha
    interpretation: str                       # human-readable summary


@dataclass
class GroupSummary:
    """Descriptive statistics for one group."""

    label: str      # "case" or "control"
    n: int          # sample size
    mean: float
    median: float
    std: float      # sample SD (ddof=1)
    min_val: float
    max_val: float
    q25: float
    q75: float
    iqr: float      # q75 − q25


@dataclass
class SubgroupAnalysis:
    """Comparison results stratified by a categorical variable."""

    subgroup_variable: str                    # e.g. "age_bin"
    subgroups: dict[str, ComparisonResult]    # label → result


@dataclass
class ComprehensiveComparison:
    """Full case-vs-control comparison output."""

    main_comparison: ComparisonResult
    case_summary: GroupSummary
    control_summary: GroupSummary
    normality_tests: dict                     # per-group D'Agostino-Pearson
    variance_test: dict                       # Levene's test
    bootstrap_ci: tuple[float, float] | None  # 95 % CI for mean diff
    subgroup_analyses: list[SubgroupAnalysis] | None
    power_analysis: dict                      # post-hoc power + MDE
    clinical_significance: str                # qualitative assessment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_AGE_BINS: list[tuple[int, int]] = [
    (0, 20), (20, 40), (40, 60), (60, 80), (80, 150),
]

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compare_groups(
    cases: list[float] | np.ndarray,
    controls: list[float] | np.ndarray,
    *,
    test: str = "auto",
    alpha: float = 0.05,
    bootstrap_n: int = 1000,
) -> ComprehensiveComparison:
    """Compare telomere lengths between a case and a control group.

    Parameters
    ----------
    cases, controls : array-like of float
        Telomere-length measurements.
    test : ``"auto"`` | ``"t_test"`` | ``"mann_whitney"``
        Test selection strategy (default ``"auto"``).
    alpha : float
        Significance threshold (default 0.05).
    bootstrap_n : int
        Bootstrap resamples for mean-difference CI (0 to skip).

    Returns
    -------
    ComprehensiveComparison
    """
    ca = np.asarray(cases, dtype=np.float64)
    ct = np.asarray(controls, dtype=np.float64)
    if len(ca) < 3 or len(ct) < 3:
        raise ValueError("Each group must contain at least 3 observations.")

    case_summ = _summarise_group(ca, "case")
    ctrl_summ = _summarise_group(ct, "control")

    # Normality ----------------------------------------------------------------
    cs_stat, cs_p = _normality_test(ca)
    ct_stat, ct_p = _normality_test(ct)
    normality: dict[str, Any] = {
        "case": {"statistic": round(cs_stat, 4), "p_value": round(cs_p, 6)},
        "control": {"statistic": round(ct_stat, 4), "p_value": round(ct_p, 6)},
        "both_normal": cs_p > alpha and ct_p > alpha,
    }

    # Levene -------------------------------------------------------------------
    lev_stat, lev_p = _levene_test(ca, ct)
    variance: dict[str, Any] = {
        "test": "Levene", "statistic": round(lev_stat, 4),
        "p_value": round(lev_p, 6), "equal_variances": lev_p > alpha,
    }

    # Select and run test ------------------------------------------------------
    chosen = test
    if chosen == "auto":
        chosen = "t_test" if normality["both_normal"] else "mann_whitney"
    if chosen == "t_test":
        stat, p_val = _welch_t_test(ca, ct)
        test_label = "Welch's t-test"
    else:
        stat, p_val = _mann_whitney_u(ca, ct)
        test_label = "Mann-Whitney U"

    es, es_ci = compute_effect_size(ca, ct, method="hedges_g")
    sig = p_val < alpha
    main = ComparisonResult(
        test_name=test_label, statistic=round(stat, 4),
        p_value=round(p_val, 8), effect_size=round(es, 4),
        effect_size_type="Hedges' g",
        confidence_interval_95=(round(es_ci[0], 4), round(es_ci[1], 4)),
        significant=sig,
        interpretation=_interpret_result(test_label, p_val, es, alpha),
    )
    boot = _bootstrap_mean_diff_ci(ca, ct, n=bootstrap_n) if bootstrap_n > 0 else None
    pwr = power_analysis(abs(es), len(ca), len(ct), alpha=alpha)

    return ComprehensiveComparison(
        main_comparison=main, case_summary=case_summ,
        control_summary=ctrl_summ, normality_tests=normality,
        variance_test=variance, bootstrap_ci=boot,
        subgroup_analyses=None, power_analysis=pwr,
        clinical_significance=_assess_clinical(es, p_val, len(ca), len(ct)),
    )


def compare_age_matched(
    case_ages: list[int] | np.ndarray,
    case_tl: list[float] | np.ndarray,
    control_ages: list[int] | np.ndarray,
    control_tl: list[float] | np.ndarray,
    age_bins: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Age-stratified case-vs-control comparison.

    Returns ``{"overall": ComprehensiveComparison, "per_bin": {...}}``.
    Default *age_bins*: ``[(0,20), (20,40), (40,60), (60,80), (80,150)]``.
    Bins with < 3 observations per group return insufficient-data dicts.
    """
    ca = np.asarray(case_ages, dtype=np.float64)
    ka = np.asarray(control_ages, dtype=np.float64)
    c_tl = np.asarray(case_tl, dtype=np.float64)
    k_tl = np.asarray(control_tl, dtype=np.float64)
    if len(ca) != len(c_tl):
        raise ValueError("case_ages and case_tl must have the same length.")
    if len(ka) != len(k_tl):
        raise ValueError("control_ages and control_tl must have the same length.")

    bins = age_bins if age_bins is not None else _DEFAULT_AGE_BINS
    per_bin: dict[str, Any] = {}
    for lo, hi in bins:
        cm, km = (ca >= lo) & (ca < hi), (ka >= lo) & (ka < hi)
        cv, kv = c_tl[cm], k_tl[km]
        if len(cv) < 3 or len(kv) < 3:
            per_bin[f"{lo}-{hi}"] = {"status": "insufficient_data",
                                     "case_n": int(len(cv)),
                                     "control_n": int(len(kv))}
        else:
            per_bin[f"{lo}-{hi}"] = compare_groups(cv, kv)
    return {"overall": compare_groups(c_tl, k_tl), "per_bin": per_bin}


def compare_sex_stratified(
    case_tl: list[float] | np.ndarray,
    case_sexes: list[str],
    control_tl: list[float] | np.ndarray,
    control_sexes: list[str],
) -> dict[str, Any]:
    """Sex-stratified case-vs-control comparison.

    Sex labels are normalised internally (``"m"``/``"male"`` and
    ``"f"``/``"female"`` accepted).  Returns ``"overall"``, ``"male"``,
    ``"female"`` comparisons.
    """
    c_tl = np.asarray(case_tl, dtype=np.float64)
    k_tl = np.asarray(control_tl, dtype=np.float64)
    c_sex = np.array([s.lower().strip() for s in case_sexes])
    k_sex = np.array([s.lower().strip() for s in control_sexes])
    if len(c_tl) != len(c_sex):
        raise ValueError("case_tl and case_sexes must have the same length.")
    if len(k_tl) != len(k_sex):
        raise ValueError("control_tl and control_sexes must have the same length.")

    result: dict[str, Any] = {"overall": compare_groups(c_tl, k_tl)}
    for label, aliases in [("male", {"m", "male"}), ("female", {"f", "female"})]:
        cm, km = np.isin(c_sex, list(aliases)), np.isin(k_sex, list(aliases))
        cv, kv = c_tl[cm], k_tl[km]
        if len(cv) < 3 or len(kv) < 3:
            result[label] = {"status": "insufficient_data",
                             "case_n": int(len(cv)), "control_n": int(len(kv))}
        else:
            result[label] = compare_groups(cv, kv)
    return result


def compute_effect_size(
    cases: list[float] | np.ndarray,
    controls: list[float] | np.ndarray,
    *,
    method: str = "hedges_g",
) -> tuple[float, tuple[float, float]]:
    """Compute a standardised effect size with 95 % CI.

    *method*: ``"cohen_d"`` | ``"hedges_g"`` (default) | ``"glass_delta"``.
    Returns ``(effect_size, (lower_95, upper_95))``.
    """
    ca = np.asarray(cases, dtype=np.float64)
    ct = np.asarray(controls, dtype=np.float64)
    n1, n2 = len(ca), len(ct)
    m1, m2 = float(np.mean(ca)), float(np.mean(ct))
    v1, v2 = float(np.var(ca, ddof=1)), float(np.var(ct, ddof=1))

    if method == "cohen_d":
        sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = (m1 - m2) / sp if sp > 0 else 0.0
    elif method == "hedges_g":
        sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d_raw = (m1 - m2) / sp if sp > 0 else 0.0
        d = d_raw * (1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0))
    elif method == "glass_delta":
        d = (m1 - m2) / (math.sqrt(v2) if v2 > 0 else 1e-12)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Approximate SE (Hedges & Olkin 1985)
    se = math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2.0 * (n1 + n2)))
    return round(d, 6), (round(d - 1.96 * se, 6), round(d + 1.96 * se, 6))


def power_analysis(
    effect_size: float, n_cases: int, n_controls: int, *, alpha: float = 0.05,
) -> dict[str, Any]:
    """Post-hoc statistical power via normal approximation.

    Returns dict with ``power``, ``minimum_detectable_effect`` (at 80 %),
    ``n_cases``, ``n_controls``, ``alpha``.
    """
    n_h = 2.0 * n_cases * n_controls / (n_cases + n_controls)
    se = math.sqrt(2.0 / n_h)
    z_a = _normal_ppf(1.0 - alpha / 2.0)
    ncp = abs(effect_size) / se
    achieved = 1.0 - _normal_cdf(z_a - ncp) + _normal_cdf(-z_a - ncp)
    mde = (z_a + _normal_ppf(0.80)) * se
    return {
        "power": round(min(max(achieved, 0.0), 1.0), 4),
        "minimum_detectable_effect": round(max(mde, 0.0), 4),
        "n_cases": n_cases, "n_controls": n_controls, "alpha": alpha,
    }


def generate_comparison_report(comparison: ComprehensiveComparison) -> str:
    """Generate a human-readable plain-text report from *comparison*."""
    mc = comparison.main_comparison
    cs, ks = comparison.case_summary, comparison.control_summary
    sep = "=" * 72
    r = _row  # shorthand

    lines: list[str] = [
        sep, "  Case vs Control — Telomere Length Comparison Report", sep, "",
        "Group Summaries", "-" * 40,
        f"  {'':12s} {'Case':>12s}  {'Control':>12s}",
        r("N", cs.n, ks.n), r("Mean", cs.mean, ks.mean),
        r("Median", cs.median, ks.median), r("SD", cs.std, ks.std),
        r("Min", cs.min_val, ks.min_val), r("Max", cs.max_val, ks.max_val),
        r("Q25", cs.q25, ks.q25), r("Q75", cs.q75, ks.q75),
        r("IQR", cs.iqr, ks.iqr), "",
        "Normality Tests", "-" * 40,
    ]
    for gl in ("case", "control"):
        nt = comparison.normality_tests[gl]
        lines.append(f"  {gl.capitalize():12s}  K2={nt['statistic']:.4f}"
                     f"  p={nt['p_value']:.6f}")
    lines.append(f"  Both normal: {comparison.normality_tests['both_normal']}")

    vt = comparison.variance_test
    lines += [
        "", "Variance (Levene)", "-" * 40,
        f"  F={vt['statistic']:.4f}  p={vt['p_value']:.6f}"
        f"  equal={vt['equal_variances']}",
        "", "Primary Test", "-" * 40,
        f"  {mc.test_name}:  stat={mc.statistic:.4f}"
        f"  p={mc.p_value:.8f}  sig={mc.significant}",
        "", "Effect Size", "-" * 40,
        f"  {mc.effect_size_type}={mc.effect_size:.4f}"
        f"  95%CI=({mc.confidence_interval_95[0]:.4f},"
        f" {mc.confidence_interval_95[1]:.4f})",
    ]
    if comparison.bootstrap_ci is not None:
        lo, hi = comparison.bootstrap_ci
        lines += ["", "Bootstrap CI (Mean Diff)", "-" * 40,
                  f"  95%CI=({lo:.4f}, {hi:.4f})"]
    pw = comparison.power_analysis
    lines += [
        "", "Power Analysis", "-" * 40,
        f"  power={pw['power']:.4f}"
        f"  MDE(80%)={pw['minimum_detectable_effect']:.4f}",
        "", "Clinical Significance", "-" * 40,
        f"  {comparison.clinical_significance}",
        "", "Interpretation", "-" * 40,
        f"  {mc.interpretation}", "", sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers — descriptive
# ---------------------------------------------------------------------------

def _summarise_group(arr: np.ndarray, label: str) -> GroupSummary:
    """Build a GroupSummary from a 1-D array."""
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return GroupSummary(
        label=label, n=len(arr),
        mean=round(float(np.mean(arr)), 4),
        median=round(float(np.median(arr)), 4),
        std=round(float(np.std(arr, ddof=1)), 4),
        min_val=round(float(np.min(arr)), 4),
        max_val=round(float(np.max(arr)), 4),
        q25=round(q25, 4), q75=round(q75, 4), iqr=round(q75 - q25, 4),
    )


def _row(lbl: str, cv: object, kv: object) -> str:
    """Format a two-column report row."""
    if isinstance(cv, int):
        return f"  {lbl:12s} {cv:12d}  {kv:12d}"
    return f"  {lbl:12s} {cv:12.4f}  {kv:12.4f}"


# ---------------------------------------------------------------------------
# Internal helpers — statistical tests
# ---------------------------------------------------------------------------

def _welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's two-sample t-test (unequal variances)."""
    n1, n2 = len(a), len(b)
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 / n1 + v2 / n2) > 0 else 1e-12
    t = (float(np.mean(a)) - float(np.mean(b))) / se
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    return t, 2.0 * _t_survival(abs(t), num / den if den > 0 else 1.0)


def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U with continuity-corrected normal approximation."""
    ranks = _average_ranks(np.concatenate([a, b]))
    n1, n2 = len(a), len(b)
    u1 = float(np.sum(ranks[:n1])) - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (abs(u1 - mu) - 0.5) / sigma if sigma > 0 else 0.0
    return u, 2.0 * (1.0 - _normal_cdf(abs(z)))


def _normality_test(x: np.ndarray) -> tuple[float, float]:
    """D'Agostino-Pearson K-squared omnibus normality test."""
    n = len(x)
    if n < 8:
        return 0.0, 1.0
    s = float(np.std(x, ddof=1))
    if s == 0.0:
        return 0.0, 1.0
    z = (x - np.mean(x)) / s
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0
    # Skewness Z-score (D'Agostino 1970)
    y = skew * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    b2 = (3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3)
          / ((n - 2) * (n + 5) * (n + 7) * (n + 9)))
    w2 = math.sqrt(2.0 * (b2 - 1.0)) - 1.0 if b2 > 1.0 else 1.0
    dl = 1.0 / math.sqrt(math.log(math.sqrt(w2))) if w2 > 0 else 1.0
    al = math.sqrt(2.0 / (w2 - 1.0)) if w2 > 1.0 else 1.0
    r = y / al if al > 0 else 0.0
    z_sk = dl * math.log(r + math.sqrt(r ** 2 + 1.0)) if al > 0 else 0.0
    # Kurtosis Z-score
    vk = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    z_ku = (kurt - (3.0 * (n - 1) / (n + 1) - 3.0)) / math.sqrt(vk) if vk > 0 else 0.0
    k2 = z_sk ** 2 + z_ku ** 2
    p = math.exp(-k2 / 2.0) if k2 < 100.0 else 0.0
    return round(k2, 4), min(1.0, max(0.0, p))


def _levene_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Levene's test for equality of variances (median-based)."""
    da, db = np.abs(a - float(np.median(a))), np.abs(b - float(np.median(b)))
    n1, n2 = len(a), len(b)
    gm = float(np.mean(np.concatenate([da, db])))
    ma, mb = float(np.mean(da)), float(np.mean(db))
    ss_b = n1 * (ma - gm) ** 2 + n2 * (mb - gm) ** 2
    ss_w = float(np.sum((da - ma) ** 2) + np.sum((db - mb) ** 2))
    if ss_w == 0.0:
        return 0.0, 1.0
    f = ss_b / (ss_w / (n1 + n2 - 2))
    return f, _f_survival(f, 1, n1 + n2 - 2)


# ---------------------------------------------------------------------------
# Internal helpers — bootstrap & ranking
# ---------------------------------------------------------------------------

def _bootstrap_mean_diff_ci(
    a: np.ndarray, b: np.ndarray, *, n: int = 1000,
) -> tuple[float, float]:
    """Bootstrap 95 % CI for mean(case) − mean(control)."""
    rng = random.Random()
    al, bl = a.tolist(), b.tolist()
    diffs = sorted(
        sum(rng.choices(al, k=len(al))) / len(al)
        - sum(rng.choices(bl, k=len(bl))) / len(bl)
        for _ in range(n)
    )
    lo = max(0, int(math.floor(0.025 * n)))
    hi = min(n - 1, int(math.floor(0.975 * n)))
    return round(diffs[lo], 4), round(diffs[hi], 4)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling."""
    order = x.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j < len(sx) and sx[j] == sx[i]:
            j += 1
        if j > i + 1:
            avg = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks


# ---------------------------------------------------------------------------
# Internal helpers — distribution functions
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF (Abramowitz & Stegun 26.2.23)."""
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    if p == 0.5:
        return 0.0
    sign, q = (-1.0, p) if p < 0.5 else (1.0, 1.0 - p)
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return sign * (t - (c0 + c1 * t + c2 * t * t)
                   / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3))


def _t_survival(t: float, df: float) -> float:
    """Upper-tail Student-t probability."""
    if df > 100:
        return 1.0 - _normal_cdf(t)
    x = df / (df + t ** 2)
    try:
        lb = math.lgamma(df / 2) + math.lgamma(0.5) - math.lgamma(df / 2 + 0.5)
        r = 0.5 * x ** (df / 2) * (1.0 - x) ** 0.5 / (df / 2 * math.exp(lb))
    except (OverflowError, ValueError):
        r = 1.0 - _normal_cdf(t)
    return min(1.0, max(0.0, r))


def _f_survival(f: float, df1: int, df2: int) -> float:
    """Approximate upper-tail F probability via incomplete beta."""
    if f <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    a, b = df2 / 2.0, df1 / 2.0
    try:
        lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        pf = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lb)
        total, term = 1.0 / a, 1.0
        for k in range(1, 60):
            term *= (k - b) * x / k
            total += term / (a + k)
            if abs(term / (a + k)) < 1e-10:
                break
        r = pf * total
    except (OverflowError, ValueError, ZeroDivisionError):
        z = math.sqrt(2.0 * f * df1 / df2) - math.sqrt(2.0 * df1 - 1.0)
        r = 1.0 - _normal_cdf(z / math.sqrt(2.0))
    return min(1.0, max(0.0, r))


# ---------------------------------------------------------------------------
# Internal helpers — interpretation
# ---------------------------------------------------------------------------


def _interpret_result(
    test_name: str, p_value: float, effect_size: float, alpha: float,
) -> str:
    """Human-readable interpretation string."""
    ad = abs(effect_size)
    mag = ("negligible" if ad < 0.20 else "small" if ad < 0.50
           else "medium" if ad < 0.80 else "large")
    direction = "higher" if effect_size > 0 else "lower"
    if p_value < alpha:
        return (f"The {test_name} indicates a statistically significant difference "
                f"(p={p_value:.4g}) with a {mag} effect size ({effect_size:+.3f}). "
                f"Cases have {direction} telomere lengths than controls.")
    return (f"The {test_name} does not detect a statistically significant "
            f"difference (p={p_value:.4g}). The effect size is {mag} "
            f"({effect_size:+.3f}).")


def _assess_clinical(
    effect_size: float, p_value: float, n1: int, n2: int,
) -> str:
    """Qualitative clinical-significance assessment."""
    ad = abs(effect_size)
    total = n1 + n2
    if ad >= 0.80 and p_value < 0.05:
        return ("Clinically significant — large effect with statistical support. "
                "This difference is likely biologically meaningful.")
    if ad >= 0.50 and p_value < 0.05:
        return ("Likely clinically relevant — moderate effect with statistical support. "
                "Consider replication in an independent cohort.")
    if ad >= 0.80:
        return ("Potentially clinically relevant — large effect but not statistically "
                "significant. Consider increasing sample size.")
    if ad < 0.20 and p_value < 0.05 and total > 200:
        return ("Statistically significant but clinically negligible. The large "
                "sample size may have inflated significance of a trivial effect.")
    if ad < 0.20:
        return "No clinically meaningful difference detected. Effect size is negligible."
    if p_value < 0.05:
        return ("Statistically significant with a small-to-moderate effect. "
                "Clinical relevance should be evaluated in context.")
    return ("Not statistically significant. Insufficient evidence for a clinically "
            "meaningful difference at the current sample size.")
