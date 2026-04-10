"""Age-telomere correlation analysis.

Provides Pearson/Spearman correlation, two-phase piecewise linear regression,
comparison with published attrition rates, and forward trajectory projection.

Published attrition rates: ~80 bp/yr childhood, ~25-30 bp/yr adulthood
(Frenck 1998, Müezzinler 2013).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

_CHILDHOOD_RATE: float = 0.080   # kb/yr (80 bp/yr)
_ADULT_RATE: float = 0.027       # kb/yr (27 bp/yr)
_BREAKPOINT: int = 18            # childhood/adult transition age


def compute_age_telomere_correlation(
    ages: list[int],
    telomere_lengths: list[float],
    sexes: list[str] | None = None,
) -> dict[str, Any]:
    """Compute correlation between age and telomere length.

    Returns Pearson r, Spearman rho, piecewise model, published comparison,
    and optionally sex-stratified statistics.
    """
    if len(ages) != len(telomere_lengths):
        raise ValueError("ages and telomere_lengths must have the same length.")
    if len(ages) < 5:
        raise ValueError("At least 5 data points are required.")

    x = np.asarray(ages, dtype=np.float64)
    y = np.asarray(telomere_lengths, dtype=np.float64)
    pr, pp = _pearson(x, y)
    sr, sp = _spearman(x, y)
    pw = _fit_piecewise(x, y)

    result: dict[str, Any] = {
        "n": len(ages), "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp,
        "piecewise_model": pw,
        "published_comparison": _compare_published(pw),
    }

    if sexes is not None:
        if len(sexes) != len(ages):
            raise ValueError("sexes must match ages in length.")
        sex_arr = np.array([s.lower().strip() for s in sexes])
        stratified: dict[str, Any] = {}
        for sv in np.unique(sex_arr):
            m = sex_arr == sv
            if np.sum(m) < 5:
                continue
            r1, p1 = _pearson(x[m], y[m])
            r2, p2 = _spearman(x[m], y[m])
            stratified[str(sv)] = {
                "n": int(np.sum(m)), "pearson_r": r1, "pearson_p": p1,
                "spearman_rho": r2, "spearman_p": p2,
                "mean_tl_kb": round(float(np.mean(y[m])), 4),
                "sd_tl_kb": round(float(np.std(y[m], ddof=1)), 4),
            }
        result["sex_stratified"] = stratified
    return result


def predict_telomere_trajectory(
    current_age: int,
    current_tl_kb: float,
    sex: str,
    years_forward: int = 20,
    attrition_override_kb_per_year: float | None = None,
) -> list[dict[str, Any]]:
    """Project future telomere length with 95% confidence intervals.

    Uses published attrition rates by default. CI widens with sqrt(time).
    Returns one dict per year: {age, predicted_tl, lower_95, upper_95}.
    """
    if years_forward < 1:
        raise ValueError("years_forward must be >= 1")

    base_sd = 0.3  # kb/year SD from population studies
    trajectory: list[dict[str, Any]] = []
    cumulative_loss = 0.0
    for dt in range(1, years_forward + 1):
        future_age = current_age + dt
        if attrition_override_kb_per_year is not None:
            rate = attrition_override_kb_per_year
        else:
            rate = _CHILDHOOD_RATE if future_age < _BREAKPOINT else _ADULT_RATE
        cumulative_loss += rate
        pred = current_tl_kb - cumulative_loss
        sd = base_sd * math.sqrt(dt)
        trajectory.append({
            "age": future_age,
            "predicted_tl": round(max(pred, 0.0), 3),
            "lower_95": round(max(pred - 1.96 * sd, 0.0), 3),
            "upper_95": round(pred + 1.96 * sd, 3),
        })
    return trajectory


# --- Internal helpers -------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r with two-tailed p-value."""
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0 or n <= 2:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r**2))
    p = 2.0 * _t_surv(abs(t), n - 2)
    return round(r, 6), round(p, 8)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation."""
    return _pearson(_rank(x), _rank(y))


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    return r


def _t_surv(t: float, df: int) -> float:
    """Upper-tail t-distribution probability (normal approx for df>30)."""
    if df > 30:
        return 1.0 - 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    x = df / (df + t**2)
    return min(1.0, max(0.0, 0.5 * x ** (df / 2.0)))


def _fit_piecewise(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Two-phase piecewise linear model at age 18 breakpoint."""
    result: dict[str, Any] = {"breakpoint_age": _BREAKPOINT}
    for label, mask in [("childhood", x < _BREAKPOINT), ("adulthood", x >= _BREAKPOINT)]:
        n = int(np.sum(mask))
        if n < 2:
            result[label] = {"slope_kb_per_year": None, "intercept_kb": None, "n": n}
            continue
        c = np.polyfit(x[mask], y[mask], 1)
        yp = np.polyval(c, x[mask])
        ss_r = float(np.sum((y[mask] - yp) ** 2))
        ss_t = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
        result[label] = {
            "slope_kb_per_year": round(float(c[0]), 5),
            "intercept_kb": round(float(c[1]), 3),
            "r_squared": round(1.0 - ss_r / ss_t if ss_t > 0 else 1.0, 4),
            "n": n,
        }
    return result


def _compare_published(pw: dict[str, Any]) -> dict[str, Any]:
    """Compare fitted attrition rates with published values."""
    comp: dict[str, Any] = {}
    for phase, pub in [("childhood", -_CHILDHOOD_RATE), ("adulthood", -_ADULT_RATE)]:
        fitted = pw.get(phase, {}).get("slope_kb_per_year")
        if fitted is None:
            comp[phase] = {"status": "insufficient_data"}
            continue
        diff = abs(fitted - pub)
        comp[phase] = {
            "fitted_kb_per_year": fitted, "published_kb_per_year": pub,
            "absolute_difference": round(diff, 5),
            "within_expected_range": diff < 0.03,
        }
    return comp
