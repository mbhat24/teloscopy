"""
teloscopy.tracking.longitudinal
===============================

Longitudinal tracking of telomere-length dynamics for individual patients.

This module implements recording, statistical analysis, anomaly detection,
and projection of telomere measurements over time.  It is backed by a
lightweight JSON-per-patient storage layer so that data remains portable
and human-readable.

Scientific references
---------------------
* Müezzinler A, Zaineddin AK, Brenner H (2013). "A systematic review of
  leukocyte telomere length and age in adults." *Ageing Research Reviews*
  12(2): 509–519.  — Meta-analysis providing age-stratified reference
  ranges used here for population-percentile calculations.
* Hastie ND, Dempster M, Dunlop MG *et al.* (1990). "Telomere reduction
  in human colorectal carcinoma and with ageing." *Nature* 346: 866–868.
  — Seminal paper establishing ~30-60 bp/year leukocyte telomere
  attrition rates in healthy adults.
* Aubert G, Lansdorp PM (2008). "Telomeres and aging."
  *Physiological Reviews* 88(2): 557–579.  — Comprehensive review
  covering measurement methods, biological variance, and clinical
  interpretation of telomere dynamics.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Population reference data
# ---------------------------------------------------------------------------
# Mean telomere length (kb) and SD by decade, derived from Müezzinler 2013
# meta-analysis of leukocyte telomere length (LTL) in adults.
# Keys are (decade_start, decade_end) inclusive.
_POPULATION_REFERENCE: dict[tuple[int, int], tuple[float, float]] = {
    (0, 9): (10.50, 1.40),
    (10, 19): (9.80, 1.30),
    (20, 29): (9.10, 1.20),
    (30, 39): (8.40, 1.15),
    (40, 49): (7.70, 1.10),
    (50, 59): (7.10, 1.05),
    (60, 69): (6.50, 1.00),
    (70, 79): (6.00, 0.95),
    (80, 89): (5.60, 0.90),
    (90, 120): (5.30, 0.85),
}

# Normal attrition rate range in bp/year (Hastie 1990, Aubert & Lansdorp 2008)
_NORMAL_ATTRITION_LOW_BP = 20.0
_NORMAL_ATTRITION_HIGH_BP = 70.0
_NORMAL_ATTRITION_MEAN_BP = 40.0

# Anomaly detection thresholds
_ANOMALY_ZSCORE_WARN = 2.0
_ANOMALY_ZSCORE_CRITICAL = 3.0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _population_stats_for_age(age: int) -> tuple[float, float]:
    """Return (mean_kb, sd_kb) for a given chronological age."""
    for (lo, hi), (mean, sd) in _POPULATION_REFERENCE.items():
        if lo <= age <= hi:
            return mean, sd
    # Extrapolate for ages outside table
    if age < 0:
        return _POPULATION_REFERENCE[(0, 9)]
    return _POPULATION_REFERENCE[(90, 120)]


def _z_score(value: float, mean: float, sd: float) -> float:
    """Standard z-score."""
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def _percentile_from_z(z: float) -> float:
    """Approximate percentile from z-score using the logistic approximation
    of the normal CDF (Bowling *et al.* 2009)."""
    return 100.0 / (1.0 + math.exp(-1.7 * z - 0.73 * z**3 / 6.0))


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple ordinary-least-squares linear regression.

    Returns (slope, intercept).
    """
    n = len(xs)
    if n < 2:
        raise ValueError("At least two data points required for regression")
    x_bar = statistics.mean(xs)
    y_bar = statistics.mean(ys)
    ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_bar) ** 2 for x in xs)
    if ss_xx == 0:
        return 0.0, y_bar
    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar
    return slope, intercept


def _residual_std_error(xs: list[float], ys: list[float], slope: float, intercept: float) -> float:
    """Root-mean-square residual error of a linear fit."""
    n = len(xs)
    if n <= 2:
        return 0.0
    residuals = [(y - (slope * x + intercept)) for x, y in zip(xs, ys)]
    ss_res = sum(r**2 for r in residuals)
    return math.sqrt(ss_res / (n - 2))


def _bootstrap_slope(
    xs: list[float],
    ys: list[float],
    n_iterations: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap estimate of slope with confidence interval.

    Returns (mean_slope, ci_lower, ci_upper).
    """
    rng = random.Random(seed if seed is not None else 42)
    n = len(xs)
    slopes: list[float] = []
    for _ in range(n_iterations):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [xs[i] for i in indices]
        by = [ys[i] for i in indices]
        # Skip degenerate samples
        if len(set(bx)) < 2:
            continue
        s, _ = _linear_regression(bx, by)
        slopes.append(s)
    if not slopes:
        s, _ = _linear_regression(xs, ys)
        return s, s, s
    slopes.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * len(slopes)))
    hi_idx = min(len(slopes) - 1, int((1 - alpha) * len(slopes)))
    return statistics.mean(slopes), slopes[lo_idx], slopes[hi_idx]


def _bayesian_changepoint_scores(ys: list[float], prior_scale: float = 1.0) -> list[float]:
    """Simplified Bayesian online change-point detection.

    Returns a score for each interior point (indices 1..n-2).  Higher
    scores indicate a more probable change-point.  This is a simplified
    version that compares the log-likelihood of a two-segment model vs.
    a single-segment model at each candidate split.

    Parameters
    ----------
    ys : list[float]
        Ordered sequence of observations.
    prior_scale : float
        Prior variance scale — larger values make the detector less
        sensitive (fewer false positives).
    """
    n = len(ys)
    if n < 4:
        return [0.0] * max(0, n - 2)

    def _segment_loglik(segment: list[float]) -> float:
        k = len(segment)
        if k < 2:
            return 0.0
        mu = statistics.mean(segment)
        var = statistics.variance(segment) + 1e-12
        return -0.5 * k * math.log(2 * math.pi * var) - sum((v - mu) ** 2 for v in segment) / (
            2 * var
        )

    full_ll = _segment_loglik(ys)
    scores: list[float] = []
    for t in range(1, n - 1):
        left = ys[: t + 1]
        right = ys[t + 1 :]
        split_ll = _segment_loglik(left) + _segment_loglik(right)
        # Log Bayes factor (split vs. no-split), penalised by prior
        log_bf = split_ll - full_ll - math.log(prior_scale)
        scores.append(max(0.0, log_bf))
    return scores


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Measurement:
    """A single telomere-length measurement for a patient.

    Attributes
    ----------
    measurement_id : str
        Unique identifier (UUID4).
    patient_id : str
        Patient identifier.
    timestamp : str
        ISO-8601 UTC timestamp of when the measurement was recorded.
    telomere_length_kb : float
        Mean telomere restriction-fragment length in kilobases.
    biological_age : int
        Estimated biological age at time of measurement.
    chronological_age : int
        Actual chronological age.
    method : str
        Assay method (e.g. ``"TRF"``, ``"qPCR"``, ``"FlowFISH"``).
    confidence : float
        Confidence/quality score in [0, 1].  Defaults to 1.0.
    metadata : dict
        Arbitrary extra fields (tissue type, lab, batch, etc.).
    """

    measurement_id: str
    patient_id: str
    timestamp: str
    telomere_length_kb: float
    biological_age: int
    chronological_age: int
    method: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PatientHistory:
    """Complete measurement history for a patient.

    Attributes
    ----------
    patient_id : str
        Patient identifier.
    measurements : list[Measurement]
        Chronologically ordered measurements.
    age_range : tuple[int, int]
        (min_age, max_age) over the recorded history.
    span_years : float
        Time span covered in fractional years.
    measurement_count : int
        Total number of measurements.
    methods_used : list[str]
        Distinct assay methods appearing in the history.
    """

    patient_id: str
    measurements: list[Measurement]
    age_range: tuple[int, int]
    span_years: float
    measurement_count: int
    methods_used: list[str]


@dataclass
class AttritionAnalysis:
    """Telomere attrition-rate analysis.

    The primary metric is *rate_bp_per_year* — the estimated annual loss
    of telomere length in base-pairs, derived from ordinary-least-squares
    regression of length on chronological age.

    Attributes
    ----------
    patient_id : str
    rate_bp_per_year : float
        Point estimate (positive = shortening).
    ci_lower_bp : float
        Lower bound of the 95 % bootstrap confidence interval.
    ci_upper_bp : float
        Upper bound.
    r_squared : float
        Coefficient of determination of the linear fit.
    residual_se_kb : float
        Residual standard error of the fit in kb.
    population_comparison : str
        Qualitative label: ``"normal"``, ``"accelerated"``, ``"decelerated"``,
        or ``"elongating"`` relative to published norms.
    n_measurements : int
        Number of data points used.
    reference : str
        Literature reference for comparison thresholds.
    """

    patient_id: str
    rate_bp_per_year: float
    ci_lower_bp: float
    ci_upper_bp: float
    r_squared: float
    residual_se_kb: float
    population_comparison: str
    n_measurements: int
    reference: str = (
        "Hastie et al. 1990, Nature 346:866; Aubert & Lansdorp 2008, Physiol Rev 88:557"
    )


@dataclass
class Prediction:
    """Projected telomere length at a future time-point.

    Attributes
    ----------
    year_offset : int
        Years ahead from the most recent measurement.
    projected_age : int
        Predicted chronological age at projection time.
    predicted_length_kb : float
        Central estimate of telomere length (kb).
    ci_lower_kb : float
        Lower bound of the 95 % prediction interval.
    ci_upper_kb : float
        Upper bound.
    percentile_at_age : float
        Expected population percentile at the projected age.
    """

    year_offset: int
    projected_age: int
    predicted_length_kb: float
    ci_lower_kb: float
    ci_upper_kb: float
    percentile_at_age: float


@dataclass
class PopulationComparison:
    """Age-adjusted population percentile ranking at each measurement.

    Reference data from Müezzinler *et al.* (2013) meta-analysis.

    Attributes
    ----------
    patient_id : str
    percentiles : list[dict]
        Each entry: ``{"age": int, "length_kb": float, "percentile": float,
        "z_score": float, "timestamp": str}``.
    mean_percentile : float
        Average percentile across all measurements.
    percentile_trend : str
        ``"stable"``, ``"declining"``, or ``"improving"``.
    trajectory_vs_expected : str
        ``"above_average"``, ``"average"``, or ``"below_average"``.
    reference : str
        Literature reference.
    """

    patient_id: str
    percentiles: list[dict]
    mean_percentile: float
    percentile_trend: str
    trajectory_vs_expected: str
    reference: str = "Müezzinler et al. 2013, Ageing Res Rev 12:509"


@dataclass
class Anomaly:
    """A detected anomalous change in a patient's telomere trajectory.

    Attributes
    ----------
    patient_id : str
    timestamp : str
        When the anomalous measurement was taken.
    measurement_index : int
        Position in the chronological measurement list.
    observed_length_kb : float
    expected_length_kb : float
        Value predicted by the linear trend at this time-point.
    deviation_kb : float
        Signed difference (observed - expected).
    z_score : float
        Deviation expressed in standard-error units.
    severity : str
        ``"warning"`` (|z| >= 2) or ``"critical"`` (|z| >= 3).
    direction : str
        ``"shortening"`` or ``"lengthening"`` (relative to trend).
    possible_causes : list[str]
        Candidate explanations for the anomaly.
    """

    patient_id: str
    timestamp: str
    measurement_index: int
    observed_length_kb: float
    expected_length_kb: float
    deviation_kb: float
    z_score: float
    severity: str
    direction: str
    possible_causes: list[str]


@dataclass
class TrendReport:
    """Comprehensive longitudinal trend report for a patient.

    Aggregates attrition, population comparison, anomalies, predictions,
    and actionable recommendations into a single document.

    Attributes
    ----------
    patient_id : str
    generated_at : str
        ISO-8601 UTC timestamp of report generation.
    history_summary : dict
        Basic stats — count, span, age range.
    attrition : AttritionAnalysis
    population : PopulationComparison
    anomalies : list[Anomaly]
    predictions : list[Prediction]
    visualization_data : dict
        Pre-computed series for plotting (timestamps, lengths, trend line,
        confidence bands, percentile bands).
    recommendations : list[str]
        Plain-language clinical/lifestyle recommendations based on the
        analysis.  Not medical advice — informational only.
    """

    patient_id: str
    generated_at: str
    history_summary: dict
    attrition: AttritionAnalysis
    population: PopulationComparison
    anomalies: list[Anomaly]
    predictions: list[Prediction]
    visualization_data: dict
    recommendations: list[str]


# ---------------------------------------------------------------------------
# TelomereTracker — main public API
# ---------------------------------------------------------------------------


class TelomereTracker:
    """Record, analyse, and project telomere-length measurements over time.

    Data is stored as one JSON file per patient under *storage_path*.

    Parameters
    ----------
    storage_path : str
        Directory where patient JSON files are persisted.
        Created automatically if it does not exist.

    Examples
    --------
    >>> tracker = TelomereTracker()
    >>> m = tracker.record_measurement("P001", 8.5, 45, 42, "qPCR")
    >>> history = tracker.get_history("P001")
    >>> attrition = tracker.calculate_attrition_rate("P001")
    """

    def __init__(self, storage_path: str = "/tmp/teloscopy_tracking") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    # -- persistence helpers ------------------------------------------------

    def _patient_file(self, patient_id: str) -> Path:
        """Return the JSON file path for a patient."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in patient_id)
        return self.storage_path / f"{safe_id}.json"

    def _load_records(self, patient_id: str) -> list[dict]:
        """Load raw measurement dicts from the patient file."""
        path = self._patient_file(patient_id)
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("measurements", [])

    def _save_records(self, patient_id: str, records: list[dict]) -> None:
        """Persist measurement dicts for a patient."""
        path = self._patient_file(patient_id)
        payload = {
            "patient_id": patient_id,
            "updated_at": datetime.now(UTC).isoformat(),
            "measurements": records,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    @staticmethod
    def _dict_to_measurement(d: dict) -> Measurement:
        """Reconstitute a *Measurement* from a plain dict."""
        return Measurement(
            measurement_id=d["measurement_id"],
            patient_id=d["patient_id"],
            timestamp=d["timestamp"],
            telomere_length_kb=float(d["telomere_length_kb"]),
            biological_age=int(d["biological_age"]),
            chronological_age=int(d["chronological_age"]),
            method=d["method"],
            confidence=float(d.get("confidence", 1.0)),
            metadata=d.get("metadata", {}),
        )

    # -- public API ---------------------------------------------------------

    def record_measurement(
        self,
        patient_id: str,
        telomere_length_kb: float,
        biological_age: int,
        chronological_age: int,
        method: str,
        metadata: dict | None = None,
    ) -> Measurement:
        """Record a new telomere-length measurement.

        Parameters
        ----------
        patient_id : str
            Unique patient identifier.
        telomere_length_kb : float
            Mean telomere restriction-fragment length in kilobases.
        biological_age : int
            Estimated biological age.
        chronological_age : int
            Actual chronological age.
        method : str
            Assay method (``"TRF"``, ``"qPCR"``, ``"FlowFISH"``, etc.).
        metadata : dict, optional
            Extra key-value pairs (tissue, lab, batch …).

        Returns
        -------
        Measurement
            The newly created measurement record.
        """
        if telomere_length_kb <= 0:
            raise ValueError("telomere_length_kb must be positive")
        if chronological_age < 0:
            raise ValueError("chronological_age must be non-negative")

        meas = Measurement(
            measurement_id=str(uuid.uuid4()),
            patient_id=patient_id,
            timestamp=datetime.now(UTC).isoformat(),
            telomere_length_kb=telomere_length_kb,
            biological_age=biological_age,
            chronological_age=chronological_age,
            method=method,
            confidence=1.0,
            metadata=metadata or {},
        )
        records = self._load_records(patient_id)
        records.append(asdict(meas))
        # Keep chronological order
        records.sort(key=lambda r: r["timestamp"])
        self._save_records(patient_id, records)
        return meas

    def get_history(self, patient_id: str) -> PatientHistory:
        """Retrieve the full measurement history for a patient.

        Returns
        -------
        PatientHistory
            Contains all measurements in chronological order along with
            summary statistics.

        Raises
        ------
        FileNotFoundError
            If no data exists for *patient_id*.
        """
        records = self._load_records(patient_id)
        if not records:
            raise FileNotFoundError(f"No measurements found for patient '{patient_id}'")
        measurements = [self._dict_to_measurement(r) for r in records]
        ages = [m.chronological_age for m in measurements]
        methods = sorted({m.method for m in measurements})

        # Compute span from timestamps
        ts = [datetime.fromisoformat(m.timestamp) for m in measurements]
        span_seconds = (max(ts) - min(ts)).total_seconds()
        span_years = span_seconds / (365.25 * 86400)

        return PatientHistory(
            patient_id=patient_id,
            measurements=measurements,
            age_range=(min(ages), max(ages)),
            span_years=round(span_years, 2),
            measurement_count=len(measurements),
            methods_used=methods,
        )

    def calculate_attrition_rate(self, patient_id: str) -> AttritionAnalysis:
        """Calculate telomere attrition rate via linear regression.

        The slope of telomere length (kb) vs. chronological age (years)
        is converted to base-pairs per year.  Bootstrap resampling
        provides a 95 % confidence interval.

        Returns
        -------
        AttritionAnalysis

        Raises
        ------
        ValueError
            If fewer than two measurements exist.
        """
        history = self.get_history(patient_id)
        if history.measurement_count < 2:
            raise ValueError("At least two measurements are required to compute attrition rate")

        ages = [float(m.chronological_age) for m in history.measurements]
        lengths = [m.telomere_length_kb for m in history.measurements]

        slope, intercept = _linear_regression(ages, lengths)
        rse = _residual_std_error(ages, lengths, slope, intercept)

        # R-squared
        y_bar = statistics.mean(lengths)
        ss_tot = sum((y - y_bar) ** 2 for y in lengths)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(ages, lengths))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Bootstrap CI on slope
        mean_slope, ci_lo, ci_hi = _bootstrap_slope(ages, lengths)

        # Convert kb/year -> bp/year (1 kb = 1000 bp), sign convention:
        # positive rate = shortening (slope is negative if shortening)
        rate_bp = -slope * 1000.0
        ci_lower_bp = -ci_hi * 1000.0  # note inversion
        ci_upper_bp = -ci_lo * 1000.0

        # Qualitative comparison to population norms
        if rate_bp < _NORMAL_ATTRITION_LOW_BP:
            comparison = "decelerated"
        elif rate_bp > _NORMAL_ATTRITION_HIGH_BP:
            comparison = "accelerated"
        elif rate_bp < 0:
            comparison = "elongating"
        else:
            comparison = "normal"

        return AttritionAnalysis(
            patient_id=patient_id,
            rate_bp_per_year=round(rate_bp, 2),
            ci_lower_bp=round(ci_lower_bp, 2),
            ci_upper_bp=round(ci_upper_bp, 2),
            r_squared=round(r_sq, 4),
            residual_se_kb=round(rse, 4),
            population_comparison=comparison,
            n_measurements=history.measurement_count,
        )

    def predict_future_length(
        self,
        patient_id: str,
        years_ahead: int = 10,
    ) -> list[Prediction]:
        """Project future telomere length from the current trend.

        Uses the linear regression model and adds a prediction interval
        derived from the residual standard error and bootstrap CI on
        the slope.

        Parameters
        ----------
        patient_id : str
        years_ahead : int
            Number of years to project (default 10).

        Returns
        -------
        list[Prediction]
            One entry per future year.
        """
        history = self.get_history(patient_id)
        if history.measurement_count < 2:
            raise ValueError("At least two measurements required for prediction")

        ages = [float(m.chronological_age) for m in history.measurements]
        lengths = [m.telomere_length_kb for m in history.measurements]

        slope, intercept = _linear_regression(ages, lengths)
        rse = _residual_std_error(ages, lengths, slope, intercept)
        _, ci_lo_slope, ci_hi_slope = _bootstrap_slope(ages, lengths)

        last_age = max(ages)
        predictions: list[Prediction] = []

        for yr in range(1, years_ahead + 1):
            future_age = int(last_age + yr)
            pred_kb = slope * future_age + intercept

            # Prediction interval widens with extrapolation distance
            x_bar = statistics.mean(ages)
            ss_xx = sum((x - x_bar) ** 2 for x in ages)
            n = len(ages)
            h = 1.0 / n + (future_age - x_bar) ** 2 / ss_xx if ss_xx > 0 else 1.0
            margin = 1.96 * rse * math.sqrt(1 + h)

            ci_lo = max(0.0, pred_kb - margin)
            ci_hi = pred_kb + margin

            # Population percentile at projected age
            pop_mean, pop_sd = _population_stats_for_age(future_age)
            z = _z_score(pred_kb, pop_mean, pop_sd)
            pct = _percentile_from_z(z)

            predictions.append(
                Prediction(
                    year_offset=yr,
                    projected_age=future_age,
                    predicted_length_kb=round(pred_kb, 3),
                    ci_lower_kb=round(ci_lo, 3),
                    ci_upper_kb=round(ci_hi, 3),
                    percentile_at_age=round(pct, 1),
                )
            )

        return predictions

    def compare_to_population(
        self,
        patient_id: str,
    ) -> PopulationComparison:
        """Compute age-adjusted population percentile at each measurement.

        Reference ranges are from Müezzinler *et al.* (2013).

        Returns
        -------
        PopulationComparison
        """
        history = self.get_history(patient_id)
        percentiles: list[dict] = []

        for m in history.measurements:
            pop_mean, pop_sd = _population_stats_for_age(m.chronological_age)
            z = _z_score(m.telomere_length_kb, pop_mean, pop_sd)
            pct = _percentile_from_z(z)
            percentiles.append(
                {
                    "age": m.chronological_age,
                    "length_kb": m.telomere_length_kb,
                    "percentile": round(pct, 1),
                    "z_score": round(z, 3),
                    "timestamp": m.timestamp,
                }
            )

        pcts = [p["percentile"] for p in percentiles]
        mean_pct = statistics.mean(pcts) if pcts else 50.0

        # Trend: compare first-half mean to second-half mean
        if len(pcts) >= 4:
            mid = len(pcts) // 2
            first_half = statistics.mean(pcts[:mid])
            second_half = statistics.mean(pcts[mid:])
            diff = second_half - first_half
            if diff < -5:
                trend = "declining"
            elif diff > 5:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"

        if mean_pct >= 60:
            trajectory = "above_average"
        elif mean_pct <= 40:
            trajectory = "below_average"
        else:
            trajectory = "average"

        return PopulationComparison(
            patient_id=patient_id,
            percentiles=percentiles,
            mean_percentile=round(mean_pct, 1),
            percentile_trend=trend,
            trajectory_vs_expected=trajectory,
        )

    def detect_anomalies(self, patient_id: str) -> list[Anomaly]:
        """Detect anomalous changes in the telomere trajectory.

        Two complementary approaches are used:

        1. **Residual screening** — points whose residual from the linear
           trend exceeds 2 SE (warning) or 3 SE (critical).
        2. **Change-point detection** — Bayesian scoring of each interior
           point to identify abrupt rate changes.

        Returns
        -------
        list[Anomaly]
            Anomalies sorted by severity (critical first).
        """
        history = self.get_history(patient_id)
        if history.measurement_count < 3:
            return []

        ages = [float(m.chronological_age) for m in history.measurements]
        lengths = [m.telomere_length_kb for m in history.measurements]

        slope, intercept = _linear_regression(ages, lengths)
        rse = _residual_std_error(ages, lengths, slope, intercept)

        anomalies: list[Anomaly] = []

        # --- 1. Residual-based anomalies ---
        if rse > 0:
            for idx, m in enumerate(history.measurements):
                expected = slope * m.chronological_age + intercept
                deviation = m.telomere_length_kb - expected
                z = deviation / rse

                if abs(z) >= _ANOMALY_ZSCORE_WARN:
                    severity = "critical" if abs(z) >= _ANOMALY_ZSCORE_CRITICAL else "warning"
                    direction = "lengthening" if deviation > 0 else "shortening"
                    causes = self._suggest_anomaly_causes(direction, severity, m)
                    anomalies.append(
                        Anomaly(
                            patient_id=patient_id,
                            timestamp=m.timestamp,
                            measurement_index=idx,
                            observed_length_kb=round(m.telomere_length_kb, 3),
                            expected_length_kb=round(expected, 3),
                            deviation_kb=round(deviation, 3),
                            z_score=round(z, 2),
                            severity=severity,
                            direction=direction,
                            possible_causes=causes,
                        )
                    )

        # --- 2. Change-point detection ---
        cp_scores = _bayesian_changepoint_scores(lengths)
        if cp_scores:
            max_score = max(cp_scores)
            threshold = max(
                2.0,
                statistics.mean(cp_scores)
                + 2.0 * (statistics.stdev(cp_scores) if len(cp_scores) > 1 else 0),
            )
            for i, score in enumerate(cp_scores):
                actual_idx = i + 1
                if score >= threshold:
                    m = history.measurements[actual_idx]
                    expected = slope * m.chronological_age + intercept
                    deviation = m.telomere_length_kb - expected
                    z_equiv = score / max_score * 3.0 if max_score > 0 else 0
                    # Avoid duplicating residual-based anomalies
                    already_flagged = any(a.measurement_index == actual_idx for a in anomalies)
                    if not already_flagged:
                        direction = "lengthening" if deviation > 0 else "shortening"
                        anomalies.append(
                            Anomaly(
                                patient_id=patient_id,
                                timestamp=m.timestamp,
                                measurement_index=actual_idx,
                                observed_length_kb=round(m.telomere_length_kb, 3),
                                expected_length_kb=round(expected, 3),
                                deviation_kb=round(deviation, 3),
                                z_score=round(z_equiv, 2),
                                severity="warning",
                                direction=direction,
                                possible_causes=[
                                    "Rate change-point detected by Bayesian model",
                                    "Consider splitting analysis into sub-periods",
                                ],
                            )
                        )

        # Sort: critical first, then by absolute z-score descending
        anomalies.sort(key=lambda a: (0 if a.severity == "critical" else 1, -abs(a.z_score)))
        return anomalies

    @staticmethod
    def _suggest_anomaly_causes(
        direction: str, severity: str, measurement: Measurement
    ) -> list[str]:
        """Generate candidate explanations for an anomalous reading."""
        causes: list[str] = []
        if direction == "shortening":
            causes.append("Acute oxidative stress or inflammation")
            causes.append("Possible assay artifact or sample degradation")
            if severity == "critical":
                causes.append("Chemotherapy or radiation exposure")
                causes.append("Severe physiological stress event")
        else:
            causes.append("Telomerase activation (transient or sustained)")
            causes.append("Assay batch effect or methodological change")
            if severity == "critical":
                causes.append("Sample labelling error — verify patient ID")
                causes.append("Alternative lengthening of telomeres (ALT)")
        # Method-specific caveats
        method_lower = measurement.method.lower()
        if method_lower == "qpcr":
            causes.append("qPCR T/S ratio variability between runs")
        elif method_lower == "flowfish":
            causes.append("FlowFISH gating artefact")
        return causes

    def generate_trend_report(self, patient_id: str) -> TrendReport:
        """Generate a comprehensive longitudinal trend report.

        Aggregates history, attrition, population comparison, anomalies,
        10-year predictions, and plain-language recommendations.

        Returns
        -------
        TrendReport
        """
        history = self.get_history(patient_id)
        attrition = self.calculate_attrition_rate(patient_id)
        population = self.compare_to_population(patient_id)
        anomalies = self.detect_anomalies(patient_id)
        predictions = self.predict_future_length(patient_id, years_ahead=10)

        # Build visualisation-ready data series
        viz: dict = {
            "timestamps": [m.timestamp for m in history.measurements],
            "ages": [m.chronological_age for m in history.measurements],
            "lengths_kb": [m.telomere_length_kb for m in history.measurements],
            "trend_line": [],
            "ci_upper": [],
            "ci_lower": [],
            "population_mean": [],
            "population_p10": [],
            "population_p90": [],
        }
        ages_f = [float(a) for a in viz["ages"]]
        slope, intercept = _linear_regression(ages_f, viz["lengths_kb"])
        rse = _residual_std_error(ages_f, viz["lengths_kb"], slope, intercept)
        for age in viz["ages"]:
            trend_val = slope * age + intercept
            viz["trend_line"].append(round(trend_val, 3))
            viz["ci_upper"].append(round(trend_val + 1.96 * rse, 3))
            viz["ci_lower"].append(round(max(0, trend_val - 1.96 * rse), 3))
            pop_mean, pop_sd = _population_stats_for_age(age)
            viz["population_mean"].append(round(pop_mean, 3))
            viz["population_p10"].append(round(pop_mean - 1.282 * pop_sd, 3))
            viz["population_p90"].append(round(pop_mean + 1.282 * pop_sd, 3))

        # Recommendations
        recommendations = self._build_recommendations(attrition, population, anomalies, history)

        history_summary = {
            "measurement_count": history.measurement_count,
            "span_years": history.span_years,
            "age_range": list(history.age_range),
            "methods_used": history.methods_used,
            "first_length_kb": history.measurements[0].telomere_length_kb,
            "last_length_kb": history.measurements[-1].telomere_length_kb,
        }

        return TrendReport(
            patient_id=patient_id,
            generated_at=datetime.now(UTC).isoformat(),
            history_summary=history_summary,
            attrition=attrition,
            population=population,
            anomalies=anomalies,
            predictions=predictions,
            visualization_data=viz,
            recommendations=recommendations,
        )

    @staticmethod
    def _build_recommendations(
        attrition: AttritionAnalysis,
        population: PopulationComparison,
        anomalies: list[Anomaly],
        history: PatientHistory,
    ) -> list[str]:
        """Generate plain-language recommendations.

        NOTE: These are informational observations, not medical advice.
        """
        recs: list[str] = []

        # Attrition-based
        if attrition.population_comparison == "accelerated":
            recs.append(
                "Telomere attrition rate is above the population norm "
                f"({attrition.rate_bp_per_year:.0f} bp/yr vs. typical "
                f"{_NORMAL_ATTRITION_LOW_BP:.0f}–{_NORMAL_ATTRITION_HIGH_BP:.0f} bp/yr). "
                "Consider evaluating modifiable risk factors: chronic stress, "
                "smoking, sedentary lifestyle, poor sleep quality."
            )
        elif attrition.population_comparison == "decelerated":
            recs.append(
                "Telomere attrition is slower than the population average — "
                "consistent with healthy ageing trajectory."
            )
        elif attrition.population_comparison == "elongating":
            recs.append(
                "Telomere length appears to be *increasing* over the "
                "observation period.  While possible in some contexts "
                "(e.g. lifestyle interventions, telomerase activators), "
                "this may also indicate measurement variability.  Confirm "
                "with repeat assays."
            )

        # Population-percentile–based
        if population.trajectory_vs_expected == "below_average":
            recs.append(
                "Overall telomere length is below the age-adjusted "
                "population average.  This is associated with increased "
                "risk of age-related diseases (Müezzinler et al. 2013).  "
                "Periodic monitoring is recommended."
            )
        elif population.trajectory_vs_expected == "above_average":
            recs.append(
                "Telomere length is above the population average for age — "
                "a favourable biomarker profile."
            )

        if population.percentile_trend == "declining":
            recs.append(
                "Population percentile is trending downward over time, "
                "suggesting accelerating relative shortening.  "
                "Follow-up measurements at 6–12 month intervals are advised."
            )

        # Anomaly-based
        critical_count = sum(1 for a in anomalies if a.severity == "critical")
        if critical_count > 0:
            recs.append(
                f"{critical_count} critical anomaly/anomalies detected.  "
                "Review assay quality control and patient medical history "
                "for potential confounders (chemotherapy, acute illness, "
                "sample handling issues)."
            )
        elif len(anomalies) > 0:
            recs.append(
                f"{len(anomalies)} mild anomaly/anomalies detected.  "
                "These may reflect normal biological variation or assay "
                "noise; continue routine monitoring."
            )

        # Measurement frequency
        if history.measurement_count < 4:
            recs.append(
                "Fewer than 4 measurements on record — statistical power "
                "is limited.  Additional time-points will improve trend "
                "estimates and prediction accuracy."
            )

        # Multi-method caveat
        if len(history.methods_used) > 1:
            recs.append(
                f"Multiple assay methods used ({', '.join(history.methods_used)}).  "
                "Inter-method variability can confound longitudinal "
                "comparisons.  Where possible, use a single method for "
                "serial measurements (Aubert & Lansdorp 2008)."
            )

        if not recs:
            recs.append(
                "Telomere dynamics are within normal ranges.  Continue "
                "periodic monitoring per standard protocol."
            )

        return recs

    # -- CSV I/O ------------------------------------------------------------

    def export_csv(self, patient_id: str, path: str) -> None:
        """Export a patient's measurement history to a CSV file.

        Columns: measurement_id, timestamp, telomere_length_kb,
        biological_age, chronological_age, method, confidence, metadata.

        Parameters
        ----------
        patient_id : str
        path : str
            Destination file path.
        """
        history = self.get_history(patient_id)
        fieldnames = [
            "measurement_id",
            "patient_id",
            "timestamp",
            "telomere_length_kb",
            "biological_age",
            "chronological_age",
            "method",
            "confidence",
            "metadata",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for m in history.measurements:
                row = asdict(m)
                row["metadata"] = json.dumps(row["metadata"])
                writer.writerow(row)

    def import_csv(self, path: str) -> str:
        """Import measurements from a CSV file.

        The CSV must contain at least the columns: ``patient_id``,
        ``telomere_length_kb``, ``biological_age``, ``chronological_age``,
        ``method``.  Optional columns: ``timestamp``, ``confidence``,
        ``metadata``, ``measurement_id``.

        Parameters
        ----------
        path : str
            Source CSV file path.

        Returns
        -------
        str
            The patient_id of the imported measurements.
        """
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            raise ValueError("CSV file is empty")

        patient_id: str | None = None
        records = self._load_records(rows[0].get("patient_id", ""))

        for row in rows:
            pid = row.get("patient_id")
            if pid is None:
                raise ValueError("CSV must contain a 'patient_id' column")
            if patient_id is None:
                patient_id = pid
                records = self._load_records(patient_id)
            elif pid != patient_id:
                raise ValueError(
                    f"CSV contains multiple patient IDs: {patient_id}, {pid}.  "
                    "Only single-patient imports are supported."
                )

            meta_raw = row.get("metadata", "{}")
            try:
                metadata = json.loads(meta_raw) if meta_raw else {}
            except json.JSONDecodeError:
                metadata = {"raw": meta_raw}

            meas = Measurement(
                measurement_id=row.get("measurement_id", str(uuid.uuid4())),
                patient_id=patient_id,
                timestamp=row.get(
                    "timestamp",
                    datetime.now(UTC).isoformat(),
                ),
                telomere_length_kb=float(row["telomere_length_kb"]),
                biological_age=int(row["biological_age"]),
                chronological_age=int(row["chronological_age"]),
                method=row["method"],
                confidence=float(row.get("confidence", 1.0)),
                metadata=metadata,
            )
            records.append(asdict(meas))

        records.sort(key=lambda r: r["timestamp"])
        assert patient_id is not None
        self._save_records(patient_id, records)
        return patient_id
