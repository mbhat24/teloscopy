"""
teloscopy.tracking — Longitudinal Tracking of Telomere Dynamics
===============================================================

This sub-package provides tools for recording, analysing, and projecting
telomere-length measurements over time.  Longitudinal tracking is essential
for translating single-point telomere assays into clinically actionable
insights: attrition rates, anomaly detection, population-percentile
trajectories, and personalised ageing forecasts.

Key capabilities
----------------
* **TelomereTracker** — the primary entry-point for recording measurements,
  querying patient histories, and running trend analyses.
* Statistical modelling of attrition via linear regression with bootstrap
  confidence intervals.
* Bayesian change-point detection for identifying abrupt shifts in
  shortening rate (e.g. post-chemotherapy, lifestyle interventions).
* Age-adjusted population-percentile ranking based on published reference
  data (Müezzinler *et al.* 2013; Hastie *et al.* 1990; Aubert & Lansdorp
  2008).

Storage is deliberately simple — one JSON file per patient — so that
datasets remain human-readable and easily portable between systems.
"""

from teloscopy.tracking.longitudinal import (  # noqa: F401
    Anomaly,
    AttritionAnalysis,
    Measurement,
    PatientHistory,
    PopulationComparison,
    Prediction,
    TelomereTracker,
    TrendReport,
)

__all__ = [
    "TelomereTracker",
    "Measurement",
    "PatientHistory",
    "AttritionAnalysis",
    "Prediction",
    "PopulationComparison",
    "Anomaly",
    "TrendReport",
]
