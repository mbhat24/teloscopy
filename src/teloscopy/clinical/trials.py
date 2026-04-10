"""
Multi-institution clinical trial coordination module.

Provides end-to-end management of multi-site clinical trials for validating
the Teloscopy telomere length analysis platform across institutions.  Handles
site registration, participant enrollment, data collection, quality control,
interim and final statistical analyses, adverse-event tracking, and regulatory
reporting required by Data Safety Monitoring Boards (DSMBs).

This module implements trial-management methodologies in accordance with:
    - ICH E6(R2): Guideline for Good Clinical Practice (2016)
    - ICH E9: Statistical Principles for Clinical Trials (1998)
    - ICH E9(R1): Estimands and Sensitivity Analysis in Clinical Trials (2019)
    - FDA Guidance: Adaptive Designs for Clinical Trials of Drugs and
      Biologics (2019)
    - 21 CFR Part 11: Electronic Records; Electronic Signatures
    - CONSORT 2010: Consolidated Standards of Reporting Trials

References:
    [1] Lan, K.K.G. and DeMets, D.L., "Discrete Sequential Boundaries for
        Clinical Trials," *Biometrika*, 70(3):659-663, 1983.
    [2] O'Brien, P.C. and Fleming, T.R., "A Multiple Testing Procedure for
        Clinical Trials," *Biometrics*, 35(3):549-556, 1979.
    [3] Pocock, S.J., "Group Sequential Methods in the Design and Analysis
        of Clinical Trials," *Biometrika*, 64(2):191-199, 1977.

Example:
    >>> protocol = TrialProtocol(
    ...     protocol_id="TELO-2024-001",
    ...     title="Multi-site Validation of qFISH Telomere Length Analysis",
    ...     phase=TrialPhase.PHASE_2,
    ...     primary_endpoint="telomere_length_correlation",
    ...     secondary_endpoints=["inter_site_reproducibility"],
    ...     inclusion_criteria=["age >= 18", "healthy_volunteer"],
    ...     exclusion_criteria=["active_malignancy"],
    ...     target_sample_size=500,
    ...     duration_months=24,
    ...     version="1.0",
    ... )
    >>> coordinator = ClinicalTrialCoordinator("TRIAL-001", protocol)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SiteStatus(Enum):
    """Operational status of an institutional trial site."""

    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class TrialPhase(Enum):
    """Phase of the clinical trial.

    Reference:
        FDA Guidance — IND Applications for Clinical Investigations (2003).
    """

    PILOT = "pilot"
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"


class ParticipantStatus(Enum):
    """Lifecycle status of a trial participant."""

    SCREENING = "screening"
    ENROLLED = "enrolled"
    ACTIVE = "active"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"
    SCREEN_FAILED = "screen_failed"


class DataQualityFlag(Enum):
    """Outcome of a data-quality validation check."""

    PASSED = "passed"
    FLAGGED = "flagged"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TELOMERE_LENGTH_MIN_BP: float = 3_000.0
_TELOMERE_LENGTH_MAX_BP: float = 20_000.0
_VALID_MEASUREMENT_METHODS: set[str] = {"qfish", "trf", "qpcr", "flowfish"}
_VALID_SEVERITIES: set[str] = {"mild", "moderate", "severe"}
_VALID_RELATEDNESS: set[str] = {
    "unrelated",
    "possibly",
    "probably",
    "definitely",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ProtocolAmendment:
    """Record of a single amendment to a trial protocol.

    Attributes:
        amendment_id: Unique identifier for this amendment.
        version: New protocol version after the amendment.
        date: Date the amendment was approved.
        description: Human-readable summary of changes.
        approved_by: Name or role of the approving authority.
    """

    amendment_id: str
    version: str
    date: date
    description: str
    approved_by: str


@dataclass
class InstitutionSite:
    """Institutional site participating in the clinical trial.

    Attributes:
        site_id: Unique site identifier.
        name: Institution name (e.g. "Mayo Clinic").
        location: City and state / region (e.g. "Rochester, MN").
        principal_investigator: Name of the PI at this site.
        irb_approval_number: IRB protocol approval reference number.
        irb_approval_date: Date IRB approval was granted.
        irb_expiry_date: Date IRB approval expires.
        status: Current operational status of the site.
        enrolled_count: Number of participants enrolled so far.
        target_enrollment: Site-level enrollment target.
        contact_email: Primary contact e-mail for the site.
        data_sharing_agreement: Whether a DSA has been executed.
    """

    site_id: str
    name: str
    location: str
    principal_investigator: str
    irb_approval_number: str
    irb_approval_date: date
    irb_expiry_date: date
    status: SiteStatus
    enrolled_count: int
    target_enrollment: int
    contact_email: str
    data_sharing_agreement: bool


@dataclass
class TrialProtocol:
    """Versioned clinical-trial protocol definition.

    Attributes:
        protocol_id: Unique protocol identifier.
        title: Full title of the trial.
        phase: Trial phase classification.
        primary_endpoint: Primary efficacy endpoint.
        secondary_endpoints: List of secondary endpoints.
        inclusion_criteria: Eligibility inclusion criteria.
        exclusion_criteria: Eligibility exclusion criteria.
        target_sample_size: Total enrollment target across all sites.
        duration_months: Planned trial duration in months.
        version: Current protocol version string.
        amendments: Ordered list of protocol amendments.
    """

    protocol_id: str
    title: str
    phase: TrialPhase
    primary_endpoint: str
    secondary_endpoints: list[str]
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    target_sample_size: int
    duration_months: int
    version: str
    amendments: list[ProtocolAmendment] = field(default_factory=list)


@dataclass
class ParticipantEnrollment:
    """Enrollment record for an individual trial participant.

    Attributes:
        participant_id: Anonymised participant identifier.
        site_id: Site at which the participant is enrolled.
        enrollment_date: Date of enrollment.
        consent_version: Version of the informed-consent document signed.
        demographics: Anonymised demographic data (age, sex, ethnicity).
        status: Current participant lifecycle status.
        withdrawal_reason: Reason for withdrawal, if applicable.
        visits_completed: Number of study visits completed.
        next_visit_date: Scheduled date of the next visit.
    """

    participant_id: str
    site_id: str
    enrollment_date: date
    consent_version: str
    demographics: dict[str, Any]
    status: ParticipantStatus
    withdrawal_reason: str | None = None
    visits_completed: int = 0
    next_visit_date: date | None = None


@dataclass
class TrialDataPoint:
    """Single telomere-length measurement collected during the trial.

    Attributes:
        participant_id: Anonymised participant identifier.
        site_id: Collecting site identifier.
        visit_number: Ordinal visit number (1-based).
        collection_date: Date the sample was collected.
        telomere_length_bp: Measured telomere length in base pairs.
        measurement_method: Method used (qfish, trf, qpcr, flowfish).
        quality_score: Analyst-assigned quality score in [0, 1].
        metadata: Additional metadata (instrument, operator, batch, etc.).
    """

    participant_id: str
    site_id: str
    visit_number: int
    collection_date: date
    telomere_length_bp: float
    measurement_method: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdverseEvent:
    """Record of an adverse event observed during the trial.

    Attributes:
        event_id: Unique adverse-event identifier.
        participant_id: Anonymised participant identifier.
        site_id: Site that reported the event.
        description: Free-text description of the event.
        severity: Severity grade (mild, moderate, severe).
        relatedness: Relationship to the study intervention.
        date_reported: Date the event was reported.
        date_resolved: Date the event was resolved, if applicable.
        resolution: Textual description of how the event was resolved.
    """

    event_id: str
    participant_id: str
    site_id: str
    description: str
    severity: str
    relatedness: str
    date_reported: date
    date_resolved: date | None = None
    resolution: str = ""


@dataclass
class TrialResult:
    """Aggregated result of an interim or final trial analysis.

    Attributes:
        trial_id: Trial identifier.
        analysis_date: Timestamp of the analysis.
        sites_included: Number of sites contributing data.
        total_participants: Number of participants included.
        primary_endpoint_met: Whether the primary endpoint was achieved.
        effect_size: Estimated effect size (Cohen's d or equivalent).
        p_value: Two-sided p-value for the primary endpoint test.
        confidence_interval: 95 % confidence interval for the effect size.
        per_site_results: Per-site summary statistics.
        adverse_events: List of adverse events recorded to date.
    """

    trial_id: str
    analysis_date: datetime
    sites_included: int
    total_participants: int
    primary_endpoint_met: bool
    effect_size: float
    p_value: float
    confidence_interval: tuple[float, float]
    per_site_results: dict[str, dict[str, Any]]
    adverse_events: list[AdverseEvent]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _generate_id(prefix: str = "") -> str:
    """Return a new UUID-based identifier with an optional prefix."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


def _now_utc() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _normal_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal.

    Uses the Abramowitz & Stegun rational approximation (formula 26.2.17)
    which provides |error| < 7.5 x 10^-8.

    Parameters
    ----------
    x : float
        Quantile value.

    Returns
    -------
    float
        P(Z <= x) where Z ~ N(0, 1).
    """
    # Coefficients for the approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0
    if x < 0:
        sign = -1.0
    x = abs(x) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def _normal_ppf(p: float) -> float:
    """Percent-point (inverse CDF) function for the standard normal.

    Uses the Beasley-Springer-Moro rational approximation.

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        z such that P(Z <= z) = p.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Rational approximation coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _two_sided_p_value(z: float) -> float:
    """Return the two-sided p-value for a standard-normal z-statistic."""
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


# ---------------------------------------------------------------------------
# ClinicalTrialCoordinator
# ---------------------------------------------------------------------------


class ClinicalTrialCoordinator:
    """Coordinates a multi-site clinical trial for telomere-length validation.

    Manages the full lifecycle of a trial: site registration, participant
    enrollment, data collection and quality control, interim and final
    statistical analyses, adverse-event tracking, randomisation, and
    regulatory reporting.

    Parameters
    ----------
    trial_id : str
        Unique trial identifier.
    protocol : TrialProtocol
        Protocol definition governing the trial.

    Attributes
    ----------
    trial_id : str
        Unique trial identifier.
    protocol : TrialProtocol
        Active protocol definition.
    """

    def __init__(self, trial_id: str, protocol: TrialProtocol) -> None:
        self.trial_id = trial_id
        self.protocol = protocol

        self._sites: dict[str, InstitutionSite] = {}
        self._participants: dict[str, ParticipantEnrollment] = {}
        self._data_points: list[TrialDataPoint] = []
        self._adverse_events: list[AdverseEvent] = []
        self._interim_count: int = 0
        self._max_interims: int = 4  # default for Lan-DeMets spending

        logger.info(
            "ClinicalTrialCoordinator initialised: trial=%s protocol=%s phase=%s",
            trial_id,
            protocol.protocol_id,
            protocol.phase.value,
        )

    # ----- site management -----------------------------------------------

    def register_site(self, site: InstitutionSite) -> str:
        """Register an institutional site for the trial.

        Parameters
        ----------
        site : InstitutionSite
            Site information to register.

        Returns
        -------
        str
            The ``site_id`` of the registered site.

        Raises
        ------
        ValueError
            If a site with the same ``site_id`` is already registered or if
            required fields are missing.
        """
        if site.site_id in self._sites:
            raise ValueError(f"Site already registered: {site.site_id}")
        if not site.data_sharing_agreement:
            raise ValueError(
                f"Data sharing agreement required for site: {site.site_id}"
            )
        self._sites[site.site_id] = site
        logger.info(
            "Site registered: site_id=%s name=%s pi=%s",
            site.site_id,
            site.name,
            site.principal_investigator,
        )
        return site.site_id

    def update_site_status(self, site_id: str, status: SiteStatus) -> None:
        """Update the operational status of a registered site.

        Parameters
        ----------
        site_id : str
            Identifier of the site to update.
        status : SiteStatus
            New status to assign.

        Raises
        ------
        KeyError
            If the site is not registered.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site not found: {site_id}")
        old_status = self._sites[site_id].status
        self._sites[site_id].status = status
        logger.info(
            "Site status updated: site_id=%s old=%s new=%s",
            site_id,
            old_status.value,
            status.value,
        )

    def get_site_summary(self) -> list[dict[str, Any]]:
        """Return a summary of all registered sites.

        Returns
        -------
        list[dict[str, Any]]
            One dictionary per site with key operational metrics.
        """
        summaries: list[dict[str, Any]] = []
        for site in self._sites.values():
            summaries.append({
                "site_id": site.site_id,
                "name": site.name,
                "location": site.location,
                "status": site.status.value,
                "enrolled_count": site.enrolled_count,
                "target_enrollment": site.target_enrollment,
                "enrollment_pct": round(
                    site.enrolled_count / max(site.target_enrollment, 1) * 100.0, 1
                ),
                "principal_investigator": site.principal_investigator,
                "irb_valid": site.irb_expiry_date >= date.today(),
            })
        return summaries

    def verify_irb_status(self, site_id: str) -> dict[str, Any]:
        """Verify the IRB approval status for a site.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        dict[str, Any]
            Dictionary containing ``is_valid``, ``approval_number``,
            ``approval_date``, ``expiry_date``, and ``days_remaining``.

        Raises
        ------
        KeyError
            If the site is not registered.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site not found: {site_id}")
        site = self._sites[site_id]
        today = date.today()
        days_remaining = (site.irb_expiry_date - today).days
        is_valid = days_remaining >= 0

        if not is_valid:
            logger.warning(
                "IRB approval expired for site %s: expired %d days ago",
                site_id,
                abs(days_remaining),
            )
        return {
            "site_id": site_id,
            "is_valid": is_valid,
            "approval_number": site.irb_approval_number,
            "approval_date": site.irb_approval_date.isoformat(),
            "expiry_date": site.irb_expiry_date.isoformat(),
            "days_remaining": days_remaining,
        }

    # ----- enrollment ----------------------------------------------------

    def enroll_participant(
        self,
        site_id: str,
        demographics: dict[str, Any],
        consent_version: str,
    ) -> str:
        """Enroll a new participant at a given site.

        Parameters
        ----------
        site_id : str
            Site at which the participant is enrolling.
        demographics : dict[str, Any]
            Anonymised demographic data (must include ``age`` and ``sex``).
        consent_version : str
            Version of the informed-consent document the participant signed.

        Returns
        -------
        str
            Generated anonymised participant identifier.

        Raises
        ------
        KeyError
            If the site is not registered.
        ValueError
            If the site is not active, IRB has expired, enrollment target is
            reached, or the participant fails eligibility checks.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site not found: {site_id}")
        site = self._sites[site_id]

        if site.status != SiteStatus.ACTIVE:
            raise ValueError(
                f"Cannot enroll at site {site_id}: status is {site.status.value}"
            )

        # Check IRB validity
        if site.irb_expiry_date < date.today():
            raise ValueError(
                f"IRB approval has expired for site {site_id}"
            )

        # Check enrollment cap
        if site.enrolled_count >= site.target_enrollment:
            raise ValueError(
                f"Site {site_id} has reached its enrollment target "
                f"({site.target_enrollment})"
            )

        # Eligibility
        eligible, reasons = self.check_eligibility(demographics)
        if not eligible:
            raise ValueError(
                f"Participant not eligible: {'; '.join(reasons)}"
            )

        participant_id = _generate_id(prefix="SUBJ-")
        enrollment = ParticipantEnrollment(
            participant_id=participant_id,
            site_id=site_id,
            enrollment_date=date.today(),
            consent_version=consent_version,
            demographics=demographics,
            status=ParticipantStatus.ENROLLED,
        )
        self._participants[participant_id] = enrollment
        site.enrolled_count += 1

        logger.info(
            "Participant enrolled: id=%s site=%s consent=%s",
            participant_id,
            site_id,
            consent_version,
        )
        return participant_id

    def withdraw_participant(self, participant_id: str, reason: str) -> None:
        """Withdraw a participant from the trial.

        Parameters
        ----------
        participant_id : str
            Identifier of the participant to withdraw.
        reason : str
            Free-text reason for withdrawal.

        Raises
        ------
        KeyError
            If the participant is not found.
        ValueError
            If the participant is already withdrawn or completed.
        """
        if participant_id not in self._participants:
            raise KeyError(f"Participant not found: {participant_id}")
        p = self._participants[participant_id]
        if p.status in (ParticipantStatus.WITHDRAWN, ParticipantStatus.COMPLETED):
            raise ValueError(
                f"Participant {participant_id} has status {p.status.value}; "
                f"cannot withdraw"
            )
        p.status = ParticipantStatus.WITHDRAWN
        p.withdrawal_reason = reason
        logger.info(
            "Participant withdrawn: id=%s reason=%s",
            participant_id,
            reason,
        )

    def get_enrollment_summary(self) -> dict[str, Any]:
        """Return a summary of enrollment status across the trial.

        Returns
        -------
        dict[str, Any]
            Contains ``total_enrolled``, ``total_target``, ``enrollment_pct``,
            ``per_site`` breakdown, and ``by_status`` counts.
        """
        total_enrolled = sum(
            1
            for p in self._participants.values()
            if p.status
            not in (ParticipantStatus.WITHDRAWN, ParticipantStatus.SCREEN_FAILED)
        )
        total_target = self.protocol.target_sample_size

        per_site: dict[str, dict[str, int]] = {}
        for site_id, site in self._sites.items():
            site_participants = [
                p for p in self._participants.values() if p.site_id == site_id
            ]
            per_site[site_id] = {
                "enrolled": len([
                    p
                    for p in site_participants
                    if p.status
                    not in (
                        ParticipantStatus.WITHDRAWN,
                        ParticipantStatus.SCREEN_FAILED,
                    )
                ]),
                "withdrawn": len([
                    p
                    for p in site_participants
                    if p.status == ParticipantStatus.WITHDRAWN
                ]),
                "target": site.target_enrollment,
            }

        by_status: dict[str, int] = {}
        for p in self._participants.values():
            by_status[p.status.value] = by_status.get(p.status.value, 0) + 1

        return {
            "trial_id": self.trial_id,
            "total_enrolled": total_enrolled,
            "total_target": total_target,
            "enrollment_pct": round(
                total_enrolled / max(total_target, 1) * 100.0, 1
            ),
            "per_site": per_site,
            "by_status": by_status,
        }

    def check_eligibility(
        self, demographics: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Check whether a prospective participant meets eligibility criteria.

        Evaluates inclusion and exclusion criteria defined in the protocol
        against the supplied demographics.

        Parameters
        ----------
        demographics : dict[str, Any]
            Anonymised demographic data.  Expected keys include ``age``
            (int) and ``sex`` (str).  Additional keys are evaluated against
            the protocol criteria.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_eligible, reasons)`` where *reasons* lists any criteria
            that were not met.
        """
        reasons: list[str] = []

        # Evaluate inclusion criteria
        for criterion in self.protocol.inclusion_criteria:
            if ">=" in criterion:
                parts = criterion.split(">=")
                field_name = parts[0].strip()
                threshold = int(parts[1].strip())
                value = demographics.get(field_name)
                if value is None or value < threshold:
                    reasons.append(f"Inclusion criterion not met: {criterion}")
            elif "<=" in criterion:
                parts = criterion.split("<=")
                field_name = parts[0].strip()
                threshold = int(parts[1].strip())
                value = demographics.get(field_name)
                if value is None or value > threshold:
                    reasons.append(f"Inclusion criterion not met: {criterion}")
            # Non-comparison criteria are treated as required flags
            elif demographics.get(criterion) is None and "==" not in criterion:
                # Check if it's a simple presence flag
                if not demographics.get(criterion, False):
                    reasons.append(f"Inclusion criterion not met: {criterion}")

        # Evaluate exclusion criteria
        for criterion in self.protocol.exclusion_criteria:
            field_name = criterion.strip()
            if demographics.get(field_name, False):
                reasons.append(f"Exclusion criterion met: {criterion}")

        is_eligible = len(reasons) == 0
        return is_eligible, reasons

    # ----- data collection -----------------------------------------------

    def submit_data_point(self, data: TrialDataPoint) -> str:
        """Submit a telomere-length data point from a site.

        Parameters
        ----------
        data : TrialDataPoint
            The measurement data to submit.

        Returns
        -------
        str
            A unique data-point receipt identifier.

        Raises
        ------
        KeyError
            If the site or participant is not registered.
        ValueError
            If the data fails quality validation with a ``REJECTED`` flag.
        """
        if data.site_id not in self._sites:
            raise KeyError(f"Site not found: {data.site_id}")
        if data.participant_id not in self._participants:
            raise KeyError(f"Participant not found: {data.participant_id}")

        flag, issues = self.validate_data_quality(data)
        if flag == DataQualityFlag.REJECTED:
            raise ValueError(
                f"Data point rejected: {'; '.join(issues)}"
            )

        self._data_points.append(data)
        receipt_id = _generate_id(prefix="DP-")

        if flag == DataQualityFlag.FLAGGED:
            logger.warning(
                "Data point flagged: receipt=%s issues=%s",
                receipt_id,
                issues,
            )
        else:
            logger.info(
                "Data point submitted: receipt=%s participant=%s site=%s",
                receipt_id,
                data.participant_id,
                data.site_id,
            )
        return receipt_id

    def validate_data_quality(
        self, data: TrialDataPoint
    ) -> tuple[DataQualityFlag, list[str]]:
        """Validate quality constraints on a telomere-length data point.

        Checks include biologically plausible telomere length range, valid
        quality score, recognised measurement method, valid visit number,
        and future-date guards.

        Parameters
        ----------
        data : TrialDataPoint
            Data point to validate.

        Returns
        -------
        tuple[DataQualityFlag, list[str]]
            ``(flag, issues)`` where *issues* lists all detected problems.
        """
        issues: list[str] = []
        has_critical = False

        # Telomere length range
        if data.telomere_length_bp < _TELOMERE_LENGTH_MIN_BP:
            issues.append(
                f"Telomere length {data.telomere_length_bp} bp below "
                f"minimum ({_TELOMERE_LENGTH_MIN_BP} bp)"
            )
            has_critical = True
        elif data.telomere_length_bp > _TELOMERE_LENGTH_MAX_BP:
            issues.append(
                f"Telomere length {data.telomere_length_bp} bp above "
                f"maximum ({_TELOMERE_LENGTH_MAX_BP} bp)"
            )
            has_critical = True

        # Quality score
        if not (0.0 <= data.quality_score <= 1.0):
            issues.append(
                f"Quality score {data.quality_score} outside valid range [0, 1]"
            )
            has_critical = True

        # Measurement method
        if data.measurement_method not in _VALID_MEASUREMENT_METHODS:
            issues.append(
                f"Unknown measurement method: {data.measurement_method!r}; "
                f"expected one of {sorted(_VALID_MEASUREMENT_METHODS)}"
            )
            has_critical = True

        # Visit number
        if data.visit_number < 1:
            issues.append(f"Invalid visit number: {data.visit_number}")
            has_critical = True

        # Future date
        if data.collection_date > date.today():
            issues.append(
                f"Collection date {data.collection_date} is in the future"
            )
            has_critical = True

        # Borderline quality (flagged, not rejected)
        if 0.0 <= data.quality_score < 0.3:
            issues.append(
                f"Low quality score: {data.quality_score}"
            )

        if has_critical:
            return DataQualityFlag.REJECTED, issues
        if issues:
            return DataQualityFlag.FLAGGED, issues
        return DataQualityFlag.PASSED, issues

    def get_site_data(self, site_id: str) -> list[TrialDataPoint]:
        """Return all data points collected at a specific site.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        list[TrialDataPoint]
            Data points for the given site.

        Raises
        ------
        KeyError
            If the site is not registered.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site not found: {site_id}")
        return [dp for dp in self._data_points if dp.site_id == site_id]

    # ----- analysis ------------------------------------------------------

    def compute_interim_analysis(self) -> TrialResult:
        """Perform an interim analysis using Lan-DeMets alpha spending.

        Implements a group-sequential design with an O'Brien-Fleming-type
        alpha-spending function to control the overall Type-I error rate at
        alpha = 0.05 across multiple interim looks.

        The O'Brien-Fleming spending function is:
            alpha*(t) = 2 - 2 * Phi(z_{alpha/2} / sqrt(t))
        where t is the information fraction and Phi is the standard-normal CDF.

        Returns
        -------
        TrialResult
            Interim analysis results including effect size, p-value, and
            whether the primary endpoint was met at the adjusted boundary.

        Reference:
            Lan, K.K.G. & DeMets, D.L. (1983). Biometrika, 70(3):659-663.
        """
        self._interim_count += 1
        return self._run_analysis(is_final=False)

    def compute_final_analysis(self) -> TrialResult:
        """Perform the final (confirmatory) analysis of the trial.

        Returns
        -------
        TrialResult
            Final analysis results.
        """
        return self._run_analysis(is_final=True)

    def _run_analysis(self, *, is_final: bool) -> TrialResult:
        """Internal analysis engine shared by interim and final analyses.

        Parameters
        ----------
        is_final : bool
            Whether this is the final analysis.

        Returns
        -------
        TrialResult
            Computed trial result.
        """
        active_sites = {
            sid
            for sid, s in self._sites.items()
            if s.status == SiteStatus.ACTIVE
        }

        # Gather measurements
        measurements = [
            dp.telomere_length_bp
            for dp in self._data_points
            if dp.site_id in active_sites
        ]
        n = len(measurements)

        # Compute effect size (one-sample against population mean of 7000 bp)
        population_mean = 7000.0
        if n >= 2:
            sample_mean = statistics.mean(measurements)
            sample_sd = statistics.stdev(measurements)
            if sample_sd > 0:
                effect_size = round((sample_mean - population_mean) / sample_sd, 6)
                z_stat = effect_size * math.sqrt(n)
                p_value = _two_sided_p_value(z_stat)
            else:
                effect_size = 0.0
                p_value = 1.0

            se = sample_sd / math.sqrt(n)
            ci_low = round(sample_mean - 1.96 * se, 2)
            ci_high = round(sample_mean + 1.96 * se, 2)
        else:
            sample_mean = measurements[0] if n == 1 else 0.0
            effect_size = 0.0
            p_value = 1.0
            ci_low = sample_mean
            ci_high = sample_mean

        # Determine alpha boundary
        if is_final:
            alpha_boundary = 0.05
        else:
            alpha_boundary = self._lan_demets_spending(
                info_fraction=self._interim_count / self._max_interims,
                alpha=0.05,
            )

        primary_met = p_value <= alpha_boundary

        # Per-site results
        per_site: dict[str, dict[str, Any]] = {}
        for sid in active_sites:
            site_vals = [
                dp.telomere_length_bp
                for dp in self._data_points
                if dp.site_id == sid
            ]
            if len(site_vals) >= 2:
                per_site[sid] = {
                    "n": len(site_vals),
                    "mean": round(statistics.mean(site_vals), 2),
                    "stdev": round(statistics.stdev(site_vals), 2),
                    "min": round(min(site_vals), 2),
                    "max": round(max(site_vals), 2),
                }
            elif len(site_vals) == 1:
                per_site[sid] = {
                    "n": 1,
                    "mean": round(site_vals[0], 2),
                    "stdev": 0.0,
                    "min": round(site_vals[0], 2),
                    "max": round(site_vals[0], 2),
                }
            else:
                per_site[sid] = {"n": 0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}

        result = TrialResult(
            trial_id=self.trial_id,
            analysis_date=_now_utc(),
            sites_included=len(active_sites),
            total_participants=n,
            primary_endpoint_met=primary_met,
            effect_size=round(effect_size, 6),
            p_value=round(p_value, 6),
            confidence_interval=(ci_low, ci_high),
            per_site_results=per_site,
            adverse_events=list(self._adverse_events),
        )

        analysis_type = "Final" if is_final else f"Interim #{self._interim_count}"
        logger.info(
            "%s analysis completed: n=%d effect=%.4f p=%.6f boundary=%.6f met=%s",
            analysis_type,
            n,
            effect_size,
            p_value,
            alpha_boundary,
            primary_met,
        )
        return result

    @staticmethod
    def _lan_demets_spending(info_fraction: float, alpha: float = 0.05) -> float:
        """Compute the cumulative alpha spent using O'Brien-Fleming spending.

        Implements the Lan-DeMets version of the O'Brien-Fleming spending
        function:

            alpha*(t) = 2 - 2 * Phi(z_{alpha/2} / sqrt(t))

        Parameters
        ----------
        info_fraction : float
            Information fraction t in (0, 1].
        alpha : float
            Overall significance level (default 0.05).

        Returns
        -------
        float
            Cumulative alpha spent up to this information fraction.

        Reference:
            Lan & DeMets (1983), Biometrika, 70(3):659-663.
        """
        if info_fraction <= 0.0:
            return 0.0
        if info_fraction >= 1.0:
            return alpha

        z_alpha_half = _normal_ppf(1.0 - alpha / 2.0)
        spent = 2.0 - 2.0 * _normal_cdf(z_alpha_half / math.sqrt(info_fraction))
        return min(spent, alpha)

    def check_stopping_criteria(self) -> tuple[bool, str]:
        """Check whether the trial should stop early.

        Implements O'Brien-Fleming-like stopping boundaries for both
        efficacy and futility.  Efficacy stopping occurs when the test
        statistic exceeds the adjusted critical value at the current
        information fraction.  Futility stopping uses a conditional-power
        threshold of 20 %.

        Returns
        -------
        tuple[bool, str]
            ``(should_stop, reason)`` — if ``should_stop`` is ``False``,
            *reason* will be ``"continue"``.

        Reference:
            O'Brien & Fleming (1979), Biometrics, 35(3):549-556.
        """
        if not self._data_points:
            return False, "continue"

        active_sites = {
            sid
            for sid, s in self._sites.items()
            if s.status == SiteStatus.ACTIVE
        }
        measurements = [
            dp.telomere_length_bp
            for dp in self._data_points
            if dp.site_id in active_sites
        ]
        n = len(measurements)
        if n < 2:
            return False, "continue"

        sample_mean = statistics.mean(measurements)
        sample_sd = statistics.stdev(measurements)
        population_mean = 7000.0

        if sample_sd == 0:
            return False, "continue"

        z_stat = (sample_mean - population_mean) / (sample_sd / math.sqrt(n))

        # Information fraction
        total_target = self.protocol.target_sample_size
        info_fraction = min(n / max(total_target, 1), 1.0)

        if info_fraction <= 0.0:
            return False, "continue"

        # O'Brien-Fleming efficacy boundary
        z_alpha_half = _normal_ppf(1.0 - 0.05 / 2.0)
        obf_boundary = z_alpha_half / math.sqrt(info_fraction)

        if abs(z_stat) >= obf_boundary:
            logger.info(
                "Stopping for efficacy: |z|=%.4f >= boundary=%.4f at t=%.3f",
                abs(z_stat),
                obf_boundary,
                info_fraction,
            )
            return True, "efficacy"

        # Futility: conditional power < 20 %
        # CP = Phi(z_stat * sqrt(1/t) - z_alpha/2 * sqrt((1 - t)/(t)))
        # simplified: project z to final
        if info_fraction < 1.0:
            z_final_proj = z_stat * math.sqrt(1.0 / info_fraction)
            conditional_power = 1.0 - _normal_cdf(
                z_alpha_half - z_final_proj * math.sqrt(info_fraction)
            )
            if conditional_power < 0.20 and info_fraction >= 0.5:
                logger.info(
                    "Stopping for futility: CP=%.4f < 0.20 at t=%.3f",
                    conditional_power,
                    info_fraction,
                )
                return True, "futility"

        return False, "continue"

    # ----- reporting -----------------------------------------------------

    def generate_dsmb_report(self) -> dict[str, Any]:
        """Generate a Data Safety Monitoring Board report.

        Produces a comprehensive summary covering enrollment, data quality,
        safety (adverse events), and efficacy signals suitable for DSMB
        review.

        Returns
        -------
        dict[str, Any]
            Structured DSMB report with sections for enrollment, safety,
            efficacy, site performance, and data quality.
        """
        enrollment = self.get_enrollment_summary()

        # Safety summary
        ae_by_severity: dict[str, int] = {}
        ae_by_relatedness: dict[str, int] = {}
        for ae in self._adverse_events:
            ae_by_severity[ae.severity] = ae_by_severity.get(ae.severity, 0) + 1
            ae_by_relatedness[ae.relatedness] = (
                ae_by_relatedness.get(ae.relatedness, 0) + 1
            )

        # Efficacy summary
        measurements = [dp.telomere_length_bp for dp in self._data_points]
        efficacy: dict[str, Any] = {}
        if len(measurements) >= 2:
            efficacy = {
                "n_measurements": len(measurements),
                "mean_telomere_length": round(statistics.mean(measurements), 2),
                "stdev_telomere_length": round(statistics.stdev(measurements), 2),
                "median_telomere_length": round(statistics.median(measurements), 2),
            }
        else:
            efficacy = {
                "n_measurements": len(measurements),
                "mean_telomere_length": measurements[0] if measurements else None,
                "stdev_telomere_length": None,
                "median_telomere_length": measurements[0] if measurements else None,
            }

        # Data-quality summary
        quality_scores = [dp.quality_score for dp in self._data_points]
        data_quality: dict[str, Any] = {}
        if quality_scores:
            data_quality = {
                "total_data_points": len(quality_scores),
                "mean_quality": round(statistics.mean(quality_scores), 4),
                "min_quality": round(min(quality_scores), 4),
                "max_quality": round(max(quality_scores), 4),
                "below_threshold": sum(1 for q in quality_scores if q < 0.5),
            }
        else:
            data_quality = {
                "total_data_points": 0,
                "mean_quality": None,
                "min_quality": None,
                "max_quality": None,
                "below_threshold": 0,
            }

        # Site performance
        site_performance: dict[str, dict[str, Any]] = {}
        for sid, site in self._sites.items():
            site_data = self.get_site_data(sid)
            site_performance[sid] = {
                "name": site.name,
                "status": site.status.value,
                "enrolled": site.enrolled_count,
                "data_points": len(site_data),
                "irb_valid": site.irb_expiry_date >= date.today(),
            }

        report = {
            "report_date": _now_utc().isoformat(),
            "trial_id": self.trial_id,
            "protocol_version": self.protocol.version,
            "enrollment": enrollment,
            "safety": {
                "total_adverse_events": len(self._adverse_events),
                "by_severity": ae_by_severity,
                "by_relatedness": ae_by_relatedness,
                "serious_events": sum(
                    1 for ae in self._adverse_events if ae.severity == "severe"
                ),
            },
            "efficacy": efficacy,
            "data_quality": data_quality,
            "site_performance": site_performance,
        }

        logger.info(
            "DSMB report generated: trial=%s sites=%d participants=%d aes=%d",
            self.trial_id,
            len(self._sites),
            len(self._participants),
            len(self._adverse_events),
        )
        return report

    def generate_site_performance_report(
        self, site_id: str
    ) -> dict[str, Any]:
        """Generate a performance report for a specific site.

        Parameters
        ----------
        site_id : str
            Identifier of the site.

        Returns
        -------
        dict[str, Any]
            Site-level performance metrics including enrollment rate,
            data quality, protocol deviations, and adverse events.

        Raises
        ------
        KeyError
            If the site is not registered.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site not found: {site_id}")
        site = self._sites[site_id]
        site_data = self.get_site_data(site_id)
        site_aes = [ae for ae in self._adverse_events if ae.site_id == site_id]
        site_participants = [
            p for p in self._participants.values() if p.site_id == site_id
        ]

        quality_scores = [dp.quality_score for dp in site_data]
        report: dict[str, Any] = {
            "site_id": site_id,
            "name": site.name,
            "location": site.location,
            "principal_investigator": site.principal_investigator,
            "status": site.status.value,
            "enrollment": {
                "enrolled": site.enrolled_count,
                "target": site.target_enrollment,
                "pct": round(
                    site.enrolled_count / max(site.target_enrollment, 1) * 100.0, 1
                ),
                "withdrawn": sum(
                    1
                    for p in site_participants
                    if p.status == ParticipantStatus.WITHDRAWN
                ),
            },
            "data": {
                "total_points": len(site_data),
                "mean_quality": (
                    round(statistics.mean(quality_scores), 4)
                    if quality_scores
                    else None
                ),
                "methods_used": list({dp.measurement_method for dp in site_data}),
            },
            "safety": {
                "total_aes": len(site_aes),
                "severe": sum(1 for ae in site_aes if ae.severity == "severe"),
            },
            "irb": self.verify_irb_status(site_id),
        }
        return report

    def export_trial_data(self, format: str = "csv") -> str:
        """Export all trial data points as a CSV or JSON string.

        Parameters
        ----------
        format : str
            Export format — ``"csv"`` or ``"json"`` (default ``"csv"``).

        Returns
        -------
        str
            Serialised data string.

        Raises
        ------
        ValueError
            If *format* is not ``"csv"`` or ``"json"``.
        """
        if format not in ("csv", "json"):
            raise ValueError(f"Unsupported export format: {format!r}")

        records: list[dict[str, Any]] = []
        for dp in self._data_points:
            records.append({
                "participant_id": dp.participant_id,
                "site_id": dp.site_id,
                "visit_number": dp.visit_number,
                "collection_date": dp.collection_date.isoformat(),
                "telomere_length_bp": dp.telomere_length_bp,
                "measurement_method": dp.measurement_method,
                "quality_score": dp.quality_score,
            })

        if format == "json":
            return json.dumps(records, indent=2)

        # CSV
        if not records:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue()

    # ----- adverse events ------------------------------------------------

    def report_adverse_event(self, event: AdverseEvent) -> str:
        """Report an adverse event.

        Parameters
        ----------
        event : AdverseEvent
            Adverse event record.

        Returns
        -------
        str
            The ``event_id`` of the recorded event.

        Raises
        ------
        KeyError
            If the site or participant is not registered.
        ValueError
            If the severity or relatedness value is invalid.
        """
        if event.site_id not in self._sites:
            raise KeyError(f"Site not found: {event.site_id}")
        if event.participant_id not in self._participants:
            raise KeyError(f"Participant not found: {event.participant_id}")
        if event.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity {event.severity!r}; "
                f"expected one of {sorted(_VALID_SEVERITIES)}"
            )
        if event.relatedness not in _VALID_RELATEDNESS:
            raise ValueError(
                f"Invalid relatedness {event.relatedness!r}; "
                f"expected one of {sorted(_VALID_RELATEDNESS)}"
            )

        self._adverse_events.append(event)
        logger.info(
            "Adverse event reported: id=%s participant=%s severity=%s",
            event.event_id,
            event.participant_id,
            event.severity,
        )
        return event.event_id

    def get_adverse_events(
        self, severity: str | None = None
    ) -> list[AdverseEvent]:
        """Retrieve adverse events, optionally filtered by severity.

        Parameters
        ----------
        severity : str or None
            If provided, only events matching this severity are returned.

        Returns
        -------
        list[AdverseEvent]
            Matching adverse events.
        """
        if severity is None:
            return list(self._adverse_events)
        return [ae for ae in self._adverse_events if ae.severity == severity]

    # ----- multi-site coordination ---------------------------------------

    def synchronize_sites(self) -> dict[str, Any]:
        """Synchronise status across all registered sites.

        Checks IRB validity, enrollment progress, and data-submission
        activity for every site and returns a consolidated status report.

        Returns
        -------
        dict[str, Any]
            Synchronisation report with per-site health indicators.
        """
        sync_results: dict[str, dict[str, Any]] = {}
        issues: list[str] = []

        for sid, site in self._sites.items():
            irb = self.verify_irb_status(sid)
            site_data = self.get_site_data(sid)

            status_ok = site.status == SiteStatus.ACTIVE
            irb_ok = irb["is_valid"]
            has_data = len(site_data) > 0

            if not irb_ok and site.status == SiteStatus.ACTIVE:
                self.update_site_status(sid, SiteStatus.SUSPENDED)
                issues.append(
                    f"Site {sid} suspended: IRB expired"
                )

            sync_results[sid] = {
                "name": site.name,
                "status": self._sites[sid].status.value,
                "irb_valid": irb_ok,
                "irb_days_remaining": irb["days_remaining"],
                "enrolled": site.enrolled_count,
                "data_points": len(site_data),
                "healthy": status_ok and irb_ok,
            }

        return {
            "sync_timestamp": _now_utc().isoformat(),
            "total_sites": len(self._sites),
            "active_sites": sum(
                1 for s in self._sites.values() if s.status == SiteStatus.ACTIVE
            ),
            "sites": sync_results,
            "issues": issues,
        }

    def generate_randomization_schedule(
        self, seed: int | None = None
    ) -> dict[str, Any]:
        """Generate a block-randomisation schedule across sites.

        Uses permuted block randomisation stratified by site to produce
        balanced group assignments (arm A / arm B).

        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        dict[str, Any]
            Randomisation schedule with assignments keyed by participant ID.
        """
        import random

        rng = random.Random(seed)

        active_participants = [
            p
            for p in self._participants.values()
            if p.status
            not in (ParticipantStatus.WITHDRAWN, ParticipantStatus.SCREEN_FAILED)
        ]

        # Group by site for stratified randomisation
        by_site: dict[str, list[str]] = {}
        for p in active_participants:
            by_site.setdefault(p.site_id, []).append(p.participant_id)

        assignments: dict[str, str] = {}
        block_size = 4  # Standard block size

        for site_id, pids in by_site.items():
            rng.shuffle(pids)
            idx = 0
            while idx < len(pids):
                block = pids[idx : idx + block_size]
                # Create balanced block
                half = len(block) // 2
                block_assignments = ["arm_A"] * half + ["arm_B"] * (len(block) - half)
                rng.shuffle(block_assignments)
                for pid, arm in zip(block, block_assignments):
                    assignments[pid] = arm
                idx += block_size

        schedule = {
            "trial_id": self.trial_id,
            "generated_at": _now_utc().isoformat(),
            "seed": seed,
            "block_size": block_size,
            "total_randomised": len(assignments),
            "arm_A_count": sum(1 for a in assignments.values() if a == "arm_A"),
            "arm_B_count": sum(1 for a in assignments.values() if a == "arm_B"),
            "assignments": assignments,
        }

        logger.info(
            "Randomisation schedule generated: n=%d arm_A=%d arm_B=%d seed=%s",
            len(assignments),
            schedule["arm_A_count"],
            schedule["arm_B_count"],
            seed,
        )
        return schedule


# ---------------------------------------------------------------------------
# Extended Data Models for TrialManager
# ---------------------------------------------------------------------------


@dataclass
class ConsentRecord:
    """Record of informed consent for a trial participant.

    Tracks the consent lifecycle including the initial signing, any
    re-consent events triggered by protocol amendments, and voluntary
    withdrawal of consent.

    Attributes:
        participant_id: Anonymised participant identifier.
        consent_version: Version of the ICF (Informed Consent Form) signed.
        date_signed: Date the consent was signed.
        witnessed_by: Name or identifier of the consent witness.
        is_active: Whether the consent is currently active.
        withdrawal_date: Date consent was withdrawn, if applicable.
        withdrawal_reason: Reason for withdrawal, if applicable.
        reconsent_history: List of (version, date) tuples for re-consent events.
    """

    participant_id: str
    consent_version: str
    date_signed: date
    witnessed_by: str
    is_active: bool = True
    withdrawal_date: date | None = None
    withdrawal_reason: str | None = None
    reconsent_history: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Institution:
    """Institutional site participating in a clinical trial.

    A simplified, high-level representation suitable for the
    ``TrialManager`` facade API, mapping to the more detailed
    ``InstitutionSite`` used internally by ``ClinicalTrialCoordinator``.

    Attributes:
        id: Unique institution identifier.
        name: Institution display name (e.g. "Mayo Clinic").
        principal_investigator: Name of the PI at this site.
        irb_number: IRB protocol approval reference number.
        site_status: Current operational status of the site.
        enrolled_count: Number of participants enrolled so far.
        location: City and state / region.
        contact_email: Primary contact e-mail.
        target_enrollment: Site-level enrollment target.
    """

    id: str
    name: str
    principal_investigator: str
    irb_number: str
    site_status: str = "pending_approval"
    enrolled_count: int = 0
    location: str = ""
    contact_email: str = ""
    target_enrollment: int = 50


@dataclass
class ClinicalTrial:
    """High-level descriptor of a multi-institution clinical trial.

    Holds the administrative metadata for a trial managed through the
    ``TrialManager`` facade.

    Attributes:
        trial_id: Unique trial identifier.
        title: Full title of the trial.
        phase: Trial phase (e.g. ``"phase_2"``).
        institutions: List of participating institution IDs.
        status: Trial lifecycle status (``"setup"``, ``"active"``,
            ``"completed"``, ``"suspended"``).
        irb_approval_date: Date of central IRB approval.
        irb_expiry_date: Expiry date of central IRB approval.
        start_date: Planned or actual start date.
        end_date: Planned or actual end date.
        target_enrollment: Total enrollment target across all sites.
        created_at: Timestamp of trial creation.
    """

    trial_id: str
    title: str
    phase: str
    institutions: list[str] = field(default_factory=list)
    status: str = "setup"
    irb_approval_date: date | None = None
    irb_expiry_date: date | None = None
    start_date: date | None = None
    end_date: date | None = None
    target_enrollment: int = 500
    created_at: str = field(default_factory=lambda: _now_utc().isoformat())


# ---------------------------------------------------------------------------
# TrialManager — high-level facade
# ---------------------------------------------------------------------------


class TrialManager:
    """High-level facade for multi-institution clinical trial management.

    Wraps :class:`ClinicalTrialCoordinator` with a simplified API and adds
    capabilities not present in the lower-level coordinator:

    - **Consent management**: Tracks informed-consent records, re-consent
      events triggered by protocol amendments, and consent withdrawal.
    - **Differential-privacy aggregation**: Integrates with
      :class:`~teloscopy.platform.federated.DifferentialPrivacy` to add
      calibrated noise when aggregating telomere-length data across sites,
      preventing reconstruction of individual-level contributions.
    - **Regulatory export**: Generates a complete FDA submission package
      incorporating trial results, site data, adverse events, and DSMB
      reports in a structured format aligned with 21 CFR Part 11.

    Parameters
    ----------
    trial_id : str
        Unique trial identifier.
    title : str
        Full title of the clinical trial.
    phase : str
        Trial phase (``"pilot"``, ``"phase_1"``, ``"phase_2"``,
        ``"phase_3"``, ``"phase_4"``).
    target_enrollment : int
        Total enrollment target across all sites.

    Example
    -------
    >>> manager = TrialManager(
    ...     trial_id="TELO-2024-001",
    ...     title="Multi-site qFISH Telomere Validation",
    ...     phase="phase_2",
    ...     target_enrollment=500,
    ... )
    >>> manager.add_institution(
    ...     name="Mayo Clinic",
    ...     pi="Dr. Smith",
    ...     irb_number="IRB-2024-0042",
    ...     location="Rochester, MN",
    ...     contact_email="smith@mayo.edu",
    ... )
    """

    _VALID_PHASES: set[str] = {p.value for p in TrialPhase}
    _VALID_STATUSES: set[str] = {"setup", "active", "completed", "suspended"}

    def __init__(
        self,
        trial_id: str,
        title: str,
        phase: str,
        target_enrollment: int = 500,
    ) -> None:
        if phase not in self._VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}; expected one of "
                f"{sorted(self._VALID_PHASES)}"
            )

        self._trial = ClinicalTrial(
            trial_id=trial_id,
            title=title,
            phase=phase,
            target_enrollment=target_enrollment,
        )

        protocol = TrialProtocol(
            protocol_id=f"{trial_id}-PROTO",
            title=title,
            phase=TrialPhase(phase),
            primary_endpoint="telomere_length_correlation",
            secondary_endpoints=["inter_site_reproducibility", "measurement_cv"],
            inclusion_criteria=["age >= 18"],
            exclusion_criteria=["active_malignancy"],
            target_sample_size=target_enrollment,
            duration_months=24,
            version="1.0",
        )
        self._coordinator = ClinicalTrialCoordinator(trial_id, protocol)
        self._consent_records: dict[str, ConsentRecord] = {}
        self._institutions: dict[str, Institution] = {}

        logger.info(
            "TrialManager created: trial=%s phase=%s target=%d",
            trial_id,
            phase,
            target_enrollment,
        )

    # -- properties -------------------------------------------------------

    @property
    def trial(self) -> ClinicalTrial:
        """Return the :class:`ClinicalTrial` metadata object."""
        return self._trial

    @property
    def coordinator(self) -> ClinicalTrialCoordinator:
        """Return the underlying :class:`ClinicalTrialCoordinator`."""
        return self._coordinator

    # -- trial lifecycle --------------------------------------------------

    @classmethod
    def create_trial(
        cls,
        trial_id: str,
        title: str,
        phase: str,
        target_enrollment: int = 500,
    ) -> "TrialManager":
        """Factory method to initialise a new clinical trial.

        Parameters
        ----------
        trial_id : str
            Unique trial identifier.
        title : str
            Full title.
        phase : str
            Trial phase string.
        target_enrollment : int
            Total enrollment target.

        Returns
        -------
        TrialManager
            A fully initialised trial manager instance.
        """
        return cls(
            trial_id=trial_id,
            title=title,
            phase=phase,
            target_enrollment=target_enrollment,
        )

    def activate_trial(self) -> None:
        """Transition the trial from ``setup`` to ``active`` status.

        Raises
        ------
        ValueError
            If no institutions have been added or if the trial is not
            in ``setup`` status.
        """
        if self._trial.status != "setup":
            raise ValueError(
                f"Cannot activate trial in status {self._trial.status!r}"
            )
        if not self._institutions:
            raise ValueError("Cannot activate trial with no institutions")
        self._trial.status = "active"
        self._trial.start_date = date.today()
        logger.info("Trial %s activated", self._trial.trial_id)

    # -- institution management -------------------------------------------

    def add_institution(
        self,
        name: str,
        pi: str,
        irb_number: str,
        location: str = "",
        contact_email: str = "",
        target_enrollment: int = 50,
    ) -> str:
        """Add a participating institution (site) to the trial.

        Creates both a high-level :class:`Institution` record and
        registers the corresponding :class:`InstitutionSite` with the
        underlying coordinator.

        Parameters
        ----------
        name : str
            Institution display name.
        pi : str
            Principal investigator name.
        irb_number : str
            IRB approval reference number.
        location : str
            City and state / region.
        contact_email : str
            Primary site contact e-mail.
        target_enrollment : int
            Site-level enrollment target (default 50).

        Returns
        -------
        str
            The generated institution / site identifier.

        Raises
        ------
        ValueError
            If a site with the same name is already registered.
        """
        for inst in self._institutions.values():
            if inst.name == name:
                raise ValueError(f"Institution already registered: {name}")

        site_id = _generate_id(prefix="SITE-")
        today = date.today()

        institution = Institution(
            id=site_id,
            name=name,
            principal_investigator=pi,
            irb_number=irb_number,
            site_status=SiteStatus.ACTIVE.value,
            location=location,
            contact_email=contact_email,
            target_enrollment=target_enrollment,
        )
        self._institutions[site_id] = institution
        self._trial.institutions.append(site_id)

        # Register with the lower-level coordinator
        site = InstitutionSite(
            site_id=site_id,
            name=name,
            location=location,
            principal_investigator=pi,
            irb_approval_number=irb_number,
            irb_approval_date=today,
            irb_expiry_date=date(today.year + 1, today.month, today.day),
            status=SiteStatus.ACTIVE,
            enrolled_count=0,
            target_enrollment=target_enrollment,
            contact_email=contact_email,
            data_sharing_agreement=True,
        )
        self._coordinator.register_site(site)

        logger.info(
            "Institution added: site_id=%s name=%s pi=%s",
            site_id,
            name,
            pi,
        )
        return site_id

    def get_institution(self, site_id: str) -> Institution:
        """Retrieve an :class:`Institution` record by its identifier.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        Institution

        Raises
        ------
        KeyError
            If the site is not registered.
        """
        if site_id not in self._institutions:
            raise KeyError(f"Institution not found: {site_id}")
        return self._institutions[site_id]

    def list_institutions(self) -> list[Institution]:
        """Return all registered institutions."""
        return list(self._institutions.values())

    # -- enrollment & consent ---------------------------------------------

    def enroll_patient(
        self,
        site_id: str,
        demographics: dict[str, Any],
        consent_version: str = "1.0",
        witnessed_by: str = "site_coordinator",
    ) -> str:
        """Enroll a patient at a given site with informed-consent tracking.

        Creates a consent record and delegates the enrollment to the
        underlying :class:`ClinicalTrialCoordinator`.

        Parameters
        ----------
        site_id : str
            Institution / site identifier.
        demographics : dict[str, Any]
            Anonymised demographic data (must include ``age`` and ``sex``).
        consent_version : str
            Version of the informed-consent form signed.
        witnessed_by : str
            Name or role of the consent witness.

        Returns
        -------
        str
            Generated anonymised participant identifier.

        Raises
        ------
        KeyError
            If the site is not registered.
        ValueError
            If the site is not active, IRB has expired, enrollment target
            is reached, or the participant fails eligibility checks.
        """
        if site_id not in self._institutions:
            raise KeyError(f"Institution not found: {site_id}")

        participant_id = self._coordinator.enroll_participant(
            site_id=site_id,
            demographics=demographics,
            consent_version=consent_version,
        )

        # Track consent
        consent = ConsentRecord(
            participant_id=participant_id,
            consent_version=consent_version,
            date_signed=date.today(),
            witnessed_by=witnessed_by,
        )
        self._consent_records[participant_id] = consent

        # Update institution enrolled count
        self._institutions[site_id].enrolled_count += 1

        logger.info(
            "Patient enrolled via TrialManager: id=%s site=%s consent=%s",
            participant_id,
            site_id,
            consent_version,
        )
        return participant_id

    def withdraw_consent(self, participant_id: str, reason: str) -> None:
        """Withdraw a participant's consent and remove them from the trial.

        Records the withdrawal in the consent record and delegates the
        participant withdrawal to the coordinator.

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        reason : str
            Free-text reason for consent withdrawal.

        Raises
        ------
        KeyError
            If the participant is not found.
        ValueError
            If consent is already withdrawn.
        """
        if participant_id not in self._consent_records:
            raise KeyError(f"Consent record not found: {participant_id}")
        consent = self._consent_records[participant_id]
        if not consent.is_active:
            raise ValueError(
                f"Consent already withdrawn for {participant_id}"
            )

        consent.is_active = False
        consent.withdrawal_date = date.today()
        consent.withdrawal_reason = reason

        self._coordinator.withdraw_participant(participant_id, reason)

        logger.info(
            "Consent withdrawn: participant=%s reason=%s",
            participant_id,
            reason,
        )

    def reconsent_participant(
        self,
        participant_id: str,
        new_version: str,
        witnessed_by: str = "site_coordinator",
    ) -> None:
        """Record a re-consent event for a participant.

        Typically triggered by a protocol amendment that requires
        participants to sign an updated informed-consent form.

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        new_version : str
            New consent form version.
        witnessed_by : str
            Name or role of the witness.

        Raises
        ------
        KeyError
            If the participant's consent record is not found.
        ValueError
            If the participant's consent is no longer active.
        """
        if participant_id not in self._consent_records:
            raise KeyError(f"Consent record not found: {participant_id}")
        consent = self._consent_records[participant_id]
        if not consent.is_active:
            raise ValueError(
                f"Cannot re-consent participant {participant_id}: "
                f"consent is withdrawn"
            )

        consent.reconsent_history.append(
            (consent.consent_version, consent.date_signed.isoformat())
        )
        consent.consent_version = new_version
        consent.date_signed = date.today()
        consent.witnessed_by = witnessed_by

        logger.info(
            "Participant re-consented: id=%s new_version=%s",
            participant_id,
            new_version,
        )

    def get_consent_record(self, participant_id: str) -> ConsentRecord:
        """Retrieve the consent record for a participant.

        Parameters
        ----------
        participant_id : str
            Participant identifier.

        Returns
        -------
        ConsentRecord

        Raises
        ------
        KeyError
            If no consent record exists for this participant.
        """
        if participant_id not in self._consent_records:
            raise KeyError(f"Consent record not found: {participant_id}")
        return self._consent_records[participant_id]

    # -- data submission --------------------------------------------------

    def submit_data(
        self,
        site_id: str,
        participant_id: str,
        telomere_length_bp: float,
        measurement_method: str = "qfish",
        visit_number: int = 1,
        quality_score: float = 0.95,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a telomere-length measurement from a site.

        Constructs a :class:`TrialDataPoint` and delegates to the
        coordinator's ``submit_data_point`` method, which performs quality
        validation.

        Parameters
        ----------
        site_id : str
            Collecting site identifier.
        participant_id : str
            Anonymised participant identifier.
        telomere_length_bp : float
            Measured telomere length in base pairs.
        measurement_method : str
            Method used (default ``"qfish"``).
        visit_number : int
            Ordinal visit number (1-based).
        quality_score : float
            Analyst-assigned quality score in [0, 1].
        metadata : dict[str, Any] or None
            Additional metadata.

        Returns
        -------
        str
            Data-point receipt identifier.

        Raises
        ------
        KeyError
            If the site or participant is not registered.
        ValueError
            If the data point fails quality validation.
        """
        data = TrialDataPoint(
            participant_id=participant_id,
            site_id=site_id,
            visit_number=visit_number,
            collection_date=date.today(),
            telomere_length_bp=telomere_length_bp,
            measurement_method=measurement_method,
            quality_score=quality_score,
            metadata=metadata or {},
        )
        return self._coordinator.submit_data_point(data)

    # -- aggregation with differential privacy ----------------------------

    def aggregate_results(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_norm: float = 1.0,
    ) -> dict[str, Any]:
        """Aggregate telomere-length results across sites with differential privacy.

        Collects per-site mean telomere lengths, applies gradient clipping
        and calibrated Gaussian noise from
        :class:`~teloscopy.platform.federated.DifferentialPrivacy`, then
        computes the noised cross-site aggregate.  This prevents the
        central coordinator from reconstructing exact per-site
        contributions while preserving statistical utility.

        Parameters
        ----------
        epsilon : float
            Privacy parameter (default 1.0).  Smaller values provide
            stronger privacy but more noise.
        delta : float
            Privacy failure probability (default 1e-5).
        max_norm : float
            Clipping norm for per-site means (default 1.0).

        Returns
        -------
        dict[str, Any]
            Aggregated result containing ``global_mean``,
            ``global_stdev``, ``n_sites``, ``total_samples``,
            ``per_site_n``, ``privacy_budget``, and ``epsilon`` / ``delta``
            parameters used.

        Notes
        -----
        The Gaussian mechanism satisfies (epsilon, delta)-differential
        privacy with noise calibrated as:

            sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

        Reference: Abadi et al., "Deep Learning with Differential
        Privacy," CCS 2016.
        """
        try:
            import numpy as np
            from teloscopy.platform.federated import DifferentialPrivacy
        except ImportError:
            logger.warning(
                "numpy or federated module not available; "
                "falling back to non-private aggregation"
            )
            return self._aggregate_no_dp()

        # Collect per-site means as a vector
        site_means: list[float] = []
        site_counts: list[int] = []
        site_ids: list[str] = []

        for sid in self._institutions:
            site_data = self._coordinator.get_site_data(sid)
            if site_data:
                vals = [dp.telomere_length_bp for dp in site_data]
                site_means.append(statistics.mean(vals))
                site_counts.append(len(vals))
                site_ids.append(sid)

        if not site_means:
            return {
                "global_mean": None,
                "global_stdev": None,
                "n_sites": 0,
                "total_samples": 0,
                "per_site_n": {},
                "privacy_budget": None,
                "epsilon": epsilon,
                "delta": delta,
            }

        means_array = np.array(site_means, dtype=np.float64)

        # Clip and add noise
        clipped = DifferentialPrivacy.clip_gradients(means_array, max_norm=max_norm * 10000.0)
        sensitivity = max_norm * 10000.0 / max(len(site_means), 1)
        noised = DifferentialPrivacy.add_noise(
            clipped,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
        )

        # Weighted aggregate
        weights = np.array(site_counts, dtype=np.float64)
        total_n = int(weights.sum())
        global_mean = float(np.average(noised, weights=weights))
        global_stdev = float(np.sqrt(
            np.average((noised - global_mean) ** 2, weights=weights)
        )) if len(noised) > 1 else 0.0

        budget = DifferentialPrivacy.compute_privacy_budget(
            n_rounds=1,
            epsilon_per_round=epsilon,
            delta=delta,
        )

        per_site_n = dict(zip(site_ids, [int(c) for c in site_counts]))

        result = {
            "global_mean": round(global_mean, 2),
            "global_stdev": round(global_stdev, 2),
            "n_sites": len(site_means),
            "total_samples": total_n,
            "per_site_n": per_site_n,
            "privacy_budget": {
                "total_epsilon": budget.total_epsilon,
                "total_delta": budget.total_delta,
                "per_round_epsilon": budget.per_round_epsilon,
            },
            "epsilon": epsilon,
            "delta": delta,
        }

        logger.info(
            "DP aggregation: sites=%d samples=%d eps=%.2f delta=%.2e mean=%.2f",
            len(site_means),
            total_n,
            epsilon,
            delta,
            global_mean,
        )
        return result

    def _aggregate_no_dp(self) -> dict[str, Any]:
        """Fallback aggregation without differential privacy.

        Used when numpy or the federated module is not available.

        Returns
        -------
        dict[str, Any]
            Same structure as :meth:`aggregate_results` but without
            privacy guarantees or budget information.
        """
        all_values: list[float] = []
        per_site_n: dict[str, int] = {}

        for sid in self._institutions:
            site_data = self._coordinator.get_site_data(sid)
            vals = [dp.telomere_length_bp for dp in site_data]
            if vals:
                all_values.extend(vals)
                per_site_n[sid] = len(vals)

        if not all_values:
            return {
                "global_mean": None,
                "global_stdev": None,
                "n_sites": 0,
                "total_samples": 0,
                "per_site_n": {},
                "privacy_budget": None,
                "epsilon": None,
                "delta": None,
            }

        return {
            "global_mean": round(statistics.mean(all_values), 2),
            "global_stdev": round(
                statistics.stdev(all_values) if len(all_values) >= 2 else 0.0, 2
            ),
            "n_sites": len(per_site_n),
            "total_samples": len(all_values),
            "per_site_n": per_site_n,
            "privacy_budget": None,
            "epsilon": None,
            "delta": None,
        }

    # -- reporting --------------------------------------------------------

    def generate_dsmb_report(self) -> dict[str, Any]:
        """Generate a Data Safety Monitoring Board report.

        Delegates to the underlying coordinator's DSMB report and
        augments it with consent-management statistics.

        Returns
        -------
        dict[str, Any]
            Structured DSMB report with consent section added.
        """
        report = self._coordinator.generate_dsmb_report()

        # Add consent summary
        active_consents = sum(
            1 for c in self._consent_records.values() if c.is_active
        )
        withdrawn_consents = sum(
            1 for c in self._consent_records.values() if not c.is_active
        )
        reconsented = sum(
            1
            for c in self._consent_records.values()
            if c.reconsent_history
        )
        report["consent"] = {
            "total_consented": len(self._consent_records),
            "active_consents": active_consents,
            "withdrawn_consents": withdrawn_consents,
            "reconsented": reconsented,
        }

        return report

    def get_site_quality_report(self, site_id: str) -> dict[str, Any]:
        """Generate a data-quality report for a specific site.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        dict[str, Any]
            Quality metrics including data-point count, mean quality,
            flagged submissions, and protocol deviations.

        Raises
        ------
        KeyError
            If the site is not registered.
        """
        if site_id not in self._institutions:
            raise KeyError(f"Institution not found: {site_id}")

        site_data = self._coordinator.get_site_data(site_id)
        quality_scores = [dp.quality_score for dp in site_data]
        methods = [dp.measurement_method for dp in site_data]

        # Check for protocol deviations (multiple methods = deviation)
        unique_methods = set(methods)
        protocol_deviations: list[str] = []
        if len(unique_methods) > 1:
            protocol_deviations.append(
                f"Multiple measurement methods used: {sorted(unique_methods)}"
            )

        # Check for low-quality submissions
        low_quality = [q for q in quality_scores if q < 0.5]

        report: dict[str, Any] = {
            "site_id": site_id,
            "name": self._institutions[site_id].name,
            "total_data_points": len(site_data),
            "mean_quality": (
                round(statistics.mean(quality_scores), 4)
                if quality_scores
                else None
            ),
            "min_quality": (
                round(min(quality_scores), 4) if quality_scores else None
            ),
            "max_quality": (
                round(max(quality_scores), 4) if quality_scores else None
            ),
            "low_quality_count": len(low_quality),
            "methods_used": sorted(unique_methods),
            "protocol_deviations": protocol_deviations,
            "enrolled": self._institutions[site_id].enrolled_count,
            "target": self._institutions[site_id].target_enrollment,
            "enrollment_pct": round(
                self._institutions[site_id].enrolled_count
                / max(self._institutions[site_id].target_enrollment, 1)
                * 100.0,
                1,
            ),
        }
        return report

    def get_site_monitoring_report(self) -> dict[str, Any]:
        """Generate a monitoring report across all sites.

        Checks enrollment targets, data-quality thresholds, IRB status,
        and protocol deviations for every registered institution.

        Returns
        -------
        dict[str, Any]
            Summary with per-site health indicators and any flagged
            issues requiring attention.
        """
        sites_report: dict[str, dict[str, Any]] = {}
        issues: list[str] = []

        for sid, inst in self._institutions.items():
            quality = self.get_site_quality_report(sid)
            irb = self._coordinator.verify_irb_status(sid)

            # Flag sites below enrollment pace
            enrollment_pct = quality["enrollment_pct"]
            if enrollment_pct < 25.0 and inst.enrolled_count < inst.target_enrollment:
                issues.append(
                    f"Site {sid} ({inst.name}): enrollment at {enrollment_pct}% "
                    f"— below 25% target pace"
                )

            # Flag quality issues
            if quality["mean_quality"] is not None and quality["mean_quality"] < 0.7:
                issues.append(
                    f"Site {sid} ({inst.name}): mean quality "
                    f"{quality['mean_quality']:.4f} below 0.7 threshold"
                )

            # Flag IRB issues
            if not irb["is_valid"]:
                issues.append(
                    f"Site {sid} ({inst.name}): IRB approval expired"
                )
            elif irb["days_remaining"] < 30:
                issues.append(
                    f"Site {sid} ({inst.name}): IRB expires in "
                    f"{irb['days_remaining']} days"
                )

            # Protocol deviations
            if quality["protocol_deviations"]:
                for dev in quality["protocol_deviations"]:
                    issues.append(f"Site {sid} ({inst.name}): {dev}")

            sites_report[sid] = {
                "name": inst.name,
                "status": inst.site_status,
                "enrollment_pct": enrollment_pct,
                "mean_quality": quality["mean_quality"],
                "data_points": quality["total_data_points"],
                "irb_valid": irb["is_valid"],
                "irb_days_remaining": irb["days_remaining"],
                "protocol_deviations": len(quality["protocol_deviations"]),
            }

        return {
            "trial_id": self._trial.trial_id,
            "report_date": _now_utc().isoformat(),
            "total_sites": len(self._institutions),
            "sites": sites_report,
            "issues": issues,
            "issues_count": len(issues),
        }

    def export_regulatory_package(self) -> dict[str, Any]:
        """Export a complete regulatory package for FDA submission.

        Assembles a structured package containing all information
        required for a regulatory submission, including:

        - Trial metadata and protocol information
        - Site registry and IRB status
        - Enrollment and consent summaries
        - DSMB report (safety, efficacy, data quality)
        - Aggregated results (with differential privacy)
        - Adverse-event listings
        - Exported trial data in JSON format
        - Site monitoring report

        The package structure is aligned with FDA 21 CFR Part 11
        requirements for electronic records.

        Returns
        -------
        dict[str, Any]
            Complete regulatory submission package as a nested dictionary.
        """
        dsmb = self.generate_dsmb_report()
        aggregated = self.aggregate_results()
        monitoring = self.get_site_monitoring_report()
        trial_data_json = self._coordinator.export_trial_data(format="json")

        # Build site registry
        site_registry: list[dict[str, Any]] = []
        for sid, inst in self._institutions.items():
            irb = self._coordinator.verify_irb_status(sid)
            site_registry.append({
                "site_id": sid,
                "name": inst.name,
                "location": inst.location,
                "principal_investigator": inst.principal_investigator,
                "irb_number": inst.irb_number,
                "irb_valid": irb["is_valid"],
                "irb_expiry": irb["expiry_date"],
                "enrolled": inst.enrolled_count,
                "target": inst.target_enrollment,
            })

        # Consent summary
        consent_summary = {
            "total_consented": len(self._consent_records),
            "active": sum(
                1 for c in self._consent_records.values() if c.is_active
            ),
            "withdrawn": sum(
                1 for c in self._consent_records.values() if not c.is_active
            ),
            "consent_versions_used": sorted(
                {c.consent_version for c in self._consent_records.values()}
            ),
        }

        package = {
            "package_type": "FDA_Regulatory_Submission",
            "generated_at": _now_utc().isoformat(),
            "format_version": "1.0",
            "cfr_compliance": "21 CFR Part 11",
            "trial_metadata": {
                "trial_id": self._trial.trial_id,
                "title": self._trial.title,
                "phase": self._trial.phase,
                "status": self._trial.status,
                "start_date": (
                    self._trial.start_date.isoformat()
                    if self._trial.start_date
                    else None
                ),
                "end_date": (
                    self._trial.end_date.isoformat()
                    if self._trial.end_date
                    else None
                ),
                "target_enrollment": self._trial.target_enrollment,
            },
            "protocol": {
                "protocol_id": self._coordinator.protocol.protocol_id,
                "version": self._coordinator.protocol.version,
                "phase": self._coordinator.protocol.phase.value,
                "primary_endpoint": self._coordinator.protocol.primary_endpoint,
                "secondary_endpoints": self._coordinator.protocol.secondary_endpoints,
            },
            "site_registry": site_registry,
            "consent_summary": consent_summary,
            "dsmb_report": dsmb,
            "aggregated_results": aggregated,
            "site_monitoring": monitoring,
            "trial_data": trial_data_json,
            "adverse_events": [
                {
                    "event_id": ae.event_id,
                    "participant_id": ae.participant_id,
                    "site_id": ae.site_id,
                    "description": ae.description,
                    "severity": ae.severity,
                    "relatedness": ae.relatedness,
                    "date_reported": ae.date_reported.isoformat(),
                    "date_resolved": (
                        ae.date_resolved.isoformat() if ae.date_resolved else None
                    ),
                }
                for ae in self._coordinator.get_adverse_events()
            ],
        }

        logger.info(
            "Regulatory package exported: trial=%s sites=%d",
            self._trial.trial_id,
            len(site_registry),
        )
        return package
