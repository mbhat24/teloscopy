"""
REST API endpoints for multi-institution clinical trial management.

Provides a FastAPI :class:`~fastapi.APIRouter` that exposes the
:class:`~teloscopy.clinical.trials.TrialManager` capabilities over HTTP.
Endpoints follow RESTful conventions and return JSON responses.

Endpoints
---------
POST /api/trials            Create a new clinical trial.
GET  /api/trials            List all managed trials.
GET  /api/trials/{id}       Retrieve trial details.
POST /api/trials/{id}/sites Add an institution (site) to a trial.
POST /api/trials/{id}/enroll  Enroll a patient at a site.
POST /api/trials/{id}/data  Submit a telomere-length measurement.
GET  /api/trials/{id}/report  Generate a DSMB report.
GET  /api/trials/{id}/export  Export a regulatory submission package.

All endpoints include input validation via Pydantic models and return
structured JSON error responses for invalid requests.

Usage::

    from fastapi import FastAPI
    from teloscopy.clinical.endpoints import trial_router

    app = FastAPI()
    app.include_router(trial_router)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from teloscopy.clinical.trials import TrialManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory trial registry (swap for DB in production)
# ---------------------------------------------------------------------------

_trials: dict[str, TrialManager] = {}


def _get_trial(trial_id: str) -> TrialManager:
    """Look up a trial by ID or raise 404.

    Parameters
    ----------
    trial_id : str
        Trial identifier.

    Returns
    -------
    TrialManager

    Raises
    ------
    HTTPException
        404 if the trial does not exist.
    """
    if trial_id not in _trials:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial not found: {trial_id}",
        )
    return _trials[trial_id]


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class CreateTrialRequest(BaseModel):
    """Request body for creating a new clinical trial."""

    trial_id: str = Field(..., description="Unique trial identifier.")
    title: str = Field(..., description="Full title of the trial.")
    phase: str = Field(
        ...,
        description="Trial phase: pilot, phase_1, phase_2, phase_3, phase_4.",
    )
    target_enrollment: int = Field(
        500, ge=1, description="Total enrollment target across all sites."
    )


class CreateTrialResponse(BaseModel):
    """Response after creating a new trial."""

    trial_id: str
    title: str
    phase: str
    status: str
    target_enrollment: int


class AddSiteRequest(BaseModel):
    """Request body for adding an institution to a trial."""

    name: str = Field(..., description="Institution display name.")
    pi: str = Field(..., description="Principal investigator name.")
    irb_number: str = Field(..., description="IRB approval reference number.")
    location: str = Field("", description="City and state / region.")
    contact_email: str = Field("", description="Primary contact e-mail.")
    target_enrollment: int = Field(50, ge=1, description="Site enrollment target.")


class AddSiteResponse(BaseModel):
    """Response after adding an institution."""

    site_id: str
    name: str
    principal_investigator: str


class EnrollPatientRequest(BaseModel):
    """Request body for enrolling a patient."""

    site_id: str = Field(..., description="Institution / site identifier.")
    demographics: dict[str, Any] = Field(
        ..., description="Anonymised demographic data (must include age, sex)."
    )
    consent_version: str = Field("1.0", description="Consent form version.")
    witnessed_by: str = Field(
        "site_coordinator", description="Consent witness."
    )


class EnrollPatientResponse(BaseModel):
    """Response after enrolling a patient."""

    participant_id: str
    site_id: str
    consent_version: str


class SubmitDataRequest(BaseModel):
    """Request body for submitting a telomere-length measurement."""

    site_id: str = Field(..., description="Collecting site identifier.")
    participant_id: str = Field(..., description="Participant identifier.")
    telomere_length_bp: float = Field(
        ..., description="Measured telomere length in base pairs."
    )
    measurement_method: str = Field("qfish", description="Measurement method.")
    visit_number: int = Field(1, ge=1, description="Visit number (1-based).")
    quality_score: float = Field(
        0.95, ge=0.0, le=1.0, description="Quality score [0, 1]."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )


class SubmitDataResponse(BaseModel):
    """Response after submitting data."""

    receipt_id: str
    site_id: str
    participant_id: str


class TrialSummary(BaseModel):
    """Compact trial summary for listing endpoints."""

    trial_id: str
    title: str
    phase: str
    status: str
    n_institutions: int
    target_enrollment: int


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

trial_router = APIRouter(
    prefix="/api/trials",
    tags=["Clinical Trials"],
    responses={
        404: {"description": "Trial not found"},
        422: {"description": "Validation error"},
    },
)


@trial_router.post(
    "",
    response_model=CreateTrialResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new clinical trial",
    description=(
        "Initialise a new multi-institution clinical trial with the "
        "specified parameters.  Returns the created trial metadata."
    ),
)
async def create_trial(body: CreateTrialRequest) -> CreateTrialResponse:
    """Create a new clinical trial.

    Parameters
    ----------
    body : CreateTrialRequest
        Trial creation parameters.

    Returns
    -------
    CreateTrialResponse
        Created trial metadata.

    Raises
    ------
    HTTPException
        409 if a trial with the same ID already exists.
        422 if the phase is invalid.
    """
    if body.trial_id in _trials:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Trial already exists: {body.trial_id}",
        )

    try:
        manager = TrialManager.create_trial(
            trial_id=body.trial_id,
            title=body.title,
            phase=body.phase,
            target_enrollment=body.target_enrollment,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    _trials[body.trial_id] = manager
    logger.info("Trial created via API: %s", body.trial_id)

    return CreateTrialResponse(
        trial_id=body.trial_id,
        title=body.title,
        phase=body.phase,
        status="setup",
        target_enrollment=body.target_enrollment,
    )


@trial_router.get(
    "",
    response_model=list[TrialSummary],
    summary="List all clinical trials",
    description="Return a compact summary of every managed trial.",
)
async def list_trials() -> list[TrialSummary]:
    """List all managed clinical trials.

    Returns
    -------
    list[TrialSummary]
        One summary per trial.
    """
    summaries: list[TrialSummary] = []
    for tid, mgr in _trials.items():
        trial = mgr.trial
        summaries.append(
            TrialSummary(
                trial_id=trial.trial_id,
                title=trial.title,
                phase=trial.phase,
                status=trial.status,
                n_institutions=len(trial.institutions),
                target_enrollment=trial.target_enrollment,
            )
        )
    return summaries


@trial_router.get(
    "/{trial_id}",
    summary="Get trial details",
    description="Retrieve full details for a specific trial.",
)
async def get_trial(trial_id: str) -> dict[str, Any]:
    """Retrieve detailed trial information.

    Parameters
    ----------
    trial_id : str
        Trial identifier.

    Returns
    -------
    dict[str, Any]
        Trial metadata, institution list, enrollment summary, and
        consent statistics.
    """
    mgr = _get_trial(trial_id)
    trial = mgr.trial
    enrollment = mgr.coordinator.get_enrollment_summary()
    institutions = [
        {
            "id": inst.id,
            "name": inst.name,
            "principal_investigator": inst.principal_investigator,
            "irb_number": inst.irb_number,
            "site_status": inst.site_status,
            "enrolled_count": inst.enrolled_count,
            "target_enrollment": inst.target_enrollment,
            "location": inst.location,
        }
        for inst in mgr.list_institutions()
    ]
    return {
        "trial_id": trial.trial_id,
        "title": trial.title,
        "phase": trial.phase,
        "status": trial.status,
        "target_enrollment": trial.target_enrollment,
        "start_date": trial.start_date.isoformat() if trial.start_date else None,
        "end_date": trial.end_date.isoformat() if trial.end_date else None,
        "created_at": trial.created_at,
        "institutions": institutions,
        "enrollment": enrollment,
    }


@trial_router.post(
    "/{trial_id}/sites",
    response_model=AddSiteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an institution",
    description="Register a new institutional site for the trial.",
)
async def add_site(trial_id: str, body: AddSiteRequest) -> AddSiteResponse:
    """Add an institution to a trial.

    Parameters
    ----------
    trial_id : str
        Trial identifier.
    body : AddSiteRequest
        Institution details.

    Returns
    -------
    AddSiteResponse
        The generated site ID and key metadata.
    """
    mgr = _get_trial(trial_id)
    try:
        site_id = mgr.add_institution(
            name=body.name,
            pi=body.pi,
            irb_number=body.irb_number,
            location=body.location,
            contact_email=body.contact_email,
            target_enrollment=body.target_enrollment,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return AddSiteResponse(
        site_id=site_id,
        name=body.name,
        principal_investigator=body.pi,
    )


@trial_router.post(
    "/{trial_id}/enroll",
    response_model=EnrollPatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a patient",
    description="Enroll a patient at a specific site with consent tracking.",
)
async def enroll_patient(
    trial_id: str, body: EnrollPatientRequest
) -> EnrollPatientResponse:
    """Enroll a patient in the trial.

    Parameters
    ----------
    trial_id : str
        Trial identifier.
    body : EnrollPatientRequest
        Enrollment details.

    Returns
    -------
    EnrollPatientResponse
        The generated participant ID.
    """
    mgr = _get_trial(trial_id)
    try:
        participant_id = mgr.enroll_patient(
            site_id=body.site_id,
            demographics=body.demographics,
            consent_version=body.consent_version,
            witnessed_by=body.witnessed_by,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return EnrollPatientResponse(
        participant_id=participant_id,
        site_id=body.site_id,
        consent_version=body.consent_version,
    )


@trial_router.post(
    "/{trial_id}/data",
    response_model=SubmitDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit site data",
    description="Submit a telomere-length measurement from a site.",
)
async def submit_data(
    trial_id: str, body: SubmitDataRequest
) -> SubmitDataResponse:
    """Submit a telomere-length data point.

    Parameters
    ----------
    trial_id : str
        Trial identifier.
    body : SubmitDataRequest
        Measurement data.

    Returns
    -------
    SubmitDataResponse
        A receipt ID confirming submission.
    """
    mgr = _get_trial(trial_id)
    try:
        receipt_id = mgr.submit_data(
            site_id=body.site_id,
            participant_id=body.participant_id,
            telomere_length_bp=body.telomere_length_bp,
            measurement_method=body.measurement_method,
            visit_number=body.visit_number,
            quality_score=body.quality_score,
            metadata=body.metadata,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return SubmitDataResponse(
        receipt_id=receipt_id,
        site_id=body.site_id,
        participant_id=body.participant_id,
    )


@trial_router.get(
    "/{trial_id}/report",
    summary="Generate DSMB report",
    description=(
        "Generate a Data Safety Monitoring Board report covering "
        "enrollment, safety, efficacy, data quality, and consent."
    ),
)
async def get_dsmb_report(trial_id: str) -> dict[str, Any]:
    """Generate and return a DSMB report.

    Parameters
    ----------
    trial_id : str
        Trial identifier.

    Returns
    -------
    dict[str, Any]
        Structured DSMB report.
    """
    mgr = _get_trial(trial_id)
    return mgr.generate_dsmb_report()


@trial_router.get(
    "/{trial_id}/export",
    summary="Export regulatory package",
    description=(
        "Export a complete regulatory submission package aligned "
        "with 21 CFR Part 11 for FDA review."
    ),
)
async def export_regulatory(trial_id: str) -> dict[str, Any]:
    """Export a regulatory package for the trial.

    Parameters
    ----------
    trial_id : str
        Trial identifier.

    Returns
    -------
    dict[str, Any]
        Complete regulatory submission package.
    """
    mgr = _get_trial(trial_id)
    return mgr.export_regulatory_package()
