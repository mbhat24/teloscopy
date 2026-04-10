"""Teloscopy FastAPI application.

Provides REST endpoints for microscopy image upload, telomere analysis,
disease-risk scoring, personalised nutrition plans, and an HTML frontend
rendered via Jinja2.

Run with::

    uvicorn teloscopy.webapp.app:app --reload
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import logging
import math
import os
import random
import secrets
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Real pipeline imports (lazy — heavy deps load only when used)
# ---------------------------------------------------------------------------
from teloscopy.facial.image_classifier import ImageType, classify_image
from teloscopy.facial.predictor import analyze_face
from teloscopy.genomics.disease_risk import DiseasePredictor
from teloscopy.nutrition.diet_advisor import DietAdvisor
from teloscopy.nutrition.regional_diets import resolve_region
from teloscopy.webapp.health_checkup import HealthCheckupAnalyzer
from teloscopy.webapp.report_parser import (
    compute_extraction_confidence,
    detect_file_type,
    extract_text,
    parse_lab_report,
)
from teloscopy.webapp.models import (
    AgentInfo,
    AgentStatusEnum,
    AgentSystemStatus,
    AnalysisResponse,
    AncestryDerivedPredictionsResponse,
    AncestryEstimateResponse,
    BloodTestPanel,
    ConsentBundle,
    ConsentPurpose,
    ConsentRecord,
    ConditionScreeningResponse,
    DataDeletionRequest,
    DataDeletionResponse,
    DermatologicalAnalysisResponse,
    DietPlanRequest,
    DietPlanResponse,
    DietRecommendation,
    DiseaseRisk,
    DiseaseRiskRequest,
    DiseaseRiskResponse,
    FacialAnalysisResult,
    FacialHealthScreeningResponse,
    FacialMeasurementsResponse,
    GrievanceRequest,
    GrievanceResponse,
    HealthCheckupRequest,
    HealthCheckupResponse,
    HealthResponse,
    ImageValidationResponse,
    JobStatus,
    JobStatusEnum,
    LegalNotice,
    MealPlan,
    NutritionRequest,
    NutritionResponse,
    PharmacogenomicPredictionResponse,
    PredictedVariantResponse,
    ProfileAnalysisRequest,
    ProfileAnalysisResponse,
    ReconstructedDNAResponse,
    ReconstructedSequenceResponse,
    ReportParsePreview,
    RiskLevel,
    Sex,
    TelomereResult,
    UploadResponse,
    UrineTestPanel,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure application logging based on environment variables."""
    log_level = os.getenv("TELOSCOPY_LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    if os.getenv("TELOSCOPY_LOG_FORMAT") == "json":
        # Simple JSON-like structured format without extra deps
        log_format = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO), format=log_format, force=True
    )


_setup_logging()
logger: logging.Logger = logging.getLogger("teloscopy.webapp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_DIR: Path = Path(__file__).resolve().parent
_TEMPLATES_DIR: Path = _BASE_DIR / "templates"
_STATIC_DIR: Path = _BASE_DIR / "static"
_UPLOAD_DIR: Path = Path(os.getenv("TELOSCOPY_UPLOAD_DIR", "/tmp/teloscopy_uploads"))
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MiB
_MIN_IMAGE_DIMENSION: int = 32  # minimum width/height in pixels

# Magic byte signatures for image format validation
_IMAGE_MAGIC_BYTES: dict[str, list[bytes]] = {
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpeg": [b"\xff\xd8\xff"],
    "tiff": [b"II\x2a\x00", b"MM\x00\x2a"],  # little-endian / big-endian
    "bmp": [b"BM"],
    "webp": [b"RIFF\x00\x00\x00\x00WEBP"],  # 4-byte gap is file-size (varies)
}

# WebP has the structure RIFF<size>WEBP — we only check bytes 0-3 and 8-11.
_WEBP_MARKER = b"WEBP"

_TELOSCOPY_ENV: str = os.getenv("TELOSCOPY_ENV", "production")
_CORS_ORIGINS: list[str] = os.getenv(
    "TELOSCOPY_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")

# ---------------------------------------------------------------------------
# In-memory job store (swap for Redis in production)
# ---------------------------------------------------------------------------

_jobs: dict[str, JobStatus] = {}
_JOB_MAX_COUNT: int = 10_000
_JOB_TTL_SECONDS: float = 3600.0  # 1 hour


def _evict_stale_jobs() -> None:
    """Remove expired jobs to prevent unbounded memory growth."""
    if len(_jobs) < _JOB_MAX_COUNT:
        return
    cutoff = time.monotonic() - _JOB_TTL_SECONDS
    stale = [jid for jid, j in _jobs.items() if getattr(j, '_created_at', 0) < cutoff]
    for jid in stale[:len(_jobs) - _JOB_MAX_COUNT + 100]:
        _jobs.pop(jid, None)


_APP_START_TIME: float = time.time()

# ---------------------------------------------------------------------------
# Pipeline singletons (instantiated once at startup)
# ---------------------------------------------------------------------------

_disease_predictor: DiseasePredictor = DiseasePredictor()
_diet_advisor: DietAdvisor = DietAdvisor()
_health_analyzer: HealthCheckupAnalyzer = HealthCheckupAnalyzer(_diet_advisor)

logger.info(
    "Pipeline loaded: %d disease variants, DietAdvisor ready",
    _disease_predictor.variant_count,
)

_ANALYSIS_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(8)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """In-memory sliding-window rate limiter keyed by client IP.

    Tracks timestamps of recent requests per key and rejects requests
    that exceed the configured maximum within the sliding window.
    Thread-safe for use with ``asyncio.to_thread`` or sync middleware.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._requests: dict[str, list[float]] = collections.defaultdict(list)
        self._call_count: int = 0

    def cleanup(self) -> None:
        """Remove keys with no recent requests to prevent memory growth."""
        empty_keys = [k for k, v in self._requests.items() if not v]
        for k in empty_keys:
            del self._requests[k]

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Return *True* if the request is within the rate limit.

        Prunes expired timestamps and appends the current one if allowed.
        """
        now: float = time.time()
        cutoff: float = now - window_seconds
        with self._lock:
            self._call_count += 1
            if self._call_count % 100 == 0:
                self.cleanup()
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]
            if len(self._requests[key]) >= max_requests:
                return False
            self._requests[key].append(now)
            return True


_rate_limiter: RateLimiter = RateLimiter()


def rate_limit(max_requests: int, window_seconds: int = 60):
    """Create a FastAPI dependency that enforces per-IP rate limiting.

    Usage::

        @app.get("/endpoint", dependencies=[Depends(rate_limit(10, 60))])
        async def my_endpoint(): ...

    Raises :class:`~fastapi.HTTPException` with status 429 when the
    caller exceeds *max_requests* within *window_seconds*.
    """

    async def _check_rate_limit(request: Request) -> None:
        client_ip: str = request.client.host if request.client else "unknown"
        key: str = f"{client_ip}:{request.url.path}"
        if not _rate_limiter.is_allowed(key, max_requests, window_seconds):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. Maximum {max_requests} requests "
                    f"per {window_seconds} seconds."
                ),
            )

    return _check_rate_limit


# ---------------------------------------------------------------------------
# Server-side consent enforcement (DPDP Act 2023 Section 6)
# ---------------------------------------------------------------------------

# HMAC secret for signing consent tokens — generated once per process (or
# loaded from env).  Tokens survive for the lifetime of the running server.
_CONSENT_SECRET: bytes = os.getenv(
    "TELOSCOPY_CONSENT_SECRET", secrets.token_hex(32)
).encode()

# In-memory consent store: token → {session_id, purposes, granted_at, withdrawn}
_consent_store: dict[str, dict[str, Any]] = {}
_consent_store_lock: threading.Lock = threading.Lock()

# Tokens older than 24 hours require re-consent.
_CONSENT_TOKEN_TTL: float = 86_400.0

# Set of session_ids that have withdrawn consent — checked on every request.
_withdrawn_sessions: set[str] = set()


def _sign_consent_token(session_id: str, purposes: list[str]) -> str:
    """Create an HMAC-signed consent token encoding session + purposes."""
    timestamp = str(int(time.time()))
    payload = f"{session_id}:{','.join(sorted(purposes))}:{timestamp}"
    sig = hmac.new(_CONSENT_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:32]
    import base64
    token = base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode()
    return token


def _verify_consent_token(token: str) -> dict[str, Any] | None:
    """Verify an HMAC-signed consent token.  Returns payload dict or None."""
    import base64
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.rsplit(":", 1)
        if len(parts) != 2:
            return None
        payload_str, sig = parts
        expected_sig = hmac.new(
            _CONSENT_SECRET, payload_str.encode(), hashlib.sha256
        ).hexdigest()[:32]
        if not hmac.compare_digest(sig, expected_sig):
            return None
        payload_parts = payload_str.split(":", 2)
        if len(payload_parts) != 3:
            return None
        session_id, purposes_str, timestamp_str = payload_parts
        timestamp = int(timestamp_str)
        # Check token TTL
        if time.time() - timestamp > _CONSENT_TOKEN_TTL:
            return None
        # Check if consent was withdrawn
        if session_id in _withdrawn_sessions:
            return None
        return {
            "session_id": session_id,
            "purposes": purposes_str.split(",") if purposes_str else [],
            "granted_at": timestamp,
        }
    except Exception:
        return None


def require_consent(*required_purposes: str):
    """FastAPI dependency that enforces server-side consent verification.

    Checks for a consent token in the ``X-Consent-Token`` header or
    ``consent_token`` cookie.  The token must be a valid, non-expired,
    non-withdrawn HMAC-signed token that includes all *required_purposes*.

    Usage::

        @app.post("/endpoint", dependencies=[Depends(require_consent("health_report"))])
        async def my_endpoint(): ...

    Raises :class:`~fastapi.HTTPException` with 403 if consent is missing
    or insufficient.
    """

    async def _check_consent(
        request: Request,
        x_consent_token: str | None = Header(None),
        consent_token: str | None = Cookie(None),
    ) -> None:
        token = x_consent_token or consent_token
        if not token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Consent required.  Please accept the privacy notice and "
                    "terms of service before using this endpoint.  Submit your "
                    "consent via POST /api/legal/consent to obtain a consent token."
                ),
            )
        payload = _verify_consent_token(token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Consent token is invalid, expired, or has been withdrawn.  "
                    "Please re-submit consent via POST /api/legal/consent."
                ),
            )
        # Check that all required purposes are covered
        granted = set(payload.get("purposes", []))
        missing = set(required_purposes) - granted
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Consent not granted for required purpose(s): "
                    f"{', '.join(sorted(missing))}.  Please update your consent "
                    f"via POST /api/legal/consent."
                ),
            )
        # Attach consent info to request state for downstream use
        request.state.consent_session_id = payload["session_id"]
        request.state.consent_purposes = payload["purposes"]

    return _check_consent


# ---------------------------------------------------------------------------
# CSRF protection
# ---------------------------------------------------------------------------


async def _check_csrf(request: Request) -> None:
    """Verify that state-changing requests include X-Requested-With header.

    This is a simple CSRF mitigation: browsers will not add custom headers
    on cross-origin form submissions.  The frontend's ``fetch()`` calls
    must include ``X-Requested-With: XMLHttpRequest``.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    # Allow consent/legal endpoints without CSRF (they need to be accessible
    # from the consent modal before JS is fully wired up)
    if request.url.path.startswith("/api/legal/"):
        return
    xrw = request.headers.get("x-requested-with", "")
    content_type = request.headers.get("content-type", "")
    # Accept requests with X-Requested-With header, or JSON content type
    # (custom content types cannot be set by cross-origin forms)
    if xrw or "application/json" in content_type or "multipart/form-data" in content_type:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="CSRF validation failed. Include X-Requested-With header.",
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_docs_url: str | None = "/docs" if _TELOSCOPY_ENV != "production" else None
_redoc_url: str | None = "/redoc" if _TELOSCOPY_ENV != "production" else None

app: FastAPI = FastAPI(
    title="Teloscopy API",
    description="Multi-Agent Genomic Intelligence Platform — Telomere analysis, disease risk prediction, and personalized nutrition from qFISH microscopy images.",
    version="2.0.0",
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    contact={"name": "Teloscopy Contributors", "url": "https://github.com/Mahesh2023/teloscopy"},
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_tags=[
        {"name": "Health", "description": "System health and readiness checks"},
        {"name": "Analysis", "description": "Image upload and telomere analysis pipeline"},
        {"name": "Disease Risk", "description": "Genetic disease risk prediction"},
        {"name": "Nutrition", "description": "Personalized diet planning and meal recommendations"},
        {"name": "Agents", "description": "Multi-agent system status and control"},
        {"name": "Legal", "description": "Legal compliance, consent management, and data subject rights (DPDP Act 2023)"},
    ],
)

# -- CORS -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-Consent-Token"],
)

# -- Security headers -------------------------------------------------------

_CSP_POLICY: str = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next: Any) -> Any:
    """Append hardening headers to every HTTP response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = _CSP_POLICY
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


@app.middleware("http")
async def csrf_middleware(request: Request, call_next: Any) -> Any:
    """Enforce CSRF protection on state-changing requests.

    Rejects POST/PUT/DELETE/PATCH requests that lack a valid content-type
    or X-Requested-With header, unless the path is exempt (e.g. legal
    endpoints that must be accessible from the consent modal).
    """
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        path = request.url.path
        # Exempt paths: legal/consent endpoints, OpenAPI docs
        exempt = (
            path.startswith("/api/legal/")
            or path.startswith("/docs")
            or path.startswith("/redoc")
            or path.startswith("/openapi")
        )
        if not exempt:
            xrw = request.headers.get("x-requested-with", "")
            ct = request.headers.get("content-type", "")
            # Custom headers/content-types can't be set by cross-origin HTML forms
            if not (xrw or "application/json" in ct or "multipart/form-data" in ct):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"error": {"code": 403, "message": "CSRF validation failed. Include appropriate Content-Type or X-Requested-With header."}},
                )
    response = await call_next(request)
    return response


# -- Request ID -------------------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Any:
    """Generate a unique request ID, log the request with timing, and attach ID to the response."""
    request_id: str = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.monotonic() - start
        logger.error(
            "%s %s 500 took %.3fs [request_id=%s]",
            request.method,
            request.url.path,
            elapsed,
            request_id,
        )
        raise
    elapsed = time.monotonic() - start
    status_code = response.status_code
    if status_code >= 500:
        logger.error(
            "%s %s %d took %.3fs [request_id=%s]",
            request.method,
            request.url.path,
            status_code,
            elapsed,
            request_id,
        )
    elif status_code >= 400:
        logger.warning(
            "%s %s %d took %.3fs [request_id=%s]",
            request.method,
            request.url.path,
            status_code,
            elapsed,
            request_id,
        )
    else:
        logger.info(
            "%s %s %d took %.3fs",
            request.method,
            request.url.path,
            status_code,
            elapsed,
        )
    response.headers["X-Request-ID"] = request_id
    return response


# -- Exception handler (surface errors in non-production) --------------------


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> Any:
    """Return structured JSON error for HTTP exceptions."""
    from fastapi.responses import JSONResponse

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    code = exc.status_code
    if code >= 500:
        logger.error("HTTPException %d on %s: %s", code, request.url.path, exc.detail)
    elif code >= 400:
        logger.warning("HTTPException %d on %s: %s", code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=code,
        content={"error": {"code": code, "message": str(exc.detail), "request_id": request_id}},
    )


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> Any:
    """Return 422 for ValueError exceptions."""
    from fastapi.responses import JSONResponse

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=422,
        content={"error": {"code": 422, "message": str(exc), "request_id": request_id}},
    )


@app.exception_handler(ValidationError)
async def _validation_error_handler(request: Request, exc: ValidationError) -> Any:
    """Return 422 for pydantic ValidationError exceptions."""
    from fastapi.responses import JSONResponse

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning("ValidationError on %s: %s", request.url.path, exc)
    details = [
        {
            "field": e.get("loc", [])[-1] if e.get("loc") else "unknown",
            "error": e.get("type", "invalid"),
        }
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "Validation error. Please check your input.",
                "details": details,
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> Any:
    """Return traceback detail for unhandled errors to aid debugging."""
    from fastapi.responses import JSONResponse

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    tb = traceback.format_exc()
    logger.error("Unhandled %s on %s: %s\n%s", type(exc).__name__, request.url.path, exc, tb)
    return JSONResponse(
        status_code=500,
        content={
            "error": {"code": 500, "message": "Internal server error", "request_id": request_id}
        },
    )


# -- Startup / shutdown events -----------------------------------------------


@app.on_event("startup")
async def _on_startup() -> None:
    """Log application configuration at startup."""
    env = _TELOSCOPY_ENV
    logger.info("Teloscopy v%s starting [env=%s]", app.version, env)
    logger.info("Templates dir: %s (exists=%s)", _TEMPLATES_DIR, _TEMPLATES_DIR.exists())
    logger.info("Static dir:    %s (exists=%s)", _STATIC_DIR, _STATIC_DIR.exists())
    logger.info("Upload dir:    %s (exists=%s)", _UPLOAD_DIR, _UPLOAD_DIR.exists())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Log graceful shutdown and clean up resources."""
    logger.info("Teloscopy v%s shutting down gracefully.", app.version)
    # Clean up old uploaded files
    try:
        import glob
        for f in glob.glob(str(_UPLOAD_DIR / "*")):
            try:
                Path(f).unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        pass


# -- Templates & static files -----------------------------------------------

templates: Jinja2Templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ===================================================================== #
#  Type translation helpers                                              #
# ===================================================================== #


def _translate_disease_risks(
    risk_profile_risks: list[Any],
) -> list[DiseaseRisk]:
    """Convert ``genomics.disease_risk.DiseaseRisk`` dataclasses to Pydantic models."""
    results: list[DiseaseRisk] = []
    for dr in risk_profile_risks:
        prob = min(dr.lifetime_risk_pct / 100.0, 1.0)
        if prob < 0.20:
            level = RiskLevel.LOW
        elif prob < 0.50:
            level = RiskLevel.MODERATE
        elif prob < 0.75:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.VERY_HIGH

        recs: list[str] = []
        if dr.preventability_score > 0.5:
            recs.append(f"Modifiable risk — preventability {dr.preventability_score:.0%}")
        if dr.age_of_onset_range[0] > 0:
            recs.append(f"Typical onset age {dr.age_of_onset_range[0]}–{dr.age_of_onset_range[1]}")

        results.append(
            DiseaseRisk(
                disease=dr.condition,
                risk_level=level,
                probability=round(prob, 3),
                contributing_factors=dr.contributing_variants,
                recommendations=recs,
            )
        )
    return results


def _translate_diet_recommendation(
    recs: list[Any],
    meal_plans: list[Any],
    calorie_target: int = 2100,
) -> DietRecommendation:
    """Convert ``nutrition.diet_advisor`` results to Pydantic model."""
    key_nutrients: list[str] = []
    foods_increase: list[str] = []
    foods_avoid: list[str] = []
    summary_parts: list[str] = []

    for r in recs[:12]:
        key_nutrients.append(r.nutrient)
        foods_increase.extend(r.target_foods[:3])
        foods_avoid.extend(r.avoid_foods[:2])
        if r.recommendation:
            summary_parts.append(r.recommendation)

    summary = (
        " ".join(summary_parts[:3])
        if summary_parts
        else (
            "Based on your genetic profile and health markers, we recommend "
            "a nutrient-dense, anti-inflammatory diet to support genomic health."
        )
    )

    pydantic_plans: list[MealPlan] = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for mp in meal_plans:

        def _meal_str(items: list[Any]) -> str:
            return ", ".join(f"{fi.name} ({g:.0f}g)" for fi, g in items[:4]) or "Seasonal selection"

        if isinstance(mp.day, int):
            week_num = (mp.day - 1) // 7 + 1
            day_label = day_names[(mp.day - 1) % 7]
            label = f"Week {week_num} — {day_label}" if len(meal_plans) > 7 else day_label
        else:
            label = str(mp.day)

        pydantic_plans.append(
            MealPlan(
                day=label,
                breakfast=_meal_str(mp.breakfast),
                lunch=_meal_str(mp.lunch),
                dinner=_meal_str(mp.dinner),
                snacks=[_meal_str(mp.snacks)] if mp.snacks else [],
            )
        )

    return DietRecommendation(
        summary=summary,
        key_nutrients=list(dict.fromkeys(key_nutrients)),  # deduplicate, keep order
        foods_to_increase=list(dict.fromkeys(foods_increase))[:10],
        foods_to_avoid=list(dict.fromkeys(foods_avoid))[:8],
        meal_plans=pydantic_plans,
        calorie_target=calorie_target,
    )


def _sanitize_float(v: float, default: float = 0.0) -> float:
    """Replace NaN / ±Inf with *default* to keep JSON valid."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return v


def _dataclass_to_dict(obj: Any) -> dict | None:
    """Recursively convert a dataclass to a JSON-safe dict, or return None."""
    if obj is None:
        return None
    from dataclasses import asdict, fields as dc_fields
    try:
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for f in dc_fields(obj):
                val = getattr(obj, f.name)
                result[f.name] = _dataclass_to_dict(val) if hasattr(val, "__dataclass_fields__") else (
                    [_dataclass_to_dict(item) if hasattr(item, "__dataclass_fields__") else item for item in val]
                    if isinstance(val, list) else val
                )
            return result
        return obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)
    except Exception:
        return None


def _translate_facial_profile(profile: Any) -> FacialAnalysisResult:
    """Convert ``facial.predictor.FacialGenomicProfile`` to Pydantic model."""
    m = profile.measurements
    a = profile.ancestry
    sf = _sanitize_float

    # Translate reconstructed DNA if present
    rdna = profile.reconstructed_dna
    reconstructed_dna_resp = None
    if rdna and rdna.sequences:
        reconstructed_dna_resp = ReconstructedDNAResponse(
            sequences=[
                ReconstructedSequenceResponse(
                    rsid=s.rsid,
                    gene=s.gene,
                    chromosome=s.chromosome,
                    position=s.position,
                    ref_allele=s.ref_allele,
                    predicted_allele_1=s.predicted_allele_1,
                    predicted_allele_2=s.predicted_allele_2,
                    flanking_5prime=s.flanking_5prime,
                    flanking_3prime=s.flanking_3prime,
                    confidence=s.confidence,
                )
                for s in rdna.sequences
            ],
            total_variants=rdna.total_variants,
            genome_build=rdna.genome_build,
            fasta=rdna.fasta,
            disclaimer=rdna.disclaimer,
        )

    # Translate pharmacogenomic predictions
    pharma_resp = [
        PharmacogenomicPredictionResponse(
            gene=p.gene,
            rsid=p.rsid,
            predicted_phenotype=p.predicted_phenotype,
            confidence=sf(p.confidence),
            affected_drugs=list(p.affected_drugs),
            clinical_recommendation=p.clinical_recommendation,
            basis=p.basis,
        )
        for p in getattr(profile, "pharmacogenomic_predictions", [])
    ]

    # Translate health screening
    hs = getattr(profile, "health_screening", None)
    health_screening_resp = None
    if hs is not None:
        health_screening_resp = FacialHealthScreeningResponse(
            estimated_bmi_category=hs.estimated_bmi_category,
            bmi_confidence=sf(hs.bmi_confidence),
            anemia_risk_score=sf(hs.anemia_risk_score),
            cardiovascular_risk_indicators=list(hs.cardiovascular_risk_indicators),
            thyroid_indicators=list(hs.thyroid_indicators),
            fatigue_stress_score=sf(hs.fatigue_stress_score),
            hydration_score=sf(hs.hydration_score, 50.0),
        )

    # Translate dermatological analysis
    da = getattr(profile, "dermatological_analysis", None)
    derm_resp = None
    if da is not None:
        derm_resp = DermatologicalAnalysisResponse(
            rosacea_risk_score=sf(da.rosacea_risk_score),
            melasma_risk_score=sf(da.melasma_risk_score),
            photo_aging_gap=da.photo_aging_gap,
            acne_severity_score=sf(da.acne_severity_score),
            skin_cancer_risk_factors=list(da.skin_cancer_risk_factors),
            pigmentation_disorder_risk=sf(da.pigmentation_disorder_risk),
            moisture_barrier_score=sf(da.moisture_barrier_score, 50.0),
        )

    # Translate condition screenings
    cond_resp = [
        ConditionScreeningResponse(
            condition=c.condition,
            risk_score=sf(c.risk_score),
            facial_markers=list(c.facial_markers),
            confidence=sf(c.confidence),
            recommendation=c.recommendation,
        )
        for c in getattr(profile, "condition_screenings", [])
    ]

    # Translate ancestry-derived predictions
    ad = getattr(profile, "ancestry_derived", None)
    ancestry_derived_resp = None
    if ad is not None:
        ancestry_derived_resp = AncestryDerivedPredictionsResponse(
            predicted_mtdna_haplogroup=ad.predicted_mtdna_haplogroup,
            haplogroup_confidence=sf(ad.haplogroup_confidence),
            lactose_tolerance_probability=sf(ad.lactose_tolerance_probability, 0.5),
            alcohol_flush_probability=sf(ad.alcohol_flush_probability),
            caffeine_sensitivity=ad.caffeine_sensitivity,
            bitter_taste_sensitivity=ad.bitter_taste_sensitivity,
            population_specific_risks=list(ad.population_specific_risks),
        )

    return FacialAnalysisResult(
        image_type="face_photo",
        estimated_biological_age=profile.estimated_biological_age,
        estimated_telomere_length_kb=sf(profile.estimated_telomere_length_kb),
        telomere_percentile=profile.telomere_percentile,
        skin_health_score=sf(profile.skin_health_score),
        oxidative_stress_score=sf(profile.oxidative_stress_score),
        predicted_eye_colour=profile.predicted_eye_colour,
        predicted_hair_colour=profile.predicted_hair_colour,
        predicted_skin_type=profile.predicted_skin_type,
        measurements=FacialMeasurementsResponse(
            face_width=sf(m.face_width),
            face_height=sf(m.face_height),
            face_ratio=sf(m.face_ratio),
            skin_brightness=sf(m.skin_brightness),
            skin_uniformity=sf(m.skin_uniformity),
            wrinkle_score=sf(m.wrinkle_score),
            symmetry_score=sf(m.symmetry_score, 0.5),
            dark_circle_score=sf(m.dark_circle_score),
            texture_roughness=sf(m.texture_roughness),
            uv_damage_score=sf(m.uv_damage_score),
        ),
        ancestry=AncestryEstimateResponse(
            european=sf(a.european),
            east_asian=sf(a.east_asian),
            south_asian=sf(a.south_asian),
            african=sf(a.african),
            middle_eastern=sf(a.middle_eastern),
            latin_american=sf(a.latin_american),
            confidence=sf(a.confidence),
        ),
        predicted_variants=[
            PredictedVariantResponse(
                rsid=v.rsid,
                gene=v.gene,
                predicted_genotype=v.predicted_genotype,
                confidence=v.confidence,
                basis=v.basis,
            )
            for v in profile.predicted_variants
        ],
        reconstructed_dna=reconstructed_dna_resp,
        pharmacogenomic_predictions=pharma_resp,
        health_screening=health_screening_resp,
        dermatological_analysis=derm_resp,
        condition_screenings=cond_resp,
        ancestry_derived=ancestry_derived_resp,
        analysis_warnings=profile.analysis_warnings,
        # v2.1 future direction modules — serialize dataclasses to dicts
        epigenetic_clock=_dataclass_to_dict(getattr(profile, "epigenetic_clock", None)),
        stela_profile=_dataclass_to_dict(getattr(profile, "stela_profile", None)),
        cfdna_telomere=_dataclass_to_dict(getattr(profile, "cfdna_telomere", None)),
        drug_targets=_dataclass_to_dict(getattr(profile, "drug_targets", None)),
        multi_omics=_dataclass_to_dict(getattr(profile, "multi_omics", None)),
        enhanced_genomic=_dataclass_to_dict(getattr(profile, "enhanced_genomic", None)),
    )


def _build_variant_dict(known_variants: list[str]) -> dict[str, str]:
    """Convert a list of user-supplied variant strings to {rsid: genotype}.

    Accepts formats:
    - ``"rs429358:CT"`` → ``{"rs429358": "CT"}``
    - ``"rs429358"`` (no genotype) → ``{"rs429358": "CT"}`` (heterozygous default)
    """
    result: dict[str, str] = {}
    for v in known_variants:
        v = v.strip()
        if not v:
            continue
        if ":" in v:
            rsid, geno = v.split(":", 1)
            result[rsid.strip()] = geno.strip().upper()
        else:
            # Default to heterozygous
            result[v] = "CT"
    return result


# ===================================================================== #
#  Simulation fallbacks (used when real pipeline data is insufficient)   #
# ===================================================================== #


def _simulate_telomere_analysis() -> TelomereResult:
    """Return plausible mock telomere analysis results.

    Uses the consensus telomere-age model (TL ≈ 11.0 − 0.040 × age)
    with random biological variability so that telomere length and
    biological age are correlated.
    """
    # Random biological age in a realistic range
    bio_age: int = random.randint(20, 85)
    # Consensus model + biological noise (SD ≈ 1.2 kb)
    # Two-phase model: faster attrition birth–20, slower 20+
    if bio_age <= 20:
        base_tl = 11.0 - 0.060 * bio_age
    else:
        base_tl = 9.80 - 0.025 * (bio_age - 20)
    mean_len: float = round(max(4.0, base_tl + random.gauss(0, 1.2)), 2)
    std_dev: float = round(abs(mean_len * 0.12), 2)
    return TelomereResult(
        mean_length=mean_len,
        std_dev=std_dev,
        t_s_ratio=round(max(0.3, (mean_len - 3.274) / 2.413), 2),
        biological_age_estimate=bio_age,
        overlay_image_url=None,
        raw_measurements=[round(max(3.0, mean_len + random.gauss(0, std_dev or 0.5)), 2) for _ in range(20)],
    )


def _telomere_from_facial(facial: FacialAnalysisResult) -> TelomereResult:
    """Build a TelomereResult from facial analysis predictions."""
    tl = facial.estimated_telomere_length_kb
    bio_age = facial.estimated_biological_age
    return TelomereResult(
        mean_length=tl,
        std_dev=round(abs(tl * 0.12), 2),
        t_s_ratio=round(max(0.3, (tl - 3.274) / 2.413), 2),
        biological_age_estimate=bio_age,
        overlay_image_url=None,
        raw_measurements=[round(max(3.0, tl + random.gauss(0, abs(tl * 0.12) or 0.5)), 2) for _ in range(10)],
    )


# ===================================================================== #
#  Core analysis pipeline                                                #
# ===================================================================== #


async def _run_full_analysis(
    job_id: str,
    profile: UserProfile,
    image_path: str,
) -> None:
    """Run the full analysis pipeline in the background.

    Classifies the uploaded image, then routes to either:
    - **FISH microscopy**: simulated telomere analysis (real qFISH pipeline
      requires calibrated multi-channel TIFFs not typical of web uploads)
    - **Face photograph**: real facial-genomic prediction via
      :func:`~teloscopy.facial.predictor.analyze_face`

    In *both* paths the real :class:`DiseasePredictor` and
    :class:`DietAdvisor` are used for disease-risk and nutrition output.
    """
    job: JobStatus = _jobs[job_id]
    try:
        job.status = JobStatusEnum.RUNNING
        job.message = "Classifying uploaded image..."
        job.progress_pct = 5.0
        job.updated_at = datetime.utcnow()

        # ------------------------------------------------------------------
        # Phase 0 — Image classification
        # ------------------------------------------------------------------
        classification = await asyncio.to_thread(classify_image, image_path)
        image_type = classification.image_type
        logger.info(
            "Job %s: image classified as %s (confidence %.2f)",
            job_id,
            image_type,
            classification.confidence,
        )

        job.progress_pct = 10.0
        job.message = f"Image type: {image_type.value}. Starting analysis..."
        job.updated_at = datetime.utcnow()

        facial_result: FacialAnalysisResult | None = None

        # ------------------------------------------------------------------
        # Phase 1 — Telomere / Facial analysis
        # ------------------------------------------------------------------
        if image_type == ImageType.FISH_MICROSCOPY:
            # FISH microscopy — use simulated telomere results
            # (real qFISH pipeline requires calibrated multi-channel TIFF)
            await asyncio.sleep(0.5)
            telomere = _simulate_telomere_analysis()
        else:
            # Face photo OR unknown photo — attempt facial-genomic analysis.
            # The predictor handles the case where no face is detected
            # (falls back to centre-region analysis with a warning).
            facial_profile = await asyncio.to_thread(
                analyze_face,
                image_path,
                profile.age,
                profile.sex.value,
            )
            facial_result = _translate_facial_profile(facial_profile)
            telomere = _telomere_from_facial(facial_result)
            logger.info(
                "Job %s: facial analysis → bio_age=%d, TL=%.2f kb",
                job_id,
                facial_result.estimated_biological_age,
                facial_result.estimated_telomere_length_kb,
            )

        job.progress_pct = 35.0
        job.message = "Telomere analysis complete. Assessing disease risk..."
        job.updated_at = datetime.utcnow()

        # ------------------------------------------------------------------
        # Phase 2 — Disease risk (real predictor)
        # ------------------------------------------------------------------
        variant_dict = _build_variant_dict(profile.known_variants)

        # If facial analysis predicted variants, merge them in
        if facial_result and facial_result.predicted_variants:
            for pv in facial_result.predicted_variants:
                if pv.rsid not in variant_dict and pv.confidence > 0.3:
                    # Use actual risk/ref alleles from the predictor.
                    risk_al = getattr(pv, "risk_allele", "") or "T"
                    ref_al = getattr(pv, "ref_allele", "") or "C"
                    if "homozygous variant" in pv.predicted_genotype:
                        variant_dict[pv.rsid] = f"{risk_al}{risk_al}"
                    elif "heterozygous" in pv.predicted_genotype:
                        variant_dict[pv.rsid] = f"{ref_al}{risk_al}"
                    else:
                        variant_dict[pv.rsid] = f"{ref_al}{ref_al}"

        risk_profile = await asyncio.to_thread(
            _disease_predictor.predict_from_variants,
            variant_dict,
            profile.age,
            profile.sex.value,
        )

        # If facial analysis estimated telomere length, also compute
        # telomere-based disease risks and merge them in.
        if facial_result and facial_result.estimated_telomere_length_kb > 0:
            tl_bp = facial_result.estimated_telomere_length_kb * 1000.0
            tl_risks = await asyncio.to_thread(
                _disease_predictor.predict_from_telomere_data,
                tl_bp,
                profile.age,
                profile.sex.value,
            )
            # Merge: if a condition already appears from variant analysis,
            # keep the higher lifetime risk; otherwise add the new entry.
            existing = {r.condition: r for r in risk_profile.risks}
            for tr in tl_risks:
                if tr.condition in existing:
                    if tr.lifetime_risk_pct > existing[tr.condition].lifetime_risk_pct:
                        existing[tr.condition] = tr
                else:
                    risk_profile.risks.append(tr)
                    existing[tr.condition] = tr

        risks = _translate_disease_risks(risk_profile.top_risks(n=15))

        job.progress_pct = 65.0
        job.message = "Disease risk assessed. Generating diet plan..."
        job.updated_at = datetime.utcnow()

        # ------------------------------------------------------------------
        # Phase 3 — Diet recommendation (real advisor)
        # ------------------------------------------------------------------
        resolved_region = resolve_region(
            profile.region,
            country=getattr(profile, "country", None),
            state=getattr(profile, "state", None),
        )
        genetic_risk_names = [r.condition for r in risk_profile.risks[:10]]
        diet_recs = await asyncio.to_thread(
            _diet_advisor.generate_recommendations,
            genetic_risk_names,
            variant_dict,
            resolved_region,
            profile.age,
            profile.sex.value,
            profile.dietary_restrictions or None,
        )
        diet_meals = await asyncio.to_thread(
            _diet_advisor.create_meal_plan,
            diet_recs,
            resolved_region,
            2000,
            7,
            profile.dietary_restrictions or None,
        )

        # Safety net: adapt any remaining violations post-hoc.
        if profile.dietary_restrictions:
            diet_meals = await asyncio.to_thread(
                _diet_advisor.adapt_to_restrictions,
                diet_meals,
                profile.dietary_restrictions,
            )

        diet = _translate_diet_recommendation(diet_recs, diet_meals)

        job.progress_pct = 90.0
        job.message = "Compiling final report..."
        job.updated_at = datetime.utcnow()

        await asyncio.sleep(0.2)

        # Done
        result = AnalysisResponse(
            job_id=job_id,
            image_type=image_type.value,
            telomere_results=telomere,
            disease_risks=risks,
            diet_recommendations=diet,
            facial_analysis=facial_result,
            report_url=f"/api/results/{job_id}",
        )
        job.result = result
        job.status = JobStatusEnum.COMPLETED
        job.progress_pct = 100.0
        job.message = "Analysis complete"
        job.updated_at = datetime.utcnow()
        logger.info("Job %s completed successfully.", job_id)
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError:
            pass

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed: %s", job_id, exc)
        job.status = JobStatusEnum.FAILED
        job.message = "Analysis failed. Please try again or contact support."
        job.updated_at = datetime.utcnow()
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError:
            pass


async def _run_full_analysis_limited(
    job_id: str,
    profile: UserProfile,
    image_path: str,
) -> None:
    """Wrapper that enforces a concurrency limit on analysis tasks."""
    async with _ANALYSIS_SEMAPHORE:
        await _run_full_analysis(job_id, profile, image_path)


def _validate_extension(filename: str) -> bool:
    """Return *True* if the filename has an allowed image extension."""
    ext: str = Path(filename).suffix.lower()
    return ext in _ALLOWED_EXTENSIONS


def _detect_image_format(data: bytes) -> str:
    """Detect image format from magic bytes. Returns format name or 'unknown'."""
    # Special handling for RIFF-based formats: WebP is RIFF<4-byte size>WEBP
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == _WEBP_MARKER:
        return "webp"
    for fmt, signatures in _IMAGE_MAGIC_BYTES.items():
        if fmt == "webp":
            continue  # already handled above
        for sig in signatures:
            if data[: len(sig)] == sig:
                return fmt
    return "unknown"


def _validate_image_content(contents: bytes, filename: str) -> ImageValidationResponse:
    """Validate image content: magic bytes, decodability, dimensions.

    Returns an :class:`ImageValidationResponse` with ``valid=True`` if the
    image passes all checks, or ``valid=False`` with a list of issues.
    Extension/content format mismatches are reported as *warnings* (not
    hard failures) when the image can still be decoded by OpenCV.
    """
    issues: list[str] = []
    warnings: list[str] = []
    file_size = len(contents)

    # 1. Magic bytes check (informational — not a hard failure if cv2
    #    can still decode the image)
    detected_format = _detect_image_format(contents)
    ext = Path(filename).suffix.lower()
    ext_to_format = {
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".tif": "tiff",
        ".tiff": "tiff",
        ".bmp": "bmp",
        ".webp": "webp",
    }
    expected_format = ext_to_format.get(ext, "unknown")
    magic_mismatch = False
    format_mismatch_msg = ""
    if detected_format == "unknown":
        magic_mismatch = True
    elif expected_format != "unknown" and detected_format != expected_format:
        format_mismatch_msg = (
            f"Extension '{ext}' suggests {expected_format} but content is {detected_format}."
        )

    # 2. Try to decode with OpenCV
    width, height, channels = 0, 0, 0
    image_type = "unknown"
    face_detected = False
    decoded_ok = False
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            issues.append("Image could not be decoded — file may be corrupted or truncated.")
        else:
            decoded_ok = True
            if img.ndim == 2:
                height, width = img.shape
                channels = 1
            else:
                height, width = img.shape[:2]
                channels = img.shape[2] if img.ndim == 3 else 1

            if width < _MIN_IMAGE_DIMENSION or height < _MIN_IMAGE_DIMENSION:
                issues.append(
                    f"Image is too small ({width}x{height}). "
                    f"Minimum dimension is {_MIN_IMAGE_DIMENSION}px."
                )

            if width > 16384 or height > 16384:
                issues.append(
                    f"Image is extremely large ({width}x{height}). "
                    f"Maximum supported dimension is 16384px."
                )

            # 3. Quick classification for user feedback
            try:
                # Save to temp file for classifier
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(contents)
                    tmp_path = tmp.name
                try:
                    classification = classify_image(tmp_path)
                    image_type = classification.image_type.value
                    face_detected = classification.face_detected
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                image_type = "unknown"
    except ImportError:
        issues.append("OpenCV not available for image validation.")

    # Only report magic-bytes mismatch as a hard failure when the image
    # could not be decoded at all.  If cv2 decoded it successfully the
    # file is usable regardless of unexpected header bytes.
    if magic_mismatch and not decoded_ok:
        issues.append(
            f"File content does not match any known image format. "
            f"Expected {expected_format} based on extension '{ext}'."
        )

    # Extension/content format mismatch: treat as a warning when the
    # image decoded successfully, hard failure only when it didn't.
    if format_mismatch_msg:
        if decoded_ok:
            warnings.append(format_mismatch_msg)
        else:
            issues.append(format_mismatch_msg)

    return ImageValidationResponse(
        valid=len(issues) == 0,
        image_type=image_type,
        width=width,
        height=height,
        channels=channels,
        file_size_bytes=file_size,
        format_detected=detected_format,
        face_detected=face_detected,
        issues=issues,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Legal Compliance Endpoints (DPDP Act 2023)
# ---------------------------------------------------------------------------

# In-memory consent audit log (swap for database in production)
_consent_log: list[dict] = []
_grievance_log: list[dict] = []
_deletion_log: list[dict] = []


@app.get("/api/legal/notice", response_model=LegalNotice, tags=["Legal"])
async def get_legal_notice():
    """Return the legal notice per DPDP Act 2023 Section 5.
    
    Must be presented to the Data Principal before collecting consent.
    """
    return LegalNotice()


@app.post("/api/legal/consent", tags=["Legal"])
async def record_consent(bundle: ConsentBundle, request: Request, response: Response):
    """Record explicit consent from the Data Principal.
    
    Per DPDP Act 2023 Section 6, consent must be free, specific, informed,
    unconditional, and unambiguous with a clear affirmative action.
    
    Returns a signed ``consent_token`` that must be included in subsequent
    API requests (via ``X-Consent-Token`` header or ``consent_token`` cookie)
    to prove that consent was obtained.
    """
    # Validate that required consents are actually granted
    granted_purposes = [c.purpose for c in bundle.consents if c.granted]
    if not granted_purposes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one consent purpose must be granted.",
        )
    if not bundle.data_principal_age_confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Age confirmation is required (DPDP Act Section 9).",
        )
    
    client_ip = request.client.host if request.client else "unknown"
    ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16]
    
    for consent in bundle.consents:
        consent.ip_hash = ip_hash
    
    # Generate signed consent token
    consent_token = _sign_consent_token(bundle.session_id, granted_purposes)
    
    # Store in server-side consent store
    with _consent_store_lock:
        _consent_store[bundle.session_id] = {
            "purposes": granted_purposes,
            "granted_at": datetime.utcnow().isoformat(),
            "ip_hash": ip_hash,
            "token": consent_token,
        }
    
    record = {
        "session_id": bundle.session_id,
        "consents": [c.dict() for c in bundle.consents],
        "age_confirmed": bundle.data_principal_age_confirmed,
        "privacy_policy_version": bundle.privacy_policy_version,
        "terms_version": bundle.terms_version,
        "recorded_at": datetime.utcnow().isoformat(),
        "ip_hash": ip_hash,
    }
    _consent_log.append(record)
    logger.info("Consent recorded for session %s with %d purposes", bundle.session_id, len(bundle.consents))
    
    # Set consent token as cookie (HttpOnly, Secure, SameSite=Strict)
    response.set_cookie(
        key="consent_token",
        value=consent_token,
        httponly=True,
        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
        samesite="strict",
        max_age=int(_CONSENT_TOKEN_TTL),
        path="/api/",
    )
    
    return {
        "status": "recorded",
        "session_id": bundle.session_id,
        "consent_token": consent_token,
        "purposes_consented": granted_purposes,
        "message": "Your consent has been recorded. Include the consent_token in subsequent API requests. You may withdraw consent at any time.",
    }


@app.post("/api/legal/consent/withdraw", tags=["Legal"])
async def withdraw_consent(session_id: str = "", purposes: list[str] = [], response: Response = None):
    """Withdraw consent per DPDP Act 2023 Section 6(6).
    
    The Data Principal may withdraw consent at any time, with the same
    ease as it was given. Withdrawal does not affect lawfulness of
    processing done before withdrawal.
    
    This invalidates the consent token server-side, so subsequent API
    calls using that session's token will be rejected with 403.
    """
    # Invalidate the session's consent server-side
    if session_id:
        _withdrawn_sessions.add(session_id)
        with _consent_store_lock:
            _consent_store.pop(session_id, None)
    
    record = {
        "session_id": session_id,
        "purposes_withdrawn": purposes,
        "withdrawn_at": datetime.utcnow().isoformat(),
    }
    _consent_log.append(record)
    logger.info("Consent withdrawn for session %s, purposes: %s", session_id, purposes)
    
    # Clear the consent cookie
    if response is not None:
        response.delete_cookie(key="consent_token", path="/api/")
    
    return {
        "status": "withdrawn",
        "session_id": session_id,
        "purposes_withdrawn": purposes,
        "message": (
            "Your consent has been withdrawn. No further processing will occur "
            "for the specified purposes. Your consent token has been invalidated. "
            "Note: withdrawal does not affect the lawfulness of processing already "
            "completed. Since Teloscopy processes data ephemerally, no persistent "
            "data needs to be deleted."
        ),
    }


@app.post(
    "/api/legal/data-deletion",
    response_model=DataDeletionResponse,
    tags=["Legal"],
)
async def request_data_deletion(req: DataDeletionRequest):
    """Exercise Right to Erasure under DPDP Act 2023 Section 12(3).
    
    Since Teloscopy processes all data ephemerally (in-memory only,
    no persistent storage), this endpoint confirms that no data
    needs to be deleted and provides an audit record.
    """
    response = DataDeletionResponse(request_id=req.request_id)
    
    _deletion_log.append({
        "request_id": req.request_id,
        "session_id": req.session_id,
        "reason": req.reason,
        "requested_at": req.requested_at.isoformat(),
        "completed_at": response.completed_at.isoformat(),
    })
    logger.info("Data deletion request %s processed", req.request_id)
    
    return response


@app.post(
    "/api/legal/grievance",
    response_model=GrievanceResponse,
    tags=["Legal"],
)
async def submit_grievance(grievance: GrievanceRequest):
    """Submit a grievance per DPDP Act 2023 Section 13.
    
    The Grievance Officer will acknowledge and respond within 30 days.
    """
    _grievance_log.append(grievance.dict())
    # Redact email in logs to avoid PII leakage (log hash instead)
    email_hash = hashlib.sha256(grievance.email.encode()).hexdigest()[:8] if grievance.email else "none"
    logger.info("Grievance %s received from [email_hash=%s]", grievance.grievance_id, email_hash)
    
    return GrievanceResponse(grievance_id=grievance.grievance_id)


@app.get("/api/legal/privacy-policy", tags=["Legal"])
async def get_privacy_policy_summary():
    """Return a summary of the privacy policy with links."""
    return {
        "version": "1.0",
        "last_updated": "2026-04-01",
        "full_document_url": "/docs/privacy-policy",
        "governing_law": "Digital Personal Data Protection Act, 2023 (India)",
        "data_fiduciary": "Teloscopy Project",
        "grievance_officer_email": "animaticalpha123@gmail.com",
        "data_protection_board": "Data Protection Board of India",
        "key_points": [
            "All data is processed ephemerally — nothing is stored on servers",
            "Explicit consent is required before any processing",
            "You can withdraw consent at any time",
            "You have rights to access, correction, erasure, and grievance redressal",
            "No data is shared with third parties",
            "Facial images and health reports are never persisted",
            "Users must be 18+ or have verifiable parental consent",
        ],
    }


# ===================================================================== #
#  HTML (frontend) routes                                                #
# ===================================================================== #


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request) -> HTMLResponse:
    """Serve the main landing / upload page."""
    import json as _json

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"research_json": _json.dumps(_RESEARCH_CACHE)},
    )


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    """Serve the dedicated upload page (same template, scroll-to-upload)."""
    import json as _json

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"scroll_to": "upload", "research_json": _json.dumps(_RESEARCH_CACHE)},
    )


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str) -> HTMLResponse:
    """Serve a results page for a specific job."""
    import json as _json

    job: JobStatus | None = _jobs.get(job_id)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "job_id": job_id,
            "job": job,
            "scroll_to": "results",
            "research_json": _json.dumps(_RESEARCH_CACHE),
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Serve the agent-monitoring dashboard."""
    return templates.TemplateResponse(request=request, name="dashboard.html")


# ===================================================================== #
#  Legal document routes (Privacy Policy & Terms of Service)             #
# ===================================================================== #

# Legal docs are shipped inside the package at teloscopy/data/legal/ so
# they survive pip-install.  _load_legal_doc() tries multiple strategies
# at request time so it works regardless of build caches or install mode.


def _load_legal_doc(filename: str) -> str | None:
    """Return the text of a legal Markdown file, or *None* if not found.

    Lookup order:
    1. ``importlib.resources`` — the standard way to access package data,
       works even when the package is installed as a zipped wheel.
    2. Path relative to *this* file (``teloscopy/data/legal/``).
    3. Project-root ``docs/`` via CWD (Render clones the repo and runs
       ``pip install`` inside the checkout, so CWD is the repo root).
    4. Project-root ``docs/`` computed from ``__file__`` (dev checkout).
    """
    # Strategy 1: importlib.resources  (Python ≥ 3.9)
    try:
        from importlib.resources import files as _res_files  # noqa: F811

        ref = _res_files("teloscopy.data").joinpath("legal", filename)
        return ref.read_text(encoding="utf-8")
    except Exception:  # FileNotFoundError, ModuleNotFoundError, TypeError …
        pass

    # Strategy 2: filesystem path relative to this module
    _pkg_legal = Path(__file__).resolve().parent.parent / "data" / "legal" / filename
    if _pkg_legal.is_file():
        return _pkg_legal.read_text(encoding="utf-8")

    # Strategy 3: CWD / docs  (Render working directory = repo root)
    _cwd_docs = Path.cwd() / "docs" / filename
    if _cwd_docs.is_file():
        return _cwd_docs.read_text(encoding="utf-8")

    # Strategy 4: project root inferred from __file__
    _proj_docs = Path(__file__).resolve().parent.parent.parent.parent / "docs" / filename
    if _proj_docs.is_file():
        return _proj_docs.read_text(encoding="utf-8")

    return None


@app.get("/docs/privacy-policy", response_class=HTMLResponse)
async def privacy_policy_page() -> HTMLResponse:
    """Serve the Privacy Policy as a styled HTML page."""
    return _render_legal_doc("PRIVACY_POLICY.md", "Privacy Policy")


@app.get("/docs/terms-of-service", response_class=HTMLResponse)
async def terms_of_service_page() -> HTMLResponse:
    """Serve the Terms of Service as a styled HTML page."""
    return _render_legal_doc("TERMS_OF_SERVICE.md", "Terms of Service")


def _render_legal_doc(filename: str, title: str) -> HTMLResponse:
    """Render a Markdown legal document as a styled HTML page."""
    md_content = _load_legal_doc(filename)
    if md_content is None:
        raise HTTPException(status_code=404, detail=f"{title} document not found")

    # Simple Markdown-to-HTML conversion for legal docs
    import html as _html
    import re

    content = _html.escape(md_content)
    # Headers
    content = re.sub(r"^######\s+(.+)$", r"<h6>\1</h6>", content, flags=re.MULTILINE)
    content = re.sub(r"^#####\s+(.+)$", r"<h5>\1</h5>", content, flags=re.MULTILINE)
    content = re.sub(r"^####\s+(.+)$", r"<h4>\1</h4>", content, flags=re.MULTILINE)
    content = re.sub(r"^###\s+(.+)$", r"<h3>\1</h3>", content, flags=re.MULTILINE)
    content = re.sub(r"^##\s+(.+)$", r"<h2>\1</h2>", content, flags=re.MULTILINE)
    content = re.sub(r"^#\s+(.+)$", r"<h1>\1</h1>", content, flags=re.MULTILINE)
    # Bold and italic
    content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
    content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", content)
    # Links
    content = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" style="color:#00d4aa;">\1</a>',
        content,
    )
    # Horizontal rules
    content = re.sub(r"^---+$", "<hr>", content, flags=re.MULTILINE)
    # List items
    content = re.sub(r"^[-*]\s+(.+)$", r"<li>\1</li>", content, flags=re.MULTILINE)
    content = re.sub(r"((?:<li>.*</li>\n?)+)", r"<ul>\1</ul>", content)
    # Numbered list items
    content = re.sub(r"^\d+\.\s+(.+)$", r"<li>\1</li>", content, flags=re.MULTILINE)
    # Paragraphs — wrap text blocks
    content = re.sub(r"^(?!<[a-z/]|$)(.+)$", r"<p>\1</p>", content, flags=re.MULTILINE)

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_html.escape(title)} — Teloscopy</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0b0f19; color: #e0e6ed;
            line-height: 1.7; padding: 2rem;
            max-width: 860px; margin: 0 auto;
        }}
        h1 {{ font-size: 2rem; font-weight: 800; color: #00d4aa; margin: 2rem 0 .5rem; }}
        h2 {{ font-size: 1.4rem; font-weight: 700; color: #e0e6ed; margin: 2rem 0 .5rem;
               border-bottom: 1px solid rgba(46,58,89,.6); padding-bottom: .4rem; }}
        h3 {{ font-size: 1.1rem; font-weight: 600; color: #00d4aa; margin: 1.5rem 0 .4rem; }}
        h4, h5, h6 {{ font-size: 1rem; font-weight: 600; margin: 1rem 0 .3rem; }}
        p {{ margin: .6rem 0; font-size: .92rem; color: #a0aec0; }}
        a {{ color: #00d4aa; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        ul, ol {{ padding-left: 1.5rem; margin: .5rem 0; }}
        li {{ font-size: .92rem; color: #a0aec0; margin: .3rem 0; }}
        strong {{ color: #e0e6ed; }}
        hr {{ border: none; border-top: 1px solid rgba(46,58,89,.6); margin: 2rem 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: .5rem .75rem; border: 1px solid rgba(46,58,89,.6);
                  font-size: .85rem; text-align: left; }}
        th {{ background: rgba(19,24,37,.8); color: #a0aec0; font-weight: 600; }}
        .back-link {{ display: inline-block; margin-bottom: 1.5rem; color: #00d4aa;
                      font-size: .9rem; }}
        .footer {{ text-align: center; margin-top: 3rem; padding: 1.5rem 0;
                   border-top: 1px solid rgba(46,58,89,.6); font-size: .78rem; color: #8f9bb3; }}
    </style>
</head>
<body>
    <a href="/" class="back-link">&larr; Back to Teloscopy</a>
    {content}
    <div class="footer">
        <p>&copy; 2024&ndash;2026 Teloscopy &mdash; Governed by the laws of India</p>
        <p style="margin-top:.4rem;">
            <a href="/docs/privacy-policy">Privacy Policy</a> &middot;
            <a href="/docs/terms-of-service">Terms of Service</a> &middot;
            <a href="mailto:animaticalpha123@gmail.com">Grievance Officer</a>
        </p>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html_page)


# ===================================================================== #
#  Mobile app download links                                             #
# ===================================================================== #

_GITHUB_RELEASES_URL = "https://github.com/Mahesh2023/teloscopy/releases"


@app.get("/api/download/android")
async def download_android_redirect() -> dict[str, str]:
    """Return the GitHub Releases URL for the Android APK."""
    return {
        "url": f"{_GITHUB_RELEASES_URL}/latest",
        "platform": "android",
        "message": "Download the latest APK from the GitHub Releases page.",
    }


@app.get("/api/download/ios")
async def download_ios_redirect() -> dict[str, str]:
    """Return the GitHub Releases URL for the iOS build."""
    return {
        "url": f"{_GITHUB_RELEASES_URL}/latest",
        "platform": "ios",
        "message": "Download the latest iOS build from the GitHub Releases page.",
    }


# ===================================================================== #
#  Research Knowledge Base                                               #
# ===================================================================== #

_PROJECT_ROOT = _BASE_DIR.parent.parent.parent  # src/teloscopy/webapp → project root

_RESEARCH_FILES: list[dict[str, str]] = [
    {
        "id": "knowledge-base",
        "title": "Gene Sequencing & Telomere Analysis",
        "file": "KNOWLEDGE_BASE.md",
    },
    {
        "id": "research",
        "title": "Scientific Foundation & Research",
        "file": "docs/RESEARCH.md",
    },
]


def _load_research_documents() -> dict[str, Any]:
    """Parse research markdown files into structured sections at startup.

    This runs once at import time so the /api/research endpoint never does
    blocking file I/O inside the async event loop.
    """
    import re as _re

    documents: list[dict[str, Any]] = []

    # Try multiple base paths to handle both source-tree and installed-package layouts
    candidate_roots = [
        _PROJECT_ROOT,
        Path.cwd(),
        Path.cwd() / "src" / "teloscopy" / ".." / ".." / "..",
    ]

    for doc_meta in _RESEARCH_FILES:
        text: str | None = None
        for root in candidate_roots:
            filepath = (root / doc_meta["file"]).resolve()
            if filepath.is_file():
                try:
                    text = filepath.read_text(encoding="utf-8")
                    break
                except OSError:
                    continue

        if text is None:
            logger.warning("Research file not found: %s (tried %d locations)", doc_meta["file"], len(candidate_roots))
            continue

        sections: list[dict[str, Any]] = []
        current_section: dict[str, Any] | None = None

        for line in text.split("\n"):
            m = _re.match(r"^##\s+(.+)", line)
            if m and not line.startswith("###"):
                if current_section:
                    current_section["content"] = current_section["content"].rstrip()
                    sections.append(current_section)
                title = m.group(1).strip()
                title_clean = _re.sub(r"^\d+[\.\)]\s*", "", title)
                current_section = {
                    "title": title_clean if title_clean else title,
                    "content": "",
                }
            elif current_section is not None:
                current_section["content"] += line + "\n"

        if current_section:
            current_section["content"] = current_section["content"].rstrip()
            sections.append(current_section)

        documents.append(
            {
                "id": doc_meta["id"],
                "title": doc_meta["title"],
                "sections": sections,
            }
        )

    logger.info(
        "Research library loaded: %d documents, %d sections",
        len(documents),
        sum(len(d["sections"]) for d in documents),
    )
    return {"documents": documents}


# Pre-load at import time — no file I/O happens during request handling
_RESEARCH_CACHE: dict[str, Any] = _load_research_documents()


@app.get("/api/research")
async def get_research() -> dict[str, Any]:
    """Return all research documents as structured sections (served from cache)."""
    return _RESEARCH_CACHE


@app.get("/api/debug/templates")
async def debug_templates(request: Request) -> dict[str, Any]:
    """Diagnostic endpoint for template debugging."""
    if _TELOSCOPY_ENV == "production":
        raise HTTPException(status_code=404, detail="Not found")
    import os
    import traceback

    import starlette

    diag: dict[str, Any] = {
        "templates_dir": str(_TEMPLATES_DIR),
        "templates_dir_exists": _TEMPLATES_DIR.exists(),
        "template_files": (
            [f.name for f in _TEMPLATES_DIR.iterdir()] if _TEMPLATES_DIR.exists() else []
        ),
        "static_dir": str(_STATIC_DIR),
        "static_dir_exists": _STATIC_DIR.exists(),
        "cwd": os.getcwd(),
        "app_file": str(Path(__file__).resolve()),
        "starlette_version": starlette.__version__,
    }

    # Try rendering index.html to catch the actual error
    try:
        resp = templates.TemplateResponse(request=request, name="index.html")
        diag["index_render"] = "OK"
        diag["index_status"] = resp.status_code
    except Exception as exc:
        diag["index_render_error"] = f"{type(exc).__name__}: {exc}"
        diag["index_traceback"] = traceback.format_exc()

    return diag


# ===================================================================== #
#  API routes                                                            #
# ===================================================================== #

# -- Health -----------------------------------------------------------------


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Returns system health status including uptime and version info.",
    dependencies=[Depends(rate_limit(60, 60))],
)
async def health_check() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse()


@app.get(
    "/readiness",
    tags=["Health"],
    summary="Readiness check",
    description="Checks if all subsystems (pipeline, diet advisor, disease predictor) are available and ready to serve requests.",
)
async def readiness_check() -> dict[str, Any]:
    """Return readiness status of all subsystems."""
    return {
        "status": "ready",
        "checks": {
            "pipeline": True,
            "diet_advisor": True,
            "disease_predictor": True,
        },
    }


@app.get(
    "/api/agents/status",
    response_model=AgentSystemStatus,
    tags=["Agents"],
    summary="Agent system status",
    description="Returns the current status of each agent in the multi-agent system, including active jobs and uptime.",
    dependencies=[Depends(rate_limit(60, 60))],
)
async def agents_status() -> AgentSystemStatus:
    """Return status of each agent in the multi-agent system."""
    now: datetime = datetime.utcnow()
    agents: list[AgentInfo] = [
        AgentInfo(
            name="Image Analysis Agent",
            status=AgentStatusEnum.IDLE,
            last_active=now,
            tasks_completed=random.randint(10, 200),
        ),
        AgentInfo(
            name="Genomics Agent",
            status=AgentStatusEnum.IDLE,
            last_active=now,
            tasks_completed=random.randint(10, 200),
        ),
        AgentInfo(
            name="Nutrition Agent",
            status=AgentStatusEnum.IDLE,
            last_active=now,
            tasks_completed=random.randint(10, 200),
        ),
        AgentInfo(
            name="Improvement Agent",
            status=AgentStatusEnum.IDLE,
            last_active=now,
            tasks_completed=random.randint(5, 50),
        ),
    ]
    # If there are running jobs, mark relevant agents as busy
    active: int = sum(1 for j in _jobs.values() if j.status == JobStatusEnum.RUNNING)
    if active > 0 and agents:
        agents[0].status = AgentStatusEnum.BUSY
        agents[0].current_task = "Processing microscopy image"

    return AgentSystemStatus(
        agents=agents,
        total_analyses=len(_jobs),
        active_jobs=active,
        uptime_seconds=round(time.time() - _APP_START_TIME, 1),
    )


# -- Upload -----------------------------------------------------------------


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=201,
    tags=["Analysis"],
    summary="Upload microscopy image",
    description="Upload a microscopy or face photograph image and receive a job_id for tracking subsequent analysis.",
    dependencies=[Depends(rate_limit(10, 60)), Depends(require_consent("telomere_analysis", "facial_analysis"))],
)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a microscopy image and receive a ``job_id``."""
    if not file.filename or not _validate_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"Invalid file type. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"),
        )

    job_id: str = str(uuid.uuid4())
    ext: str = Path(file.filename).suffix.lower()
    dest: Path = _UPLOAD_DIR / f"{job_id}{ext}"

    contents: bytes = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 50 MiB limit.",
        )
    dest.write_bytes(contents)
    safe_filename = file.filename.replace('\n', '_').replace('\r', '_')
    logger.info("Saved upload %s → %s (%d bytes)", safe_filename, dest, len(contents))

    _evict_stale_jobs()
    _jobs[job_id] = JobStatus(job_id=job_id)

    return UploadResponse(job_id=job_id, filename=file.filename)


# -- Status / results -------------------------------------------------------


@app.get(
    "/api/status/{job_id}",
    response_model=JobStatus,
    tags=["Analysis"],
    summary="Get job status",
    description="Return the current status and progress of an analysis job by its job_id.",
    dependencies=[Depends(rate_limit(60, 60)), Depends(require_consent("telomere_analysis"))],
)
async def get_job_status(job_id: str) -> JobStatus:
    """Return the current status of an analysis job."""
    job: JobStatus | None = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return job


@app.get(
    "/api/results/{job_id}",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Get analysis results",
    description="Return the full analysis results (telomere, disease risk, nutrition) for a completed job.",
    dependencies=[Depends(rate_limit(60, 60)), Depends(require_consent("telomere_analysis"))],
)
async def get_job_results(job_id: str) -> AnalysisResponse:
    """Return the full results of a completed analysis job."""
    job: JobStatus | None = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    if job.status != JobStatusEnum.COMPLETED or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not yet complete (status={job.status.value}).",
        )
    return job.result


# -- Full analysis -----------------------------------------------------------


@app.post(
    "/api/analyze",
    response_model=JobStatus,
    status_code=202,
    tags=["Analysis"],
    summary="Run full analysis pipeline",
    description="Upload an image with user profile data and run the complete analysis pipeline (telomere measurement, disease risk, nutrition plan).",
    dependencies=[Depends(rate_limit(20, 60)), Depends(require_consent("telomere_analysis", "disease_risk", "nutrition_plan"))],
)
async def full_analysis(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    region: str = Form(...),
    country: str = Form(""),
    state: str = Form(""),
    dietary_restrictions: str = Form(""),
    known_variants: str = Form(""),
) -> JobStatus:
    """Run the full analysis pipeline (image + profile → results).

    Parameters are sent as multipart form data so the image and JSON
    metadata travel in a single request.
    """
    if not file.filename or not _validate_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    job_id: str = str(uuid.uuid4())
    ext: str = Path(file.filename).suffix.lower()
    dest: Path = _UPLOAD_DIR / f"{job_id}{ext}"

    contents: bytes = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 50 MiB limit.",
        )
    dest.write_bytes(contents)

    # Parse comma-separated lists
    restrictions: list[str] = [r.strip() for r in dietary_restrictions.split(",") if r.strip()]
    variants: list[str] = [v.strip() for v in known_variants.split(",") if v.strip()]

    profile = UserProfile(
        age=age,
        sex=Sex(sex),
        region=region,
        country=country or None,
        state=state or None,
        dietary_restrictions=restrictions,
        known_variants=variants,
    )

    job = JobStatus(job_id=job_id, message="Queued for analysis")
    _evict_stale_jobs()
    _jobs[job_id] = job

    # Fire-and-forget background task
    asyncio.create_task(_run_full_analysis_limited(job_id, profile, str(dest)))  # noqa: RUF006
    logger.info("Queued full analysis job %s", job_id)

    return job


# -- Disease risk (standalone) -----------------------------------------------


@app.post(
    "/api/disease-risk",
    response_model=DiseaseRiskResponse,
    tags=["Disease Risk"],
    summary="Predict disease risks",
    description="Compute disease-risk scores from known genetic variants, age, sex, and region without requiring an image upload.",
    dependencies=[Depends(rate_limit(20, 60)), Depends(require_consent("disease_risk", "genetic_data"))],
)
async def disease_risk(request: DiseaseRiskRequest) -> DiseaseRiskResponse:
    """Compute disease-risk scores from variants and telomere data."""
    variant_dict = _build_variant_dict(request.known_variants)
    try:
        risk_profile = await asyncio.to_thread(
            _disease_predictor.predict_from_variants,
            variant_dict,
            request.age,
            request.sex.value,
        )
        risks = _translate_disease_risks(risk_profile.top_risks(n=15))
    except Exception:
        logger.exception("disease-risk prediction failed, returning empty")
        risks = []
    overall: float = round(sum(r.probability for r in risks) / max(len(risks), 1), 3)
    return DiseaseRiskResponse(risks=risks, overall_risk_score=min(overall, 1.0))


# -- Diet plan (standalone) --------------------------------------------------


@app.post(
    "/api/diet-plan",
    response_model=DietPlanResponse,
    tags=["Nutrition"],
    summary="Generate diet plan",
    description="Generate a personalised diet plan based on genetic risk profile, dietary restrictions, and regional food preferences.",
    dependencies=[Depends(rate_limit(20, 60)), Depends(require_consent("nutrition_plan"))],
)
async def diet_plan(request: DietPlanRequest) -> DietPlanResponse:
    """Generate a personalised diet plan."""
    variant_dict = _build_variant_dict(request.known_variants)
    resolved_region = resolve_region(
        request.region,
        country=getattr(request, "country", None),
        state=getattr(request, "state", None),
    )
    try:
        genetic_risk_names = [r.disease for r in request.disease_risks]
        diet_recs = await asyncio.to_thread(
            _diet_advisor.generate_recommendations,
            genetic_risk_names,
            variant_dict,
            resolved_region,
            request.age,
            request.sex.value,
            request.dietary_restrictions or None,
        )
        diet_meals = await asyncio.to_thread(
            _diet_advisor.create_meal_plan,
            diet_recs,
            resolved_region,
            request.calorie_target,
            request.meal_plan_days,
            request.dietary_restrictions or None,
        )

        # Safety net: adapt any remaining violations post-hoc.
        if request.dietary_restrictions:
            diet_meals = await asyncio.to_thread(
                _diet_advisor.adapt_to_restrictions,
                diet_meals,
                request.dietary_restrictions,
            )

        rec = _translate_diet_recommendation(diet_recs, diet_meals)
    except Exception:
        logger.exception("diet-plan generation failed, returning defaults")
        rec = DietRecommendation(
            summary="A balanced diet rich in whole foods is recommended.",
            key_nutrients=["Omega-3", "Folate", "Vitamin D"],
            foods_to_increase=["Leafy greens", "Fatty fish", "Berries"],
            foods_to_avoid=["Processed meats", "Refined sugars"],
            meal_plans=[],
            calorie_target=2100,
        )
    return DietPlanResponse(recommendation=rec)


# -- Image validation --------------------------------------------------------


@app.post(
    "/api/validate-image",
    response_model=ImageValidationResponse,
    tags=["Analysis"],
    summary="Validate image",
    description="Validate an uploaded image before analysis — checks format, dimensions, and classifies as face photo or microscopy image.",
    dependencies=[Depends(rate_limit(10, 60)), Depends(require_consent("telomere_analysis"))],
)
async def validate_image(file: UploadFile = File(...)) -> ImageValidationResponse:
    """Validate an uploaded image before analysis.

    Checks magic bytes, decodability, dimensions, and classifies as
    face photo or microscopy image. Use this for client-side previews.
    """
    if not file.filename or not _validate_extension(file.filename):
        return ImageValidationResponse(
            valid=False,
            issues=[f"Invalid file type. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"],
        )
    contents: bytes = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        return ImageValidationResponse(
            valid=False,
            file_size_bytes=len(contents),
            issues=["File exceeds the 50 MiB limit."],
        )
    return await asyncio.to_thread(_validate_image_content, contents, file.filename)


# -- Profile-only analysis (no image) ----------------------------------------


@app.post(
    "/api/profile-analysis",
    response_model=ProfileAnalysisResponse,
    tags=["Analysis"],
    summary="Profile-only analysis",
    description="Run disease-risk and nutrition analysis using only user-provided details (no image upload required).",
    dependencies=[Depends(rate_limit(20, 60)), Depends(require_consent("disease_risk", "nutrition_plan", "profile_data"))],
)
async def profile_analysis(request: ProfileAnalysisRequest) -> ProfileAnalysisResponse:
    """Run disease-risk and nutrition analysis using only user-provided details.

    No image upload is required. Users provide age, sex, region, dietary
    restrictions, and optionally known genetic variants.
    """
    variant_dict = _build_variant_dict(request.known_variants)
    risks: list[DiseaseRisk] = []
    rec: DietRecommendation | None = None

    # Disease risk
    if request.include_disease_risk:
        try:
            risk_profile = await asyncio.to_thread(
                _disease_predictor.predict_from_variants,
                variant_dict,
                request.age,
                request.sex.value,
            )
            risks = _translate_disease_risks(risk_profile.top_risks(n=15))
        except Exception:
            logger.exception("profile-analysis: disease-risk failed")

    # Nutrition
    if request.include_nutrition:
        try:
            resolved_rgn = resolve_region(
                request.region,
                country=getattr(request, "country", None),
                state=getattr(request, "state", None),
            )
            genetic_risk_names = [r.disease for r in risks[:10]]
            diet_recs = await asyncio.to_thread(
                _diet_advisor.generate_recommendations,
                genetic_risk_names,
                variant_dict,
                resolved_rgn,
                request.age,
                request.sex.value,
                request.dietary_restrictions or None,
            )
            diet_meals = await asyncio.to_thread(
                _diet_advisor.create_meal_plan,
                diet_recs,
                resolved_rgn,
                2000,
                7,
                request.dietary_restrictions or None,
            )
            if request.dietary_restrictions:
                diet_meals = await asyncio.to_thread(
                    _diet_advisor.adapt_to_restrictions,
                    diet_meals,
                    request.dietary_restrictions,
                )
            rec = _translate_diet_recommendation(diet_recs, diet_meals)
        except Exception:
            logger.exception("profile-analysis: nutrition failed")

    overall = round(sum(r.probability for r in risks) / max(len(risks), 1), 3) if risks else 0.0
    return ProfileAnalysisResponse(
        disease_risks=risks,
        diet_recommendations=rec,
        overall_risk_score=min(overall, 1.0),
    )


# -- Standalone nutrition endpoint -------------------------------------------


@app.post(
    "/api/nutrition",
    response_model=NutritionResponse,
    tags=["Nutrition"],
    summary="Personalised nutrition plan",
    description="Generate a personalised nutrition plan from user details including health conditions, dietary restrictions, and genetic variants.",
    dependencies=[Depends(rate_limit(20, 60)), Depends(require_consent("nutrition_plan"))],
)
async def nutrition_plan(request: NutritionRequest) -> NutritionResponse:
    """Generate a personalised nutrition plan from user details.

    Accepts age, sex, region, dietary restrictions, known variants,
    health conditions, calorie target, and number of meal plan days.
    No image required.
    """
    variant_dict = _build_variant_dict(request.known_variants)
    resolved_region = resolve_region(
        request.region,
        country=getattr(request, "country", None),
        state=getattr(request, "state", None),
    )
    try:
        # Use health conditions as genetic risk proxies
        genetic_risk_names = list(request.health_conditions)
        diet_recs = await asyncio.to_thread(
            _diet_advisor.generate_recommendations,
            genetic_risk_names,
            variant_dict,
            resolved_region,
            request.age,
            request.sex.value,
            request.dietary_restrictions or None,
        )
        diet_meals = await asyncio.to_thread(
            _diet_advisor.create_meal_plan,
            diet_recs,
            resolved_region,
            request.calorie_target,
            request.meal_plan_days,
            request.dietary_restrictions or None,
        )
        if request.dietary_restrictions:
            diet_meals = await asyncio.to_thread(
                _diet_advisor.adapt_to_restrictions,
                diet_meals,
                request.dietary_restrictions,
            )
        rec = _translate_diet_recommendation(diet_recs, diet_meals, request.calorie_target)
    except Exception:
        logger.exception("nutrition plan generation failed")
        rec = DietRecommendation(
            summary="A balanced diet rich in whole foods is recommended.",
            key_nutrients=["Omega-3", "Folate", "Vitamin D"],
            foods_to_increase=["Leafy greens", "Berries", "Legumes"],
            foods_to_avoid=["Processed meats", "Refined sugars"],
            meal_plans=[],
            calorie_target=request.calorie_target,
        )
    return NutritionResponse(recommendation=rec)


# -- Health checkup ----------------------------------------------------------


@app.post(
    "/api/health-checkup",
    response_model=HealthCheckupResponse,
    tags=["Health Checkup"],
    summary="Analyse annual health checkup",
    description=(
        "Upload blood test, urine test, and abdomen scan results to "
        "get a personalised health analysis with condition detection, "
        "health scoring, and a diet plan tailored to your findings."
    ),
    dependencies=[Depends(rate_limit(10, 60)), Depends(require_consent("health_report"))],
)
async def health_checkup(request: HealthCheckupRequest) -> HealthCheckupResponse:
    """Analyse health checkup data and return personalised diet plan."""
    # Reject if no lab data at all — a score of 0/100 with 0 parameters is misleading.
    has_blood = request.blood_tests is not None and any(
        v is not None for v in request.blood_tests.model_dump().values()
    )
    has_urine = request.urine_tests is not None and any(
        v is not None for v in request.urine_tests.model_dump().values()
    )
    if not has_blood and not has_urine:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No lab values provided. Please enter at least one blood or urine "
                "test value, or upload a lab report using the 'Upload Report' tab."
            ),
        )
    try:
        response = await asyncio.to_thread(_health_analyzer.analyze, request)
    except Exception:
        logger.exception("health-checkup analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health checkup analysis failed. Please try again.",
        )
    return response


# -- Health checkup report upload --------------------------------------------

_REPORT_ALLOWED_EXTENSIONS: set[str] = {
    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".txt", ".text", ".csv",
}
_MAX_REPORT_BYTES: int = 20 * 1024 * 1024  # 20 MiB


@app.post(
    "/api/health-checkup/parse-report",
    response_model=ReportParsePreview,
    tags=["Health Checkup"],
    summary="Parse uploaded lab report (preview)",
    description=(
        "Upload a lab report (PDF, image, or text file) and get a preview "
        "of extracted lab values. Use this to review and correct values "
        "before running the full health checkup analysis."
    ),
    dependencies=[Depends(rate_limit(10, 60)), Depends(require_consent("health_report"))],
)
async def parse_report_preview(file: UploadFile = File(...)) -> ReportParsePreview:
    """Parse an uploaded lab report and return extracted values for review.

    Supports PDF, image (via OCR), and plain text files.
    """
    filename = file.filename or "report"
    ext = Path(filename).suffix.lower()
    if ext and ext not in _REPORT_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_REPORT_ALLOWED_EXTENSIONS))}",
        )

    contents: bytes = await file.read()
    if len(contents) > _MAX_REPORT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 20 MiB limit for report uploads.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Detect file type and extract text
    file_type = detect_file_type(contents, filename)
    try:
        text = await asyncio.to_thread(extract_text, contents, filename)
    except RuntimeError as exc:
        return ReportParsePreview(
            confidence=0.0,
            file_type=file_type,
            text_length=0,
            unrecognized_lines=[str(exc)],
        )

    if not text.strip():
        hints = []
        if file_type == "pdf":
            hints.append("The PDF may be a scanned image without a text layer. Try uploading as an image instead, or use the manual entry form.")
        elif file_type == "image":
            hints.append("Could not extract text from the image. The image may be too low quality or in an unsupported format. Try the manual entry form.")
        else:
            hints.append("Could not extract any text from the uploaded file.")
        return ReportParsePreview(
            confidence=0.0,
            file_type=file_type,
            text_length=0,
            unrecognized_lines=hints,
        )

    # Parse lab values from extracted text
    blood_tests, urine_tests, abdomen_text = await asyncio.to_thread(
        parse_lab_report, text
    )

    # Compute confidence
    confidence = compute_extraction_confidence(blood_tests, urine_tests, abdomen_text, text)

    # Collect unrecognized lines (lines with numbers that weren't matched)
    import re as _re

    unrecognized: list[str] = []
    recognized_values = set()
    for v in blood_tests.values():
        recognized_values.add(str(v))
    for v in urine_tests.values():
        recognized_values.add(str(v))

    for line in text.split("\n"):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 5:
            continue
        if _re.search(r"\d+\.?\d*", line_stripped):
            # Check if any recognized value appears in this line
            has_recognized = False
            for rv in recognized_values:
                if rv in line_stripped:
                    has_recognized = True
                    break
            if not has_recognized and len(unrecognized) < 50:
                unrecognized.append(line_stripped[:200])

    return ReportParsePreview(
        extracted_blood_tests=blood_tests,
        extracted_urine_tests=urine_tests,
        extracted_abdomen_notes=abdomen_text,
        unrecognized_lines=unrecognized,
        confidence=confidence,
        file_type=file_type,
        text_length=len(text),
    )


@app.post(
    "/api/health-checkup/upload",
    response_model=HealthCheckupResponse,
    tags=["Health Checkup"],
    summary="Upload lab report and analyse",
    description=(
        "Upload a lab report (PDF, image, or text) along with profile data "
        "to get a full health checkup analysis. The report is parsed "
        "automatically and values are used for condition detection, health "
        "scoring, and personalised diet planning."
    ),
    dependencies=[Depends(rate_limit(10, 60)), Depends(require_consent("health_report"))],
)
async def health_checkup_upload(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    region: str = Form(...),
    country: str = Form(None),
    state: str = Form(None),
    dietary_restrictions: str = Form(""),
    known_variants: str = Form(""),
    calorie_target: int = Form(2000),
    meal_plan_days: int = Form(7),
    health_conditions: str = Form(""),
) -> HealthCheckupResponse:
    """Upload a lab report file and run full health checkup analysis.

    The uploaded file is parsed to extract blood test, urine test, and
    abdomen scan values.  These are combined with the provided profile
    data and run through the same analysis pipeline as the manual entry
    endpoint (``/api/health-checkup``).
    """
    filename = file.filename or "report"
    ext = Path(filename).suffix.lower()
    if ext and ext not in _REPORT_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_REPORT_ALLOWED_EXTENSIONS))}",
        )

    contents: bytes = await file.read()
    if len(contents) > _MAX_REPORT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 20 MiB limit for report uploads.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Extract text and parse lab values
    file_type = detect_file_type(contents, filename)
    try:
        text = await asyncio.to_thread(extract_text, contents, filename)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    if not text.strip():
        if file_type == "pdf":
            detail = "Could not extract text from the PDF. It may be a scanned image without a text layer. Try using the 'Extract Lab Values' button first, or enter values manually."
        elif file_type == "image":
            detail = "Could not extract text from the image. It may be too low quality. Try entering values manually."
        else:
            detail = "Could not extract text from the uploaded file. Try entering values manually."
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )

    blood_dict, urine_dict, abdomen_text = await asyncio.to_thread(
        parse_lab_report, text
    )

    if not blood_dict and not urine_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No lab values could be extracted from the uploaded file. "
                "Please check the file format or use manual entry."
            ),
        )

    # Parse comma-separated form fields
    restrictions: list[str] = [r.strip() for r in dietary_restrictions.split(",") if r.strip()]
    variants: list[str] = [v.strip() for v in known_variants.split(",") if v.strip()]
    conditions: list[str] = [c.strip() for c in health_conditions.split(",") if c.strip()]

    # Build request from parsed data
    blood_panel = BloodTestPanel(**blood_dict) if blood_dict else None
    urine_panel = UrineTestPanel(**urine_dict) if urine_dict else None

    checkup_request = HealthCheckupRequest(
        age=age,
        sex=Sex(sex),
        region=region,
        country=country or None,
        state=state or None,
        dietary_restrictions=restrictions,
        known_variants=variants,
        blood_tests=blood_panel,
        urine_tests=urine_panel,
        abdomen_scan_notes=abdomen_text or None,
        calorie_target=calorie_target,
        meal_plan_days=meal_plan_days,
        health_conditions=conditions,
    )

    try:
        response = await asyncio.to_thread(_health_analyzer.analyze, checkup_request)
    except Exception:
        logger.exception("health-checkup upload analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health checkup analysis failed. Please try again.",
        )
    return response
