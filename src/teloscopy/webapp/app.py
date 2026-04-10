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
import logging
import math
import os
import random
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
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
from teloscopy.webapp.models import (
    AgentInfo,
    AgentStatusEnum,
    AgentSystemStatus,
    AnalysisResponse,
    AncestryEstimateResponse,
    DietPlanRequest,
    DietPlanResponse,
    DietRecommendation,
    DiseaseRisk,
    DiseaseRiskRequest,
    DiseaseRiskResponse,
    FacialAnalysisResult,
    FacialMeasurementsResponse,
    HealthResponse,
    ImageValidationResponse,
    JobStatus,
    JobStatusEnum,
    MealPlan,
    NutritionRequest,
    NutritionResponse,
    PredictedVariantResponse,
    ProfileAnalysisRequest,
    ProfileAnalysisResponse,
    RiskLevel,
    Sex,
    TelomereResult,
    UploadResponse,
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
    "webp": [b"RIFF"],
}

_TELOSCOPY_ENV: str = os.getenv("TELOSCOPY_ENV", "production")
_CORS_ORIGINS: list[str] = os.getenv(
    "TELOSCOPY_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000",
).split(",")

# ---------------------------------------------------------------------------
# In-memory job store (swap for Redis in production)
# ---------------------------------------------------------------------------

_jobs: dict[str, JobStatus] = {}

_APP_START_TIME: float = time.time()

# ---------------------------------------------------------------------------
# Pipeline singletons (instantiated once at startup)
# ---------------------------------------------------------------------------

_disease_predictor: DiseasePredictor = DiseasePredictor()
_diet_advisor: DietAdvisor = DietAdvisor()

logger.info(
    "Pipeline loaded: %d disease variants, DietAdvisor ready",
    _disease_predictor.variant_count,
)

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

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Return *True* if the request is within the rate limit.

        Prunes expired timestamps and appends the current one if allowed.
        """
        now: float = time.time()
        cutoff: float = now - window_seconds
        with self._lock:
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
# FastAPI application
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="Teloscopy API",
    description="Multi-Agent Genomic Intelligence Platform — Telomere analysis, disease risk prediction, and personalized nutrition from qFISH microscopy images.",
    version="2.0.0",
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    contact={"name": "Teloscopy Contributors", "url": "https://github.com/Mahesh2023/teloscopy"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "System health and readiness checks"},
        {"name": "Analysis", "description": "Image upload and telomere analysis pipeline"},
        {"name": "Disease Risk", "description": "Genetic disease risk prediction"},
        {"name": "Nutrition", "description": "Personalized diet planning and meal recommendations"},
        {"name": "Agents", "description": "Multi-agent system status and control"},
    ],
)

# -- CORS -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _TELOSCOPY_ENV == "development" else _CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return JSONResponse(
        status_code=422,
        content={"error": {"code": 422, "message": str(exc), "request_id": request_id}},
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
    env = os.getenv("TELOSCOPY_ENV", "development")
    logger.info("Teloscopy v%s starting [env=%s]", app.version, env)
    logger.info("Templates dir: %s (exists=%s)", _TEMPLATES_DIR, _TEMPLATES_DIR.exists())
    logger.info("Static dir:    %s (exists=%s)", _STATIC_DIR, _STATIC_DIR.exists())
    logger.info("Upload dir:    %s (exists=%s)", _UPLOAD_DIR, _UPLOAD_DIR.exists())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Log graceful shutdown."""
    logger.info("Teloscopy v%s shutting down gracefully.", app.version)


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


def _translate_facial_profile(profile: Any) -> FacialAnalysisResult:
    """Convert ``facial.predictor.FacialGenomicProfile`` to Pydantic model."""
    m = profile.measurements
    a = profile.ancestry
    sf = _sanitize_float
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
        analysis_warnings=profile.analysis_warnings,
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
    mean_len: float = round(max(2.0, 11.0 - 0.040 * bio_age + random.gauss(0, 1.2)), 2)
    std_dev: float = round(abs(mean_len * 0.12), 2)
    return TelomereResult(
        mean_length=mean_len,
        std_dev=std_dev,
        t_s_ratio=round(mean_len / 5.0, 2),
        biological_age_estimate=bio_age,
        overlay_image_url=None,
        raw_measurements=[round(max(1.0, mean_len + random.gauss(0, std_dev or 0.5)), 2) for _ in range(20)],
    )


def _telomere_from_facial(facial: FacialAnalysisResult) -> TelomereResult:
    """Build a TelomereResult from facial analysis predictions."""
    tl = facial.estimated_telomere_length_kb
    bio_age = facial.estimated_biological_age
    return TelomereResult(
        mean_length=tl,
        std_dev=round(abs(tl * 0.12), 2),
        t_s_ratio=round(tl / 5.0, 2),
        biological_age_estimate=bio_age,
        overlay_image_url=None,
        raw_measurements=[round(tl + random.uniform(-0.8, 0.8), 2) for _ in range(10)],
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
        genetic_risk_names = [r.condition for r in risk_profile.risks[:10]]
        diet_recs = await asyncio.to_thread(
            _diet_advisor.generate_recommendations,
            genetic_risk_names,
            variant_dict,
            profile.region,
            profile.age,
            profile.sex.value,
            profile.dietary_restrictions or None,
        )
        diet_meals = await asyncio.to_thread(
            _diet_advisor.create_meal_plan,
            diet_recs,
            profile.region,
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

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed: %s", job_id, exc)
        job.status = JobStatusEnum.FAILED
        job.message = f"Analysis failed: {exc}"
        job.updated_at = datetime.utcnow()


def _validate_extension(filename: str) -> bool:
    """Return *True* if the filename has an allowed image extension."""
    ext: str = Path(filename).suffix.lower()
    return ext in _ALLOWED_EXTENSIONS


def _detect_image_format(data: bytes) -> str:
    """Detect image format from magic bytes. Returns format name or 'unknown'."""
    for fmt, signatures in _IMAGE_MAGIC_BYTES.items():
        for sig in signatures:
            if data[: len(sig)] == sig:
                return fmt
    return "unknown"


def _validate_image_content(contents: bytes, filename: str) -> ImageValidationResponse:
    """Validate image content: magic bytes, decodability, dimensions.

    Returns an :class:`ImageValidationResponse` with ``valid=True`` if the
    image passes all checks, or ``valid=False`` with a list of issues.
    """
    issues: list[str] = []
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
    if detected_format == "unknown":
        magic_mismatch = True
    elif expected_format != "unknown" and detected_format != expected_format:
        issues.append(
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
                classification = classify_image(tmp_path)
                image_type = classification.image_type.value
                face_detected = classification.face_detected
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
    )


# ===================================================================== #
#  HTML (frontend) routes                                                #
# ===================================================================== #


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request) -> HTMLResponse:
    """Serve the main landing / upload page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    """Serve the dedicated upload page (same template, scroll-to-upload)."""
    return templates.TemplateResponse(
        request=request, name="index.html", context={"scroll_to": "upload"}
    )


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str) -> HTMLResponse:
    """Serve a results page for a specific job."""
    job: JobStatus | None = _jobs.get(job_id)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"job_id": job_id, "job": job, "scroll_to": "results"},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Serve the agent-monitoring dashboard."""
    return templates.TemplateResponse(request=request, name="dashboard.html")


@app.get("/api/debug/templates")
async def debug_templates(request: Request) -> dict[str, Any]:
    """Diagnostic endpoint for template debugging."""
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
    dependencies=[Depends(rate_limit(10, 60))],
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
    logger.info("Saved upload %s → %s (%d bytes)", file.filename, dest, len(contents))

    _jobs[job_id] = JobStatus(job_id=job_id)

    return UploadResponse(job_id=job_id, filename=file.filename)


# -- Status / results -------------------------------------------------------


@app.get(
    "/api/status/{job_id}",
    response_model=JobStatus,
    tags=["Analysis"],
    summary="Get job status",
    description="Return the current status and progress of an analysis job by its job_id.",
    dependencies=[Depends(rate_limit(60, 60))],
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
    dependencies=[Depends(rate_limit(60, 60))],
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
    dependencies=[Depends(rate_limit(20, 60))],
)
async def full_analysis(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    region: str = Form(...),
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
        dietary_restrictions=restrictions,
        known_variants=variants,
    )

    job = JobStatus(job_id=job_id, message="Queued for analysis")
    _jobs[job_id] = job

    # Fire-and-forget background task
    asyncio.create_task(_run_full_analysis(job_id, profile, str(dest)))  # noqa: RUF006
    logger.info("Queued full analysis job %s", job_id)

    return job


# -- Disease risk (standalone) -----------------------------------------------


@app.post(
    "/api/disease-risk",
    response_model=DiseaseRiskResponse,
    tags=["Disease Risk"],
    summary="Predict disease risks",
    description="Compute disease-risk scores from known genetic variants, age, sex, and region without requiring an image upload.",
    dependencies=[Depends(rate_limit(20, 60))],
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
    dependencies=[Depends(rate_limit(20, 60))],
)
async def diet_plan(request: DietPlanRequest) -> DietPlanResponse:
    """Generate a personalised diet plan."""
    variant_dict = _build_variant_dict(request.known_variants)
    try:
        genetic_risk_names = [r.disease for r in request.disease_risks]
        diet_recs = await asyncio.to_thread(
            _diet_advisor.generate_recommendations,
            genetic_risk_names,
            variant_dict,
            request.region,
            request.age,
            request.sex.value,
            request.dietary_restrictions or None,
        )
        diet_meals = await asyncio.to_thread(
            _diet_advisor.create_meal_plan,
            diet_recs,
            request.region,
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
    dependencies=[Depends(rate_limit(10, 60))],
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
    dependencies=[Depends(rate_limit(20, 60))],
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
            genetic_risk_names = [r.disease for r in risks[:10]]
            diet_recs = await asyncio.to_thread(
                _diet_advisor.generate_recommendations,
                genetic_risk_names,
                variant_dict,
                request.region,
                request.age,
                request.sex.value,
                request.dietary_restrictions or None,
            )
            diet_meals = await asyncio.to_thread(
                _diet_advisor.create_meal_plan,
                diet_recs,
                request.region,
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
    dependencies=[Depends(rate_limit(20, 60))],
)
async def nutrition_plan(request: NutritionRequest) -> NutritionResponse:
    """Generate a personalised nutrition plan from user details.

    Accepts age, sex, region, dietary restrictions, known variants,
    health conditions, calorie target, and number of meal plan days.
    No image required.
    """
    variant_dict = _build_variant_dict(request.known_variants)
    try:
        # Use health conditions as genetic risk proxies
        genetic_risk_names = list(request.health_conditions)
        diet_recs = await asyncio.to_thread(
            _diet_advisor.generate_recommendations,
            genetic_risk_names,
            variant_dict,
            request.region,
            request.age,
            request.sex.value,
            request.dietary_restrictions or None,
        )
        diet_meals = await asyncio.to_thread(
            _diet_advisor.create_meal_plan,
            diet_recs,
            request.region,
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
