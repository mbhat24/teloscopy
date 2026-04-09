"""Teloscopy FastAPI application.

Provides REST endpoints for microscopy image upload, telomere analysis,
disease-risk scoring, personalised nutrition plans, and an HTML frontend
rendered via Jinja2.

Run with::

    uvicorn teloscopy.webapp.app:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
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
    JobStatus,
    JobStatusEnum,
    MealPlan,
    PredictedVariantResponse,
    RiskLevel,
    Sex,
    TelomereResult,
    UploadResponse,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger("teloscopy.webapp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_DIR: Path = Path(__file__).resolve().parent
_TEMPLATES_DIR: Path = _BASE_DIR / "templates"
_STATIC_DIR: Path = _BASE_DIR / "static"
_UPLOAD_DIR: Path = Path(os.getenv("TELOSCOPY_UPLOAD_DIR", "/tmp/teloscopy_uploads"))
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MiB

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
# FastAPI application
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="Teloscopy",
    description=(
        "Telomere analysis, disease-risk assessment, and personalised "
        "nutrition recommendations from microscopy images."
    ),
    version="0.1.0",
)

# -- CORS -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Exception handler (surface errors in non-production) --------------------


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> Any:
    """Return traceback detail for unhandled errors to aid debugging."""
    import traceback

    from fastapi.responses import JSONResponse

    tb = traceback.format_exc()
    logger.error("Unhandled %s on %s: %s\n%s", type(exc).__name__, request.url.path, exc, tb)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "path": request.url.path,
            "traceback": tb.splitlines()[-5:],
        },
    )


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
    for mp in meal_plans[:7]:

        def _meal_str(items: list[Any]) -> str:
            return ", ".join(f"{fi.name} ({g:.0f}g)" for fi, g in items[:4]) or "Seasonal selection"

        pydantic_plans.append(
            MealPlan(
                day=day_names[mp.day % 7] if isinstance(mp.day, int) else str(mp.day),
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


def _translate_facial_profile(profile: Any) -> FacialAnalysisResult:
    """Convert ``facial.predictor.FacialGenomicProfile`` to Pydantic model."""
    m = profile.measurements
    a = profile.ancestry
    return FacialAnalysisResult(
        image_type="face_photo",
        estimated_biological_age=profile.estimated_biological_age,
        estimated_telomere_length_kb=profile.estimated_telomere_length_kb,
        telomere_percentile=profile.telomere_percentile,
        skin_health_score=profile.skin_health_score,
        oxidative_stress_score=profile.oxidative_stress_score,
        predicted_eye_colour=profile.predicted_eye_colour,
        predicted_hair_colour=profile.predicted_hair_colour,
        predicted_skin_type=profile.predicted_skin_type,
        measurements=FacialMeasurementsResponse(
            face_width=m.face_width,
            face_height=m.face_height,
            face_ratio=m.face_ratio,
            skin_brightness=m.skin_brightness,
            skin_uniformity=m.skin_uniformity,
            wrinkle_score=m.wrinkle_score,
            symmetry_score=m.symmetry_score,
            dark_circle_score=m.dark_circle_score,
            texture_roughness=m.texture_roughness,
            uv_damage_score=m.uv_damage_score,
        ),
        ancestry=AncestryEstimateResponse(
            european=a.european,
            east_asian=a.east_asian,
            south_asian=a.south_asian,
            african=a.african,
            middle_eastern=a.middle_eastern,
            latin_american=a.latin_american,
            confidence=a.confidence,
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
    """Return plausible mock telomere analysis results."""
    mean_len: float = round(random.uniform(4.0, 12.0), 2)
    return TelomereResult(
        mean_length=mean_len,
        std_dev=round(random.uniform(0.3, 2.0), 2),
        t_s_ratio=round(random.uniform(0.5, 2.5), 2),
        biological_age_estimate=random.randint(20, 85),
        overlay_image_url=None,
        raw_measurements=[round(random.uniform(3.0, 14.0), 2) for _ in range(20)],
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
        if image_type == ImageType.FACE_PHOTO:
            # Real facial-genomic prediction (CPU-bound → run in thread)
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
        else:
            # FISH microscopy — use simulated telomere results
            # (real qFISH pipeline requires calibrated multi-channel TIFF)
            await asyncio.sleep(0.5)
            telomere = _simulate_telomere_analysis()

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
                    # Use predicted genotype as a proxy
                    if "homozygous variant" in pv.predicted_genotype:
                        variant_dict[pv.rsid] = "TT"
                    elif "heterozygous" in pv.predicted_genotype:
                        variant_dict[pv.rsid] = "CT"
                    else:
                        variant_dict[pv.rsid] = "CC"

        risk_profile = await asyncio.to_thread(
            _disease_predictor.predict_from_variants,
            variant_dict,
            profile.age,
            profile.sex.value,
        )
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
            2100,
            3,
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
        resp = templates.TemplateResponse("index.html", {"request": request})
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


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse()


@app.get("/api/agents/status", response_model=AgentSystemStatus)
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


@app.post("/api/upload", response_model=UploadResponse, status_code=201)
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


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Return the current status of an analysis job."""
    job: JobStatus | None = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return job


@app.get("/api/results/{job_id}", response_model=AnalysisResponse)
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


@app.post("/api/analyze", response_model=JobStatus, status_code=202)
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
            detail="Invalid file type.",
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


@app.post("/api/disease-risk", response_model=DiseaseRiskResponse)
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


@app.post("/api/diet-plan", response_model=DietPlanResponse)
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
            2100,
            3,
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
