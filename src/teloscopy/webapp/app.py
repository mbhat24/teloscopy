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

from teloscopy.webapp.models import (
    AgentInfo,
    AgentStatusEnum,
    AgentSystemStatus,
    AnalysisResponse,
    DietPlanRequest,
    DietPlanResponse,
    DietRecommendation,
    DiseaseRisk,
    DiseaseRiskRequest,
    DiseaseRiskResponse,
    HealthResponse,
    JobStatus,
    JobStatusEnum,
    MealPlan,
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

# -- Templates & static files -----------------------------------------------

templates: Jinja2Templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ===================================================================== #
#  Helper / simulation functions                                         #
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


def _simulate_disease_risks(
    variants: list[str],
    telomere_length: float | None = None,
    age: int = 40,
) -> list[DiseaseRisk]:
    """Return mock disease-risk entries."""
    diseases: list[dict[str, Any]] = [
        {
            "disease": "Cardiovascular Disease",
            "factors": ["age", "telomere shortening", "APOE variants"],
            "recs": [
                "Increase omega-3 intake",
                "Regular aerobic exercise",
                "Monitor blood pressure",
            ],
        },
        {
            "disease": "Type 2 Diabetes",
            "factors": ["insulin resistance markers", "TCF7L2 variant"],
            "recs": [
                "Reduce refined carbohydrates",
                "Increase fibre intake",
                "Regular glucose monitoring",
            ],
        },
        {
            "disease": "Alzheimer's Disease",
            "factors": ["APOE-e4 allele", "telomere attrition"],
            "recs": [
                "Mediterranean diet",
                "Cognitive exercises",
                "Adequate sleep hygiene",
            ],
        },
        {
            "disease": "Colorectal Cancer",
            "factors": ["APC variant", "dietary factors"],
            "recs": [
                "High-fibre diet",
                "Regular screening colonoscopy",
                "Limit red meat consumption",
            ],
        },
        {
            "disease": "Osteoporosis",
            "factors": ["age", "VDR variant", "calcium metabolism"],
            "recs": [
                "Calcium and vitamin D supplementation",
                "Weight-bearing exercise",
                "Bone density screening",
            ],
        },
    ]
    risks: list[DiseaseRisk] = []
    for d in diseases:
        level: RiskLevel = random.choice(list(RiskLevel))
        prob: float = {
            RiskLevel.LOW: round(random.uniform(0.01, 0.20), 3),
            RiskLevel.MODERATE: round(random.uniform(0.20, 0.50), 3),
            RiskLevel.HIGH: round(random.uniform(0.50, 0.75), 3),
            RiskLevel.VERY_HIGH: round(random.uniform(0.75, 0.95), 3),
        }[level]
        risks.append(
            DiseaseRisk(
                disease=d["disease"],
                risk_level=level,
                probability=prob,
                contributing_factors=d["factors"],
                recommendations=d["recs"],
            )
        )
    return risks


def _simulate_diet_recommendation(
    restrictions: list[str] | None = None,
) -> DietRecommendation:
    """Return a mock diet recommendation."""
    return DietRecommendation(
        summary=(
            "Based on your telomere profile and genetic markers, we recommend "
            "an anti-inflammatory, nutrient-dense diet rich in antioxidants "
            "and omega-3 fatty acids to support telomere maintenance."
        ),
        key_nutrients=[
            "Omega-3 fatty acids",
            "Vitamin D",
            "Folate",
            "Zinc",
            "Vitamin C",
            "Selenium",
            "Polyphenols",
        ],
        foods_to_increase=[
            "Fatty fish (salmon, mackerel)",
            "Leafy greens (spinach, kale)",
            "Berries (blueberries, strawberries)",
            "Nuts and seeds (walnuts, flaxseed)",
            "Legumes (lentils, chickpeas)",
            "Whole grains (quinoa, oats)",
            "Green tea",
        ],
        foods_to_avoid=[
            "Processed meats",
            "Refined sugars",
            "Trans fats",
            "Excessive alcohol",
            "High-sodium processed foods",
        ],
        meal_plans=[
            MealPlan(
                day="Monday",
                breakfast="Overnight oats with blueberries, walnuts, and chia seeds",
                lunch="Grilled salmon salad with spinach, avocado, and olive oil dressing",
                dinner="Lentil soup with kale and whole-grain bread",
                snacks=["Green tea", "Mixed nuts", "Apple slices with almond butter"],
            ),
            MealPlan(
                day="Tuesday",
                breakfast="Spinach and mushroom omelette with whole-grain toast",
                lunch="Quinoa bowl with roasted vegetables, chickpeas, and tahini",
                dinner="Baked mackerel with sweet potato and steamed broccoli",
                snacks=["Greek yoghurt with berries", "Carrot sticks with hummus"],
            ),
            MealPlan(
                day="Wednesday",
                breakfast="Smoothie with kale, banana, flaxseed, and almond milk",
                lunch="Turkey and avocado wrap with mixed greens",
                dinner="Stir-fried tofu with brown rice and mixed vegetables",
                snacks=["Trail mix", "Orange slices"],
            ),
        ],
        calorie_target=2100,
    )


async def _run_full_analysis(job_id: str, profile: UserProfile) -> None:
    """Simulate a long-running analysis pipeline in the background."""
    job: JobStatus = _jobs[job_id]
    try:
        job.status = JobStatusEnum.RUNNING
        job.message = "Starting telomere image analysis..."
        job.progress_pct = 5.0
        job.updated_at = datetime.utcnow()

        # Phase 1 – image analysis
        await asyncio.sleep(1.5)
        telomere: TelomereResult = _simulate_telomere_analysis()
        job.progress_pct = 35.0
        job.message = "Telomere analysis complete. Assessing disease risk..."
        job.updated_at = datetime.utcnow()

        # Phase 2 – disease risk
        await asyncio.sleep(1.0)
        risks: list[DiseaseRisk] = _simulate_disease_risks(
            variants=profile.known_variants,
            telomere_length=telomere.mean_length,
            age=profile.age,
        )
        job.progress_pct = 65.0
        job.message = "Disease risk assessed. Generating diet plan..."
        job.updated_at = datetime.utcnow()

        # Phase 3 – diet plan
        await asyncio.sleep(1.0)
        diet: DietRecommendation = _simulate_diet_recommendation(
            restrictions=profile.dietary_restrictions,
        )
        job.progress_pct = 90.0
        job.message = "Compiling final report..."
        job.updated_at = datetime.utcnow()

        await asyncio.sleep(0.5)

        # Done
        result = AnalysisResponse(
            job_id=job_id,
            telomere_results=telomere,
            disease_risks=risks,
            diet_recommendations=diet,
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
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    """Serve the dedicated upload page (same template, scroll-to-upload)."""
    return templates.TemplateResponse("index.html", {"request": request, "scroll_to": "upload"})


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str) -> HTMLResponse:
    """Serve a results page for a specific job."""
    job: JobStatus | None = _jobs.get(job_id)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "job_id": job_id, "job": job, "scroll_to": "results"},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Serve the agent-monitoring dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


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
    asyncio.create_task(_run_full_analysis(job_id, profile))  # noqa: RUF006
    logger.info("Queued full analysis job %s", job_id)

    return job


# -- Disease risk (standalone) -----------------------------------------------


@app.post("/api/disease-risk", response_model=DiseaseRiskResponse)
async def disease_risk(request: DiseaseRiskRequest) -> DiseaseRiskResponse:
    """Compute disease-risk scores from variants and telomere data."""
    risks: list[DiseaseRisk] = _simulate_disease_risks(
        variants=request.known_variants,
        telomere_length=request.telomere_length,
        age=request.age,
    )
    overall: float = round(sum(r.probability for r in risks) / max(len(risks), 1), 3)
    return DiseaseRiskResponse(risks=risks, overall_risk_score=min(overall, 1.0))


# -- Diet plan (standalone) --------------------------------------------------


@app.post("/api/diet-plan", response_model=DietPlanResponse)
async def diet_plan(request: DietPlanRequest) -> DietPlanResponse:
    """Generate a personalised diet plan."""
    rec: DietRecommendation = _simulate_diet_recommendation(
        restrictions=request.dietary_restrictions,
    )
    return DietPlanResponse(recommendation=rec)
