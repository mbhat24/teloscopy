"""Pydantic models for the Teloscopy web API.

Defines request/response schemas for every endpoint including user
profiles, analysis pipelines, disease-risk scoring, and diet planning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Sex(StrEnum):
    """Biological sex options."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class JobStatusEnum(StrEnum):
    """Possible states for an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(StrEnum):
    """Categorical disease-risk level."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AgentStatusEnum(StrEnum):
    """Status of an individual agent in the multi-agent system."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


# ---------------------------------------------------------------------------
# User Profile
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Demographic and health information supplied by the user."""

    age: int = Field(..., ge=1, le=150, description="Age in years")
    sex: Sex = Field(..., description="Biological sex")
    region: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Geographic region (e.g. 'East Asia', 'Northern Europe')",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions such as 'vegetarian', 'gluten-free'",
    )
    known_variants: list[str] = Field(
        default_factory=list,
        description="Known genetic variants (rsIDs or gene names)",
    )


# ---------------------------------------------------------------------------
# Analysis (full pipeline)
# ---------------------------------------------------------------------------


class AnalysisRequest(BaseModel):
    """Request body for the full analysis endpoint.

    The microscopy image is uploaded as a multipart file, so it is *not*
    part of this schema.  This model captures the JSON metadata sent
    alongside the file.
    """

    user_profile: UserProfile


class TelomereResult(BaseModel):
    """Results from telomere image analysis."""

    mean_length: float = Field(..., description="Mean telomere length in kb")
    std_dev: float = Field(..., description="Standard deviation of length")
    t_s_ratio: float = Field(..., description="Telomere-to-single-copy gene ratio")
    biological_age_estimate: int = Field(..., description="Estimated biological age")
    overlay_image_url: str | None = Field(None, description="URL of the annotated overlay image")
    raw_measurements: list[float] = Field(
        default_factory=list,
        description="Individual telomere length measurements",
    )


class DiseaseRisk(BaseModel):
    """A single disease-risk entry."""

    disease: str
    risk_level: RiskLevel
    probability: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class MealPlan(BaseModel):
    """One day's meal plan."""

    day: str
    breakfast: str
    lunch: str
    dinner: str
    snacks: list[str] = Field(default_factory=list)


class DietRecommendation(BaseModel):
    """Dietary recommendations tied to genomic / telomere findings."""

    summary: str
    key_nutrients: list[str] = Field(default_factory=list)
    foods_to_increase: list[str] = Field(default_factory=list)
    foods_to_avoid: list[str] = Field(default_factory=list)
    meal_plans: list[MealPlan] = Field(default_factory=list)
    calorie_target: int | None = None


class _AnalysisResponseBase(BaseModel):
    """Deprecated — replaced by AnalysisResponse with facial_analysis field."""

    pass


# ---------------------------------------------------------------------------
# Disease Risk (standalone)
# ---------------------------------------------------------------------------


class DiseaseRiskRequest(BaseModel):
    """Standalone disease-risk request (no image required)."""

    known_variants: list[str] = Field(
        default_factory=list,
        description="Genetic variants (rsIDs or gene names)",
    )
    telomere_length: float | None = Field(None, description="Mean telomere length in kb")
    age: int = Field(..., ge=1, le=150)
    sex: Sex = Field(...)
    region: str = Field(..., min_length=1, max_length=128)


class DiseaseRiskResponse(BaseModel):
    """Response from the standalone disease-risk endpoint."""

    risks: list[DiseaseRisk] = Field(default_factory=list)
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite risk score")
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Diet Plan (standalone)
# ---------------------------------------------------------------------------


class DietPlanRequest(BaseModel):
    """Standalone diet plan request."""

    age: int = Field(..., ge=1, le=150)
    sex: Sex = Field(...)
    region: str = Field(..., min_length=1, max_length=128)
    dietary_restrictions: list[str] = Field(default_factory=list)
    known_variants: list[str] = Field(default_factory=list)
    telomere_length: float | None = Field(None, description="Mean telomere length in kb")
    disease_risks: list[DiseaseRisk] = Field(default_factory=list)


class DietPlanResponse(BaseModel):
    """Response from the standalone diet plan endpoint."""

    recommendation: DietRecommendation
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Job Status
# ---------------------------------------------------------------------------


class JobStatus(BaseModel):
    """Status of an asynchronous analysis job."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatusEnum = Field(default=JobStatusEnum.PENDING)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = Field(default="Job created")
    result: AnalysisResponse | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Agent Status
# ---------------------------------------------------------------------------


class AgentInfo(BaseModel):
    """Status information for a single agent."""

    name: str
    status: AgentStatusEnum = AgentStatusEnum.IDLE
    last_active: datetime | None = None
    tasks_completed: int = 0
    current_task: str | None = None


class AgentSystemStatus(BaseModel):
    """Aggregated status of the multi-agent system."""

    agents: list[AgentInfo] = Field(default_factory=list)
    total_analyses: int = 0
    active_jobs: int = 0
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Upload response
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Response returned after a successful image upload."""

    job_id: str
    filename: str
    message: str = "Image uploaded successfully"


# ---------------------------------------------------------------------------
# Facial Analysis
# ---------------------------------------------------------------------------


class FacialMeasurementsResponse(BaseModel):
    """Facial feature measurements extracted from photograph."""

    face_width: float = 0.0
    face_height: float = 0.0
    face_ratio: float = 0.0
    skin_brightness: float = 0.0
    skin_uniformity: float = 0.0
    wrinkle_score: float = 0.0
    symmetry_score: float = 0.0
    dark_circle_score: float = 0.0
    texture_roughness: float = 0.0
    uv_damage_score: float = 0.0


class AncestryEstimateResponse(BaseModel):
    """Estimated ancestral composition from facial features."""

    european: float = 0.0
    east_asian: float = 0.0
    south_asian: float = 0.0
    african: float = 0.0
    middle_eastern: float = 0.0
    latin_american: float = 0.0
    confidence: float = 0.0


class PredictedVariantResponse(BaseModel):
    """A predicted genetic variant from facial analysis."""

    rsid: str
    gene: str
    predicted_genotype: str
    confidence: float
    basis: str


class FacialAnalysisResult(BaseModel):
    """Complete facial-genomic analysis result."""

    image_type: str = "face_photo"
    estimated_biological_age: int = 0
    estimated_telomere_length_kb: float = 0.0
    telomere_percentile: int = 50
    skin_health_score: float = 0.0
    oxidative_stress_score: float = 0.0
    predicted_eye_colour: str = "unknown"
    predicted_hair_colour: str = "unknown"
    predicted_skin_type: str = "unknown"
    measurements: FacialMeasurementsResponse = Field(default_factory=FacialMeasurementsResponse)
    ancestry: AncestryEstimateResponse = Field(default_factory=AncestryEstimateResponse)
    predicted_variants: list[PredictedVariantResponse] = Field(default_factory=list)
    analysis_warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Full analysis response returned once all agents have completed."""

    job_id: str
    image_type: str = Field(default="fish_microscopy", description="fish_microscopy or face_photo")
    telomere_results: TelomereResult
    disease_risks: list[DiseaseRisk] = Field(default_factory=list)
    diet_recommendations: DietRecommendation
    facial_analysis: FacialAnalysisResult | None = None
    report_url: str | None = None
    csv_url: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
