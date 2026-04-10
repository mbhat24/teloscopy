"""Pydantic models for the Teloscopy web API.

Defines request/response schemas for every endpoint including user
profiles, analysis pipelines, disease-risk scoring, and diet planning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
import enum
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, enum.Enum):
        """Backport of StrEnum for Python < 3.11."""

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
    country: str | None = Field(
        None,
        max_length=128,
        description="Country (e.g. 'India', 'USA', 'Japan'). Enables country-specific diet plans.",
    )
    state: str | None = Field(
        None,
        max_length=128,
        description="State or province (e.g. 'Kerala', 'California'). Enables state-level diet personalisation.",
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "known_variants": ["rs429358:CT", "rs7412:CC", "rs1801133:AG"],
                    "telomere_length": 6.8,
                    "age": 45,
                    "sex": "female",
                    "region": "Northern Europe",
                }
            ]
        }
    }


class DiseaseRiskResponse(BaseModel):
    """Response from the standalone disease-risk endpoint."""

    risks: list[DiseaseRisk] = Field(default_factory=list)
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite risk score")
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = Field(
        default="For research/educational use only — not clinical advice. Results must NOT be used for medical decisions.",
        description="Legal and scientific disclaimer",
    )


# ---------------------------------------------------------------------------
# Diet Plan (standalone)
# ---------------------------------------------------------------------------


class DietPlanRequest(BaseModel):
    """Standalone diet plan request."""

    age: int = Field(..., ge=1, le=150)
    sex: Sex = Field(...)
    region: str = Field(..., min_length=1, max_length=128)
    country: str | None = Field(None, max_length=128, description="Country for regional diet")
    state: str | None = Field(None, max_length=128, description="State/province for local diet")
    dietary_restrictions: list[str] = Field(default_factory=list)
    known_variants: list[str] = Field(default_factory=list)
    telomere_length: float | None = Field(None, description="Mean telomere length in kb")
    disease_risks: list[DiseaseRisk] = Field(default_factory=list)
    meal_plan_days: int = Field(7, ge=1, le=30)
    calorie_target: int = Field(2000, ge=800, le=5000)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 34,
                    "sex": "male",
                    "region": "East Asia",
                    "dietary_restrictions": ["vegetarian", "gluten-free"],
                    "known_variants": ["rs1801133:AG"],
                    "telomere_length": 7.2,
                    "disease_risks": [],
                    "meal_plan_days": 7,
                    "calorie_target": 2200,
                }
            ]
        }
    }


class DietPlanResponse(BaseModel):
    """Response from the standalone diet plan endpoint."""

    recommendation: DietRecommendation
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = Field(
        default="For research/educational use only — not clinical advice. Results must NOT be used for medical decisions.",
        description="Legal and scientific disclaimer",
    )


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
    risk_allele: str = ""
    ref_allele: str = ""


class ReconstructedSequenceResponse(BaseModel):
    """A single reconstructed DNA sequence fragment around a predicted SNP."""

    rsid: str
    gene: str
    chromosome: str
    position: int
    ref_allele: str
    predicted_allele_1: str
    predicted_allele_2: str
    flanking_5prime: str
    flanking_3prime: str
    confidence: float


class ReconstructedDNAResponse(BaseModel):
    """Reconstructed partial genome from predicted variants."""

    sequences: list[ReconstructedSequenceResponse] = Field(default_factory=list)
    total_variants: int = 0
    genome_build: str = "GRCh38/hg38"
    fasta: str = ""
    disclaimer: str = (
        "RECONSTRUCTED SEQUENCE — This is a statistical reconstruction based "
        "on facial-genomic predictions, NOT actual DNA sequencing. Predicted "
        "genotypes are derived from population-level allele frequencies and "
        "phenotypic correlations. Do not use for clinical decisions."
    )


class PharmacogenomicPredictionResponse(BaseModel):
    """A predicted pharmacogenomic interaction from facial-genomic analysis."""

    gene: str
    rsid: str
    predicted_phenotype: str
    confidence: float
    affected_drugs: list[str] = []
    clinical_recommendation: str = ""
    basis: str = ""


class FacialHealthScreeningResponse(BaseModel):
    """Health screening indicators derived from facial analysis."""

    estimated_bmi_category: str = "Unknown"
    bmi_confidence: float = 0.0
    anemia_risk_score: float = 0.0
    cardiovascular_risk_indicators: list[str] = []
    thyroid_indicators: list[str] = []
    fatigue_stress_score: float = 0.0
    hydration_score: float = 50.0


class DermatologicalAnalysisResponse(BaseModel):
    """Dermatological risk indicators from facial analysis."""

    rosacea_risk_score: float = 0.0
    melasma_risk_score: float = 0.0
    photo_aging_gap: int = 0
    acne_severity_score: float = 0.0
    skin_cancer_risk_factors: list[str] = []
    pigmentation_disorder_risk: float = 0.0
    moisture_barrier_score: float = 50.0


class ConditionScreeningResponse(BaseModel):
    """A single condition screening result from facial-genomic analysis."""

    condition: str
    risk_score: float = 0.0
    facial_markers: list[str] = []
    confidence: float = 0.0
    recommendation: str = ""


class AncestryDerivedPredictionsResponse(BaseModel):
    """Ancestry-derived metabolic and haplogroup predictions."""

    predicted_mtdna_haplogroup: str = "Unknown"
    haplogroup_confidence: float = 0.0
    lactose_tolerance_probability: float = 0.5
    alcohol_flush_probability: float = 0.0
    caffeine_sensitivity: str = "Unknown"
    bitter_taste_sensitivity: str = "Unknown"
    population_specific_risks: list[str] = []


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
    reconstructed_dna: ReconstructedDNAResponse | None = None
    pharmacogenomic_predictions: list[PharmacogenomicPredictionResponse] = []
    health_screening: FacialHealthScreeningResponse | None = None
    dermatological_analysis: DermatologicalAnalysisResponse | None = None
    condition_screenings: list[ConditionScreeningResponse] = []
    ancestry_derived: AncestryDerivedPredictionsResponse | None = None
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
    disclaimer: str = Field(
        default="For research/educational use only — not clinical advice. Results must NOT be used for medical decisions.",
        description="Legal and scientific disclaimer",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------


class ImageValidationResponse(BaseModel):
    """Result of image content validation before analysis."""

    valid: bool = True
    image_type: str = "unknown"
    width: int = 0
    height: int = 0
    channels: int = 0
    file_size_bytes: int = 0
    format_detected: str = "unknown"
    face_detected: bool = False
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Profile-only analysis (no image required)
# ---------------------------------------------------------------------------


class ProfileAnalysisRequest(BaseModel):
    """Request body for analysis using only user-provided details."""

    age: int = Field(..., ge=1, le=150)
    sex: Sex = Field(...)
    region: str = Field(..., min_length=1, max_length=128)
    country: str | None = Field(None, max_length=128, description="Country for regional diet")
    state: str | None = Field(None, max_length=128, description="State/province for local diet")
    dietary_restrictions: list[str] = Field(default_factory=list)
    known_variants: list[str] = Field(default_factory=list)
    telomere_length_kb: float | None = Field(
        None, description="Self-reported telomere length in kb (optional)"
    )
    include_nutrition: bool = Field(True, description="Include nutrition recommendations")
    include_disease_risk: bool = Field(True, description="Include disease risk assessment")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 52,
                    "sex": "female",
                    "region": "Southern Europe",
                    "dietary_restrictions": ["lactose-free"],
                    "known_variants": ["rs429358:CT", "rs7412:CC"],
                    "telomere_length_kb": 5.9,
                    "include_nutrition": True,
                    "include_disease_risk": True,
                }
            ]
        }
    }


class ProfileAnalysisResponse(BaseModel):
    """Response for profile-only analysis."""

    disease_risks: list[DiseaseRisk] = Field(default_factory=list)
    diet_recommendations: DietRecommendation | None = None
    overall_risk_score: float = Field(0.0, ge=0.0, le=1.0)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = Field(
        default="For research/educational use only — not clinical advice. Results must NOT be used for medical decisions.",
        description="Legal and scientific disclaimer",
    )


# ---------------------------------------------------------------------------
# Standalone nutrition request
# ---------------------------------------------------------------------------


class NutritionRequest(BaseModel):
    """Standalone nutrition/diet plan request with full user details."""

    age: int = Field(..., ge=1, le=150)
    sex: Sex = Field(...)
    region: str = Field(..., min_length=1, max_length=128)
    country: str | None = Field(None, max_length=128, description="Country for regional diet")
    state: str | None = Field(None, max_length=128, description="State/province for local diet")
    dietary_restrictions: list[str] = Field(default_factory=list)
    known_variants: list[str] = Field(default_factory=list)
    health_conditions: list[str] = Field(
        default_factory=list,
        description="Known health conditions (e.g. 'diabetes', 'hypertension')",
    )
    calorie_target: int = Field(2000, ge=800, le=5000)
    meal_plan_days: int = Field(7, ge=1, le=30)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 29,
                    "sex": "male",
                    "region": "South Asia",
                    "dietary_restrictions": ["vegetarian"],
                    "known_variants": ["rs1801133:AG", "rs4988235:CT"],
                    "health_conditions": ["hypertension", "prediabetes"],
                    "calorie_target": 1800,
                    "meal_plan_days": 14,
                }
            ]
        }
    }


class NutritionResponse(BaseModel):
    """Response for standalone nutrition endpoint."""

    recommendation: DietRecommendation
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = Field(
        default="For research/educational use only — not clinical advice. Results must NOT be used for medical decisions.",
        description="Legal and scientific disclaimer",
    )


# ---------------------------------------------------------------------------
# Health Checkup — blood tests, urine tests, abdomen scan
# ---------------------------------------------------------------------------


class BloodTestPanel(BaseModel):
    """Structured blood test results.

    All values are optional — users submit whatever parameters they have.
    The backend interprets each provided value against age/sex ranges.
    """

    # CBC (Complete Blood Count)
    hemoglobin: float | None = Field(None, description="g/dL")
    rbc_count: float | None = Field(None, description="million cells/mcL")
    wbc_count: float | None = Field(None, description="thousand cells/mcL")
    platelet_count: float | None = Field(None, description="thousand/mcL")
    hematocrit: float | None = Field(None, description="%")
    mcv: float | None = Field(None, description="fL")
    mch: float | None = Field(None, description="pg")
    mchc: float | None = Field(None, description="g/dL")
    rdw: float | None = Field(None, description="%")
    neutrophils: float | None = Field(None, description="%")
    lymphocytes: float | None = Field(None, description="%")
    monocytes: float | None = Field(None, description="%")
    eosinophils: float | None = Field(None, description="%")
    basophils: float | None = Field(None, description="%")

    # Lipid Panel
    total_cholesterol: float | None = Field(None, description="mg/dL")
    ldl_cholesterol: float | None = Field(None, description="mg/dL")
    hdl_cholesterol: float | None = Field(None, description="mg/dL")
    triglycerides: float | None = Field(None, description="mg/dL")
    vldl: float | None = Field(None, description="mg/dL")
    total_cholesterol_hdl_ratio: float | None = Field(None, description="ratio")

    # Liver Function (LFT)
    sgot_ast: float | None = Field(None, description="U/L")
    sgpt_alt: float | None = Field(None, description="U/L")
    alkaline_phosphatase: float | None = Field(None, description="U/L")
    total_bilirubin: float | None = Field(None, description="mg/dL")
    direct_bilirubin: float | None = Field(None, description="mg/dL")
    ggt: float | None = Field(None, description="U/L")
    total_protein: float | None = Field(None, description="g/dL")
    albumin: float | None = Field(None, description="g/dL")
    globulin: float | None = Field(None, description="g/dL")
    ag_ratio: float | None = Field(None, description="ratio")

    # Kidney Function (KFT)
    blood_urea: float | None = Field(None, description="mg/dL")
    serum_creatinine: float | None = Field(None, description="mg/dL")
    uric_acid: float | None = Field(None, description="mg/dL")
    bun: float | None = Field(None, description="mg/dL")
    egfr: float | None = Field(None, description="mL/min/1.73m²")

    # Diabetes Panel
    fasting_glucose: float | None = Field(None, description="mg/dL")
    hba1c: float | None = Field(None, description="%")
    postprandial_glucose: float | None = Field(None, description="mg/dL")
    fasting_insulin: float | None = Field(None, description="µIU/mL")

    # Thyroid
    tsh: float | None = Field(None, description="µIU/mL")
    t3: float | None = Field(None, description="ng/dL")
    t4: float | None = Field(None, description="µg/dL")
    free_t3: float | None = Field(None, description="pg/mL")
    free_t4: float | None = Field(None, description="ng/dL")

    # Vitamins
    vitamin_d: float | None = Field(None, description="ng/mL")
    vitamin_b12: float | None = Field(None, description="pg/mL")
    folate: float | None = Field(None, description="ng/mL")

    # Minerals / Electrolytes
    iron: float | None = Field(None, description="µg/dL")
    ferritin: float | None = Field(None, description="ng/mL")
    tibc: float | None = Field(None, description="µg/dL")
    transferrin_saturation: float | None = Field(None, description="%")
    calcium: float | None = Field(None, description="mg/dL")
    phosphorus: float | None = Field(None, description="mg/dL")
    magnesium: float | None = Field(None, description="mg/dL")
    sodium: float | None = Field(None, description="mEq/L")
    potassium: float | None = Field(None, description="mEq/L")
    chloride: float | None = Field(None, description="mEq/L")

    # Inflammation
    crp: float | None = Field(None, description="mg/L")
    esr: float | None = Field(None, description="mm/hr")
    homocysteine: float | None = Field(None, description="µmol/L")


class UrineTestPanel(BaseModel):
    """Structured urine test results."""

    ph: float | None = Field(None, description="pH units")
    specific_gravity: float | None = Field(None, description="ratio")
    protein: float | None = Field(None, description="mg/dL (0=negative)")
    glucose: float | None = Field(None, description="mg/dL (0=negative)")
    ketones: float | None = Field(None, description="mg/dL (0=negative)")
    bilirubin: float | None = Field(None, description="mg/dL (0=negative)")
    urobilinogen: float | None = Field(None, description="mg/dL")
    blood: float | None = Field(None, description="RBC/HPF (0=negative)")
    nitrites: float | None = Field(None, description="0=negative, 1=positive")
    leukocytes: float | None = Field(None, description="WBC/HPF (0=negative)")
    rbc_urine: float | None = Field(None, description="cells/HPF")
    wbc_urine: float | None = Field(None, description="cells/HPF")
    epithelial_cells: float | None = Field(None, description="cells/HPF")


class HealthCheckupRequest(BaseModel):
    """Full health checkup request with lab data, scan, and profile."""

    # User profile
    age: int = Field(..., ge=1, le=150)
    sex: Sex = Field(...)
    region: str = Field(..., min_length=1, max_length=128)
    country: str | None = Field(None, max_length=128)
    state: str | None = Field(None, max_length=128)
    dietary_restrictions: list[str] = Field(default_factory=list)
    known_variants: list[str] = Field(default_factory=list)

    # Lab data
    blood_tests: BloodTestPanel | None = None
    urine_tests: UrineTestPanel | None = None
    abdomen_scan_notes: str | None = Field(
        None,
        max_length=5000,
        description="Free-text abdomen scan / ultrasound findings from the doctor's report",
    )

    # Diet preferences
    calorie_target: int = Field(2000, ge=800, le=5000)
    meal_plan_days: int = Field(7, ge=1, le=30)
    health_conditions: list[str] = Field(
        default_factory=list,
        description="User-reported health conditions (in addition to auto-detected from labs)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 42,
                    "sex": "male",
                    "region": "South Asia",
                    "country": "India",
                    "state": "Karnataka",
                    "blood_tests": {
                        "hemoglobin": 13.2,
                        "fasting_glucose": 118,
                        "hba1c": 6.1,
                        "total_cholesterol": 232,
                        "ldl_cholesterol": 155,
                        "hdl_cholesterol": 38,
                        "triglycerides": 198,
                        "vitamin_d": 14.5,
                        "vitamin_b12": 180,
                        "sgpt_alt": 52,
                        "sgot_ast": 48,
                        "crp": 4.2,
                        "tsh": 5.8,
                        "uric_acid": 8.1,
                    },
                    "urine_tests": {
                        "protein": 15,
                        "glucose": 30,
                    },
                    "abdomen_scan_notes": "Mild hepatomegaly with grade 1 fatty liver. Both kidneys normal.",
                    "calorie_target": 1800,
                    "meal_plan_days": 7,
                    "dietary_restrictions": ["vegetarian"],
                }
            ]
        }
    }


class LabResultResponse(BaseModel):
    """Single lab result in the response."""

    parameter: str
    display_name: str
    value: float
    unit: str
    status: str  # "low", "normal", "high", "critical_low", "critical_high"
    reference_low: float
    reference_high: float
    category: str


class HealthFindingResponse(BaseModel):
    """A detected health finding in the response."""

    condition: str
    display_name: str
    severity: str
    evidence: list[str]
    dietary_impact: str
    nutrients_to_increase: list[str] = Field(default_factory=list)
    nutrients_to_decrease: list[str] = Field(default_factory=list)
    foods_to_increase: list[str] = Field(default_factory=list)
    foods_to_avoid: list[str] = Field(default_factory=list)


class AbdomenFindingResponse(BaseModel):
    """Abdomen scan finding in the response."""

    organ: str
    finding: str
    severity: str
    dietary_impact: str
    foods_to_avoid: list[str] = Field(default_factory=list)
    foods_to_increase: list[str] = Field(default_factory=list)


class HealthCheckupResponse(BaseModel):
    """Complete health checkup analysis with personalised diet plan."""

    # Lab interpretation
    lab_results: list[LabResultResponse] = Field(default_factory=list)
    abnormal_count: int = 0
    total_tested: int = 0

    # Health findings
    findings: list[HealthFindingResponse] = Field(default_factory=list)
    abdomen_findings: list[AbdomenFindingResponse] = Field(default_factory=list)
    detected_conditions: list[str] = Field(default_factory=list)

    # Health score
    overall_health_score: float = Field(0.0, ge=0.0, le=100.0)
    health_score_breakdown: dict[str, float] = Field(default_factory=dict)

    # Diet plan (reuse existing DietRecommendation)
    diet_recommendation: DietRecommendation | None = None
    dietary_modifications: list[str] = Field(default_factory=list)
    calorie_adjustment: int = 0

    # Metadata
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = Field(
        default="For research/educational use only — not clinical or diagnostic advice. "
        "Lab interpretation is approximate. Always consult your physician for medical decisions.",
    )
