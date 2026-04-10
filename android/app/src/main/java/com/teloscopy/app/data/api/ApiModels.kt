package com.teloscopy.app.data.api

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

// ---------------------------------------------------------------------------
// Telomere Results
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class TelomereResult(
    @Json(name = "mean_length") val meanLength: Double,
    @Json(name = "std_dev") val stdDev: Double,
    @Json(name = "t_s_ratio") val tsRatio: Double,
    @Json(name = "biological_age_estimate") val biologicalAgeEstimate: Int,
    @Json(name = "overlay_image_url") val overlayImageUrl: String? = null,
    @Json(name = "raw_measurements") val rawMeasurements: List<Double> = emptyList()
)

// ---------------------------------------------------------------------------
// Disease Risk
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class DiseaseRisk(
    @Json(name = "disease") val disease: String,
    @Json(name = "risk_level") val riskLevel: String,
    @Json(name = "probability") val probability: Double,
    @Json(name = "contributing_factors") val contributingFactors: List<String> = emptyList(),
    @Json(name = "recommendations") val recommendations: List<String> = emptyList()
)

// ---------------------------------------------------------------------------
// Diet / Nutrition
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class MealPlan(
    @Json(name = "day") val day: String,
    @Json(name = "breakfast") val breakfast: String,
    @Json(name = "lunch") val lunch: String,
    @Json(name = "dinner") val dinner: String,
    @Json(name = "snacks") val snacks: List<String> = emptyList()
)

@JsonClass(generateAdapter = true)
data class DietRecommendation(
    @Json(name = "summary") val summary: String,
    @Json(name = "key_nutrients") val keyNutrients: List<String> = emptyList(),
    @Json(name = "foods_to_increase") val foodsToIncrease: List<String> = emptyList(),
    @Json(name = "foods_to_avoid") val foodsToAvoid: List<String> = emptyList(),
    @Json(name = "meal_plans") val mealPlans: List<MealPlan> = emptyList(),
    @Json(name = "calorie_target") val calorieTarget: Int? = null
)

// ---------------------------------------------------------------------------
// Facial Analysis
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class FacialMeasurementsResponse(
    @Json(name = "face_width") val faceWidth: Double = 0.0,
    @Json(name = "face_height") val faceHeight: Double = 0.0,
    @Json(name = "face_ratio") val faceRatio: Double = 0.0,
    @Json(name = "skin_brightness") val skinBrightness: Double = 0.0,
    @Json(name = "skin_uniformity") val skinUniformity: Double = 0.0,
    @Json(name = "wrinkle_score") val wrinkleScore: Double = 0.0,
    @Json(name = "symmetry_score") val symmetryScore: Double = 0.0,
    @Json(name = "dark_circle_score") val darkCircleScore: Double = 0.0,
    @Json(name = "texture_roughness") val textureRoughness: Double = 0.0,
    @Json(name = "uv_damage_score") val uvDamageScore: Double = 0.0
)

@JsonClass(generateAdapter = true)
data class AncestryEstimateResponse(
    @Json(name = "european") val european: Double = 0.0,
    @Json(name = "east_asian") val eastAsian: Double = 0.0,
    @Json(name = "south_asian") val southAsian: Double = 0.0,
    @Json(name = "african") val african: Double = 0.0,
    @Json(name = "middle_eastern") val middleEastern: Double = 0.0,
    @Json(name = "latin_american") val latinAmerican: Double = 0.0,
    @Json(name = "confidence") val confidence: Double = 0.0
)

@JsonClass(generateAdapter = true)
data class PredictedVariantResponse(
    @Json(name = "rsid") val rsid: String,
    @Json(name = "gene") val gene: String,
    @Json(name = "predicted_genotype") val predictedGenotype: String,
    @Json(name = "confidence") val confidence: Double,
    @Json(name = "basis") val basis: String,
    @Json(name = "risk_allele") val riskAllele: String = "",
    @Json(name = "ref_allele") val refAllele: String = ""
)

@JsonClass(generateAdapter = true)
data class ReconstructedSequenceResponse(
    @Json(name = "rsid") val rsid: String,
    @Json(name = "gene") val gene: String,
    @Json(name = "chromosome") val chromosome: String,
    @Json(name = "position") val position: Long,
    @Json(name = "ref_allele") val refAllele: String,
    @Json(name = "predicted_allele_1") val predictedAllele1: String,
    @Json(name = "predicted_allele_2") val predictedAllele2: String,
    @Json(name = "flanking_5prime") val flanking5prime: String = "",
    @Json(name = "flanking_3prime") val flanking3prime: String = "",
    @Json(name = "confidence") val confidence: Double = 0.0
)

@JsonClass(generateAdapter = true)
data class ReconstructedDNAResponse(
    @Json(name = "sequences") val sequences: List<ReconstructedSequenceResponse> = emptyList(),
    @Json(name = "total_variants") val totalVariants: Int = 0,
    @Json(name = "genome_build") val genomeBuild: String = "GRCh38/hg38",
    @Json(name = "fasta") val fasta: String = "",
    @Json(name = "disclaimer") val disclaimer: String = ""
)

@JsonClass(generateAdapter = true)
data class PharmacogenomicPredictionResponse(
    @Json(name = "gene") val gene: String = "",
    @Json(name = "rsid") val rsid: String = "",
    @Json(name = "predicted_phenotype") val predictedPhenotype: String = "",
    @Json(name = "confidence") val confidence: Double = 0.0,
    @Json(name = "affected_drugs") val affectedDrugs: List<String> = emptyList(),
    @Json(name = "clinical_recommendation") val clinicalRecommendation: String = "",
    @Json(name = "basis") val basis: String = ""
)

@JsonClass(generateAdapter = true)
data class FacialHealthScreeningResponse(
    @Json(name = "estimated_bmi_category") val estimatedBmiCategory: String = "Unknown",
    @Json(name = "bmi_confidence") val bmiConfidence: Double = 0.0,
    @Json(name = "anemia_risk_score") val anemiaRiskScore: Double = 0.0,
    @Json(name = "cardiovascular_risk_indicators") val cardiovascularRiskIndicators: List<String> = emptyList(),
    @Json(name = "thyroid_indicators") val thyroidIndicators: List<String> = emptyList(),
    @Json(name = "fatigue_stress_score") val fatigueStressScore: Double = 0.0,
    @Json(name = "hydration_score") val hydrationScore: Double = 50.0
)

@JsonClass(generateAdapter = true)
data class DermatologicalAnalysisResponse(
    @Json(name = "rosacea_risk_score") val rosaceaRiskScore: Double = 0.0,
    @Json(name = "melasma_risk_score") val melasmaRiskScore: Double = 0.0,
    @Json(name = "photo_aging_gap") val photoAgingGap: Int = 0,
    @Json(name = "acne_severity_score") val acneSeverityScore: Double = 0.0,
    @Json(name = "skin_cancer_risk_factors") val skinCancerRiskFactors: List<String> = emptyList(),
    @Json(name = "pigmentation_disorder_risk") val pigmentationDisorderRisk: Double = 0.0,
    @Json(name = "moisture_barrier_score") val moistureBarrierScore: Double = 50.0
)

@JsonClass(generateAdapter = true)
data class ConditionScreeningResponse(
    @Json(name = "condition") val condition: String = "",
    @Json(name = "risk_score") val riskScore: Double = 0.0,
    @Json(name = "facial_markers") val facialMarkers: List<String> = emptyList(),
    @Json(name = "confidence") val confidence: Double = 0.0,
    @Json(name = "recommendation") val recommendation: String = ""
)

@JsonClass(generateAdapter = true)
data class AncestryDerivedPredictionsResponse(
    @Json(name = "predicted_mtdna_haplogroup") val predictedMtdnaHaplogroup: String = "Unknown",
    @Json(name = "haplogroup_confidence") val haplogroupConfidence: Double = 0.0,
    @Json(name = "lactose_tolerance_probability") val lactoseToleranceProbability: Double = 0.5,
    @Json(name = "alcohol_flush_probability") val alcoholFlushProbability: Double = 0.0,
    @Json(name = "caffeine_sensitivity") val caffeineSensitivity: String = "Unknown",
    @Json(name = "bitter_taste_sensitivity") val bitterTasteSensitivity: String = "Unknown",
    @Json(name = "population_specific_risks") val populationSpecificRisks: List<String> = emptyList()
)

@JsonClass(generateAdapter = true)
data class FacialAnalysisResult(
    @Json(name = "image_type") val imageType: String = "face_photo",
    @Json(name = "estimated_biological_age") val estimatedBiologicalAge: Int = 0,
    @Json(name = "estimated_telomere_length_kb") val estimatedTelomereLengthKb: Double = 0.0,
    @Json(name = "telomere_percentile") val telomerePercentile: Int = 50,
    @Json(name = "skin_health_score") val skinHealthScore: Double = 0.0,
    @Json(name = "oxidative_stress_score") val oxidativeStressScore: Double = 0.0,
    @Json(name = "predicted_eye_colour") val predictedEyeColour: String = "unknown",
    @Json(name = "predicted_hair_colour") val predictedHairColour: String = "unknown",
    @Json(name = "predicted_skin_type") val predictedSkinType: String = "unknown",
    @Json(name = "measurements") val measurements: FacialMeasurementsResponse = FacialMeasurementsResponse(),
    @Json(name = "ancestry") val ancestry: AncestryEstimateResponse = AncestryEstimateResponse(),
    @Json(name = "predicted_variants") val predictedVariants: List<PredictedVariantResponse> = emptyList(),
    @Json(name = "reconstructed_dna") val reconstructedDna: ReconstructedDNAResponse? = null,
    @Json(name = "pharmacogenomic_predictions") val pharmacogenomicPredictions: List<PharmacogenomicPredictionResponse> = emptyList(),
    @Json(name = "health_screening") val healthScreening: FacialHealthScreeningResponse? = null,
    @Json(name = "dermatological_analysis") val dermatologicalAnalysis: DermatologicalAnalysisResponse? = null,
    @Json(name = "condition_screenings") val conditionScreenings: List<ConditionScreeningResponse> = emptyList(),
    @Json(name = "ancestry_derived") val ancestryDerived: AncestryDerivedPredictionsResponse? = null,
    @Json(name = "analysis_warnings") val analysisWarnings: List<String> = emptyList()
)

// ---------------------------------------------------------------------------
// Full Analysis Response
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class AnalysisResponse(
    @Json(name = "job_id") val jobId: String,
    @Json(name = "image_type") val imageType: String = "fish_microscopy",
    @Json(name = "telomere_results") val telomereResults: TelomereResult,
    @Json(name = "disease_risks") val diseaseRisks: List<DiseaseRisk> = emptyList(),
    @Json(name = "diet_recommendations") val dietRecommendations: DietRecommendation,
    @Json(name = "facial_analysis") val facialAnalysis: FacialAnalysisResult? = null,
    @Json(name = "report_url") val reportUrl: String? = null,
    @Json(name = "csv_url") val csvUrl: String? = null,
    @Json(name = "created_at") val createdAt: String = "",
    @Json(name = "disclaimer") val disclaimer: String = ""
)

// ---------------------------------------------------------------------------
// Job Status (for polling)
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class JobStatus(
    @Json(name = "job_id") val jobId: String,
    @Json(name = "status") val status: String = "pending",
    @Json(name = "progress_pct") val progressPct: Double = 0.0,
    @Json(name = "message") val message: String = "Job created",
    @Json(name = "result") val result: AnalysisResponse? = null,
    @Json(name = "created_at") val createdAt: String? = null,
    @Json(name = "updated_at") val updatedAt: String? = null
)

// ---------------------------------------------------------------------------
// Image Validation
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class ImageValidationResponse(
    @Json(name = "valid") val valid: Boolean = true,
    @Json(name = "image_type") val imageType: String = "unknown",
    @Json(name = "width") val width: Int = 0,
    @Json(name = "height") val height: Int = 0,
    @Json(name = "channels") val channels: Int = 0,
    @Json(name = "file_size_bytes") val fileSizeBytes: Int = 0,
    @Json(name = "format_detected") val formatDetected: String = "unknown",
    @Json(name = "face_detected") val faceDetected: Boolean = false,
    @Json(name = "issues") val issues: List<String> = emptyList()
)

// ---------------------------------------------------------------------------
// Health Check
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class HealthResponse(
    @Json(name = "status") val status: String = "ok",
    @Json(name = "version") val version: String = "",
    @Json(name = "timestamp") val timestamp: String = ""
)

// ---------------------------------------------------------------------------
// Profile-Only Analysis
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class ProfileAnalysisRequest(
    @Json(name = "age") val age: Int,
    @Json(name = "sex") val sex: String,
    @Json(name = "region") val region: String,
    @Json(name = "dietary_restrictions") val dietaryRestrictions: List<String> = emptyList(),
    @Json(name = "known_variants") val knownVariants: List<String> = emptyList(),
    @Json(name = "telomere_length_kb") val telomereLengthKb: Double? = null,
    @Json(name = "include_nutrition") val includeNutrition: Boolean = true,
    @Json(name = "include_disease_risk") val includeDiseaseRisk: Boolean = true
)

@JsonClass(generateAdapter = true)
data class ProfileAnalysisResponse(
    @Json(name = "disease_risks") val diseaseRisks: List<DiseaseRisk> = emptyList(),
    @Json(name = "diet_recommendations") val dietRecommendations: DietRecommendation? = null,
    @Json(name = "overall_risk_score") val overallRiskScore: Double = 0.0,
    @Json(name = "assessed_at") val assessedAt: String = "",
    @Json(name = "disclaimer") val disclaimer: String = ""
)

// ---------------------------------------------------------------------------
// Disease Risk (standalone)
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class DiseaseRiskRequest(
    @Json(name = "known_variants") val knownVariants: List<String> = emptyList(),
    @Json(name = "telomere_length") val telomereLength: Double? = null,
    @Json(name = "age") val age: Int,
    @Json(name = "sex") val sex: String,
    @Json(name = "region") val region: String
)

@JsonClass(generateAdapter = true)
data class DiseaseRiskResponse(
    @Json(name = "risks") val risks: List<DiseaseRisk> = emptyList(),
    @Json(name = "overall_risk_score") val overallRiskScore: Double = 0.0,
    @Json(name = "assessed_at") val assessedAt: String = "",
    @Json(name = "disclaimer") val disclaimer: String = ""
)

// ---------------------------------------------------------------------------
// Nutrition (standalone)
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class NutritionRequest(
    @Json(name = "age") val age: Int,
    @Json(name = "sex") val sex: String,
    @Json(name = "region") val region: String,
    @Json(name = "dietary_restrictions") val dietaryRestrictions: List<String> = emptyList(),
    @Json(name = "known_variants") val knownVariants: List<String> = emptyList(),
    @Json(name = "health_conditions") val healthConditions: List<String> = emptyList(),
    @Json(name = "calorie_target") val calorieTarget: Int = 2000,
    @Json(name = "meal_plan_days") val mealPlanDays: Int = 7
)

@JsonClass(generateAdapter = true)
data class NutritionResponse(
    @Json(name = "recommendation") val recommendation: DietRecommendation,
    @Json(name = "generated_at") val generatedAt: String = "",
    @Json(name = "disclaimer") val disclaimer: String = ""
)

// ---------------------------------------------------------------------------
// Health Checkup – Report Parse Preview
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class ReportParsePreview(
    @Json(name = "extracted_blood_tests") val extractedBloodTests: Map<String, Double> = emptyMap(),
    @Json(name = "extracted_urine_tests") val extractedUrineTests: Map<String, Double> = emptyMap(),
    @Json(name = "extracted_abdomen_notes") val extractedAbdomenNotes: String = "",
    @Json(name = "unrecognized_lines") val unrecognizedLines: List<String> = emptyList(),
    @Json(name = "confidence") val confidence: Double = 0.0,
    @Json(name = "file_type") val fileType: String = "",
    @Json(name = "text_length") val textLength: Int = 0
)

// ---------------------------------------------------------------------------
// Health Checkup – Upload Response
// ---------------------------------------------------------------------------

@JsonClass(generateAdapter = true)
data class HealthCheckupResponse(
    @Json(name = "lab_results") val labResults: List<Map<String, Any>> = emptyList(),
    @Json(name = "abnormal_count") val abnormalCount: Int = 0,
    @Json(name = "total_tested") val totalTested: Int = 0,
    @Json(name = "findings") val findings: List<Map<String, Any>> = emptyList(),
    @Json(name = "detected_conditions") val detectedConditions: List<String> = emptyList(),
    @Json(name = "overall_health_score") val overallHealthScore: Double = 0.0,
    @Json(name = "diet_recommendation") val dietRecommendation: DietRecommendation? = null,
    @Json(name = "disclaimer") val disclaimer: String = ""
)
