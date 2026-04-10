// APIModels.swift
// Teloscopy
//
// All API request and response models matching the FastAPI backend.
// Mirrors Android app's ApiModels.kt

import Foundation

// MARK: - Color Theme Constants

enum TeloscopyColors {
    static let backgroundPrimary = "#0B0F19"
    static let backgroundSecondary = "#111827"
    static let surface = "#1F2937"
    static let surfaceVariant = "#1E2235"
    static let surfaceElevated = "#283040"
    static let accent = "#00D4AA"
    static let primaryCyan = "#00E5FF"
    static let secondaryPurple = "#7C4DFF"
    static let tertiaryGreen = "#69F0AE"
    static let textPrimary = "#F9FAFB"
    static let textSecondary = "#9CA3AF"
    static let error = "#EF4444"
    static let warning = "#F59E0B"
    static let riskLow = "#4ADE80"
    static let riskModerate = "#FB923C"
    static let riskHigh = "#EF4444"
    static let riskVeryHigh = "#DC2626"
}

// MARK: - Telomere Analysis Result (API Response DTO)
// NOTE: The canonical TelomereResult is defined in Analysis.swift.
// This lightweight version maps the /api/analyze endpoint response.

struct TelomereAPIResult: Codable, Equatable {
    let meanLengthBp: Double?
    let medianLengthBp: Double?
    let stdDev: Double?
    let cvPercent: Double?
    let shortTelomerePct: Double?
    let spotsDetected: Int?
    let biologicalAgeEstimate: Double?
    let percentileRank: Double?
    let tsRatio: Double?
    let overlayImageUrl: String?
    let rawMeasurements: [Double]?

    enum CodingKeys: String, CodingKey {
        case meanLengthBp = "mean_length_bp"
        case medianLengthBp = "median_length_bp"
        case stdDev = "std_dev"
        case cvPercent = "cv_percent"
        case shortTelomerePct = "short_telomere_pct"
        case spotsDetected = "spots_detected"
        case biologicalAgeEstimate = "biological_age_estimate"
        case percentileRank = "percentile_rank"
        case tsRatio = "ts_ratio"
        case overlayImageUrl = "overlay_image_url"
        case rawMeasurements = "raw_measurements"
    }
}

// MARK: - Disease Risk

struct DiseaseRisk: Codable, Identifiable, Equatable {
    var id: String { disease }
    let disease: String
    let lifetimeRiskPct: Double?
    let relativeRisk: Double?
    let riskLevel: String
    let probability: Double?
    let contributingVariants: [String]?
    let contributingFactors: [String]?
    let recommendations: [String]?

    enum CodingKeys: String, CodingKey {
        case disease
        case lifetimeRiskPct = "lifetime_risk_pct"
        case relativeRisk = "relative_risk"
        case riskLevel = "risk_level"
        case probability
        case contributingVariants = "contributing_variants"
        case contributingFactors = "contributing_factors"
        case recommendations
    }

    var displayRisk: Double {
        lifetimeRiskPct ?? (probability.map { $0 * 100 }) ?? 0
    }

    var displayFactors: [String] {
        contributingFactors ?? contributingVariants ?? []
    }
}

// MARK: - Diet & Nutrition

struct MealPlan: Codable, Identifiable, Equatable {
    var id: String { "day-\(day)" }
    let day: Int
    let breakfast: [String]
    let lunch: [String]
    let dinner: [String]
    let snacks: [String]
}

struct DietRecommendation: Codable, Equatable {
    let summary: String?
    let dailyCalories: Int?
    let calorieTarget: Int?
    let keyNutrients: [String]?
    let targetFoods: [String]?
    let foodsToIncrease: [String]?
    let avoidFoods: [String]?
    let foodsToAvoid: [String]?
    let mealPlans: [MealPlan]?

    enum CodingKeys: String, CodingKey {
        case summary
        case dailyCalories = "daily_calories"
        case calorieTarget = "calorie_target"
        case keyNutrients = "key_nutrients"
        case targetFoods = "target_foods"
        case foodsToIncrease = "foods_to_increase"
        case avoidFoods = "avoid_foods"
        case foodsToAvoid = "foods_to_avoid"
        case mealPlans = "meal_plans"
    }

    var displayCalories: Int {
        dailyCalories ?? calorieTarget ?? 2000
    }

    var displayNutrients: [String] {
        keyNutrients ?? []
    }

    var displayTargetFoods: [String] {
        targetFoods ?? foodsToIncrease ?? []
    }

    var displayAvoidFoods: [String] {
        avoidFoods ?? foodsToAvoid ?? []
    }
}

// MARK: - Facial Analysis Models

struct FacialMeasurementsResponse: Codable, Equatable {
    let faceWidthPx: Double?
    let faceHeightPx: Double?
    let faceAspectRatio: Double?
    let skinBrightness: Double?
    let wrinkleScore: Double?
    let symmetryScore: Double?
    let darkCircleScore: Double?
    let textureRoughness: Double?
    let uvDamageScore: Double?

    enum CodingKeys: String, CodingKey {
        case faceWidthPx = "face_width_px"
        case faceHeightPx = "face_height_px"
        case faceAspectRatio = "face_aspect_ratio"
        case skinBrightness = "skin_brightness"
        case wrinkleScore = "wrinkle_score"
        case symmetryScore = "symmetry_score"
        case darkCircleScore = "dark_circle_score"
        case textureRoughness = "texture_roughness"
        case uvDamageScore = "uv_damage_score"
    }
}

struct AncestryEstimateResponse: Codable, Equatable {
    let european: Double?
    let eastAsian: Double?
    let southAsian: Double?
    let african: Double?
    let middleEastern: Double?
    let latinAmerican: Double?
    let confidence: Double?

    enum CodingKeys: String, CodingKey {
        case european
        case eastAsian = "east_asian"
        case southAsian = "south_asian"
        case african
        case middleEastern = "middle_eastern"
        case latinAmerican = "latin_american"
        case confidence
    }
}

struct PredictedVariantResponse: Codable, Identifiable, Equatable {
    var id: String { rsid }
    let rsid: String
    let gene: String?
    let predictedGenotype: String?
    let confidence: Double?
    let basis: String?
    let riskAllele: String?
    let refAllele: String?

    enum CodingKeys: String, CodingKey {
        case rsid
        case gene
        case predictedGenotype = "predicted_genotype"
        case confidence
        case basis
        case riskAllele = "risk_allele"
        case refAllele = "ref_allele"
    }
}

struct ReconstructedSequenceResponse: Codable, Identifiable, Equatable {
    var id: String { rsid }
    let rsid: String
    let gene: String?
    let chromosome: String?
    let position: Int?
    let refAllele: String?
    let predictedAllele1: String?
    let predictedAllele2: String?
    let flankingLeft: String?
    let flankingRight: String?
    let confidence: Double?

    enum CodingKeys: String, CodingKey {
        case rsid
        case gene
        case chromosome
        case position
        case refAllele = "ref_allele"
        case predictedAllele1 = "predicted_allele_1"
        case predictedAllele2 = "predicted_allele_2"
        case flankingLeft = "flanking_left"
        case flankingRight = "flanking_right"
        case confidence
    }
}

struct ReconstructedDNAResponse: Codable, Equatable {
    let sequences: [ReconstructedSequenceResponse]?
    let totalVariants: Int?
    let genomeBuild: String?
    let fasta: String?
    let disclaimer: String?

    enum CodingKeys: String, CodingKey {
        case sequences
        case totalVariants = "total_variants"
        case genomeBuild = "genome_build"
        case fasta
        case disclaimer
    }
}

struct PharmacogenomicPredictionResponse: Codable, Identifiable, Equatable {
    var id: String { "\(gene ?? "")-\(rsid ?? "")" }
    let gene: String?
    let rsid: String?
    let predictedPhenotype: String?
    let confidence: Double?
    let affectedDrugs: [String]?
    let clinicalRecommendation: String?
    let basis: String?

    enum CodingKeys: String, CodingKey {
        case gene
        case rsid
        case predictedPhenotype = "predicted_phenotype"
        case confidence
        case affectedDrugs = "affected_drugs"
        case clinicalRecommendation = "clinical_recommendation"
        case basis
    }
}

struct FacialHealthScreeningResponse: Codable, Equatable {
    let estimatedBmiCategory: String?
    let anemiaRiskScore: Double?
    let cardiovascularRiskIndicators: [String]?
    let thyroidIndicators: [String]?
    let fatigueStressScore: Double?
    let hydrationScore: Double?

    enum CodingKeys: String, CodingKey {
        case estimatedBmiCategory = "estimated_bmi_category"
        case anemiaRiskScore = "anemia_risk_score"
        case cardiovascularRiskIndicators = "cardiovascular_risk_indicators"
        case thyroidIndicators = "thyroid_indicators"
        case fatigueStressScore = "fatigue_stress_score"
        case hydrationScore = "hydration_score"
    }
}

struct DermatologicalAnalysisResponse: Codable, Equatable {
    let rosaceaRiskScore: Double?
    let melasmaRiskScore: Double?
    let photoAgingGap: Double?
    let acneSeverityScore: Double?
    let skinCancerRiskFactors: [String]?
    let pigmentationDisorderRisk: Double?
    let moistureBarrierScore: Double?

    enum CodingKeys: String, CodingKey {
        case rosaceaRiskScore = "rosacea_risk_score"
        case melasmaRiskScore = "melasma_risk_score"
        case photoAgingGap = "photo_aging_gap"
        case acneSeverityScore = "acne_severity_score"
        case skinCancerRiskFactors = "skin_cancer_risk_factors"
        case pigmentationDisorderRisk = "pigmentation_disorder_risk"
        case moistureBarrierScore = "moisture_barrier_score"
    }
}

struct ConditionScreeningResponse: Codable, Identifiable, Equatable {
    var id: String { condition }
    let condition: String
    let riskScore: Double?
    let facialMarkers: [String]?
    let confidence: Double?
    let recommendation: String?

    enum CodingKeys: String, CodingKey {
        case condition
        case riskScore = "risk_score"
        case facialMarkers = "facial_markers"
        case confidence
        case recommendation
    }
}

struct AncestryDerivedPredictionsResponse: Codable, Equatable {
    let predictedMtdnaHaplogroup: String?
    let lactoseToleranceProbability: Double?
    let alcoholFlushProbability: Double?
    let caffeineSensitivity: String?
    let bitterTasteSensitivity: String?
    let populationSpecificRisks: [String]?

    enum CodingKeys: String, CodingKey {
        case predictedMtdnaHaplogroup = "predicted_mtdna_haplogroup"
        case lactoseToleranceProbability = "lactose_tolerance_probability"
        case alcoholFlushProbability = "alcohol_flush_probability"
        case caffeineSensitivity = "caffeine_sensitivity"
        case bitterTasteSensitivity = "bitter_taste_sensitivity"
        case populationSpecificRisks = "population_specific_risks"
    }
}

struct FacialAnalysisResult: Codable, Equatable {
    let measurements: FacialMeasurementsResponse?
    let ancestry: AncestryEstimateResponse?
    let predictedVariants: [PredictedVariantResponse]?
    let reconstructedDna: ReconstructedDNAResponse?
    let pharmacogenomics: [PharmacogenomicPredictionResponse]?
    let healthScreening: FacialHealthScreeningResponse?
    let dermatological: DermatologicalAnalysisResponse?
    let conditionScreening: [ConditionScreeningResponse]?
    let ancestryDerived: AncestryDerivedPredictionsResponse?
    let estimatedBiologicalAge: Double?
    let estimatedTelomereLengthKb: Double?
    let skinHealthScore: Double?
    let oxidativeStressScore: Double?
    let predictedEyeColour: String?
    let predictedHairColour: String?

    enum CodingKeys: String, CodingKey {
        case measurements
        case ancestry
        case predictedVariants = "predicted_variants"
        case reconstructedDna = "reconstructed_dna"
        case pharmacogenomics
        case healthScreening = "health_screening"
        case dermatological
        case conditionScreening = "condition_screening"
        case ancestryDerived = "ancestry_derived"
        case estimatedBiologicalAge = "estimated_biological_age"
        case estimatedTelomereLengthKb = "estimated_telomere_length_kb"
        case skinHealthScore = "skin_health_score"
        case oxidativeStressScore = "oxidative_stress_score"
        case predictedEyeColour = "predicted_eye_colour"
        case predictedHairColour = "predicted_hair_colour"
    }
}

// MARK: - Top-Level Response Models

struct AnalysisResponse: Codable, Equatable {
    let jobId: String?
    let imageType: String?
    let telomereResults: TelomereAPIResult?
    let diseaseRisks: [DiseaseRisk]?
    let dietRecommendations: DietRecommendation?
    let facialAnalysis: FacialAnalysisResult?
    let reportUrl: String?
    let csvUrl: String?
    let createdAt: String?
    let disclaimer: String?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case imageType = "image_type"
        case telomereResults = "telomere_results"
        case diseaseRisks = "disease_risks"
        case dietRecommendations = "diet_recommendations"
        case facialAnalysis = "facial_analysis"
        case reportUrl = "report_url"
        case csvUrl = "csv_url"
        case createdAt = "created_at"
        case disclaimer
    }
}

struct JobStatus: Codable, Equatable {
    let jobId: String
    let status: String
    let progress: Double?
    let progressPct: Double?
    let message: String?
    let result: AnalysisResponse?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
        case progress
        case progressPct = "progress_pct"
        case message
        case result
    }

    var displayProgress: Double {
        progressPct ?? progress ?? 0
    }
}

struct ImageValidationResponse: Codable, Equatable {
    let valid: Bool
    let imageType: String?
    let width: Int?
    let height: Int?
    let channels: Int?
    let fileSizeBytes: Int?
    let formatDetected: String?
    let faceDetected: Bool?
    let issues: [String]?

    enum CodingKeys: String, CodingKey {
        case valid
        case imageType = "image_type"
        case width, height, channels
        case fileSizeBytes = "file_size_bytes"
        case formatDetected = "format_detected"
        case faceDetected = "face_detected"
        case issues
    }
}

// MARK: - Request Models

struct ProfileAnalysisRequest: Codable {
    let age: Int
    let sex: String
    let region: String
    let country: String?
    let state: String?
    let geneticRisks: [String]
    let variants: [String: String]
    let dietaryRestrictions: [String]

    enum CodingKeys: String, CodingKey {
        case age, sex, region, country, state
        case geneticRisks = "genetic_risks"
        case variants
        case dietaryRestrictions = "dietary_restrictions"
    }
}

struct ProfileAnalysisResponse: Codable, Equatable {
    let diseaseRisks: [DiseaseRisk]?
    let dietRecommendations: DietRecommendation?
    let telomereEstimate: TelomereAPIResult?
    let summary: String?

    enum CodingKeys: String, CodingKey {
        case diseaseRisks = "disease_risks"
        case dietRecommendations = "diet_recommendations"
        case telomereEstimate = "telomere_estimate"
        case summary
    }
}

struct NutritionRequest: Codable {
    let age: Int
    let sex: String
    let region: String
    let country: String?
    let state: String?
    let geneticRisks: [String]
    let variants: [String: String]
    let dietaryRestrictions: [String]
    let caloriesTarget: Int?
    let days: Int?

    enum CodingKeys: String, CodingKey {
        case age, sex, region, country, state
        case geneticRisks = "genetic_risks"
        case variants
        case dietaryRestrictions = "dietary_restrictions"
        case caloriesTarget = "calories_target"
        case days
    }
}

struct NutritionResponse: Codable, Equatable {
    let dietRecommendations: DietRecommendation?
    let mealPlans: [MealPlan]?

    enum CodingKeys: String, CodingKey {
        case dietRecommendations = "diet_recommendations"
        case mealPlans = "meal_plans"
    }
}

struct DiseaseRiskRequest: Codable {
    let variants: [String: String]
    let age: Int
    let sex: String
    let region: String

    enum CodingKeys: String, CodingKey {
        case variants, age, sex, region
    }
}

struct DiseaseRiskResponse: Codable, Equatable {
    let diseaseRisks: [DiseaseRisk]?

    enum CodingKeys: String, CodingKey {
        case diseaseRisks = "disease_risks"
    }
}

// NOTE: The canonical UserProfile is defined in Analysis.swift.
// This lightweight version is used for request payloads (profile-based analysis).

struct UserProfileRequest: Codable {
    let age: Int
    let sex: String
    let region: String
    let country: String?
    let state: String?
    let geneticRisks: [String]?
    let variants: [String: String]?

    enum CodingKeys: String, CodingKey {
        case age, sex, region, country, state
        case geneticRisks = "genetic_risks"
        case variants
    }
}

struct BloodTestPanel: Codable {
    let hemoglobin: Double?
    let whiteBloodCells: Double?
    let platelets: Double?
    let glucose: Double?
    let cholesterolTotal: Double?
    let cholesterolHdl: Double?
    let cholesterolLdl: Double?
    let triglycerides: Double?
    let creatinine: Double?
    let alt: Double?
    let ast: Double?
    let tsh: Double?

    enum CodingKeys: String, CodingKey {
        case hemoglobin
        case whiteBloodCells = "white_blood_cells"
        case platelets, glucose
        case cholesterolTotal = "cholesterol_total"
        case cholesterolHdl = "cholesterol_hdl"
        case cholesterolLdl = "cholesterol_ldl"
        case triglycerides, creatinine, alt, ast, tsh
    }
}

struct UrineTestPanel: Codable {
    let ph: Double?
    let specificGravity: Double?
    let protein: String?
    let glucose: String?
    let ketones: String?
    let blood: String?

    enum CodingKeys: String, CodingKey {
        case ph
        case specificGravity = "specific_gravity"
        case protein, glucose, ketones, blood
    }
}

struct HealthCheckupRequest: Codable {
    let profile: UserProfileRequest
    let bloodTests: BloodTestPanel?
    let urineTests: UrineTestPanel?

    enum CodingKeys: String, CodingKey {
        case profile
        case bloodTests = "blood_tests"
        case urineTests = "urine_tests"
    }
}

struct HealthCheckupResponse: Codable, Equatable {
    let summary: String?
    let diseaseRisks: [DiseaseRisk]?
    let dietRecommendations: DietRecommendation?
    let warnings: [String]?

    enum CodingKeys: String, CodingKey {
        case summary
        case diseaseRisks = "disease_risks"
        case dietRecommendations = "diet_recommendations"
        case warnings
    }
}

struct HealthResponse: Codable, Equatable {
    let status: String
    let version: String?
    let uptime: Double?
}

struct AgentStatusResponse: Codable, Equatable {
    let status: String
    let agents: [String: String]?
    let activeJobs: Int?

    enum CodingKeys: String, CodingKey {
        case status
        case agents
        case activeJobs = "active_jobs"
    }
}

// MARK: - API Error

struct APIErrorResponse: Codable {
    let detail: String?
    let message: String?

    var displayMessage: String {
        detail ?? message ?? "Unknown error occurred"
    }
}

enum AppNetworkError: LocalizedError {
    case invalidURL
    case networkError(Error)
    case httpError(Int, String)
    case decodingError(Error)
    case invalidResponse
    case timeout
    case serverUnreachable

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid server URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .httpError(let code, let message):
            return "Server error (\(code)): \(message)"
        case .decodingError(let error):
            return "Failed to parse response: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid response from server"
        case .timeout:
            return "Request timed out"
        case .serverUnreachable:
            return "Cannot reach server. Check your server URL in Settings."
        }
    }
}
