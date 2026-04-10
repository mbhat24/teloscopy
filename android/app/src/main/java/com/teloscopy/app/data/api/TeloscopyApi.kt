package com.teloscopy.app.data.api

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path

interface TeloscopyApi {

    /**
     * Submit an image for analysis along with user profile metadata.
     * Returns 202 with a [JobStatus] containing the job ID for polling.
     */
    @Multipart
    @POST("api/analyze")
    suspend fun analyze(
        @Part file: MultipartBody.Part,
        @Part("age") age: RequestBody,
        @Part("sex") sex: RequestBody,
        @Part("region") region: RequestBody,
        @Part("dietary_restrictions") dietaryRestrictions: RequestBody,
        @Part("known_variants") knownVariants: RequestBody
    ): Response<JobStatus>

    /**
     * Poll the processing status of a submitted analysis job.
     * [progressPct] in the response ranges from 0 to 100.
     */
    @GET("api/status/{job_id}")
    suspend fun getStatus(
        @Path("job_id") jobId: String
    ): Response<JobStatus>

    /**
     * Retrieve completed analysis results.
     * Returns 409 Conflict if the job has not finished processing.
     */
    @GET("api/results/{job_id}")
    suspend fun getResults(
        @Path("job_id") jobId: String
    ): Response<AnalysisResponse>

    /**
     * Run a profile-based analysis combining disease-risk and/or nutrition
     * recommendations without an image upload.
     */
    @POST("api/profile-analysis")
    suspend fun analyzeProfile(
        @Body request: ProfileAnalysisRequest
    ): Response<ProfileAnalysisResponse>

    /**
     * Compute disease-risk scores for the given genetic variants and
     * demographic information.
     */
    @POST("api/disease-risk")
    suspend fun getDiseaseRisk(
        @Body request: DiseaseRiskRequest
    ): Response<DiseaseRiskResponse>

    /**
     * Generate personalised nutrition recommendations and meal plans.
     */
    @POST("api/nutrition")
    suspend fun getNutrition(
        @Body request: NutritionRequest
    ): Response<NutritionResponse>

    /**
     * Pre-validate an image before full analysis to check format,
     * dimensions, and suitability.
     */
    @Multipart
    @POST("api/validate-image")
    suspend fun validateImage(
        @Part file: MultipartBody.Part
    ): Response<ImageValidationResponse>

    /**
     * Health-check endpoint for connectivity verification.
     */
    @GET("api/health")
    suspend fun checkHealth(): Response<HealthResponse>

    // ----- Health Checkup (Report Upload) -----

    /**
     * Parse a health-checkup report (PDF/image) and return a preview of
     * extracted lab values before committing the full analysis.
     */
    @Multipart
    @POST("api/health-checkup/parse-report")
    suspend fun parseHealthReport(
        @Part file: MultipartBody.Part
    ): ReportParsePreview

    /**
     * Upload a health-checkup report with demographic context and receive
     * the full analysis (lab results, findings, diet recommendation, etc.).
     */
    @Multipart
    @POST("api/health-checkup/upload")
    suspend fun uploadHealthReport(
        @Part file: MultipartBody.Part,
        @Part("age") age: RequestBody,
        @Part("sex") sex: RequestBody,
        @Part("region") region: RequestBody,
        @Part("country") country: RequestBody?,
        @Part("state") state: RequestBody?,
        @Part("dietary_restrictions") dietaryRestrictions: RequestBody?,
        @Part("known_variants") knownVariants: RequestBody?,
        @Part("calorie_target") calorieTarget: RequestBody?,
        @Part("meal_plan_days") mealPlanDays: RequestBody?,
        @Part("health_conditions") healthConditions: RequestBody?
    ): HealthCheckupResponse
}
