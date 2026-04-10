package com.teloscopy.app.data.repository

import com.teloscopy.app.data.api.AnalysisResponse
import com.teloscopy.app.data.api.DiseaseRiskRequest
import com.teloscopy.app.data.api.DiseaseRiskResponse
import com.teloscopy.app.data.api.ImageValidationResponse
import com.teloscopy.app.data.api.JobStatus
import com.teloscopy.app.data.api.NutritionRequest
import com.teloscopy.app.data.api.NutritionResponse
import com.teloscopy.app.data.api.ProfileAnalysisRequest
import com.teloscopy.app.data.api.ProfileAnalysisResponse
import com.teloscopy.app.data.api.TeloscopyApi
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Response
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository that mediates between the ViewModels and the Teloscopy
 * FastAPI backend via [TeloscopyApi].
 *
 * Every public method returns [Result]<T> so callers can use
 * `.onSuccess` / `.onFailure` without try-catch boilerplate.
 */
@Singleton
class AnalysisRepository @Inject constructor(
    private val api: TeloscopyApi
) {

    // ---------------------------------------------------------------------
    // Full analysis (image + profile)
    // ---------------------------------------------------------------------

    /**
     * Upload an image together with the user profile and start an
     * asynchronous analysis job on the backend.
     *
     * @return [Result]<[JobStatus]> containing the `jobId` for polling.
     */
    suspend fun analyzeWithImage(
        imageFile: File,
        age: Int,
        sex: String,
        region: String,
        restrictions: List<String>,
        variants: List<String>
    ): Result<JobStatus> = safeApiCall {
        val mimeType = mimeTypeForFile(imageFile)

        val imagePart = MultipartBody.Part.createFormData(
            "file",
            imageFile.name,
            imageFile.asRequestBody(mimeType.toMediaTypeOrNull())
        )

        val textType = "text/plain".toMediaTypeOrNull()

        api.analyze(
            file = imagePart,
            age = age.toString().toRequestBody(textType),
            sex = sex.toRequestBody(textType),
            region = region.toRequestBody(textType),
            dietaryRestrictions = restrictions.joinToString(",").toRequestBody(textType),
            knownVariants = variants.joinToString(",").toRequestBody(textType)
        )
    }

    // ---------------------------------------------------------------------
    // Job polling
    // ---------------------------------------------------------------------

    /**
     * Poll the backend for the current status of an analysis job.
     */
    suspend fun pollStatus(jobId: String): Result<JobStatus> = safeApiCall {
        api.getStatus(jobId)
    }

    // ---------------------------------------------------------------------
    // Fetch completed results
    // ---------------------------------------------------------------------

    /**
     * Retrieve the full analysis results for a completed job.
     */
    suspend fun getResults(jobId: String): Result<AnalysisResponse> = safeApiCall {
        api.getResults(jobId)
    }

    // ---------------------------------------------------------------------
    // Image validation
    // ---------------------------------------------------------------------

    /**
     * Validate an image file before submitting for analysis.
     * Checks format, dimensions, and classifies as face photo or microscopy.
     */
    suspend fun validateImage(imageFile: File): Result<ImageValidationResponse> = safeApiCall {
        val mimeType = mimeTypeForFile(imageFile)

        val imagePart = MultipartBody.Part.createFormData(
            "file",
            imageFile.name,
            imageFile.asRequestBody(mimeType.toMediaTypeOrNull())
        )

        api.validateImage(imagePart)
    }

    // ---------------------------------------------------------------------
    // Profile-only analysis (no image)
    // ---------------------------------------------------------------------

    /**
     * Run disease-risk and nutrition analysis using only user-provided
     * profile details (no image upload required).
     */
    suspend fun analyzeProfile(
        age: Int,
        sex: String,
        region: String,
        restrictions: List<String>,
        variants: List<String>,
        telomereLength: Double?,
        includeNutrition: Boolean,
        includeDiseaseRisk: Boolean
    ): Result<ProfileAnalysisResponse> = safeApiCall {
        val request = ProfileAnalysisRequest(
            age = age,
            sex = sex,
            region = region,
            dietaryRestrictions = restrictions,
            knownVariants = variants,
            telomereLengthKb = telomereLength,
            includeNutrition = includeNutrition,
            includeDiseaseRisk = includeDiseaseRisk
        )
        api.analyzeProfile(request)
    }

    // ---------------------------------------------------------------------
    // Standalone disease risk
    // ---------------------------------------------------------------------

    /**
     * Compute disease-risk scores from variants and optional telomere data.
     */
    suspend fun getDiseaseRisk(
        variants: List<String>,
        telomereLength: Double?,
        age: Int,
        sex: String,
        region: String
    ): Result<DiseaseRiskResponse> = safeApiCall {
        val request = DiseaseRiskRequest(
            knownVariants = variants,
            telomereLength = telomereLength,
            age = age,
            sex = sex,
            region = region
        )
        api.getDiseaseRisk(request)
    }

    // ---------------------------------------------------------------------
    // Standalone nutrition
    // ---------------------------------------------------------------------

    /**
     * Generate a personalised nutrition plan from user details.
     */
    suspend fun getNutrition(
        age: Int,
        sex: String,
        region: String,
        restrictions: List<String>,
        variants: List<String>,
        conditions: List<String>,
        calories: Int,
        days: Int
    ): Result<NutritionResponse> = safeApiCall {
        val request = NutritionRequest(
            age = age,
            sex = sex,
            region = region,
            dietaryRestrictions = restrictions,
            knownVariants = variants,
            healthConditions = conditions,
            calorieTarget = calories,
            mealPlanDays = days
        )
        api.getNutrition(request)
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    /**
     * Wraps a Retrofit [Response] call in a [Result], mapping HTTP errors
     * to descriptive failure messages.
     */
    private suspend inline fun <T> safeApiCall(
        crossinline call: suspend () -> Response<T>
    ): Result<T> = runCatching {
        val response = call()
        if (response.isSuccessful) {
            response.body() ?: throw IllegalStateException(
                "Server returned success but the response body was empty."
            )
        } else {
            val errorBody = response.errorBody()?.string() ?: "Unknown error"
            throw ApiException(response.code(), errorBody)
        }
    }

    /**
     * Determine the MIME type for an image file based on its extension.
     */
    private fun mimeTypeForFile(file: File): String = when (file.extension.lowercase()) {
        "png" -> "image/png"
        "jpg", "jpeg" -> "image/jpeg"
        "tif", "tiff" -> "image/tiff"
        "bmp" -> "image/bmp"
        "webp" -> "image/webp"
        else -> "application/octet-stream"
    }
}

/**
 * Exception thrown when the API returns a non-2xx HTTP status.
 */
class ApiException(
    val httpCode: Int,
    val errorBody: String
) : Exception("HTTP $httpCode: $errorBody")
