package com.teloscopy.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.teloscopy.app.data.api.DiseaseRiskResponse
import com.teloscopy.app.data.api.NutritionResponse
import com.teloscopy.app.data.api.ProfileAnalysisResponse
import com.teloscopy.app.data.repository.AnalysisRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * UI state for the profile-only analysis screen (no image required).
 *
 * Supports three independent analysis paths:
 * - Combined profile analysis (disease risk + nutrition)
 * - Standalone disease risk
 * - Standalone nutrition plan
 */
data class ProfileUiState(
    val isLoading: Boolean = false,
    val result: ProfileAnalysisResponse? = null,
    val diseaseRiskResult: DiseaseRiskResponse? = null,
    val nutritionResult: NutritionResponse? = null,
    val error: String? = null
)

/**
 * ViewModel for profile-only analysis.
 *
 * Users can submit demographic and genetic information without an
 * image to receive disease-risk assessments and/or personalised
 * nutrition plans from the Teloscopy backend.
 */
@HiltViewModel
class ProfileViewModel @Inject constructor(
    private val repository: AnalysisRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(ProfileUiState())
    val uiState: StateFlow<ProfileUiState> = _uiState.asStateFlow()

    // ---------------------------------------------------------------------
    // Combined profile analysis
    // ---------------------------------------------------------------------

    /**
     * Run a combined profile analysis (disease risk + nutrition) using
     * only user-provided details.
     *
     * @param age              User's age in years
     * @param sex              Biological sex ("male", "female", "other")
     * @param region           Geographic region (e.g. "Northern Europe")
     * @param restrictions     Dietary restrictions (e.g. "vegetarian")
     * @param variants         Known genetic variants (e.g. "rs429358:CT")
     * @param telomereLength   Optional self-reported telomere length in kb
     * @param includeNutrition Whether to include nutrition recommendations
     * @param includeDiseaseRisk Whether to include disease risk assessment
     */
    fun analyzeProfile(
        age: Int,
        sex: String,
        region: String,
        restrictions: List<String>,
        variants: List<String>,
        telomereLength: Double?,
        includeNutrition: Boolean,
        includeDiseaseRisk: Boolean
    ) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(isLoading = true, error = null, result = null)
            }

            repository.analyzeProfile(
                age = age,
                sex = sex,
                region = region,
                restrictions = restrictions,
                variants = variants,
                telomereLength = telomereLength,
                includeNutrition = includeNutrition,
                includeDiseaseRisk = includeDiseaseRisk
            ).onSuccess { response ->
                _uiState.update {
                    it.copy(isLoading = false, result = response)
                }
            }.onFailure { throwable ->
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        error = throwable.message
                            ?: "Profile analysis failed. Please try again."
                    )
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Standalone disease risk
    // ---------------------------------------------------------------------

    /**
     * Compute disease-risk scores from variants and optional telomere
     * data without requiring an image.
     *
     * @param variants       Known genetic variants (rsID:genotype format)
     * @param telomereLength Optional telomere length in kb
     * @param age            User's age
     * @param sex            Biological sex
     * @param region         Geographic region
     */
    fun getDiseaseRisk(
        variants: List<String>,
        telomereLength: Double?,
        age: Int,
        sex: String,
        region: String
    ) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(isLoading = true, error = null, diseaseRiskResult = null)
            }

            repository.getDiseaseRisk(
                variants = variants,
                telomereLength = telomereLength,
                age = age,
                sex = sex,
                region = region
            ).onSuccess { response ->
                _uiState.update {
                    it.copy(isLoading = false, diseaseRiskResult = response)
                }
            }.onFailure { throwable ->
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        error = throwable.message
                            ?: "Disease risk analysis failed. Please try again."
                    )
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Standalone nutrition
    // ---------------------------------------------------------------------

    /**
     * Generate a personalised nutrition plan from user details.
     *
     * @param age          User's age
     * @param sex          Biological sex
     * @param region       Geographic region
     * @param restrictions Dietary restrictions
     * @param variants     Known genetic variants
     * @param conditions   Known health conditions (e.g. "diabetes")
     * @param calories     Daily calorie target (800-5000)
     * @param days         Number of meal plan days (1-30)
     */
    fun getNutrition(
        age: Int,
        sex: String,
        region: String,
        restrictions: List<String>,
        variants: List<String>,
        conditions: List<String>,
        calories: Int,
        days: Int
    ) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(isLoading = true, error = null, nutritionResult = null)
            }

            repository.getNutrition(
                age = age,
                sex = sex,
                region = region,
                restrictions = restrictions,
                variants = variants,
                conditions = conditions,
                calories = calories,
                days = days
            ).onSuccess { response ->
                _uiState.update {
                    it.copy(isLoading = false, nutritionResult = response)
                }
            }.onFailure { throwable ->
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        error = throwable.message
                            ?: "Nutrition plan generation failed. Please try again."
                    )
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------------------

    /** Clear the current error message. */
    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    /** Reset the entire UI state to its initial values. */
    fun resetState() {
        _uiState.value = ProfileUiState()
    }
}
