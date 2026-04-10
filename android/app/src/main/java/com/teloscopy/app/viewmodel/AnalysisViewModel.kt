package com.teloscopy.app.viewmodel

import android.content.Context
import android.net.Uri
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.teloscopy.app.data.api.AnalysisResponse
import com.teloscopy.app.data.api.ImageValidationResponse
import com.teloscopy.app.data.repository.AnalysisRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import javax.inject.Inject

/**
 * UI state for the image-based analysis screen.
 *
 * Tracks the full lifecycle: idle -> uploading -> processing (with
 * progress) -> completed / failed.
 */
data class AnalysisUiState(
    val isLoading: Boolean = false,
    val jobId: String? = null,
    val jobStatus: String = "idle",
    val progressPct: Int = 0,
    val statusMessage: String = "",
    val result: AnalysisResponse? = null,
    val error: String? = null,
    val imageValidation: ImageValidationResponse? = null
)

/**
 * ViewModel for the full analysis flow (image upload + profile -> results).
 *
 * Handles:
 * 1. Image validation (optional pre-check)
 * 2. Submitting the image + profile to the backend
 * 3. Polling the backend for job progress every 2 seconds
 * 4. Fetching final results once the job completes
 */
@HiltViewModel
class AnalysisViewModel @Inject constructor(
    private val repository: AnalysisRepository,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val _uiState = MutableStateFlow(AnalysisUiState())
    val uiState: StateFlow<AnalysisUiState> = _uiState.asStateFlow()

    /** Handle to the active polling coroutine so it can be cancelled. */
    private var pollingJob: Job? = null

    companion object {
        /** Interval between status polls in milliseconds. */
        private const val POLL_INTERVAL_MS = 2000L

        /** Maximum number of poll iterations (~4 minutes at 2 s intervals). */
        private const val MAX_POLL_ITERATIONS = 120
    }

    // ---------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------

    /**
     * Submit an image together with user profile data for analysis.
     *
     * The [imageUri] is resolved via the [context]'s content resolver,
     * copied to a temporary file, and uploaded as multipart form data.
     */
    fun submitAnalysis(
        imageUri: Uri?,
        context: Context,
        age: Int,
        sex: String,
        region: String,
        restrictions: List<String>,
        variants: List<String>
    ) {
        if (imageUri == null) {
            _uiState.update { it.copy(error = "No image selected. Please choose an image.") }
            return
        }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isLoading = true,
                    jobStatus = "uploading",
                    statusMessage = "Preparing image for upload...",
                    error = null,
                    result = null,
                    progressPct = 0
                )
            }

            val tempFile = uriToTempFile(imageUri, context)
            if (tempFile == null) {
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        jobStatus = "failed",
                        error = "Unable to read the selected image."
                    )
                }
                return@launch
            }

            _uiState.update {
                it.copy(statusMessage = "Uploading image and profile data...")
            }

            repository.analyzeWithImage(
                imageFile = tempFile,
                age = age,
                sex = sex,
                region = region,
                restrictions = restrictions,
                variants = variants
            ).onSuccess { jobResponse ->
                // Clean up temp file after successful upload
                tempFile.delete()

                val jobId = jobResponse.jobId
                _uiState.update {
                    it.copy(
                        jobId = jobId,
                        jobStatus = "processing",
                        statusMessage = jobResponse.message,
                        progressPct = jobResponse.progressPct.toInt()
                    )
                }

                // Save jobId to SavedStateHandle for process-death recovery
                savedStateHandle["jobId"] = jobId

                pollJobStatus(jobId)

            }.onFailure { throwable ->
                tempFile.delete()
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        jobStatus = "failed",
                        error = throwable.message
                            ?: "Failed to submit analysis. Please try again."
                    )
                }
            }
        }
    }

    /**
     * Validate an image before analysis submission.
     *
     * Checks format, dimensions, and classifies the image type (face
     * photo vs. microscopy) on the server side.
     */
    fun validateImage(uri: Uri, context: Context) {
        viewModelScope.launch {
            val tempFile = uriToTempFile(uri, context)
            if (tempFile == null) {
                _uiState.update {
                    it.copy(
                        imageValidation = null,
                        error = "Unable to read the selected image for validation."
                    )
                }
                return@launch
            }

            repository.validateImage(tempFile)
                .onSuccess { validation ->
                    tempFile.delete()
                    _uiState.update { it.copy(imageValidation = validation) }
                }
                .onFailure { throwable ->
                    tempFile.delete()
                    _uiState.update {
                        it.copy(
                            imageValidation = null,
                            error = "Image validation failed: ${throwable.message}"
                        )
                    }
                }
        }
    }

    /** Clear the current error message. */
    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    /** Reset the entire UI state to its initial values. */
    fun resetState() {
        pollingJob?.cancel()
        pollingJob = null
        _uiState.value = AnalysisUiState()
    }

    /** Clear the cached result so the screen can be re-used. */
    fun clearResult() {
        _uiState.update { it.copy(result = null) }
    }

    /** Cancel a running analysis and stop polling. */
    fun cancelAnalysis() {
        pollingJob?.cancel()
        pollingJob = null
        _uiState.update {
            it.copy(
                isLoading = false,
                jobStatus = "cancelled",
                statusMessage = "Analysis cancelled"
            )
        }
    }

    /**
     * Public entry point to fetch completed results for the given [jobId].
     *
     * Used by the results screen when navigated to directly (e.g. deep link
     * or when the ViewModel instance does not already carry a cached result).
     */
    fun loadResults(jobId: String) {
        if (_uiState.value.isLoading) return
        _uiState.update { it.copy(isLoading = true, error = null) }
        viewModelScope.launch {
            fetchResults(jobId)
        }
    }

    // ---------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------

    /**
     * Poll the backend for job progress every [POLL_INTERVAL_MS]
     * milliseconds until the job completes, fails, or the maximum
     * number of iterations is reached.
     */
    private fun pollJobStatus(jobId: String) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            var iteration = 0

            while (iteration < MAX_POLL_ITERATIONS) {
                delay(POLL_INTERVAL_MS)
                iteration++

                repository.pollStatus(jobId)
                    .onSuccess { status ->
                        val progressPct = status.progressPct.toInt().coerceIn(0, 100)

                        _uiState.update {
                            it.copy(
                                progressPct = progressPct,
                                statusMessage = status.message,
                                jobStatus = status.status
                            )
                        }

                        when (status.status) {
                            "completed" -> {
                                fetchResults(jobId)
                                return@launch
                            }
                            "failed" -> {
                                _uiState.update {
                                    it.copy(
                                        isLoading = false,
                                        jobStatus = "failed",
                                        error = status.message.ifBlank {
                                            "Analysis failed on the server."
                                        }
                                    )
                                }
                                return@launch
                            }
                        }
                    }
                    .onFailure {
                        // Transient network errors during polling are not fatal;
                        // keep trying until the iteration limit is reached.
                        _uiState.update {
                            it.copy(statusMessage = "Reconnecting to server...")
                        }
                    }
            }

            // Exceeded maximum poll iterations -- treat as timeout
            _uiState.update {
                it.copy(
                    isLoading = false,
                    jobStatus = "failed",
                    error = "Analysis timed out after ${MAX_POLL_ITERATIONS * POLL_INTERVAL_MS / 1000} seconds. Please try again."
                )
            }
        }
    }

    /**
     * Fetch the completed analysis results and update UI state.
     */
    private suspend fun fetchResults(jobId: String) {
        repository.getResults(jobId)
            .onSuccess { analysisResult ->
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        jobStatus = "completed",
                        progressPct = 100,
                        statusMessage = "Analysis complete",
                        result = analysisResult
                    )
                }
            }
            .onFailure { throwable ->
                _uiState.update {
                    it.copy(
                        isLoading = false,
                        jobStatus = "failed",
                        error = "Failed to retrieve results: ${throwable.message}"
                    )
                }
            }
    }

    /**
     * Copy a content [Uri] into a temporary file using the application
     * context's cache directory.  Returns the temp [File], or null if
     * the URI cannot be opened.
     */
    private fun uriToTempFile(uri: Uri, context: Context): File? {
        val inputStream = context.contentResolver.openInputStream(uri) ?: return null

        val fileName = "teloscopy_upload_${System.currentTimeMillis()}.tmp"
        val tempFile = File(context.cacheDir, fileName)

        inputStream.use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        }

        return tempFile
    }

    override fun onCleared() {
        super.onCleared()
        pollingJob?.cancel()
    }
}
