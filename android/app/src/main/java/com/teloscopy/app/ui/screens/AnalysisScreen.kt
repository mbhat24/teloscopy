package com.teloscopy.app.ui.screens

import android.Manifest
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.teloscopy.app.viewmodel.AnalysisViewModel
import java.io.File

// ── Region options ───────────────────────────────────────────────────────────
private val regionOptions = listOf(
    "Northern Europe",
    "Southern Europe",
    "East Asia",
    "South Asia",
    "West Africa",
    "East Africa",
    "Middle East",
    "Central America",
    "South America",
    "Oceania"
)

// ── Dietary restriction options ──────────────────────────────────────────────
private val dietaryOptions = listOf(
    "Vegetarian",
    "Vegan",
    "Pescatarian",
    "Gluten-Free",
    "Lactose-Free",
    "Low-Sodium",
    "Halal",
    "Kosher",
    "Diabetic-Friendly"
)

// ── Sex options ──────────────────────────────────────────────────────────────
private val sexOptions = listOf("Male", "Female", "Other")

// ═════════════════════════════════════════════════════════════════════════════
// AnalysisScreen
// ═════════════════════════════════════════════════════════════════════════════

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun AnalysisScreen(
    viewModel: AnalysisViewModel = hiltViewModel(),
    onNavigateToResults: (String) -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val snackbarHostState = remember { SnackbarHostState() }

    // ── Local form state ─────────────────────────────────────────────────────
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var age by remember { mutableStateOf("") }
    var selectedSex by remember { mutableStateOf("") }
    var selectedRegion by remember { mutableStateOf("") }
    var variants by remember { mutableStateOf("") }
    var selectedRestrictions by remember { mutableStateOf(setOf<String>()) }

    // ── ViewModel state ──────────────────────────────────────────────────────
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    // ── Region dropdown expanded state ───────────────────────────────────────
    var regionDropdownExpanded by remember { mutableStateOf(false) }

    // ── Camera temp-file URI ─────────────────────────────────────────────────
    val cameraImageUri = remember {
        val tempFile = File.createTempFile(
            "camera_capture_",
            ".jpg",
            context.cacheDir
        )
        FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            tempFile
        )
    }

    // ── Camera permission + capture launchers ────────────────────────────────
    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            selectedImageUri = cameraImageUri
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            takePictureLauncher.launch(cameraImageUri)
        }
    }

    // ── Gallery picker launcher ──────────────────────────────────────────────
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { selectedImageUri = it }
    }

    // ── Auto-navigate to results when analysis completes ─────────────────────
    LaunchedEffect(uiState.result) {
        val result = uiState.result
        if (result != null) {
            val navId = uiState.jobId ?: result.jobId
            viewModel.clearResult()
            onNavigateToResults(navId)
        }
    }

    // ── Show Snackbar on error ───────────────────────────────────────────────
    LaunchedEffect(uiState.error) {
        val errorMsg = uiState.error
        if (errorMsg != null) {
            val snackResult = snackbarHostState.showSnackbar(
                message = errorMsg,
                actionLabel = "Retry",
                duration = SnackbarDuration.Long
            )
            if (snackResult == SnackbarResult.ActionPerformed) {
                viewModel.clearError()
            }
        }
    }

    // ── Form validation ──────────────────────────────────────────────────────
    val ageValue = age.toIntOrNull()
    val isAgeValid = ageValue != null && ageValue in 1..120
    val canSubmit = selectedImageUri != null && age.isNotBlank() && isAgeValid

    // ═════════════════════════════════════════════════════════════════════════
    // UI
    // ═════════════════════════════════════════════════════════════════════════

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Facial Analysis",
                        style = MaterialTheme.typography.titleLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        },
        containerColor = MaterialTheme.colorScheme.background
    ) { innerPadding ->

        Box(modifier = Modifier.fillMaxSize()) {

            // ── Main scrollable content ──────────────────────────────────
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // ── Section 1: Image Upload ──────────────────────────────
                item {
                    Spacer(modifier = Modifier.height(8.dp))
                    SectionHeader(title = "Upload Photo")
                }

                item {
                    ImageUploadArea(
                        selectedImageUri = selectedImageUri,
                        onTap = { galleryLauncher.launch("image/*") }
                    )
                }

                item {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        OutlinedButton(
                            onClick = {
                                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                            },
                            modifier = Modifier.weight(1f),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.primary
                            )
                        ) {
                            Icon(
                                imageVector = Icons.Filled.AddAPhoto,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Take Photo")
                        }

                        OutlinedButton(
                            onClick = { galleryLauncher.launch("image/*") },
                            modifier = Modifier.weight(1f),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.primary
                            )
                        ) {
                            Icon(
                                imageVector = Icons.Filled.PhotoLibrary,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Choose Gallery")
                        }
                    }
                }

                // ── Section 2: Profile Form ──────────────────────────────
                item {
                    SectionHeader(title = "Profile Information")
                }

                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surface
                        ),
                        shape = RoundedCornerShape(16.dp),
                        border = BorderStroke(
                            1.dp,
                            MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                        )
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            // ── Age ──────────────────────────────────────
                            OutlinedTextField(
                                value = age,
                                onValueChange = { newValue ->
                                    // Allow only digits, max 3 chars
                                    if (newValue.length <= 3 && newValue.all { it.isDigit() }) {
                                        age = newValue
                                    }
                                },
                                label = { Text("Age") },
                                placeholder = { Text("Enter your age") },
                                keyboardOptions = KeyboardOptions(
                                    keyboardType = KeyboardType.Number
                                ),
                                singleLine = true,
                                isError = age.isNotBlank() && !isAgeValid,
                                supportingText = if (age.isNotBlank() && !isAgeValid) {
                                    { Text("Age must be between 1 and 120") }
                                } else {
                                    null
                                },
                                modifier = Modifier.fillMaxWidth(),
                                colors = analysisTextFieldColors()
                            )

                            // ── Sex ──────────────────────────────────────
                            Text(
                                text = "Sex",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurface,
                                fontWeight = FontWeight.Medium
                            )

                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                sexOptions.forEach { option ->
                                    FilterChip(
                                        selected = selectedSex == option,
                                        onClick = { selectedSex = option },
                                        label = { Text(option) },
                                        colors = analysisFilterChipColors()
                                    )
                                }
                            }

                            // ── Region ───────────────────────────────────
                            ExposedDropdownMenuBox(
                                expanded = regionDropdownExpanded,
                                onExpandedChange = { regionDropdownExpanded = it }
                            ) {
                                OutlinedTextField(
                                    value = selectedRegion,
                                    onValueChange = {},
                                    readOnly = true,
                                    label = { Text("Region / Ancestry") },
                                    placeholder = { Text("Select region") },
                                    trailingIcon = {
                                        ExposedDropdownMenuDefaults.TrailingIcon(
                                            expanded = regionDropdownExpanded
                                        )
                                    },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .menuAnchor(MenuAnchorType.PrimaryNotEditable),
                                    colors = analysisTextFieldColors()
                                )

                                ExposedDropdownMenu(
                                    expanded = regionDropdownExpanded,
                                    onDismissRequest = { regionDropdownExpanded = false }
                                ) {
                                    regionOptions.forEach { region ->
                                        DropdownMenuItem(
                                            text = {
                                                Text(
                                                    text = region,
                                                    color = MaterialTheme.colorScheme.onSurface
                                                )
                                            },
                                            onClick = {
                                                selectedRegion = region
                                                regionDropdownExpanded = false
                                            },
                                            contentPadding = ExposedDropdownMenuDefaults.ItemContentPadding
                                        )
                                    }
                                }
                            }

                            // ── Known Variants ───────────────────────────
                            OutlinedTextField(
                                value = variants,
                                onValueChange = { variants = it },
                                label = { Text("Known Variants") },
                                placeholder = { Text("rs429358:CT, rs7412:CC") },
                                singleLine = false,
                                minLines = 2,
                                maxLines = 4,
                                modifier = Modifier.fillMaxWidth(),
                                colors = analysisTextFieldColors()
                            )

                            // ── Dietary Restrictions ─────────────────────
                            Text(
                                text = "Dietary Restrictions",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurface,
                                fontWeight = FontWeight.Medium
                            )

                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                dietaryOptions.forEach { restriction ->
                                    val isSelected = restriction in selectedRestrictions
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = {
                                            selectedRestrictions = if (isSelected) {
                                                selectedRestrictions - restriction
                                            } else {
                                                selectedRestrictions + restriction
                                            }
                                        },
                                        label = { Text(restriction) },
                                        colors = analysisFilterChipColors()
                                    )
                                }
                            }
                        }
                    }
                }

                // ── Section 3: Submit Button ─────────────────────────────
                item {
                    Button(
                        onClick = {
                            selectedImageUri?.let { uri ->
                                viewModel.submitAnalysis(
                                    imageUri = uri,
                                    context = context,
                                    age = ageValue ?: 0,
                                    sex = selectedSex,
                                    region = selectedRegion,
                                    restrictions = selectedRestrictions.toList(),
                                    variants = variants.split(",").map { it.trim() }.filter { it.isNotEmpty() }
                                )
                            }
                        },
                        enabled = canSubmit && !uiState.isLoading,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary,
                            contentColor = MaterialTheme.colorScheme.onPrimary,
                            disabledContainerColor = MaterialTheme.colorScheme.primary
                                .copy(alpha = 0.3f),
                            disabledContentColor = MaterialTheme.colorScheme.onPrimary
                                .copy(alpha = 0.5f)
                        )
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Science,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Analyze",
                            style = MaterialTheme.typography.titleLarge
                        )
                    }
                }

                // ── Section 4: Error display card ────────────────────────
                if (uiState.error != null) {
                    item {
                        ErrorCard(
                            errorMessage = uiState.error!!,
                            onRetry = { viewModel.clearError() },
                            onDismiss = { viewModel.clearError() }
                        )
                    }
                }

                // Bottom spacing
                item {
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }

            // ── Progress Overlay ─────────────────────────────────────────
            AnimatedVisibility(
                visible = uiState.isLoading,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                ProgressOverlay(
                    progressPct = uiState.progressPct,
                    statusMessage = uiState.statusMessage,
                    onCancel = { viewModel.cancelAnalysis() }
                )
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Sub-composables
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Section header label used to separate form areas.
 */
@Composable
private fun SectionHeader(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.headlineMedium,
        color = MaterialTheme.colorScheme.primary,
        fontWeight = FontWeight.SemiBold
    )
}

/**
 * Dashed-border box that acts as the image preview / tap-to-select area.
 * Shows the selected image via Coil [AsyncImage] or a placeholder prompt.
 */
@Composable
private fun ImageUploadArea(
    selectedImageUri: Uri?,
    onTap: () -> Unit
) {
    val borderColor = MaterialTheme.colorScheme.outline
    val dashedStroke = remember {
        Stroke(
            width = 4f,
            pathEffect = PathEffect.dashPathEffect(floatArrayOf(20f, 10f), 0f)
        )
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surface)
            .clickable(onClick = onTap),
        contentAlignment = Alignment.Center
    ) {
        // Draw dashed border as canvas overlay
        Canvas(modifier = Modifier.fillMaxSize()) {
            drawRoundRect(
                color = borderColor,
                style = dashedStroke,
                cornerRadius = CornerRadius(16.dp.toPx())
            )
        }

        if (selectedImageUri != null) {
            AsyncImage(
                model = ImageRequest.Builder(LocalContext.current)
                    .data(selectedImageUri)
                    .crossfade(true)
                    .build(),
                contentDescription = "Selected face photo",
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(16.dp))
            )
        } else {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Image,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Tap to select image",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                )
            }
        }
    }
}

/**
 * Semi-transparent overlay displayed during analysis processing.
 * Shows a [LinearProgressIndicator], status text, and a cancel button.
 */
@Composable
private fun ProgressOverlay(
    progressPct: Int,
    statusMessage: String,
    onCancel: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.75f))
            .clickable(enabled = false) { /* consume taps */ },
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(32.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            shape = RoundedCornerShape(20.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "Analyzing...",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold
                )

                LinearProgressIndicator(
                    progress = { progressPct / 100f },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = MaterialTheme.colorScheme.primary,
                    trackColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                )

                Text(
                    text = "$progressPct%",
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = statusMessage,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
                )

                OutlinedButton(
                    onClick = onCancel,
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.error),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Icon(
                        imageVector = Icons.Filled.Cancel,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Cancel")
                }
            }
        }
    }
}

/**
 * Error card with the error message, a retry button, and a dismiss action.
 */
@Composable
private fun ErrorCard(
    errorMessage: String,
    onRetry: () -> Unit,
    onDismiss: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.Warning,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onErrorContainer,
                    modifier = Modifier.size(24.dp)
                )
                Text(
                    text = "Analysis Error",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    fontWeight = FontWeight.SemiBold
                )
            }

            Text(
                text = errorMessage,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onErrorContainer
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = onDismiss,
                    border = BorderStroke(
                        1.dp,
                        MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.5f)
                    ),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.onErrorContainer
                    )
                ) {
                    Text("Dismiss")
                }

                Spacer(modifier = Modifier.width(8.dp))

                Button(
                    onClick = onRetry,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                        contentColor = MaterialTheme.colorScheme.onError
                    )
                ) {
                    Text("Retry")
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Shared style helpers
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Consistent [OutlinedTextField] colours for the dark analysis theme.
 */
@Composable
private fun analysisTextFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedTextColor = MaterialTheme.colorScheme.onSurface,
    unfocusedTextColor = MaterialTheme.colorScheme.onSurface,
    focusedBorderColor = MaterialTheme.colorScheme.primary,
    unfocusedBorderColor = MaterialTheme.colorScheme.outline,
    cursorColor = MaterialTheme.colorScheme.primary,
    focusedLabelColor = MaterialTheme.colorScheme.primary,
    unfocusedLabelColor = MaterialTheme.colorScheme.onSurfaceVariant,
    focusedPlaceholderColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
    unfocusedPlaceholderColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
    errorBorderColor = MaterialTheme.colorScheme.error,
    errorLabelColor = MaterialTheme.colorScheme.error,
    errorSupportingTextColor = MaterialTheme.colorScheme.error
)

/**
 * Consistent [FilterChip] colours matching the cyan-on-dark theme.
 */
@Composable
private fun analysisFilterChipColors() = FilterChipDefaults.filterChipColors(
    containerColor = MaterialTheme.colorScheme.surface,
    labelColor = MaterialTheme.colorScheme.onSurfaceVariant,
    selectedContainerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.2f),
    selectedLabelColor = MaterialTheme.colorScheme.primary
)
