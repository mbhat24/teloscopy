package com.teloscopy.app.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.Healing
import androidx.compose.material.icons.filled.Restaurant
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.teloscopy.app.ui.components.DiseaseRiskCard
import com.teloscopy.app.ui.components.MealPlanCard
import com.teloscopy.app.ui.components.SectionHeader
import com.teloscopy.app.ui.components.StatCard
import com.teloscopy.app.viewmodel.ProfileViewModel

// ── Available regions for the dropdown ─────────────────────────────────────
private val REGIONS = listOf(
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

// ── Dietary restriction options ────────────────────────────────────────────
private val DIETARY_RESTRICTIONS = listOf(
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

/**
 * Profile-only analysis screen.
 *
 * Presents a form for entering user profile data (age, sex, region,
 * genetic variants, dietary restrictions, optional telomere length)
 * and submits it for disease-risk and/or nutrition analysis without
 * requiring an image upload.
 *
 * @param viewModel The [ProfileViewModel] that manages state and API calls.
 * @param onBack    Callback invoked when the user presses the back arrow.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun ProfileAnalysisScreen(
    viewModel: ProfileViewModel = hiltViewModel(),
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val snackbarHostState = remember { SnackbarHostState() }
    val scrollState = rememberScrollState()

    // ── Form state ─────────────────────────────────────────────────────
    var age by remember { mutableStateOf("") }
    var selectedSex by remember { mutableStateOf("Male") }
    var selectedRegion by remember { mutableStateOf(REGIONS.first()) }
    var regionDropdownExpanded by remember { mutableStateOf(false) }
    var knownVariants by remember { mutableStateOf("") }
    val selectedRestrictions = remember { mutableStateListOf<String>() }
    var telomereLength by remember { mutableStateOf("") }
    var includeDiseaseRisk by remember { mutableStateOf(true) }
    var includeNutrition by remember { mutableStateOf(true) }

    // ── Error snackbar ─────────────────────────────────────────────────
    LaunchedEffect(uiState.error) {
        uiState.error?.let { errorMsg ->
            snackbarHostState.showSnackbar(errorMsg)
            viewModel.clearError()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Profile Analysis",
                        style = MaterialTheme.typography.titleLarge.copy(
                            fontWeight = FontWeight.Bold
                        )
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface,
                    navigationIconContentColor = MaterialTheme.colorScheme.onSurface
                )
            )
        },
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState) { data ->
                Snackbar(
                    snackbarData = data,
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer
                )
            }
        },
        containerColor = MaterialTheme.colorScheme.background
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                // ────────────────────────────────────────────────────────
                // FORM CARD
                // ────────────────────────────────────────────────────────
                ElevatedCard(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.elevatedCardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    ),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        // ── Age ─────────────────────────────────────────
                        OutlinedTextField(
                            value = age,
                            onValueChange = { input ->
                                if (input.all { it.isDigit() } && input.length <= 3) {
                                    age = input
                                }
                            },
                            label = { Text("Age") },
                            placeholder = { Text("Enter your age") },
                            keyboardOptions = KeyboardOptions(
                                keyboardType = KeyboardType.Number
                            ),
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = MaterialTheme.colorScheme.primary,
                                unfocusedBorderColor = MaterialTheme.colorScheme.outline,
                                focusedLabelColor = MaterialTheme.colorScheme.primary,
                                cursorColor = MaterialTheme.colorScheme.primary
                            )
                        )

                        // ── Sex ─────────────────────────────────────────
                        Text(
                            text = "Sex",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            listOf("Male", "Female", "Other").forEach { sex ->
                                FilterChip(
                                    selected = selectedSex == sex,
                                    onClick = { selectedSex = sex },
                                    label = { Text(sex) },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = MaterialTheme.colorScheme.primary.copy(
                                            alpha = 0.2f
                                        ),
                                        selectedLabelColor = MaterialTheme.colorScheme.primary
                                    )
                                )
                            }
                        }

                        // ── Region dropdown ─────────────────────────────
                        ExposedDropdownMenuBox(
                            expanded = regionDropdownExpanded,
                            onExpandedChange = { regionDropdownExpanded = it }
                        ) {
                            OutlinedTextField(
                                value = selectedRegion,
                                onValueChange = {},
                                readOnly = true,
                                label = { Text("Region") },
                                trailingIcon = {
                                    ExposedDropdownMenuDefaults.TrailingIcon(
                                        expanded = regionDropdownExpanded
                                    )
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .menuAnchor(MenuAnchorType.PrimaryNotEditable),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                                    unfocusedBorderColor = MaterialTheme.colorScheme.outline,
                                    focusedLabelColor = MaterialTheme.colorScheme.primary
                                )
                            )
                            ExposedDropdownMenu(
                                expanded = regionDropdownExpanded,
                                onDismissRequest = { regionDropdownExpanded = false }
                            ) {
                                REGIONS.forEach { region ->
                                    DropdownMenuItem(
                                        text = { Text(region) },
                                        onClick = {
                                            selectedRegion = region
                                            regionDropdownExpanded = false
                                        },
                                        contentPadding = ExposedDropdownMenuDefaults.ItemContentPadding
                                    )
                                }
                            }
                        }

                        // ── Known variants ──────────────────────────────
                        OutlinedTextField(
                            value = knownVariants,
                            onValueChange = { knownVariants = it },
                            label = { Text("Known Variants") },
                            placeholder = { Text("e.g. rs429358:CT, rs7412:CC") },
                            singleLine = false,
                            minLines = 2,
                            maxLines = 4,
                            modifier = Modifier.fillMaxWidth(),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = MaterialTheme.colorScheme.primary,
                                unfocusedBorderColor = MaterialTheme.colorScheme.outline,
                                focusedLabelColor = MaterialTheme.colorScheme.primary,
                                cursorColor = MaterialTheme.colorScheme.primary
                            )
                        )

                        // ── Dietary restrictions ────────────────────────
                        Text(
                            text = "Dietary Restrictions",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            DIETARY_RESTRICTIONS.forEach { restriction ->
                                val isSelected = restriction in selectedRestrictions
                                FilterChip(
                                    selected = isSelected,
                                    onClick = {
                                        if (isSelected) {
                                            selectedRestrictions.remove(restriction)
                                        } else {
                                            selectedRestrictions.add(restriction)
                                        }
                                    },
                                    label = { Text(restriction) },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = MaterialTheme.colorScheme.tertiary.copy(
                                            alpha = 0.2f
                                        ),
                                        selectedLabelColor = MaterialTheme.colorScheme.tertiary
                                    )
                                )
                            }
                        }

                        // ── Telomere length ─────────────────────────────
                        OutlinedTextField(
                            value = telomereLength,
                            onValueChange = { input ->
                                // Allow digits and a single decimal point
                                if (input.isEmpty() ||
                                    input.matches(Regex("^\\d*\\.?\\d*$"))
                                ) {
                                    telomereLength = input
                                }
                            },
                            label = { Text("Telomere Length") },
                            placeholder = { Text("kb, if known") },
                            supportingText = { Text("Optional — measured telomere length in kilobases") },
                            keyboardOptions = KeyboardOptions(
                                keyboardType = KeyboardType.Decimal
                            ),
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = MaterialTheme.colorScheme.primary,
                                unfocusedBorderColor = MaterialTheme.colorScheme.outline,
                                focusedLabelColor = MaterialTheme.colorScheme.primary,
                                cursorColor = MaterialTheme.colorScheme.primary
                            )
                        )

                        // ── Include Disease Risk switch ─────────────────
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "Include Disease Risk",
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = "Assess genetic disease risk factors",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Switch(
                                checked = includeDiseaseRisk,
                                onCheckedChange = { includeDiseaseRisk = it },
                                colors = SwitchDefaults.colors(
                                    checkedThumbColor = MaterialTheme.colorScheme.primary,
                                    checkedTrackColor = MaterialTheme.colorScheme.primary.copy(
                                        alpha = 0.3f
                                    )
                                )
                            )
                        }

                        // ── Include Nutrition switch ────────────────────
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "Include Nutrition",
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Text(
                                    text = "Generate personalised diet recommendations",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Switch(
                                checked = includeNutrition,
                                onCheckedChange = { includeNutrition = it },
                                colors = SwitchDefaults.colors(
                                    checkedThumbColor = MaterialTheme.colorScheme.primary,
                                    checkedTrackColor = MaterialTheme.colorScheme.primary.copy(
                                        alpha = 0.3f
                                    )
                                )
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // ────────────────────────────────────────────────────────
                // ANALYZE BUTTON
                // ────────────────────────────────────────────────────────
                Button(
                    onClick = {
                        val parsedAge = age.toIntOrNull() ?: 0
                        val variantList = knownVariants
                            .split(",")
                            .map { it.trim() }
                            .filter { it.isNotEmpty() }
                        val telLen = telomereLength.toDoubleOrNull()

                        viewModel.analyzeProfile(
                            age = parsedAge,
                            sex = selectedSex,
                            region = selectedRegion,
                            restrictions = selectedRestrictions.toList(),
                            variants = variantList,
                            telomereLength = telLen,
                            includeNutrition = includeNutrition,
                            includeDiseaseRisk = includeDiseaseRisk
                        )
                    },
                    enabled = !uiState.isLoading &&
                            age.isNotEmpty() &&
                            (includeDiseaseRisk || includeNutrition),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                        disabledContainerColor = MaterialTheme.colorScheme.outline.copy(
                            alpha = 0.3f
                        )
                    )
                ) {
                    if (uiState.isLoading) {
                        CircularProgressIndicator(
                            color = MaterialTheme.colorScheme.onPrimary,
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .height(20.dp)
                                .width(20.dp),
                            strokeWidth = 2.dp
                        )
                    }
                    Icon(
                        imageVector = Icons.Filled.Analytics,
                        contentDescription = null,
                        modifier = Modifier.padding(end = 8.dp)
                    )
                    Text(
                        text = if (uiState.isLoading) "Analyzing…" else "Analyze Profile",
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontWeight = FontWeight.Bold
                        )
                    )
                }

                // ────────────────────────────────────────────────────────
                // LOADING OVERLAY
                // ────────────────────────────────────────────────────────
                AnimatedVisibility(
                    visible = uiState.isLoading,
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 32.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(
                                color = MaterialTheme.colorScheme.primary
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = "Running analysis on your profile…",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                // ────────────────────────────────────────────────────────
                // RESULTS SECTION
                // ────────────────────────────────────────────────────────
                val result = uiState.result
                if (result != null && !uiState.isLoading) {
                    Spacer(modifier = Modifier.height(24.dp))

                    // ── Disease Risk Results ────────────────────────────
                    val diseaseRisks = uiState.diseaseRiskResult
                    if (diseaseRisks != null) {
                        SectionHeader(
                            title = "Disease Risk Assessment",
                            icon = Icons.Filled.Healing,
                            color = MaterialTheme.colorScheme.secondary
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        // Overall risk score stat card
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center
                        ) {
                            StatCard(
                                label = "Overall Risk Score",
                                value = String.format(
                                    "%.1f",
                                    diseaseRisks.overallRiskScore
                                ),
                                unit = "/ 10",
                                color = MaterialTheme.colorScheme.secondary,
                                modifier = Modifier.width(180.dp)
                            )
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        // Individual disease risk cards
                        diseaseRisks.risks.forEach { risk ->
                            DiseaseRiskCard(
                                risk = risk,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                        }

                        // Disclaimer
                        if (diseaseRisks.disclaimer.isNotBlank()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            DisclaimerRow(text = diseaseRisks.disclaimer)
                        }

                        Spacer(modifier = Modifier.height(24.dp))
                    }

                    // ── Nutrition Results ────────────────────────────────
                    val nutrition = uiState.nutritionResult
                    if (nutrition != null) {
                        SectionHeader(
                            title = "Nutrition Plan",
                            icon = Icons.Filled.Restaurant,
                            color = MaterialTheme.colorScheme.tertiary
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        val recommendation = nutrition.recommendation

                        // Summary
                        Text(
                            text = recommendation.summary,
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )

                        // Calorie target stat card (if present)
                        recommendation.calorieTarget?.let { cal ->
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.Center
                            ) {
                                StatCard(
                                    label = "Daily Calorie Target",
                                    value = cal.toString(),
                                    unit = "kcal",
                                    color = MaterialTheme.colorScheme.tertiary,
                                    modifier = Modifier.width(200.dp)
                                )
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                        }

                        // Key nutrients
                        if (recommendation.keyNutrients.isNotEmpty()) {
                            Text(
                                text = "Key Nutrients",
                                style = MaterialTheme.typography.titleSmall.copy(
                                    fontWeight = FontWeight.Bold
                                ),
                                color = MaterialTheme.colorScheme.tertiary
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                recommendation.keyNutrients.forEach { nutrient ->
                                    FilterChip(
                                        selected = true,
                                        onClick = {},
                                        label = { Text(nutrient) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.tertiary.copy(
                                                alpha = 0.15f
                                            ),
                                            selectedLabelColor = MaterialTheme.colorScheme.tertiary
                                        )
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                        }

                        // Foods to increase
                        if (recommendation.foodsToIncrease.isNotEmpty()) {
                            Text(
                                text = "Foods to Increase",
                                style = MaterialTheme.typography.titleSmall.copy(
                                    fontWeight = FontWeight.Bold
                                ),
                                color = MaterialTheme.colorScheme.tertiary
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            recommendation.foodsToIncrease.forEach { food ->
                                Row(
                                    modifier = Modifier.padding(
                                        start = 8.dp,
                                        top = 2.dp
                                    )
                                ) {
                                    Text(
                                        text = "▲",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.tertiary,
                                        modifier = Modifier.padding(end = 8.dp)
                                    )
                                    Text(
                                        text = food,
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                        }

                        // Foods to avoid
                        if (recommendation.foodsToAvoid.isNotEmpty()) {
                            Text(
                                text = "Foods to Avoid",
                                style = MaterialTheme.typography.titleSmall.copy(
                                    fontWeight = FontWeight.Bold
                                ),
                                color = MaterialTheme.colorScheme.error
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            recommendation.foodsToAvoid.forEach { food ->
                                Row(
                                    modifier = Modifier.padding(
                                        start = 8.dp,
                                        top = 2.dp
                                    )
                                ) {
                                    Text(
                                        text = "▼",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.error,
                                        modifier = Modifier.padding(end = 8.dp)
                                    )
                                    Text(
                                        text = food,
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                        }

                        // Meal plans
                        if (recommendation.mealPlans.isNotEmpty()) {
                            Text(
                                text = "Meal Plans",
                                style = MaterialTheme.typography.titleSmall.copy(
                                    fontWeight = FontWeight.Bold
                                ),
                                color = MaterialTheme.colorScheme.tertiary
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            recommendation.mealPlans.forEach { mealPlan ->
                                MealPlanCard(
                                    mealPlan = mealPlan,
                                    modifier = Modifier.padding(vertical = 4.dp)
                                )
                            }
                        }

                        // Disclaimer
                        if (nutrition.disclaimer.isNotBlank()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            DisclaimerRow(text = nutrition.disclaimer)
                        }
                    }

                    // Overall disclaimer from the response
                    if (result.disclaimer.isNotBlank()) {
                        Spacer(modifier = Modifier.height(12.dp))
                        DisclaimerRow(text = result.disclaimer)
                    }

                    // Bottom spacer for scroll clearance
                    Spacer(modifier = Modifier.height(32.dp))
                }
            }
        }
    }
}

/**
 * A small disclaimer row with a warning icon and muted text.
 */
@Composable
private fun DisclaimerRow(text: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp),
        verticalAlignment = Alignment.Top
    ) {
        Icon(
            imageVector = Icons.Filled.Warning,
            contentDescription = "Disclaimer",
            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
            modifier = Modifier
                .padding(top = 2.dp, end = 8.dp)
                .height(16.dp)
                .width(16.dp)
        )
        Text(
            text = text,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
        )
    }
}
