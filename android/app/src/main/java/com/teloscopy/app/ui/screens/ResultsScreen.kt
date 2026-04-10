package com.teloscopy.app.ui.screens

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.Biotech
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Science
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material.icons.outlined.Face
import androidx.compose.material.icons.outlined.Restaurant
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.teloscopy.app.data.api.AnalysisResponse
import com.teloscopy.app.data.api.AncestryDerivedPredictionsResponse
import com.teloscopy.app.data.api.ConditionScreeningResponse
import com.teloscopy.app.data.api.DermatologicalAnalysisResponse
import com.teloscopy.app.data.api.DietRecommendation
import com.teloscopy.app.data.api.FacialAnalysisResult
import com.teloscopy.app.data.api.FacialHealthScreeningResponse
import com.teloscopy.app.data.api.PharmacogenomicPredictionResponse
import com.teloscopy.app.data.api.ReconstructedDNAResponse
import com.teloscopy.app.data.api.TelomereResult
import com.teloscopy.app.ui.components.DiseaseRiskCard
import com.teloscopy.app.ui.components.MealPlanCard
import com.teloscopy.app.ui.theme.Primary
import com.teloscopy.app.ui.theme.RiskHigh
import com.teloscopy.app.ui.theme.RiskLow
import com.teloscopy.app.ui.theme.RiskModerate
import com.teloscopy.app.ui.theme.Secondary
import com.teloscopy.app.ui.theme.SurfaceVariant
import com.teloscopy.app.ui.theme.Tertiary
import com.teloscopy.app.viewmodel.AnalysisViewModel

// ─────────────────────────────────────────────────────────────────────────────
// ResultsScreen – top-level composable
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Displays the full analysis results after the pipeline completes.
 *
 * @param jobId     The analysis job identifier used to fetch results.
 * @param viewModel Shared [AnalysisViewModel] (Hilt-provided).
 * @param onBack    Callback invoked when the user presses the back arrow.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultsScreen(
    jobId: String,
    viewModel: AnalysisViewModel = hiltViewModel(),
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    // Fetch results if they are not already present (e.g. direct navigation).
    LaunchedEffect(jobId) {
        if (uiState.result == null && !uiState.isLoading) {
            viewModel.loadResults(jobId)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Analysis Results",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
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
                    navigationIconContentColor = MaterialTheme.colorScheme.primary
                )
            )
        },
        containerColor = MaterialTheme.colorScheme.background
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when {
                // Loading – no result yet and no error
                uiState.isLoading && uiState.result == null -> {
                    LoadingContent()
                }
                // Error – no result available
                uiState.error != null && uiState.result == null -> {
                    ErrorContent(
                        error = uiState.error!!,
                        onRetry = { viewModel.loadResults(jobId) }
                    )
                }
                // Result available
                uiState.result != null -> {
                    ResultsContent(result = uiState.result!!)
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading state
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun LoadingContent() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(
                color = MaterialTheme.colorScheme.primary,
                strokeWidth = 3.dp
            )
            Spacer(Modifier.height(16.dp))
            Text(
                text = "Loading results\u2026",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error state
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun ErrorContent(error: String, onRetry: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        ElevatedCard(
            colors = CardDefaults.elevatedCardColors(
                containerColor = MaterialTheme.colorScheme.errorContainer
            )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Outlined.ErrorOutline,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.error,
                    modifier = Modifier.size(48.dp)
                )
                Spacer(Modifier.height(16.dp))
                Text(
                    text = "Something went wrong",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = error,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    textAlign = TextAlign.Center
                )
                Spacer(Modifier.height(20.dp))
                Button(onClick = onRetry) {
                    Text("Retry")
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main results content (LazyColumn)
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResultsContent(result: AnalysisResponse) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // ── (a) Summary card ─────────────────────────────────────────────
        item(key = "summary") {
            SummaryCard(result)
        }

        // ── (b) Telomere results (always present – non-nullable) ─────────
        item(key = "telomere_header") {
            SectionHeader(
                icon = Icons.Outlined.Biotech,
                title = "Telomere Results"
            )
        }
        item(key = "telomere_content") {
            TelomereResultsSection(telomere = result.telomereResults)
        }

        // ── (c) Facial analysis (nullable) ───────────────────────────────
        if (result.facialAnalysis != null) {
            item(key = "facial_header") {
                SectionHeader(
                    icon = Icons.Outlined.Face,
                    title = "Facial Analysis"
                )
            }
            item(key = "facial_content") {
                FacialAnalysisSection(facial = result.facialAnalysis!!)
            }

            // ── (c2) DNA Reconstruction (inside facial analysis) ─────────
            val dna = result.facialAnalysis!!.reconstructedDna
            if (dna != null && dna.sequences.isNotEmpty()) {
                item(key = "dna_header") {
                    SectionHeader(
                        icon = Icons.Outlined.Science,
                        title = "DNA Reconstruction"
                    )
                }
                item(key = "dna_content") {
                    DnaReconstructionSection(dna = dna)
                }
            }

            // ── (c3) Pharmacogenomic Profile ─────────────────────────────
            val facial = result.facialAnalysis!!
            if (facial.pharmacogenomicPredictions.isNotEmpty()) {
                item(key = "pharmacogenomics_content") {
                    PharmacogenomicsSection(predictions = facial.pharmacogenomicPredictions)
                }
            }

            // ── (c4) Health Screening ────────────────────────────────────
            if (facial.healthScreening != null) {
                item(key = "health_screening_content") {
                    HealthScreeningSection(screening = facial.healthScreening!!)
                }
            }

            // ── (c5) Dermatological / Skin Analysis ──────────────────────
            if (facial.dermatologicalAnalysis != null) {
                item(key = "dermatological_content") {
                    DermatologicalAnalysisSection(derm = facial.dermatologicalAnalysis!!)
                }
            }

            // ── (c6) Condition Screening ─────────────────────────────────
            if (facial.conditionScreenings.isNotEmpty()) {
                item(key = "condition_screening_content") {
                    ConditionScreeningSection(screenings = facial.conditionScreenings)
                }
            }

            // ── (c7) Ancestry-Derived Predictions ────────────────────────
            if (facial.ancestryDerived != null) {
                item(key = "ancestry_derived_content") {
                    AncestryDerivedSection(ancestry = facial.ancestryDerived!!)
                }
            }
        }

        // ── (d) Disease risks ────────────────────────────────────────────
        if (result.diseaseRisks.isNotEmpty()) {
            item(key = "disease_header") {
                SectionHeader(
                    icon = Icons.Outlined.Warning,
                    title = "Disease Risks"
                )
            }
            items(
                items = result.diseaseRisks,
                key = { it.disease }
            ) { risk ->
                DiseaseRiskCard(risk = risk)
            }
        }

        // ── (e) Diet recommendations (always present – non-nullable) ─────
        item(key = "diet_header") {
            SectionHeader(
                icon = Icons.Outlined.Restaurant,
                title = "Diet Recommendations"
            )
        }
        item(key = "diet_content") {
            DietRecommendationsSection(diet = result.dietRecommendations)
        }

        // ── (f) Disclaimer ───────────────────────────────────────────────
        item(key = "disclaimer") {
            DisclaimerSection(text = result.disclaimer)
        }

        // Bottom spacer for safe-area / nav-bar clearance
        item(key = "bottom_spacer") {
            Spacer(Modifier.height(32.dp))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section header
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun SectionHeader(icon: ImageVector, title: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(top = 8.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(24.dp)
        )
        Spacer(Modifier.width(10.dp))
        Text(
            text = title,
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onBackground
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (a) Summary card
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun SummaryCard(result: AnalysisResponse) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Image type badge
                val isFacePhoto = result.imageType.lowercase() == "face_photo"
                val badgeColor = if (isFacePhoto) {
                    MaterialTheme.colorScheme.secondary
                } else {
                    MaterialTheme.colorScheme.primary
                }
                val badgeLabel = if (isFacePhoto) "Face Photo" else "FISH Microscopy"

                Surface(
                    color = badgeColor.copy(alpha = 0.15f),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = badgeLabel,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        color = badgeColor,
                        style = MaterialTheme.typography.labelMedium.copy(
                            fontWeight = FontWeight.Bold
                        )
                    )
                }

                // Created-at timestamp
                Text(
                    text = formatTimestamp(result.createdAt),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(Modifier.height(10.dp))

            // Job ID in small muted text
            Text(
                text = "Job ID: ${result.jobId}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) Telomere results section
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun TelomereResultsSection(telomere: TelomereResult) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Biological Age – large prominent display
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.elevatedCardColors(containerColor = SurfaceVariant)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Biological Age Estimate",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "${telomere.biologicalAgeEstimate}",
                    style = MaterialTheme.typography.displayLarge.copy(
                        fontSize = 56.sp,
                        fontWeight = FontWeight.Bold
                    ),
                    color = Tertiary
                )
                Text(
                    text = "years",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // Three stat cards in a row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            StatCard(
                value = String.format("%.2f", telomere.meanLength),
                label = "Mean Length\n(kb)",
                modifier = Modifier.weight(1f)
            )
            StatCard(
                value = String.format("%.2f", telomere.stdDev),
                label = "Std Dev",
                modifier = Modifier.weight(1f)
            )
            StatCard(
                value = String.format("%.2f", telomere.tsRatio),
                label = "T/S Ratio",
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
private fun StatCard(
    value: String,
    label: String,
    modifier: Modifier = Modifier
) {
    ElevatedCard(
        modifier = modifier,
        colors = CardDefaults.elevatedCardColors(containerColor = SurfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                color = Primary
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (c) Facial analysis section
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun FacialAnalysisSection(facial: FacialAnalysisResult) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Grid of info chips
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            InfoChip(
                label = "Bio Age",
                value = "${facial.estimatedBiologicalAge}"
            )
            InfoChip(
                label = "Telomere",
                value = String.format("%.2f kb", facial.estimatedTelomereLengthKb)
            )
            InfoChip(
                label = "Percentile",
                value = "${facial.telomerePercentile}th"
            )
            InfoChip(
                label = "Skin Health",
                value = String.format("%.1f", facial.skinHealthScore)
            )
            InfoChip(
                label = "Oxidative Stress",
                value = String.format("%.1f", facial.oxidativeStressScore)
            )
        }

        HorizontalDivider(
            color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f),
            modifier = Modifier.padding(vertical = 4.dp)
        )

        // Predicted traits row
        Text(
            text = "Predicted Traits",
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            TraitBadge(text = "Eye: ${facial.predictedEyeColour}")
            TraitBadge(text = "Hair: ${facial.predictedHairColour}")
            TraitBadge(text = "Skin: ${facial.predictedSkinType}")
        }
    }
}

@Composable
private fun InfoChip(label: String, value: String) {
    Surface(
        color = SurfaceVariant,
        shape = RoundedCornerShape(10.dp)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge.copy(
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                ),
                color = Primary
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun TraitBadge(text: String) {
    Surface(
        color = MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f),
        shape = RoundedCornerShape(8.dp)
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Medium),
            color = MaterialTheme.colorScheme.secondary
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (c2) DNA Reconstruction section
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DnaReconstructionSection(dna: ReconstructedDNAResponse) {
    val context = LocalContext.current
    var showFasta by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.animateContentSize(),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Summary card
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.elevatedCardColors(containerColor = SurfaceVariant)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "${dna.totalVariants}",
                        style = MaterialTheme.typography.headlineMedium.copy(
                            fontWeight = FontWeight.Bold
                        ),
                        color = Secondary
                    )
                    Text(
                        text = "SNPs",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = dna.genomeBuild,
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontWeight = FontWeight.Bold
                        ),
                        color = Primary
                    )
                    Text(
                        text = "Genome Build",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        // Variant table
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.elevatedCardColors(containerColor = SurfaceVariant)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = "Reconstructed Variants",
                    style = MaterialTheme.typography.titleMedium.copy(
                        fontWeight = FontWeight.SemiBold
                    ),
                    color = MaterialTheme.colorScheme.onSurface
                )
                Spacer(Modifier.height(8.dp))

                // Table header
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                        .background(
                            MaterialTheme.colorScheme.outline.copy(alpha = 0.15f),
                            RoundedCornerShape(6.dp)
                        )
                        .padding(horizontal = 10.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    DnaTableHeaderCell("SNP", Modifier.width(100.dp))
                    DnaTableHeaderCell("Gene", Modifier.width(80.dp))
                    DnaTableHeaderCell("Chr", Modifier.width(50.dp))
                    DnaTableHeaderCell("Genotype", Modifier.width(80.dp))
                    DnaTableHeaderCell("Conf.", Modifier.width(55.dp))
                }

                // Table rows
                dna.sequences.forEach { seq ->
                    val isHomozygous = seq.predictedAllele1 == seq.predictedAllele2
                    val genotypeText = "${seq.predictedAllele1}/${seq.predictedAllele2}"
                    val genotypeColor = if (isHomozygous) {
                        if (seq.predictedAllele1 == seq.refAllele) RiskLow else RiskHigh
                    } else {
                        RiskModerate
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState())
                            .padding(horizontal = 10.dp, vertical = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = seq.rsid,
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Medium
                            ),
                            color = Primary,
                            modifier = Modifier.width(100.dp)
                        )
                        Text(
                            text = seq.gene,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.width(80.dp)
                        )
                        Text(
                            text = seq.chromosome,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.width(50.dp)
                        )
                        Surface(
                            color = genotypeColor.copy(alpha = 0.15f),
                            shape = RoundedCornerShape(4.dp),
                            modifier = Modifier.width(80.dp)
                        ) {
                            Text(
                                text = genotypeText,
                                style = MaterialTheme.typography.bodySmall.copy(
                                    fontFamily = FontFamily.Monospace,
                                    fontWeight = FontWeight.Bold
                                ),
                                color = genotypeColor,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.padding(
                                    horizontal = 8.dp,
                                    vertical = 4.dp
                                )
                            )
                        }
                        Text(
                            text = String.format("%.0f%%", seq.confidence * 100),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.width(55.dp),
                            textAlign = TextAlign.End
                        )
                    }

                    HorizontalDivider(
                        color = MaterialTheme.colorScheme.outline.copy(alpha = 0.15f)
                    )
                }
            }
        }

        // FASTA section
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedButton(
                onClick = { showFasta = !showFasta },
                modifier = Modifier.weight(1f)
            ) {
                Text(if (showFasta) "Hide FASTA" else "Show FASTA")
            }
            OutlinedButton(
                onClick = {
                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE)
                            as ClipboardManager
                    clipboard.setPrimaryClip(
                        ClipData.newPlainText("DNA FASTA", dna.fasta)
                    )
                    Toast.makeText(context, "FASTA copied to clipboard", Toast.LENGTH_SHORT).show()
                }
            ) {
                Icon(
                    imageVector = Icons.Outlined.ContentCopy,
                    contentDescription = "Copy",
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(4.dp))
                Text("Copy")
            }
            OutlinedButton(
                onClick = {
                    val shareIntent = Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        putExtra(Intent.EXTRA_TEXT, dna.fasta)
                        putExtra(Intent.EXTRA_SUBJECT, "Teloscopy DNA Reconstruction")
                    }
                    context.startActivity(Intent.createChooser(shareIntent, "Share FASTA"))
                }
            ) {
                Icon(
                    imageVector = Icons.Outlined.Share,
                    contentDescription = "Share",
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(4.dp))
                Text("Share")
            }
        }

        // Expandable FASTA viewer
        if (showFasta && dna.fasta.isNotBlank()) {
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp)
                        .horizontalScroll(rememberScrollState())
                ) {
                    Text(
                        text = dna.fasta,
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = FontFamily.Monospace,
                            lineHeight = 18.sp,
                            fontSize = 11.sp
                        ),
                        color = Tertiary
                    )
                }
            }
        }

        // DNA disclaimer
        if (dna.disclaimer.isNotBlank()) {
            Text(
                text = dna.disclaimer,
                style = MaterialTheme.typography.labelSmall.copy(
                    fontSize = 10.sp,
                    lineHeight = 14.sp
                ),
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

@Composable
private fun DnaTableHeaderCell(text: String, modifier: Modifier = Modifier) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = modifier
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Risk-score → colour helper (shared by new v2 sections)
// ─────────────────────────────────────────────────────────────────────────────

/** Maps a 0-1 risk score to a traffic-light colour. */
private fun riskColor(score: Double): Color = when {
    score < 0.2  -> RiskLow
    score < 0.5  -> RiskModerate
    score < 0.75 -> Color(0xFFFF9800) // orange
    else         -> RiskHigh
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared composable: coloured progress bar
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun RiskProgressBar(
    label: String,
    score: Double,
    modifier: Modifier = Modifier
) {
    val clampedScore = score.coerceIn(0.0, 1.0)
    val barColor = riskColor(clampedScore)
    Column(modifier = modifier) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = String.format("%.0f%%", clampedScore * 100),
                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Bold),
                color = barColor
            )
        }
        Spacer(Modifier.height(4.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .background(
                    MaterialTheme.colorScheme.outline.copy(alpha = 0.15f),
                    RoundedCornerShape(4.dp)
                )
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(clampedScore.toFloat())
                    .height(8.dp)
                    .background(barColor, RoundedCornerShape(4.dp))
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared composable: expandable card wrapper (matches existing card style)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun ExpandableResultCard(
    title: String,
    icon: ImageVector,
    content: @Composable () -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    ElevatedCard(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.elevatedCardColors(containerColor = SurfaceVariant)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { expanded = !expanded },
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = icon,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(22.dp)
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontWeight = FontWeight.SemiBold
                        ),
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                Text(
                    text = if (expanded) "Hide" else "Show",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            if (expanded) {
                Spacer(Modifier.height(12.dp))
                HorizontalDivider(
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                )
                Spacer(Modifier.height(12.dp))
                content()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared composable: bullet-point list
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun BulletList(
    items: List<String>,
    tintColor: Color = MaterialTheme.colorScheme.onSurface
) {
    Column {
        items.forEach { item ->
            Row(
                modifier = Modifier.padding(vertical = 3.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .background(
                            color = tintColor.copy(alpha = 0.7f),
                            shape = RoundedCornerShape(3.dp)
                        )
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    text = item,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (c3) Pharmacogenomic Profile section
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PharmacogenomicsSection(
    predictions: List<PharmacogenomicPredictionResponse>
) {
    ExpandableResultCard(
        title = "Pharmacogenomic Profile",
        icon = Icons.Outlined.Biotech
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            predictions.forEach { pred ->
                PharmacogenomicPredictionCard(pred)
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PharmacogenomicPredictionCard(pred: PharmacogenomicPredictionResponse) {
    var showRecommendation by remember { mutableStateOf(false) }

    val phenotypeColor = when (pred.predictedPhenotype.lowercase()) {
        "normal metabolizer", "normal"         -> RiskLow
        "intermediate metabolizer", "intermediate" -> RiskModerate
        "poor metabolizer", "poor"             -> RiskHigh
        "ultra-rapid metabolizer", "ultra-rapid"   -> Color(0xFFD32F2F)
        else                                   -> MaterialTheme.colorScheme.onSurface
    }

    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            // Gene + rsid header row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = pred.gene,
                    style = MaterialTheme.typography.titleMedium.copy(
                        fontWeight = FontWeight.Bold
                    ),
                    color = Primary
                )
                Text(
                    text = pred.rsid,
                    style = MaterialTheme.typography.bodySmall.copy(
                        fontFamily = FontFamily.Monospace
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(Modifier.height(6.dp))

            // Phenotype badge
            Surface(
                color = phenotypeColor.copy(alpha = 0.15f),
                shape = RoundedCornerShape(6.dp)
            ) {
                Text(
                    text = pred.predictedPhenotype,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelMedium.copy(
                        fontWeight = FontWeight.Bold
                    ),
                    color = phenotypeColor
                )
            }

            Spacer(Modifier.height(4.dp))

            // Confidence
            Text(
                text = "Confidence: ${String.format("%.0f%%", pred.confidence * 100)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            // Basis
            if (pred.basis.isNotBlank()) {
                Text(
                    text = "Basis: ${pred.basis}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                )
            }

            // Affected drugs as chips
            if (pred.affectedDrugs.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Affected Drugs",
                    style = MaterialTheme.typography.labelSmall.copy(
                        fontWeight = FontWeight.SemiBold
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(4.dp))
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    pred.affectedDrugs.forEach { drug ->
                        Surface(
                            color = Secondary.copy(alpha = 0.12f),
                            shape = RoundedCornerShape(6.dp)
                        ) {
                            Text(
                                text = drug,
                                modifier = Modifier.padding(
                                    horizontal = 10.dp,
                                    vertical = 4.dp
                                ),
                                style = MaterialTheme.typography.labelSmall,
                                color = Secondary
                            )
                        }
                    }
                }
            }

            // Expandable clinical recommendation
            if (pred.clinicalRecommendation.isNotBlank()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = if (showRecommendation) "Hide Recommendation" else "Show Recommendation",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.clickable {
                        showRecommendation = !showRecommendation
                    }
                )
                if (showRecommendation) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = pred.clinicalRecommendation,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (c4) Health Screening section
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun HealthScreeningSection(screening: FacialHealthScreeningResponse) {
    ExpandableResultCard(
        title = "Health Screening",
        icon = Icons.Outlined.Face
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            // BMI category
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Estimated BMI Category",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Column(horizontalAlignment = Alignment.End) {
                        Text(
                            text = screening.estimatedBmiCategory,
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontWeight = FontWeight.Bold
                            ),
                            color = Primary
                        )
                        Text(
                            text = "Confidence: ${String.format("%.0f%%", screening.bmiConfidence * 100)}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // Anemia risk bar
            RiskProgressBar(
                label = "Anemia Risk",
                score = screening.anemiaRiskScore
            )

            // Fatigue / Stress bar
            RiskProgressBar(
                label = "Fatigue / Stress",
                score = screening.fatigueStressScore
            )

            // Hydration bar
            RiskProgressBar(
                label = "Hydration",
                score = screening.hydrationScore / 100.0
            )

            // Cardiovascular risk indicators
            if (screening.cardiovascularRiskIndicators.isNotEmpty()) {
                Text(
                    text = "Cardiovascular Risk Indicators",
                    style = MaterialTheme.typography.labelMedium.copy(
                        fontWeight = FontWeight.SemiBold
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                BulletList(
                    items = screening.cardiovascularRiskIndicators,
                    tintColor = RiskHigh
                )
            }

            // Thyroid indicators
            if (screening.thyroidIndicators.isNotEmpty()) {
                Text(
                    text = "Thyroid Indicators",
                    style = MaterialTheme.typography.labelMedium.copy(
                        fontWeight = FontWeight.SemiBold
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                BulletList(
                    items = screening.thyroidIndicators,
                    tintColor = RiskModerate
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (c5) Dermatological / Skin Analysis section
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun DermatologicalAnalysisSection(derm: DermatologicalAnalysisResponse) {
    ExpandableResultCard(
        title = "Skin Analysis",
        icon = Icons.Outlined.Face
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            // Risk score progress bars in a grid-like layout
            RiskProgressBar(label = "Rosacea Risk", score = derm.rosaceaRiskScore)
            RiskProgressBar(label = "Melasma Risk", score = derm.melasmaRiskScore)
            RiskProgressBar(label = "Acne Severity", score = derm.acneSeverityScore)
            RiskProgressBar(
                label = "Pigmentation Disorder Risk",
                score = derm.pigmentationDisorderRisk
            )

            // Moisture barrier score
            RiskProgressBar(
                label = "Moisture Barrier",
                score = derm.moistureBarrierScore / 100.0
            )

            // Photo-aging gap as +/- years text
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Photo-Aging Gap",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    val gapText = if (derm.photoAgingGap >= 0) {
                        "+${derm.photoAgingGap} years"
                    } else {
                        "${derm.photoAgingGap} years"
                    }
                    val gapColor = if (derm.photoAgingGap > 0) RiskHigh else RiskLow
                    Text(
                        text = gapText,
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontWeight = FontWeight.Bold
                        ),
                        color = gapColor
                    )
                }
            }

            // Skin cancer risk factors
            if (derm.skinCancerRiskFactors.isNotEmpty()) {
                Text(
                    text = "Skin Cancer Risk Factors",
                    style = MaterialTheme.typography.labelMedium.copy(
                        fontWeight = FontWeight.SemiBold
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                BulletList(
                    items = derm.skinCancerRiskFactors,
                    tintColor = RiskHigh
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (c6) Condition Screening section
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ConditionScreeningSection(
    screenings: List<ConditionScreeningResponse>
) {
    ExpandableResultCard(
        title = "Condition Screening",
        icon = Icons.Outlined.Warning
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            screenings.forEach { screening ->
                ConditionScreeningCard(screening)
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ConditionScreeningCard(screening: ConditionScreeningResponse) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            // Condition name
            Text(
                text = screening.condition,
                style = MaterialTheme.typography.titleMedium.copy(
                    fontWeight = FontWeight.Bold
                ),
                color = MaterialTheme.colorScheme.onSurface
            )

            Spacer(Modifier.height(8.dp))

            // Risk bar
            RiskProgressBar(label = "Risk Score", score = screening.riskScore)

            Spacer(Modifier.height(4.dp))

            // Confidence
            Text(
                text = "Confidence: ${String.format("%.0f%%", screening.confidence * 100)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            // Facial markers
            if (screening.facialMarkers.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Facial Markers",
                    style = MaterialTheme.typography.labelSmall.copy(
                        fontWeight = FontWeight.SemiBold
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(4.dp))
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    screening.facialMarkers.forEach { marker ->
                        Surface(
                            color = Tertiary.copy(alpha = 0.12f),
                            shape = RoundedCornerShape(6.dp)
                        ) {
                            Text(
                                text = marker,
                                modifier = Modifier.padding(
                                    horizontal = 10.dp,
                                    vertical = 4.dp
                                ),
                                style = MaterialTheme.typography.labelSmall,
                                color = Tertiary
                            )
                        }
                    }
                }
            }

            // Recommendation
            if (screening.recommendation.isNotBlank()) {
                Spacer(Modifier.height(8.dp))
                HorizontalDivider(
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    text = screening.recommendation,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (c7) Ancestry-Derived Predictions section
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun AncestryDerivedSection(
    ancestry: AncestryDerivedPredictionsResponse
) {
    ExpandableResultCard(
        title = "Ancestry-Derived Predictions",
        icon = Icons.Outlined.Science
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            // mtDNA Haplogroup
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Predicted mtDNA Haplogroup",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Column(horizontalAlignment = Alignment.End) {
                        Text(
                            text = ancestry.predictedMtdnaHaplogroup,
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace
                            ),
                            color = Primary
                        )
                        Text(
                            text = "Confidence: ${String.format("%.0f%%", ancestry.haplogroupConfidence * 100)}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // Tolerance / sensitivity bars
            RiskProgressBar(
                label = "Lactose Tolerance",
                score = ancestry.lactoseToleranceProbability
            )
            RiskProgressBar(
                label = "Alcohol Flush Probability",
                score = ancestry.alcoholFlushProbability
            )

            // Caffeine & bitter taste as text rows
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Caffeine Sensitivity",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = ancestry.caffeineSensitivity,
                    style = MaterialTheme.typography.bodySmall.copy(
                        fontWeight = FontWeight.Bold
                    ),
                    color = Primary
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Bitter Taste Sensitivity",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = ancestry.bitterTasteSensitivity,
                    style = MaterialTheme.typography.bodySmall.copy(
                        fontWeight = FontWeight.Bold
                    ),
                    color = Primary
                )
            }

            // Population-specific risks
            if (ancestry.populationSpecificRisks.isNotEmpty()) {
                Text(
                    text = "Population-Specific Risks",
                    style = MaterialTheme.typography.labelMedium.copy(
                        fontWeight = FontWeight.SemiBold
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                BulletList(
                    items = ancestry.populationSpecificRisks,
                    tintColor = RiskModerate
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (e) Diet recommendations section
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DietRecommendationsSection(diet: DietRecommendation) {
    Column(
        modifier = Modifier.animateContentSize(),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // Summary text
        Text(
            text = diet.summary,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground
        )

        // Key Nutrients as chips
        if (diet.keyNutrients.isNotEmpty()) {
            Text(
                text = "Key Nutrients",
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                diet.keyNutrients.forEach { nutrient ->
                    NutrientChip(text = nutrient)
                }
            }
        }

        // Foods to Increase (green-tinted)
        if (diet.foodsToIncrease.isNotEmpty()) {
            FoodList(
                title = "Foods to Increase",
                items = diet.foodsToIncrease,
                tintColor = RiskLow
            )
        }

        // Foods to Avoid (red-tinted)
        if (diet.foodsToAvoid.isNotEmpty()) {
            FoodList(
                title = "Foods to Avoid",
                items = diet.foodsToAvoid,
                tintColor = RiskHigh
            )
        }

        // Calorie target
        if (diet.calorieTarget != null) {
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(containerColor = SurfaceVariant)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Daily Calorie Target",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = "${diet.calorieTarget} kcal",
                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                        color = Primary
                    )
                }
            }
        }

        // Meal plans
        if (diet.mealPlans.isNotEmpty()) {
            HorizontalDivider(
                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f),
                modifier = Modifier.padding(vertical = 4.dp)
            )
            Text(
                text = "Meal Plans",
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onBackground
            )
            diet.mealPlans.forEach { plan ->
                MealPlanCard(mealPlan = plan)
            }
        }
    }
}

@Composable
private fun NutrientChip(text: String) {
    Surface(
        color = Tertiary.copy(alpha = 0.12f),
        shape = RoundedCornerShape(8.dp)
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Medium),
            color = Tertiary
        )
    }
}

@Composable
private fun FoodList(
    title: String,
    items: List<String>,
    tintColor: Color
) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(
            containerColor = tintColor.copy(alpha = 0.06f)
        )
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = tintColor
            )
            Spacer(Modifier.height(8.dp))
            items.forEach { item ->
                Row(
                    modifier = Modifier.padding(vertical = 3.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .background(
                                color = tintColor.copy(alpha = 0.7f),
                                shape = RoundedCornerShape(3.dp)
                            )
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        text = item,
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (f) Disclaimer
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun DisclaimerSection(text: String) {
    Column(modifier = Modifier.padding(top = 8.dp)) {
        HorizontalDivider(
            color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
        )
        Spacer(Modifier.height(12.dp))
        Text(
            text = text,
            style = MaterialTheme.typography.labelMedium.copy(
                fontSize = 11.sp,
                lineHeight = 16.sp
            ),
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Best-effort formatting of an ISO-8601 timestamp string into a more
 * readable date. Falls back to the raw string if parsing fails.
 */
private fun formatTimestamp(raw: String): String {
    return try {
        // Handle "2024-03-15T10:30:00Z" or similar ISO format
        val dateTimePart = raw.substringBefore("T")
        val parts = dateTimePart.split("-")
        if (parts.size == 3) {
            val months = arrayOf(
                "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            )
            val month = parts[1].toIntOrNull() ?: return raw
            val day = parts[2].toIntOrNull() ?: return raw
            val year = parts[0]
            "${months.getOrElse(month) { "???" }} $day, $year"
        } else {
            raw
        }
    } catch (_: Exception) {
        raw
    }
}
