package com.teloscopy.app.ui.screens

import android.app.Activity
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Gavel
import androidx.compose.material.icons.outlined.Mail
import androidx.compose.material.icons.outlined.PrivacyTip
import androidx.compose.material.icons.outlined.Science
import androidx.compose.material.icons.outlined.Security
import androidx.compose.material.icons.outlined.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import com.teloscopy.app.ui.theme.Background
import com.teloscopy.app.ui.theme.OnBackground
import com.teloscopy.app.ui.theme.OnSurfaceVariant
import com.teloscopy.app.ui.theme.Primary
import com.teloscopy.app.ui.theme.Secondary
import com.teloscopy.app.ui.theme.Surface
import com.teloscopy.app.ui.theme.Tertiary
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

// ─────────────────────────────────────────────────────────────────────────────
// Consent DataStore Keys
// ─────────────────────────────────────────────────────────────────────────────

/** DataStore key indicating the user has accepted the consent form. */
val CONSENT_ACCEPTED_KEY = booleanPreferencesKey("consent_accepted")

/** Epoch-millis timestamp of when consent was granted. */
val CONSENT_TIMESTAMP_KEY = stringPreferencesKey("consent_timestamp")

/** Semantic version of the consent text that was accepted. */
val CONSENT_VERSION_KEY = stringPreferencesKey("consent_version")

/** Whether the user opted in to anonymised research data sharing. */
val CONSENT_RESEARCH_KEY = booleanPreferencesKey("consent_research")

/** Current consent document version. Bump when legal text changes. */
private const val CURRENT_CONSENT_VERSION = "1.0"

// ─────────────────────────────────────────────────────────────────────────────
// Consent Screen
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Legal consent screen compliant with the Indian Digital Personal Data
 * Protection (DPDP) Act 2023 and IT Act 2000.
 *
 * Displays a scrollable consent form that must be acknowledged before the
 * user can proceed to the rest of the application. Consent state is
 * persisted to [DataStore] so the form is shown only once (unless the
 * user withdraws consent from Settings).
 *
 * @param dataStore   The Hilt-provided [DataStore] for persisting consent.
 * @param onConsentGranted Called after consent is successfully stored.
 */
@Composable
fun ConsentScreen(
    dataStore: DataStore<Preferences>,
    onConsentGranted: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // ── Required checkbox states ─────────────────────────────────────────
    var ageConfirmed by remember { mutableStateOf(false) }
    var privacyAccepted by remember { mutableStateOf(false) }
    var termsAccepted by remember { mutableStateOf(false) }
    var dataProcessingAccepted by remember { mutableStateOf(false) }

    // ── Optional checkbox state ──────────────────────────────────────────
    var researchConsent by remember { mutableStateOf(false) }

    // ── Decline dialog ───────────────────────────────────────────────────
    var showDeclineDialog by remember { mutableStateOf(false) }

    val allRequiredChecked =
        ageConfirmed && privacyAccepted && termsAccepted && dataProcessingAccepted

    Scaffold(containerColor = Background) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
        ) {
            // ── Gradient Header ──────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(
                                Primary.copy(alpha = 0.10f),
                                Color.Transparent
                            )
                        )
                    )
                    .padding(horizontal = 24.dp, vertical = 32.dp)
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.fillMaxWidth()) {
                    Icon(
                        imageVector = Icons.Outlined.Security,
                        contentDescription = null,
                        tint = Primary,
                        modifier = Modifier.size(48.dp)
                    )
                    Spacer(Modifier.height(12.dp))
                    Text(
                        text = "Privacy & Consent",
                        style = MaterialTheme.typography.headlineMedium.copy(
                            fontWeight = FontWeight.Bold
                        ),
                        color = OnBackground,
                        textAlign = TextAlign.Center
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Teloscopy Genomic Intelligence Platform",
                        style = MaterialTheme.typography.bodyMedium,
                        color = OnSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                }
            }

            Column(modifier = Modifier.padding(horizontal = 20.dp)) {

                // ── DPDP Notice ──────────────────────────────────────────
                ConsentSectionHeader(
                    title = "Privacy Notice",
                    icon = Icons.Outlined.PrivacyTip
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "In accordance with the Digital Personal Data Protection " +
                            "(DPDP) Act, 2023 and the Information Technology Act, 2000, " +
                            "this notice explains how Teloscopy collects, processes, and " +
                            "protects your personal and sensitive personal data.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = OnSurfaceVariant,
                    lineHeight = 22.sp
                )

                Spacer(Modifier.height(20.dp))

                // ── Data We Collect ──────────────────────────────────────
                ConsentSectionHeader(
                    title = "Data We Collect",
                    icon = Icons.Outlined.Science
                )
                Spacer(Modifier.height(8.dp))
                DataCollectionCard()

                Spacer(Modifier.height(20.dp))

                // ── Your Rights ──────────────────────────────────────────
                ConsentSectionHeader(
                    title = "Your Rights as a Data Principal",
                    icon = Icons.Outlined.Gavel
                )
                Spacer(Modifier.height(8.dp))
                DataPrincipalRightsCard()

                Spacer(Modifier.height(20.dp))

                // ── Medical Disclaimer ───────────────────────────────────
                MedicalDisclaimerCard()

                Spacer(Modifier.height(20.dp))

                // ── Required Consents ────────────────────────────────────
                ConsentSectionHeader(
                    title = "Required Consents",
                    icon = Icons.Outlined.CheckCircle
                )
                Spacer(Modifier.height(8.dp))

                ElevatedCard(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.elevatedCardColors(containerColor = Surface),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        ConsentCheckbox(
                            checked = ageConfirmed,
                            onCheckedChange = { ageConfirmed = it },
                            label = "I confirm that I am 18 years of age or older, as " +
                                    "required under Section 9 of the DPDP Act, 2023.",
                            required = true
                        )

                        HorizontalDivider(
                            color = Color(0xFF3A3F55).copy(alpha = 0.5f),
                            modifier = Modifier.padding(vertical = 4.dp)
                        )

                        ConsentCheckbox(
                            checked = privacyAccepted,
                            onCheckedChange = { privacyAccepted = it },
                            label = "I have read and understood the Privacy Policy, " +
                                    "including how my sensitive personal data is " +
                                    "collected, processed, stored, and shared.",
                            required = true
                        )

                        HorizontalDivider(
                            color = Color(0xFF3A3F55).copy(alpha = 0.5f),
                            modifier = Modifier.padding(vertical = 4.dp)
                        )

                        ConsentCheckbox(
                            checked = termsAccepted,
                            onCheckedChange = { termsAccepted = it },
                            label = "I agree to the Terms of Service governing my " +
                                    "use of the Teloscopy platform and services.",
                            required = true
                        )

                        HorizontalDivider(
                            color = Color(0xFF3A3F55).copy(alpha = 0.5f),
                            modifier = Modifier.padding(vertical = 4.dp)
                        )

                        ConsentCheckbox(
                            checked = dataProcessingAccepted,
                            onCheckedChange = { dataProcessingAccepted = it },
                            label = "I consent to the processing of my personal and " +
                                    "sensitive personal data (including facial images, " +
                                    "health data, and genetic information) for the " +
                                    "purposes described in the Privacy Policy.",
                            required = true
                        )
                    }
                }

                Spacer(Modifier.height(16.dp))

                // ── Optional Research Consent ────────────────────────────
                ConsentSectionHeader(
                    title = "Optional Research Consent",
                    icon = Icons.Outlined.Science
                )
                Spacer(Modifier.height(8.dp))

                ElevatedCard(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.elevatedCardColors(
                        containerColor = Secondary.copy(alpha = 0.08f)
                    ),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        ConsentCheckbox(
                            checked = researchConsent,
                            onCheckedChange = { researchConsent = it },
                            label = "I voluntarily consent to the use of my anonymised " +
                                    "and de-identified data for research purposes and " +
                                    "model improvement. This is entirely optional and " +
                                    "does not affect your use of the app.",
                            required = false
                        )
                    }
                }

                Spacer(Modifier.height(24.dp))

                // ── Legal Links ──────────────────────────────────────────
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center
                ) {
                    TextButton(onClick = {
                        val intent = Intent(
                            Intent.ACTION_VIEW,
                            Uri.parse("https://teloscopy.app/privacy")
                        )
                        context.startActivity(intent)
                    }) {
                        Text(
                            text = "Privacy Policy",
                            color = Primary,
                            style = MaterialTheme.typography.labelLarge,
                            textDecoration = TextDecoration.Underline
                        )
                    }
                    Spacer(Modifier.width(16.dp))
                    TextButton(onClick = {
                        val intent = Intent(
                            Intent.ACTION_VIEW,
                            Uri.parse("https://teloscopy.app/terms")
                        )
                        context.startActivity(intent)
                    }) {
                        Text(
                            text = "Terms of Service",
                            color = Primary,
                            style = MaterialTheme.typography.labelLarge,
                            textDecoration = TextDecoration.Underline
                        )
                    }
                }

                Spacer(Modifier.height(16.dp))

                // ── Action Buttons ───────────────────────────────────────
                Button(
                    onClick = {
                        scope.launch {
                            dataStore.edit { prefs ->
                                prefs[CONSENT_ACCEPTED_KEY] = true
                                prefs[CONSENT_TIMESTAMP_KEY] =
                                    System.currentTimeMillis().toString()
                                prefs[CONSENT_VERSION_KEY] = CURRENT_CONSENT_VERSION
                                prefs[CONSENT_RESEARCH_KEY] = researchConsent
                            }
                            onConsentGranted()
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    enabled = allRequiredChecked,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Primary,
                        contentColor = Color.Black,
                        disabledContainerColor = Primary.copy(alpha = 0.3f),
                        disabledContentColor = Color.Black.copy(alpha = 0.4f)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.CheckCircle,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "I Agree & Continue",
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                }

                Spacer(Modifier.height(12.dp))

                OutlinedButton(
                    onClick = { showDeclineDialog = true },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = OnSurfaceVariant
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = "Decline",
                        fontWeight = FontWeight.Medium
                    )
                }

                Spacer(Modifier.height(24.dp))

                // ── Grievance Officer ────────────────────────────────────
                GrievanceOfficerCard()

                Spacer(Modifier.height(16.dp))

                // ── Footer ───────────────────────────────────────────────
                Text(
                    text = "Consent Version $CURRENT_CONSENT_VERSION  \u2022  " +
                            "DPDP Act 2023  \u2022  IT Act 2000",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color(0xFF4A4A4A),
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(24.dp))
            }
        }
    }

    // ── Decline Dialog ───────────────────────────────────────────────────
    if (showDeclineDialog) {
        AlertDialog(
            onDismissRequest = { showDeclineDialog = false },
            containerColor = Surface,
            title = {
                Text(
                    text = "Consent Required",
                    color = OnBackground,
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Text(
                    text = "Teloscopy requires your consent to process personal and " +
                            "sensitive personal data in order to provide genomic " +
                            "analysis services.\n\n" +
                            "Under the DPDP Act 2023, you have the right to decline. " +
                            "However, without consent, we cannot provide our services " +
                            "and the app will close.\n\n" +
                            "You may return at any time to grant consent.",
                    color = OnSurfaceVariant,
                    lineHeight = 22.sp
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        showDeclineDialog = false
                        // Close the app
                        (context as? Activity)?.finishAffinity()
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFFF5252),
                        contentColor = Color.White
                    )
                ) {
                    Text("Exit App", fontWeight = FontWeight.SemiBold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeclineDialog = false }) {
                    Text("Go Back", color = Primary)
                }
            }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Section header matching the SettingsScreen style — uppercase label with
 * an icon and the primary colour.
 */
@Composable
private fun ConsentSectionHeader(
    title: String,
    icon: ImageVector? = null
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 4.dp)
    ) {
        if (icon != null) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = Primary,
                modifier = Modifier.size(16.dp)
            )
            Spacer(Modifier.width(8.dp))
        }
        Text(
            text = title.uppercase(),
            style = MaterialTheme.typography.labelMedium.copy(
                letterSpacing = 1.5.sp,
                fontSize = 12.sp
            ),
            color = Primary,
            fontWeight = FontWeight.Bold
        )
    }
}

/**
 * Card listing every category of personal data the app collects.
 */
@Composable
private fun DataCollectionCard() {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(containerColor = Surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "We collect and process the following data categories:",
                style = MaterialTheme.typography.bodyMedium,
                color = OnSurfaceVariant,
                lineHeight = 20.sp
            )
            Spacer(Modifier.height(12.dp))

            val dataTypes = listOf(
                "Facial images" to "Used for biological age estimation and facial analysis",
                "Health reports" to "Blood work, medical history, and health markers",
                "Genetic variants" to "SNP data for disease risk prediction and ancestry",
                "Demographics" to "Age, sex, ethnicity, and ancestral region",
                "Telomere measurements" to "Telomere length data for cellular aging assessment",
                "Dietary information" to "Dietary preferences and nutritional data for personalized recommendations"
            )

            dataTypes.forEach { (title, description) ->
                DataTypeItem(title = title, description = description)
                if (title != dataTypes.last().first) {
                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

/** A single row inside the data-collection card. */
@Composable
private fun DataTypeItem(title: String, description: String) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Box(
            modifier = Modifier
                .padding(top = 4.dp)
                .size(6.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(Primary)
        )
        Spacer(Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyMedium,
                color = OnBackground,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = OnSurfaceVariant,
                lineHeight = 18.sp
            )
        }
    }
}

/**
 * Card enumerating the rights of the data principal under the DPDP Act.
 */
@Composable
private fun DataPrincipalRightsCard() {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(containerColor = Surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Under the DPDP Act 2023, you have the right to:",
                style = MaterialTheme.typography.bodyMedium,
                color = OnSurfaceVariant,
                lineHeight = 20.sp
            )
            Spacer(Modifier.height(12.dp))

            val rights = listOf(
                "Right to Access" to "Request a summary of your personal data and processing activities",
                "Right to Correction" to "Request correction of inaccurate or misleading personal data",
                "Right to Erasure" to "Request deletion of your personal data that is no longer necessary",
                "Right to Grievance Redressal" to "File a complaint with our Grievance Officer or the Data Protection Board",
                "Right to Nomination" to "Nominate any individual to exercise your rights in case of death or incapacity",
                "Right to Withdrawal" to "Withdraw consent at any time through the Settings screen"
            )

            rights.forEach { (title, description) ->
                RightItem(title = title, description = description)
                if (title != rights.last().first) {
                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

/** A single row inside the rights card. */
@Composable
private fun RightItem(title: String, description: String) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Icon(
            imageVector = Icons.Outlined.CheckCircle,
            contentDescription = null,
            tint = Tertiary,
            modifier = Modifier
                .padding(top = 2.dp)
                .size(14.dp)
        )
        Spacer(Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyMedium,
                color = OnBackground,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = OnSurfaceVariant,
                lineHeight = 18.sp
            )
        }
    }
}

/**
 * Warning-styled card with the medical / research disclaimer.
 */
@Composable
private fun MedicalDisclaimerCard() {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(
            containerColor = Color(0xFF1A1520)
        ),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.Top
        ) {
            Icon(
                imageVector = Icons.Outlined.Warning,
                contentDescription = null,
                tint = Color(0xFFFF9800),
                modifier = Modifier.size(24.dp)
            )
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Medical Disclaimer",
                    style = MaterialTheme.typography.titleMedium,
                    color = Color(0xFFFF9800),
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "Teloscopy is intended for research and educational purposes " +
                            "only. It is NOT a medical device and is NOT intended for " +
                            "the diagnosis, treatment, cure, or prevention of any " +
                            "disease or medical condition.\n\n" +
                            "Results generated by the app should not be considered " +
                            "medical advice. Always consult a qualified healthcare " +
                            "professional before making any health-related decisions " +
                            "based on information provided by this application.",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFFBDBDBD),
                    lineHeight = 20.sp
                )
            }
        }
    }
}

/**
 * A labelled checkbox row used for consent items.
 *
 * @param checked        Current check state.
 * @param onCheckedChange Callback when the user toggles the checkbox.
 * @param label          Descriptive text for this consent item.
 * @param required       When `true`, a "(Required)" tag is appended.
 */
@Composable
private fun ConsentCheckbox(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    label: String,
    required: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.Top
    ) {
        Checkbox(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = CheckboxDefaults.colors(
                checkedColor = Primary,
                uncheckedColor = OnSurfaceVariant,
                checkmarkColor = Color.Black
            )
        )
        Spacer(Modifier.width(4.dp))
        Column(modifier = Modifier.weight(1f).padding(top = 12.dp)) {
            Text(
                text = buildAnnotatedString {
                    append(label)
                    if (required) {
                        append(" ")
                        withStyle(
                            SpanStyle(
                                color = Color(0xFFFF5252),
                                fontWeight = FontWeight.Bold,
                                fontSize = 11.sp
                            )
                        ) {
                            append("(Required)")
                        }
                    } else {
                        append(" ")
                        withStyle(
                            SpanStyle(
                                color = Tertiary,
                                fontWeight = FontWeight.Medium,
                                fontSize = 11.sp
                            )
                        ) {
                            append("(Optional)")
                        }
                    }
                },
                style = MaterialTheme.typography.bodySmall,
                color = OnBackground,
                lineHeight = 20.sp
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Consent Gate
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Consent gate that verifies consent is still valid before showing content.
 * Redirects to consent screen if consent has been withdrawn or not granted.
 */
@Composable
fun RequireConsent(
    dataStore: DataStore<Preferences>,
    onConsentRequired: () -> Unit,
    content: @Composable () -> Unit
) {
    val consentAccepted by dataStore.data
        .map { prefs -> prefs[CONSENT_ACCEPTED_KEY] ?: false }
        .collectAsState(initial = null)

    when (consentAccepted) {
        true -> content()
        false -> {
            LaunchedEffect(Unit) { onConsentRequired() }
        }
        null -> {
            // Loading state — show nothing while checking
            Box(modifier = Modifier.fillMaxSize())
        }
    }
}

/**
 * Contact card for the Grievance Officer as required by DPDP Act.
 */
@Composable
private fun GrievanceOfficerCard() {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(containerColor = Surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Outlined.Mail,
                contentDescription = null,
                tint = Primary,
                modifier = Modifier.size(20.dp)
            )
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Grievance Officer",
                    style = MaterialTheme.typography.titleMedium,
                    color = OnBackground,
                    fontWeight = FontWeight.Medium
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = "For data protection concerns or to exercise your rights, " +
                            "contact our Grievance Officer:",
                    style = MaterialTheme.typography.bodySmall,
                    color = OnSurfaceVariant,
                    lineHeight = 18.sp
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "animaticalpha123@gmail.com",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Primary,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}
