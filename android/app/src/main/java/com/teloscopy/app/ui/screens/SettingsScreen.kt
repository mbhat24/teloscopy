package com.teloscopy.app.ui.screens

import android.content.Context
import androidx.compose.animation.animateColorAsState
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Cloud
import androidx.compose.material.icons.outlined.DarkMode
import androidx.compose.material.icons.outlined.Error
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Palette
import androidx.compose.material.icons.outlined.PrivacyTip
import androidx.compose.material.icons.outlined.Save
import androidx.compose.material.icons.outlined.Science
import androidx.compose.material.icons.outlined.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.teloscopy.app.ui.theme.Background
import com.teloscopy.app.ui.theme.OnBackground
import com.teloscopy.app.ui.theme.OnSurfaceVariant
import com.teloscopy.app.ui.theme.Primary
import com.teloscopy.app.ui.theme.Secondary
import com.teloscopy.app.ui.theme.Surface
import com.teloscopy.app.ui.theme.Tertiary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

private const val PREFS_NAME = "teloscopy_settings"
private const val KEY_SERVER_URL = "server_url"
private const val KEY_DARK_MODE = "dark_mode"
private const val DEFAULT_SERVER_URL = "http://10.0.2.2:8000"
private const val APP_VERSION = "2.0.0"
private const val APP_BUILD = "2024.06"

// ─────────────────────────────────────────────────────────────────────────────
// Connection Status
// ─────────────────────────────────────────────────────────────────────────────

private sealed class ConnectionStatus {
    data object Unknown : ConnectionStatus()
    data object Testing : ConnectionStatus()
    data class Connected(val version: String) : ConnectionStatus()
    data class Failed(val reason: String) : ConnectionStatus()
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings Screen
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Settings screen for configuring the Teloscopy backend server URL,
 * testing connectivity, toggling appearance, and viewing app information.
 *
 * Persists the server URL via [android.content.SharedPreferences] so it
 * survives process death and is immediately available to other components
 * (e.g. the Retrofit base-URL interceptor) without requiring DataStore's
 * coroutine machinery.
 *
 * @param onBack Called when the user taps the back navigation arrow.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Load saved settings from SharedPreferences
    val prefs = remember {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }
    var serverUrl by remember {
        mutableStateOf(prefs.getString(KEY_SERVER_URL, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL)
    }
    var connectionStatus by remember { mutableStateOf<ConnectionStatus>(ConnectionStatus.Unknown) }
    var saveMessage by remember { mutableStateOf<String?>(null) }

    // Dark mode toggle state (placeholder – not wired to actual theme yet)
    var isDarkMode by remember {
        mutableStateOf(prefs.getBoolean(KEY_DARK_MODE, true))
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Settings",
                        color = OnBackground,
                        fontWeight = FontWeight.SemiBold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = OnBackground
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Background
                )
            )
        },
        containerColor = Background
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            // ═════════════════════════════════════════════════════════════
            // SECTION 1: Server Configuration
            // ═════════════════════════════════════════════════════════════

            SettingsSectionHeader(
                title = "Server Configuration",
                icon = Icons.Outlined.Cloud
            )

            Spacer(modifier = Modifier.height(12.dp))

            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(containerColor = Surface),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "Backend Server URL",
                        style = MaterialTheme.typography.titleMedium,
                        color = OnBackground,
                        fontWeight = FontWeight.Medium
                    )

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = "The URL of the Teloscopy analysis backend. Use 10.0.2.2 " +
                                "for the Android emulator or your server\u2019s IP address.",
                        style = MaterialTheme.typography.bodySmall,
                        color = OnSurfaceVariant,
                        lineHeight = 18.sp
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    OutlinedTextField(
                        value = serverUrl,
                        onValueChange = {
                            serverUrl = it
                            saveMessage = null
                        },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Server URL") },
                        placeholder = { Text(DEFAULT_SERVER_URL) },
                        singleLine = true,
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = OnBackground,
                            unfocusedTextColor = OnBackground,
                            cursorColor = Primary,
                            focusedBorderColor = Primary,
                            unfocusedBorderColor = Color(0xFF3A3F55),
                            focusedLabelColor = Primary,
                            unfocusedLabelColor = OnSurfaceVariant,
                            focusedPlaceholderColor = Color(0xFF616161),
                            unfocusedPlaceholderColor = Color(0xFF616161)
                        )
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Save Button
                        Button(
                            onClick = {
                                prefs.edit()
                                    .putString(KEY_SERVER_URL, serverUrl.trimEnd('/'))
                                    .apply()
                                saveMessage = "Saved"
                            },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Primary,
                                contentColor = Color.Black
                            ),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.Save,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Save",
                                fontWeight = FontWeight.SemiBold
                            )
                        }

                        // Health Check Button
                        FilledTonalButton(
                            onClick = {
                                connectionStatus = ConnectionStatus.Testing
                                saveMessage = null
                                scope.launch {
                                    connectionStatus = testConnection(serverUrl.trimEnd('/'))
                                }
                            },
                            modifier = Modifier.weight(1f),
                            enabled = connectionStatus !is ConnectionStatus.Testing,
                            colors = ButtonDefaults.filledTonalButtonColors(
                                containerColor = Secondary.copy(alpha = 0.2f),
                                contentColor = Secondary
                            ),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            if (connectionStatus is ConnectionStatus.Testing) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(18.dp),
                                    color = Secondary,
                                    strokeWidth = 2.dp
                                )
                            } else {
                                Icon(
                                    imageVector = Icons.Outlined.Cloud,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Test",
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }

                    // Save confirmation
                    if (saveMessage != null) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = saveMessage!!,
                            style = MaterialTheme.typography.labelMedium,
                            color = Tertiary
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Server Status Card
            ServerStatusCard(status = connectionStatus)

            Spacer(modifier = Modifier.height(24.dp))

            // ═════════════════════════════════════════════════════════════
            // SECTION 2: Appearance
            // ═════════════════════════════════════════════════════════════

            SettingsSectionHeader(
                title = "Appearance",
                icon = Icons.Outlined.Palette
            )

            Spacer(modifier = Modifier.height(12.dp))

            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(containerColor = Surface),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    // Dark mode toggle
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.DarkMode,
                                contentDescription = null,
                                tint = Primary,
                                modifier = Modifier.size(20.dp)
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Column {
                                Text(
                                    text = "Dark Mode",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = OnBackground,
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    text = if (isDarkMode) "Dark theme enabled"
                                    else "Light theme enabled",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = OnSurfaceVariant
                                )
                            }
                        }

                        Switch(
                            checked = isDarkMode,
                            onCheckedChange = { checked ->
                                isDarkMode = checked
                                // Persist preference (UI-only placeholder for now)
                                prefs.edit().putBoolean(KEY_DARK_MODE, checked).apply()
                            },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = Primary,
                                checkedTrackColor = Primary.copy(alpha = 0.3f),
                                uncheckedThumbColor = OnSurfaceVariant,
                                uncheckedTrackColor = Color(0xFF3A3F55)
                            )
                        )
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = "Note: Theme switching will take effect after app restart.",
                        style = MaterialTheme.typography.labelSmall,
                        color = OnSurfaceVariant.copy(alpha = 0.6f)
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // ═════════════════════════════════════════════════════════════
            // SECTION 3: Data & Privacy
            // ═════════════════════════════════════════════════════════════

            SettingsSectionHeader(
                title = "Data & Privacy",
                icon = Icons.Outlined.PrivacyTip
            )

            Spacer(modifier = Modifier.height(12.dp))

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
                        imageVector = Icons.Outlined.PrivacyTip,
                        contentDescription = null,
                        tint = OnSurfaceVariant,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Privacy & Legal",
                            style = MaterialTheme.typography.titleMedium,
                            color = OnBackground,
                            fontWeight = FontWeight.Medium
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Your genetic data is processed locally and on your " +
                                    "configured server. No data is sent to third parties. " +
                                    "Review our privacy policy for details.",
                            style = MaterialTheme.typography.bodySmall,
                            color = OnSurfaceVariant,
                            lineHeight = 18.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Disclaimer
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
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Disclaimer",
                            style = MaterialTheme.typography.titleMedium,
                            color = Color(0xFFFF9800),
                            fontWeight = FontWeight.SemiBold
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "For research and educational purposes only. Not intended " +
                                    "for medical diagnosis or treatment. Results should not " +
                                    "replace professional medical advice. Always consult a " +
                                    "qualified healthcare provider.",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color(0xFFBDBDBD),
                            lineHeight = 18.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // ═════════════════════════════════════════════════════════════
            // SECTION 4: About
            // ═════════════════════════════════════════════════════════════

            SettingsSectionHeader(
                title = "About",
                icon = Icons.Outlined.Info
            )

            Spacer(modifier = Modifier.height(12.dp))

            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(containerColor = Surface),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    // App icon / name row
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(RoundedCornerShape(10.dp))
                                .background(Primary.copy(alpha = 0.15f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.Science,
                                contentDescription = null,
                                tint = Primary,
                                modifier = Modifier.size(22.dp)
                            )
                        }
                        Spacer(modifier = Modifier.width(12.dp))
                        Column {
                            Text(
                                text = "Teloscopy",
                                style = MaterialTheme.typography.titleLarge,
                                color = Primary,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "Genomic Intelligence Platform",
                                style = MaterialTheme.typography.labelSmall,
                                color = OnSurfaceVariant
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    HorizontalDivider(color = Color(0xFF3A3F55).copy(alpha = 0.5f))

                    Spacer(modifier = Modifier.height(12.dp))

                    // Info rows
                    SettingsInfoRow(label = "Version", value = APP_VERSION)
                    SettingsInfoRow(label = "Build", value = APP_BUILD)
                    SettingsInfoRow(
                        label = "Description",
                        value = "Biological age estimation, disease risk prediction, " +
                                "and personalized nutrition recommendations powered " +
                                "by genomic and facial analysis."
                    )
                    SettingsInfoRow(label = "Engine", value = "Teloscopy Core v2.0")
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // ── Footer ──────────────────────────────────────────────────────
            Text(
                text = "\u00a9 2024 Teloscopy Project. All rights reserved.",
                style = MaterialTheme.typography.labelMedium,
                color = Color(0xFF4A4A4A),
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components (private to this file)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Section header with an icon and uppercase label.
 * Groups related settings visually.
 */
@Composable
private fun SettingsSectionHeader(
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
            Spacer(modifier = Modifier.width(8.dp))
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

/** A label / value pair row used in the About section. */
@Composable
private fun SettingsInfoRow(label: String, value: String) {
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = OnSurfaceVariant,
            fontWeight = FontWeight.Medium
        )
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = OnBackground,
            lineHeight = 20.sp
        )
    }
    Spacer(modifier = Modifier.height(4.dp))
}

/**
 * Card showing the current server connection status with an animated
 * colour indicator, icon, and descriptive text.
 */
@Composable
private fun ServerStatusCard(status: ConnectionStatus) {
    val statusColor by animateColorAsState(
        targetValue = when (status) {
            is ConnectionStatus.Unknown -> OnSurfaceVariant
            is ConnectionStatus.Testing -> Secondary
            is ConnectionStatus.Connected -> Tertiary
            is ConnectionStatus.Failed -> Color(0xFFFF5252)
        },
        label = "statusColor"
    )

    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(containerColor = Surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Status indicator circle
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(statusColor.copy(alpha = 0.15f)),
                contentAlignment = Alignment.Center
            ) {
                when (status) {
                    is ConnectionStatus.Testing -> {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            color = statusColor,
                            strokeWidth = 2.dp
                        )
                    }
                    is ConnectionStatus.Connected -> {
                        Icon(
                            imageVector = Icons.Outlined.CheckCircle,
                            contentDescription = "Connected",
                            tint = statusColor,
                            modifier = Modifier.size(22.dp)
                        )
                    }
                    is ConnectionStatus.Failed -> {
                        Icon(
                            imageVector = Icons.Outlined.Error,
                            contentDescription = "Connection failed",
                            tint = statusColor,
                            modifier = Modifier.size(22.dp)
                        )
                    }
                    is ConnectionStatus.Unknown -> {
                        Icon(
                            imageVector = Icons.Outlined.Cloud,
                            contentDescription = "Not tested",
                            tint = statusColor,
                            modifier = Modifier.size(22.dp)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Server Status",
                    style = MaterialTheme.typography.titleMedium,
                    color = OnBackground,
                    fontWeight = FontWeight.Medium
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = when (status) {
                        is ConnectionStatus.Unknown -> "Not tested yet"
                        is ConnectionStatus.Testing -> "Testing connection\u2026"
                        is ConnectionStatus.Connected -> "Connected (v${status.version})"
                        is ConnectionStatus.Failed -> status.reason
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = statusColor,
                    lineHeight = 18.sp
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Network helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tests the connection to the Teloscopy backend by hitting the `/api/health`
 * endpoint. Runs on [Dispatchers.IO] to avoid blocking the main thread.
 *
 * @param baseUrl The base URL of the backend server (without trailing slash).
 * @return A [ConnectionStatus] representing the result of the health check.
 */
private suspend fun testConnection(baseUrl: String): ConnectionStatus {
    return withContext(Dispatchers.IO) {
        try {
            val url = URL("${baseUrl.trimEnd('/')}/api/health")
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 5_000
            connection.readTimeout = 5_000
            connection.setRequestProperty("Accept", "application/json")

            try {
                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val body = connection.inputStream.bufferedReader().use { it.readText() }
                    // Parse version from JSON: {"status":"ok","version":"2.0.0",...}
                    val versionRegex = """"version"\s*:\s*"([^"]+)"""".toRegex()
                    val version = versionRegex.find(body)?.groupValues?.get(1) ?: "unknown"
                    ConnectionStatus.Connected(version)
                } else {
                    ConnectionStatus.Failed("HTTP $responseCode")
                }
            } finally {
                connection.disconnect()
            }
        } catch (e: java.net.ConnectException) {
            ConnectionStatus.Failed("Connection refused")
        } catch (e: java.net.SocketTimeoutException) {
            ConnectionStatus.Failed("Connection timed out")
        } catch (e: java.net.UnknownHostException) {
            ConnectionStatus.Failed("Unknown host")
        } catch (e: Exception) {
            ConnectionStatus.Failed(e.message ?: "Unknown error")
        }
    }
}
