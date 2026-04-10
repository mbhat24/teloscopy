package com.teloscopy.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.outlined.Bedtime
import androidx.compose.material.icons.outlined.Biotech
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.FitnessCenter
import androidx.compose.material.icons.outlined.HealthAndSafety
import androidx.compose.material.icons.outlined.LocalDrink
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.Restaurant
import androidx.compose.material.icons.outlined.Schedule
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.TrendingUp
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.pulltorefresh.PullToRefreshContainer
import androidx.compose.material3.pulltorefresh.rememberPullToRefreshState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.teloscopy.app.ui.theme.Background
import com.teloscopy.app.ui.theme.GradientAmberStart
import com.teloscopy.app.ui.theme.GradientCyanEnd
import com.teloscopy.app.ui.theme.GradientCyanStart
import com.teloscopy.app.ui.theme.GradientGreenEnd
import com.teloscopy.app.ui.theme.GradientGreenStart
import com.teloscopy.app.ui.theme.GradientPurpleEnd
import com.teloscopy.app.ui.theme.GradientPurpleStart
import com.teloscopy.app.ui.theme.OnBackground
import com.teloscopy.app.ui.theme.OnSurfaceVariant
import com.teloscopy.app.ui.theme.Primary
import com.teloscopy.app.ui.theme.Secondary
import com.teloscopy.app.ui.theme.Surface
import com.teloscopy.app.ui.theme.SurfaceVariant
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main home screen for the Teloscopy Genomic Intelligence app.
 *
 * Displays a health-dashboard layout with a greeting, quick stats,
 * action cards, health tips, and recent activity. Pull-to-refresh
 * is supported for future integration with live data sources.
 *
 * @param onNavigateToAnalysis Called when the user taps the Facial Analysis card.
 * @param onNavigateToProfile  Called when the user taps the Profile Analysis card.
 * @param onNavigateToSettings Called when the user taps the settings gear icon.
 * @param onOpenDrawer         Called when the user taps the hamburger menu icon.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onNavigateToAnalysis: () -> Unit,
    onNavigateToProfile: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onOpenDrawer: () -> Unit = {}
) {
    val pullToRefreshState = rememberPullToRefreshState()

    // Simulate a refresh when the user pulls down
    if (pullToRefreshState.isRefreshing) {
        LaunchedEffect(true) {
            delay(1500L)
            pullToRefreshState.endRefresh()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Teloscopy",
                        color = Primary,
                        fontWeight = FontWeight.Bold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onOpenDrawer) {
                        Icon(
                            imageVector = Icons.Filled.Menu,
                            contentDescription = "Open menu",
                            tint = OnBackground
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(
                            imageVector = Icons.Outlined.Settings,
                            contentDescription = "Settings",
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
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .nestedScroll(pullToRefreshState.nestedScrollConnection)
        ) {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // ── 0. Gradient Header ───────────────────────────────────
                item { GradientHeader() }

                // ── 1. Greeting ──────────────────────────────────────────
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        GreetingSection()
                    }
                }

                // ── 1.5 Quick Stat Cards ─────────────────────────────────
                item {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        QuickStatCard(
                            modifier = Modifier.weight(1f),
                            label = "Analyses",
                            value = "--",
                            color = MaterialTheme.colorScheme.primary
                        )
                        QuickStatCard(
                            modifier = Modifier.weight(1f),
                            label = "Bio Age",
                            value = "--",
                            color = MaterialTheme.colorScheme.tertiary
                        )
                        QuickStatCard(
                            modifier = Modifier.weight(1f),
                            label = "Telomere",
                            value = "--",
                            color = MaterialTheme.colorScheme.secondary
                        )
                    }
                    Spacer(Modifier.height(24.dp))
                }

                // ── 2. Quick Stats ───────────────────────────────────────
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        QuickStatsRow()
                    }
                }

                // ── 3. Quick Actions ─────────────────────────────────────
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        SectionTitle("Quick Actions")
                    }
                }
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        QuickActionsRow(
                            onNavigateToAnalysis = onNavigateToAnalysis,
                            onNavigateToProfile = onNavigateToProfile
                        )
                    }
                }

                // ── 4. Health Tips ───────────────────────────────────────
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        SectionTitle("Health Tips")
                    }
                }
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        HealthTipsRow()
                    }
                }

                // ── 5. Recent Activity ───────────────────────────────────
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        SectionTitle("Recent Activity")
                    }
                }
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        RecentActivityEmptyState()
                    }
                }

                // ── 6. Footer ────────────────────────────────────────────
                item {
                    Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                        FooterDisclaimer()
                    }
                }

                // Bottom spacer so content isn't clipped behind nav bar
                item { Spacer(modifier = Modifier.height(8.dp)) }
            }

            PullToRefreshContainer(
                state = pullToRefreshState,
                modifier = Modifier.align(Alignment.TopCenter),
                containerColor = SurfaceVariant,
                contentColor = Primary
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient Header
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun GradientHeader() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.primary.copy(alpha = 0.08f),
                        Color.Transparent
                    )
                )
            )
            .padding(horizontal = 24.dp, vertical = 32.dp)
    ) {
        Column {
            Text(
                text = "Welcome to",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Teloscopy",
                style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.primary
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Telomere analysis & personalised health insights",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Greeting Section
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun GreetingSection() {
    val dateFormat = SimpleDateFormat("EEEE, MMMM d", Locale.getDefault())
    val currentDate = dateFormat.format(Date())

    Column(modifier = Modifier.padding(top = 4.dp)) {
        Text(
            text = "Welcome back",
            style = MaterialTheme.typography.headlineMedium,
            color = OnBackground,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = currentDate,
            style = MaterialTheme.typography.bodyLarge,
            color = OnSurfaceVariant
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Quick Stats Row  — 3 gradient cards
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun QuickStatsRow() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        GradientStatCard(
            modifier = Modifier.weight(1f),
            title = "Last Analysis",
            value = "N/A",
            icon = Icons.Outlined.Schedule,
            gradientStart = GradientCyanStart,
            gradientEnd = GradientCyanEnd
        )
        GradientStatCard(
            modifier = Modifier.weight(1f),
            title = "Bio Age",
            value = "-- yrs",
            icon = Icons.Outlined.Biotech,
            gradientStart = GradientPurpleStart,
            gradientEnd = GradientPurpleEnd
        )
        GradientStatCard(
            modifier = Modifier.weight(1f),
            title = "Risk Score",
            value = "--",
            icon = Icons.Outlined.TrendingUp,
            gradientStart = GradientGreenStart,
            gradientEnd = GradientGreenEnd
        )
    }
}

@Composable
private fun GradientStatCard(
    modifier: Modifier = Modifier,
    title: String,
    value: String,
    icon: ImageVector,
    gradientStart: Color,
    gradientEnd: Color
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(
                brush = Brush.linearGradient(
                    colors = listOf(
                        gradientStart.copy(alpha = 0.25f),
                        gradientEnd.copy(alpha = 0.10f)
                    )
                )
            )
            .padding(12.dp)
    ) {
        Column {
            Icon(
                imageVector = icon,
                contentDescription = title,
                tint = gradientStart,
                modifier = Modifier.size(22.dp)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge,
                color = OnBackground,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.labelMedium,
                color = OnSurfaceVariant,
                fontSize = 11.sp
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section Title
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun SectionTitle(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleLarge,
        color = OnBackground,
        fontWeight = FontWeight.SemiBold,
        fontSize = 18.sp
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Quick Actions  — two side-by-side cards
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun QuickActionsRow(
    onNavigateToAnalysis: () -> Unit,
    onNavigateToProfile: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        QuickActionCard(
            modifier = Modifier.weight(1f),
            title = "Facial Analysis",
            subtitle = "Capture or upload a photo",
            icon = Icons.Outlined.CameraAlt,
            iconTint = Primary,
            onClick = onNavigateToAnalysis
        )
        QuickActionCard(
            modifier = Modifier.weight(1f),
            title = "Profile Analysis",
            subtitle = "Analyze your health data",
            icon = Icons.Outlined.Person,
            iconTint = Secondary,
            onClick = onNavigateToProfile
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun QuickActionCard(
    modifier: Modifier = Modifier,
    title: String,
    subtitle: String,
    icon: ImageVector,
    iconTint: Color,
    onClick: () -> Unit
) {
    ElevatedCard(
        onClick = onClick,
        modifier = modifier.height(140.dp),
        colors = CardDefaults.elevatedCardColors(containerColor = Surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.Center
        ) {
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(iconTint.copy(alpha = 0.15f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = title,
                    tint = iconTint,
                    modifier = Modifier.size(24.dp)
                )
            }
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.titleLarge.copy(fontSize = 15.sp),
                color = OnBackground,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = subtitle,
                style = MaterialTheme.typography.labelMedium,
                color = OnSurfaceVariant,
                fontSize = 11.sp
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Health Tips  — horizontal scrollable cards
// ─────────────────────────────────────────────────────────────────────────────

private data class HealthTip(
    val title: String,
    val description: String,
    val icon: ImageVector,
    val accentColor: Color
)

private val healthTips = listOf(
    HealthTip(
        title = "Stay Hydrated",
        description = "Drink 8+ glasses of water daily to support cellular health.",
        icon = Icons.Outlined.LocalDrink,
        accentColor = GradientCyanStart
    ),
    HealthTip(
        title = "Quality Sleep",
        description = "7–9 hours of sleep helps repair DNA and slow telomere shortening.",
        icon = Icons.Outlined.Bedtime,
        accentColor = GradientPurpleStart
    ),
    HealthTip(
        title = "Balanced Diet",
        description = "Antioxidant-rich foods protect against oxidative damage to DNA.",
        icon = Icons.Outlined.Restaurant,
        accentColor = GradientGreenStart
    ),
    HealthTip(
        title = "Regular Exercise",
        description = "150 min/week of moderate exercise is linked to longer telomeres.",
        icon = Icons.Outlined.FitnessCenter,
        accentColor = GradientAmberStart
    )
)

@Composable
private fun HealthTipsRow() {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = PaddingValues(end = 4.dp)
    ) {
        items(healthTips) { tip ->
            HealthTipCard(tip = tip)
        }
    }
}

@Composable
private fun HealthTipCard(tip: HealthTip) {
    ElevatedCard(
        modifier = Modifier
            .width(200.dp)
            .height(150.dp),
        colors = CardDefaults.elevatedCardColors(containerColor = Surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(14.dp)
        ) {
            Icon(
                imageVector = tip.icon,
                contentDescription = tip.title,
                tint = tip.accentColor,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = tip.title,
                style = MaterialTheme.typography.titleLarge.copy(fontSize = 14.sp),
                color = OnBackground,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = tip.description,
                style = MaterialTheme.typography.labelMedium,
                color = OnSurfaceVariant,
                fontSize = 11.sp,
                lineHeight = 15.sp,
                maxLines = 3
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recent Activity — Empty State
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun RecentActivityEmptyState() {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(containerColor = Surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Outlined.HealthAndSafety,
                contentDescription = "No recent activity",
                tint = OnSurfaceVariant.copy(alpha = 0.5f),
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "No analyses yet",
                style = MaterialTheme.typography.titleLarge.copy(fontSize = 16.sp),
                color = OnBackground,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Start your first facial or profile analysis\nto see your health insights here.",
                style = MaterialTheme.typography.bodyLarge.copy(fontSize = 13.sp),
                color = OnSurfaceVariant,
                textAlign = TextAlign.Center,
                lineHeight = 18.sp
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Footer Disclaimer
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun FooterDisclaimer() {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "For research and educational purposes only. Not intended for " +
                    "medical diagnosis, treatment, or prevention of any disease. " +
                    "Consult a qualified healthcare professional for medical advice.",
            style = MaterialTheme.typography.labelMedium,
            color = Color(0xFF616161),
            textAlign = TextAlign.Center,
            lineHeight = 16.sp
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Powered by Teloscopy v2.0",
            style = MaterialTheme.typography.labelMedium,
            color = Color(0xFF4A4A4A),
            textAlign = TextAlign.Center
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Quick Stat Card (dashboard metric)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun QuickStatCard(
    modifier: Modifier = Modifier,
    label: String,
    value: String,
    color: Color
) {
    ElevatedCard(
        modifier = modifier,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
                color = color
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
