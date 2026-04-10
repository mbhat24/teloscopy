package com.teloscopy.app.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.MedicalServices
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.teloscopy.app.ui.screens.AnalysisScreen
import com.teloscopy.app.ui.screens.CONSENT_ACCEPTED_KEY
import com.teloscopy.app.ui.screens.ConsentScreen
import com.teloscopy.app.ui.screens.HealthCheckupScreen
import com.teloscopy.app.ui.screens.HomeScreen
import com.teloscopy.app.ui.screens.ProfileAnalysisScreen
import com.teloscopy.app.ui.screens.RequireConsent
import com.teloscopy.app.ui.screens.ResultsScreen
import com.teloscopy.app.ui.screens.SettingsScreen
import com.teloscopy.app.viewmodel.AnalysisViewModel
import com.teloscopy.app.viewmodel.HealthCheckupViewModel
import com.teloscopy.app.viewmodel.ProfileViewModel
import kotlinx.coroutines.flow.map

/**
 * Sealed class defining every navigable screen in the Teloscopy app.
 *
 * Each subclass carries its Compose Navigation route string. Screens
 * that accept arguments expose a helper function to build the concrete
 * route with the argument value substituted in.
 */
sealed class Screen(val route: String) {

    /** Landing / home screen with navigation to all features. */
    data object Home : Screen("home")

    /** Full analysis screen (image upload + user profile). */
    data object Analysis : Screen("analysis")

    /**
     * Results screen for a completed (or in-progress) analysis job.
     *
     * The route contains a `{jobId}` path parameter.  Use [createRoute]
     * to build a concrete route for navigation.
     */
    data object Results : Screen("results/{jobId}") {
        /** Build a concrete route with the given [jobId]. */
        fun createRoute(jobId: String): String = "results/$jobId"
    }

    /** Profile-only analysis (no image required). */
    data object ProfileAnalysis : Screen("profile_analysis")

    /** Application settings. */
    data object Settings : Screen("settings")

    /** Health checkup with lab report upload and analysis. */
    data object HealthCheckup : Screen("health_checkup")

    /** Legal consent / DPDP compliance screen (shown before first use). */
    data object Consent : Screen("consent")
}

// ─────────────────────────────────────────────────────────────────────────────
// Bottom Navigation Items
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Represents a single item in the bottom navigation bar.
 *
 * @param route           The navigation route this item targets.
 * @param label           The text label shown below the icon.
 * @param selectedIcon    Icon displayed when this item is selected.
 * @param unselectedIcon  Icon displayed when this item is not selected.
 */
data class BottomNavItem(
    val route: String,
    val label: String,
    val selectedIcon: ImageVector,
    val unselectedIcon: ImageVector
)

/** The five bottom navigation bar items shown throughout the app. */
val bottomNavItems = listOf(
    BottomNavItem(
        route = Screen.Home.route,
        label = "Home",
        selectedIcon = Icons.Filled.Home,
        unselectedIcon = Icons.Outlined.Home
    ),
    BottomNavItem(
        route = Screen.Analysis.route,
        label = "Analyze",
        selectedIcon = Icons.Filled.CameraAlt,
        unselectedIcon = Icons.Outlined.CameraAlt
    ),
    BottomNavItem(
        route = Screen.HealthCheckup.route,
        label = "Checkup",
        selectedIcon = Icons.Filled.MedicalServices,
        unselectedIcon = Icons.Outlined.MedicalServices
    ),
    BottomNavItem(
        route = Screen.ProfileAnalysis.route,
        label = "Profile",
        selectedIcon = Icons.Filled.Person,
        unselectedIcon = Icons.Outlined.Person
    ),
    BottomNavItem(
        route = Screen.Settings.route,
        label = "Settings",
        selectedIcon = Icons.Filled.Settings,
        unselectedIcon = Icons.Outlined.Settings
    )
)

// ─────────────────────────────────────────────────────────────────────────────
// Navigation Host
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Top-level navigation host for the Teloscopy app.
 *
 * Wires each [Screen] route to its composable destination, creates
 * Hilt-scoped ViewModels where needed, and forwards navigation
 * callbacks so screens remain decoupled from the [NavHostController].
 *
 * On first launch (or after consent withdrawal) the user is directed to
 * the [ConsentScreen] before they can access any other feature.
 *
 * @param navController The [NavHostController] that drives navigation.
 * @param dataStore     The Hilt-provided [DataStore] for reading/writing consent state.
 * @param modifier      Optional [Modifier] applied to the [NavHost].
 * @param onOpenDrawer  Callback invoked when the user taps the hamburger
 *                      icon to open the navigation drawer.
 */
@Composable
fun TeloscopyNavHost(
    navController: NavHostController,
    dataStore: DataStore<Preferences>,
    modifier: Modifier = Modifier,
    onOpenDrawer: () -> Unit = {}
) {
    // Observe consent state from DataStore
    val consentAccepted by dataStore.data
        .map { prefs -> prefs[CONSENT_ACCEPTED_KEY] ?: false }
        .collectAsState(initial = false)

    val startDestination = if (consentAccepted) {
        Screen.Home.route
    } else {
        Screen.Consent.route
    }

    NavHost(
        navController = navController,
        startDestination = startDestination,
        modifier = modifier
    ) {
        // -- Consent (DPDP Act compliance) ----------------------------------
        composable(Screen.Consent.route) {
            ConsentScreen(
                dataStore = dataStore,
                onConsentGranted = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Consent.route) { inclusive = true }
                    }
                }
            )
        }

        // -- Home -----------------------------------------------------------
        composable(Screen.Home.route) {
            HomeScreen(
                onNavigateToAnalysis = {
                    navController.navigate(Screen.Analysis.route)
                },
                onNavigateToProfile = {
                    navController.navigate(Screen.ProfileAnalysis.route)
                },
                onNavigateToSettings = {
                    navController.navigate(Screen.Settings.route)
                },
                onOpenDrawer = onOpenDrawer
            )
        }

        // -- Analysis (image + profile) -------------------------------------
        composable(Screen.Analysis.route) {
            RequireConsent(
                dataStore = dataStore,
                onConsentRequired = {
                    navController.navigate(Screen.Consent.route) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            ) {
                val viewModel: AnalysisViewModel = hiltViewModel()
                AnalysisScreen(
                    viewModel = viewModel,
                    onNavigateToResults = { jobId ->
                        navController.navigate(Screen.Results.createRoute(jobId))
                    },
                    onBack = {
                        navController.popBackStack()
                    }
                )
            }
        }

        // -- Results --------------------------------------------------------
        composable(
            route = Screen.Results.route,
            arguments = listOf(
                navArgument("jobId") { type = NavType.StringType }
            )
        ) { backStackEntry ->
            RequireConsent(
                dataStore = dataStore,
                onConsentRequired = {
                    navController.navigate(Screen.Consent.route) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            ) {
                val jobId = backStackEntry.arguments?.getString("jobId") ?: ""
                val viewModel: AnalysisViewModel = hiltViewModel()
                ResultsScreen(
                    jobId = jobId,
                    viewModel = viewModel,
                    onBack = {
                        navController.popBackStack()
                    }
                )
            }
        }

        // -- Profile Analysis (no image) ------------------------------------
        composable(Screen.ProfileAnalysis.route) {
            RequireConsent(
                dataStore = dataStore,
                onConsentRequired = {
                    navController.navigate(Screen.Consent.route) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            ) {
                val viewModel: ProfileViewModel = hiltViewModel()
                ProfileAnalysisScreen(
                    viewModel = viewModel,
                    onBack = {
                        navController.popBackStack()
                    }
                )
            }
        }

        // -- Health Checkup (lab report upload) ------------------------------
        composable(Screen.HealthCheckup.route) {
            RequireConsent(
                dataStore = dataStore,
                onConsentRequired = {
                    navController.navigate(Screen.Consent.route) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            ) {
                val viewModel: HealthCheckupViewModel = hiltViewModel()
                HealthCheckupScreen(
                    viewModel = viewModel,
                    onBack = {
                        navController.popBackStack()
                    }
                )
            }
        }

        // -- Settings -------------------------------------------------------
        composable(Screen.Settings.route) {
            SettingsScreen(
                onBack = {
                    navController.popBackStack()
                },
                dataStore = dataStore,
                onWithdrawConsent = {
                    navController.navigate(Screen.Consent.route) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            )
        }
    }
}
