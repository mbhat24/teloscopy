package com.teloscopy.app.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.teloscopy.app.ui.screens.AnalysisScreen
import com.teloscopy.app.ui.screens.HomeScreen
import com.teloscopy.app.ui.screens.ProfileAnalysisScreen
import com.teloscopy.app.ui.screens.ResultsScreen
import com.teloscopy.app.ui.screens.SettingsScreen
import com.teloscopy.app.viewmodel.AnalysisViewModel
import com.teloscopy.app.viewmodel.ProfileViewModel

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

/** The four bottom navigation bar items shown throughout the app. */
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
 * @param navController The [NavHostController] that drives navigation.
 * @param modifier      Optional [Modifier] applied to the [NavHost].
 * @param onOpenDrawer  Callback invoked when the user taps the hamburger
 *                      icon to open the navigation drawer.
 */
@Composable
fun TeloscopyNavHost(
    navController: NavHostController,
    modifier: Modifier = Modifier,
    onOpenDrawer: () -> Unit = {}
) {
    NavHost(
        navController = navController,
        startDestination = Screen.Home.route,
        modifier = modifier
    ) {
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

        // -- Results --------------------------------------------------------
        composable(
            route = Screen.Results.route,
            arguments = listOf(
                navArgument("jobId") { type = NavType.StringType }
            )
        ) { backStackEntry ->
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

        // -- Profile Analysis (no image) ------------------------------------
        composable(Screen.ProfileAnalysis.route) {
            val viewModel: ProfileViewModel = hiltViewModel()
            ProfileAnalysisScreen(
                viewModel = viewModel,
                onBack = {
                    navController.popBackStack()
                }
            )
        }

        // -- Settings -------------------------------------------------------
        composable(Screen.Settings.route) {
            SettingsScreen(
                onBack = {
                    navController.popBackStack()
                }
            )
        }
    }
}
