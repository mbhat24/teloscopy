package com.teloscopy.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Help
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.Science
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.teloscopy.app.ui.navigation.Screen
import com.teloscopy.app.ui.navigation.TeloscopyNavHost
import com.teloscopy.app.ui.theme.Background
import com.teloscopy.app.ui.theme.DrawerSurface
import com.teloscopy.app.ui.theme.NavBarSurface
import com.teloscopy.app.ui.theme.OnBackground
import com.teloscopy.app.ui.theme.OnSurfaceVariant
import com.teloscopy.app.ui.theme.Primary
import com.teloscopy.app.ui.theme.TeloscopyTheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        setContent {
            TeloscopyTheme {
                val navController = rememberNavController()
                val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
                val scope = rememberCoroutineScope()
                val currentBackStackEntry by navController.currentBackStackEntryAsState()
                val currentRoute = currentBackStackEntry?.destination?.route

                ModalNavigationDrawer(
                    drawerState = drawerState,
                    gesturesEnabled = currentRoute == Screen.Home.route,
                    drawerContent = {
                        DrawerContent(
                            currentRoute = currentRoute,
                            onNavigate = { route ->
                                scope.launch { drawerState.close() }
                                navController.navigate(route) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            onCloseDrawer = {
                                scope.launch { drawerState.close() }
                            }
                        )
                    }
                ) {
                    Scaffold(
                        containerColor = Background,
                        bottomBar = {
                            if (currentRoute in listOf(
                                    Screen.Home.route,
                                    Screen.Analysis.route,
                                    Screen.ProfileAnalysis.route,
                                    Screen.Settings.route
                                )
                            ) {
                                NavigationBar(
                                    containerColor = MaterialTheme.colorScheme.surface,
                                    tonalElevation = 0.dp
                                ) {
                                    NavigationBarItem(
                                        selected = currentRoute == Screen.Home.route,
                                        onClick = {
                                            navController.navigate(Screen.Home.route) {
                                                popUpTo(Screen.Home.route) { inclusive = true }
                                                launchSingleTop = true
                                            }
                                        },
                                        icon = {
                                            Icon(
                                                Icons.Outlined.Home,
                                                contentDescription = "Home"
                                            )
                                        },
                                        label = { Text("Home") },
                                        colors = NavigationBarItemDefaults.colors(
                                            selectedIconColor = MaterialTheme.colorScheme.primary,
                                            selectedTextColor = MaterialTheme.colorScheme.primary,
                                            indicatorColor = MaterialTheme.colorScheme.primary.copy(
                                                alpha = 0.12f
                                            )
                                        )
                                    )
                                    NavigationBarItem(
                                        selected = currentRoute == Screen.Analysis.route,
                                        onClick = {
                                            navController.navigate(Screen.Analysis.route) {
                                                popUpTo(Screen.Home.route)
                                                launchSingleTop = true
                                            }
                                        },
                                        icon = {
                                            Icon(
                                                Icons.Outlined.Science,
                                                contentDescription = "Analyze"
                                            )
                                        },
                                        label = { Text("Analyze") },
                                        colors = NavigationBarItemDefaults.colors(
                                            selectedIconColor = MaterialTheme.colorScheme.primary,
                                            selectedTextColor = MaterialTheme.colorScheme.primary,
                                            indicatorColor = MaterialTheme.colorScheme.primary.copy(
                                                alpha = 0.12f
                                            )
                                        )
                                    )
                                    NavigationBarItem(
                                        selected = currentRoute == Screen.Settings.route,
                                        onClick = {
                                            navController.navigate(Screen.Settings.route) {
                                                popUpTo(Screen.Home.route)
                                                launchSingleTop = true
                                            }
                                        },
                                        icon = {
                                            Icon(
                                                Icons.Outlined.Settings,
                                                contentDescription = "Settings"
                                            )
                                        },
                                        label = { Text("Settings") },
                                        colors = NavigationBarItemDefaults.colors(
                                            selectedIconColor = MaterialTheme.colorScheme.primary,
                                            selectedTextColor = MaterialTheme.colorScheme.primary,
                                            indicatorColor = MaterialTheme.colorScheme.primary.copy(
                                                alpha = 0.12f
                                            )
                                        )
                                    )
                                }
                            }
                        }
                    ) { innerPadding ->
                        Surface(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(innerPadding),
                            color = MaterialTheme.colorScheme.background
                        ) {
                            TeloscopyNavHost(
                                navController = navController,
                                onOpenDrawer = {
                                    scope.launch { drawerState.open() }
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation Drawer Content
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Drawer item descriptor used to build the sidebar menu.
 */
private data class DrawerItem(
    val route: String?,
    val label: String,
    val icon: ImageVector
)

/** Primary navigation items shown in the drawer (matching bottom bar). */
private val primaryDrawerItems = listOf(
    DrawerItem(Screen.Home.route, "Home", Icons.Outlined.Home),
    DrawerItem(Screen.Analysis.route, "Analyze", Icons.Outlined.CameraAlt),
    DrawerItem(Screen.ProfileAnalysis.route, "Profile", Icons.Outlined.Person),
    DrawerItem(Screen.Settings.route, "Settings", Icons.Outlined.Settings)
)

/** Extra drawer-only items. Route is null because they are non-navigating. */
private val secondaryDrawerItems = listOf(
    DrawerItem(null, "About", Icons.Outlined.Info),
    DrawerItem(null, "Help", Icons.Outlined.Help)
)

/**
 * Content composable for the [ModalNavigationDrawer].
 *
 * Displays the app brand at the top, followed by primary navigation items
 * (matching the bottom bar) and secondary utility items (About, Help).
 */
@Composable
private fun DrawerContent(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onCloseDrawer: () -> Unit
) {
    ModalDrawerSheet(
        drawerContainerColor = DrawerSurface
    ) {
        // ── App brand ────────────────────────────────────────────────────
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "Teloscopy",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = Primary,
            modifier = Modifier.padding(horizontal = 28.dp)
        )
        Text(
            text = "Genomic Intelligence",
            fontSize = 14.sp,
            color = OnSurfaceVariant,
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 4.dp)
        )
        Spacer(modifier = Modifier.height(16.dp))

        // ── Primary navigation items ─────────────────────────────────────
        primaryDrawerItems.forEach { item ->
            NavigationDrawerItem(
                label = { Text(item.label, color = OnBackground) },
                selected = currentRoute == item.route,
                onClick = {
                    item.route?.let { onNavigate(it) }
                },
                icon = {
                    Icon(
                        imageVector = item.icon,
                        contentDescription = item.label,
                        tint = if (currentRoute == item.route) Primary
                        else OnSurfaceVariant
                    )
                },
                colors = NavigationDrawerItemDefaults.colors(
                    selectedContainerColor = Primary.copy(alpha = 0.12f),
                    unselectedContainerColor = Color.Transparent
                ),
                modifier = Modifier.padding(horizontal = 12.dp)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // ── Secondary items (About, Help) ────────────────────────────────
        secondaryDrawerItems.forEach { item ->
            NavigationDrawerItem(
                label = { Text(item.label, color = OnBackground) },
                selected = false,
                onClick = onCloseDrawer,
                icon = {
                    Icon(
                        imageVector = item.icon,
                        contentDescription = item.label,
                        tint = OnSurfaceVariant
                    )
                },
                colors = NavigationDrawerItemDefaults.colors(
                    unselectedContainerColor = Color.Transparent
                ),
                modifier = Modifier.padding(horizontal = 12.dp)
            )
        }
    }
}
