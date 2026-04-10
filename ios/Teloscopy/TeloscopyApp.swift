// TeloscopyApp.swift
// Teloscopy
//
// Main application entry point with TabView navigation.
// Configures the app environment, services, and appearance.
//

import SwiftUI
import Combine

// MARK: - App Theme

struct TeloscopyTheme {
    static let primaryBlue = Color(red: 0.129, green: 0.459, blue: 0.855)
    static let darkBlue = Color(red: 0.059, green: 0.298, blue: 0.647)
    static let lightBlue = Color(red: 0.878, green: 0.925, blue: 0.980)
    static let accentTeal = Color(red: 0.180, green: 0.710, blue: 0.769)
    static let medicalWhite = Color(red: 0.969, green: 0.976, blue: 0.988)
    static let warningOrange = Color(red: 0.957, green: 0.620, blue: 0.188)
    static let successGreen = Color(red: 0.224, green: 0.749, blue: 0.467)
    static let errorRed = Color(red: 0.890, green: 0.243, blue: 0.282)
    static let textPrimary = Color(red: 0.133, green: 0.157, blue: 0.208)
    static let textSecondary = Color(red: 0.467, green: 0.506, blue: 0.573)
    static let cardBackground = Color(UIColor.systemBackground)
    static let surfaceBackground = Color(UIColor.secondarySystemBackground)
    
    static let cardShadow = Color.black.opacity(0.06)
    static let cornerRadius: CGFloat = 14
    static let smallCornerRadius: CGFloat = 8
}

// MARK: - Global Cancellable Storage

final class AppCancellables {
    static let shared = AppCancellables()
    var cancellables = Set<AnyCancellable>()
    private init() {}
}

// MARK: - App Entry Point

@main
struct TeloscopyApp: App {
    @ObservedObject private var apiService = APIService.shared
    @ObservedObject private var syncManager = SyncManager.shared
    @AppStorage("appearance_mode") private var appearanceMode: String = "system"
    @AppStorage("consent_accepted") private var consentAccepted = false
    
    private static let currentConsentVersion = "1.0"
    
    init() {
        configureAppearance()
    }
    
    var body: some Scene {
        WindowGroup {
            if consentAccepted && UserDefaults.standard.string(forKey: "consent_version") == Self.currentConsentVersion {
                MainTabView()
                    .environmentObject(apiService)
                    .environmentObject(syncManager)
                    .preferredColorScheme(colorScheme)
                    .onAppear {
                        apiService.checkServerHealth()
                            .sink(receiveValue: { _ in })
                            .store(in: &AppCancellables.shared.cancellables)
                    }
            } else {
                ConsentView(onAccept: {
                    consentAccepted = true
                })
                .preferredColorScheme(colorScheme)
            }
        }
    }
    
    private var colorScheme: ColorScheme? {
        switch appearanceMode {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }
    
    private func configureAppearance() {
        // Tab bar appearance
        let tabBarAppearance = UITabBarAppearance()
        tabBarAppearance.configureWithDefaultBackground()
        UITabBar.appearance().scrollEdgeAppearance = tabBarAppearance
        UITabBar.appearance().standardAppearance = tabBarAppearance
        
        // Navigation bar appearance
        let navAppearance = UINavigationBarAppearance()
        navAppearance.configureWithDefaultBackground()
        navAppearance.titleTextAttributes = [
            .foregroundColor: UIColor(TeloscopyTheme.textPrimary)
        ]
        navAppearance.largeTitleTextAttributes = [
            .foregroundColor: UIColor(TeloscopyTheme.textPrimary)
        ]
        UINavigationBar.appearance().standardAppearance = navAppearance
        UINavigationBar.appearance().scrollEdgeAppearance = navAppearance
    }
}

// MARK: - Main Tab View

struct MainTabView: View {
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var syncManager: SyncManager
    @State private var selectedTab = 0
    @State private var showLoginSheet = false
    
    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                HomeView()
            }
            .tabItem {
                Label("Home", systemImage: "house.fill")
            }
            .tag(0)
            
            NavigationStack {
                AnalysisView()
            }
            .tabItem {
                Label("Analyze", systemImage: "microscope")
            }
            .tag(1)
            
            NavigationStack {
                HealthCheckupView()
            }
            .tabItem {
                Label("Checkup", systemImage: "cross.case.fill")
            }
            .tag(2)
            
            NavigationStack {
                ResultsView()
            }
            .tabItem {
                Label("Results", systemImage: "chart.bar.xaxis")
            }
            .tag(3)
            
            NavigationStack {
                ProfileView()
            }
            .tabItem {
                Label("Profile", systemImage: "person.crop.circle")
            }
            .tag(4)
            
            NavigationStack {
                SettingsView()
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape.fill")
            }
            .tag(5)
        }
        .tint(TeloscopyTheme.primaryBlue)
        .sheet(isPresented: $showLoginSheet) {
            LoginView()
                .environmentObject(apiService)
        }
    }
}

// MARK: - Login View

struct LoginView: View {
    @EnvironmentObject var apiService: APIService
    @Environment(\.dismiss) var dismiss
    
    @State private var username = ""
    @State private var password = ""
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var cancellables = Set<AnyCancellable>()
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 32) {
                    // Logo area
                    VStack(spacing: 12) {
                        Image(systemName: "dna")
                            .font(.system(size: 60))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [TeloscopyTheme.primaryBlue, TeloscopyTheme.accentTeal],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                        
                        Text("Teloscopy")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(TeloscopyTheme.textPrimary)
                        
                        Text("Genomic Telomere Analysis")
                            .font(.subheadline)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                    .padding(.top, 40)
                    
                    // Form fields
                    VStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Username")
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                            
                            TextField("Enter your username", text: $username)
                                .textFieldStyle(.plain)
                                .padding(14)
                                .background(TeloscopyTheme.surfaceBackground)
                                .cornerRadius(TeloscopyTheme.smallCornerRadius)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()
                        }
                        
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Password")
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                            
                            SecureField("Enter your password", text: $password)
                                .textFieldStyle(.plain)
                                .padding(14)
                                .background(TeloscopyTheme.surfaceBackground)
                                .cornerRadius(TeloscopyTheme.smallCornerRadius)
                        }
                    }
                    .padding(.horizontal)
                    
                    // Error message
                    if let error = errorMessage {
                        HStack {
                            Image(systemName: "exclamationmark.circle.fill")
                                .foregroundColor(TeloscopyTheme.errorRed)
                            Text(error)
                                .font(.caption)
                                .foregroundColor(TeloscopyTheme.errorRed)
                        }
                        .padding(.horizontal)
                    }
                    
                    // Login button
                    Button(action: performLogin) {
                        HStack {
                            if isLoading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.9)
                            }
                            Text(isLoading ? "Signing In..." : "Sign In")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            LinearGradient(
                                colors: [TeloscopyTheme.primaryBlue, TeloscopyTheme.darkBlue],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .foregroundColor(.white)
                        .cornerRadius(TeloscopyTheme.cornerRadius)
                    }
                    .disabled(username.isEmpty || password.isEmpty || isLoading)
                    .opacity(username.isEmpty || password.isEmpty ? 0.6 : 1.0)
                    .padding(.horizontal)
                    
                    // Skip button
                    Button("Continue without signing in") {
                        dismiss()
                    }
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
    
    private func performLogin() {
        isLoading = true
        errorMessage = nil
        
        apiService.login(username: username, password: password)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { completion in
                    isLoading = false
                    if case .failure(let error) = completion {
                        errorMessage = error.localizedDescription
                    }
                },
                receiveValue: { _ in
                    dismiss()
                }
            )
            .store(in: &cancellables)
    }
}

// MARK: - Reusable Card Modifier

struct CardModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(TeloscopyTheme.cardBackground)
            .cornerRadius(TeloscopyTheme.cornerRadius)
            .shadow(color: TeloscopyTheme.cardShadow, radius: 8, x: 0, y: 2)
    }
}

extension View {
    func cardStyle() -> some View {
        modifier(CardModifier())
    }
}
