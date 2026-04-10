// ContentView.swift
// Teloscopy
//
// Root tab-based navigation.
// Mirrors Android's NavGraph.kt + MainActivity.kt bottom navigation.

import SwiftUI

enum AppTab: String, CaseIterable {
    case home = "Home"
    case analysis = "Analyze"
    case settings = "Settings"

    var icon: String {
        switch self {
        case .home: return "house.fill"
        case .analysis: return "camera.fill"
        case .settings: return "gearshape.fill"
        }
    }

    var iconOutline: String {
        switch self {
        case .home: return "house"
        case .analysis: return "camera"
        case .settings: return "gearshape"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var settings: SettingsStore
    @State private var selectedTab: AppTab = .home
    @State private var resultsJobId: String?
    @State private var showResults = false

    var body: some View {
        ZStack(alignment: .bottom) {
            // Tab content
            Group {
                switch selectedTab {
                case .home:
                    HomeView(
                        onNavigateToAnalysis: { selectedTab = .analysis },
                        onNavigateToProfile: { navigateToProfile() }
                    )
                case .analysis:
                    AnalysisView(onNavigateToResults: { jobId in
                        resultsJobId = jobId
                        showResults = true
                    })
                case .settings:
                    SettingsView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Custom bottom tab bar
            customTabBar
        }
        .ignoresSafeArea(.keyboard)
        .sheet(isPresented: $showResults) {
            if let jobId = resultsJobId {
                ResultsView(jobId: jobId)
            }
        }
    }

    @State private var showProfileSheet = false

    private func navigateToProfile() {
        showProfileSheet = true
    }

    // MARK: - Custom Tab Bar

    private var customTabBar: some View {
        HStack {
            ForEach(AppTab.allCases, id: \.self) { tab in
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { selectedTab = tab } }) {
                    VStack(spacing: 4) {
                        Image(systemName: selectedTab == tab ? tab.icon : tab.iconOutline)
                            .font(.system(size: 20, weight: selectedTab == tab ? .semibold : .regular))
                            .foregroundColor(selectedTab == tab ? .tsAccent : .tsTextSecondary)

                        Text(tab.rawValue)
                            .font(.system(size: 10, weight: selectedTab == tab ? .semibold : .regular))
                            .foregroundColor(selectedTab == tab ? .tsAccent : .tsTextSecondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.top, 8)
        .padding(.bottom, 20)
        .background(
            Rectangle()
                .fill(Color(hex: 0x111528))
                .shadow(color: .black.opacity(0.3), radius: 8, y: -4)
                .ignoresSafeArea(edges: .bottom)
        )
        .sheet(isPresented: $showProfileSheet) {
            ProfileAnalysisView(onNavigateToResults: { jobId in
                showProfileSheet = false
                resultsJobId = jobId
                showResults = true
            })
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(SettingsStore.shared)
}
