// ContentView.swift
// Teloscopy
//
// Alternate root tab-based navigation (not used at runtime — see TeloscopyApp.swift).
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
    @State private var selectedTab: AppTab = .home

    var body: some View {
        ZStack(alignment: .bottom) {
            Group {
                switch selectedTab {
                case .home:
                    HomeView()
                case .analysis:
                    AnalysisView()
                case .settings:
                    SettingsView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Custom bottom tab bar
            customTabBar
        }
        .ignoresSafeArea(.keyboard)
    }

    // MARK: - Custom Tab Bar

    private var customTabBar: some View {
        HStack {
            ForEach(AppTab.allCases, id: \.self) { tab in
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { selectedTab = tab } }) {
                    VStack(spacing: 4) {
                        Image(systemName: selectedTab == tab ? tab.icon : tab.iconOutline)
                            .font(.system(size: 20, weight: selectedTab == tab ? .semibold : .regular))
                            .foregroundColor(selectedTab == tab ? .accentColor : .secondary)

                        Text(tab.rawValue)
                            .font(.system(size: 10, weight: selectedTab == tab ? .semibold : .regular))
                            .foregroundColor(selectedTab == tab ? .accentColor : .secondary)
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
                .fill(Color(red: 0.067, green: 0.082, blue: 0.157))
                .shadow(color: .black.opacity(0.3), radius: 8, y: -4)
                .ignoresSafeArea(edges: .bottom)
        )
    }
}
