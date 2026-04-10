// SettingsView.swift
// Teloscopy
//
// Server URL configuration + about section.
// Mirrors Android's SettingsScreen.kt composable.

import SwiftUI

// MARK: - Connection Status

private enum ConnectionStatus: Equatable {
    case unknown
    case testing
    case connected(version: String)
    case failed(reason: String)

    var displayText: String {
        switch self {
        case .unknown: return "Not tested yet"
        case .testing: return "Testing connection..."
        case .connected(let version): return "Connected (v\(version))"
        case .failed(let reason): return reason
        }
    }

    var color: Color {
        switch self {
        case .unknown: return .tsTextSecondary
        case .testing: return .tsPurple
        case .connected: return .tsGreen
        case .failed: return .tsError
        }
    }

    var icon: String {
        switch self {
        case .unknown: return "cloud"
        case .testing: return "arrow.triangle.2.circlepath"
        case .connected: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.circle.fill"
        }
    }
}

private let APP_VERSION = "2.0.0"
private let APP_BUILD = "2024.06"

// MARK: - Settings View

struct SettingsView: View {
    @EnvironmentObject var settings: SettingsStore
    @State private var serverURL: String = ""
    @State private var connectionStatus: ConnectionStatus = .unknown
    @State private var saveMessage: String?
    @State private var showResetConfirm = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    serverConfigSection
                    serverStatusCard
                    dataPrivacySection
                    disclaimerSection
                    aboutSection
                    footerSection
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                .padding(.bottom, 32)
            }
            .background(Color.tsBackground)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Color.tsBackground, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .onAppear {
                serverURL = settings.serverURL
            }
            .alert("Reset Settings", isPresented: $showResetConfirm) {
                Button("Cancel", role: .cancel) { }
                Button("Reset", role: .destructive) {
                    settings.resetToDefaults()
                    serverURL = settings.serverURL
                    connectionStatus = .unknown
                }
            } message: {
                Text("This will reset all settings to their default values.")
            }
        }
    }

    // MARK: - Server Configuration

    private var serverConfigSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(title: "SERVER CONFIGURATION", icon: "cloud.fill")

            VStack(alignment: .leading, spacing: 16) {
                Text("Backend Server URL")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.tsTextPrimary)

                Text("The URL of the Teloscopy analysis backend. Use localhost for simulator or your server's IP address.")
                    .font(.system(size: 13))
                    .foregroundColor(.tsTextSecondary)

                TextField("http://localhost:8000", text: $serverURL)
                    .textFieldStyle(TeloscopyTextFieldStyle())
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
                    .keyboardType(.URL)

                HStack(spacing: 12) {
                    // Save button
                    Button(action: saveServerURL) {
                        HStack(spacing: 6) {
                            Image(systemName: "square.and.arrow.down")
                                .font(.system(size: 14))
                            Text("Save")
                                .font(.system(size: 14, weight: .semibold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .foregroundColor(.tsBackground)
                        .background(RoundedRectangle(cornerRadius: 10).fill(Color.tsCyan))
                    }

                    // Test button
                    Button(action: testConnection) {
                        HStack(spacing: 6) {
                            if connectionStatus == .testing {
                                ProgressView()
                                    .tint(.tsPurple)
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "cloud")
                                    .font(.system(size: 14))
                            }
                            Text("Test")
                                .font(.system(size: 14, weight: .semibold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .foregroundColor(.tsPurple)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color.tsPurple.opacity(0.2))
                        )
                    }
                    .disabled(connectionStatus == .testing)
                }

                if let saveMessage = saveMessage {
                    Text(saveMessage)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.tsGreen)
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.tsSurface)
            )
        }
    }

    // MARK: - Server Status Card

    private var serverStatusCard: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(connectionStatus.color.opacity(0.15))
                    .frame(width: 40, height: 40)

                if connectionStatus == .testing {
                    ProgressView()
                        .tint(connectionStatus.color)
                        .scaleEffect(0.8)
                } else {
                    Image(systemName: connectionStatus.icon)
                        .font(.system(size: 18))
                        .foregroundColor(connectionStatus.color)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("Server Status")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.tsTextPrimary)

                Text(connectionStatus.displayText)
                    .font(.system(size: 13))
                    .foregroundColor(connectionStatus.color)
            }

            Spacer()
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.tsSurface)
        )
    }

    // MARK: - Data & Privacy

    private var dataPrivacySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(title: "DATA & PRIVACY", icon: "hand.raised.fill")

            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "hand.raised.fill")
                    .font(.system(size: 16))
                    .foregroundColor(.tsTextSecondary)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Privacy & Legal")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.tsTextPrimary)

                    Text("Your genetic data is processed locally and on your configured server. No data is sent to third parties. Review our privacy policy for details.")
                        .font(.system(size: 13))
                        .foregroundColor(.tsTextSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.tsSurface)
            )
        }
    }

    // MARK: - Disclaimer

    private var disclaimerSection: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 16))
                .foregroundColor(.tsWarning)

            VStack(alignment: .leading, spacing: 6) {
                Text("Disclaimer")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.tsWarning)

                Text("For research and educational purposes only. Not intended for medical diagnosis or treatment. Results should not replace professional medical advice. Always consult a qualified healthcare provider.")
                    .font(.system(size: 13))
                    .foregroundColor(.tsTextSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(hex: 0x1A1520))
        )
    }

    // MARK: - About

    private var aboutSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(title: "ABOUT", icon: "info.circle.fill")

            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color.tsCyan.opacity(0.15))
                            .frame(width: 40, height: 40)
                        Image(systemName: "flask.fill")
                            .font(.system(size: 18))
                            .foregroundColor(.tsCyan)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Teloscopy")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.tsCyan)
                        Text("Genomic Intelligence Platform")
                            .font(.system(size: 12))
                            .foregroundColor(.tsTextSecondary)
                    }
                }

                Divider()
                    .background(Color.tsTextSecondary.opacity(0.2))

                infoRow(label: "Version", value: APP_VERSION)
                infoRow(label: "Build", value: APP_BUILD)
                infoRow(label: "Platform", value: "iOS \(UIDevice.current.systemVersion)")
                infoRow(label: "Description", value: "Biological age estimation, disease risk prediction, and personalized nutrition recommendations powered by genomic and facial analysis.")
                infoRow(label: "Engine", value: "Teloscopy Core v2.0")
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.tsSurface)
            )
        }
    }

    // MARK: - Footer

    private var footerSection: some View {
        VStack(spacing: 12) {
            Button(action: { showResetConfirm = true }) {
                Text("Reset to Defaults")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.tsError)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.tsError.opacity(0.5), lineWidth: 1)
                    )
            }

            Text("© 2024 Teloscopy Project. All rights reserved.")
                .font(.system(size: 12))
                .foregroundColor(.tsTextSecondary.opacity(0.5))
        }
    }

    // MARK: - Helper Views

    private func sectionHeader(title: String, icon: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 12))
                .foregroundColor(.tsCyan)
            Text(title)
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(.tsCyan)
                .tracking(1.5)
        }
    }

    private func infoRow(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.tsTextSecondary)
            Text(value)
                .font(.system(size: 14))
                .foregroundColor(.tsTextPrimary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Actions

    private func saveServerURL() {
        let trimmed = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
        settings.serverURL = trimmed.hasSuffix("/") ? String(trimmed.dropLast()) : trimmed
        saveMessage = "Saved"
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            saveMessage = nil
        }
    }

    private func testConnection() {
        connectionStatus = .testing
        saveMessage = nil

        Task {
            do {
                let health = try await APIService.shared.healthCheck()
                connectionStatus = .connected(version: health.version ?? "unknown")
            } catch let error as APIError {
                connectionStatus = .failed(reason: error.localizedDescription)
            } catch {
                connectionStatus = .failed(reason: error.localizedDescription)
            }
        }
    }
}

#Preview {
    SettingsView()
        .environmentObject(SettingsStore.shared)
}
