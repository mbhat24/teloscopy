// SettingsStore.swift
// Teloscopy
//
// UserDefaults wrapper for app settings.
// Mirrors Android's SharedPreferences / DataStore usage.

import Foundation
import SwiftUI

@MainActor
class SettingsStore: ObservableObject {
    static let shared = SettingsStore()

    private let defaults = UserDefaults.standard

    private enum Keys {
        static let serverURL = "server_url"
        static let hasCompletedOnboarding = "has_completed_onboarding"
        static let lastJobId = "last_job_id"
    }

    @Published var serverURL: String {
        didSet {
            defaults.set(serverURL, forKey: Keys.serverURL)
        }
    }

    @Published var hasCompletedOnboarding: Bool {
        didSet {
            defaults.set(hasCompletedOnboarding, forKey: Keys.hasCompletedOnboarding)
        }
    }

    @Published var lastJobId: String? {
        didSet {
            defaults.set(lastJobId, forKey: Keys.lastJobId)
        }
    }

    var baseURL: URL? {
        var urlString = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if urlString.hasSuffix("/") {
            urlString = String(urlString.dropLast())
        }
        return URL(string: urlString)
    }

    init() {
        self.serverURL = defaults.string(forKey: Keys.serverURL) ?? "http://localhost:8000"
        self.hasCompletedOnboarding = defaults.bool(forKey: Keys.hasCompletedOnboarding)
        self.lastJobId = defaults.string(forKey: Keys.lastJobId)
    }

    func resetToDefaults() {
        serverURL = "http://localhost:8000"
        hasCompletedOnboarding = false
        lastJobId = nil
    }
}
