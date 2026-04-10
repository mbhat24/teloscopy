// TeloscopyApp.swift
// Teloscopy
//
// App entry point.
// Mirrors Android's TeloscopyApp.kt / MainActivity.kt

import SwiftUI

@main
struct TeloscopyApp: App {
    @StateObject private var settings = SettingsStore.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .preferredColorScheme(.dark)
        }
    }
}
