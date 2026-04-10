// SettingsView.swift
// Teloscopy
//
// Application settings: server configuration, notifications,
// appearance, data management, and about information.
//

import SwiftUI
import Combine

struct SettingsView: View {
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var syncManager: SyncManager
    
    @AppStorage("server_url") private var serverURL = APIConfiguration.defaultBaseURL
    @AppStorage("appearance_mode") private var appearanceMode = "system"
    @AppStorage("notifications_enabled") private var notificationsEnabled = true
    @AppStorage("notification_analysis_complete") private var notifyAnalysisComplete = true
    @AppStorage("notification_sync_errors") private var notifySyncErrors = true
    @AppStorage("auto_sync_enabled") private var autoSyncEnabled = true
    @AppStorage("auto_upload_on_wifi") private var autoUploadOnWifi = true
    @AppStorage("image_compression_quality") private var compressionQuality = 0.85
    @AppStorage("keep_local_images") private var keepLocalImages = true
    
    @State private var showServerURLEditor = false
    @State private var tempServerURL = ""
    @State private var isTestingConnection = false
    @State private var connectionTestResult: ConnectionTestResult?
    @State private var showClearCacheConfirm = false
    @State private var showResetConfirm = false
    @State private var showLogoutConfirm = false
    @State private var cancellables = Set<AnyCancellable>()
    
    enum ConnectionTestResult {
        case success
        case failure(String)
    }
    
    var body: some View {
        List {
            serverSection
            accountSection
            syncSection
            notificationSection
            appearanceSection
            dataSection
            aboutSection
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Settings")
        .sheet(isPresented: $showServerURLEditor) {
            serverURLEditor
        }
        .alert("Clear Cache?", isPresented: $showClearCacheConfirm) {
            Button("Cancel", role: .cancel) { }
            Button("Clear", role: .destructive) {
                syncManager.clearCache()
            }
        } message: {
            Text("This will remove all locally cached analyses and results. Synced data on the server will not be affected.")
        }
        .alert("Reset All Settings?", isPresented: $showResetConfirm) {
            Button("Cancel", role: .cancel) { }
            Button("Reset", role: .destructive) { resetAllSettings() }
        } message: {
            Text("This will reset all settings to their default values. Your analysis data will not be affected.")
        }
        .alert("Sign Out?", isPresented: $showLogoutConfirm) {
            Button("Cancel", role: .cancel) { }
            Button("Sign Out", role: .destructive) {
                apiService.logout()
            }
        } message: {
            Text("You'll need to sign in again to sync your data.")
        }
    }
    
    // MARK: - Server Configuration
    
    private var serverSection: some View {
        Section {
            // Server URL
            Button(action: {
                tempServerURL = serverURL
                showServerURLEditor = true
            }) {
                HStack {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Server URL")
                                .font(.subheadline)
                                .foregroundColor(TeloscopyTheme.textPrimary)
                            Text(serverURL)
                                .font(.caption)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                                .lineLimit(1)
                        }
                    } icon: {
                        Image(systemName: "server.rack")
                            .foregroundColor(TeloscopyTheme.primaryBlue)
                    }
                    
                    Spacer()
                    
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
            }
            
            // Connection status
            HStack {
                Label {
                    Text("Connection Status")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: apiService.isServerReachable ? "wifi" : "wifi.slash")
                        .foregroundColor(apiService.isServerReachable ? TeloscopyTheme.successGreen : TeloscopyTheme.errorRed)
                }
                
                Spacer()
                
                if isTestingConnection {
                    ProgressView()
                        .scaleEffect(0.8)
                } else {
                    Text(apiService.isServerReachable ? "Connected" : "Disconnected")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(apiService.isServerReachable ? TeloscopyTheme.successGreen : TeloscopyTheme.errorRed)
                }
            }
            
            // Test connection button
            Button(action: testConnection) {
                Label("Test Connection", systemImage: "arrow.triangle.2.circlepath")
                    .font(.subheadline)
            }
            .disabled(isTestingConnection)
            
            if let result = connectionTestResult {
                switch result {
                case .success:
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(TeloscopyTheme.successGreen)
                        Text("Connection successful")
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.successGreen)
                    }
                case .failure(let message):
                    HStack {
                        Image(systemName: "exclamationmark.circle.fill")
                            .foregroundColor(TeloscopyTheme.errorRed)
                        Text(message)
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.errorRed)
                    }
                }
            }
        } header: {
            Text("Server Configuration")
        } footer: {
            Text("Configure the Teloscopy analysis server address. Default: \(APIConfiguration.defaultBaseURL)")
        }
    }
    
    // MARK: - Account Section
    
    private var accountSection: some View {
        Section("Account") {
            if apiService.isAuthenticated, let user = apiService.currentUser {
                HStack {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(user.fullName)
                                .font(.subheadline)
                                .foregroundColor(TeloscopyTheme.textPrimary)
                            Text(user.email)
                                .font(.caption)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                        }
                    } icon: {
                        ZStack {
                            Circle()
                                .fill(TeloscopyTheme.primaryBlue.opacity(0.15))
                                .frame(width: 32, height: 32)
                            Text(user.fullName.prefix(1).uppercased())
                                .font(.caption)
                                .fontWeight(.bold)
                                .foregroundColor(TeloscopyTheme.primaryBlue)
                        }
                    }
                }
                
                Button(role: .destructive, action: { showLogoutConfirm = true }) {
                    Label("Sign Out", systemImage: "rectangle.portrait.and.arrow.right")
                        .font(.subheadline)
                }
            } else {
                Button(action: { }) {
                    Label("Sign In", systemImage: "person.crop.circle")
                        .font(.subheadline)
                }
            }
        }
    }
    
    // MARK: - Sync Section
    
    private var syncSection: some View {
        Section {
            Toggle(isOn: $autoSyncEnabled) {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Auto Sync")
                            .font(.subheadline)
                        Text("Periodically sync with server")
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                } icon: {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .foregroundColor(TeloscopyTheme.primaryBlue)
                }
            }
            .tint(TeloscopyTheme.primaryBlue)
            
            Toggle(isOn: $autoUploadOnWifi) {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Upload on Wi-Fi Only")
                            .font(.subheadline)
                        Text("Save cellular data")
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                } icon: {
                    Image(systemName: "wifi")
                        .foregroundColor(TeloscopyTheme.accentTeal)
                }
            }
            .tint(TeloscopyTheme.primaryBlue)
            
            // Pending uploads
            HStack {
                Label {
                    Text("Pending Uploads")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "arrow.up.circle")
                        .foregroundColor(TeloscopyTheme.warningOrange)
                }
                
                Spacer()
                
                Text("\(syncManager.pendingUploads.count)")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            // Sync now button
            Button(action: { syncManager.performSync() }) {
                HStack {
                    Label("Sync Now", systemImage: "arrow.clockwise")
                        .font(.subheadline)
                    
                    Spacer()
                    
                    if syncManager.syncState == .syncing {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Text(syncManager.syncState.displayName)
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                }
            }
            .disabled(syncManager.syncState == .syncing)
        } header: {
            Text("Data Sync")
        }
    }
    
    // MARK: - Notifications Section
    
    private var notificationSection: some View {
        Section {
            Toggle(isOn: $notificationsEnabled) {
                Label {
                    Text("Enable Notifications")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "bell.fill")
                        .foregroundColor(TeloscopyTheme.warningOrange)
                }
            }
            .tint(TeloscopyTheme.primaryBlue)
            
            if notificationsEnabled {
                Toggle(isOn: $notifyAnalysisComplete) {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Analysis Complete")
                                .font(.subheadline)
                            Text("Notify when results are ready")
                                .font(.caption)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                        }
                    } icon: {
                        Image(systemName: "checkmark.circle")
                            .foregroundColor(TeloscopyTheme.successGreen)
                    }
                }
                .tint(TeloscopyTheme.primaryBlue)
                
                Toggle(isOn: $notifySyncErrors) {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Sync Errors")
                                .font(.subheadline)
                            Text("Notify on upload or sync failures")
                                .font(.caption)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                        }
                    } icon: {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundColor(TeloscopyTheme.errorRed)
                    }
                }
                .tint(TeloscopyTheme.primaryBlue)
            }
        } header: {
            Text("Notifications")
        }
    }
    
    // MARK: - Appearance Section
    
    private var appearanceSection: some View {
        Section {
            Picker(selection: $appearanceMode) {
                Text("System").tag("system")
                Text("Light").tag("light")
                Text("Dark").tag("dark")
            } label: {
                Label {
                    Text("Appearance")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "paintbrush.fill")
                        .foregroundColor(TeloscopyTheme.primaryBlue)
                }
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Label {
                    Text("Image Compression")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "photo")
                        .foregroundColor(TeloscopyTheme.accentTeal)
                }
                
                HStack {
                    Slider(value: $compressionQuality, in: 0.5...1.0, step: 0.05)
                        .tint(TeloscopyTheme.primaryBlue)
                    
                    Text("\(Int(compressionQuality * 100))%")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                        .frame(width: 40)
                }
            }
            
            Toggle(isOn: $keepLocalImages) {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Keep Local Images")
                            .font(.subheadline)
                        Text("Retain images after upload")
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                } icon: {
                    Image(systemName: "internaldrive")
                        .foregroundColor(TeloscopyTheme.warningOrange)
                }
            }
            .tint(TeloscopyTheme.primaryBlue)
        } header: {
            Text("Appearance & Storage")
        }
    }
    
    // MARK: - Data Management Section
    
    private var dataSection: some View {
        Section {
            // Cache size
            HStack {
                Label {
                    Text("Cache Size")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "internaldrive")
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
                
                Spacer()
                
                Text(syncManager.cacheSize)
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            Button(action: { showClearCacheConfirm = true }) {
                Label("Clear Cache", systemImage: "trash")
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.warningOrange)
            }
            
            if !syncManager.pendingUploads.isEmpty {
                Button(action: { syncManager.clearUploadQueue() }) {
                    Label("Clear Upload Queue", systemImage: "xmark.circle")
                        .font(.subheadline)
                        .foregroundColor(TeloscopyTheme.errorRed)
                }
            }
            
            Button(action: { showResetConfirm = true }) {
                Label("Reset All Settings", systemImage: "arrow.counterclockwise")
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.errorRed)
            }
        } header: {
            Text("Data Management")
        }
    }
    
    // MARK: - About Section
    
    private var aboutSection: some View {
        Section {
            HStack {
                Label {
                    Text("Version")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "info.circle")
                        .foregroundColor(TeloscopyTheme.primaryBlue)
                }
                Spacer()
                Text("1.0.0 (1)")
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            HStack {
                Label {
                    Text("Bundle ID")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "shippingbox")
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
                Spacer()
                Text("com.teloscopy.ios")
                    .font(.caption)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            HStack {
                Label {
                    Text("iOS Target")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "iphone")
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
                Spacer()
                Text("iOS 16.0+")
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            NavigationLink {
                licensesView
            } label: {
                Label {
                    Text("Open Source Licenses")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "doc.text")
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
            }
            
            NavigationLink {
                privacyPolicyView
            } label: {
                Label {
                    Text("Privacy Policy")
                        .font(.subheadline)
                } icon: {
                    Image(systemName: "hand.raised.fill")
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
            }
        } header: {
            Text("About")
        } footer: {
            VStack(spacing: 4) {
                Text("Teloscopy - Genomic Telomere Analysis")
                Text("Built with SwiftUI for iOS 16+")
            }
            .font(.caption2)
            .frame(maxWidth: .infinity)
            .padding(.top, 12)
        }
    }
    
    // MARK: - Server URL Editor Sheet
    
    private var serverURLEditor: some View {
        NavigationStack {
            VStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Server URL")
                        .font(.headline)
                    
                    Text("Enter the address of your Teloscopy analysis server.")
                        .font(.subheadline)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                    
                    TextField("https://your-server.example.com", text: $tempServerURL)
                        .textFieldStyle(.plain)
                        .padding(14)
                        .background(TeloscopyTheme.surfaceBackground)
                        .cornerRadius(TeloscopyTheme.smallCornerRadius)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    
                    Text("Default: \(APIConfiguration.defaultBaseURL)")
                        .font(.caption)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
                
                Button("Reset to Default") {
                    tempServerURL = APIConfiguration.defaultBaseURL
                }
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.primaryBlue)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Server URL")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        showServerURLEditor = false
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        serverURL = tempServerURL
                        showServerURLEditor = false
                        // Recheck connection with new URL
                        testConnection()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
    
    // MARK: - Licenses View
    
    private var licensesView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Open Source Licenses")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text("Teloscopy iOS uses the following open source technologies:")
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                
                VStack(alignment: .leading, spacing: 12) {
                    LicenseRow(name: "SwiftUI", license: "Apple Inc. - Proprietary")
                    LicenseRow(name: "Combine", license: "Apple Inc. - Proprietary")
                    LicenseRow(name: "Foundation", license: "Apple Inc. - Proprietary")
                }
                
                Text("This application is built entirely using Apple's first-party frameworks with no third-party dependencies.")
                    .font(.caption)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                    .padding(.top)
            }
            .padding()
        }
        .navigationTitle("Licenses")
        .navigationBarTitleDisplayMode(.inline)
    }
    
    private var privacyPolicyView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Privacy Policy")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Group {
                    Text("Data Collection")
                        .font(.headline)
                    Text("Teloscopy collects microscope images and analysis metadata that you explicitly provide. No data is collected automatically without your consent.")
                    
                    Text("Data Storage")
                        .font(.headline)
                    Text("Your data is stored locally on your device and optionally synchronized with your configured Teloscopy server. You maintain full control over your data.")
                    
                    Text("Camera & Photos")
                        .font(.headline)
                    Text("Camera and photo library access is used exclusively for capturing or selecting microscope images for telomere analysis. Images are not shared with third parties.")
                    
                    Text("Network Communication")
                        .font(.headline)
                    Text("The app communicates only with your configured Teloscopy server. No data is sent to any other service or analytics platform.")
                }
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
            }
            .padding()
        }
        .navigationTitle("Privacy Policy")
        .navigationBarTitleDisplayMode(.inline)
    }
    
    // MARK: - Actions
    
    private func testConnection() {
        isTestingConnection = true
        connectionTestResult = nil
        
        apiService.checkServerHealth()
            .receive(on: DispatchQueue.main)
            .sink { reachable in
                isTestingConnection = false
                connectionTestResult = reachable ? .success : .failure("Could not reach the server. Verify the URL and that the server is running.")
            }
            .store(in: &cancellables)
    }
    
    private func resetAllSettings() {
        serverURL = APIConfiguration.defaultBaseURL
        appearanceMode = "system"
        notificationsEnabled = true
        notifyAnalysisComplete = true
        notifySyncErrors = true
        autoSyncEnabled = true
        autoUploadOnWifi = true
        compressionQuality = 0.85
        keepLocalImages = true
    }
}

// MARK: - License Row

struct LicenseRow: View {
    let name: String
    let license: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(name)
                .font(.subheadline)
                .fontWeight(.semibold)
            Text(license)
                .font(.caption)
                .foregroundColor(TeloscopyTheme.textSecondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(TeloscopyTheme.surfaceBackground)
        .cornerRadius(TeloscopyTheme.smallCornerRadius)
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        SettingsView()
    }
    .environmentObject(APIService.shared)
    .environmentObject(SyncManager.shared)
}
