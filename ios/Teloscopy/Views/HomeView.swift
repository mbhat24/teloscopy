// HomeView.swift
// Teloscopy
//
// Dashboard view showing recent analyses, quick actions, and system status.
//

import SwiftUI
import Combine

struct HomeView: View {
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var syncManager: SyncManager
    
    @State private var recentAnalyses: [Analysis] = Analysis.sampleData
    @State private var isRefreshing = false
    @State private var showNewAnalysis = false
    @State private var cancellables = Set<AnyCancellable>()
    
    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12)
    ]
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                connectionStatusBanner
                quickActionsGrid
                statisticsCards
                recentAnalysesSection
            }
            .padding()
        }
        .background(TeloscopyTheme.surfaceBackground.ignoresSafeArea())
        .navigationTitle("Teloscopy")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: refreshData) {
                    if isRefreshing {
                        ProgressView()
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
        .onAppear(perform: loadData)
        .refreshable { refreshData() }
        .sheet(isPresented: $showNewAnalysis) {
            NavigationStack {
                AnalysisView()
            }
        }
    }
    
    // MARK: - Header
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(greeting)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(TeloscopyTheme.textPrimary)
                    
                    Text("Telomere Analysis Dashboard")
                        .font(.subheadline)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
                
                Spacer()
                
                Image(systemName: "dna")
                    .font(.system(size: 36))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [TeloscopyTheme.primaryBlue, TeloscopyTheme.accentTeal],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }
        }
        .padding()
        .cardStyle()
    }
    
    private var greeting: String {
        let hour = Calendar.current.component(.hour, from: Date())
        let name = apiService.currentUser?.fullName.components(separatedBy: " ").first ?? ""
        let prefix = hour < 12 ? "Good Morning" : (hour < 17 ? "Good Afternoon" : "Good Evening")
        return name.isEmpty ? prefix : "\(prefix), \(name)"
    }
    
    // MARK: - Connection Status
    
    private var connectionStatusBanner: some View {
        Group {
            if !syncManager.isNetworkAvailable {
                HStack(spacing: 10) {
                    Image(systemName: "wifi.slash")
                        .foregroundColor(.white)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Offline Mode")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                        
                        Text("Changes will sync when connection is restored")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.85))
                    }
                    
                    Spacer()
                    
                    if !syncManager.pendingUploads.isEmpty {
                        Text("\(syncManager.pendingUploads.count) pending")
                            .font(.caption)
                            .fontWeight(.medium)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.white.opacity(0.2))
                            .cornerRadius(6)
                            .foregroundColor(.white)
                    }
                }
                .padding()
                .background(
                    LinearGradient(
                        colors: [TeloscopyTheme.warningOrange, .orange],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(TeloscopyTheme.cornerRadius)
            } else if !apiService.isServerReachable {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.icloud")
                        .foregroundColor(.white)
                    
                    Text("Server unreachable - Check Settings")
                        .font(.subheadline)
                        .foregroundColor(.white)
                    
                    Spacer()
                }
                .padding()
                .background(TeloscopyTheme.errorRed)
                .cornerRadius(TeloscopyTheme.cornerRadius)
            }
        }
    }
    
    // MARK: - Quick Actions
    
    private var quickActionsGrid: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Actions")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            LazyVGrid(columns: columns, spacing: 12) {
                QuickActionCard(
                    title: "New Analysis",
                    subtitle: "Start scanning",
                    icon: "plus.circle.fill",
                    color: TeloscopyTheme.primaryBlue
                ) {
                    showNewAnalysis = true
                }
                
                QuickActionCard(
                    title: "Camera Capture",
                    subtitle: "Take photo",
                    icon: "camera.fill",
                    color: TeloscopyTheme.accentTeal
                ) {
                    showNewAnalysis = true
                }
                
                QuickActionCard(
                    title: "Import Data",
                    subtitle: "From files",
                    icon: "square.and.arrow.down.fill",
                    color: TeloscopyTheme.successGreen
                ) {
                    // Import action
                }
                
                QuickActionCard(
                    title: "Sync Data",
                    subtitle: syncManager.syncState.displayName,
                    icon: "arrow.triangle.2.circlepath",
                    color: TeloscopyTheme.warningOrange
                ) {
                    syncManager.performSync()
                }
            }
        }
    }
    
    // MARK: - Statistics
    
    private var statisticsCards: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Overview")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            HStack(spacing: 12) {
                HomeStatCard(
                    value: "\(recentAnalyses.count)",
                    label: "Total Analyses",
                    icon: "doc.text.magnifyingglass",
                    color: TeloscopyTheme.primaryBlue
                )
                
                HomeStatCard(
                    value: "\(recentAnalyses.filter { $0.status == .completed }.count)",
                    label: "Completed",
                    icon: "checkmark.circle",
                    color: TeloscopyTheme.successGreen
                )
                
                HomeStatCard(
                    value: "\(recentAnalyses.filter { $0.status == .processing }.count)",
                    label: "In Progress",
                    icon: "hourglass",
                    color: TeloscopyTheme.warningOrange
                )
            }
        }
    }
    
    // MARK: - Recent Analyses
    
    private var recentAnalysesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Analyses")
                    .font(.headline)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                
                Spacer()
                
                NavigationLink("See All") {
                    AllAnalysesView(analyses: recentAnalyses)
                }
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.primaryBlue)
            }
            
            if recentAnalyses.isEmpty {
                emptyStateView
            } else {
                ForEach(recentAnalyses.prefix(5)) { analysis in
                    NavigationLink(destination: AnalysisDetailView(analysis: analysis)) {
                        AnalysisRowCard(analysis: analysis)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "microscope")
                .font(.system(size: 48))
                .foregroundColor(TeloscopyTheme.textSecondary.opacity(0.5))
            
            Text("No analyses yet")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textSecondary)
            
            Text("Start your first telomere analysis to see results here.")
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
                .multilineTextAlignment(.center)
            
            Button(action: { showNewAnalysis = true }) {
                Label("Start Analysis", systemImage: "plus")
                    .fontWeight(.semibold)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                    .background(TeloscopyTheme.primaryBlue)
                    .foregroundColor(.white)
                    .cornerRadius(TeloscopyTheme.cornerRadius)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
        .cardStyle()
    }
    
    // MARK: - Data Loading
    
    private func loadData() {
        recentAnalyses = syncManager.cachedAnalyses.isEmpty ? Analysis.sampleData : syncManager.cachedAnalyses
    }
    
    private func refreshData() {
        isRefreshing = true
        syncManager.performSync()
        
        apiService.fetchAnalyses()
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { _ in
                    isRefreshing = false
                },
                receiveValue: { analyses in
                    recentAnalyses = analyses
                    syncManager.saveAnalysesLocally(analyses)
                }
            )
            .store(in: &cancellables)
    }
}

// MARK: - Quick Action Card

struct QuickActionCard: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 10) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(TeloscopyTheme.textPrimary)
                    
                    Text(subtitle)
                        .font(.caption)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .cardStyle()
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Stat Card (Home-specific)

struct HomeStatCard: View {
    let value: String
    let label: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
            
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            Text(label)
                .font(.caption2)
                .foregroundColor(TeloscopyTheme.textSecondary)
                .multilineTextAlignment(.center)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .padding(.horizontal, 8)
        .cardStyle()
    }
}

// MARK: - Analysis Row Card

struct AnalysisRowCard: View {
    let analysis: Analysis
    
    var body: some View {
        HStack(spacing: 14) {
            // Status icon
            ZStack {
                Circle()
                    .fill(analysis.status.color.opacity(0.15))
                    .frame(width: 44, height: 44)
                
                Image(systemName: analysis.analysisType.iconName)
                    .font(.system(size: 18))
                    .foregroundColor(analysis.status.color)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(analysis.name)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                    .lineLimit(1)
                
                HStack(spacing: 8) {
                    Label(analysis.analysisType.displayName, systemImage: analysis.status.iconName)
                        .font(.caption)
                        .foregroundColor(analysis.status.color)
                    
                    if let sampleId = analysis.sampleId {
                        Text(sampleId)
                            .font(.caption2)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(TeloscopyTheme.surfaceBackground)
                            .cornerRadius(4)
                    }
                }
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(analysis.formattedDate)
                    .font(.caption2)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                
                if !analysis.isSynced {
                    Image(systemName: "icloud.slash")
                        .font(.caption2)
                        .foregroundColor(TeloscopyTheme.warningOrange)
                }
            }
            
            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(TeloscopyTheme.textSecondary)
        }
        .padding()
        .cardStyle()
    }
}

// MARK: - Analysis Detail View (Placeholder)

struct AnalysisDetailView: View {
    let analysis: Analysis
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Status header
                VStack(spacing: 12) {
                    Image(systemName: analysis.status.iconName)
                        .font(.system(size: 40))
                        .foregroundColor(analysis.status.color)
                    
                    Text(analysis.status.displayName)
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(analysis.status.color)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
                .cardStyle()
                
                // Details
                VStack(alignment: .leading, spacing: 16) {
                    DetailRow(label: "Analysis Type", value: analysis.analysisType.displayName)
                    DetailRow(label: "Sample ID", value: analysis.sampleId ?? "N/A")
                    DetailRow(label: "Patient ID", value: analysis.patientId ?? "N/A")
                    DetailRow(label: "Images", value: "\(analysis.imageCount)")
                    DetailRow(label: "Created", value: analysis.formattedDate)
                    
                    if let notes = analysis.notes {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Notes")
                                .font(.caption)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                            Text(notes)
                                .font(.subheadline)
                                .foregroundColor(TeloscopyTheme.textPrimary)
                        }
                    }
                }
                .padding()
                .cardStyle()
            }
            .padding()
        }
        .background(TeloscopyTheme.surfaceBackground.ignoresSafeArea())
        .navigationTitle(analysis.name)
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct DetailRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
            Spacer()
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(TeloscopyTheme.textPrimary)
        }
    }
}

// MARK: - All Analyses View

struct AllAnalysesView: View {
    let analyses: [Analysis]
    @State private var searchText = ""
    @State private var selectedFilter: AnalysisStatus? = nil
    
    var filteredAnalyses: [Analysis] {
        var result = analyses
        if let filter = selectedFilter {
            result = result.filter { $0.status == filter }
        }
        if !searchText.isEmpty {
            result = result.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                ($0.sampleId?.localizedCaseInsensitiveContains(searchText) ?? false)
            }
        }
        return result
    }
    
    var body: some View {
        List {
            // Filter chips
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    FilterChip(title: "All", isSelected: selectedFilter == nil) {
                        selectedFilter = nil
                    }
                    ForEach(AnalysisStatus.allCases, id: \.rawValue) { status in
                        FilterChip(title: status.displayName, isSelected: selectedFilter == status) {
                            selectedFilter = status
                        }
                    }
                }
            }
            .listRowInsets(EdgeInsets())
            .listRowBackground(Color.clear)
            
            ForEach(filteredAnalyses) { analysis in
                NavigationLink(destination: AnalysisDetailView(analysis: analysis)) {
                    AnalysisListRow(analysis: analysis)
                }
            }
        }
        .searchable(text: $searchText, prompt: "Search analyses...")
        .navigationTitle("All Analyses")
    }
}

struct FilterChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.caption)
                .fontWeight(.medium)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(isSelected ? TeloscopyTheme.primaryBlue : TeloscopyTheme.surfaceBackground)
                .foregroundColor(isSelected ? .white : TeloscopyTheme.textPrimary)
                .cornerRadius(20)
        }
    }
}

struct AnalysisListRow: View {
    let analysis: Analysis
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: analysis.analysisType.iconName)
                .foregroundColor(analysis.status.color)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(analysis.name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(analysis.formattedDate)
                    .font(.caption)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            Spacer()
            
            Text(analysis.status.displayName)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(analysis.status.color)
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        HomeView()
    }
    .environmentObject(APIService.shared)
    .environmentObject(SyncManager.shared)
}
