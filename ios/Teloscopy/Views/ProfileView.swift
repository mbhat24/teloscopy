// ProfileView.swift
// Teloscopy
//
// User profile, analysis history, and longitudinal telomere tracking.
//

import SwiftUI
import Combine

struct ProfileView: View {
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var syncManager: SyncManager
    
    @State private var showLoginSheet = false
    @State private var analysisHistory: [Analysis] = Analysis.sampleData
    @State private var longitudinalData: [LongitudinalDataPoint] = []
    @State private var selectedTimeRange: TimeRange = .sixMonths
    @State private var cancellables = Set<AnyCancellable>()
    
    enum TimeRange: String, CaseIterable {
        case threeMonths = "3M"
        case sixMonths = "6M"
        case oneYear = "1Y"
        case allTime = "All"
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                profileHeader
                
                if apiService.isAuthenticated {
                    analysisStatsSection
                    longitudinalTrackingSection
                    analysisHistorySection
                } else {
                    signInPrompt
                }
            }
            .padding()
        }
        .background(TeloscopyTheme.surfaceBackground.ignoresSafeArea())
        .navigationTitle("Profile")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                if apiService.isAuthenticated {
                    Menu {
                        Button(action: { }) {
                            Label("Edit Profile", systemImage: "pencil")
                        }
                        Button(action: { }) {
                            Label("Export Data", systemImage: "square.and.arrow.up")
                        }
                        Divider()
                        Button(role: .destructive, action: {
                            apiService.logout()
                        }) {
                            Label("Sign Out", systemImage: "rectangle.portrait.and.arrow.right")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .sheet(isPresented: $showLoginSheet) {
            LoginView()
                .environmentObject(apiService)
        }
        .onAppear(perform: loadProfileData)
    }
    
    // MARK: - Profile Header
    
    private var profileHeader: some View {
        VStack(spacing: 16) {
            // Avatar
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [TeloscopyTheme.primaryBlue, TeloscopyTheme.accentTeal],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 80, height: 80)
                
                if let user = apiService.currentUser {
                    Text(user.fullName.prefix(2).uppercased())
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                } else {
                    Image(systemName: "person.fill")
                        .font(.system(size: 32))
                        .foregroundColor(.white)
                }
            }
            
            if let user = apiService.currentUser {
                VStack(spacing: 4) {
                    Text(user.fullName)
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(TeloscopyTheme.textPrimary)
                    
                    Text(user.email)
                        .font(.subheadline)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                    
                    if let institution = user.institution {
                        HStack(spacing: 4) {
                            Image(systemName: "building.2")
                                .font(.caption2)
                            Text(institution)
                                .font(.caption)
                        }
                        .foregroundColor(TeloscopyTheme.textSecondary)
                        .padding(.top, 2)
                    }
                    
                    if let role = user.role {
                        Text(role)
                            .font(.caption)
                            .fontWeight(.medium)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 4)
                            .background(TeloscopyTheme.primaryBlue.opacity(0.1))
                            .foregroundColor(TeloscopyTheme.primaryBlue)
                            .cornerRadius(12)
                            .padding(.top, 4)
                    }
                }
            } else {
                VStack(spacing: 4) {
                    Text("Guest User")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(TeloscopyTheme.textPrimary)
                    
                    Text("Sign in to sync your data")
                        .font(.subheadline)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
            }
            
            // Sync status
            if let lastSync = syncManager.lastSyncDate {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.caption2)
                    Text("Last synced: \(lastSync, style: .relative) ago")
                        .font(.caption2)
                }
                .foregroundColor(TeloscopyTheme.textSecondary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .cardStyle()
    }
    
    // MARK: - Sign In Prompt
    
    private var signInPrompt: some View {
        VStack(spacing: 20) {
            Image(systemName: "person.crop.circle.badge.plus")
                .font(.system(size: 48))
                .foregroundColor(TeloscopyTheme.primaryBlue.opacity(0.5))
            
            Text("Sign In to Access Full Features")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            Text("Create an account or sign in to sync your analyses across devices, track longitudinal data, and collaborate with your team.")
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            Button(action: { showLoginSheet = true }) {
                Text("Sign In")
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(TeloscopyTheme.primaryBlue)
                    .foregroundColor(.white)
                    .cornerRadius(TeloscopyTheme.cornerRadius)
            }
            .padding(.horizontal, 32)
        }
        .padding(.vertical, 40)
        .cardStyle()
    }
    
    // MARK: - Analysis Statistics
    
    private var analysisStatsSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Your Statistics", systemImage: "chart.pie")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                ProfileStatCard(
                    value: "\(analysisHistory.count)",
                    label: "Total",
                    icon: "doc.text",
                    color: TeloscopyTheme.primaryBlue
                )
                
                ProfileStatCard(
                    value: "\(analysisHistory.filter { $0.status == .completed }.count)",
                    label: "Completed",
                    icon: "checkmark.circle",
                    color: TeloscopyTheme.successGreen
                )
                
                let thisMonth = analysisHistory.filter {
                    Calendar.current.isDate($0.createdAt, equalTo: Date(), toGranularity: .month)
                }.count
                ProfileStatCard(
                    value: "\(thisMonth)",
                    label: "This Month",
                    icon: "calendar",
                    color: TeloscopyTheme.accentTeal
                )
            }
            
            // Analysis type breakdown
            VStack(spacing: 10) {
                ForEach(AnalysisType.allCases) { type in
                    let count = analysisHistory.filter { $0.analysisType == type }.count
                    if count > 0 {
                        HStack {
                            Image(systemName: type.iconName)
                                .foregroundColor(TeloscopyTheme.primaryBlue)
                                .frame(width: 24)
                            
                            Text(type.displayName)
                                .font(.subheadline)
                                .foregroundColor(TeloscopyTheme.textPrimary)
                            
                            Spacer()
                            
                            Text("\(count)")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(TeloscopyTheme.textPrimary)
                            
                            // Mini progress bar
                            let ratio = Double(count) / Double(max(analysisHistory.count, 1))
                            RoundedRectangle(cornerRadius: 3)
                                .fill(TeloscopyTheme.primaryBlue.opacity(0.2))
                                .frame(width: 60, height: 6)
                                .overlay(alignment: .leading) {
                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(TeloscopyTheme.primaryBlue)
                                        .frame(width: 60 * CGFloat(ratio), height: 6)
                                }
                        }
                    }
                }
            }
            .padding()
            .background(TeloscopyTheme.surfaceBackground)
            .cornerRadius(TeloscopyTheme.smallCornerRadius)
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Longitudinal Tracking
    
    private var longitudinalTrackingSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Telomere Length Trend", systemImage: "chart.xyaxis.line")
                    .font(.headline)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                
                Spacer()
            }
            
            // Time range selector
            HStack(spacing: 0) {
                ForEach(TimeRange.allCases, id: \.rawValue) { range in
                    Button(action: { selectedTimeRange = range }) {
                        Text(range.rawValue)
                            .font(.caption)
                            .fontWeight(.medium)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(selectedTimeRange == range ? TeloscopyTheme.primaryBlue : Color.clear)
                            .foregroundColor(selectedTimeRange == range ? .white : TeloscopyTheme.textSecondary)
                    }
                }
            }
            .background(TeloscopyTheme.surfaceBackground)
            .cornerRadius(8)
            
            // Trend chart placeholder
            if longitudinalData.isEmpty {
                // Show sample trend line
                VStack(spacing: 8) {
                    sampleTrendChart
                    
                    Text("Sample trend data shown. Complete more analyses for real tracking.")
                        .font(.caption2)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                        .multilineTextAlignment(.center)
                }
            } else {
                trendChart
            }
            
            // Summary
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Trend")
                        .font(.caption2)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                    
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.down.right")
                            .font(.caption)
                        Text("-0.3 kb/year")
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(TeloscopyTheme.warningOrange)
                }
                
                Divider()
                    .frame(height: 30)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Rate")
                        .font(.caption2)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                    Text("Normal range")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(TeloscopyTheme.successGreen)
                }
                
                Divider()
                    .frame(height: 30)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Data Points")
                        .font(.caption2)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                    Text("\(max(longitudinalData.count, 6))")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(TeloscopyTheme.textPrimary)
                }
            }
        }
        .padding()
        .cardStyle()
    }
    
    private var sampleTrendChart: some View {
        let points: [(CGFloat, CGFloat)] = [
            (0.0, 0.8), (0.15, 0.75), (0.3, 0.72), (0.45, 0.68),
            (0.6, 0.65), (0.75, 0.62), (0.9, 0.58), (1.0, 0.55)
        ]
        
        return GeometryReader { geometry in
            let w = geometry.size.width
            let h = geometry.size.height
            
            ZStack {
                // Grid lines
                ForEach(0..<5) { i in
                    let y = h * CGFloat(i) / 4
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: w, y: y))
                    }
                    .stroke(Color.gray.opacity(0.1), lineWidth: 0.5)
                }
                
                // Area fill
                Path { path in
                    path.move(to: CGPoint(x: 0, y: h))
                    for point in points {
                        path.addLine(to: CGPoint(x: w * point.0, y: h * (1 - point.1)))
                    }
                    path.addLine(to: CGPoint(x: w, y: h))
                    path.closeSubpath()
                }
                .fill(
                    LinearGradient(
                        colors: [TeloscopyTheme.primaryBlue.opacity(0.2), TeloscopyTheme.primaryBlue.opacity(0.02)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                
                // Line
                Path { path in
                    for (index, point) in points.enumerated() {
                        let x = w * point.0
                        let y = h * (1 - point.1)
                        if index == 0 {
                            path.move(to: CGPoint(x: x, y: y))
                        } else {
                            path.addLine(to: CGPoint(x: x, y: y))
                        }
                    }
                }
                .stroke(TeloscopyTheme.primaryBlue, style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                
                // Data points
                ForEach(0..<points.count, id: \.self) { index in
                    let point = points[index]
                    Circle()
                        .fill(TeloscopyTheme.primaryBlue)
                        .frame(width: 6, height: 6)
                        .position(x: w * point.0, y: h * (1 - point.1))
                }
            }
        }
        .frame(height: 140)
    }
    
    private var trendChart: some View {
        Text("Longitudinal trend chart")
            .frame(height: 140)
            .frame(maxWidth: .infinity)
    }
    
    // MARK: - Analysis History
    
    private var analysisHistorySection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Analysis History", systemImage: "clock.arrow.circlepath")
                    .font(.headline)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                
                Spacer()
                
                NavigationLink("View All") {
                    AllAnalysesView(analyses: analysisHistory)
                }
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.primaryBlue)
            }
            
            ForEach(analysisHistory.prefix(5)) { analysis in
                HistoryRow(analysis: analysis)
            }
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Data Loading
    
    private func loadProfileData() {
        analysisHistory = syncManager.cachedAnalyses.isEmpty ? Analysis.sampleData : syncManager.cachedAnalyses
        
        // Generate sample longitudinal data
        if longitudinalData.isEmpty {
            longitudinalData = (0..<8).map { index in
                LongitudinalDataPoint(
                    date: Calendar.current.date(byAdding: .month, value: -index * 2, to: Date()) ?? Date(),
                    meanTelomereLength: 7.5 - Double(index) * 0.15 + Double.random(in: -0.2...0.2),
                    analysisId: UUID()
                )
            }.reversed()
        }
        
        guard apiService.isAuthenticated else { return }
        
        apiService.fetchProfile()
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { _ in }, receiveValue: { _ in })
            .store(in: &cancellables)
        
        apiService.fetchLongitudinalData()
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { _ in },
                receiveValue: { data in
                    if !data.isEmpty {
                        longitudinalData = data
                    }
                }
            )
            .store(in: &cancellables)
    }
}

// MARK: - Supporting Views

struct ProfileStatCard: View {
    let value: String
    let label: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
            
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            Text(label)
                .font(.caption2)
                .foregroundColor(TeloscopyTheme.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(color.opacity(0.05))
        .cornerRadius(TeloscopyTheme.smallCornerRadius)
    }
}

struct HistoryRow: View {
    let analysis: Analysis
    
    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(analysis.status.color.opacity(0.15))
                .frame(width: 36, height: 36)
                .overlay(
                    Image(systemName: analysis.status.iconName)
                        .font(.system(size: 14))
                        .foregroundColor(analysis.status.color)
                )
            
            VStack(alignment: .leading, spacing: 2) {
                Text(analysis.name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                    .lineLimit(1)
                
                Text(analysis.analysisType.displayName)
                    .font(.caption)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 2) {
                Text(analysis.formattedDate)
                    .font(.caption2)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                
                Text(analysis.status.displayName)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundColor(analysis.status.color)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        ProfileView()
    }
    .environmentObject(APIService.shared)
    .environmentObject(SyncManager.shared)
}
