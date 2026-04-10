// ResultsView.swift
// Teloscopy
//
// Displays analysis results including telomere length charts,
// chromosome-level data, distribution histograms, and health indicators.
//

import SwiftUI
import Combine

struct ResultsView: View {
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var syncManager: SyncManager
    
    @State private var completedAnalyses: [Analysis] = Analysis.sampleData.filter { $0.status == .completed }
    @State private var selectedAnalysis: Analysis?
    @State private var result: TelomereResult? = TelomereResult.sampleResult
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var cancellables = Set<AnyCancellable>()
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                if completedAnalyses.isEmpty {
                    noResultsView
                } else {
                    analysisPicker
                    
                    if let result = result {
                        summaryCard(result: result)
                        healthIndicatorCard(result: result)
                        telomereLengthChart(result: result)
                        chromosomeDataSection(result: result)
                        distributionChart(result: result)
                        detailedMetrics(result: result)
                    } else if isLoading {
                        loadingView
                    } else if let error = errorMessage {
                        errorView(message: error)
                    }
                }
            }
            .padding()
        }
        .background(TeloscopyTheme.surfaceBackground.ignoresSafeArea())
        .navigationTitle("Results")
        .onAppear {
            if selectedAnalysis == nil, let first = completedAnalyses.first {
                selectedAnalysis = first
            }
        }
    }
    
    // MARK: - Analysis Picker
    
    private var analysisPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Select Analysis")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(TeloscopyTheme.textSecondary)
            
            Menu {
                ForEach(completedAnalyses) { analysis in
                    Button(action: {
                        selectedAnalysis = analysis
                        loadResult(for: analysis)
                    }) {
                        Label(analysis.name, systemImage: analysis.analysisType.iconName)
                    }
                }
            } label: {
                HStack {
                    Image(systemName: selectedAnalysis?.analysisType.iconName ?? "doc")
                        .foregroundColor(TeloscopyTheme.primaryBlue)
                    
                    Text(selectedAnalysis?.name ?? "Choose analysis")
                        .foregroundColor(TeloscopyTheme.textPrimary)
                    
                    Spacer()
                    
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                }
                .padding()
                .background(TeloscopyTheme.cardBackground)
                .cornerRadius(TeloscopyTheme.smallCornerRadius)
                .overlay(
                    RoundedRectangle(cornerRadius: TeloscopyTheme.smallCornerRadius)
                        .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                )
            }
        }
    }
    
    // MARK: - Summary Card
    
    private func summaryCard(result: TelomereResult) -> some View {
        VStack(spacing: 16) {
            HStack {
                Label("Analysis Summary", systemImage: "chart.bar.doc.horizontal")
                    .font(.headline)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                Spacer()
                
                Text("Quality: \(String(format: "%.0f%%", result.qualityScore * 100))")
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(qualityColor(result.qualityScore).opacity(0.15))
                    .foregroundColor(qualityColor(result.qualityScore))
                    .cornerRadius(12)
            }
            
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 14) {
                MetricCard(
                    title: "Mean Length",
                    value: String(format: "%.1f kb", result.meanTelomereLength),
                    subtitle: "Kilobases",
                    color: TeloscopyTheme.primaryBlue
                )
                
                MetricCard(
                    title: "Median Length",
                    value: String(format: "%.1f kb", result.medianTelomereLength),
                    subtitle: "Kilobases",
                    color: TeloscopyTheme.accentTeal
                )
                
                MetricCard(
                    title: "Std Deviation",
                    value: String(format: "%.2f", result.standardDeviation),
                    subtitle: "Variability",
                    color: TeloscopyTheme.warningOrange
                )
                
                MetricCard(
                    title: "Cells Analyzed",
                    value: "\(result.totalCellsAnalyzed)",
                    subtitle: "Total count",
                    color: TeloscopyTheme.successGreen
                )
            }
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Health Indicator
    
    private func healthIndicatorCard(result: TelomereResult) -> some View {
        VStack(spacing: 16) {
            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .fill(result.healthIndicator.color.opacity(0.15))
                        .frame(width: 56, height: 56)
                    
                    Image(systemName: result.healthIndicator.iconName)
                        .font(.system(size: 24))
                        .foregroundColor(result.healthIndicator.color)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Telomere Health Status")
                        .font(.caption)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                    
                    Text(result.healthIndicator.rawValue)
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(result.healthIndicator.color)
                }
                
                Spacer()
            }
            
            // Range visualization
            GeometryReader { geometry in
                let width = geometry.size.width
                let position = min(max(CGFloat((result.meanTelomereLength - 2) / 12), 0), 1)
                
                ZStack(alignment: .leading) {
                    // Background gradient bar
                    LinearGradient(
                        colors: [.red, .orange, .blue, .green],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                    .frame(height: 8)
                    .cornerRadius(4)
                    
                    // Marker
                    Circle()
                        .fill(.white)
                        .frame(width: 16, height: 16)
                        .shadow(color: .black.opacity(0.2), radius: 2)
                        .offset(x: width * position - 8)
                }
            }
            .frame(height: 16)
            
            HStack {
                Text("Critical")
                    .font(.caption2)
                    .foregroundColor(.red)
                Spacer()
                Text("Below Avg")
                    .font(.caption2)
                    .foregroundColor(.orange)
                Spacer()
                Text("Normal")
                    .font(.caption2)
                    .foregroundColor(.blue)
                Spacer()
                Text("Excellent")
                    .font(.caption2)
                    .foregroundColor(.green)
            }
            
            // Age estimate
            if let ageEstimate = result.ageEstimate, let offset = result.biologicalAgeOffset {
                Divider()
                
                HStack(spacing: 20) {
                    VStack(spacing: 2) {
                        Text("Estimated Bio Age")
                            .font(.caption2)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                        Text(String(format: "%.0f years", ageEstimate))
                            .font(.headline)
                            .foregroundColor(TeloscopyTheme.textPrimary)
                    }
                    
                    Divider()
                        .frame(height: 36)
                    
                    VStack(spacing: 2) {
                        Text("Offset")
                            .font(.caption2)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                        Text(String(format: "%+.1f years", offset))
                            .font(.headline)
                            .foregroundColor(offset < 0 ? TeloscopyTheme.successGreen : TeloscopyTheme.warningOrange)
                    }
                }
            }
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Telomere Length Bar Chart
    
    private func telomereLengthChart(result: TelomereResult) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Chromosome Telomere Lengths", systemImage: "chart.bar.fill")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .bottom, spacing: 6) {
                    ForEach(result.chromosomeData) { chromosome in
                        VStack(spacing: 4) {
                            // P-arm and Q-arm stacked bar
                            VStack(spacing: 1) {
                                Rectangle()
                                    .fill(TeloscopyTheme.accentTeal.opacity(0.7))
                                    .frame(
                                        width: 22,
                                        height: max(4, CGFloat(chromosome.pArmLength) * 10)
                                    )
                                
                                Rectangle()
                                    .fill(TeloscopyTheme.primaryBlue)
                                    .frame(
                                        width: 22,
                                        height: max(4, CGFloat(chromosome.qArmLength) * 10)
                                    )
                            }
                            .cornerRadius(3)
                            
                            // Chromosome label
                            Text(chromosome.displayName)
                                .font(.system(size: 9, weight: .medium))
                                .foregroundColor(chromosome.aberrationDetected
                                    ? TeloscopyTheme.errorRed
                                    : TeloscopyTheme.textSecondary
                                )
                        }
                    }
                }
                .frame(height: 160)
                .padding(.bottom, 4)
            }
            
            // Legend
            HStack(spacing: 16) {
                LegendItem(color: TeloscopyTheme.accentTeal.opacity(0.7), label: "p-arm")
                LegendItem(color: TeloscopyTheme.primaryBlue, label: "q-arm")
                LegendItem(color: TeloscopyTheme.errorRed, label: "Aberration")
            }
            .font(.caption2)
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Chromosome Detail Section
    
    private func chromosomeDataSection(result: TelomereResult) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Chromosome Details", systemImage: "list.bullet.rectangle")
                    .font(.headline)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                
                Spacer()
                
                let aberrationCount = result.chromosomeData.filter { $0.aberrationDetected }.count
                if aberrationCount > 0 {
                    Text("\(aberrationCount) aberration\(aberrationCount > 1 ? "s" : "")")
                        .font(.caption)
                        .fontWeight(.medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(TeloscopyTheme.errorRed.opacity(0.1))
                        .foregroundColor(TeloscopyTheme.errorRed)
                        .cornerRadius(8)
                }
            }
            
            ForEach(result.chromosomeData.filter { $0.aberrationDetected }) { chromosome in
                HStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(TeloscopyTheme.errorRed)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Chromosome \(chromosome.displayName)")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        
                        if let type = chromosome.aberrationType {
                            Text(type)
                                .font(.caption)
                                .foregroundColor(TeloscopyTheme.textSecondary)
                        }
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(String(format: "%.1f kb", chromosome.averageLength))
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        Text(String(format: "%.0f%% conf.", chromosome.confidenceScore * 100))
                            .font(.caption2)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                }
                .padding()
                .background(TeloscopyTheme.errorRed.opacity(0.05))
                .cornerRadius(TeloscopyTheme.smallCornerRadius)
            }
            
            // Top 5 shortest telomeres
            VStack(alignment: .leading, spacing: 8) {
                Text("Shortest Telomeres")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                
                ForEach(result.chromosomeData.sorted { $0.averageLength < $1.averageLength }.prefix(5)) { chromosome in
                    HStack {
                        Text("Chr \(chromosome.displayName)")
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.textPrimary)
                            .frame(width: 50, alignment: .leading)
                        
                        // Progress bar
                        GeometryReader { geometry in
                            let maxLen = result.chromosomeData.map(\.averageLength).max() ?? 1
                            let ratio = chromosome.averageLength / maxLen
                            
                            ZStack(alignment: .leading) {
                                Rectangle()
                                    .fill(Color.gray.opacity(0.1))
                                    .frame(height: 12)
                                    .cornerRadius(6)
                                
                                Rectangle()
                                    .fill(chromosome.lengthCategory.color)
                                    .frame(width: geometry.size.width * CGFloat(ratio), height: 12)
                                    .cornerRadius(6)
                            }
                        }
                        .frame(height: 12)
                        
                        Text(String(format: "%.1f kb", chromosome.averageLength))
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(chromosome.lengthCategory.color)
                            .frame(width: 55, alignment: .trailing)
                    }
                }
            }
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Distribution Chart
    
    private func distributionChart(result: TelomereResult) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Length Distribution", systemImage: "chart.bar.xaxis")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            if result.telomereDistribution.isEmpty {
                Text("No distribution data available")
                    .font(.subheadline)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 20)
            } else {
                // Histogram
                HStack(alignment: .bottom, spacing: 4) {
                    ForEach(result.telomereDistribution) { bin in
                        VStack(spacing: 4) {
                            Text("\(bin.count)")
                                .font(.system(size: 8))
                                .foregroundColor(TeloscopyTheme.textSecondary)
                            
                            Rectangle()
                                .fill(
                                    LinearGradient(
                                        colors: [TeloscopyTheme.primaryBlue.opacity(0.6), TeloscopyTheme.primaryBlue],
                                        startPoint: .top,
                                        endPoint: .bottom
                                    )
                                )
                                .frame(height: max(4, CGFloat(bin.percentage) * 6))
                                .cornerRadius(3)
                            
                            Text(bin.label)
                                .font(.system(size: 7))
                                .foregroundColor(TeloscopyTheme.textSecondary)
                                .rotationEffect(.degrees(-45))
                                .frame(width: 30, height: 20)
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                .frame(height: 140)
                .padding(.top, 8)
                
                // Percentile markers
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("10th percentile")
                            .font(.caption2)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                        Text(String(format: "%.1f kb", result.percentileTenth))
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                    
                    Spacer()
                    
                    VStack(spacing: 2) {
                        Text("Median")
                            .font(.caption2)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                        Text(String(format: "%.1f kb", result.medianTelomereLength))
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("90th percentile")
                            .font(.caption2)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                        Text(String(format: "%.1f kb", result.percentileNinetieth))
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                }
            }
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Detailed Metrics
    
    private func detailedMetrics(result: TelomereResult) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Detailed Metrics", systemImage: "tablecells")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            VStack(spacing: 12) {
                MetricRow(label: "Shortest Telomere", value: String(format: "%.2f kb", result.shortestTelomere))
                Divider()
                MetricRow(label: "Longest Telomere", value: String(format: "%.2f kb", result.longestTelomere))
                Divider()
                MetricRow(label: "Range", value: String(format: "%.2f kb", result.longestTelomere - result.shortestTelomere))
                Divider()
                MetricRow(label: "10th Percentile", value: String(format: "%.2f kb", result.percentileTenth))
                Divider()
                MetricRow(label: "90th Percentile", value: String(format: "%.2f kb", result.percentileNinetieth))
                Divider()
                MetricRow(label: "Interquartile Range", value: String(format: "%.2f kb", result.percentileNinetieth - result.percentileTenth))
                Divider()
                MetricRow(label: "Coefficient of Variation", value: String(format: "%.1f%%", (result.standardDeviation / result.meanTelomereLength) * 100))
            }
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Empty / Loading / Error States
    
    private var noResultsView: some View {
        VStack(spacing: 20) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 56))
                .foregroundColor(TeloscopyTheme.textSecondary.opacity(0.4))
            
            Text("No Results Yet")
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            Text("Complete an analysis to see telomere length results, chromosome data, and health indicators here.")
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 80)
    }
    
    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)
            Text("Loading results...")
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }
    
    private func errorView(message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundColor(TeloscopyTheme.warningOrange)
            
            Text(message)
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
                .multilineTextAlignment(.center)
            
            Button("Retry") {
                if let analysis = selectedAnalysis {
                    loadResult(for: analysis)
                }
            }
            .fontWeight(.medium)
            .foregroundColor(TeloscopyTheme.primaryBlue)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
        .cardStyle()
    }
    
    // MARK: - Helpers
    
    private func loadResult(for analysis: Analysis) {
        // Try cached result first
        if let cached = syncManager.loadCachedResult(for: analysis.id) {
            result = cached
            return
        }
        
        guard let serverId = analysis.serverAnalysisId else {
            result = TelomereResult.sampleResult
            return
        }
        
        isLoading = true
        errorMessage = nil
        
        apiService.fetchResult(analysisId: serverId)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { completion in
                    isLoading = false
                    if case .failure(let error) = completion {
                        errorMessage = error.localizedDescription
                    }
                },
                receiveValue: { fetchedResult in
                    result = fetchedResult
                    syncManager.saveResultLocally(fetchedResult, for: analysis.id)
                }
            )
            .store(in: &cancellables)
    }
    
    private func qualityColor(_ score: Double) -> Color {
        if score >= 0.9 { return TeloscopyTheme.successGreen }
        if score >= 0.7 { return TeloscopyTheme.warningOrange }
        return TeloscopyTheme.errorRed
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: String
    let subtitle: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundColor(TeloscopyTheme.textSecondary)
            
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(subtitle)
                .font(.caption2)
                .foregroundColor(TeloscopyTheme.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(color.opacity(0.05))
        .cornerRadius(TeloscopyTheme.smallCornerRadius)
    }
}

struct MetricRow: View {
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
                .fontWeight(.semibold)
                .foregroundColor(TeloscopyTheme.textPrimary)
        }
    }
}

struct LegendItem: View {
    let color: Color
    let label: String
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .foregroundColor(TeloscopyTheme.textSecondary)
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        ResultsView()
    }
    .environmentObject(APIService.shared)
    .environmentObject(SyncManager.shared)
}
