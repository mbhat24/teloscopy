// StatCard.swift
// Teloscopy
//
// Reusable stat display card component.
// Mirrors Android's StatCard.kt composable.

import SwiftUI

struct StatCard: View {
    let title: String
    let value: String
    let subtitle: String?
    let icon: String
    let gradientColors: [Color]

    init(
        title: String,
        value: String,
        subtitle: String? = nil,
        icon: String = "chart.bar.fill",
        gradientColors: [Color] = [Color.tsCyan, Color.tsCyan.opacity(0.6)]
    ) {
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.icon = icon
        self.gradientColors = gradientColors
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: gradientColors,
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Spacer()
            }

            Text(value)
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundColor(.tsTextPrimary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)

            Text(title)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.tsTextSecondary)
                .lineLimit(1)

            if let subtitle = subtitle {
                Text(subtitle)
                    .font(.system(size: 10, weight: .regular))
                    .foregroundColor(.tsTextSecondary.opacity(0.7))
                    .lineLimit(2)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.tsSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(
                            LinearGradient(
                                colors: [gradientColors[0].opacity(0.3), Color.clear],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        )
    }
}

// MARK: - Color Extensions

extension Color {
    static let tsBackground = Color(hex: 0x0B0F19)
    static let tsBackgroundSecondary = Color(hex: 0x111827)
    static let tsSurface = Color(hex: 0x1F2937)
    static let tsSurfaceVariant = Color(hex: 0x1E2235)
    static let tsSurfaceElevated = Color(hex: 0x283040)
    static let tsCyan = Color(hex: 0x00E5FF)
    static let tsAccent = Color(hex: 0x00D4AA)
    static let tsPurple = Color(hex: 0x7C4DFF)
    static let tsGreen = Color(hex: 0x69F0AE)
    static let tsAmber = Color(hex: 0xFFB74D)
    static let tsTextPrimary = Color(hex: 0xF9FAFB)
    static let tsTextSecondary = Color(hex: 0x9CA3AF)
    static let tsError = Color(hex: 0xEF4444)
    static let tsWarning = Color(hex: 0xF59E0B)
    static let tsRiskLow = Color(hex: 0x4ADE80)
    static let tsRiskModerate = Color(hex: 0xFB923C)
    static let tsRiskHigh = Color(hex: 0xEF4444)
    static let tsRiskVeryHigh = Color(hex: 0xDC2626)

    init(hex: UInt, alpha: Double = 1.0) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xff) / 255,
            green: Double((hex >> 8) & 0xff) / 255,
            blue: Double(hex & 0xff) / 255,
            opacity: alpha
        )
    }

    static func riskColor(for level: String) -> Color {
        switch level.lowercased() {
        case "low", "minimal":
            return .tsRiskLow
        case "moderate", "medium":
            return .tsRiskModerate
        case "high", "elevated":
            return .tsRiskHigh
        case "very high", "critical", "severe":
            return .tsRiskVeryHigh
        default:
            return .tsTextSecondary
        }
    }
}

// MARK: - ChipButton

struct ChipButton: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(isSelected ? .tsBackground : .tsTextSecondary)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(
                    Capsule()
                        .fill(isSelected ? Color.tsCyan : Color.tsSurface)
                )
                .overlay(
                    Capsule()
                        .stroke(isSelected ? Color.clear : Color.tsTextSecondary.opacity(0.3), lineWidth: 1)
                )
        }
    }
}

// MARK: - TeloscopyTextFieldStyle

struct TeloscopyTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color.tsSurfaceVariant)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.tsTextSecondary.opacity(0.3), lineWidth: 1)
            )
            .foregroundColor(.tsTextPrimary)
    }
}

#Preview {
    VStack(spacing: 16) {
        HStack(spacing: 12) {
            StatCard(
                title: "Biological Age",
                value: "34.2",
                subtitle: "years",
                icon: "clock.fill",
                gradientColors: [.tsCyan, .tsCyan.opacity(0.6)]
            )
            StatCard(
                title: "Telomere Length",
                value: "6.8 kb",
                subtitle: "mean",
                icon: "dna",
                gradientColors: [.tsPurple, .tsPurple.opacity(0.6)]
            )
        }
    }
    .padding()
    .background(Color.tsBackground)
}
