// DiseaseRiskCard.swift
// Teloscopy
//
// Disease risk display card component.
// Mirrors Android's DiseaseRiskCard.kt composable.

import SwiftUI

struct DiseaseRiskCard: View {
    let risk: DiseaseRisk
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button(action: { withAnimation(.spring(response: 0.3)) { isExpanded.toggle() } }) {
                HStack(spacing: 12) {
                    // Risk level indicator
                    Circle()
                        .fill(Color.riskColor(for: risk.riskLevel))
                        .frame(width: 12, height: 12)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(risk.disease)
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundColor(.tsTextPrimary)
                            .lineLimit(1)

                        Text(risk.riskLevel.capitalized)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(Color.riskColor(for: risk.riskLevel))
                    }

                    Spacer()

                    // Risk percentage
                    if risk.displayRisk > 0 {
                        Text(String(format: "%.1f%%", risk.displayRisk))
                            .font(.system(size: 18, weight: .bold, design: .rounded))
                            .foregroundColor(Color.riskColor(for: risk.riskLevel))
                    }

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.tsTextSecondary)
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            // Risk progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.tsSurfaceVariant)
                        .frame(height: 4)

                    Rectangle()
                        .fill(Color.riskColor(for: risk.riskLevel))
                        .frame(width: geometry.size.width * min(risk.displayRisk / 100.0, 1.0), height: 4)
                }
            }
            .frame(height: 4)
            .padding(.horizontal, 16)

            // Expanded details
            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    // Relative risk
                    if let relativeRisk = risk.relativeRisk, relativeRisk > 0 {
                        HStack {
                            Text("Relative Risk:")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.tsTextSecondary)
                            Text(String(format: "%.2fx", relativeRisk))
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(.tsTextPrimary)
                        }
                    }

                    // Contributing factors
                    let factors = risk.displayFactors
                    if !factors.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Contributing Factors")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(.tsTextSecondary)

                            FlowLayout(spacing: 6) {
                                ForEach(factors, id: \.self) { factor in
                                    Text(factor)
                                        .font(.system(size: 11, weight: .medium))
                                        .foregroundColor(.tsCyan)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(
                                            Capsule()
                                                .fill(Color.tsCyan.opacity(0.15))
                                        )
                                }
                            }
                        }
                    }

                    // Recommendations
                    if let recommendations = risk.recommendations, !recommendations.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Recommendations")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(.tsTextSecondary)

                            ForEach(recommendations, id: \.self) { rec in
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.system(size: 12))
                                        .foregroundColor(.tsGreen)
                                        .padding(.top, 2)

                                    Text(rec)
                                        .font(.system(size: 13))
                                        .foregroundColor(.tsTextPrimary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                    }
                }
                .padding(16)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.tsSurface)
        )
    }
}

// MARK: - Flow Layout for Chips

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = layout(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = layout(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y), proposal: .unspecified)
        }
    }

    private func layout(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var totalHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)

            if currentX + size.width > maxWidth, currentX > 0 {
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }

            positions.append(CGPoint(x: currentX, y: currentY))
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            totalHeight = currentY + lineHeight
        }

        return (CGSize(width: maxWidth, height: totalHeight), positions)
    }
}

#Preview {
    ScrollView {
        VStack(spacing: 12) {
            DiseaseRiskCard(risk: DiseaseRisk(
                disease: "Type 2 Diabetes",
                lifetimeRiskPct: 23.5,
                relativeRisk: 1.8,
                riskLevel: "moderate",
                probability: nil,
                contributingVariants: ["TCF7L2", "PPARG", "KCNJ11"],
                contributingFactors: nil,
                recommendations: ["Regular blood glucose monitoring", "Mediterranean diet recommended", "150 min/week moderate exercise"]
            ))
            DiseaseRiskCard(risk: DiseaseRisk(
                disease: "Cardiovascular Disease",
                lifetimeRiskPct: 45.2,
                relativeRisk: 2.1,
                riskLevel: "high",
                probability: nil,
                contributingVariants: ["APOE", "LDLR"],
                contributingFactors: nil,
                recommendations: ["Statin therapy discussion with physician"]
            ))
        }
        .padding()
    }
    .background(Color.tsBackground)
}
