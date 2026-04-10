// ShimmerEffect.swift
// Teloscopy
//
// Loading shimmer/skeleton placeholder effect.
// Mirrors Android's ShimmerEffect.kt composable.

import SwiftUI

struct ShimmerEffect: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .overlay(
                GeometryReader { geometry in
                    LinearGradient(
                        gradient: Gradient(stops: [
                            .init(color: Color.clear, location: max(0, phase - 0.3)),
                            .init(color: Color.white.opacity(0.1), location: phase),
                            .init(color: Color.clear, location: min(1, phase + 0.3))
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                }
                .allowsHitTesting(false)
            )
            .onAppear {
                withAnimation(
                    .linear(duration: 1.5)
                    .repeatForever(autoreverses: false)
                ) {
                    phase = 1.3
                }
            }
    }
}

extension View {
    func shimmer() -> some View {
        modifier(ShimmerEffect())
    }
}

// MARK: - Skeleton Loading Views

struct ShimmerBox: View {
    let width: CGFloat?
    let height: CGFloat

    init(width: CGFloat? = nil, height: CGFloat = 16) {
        self.width = width
        self.height = height
    }

    var body: some View {
        RoundedRectangle(cornerRadius: height / 2)
            .fill(Color.tsSurfaceVariant)
            .frame(width: width, height: height)
            .shimmer()
    }
}

struct SkeletonStatCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ShimmerBox(width: 24, height: 24)
            ShimmerBox(width: 80, height: 28)
            ShimmerBox(width: 60, height: 14)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.tsSurface)
        )
    }
}

struct SkeletonRiskCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                ShimmerBox(width: 12, height: 12)
                VStack(alignment: .leading, spacing: 4) {
                    ShimmerBox(width: 160, height: 16)
                    ShimmerBox(width: 60, height: 12)
                }
                Spacer()
                ShimmerBox(width: 50, height: 20)
            }
            ShimmerBox(height: 4)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.tsSurface)
        )
    }
}

struct SkeletonResultsView: View {
    var body: some View {
        VStack(spacing: 16) {
            // Stat cards grid
            HStack(spacing: 12) {
                SkeletonStatCard()
                SkeletonStatCard()
            }

            HStack(spacing: 12) {
                SkeletonStatCard()
                SkeletonStatCard()
            }

            // Risk cards
            SkeletonRiskCard()
            SkeletonRiskCard()
            SkeletonRiskCard()
        }
        .padding()
    }
}

#Preview {
    ScrollView {
        SkeletonResultsView()
    }
    .background(Color.tsBackground)
}
