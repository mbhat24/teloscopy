// MealPlanCard.swift
// Teloscopy
//
// Meal plan display card component.
// Mirrors Android's MealPlanCard.kt composable.

import SwiftUI

struct MealPlanCard: View {
    let mealPlan: MealPlan
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button(action: { withAnimation(.spring(response: 0.3)) { isExpanded.toggle() } }) {
                HStack {
                    Image(systemName: "calendar")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.tsGreen)

                    Text("Day \(mealPlan.day)")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.tsTextPrimary)

                    Spacer()

                    let totalItems = mealPlan.breakfast.count + mealPlan.lunch.count + mealPlan.dinner.count + mealPlan.snacks.count
                    Text("\(totalItems) items")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.tsTextSecondary)

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.tsTextSecondary)
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 16) {
                    MealSection(title: "Breakfast", icon: "sunrise.fill", items: mealPlan.breakfast, color: .tsAmber)
                    MealSection(title: "Lunch", icon: "sun.max.fill", items: mealPlan.lunch, color: .tsCyan)
                    MealSection(title: "Dinner", icon: "moon.stars.fill", items: mealPlan.dinner, color: .tsPurple)

                    if !mealPlan.snacks.isEmpty {
                        MealSection(title: "Snacks", icon: "leaf.fill", items: mealPlan.snacks, color: .tsGreen)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.tsSurface)
        )
    }
}

struct MealSection: View {
    let title: String
    let icon: String
    let items: [String]
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(color)

                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(color)
            }

            ForEach(items, id: \.self) { item in
                HStack(alignment: .top, spacing: 8) {
                    Circle()
                        .fill(color.opacity(0.5))
                        .frame(width: 4, height: 4)
                        .padding(.top, 6)

                    Text(item)
                        .font(.system(size: 13))
                        .foregroundColor(.tsTextPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.tsSurfaceVariant.opacity(0.5))
        )
    }
}

#Preview {
    ScrollView {
        VStack(spacing: 12) {
            MealPlanCard(mealPlan: MealPlan(
                day: 1,
                breakfast: ["Oatmeal with berries", "Green tea", "Boiled eggs"],
                lunch: ["Grilled salmon salad", "Quinoa", "Avocado"],
                dinner: ["Lean chicken breast", "Steamed vegetables", "Brown rice"],
                snacks: ["Mixed nuts", "Greek yogurt"]
            ))
        }
        .padding()
    }
    .background(Color.tsBackground)
}
