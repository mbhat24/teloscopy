// ProfileAnalysisView.swift
// Teloscopy
//
// Profile-only analysis form (no image required).
// Mirrors Android's ProfileAnalysisScreen.kt composable.

import SwiftUI

// MARK: - Profile Analysis View

struct ProfileAnalysisView: View {
    @EnvironmentObject var settings: SettingsStore
    @StateObject private var viewModel = ProfileAnalysisViewModel()

    let onNavigateToResults: (String) -> Void

    // Form state
    @State private var age = ""
    @State private var selectedSex = ""
    @State private var selectedRegion = ""
    @State private var variants = ""
    @State private var selectedRestrictions: Set<String> = []
    @State private var includeNutrition = true
    @State private var includeDiseaseRisk = true
    @State private var calorieTarget = "2000"
    @State private var mealPlanDays = "7"
    @State private var activeTab = 0 // 0 = combined, 1 = disease risk, 2 = nutrition

    private let regionOptions = [
        "Northern Europe", "Southern Europe", "East Asia", "South Asia",
        "West Africa", "East Africa", "Middle East", "Central America",
        "South America", "Oceania"
    ]

    private let dietaryOptions = [
        "Vegetarian", "Vegan", "Pescatarian", "Gluten-Free",
        "Lactose-Free", "Low-Sodium", "Halal", "Kosher", "Diabetic-Friendly"
    ]

    private let sexOptions = ["Male", "Female", "Other"]

    private var ageValue: Int? { Int(age) }
    private var isAgeValid: Bool {
        guard let v = ageValue else { return false }
        return v >= 1 && v <= 120
    }
    private var canSubmit: Bool {
        !age.isEmpty && isAgeValid && !selectedSex.isEmpty && !selectedRegion.isEmpty
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Analysis type tabs
                    analysisTypePicker

                    // Form
                    profileForm

                    // Submit button
                    submitButton

                    // Results
                    if viewModel.isLoading {
                        SkeletonResultsView()
                    }

                    if let error = viewModel.error {
                        errorBanner(message: error)
                    }

                    // Profile results
                    if let profileResult = viewModel.profileResult {
                        profileResultsSection(profileResult)
                    }

                    if let diseaseResult = viewModel.diseaseRiskResult {
                        diseaseResultsSection(diseaseResult)
                    }

                    if let nutritionResult = viewModel.nutritionResult {
                        nutritionResultsSection(nutritionResult)
                    }

                    Spacer(minLength: 32)
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
            }
            .background(Color.tsBackground)
            .navigationTitle("Profile Analysis")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Color.tsBackground, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
        }
    }

    // MARK: - Analysis Type Picker

    private var analysisTypePicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Analysis Type")
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(.tsCyan)

            HStack(spacing: 0) {
                SegmentButton(title: "Combined", isSelected: activeTab == 0) { activeTab = 0 }
                SegmentButton(title: "Disease Risk", isSelected: activeTab == 1) { activeTab = 1 }
                SegmentButton(title: "Nutrition", isSelected: activeTab == 2) { activeTab = 2 }
            }
            .background(Color.tsSurface)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    // MARK: - Profile Form

    private var profileForm: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Profile Information")
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(.tsCyan)

            VStack(spacing: 16) {
                // Age
                VStack(alignment: .leading, spacing: 4) {
                    Text("Age *")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.tsTextSecondary)
                    TextField("Enter your age", text: $age)
                        .keyboardType(.numberPad)
                        .textFieldStyle(TeloscopyTextFieldStyle())
                    if !age.isEmpty && !isAgeValid {
                        Text("Age must be between 1 and 120")
                            .font(.system(size: 12))
                            .foregroundColor(.tsError)
                    }
                }

                // Sex
                VStack(alignment: .leading, spacing: 8) {
                    Text("Sex *")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.tsTextSecondary)
                    HStack(spacing: 8) {
                        ForEach(sexOptions, id: \.self) { option in
                            ChipButton(
                                title: option,
                                isSelected: selectedSex == option,
                                action: { selectedSex = option }
                            )
                        }
                    }
                }

                // Region
                VStack(alignment: .leading, spacing: 4) {
                    Text("Region / Ancestry *")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.tsTextSecondary)
                    Menu {
                        ForEach(regionOptions, id: \.self) { region in
                            Button(region) { selectedRegion = region }
                        }
                    } label: {
                        HStack {
                            Text(selectedRegion.isEmpty ? "Select region" : selectedRegion)
                                .foregroundColor(selectedRegion.isEmpty ? .tsTextSecondary.opacity(0.5) : .tsTextPrimary)
                            Spacer()
                            Image(systemName: "chevron.down")
                                .foregroundColor(.tsTextSecondary)
                        }
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.tsTextSecondary.opacity(0.3), lineWidth: 1)
                        )
                    }
                }

                // Known variants
                VStack(alignment: .leading, spacing: 4) {
                    Text("Known Variants")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.tsTextSecondary)
                    TextField("rs429358:CT, rs7412:CC", text: $variants, axis: .vertical)
                        .lineLimit(2...4)
                        .textFieldStyle(TeloscopyTextFieldStyle())
                }

                // Dietary restrictions
                VStack(alignment: .leading, spacing: 8) {
                    Text("Dietary Restrictions")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.tsTextSecondary)

                    FlowLayout(spacing: 8) {
                        ForEach(dietaryOptions, id: \.self) { restriction in
                            ChipButton(
                                title: restriction,
                                isSelected: selectedRestrictions.contains(restriction),
                                action: {
                                    if selectedRestrictions.contains(restriction) {
                                        selectedRestrictions.remove(restriction)
                                    } else {
                                        selectedRestrictions.insert(restriction)
                                    }
                                }
                            )
                        }
                    }
                }

                // Nutrition-specific options
                if activeTab == 0 || activeTab == 2 {
                    VStack(alignment: .leading, spacing: 12) {
                        Divider()
                            .background(Color.tsTextSecondary.opacity(0.2))

                        Text("Nutrition Options")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(.tsTextSecondary)

                        HStack(spacing: 16) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Calorie Target")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(.tsTextSecondary)
                                TextField("2000", text: $calorieTarget)
                                    .keyboardType(.numberPad)
                                    .textFieldStyle(TeloscopyTextFieldStyle())
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Meal Plan Days")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(.tsTextSecondary)
                                TextField("7", text: $mealPlanDays)
                                    .keyboardType(.numberPad)
                                    .textFieldStyle(TeloscopyTextFieldStyle())
                            }
                        }
                    }
                }

                // Toggles for combined analysis
                if activeTab == 0 {
                    VStack(spacing: 8) {
                        Toggle("Include Disease Risk", isOn: $includeDiseaseRisk)
                            .tint(.tsCyan)
                            .foregroundColor(.tsTextPrimary)
                        Toggle("Include Nutrition Plan", isOn: $includeNutrition)
                            .tint(.tsCyan)
                            .foregroundColor(.tsTextPrimary)
                    }
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.tsSurface)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.tsTextSecondary.opacity(0.15), lineWidth: 1)
                    )
            )
        }
    }

    // MARK: - Submit Button

    private var submitButton: some View {
        Button(action: submit) {
            HStack(spacing: 8) {
                Image(systemName: activeTab == 0 ? "person.fill.checkmark" : activeTab == 1 ? "shield.checkered" : "fork.knife")
                    .font(.system(size: 16))
                Text(activeTab == 0 ? "Run Analysis" : activeTab == 1 ? "Assess Risk" : "Generate Plan")
                    .font(.system(size: 17, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .foregroundColor(canSubmit && !viewModel.isLoading ? Color.tsBackground : Color.tsBackground.opacity(0.5))
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(canSubmit && !viewModel.isLoading ? Color.tsCyan : Color.tsCyan.opacity(0.3))
            )
        }
        .disabled(!canSubmit || viewModel.isLoading)
    }

    // MARK: - Error Banner

    private func errorBanner(message: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.tsError)
            Text(message)
                .font(.system(size: 13))
                .foregroundColor(.tsTextPrimary)
            Spacer()
            Button(action: { viewModel.clearError() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 12))
                    .foregroundColor(.tsTextSecondary)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.tsError.opacity(0.1))
        )
    }

    // MARK: - Profile Results

    private func profileResultsSection(_ result: ProfileAnalysisResponse) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Profile Analysis Results")
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.tsTextPrimary)

            if let summary = result.summary {
                Text(summary)
                    .font(.system(size: 14))
                    .foregroundColor(.tsTextSecondary)
                    .padding(12)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Color.tsSurface))
            }

            if let risks = result.diseaseRisks, !risks.isEmpty {
                Text("Disease Risks")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundColor(.tsTextPrimary)
                ForEach(risks) { risk in
                    DiseaseRiskCard(risk: risk)
                }
            }

            if let diet = result.dietRecommendations {
                dietRecommendationSection(diet)
            }
        }
    }

    // MARK: - Disease Results

    private func diseaseResultsSection(_ result: DiseaseRiskResponse) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Disease Risk Assessment")
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.tsTextPrimary)

            if let risks = result.diseaseRisks, !risks.isEmpty {
                ForEach(risks) { risk in
                    DiseaseRiskCard(risk: risk)
                }
            } else {
                Text("No disease risks identified")
                    .font(.system(size: 14))
                    .foregroundColor(.tsTextSecondary)
            }
        }
    }

    // MARK: - Nutrition Results

    private func nutritionResultsSection(_ result: NutritionResponse) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Nutrition Plan")
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.tsTextPrimary)

            if let diet = result.dietRecommendations {
                dietRecommendationSection(diet)
            }

            if let plans = result.mealPlans, !plans.isEmpty {
                ForEach(plans) { plan in
                    MealPlanCard(mealPlan: plan)
                }
            }
        }
    }

    // MARK: - Diet Recommendation

    private func dietRecommendationSection(_ diet: DietRecommendation) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if let summary = diet.summary {
                Text(summary)
                    .font(.system(size: 14))
                    .foregroundColor(.tsTextSecondary)
            }

            HStack(spacing: 12) {
                StatCard(
                    title: "Daily Calories",
                    value: "\(diet.displayCalories)",
                    icon: "flame.fill",
                    gradientColors: [.tsAmber, .tsAmber.opacity(0.6)]
                )
                StatCard(
                    title: "Key Nutrients",
                    value: "\(diet.displayNutrients.count)",
                    icon: "leaf.fill",
                    gradientColors: [.tsGreen, .tsGreen.opacity(0.6)]
                )
            }

            if !diet.displayTargetFoods.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Foods to Increase")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.tsGreen)
                    FlowLayout(spacing: 6) {
                        ForEach(diet.displayTargetFoods, id: \.self) { food in
                            Text(food)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.tsGreen)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Capsule().fill(Color.tsGreen.opacity(0.15)))
                        }
                    }
                }
            }

            if !diet.displayAvoidFoods.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Foods to Avoid")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.tsError)
                    FlowLayout(spacing: 6) {
                        ForEach(diet.displayAvoidFoods, id: \.self) { food in
                            Text(food)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.tsError)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Capsule().fill(Color.tsError.opacity(0.15)))
                        }
                    }
                }
            }

            if let plans = diet.mealPlans, !plans.isEmpty {
                ForEach(plans) { plan in
                    MealPlanCard(mealPlan: plan)
                }
            }
        }
    }

    // MARK: - Submit Logic

    private func submit() {
        let parsedVariants = variants
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        switch activeTab {
        case 0:
            viewModel.runProfileAnalysis(
                age: ageValue ?? 30,
                sex: selectedSex,
                region: selectedRegion,
                restrictions: Array(selectedRestrictions),
                variants: parsedVariants,
                includeNutrition: includeNutrition,
                includeDiseaseRisk: includeDiseaseRisk
            )
        case 1:
            viewModel.runDiseaseRisk(
                age: ageValue ?? 30,
                sex: selectedSex,
                region: selectedRegion,
                variants: parsedVariants
            )
        case 2:
            viewModel.runNutrition(
                age: ageValue ?? 30,
                sex: selectedSex,
                region: selectedRegion,
                restrictions: Array(selectedRestrictions),
                variants: parsedVariants,
                calorieTarget: Int(calorieTarget) ?? 2000,
                days: Int(mealPlanDays) ?? 7
            )
        default:
            break
        }
    }
}

// MARK: - Segment Button

struct SegmentButton: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(isSelected ? .tsBackground : .tsTextSecondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(isSelected ? Color.tsCyan : Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }
}

// MARK: - Profile Analysis ViewModel

@MainActor
class ProfileAnalysisViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var error: String?
    @Published var profileResult: ProfileAnalysisResponse?
    @Published var diseaseRiskResult: DiseaseRiskResponse?
    @Published var nutritionResult: NutritionResponse?

    func runProfileAnalysis(
        age: Int, sex: String, region: String,
        restrictions: [String], variants: [String],
        includeNutrition: Bool, includeDiseaseRisk: Bool
    ) {
        isLoading = true
        error = nil
        profileResult = nil

        Task {
            do {
                let request = ProfileAnalysisRequest(
                    age: age,
                    sex: sex,
                    region: region,
                    country: nil,
                    state: nil,
                    geneticRisks: variants,
                    variants: [:],
                    dietaryRestrictions: restrictions
                )
                profileResult = try await APIService.shared.profileAnalysis(request: request)
                isLoading = false
            } catch {
                isLoading = false
                self.error = error.localizedDescription
            }
        }
    }

    func runDiseaseRisk(age: Int, sex: String, region: String, variants: [String]) {
        isLoading = true
        error = nil
        diseaseRiskResult = nil

        Task {
            do {
                let request = DiseaseRiskRequest(
                    variants: Dictionary(uniqueKeysWithValues: variants.compactMap { v -> (String, String)? in
                        let parts = v.split(separator: ":")
                        guard parts.count == 2 else { return nil }
                        return (String(parts[0]), String(parts[1]))
                    }),
                    age: age,
                    sex: sex,
                    region: region
                )
                diseaseRiskResult = try await APIService.shared.diseaseRisk(request: request)
                isLoading = false
            } catch {
                isLoading = false
                self.error = error.localizedDescription
            }
        }
    }

    func runNutrition(
        age: Int, sex: String, region: String,
        restrictions: [String], variants: [String],
        calorieTarget: Int, days: Int
    ) {
        isLoading = true
        error = nil
        nutritionResult = nil

        Task {
            do {
                let request = NutritionRequest(
                    age: age,
                    sex: sex,
                    region: region,
                    country: nil,
                    state: nil,
                    geneticRisks: variants,
                    variants: [:],
                    dietaryRestrictions: restrictions,
                    caloriesTarget: calorieTarget,
                    days: days
                )
                nutritionResult = try await APIService.shared.nutrition(request: request)
                isLoading = false
            } catch {
                isLoading = false
                self.error = error.localizedDescription
            }
        }
    }

    func clearError() {
        error = nil
    }
}

#Preview {
    ProfileAnalysisView(onNavigateToResults: { _ in })
        .environmentObject(SettingsStore.shared)
}
