# Teloscopy iOS

iOS companion app for the Teloscopy genomic analysis platform, built with Swift and SwiftUI.

## Overview

Teloscopy iOS mirrors the Android companion app, providing a native iOS experience for:

- **Telomere Analysis** — Upload microscopy or facial images for AI-powered telomere length estimation
- **Profile-Based Analysis** — Get genomic risk assessments based on demographic and genetic profile
- **Disease Risk Assessment** — Standalone disease risk predictions from genetic variants
- **Nutrition Planning** — Personalized meal plans based on genetic profile
- **Health Checkup** — Comprehensive health analysis with blood/urine panel integration

## Requirements

- iOS 16.0+
- Xcode 15.0+
- Swift 5.9+

## Architecture

The app follows a clean MVVM architecture with SwiftUI:

```
Teloscopy/
├── TeloscopyApp.swift          — App entry point
├── ContentView.swift           — Root tab navigation
├── Models/
│   └── APIModels.swift         — Codable request/response models
├── Services/
│   ├── APIService.swift        — URLSession-based network layer
│   └── SettingsStore.swift     — UserDefaults wrapper
├── Views/
│   ├── HomeView.swift          — Landing screen with feature cards
│   ├── AnalysisView.swift      — Image upload + camera + analysis
│   ├── ProfileAnalysisView.swift — Profile-only analysis
│   ├── ResultsView.swift       — Tabbed results display
│   └── SettingsView.swift      — Server config + about
├── Components/
│   ├── StatCard.swift          — Reusable stat display
│   ├── DiseaseRiskCard.swift   — Disease risk card
│   ├── MealPlanCard.swift      — Meal plan display
│   └── ShimmerEffect.swift     — Loading shimmer placeholder
└── Assets.xcassets/            — Colors and app icon
```

## Setup

1. Open `Teloscopy.xcodeproj` in Xcode
2. Select your development team under Signing & Capabilities
3. Build and run on a simulator or device
4. Configure the backend server URL in Settings (default: `http://localhost:8000`)

## Backend

The app connects to the Teloscopy FastAPI backend. See the main project README for backend setup instructions.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/analyze | Upload image for analysis |
| GET | /api/status/{jobId} | Poll job status |
| GET | /api/results/{jobId} | Get analysis results |
| POST | /api/profile-analysis | Profile-only analysis |
| POST | /api/validate-image | Validate image before upload |
| GET | /api/health | Health check |
| GET | /api/agents/status | Agent system status |
| POST | /api/nutrition | Nutrition plan |
| POST | /api/disease-risk | Disease risk assessment |
| POST | /api/health-checkup | Health checkup |

## Design

- Dark genomic theme matching the web application
- Background: `#0B0F19` / Surface: `#1F2937` / Accent: `#00D4AA`
- Tab-based navigation (Home, Analyze, Settings)
- Consistent with Material 3 design language on Android

## License

Proprietary — Teloscopy © 2024
