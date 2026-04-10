# Teloscopy Android App

A genomic health analysis companion app built with Jetpack Compose. Upload facial photos or enter profile data to receive biological age estimates, disease-risk assessments, and personalised nutrition plans powered by the Teloscopy backend.

## Features

- **Facial Analysis** -- Take or pick a photo for biological age estimation and telomere-length inference via computer vision.
- **Profile-Only Analysis** -- Enter age, sex, ancestry, known genetic variants, and optional telomere length for disease-risk and nutrition analysis without an image.
- **Disease Risk Assessment** -- Colour-coded risk cards with probability bars, contributing factors, and actionable recommendations.
- **Personalised Nutrition** -- Calorie targets, key nutrients, foods to increase/avoid, and multi-day meal plans that respect dietary restrictions.
- **Dark Theme** -- Material 3 dark-cyan theme optimised for OLED displays.

## Tech Stack

| Layer | Library |
|-------|---------|
| UI | Jetpack Compose + Material 3 |
| DI | Hilt |
| Networking | Retrofit + Moshi + OkHttp |
| Image loading | Coil |
| Camera | CameraX (via FileProvider) |
| Preferences | Jetpack DataStore |
| Charts | Vico |
| Min SDK | 26 (Android 8.0) |
| Target SDK | 34 (Android 14) |

## Project Structure

```
app/src/main/java/com/teloscopy/app/
  +-- TeloscopyApp.kt           # @HiltAndroidApp
  +-- MainActivity.kt           # @AndroidEntryPoint, single-activity host
  +-- data/
  |   +-- api/
  |   |   +-- ApiModels.kt      # Moshi data classes (14+ models)
  |   |   +-- TeloscopyApi.kt   # Retrofit interface (8 endpoints)
  |   +-- repository/
  |       +-- AnalysisRepository.kt  # Result<T> wrapper
  +-- di/
  |   +-- AppModule.kt          # Hilt @Module providing Retrofit, Moshi, etc.
  +-- viewmodel/
  |   +-- AnalysisViewModel.kt  # Image analysis flow + job polling
  |   +-- ProfileViewModel.kt   # Profile-only analysis
  +-- ui/
      +-- navigation/NavGraph.kt
      +-- theme/{Color,Type,Theme}.kt
      +-- screens/{Home,Analysis,Results,ProfileAnalysis,Settings}Screen.kt
      +-- components/{SectionHeader,StatCard,DiseaseRiskCard,MealPlanCard}.kt
```

## Building

```bash
# Debug build
./gradlew assembleDebug

# Release build (requires signing config)
./gradlew assembleRelease
```

The app connects to the Teloscopy backend at `http://10.0.2.2:8000/` by default (Android emulator localhost alias). For physical devices, update the `BASE_URL` in `AppModule.kt`.

## Backend Setup

The companion backend lives in the parent directory. Start it with:

```bash
cd ..
pip install -e .
uvicorn teloscopy.webapp.app:app --host 0.0.0.0 --port 8000
```

## Play Store Publishing

1. Create a [Google Play Developer account](https://play.google.com/console) ($25 one-time fee).
2. Complete identity verification.
3. Generate a signed release APK/AAB with your upload key.
4. Create a new app listing and fill in the store metadata (see `STORE_LISTING.md`).
5. Upload the AAB to the internal testing track first, then promote to production.

## License

Proprietary -- all rights reserved.
