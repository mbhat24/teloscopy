# Teloscopy Architecture

> **Version 2.0** — Multi-Agent Genomic Intelligence Platform

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Project Structure](#3-project-structure)
4. [Multi-Agent System](#4-multi-agent-system)
5. [Core Modules Reference](#5-core-modules-reference)
6. [Web Application](#6-web-application)
7. [Android Application](#7-android-application)
8. [Data Layer — JSON Data Files](#8-data-layer--json-data-files)
9. [Disease Risk Prediction Engine](#9-disease-risk-prediction-engine)
10. [Nutrigenomic Diet Advisor](#10-nutrigenomic-diet-advisor)
11. [Health Checkup System](#11-health-checkup-system)
12. [Facial-Genomic Analysis](#12-facial-genomic-analysis)
13. [Image Analysis Pipeline](#13-image-analysis-pipeline)
14. [API Reference](#14-api-reference)
15. [Test Suite](#15-test-suite)
16. [Deployment Architecture](#16-deployment-architecture)
17. [Security & Privacy](#17-security--privacy)
18. [Scientific Background](#18-scientific-background)
19. [Performance Considerations](#19-performance-considerations)
20. [Future Roadmap](#20-future-roadmap)

---

## 1. System Overview

Teloscopy is a **multi-agent genomic intelligence platform** that:

1. Accepts fluorescence microscopy images (qFISH) or face photos via web upload, CLI, Android app, or iOS app
2. Analyzes telomere length at each chromosome end using computer vision
3. Predicts disease risk using telomere data + 560 genetic variants (SNPs)
4. Generates personalized diet plans based on genetics + geographic food availability across 35 regions
5. Performs health checkup analysis from blood/urine lab reports with 24 condition detectors
6. Reconstructs partial DNA sequences and pharmacogenomic profiles from facial analysis
7. Continuously self-improves through an autonomous 6-agent orchestration system

```
                           TELOSCOPY v2.0
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   USER ──────► WEB UI / CLI / Android / iOS / Python API   │
    │                      │                                  │
    │                      ▼                                  │
    │            ┌─────────────────┐                          │
    │            │  ORCHESTRATOR   │  Multi-Agent Controller  │
    │            │     AGENT       │                          │
    │            └────┬──┬──┬──┬──┘                          │
    │                 │  │  │  │                              │
    │       ┌─────┐  │  │  │  └──────┐                      │
    │       ▼     ▼  ▼  ▼  ▼        ▼                      │
    │    ┌─────┐┌─────┐┌─────┐┌──────────┐┌────────┐       │
    │    │IMAGE││GENO-││NUTRI││IMPROVE-  ││REPORT  │       │
    │    │AGENT││MICS ││TION ││MENT     ││AGENT   │       │
    │    │     ││AGENT││AGENT││AGENT    ││        │       │
    │    └──┬──┘└──┬──┘└──┬──┘└────┬─────┘└───┬────┘       │
    │       │      │      │        │           │            │
    │       ▼      ▼      ▼        ▼           ▼            │
    │   ┌───────────────────────────────────────────┐       │
    │   │           CORE MODULES                     │       │
    │   │  Preprocessing │ Segmentation │ Detection  │       │
    │   │  Association   │ Quantification│ Synthetic │       │
    │   │  Disease Risk  │ Diet Advisor  │ Statistics │       │
    │   │  Facial/Genomic│ Health Checkup│ i18n       │       │
    │   │  Sequencing    │ Visualization │ Pipeline   │       │
    │   └──────────────────────┬────────────────────┘       │
    │                          │                             │
    │   ┌──────────────────────▼────────────────────┐       │
    │   │           DATA LAYER (43 JSON files)       │       │
    │   │  560 SNP variants │ 650 foods │ 35 regions │       │
    │   │  125 disease baselines │ 24 condition rules│       │
    │   │  10 languages │ 30 country profiles        │       │
    │   └───────────────────────────────────────────┘       │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

### Project Metrics

| Metric | Value |
|--------|-------|
| Python source files | 63 files across 16 subpackages |
| Python lines of code | ~37,000 |
| JSON data files | 43 files (~46,400 lines) |
| Android Kotlin files | 22 files (~7,700 lines) |
| iOS Swift files | 10+ files (~3,000 lines) |
| HTML templates | 2 files (~5,400 lines) |
| Test files | 11 files (530+ tests, ~5,200 lines) |
| **Total codebase** | **~105,000+ lines** |

---

## 2. High-Level Architecture

### 2.1 Architecture Pattern

Teloscopy uses a **microkernel + multi-agent** architecture:

- **Microkernel**: Core image analysis pipeline (preprocessing → segmentation → detection → quantification)
- **Agents**: Autonomous specialist modules that communicate via async message passing
- **Plugin System**: New analysis methods can be added without modifying core code
- **Event-Driven**: Agents react to events (image uploaded, analysis complete, etc.)

### 2.2 Layer Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                        │
│  Web UI (FastAPI+Jinja2) │ CLI (Click) │ Android (Compose)   │
│  iOS (SwiftUI)                                                │
├──────────────────────────────────────────────────────────────┤
│                     ORCHESTRATION LAYER                       │
│  OrchestratorAgent  │  Workflow Engine  │  Message Router     │
├──────────────────────────────────────────────────────────────┤
│                     INTELLIGENCE LAYER                        │
│  ImageAgent │ GenomicsAgent │ NutritionAgent │ ReportAgent    │
│  ImprovementAgent │ FacialPredictor │ HealthCheckupAnalyzer   │
├──────────────────────────────────────────────────────────────┤
│                       DOMAIN LAYER                            │
│  Telomere Pipeline  │ Disease Risk │ Diet Advisor │ i18n      │
│  Sequencing │ Statistics │ Visualization │ Report Parser      │
├──────────────────────────────────────────────────────────────┤
│                        DATA LAYER                             │
│  43 JSON data files (SNPs, foods, regions, lab ranges, i18n)  │
├──────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                        │
│  NumPy/SciPy/Pandas │ scikit-image │ OpenCV │ tifffile        │
│  FastAPI/Uvicorn │ Cellpose (opt) │ Biopython (opt)           │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Web Frontend | Vanilla HTML/CSS/JS + Chart.js | Upload UI, results display, dashboard |
| Android | Jetpack Compose + Material3 + Hilt | Native mobile client |
| iOS | SwiftUI + Combine | Native iOS client |
| API Server | FastAPI + Uvicorn | REST API, async processing |
| Image Processing | scikit-image, OpenCV, tifffile | Core CV pipeline |
| Deep Learning | Cellpose (optional) | Chromosome segmentation |
| Scientific Computing | NumPy, SciPy, pandas | Data processing |
| Visualization | Matplotlib, Seaborn | Plots and overlays |
| CLI | Click + Rich | Terminal interface |
| Containerization | Docker + Docker Compose | Deployment |
| CI/CD | GitHub Actions | Lint → Test (3.11/3.12/3.13 matrix) → Docker |
| Testing | pytest (520 tests) | Unit + integration tests |
| Linting | ruff | Code quality |

---

## 3. Project Structure

### 3.1 Complete File Tree

```
teloscopy/
├── pyproject.toml                          # Build config, deps, metadata
├── requirements.txt                        # Flat requirements (alternative)
├── Makefile                                # 15 build/dev targets
├── Dockerfile                              # Multi-stage (python:3.12-slim)
├── docker-compose.yml                      # Single service, 2G memory limit
├── render.yaml                             # Render.com deployment config
├── setup.sh                                # Interactive installer (Docker/venv)
├── .env.example                            # Environment variable reference
├── .github/workflows/ci.yml               # CI pipeline (lint→test→docker)
├── ARCHITECTURE.md                         # This document
├── KNOWLEDGE_BASE.md                       # Scientific background (1,258 lines)
├── README.md                               # Getting started guide
├── LICENSE                                 # MIT
│
├── src/teloscopy/
│   ├── __init__.py                 [  25]  # Package root, __version__="2.0.0"
│   ├── cli.py                      [ 400]  # Click CLI: analyze, batch, generate, report
│   │
│   ├── telomere/                           # Core qFISH image analysis pipeline
│   │   ├── __init__.py             [  18]
│   │   ├── preprocessing.py        [ 294]  # Image loading, background subtraction, denoising
│   │   ├── segmentation.py         [ 288]  # Otsu+watershed / Cellpose segmentation
│   │   ├── spot_detection.py       [ 319]  # LoG / DoG / DoH blob detection
│   │   ├── association.py          [ 207]  # KDTree spot-to-chromosome tip matching
│   │   ├── quantification.py       [ 387]  # Aperture photometry + calibration
│   │   ├── pipeline.py             [ 672]  # End-to-end 9-step orchestrator
│   │   └── synthetic.py            [ 474]  # Synthetic qFISH test image generator
│   │
│   ├── genomics/                           # Genetic disease risk prediction
│   │   ├── __init__.py             [  23]
│   │   └── disease_risk.py         [ 969]  # 560 SNPs, 26 disease categories, PRS
│   │
│   ├── nutrition/                          # Diet recommendation engine
│   │   ├── __init__.py             [  55]
│   │   ├── diet_advisor.py         [1377]  # Nutrigenomics, 650 foods, 35 regions
│   │   ├── health_checkup.py       [2651]  # Lab interpretation, 24 condition detectors
│   │   ├── i18n.py                 [ 278]  # 10-language translation
│   │   └── regional_diets.py       [ 169]  # Country/state → region resolution
│   │
│   ├── facial/                             # Facial-genomic analysis
│   │   ├── __init__.py             [  12]
│   │   ├── predictor.py            [1639]  # 59 SNP prediction, ancestry, pharmacogenomics
│   │   └── image_classifier.py     [ 268]  # FISH vs face photo classification
│   │
│   ├── agents/                             # Multi-agent orchestration system
│   │   ├── __init__.py             [  28]
│   │   ├── base.py                 [ 319]  # BaseAgent, AgentMessage, AgentState, _MessageRouter
│   │   ├── orchestrator.py         [ 535]  # Central coordinator, workflow engine
│   │   ├── image_agent.py          [ 381]  # Image analysis specialist
│   │   ├── genomics_agent.py       [ 468]  # Disease risk specialist
│   │   ├── nutrition_agent.py      [2340]  # Diet planning specialist
│   │   ├── improvement_agent.py    [ 616]  # Self-optimization agent
│   │   └── report_agent.py         [ 529]  # Report generation agent
│   │
│   ├── webapp/                             # Web application
│   │   ├── __init__.py             [  10]
│   │   ├── app.py                  [1886]  # FastAPI server, 20 REST endpoints
│   │   ├── models.py               [ 914]  # 45 Pydantic request/response models
│   │   ├── health_checkup.py       [1038]  # Webapp health checkup analyzer
│   │   ├── report_parser.py        [ 567]  # PDF/image/text lab report extraction
│   │   └── templates/
│   │       ├── index.html          [3851]  # Main app (6 sections, 36 JS functions)
│   │       └── dashboard.html      [1520]  # Agent monitoring dashboard
│   │
│   ├── integrations/                       # External system integrations
│   │   ├── __init__.py             [  43]
│   │   ├── fhir.py                 [1855]  # HL7 FHIR export + HIPAA compliance
│   │   ├── genotype_import.py      [ 809]  # 23andMe/AncestryDNA/VCF import
│   │   ├── llm_reports.py          [ 772]  # Ollama/OpenAI report generation
│   │   └── wgs.py                  [1518]  # Whole genome sequencing analysis
│   │
│   ├── clinical/                           # Clinical validation & trials
│   │   ├── __init__.py             [  17]
│   │   ├── validation.py           [1491]  # ClinicalValidator, FDAPathway (510k)
│   │   ├── trials.py                       # Multi-institution trial management
│   │   └── endpoints.py                    # REST API for clinical trials
│   │
│   ├── tracking/                           # User feedback & longitudinal
│   │   ├── __init__.py             [  47]
│   │   ├── feedback.py             [ 736]  # FeedbackCollector, ModelRetrainer
│   │   └── longitudinal.py         [1185]  # TelomereTracker, trend analysis
│   │
│   ├── platform/                           # Platform extensions
│   │   ├── __init__.py             [  19]
│   │   ├── federated.py            [1139]  # Federated learning coordinator
│   │   ├── mobile_api.py           [ 771]  # Mobile API (token, sync, push)
│   │   ├── plugin_system.py        [ 781]  # Plugin manager + plugin types
│   │   └── research_tools.py       [ 625]  # Research exporter, cohort builder
│   │
│   ├── ml/                                 # Machine learning
│   │   ├── __init__.py             [   5]
│   │   └── cnn_spot_detector.py    [1369]  # Pure-numpy UNet for spot detection
│   │
│   ├── sequencing/                         # Sequence-based telomere analysis
│   │   ├── __init__.py             [   1]
│   │   └── telomere_seq.py         [ 314]  # TTAGGG counting from BAM/FASTQ
│   │
│   ├── analysis/                           # Statistical analysis
│   │   ├── __init__.py             [   1]
│   │   └── statistics.py           [ 110]  # Per-cell/sample stats, DataFrame export
│   │
│   ├── visualisation/                      # Plotting and visualization
│   │   ├── __init__.py             [   1]
│   │   └── plots.py                [ 463]  # Overlays, histograms, heatmaps, galleries
│   │
│   ├── data/                               # Data management
│   │   ├── __init__.py             [  25]
│   │   ├── benchmarks.py           [ 737]  # BenchmarkSuite for validation
│   │   ├── datasets.py             [ 748]  # DatasetManager, auto-downloading
│   │   ├── training.py                     # Training dataset generation & loading
│   │   └── json/                           # 43 JSON data files (see Section 8)
│   │       ├── food_database.json          [359K, 650 foods]
│   │       ├── builtin_variant_db.json     [162K, 560 SNP variants]
│   │       ├── geographic_profiles.json    [ 65K, 35 regions]
│   │       ├── country_profiles.json       [ 76K, 30 countries]
│   │       └── ... (39 more files)
│   │
│   └── models/
│       └── __init__.py             [   0]
│
├── scripts/                                   # Utility scripts
│   ├── run_benchmarks.py                      # Benchmark CLI runner
│   └── generate_training_data.py              # Training dataset generator
│
├── benchmark_results/                         # Published benchmark outputs
│   ├── results.json                           # Machine-readable metrics
│   └── BENCHMARKS.md                          # Human-readable report
│
├── tests/                                  # 520 tests in 10 files
│   ├── __init__.py
│   ├── test_agents.py              [1067]  # 54 tests — multi-agent system
│   ├── test_disease_risk.py        [ 614]  # 54 tests — disease risk prediction
│   ├── test_health_checkup.py      [ 905]  # 99 tests — health checkup analyzer
│   ├── test_nutrition.py           [ 633]  # 51 tests — nutrigenomics engine
│   ├── test_diet_restrictions.py   [ 350]  # 15 tests — dietary restriction filtering
│   ├── test_diet_variety.py        [ 653]  # 25 tests — meal plan variety
│   ├── test_pipeline.py            [ 181]  # 14 tests — telomere analysis pipeline
│   ├── test_synthetic.py           [ 123]  # 16 tests — synthetic image generation
│   └── test_webapp.py              [ 322]  # 19 tests — FastAPI endpoints
│
└── android/                                # Android app (Jetpack Compose)
    ├── build.gradle.kts                    # Root Gradle config
    ├── settings.gradle.kts                 # Single :app module
    ├── gradle/libs.versions.toml           # Version catalog (106 lines)
    └── app/
        ├── build.gradle.kts                # compileSdk 34, minSdk 26
        └── src/main/
            ├── AndroidManifest.xml
            └── java/com/teloscopy/app/     # 22 Kotlin files (see Section 7)

└── ios/                                    # iOS app (SwiftUI)
    └── Teloscopy/
        ├── TeloscopyApp.swift              # App entry point with TabView
        ├── Views/                          # 5 SwiftUI view files
        ├── Models/                         # Data models
        ├── Services/                       # API + sync services
        ├── Assets.xcassets/                # Asset catalog
        └── Info.plist                      # App configuration
```

### 3.2 Dependencies

#### Core Dependencies
```
numpy>=1.26    scipy>=1.11    pandas>=2.1    scikit-image>=0.22
opencv-python-headless>=4.8    tifffile>=2023.7    matplotlib>=3.8
seaborn>=0.13    click>=8.1    rich>=13.0
```

#### Optional Dependency Groups
| Group | Packages | Purpose |
|-------|----------|---------|
| `cellpose` | cellpose>=3.0 | Advanced chromosome segmentation |
| `sequencing` | biopython>=1.83, pysam>=0.22 | BAM/FASTQ telomere analysis |
| `interactive` | plotly>=5.18, napari>=0.4 | Interactive visualization |
| `webapp` | fastapi>=0.110, uvicorn>=0.27, python-multipart, jinja2>=3.1 | Web server |
| `dev` | pytest>=8.0, pytest-cov, pytest-asyncio, httpx, ruff>=0.3 | Development |

### 3.3 Dependency Graph

```
webapp.app ─────────────────────────────────────────────────────────
    ├── facial.image_classifier ──── classify_image()
    ├── facial.predictor ─────────── analyze_face()
    │       ├── loads: population_snp_frequencies.json
    │       ├── loads: snp_alleles.json, snp_genomic_context.json
    │       ├── loads: pharmacogenomic_map.json
    │       └── loads: mtdna_haplogroup_priors.json
    ├── genomics.disease_risk ────── DiseasePredictor
    │       ├── loads: builtin_variant_db.json (560 variants)
    │       ├── loads: baseline_incidence.json (125 conditions)
    │       ├── loads: onset_ranges.json, preventability_scores.json
    │       ├── loads: telomere_risk_modifiers.json
    │       └── loads: screening_recommendations.json
    ├── nutrition.diet_advisor ───── DietAdvisor
    │       ├── loads: food_database.json (650 foods)
    │       ├── loads: nutrigenomics_database.json (110 mappings)
    │       ├── loads: geographic_profiles.json (35 regions)
    │       ├── loads: dietary_filter_keywords.json
    │       ├── loads: telomere_protective_nutrients.json
    │       └── loads: nutrient_micro_keys.json, restriction_excluded_groups.json, priority_rank.json
    ├── nutrition.regional_diets ─── resolve_region()
    │       ├── loads: frontend_region_map.json, region_countries.json
    │       ├── loads: country_profiles.json (30 countries)
    │       └── loads: country_states.json, country_region_override.json, state_region_override.json
    ├── webapp.health_checkup ────── HealthCheckupAnalyzer
    │       ├── loads: blood_reference_ranges.json, sex_specific_overrides.json
    │       ├── loads: urine_reference_ranges.json
    │       ├── loads: condition_rules.json (24 rules)
    │       ├── loads: abdomen_patterns.json, category_weights.json
    │       └── loads: condition_risk_mapping.json
    └── webapp.report_parser ─────── parse_lab_report()
            └── loads: parameter_aliases.json (403 aliases)

agents.orchestrator
    ├── agents.image_agent ──── telomere.pipeline (9-step analysis)
    ├── agents.genomics_agent ── genomics.disease_risk
    ├── agents.nutrition_agent ── nutrition.diet_advisor
    ├── agents.improvement_agent ── analysis.statistics
    └── agents.report_agent ──── visualisation.plots

nutrition.health_checkup (standalone, 2651 lines)
    ├── loads: parameter_metadata.json
    ├── loads: urine_ranges_nutrition.json
    ├── loads: abdomen_patterns_nutrition.json
    ├── loads: condition_advice.json
    └── loads: abdomen_advice.json
```

---

## 4. Multi-Agent System

### 4.1 Agent Architecture

Each agent follows the **Actor Model** pattern — autonomous entities that communicate via asynchronous message passing through `asyncio.Queue`.

```
┌──────────────────────────────────────────────────┐
│                   BaseAgent (ABC)                  │
│                                                    │
│  ┌─────────────┐  ┌────────────────────────────┐ │
│  │ Message      │  │ State Machine              │ │
│  │ Queue        │  │ IDLE → RUNNING → WAITING   │ │
│  │ (asyncio)    │  │ → COMPLETED / ERROR        │ │
│  └──────┬──────┘  └────────────────────────────┘ │
│         │                                         │
│  ┌──────▼──────┐  Fields:                         │
│  │ handle_     │  - name: str                     │
│  │ message()   │  - state: AgentState (Enum)      │
│  └─────────────┘  - _inbox: asyncio.Queue         │
│  ┌─────────────┐  - _router: _MessageRouter       │
│  │ send_       │  - max_retries: 3                │
│  │ message()   │  - timeout: 300s                 │
│  └─────────────┘                                  │
│                                                    │
│  AgentMessage dataclass:                          │
│  - sender, receiver, action, payload              │
│  - correlation_id (UUID), timestamp               │
│  - message_type: request|response|event           │
└──────────────────────────────────────────────────┘
```

### 4.2 Agent Registry

| Agent | Class | Lines | Role | Key Methods |
|-------|-------|-------|------|-------------|
| **Orchestrator** | `OrchestratorAgent` | 535 | Central coordinator | `process_full_analysis()`, workflow management, error recovery |
| **Image Analysis** | `ImageAnalysisAgent` | 381 | Image processing | `analyze_image()`, `validate_results()`, `suggest_improvements()` |
| **Genomics** | `GenomicsAgent` | 468 | Genetic analysis | `assess_risk()`, `project_timeline()`, `prevention_recommendations()` |
| **Nutrition** | `NutritionAgent` | 2,340 | Diet planning | `generate_diet_plan()`, `protective_foods()`, `adapt_restrictions()` |
| **Improvement** | `ContinuousImprovementAgent` | 616 | Self-optimization | `evaluate_quality()`, `track_metrics()`, `parameter_tuning()`, `auto_tune()` |
| **Report** | `ReportAgent` | 529 | Report generation | `generate_full_report()`, `format_html()`, `format_json()` |

### 4.3 Message Flow — Full Analysis

```
User uploads image + profile
        │
        ▼
  OrchestratorAgent
        │
        ├──► ImageAgent.analyze_image()
        │         │  Preprocessing → Segmentation → Detection → Quantification
        │         ▼
        │    {telomere_results, spots, chromosomes, statistics}
        │
        ├──► GenomicsAgent.assess_risk(telomere_data + SNP variants)
        │         │  Polygenic risk scores + telomere-disease correlation
        │         ▼
        │    {disease_risks[], timeline, actionable_insights}
        │
        ├──► NutritionAgent.generate_diet_plan(risks + region + restrictions)
        │         │  Nutrigenomics mapping → geographic foods → meal plans
        │         ▼
        │    {recommendations[], meal_plans[]}
        │
        ├──► ReportAgent.generate_full_report(all_results)
        │         │  HTML + JSON + CSV + visualizations
        │         ▼
        │    {report_html, report_json, csv_data}
        │
        └──► ImprovementAgent.track_metrics(results)
                  │  Quality scoring + parameter suggestions
                  ▼
             {quality_score, improvement_suggestions[]}
```

### 4.4 Continuous Improvement Loop

The ImprovementAgent runs after every analysis:

1. **Collect metrics** — spot detection confidence, segmentation quality (overlap IoU), association success rate, user feedback
2. **Evaluate quality** — compare methods (LoG vs DoG vs DoH), track per-image scores, identify systematic failures
3. **Suggest tuning** — adjust blob_log thresholds, watershed min_distance, background subtraction params
4. **Auto-apply** — update default config with audit trail (requires approval)

---

## 5. Core Modules Reference

### 5.1 `teloscopy/__init__.py` (25 lines)
- **Version**: `__version__ = "2.0.0"`
- **Exports**: 15 subpackage names in `__all__`

### 5.2 `teloscopy/cli.py` (400 lines)
Click-based CLI with 4 commands:

| Command | Arguments | Purpose |
|---------|-----------|---------|
| `analyze` | `IMAGE_PATH`, `--config`, `--output`, `--method`, `--save-overlay` | Single qFISH image analysis |
| `batch` | `INPUT_DIR`, `--config`, `--output`, `--pattern` | Batch process directory |
| `generate` | `--output`, `--n-images`, `--seed` | Generate synthetic test images |
| `report` | `CSV_PATH`, `--output` | Generate report from CSV data |

### 5.3 Telomere Pipeline (`telomere/`)

| Module | Lines | Key Functions |
|--------|-------|---------------|
| `preprocessing.py` | 294 | `load_image()`, `subtract_background()`, `denoise()`, `preprocess()` |
| `segmentation.py` | 288 | `segment_otsu_watershed()`, `segment_cellpose()`, `segment()`, `get_chromosome_properties()` |
| `spot_detection.py` | 319 | `detect_spots_log()`, `detect_spots_dog()`, `detect_spots_doh()`, `detect_spots()`, `filter_spots()` |
| `association.py` | 207 | `associate_spots_to_chromosomes()`, `summarize_associations()` |
| `quantification.py` | 387 | `measure_spot_intensity()`, `quantify_all_spots()`, `Calibration` class (linear/polynomial) |
| `pipeline.py` | 672 | `analyze_image()` (9-step orchestrator), `analyze_batch()`, `get_default_config()` |
| `synthetic.py` | 474 | `generate_metaphase_spread()`, `generate_chromosome()`, `generate_telomere_spot()`, `save_synthetic_image()` |

### 5.4 Genomics (`genomics/disease_risk.py`, 969 lines)

| Class/Constant | Type | Purpose |
|----------------|------|---------|
| `GeneticVariant` | frozen dataclass | rsid, gene, chromosome, position, risk_allele, effect_size, condition, evidence_level |
| `DiseaseRisk` | dataclass | condition, category, lifetime_risk_pct, relative_risk, confidence, contributing_variants |
| `RiskProfile` | class | Container with `top_risks(n)`, `filter_by_category()`, `filter_by_confidence()`, `summary() → DataFrame` |
| `DiseasePredictor` | class | Main entry: `predict_from_variants()`, `predict_from_telomere_data()`, `predict_from_image_analysis()`, `calculate_polygenic_risk()`, `project_risk_over_time()`, `get_actionable_insights()` |
| `BUILTIN_VARIANT_DB` | list[GeneticVariant] | 560 SNP-disease associations loaded from JSON |
| `BASELINE_INCIDENCE` | dict | 125 condition baselines by sex |

**Loads 6 JSON files**: `builtin_variant_db.json`, `baseline_incidence.json`, `onset_ranges.json`, `preventability_scores.json`, `telomere_risk_modifiers.json`, `screening_recommendations.json`

### 5.5 Nutrition (`nutrition/`)

#### `diet_advisor.py` (1,377 lines)

| Class | Key Methods |
|-------|-------------|
| `DietAdvisor` | `calculate_nutrient_needs()`, `generate_recommendations()`, `create_meal_plan()`, `get_region_specific_foods()`, `get_telomere_protective_diet()`, `adapt_to_restrictions()` |

Supporting dataclasses: `NutrientNeed`, `FoodItem`, `MealPlan`, `DietaryRecommendation`, `GeographicProfile`

**Loads 8 JSON files**: `food_database.json`, `nutrigenomics_database.json`, `geographic_profiles.json`, `dietary_filter_keywords.json`, `telomere_protective_nutrients.json`, `nutrient_micro_keys.json`, `restriction_excluded_groups.json`, `priority_rank.json`

#### `regional_diets.py` (169 lines)

| Function | Purpose |
|----------|---------|
| `resolve_region(frontend_region, country, state)` | Resolves UI input → internal region ID |
| `get_country_profile(country) → CountryProfile` | Full country dietary profile |
| `get_state_profile(country, state) → StateProfile` | State-level dietary detail |
| `list_countries_for_region()` / `list_states_for_country()` | Dropdown population |

**Loads 6 JSON files**: `frontend_region_map.json`, `region_countries.json`, `country_states.json`, `country_profiles.json`, `country_region_override.json`, `state_region_override.json`

#### `i18n.py` (278 lines)

`DietTranslator` class supporting 10 languages: English, Spanish, French, German, Chinese, Hindi, Arabic, Portuguese, Japanese, Korean.

Methods: `translate_recommendation()`, `translate_meal_plan()`, `translate_full_report()`, `available_languages()`

**Loads 4 JSON files**: `i18n_labels.json`, `i18n_day_names.json`, `i18n_food_translations.json`, `i18n_nutrient_translations.json`

#### `health_checkup.py` (2,651 lines — largest file)

Full health checkup analysis from blood/urine results + abdomen scans.

Main entry: `process_health_checkup(blood_data, urine_data, abdomen_text, age, sex) → HealthCheckupResult`

24 private condition detector functions covering: iron-deficiency anemia, macrocytic anemia, dyslipidemia, diabetes/prediabetes, hypothyroidism, hyperthyroidism, vitamin D/B12/A/E deficiency, folate deficiency, liver stress, fatty liver, kidney impairment, hyperuricemia, inflammation, electrolyte imbalance, proteinuria, insulin resistance, prehypertension, cardiac risk, zinc deficiency, calcium/magnesium deficiency.

**Loads 5 JSON files**: `parameter_metadata.json`, `urine_ranges_nutrition.json`, `abdomen_patterns_nutrition.json`, `condition_advice.json`, `abdomen_advice.json`

### 5.6 Facial-Genomic Analysis (`facial/`)

#### `predictor.py` (1,639 lines)

Main function: `analyze_face(image_path, chronological_age, sex) → FacialGenomicProfile`

12 dataclasses: `FacialMeasurements`, `AncestryEstimate`, `PredictedVariant`, `ReconstructedSequence`, `ReconstructedDNA`, `PharmacogenomicPrediction`, `FacialHealthScreening`, `DermatologicalAnalysis`, `ConditionScreening`, `AncestryDerivedPredictions`, `FacialGenomicProfile`

Analysis pipeline:
1. Face detection and measurement extraction (OpenCV)
2. Biological age estimation from facial features
3. Ancestry estimation (6-population model)
4. 59 SNP variant prediction from ancestry + facial markers
5. DNA sequence reconstruction with flanking regions
6. Pharmacogenomic phenotype prediction (8 genes)
7. Health screening (BMI, anemia, cardiovascular, thyroid)
8. Dermatological analysis (rosacea, melasma, acne, UV damage)
9. Condition screening (10+ conditions)
10. Ancestry-derived predictions (haplogroup, lactose, alcohol flush)

**Loads 5 JSON files**: `population_snp_frequencies.json`, `snp_alleles.json`, `snp_genomic_context.json`, `pharmacogenomic_map.json`, `mtdna_haplogroup_priors.json`

#### `image_classifier.py` (268 lines)

`classify_image(image_path) → ClassificationResult`

Returns: `image_type` (FISH_MICROSCOPY / FACE_PHOTO / UNKNOWN_PHOTO), `confidence`, `face_detected`, `is_fluorescence`

### 5.7 Other Subpackages

| Subpackage | Key Classes/Functions | Lines | Purpose |
|------------|----------------------|-------|---------|
| `integrations/fhir.py` | `FHIRExporter`, `HIPAACompliance` | 1,855 | HL7 FHIR resource export, HIPAA audit |
| `integrations/genotype_import.py` | `GenotypeImporter`, `GenotypeData` | 809 | Import 23andMe/AncestryDNA/VCF files |
| `integrations/llm_reports.py` | `ReportGenerator`, `OllamaClient`, `OpenAIClient` | 772 | LLM-powered narrative reports |
| `integrations/wgs.py` | `WGSAnalyzer`, `WGSData` | 1,518 | Whole genome sequencing analysis |
| `clinical/validation.py` | `ClinicalValidator`, `FDAPathway` | 1,491 | Clinical validation, FDA 510(k) assessment |
| `tracking/feedback.py` | `FeedbackCollector`, `ModelRetrainer` | 736 | User feedback loop, model improvement |
| `tracking/longitudinal.py` | `TelomereTracker`, `PatientHistory` | 1,185 | Longitudinal telomere tracking |
| `platform/federated.py` | `FederatedLearningCoordinator` | 1,139 | Multi-institution federated learning |
| `platform/mobile_api.py` | `MobileAPIController`, `TokenManager` | 771 | Mobile SDK (auth, sync, push) |
| `platform/plugin_system.py` | `PluginManager`, `PluginBase` (4 types) | 781 | Plugin architecture |
| `platform/research_tools.py` | `ResearchExporter`, `CohortBuilder`, `CitationGenerator` | 625 | Research collaboration tools |
| `ml/cnn_spot_detector.py` | `CNNSpotDetector`, `_UNet` | 1,369 | Pure-numpy CNN for spot detection |
| `sequencing/telomere_seq.py` | `count_telomere_repeats()`, `estimate_from_bam()` | 314 | TTAGGG counting from BAM/FASTQ |
| `analysis/statistics.py` | `compute_cell_statistics()`, `create_results_dataframe()` | 110 | Per-cell/sample statistics |
| `visualisation/plots.py` | `plot_telomere_overlay()`, `plot_intensity_histogram()`, etc. | 463 | 5 plot types |
| `data/benchmarks.py` | `BenchmarkSuite` | 737 | Spot detection/segmentation benchmarks |
| `data/datasets.py` | `DatasetManager` | 748 | Dataset catalog + auto-downloading |

---

## 6. Web Application

### 6.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          BROWSER                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  index.html (3,851 lines)                              │  │
│  │  ├── Hero landing section                              │  │
│  │  ├── Upload & Profile form (image + 8 profile fields)  │  │
│  │  ├── Nutrition Planner (standalone form)                │  │
│  │  ├── Health Checkup (5-tab: profile/upload/blood/urine/│  │
│  │  │   abdomen — 62 blood + 13 urine parameters)         │  │
│  │  ├── Results (6-tab: telomere/facial/DNA/risk/diet/     │  │
│  │  │   charts — with Chart.js visualizations)             │  │
│  │  └── Agents section (multi-agent status)                │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  dashboard.html (1,520 lines)                          │  │
│  │  ├── Metrics row (5 cards)                             │  │
│  │  ├── Agent cards grid (dynamic)                        │  │
│  │  ├── Performance grid (CPU/Mem/Queue/Response)         │  │
│  │  ├── Jobs table + Activity log                         │  │
│  │  └── Improvement suggestions                           │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │ HTTP/REST (20 endpoints)              │
└───────────────────────┼──────────────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────────────┐
│  FastAPI Server       ▼                                       │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Infrastructure                                      │     │
│  │  ├── Rate Limiter (in-memory sliding window per IP) │     │
│  │  ├── Security Headers middleware (CSP, XSS, etc.)   │     │
│  │  ├── Request ID middleware (UUID + timing)           │     │
│  │  ├── CORS (configurable, * in dev)                  │     │
│  │  └── Upload limits: 50 MiB (images), 20 MiB (reports)   │
│  ├─────────────────────────────────────────────────────┤     │
│  │  Pipeline Singletons (initialized at startup)       │     │
│  │  ├── DiseasePredictor                               │     │
│  │  ├── DietAdvisor                                    │     │
│  │  └── HealthCheckupAnalyzer                          │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │  Job Store (in-memory dict[str, JobStatus])         │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Pydantic Models (45 models in `models.py`, 914 lines)

#### Enumerations
| Enum | Values |
|------|--------|
| `Sex` | male, female, other |
| `JobStatusEnum` | pending, running, completed, failed |
| `RiskLevel` | low, moderate, high, very_high |
| `AgentStatusEnum` | idle, busy, error, offline |

#### Key Request Models
| Model | Key Fields |
|-------|------------|
| `UserProfile` | age (1-150), sex, region, country?, state?, dietary_restrictions[], known_variants[] |
| `ProfileAnalysisRequest` | UserProfile fields + include_nutrition, include_disease_risk toggles |
| `DiseaseRiskRequest` | known_variants[], telomere_length?, age, sex, region |
| `DietPlanRequest` | profile + disease_risks[], meal_plan_days (1-30), calorie_target (800-5000) |
| `NutritionRequest` | profile + health_conditions[], calorie_target, meal_plan_days |
| `HealthCheckupRequest` | profile + BloodTestPanel (62 fields) + UrineTestPanel (13 fields) + abdomen_scan_notes |
| `BloodTestPanel` | 62 optional float fields: CBC(14), Lipid(6), Liver(9), Kidney(5), Diabetes(5), Thyroid(5), Vitamins(5), Minerals(7), Electrolytes(4), Inflammation(3) |

#### Key Response Models
| Model | Key Fields |
|-------|------------|
| `AnalysisResponse` | job_id, image_type, telomere_results, disease_risks[], diet_recommendations, facial_analysis? |
| `FacialAnalysisResult` | bio_age, telomere_length_kb, ancestry, predicted_variants[], reconstructed_dna, pharmacogenomic_predictions[], health_screening, dermatological_analysis, condition_screenings[] |
| `HealthCheckupResponse` | lab_results[], overall_health_score (0-100), health_score_breakdown, findings[], abdomen_findings[], diet_recommendation |
| `ReportParsePreview` | extracted_blood_tests, extracted_urine_tests, extracted_abdomen_notes, confidence, unrecognized_lines[] |

### 6.3 Report Parser (`report_parser.py`, 567 lines)

Extracts lab values from uploaded PDF/image/text reports.

| Feature | Detail |
|---------|--------|
| **PDF parsing** | PyMuPDF (fitz) → pdfplumber → pypdf/PyPDF2 (fallback chain) |
| **Image OCR** | pytesseract + Pillow |
| **Text parsing** | UTF-8/Latin-1 decode |
| **Alias system** | 403 aliases across 75 parameters (loaded from `parameter_aliases.json`) |
| **3 regex strategies** | 1) Pipe-separated tables, 2) Colon/equals patterns, 3) Whitespace-separated |
| **Section detection** | Urine section markers, abdomen section extraction |
| **Confidence score** | Formula: `min(ratio × 0.5 + volume_score + key_bonus + abdomen_bonus, 1.0)` |

### 6.4 Webapp Health Checkup (`webapp/health_checkup.py`, 1,038 lines)

The `HealthCheckupAnalyzer` class orchestrates lab interpretation:

1. **`_interpret_labs()`** — classifies each value as low/normal/high/critical using age/sex-adjusted ranges
2. **`_detect_conditions()`** — runs 24 `_check_*` methods from `condition_rules.json`
3. **`_parse_abdomen()`** — regex-scans abdomen scan notes against `abdomen_patterns.json`
4. **`_compute_health_score()`** — weighted category scoring (0-100) via `category_weights.json`
5. **`_generate_diet()`** — calls DietAdvisor with detected conditions mapped via `condition_risk_mapping.json`

**24 Condition Detectors:**

| Condition | Key Thresholds |
|-----------|---------------|
| Prediabetes | FG 100-125, HbA1c 5.7-6.4, PP glucose 140-199 |
| Diabetes | FG ≥126, HbA1c ≥6.5, PP glucose ≥200 |
| Dyslipidemia | TC >200, LDL >100, HDL <40, TG >150 |
| Liver stress | ALT >56, AST >40, GGT >45, ALP >147 |
| Fatty liver | Liver marker + metabolic marker (TG>150 or GGT>45) |
| Kidney impairment | Creatinine >1.3♂/1.1♀, eGFR <90, BUN >20 |
| Hyperuricemia | Uric acid >7.2♂/6.0♀ |
| Hypothyroidism | TSH >4.0, Free T4 <0.8, Free T3 <2.0 |
| Hyperthyroidism | TSH <0.4, Free T4 >1.8, Free T3 >4.4 |
| Anemia | Hb <13♂/12♀, Hct <38♂/36♀, MCV <80 |
| Vitamin D deficiency | <20 deficient, <10 severe, 20-29 insufficient |
| B12 deficiency | <200 deficient, <150 severe |
| Iron deficiency | Iron <65♂/50♀, Ferritin <20♂/12♀, Transferrin sat <20% |
| Inflammation | CRP >3.0, ESR >20, Homocysteine >15 |
| Electrolyte imbalance | Na <136/>145, K <3.5/>5.0, Cl <98/>106 |
| Proteinuria | Urine protein >14 |
| Insulin resistance | Fasting insulin >25, HOMA-IR >2.5 |
| Folate deficiency | Folate <3.0, or Homocysteine >15 + Folate <5.0 |
| Prehypertension | Na >145, K <3.5, Homocysteine >12 |
| Cardiac risk | CRP>3 + lipid risk, or Homocysteine >15 |
| Vitamin A deficiency | Vit A <20 |
| Vitamin E deficiency | Vit E <5.0 |
| Zinc deficiency | Zinc <70 |
| Ca/Mg deficiency | Ca <8.5, Mg <1.7 |

---

## 7. Android Application

### 7.1 Architecture Overview

**MVVM + Repository pattern** with **Hilt dependency injection**:

```
UI (Compose Screens) → ViewModels (StateFlow) → Repository (Result<T>) → Retrofit API → FastAPI Backend
```

| Property | Value |
|----------|-------|
| Package | `com.teloscopy.app` |
| compileSdk / targetSdk | 34 |
| minSdk | 26 (Android 8.0) |
| Kotlin | 2.0.0 |
| Compose BOM | 2024.06.00 |
| Architecture | Single Activity + Compose Navigation (5 screens) |
| Theme | Dark-first (cyan/purple/green genomic palette) |

### 7.2 File Structure (22 Kotlin files, 7,718 lines)

```
com.teloscopy.app/
├── MainActivity.kt              [309]  # Single activity, modal drawer + bottom nav
├── TeloscopyApp.kt              [ 12]  # @HiltAndroidApp entry point
├── data/
│   ├── api/
│   │   ├── ApiModels.kt         [354]  # 25 Moshi data classes matching server models
│   │   └── TeloscopyApi.kt      [121]  # Retrofit interface (10 endpoints)
│   └── repository/
│       └── AnalysisRepository.kt[240]  # Result<T> wrapper, safeApiCall()
├── di/
│   └── AppModule.kt             [ 88]  # Hilt: Moshi, OkHttp, Retrofit, DataStore
├── ui/
│   ├── components/
│   │   ├── DiseaseRiskCard.kt   [206]  # Expandable risk card with probability bar
│   │   ├── MealPlanCard.kt      [147]  # Expandable day card (meals + snacks)
│   │   ├── SectionHeader.kt     [ 63]  # Icon + title + colored divider
│   │   ├── ShimmerEffect.kt     [171]  # Loading skeleton animations
│   │   └── StatCard.kt          [ 78]  # Compact metric card
│   ├── navigation/
│   │   └── NavGraph.kt          [204]  # 5 routes: Home, Analysis, Results, Profile, Settings
│   ├── screens/
│   │   ├── HomeScreen.kt        [707]  # Dashboard with quick stats + actions
│   │   ├── AnalysisScreen.kt    [830]  # Camera/gallery upload + profile form
│   │   ├── ResultsScreen.kt     [1845] # 8+ result sections (largest file)
│   │   ├── ProfileAnalysisScreen[773]  # Profile-only analysis form + results
│   │   └── SettingsScreen.kt    [798]  # Server config, appearance, about
│   └── theme/
│       ├── Color.kt             [ 66]  # Dark palette + risk-level colors
│       ├── Theme.kt             [ 78]  # Material3 dark theme
│       └── Type.kt              [ 45]  # Custom typography
└── viewmodel/
    ├── AnalysisViewModel.kt     [361]  # Image analysis lifecycle (polling every 2s)
    └── ProfileViewModel.kt      [222]  # Profile/disease-risk/nutrition analysis
```

### 7.3 Android API Interface

| Endpoint | HTTP | Request | Response |
|----------|------|---------|----------|
| `api/analyze` | POST | Multipart (image + profile) | `JobStatus` (202) |
| `api/status/{job_id}` | GET | — | `JobStatus` |
| `api/results/{job_id}` | GET | — | `AnalysisResponse` |
| `api/profile-analysis` | POST | JSON `ProfileAnalysisRequest` | `ProfileAnalysisResponse` |
| `api/disease-risk` | POST | JSON `DiseaseRiskRequest` | `DiseaseRiskResponse` |
| `api/nutrition` | POST | JSON `NutritionRequest` | `NutritionResponse` |
| `api/validate-image` | POST | Multipart file | `ImageValidationResponse` |
| `api/health` | GET | — | `HealthResponse` |
| `api/health-checkup/parse-report` | POST | Multipart file | `ReportParsePreview` |
| `api/health-checkup/upload` | POST | Multipart (file + profile) | `HealthCheckupResponse` |

### 7.4 Key Dependencies

Compose + Material3, Navigation Compose, Hilt (DI), Retrofit + OkHttp + Moshi (networking/JSON), CameraX, Coil (image loading), Vico Charts, DataStore Preferences, Accompanist (permissions)

---

## 8. Data Layer — JSON Data Files

### 8.1 Overview

43 JSON data files totaling ~46,400 lines in `src/teloscopy/data/json/`.

### 8.2 Complete Catalog

#### Genomics / Disease Risk (7 files)

| File | Size | Records | Purpose |
|------|------|---------|---------|
| `builtin_variant_db.json` | 162K | ~560 variants | SNP-disease associations (rsid, gene, effect_size, evidence_level) |
| `baseline_incidence.json` | 8.2K | ~125 conditions | Population incidence rates by sex (per 100K person-years) |
| `onset_ranges.json` | 11K | ~178 conditions | Typical age-of-onset windows (min, max) |
| `preventability_scores.json` | 5.5K | 178 conditions | How preventable each condition is (0.0–1.0) |
| `telomere_risk_modifiers.json` | 979B | 33 conditions | Telomere shortening impact factors (1.04–1.50) |
| `screening_recommendations.json` | 11K | 24 categories | Clinical screening recommendations (action, frequency, detail) |
| `condition_risk_mapping.json` | 1.1K | 23 mappings | Maps blood-test conditions → disease risk engine names |

**Loaded by**: `genomics/disease_risk.py`, `webapp/health_checkup.py`

#### Nutrition / Diet (10 files)

| File | Size | Records | Purpose |
|------|------|---------|---------|
| `food_database.json` | 359K | ~650 foods | Macros, micros, food groups, regional tags |
| `nutrigenomics_database.json` | 41K | ~110 entries | Gene-nutrient interactions (rsid → nutrient → daily target) |
| `geographic_profiles.json` | 65K | ~35 regions | Regional food profiles (grains, proteins, vegetables, dishes) |
| `dietary_filter_keywords.json` | 5.1K | ~250 keywords | Food filtering for 8 dietary restrictions |
| `telomere_protective_nutrients.json` | 1.3K | 9 nutrients | Telomere-protective nutrients with confidence scores |
| `nutrient_micro_keys.json` | 766B | 14 categories | Nutrient category → micronutrient field name mappings |
| `restriction_excluded_groups.json` | 348B | 7 restrictions | Food groups excluded by each restriction type |
| `priority_rank.json` | 61B | 4 levels | Priority ranking: critical=4, high=3, moderate=2, low=1 |
| `category_weights.json` | 243B | 11 categories | Health score weights by lab category |
| `sex_specific_overrides.json` | 1.1K | 8 parameters | Sex-specific lab reference range overrides |

**Loaded by**: `nutrition/diet_advisor.py`, `webapp/health_checkup.py`

#### Geographic / Regional (7 files)

| File | Size | Records | Purpose |
|------|------|---------|---------|
| `country_profiles.json` | 76K | ~30 countries | Full dietary profiles with state-level detail (India: 16 states) |
| `frontend_region_map.json` | 1.6K | 46 mappings | Normalizes UI region labels → internal IDs |
| `region_countries.json` | 854B | 15 regions | Region → country list (for dropdowns) |
| `country_states.json` | 793B | 8 countries | Country → state list (for cascading dropdowns) |
| `country_region_override.json` | 1.9K | 64 mappings | Country → sub-regional dietary zone override |
| `state_region_override.json` | 3.9K | 7 countries | State-level sub-regional overrides (e.g., Kerala → india_kerala) |
| `parameter_metadata.json` | 6.7K | ~85 params | Display names and categories for lab parameters |

**Loaded by**: `nutrition/regional_diets.py`, `nutrition/health_checkup.py`

#### Blood / Urine Lab Ranges (5 files)

| File | Size | Records | Purpose |
|------|------|---------|---------|
| `blood_reference_ranges.json` | 9.0K | 57 params | Default blood test reference ranges |
| `blood_ranges_by_age_sex.json` | 106K | ~65 params × 8 groups | Age/sex-stratified ranges (male/female × 4 age brackets) |
| `urine_reference_ranges.json` | 1.8K | 13 params | Default urine reference ranges |
| `urine_ranges_nutrition.json` | 2.0K | 14 params | Extended urine ranges with critical thresholds |
| `parameter_aliases.json` | 7.8K | 75 params, 403 aliases | Lab report name normalization (e.g., "hb"/"hgb" → "hemoglobin") |

**Loaded by**: `webapp/health_checkup.py`, `nutrition/health_checkup.py`, `webapp/report_parser.py`

#### Health Checkup Rules (4 files)

| File | Size | Records | Purpose |
|------|------|---------|---------|
| `condition_rules.json` | 15K | 24 rules | Condition detection rules (check method, dietary impact, foods) |
| `condition_advice.json` | 4.4K | 24 conditions | Actionable dietary advice per condition |
| `abdomen_patterns.json` | 3.6K | 8 patterns | Regex patterns for abdomen scan findings |
| `abdomen_patterns_nutrition.json` | 6.4K | 12 patterns | Extended abdomen patterns for nutrition module |
| `abdomen_advice.json` | 961B | 9 conditions | Dietary advice for abdominal findings |

**Loaded by**: `webapp/health_checkup.py`, `nutrition/health_checkup.py`

#### Facial / Pharmacogenomics (5 files)

| File | Size | Records | Purpose |
|------|------|---------|---------|
| `population_snp_frequencies.json` | 32K | ~60 SNPs × 6 populations | Cross-population allele frequencies |
| `snp_alleles.json` | 2.3K | ~60 SNPs | Allele pairs per SNP (risk + protective) |
| `snp_genomic_context.json` | 6.4K | ~60 SNPs | Chromosome, position, 25bp flanking sequences (GRCh38) |
| `pharmacogenomic_map.json` | 3.5K | 8 pharmacogenes | Drug metabolism predictions (CYP2C19, CYP2D6, VKORC1, etc.) |
| `mtdna_haplogroup_priors.json` | 887B | 6 populations | Mitochondrial haplogroup prior probabilities |

**Loaded by**: `facial/predictor.py`

#### Internationalization (4 files)

| File | Size | Languages | Purpose |
|------|------|-----------|---------|
| `i18n_labels.json` | 3.9K | 10 | Section titles + disclaimer translations |
| `i18n_day_names.json` | 1.4K | 10 | Localized day names for meal plans |
| `i18n_food_translations.json` | 7.3K | 9 | ~31 food names in 9 languages |
| `i18n_nutrient_translations.json` | 2.7K | 10 | 10 nutrient names in 10 languages |

**Loaded by**: `nutrition/i18n.py`

### 8.3 Module → JSON Loading Map

| Python Module | JSON Files Loaded |
|---------------|-------------------|
| `genomics/disease_risk.py` | builtin_variant_db, baseline_incidence, onset_ranges, preventability_scores, telomere_risk_modifiers, screening_recommendations |
| `nutrition/diet_advisor.py` | food_database, nutrigenomics_database, geographic_profiles, dietary_filter_keywords, telomere_protective_nutrients, nutrient_micro_keys, restriction_excluded_groups, priority_rank |
| `nutrition/regional_diets.py` | frontend_region_map, region_countries, country_states, country_profiles, country_region_override, state_region_override |
| `nutrition/i18n.py` | i18n_labels, i18n_day_names, i18n_food_translations, i18n_nutrient_translations |
| `nutrition/health_checkup.py` | parameter_metadata, urine_ranges_nutrition, abdomen_patterns_nutrition, condition_advice, abdomen_advice |
| `facial/predictor.py` | population_snp_frequencies, snp_alleles, snp_genomic_context, pharmacogenomic_map, mtdna_haplogroup_priors |
| `webapp/health_checkup.py` | blood_reference_ranges, sex_specific_overrides, urine_reference_ranges, condition_rules, abdomen_patterns, category_weights, condition_risk_mapping |
| `webapp/report_parser.py` | parameter_aliases |

---

## 9. Disease Risk Prediction Engine

### 9.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              DISEASE RISK PREDICTION ENGINE                    │
│                                                                │
│  INPUTS                                                        │
│  ├── Telomere Length Data (mean, CV, short fraction)           │
│  ├── Genetic Variants (rsID → genotype, optional)             │
│  └── Demographics (age, sex)                                   │
│                                                                │
│  DATABASE: 560 SNP-disease associations across 26 categories   │
│  ├── Evidence levels: strong (1.0), moderate (0.6), suggestive │
│  ├── Population frequencies for 6 ancestral groups             │
│  └── 125 baseline incidence rates by sex                       │
│                                                                │
│  CALCULATION                                                   │
│  ├── Multiplicative OR model for SNP combinations              │
│  ├── Telomere length risk adjustment:                          │
│  │   ├── < 10th %ile → 1.5× cancer, 1.8× CVD                │
│  │   ├── CV > 0.30 → +0.2× genomic instability              │
│  │   └── Short fraction > 15% → +0.3× accelerated aging     │
│  ├── Age-dependent baseline incidence                          │
│  ├── Sex-specific modifications                                │
│  └── Confidence = evidence_level × variant_count_factor        │
│                                                                │
│  OUTPUTS                                                       │
│  ├── RiskProfile (per-condition lifetime risk %, relative risk) │
│  ├── Year-by-year risk projection (up to 30 years)            │
│  ├── Actionable prevention recommendations                     │
│  └── Screening recommendations by category                     │
└──────────────────────────────────────────────────────────────┘
```

### 9.2 Disease Categories

| Category | Conditions | Key Genes |
|----------|-----------|-----------|
| Cardiovascular | CHD, stroke, hypertension, AFib | APOE, PCSK9, LPA, LDLR |
| Cancer | Breast, colorectal, prostate, lung | BRCA1/2, TP53, APC, MLH1 |
| Metabolic | T2D, obesity, metabolic syndrome | TCF7L2, FTO, PPARG |
| Neurological | Alzheimer's, Parkinson's | APOE-e4, LRRK2, CLU |
| Autoimmune | RA, T1D, celiac, lupus | HLA-DRB1, CTLA4, PTPN22 |
| Eye | AMD, glaucoma | CFH, ARMS2 |
| Bone | Osteoporosis | ESR1, VDR, COL1A1 |
| Blood | Hemochromatosis, sickle cell | HFE, HBB |
| Pharmacogenomics | Drug metabolism variants | CYP2C19, CYP2D6, VKORC1 |
| + 17 more | Respiratory, kidney, liver, mental health, reproductive, dermatological, GI, endocrine, etc. | Various |

---

## 10. Nutrigenomic Diet Advisor

### 10.1 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                NUTRIGENOMIC DIET ADVISOR                   │
│                                                            │
│  STEP 1: GENETIC PROFILE → NUTRIENT NEEDS                 │
│  ├── MTHFR C677T ──► Folate ↑ (leafy greens)             │
│  ├── FTO rs9939609 ─► Calorie control                     │
│  ├── LCT rs4988235 ─► Lactose: dairy/alt                  │
│  ├── CYP1A2 ────────► Caffeine: fast/slow                 │
│  ├── APOE e4 ────────► Saturated fat ↓                    │
│  ├── VDR ────────────► Vitamin D ↑                        │
│  └── + 100 more gene-nutrient mappings                     │
│                                                            │
│  STEP 2: DISEASE RISKS → PROTECTIVE DIET                   │
│  ├── Cancer risk ──────► Antioxidants, fiber               │
│  ├── CVD risk ─────────► Omega-3, low sodium               │
│  ├── Short telomeres ──► Telomere-protective foods         │
│  │                       (omega-3, folate, polyphenols)    │
│  └── 24 blood-test conditions → specific dietary mods      │
│                                                            │
│  STEP 3: GEOGRAPHIC FOOD MAPPING                           │
│  ├── 650 foods × 35 geographic regions                     │
│  ├── South India: drumstick leaves, amaranth, dosa         │
│  ├── Japan: edamame, natto, seaweed                        │
│  ├── Mediterranean: lentils, olive oil, chickpeas          │
│  └── 30 countries with state-level specificity             │
│                                                            │
│  STEP 4: MEAL PLAN GENERATION                              │
│  ├── 1–30 day plans, 800–5000 kcal                        │
│  ├── Breakfast + lunch + dinner + snacks per day           │
│  ├── Macro/micro nutrient optimization                     │
│  └── 8 dietary restrictions: vegetarian, vegan, GF,        │
│      halal, kosher, nut-free, dairy-free, pescatarian      │
│                                                            │
│  STEP 5: TRANSLATION (10 languages)                        │
│  └── DietTranslator: en/es/fr/de/zh/hi/ar/pt/ja/ko       │
└──────────────────────────────────────────────────────────┘
```

### 10.2 Geographic Coverage

| Region | Sub-Regions | Countries with State Detail |
|--------|-------------|---------------------------|
| South Asia | North/South/East/West India | India (16 states) |
| East Asia | China, Japan, Korea | China (6 provinces), Japan (4 regions) |
| Southeast Asia | Thailand, Vietnam, Indonesia | — |
| Middle East | Levant, Gulf, North Africa | — |
| Mediterranean | Greece, Italy, Spain | Italy (7 regions) |
| Northern Europe | UK, Scandinavia, Germany | — |
| Sub-Saharan Africa | West, East, Southern | Nigeria (3 regions) |
| Latin America | Mexico, Brazil, Andes | Mexico (8 states), Brazil (7 states) |
| North America | General US/Canada | USA (40+ states) |

---

## 11. Health Checkup System

Two parallel implementations:

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Nutrition Module** | `nutrition/health_checkup.py` | 2,651 | Standalone lab analysis with age/sex-stratified ranges |
| **Webapp Module** | `webapp/health_checkup.py` | 1,038 | Web-facing analyzer with rule engine + diet integration |
| **Report Parser** | `webapp/report_parser.py` | 567 | PDF/image/text lab report extraction |

### Input Paths

```
Path 1: Manual Entry
  User enters values in 5-tab form (Profile/Blood/Urine/Abdomen) → POST /api/health-checkup

Path 2: Document Upload
  User uploads lab report (PDF/image/text) → POST /api/health-checkup/parse-report
    → Preview extracted values → User confirms → POST /api/health-checkup/upload

Path 3: Combined Upload + Profile
  Upload file + profile fields in one request → POST /api/health-checkup/upload
```

### Output: `HealthCheckupResponse`
- **Lab results** with status (low/normal/high/critical) for each parameter
- **Overall health score** (0-100) with category breakdown
- **Detected conditions** (up to 24 conditions)
- **Findings** with severity, evidence, dietary impact
- **Personalized diet plan** addressing all detected conditions

---

## 12. Facial-Genomic Analysis

### Pipeline (10 steps in `facial/predictor.py`)

```
Input: Face photo + chronological_age + sex
                    │
    ┌───────────────▼───────────────────────┐
    │ 1. Face Detection (OpenCV Haar/DNN)   │
    │ 2. Facial Measurements (12 metrics)   │
    │ 3. Biological Age Estimation          │
    │ 4. Telomere Length Inference           │
    │ 5. Ancestry Estimation (6 populations)│
    │ 6. 59 SNP Variant Prediction          │
    │ 7. DNA Sequence Reconstruction        │
    │ 8. Pharmacogenomic Profiling (8 genes)│
    │ 9. Health + Dermatological Screening   │
    │ 10. Ancestry-Derived Predictions      │
    └───────────────┬───────────────────────┘
                    │
    Output: FacialGenomicProfile
    ├── estimated_biological_age, telomere_length_kb
    ├── ancestry: {european, east_asian, south_asian, african, ...}
    ├── predicted_variants: [{rsid, gene, genotype, confidence}, ...]
    ├── reconstructed_dna: {sequences[], fasta, genome_build}
    ├── pharmacogenomic_predictions: [{gene, phenotype, drugs[]}, ...]
    ├── health_screening: {bmi_category, anemia_risk, cv_indicators}
    ├── dermatological_analysis: {rosacea, melasma, photo_aging}
    ├── condition_screenings: [{condition, risk_score, markers}, ...]
    └── ancestry_derived: {haplogroup, lactose_tolerance, alcohol_flush}
```

---

## 13. Image Analysis Pipeline

### 9-Step Pipeline (telomere/)

```
Step 1: LOAD IMAGE
├── .tif/.tiff (16-bit), .png, .jpg via tifffile/OpenCV
├── Extract DAPI (channel 0) + Cy3 (channel 1)
└── Output: dapi[H,W], cy3[H,W] float64

Step 2: BACKGROUND SUBTRACTION
├── Rolling Ball (morphological opening, r=50)
├── Top-Hat (white_tophat, disk selem, r=50)
└── Gaussian (subtract blurred, sigma=50)

Step 3: DENOISING → Gaussian filter (sigma=1.0)

Step 4: CHROMOSOME SEGMENTATION
├── Otsu + Watershed (distance transform, markers, min_area=500)
└── Cellpose (optional, model_type="cyto3", auto diameter)

Step 5: TIP DETECTION
└── Per chromosome: boundary → convex hull → 2 most distant points → p/q arm tips

Step 6: SPOT DETECTION (Cy3 channel)
├── blob_log (LoG) — most accurate, slowest
├── blob_dog (DoG) — balanced
└── blob_doh (DoH) — fastest, least accurate

Step 7: SPOT-CHROMOSOME ASSOCIATION
├── KDTree from all chromosome tips
├── Nearest-tip query per spot (max_distance=15px)
└── Conflict resolution: multiple spots → same tip → keep brightest

Step 8: INTENSITY QUANTIFICATION
├── Circular aperture (r=5): sum pixel values
├── Annular background (inner=7, outer=12): median
├── Corrected intensity = aperture_sum − (bg_median × area)
└── SNR = corrected_intensity / bg_std

Step 9: CALIBRATION (optional)
├── Reference standards → linear/polynomial regression
└── Apply: length_bp = f(corrected_intensity)
```

### Quality Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| Spots per chromosome | 2-4 | 1-6 | 0 or >6 |
| Association rate | >80% | 60-80% | <60% |
| Mean SNR | >5.0 | 3.0-5.0 | <3.0 |
| CV of intensities | <0.40 | 0.40-0.60 | >0.60 |

---

## 14. API Reference

### 14.1 All REST Endpoints (20 routes)

#### HTML Pages
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Main landing page (`index.html`) |
| GET | `/upload` | Upload page (scrolls to upload section) |
| GET | `/results/{job_id}` | Results page for specific job |
| GET | `/dashboard` | Agent monitoring dashboard |
| GET | `/api/debug/templates` | Diagnostic: template/static directory listing |

#### Health & Status APIs
| Method | Path | Rate Limit | Response Model | Description |
|--------|------|-----------|----------------|-------------|
| GET | `/api/health` | 60/60s | `HealthResponse` | Liveness probe (status, version, timestamp) |
| GET | `/readiness` | — | `dict` | Readiness check for all subsystems |
| GET | `/api/agents/status` | 60/60s | `AgentSystemStatus` | Status of 4 agents, active jobs, uptime |

#### Analysis APIs
| Method | Path | Rate Limit | Request | Response | Description |
|--------|------|-----------|---------|----------|-------------|
| POST | `/api/upload` | 10/60s | Multipart file | `UploadResponse` (201) | Upload image, create pending job |
| POST | `/api/analyze` | 20/60s | Multipart (file + profile) | `JobStatus` (202) | Full analysis pipeline (background) |
| GET | `/api/status/{job_id}` | 60/60s | — | `JobStatus` | Poll job progress |
| GET | `/api/results/{job_id}` | 60/60s | — | `AnalysisResponse` | Fetch completed results |
| POST | `/api/validate-image` | 10/60s | Multipart file | `ImageValidationResponse` | Pre-analysis image validation |
| POST | `/api/profile-analysis` | 20/60s | JSON `ProfileAnalysisRequest` | `ProfileAnalysisResponse` | Profile-only analysis (no image) |

#### Standalone Prediction APIs
| Method | Path | Rate Limit | Request | Response | Description |
|--------|------|-----------|---------|----------|-------------|
| POST | `/api/disease-risk` | 20/60s | JSON `DiseaseRiskRequest` | `DiseaseRiskResponse` | Standalone disease risk from variants |
| POST | `/api/diet-plan` | 20/60s | JSON `DietPlanRequest` | `DietPlanResponse` | Standalone diet plan generation |
| POST | `/api/nutrition` | 20/60s | JSON `NutritionRequest` | `NutritionResponse` | Nutrition plan from profile + conditions |

#### Health Checkup APIs
| Method | Path | Rate Limit | Request | Response | Description |
|--------|------|-----------|---------|----------|-------------|
| POST | `/api/health-checkup` | 10/60s | JSON `HealthCheckupRequest` | `HealthCheckupResponse` | Full checkup from manual lab entry |
| POST | `/api/health-checkup/parse-report` | 10/60s | Multipart file (PDF/image/text) | `ReportParsePreview` | Extract lab values for preview |
| POST | `/api/health-checkup/upload` | 10/60s | Multipart (file + profile) | `HealthCheckupResponse` | Upload report + auto-analyze |

### 14.2 Python API

```python
# Image Analysis
from teloscopy.telomere.pipeline import analyze_image
result = analyze_image("image.tif")

# Disease Risk
from teloscopy.genomics.disease_risk import DiseasePredictor
predictor = DiseasePredictor()
risks = predictor.predict_from_variants(
    variants={"rs429358": "CT", "rs7412": "CC"},
    age=45, sex="female"
)

# Diet Plan
from teloscopy.nutrition.diet_advisor import DietAdvisor
advisor = DietAdvisor()
recs = advisor.generate_recommendations(
    genetic_risks=["Cardiovascular Disease"],
    variants={"rs429358": "CT"}, region="south_asia_south",
    age=45, sex="female", dietary_restrictions=["vegetarian"]
)
plans = advisor.create_meal_plan(recs, region="south_asia_south", days=7)

# Face Analysis
from teloscopy.facial.predictor import analyze_face
profile = analyze_face("photo.jpg", chronological_age=35, sex="female")

# Health Checkup
from teloscopy.nutrition.health_checkup import process_health_checkup
result = process_health_checkup(
    blood_data={"hemoglobin": 12.5, "fasting_glucose": 110},
    urine_data={"ph": 6.5},
    abdomen_text="Mild fatty liver grade 1",
    age=45, sex="female"
)

# Multi-Agent (async)
from teloscopy.agents.orchestrator import OrchestratorAgent
orchestrator = OrchestratorAgent()
result = await orchestrator.process_full_analysis(
    image_path="image.tif",
    user_profile={"age": 45, "sex": "female", "region": "south_india"}
)

# Web Server
# uvicorn teloscopy.webapp.app:app --host 0.0.0.0 --port 8000
```

---

## 15. Test Suite

### 15.1 Overview

**520 tests across 10 files, 69 test classes, 4,848 lines**

All tests run with `pytest` and pass consistently. No `conftest.py` — each file uses inline fixtures.

### 15.2 Test Coverage by Module

| Test File | Lines | Tests | Classes | Module Tested |
|-----------|-------|-------|---------|---------------|
| `test_agents.py` | 1,067 | 54 | 10 | Multi-agent system (all 6 agents) |
| `test_disease_risk.py` | 614 | 54 | 10 | Disease risk prediction |
| `test_health_checkup.py` | 905 | 99 | 9 | Health checkup analyzer |
| `test_nutrition.py` | 633 | 51 | 8 | Nutrigenomics diet engine |
| `test_diet_restrictions.py` | 350 | 15 | 9 | Dietary restriction filtering (10 regions × 8 types) |
| `test_diet_variety.py` | 653 | 25 | 6 | Meal plan variety (30-day uniqueness) |
| `test_pipeline.py` | 181 | 14 | 7 | Full telomere analysis pipeline |
| `test_synthetic.py` | 123 | 16 | 3 | Synthetic qFISH image generation |
| `test_webapp.py` | 322 | 19 | 7 | FastAPI endpoints (9 route groups) |

### 15.3 Testing Highlights

- **Parametrized diet tests** across 10–30 geographic regions
- **Boundary value testing** for lab thresholds (e.g., glucose 99/100/126)
- **Regression tests** (e.g., "butter chicken must never appear in vegetarian plan")
- **Async agent tests** via `asyncio.run()` wrapper
- **Integration tests** covering full end-to-end pipeline with synthetic images
- **No Android tests** — only backend Python tests

### 15.4 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_health_checkup.py -v

# With coverage
pytest tests/ --cov=teloscopy --cov-report=html
```

---

## 16. Deployment Architecture

### 16.1 Installation Options

```bash
# Option A: pip (simplest)
pip install -e ".[all,webapp]"
teloscopy serve

# Option B: setup.sh (guided, creates venv, runs tests)
curl -sSL https://raw.githubusercontent.com/Mahesh2023/teloscopy/main/setup.sh | bash

# Option C: Docker (recommended for production)
docker-compose up

# Option D: Makefile (developer)
make install-dev && make test && make run
```

### 16.2 Docker Architecture

```
┌──────────────────────────────────────────────────┐
│  Docker Container (python:3.12-slim)             │
│  ├── Non-root user (tini init)                   │
│  ├── FastAPI + Uvicorn on port 8000              │
│  ├── Health check: /api/health every 30s         │
│  └── Memory limit: 2G                            │
│                                                   │
│  Volumes: ./data:/app/data, ./output:/app/output │
└──────────────────────────────────────────────────┘
```

### 16.3 Render.com Deployment

Configured via `render.yaml` (free tier, web service).

### 16.4 CI/CD Pipeline (GitHub Actions)

```
Trigger: push/PR to main
  ├── lint: ruff check src/ tests/
  ├── test: pytest (matrix: Python 3.11, 3.12, 3.13)
  └── docker: build + smoke test
```

### 16.5 Android Build

```bash
# Requirements: JDK 17, Android SDK (compileSdk 34)
cd android && ./gradlew assembleDebug
# Output: app/build/outputs/apk/debug/app-debug.apk (~21 MB)
```

---

## 17. Security & Privacy

### Data Handling
- **No external API calls**: All analysis runs locally
- **No persistent storage**: Results stored in-memory during session only
- **Image privacy**: Uploaded images processed and deleted; never stored permanently
- **Genetic data**: SNP data processed in-memory, never written to disk

### Web Application Security
- **Security headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, CSP, Referrer-Policy
- **Rate limiting**: In-memory sliding-window per client IP (configurable)
- **CORS**: Configurable via `TELOSCOPY_CORS_ORIGINS` (default: `*` in dev, restricted in prod)
- **Upload limits**: 50 MiB for images, 20 MiB for reports
- **Input validation**: Pydantic models with strict constraints on all endpoints

### Medical Disclaimer

> **IMPORTANT**: Teloscopy is an educational and research tool. Disease risk predictions
> are based on published population-level associations and should NOT be used for clinical
> decision-making. Always consult qualified healthcare professionals for medical advice.
> Genetic risk scores are probabilistic estimates, not diagnoses.

---

## 18. Scientific Background

### 18.1 Telomere Biology
- **Normal range**: 5,000–15,000 base pairs (newborn ~11,000; elderly ~4,000)
- **Shortening rate**: ~50–100 bp/year in leukocytes
- **Critical length**: ~3,000–5,000 bp triggers cellular senescence

### 18.2 Telomere-Disease Evidence

| Association | Evidence Level | Key Studies |
|------------|---------------|-------------|
| Short telomeres → Cancer | Strong | Haycock et al., BMJ 2014 |
| Short telomeres → CVD | Strong | D'Mello et al., JAHA 2015 |
| Short telomeres → T2D | Moderate | Zhao et al., Diabetes Care 2013 |
| Short telomeres → Alzheimer's | Moderate | Forero et al., J Alzheimer's Dis 2016 |
| Diet → Telomere length | Moderate | Crous-Bou et al., BMJ 2014 |
| Stress → Telomere shortening | Strong | Epel et al., PNAS 2004 |

### 18.3 Nutrigenomics Evidence

| Gene-Nutrient Interaction | Evidence |
|--------------------------|----------|
| MTHFR C677T → Folate need ↑ | Strong (40% reduced enzyme activity with TT) |
| FTO rs9939609 → Obesity risk | Strong (~3 kg higher per risk allele) |
| LCT rs4988235 → Lactose tolerance | Definitive (single variant) |
| CYP1A2 → Caffeine metabolism | Strong (slow metabolizers: MI risk with >2 cups/day) |
| APOE e4 → Fat metabolism | Strong (benefit from low-saturated-fat diet) |

### 18.4 Competitive Landscape

| Tool | qFISH | Disease Risk | Diet | Multi-Agent | Open Source |
|------|-------|-------------|------|-------------|-------------|
| **Teloscopy** | Yes | Yes (560 SNPs) | Yes (650 foods) | Yes (6 agents) | Yes |
| TeloScope | Yes | No | No | No | Yes |
| TelomereHunter | No (WGS) | No | No | No | Yes |
| 23andMe | No | Partial | No | No | No |
| Nebula Genomics | No | Yes | No | No | No |

---

## 19. Performance Considerations

### Processing Times (estimated)

| Stage | 512×512 | 2048×2048 |
|-------|---------|-----------|
| Preprocessing | ~50ms | ~200ms |
| Segmentation (Otsu) | ~100ms | ~500ms |
| Segmentation (Cellpose) | ~2s | ~10s |
| Spot Detection (LoG) | ~200ms | ~1s |
| Association | ~10ms | ~50ms |
| Quantification | ~50ms | ~200ms |
| Disease Risk | ~20ms | ~20ms |
| Diet Plan | ~50ms | ~50ms |
| **Total (Otsu)** | **~500ms** | **~2s** |
| **Total (Cellpose)** | **~2.5s** | **~12s** |

### Memory Usage
- Single 512×512 image: ~50 MB peak
- Single 2048×2048 image: ~500 MB peak
- Cellpose model loading: ~200 MB additional
- Agent system + JSON data: ~50 MB

---

## 20. Future Roadmap

### Completed Phases

**Phase 1: Core Platform** ✓
- [x] qFISH image analysis pipeline
- [x] Synthetic test image generator
- [x] CLI interface
- [x] Disease risk prediction (560 SNPs, 26 categories)
- [x] Diet recommendation engine (650 foods, 35 regions)
- [x] Multi-agent system (6 agents)
- [x] Web UI with image/face upload
- [x] Health checkup analysis (24 conditions)
- [x] Lab report document upload (PDF/image/text)
- [x] Android app (Jetpack Compose, 5 screens)
- [x] Docker + pip + Render deployment

**Phase 2: Enhanced Intelligence** ✓
- [x] ML-based spot detection (CNN) — `ml/cnn_spot_detector.py`
- [x] LLM-powered reports (Ollama/OpenAI) — `integrations/llm_reports.py`
- [x] User feedback loop — `tracking/feedback.py`
- [x] Multi-language diet plans (10 languages) — `nutrition/i18n.py`
- [x] 23andMe/AncestryDNA import — `integrations/genotype_import.py`
- [x] Facial-genomic analysis (59 SNPs) — `facial/predictor.py`

**Phase 3: Clinical-Grade** ✓
- [x] HIPAA compliance — `integrations/fhir.py`
- [x] HL7 FHIR integration — `integrations/fhir.py`
- [x] Whole genome sequencing — `integrations/wgs.py`
- [x] Longitudinal tracking — `tracking/longitudinal.py`
- [x] Clinical validation — `clinical/validation.py`
- [x] FDA 510(k) pathway — `clinical/validation.py`

**Phase 4: Platform** ✓
- [x] Plugin system — `platform/plugin_system.py`
- [x] Federated learning — `platform/federated.py`
- [x] Mobile API — `platform/mobile_api.py`
- [x] Research tools — `platform/research_tools.py`

**Phase 5: Production Readiness** ✓
- [x] Real microscopy training dataset — `data/training.py`, `scripts/generate_training_data.py`
- [x] Published benchmarks — `data/benchmarks.py`, `scripts/run_benchmarks.py`, `benchmark_results/`
- [x] iOS companion app (SwiftUI, 5 screens) — `ios/Teloscopy/`
- [x] Multi-institution clinical trial integration — `clinical/trials.py`, `clinical/endpoints.py`
- [x] Android CI/CD (GitHub Actions) — `.github/workflows/android-ci.yml`
- [x] Mobile detection + APK download banner — `webapp/templates/index.html`, `webapp/app.py`

---

*Architecture document maintained by the Teloscopy development team.*
*Last updated: April 2026 — v2.0.0*
