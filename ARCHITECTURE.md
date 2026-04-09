# Teloscopy Architecture

> **Version 2.0** — Multi-Agent Genomic Intelligence Platform

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Multi-Agent System](#3-multi-agent-system)
4. [Module Architecture](#4-module-architecture)
5. [Data Flow](#5-data-flow)
6. [Web Application](#6-web-application)
7. [Disease Risk Prediction Engine](#7-disease-risk-prediction-engine)
8. [Nutrigenomic Diet Advisor](#8-nutrigenomic-diet-advisor)
9. [Image Analysis Pipeline](#9-image-analysis-pipeline)
10. [Deployment Architecture](#10-deployment-architecture)
11. [API Reference](#11-api-reference)
12. [Security & Privacy](#12-security--privacy)
13. [Project-Specific Insights](#13-project-specific-insights)
14. [Performance Considerations](#14-performance-considerations)
15. [Future Roadmap](#15-future-roadmap)

---

## 1. System Overview

Teloscopy is a **multi-agent genomic intelligence platform** that:

1. Accepts fluorescence microscopy images (qFISH) via web upload or CLI
2. Analyzes telomere length at each chromosome end using computer vision
3. Predicts disease risk using telomere data + genetic variants (SNPs)
4. Generates personalized diet plans based on genetics + geographic food availability
5. Continuously self-improves through an autonomous agent system

```
                           TELOSCOPY v2.0
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   USER ──────► WEB UI / CLI / Python API                │
    │                      │                                  │
    │                      ▼                                  │
    │            ┌─────────────────┐                          │
    │            │  ORCHESTRATOR   │  Multi-Agent Controller   │
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
    │   │  Sequencing    │ Visualization │ Pipeline   │       │
    │   └───────────────────────────────────────────┘       │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

## 2. High-Level Architecture

### 2.1 Architecture Pattern

Teloscopy uses a **microkernel + multi-agent** architecture:

- **Microkernel**: Core image analysis pipeline (preprocessing → segmentation → detection → quantification)
- **Agents**: Autonomous specialist modules that communicate via message passing
- **Plugin System**: New analysis methods can be added without modifying core code
- **Event-Driven**: Agents react to events (image uploaded, analysis complete, etc.)

### 2.2 Layer Diagram

```
┌─────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                  │
│   Web UI (FastAPI + Jinja2)  │  CLI (Click)  │ API  │
├─────────────────────────────────────────────────────┤
│                  ORCHESTRATION LAYER                 │
│   OrchestratorAgent  │  Workflow Engine  │  Router   │
├─────────────────────────────────────────────────────┤
│                  INTELLIGENCE LAYER                  │
│  ImageAgent │ GenomicsAgent │ NutritionAgent │ ...   │
├─────────────────────────────────────────────────────┤
│                    DOMAIN LAYER                      │
│  Telomere Pipeline  │  Disease Risk  │  Diet Advisor │
│  Sequencing         │  Statistics    │  Visualization │
├─────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE LAYER                │
│  File I/O  │  Image Codecs  │  NumPy/SciPy/Pandas   │
│  scikit-image  │  OpenCV  │  tifffile  │  Cellpose   │
└─────────────────────────────────────────────────────┘
```

### 2.3 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Vanilla HTML/CSS/JS | Upload UI, results display |
| API Server | FastAPI + Uvicorn | REST API, async processing |
| Image Processing | scikit-image, OpenCV, tifffile | Core CV pipeline |
| Deep Learning | Cellpose (optional) | Chromosome segmentation |
| Scientific Computing | NumPy, SciPy, pandas | Data processing |
| Visualization | Matplotlib, Seaborn | Plots and overlays |
| CLI | Click + Rich | Terminal interface |
| Containerization | Docker + Docker Compose | Deployment |
| CI/CD | GitHub Actions | Automated testing |
| Testing | pytest | Unit + integration tests |
| Linting | ruff | Code quality |

---

## 3. Multi-Agent System

### 3.1 Agent Architecture

Each agent follows the **Actor Model** pattern — autonomous entities that communicate via asynchronous message passing.

```
┌──────────────────────────────────────────────────┐
│                   BaseAgent                       │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ Message      │  │ State        │               │
│  │ Queue        │  │ Machine      │               │
│  │ (asyncio)    │  │ IDLE→RUNNING │               │
│  └──────┬──────┘  │ →WAITING     │               │
│         │         │ →COMPLETED   │               │
│         ▼         └──────────────┘               │
│  ┌─────────────┐                                  │
│  │ handle_     │  ← Override in subclasses        │
│  │ message()   │                                  │
│  └─────────────┘                                  │
│  ┌─────────────┐                                  │
│  │ send_       │  → Routes via Orchestrator       │
│  │ message()   │                                  │
│  └─────────────┘                                  │
└──────────────────────────────────────────────────┘
```

### 3.2 Agent Registry

| Agent | Role | Capabilities |
|-------|------|-------------|
| **OrchestratorAgent** | Central coordinator | Workflow management, routing, error recovery |
| **ImageAnalysisAgent** | Image processing | Preprocessing, segmentation, detection, quantification |
| **GenomicsAgent** | Genetic analysis | Disease risk, SNP analysis, telomere-disease correlation |
| **NutritionAgent** | Diet planning | Nutrigenomics, regional food mapping, meal planning |
| **ImprovementAgent** | Self-optimization | Parameter tuning, quality metrics, A/B testing methods |
| **ReportAgent** | Report generation | HTML/JSON reports, visualizations, summaries |

### 3.3 Message Flow

```
User uploads image
        │
        ▼
  Orchestrator
        │
        ├──► ImageAgent.analyze_image()
        │         │
        │         ▼
        │    {telomere_results}
        │         │
        ├─────────┤
        │         │
        ├──► GenomicsAgent.assess_risk(telomere_data + variants)
        │         │
        │         ▼
        │    {disease_risks}
        │         │
        ├──► NutritionAgent.generate_diet_plan(risks + region)
        │         │
        │         ▼
        │    {diet_recommendations}
        │         │
        ├──► ReportAgent.generate_full_report(all_results)
        │         │
        │         ▼
        │    {html_report, json_data, visualizations}
        │         │
        └──► ImprovementAgent.track_metrics(results)
                  │
                  ▼
             {quality_score, improvement_suggestions}
```

### 3.4 Continuous Improvement Loop

The ImprovementAgent runs continuously in the background:

```
┌──────────────────────────────────────────────┐
│           CONTINUOUS IMPROVEMENT LOOP          │
│                                                │
│  1. Collect metrics from each analysis run     │
│     ├── Spot detection confidence scores       │
│     ├── Segmentation quality (overlap IoU)     │
│     ├── Association success rate                │
│     └── User feedback (if provided)            │
│                                                │
│  2. Evaluate pipeline quality                  │
│     ├── Compare methods (LoG vs DoG vs DoH)    │
│     ├── Track per-image quality scores         │
│     └── Identify systematic failures           │
│                                                │
│  3. Suggest parameter tuning                   │
│     ├── Adjust blob_log thresholds             │
│     ├── Tune watershed min_distance            │
│     ├── Optimize background subtraction        │
│     └── Select best method per image type      │
│                                                │
│  4. Auto-apply improvements (with approval)    │
│     ├── Update default config parameters       │
│     ├── Switch detection methods                │
│     └── Log all changes for audit trail        │
│                                                │
│  Repeat every N analyses or on schedule        │
└──────────────────────────────────────────────┘
```

---

## 4. Module Architecture

### 4.1 Complete Module Map

```
src/teloscopy/
│
├── __init__.py                    # Package root, version
├── cli.py                         # Click CLI (analyze, batch, generate, report, serve)
│
├── telomere/                      # Core image analysis pipeline
│   ├── __init__.py
│   ├── preprocessing.py           # Image loading, background subtraction, denoising
│   ├── segmentation.py            # Chromosome segmentation (Otsu+watershed, Cellpose)
│   ├── spot_detection.py          # Telomere spot detection (LoG, DoG, DoH)
│   ├── association.py             # Spot-to-chromosome tip matching (KDTree)
│   ├── quantification.py          # Aperture photometry, calibration
│   ├── pipeline.py                # End-to-end orchestrator (9-step pipeline)
│   └── synthetic.py               # Synthetic test image generator
│
├── sequencing/                    # Sequence-based telomere analysis
│   ├── __init__.py
│   └── telomere_seq.py            # TTAGGG counting from BAM/FASTQ
│
├── genomics/                      # NEW: Genetic disease risk prediction
│   ├── __init__.py
│   └── disease_risk.py            # SNP database, risk calculator, PRS
│
├── nutrition/                     # NEW: Diet recommendation engine
│   ├── __init__.py
│   └── diet_advisor.py            # Nutrigenomics, regional foods, meal planning
│
├── agents/                        # NEW: Multi-agent orchestration
│   ├── __init__.py
│   ├── base.py                    # BaseAgent, AgentMessage, AgentState
│   ├── orchestrator.py            # Central coordinator
│   ├── image_agent.py             # Image analysis specialist
│   ├── genomics_agent.py          # Disease risk specialist
│   ├── nutrition_agent.py         # Diet planning specialist
│   ├── improvement_agent.py       # Self-optimization agent
│   └── report_agent.py            # Report generation agent
│
├── analysis/                      # Statistical analysis
│   ├── __init__.py
│   └── statistics.py              # Per-cell/sample statistics, DataFrame export
│
├── visualisation/                 # Plotting and reports
│   ├── __init__.py
│   └── plots.py                   # Overlays, histograms, heatmaps, galleries
│
├── webapp/                        # NEW: Web application
│   ├── __init__.py
│   ├── app.py                     # FastAPI application
│   ├── models.py                  # Pydantic request/response models
│   ├── templates/
│   │   ├── index.html             # Main upload + results page
│   │   └── dashboard.html         # Agent monitoring dashboard
│   └── static/                    # Static assets
│
└── models/                        # Data models
    └── __init__.py
```

### 4.2 Dependency Graph

```
webapp.app
    ├── agents.orchestrator
    │       ├── agents.image_agent
    │       │       └── telomere.pipeline
    │       │               ├── telomere.preprocessing
    │       │               ├── telomere.segmentation
    │       │               ├── telomere.spot_detection
    │       │               ├── telomere.association
    │       │               └── telomere.quantification
    │       ├── agents.genomics_agent
    │       │       └── genomics.disease_risk
    │       ├── agents.nutrition_agent
    │       │       └── nutrition.diet_advisor
    │       ├── agents.improvement_agent
    │       │       └── analysis.statistics
    │       └── agents.report_agent
    │               └── visualisation.plots
    ├── genomics.disease_risk
    └── nutrition.diet_advisor

telomere.synthetic (standalone - test data generation)
sequencing.telomere_seq (standalone - BAM/FASTQ analysis)
```

---

## 5. Data Flow

### 5.1 End-to-End Analysis Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW                             │
│                                                                   │
│  INPUT                                                            │
│  ├── Microscopy Image (.tif, .png, .jpg)                         │
│  ├── User Profile (age, sex, region)                             │
│  ├── Known Genetic Variants (optional, rsid → genotype)          │
│  └── Dietary Restrictions (optional)                             │
│                                                                   │
│  STAGE 1: IMAGE ANALYSIS (ImageAgent)                            │
│  ├── Load image → 2-channel array (DAPI + Cy3)                  │
│  ├── Background subtraction (rolling ball / tophat)              │
│  ├── Chromosome segmentation → labeled mask                      │
│  ├── Tip detection → 2 tips per chromosome                      │
│  ├── Telomere spot detection (LoG) → (y, x, sigma) list         │
│  ├── Spot-tip association → each spot tagged to chromosome arm   │
│  ├── Aperture photometry → corrected intensity per spot          │
│  └── OUTPUT: {spots: [...], chromosomes: [...], statistics: {}}  │
│                                                                   │
│  STAGE 2: DISEASE RISK PREDICTION (GenomicsAgent)                │
│  ├── INPUT: telomere statistics + optional SNP variants          │
│  ├── Telomere-based risks:                                       │
│  │   ├── Short telomeres → elevated cancer risk                  │
│  │   ├── Short telomeres → cardiovascular disease                │
│  │   ├── High CV → genomic instability                           │
│  │   └── Age-adjusted telomere percentile                        │
│  ├── SNP-based risks (if variants provided):                     │
│  │   ├── Polygenic risk scores per condition                     │
│  │   ├── Multiplicative odds ratio model                         │
│  │   └── Age + sex adjustment                                   │
│  ├── Combined risk profile with confidence scores                │
│  └── OUTPUT: {risks: [...], timeline: {...}, insights: [...]}    │
│                                                                   │
│  STAGE 3: DIET RECOMMENDATIONS (NutritionAgent)                  │
│  ├── INPUT: disease risks + genetic variants + region            │
│  ├── Nutrigenomics mapping:                                      │
│  │   ├── Gene → nutrient needs                                  │
│  │   ├── Risk → protective foods                                │
│  │   └── Telomere protection foods (antioxidants, omega-3)      │
│  ├── Geographic adaptation:                                      │
│  │   ├── Map nutrients to locally available foods                │
│  │   ├── Traditional cuisine integration                        │
│  │   └── Seasonal availability                                  │
│  ├── Meal plan generation (7-day):                               │
│  │   ├── Breakfast, lunch, dinner, snacks                       │
│  │   ├── Macro/micro nutrient targets                           │
│  │   └── Dietary restriction compliance                         │
│  └── OUTPUT: {recommendations: [...], meal_plans: [...]}         │
│                                                                   │
│  STAGE 4: REPORT GENERATION (ReportAgent)                        │
│  ├── Compile all results into structured report                  │
│  ├── Generate visualizations (overlay, histograms, risk charts)  │
│  ├── Format as HTML + JSON + CSV                                 │
│  └── OUTPUT: {report_html, report_json, csv_data, plot_paths}    │
│                                                                   │
│  STAGE 5: CONTINUOUS IMPROVEMENT (ImprovementAgent)              │
│  ├── Track quality metrics for this analysis                     │
│  ├── Compare with historical performance                         │
│  ├── Suggest parameter adjustments if quality is low             │
│  └── OUTPUT: {quality_score, suggestions: [...]}                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Models

```python
# Core data structures flowing through the system

TelomereSpot = {
    "id": int,
    "y": float, "x": float, "sigma": float,
    "chromosome_label": int,
    "arm": "p" | "q",
    "raw_intensity": float,
    "corrected_intensity": float,
    "snr": float,
    "associated": bool,
    "valid": bool,
    "length_bp": float | None,  # if calibrated
}

DiseaseRisk = {
    "condition": str,           # e.g., "Cardiovascular Disease"
    "category": str,            # e.g., "cardiovascular"
    "lifetime_risk_pct": float, # e.g., 23.5
    "relative_risk": float,     # e.g., 1.8
    "confidence": str,          # "high" | "moderate" | "low"
    "contributing_variants": [str],
    "age_of_onset_range": (int, int),
    "preventability_score": float,  # 0.0 to 1.0
}

DietaryRecommendation = {
    "title": str,
    "description": str,
    "foods_to_increase": [str],
    "foods_to_decrease": [str],
    "rationale": str,
    "genetic_basis": str,
    "confidence": str,
}

MealPlan = {
    "day": int,
    "breakfast": {"items": [str], "calories": int},
    "lunch": {"items": [str], "calories": int},
    "dinner": {"items": [str], "calories": int},
    "snacks": {"items": [str], "calories": int},
    "daily_calories": int,
    "macros": {"protein_g": float, "carbs_g": float, "fat_g": float},
}
```

---

## 6. Web Application

### 6.1 Architecture

```
┌──────────────────────────────────────────────────┐
│                 BROWSER                            │
│  ┌────────────────────────────────────────────┐   │
│  │  index.html                                 │   │
│  │  ├── Drag-and-drop image upload            │   │
│  │  ├── User profile form                     │   │
│  │  ├── Real-time progress bar                │   │
│  │  ├── Results display (telomere + risk)     │   │
│  │  └── Diet plan viewer                      │   │
│  └────────────────┬───────────────────────────┘   │
│                   │ HTTP/REST                      │
└───────────────────┼──────────────────────────────┘
                    │
┌───────────────────┼──────────────────────────────┐
│                   ▼                                │
│  ┌────────────────────────────────────────────┐   │
│  │  FastAPI Server (uvicorn)                   │   │
│  │  ├── POST /api/upload     → save image     │   │
│  │  ├── POST /api/analyze    → full pipeline  │   │
│  │  ├── GET  /api/status/:id → job progress   │   │
│  │  ├── GET  /api/results/:id→ full results   │   │
│  │  ├── POST /api/disease-risk               │   │
│  │  ├── POST /api/diet-plan                  │   │
│  │  └── GET  /api/agents/status              │   │
│  └────────────────┬───────────────────────────┘   │
│                   │                                │
│  ┌────────────────▼───────────────────────────┐   │
│  │  Multi-Agent System                         │   │
│  │  OrchestratorAgent → routes to specialists  │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  FASTAPI SERVER                                    │
└────────────────────────────────────────────────────┘
```

### 6.2 User Workflow

1. **Upload**: User drags microscopy image onto the upload zone
2. **Profile**: Fills in age, sex, geographic region, dietary restrictions
3. **Optionally**: Enters known genetic variants (rsID → genotype)
4. **Analyze**: Clicks "Run Analysis" — starts the multi-agent pipeline
5. **Progress**: Real-time progress bar shows pipeline stages
6. **Results**: Interactive display with:
   - Telomere overlay image with detected spots highlighted
   - Per-chromosome telomere length table
   - Disease risk cards (color-coded by severity)
   - Year-by-year risk projection chart
   - 7-day meal plan with local foods
   - Downloadable CSV + JSON

---

## 7. Disease Risk Prediction Engine

### 7.1 Architecture

```
┌──────────────────────────────────────────────────────┐
│            DISEASE RISK PREDICTION ENGINE              │
│                                                        │
│  INPUT SOURCES                                         │
│  ├── Telomere Length Data                              │
│  │   ├── Mean telomere fluorescence intensity          │
│  │   ├── Telomere length distribution (CV, percentiles)│
│  │   └── Short telomere fraction (< 10th percentile)   │
│  │                                                     │
│  ├── Genetic Variants (optional)                       │
│  │   ├── SNP genotypes (rsID → alleles)               │
│  │   └── Known pathogenic variants                     │
│  │                                                     │
│  └── Demographics                                      │
│      ├── Age, Sex                                      │
│      └── Ethnicity (for population-specific baselines) │
│                                                        │
│  BUILT-IN DATABASE                                     │
│  ├── 50+ SNP-disease associations                      │
│  ├── Population allele frequencies                     │
│  ├── Baseline disease incidence rates (by age/sex)     │
│  └── Telomere-disease correlation coefficients         │
│                                                        │
│  RISK CALCULATION                                      │
│  ├── Multiplicative OR model for SNP combinations      │
│  ├── Telomere length risk adjustment                   │
│  │   ├── < 10th %ile → 1.5x cancer risk              │
│  │   ├── < 10th %ile → 1.8x CVD risk                 │
│  │   └── High CV → 1.3x genomic instability          │
│  ├── Age-dependent baseline incidence                  │
│  ├── Sex-specific modifications                        │
│  └── Confidence scoring (evidence level + N variants)  │
│                                                        │
│  OUTPUT                                                │
│  ├── Risk Profile (per-condition)                      │
│  │   ├── Lifetime risk percentage                      │
│  │   ├── Relative risk vs population                   │
│  │   ├── Contributing factors                          │
│  │   └── Preventability score                          │
│  ├── Year-by-Year Risk Projection                      │
│  └── Actionable Prevention Recommendations             │
│                                                        │
│  DISCLAIMER: Educational/research use only.            │
│  Not a substitute for clinical genetic testing.        │
└──────────────────────────────────────────────────────┘
```

### 7.2 Disease Categories Covered

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

### 7.3 Telomere-Disease Correlation Model

```
Telomere Length Quintile    Cancer Risk Multiplier    CVD Risk Multiplier
─────────────────────────────────────────────────────────────────────────
Q1 (shortest 20%)           1.50x                     1.80x
Q2                          1.20x                     1.40x
Q3 (median)                 1.00x (baseline)          1.00x (baseline)
Q4                          0.90x                     0.85x
Q5 (longest 20%)            0.80x                     0.70x

Additional factors:
- CV > 0.30 → genomic instability marker → +0.2x all risks
- Short telomere fraction > 15% → accelerated aging → +0.3x all risks
```

---

## 8. Nutrigenomic Diet Advisor

### 8.1 Architecture

```
┌──────────────────────────────────────────────────────┐
│              NUTRIGENOMIC DIET ADVISOR                 │
│                                                        │
│  ┌─────────────────────────────────────────────┐      │
│  │  GENETIC PROFILE → NUTRIENT NEEDS MAPPING    │      │
│  │                                               │      │
│  │  MTHFR C677T ──► Folate ↑ (leafy greens)    │      │
│  │  FTO rs9939609 ─► Calorie control            │      │
│  │  LCT rs4988235 ─► Lactose: dairy/alt         │      │
│  │  CYP1A2 ────────► Caffeine: fast/slow        │      │
│  │  APOE e4 ────────► Saturated fat ↓           │      │
│  │  VDR ────────────► Vitamin D ↑               │      │
│  │  HFE ────────────► Iron intake adjust        │      │
│  │  FADS1/2 ────────► Omega-3 ↑ (fish/flax)    │      │
│  │  TCF7L2 ─────────► Low glycemic diet          │      │
│  │  + 16 more gene-nutrient mappings             │      │
│  └─────────────────────────────────────────────┘      │
│                        │                               │
│                        ▼                               │
│  ┌─────────────────────────────────────────────┐      │
│  │  DISEASE RISKS → PROTECTIVE DIET             │      │
│  │                                               │      │
│  │  Cancer risk ──────► Antioxidants, fiber     │      │
│  │  CVD risk ─────────► Omega-3, low sodium     │      │
│  │  Diabetes risk ────► Low GI, fiber           │      │
│  │  Alzheimer's risk ─► Mediterranean diet      │      │
│  │  Osteoporosis ─────► Calcium, Vitamin D      │      │
│  │  Short telomeres ──► Telomere-protective     │      │
│  │                      (antioxidants, folate,  │      │
│  │                       omega-3, green tea)     │      │
│  └─────────────────────────────────────────────┘      │
│                        │                               │
│                        ▼                               │
│  ┌─────────────────────────────────────────────┐      │
│  │  GEOGRAPHIC FOOD MAPPING                      │      │
│  │                                               │      │
│  │  Need: Folate → Global food: Spinach          │      │
│  │                                               │      │
│  │  South India: Drumstick leaves, amaranth     │      │
│  │  North India: Methi (fenugreek), sarson      │      │
│  │  Japan: Edamame, natto, seaweed              │      │
│  │  Mediterranean: Lentils, chickpeas           │      │
│  │  Mexico: Black beans, nopales               │      │
│  │  West Africa: Okra, cassava leaves           │      │
│  │                                               │      │
│  │  12 geographic regions × 100+ foods           │      │
│  └─────────────────────────────────────────────┘      │
│                        │                               │
│                        ▼                               │
│  ┌─────────────────────────────────────────────┐      │
│  │  7-DAY MEAL PLAN GENERATOR                    │      │
│  │                                               │      │
│  │  Day 1:                                       │      │
│  │  Breakfast: [region-specific items]           │      │
│  │  Lunch:     [balances macros + micronutrients]│      │
│  │  Dinner:    [addresses genetic nutrient needs]│      │
│  │  Snacks:    [telomere-protective foods]       │      │
│  │                                               │      │
│  │  Respects: vegetarian, vegan, gluten-free,    │      │
│  │  halal, kosher, nut-free, low-FODMAP         │      │
│  └─────────────────────────────────────────────┘      │
│                                                        │
└──────────────────────────────────────────────────────┘
```

### 8.2 Geographic Coverage

| Region | Sub-Regions | Key Foods |
|--------|------------|-----------|
| South Asia | North India, South India, East India, West India | Dal, roti, rice, turmeric, ghee, paneer, dosa |
| East Asia | China, Japan, Korea | Rice, soy, miso, kimchi, seaweed, green tea |
| Southeast Asia | Thailand, Vietnam, Indonesia | Rice noodles, coconut, lemongrass, fish sauce |
| Middle East | Levant, Gulf, North Africa | Hummus, tahini, dates, olive oil, lamb |
| Mediterranean | Greece, Italy, Spain | Olive oil, tomatoes, feta, legumes, fish |
| Northern Europe | UK, Scandinavia, Germany | Oats, rye, herring, root vegetables, dairy |
| Sub-Saharan Africa | West, East, Southern | Plantain, cassava, millet, groundnuts, okra |
| Latin America | Mexico, Brazil, Andes | Beans, corn, avocado, quinoa, chili |
| North America | General US/Canada | Mixed (provides all global alternatives) |

---

## 9. Image Analysis Pipeline

### 9.1 Pipeline Steps (Detail)

```
Step 1: LOAD IMAGE
├── Input: .tif/.tiff (16-bit multi-channel), .png, .jpg
├── Library: tifffile (TIFF), OpenCV (PNG/JPG)
├── Extract: DAPI channel (index 0), Cy3 channel (index 1)
└── Output: dapi[H,W] float64, cy3[H,W] float64

Step 2: BACKGROUND SUBTRACTION
├── Method A: Rolling Ball (morphological opening, radius=50)
├── Method B: Top-Hat (white_tophat, disk selem, radius=50)
├── Method C: Gaussian (subtract blurred version, sigma=50)
└── Output: background-corrected dapi, cy3

Step 3: DENOISING
├── Gaussian filter (sigma=1.0, configurable)
└── Output: denoised dapi, cy3

Step 4: CHROMOSOME SEGMENTATION
├── Method A: Otsu + Watershed
│   ├── Otsu threshold on DAPI
│   ├── Distance transform
│   ├── Local maxima detection (min_distance=20)
│   ├── Watershed from markers
│   └── Filter: min_chromosome_area=500
├── Method B: Cellpose (optional)
│   ├── model_type="cyto3"
│   ├── channels=[0,0] (grayscale)
│   └── diameter=None (auto-detect)
└── Output: labels[H,W] int32 (0=background, 1..N=chromosomes)

Step 5: TIP DETECTION
├── For each chromosome region:
│   ├── Extract boundary points
│   ├── Compute convex hull
│   ├── Find two most distant points (diameter endpoints)
│   └── These are p-arm and q-arm tips
└── Output: tips_dict {label: [(y1,x1), (y2,x2)]}

Step 6: SPOT DETECTION (Cy3 channel)
├── Method A: blob_log (Laplacian of Gaussian)
│   ├── min_sigma=1.0, max_sigma=5.0
│   ├── num_sigma=10, threshold=0.05
│   └── Most accurate, slowest
├── Method B: blob_dog (Difference of Gaussian)
│   ├── min_sigma=1.0, max_sigma=5.0
│   └── threshold=0.05
├── Method C: blob_doh (Determinant of Hessian)
│   ├── min_sigma=1.0, max_sigma=5.0
│   └── threshold=0.005
└── Output: spots [(y, x, sigma), ...]

Step 7: SPOT-CHROMOSOME ASSOCIATION
├── Build KDTree from all chromosome tips
├── For each spot: query nearest tip
├── Filter: distance < max_distance (15 px default)
├── Resolve conflicts: if multiple spots → same tip, keep brightest
└── Output: spots with chromosome_label + arm tag

Step 8: INTENSITY QUANTIFICATION
├── For each spot:
│   ├── Circular aperture (radius=5): sum pixel values
│   ├── Annular background (inner=7, outer=12): median
│   ├── Corrected intensity = aperture_sum - (bg_median × aperture_area)
│   └── SNR = corrected_intensity / bg_std
└── Output: spots with corrected_intensity, snr, valid flag

Step 9: CALIBRATION (optional)
├── Reference standards: [(intensity1, bp1), (intensity2, bp2), ...]
├── Fit: linear regression or polynomial
├── Apply: length_bp = calibration_fn(corrected_intensity)
└── Output: spots with length_bp field
```

### 9.2 Quality Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| Spots per chromosome | 2-4 | 1-6 | 0 or >6 |
| Association rate | >80% | 60-80% | <60% |
| Mean SNR | >5.0 | 3.0-5.0 | <3.0 |
| CV of intensities | <0.40 | 0.40-0.60 | >0.60 |
| Short telomere % | <15% | 15-30% | >30% |

---

## 10. Deployment Architecture

### 10.1 One-Click Installation Options

```
Option A: pip install (simplest)
─────────────────────────────────
  pip install -e ".[all,webapp]"
  teloscopy serve

Option B: setup.sh (guided)
──────────────────────────────
  curl -sSL https://raw.githubusercontent.com/Mahesh2023/teloscopy/main/setup.sh | bash
  # Creates venv, installs deps, generates sample data, runs tests, starts server

Option C: Docker (recommended for production)
────────────────────────────────────────────────
  docker-compose up
  # Builds image, starts FastAPI server on port 8000

Option D: Makefile (developer)
──────────────────────────────
  make install-dev   # Install with dev dependencies
  make test          # Run all tests
  make run           # Start web server
```

### 10.2 Docker Architecture

```
┌──────────────────────────────────────────┐
│  Docker Container                         │
│  ┌────────────────────────────────────┐  │
│  │  python:3.12-slim                   │  │
│  │  ┌──────────────────────────────┐  │  │
│  │  │  Teloscopy Application        │  │  │
│  │  │  ├── FastAPI (uvicorn)       │  │  │
│  │  │  ├── Multi-Agent System      │  │  │
│  │  │  └── All analysis modules    │  │  │
│  │  └──────────────────────────────┘  │  │
│  │  Port: 8000                        │  │
│  └────────────────────────────────────┘  │
│                                           │
│  Volumes:                                │
│  ├── ./data:/app/data     (images)       │
│  └── ./output:/app/output (results)      │
└──────────────────────────────────────────┘
```

### 10.3 CI/CD Pipeline

```
GitHub Actions Workflow
───────────────────────
  Trigger: push to main, PR to main

  Jobs:
  ├── lint:
  │   └── ruff check src/ tests/
  │
  ├── test (matrix: 3.11, 3.12, 3.13):
  │   ├── pip install -e ".[all,webapp,dev]"
  │   └── pytest --cov=teloscopy
  │
  └── docker:
      ├── docker build -t teloscopy .
      └── docker run teloscopy pytest (smoke test)
```

---

## 11. API Reference

### 11.1 REST API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/` | Main web UI | None |
| `GET` | `/upload` | Upload page | None |
| `GET` | `/dashboard` | Agent dashboard | None |
| `GET` | `/results/{job_id}` | Results page | None |
| `GET` | `/api/health` | Health check | None |
| `POST` | `/api/upload` | Upload image file | None |
| `POST` | `/api/analyze` | Full analysis pipeline | None |
| `GET` | `/api/status/{job_id}` | Job status + progress | None |
| `GET` | `/api/results/{job_id}` | Full results JSON | None |
| `POST` | `/api/disease-risk` | Disease risk assessment | None |
| `POST` | `/api/diet-plan` | Diet recommendations | None |
| `GET` | `/api/agents/status` | Agent system status | None |

### 11.2 Python API

```python
# Image Analysis
from teloscopy.telomere.pipeline import analyze_image
result = analyze_image("image.tif")

# Disease Risk
from teloscopy.genomics.disease_risk import DiseasePredictor
predictor = DiseasePredictor()
risks = predictor.predict_from_telomere_data(
    mean_length_bp=8500, age=45, sex="female"
)

# Diet Advisor
from teloscopy.nutrition.diet_advisor import DietAdvisor
advisor = DietAdvisor()
plan = advisor.generate_recommendations(
    genetic_risks=risks, variants={}, region="south_india",
    age=45, sex="female", dietary_restrictions=["vegetarian"]
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

## 12. Security & Privacy

### 12.1 Data Handling

- **No external API calls**: All analysis runs locally, no data leaves the machine
- **No persistent storage**: Results stored in-memory during session only
- **Image privacy**: Uploaded images processed and results returned; originals can be auto-deleted
- **Genetic data**: SNP data processed in-memory, never written to disk unless user requests export

### 12.2 Medical Disclaimer

> **IMPORTANT**: Teloscopy is an educational and research tool. Disease risk predictions
> are based on published population-level associations and should NOT be used for clinical
> decision-making. Always consult qualified healthcare professionals for medical advice.
> Genetic risk scores are probabilistic estimates, not diagnoses.

### 12.3 Deployment Security

- Docker runs as non-root user
- No default credentials
- CORS configured for localhost by default (production: configure allowed origins)
- File upload size limits enforced
- Input validation on all API endpoints

---

## 13. Project-Specific Insights

### 13.1 Why Telomere Length Matters

Telomeres are the protective caps at chromosome ends (TTAGGG repeats in humans). They shorten with each cell division, acting as a biological clock. Key insights:

- **Normal range**: 5,000-15,000 base pairs (newborn: ~11,000; elderly: ~4,000)
- **Shortening rate**: ~50-100 bp per year in leukocytes
- **Critical length**: ~3,000-5,000 bp triggers cellular senescence
- **Clinical relevance**: Short telomeres associated with cancer, CVD, diabetes, Alzheimer's

### 13.2 Why qFISH Over Other Methods

| Method | Resolution | Throughput | Cost | What Teloscopy Does |
|--------|-----------|-----------|------|-------------------|
| qFISH | Single chromosome | Low-Medium | $$$ | Automates this |
| Southern Blot | Population average | Low | $$ | Not applicable |
| qPCR | Relative T/S ratio | High | $ | Supported via sequencing module |
| Flow-FISH | Per-cell distribution | Medium | $$$ | Not applicable |
| STELA | Allele-specific | Very Low | $$$$ | Not applicable |

### 13.3 Telomere-Disease Evidence Base

| Association | Evidence Level | Key Studies |
|------------|---------------|-------------|
| Short telomeres → Cancer | Strong | Haycock et al., BMJ 2014 (meta-analysis, 47 studies) |
| Short telomeres → CVD | Strong | D'Mello et al., JAHA 2015 (meta-analysis) |
| Short telomeres → T2D | Moderate | Zhao et al., Diabetes Care 2013 |
| Short telomeres → Alzheimer's | Moderate | Forero et al., J Alzheimer's Dis 2016 |
| Telomere CV → Instability | Suggestive | Baird et al., Nature Genetics 2003 |
| Diet → Telomere length | Moderate | Crous-Bou et al., BMJ 2014 (Mediterranean diet) |
| Exercise → Telomere length | Moderate | Puterman et al., PLoS ONE 2010 |
| Stress → Telomere shortening | Strong | Epel et al., PNAS 2004 |

### 13.4 Nutrigenomics Evidence

| Gene-Nutrient Interaction | Evidence | Impact |
|--------------------------|----------|--------|
| MTHFR C677T → Folate need ↑ | Strong | 40% reduced enzyme activity with TT genotype |
| FTO rs9939609 → Obesity risk | Strong | ~3 kg higher body weight per risk allele |
| LCT rs4988235 → Lactose tolerance | Definitive | Single variant determines adult lactase persistence |
| CYP1A2 → Caffeine metabolism | Strong | Slow metabolizers: >2 cups/day increases MI risk |
| APOE e4 → Fat metabolism | Strong | Carriers benefit more from low-saturated-fat diet |
| FADS1/2 → Omega-3 conversion | Moderate | Poor converters need preformed EPA/DHA |

### 13.5 Competitive Landscape (as of 2025)

| Tool | Language | qFISH | Disease Risk | Diet | Multi-Agent |
|------|----------|-------|-------------|------|-------------|
| **Teloscopy** | Python | Yes | Yes | Yes | Yes |
| TeloScope | ImageJ macro | Yes | No | No | No |
| TelomereHunter | Python | No (WGS) | No | No | No |
| Computel | C++ | No (WGS) | No | No | No |
| 23andMe | SaaS | No | Partial | No | No |
| Nebula Genomics | SaaS | No | Yes | No | No |

**Teloscopy is the only open-source tool combining qFISH image analysis, disease risk prediction, and personalized nutrition.**

---

## 14. Performance Considerations

### 14.1 Processing Times (estimated)

| Stage | Time (512x512) | Time (2048x2048) |
|-------|---------------|-------------------|
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

### 14.2 Memory Usage

- Single 512x512 image: ~50 MB peak
- Single 2048x2048 image: ~500 MB peak
- Cellpose model loading: ~200 MB additional
- Agent system overhead: ~10 MB

### 14.3 Scaling

- **Single image**: All processing in-memory, sequential pipeline
- **Batch**: Parallel processing with `concurrent.futures` (configurable workers)
- **Web server**: Async FastAPI handles concurrent requests; analysis jobs run in background tasks

---

## 15. Future Roadmap

### Phase 1 (Current): Core Platform
- [x] qFISH image analysis pipeline
- [x] Synthetic test image generator
- [x] CLI interface
- [x] Disease risk prediction
- [x] Diet recommendation engine
- [x] Multi-agent system
- [x] Web UI with photo upload
- [x] One-click installation (Docker + pip)

### Phase 2: Enhanced Intelligence
- [ ] ML-based spot detection (CNN trained on qFISH data)
- [ ] Real microscopy image training dataset
- [ ] LLM-powered report generation (local Ollama integration)
- [ ] User feedback loop for continuous improvement
- [ ] Multi-language diet plans
- [ ] Integration with 23andMe/AncestryDNA raw data import

### Phase 3: Clinical-Grade
- [ ] HIPAA-compliant deployment option
- [ ] HL7 FHIR integration for EHR systems
- [ ] Whole genome sequencing integration
- [ ] Longitudinal tracking (track telomere changes over time)
- [ ] Clinical validation studies
- [ ] FDA 510(k) pathway assessment

### Phase 4: Platform
- [ ] Plugin marketplace for custom analysis modules
- [ ] Multi-institution data sharing (federated learning)
- [ ] Mobile app for results viewing
- [ ] Research collaboration tools
- [ ] Published benchmarks and validation datasets

---

*Architecture document maintained by the Teloscopy development team.*
*Last updated: 2025*
