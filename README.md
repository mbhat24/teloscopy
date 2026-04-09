# Teloscopy v2.0

**Multi-Agent Genomic Intelligence Platform** — Telomere analysis, disease risk prediction, and personalized nutrition from qFISH microscopy images.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Mahesh2023/teloscopy)

```
  USER ──► Upload microscopy image + profile
               │
               ▼
       ┌───────────────┐
       │  ORCHESTRATOR  │  Multi-Agent Controller
       └──┬──┬──┬──┬──┘
          │  │  │  │
    ┌─────┘  │  │  └─────┐
    ▼        ▼  ▼        ▼
 ┌──────┐┌──────┐┌──────┐┌──────┐
 │IMAGE ││GENO- ││NUTRI-││REPORT│
 │AGENT ││MICS  ││TION  ││AGENT │
 └──┬───┘└──┬───┘└──┬───┘└──┬───┘
    │       │       │       │
    ▼       ▼       ▼       ▼
 Telomere  Disease  Diet    HTML/
 lengths   risks    plan    JSON
```

## What It Does

| Step | Input | Output |
|------|-------|--------|
| 1. **Image Analysis** | Upload a qFISH microscopy image | Per-chromosome telomere lengths |
| 2. **Disease Prediction** | Telomere data + optional genetic variants | Disease risk profile over next 30 years |
| 3. **Diet Recommendations** | Risk profile + your geographic region | 7-day meal plan with local foods |
| 4. **Report** | Everything above | Downloadable HTML/JSON/CSV report |

## Why Teloscopy?

| Problem | Existing Tools | Teloscopy |
|---------|---------------|-----------|
| qFISH analysis | ImageJ macros (manual, Windows-only) | Python (cross-platform, automated) |
| Disease prediction | Separate paid services (23andMe) | Integrated, free, open-source |
| Diet planning | Generic apps, not gene-aware | Nutrigenomics-driven, region-specific |
| Batch processing | Click through each image | `teloscopy batch ./images/` |
| Continuous improvement | None | Self-optimizing multi-agent system |

## One-Click Installation

### Option A: Quick Setup Script (recommended)

```bash
curl -sSL https://raw.githubusercontent.com/Mahesh2023/teloscopy/main/setup.sh | bash
```

### Option B: pip install

```bash
git clone https://github.com/Mahesh2023/teloscopy.git
cd teloscopy
pip install -e ".[all,webapp,dev]"
```

### Option C: Docker

```bash
git clone https://github.com/Mahesh2023/teloscopy.git
cd teloscopy
docker-compose up
```

### Option D: Makefile

```bash
make install      # Install with all dependencies
make run          # Start web server at http://localhost:8000
make test         # Run all tests
```

## Quick Start

### Web UI (recommended)

```bash
# Start the web server
uvicorn teloscopy.webapp.app:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000 in your browser
# Upload an image, fill in your profile, click Analyze
```

### CLI

```bash
# Generate synthetic test images
teloscopy generate -n 5 -o data/sample_images/

# Analyze a single image
teloscopy analyze data/sample_images/synthetic_qfish_000.tif -o output/

# Batch process all images
teloscopy batch data/sample_images/ -o output/ -p "*.tif"
```

### Python API

```python
from teloscopy.telomere.pipeline import analyze_image
from teloscopy.genomics.disease_risk import DiseasePredictor
from teloscopy.nutrition.diet_advisor import DietAdvisor

# Step 1: Analyze microscopy image
results = analyze_image("metaphase_001.tif")
print(f"Found {len(results['spots'])} telomere spots")

# Step 2: Predict disease risks from telomere data
predictor = DiseasePredictor()
risks = predictor.predict_from_telomere_data(
    mean_length_bp=7500, age=45, sex="female"
)
for risk in risks:
    print(f"  {risk.condition}: {risk.lifetime_risk_pct:.1f}% lifetime risk")

# Step 3: Get personalized diet plan
advisor = DietAdvisor()
recommendations = advisor.generate_recommendations(
    genetic_risks=risks, variants={},
    region="south_india", age=45, sex="female",
    dietary_restrictions=["vegetarian"]
)
for rec in recommendations:
    print(f"  {rec.title}: {rec.description}")

# Step 4: Generate 7-day meal plan with local foods
meal_plans = advisor.create_meal_plan(
    recommendations, region="south_india", calories=1800, days=7
)
```

### Multi-Agent System (async)

```python
import asyncio
from teloscopy.agents.orchestrator import OrchestratorAgent

async def main():
    orchestrator = OrchestratorAgent()
    result = await orchestrator.process_full_analysis(
        image_path="metaphase_001.tif",
        user_profile={
            "age": 45, "sex": "female",
            "region": "south_india",
            "dietary_restrictions": ["vegetarian"],
        }
    )
    print(result["report"]["summary"])

asyncio.run(main())
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete system architecture with diagrams.

### System Components

```
src/teloscopy/
├── telomere/           # Core image analysis pipeline (7 modules)
│   ├── preprocessing   # Load, background subtract, denoise
│   ├── segmentation    # Otsu+watershed / Cellpose
│   ├── spot_detection  # LoG/DoG/DoH blob detection
│   ├── association     # KDTree spot-to-chromosome matching
│   ├── quantification  # Aperture photometry + calibration
│   ├── pipeline        # End-to-end orchestrator
│   └── synthetic       # Synthetic test image generator
│
├── genomics/           # Disease risk prediction
│   └── disease_risk    # 519-SNP database, polygenic risk scores
│
├── nutrition/          # Diet recommendation engine
│   └── diet_advisor    # Nutrigenomics + geographic food mapping
│
├── agents/             # Multi-agent orchestration (7 agents)
│   ├── orchestrator    # Central coordinator
│   ├── image_agent     # Image analysis specialist
│   ├── genomics_agent  # Disease risk specialist
│   ├── nutrition_agent # Diet planning specialist
│   ├── improvement     # Self-optimization agent
│   └── report_agent    # Report generation
│
├── webapp/             # Web application
│   ├── app.py          # FastAPI server with REST API
│   ├── models.py       # Pydantic request/response models
│   └── templates/      # HTML pages (upload, results, dashboard)
│
├── sequencing/         # Sequence-based telomere analysis
├── analysis/           # Statistical analysis
└── visualisation/      # Plotting and reports
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Image Processing | scikit-image, OpenCV, tifffile |
| Deep Learning | Cellpose (optional) |
| Web Server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (dark theme) |
| Scientific Computing | NumPy, SciPy, pandas |
| CLI | Click + Rich |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Disease Risk Prediction

Teloscopy predicts disease risk using two data sources:

### From Telomere Data (no genetic testing needed)
Short telomeres are associated with increased risk of cancer, cardiovascular disease, and accelerated aging. Teloscopy uses published population-level correlations.

### From Genetic Variants (optional)
If you provide SNP genotypes (e.g., from 23andMe raw data), Teloscopy uses a built-in database of 63 well-studied variants across 10 disease categories:

| Category | Conditions | Key Genes |
|----------|-----------|-----------|
| Cardiovascular | CHD, stroke, hypertension | APOE, PCSK9, LPA, LDLR |
| Cancer | Breast, colorectal, prostate | BRCA1/2, TP53, APC, MLH1 |
| Metabolic | Type 2 diabetes, obesity | TCF7L2, FTO, PPARG |
| Neurological | Alzheimer's, Parkinson's | APOE-e4, LRRK2, CLU |
| Autoimmune | RA, T1D, celiac | HLA-DRB1, CTLA4, PTPN22 |
| Eye | Macular degeneration | CFH, ARMS2 |
| Bone | Osteoporosis | ESR1, VDR, COL1A1 |
| Blood | Hemochromatosis | HFE, HBB |

> **Disclaimer**: Risk predictions are for educational/research purposes only. Not a substitute for clinical genetic testing or medical advice. Always consult healthcare professionals.

## Diet Recommendations

Teloscopy generates personalized nutrition plans based on:

1. **Genetic profile** — 120+ gene-nutrient interactions (MTHFR→folate, FTO→calories, LCT→lactose, CYP1A2→caffeine, etc.)
2. **Disease risks** — Protective foods for identified conditions
3. **Telomere health** — Antioxidant-rich foods that protect telomeres
4. **Geographic region** — Locally available foods from 30 regions:

| Region | Sub-Regions | Example Foods |
|--------|------------|---------------|
| South Asia | N/S/E/W India | Dal, turmeric, roti, dosa, paneer |
| East Asia | China, Japan, Korea | Miso, kimchi, seaweed, green tea |
| Southeast Asia | Thailand, Vietnam | Coconut, lemongrass, rice noodles |
| Mediterranean | Greece, Italy, Spain | Olive oil, feta, legumes, fish |
| Middle East | Levant, Gulf | Hummus, tahini, dates, olive oil |
| Northern Europe | UK, Scandinavia | Oats, rye, herring, root veg |
| Sub-Saharan Africa | West, East, South | Plantain, millet, groundnuts |
| Latin America | Mexico, Brazil | Beans, corn, quinoa, avocado |

Supports dietary restrictions: vegetarian, vegan, gluten-free, halal, kosher, nut-free, low-FODMAP.

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload microscopy image |
| `POST` | `/api/analyze` | Full analysis pipeline |
| `GET` | `/api/status/{id}` | Check job progress |
| `GET` | `/api/results/{id}` | Get full results |
| `POST` | `/api/disease-risk` | Disease risk assessment |
| `POST` | `/api/diet-plan` | Diet recommendations |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/agents/status` | Agent system status |

## Deployment

### Render.com (Free)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Mahesh2023/teloscopy)

Or manually:
1. Fork this repo
2. Sign up at [render.com](https://render.com)
3. New Web Service → connect your GitHub → select teloscopy
4. Settings: Python 3, Build: `pip install -e ".[all,webapp]"`, Start: `uvicorn teloscopy.webapp.app:app --host 0.0.0.0 --port $PORT`

### Docker (self-hosted)

```bash
docker-compose up -d
# Access at http://localhost:8000
```

## Development

```bash
# Install dev dependencies
pip install -e ".[all,webapp,dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Start dev server with auto-reload
uvicorn teloscopy.webapp.app:app --reload --port 8000
```

## Scientific Background

Telomeres are repetitive DNA sequences (TTAGGG in humans) at chromosome ends that shorten with each cell division (~50-100 bp/year). They act as a biological clock:

- **Normal range**: 5,000-15,000 base pairs
- **Critical length**: ~3,000-5,000 bp triggers cellular senescence
- **Clinical relevance**: Short telomeres → cancer, CVD, diabetes, Alzheimer's risk

Quantitative FISH (qFISH) measures telomere length by:
1. Hybridizing fluorescent PNA probes to telomere repeats
2. Imaging under fluorescence microscopy
3. **Teloscopy automates**: Measuring intensity at each chromosome end → converting to base pairs

## Project Stats

| Metric | Value |
|--------|-------|
| Total Python files | 31+ |
| Total lines of code | 15,000+ |
| Disease variants in DB | 63 |
| Geographic food regions | 12+ |
| Gene-nutrient mappings | 25+ |
| Food items in database | 100+ |
| Agent types | 7 |
| API endpoints | 12 |
| Test cases | 50+ |

## License

MIT

## References

- Lansdorp, P. M. et al. (1996). "Heterogeneity in telomere length of human chromosomes." *Human Molecular Genetics*, 5(5), 685-691.
- Haycock, P. C. et al. (2014). "Leucocyte telomere length and risk of cardiovascular disease." *BMJ*, 349, g4227.
- Crous-Bou, M. et al. (2014). "Mediterranean diet and telomere length." *BMJ*, 349, g6674.
- Stringer, C. et al. (2021). "Cellpose: a generalist algorithm for cellular segmentation." *Nature Methods*, 18, 100-106.
- van der Walt, S. et al. (2014). "scikit-image: image processing in Python." *PeerJ*, 2, e453.
