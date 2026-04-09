# =============================================================================
# Teloscopy – Project Makefile
# =============================================================================
# One-click commands for development, testing, and deployment.
#
#   make help          Show all available targets
#   make install       Install the project for local use
#   make docker-run    Build & start via Docker Compose
# =============================================================================

.DEFAULT_GOAL := help
SHELL         := /bin/bash

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
APP_NAME      := teloscopy
PYTHON        := python3
PIP           := pip
DOCKER        := docker
COMPOSE       := docker compose
UVICORN       := uvicorn
PYTEST        := pytest
RUFF          := ruff
PORT          := 8000

# Detect whether we are inside a virtualenv
VENV_ACTIVE   := $(if $(VIRTUAL_ENV),yes,no)

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Install the project with all extras (production)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[all,webapp]"

.PHONY: install-dev
install-dev: ## Install the project with dev/test dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[all,webapp,dev]"
	@echo "✔ Dev dependencies installed"

.PHONY: venv
venv: ## Create a virtual environment in .venv/
	$(PYTHON) -m venv .venv
	@echo "Activate with:  source .venv/bin/activate"

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

.PHONY: run
run: ## Start the web app with hot-reload (development)
	$(UVICORN) $(APP_NAME).webapp.app:app --host 0.0.0.0 --port $(PORT) --reload

.PHONY: run-prod
run-prod: ## Start the web app in production mode
	$(UVICORN) $(APP_NAME).webapp.app:app --host 0.0.0.0 --port $(PORT) --workers 4

.PHONY: generate
generate: ## Generate sample telomere sequencing data
	$(APP_NAME) generate

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build the Docker image
	$(DOCKER) build -t $(APP_NAME):latest .

.PHONY: docker-run
docker-run: ## Build & start services via Docker Compose
	$(COMPOSE) up --build -d
	@echo "✔ Teloscopy is running at http://localhost:$(PORT)"

.PHONY: docker-stop
docker-stop: ## Stop Docker Compose services
	$(COMPOSE) down

.PHONY: docker-logs
docker-logs: ## Tail Docker Compose logs
	$(COMPOSE) logs -f

.PHONY: docker-clean
docker-clean: ## Remove containers, images, and volumes
	$(COMPOSE) down --rmi all --volumes --remove-orphans

# ---------------------------------------------------------------------------
# Quality & Testing
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run the test suite with pytest
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --tb=short --cov=$(APP_NAME) --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Lint source code with ruff
	$(RUFF) check src/ tests/

.PHONY: lint-fix
lint-fix: ## Auto-fix lint issues
	$(RUFF) check --fix src/ tests/

.PHONY: format
format: ## Format source code with ruff
	$(RUFF) format src/ tests/

.PHONY: typecheck
typecheck: ## Run mypy type checking
	mypy src/$(APP_NAME)

.PHONY: check
check: lint typecheck test ## Run all quality checks (lint + types + tests)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove build artefacts, caches, and temp files
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "✔ Clean complete"

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "  Teloscopy – Available Make Targets"
	@echo "  ───────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
