#!/usr/bin/env bash
# =============================================================================
# Teloscopy – One-Click Setup Script
# =============================================================================
# Installs Teloscopy and all dependencies, generates sample data, and
# optionally starts the web server.
#
# Usage (remote):
#   curl -sSL https://raw.githubusercontent.com/Mahesh2023/teloscopy/main/setup.sh | bash
#
# Usage (local):
#   chmod +x setup.sh && ./setup.sh
#
# Options:
#   --docker    Force Docker installation path
#   --local     Force local (virtualenv) installation path
#   --no-test   Skip the verification test suite
#   --yes       Accept all defaults non-interactively
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers & UI
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

info()    { echo -e "${BLUE}[ℹ]${NC} $*"; }
success() { echo -e "${GREEN}[✔]${NC} $*"; }
warn()    { echo -e "${YELLOW}[⚠]${NC} $*"; }
error()   { echo -e "${RED}[✖]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}\n"; }

spinner() {
    local pid=$1 msg=$2
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ${BLUE}%s${NC} %s" "${spin:i++%${#spin}:1}" "$msg"
        sleep 0.1
    done
    printf "\r"
}

# ---------------------------------------------------------------------------
# Defaults & argument parsing
# ---------------------------------------------------------------------------
INSTALL_MODE=""       # "docker" | "local" – empty = auto-detect
SKIP_TESTS=false
NON_INTERACTIVE=false
VENV_DIR=".venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11
REQUIRED_PYTHON="${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker)  INSTALL_MODE="docker";  shift ;;
        --local)   INSTALL_MODE="local";   shift ;;
        --no-test) SKIP_TESTS=true;        shift ;;
        --yes|-y)  NON_INTERACTIVE=true;   shift ;;
        -h|--help)
            echo "Usage: $0 [--docker|--local] [--no-test] [--yes]"
            exit 0
            ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
command_exists() { command -v "$1" &>/dev/null; }

prompt_yn() {
    # prompt_yn "Question?" default(y/n)
    if $NON_INTERACTIVE; then
        [[ "${2:-y}" == "y" ]]
        return
    fi
    local default="${2:-y}"
    local prompt
    if [[ "$default" == "y" ]]; then prompt="[Y/n]"; else prompt="[y/N]"; fi
    read -rp "$(echo -e "${YELLOW}[?]${NC} $1 ${prompt} ")" answer
    answer="${answer:-$default}"
    [[ "$answer" =~ ^[Yy] ]]
}

check_python_version() {
    local py_cmd="$1"
    if ! command_exists "$py_cmd"; then return 1; fi
    local version
    version=$("$py_cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || return 1
    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    (( major > MIN_PYTHON_MAJOR || (major == MIN_PYTHON_MAJOR && minor >= MIN_PYTHON_MINOR) ))
}

find_python() {
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if check_python_version "$cmd"; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}"
cat << 'BANNER'

  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   ████████╗███████╗██╗      ██████╗ ███████╗ ██████╗██████╗  ║
  ║      ██╔══╝██╔════╝██║     ██╔═══██╗██╔════╝██╔════╝██╔══██╗ ║
  ║      ██║   █████╗  ██║     ██║   ██║███████╗██║     ██████╔╝ ║
  ║      ██║   ██╔══╝  ██║     ██║   ██║╚════██║██║     ██╔═══╝  ║
  ║      ██║   ███████╗███████╗╚██████╔╝███████║╚██████╗██║      ║
  ║      ╚═╝   ╚══════╝╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝      ║
  ║                                                              ║
  ║          Telomere Sequence Analysis Platform                 ║
  ║              One-Click Setup Installer                       ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝

BANNER
echo -e "${NC}"

# ---------------------------------------------------------------------------
# Step 0: Pre-flight checks
# ---------------------------------------------------------------------------
header "Pre-flight Checks"

HAS_DOCKER=false
HAS_COMPOSE=false

if command_exists docker && docker info &>/dev/null; then
    HAS_DOCKER=true
    success "Docker detected ($(docker --version | head -1))"
else
    info "Docker not detected or not running"
fi

if $HAS_DOCKER && (command_exists "docker-compose" || docker compose version &>/dev/null 2>&1); then
    HAS_COMPOSE=true
    success "Docker Compose detected"
else
    info "Docker Compose not detected"
fi

# ---------------------------------------------------------------------------
# Step 1: Choose installation path
# ---------------------------------------------------------------------------
header "Installation Path"

if [[ -z "$INSTALL_MODE" ]]; then
    if $HAS_DOCKER && $HAS_COMPOSE; then
        info "Docker is available. Docker builds are reproducible and isolated."
        if prompt_yn "Install with Docker? (recommended)" "y"; then
            INSTALL_MODE="docker"
        else
            INSTALL_MODE="local"
        fi
    else
        info "Docker not available – using local installation."
        INSTALL_MODE="local"
    fi
fi

success "Installation mode: ${BOLD}${INSTALL_MODE}${NC}"

# ═══════════════════════════════════════════════════════════════════════════
# Docker Installation Path
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$INSTALL_MODE" == "docker" ]]; then
    header "Docker Setup"

    # Create local directories for bind mounts
    info "Creating data directories …"
    mkdir -p data/uploads output logs
    success "Directories created: data/ output/ logs/"

    # Copy .env.example → .env if it doesn't exist
    if [[ -f .env.example ]] && [[ ! -f .env ]]; then
        cp .env.example .env
        success "Created .env from .env.example"
    fi

    # Build the image
    info "Building Docker image (this may take a few minutes) …"
    if docker compose build 2>&1 | tail -5; then
        success "Docker image built successfully"
    else
        error "Docker build failed. Check the output above."
        exit 1
    fi

    # Start the service
    if prompt_yn "Start Teloscopy now?" "y"; then
        info "Starting services …"
        docker compose up -d
        success "Teloscopy is running!"
        echo ""
        echo -e "  ${GREEN}➜${NC}  Web UI:   ${BOLD}http://localhost:8000${NC}"
        echo -e "  ${GREEN}➜${NC}  Logs:     ${BOLD}docker compose logs -f${NC}"
        echo -e "  ${GREEN}➜${NC}  Stop:     ${BOLD}docker compose down${NC}"
        echo ""
    else
        info "You can start later with: ${BOLD}docker compose up -d${NC}"
    fi

    success "Docker setup complete!"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════
# Local Installation Path
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Step 2: Check Python version
# ---------------------------------------------------------------------------
header "Python Environment"

PYTHON_CMD=$(find_python) || {
    error "Python >= ${REQUIRED_PYTHON} is required but was not found."
    echo ""
    echo "  Install Python ${REQUIRED_PYTHON}+ from:"
    echo "    • https://www.python.org/downloads/"
    echo "    • brew install python@3.12   (macOS)"
    echo "    • sudo apt install python3.12 (Debian/Ubuntu)"
    echo ""
    exit 1
}

PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
success "Found Python ${PYTHON_VERSION} (${PYTHON_CMD})"

# ---------------------------------------------------------------------------
# Step 3: Create virtual environment
# ---------------------------------------------------------------------------
header "Virtual Environment"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    info "Already inside a virtualenv: ${VIRTUAL_ENV}"
    VENV_PYTHON="python"
elif [[ -d "$VENV_DIR" ]]; then
    info "Existing virtualenv found at ${VENV_DIR}/"
    if prompt_yn "Re-use it?" "y"; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        success "Activated ${VENV_DIR}"
        VENV_PYTHON="python"
    else
        rm -rf "$VENV_DIR"
        info "Creating fresh virtualenv …"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        success "Created & activated ${VENV_DIR}"
        VENV_PYTHON="python"
    fi
else
    info "Creating virtualenv in ${VENV_DIR}/ …"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    success "Created & activated ${VENV_DIR}"
    VENV_PYTHON="python"
fi

# ---------------------------------------------------------------------------
# Step 4: Install dependencies
# ---------------------------------------------------------------------------
header "Installing Dependencies"

info "Upgrading pip, setuptools, wheel …"
$VENV_PYTHON -m pip install --upgrade pip setuptools wheel --quiet &
spinner $! "Upgrading pip …"
success "pip upgraded"

info "Installing Teloscopy with all extras …"
if [[ -f pyproject.toml ]] || [[ -f setup.py ]]; then
    $VENV_PYTHON -m pip install -e ".[all,webapp,dev]" --quiet 2>&1 | tail -3 &
    spinner $! "Installing packages (this may take a minute) …"
    success "All dependencies installed"
else
    warn "No pyproject.toml or setup.py found – installing from requirements.txt"
    if [[ -f requirements.txt ]]; then
        $VENV_PYTHON -m pip install -r requirements.txt --quiet &
        spinner $! "Installing packages …"
        success "Requirements installed"
    else
        error "No installable package configuration found!"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: Create data directories & .env
# ---------------------------------------------------------------------------
header "Project Configuration"

mkdir -p data/uploads output logs
success "Created data directories: data/ output/ logs/"

if [[ -f .env.example ]] && [[ ! -f .env ]]; then
    cp .env.example .env
    success "Created .env from .env.example"
elif [[ ! -f .env ]]; then
    info "No .env.example found – skipping .env creation"
fi

# ---------------------------------------------------------------------------
# Step 6: Generate sample data
# ---------------------------------------------------------------------------
header "Sample Data Generation"

if command_exists teloscopy; then
    info "Generating sample telomere sequencing data …"
    if teloscopy generate 2>&1 | tail -3; then
        success "Sample data generated in data/"
    else
        warn "Sample data generation encountered issues (non-critical)"
    fi
else
    info "CLI not found on PATH – trying via Python module …"
    if $VENV_PYTHON -m teloscopy generate 2>&1 | tail -3; then
        success "Sample data generated"
    else
        warn "Could not generate sample data (the CLI may not be installed yet)"
    fi
fi

# ---------------------------------------------------------------------------
# Step 7: Run verification tests
# ---------------------------------------------------------------------------
if ! $SKIP_TESTS; then
    header "Verification Tests"

    info "Running test suite …"
    if $VENV_PYTHON -m pytest tests/ -v --tb=short 2>&1 | tail -20; then
        success "All tests passed!"
    else
        warn "Some tests failed – the installation may still be usable."
        warn "Run 'make test' to see full output."
    fi
else
    info "Tests skipped (--no-test flag)."
fi

# ---------------------------------------------------------------------------
# Step 8: Done!
# ---------------------------------------------------------------------------
header "Setup Complete!"

echo -e "${GREEN}${BOLD}"
cat << 'DONE'
  ┌─────────────────────────────────────────────────────────┐
  │           Teloscopy installed successfully!              │
  └─────────────────────────────────────────────────────────┘
DONE
echo -e "${NC}"

echo -e "  ${CYAN}Quick Start:${NC}"
echo ""
echo -e "    ${GREEN}1.${NC} Activate the virtualenv (if not already):"
echo -e "       ${BOLD}source ${VENV_DIR}/bin/activate${NC}"
echo ""
echo -e "    ${GREEN}2.${NC} Start the web application:"
echo -e "       ${BOLD}make run${NC}  or  ${BOLD}uvicorn teloscopy.webapp.app:app --reload${NC}"
echo ""
echo -e "    ${GREEN}3.${NC} Open in your browser:"
echo -e "       ${BOLD}http://localhost:8000${NC}"
echo ""
echo -e "  ${CYAN}Other useful commands:${NC}"
echo -e "    ${BOLD}make test${NC}           Run the test suite"
echo -e "    ${BOLD}make lint${NC}           Lint the codebase"
echo -e "    ${BOLD}make generate${NC}       Generate sample data"
echo -e "    ${BOLD}make docker-run${NC}     Run via Docker"
echo -e "    ${BOLD}make help${NC}           Show all Make targets"
echo ""

# Optionally start the server right away
if prompt_yn "Start the web server now?" "n"; then
    header "Starting Teloscopy"
    info "Server starting at http://localhost:8000  (Ctrl+C to stop)"
    $VENV_PYTHON -m uvicorn teloscopy.webapp.app:app --host 0.0.0.0 --port 8000 --reload
fi
