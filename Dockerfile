# =============================================================================
# Teloscopy - Multi-Stage Dockerfile
# =============================================================================
# Produces a minimal, secure production image for the Teloscopy gene-sequencing
# web application.  Two stages keep the final image free of build tooling.
#
# Build:  docker build -t teloscopy .
# Run:    docker run -p 8000:8000 teloscopy
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder – compile wheels & install Python packages
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System packages required to build native extensions (numpy, opencv, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgl1 \
        libglib2.0-0 \
        libglib2.0-dev \
        pkg-config \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies first (layer caching optimisation)
COPY requirements*.txt pyproject.toml ./
COPY src/ ./src/

# Build a wheel cache so the runtime stage can install without a compiler
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -e ".[all,webapp]"

# ---------------------------------------------------------------------------
# Stage 2: Runtime – lean image with only what is needed to run the app
# ---------------------------------------------------------------------------
FROM python:3.12-slim

LABEL maintainer="Teloscopy Contributors" \
      description="Teloscopy gene-sequencing analysis platform" \
      version="2.0.0"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime-only system libraries (OpenCV needs libgl1 & libglib2.0-0)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd --gid 1000 teloscopy && \
    useradd --uid 1000 --gid teloscopy --shell /bin/bash --create-home teloscopy

WORKDIR /app

# Copy pre-built wheels from the builder stage and install them
COPY --from=builder /build/wheels /tmp/wheels
COPY --from=builder /build/src ./src
COPY --from=builder /build/pyproject.toml ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links /tmp/wheels /tmp/wheels/*.whl && \
    pip install --no-cache-dir -e ".[all,webapp]" && \
    rm -rf /tmp/wheels

# Create data directories and hand ownership to the non-root user
RUN mkdir -p /app/data/uploads /app/output /app/logs && \
    chown -R teloscopy:teloscopy /app

# Copy any remaining project files (configs, templates, static assets, etc.)
COPY --chown=teloscopy:teloscopy . .

# Drop to non-root user
USER teloscopy

# Expose the default application port
EXPOSE 8000

# Health check – fail if the HTTP server is unreachable for 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as PID 1 to handle signals properly
ENTRYPOINT ["tini", "--"]

# Launch the ASGI server
CMD ["uvicorn", "teloscopy.webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
