"""Tests for the Teloscopy FastAPI web application."""

from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# FastAPI and related dependencies are optional; skip the entire module
# gracefully if they are not installed.
try:
    from fastapi.testclient import TestClient

    from teloscopy.webapp.app import app
    from teloscopy.webapp.models import (
        AgentStatusEnum,
        JobStatusEnum,
    )

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(
    not _HAS_FASTAPI,
    reason="FastAPI or its dependencies are not installed.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Return a synchronous TestClient for the FastAPI app."""
    return TestClient(app)


def _dummy_image_bytes(filename: str = "test.tif") -> tuple[str, io.BytesIO, str]:
    """Return a minimal fake image file for upload tests."""
    content = b"\x00" * 100  # not a real TIFF, but sufficient for upload validation
    return (filename, io.BytesIO(content), "image/tiff")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        """Health check should return HTTP 200."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_body_structure(self, client):
        """Response should contain status, version, and timestamp."""
        data = client.get("/api/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    """Tests for POST /api/upload."""

    def test_upload_valid_image(self, client):
        """Uploading a .tif file should return 201 with job_id."""
        resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("sample.tif")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data
        assert data["filename"] == "sample.tif"

    def test_upload_invalid_extension(self, client):
        """A disallowed extension should return 400."""
        resp = client.post(
            "/api/upload",
            files={"file": ("bad.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_png(self, client):
        """PNG is an allowed extension."""
        resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("sample.png")},
        )
        assert resp.status_code == 201

    def test_upload_jpg(self, client):
        """JPG is an allowed extension."""
        resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("photo.jpg")},
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    """Tests for GET /api/status/{job_id}."""

    def test_nonexistent_job_returns_404(self, client):
        """Querying a random job_id should return 404."""
        resp = client.get("/api/status/nonexistent-job-id-12345")
        assert resp.status_code == 404

    def test_existing_job_returns_200(self, client):
        """After uploading, the job should be queryable."""
        upload_resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("test.tif")},
        )
        job_id = upload_resp.json()["job_id"]

        status_resp = client.get(f"/api/status/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in [e.value for e in JobStatusEnum]


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    """Tests for POST /api/analyze."""

    def test_analyze_valid_input(self, client):
        """A valid multipart form should return 202 (accepted)."""
        resp = client.post(
            "/api/analyze",
            files={"file": _dummy_image_bytes("analysis.tif")},
            data={
                "age": "45",
                "sex": "male",
                "region": "Northern Europe",
                "dietary_restrictions": "vegetarian,gluten_free",
                "known_variants": "rs429358,rs7903146",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == JobStatusEnum.PENDING.value

    def test_analyze_invalid_file(self, client):
        """An invalid file type should return 400."""
        resp = client.post(
            "/api/analyze",
            files={"file": ("bad.pdf", io.BytesIO(b"pdf"), "application/pdf")},
            data={
                "age": "30",
                "sex": "female",
                "region": "East Asia",
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Disease risk endpoint
# ---------------------------------------------------------------------------


class TestDiseaseRiskEndpoint:
    """Tests for POST /api/disease-risk."""

    def test_disease_risk_valid(self, client):
        """A valid request should return risk scores."""
        payload = {
            "known_variants": ["rs429358", "rs7903146"],
            "telomere_length": 6.5,
            "age": 55,
            "sex": "male",
            "region": "Western Europe",
        }
        resp = client.post("/api/disease-risk", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "risks" in data
        assert "overall_risk_score" in data
        assert isinstance(data["risks"], list)
        assert len(data["risks"]) > 0

    def test_disease_risk_minimal(self, client):
        """Request with no variants should still succeed."""
        payload = {
            "known_variants": [],
            "age": 30,
            "sex": "female",
            "region": "Global",
        }
        resp = client.post("/api/disease-risk", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Diet plan endpoint
# ---------------------------------------------------------------------------


class TestDietPlanEndpoint:
    """Tests for POST /api/diet-plan."""

    def test_diet_plan_valid(self, client):
        """A valid request should return a diet recommendation."""
        payload = {
            "age": 40,
            "sex": "female",
            "region": "Mediterranean",
            "dietary_restrictions": ["vegetarian"],
            "known_variants": [],
        }
        resp = client.post("/api/diet-plan", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendation" in data
        rec = data["recommendation"]
        assert "summary" in rec
        assert "key_nutrients" in rec
        assert "meal_plans" in rec

    def test_diet_plan_with_risks(self, client):
        """A request including disease_risks should succeed."""
        payload = {
            "age": 60,
            "sex": "male",
            "region": "East Asia",
            "dietary_restrictions": [],
            "known_variants": ["rs429358"],
            "disease_risks": [
                {
                    "disease": "Cardiovascular Disease",
                    "risk_level": "high",
                    "probability": 0.65,
                    "contributing_factors": ["APOE"],
                    "recommendations": ["Exercise"],
                }
            ],
        }
        resp = client.post("/api/diet-plan", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Agents status endpoint
# ---------------------------------------------------------------------------


class TestAgentsStatusEndpoint:
    """Tests for GET /api/agents/status."""

    def test_agents_status_returns_200(self, client):
        """Agents status endpoint should return 200."""
        resp = client.get("/api/agents/status")
        assert resp.status_code == 200

    def test_agents_status_structure(self, client):
        """Response should contain a list of agents and aggregate metrics."""
        data = client.get("/api/agents/status").json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
        assert len(data["agents"]) >= 4
        assert "total_analyses" in data
        assert "active_jobs" in data
        assert "uptime_seconds" in data

    def test_agent_info_fields(self, client):
        """Each agent entry should have name and status."""
        data = client.get("/api/agents/status").json()
        for agent in data["agents"]:
            assert "name" in agent
            assert "status" in agent
            assert agent["status"] in [e.value for e in AgentStatusEnum]


# ---------------------------------------------------------------------------
# Results endpoint
# ---------------------------------------------------------------------------


class TestResultsEndpoint:
    """Tests for GET /api/results/{job_id}."""

    def test_results_nonexistent_returns_404(self, client):
        """A non-existent job should return 404."""
        resp = client.get("/api/results/does-not-exist")
        assert resp.status_code == 404

    def test_results_pending_returns_409(self, client):
        """A pending (not completed) job should return 409."""
        upload_resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("test.tif")},
        )
        job_id = upload_resp.json()["job_id"]
        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 409
