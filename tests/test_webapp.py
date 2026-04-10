"""Tests for the Teloscopy FastAPI web application."""

from __future__ import annotations

import io
import os
import sys
import uuid

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


def _obtain_consent(client: TestClient) -> str:
    """Submit full consent and return the consent token."""
    purposes = [
        "telomere_analysis", "disease_risk", "nutrition_plan",
        "facial_analysis", "health_report", "genetic_data", "profile_data",
    ]
    bundle = {
        "session_id": str(uuid.uuid4()),
        "consents": [{"purpose": p, "granted": True, "notice_version": "1.0"} for p in purposes],
        "data_principal_age_confirmed": True,
        "privacy_policy_version": "1.0",
        "terms_version": "1.0",
    }
    resp = client.post("/api/legal/consent", json=bundle)
    assert resp.status_code == 200
    return resp.json()["consent_token"]


def _ch(token: str) -> dict[str, str]:
    """Return consent headers dict."""
    return {"X-Consent-Token": token}


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
        token = _obtain_consent(client)
        resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("sample.tif")},
            headers=_ch(token),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "job_id" in data
        assert data["filename"] == "sample.tif"

    def test_upload_invalid_extension(self, client):
        """A disallowed extension should return 400."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/upload",
            files={"file": ("bad.txt", io.BytesIO(b"hello"), "text/plain")},
            headers=_ch(token),
        )
        assert resp.status_code == 400

    def test_upload_png(self, client):
        """PNG is an allowed extension."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("sample.png")},
            headers=_ch(token),
        )
        assert resp.status_code == 201

    def test_upload_jpg(self, client):
        """JPG is an allowed extension."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("photo.jpg")},
            headers=_ch(token),
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
        token = _obtain_consent(client)
        upload_resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("test.tif")},
            headers=_ch(token),
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
        token = _obtain_consent(client)
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
            headers=_ch(token),
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == JobStatusEnum.PENDING.value

    def test_analyze_invalid_file(self, client):
        """An invalid file type should return 400."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/analyze",
            files={"file": ("bad.pdf", io.BytesIO(b"pdf"), "application/pdf")},
            data={
                "age": "30",
                "sex": "female",
                "region": "East Asia",
            },
            headers=_ch(token),
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Disease risk endpoint
# ---------------------------------------------------------------------------


class TestDiseaseRiskEndpoint:
    """Tests for POST /api/disease-risk."""

    def test_disease_risk_valid(self, client):
        """A valid request should return risk scores."""
        token = _obtain_consent(client)
        payload = {
            "known_variants": ["rs429358", "rs7903146"],
            "telomere_length": 6.5,
            "age": 55,
            "sex": "male",
            "region": "Western Europe",
        }
        resp = client.post("/api/disease-risk", json=payload, headers=_ch(token))
        assert resp.status_code == 200
        data = resp.json()
        assert "risks" in data
        assert "overall_risk_score" in data
        assert isinstance(data["risks"], list)
        assert len(data["risks"]) > 0

    def test_disease_risk_minimal(self, client):
        """Request with no variants should still succeed."""
        token = _obtain_consent(client)
        payload = {
            "known_variants": [],
            "age": 30,
            "sex": "female",
            "region": "Global",
        }
        resp = client.post("/api/disease-risk", json=payload, headers=_ch(token))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Diet plan endpoint
# ---------------------------------------------------------------------------


class TestDietPlanEndpoint:
    """Tests for POST /api/diet-plan."""

    def test_diet_plan_valid(self, client):
        """A valid request should return a diet recommendation."""
        token = _obtain_consent(client)
        payload = {
            "age": 40,
            "sex": "female",
            "region": "Mediterranean",
            "dietary_restrictions": ["vegetarian"],
            "known_variants": [],
        }
        resp = client.post("/api/diet-plan", json=payload, headers=_ch(token))
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendation" in data
        rec = data["recommendation"]
        assert "summary" in rec
        assert "key_nutrients" in rec
        assert "meal_plans" in rec

    def test_diet_plan_with_risks(self, client):
        """A request including disease_risks should succeed."""
        token = _obtain_consent(client)
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
        resp = client.post("/api/diet-plan", json=payload, headers=_ch(token))
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
        token = _obtain_consent(client)
        resp = client.get("/api/results/does-not-exist", headers=_ch(token))
        assert resp.status_code == 404

    def test_results_pending_returns_409(self, client):
        """A pending (not completed) job should return 409."""
        token = _obtain_consent(client)
        upload_resp = client.post(
            "/api/upload",
            files={"file": _dummy_image_bytes("test.tif")},
            headers=_ch(token),
        )
        job_id = upload_resp.json()["job_id"]
        resp = client.get(f"/api/results/{job_id}", headers=_ch(token))
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Health checkup report upload endpoints
# ---------------------------------------------------------------------------


def _make_pdf_bytes():
    """Create a minimal test PDF with lab values using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        return None
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Lab Report\n"
        "Hemoglobin: 14.5 g/dL\n"
        "Fasting Glucose: 95 mg/dL\n"
        "Total Cholesterol: 210 mg/dL\n"
        "Serum Creatinine: 0.9 mg/dL\n"
        "TSH: 2.5 mIU/L\n"
    )
    page.insert_text((72, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestHealthCheckupParseReport:
    """Tests for POST /api/health-checkup/parse-report."""

    def test_parse_pdf_report(self, client):
        """Uploading a PDF lab report should extract values."""
        pdf_bytes = _make_pdf_bytes()
        if pdf_bytes is None:
            pytest.skip("PyMuPDF not available")
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup/parse-report",
            files={"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers=_ch(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_type"] == "pdf"
        assert data["text_length"] > 0
        blood = data["extracted_blood_tests"]
        assert "hemoglobin" in blood
        assert blood["hemoglobin"] == 14.5

    def test_parse_text_report(self, client):
        """Uploading a plain text lab report should extract values."""
        text_content = (
            "Lab Report\n"
            "Hemoglobin: 13.2 g/dL\n"
            "WBC Count: 7500 cells/mcL\n"
            "Platelet Count: 230000 /mcL\n"
            "HDL Cholesterol: 60 mg/dL\n"
        ).encode("utf-8")
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup/parse-report",
            files={"file": ("report.txt", io.BytesIO(text_content), "text/plain")},
            headers=_ch(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_type"] == "text"
        blood = data["extracted_blood_tests"]
        assert "hemoglobin" in blood
        assert blood["hemoglobin"] == 13.2

    def test_parse_empty_file_returns_zero_confidence(self, client):
        """An empty file should return 0% confidence."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup/parse-report",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
            headers=_ch(token),
        )
        assert resp.status_code == 400  # empty file check

    def test_parse_unsupported_extension(self, client):
        """An unsupported file type should return 400."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup/parse-report",
            files={"file": ("report.docx", io.BytesIO(b"fake"), "application/vnd.openxmlformats")},
            headers=_ch(token),
        )
        assert resp.status_code == 400

    def test_parse_requires_consent(self, client):
        """Parse endpoint should require consent token."""
        resp = client.post(
            "/api/health-checkup/parse-report",
            files={"file": ("report.txt", io.BytesIO(b"test"), "text/plain")},
        )
        assert resp.status_code == 403


class TestHealthCheckupManual:
    """Tests for POST /api/health-checkup (manual JSON entry)."""

    def test_rejects_empty_lab_data(self, client):
        """Submitting with no blood or urine tests should return 422."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup",
            json={
                "age": 30,
                "sex": "male",
                "region": "Global",
            },
            headers={**_ch(token), "Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        body = resp.json()
        msg = body.get("detail", "") or body.get("error", {}).get("message", "")
        assert "No lab values" in msg

    def test_rejects_all_none_blood_panel(self, client):
        """BloodTestPanel with all None values should still be rejected."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup",
            json={
                "age": 30,
                "sex": "male",
                "region": "Global",
                "blood_tests": {},
                "urine_tests": {},
            },
            headers={**_ch(token), "Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_accepts_valid_lab_data(self, client):
        """Submitting with at least one lab value should succeed."""
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup",
            json={
                "age": 42,
                "sex": "male",
                "region": "South Asia",
                "blood_tests": {
                    "hemoglobin": 14.5,
                    "fasting_glucose": 95,
                    "total_cholesterol": 200,
                },
            },
            headers={**_ch(token), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tested"] >= 3
        assert data["overall_health_score"] > 0


class TestHealthCheckupUpload:
    """Tests for POST /api/health-checkup/upload."""

    def test_upload_pdf_and_analyse(self, client):
        """Uploading a PDF with profile data should return full analysis."""
        pdf_bytes = _make_pdf_bytes()
        if pdf_bytes is None:
            pytest.skip("PyMuPDF not available")
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup/upload",
            files={"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={
                "age": "45",
                "sex": "male",
                "region": "South Asia",
                "dietary_restrictions": "",
                "known_variants": "",
                "calorie_target": "2000",
                "meal_plan_days": "7",
            },
            headers=_ch(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "lab_results" in data
        assert "overall_health_score" in data

    def test_upload_text_report_and_analyse(self, client):
        """Text file upload with profile should work."""
        text_content = (
            "Hemoglobin: 12.0 g/dL\n"
            "Fasting Glucose: 110 mg/dL\n"
            "Total Cholesterol: 250 mg/dL\n"
        ).encode("utf-8")
        token = _obtain_consent(client)
        resp = client.post(
            "/api/health-checkup/upload",
            files={"file": ("report.txt", io.BytesIO(text_content), "text/plain")},
            data={
                "age": "50",
                "sex": "female",
                "region": "Global",
            },
            headers=_ch(token),
        )
        assert resp.status_code == 200

    def test_upload_requires_consent(self, client):
        """Upload endpoint should require consent token."""
        resp = client.post(
            "/api/health-checkup/upload",
            files={"file": ("report.txt", io.BytesIO(b"test"), "text/plain")},
            data={"age": "30", "sex": "male", "region": "Global"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Android APK download
# ---------------------------------------------------------------------------


class TestAndroidDownload:
    """Tests for Android APK download endpoint."""

    def test_apk_download_returns_file(self, client):
        """GET /api/download/android should serve a file."""
        resp = client.get("/api/download/android")
        assert resp.status_code == 200
        assert "application/vnd.android.package-archive" in resp.headers.get("content-type", "")

    def test_apk_status_shows_size(self, client):
        """GET /api/download/android/status should show file info."""
        resp = client.get("/api/download/android/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["size_bytes"] is not None
        assert data["size_bytes"] > 0


# ---------------------------------------------------------------------------
# Legal document pages
# ---------------------------------------------------------------------------


class TestLegalPages:
    """Tests for Privacy Policy and Terms of Service pages."""

    def test_privacy_policy_returns_html(self, client):
        """GET /docs/privacy-policy should return styled HTML with content."""
        resp = client.get("/docs/privacy-policy")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<h1>" in resp.text
        assert "Privacy Policy" in resp.text
        assert len(resp.text) > 1000

    def test_terms_of_service_returns_html(self, client):
        """GET /docs/terms-of-service should return styled HTML with content."""
        resp = client.get("/docs/terms-of-service")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<h1>" in resp.text
        assert "Terms of Service" in resp.text
        assert len(resp.text) > 1000
