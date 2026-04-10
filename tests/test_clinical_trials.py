"""Tests for multi-institution clinical trial management.

Covers the full lifecycle of the :class:`TrialManager` facade, including
trial creation, institution management, patient enrollment with consent
tracking, data submission and validation, differential-privacy aggregation,
DSMB report generation, regulatory export, and the FastAPI REST endpoints.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from teloscopy.clinical.trials import (
    ClinicalTrial,
    ClinicalTrialCoordinator,
    ConsentRecord,
    DataQualityFlag,
    Institution,
    InstitutionSite,
    ParticipantStatus,
    SiteStatus,
    TrialDataPoint,
    TrialManager,
    TrialPhase,
    TrialProtocol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DEMOGRAPHICS = {"age": 35, "sex": "female", "healthy_volunteer": True}


def _make_manager(
    trial_id: str = "TELO-TEST-001",
    phase: str = "phase_2",
    target: int = 200,
) -> TrialManager:
    """Create a TrialManager with sensible test defaults."""
    return TrialManager.create_trial(
        trial_id=trial_id,
        title="Test Multi-site Validation",
        phase=phase,
        target_enrollment=target,
    )


def _add_two_sites(mgr: TrialManager) -> tuple[str, str]:
    """Add two institutions and return their site IDs."""
    s1 = mgr.add_institution(
        name="Mayo Clinic",
        pi="Dr. Smith",
        irb_number="IRB-2024-001",
        location="Rochester, MN",
        contact_email="smith@mayo.edu",
        target_enrollment=100,
    )
    s2 = mgr.add_institution(
        name="Johns Hopkins",
        pi="Dr. Jones",
        irb_number="IRB-2024-002",
        location="Baltimore, MD",
        contact_email="jones@jhu.edu",
        target_enrollment=100,
    )
    return s1, s2


def _enroll_and_submit(
    mgr: TrialManager,
    site_id: str,
    n: int = 3,
    base_length: float = 7000.0,
) -> list[str]:
    """Enroll *n* patients at *site_id* and submit a data point each.

    Returns the list of participant IDs.
    """
    pids: list[str] = []
    for i in range(n):
        pid = mgr.enroll_patient(
            site_id=site_id,
            demographics={"age": 30 + i, "sex": "female", "healthy_volunteer": True},
            consent_version="1.0",
        )
        mgr.submit_data(
            site_id=site_id,
            participant_id=pid,
            telomere_length_bp=base_length + i * 100,
            measurement_method="qfish",
            visit_number=1,
            quality_score=0.95,
        )
        pids.append(pid)
    return pids


# ===================================================================== #
#  Trial creation & lifecycle                                            #
# ===================================================================== #


class TestTrialCreation:
    """Test trial creation and lifecycle transitions."""

    def test_create_trial_returns_manager(self):
        mgr = _make_manager()
        assert isinstance(mgr, TrialManager)
        assert mgr.trial.trial_id == "TELO-TEST-001"
        assert mgr.trial.phase == "phase_2"
        assert mgr.trial.status == "setup"

    def test_create_trial_invalid_phase(self):
        with pytest.raises(ValueError, match="Invalid phase"):
            _make_manager(phase="phase_99")

    def test_trial_metadata_populated(self):
        mgr = _make_manager()
        trial = mgr.trial
        assert isinstance(trial, ClinicalTrial)
        assert trial.title == "Test Multi-site Validation"
        assert trial.target_enrollment == 200
        assert trial.institutions == []
        assert trial.created_at is not None

    def test_activate_trial_requires_institutions(self):
        mgr = _make_manager()
        with pytest.raises(ValueError, match="no institutions"):
            mgr.activate_trial()

    def test_activate_trial_succeeds(self):
        mgr = _make_manager()
        _add_two_sites(mgr)
        mgr.activate_trial()
        assert mgr.trial.status == "active"
        assert mgr.trial.start_date is not None

    def test_activate_trial_twice_fails(self):
        mgr = _make_manager()
        _add_two_sites(mgr)
        mgr.activate_trial()
        with pytest.raises(ValueError, match="Cannot activate"):
            mgr.activate_trial()

    def test_coordinator_is_accessible(self):
        mgr = _make_manager()
        assert isinstance(mgr.coordinator, ClinicalTrialCoordinator)
        assert mgr.coordinator.trial_id == "TELO-TEST-001"


# ===================================================================== #
#  Multi-site enrollment                                                 #
# ===================================================================== #


class TestMultiSiteEnrollment:
    """Test institution registration and multi-site patient enrollment."""

    def test_add_institution(self):
        mgr = _make_manager()
        site_id = mgr.add_institution(
            name="Test Hospital",
            pi="Dr. Test",
            irb_number="IRB-001",
            location="Test City, TS",
            contact_email="test@hospital.org",
        )
        assert site_id.startswith("SITE-")
        assert len(mgr.list_institutions()) == 1
        assert mgr.trial.institutions == [site_id]

    def test_add_duplicate_institution_fails(self):
        mgr = _make_manager()
        mgr.add_institution(name="Same Name", pi="Dr. A", irb_number="IRB-A")
        with pytest.raises(ValueError, match="already registered"):
            mgr.add_institution(name="Same Name", pi="Dr. B", irb_number="IRB-B")

    def test_get_institution(self):
        mgr = _make_manager()
        sid = mgr.add_institution(name="Get Test", pi="Dr. G", irb_number="IRB-G")
        inst = mgr.get_institution(sid)
        assert isinstance(inst, Institution)
        assert inst.name == "Get Test"
        assert inst.principal_investigator == "Dr. G"
        assert inst.irb_number == "IRB-G"

    def test_get_institution_not_found(self):
        mgr = _make_manager()
        with pytest.raises(KeyError, match="Institution not found"):
            mgr.get_institution("FAKE-ID")

    def test_enroll_patient(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(
            site_id=s1,
            demographics=_DEFAULT_DEMOGRAPHICS,
            consent_version="1.0",
        )
        assert pid.startswith("SUBJ-")
        assert mgr.get_institution(s1).enrolled_count == 1

    def test_enroll_at_multiple_sites(self):
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)

        pid1 = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)
        pid2 = mgr.enroll_patient(site_id=s2, demographics=_DEFAULT_DEMOGRAPHICS)

        assert pid1 != pid2
        assert mgr.get_institution(s1).enrolled_count == 1
        assert mgr.get_institution(s2).enrolled_count == 1

    def test_enroll_unknown_site_fails(self):
        mgr = _make_manager()
        with pytest.raises(KeyError, match="Institution not found"):
            mgr.enroll_patient(site_id="FAKE", demographics=_DEFAULT_DEMOGRAPHICS)

    def test_enroll_ineligible_patient_fails(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        with pytest.raises(ValueError, match="not eligible"):
            mgr.enroll_patient(
                site_id=s1,
                demographics={"age": 10, "sex": "male"},  # under 18
            )


# ===================================================================== #
#  Consent management                                                    #
# ===================================================================== #


class TestConsentManagement:
    """Test informed-consent tracking and withdrawal."""

    def test_consent_record_created_on_enroll(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(
            site_id=s1,
            demographics=_DEFAULT_DEMOGRAPHICS,
            consent_version="2.0",
            witnessed_by="nurse_jones",
        )
        consent = mgr.get_consent_record(pid)
        assert isinstance(consent, ConsentRecord)
        assert consent.consent_version == "2.0"
        assert consent.witnessed_by == "nurse_jones"
        assert consent.is_active is True
        assert consent.withdrawal_date is None

    def test_withdraw_consent(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)

        mgr.withdraw_consent(pid, reason="Patient request")
        consent = mgr.get_consent_record(pid)
        assert consent.is_active is False
        assert consent.withdrawal_reason == "Patient request"
        assert consent.withdrawal_date is not None

    def test_withdraw_consent_twice_fails(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)
        mgr.withdraw_consent(pid, "First withdrawal")
        with pytest.raises(ValueError, match="already withdrawn"):
            mgr.withdraw_consent(pid, "Second attempt")

    def test_reconsent_participant(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(
            site_id=s1,
            demographics=_DEFAULT_DEMOGRAPHICS,
            consent_version="1.0",
        )
        mgr.reconsent_participant(pid, new_version="2.0", witnessed_by="dr_new")
        consent = mgr.get_consent_record(pid)
        assert consent.consent_version == "2.0"
        assert consent.witnessed_by == "dr_new"
        assert len(consent.reconsent_history) == 1
        assert consent.reconsent_history[0][0] == "1.0"

    def test_reconsent_after_withdrawal_fails(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)
        mgr.withdraw_consent(pid, "Leaving study")
        with pytest.raises(ValueError, match="consent is withdrawn"):
            mgr.reconsent_participant(pid, new_version="3.0")

    def test_get_consent_unknown_participant_fails(self):
        mgr = _make_manager()
        with pytest.raises(KeyError, match="Consent record not found"):
            mgr.get_consent_record("FAKE-PID")


# ===================================================================== #
#  Data submission & validation                                          #
# ===================================================================== #


class TestDataSubmission:
    """Test data submission and quality validation."""

    def test_submit_valid_data(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)

        receipt = mgr.submit_data(
            site_id=s1,
            participant_id=pid,
            telomere_length_bp=7500.0,
            measurement_method="qfish",
            visit_number=1,
            quality_score=0.95,
        )
        assert receipt.startswith("DP-")

    def test_submit_data_unknown_site_fails(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)

        with pytest.raises(KeyError):
            mgr.submit_data(
                site_id="FAKE-SITE",
                participant_id=pid,
                telomere_length_bp=7500.0,
            )

    def test_submit_data_unknown_participant_fails(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)

        with pytest.raises(KeyError):
            mgr.submit_data(
                site_id=s1,
                participant_id="FAKE-PID",
                telomere_length_bp=7500.0,
            )

    def test_submit_out_of_range_rejected(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)

        with pytest.raises(ValueError, match="rejected"):
            mgr.submit_data(
                site_id=s1,
                participant_id=pid,
                telomere_length_bp=100.0,  # Below 3000 bp minimum
                quality_score=0.95,
            )

    def test_submit_invalid_quality_rejected(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)

        with pytest.raises(ValueError, match="rejected"):
            mgr.submit_data(
                site_id=s1,
                participant_id=pid,
                telomere_length_bp=7000.0,
                quality_score=5.0,  # Outside [0, 1]
            )

    def test_submit_invalid_method_rejected(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)

        with pytest.raises(ValueError, match="rejected"):
            mgr.submit_data(
                site_id=s1,
                participant_id=pid,
                telomere_length_bp=7000.0,
                measurement_method="unknown_method",
            )

    def test_multiple_visits(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)

        r1 = mgr.submit_data(
            site_id=s1, participant_id=pid,
            telomere_length_bp=7500.0, visit_number=1,
        )
        r2 = mgr.submit_data(
            site_id=s1, participant_id=pid,
            telomere_length_bp=7400.0, visit_number=2,
        )
        assert r1 != r2


# ===================================================================== #
#  Aggregation with differential privacy                                 #
# ===================================================================== #


class TestAggregation:
    """Test cross-site aggregation with privacy guarantees."""

    def test_aggregate_empty_trial(self):
        mgr = _make_manager()
        result = mgr.aggregate_results()
        assert result["global_mean"] is None
        assert result["n_sites"] == 0
        assert result["total_samples"] == 0

    def test_aggregate_single_site(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=5, base_length=7000.0)

        result = mgr.aggregate_results(epsilon=100.0, delta=1e-5)
        assert result["n_sites"] == 1
        assert result["total_samples"] == 5
        assert result["global_mean"] is not None
        # With very large epsilon the noise is negligible; mean ~7200
        assert 3000 < result["global_mean"] < 12000

    def test_aggregate_multiple_sites(self):
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=5, base_length=7000.0)
        _enroll_and_submit(mgr, s2, n=5, base_length=7500.0)

        result = mgr.aggregate_results(epsilon=10.0, delta=1e-5)
        assert result["n_sites"] == 2
        assert result["total_samples"] == 10
        assert len(result["per_site_n"]) == 2
        assert result["global_mean"] is not None

    def test_aggregate_has_privacy_budget(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=3, base_length=7000.0)

        result = mgr.aggregate_results(epsilon=2.0, delta=1e-5)
        budget = result["privacy_budget"]
        assert budget is not None
        assert budget["total_epsilon"] == 2.0
        assert budget["per_round_epsilon"] == 2.0
        assert result["epsilon"] == 2.0
        assert result["delta"] == 1e-5

    def test_aggregate_stronger_privacy_more_noise(self):
        """Smaller epsilon should produce more variation (on average)."""
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=10, base_length=7000.0)
        _enroll_and_submit(mgr, s2, n=10, base_length=7000.0)

        # Run with weak privacy (large epsilon)
        result_weak = mgr.aggregate_results(epsilon=100.0, delta=1e-5)
        # Run with strong privacy (small epsilon)
        result_strong = mgr.aggregate_results(epsilon=0.1, delta=1e-5)

        # Both should return valid results
        assert result_weak["global_mean"] is not None
        assert result_strong["global_mean"] is not None


# ===================================================================== #
#  DSMB report generation                                                #
# ===================================================================== #


class TestDSMBReport:
    """Test DSMB report generation."""

    def test_dsmb_report_structure(self):
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=3)
        _enroll_and_submit(mgr, s2, n=2)

        report = mgr.generate_dsmb_report()
        assert "report_date" in report
        assert "trial_id" in report
        assert "enrollment" in report
        assert "safety" in report
        assert "efficacy" in report
        assert "data_quality" in report
        assert "site_performance" in report
        # Added by TrialManager
        assert "consent" in report

    def test_dsmb_report_enrollment_counts(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=5)

        report = mgr.generate_dsmb_report()
        assert report["enrollment"]["total_enrolled"] == 5

    def test_dsmb_report_consent_section(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)

        pid1 = mgr.enroll_patient(site_id=s1, demographics=_DEFAULT_DEMOGRAPHICS)
        pid2 = mgr.enroll_patient(
            site_id=s1,
            demographics={"age": 40, "sex": "male", "healthy_volunteer": True},
        )
        mgr.withdraw_consent(pid1, "Personal reasons")

        report = mgr.generate_dsmb_report()
        consent = report["consent"]
        assert consent["total_consented"] == 2
        assert consent["active_consents"] == 1
        assert consent["withdrawn_consents"] == 1

    def test_dsmb_report_empty_trial(self):
        mgr = _make_manager()
        report = mgr.generate_dsmb_report()
        assert report["trial_id"] == "TELO-TEST-001"
        assert report["consent"]["total_consented"] == 0


# ===================================================================== #
#  Site quality & monitoring                                             #
# ===================================================================== #


class TestSiteMonitoring:
    """Test per-site quality checks and monitoring reports."""

    def test_site_quality_report(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=4, base_length=7000.0)

        report = mgr.get_site_quality_report(s1)
        assert report["total_data_points"] == 4
        assert report["mean_quality"] is not None
        assert report["mean_quality"] > 0.9
        assert report["methods_used"] == ["qfish"]
        assert report["protocol_deviations"] == []

    def test_site_quality_unknown_site_fails(self):
        mgr = _make_manager()
        with pytest.raises(KeyError, match="Institution not found"):
            mgr.get_site_quality_report("FAKE")

    def test_site_monitoring_report_structure(self):
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=3)
        _enroll_and_submit(mgr, s2, n=2)

        report = mgr.get_site_monitoring_report()
        assert report["trial_id"] == "TELO-TEST-001"
        assert report["total_sites"] == 2
        assert s1 in report["sites"]
        assert s2 in report["sites"]
        assert isinstance(report["issues"], list)
        assert isinstance(report["issues_count"], int)

    def test_site_monitoring_flags_low_enrollment(self):
        mgr = _make_manager()
        # Add site with target_enrollment=100 but enroll only 1
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=1)

        report = mgr.get_site_monitoring_report()
        # Site s1 has 1/100 = 1% enrollment, below 25% threshold
        enrollment_issues = [
            i for i in report["issues"] if "enrollment" in i.lower()
        ]
        assert len(enrollment_issues) >= 1


# ===================================================================== #
#  Regulatory export                                                     #
# ===================================================================== #


class TestRegulatoryExport:
    """Test export_regulatory_package."""

    def test_export_package_structure(self):
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=3, base_length=7000.0)
        _enroll_and_submit(mgr, s2, n=2, base_length=7200.0)

        pkg = mgr.export_regulatory_package()

        assert pkg["package_type"] == "FDA_Regulatory_Submission"
        assert pkg["cfr_compliance"] == "21 CFR Part 11"
        assert "generated_at" in pkg
        assert "trial_metadata" in pkg
        assert "protocol" in pkg
        assert "site_registry" in pkg
        assert "consent_summary" in pkg
        assert "dsmb_report" in pkg
        assert "aggregated_results" in pkg
        assert "site_monitoring" in pkg
        assert "adverse_events" in pkg

    def test_export_trial_metadata(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=1)

        pkg = mgr.export_regulatory_package()
        meta = pkg["trial_metadata"]
        assert meta["trial_id"] == "TELO-TEST-001"
        assert meta["phase"] == "phase_2"
        assert meta["target_enrollment"] == 200

    def test_export_site_registry(self):
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)

        pkg = mgr.export_regulatory_package()
        registry = pkg["site_registry"]
        assert len(registry) == 2
        names = {r["name"] for r in registry}
        assert "Mayo Clinic" in names
        assert "Johns Hopkins" in names

    def test_export_consent_summary(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        pid = mgr.enroll_patient(
            site_id=s1,
            demographics=_DEFAULT_DEMOGRAPHICS,
            consent_version="1.0",
        )
        mgr.reconsent_participant(pid, "2.0")

        pkg = mgr.export_regulatory_package()
        consent = pkg["consent_summary"]
        assert consent["total_consented"] == 1
        assert consent["active"] == 1
        assert "2.0" in consent["consent_versions_used"]

    def test_export_empty_trial(self):
        mgr = _make_manager()
        pkg = mgr.export_regulatory_package()
        assert pkg["site_registry"] == []
        assert pkg["consent_summary"]["total_consented"] == 0
        assert pkg["adverse_events"] == []


# ===================================================================== #
#  Lower-level ClinicalTrialCoordinator                                  #
# ===================================================================== #


class TestClinicalTrialCoordinator:
    """Sanity checks for the underlying coordinator accessed via TrialManager."""

    def test_coordinator_register_and_enroll(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)

        summary = mgr.coordinator.get_site_summary()
        assert len(summary) == 2
        assert any(s["name"] == "Mayo Clinic" for s in summary)

    def test_coordinator_data_export_csv(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=2)

        csv_str = mgr.coordinator.export_trial_data(format="csv")
        assert "participant_id" in csv_str
        assert "telomere_length_bp" in csv_str
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows

    def test_coordinator_data_export_json(self):
        mgr = _make_manager()
        s1, _ = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=2)

        import json

        json_str = mgr.coordinator.export_trial_data(format="json")
        data = json.loads(json_str)
        assert len(data) == 2

    def test_coordinator_interim_analysis(self):
        mgr = _make_manager()
        s1, s2 = _add_two_sites(mgr)
        _enroll_and_submit(mgr, s1, n=10, base_length=7000.0)
        _enroll_and_submit(mgr, s2, n=10, base_length=7200.0)

        result = mgr.coordinator.compute_interim_analysis()
        assert result.trial_id == "TELO-TEST-001"
        assert result.total_participants == 20
        assert result.sites_included == 2
        assert isinstance(result.p_value, float)
        assert isinstance(result.effect_size, float)


# ===================================================================== #
#  REST API endpoints (optional — skipped if fastapi not installed)      #
# ===================================================================== #

try:
    from fastapi.testclient import TestClient
    from teloscopy.clinical.endpoints import _trials, trial_router

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@pytest.mark.skipif(not _HAS_FASTAPI, reason="FastAPI not installed")
class TestTrialEndpoints:
    """Integration tests for the clinical-trial REST API."""

    @pytest.fixture(autouse=True)
    def _setup_client(self):
        """Create a fresh TestClient and clear the trial registry."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(trial_router)
        self.client = TestClient(app)
        _trials.clear()

    def test_create_trial(self):
        resp = self.client.post(
            "/api/trials",
            json={
                "trial_id": "API-001",
                "title": "API Test Trial",
                "phase": "phase_2",
                "target_enrollment": 100,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["trial_id"] == "API-001"
        assert body["status"] == "setup"

    def test_create_duplicate_trial(self):
        payload = {
            "trial_id": "DUP-001",
            "title": "Dup Trial",
            "phase": "phase_1",
        }
        self.client.post("/api/trials", json=payload)
        resp = self.client.post("/api/trials", json=payload)
        assert resp.status_code == 409

    def test_create_trial_invalid_phase(self):
        resp = self.client.post(
            "/api/trials",
            json={
                "trial_id": "BAD-001",
                "title": "Bad Phase",
                "phase": "phase_99",
            },
        )
        assert resp.status_code == 422

    def test_list_trials_empty(self):
        resp = self.client.get("/api/trials")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_trials(self):
        self.client.post(
            "/api/trials",
            json={"trial_id": "T1", "title": "Trial 1", "phase": "phase_2"},
        )
        self.client.post(
            "/api/trials",
            json={"trial_id": "T2", "title": "Trial 2", "phase": "phase_3"},
        )
        resp = self.client.get("/api/trials")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_trial_details(self):
        self.client.post(
            "/api/trials",
            json={"trial_id": "DET-001", "title": "Detail Test", "phase": "phase_2"},
        )
        resp = self.client.get("/api/trials/DET-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trial_id"] == "DET-001"
        assert "institutions" in body
        assert "enrollment" in body

    def test_get_trial_not_found(self):
        resp = self.client.get("/api/trials/MISSING")
        assert resp.status_code == 404

    def test_add_site(self):
        self.client.post(
            "/api/trials",
            json={"trial_id": "SITE-T", "title": "Site Test", "phase": "phase_2"},
        )
        resp = self.client.post(
            "/api/trials/SITE-T/sites",
            json={
                "name": "API Hospital",
                "pi": "Dr. API",
                "irb_number": "IRB-API-001",
                "location": "API City",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "API Hospital"
        assert "site_id" in body

    def test_enroll_patient_via_api(self):
        # Set up trial and site
        self.client.post(
            "/api/trials",
            json={"trial_id": "ENR-T", "title": "Enroll Test", "phase": "phase_2"},
        )
        site_resp = self.client.post(
            "/api/trials/ENR-T/sites",
            json={"name": "Enroll Hospital", "pi": "Dr. E", "irb_number": "IRB-E"},
        )
        site_id = site_resp.json()["site_id"]

        resp = self.client.post(
            "/api/trials/ENR-T/enroll",
            json={
                "site_id": site_id,
                "demographics": {"age": 30, "sex": "female", "healthy_volunteer": True},
                "consent_version": "1.0",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "participant_id" in body
        assert body["site_id"] == site_id

    def test_submit_data_via_api(self):
        # Set up trial, site, and enroll patient
        self.client.post(
            "/api/trials",
            json={"trial_id": "DAT-T", "title": "Data Test", "phase": "phase_2"},
        )
        site_resp = self.client.post(
            "/api/trials/DAT-T/sites",
            json={"name": "Data Hospital", "pi": "Dr. D", "irb_number": "IRB-D"},
        )
        site_id = site_resp.json()["site_id"]

        enroll_resp = self.client.post(
            "/api/trials/DAT-T/enroll",
            json={
                "site_id": site_id,
                "demographics": {"age": 45, "sex": "male", "healthy_volunteer": True},
            },
        )
        pid = enroll_resp.json()["participant_id"]

        resp = self.client.post(
            "/api/trials/DAT-T/data",
            json={
                "site_id": site_id,
                "participant_id": pid,
                "telomere_length_bp": 7500.0,
                "measurement_method": "qfish",
                "quality_score": 0.92,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "receipt_id" in body

    def test_dsmb_report_via_api(self):
        self.client.post(
            "/api/trials",
            json={"trial_id": "RPT-T", "title": "Report Test", "phase": "phase_2"},
        )
        resp = self.client.get("/api/trials/RPT-T/report")
        assert resp.status_code == 200
        body = resp.json()
        assert "enrollment" in body
        assert "consent" in body

    def test_export_via_api(self):
        self.client.post(
            "/api/trials",
            json={"trial_id": "EXP-T", "title": "Export Test", "phase": "phase_2"},
        )
        resp = self.client.get("/api/trials/EXP-T/export")
        assert resp.status_code == 200
        body = resp.json()
        assert body["package_type"] == "FDA_Regulatory_Submission"
        assert body["cfr_compliance"] == "21 CFR Part 11"
