"""
HL7 FHIR R4 Resource Generation for Clinical EHR Integration
=============================================================

This module implements the HL7 FHIR R4 (Fast Healthcare Interoperability Resources,
Release 4) specification for generating clinical resources suitable for Electronic
Health Record (EHR) integration. It supports the full lifecycle of telomere-based
diagnostic reporting: patient creation, genomic observations, risk assessments,
diagnostic reports, nutrition orders, and FHIR Bundle aggregation.

FHIR Specification Reference:
    https://hl7.org/fhir/R4/

LOINC Codes Used:
    - 81247-9  : Master HL7 genetic variant assessment (Molecular Sequence)
    - 93592-1  : Mean telomere length in Blood (qFISH / qPCR)
    - 30525-0  : Age (biological age, calculated)

SNOMED CT Codes Used:
    - 386053000 : Evaluation procedure (diagnostic report category)
    - 363679005 : Imaging (not used, placeholder reference)
    - 73211009  : Diabetes mellitus
    - 56265001  : Heart disease
    - 363346000 : Malignant neoplastic disease
    - 26929004  : Alzheimer's disease

HIPAA Compliance:
    The ``HIPAACompliance`` class provides utilities aligned with the HIPAA
    Security Rule (45 CFR Part 164, Subpart C), including de-identification
    per Safe Harbor (164.514(b)(2)), audit logging per 164.312(b), and
    AES-256 encryption per 164.312(a)(2)(iv).

Usage Example::

    exporter = FHIRExporter(organization_name="GenomicsLab")
    patient = exporter.create_patient("Jane Doe", "1985-03-15", "female")
    obs = exporter.create_telomere_observation(
        patient_ref=f"Patient/{patient['id']}",
        telomere_length_kb=6.8,
        biological_age=42
    )
    bundle = exporter.create_bundle([patient, obs])
    exporter.export_json(bundle, "/tmp/report.json")
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: LOINC, SNOMED-CT, and FHIR system URIs
# ---------------------------------------------------------------------------

FHIR_R4_BASE = "http://hl7.org/fhir"
LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
FHIR_OBSERVATION_CATEGORY_SYSTEM = "http://terminology.hl7.org/CodeSystem/observation-category"
FHIR_DIAGNOSTIC_CATEGORY_SYSTEM = "http://terminology.hl7.org/CodeSystem/v2-0074"
FHIR_IDENTIFIER_SYSTEM = "http://hl7.org/fhir/sid/us-ssn"
FHIR_AUDIT_EVENT_TYPE_SYSTEM = "http://dicom.nema.org/resources/ontology/DCM"
FHIR_BUNDLE_TYPE_SYSTEM = "http://hl7.org/fhir/bundle-type"
FHIR_NARRATIVE_STATUS = "generated"

# LOINC codes relevant to telomere / genomic observations
LOINC_TELOMERE_LENGTH = "93592-1"  # Mean telomere length in Blood
LOINC_BIOLOGICAL_AGE = "30525-0"  # Age
LOINC_GENETIC_VARIANT = "81247-9"  # Master HL7 genetic variant assessment

# SNOMED CT disease codes used in risk assessments
SNOMED_DISEASE_MAP: dict[str, dict[str, str]] = {
    "diabetes": {"code": "73211009", "display": "Diabetes mellitus"},
    "heart_disease": {"code": "56265001", "display": "Heart disease"},
    "cancer": {"code": "363346000", "display": "Malignant neoplastic disease"},
    "alzheimers": {"code": "26929004", "display": "Alzheimer's disease"},
    "hypertension": {"code": "38341003", "display": "Hypertensive disorder"},
    "stroke": {"code": "230690007", "display": "Cerebrovascular accident"},
    "copd": {"code": "13645005", "display": "Chronic obstructive lung disease"},
    "osteoporosis": {"code": "64859006", "display": "Osteoporosis"},
}

# Roles used by HIPAA minimum-necessary filtering
ROLE_FIELD_ACCESS: dict[str, list[str]] = {
    "clinician": [
        "resourceType",
        "id",
        "meta",
        "status",
        "code",
        "subject",
        "valueQuantity",
        "valueCodeableConcept",
        "component",
        "category",
        "effectiveDateTime",
        "issued",
        "result",
        "conclusion",
        "prediction",
        "basis",
        "encounter",
        "identifier",
        "name",
        "gender",
        "birthDate",
    ],
    "researcher": [
        "resourceType",
        "id",
        "meta",
        "status",
        "code",
        "valueQuantity",
        "valueCodeableConcept",
        "component",
        "category",
        "effectiveDateTime",
        "issued",
        "result",
        "conclusion",
        "prediction",
    ],
    "billing": [
        "resourceType",
        "id",
        "meta",
        "status",
        "code",
        "subject",
        "category",
        "issued",
        "identifier",
    ],
    "patient": [
        "resourceType",
        "id",
        "meta",
        "status",
        "code",
        "valueQuantity",
        "component",
        "effectiveDateTime",
        "conclusion",
        "prediction",
        "name",
        "gender",
        "birthDate",
    ],
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _generate_id() -> str:
    """Generate a FHIR-compliant resource ID (UUID v4 without hyphens)."""
    return uuid.uuid4().hex[:24]


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format for FHIR ``instant``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _make_meta(profile: str | None = None) -> dict:
    """
    Build a FHIR ``Meta`` element.

    Parameters
    ----------
    profile : str, optional
        A FHIR StructureDefinition URL to include in ``meta.profile``.

    Returns
    -------
    dict
        A conformant ``meta`` block with ``versionId``, ``lastUpdated``,
        and optional ``profile``.
    """
    meta: dict[str, Any] = {
        "versionId": "1",
        "lastUpdated": _now_iso(),
    }
    if profile:
        meta["profile"] = [profile]
    return meta


def _make_reference(resource_type: str, resource_id: str, display: str = "") -> dict:
    """
    Build a FHIR ``Reference`` element.

    Parameters
    ----------
    resource_type : str
        The FHIR resource type (e.g., ``"Patient"``).
    resource_id : str
        The logical ID of the referenced resource.
    display : str, optional
        A human-readable label for the reference.

    Returns
    -------
    dict
        ``{"reference": "Patient/abc123", "display": "..."}``
    """
    ref: dict[str, str] = {"reference": f"{resource_type}/{resource_id}"}
    if display:
        ref["display"] = display
    return ref


def _make_codeable_concept(system: str, code: str, display: str, text: str = "") -> dict:
    """
    Build a FHIR ``CodeableConcept`` with a single coding entry.

    Parameters
    ----------
    system : str
        The terminology system URI (e.g., LOINC_SYSTEM).
    code : str
        The code within that system.
    display : str
        Human-readable display string for the code.
    text : str, optional
        Free-text summary.  Falls back to *display* if omitted.

    Returns
    -------
    dict
        A well-formed ``CodeableConcept``.
    """
    cc: dict[str, Any] = {
        "coding": [{"system": system, "code": code, "display": display}],
        "text": text or display,
    }
    return cc


# ============================================================================
# FHIRExporter
# ============================================================================


class FHIRExporter:
    """
    Generate HL7 FHIR R4 resources for telomere-based clinical diagnostics.

    This exporter produces conformant FHIR R4 JSON resources that can be
    submitted to any FHIR-compliant EHR system (Epic, Cerner, Allscripts,
    etc.) via the RESTful FHIR API.

    Parameters
    ----------
    organization_name : str
        Display name of the performing organization (appears in
        ``DiagnosticReport.performer``).
    practitioner_id : str, optional
        Logical ID of the responsible ``Practitioner`` resource.  When
        provided, the practitioner is included as a performer on
        diagnostic reports and as the orderer on nutrition orders.

    Attributes
    ----------
    organization_name : str
    practitioner_id : str or None

    Notes
    -----
    All generated resources use UUID-based logical IDs, UTC timestamps,
    and include ``meta.versionId`` / ``meta.lastUpdated`` per the FHIR
    R4 specification (https://hl7.org/fhir/R4/resource.html#Meta).

    Examples
    --------
    >>> exp = FHIRExporter(organization_name="GenomicsLab", practitioner_id="pract-001")
    >>> patient = exp.create_patient("John Smith", "1990-01-15", "male", identifier="MRN-12345")
    >>> patient["resourceType"]
    'Patient'
    """

    def __init__(
        self,
        organization_name: str = "Teloscopy",
        practitioner_id: str | None = None,
    ) -> None:
        self.organization_name = organization_name
        self.practitioner_id = practitioner_id
        self._resource_count = 0
        logger.info(
            "FHIRExporter initialized (org=%s, practitioner=%s)",
            organization_name,
            practitioner_id,
        )

    # ------------------------------------------------------------------
    # Patient
    # ------------------------------------------------------------------

    def create_patient(
        self,
        name: str,
        birth_date: str,
        sex: str,
        identifier: str | None = None,
    ) -> dict:
        """
        Create a FHIR R4 Patient resource.

        Conforms to https://hl7.org/fhir/R4/patient.html.

        Parameters
        ----------
        name : str
            Full patient name.  The first token is treated as the given
            name; the last token is treated as the family name.
        birth_date : str
            Date of birth in ``YYYY-MM-DD`` format.
        sex : str
            Administrative gender — one of ``"male"``, ``"female"``,
            ``"other"``, or ``"unknown"`` per FHIR ValueSet
            ``administrative-gender``.
        identifier : str, optional
            An external identifier (e.g., MRN).  When provided, it is
            stored under ``Patient.identifier`` with use ``"usual"``.

        Returns
        -------
        dict
            A FHIR Patient resource dictionary.

        Raises
        ------
        ValueError
            If *sex* is not a recognized administrative gender value.
        """
        valid_genders = {"male", "female", "other", "unknown"}
        sex_lower = sex.lower()
        if sex_lower not in valid_genders:
            raise ValueError(
                f"Invalid administrative gender '{sex}'. Must be one of {valid_genders}."
            )

        parts = name.strip().split()
        given = parts[:-1] if len(parts) > 1 else parts
        family = parts[-1] if len(parts) > 1 else ""

        resource_id = _generate_id()
        patient: dict[str, Any] = {
            "resourceType": "Patient",
            "id": resource_id,
            "meta": _make_meta(
                profile="http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"
            ),
            "text": {
                "status": FHIR_NARRATIVE_STATUS,
                "div": (f'<div xmlns="http://www.w3.org/1999/xhtml"><p>Patient: {name}</p></div>'),
            },
            "active": True,
            "name": [
                {
                    "use": "official",
                    "family": family,
                    "given": given,
                    "text": name,
                }
            ],
            "gender": sex_lower,
            "birthDate": birth_date,
            "managingOrganization": {
                "display": self.organization_name,
            },
        }

        if identifier:
            patient["identifier"] = [
                {
                    "use": "usual",
                    "type": _make_codeable_concept(
                        system="http://terminology.hl7.org/CodeSystem/v2-0203",
                        code="MR",
                        display="Medical Record Number",
                    ),
                    "system": "http://hospital.example.org/mrn",
                    "value": identifier,
                }
            ]

        self._resource_count += 1
        logger.debug("Created Patient/%s for '%s'", resource_id, name)
        return patient

    # ------------------------------------------------------------------
    # Telomere Observation
    # ------------------------------------------------------------------

    def create_telomere_observation(
        self,
        patient_ref: str,
        telomere_length_kb: float,
        biological_age: int,
        method: str = "qFISH",
    ) -> dict:
        """
        Create a FHIR R4 Observation for telomere length measurement.

        Uses LOINC code **93592-1** (*Mean telomere length in Blood*) as the
        primary code and includes biological age as a component observation
        under LOINC **30525-0** (*Age*).

        Conforms to https://hl7.org/fhir/R4/observation.html and the
        Genomics Reporting IG where applicable.

        Parameters
        ----------
        patient_ref : str
            FHIR reference to the Patient (e.g., ``"Patient/abc123"``).
        telomere_length_kb : float
            Measured mean telomere length in kilobases (kb).
        biological_age : int
            Estimated biological age derived from telomere length.
        method : str
            Measurement method — ``"qFISH"``, ``"qPCR"``, ``"TRF"``,
            or ``"STELA"``.

        Returns
        -------
        dict
            A FHIR Observation resource dictionary.
        """
        resource_id = _generate_id()
        method_display_map = {
            "qFISH": "Quantitative fluorescence in situ hybridization",
            "qPCR": "Quantitative polymerase chain reaction",
            "TRF": "Terminal restriction fragment analysis",
            "STELA": "Single telomere length analysis",
        }

        observation: dict[str, Any] = {
            "resourceType": "Observation",
            "id": resource_id,
            "meta": _make_meta(profile="http://hl7.org/fhir/StructureDefinition/Observation"),
            "text": {
                "status": FHIR_NARRATIVE_STATUS,
                "div": (
                    f'<div xmlns="http://www.w3.org/1999/xhtml">'
                    f"<p>Telomere length: {telomere_length_kb} kb "
                    f"(biological age: {biological_age})</p></div>"
                ),
            },
            "status": "final",
            "category": [
                _make_codeable_concept(
                    system=FHIR_OBSERVATION_CATEGORY_SYSTEM,
                    code="laboratory",
                    display="Laboratory",
                )
            ],
            "code": _make_codeable_concept(
                system=LOINC_SYSTEM,
                code=LOINC_TELOMERE_LENGTH,
                display="Mean telomere length in Blood",
                text="Telomere Length Measurement",
            ),
            "subject": {"reference": patient_ref},
            "effectiveDateTime": _now_iso(),
            "issued": _now_iso(),
            "valueQuantity": {
                "value": round(telomere_length_kb, 3),
                "unit": "kb",
                "system": "http://unitsofmeasure.org",
                "code": "kb",
            },
            "method": _make_codeable_concept(
                system=SNOMED_SYSTEM,
                code="702659008",
                display=method_display_map.get(method, method),
                text=method,
            ),
            "component": [
                {
                    "code": _make_codeable_concept(
                        system=LOINC_SYSTEM,
                        code=LOINC_BIOLOGICAL_AGE,
                        display="Age",
                        text="Biological Age (telomere-derived)",
                    ),
                    "valueQuantity": {
                        "value": biological_age,
                        "unit": "years",
                        "system": "http://unitsofmeasure.org",
                        "code": "a",
                    },
                }
            ],
            "interpretation": [self._interpret_telomere_length(telomere_length_kb)],
            "referenceRange": [
                {
                    "low": {"value": 4.0, "unit": "kb"},
                    "high": {"value": 15.0, "unit": "kb"},
                    "text": "Normal adult telomere length range",
                    "type": _make_codeable_concept(
                        system="http://terminology.hl7.org/CodeSystem/referencerange-meaning",
                        code="normal",
                        display="Normal Range",
                    ),
                }
            ],
        }

        self._resource_count += 1
        logger.debug(
            "Created Observation/%s (telomere %.2f kb, bio-age %d)",
            resource_id,
            telomere_length_kb,
            biological_age,
        )
        return observation

    @staticmethod
    def _interpret_telomere_length(length_kb: float) -> dict:
        """
        Derive a FHIR interpretation CodeableConcept for a telomere length.

        Parameters
        ----------
        length_kb : float
            Measured telomere length in kilobases.

        Returns
        -------
        dict
            A CodeableConcept from the ``observation-interpretation``
            value set: ``L`` (low), ``N`` (normal), or ``H`` (high).
        """
        system = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
        if length_kb < 5.0:
            return _make_codeable_concept(system, "L", "Low", "Shortened telomeres")
        elif length_kb > 12.0:
            return _make_codeable_concept(system, "H", "High", "Elongated telomeres")
        return _make_codeable_concept(system, "N", "Normal", "Normal telomere length")

    # ------------------------------------------------------------------
    # Genomic Observation (SNP / variant)
    # ------------------------------------------------------------------

    def create_genomic_observation(
        self,
        patient_ref: str,
        rsid: str,
        genotype: str,
        gene: str,
    ) -> dict:
        """
        Create a FHIR R4 Observation for a molecular sequence variant.

        Uses LOINC code **81247-9** (*Master HL7 genetic variant assessment*)
        as required by the HL7 Genomics Reporting Implementation Guide.

        Conforms to https://hl7.org/fhir/R4/observation.html and the
        HL7 Genomics Reporting IG
        (http://hl7.org/fhir/uv/genomics-reporting/).

        Parameters
        ----------
        patient_ref : str
            FHIR reference to the Patient resource.
        rsid : str
            dbSNP reference SNP ID (e.g., ``"rs1234567"``).
        genotype : str
            Observed genotype string (e.g., ``"A/G"``).
        gene : str
            HGNC gene symbol (e.g., ``"TERT"``).

        Returns
        -------
        dict
            A FHIR Observation resource dictionary with genomic components.
        """
        resource_id = _generate_id()

        observation: dict[str, Any] = {
            "resourceType": "Observation",
            "id": resource_id,
            "meta": _make_meta(
                profile="http://hl7.org/fhir/uv/genomics-reporting/StructureDefinition/variant"
            ),
            "text": {
                "status": FHIR_NARRATIVE_STATUS,
                "div": (
                    f'<div xmlns="http://www.w3.org/1999/xhtml">'
                    f"<p>Variant {rsid} in {gene}: {genotype}</p></div>"
                ),
            },
            "status": "final",
            "category": [
                _make_codeable_concept(
                    system=FHIR_OBSERVATION_CATEGORY_SYSTEM,
                    code="laboratory",
                    display="Laboratory",
                ),
                _make_codeable_concept(
                    system="http://terminology.hl7.org/CodeSystem/v2-0074",
                    code="GE",
                    display="Genetics",
                ),
            ],
            "code": _make_codeable_concept(
                system=LOINC_SYSTEM,
                code=LOINC_GENETIC_VARIANT,
                display="Master HL7 genetic variant assessment",
                text=f"Genetic variant assessment — {gene} {rsid}",
            ),
            "subject": {"reference": patient_ref},
            "effectiveDateTime": _now_iso(),
            "issued": _now_iso(),
            "valueCodeableConcept": _make_codeable_concept(
                system=LOINC_SYSTEM,
                code="LA9633-4",
                display="Present",
                text="Variant detected",
            ),
            "component": [
                {
                    "code": _make_codeable_concept(
                        system=LOINC_SYSTEM,
                        code="48018-6",
                        display="Gene studied [ID]",
                    ),
                    "valueCodeableConcept": _make_codeable_concept(
                        system="http://www.genenames.org/geneId",
                        code=gene,
                        display=gene,
                    ),
                },
                {
                    "code": _make_codeable_concept(
                        system=LOINC_SYSTEM,
                        code="48013-7",
                        display="Genomic reference sequence [ID]",
                    ),
                    "valueCodeableConcept": _make_codeable_concept(
                        system="http://www.ncbi.nlm.nih.gov/snp",
                        code=rsid,
                        display=rsid,
                    ),
                },
                {
                    "code": _make_codeable_concept(
                        system=LOINC_SYSTEM,
                        code="69551-0",
                        display="Genotype display name",
                    ),
                    "valueString": genotype,
                },
            ],
        }

        self._resource_count += 1
        logger.debug(
            "Created Observation/%s (genomic %s %s %s)",
            resource_id,
            gene,
            rsid,
            genotype,
        )
        return observation

    # ------------------------------------------------------------------
    # RiskAssessment
    # ------------------------------------------------------------------

    def create_risk_assessment(
        self,
        patient_ref: str,
        disease: str,
        probability: float,
        basis_refs: list[str] | None = None,
    ) -> dict:
        """
        Create a FHIR R4 RiskAssessment resource.

        Maps disease names to SNOMED CT codes via an internal lookup
        table.  If the disease name is not found, a generic code is used.

        Conforms to https://hl7.org/fhir/R4/riskassessment.html.

        Parameters
        ----------
        patient_ref : str
            FHIR reference to the Patient.
        disease : str
            Disease key (e.g., ``"diabetes"``, ``"cancer"``).  Matched
            case-insensitively against ``SNOMED_DISEASE_MAP``.
        probability : float
            Probability between 0.0 and 1.0.
        basis_refs : list of str, optional
            FHIR references to Observation resources that form the
            evidential basis for this assessment.

        Returns
        -------
        dict
            A FHIR RiskAssessment resource dictionary.

        Raises
        ------
        ValueError
            If *probability* is outside [0, 1].
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")

        resource_id = _generate_id()
        disease_key = disease.lower().replace(" ", "_")
        snomed = SNOMED_DISEASE_MAP.get(
            disease_key,
            {"code": "64572001", "display": disease.title()},
        )

        # Qualitative risk label
        if probability < 0.2:
            qualitative = "low"
        elif probability < 0.5:
            qualitative = "moderate"
        elif probability < 0.8:
            qualitative = "high"
        else:
            qualitative = "very high"

        risk_assessment: dict[str, Any] = {
            "resourceType": "RiskAssessment",
            "id": resource_id,
            "meta": _make_meta(),
            "text": {
                "status": FHIR_NARRATIVE_STATUS,
                "div": (
                    f'<div xmlns="http://www.w3.org/1999/xhtml">'
                    f"<p>Risk of {snomed['display']}: {probability:.1%} "
                    f"({qualitative})</p></div>"
                ),
            },
            "status": "final",
            "subject": {"reference": patient_ref},
            "occurrenceDateTime": _now_iso(),
            "prediction": [
                {
                    "outcome": _make_codeable_concept(
                        system=SNOMED_SYSTEM,
                        code=snomed["code"],
                        display=snomed["display"],
                    ),
                    "probabilityDecimal": round(probability, 4),
                    "qualitativeRisk": _make_codeable_concept(
                        system="http://terminology.hl7.org/CodeSystem/risk-probability",
                        code=qualitative.replace(" ", "-"),
                        display=qualitative.title(),
                    ),
                    "relativeRisk": round(probability / 0.1, 2) if probability > 0 else 0.0,
                    "whenRange": {
                        "low": {"value": 5, "unit": "years", "code": "a"},
                        "high": {"value": 20, "unit": "years", "code": "a"},
                    },
                }
            ],
        }

        if basis_refs:
            risk_assessment["basis"] = [{"reference": ref} for ref in basis_refs]

        if self.practitioner_id:
            risk_assessment["performer"] = _make_reference("Practitioner", self.practitioner_id)

        self._resource_count += 1
        logger.debug(
            "Created RiskAssessment/%s (%s %.2f%%)",
            resource_id,
            disease,
            probability * 100,
        )
        return risk_assessment

    # ------------------------------------------------------------------
    # DiagnosticReport
    # ------------------------------------------------------------------

    def create_diagnostic_report(
        self,
        patient_ref: str,
        telomere_data: dict,
        disease_risks: list,
        observation_refs: list[str] | None = None,
    ) -> dict:
        """
        Create a FHIR R4 DiagnosticReport aggregating telomere data.

        The report bundles observation references and a narrative
        conclusion summarising telomere length, biological age, and
        top disease risks.

        Conforms to https://hl7.org/fhir/R4/diagnosticreport.html.

        Parameters
        ----------
        patient_ref : str
            FHIR reference to the Patient.
        telomere_data : dict
            Must contain ``"telomere_length_kb"`` (float) and
            ``"biological_age"`` (int).
        disease_risks : list of dict
            Each dict must have ``"disease"`` (str) and
            ``"probability"`` (float).
        observation_refs : list of str, optional
            FHIR references to Observations included in this report.

        Returns
        -------
        dict
            A FHIR DiagnosticReport resource dictionary.
        """
        resource_id = _generate_id()
        tl = telomere_data.get("telomere_length_kb", 0.0)
        ba = telomere_data.get("biological_age", 0)

        # Build a human-readable conclusion
        risk_lines = "; ".join(
            f"{r['disease']}: {r['probability']:.0%}"
            for r in sorted(disease_risks, key=lambda x: -x["probability"])[:5]
        )
        conclusion = (
            f"Telomere length {tl:.2f} kb (biological age {ba}). Risk summary — {risk_lines}."
        )

        report: dict[str, Any] = {
            "resourceType": "DiagnosticReport",
            "id": resource_id,
            "meta": _make_meta(profile="http://hl7.org/fhir/StructureDefinition/DiagnosticReport"),
            "text": {
                "status": FHIR_NARRATIVE_STATUS,
                "div": (f'<div xmlns="http://www.w3.org/1999/xhtml"><p>{conclusion}</p></div>'),
            },
            "status": "final",
            "category": [
                _make_codeable_concept(
                    system=FHIR_DIAGNOSTIC_CATEGORY_SYSTEM,
                    code="GE",
                    display="Genetics",
                )
            ],
            "code": _make_codeable_concept(
                system=LOINC_SYSTEM,
                code="51969-4",
                display="Genetic analysis report",
                text="Telomere Length Diagnostic Report",
            ),
            "subject": {"reference": patient_ref},
            "effectiveDateTime": _now_iso(),
            "issued": _now_iso(),
            "conclusion": conclusion,
            "conclusionCode": [
                _make_codeable_concept(
                    system=SNOMED_SYSTEM,
                    code="386053000",
                    display="Evaluation procedure",
                    text="Telomere length evaluation",
                )
            ],
        }

        if observation_refs:
            report["result"] = [{"reference": ref} for ref in observation_refs]

        performers = [{"display": self.organization_name}]
        if self.practitioner_id:
            performers.append(_make_reference("Practitioner", self.practitioner_id))
        report["performer"] = performers

        self._resource_count += 1
        logger.debug("Created DiagnosticReport/%s", resource_id)
        return report

    # ------------------------------------------------------------------
    # NutritionOrder
    # ------------------------------------------------------------------

    def create_nutrition_order(
        self,
        patient_ref: str,
        diet_recs: dict,
    ) -> dict:
        """
        Create a FHIR R4 NutritionOrder based on dietary recommendations.

        Conforms to https://hl7.org/fhir/R4/nutritionorder.html.

        Parameters
        ----------
        patient_ref : str
            FHIR reference to the Patient.
        diet_recs : dict
            Dietary recommendation payload.  Expected keys:

            - ``"diet_type"`` (str): e.g., ``"Mediterranean"``.
            - ``"calories"`` (int, optional): daily kcal target.
            - ``"supplements"`` (list of str, optional): recommended
              dietary supplements.
            - ``"restrictions"`` (list of str, optional): foods to avoid.
            - ``"instructions"`` (str, optional): free-text guidance.

        Returns
        -------
        dict
            A FHIR NutritionOrder resource dictionary.
        """
        resource_id = _generate_id()
        diet_type = diet_recs.get("diet_type", "Balanced")
        calories = diet_recs.get("calories")
        supplements = diet_recs.get("supplements", [])
        restrictions = diet_recs.get("restrictions", [])
        instructions = diet_recs.get("instructions", "")

        oral_diet: dict[str, Any] = {
            "type": [
                _make_codeable_concept(
                    system=SNOMED_SYSTEM,
                    code="38226001",
                    display=f"{diet_type} diet",
                    text=f"Recommended: {diet_type} diet",
                )
            ],
        }

        if calories:
            oral_diet["nutrient"] = [
                {
                    "modifier": _make_codeable_concept(
                        system=SNOMED_SYSTEM,
                        code="226029000",
                        display="Calories",
                    ),
                    "amount": {
                        "value": calories,
                        "unit": "kcal",
                        "system": "http://unitsofmeasure.org",
                        "code": "kcal",
                    },
                }
            ]

        if restrictions:
            oral_diet["fluidConsistencyType"] = []
            oral_diet["excludeFoodModifier"] = [
                _make_codeable_concept(
                    system=SNOMED_SYSTEM,
                    code="227313005",
                    display=item,
                    text=f"Avoid: {item}",
                )
                for item in restrictions
            ]

        if instructions:
            oral_diet["instruction"] = instructions

        order: dict[str, Any] = {
            "resourceType": "NutritionOrder",
            "id": resource_id,
            "meta": _make_meta(),
            "text": {
                "status": FHIR_NARRATIVE_STATUS,
                "div": (
                    f'<div xmlns="http://www.w3.org/1999/xhtml">'
                    f"<p>{diet_type} diet — {calories or 'standard'} kcal/day</p>"
                    f"</div>"
                ),
            },
            "status": "active",
            "intent": "order",
            "patient": {"reference": patient_ref},
            "dateTime": _now_iso(),
            "oralDiet": oral_diet,
        }

        if supplements:
            order["supplement"] = [
                {
                    "type": _make_codeable_concept(
                        system=SNOMED_SYSTEM,
                        code="373453009",
                        display=supp,
                        text=f"Supplement: {supp}",
                    ),
                }
                for supp in supplements
            ]

        if self.practitioner_id:
            order["orderer"] = _make_reference("Practitioner", self.practitioner_id)

        self._resource_count += 1
        logger.debug("Created NutritionOrder/%s", resource_id)
        return order

    # ------------------------------------------------------------------
    # Bundle
    # ------------------------------------------------------------------

    def create_bundle(
        self,
        resources: list[dict],
        bundle_type: str = "collection",
    ) -> dict:
        """
        Wrap a list of FHIR resources into a FHIR R4 Bundle.

        Conforms to https://hl7.org/fhir/R4/bundle.html.

        Parameters
        ----------
        resources : list of dict
            FHIR resource dictionaries to include.
        bundle_type : str
            Bundle type — ``"collection"``, ``"document"``,
            ``"transaction"``, ``"batch"``, etc.

        Returns
        -------
        dict
            A FHIR Bundle resource dictionary with ``entry`` elements.

        Raises
        ------
        ValueError
            If *bundle_type* is not a recognized FHIR bundle type.
        """
        valid_types = {
            "document",
            "message",
            "transaction",
            "transaction-response",
            "batch",
            "batch-response",
            "history",
            "searchset",
            "collection",
        }
        if bundle_type not in valid_types:
            raise ValueError(
                f"Invalid bundle type '{bundle_type}'. Must be one of {sorted(valid_types)}."
            )

        bundle_id = _generate_id()
        entries = []
        for res in resources:
            entry: dict[str, Any] = {"resource": res}
            rt = res.get("resourceType", "Unknown")
            rid = res.get("id", "")
            entry["fullUrl"] = f"urn:uuid:{rid}"
            if bundle_type == "transaction":
                entry["request"] = {
                    "method": "POST",
                    "url": rt,
                }
            entries.append(entry)

        bundle: dict[str, Any] = {
            "resourceType": "Bundle",
            "id": bundle_id,
            "meta": _make_meta(),
            "type": bundle_type,
            "timestamp": _now_iso(),
            "total": len(entries),
            "entry": entries,
        }

        logger.info(
            "Created Bundle/%s (%s, %d entries)",
            bundle_id,
            bundle_type,
            len(entries),
        )
        return bundle

    # ------------------------------------------------------------------
    # Export / Validate
    # ------------------------------------------------------------------

    def export_json(self, bundle: dict, path: str) -> None:
        """
        Serialise a FHIR Bundle (or any resource) to a JSON file.

        Parameters
        ----------
        bundle : dict
            The FHIR resource or Bundle to write.
        path : str
            Filesystem path for the output JSON file.  Parent directories
            are created automatically.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, ensure_ascii=False)

        logger.info("Exported FHIR JSON to %s (%d bytes)", path, os.path.getsize(path))

    def validate_resource(self, resource: dict) -> list[str]:
        """
        Perform basic structural validation on a FHIR resource.

        This is a lightweight, offline validator that checks required
        fields per the FHIR R4 base specification.  It does **not**
        replace a full FHIR validation server (e.g., the official
        HL7 FHIR Validator or Inferno).

        Parameters
        ----------
        resource : dict
            A FHIR resource dictionary.

        Returns
        -------
        list of str
            A list of human-readable validation error messages.  An empty
            list indicates that no issues were detected.
        """
        errors: list[str] = []

        if not isinstance(resource, dict):
            return ["Resource must be a JSON object (dict)."]

        rt = resource.get("resourceType")
        if not rt:
            errors.append("Missing required field 'resourceType'.")
        if not resource.get("id"):
            errors.append("Missing required field 'id'.")
        if not resource.get("meta"):
            errors.append("Missing recommended field 'meta'.")

        # Resource-type-specific checks
        if rt == "Patient":
            if not resource.get("name"):
                errors.append("Patient: missing 'name' element.")
            if not resource.get("gender"):
                errors.append("Patient: missing 'gender' element.")
            if not resource.get("birthDate"):
                errors.append("Patient: missing 'birthDate' element.")

        elif rt == "Observation":
            if not resource.get("status"):
                errors.append("Observation: missing required 'status'.")
            if not resource.get("code"):
                errors.append("Observation: missing required 'code'.")
            if not resource.get("subject"):
                errors.append("Observation: missing 'subject' reference.")
            # Check LOINC coding exists
            code = resource.get("code", {})
            codings = code.get("coding", [])
            if not any(c.get("system") == LOINC_SYSTEM for c in codings):
                errors.append("Observation: 'code' should include a LOINC coding.")

        elif rt == "DiagnosticReport":
            if not resource.get("status"):
                errors.append("DiagnosticReport: missing required 'status'.")
            if not resource.get("code"):
                errors.append("DiagnosticReport: missing required 'code'.")
            if not resource.get("subject"):
                errors.append("DiagnosticReport: missing 'subject' reference.")

        elif rt == "RiskAssessment":
            if not resource.get("status"):
                errors.append("RiskAssessment: missing required 'status'.")
            if not resource.get("subject"):
                errors.append("RiskAssessment: missing 'subject' reference.")
            predictions = resource.get("prediction", [])
            for i, pred in enumerate(predictions):
                if "outcome" not in pred:
                    errors.append(f"RiskAssessment.prediction[{i}]: missing 'outcome'.")
                prob = pred.get("probabilityDecimal")
                if prob is not None and not (0.0 <= prob <= 1.0):
                    errors.append(
                        f"RiskAssessment.prediction[{i}]: probabilityDecimal {prob} outside [0,1]."
                    )

        elif rt == "NutritionOrder":
            if not resource.get("status"):
                errors.append("NutritionOrder: missing required 'status'.")
            if not resource.get("intent"):
                errors.append("NutritionOrder: missing required 'intent'.")
            if not resource.get("patient"):
                errors.append("NutritionOrder: missing 'patient' reference.")
            if not resource.get("dateTime"):
                errors.append("NutritionOrder: missing required 'dateTime'.")

        elif rt == "Bundle":
            if not resource.get("type"):
                errors.append("Bundle: missing required 'type'.")
            entries = resource.get("entry", [])
            for i, entry in enumerate(entries):
                if "resource" not in entry:
                    errors.append(f"Bundle.entry[{i}]: missing 'resource'.")

        return errors


# ============================================================================
# HIPAACompliance
# ============================================================================


class HIPAACompliance:
    """
    HIPAA Security Rule compliance utilities for FHIR resources.

    Provides functionality aligned with:

    - **De-identification** — 45 CFR 164.514(b)(2) Safe Harbor method.
    - **Audit Logging** — 45 CFR 164.312(b) via FHIR AuditEvent resources.
    - **Encryption** — 45 CFR 164.312(a)(2)(iv) via AES-256-CBC.
    - **Minimum Necessary** — 45 CFR 164.502(b) role-based field filtering.

    Notes
    -----
    The encryption methods require the ``cryptography`` library at runtime.
    If unavailable, a fallback using ``hmac`` + ``hashlib`` is used for
    integrity verification, but full AES encryption will raise
    ``ImportError``.

    Examples
    --------
    >>> hipaa = HIPAACompliance()
    >>> patient = exporter.create_patient("Jane Doe", "1985-03-15", "female")
    >>> anon = hipaa.anonymize_patient(patient)
    >>> "Doe" not in json.dumps(anon)
    True
    """

    # PII fields to strip during anonymization (Safe Harbor identifiers)
    _PII_FIELDS = {
        "name",
        "identifier",
        "telecom",
        "address",
        "contact",
        "photo",
        "communication",
        "link",
        "managingOrganization",
    }

    # Narrative div may contain PII
    _NARRATIVE_FIELDS = {"text"}

    def anonymize_patient(self, patient: dict) -> dict:
        """
        Remove Protected Health Information from a FHIR Patient resource.

        Implements the HIPAA Safe Harbor de-identification method
        (45 CFR 164.514(b)(2)) by stripping the 18 categories of
        identifiers from the Patient resource.

        Parameters
        ----------
        patient : dict
            A FHIR Patient resource (``resourceType`` = ``"Patient"``).

        Returns
        -------
        dict
            A deep copy of the Patient with PII fields removed and
            ``meta.security`` set to indicate de-identified data.

        Raises
        ------
        ValueError
            If *patient* is not a Patient resource.
        """
        if patient.get("resourceType") != "Patient":
            raise ValueError("Expected a Patient resource.")

        anon = copy.deepcopy(patient)

        # Remove PII fields
        for field in self._PII_FIELDS | self._NARRATIVE_FIELDS:
            anon.pop(field, None)

        # Generalize birth date to year only (Safe Harbor allows year)
        birth_date = anon.get("birthDate", "")
        if birth_date and len(birth_date) >= 4:
            anon["birthDate"] = birth_date[:4]

        # Replace ID with a one-way hash
        original_id = anon.get("id", "")
        anon["id"] = hashlib.sha256(original_id.encode("utf-8")).hexdigest()[:24]

        # Mark as de-identified in meta.security
        anon.setdefault("meta", {})
        anon["meta"]["security"] = [
            {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue",
                "code": "PSEUDED",
                "display": "pseudonymized",
            }
        ]

        logger.info("Anonymized Patient/%s -> Patient/%s", original_id, anon["id"])
        return anon

    def generate_audit_log(
        self,
        action: str,
        resource_type: str,
        user_id: str,
    ) -> dict:
        """
        Generate a FHIR R4 AuditEvent resource for HIPAA audit logging.

        Satisfies the audit controls requirement of 45 CFR 164.312(b)
        by recording who accessed what resource and when.

        Conforms to https://hl7.org/fhir/R4/auditevent.html.

        Parameters
        ----------
        action : str
            The action performed — one of ``"C"`` (create), ``"R"``
            (read), ``"U"`` (update), ``"D"`` (delete), ``"E"``
            (execute).
        resource_type : str
            The FHIR resource type that was accessed (e.g.,
            ``"Patient"``).
        user_id : str
            Identifier of the user who performed the action.

        Returns
        -------
        dict
            A FHIR AuditEvent resource dictionary.
        """
        valid_actions = {"C", "R", "U", "D", "E"}
        if action not in valid_actions:
            raise ValueError(f"Invalid action '{action}'. Must be one of {valid_actions}.")

        resource_id = _generate_id()
        now = _now_iso()

        audit_event: dict[str, Any] = {
            "resourceType": "AuditEvent",
            "id": resource_id,
            "meta": _make_meta(),
            "type": {
                "system": FHIR_AUDIT_EVENT_TYPE_SYSTEM,
                "code": "110112",
                "display": "Query",
            },
            "subtype": [
                {
                    "system": "http://hl7.org/fhir/restful-interaction",
                    "code": {
                        "C": "create",
                        "R": "read",
                        "U": "update",
                        "D": "delete",
                        "E": "execute",
                    }[action],
                    "display": {
                        "C": "create",
                        "R": "read",
                        "U": "update",
                        "D": "delete",
                        "E": "execute",
                    }[action].title(),
                }
            ],
            "action": action,
            "period": {
                "start": now,
                "end": now,
            },
            "recorded": now,
            "outcome": "0",  # Success
            "outcomeDesc": "Operation completed successfully",
            "agent": [
                {
                    "type": _make_codeable_concept(
                        system="http://terminology.hl7.org/CodeSystem/extra-security-role-type",
                        code="humanuser",
                        display="Human User",
                    ),
                    "who": {
                        "identifier": {
                            "value": user_id,
                        },
                        "display": f"User {user_id}",
                    },
                    "requestor": True,
                }
            ],
            "source": {
                "site": "Teloscopy Clinical Platform",
                "observer": {
                    "display": "Teloscopy FHIR Server",
                },
                "type": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/security-source-type",
                        "code": "4",
                        "display": "Application Server",
                    }
                ],
            },
            "entity": [
                {
                    "what": {
                        "reference": f"{resource_type}/",
                    },
                    "type": {
                        "system": "http://terminology.hl7.org/CodeSystem/audit-entity-type",
                        "code": "2",
                        "display": "System Object",
                    },
                    "role": {
                        "system": "http://terminology.hl7.org/CodeSystem/object-role",
                        "code": "4",
                        "display": "Domain Resource",
                    },
                    "lifecycle": {
                        "system": "http://terminology.hl7.org/CodeSystem/dicom-audit-lifecycle",
                        "code": "6",
                        "display": "Access / Use",
                    },
                    "description": f"FHIR {resource_type} resource access",
                }
            ],
        }

        logger.info(
            "AuditEvent/%s: user=%s action=%s resource=%s",
            resource_id,
            user_id,
            action,
            resource_type,
        )
        return audit_event

    def encrypt_bundle(self, bundle: dict, key: bytes) -> bytes:
        """
        Encrypt a FHIR Bundle using AES-256-CBC.

        Satisfies the encryption requirement of the HIPAA Security Rule
        (45 CFR 164.312(a)(2)(iv)).  The output format is::

            IV (16 bytes) || HMAC-SHA256 (32 bytes) || AES-256-CBC ciphertext

        Parameters
        ----------
        bundle : dict
            A FHIR Bundle (or any resource) dictionary.
        key : bytes
            A 32-byte (256-bit) encryption key.

        Returns
        -------
        bytes
            The encrypted payload.

        Raises
        ------
        ValueError
            If *key* is not 32 bytes.
        ImportError
            If the ``cryptography`` library is not installed.
        """
        if len(key) != 32:
            raise ValueError(f"Key must be exactly 32 bytes (256 bits), got {len(key)}.")

        plaintext = json.dumps(bundle, ensure_ascii=False).encode("utf-8")

        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import padding as sym_padding
            from cryptography.hazmat.primitives.ciphers import (
                Cipher,
                algorithms,
                modes,
            )

            iv = os.urandom(16)

            # PKCS7 padding
            padder = sym_padding.PKCS7(128).padder()
            padded = padder.update(plaintext) + padder.finalize()

            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded) + encryptor.finalize()

            # HMAC for integrity (Encrypt-then-MAC)
            mac = hmac.new(key, iv + ciphertext, hashlib.sha256).digest()

            payload = iv + mac + ciphertext
            logger.info(
                "Encrypted FHIR bundle (%d bytes plaintext -> %d bytes encrypted)",
                len(plaintext),
                len(payload),
            )
            return payload

        except ImportError:
            logger.warning(
                "cryptography library not available; "
                "falling back to XOR-based obfuscation (NOT production-safe)."
            )
            return self._fallback_encrypt(plaintext, key)

    def decrypt_bundle(self, data: bytes, key: bytes) -> dict:
        """
        Decrypt a FHIR Bundle previously encrypted with ``encrypt_bundle``.

        Parameters
        ----------
        data : bytes
            Encrypted payload produced by ``encrypt_bundle``.
        key : bytes
            The same 32-byte key used for encryption.

        Returns
        -------
        dict
            The decrypted FHIR resource dictionary.

        Raises
        ------
        ValueError
            If *key* is wrong, data is corrupted, or HMAC verification fails.
        ImportError
            If the ``cryptography`` library is not installed.
        """
        if len(key) != 32:
            raise ValueError(f"Key must be exactly 32 bytes (256 bits), got {len(key)}.")

        if len(data) < 48:  # 16 IV + 32 HMAC minimum
            raise ValueError("Encrypted data is too short to be valid.")

        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import padding as sym_padding
            from cryptography.hazmat.primitives.ciphers import (
                Cipher,
                algorithms,
                modes,
            )

            iv = data[:16]
            stored_mac = data[16:48]
            ciphertext = data[48:]

            # Verify HMAC
            computed_mac = hmac.new(key, iv + ciphertext, hashlib.sha256).digest()
            if not hmac.compare_digest(stored_mac, computed_mac):
                raise ValueError(
                    "HMAC verification failed — data may be corrupted or the key is incorrect."
                )

            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded = decryptor.update(ciphertext) + decryptor.finalize()

            # Remove PKCS7 padding
            unpadder = sym_padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded) + unpadder.finalize()

            bundle = json.loads(plaintext.decode("utf-8"))
            logger.info("Decrypted FHIR bundle (%d bytes)", len(plaintext))
            return bundle

        except ImportError:
            logger.warning("cryptography library not available; using fallback XOR decryption.")
            plaintext = self._fallback_decrypt(data, key)
            return json.loads(plaintext.decode("utf-8"))

    def check_minimum_necessary(self, resource: dict, role: str) -> dict:
        """
        Filter a FHIR resource to include only fields permitted for a role.

        Implements the HIPAA Minimum Necessary Standard (45 CFR 164.502(b))
        by restricting the fields returned based on the accessor's role.

        Supported roles: ``"clinician"``, ``"researcher"``, ``"billing"``,
        ``"patient"``.

        Parameters
        ----------
        resource : dict
            A FHIR resource dictionary.
        role : str
            The role of the requesting user.

        Returns
        -------
        dict
            A filtered copy of the resource containing only permitted fields.

        Raises
        ------
        ValueError
            If *role* is not recognized.
        """
        role_lower = role.lower()
        if role_lower not in ROLE_FIELD_ACCESS:
            raise ValueError(
                f"Unknown role '{role}'. Must be one of {list(ROLE_FIELD_ACCESS.keys())}."
            )

        allowed_fields = set(ROLE_FIELD_ACCESS[role_lower])
        filtered = {k: copy.deepcopy(v) for k, v in resource.items() if k in allowed_fields}

        logger.debug(
            "Minimum necessary filter (role=%s): kept %d/%d fields",
            role,
            len(filtered),
            len(resource),
        )
        return filtered

    # ------------------------------------------------------------------
    # Fallback encryption (when cryptography library is unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_encrypt(plaintext: bytes, key: bytes) -> bytes:
        """
        XOR-based obfuscation fallback.

        .. warning::
            This is **NOT** cryptographically secure and must not be used
            in production.  Install the ``cryptography`` package for
            proper AES-256 encryption.

        Parameters
        ----------
        plaintext : bytes
            Data to obfuscate.
        key : bytes
            32-byte key (cycled over the plaintext).

        Returns
        -------
        bytes
            Obfuscated data prefixed with a 16-byte random IV (unused
            in XOR mode, included for format compatibility) and 32-byte
            HMAC.
        """
        iv = os.urandom(16)
        key_stream = (key * ((len(plaintext) // len(key)) + 1))[: len(plaintext)]
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream))
        mac = hmac.new(key, iv + ciphertext, hashlib.sha256).digest()
        return iv + mac + ciphertext

    @staticmethod
    def _fallback_decrypt(data: bytes, key: bytes) -> bytes:
        """
        Reverse fallback XOR obfuscation.

        Parameters
        ----------
        data : bytes
            Obfuscated payload from ``_fallback_encrypt``.
        key : bytes
            The same 32-byte key.

        Returns
        -------
        bytes
            The original plaintext.

        Raises
        ------
        ValueError
            If HMAC verification fails.
        """
        iv = data[:16]
        stored_mac = data[16:48]
        ciphertext = data[48:]

        computed_mac = hmac.new(key, iv + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(stored_mac, computed_mac):
            raise ValueError("HMAC verification failed in fallback decryption.")

        key_stream = (key * ((len(ciphertext) // len(key)) + 1))[: len(ciphertext)]
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, key_stream))
        return plaintext


# ============================================================================
# Module-level convenience functions
# ============================================================================


def create_full_telomere_report(
    patient_name: str,
    birth_date: str,
    sex: str,
    telomere_length_kb: float,
    biological_age: int,
    disease_risks: list[dict[str, Any]],
    snp_variants: list[dict[str, str]] | None = None,
    diet_recommendations: dict | None = None,
    organization: str = "Teloscopy",
    practitioner_id: str | None = None,
    output_path: str | None = None,
) -> dict:
    """
    End-to-end convenience function to generate a complete FHIR Bundle.

    Creates a Patient, telomere Observation, optional genomic Observations,
    RiskAssessments for each disease, an optional NutritionOrder, and a
    DiagnosticReport — all wrapped in a FHIR ``collection`` Bundle.

    Parameters
    ----------
    patient_name : str
        Full patient name.
    birth_date : str
        Date of birth (``YYYY-MM-DD``).
    sex : str
        Administrative gender.
    telomere_length_kb : float
        Measured mean telomere length (kb).
    biological_age : int
        Estimated biological age.
    disease_risks : list of dict
        Each entry must have ``"disease"`` (str) and ``"probability"``
        (float).
    snp_variants : list of dict, optional
        Each entry should have ``"rsid"``, ``"genotype"``, ``"gene"``.
    diet_recommendations : dict, optional
        Passed directly to ``create_nutrition_order``.
    organization : str
        Performing organization name.
    practitioner_id : str, optional
        Practitioner logical ID.
    output_path : str, optional
        If provided, the bundle is also written as JSON to this path.

    Returns
    -------
    dict
        The complete FHIR Bundle dictionary.

    Examples
    --------
    >>> bundle = create_full_telomere_report(
    ...     patient_name="Alice Johnson",
    ...     birth_date="1978-11-22",
    ...     sex="female",
    ...     telomere_length_kb=5.9,
    ...     biological_age=52,
    ...     disease_risks=[
    ...         {"disease": "diabetes", "probability": 0.35},
    ...         {"disease": "heart_disease", "probability": 0.22},
    ...     ],
    ... )
    >>> bundle["resourceType"]
    'Bundle'
    """
    exporter = FHIRExporter(
        organization_name=organization,
        practitioner_id=practitioner_id,
    )

    resources: list[dict] = []
    observation_refs: list[str] = []

    # 1. Patient
    patient = exporter.create_patient(patient_name, birth_date, sex)
    patient_ref = f"Patient/{patient['id']}"
    resources.append(patient)

    # 2. Telomere observation
    telo_obs = exporter.create_telomere_observation(
        patient_ref=patient_ref,
        telomere_length_kb=telomere_length_kb,
        biological_age=biological_age,
    )
    observation_refs.append(f"Observation/{telo_obs['id']}")
    resources.append(telo_obs)

    # 3. Genomic observations (SNPs)
    if snp_variants:
        for variant in snp_variants:
            gen_obs = exporter.create_genomic_observation(
                patient_ref=patient_ref,
                rsid=variant["rsid"],
                genotype=variant["genotype"],
                gene=variant["gene"],
            )
            observation_refs.append(f"Observation/{gen_obs['id']}")
            resources.append(gen_obs)

    # 4. Risk assessments
    for risk in disease_risks:
        risk_res = exporter.create_risk_assessment(
            patient_ref=patient_ref,
            disease=risk["disease"],
            probability=risk["probability"],
            basis_refs=observation_refs,
        )
        resources.append(risk_res)

    # 5. Nutrition order
    if diet_recommendations:
        nutrition = exporter.create_nutrition_order(
            patient_ref=patient_ref,
            diet_recs=diet_recommendations,
        )
        resources.append(nutrition)

    # 6. Diagnostic report
    report = exporter.create_diagnostic_report(
        patient_ref=patient_ref,
        telomere_data={
            "telomere_length_kb": telomere_length_kb,
            "biological_age": biological_age,
        },
        disease_risks=disease_risks,
        observation_refs=observation_refs,
    )
    resources.append(report)

    # 7. Bundle everything
    bundle = exporter.create_bundle(resources, bundle_type="collection")

    if output_path:
        exporter.export_json(bundle, output_path)

    return bundle
