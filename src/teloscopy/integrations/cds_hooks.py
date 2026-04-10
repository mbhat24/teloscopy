"""CDS Hooks and FHIR Subscriptions for Real-Time Clinical Alerting.

Implements CDS Hooks v1.1 and FHIR R4 Subscription resources for clinical
decision support within EHR systems.  Companion to :mod:`teloscopy.integrations.fhir`.
Covers RESEARCH.md Section 11.4: service discovery, hook processing, PGx alerting,
FHIR Subscriptions, and longitudinal telomere tracking.

Refs: https://cds-hooks.hl7.org/1.1/  ·  https://hl7.org/fhir/R4/subscription.html
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# -- Constants & helpers -----------------------------------------------------
_TELOSCOPY_SOURCE = {"label": "Teloscopy", "url": "https://teloscopy.ai", "icon": "https://teloscopy.ai/assets/icon-cds.png"}
LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
FHIR_OBS_CAT = "http://terminology.hl7.org/CodeSystem/observation-category"
LOINC_TELOMERE_LENGTH = "93592-1"
LOINC_BIOLOGICAL_AGE = "30525-0"
LOINC_GENETIC_VARIANT = "81247-9"

def _gen_id() -> str:
    return uuid.uuid4().hex[:24]

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

# ============================================================================
# 1. CDS Hooks Service Discovery
# ============================================================================

@dataclass
class CDSHookService:
    """Single CDS Hooks service descriptor."""
    hook: str  # "patient-view", "order-select", "order-sign", "medication-prescribe"
    title: str
    description: str
    id: str  # unique service ID
    prefetch: dict[str, str]  # FHIR query templates for prefetch


_TELOSCOPY_CDS_SERVICES = [
    CDSHookService(
        hook="patient-view",
        title="Teloscopy Telomere Health Summary",
        description="Displays telomere length status, biological age, and aging trajectory when viewing a patient chart.",
        id="teloscopy-telomere-summary",
        prefetch={"patient": "Patient/{{context.patientId}}", "observations": "Observation?patient={{context.patientId}}&code=telomere-length"},
    ),
    CDSHookService(
        hook="medication-prescribe",
        title="Teloscopy Pharmacogenomic Alert",
        description="Alerts prescribers when a medication interacts with the patient's predicted pharmacogenomic profile.",
        id="teloscopy-pgx-alert",
        prefetch={"patient": "Patient/{{context.patientId}}", "medication": "MedicationRequest/{{context.draftOrders.MedicationRequest[0]}}"},
    ),
    CDSHookService(
        hook="order-select",
        title="Teloscopy Genomic Test Recommendation",
        description="Recommends confirmatory genomic tests based on facial-genomic predictions.",
        id="teloscopy-genomic-test-rec",
        prefetch={"patient": "Patient/{{context.patientId}}"},
    ),
]


def get_cds_discovery_response() -> dict:
    """Return CDS Hooks discovery endpoint response (``GET /cds-services``)."""
    services = [
        {"hook": s.hook, "title": s.title, "description": s.description, "id": s.id, "prefetch": s.prefetch}
        for s in _TELOSCOPY_CDS_SERVICES
    ]
    logger.info("CDS discovery: returning %d services", len(services))
    return {"services": services}

# ============================================================================
# 2. CDS Card Model & Hook Processors
# ============================================================================

@dataclass
class CDSCard:
    """CDS Hooks response card with urgency indicator."""
    summary: str       # one-line summary (max 140 chars)
    detail: str        # markdown body
    indicator: str     # "info", "warning", "critical"
    source: dict = field(default_factory=lambda: dict(_TELOSCOPY_SOURCE))
    suggestions: list[dict] | None = None  # suggested actions
    links: list[dict] | None = None        # external links

    def to_dict(self) -> dict[str, Any]:
        card: dict[str, Any] = {"summary": self.summary[:140], "detail": self.detail, "indicator": self.indicator, "source": self.source}
        if self.suggestions:
            card["suggestions"] = self.suggestions
        if self.links:
            card["links"] = self.links
        return card

# -- 2a. patient-view -------------------------------------------------------

def process_patient_view_hook(
    patient_id: str,
    telomere_data: dict | None = None,
    facial_analysis: dict | None = None,
) -> list[CDSCard]:
    """Generate cards for the ``patient-view`` hook.

    *telomere_data* keys: ``telomere_length_kb``, ``percentile``,
    ``biological_age``, ``chronological_age``, ``trajectory``.

    *facial_analysis* keys: ``predicted_tl_kb``, ``confidence``.
    """
    cards: list[CDSCard] = []

    if telomere_data:
        tl = telomere_data.get("telomere_length_kb", 0.0)
        pctl = telomere_data.get("percentile", 50)
        bio = telomere_data.get("biological_age", 0)
        chrono = telomere_data.get("chronological_age", 0)
        traj = telomere_data.get("trajectory", "normal")
        age_diff = bio - chrono

        # --- Telomere status card ---
        indicator = "warning" if pctl < 10 else "info"
        status = "Below normal" if pctl < 10 else ("Low-normal" if pctl < 25 else "Within normal range")
        age_note = f"{abs(age_diff)} yrs older" if age_diff > 0 else (f"{abs(age_diff)} yrs younger" if age_diff < 0 else "concordant")
        cards.append(CDSCard(
            summary=f"Telomere length {tl:.1f} kb — {status} ({pctl}th percentile)",
            detail=(f"### Telomere Health Summary\n\n| Metric | Value |\n|---|---|\n"
                    f"| Mean TL | **{tl:.2f} kb** |\n| Percentile | {pctl}th |\n"
                    f"| Biological age | {bio} (chronological {chrono}) |\n"
                    f"| Age difference | {age_note} |\n| Trajectory | {traj} |"),
            indicator=indicator,
            links=[{"label": "View full Teloscopy report", "url": f"https://teloscopy.ai/report/{patient_id}", "type": "absolute"}],
        ))

        # --- Critically short telomere warning ---
        if tl < 5.0 or pctl < 5:
            cards.append(CDSCard(
                summary=f"CRITICAL: Telomere length critically short ({tl:.1f} kb, {pctl}th %ile)",
                detail=(f"### Critically Short Telomeres\n\nPatient **{patient_id}**: TL "
                        f"**{tl:.2f} kb** ({pctl}th %ile) — below critical threshold.\n\n"
                        f"**Significance:** accelerated senescence, elevated CVD/cancer/neurodegeneration risk.\n\n"
                        f"**Actions:** 1) Confirmatory qFISH/TRF  2) Metabolic & inflammatory panel  "
                        f"3) Review modifiable risk factors"),
                indicator="critical",
                suggestions=[{"label": "Order confirmatory TL measurement", "uuid": _gen_id(), "actions": [
                    {"type": "create", "description": "Order qFISH telomere length assay",
                     "resource": {"resourceType": "ServiceRequest", "status": "draft", "intent": "proposal",
                                  "code": {"coding": [{"system": LOINC_SYSTEM, "code": LOINC_TELOMERE_LENGTH, "display": "Mean telomere length in Blood"}]},
                                  "subject": {"reference": f"Patient/{patient_id}"}}}]}],
            ))

        # --- Senolytic therapy candidate ---
        if age_diff > 5 and tl < 6.0:
            cards.append(CDSCard(
                summary=f"Potential senolytic therapy candidate (bio-age +{age_diff} yrs)",
                detail=(f"### Senolytic Therapy Candidacy\n\nBiological age exceeds chronological by "
                        f"**{age_diff} yrs** with shortened telomeres (**{tl:.2f} kb**), suggesting "
                        f"elevated senescent cell burden.\n\nConsider: SASP biomarker panel, longevity "
                        f"medicine referral, lifestyle intervention (exercise, nutrition, stress)."),
                indicator="info",
                links=[{"label": "Senolytic evidence review", "url": "https://teloscopy.ai/evidence/senolytics", "type": "absolute"}],
            ))

    if facial_analysis:
        pred_tl = facial_analysis.get("predicted_tl_kb", 0.0)
        conf = facial_analysis.get("confidence", 0.0)
        if pred_tl < 5.5 and conf > 0.6:
            cards.append(CDSCard(
                summary=f"Facial analysis suggests short telomeres ({pred_tl:.1f} kb est.)",
                detail=(f"### Facial-Genomic Telomere Estimate\n\nPredicted TL: **{pred_tl:.2f} kb** "
                        f"(confidence {conf:.0%}). Confirmatory blood-based assay (qFISH/qPCR) recommended."),
                indicator="info",
            ))

    logger.info("patient-view for Patient/%s: %d card(s)", patient_id, len(cards))
    return cards

# ============================================================================
# 3. Drug-Gene Interaction Map (PGx Alerts)
# ============================================================================
_DRUG_PGX_INTERACTIONS: dict[str, dict[str, str]] = {
    # RxNorm ingredient-level CUI → {gene, drug, interaction, pm_action, um_action}
    "2670": {"gene": "CYP2D6", "drug": "codeine", "interaction": "Prodrug — CYP2D6 converts to morphine. Poor metabolizers: no analgesic effect. Ultra-rapid: toxicity risk.", "pm_action": "Use alternative analgesic (morphine, oxycodone)", "um_action": "Reduce dose 50% or use alternative"},
    "32968": {"gene": "CYP2C19", "drug": "clopidogrel", "interaction": "CYP2C19 converts to active metabolite. Poor metabolizers: reduced antiplatelet effect.", "pm_action": "Use prasugrel or ticagrelor instead", "um_action": "Standard dosing"},
    "11289": {"gene": "VKORC1", "drug": "warfarin", "interaction": "VKORC1 -1639G>A increases sensitivity. Requires INR-guided dose reduction.", "pm_action": "Reduce initial dose to 2-3 mg/day", "um_action": "Standard dosing"},
    "10324": {"gene": "CYP2D6", "drug": "tamoxifen", "interaction": "CYP2D6 converts to endoxifen (active). Poor metabolizers: reduced efficacy.", "pm_action": "Consider aromatase inhibitor alternative", "um_action": "Standard dosing"},
    "36567": {"gene": "SLCO1B1", "drug": "simvastatin", "interaction": "SLCO1B1*5 reduces hepatic uptake → increased plasma levels → myopathy risk.", "pm_action": "Use pravastatin or rosuvastatin; max 20mg simvastatin", "um_action": "Standard dosing"},
    "7646": {"gene": "CYP2C19", "drug": "omeprazole", "interaction": "CYP2C19 metabolizes PPIs. Ultra-rapid metabolizers may need higher doses.", "pm_action": "Standard dosing", "um_action": "Consider dose increase or alternative PPI"},
    "42316": {"gene": "CYP3A5", "drug": "tacrolimus", "interaction": "CYP3A5 expressors metabolize faster. Non-expressors need lower doses.", "pm_action": "Start at reduced dose, monitor trough levels", "um_action": "May need higher doses; monitor trough levels"},
    "195085": {"gene": "CYP2B6", "drug": "efavirenz", "interaction": "CYP2B6*6 reduces metabolism. Slow metabolizers at risk of CNS toxicity.", "pm_action": "Reduce dose to 400mg or use alternative", "um_action": "Standard 600mg dosing"},
}

# High-risk drug set — critical rather than warning for poor metabolizers
_HIGH_RISK_PGX_DRUGS = {"codeine", "clopidogrel", "warfarin", "tacrolimus"}

# -- 2b. medication-prescribe -----------------------------------------------

def process_medication_prescribe_hook(
    patient_id: str,
    medication_code: str,  # RxNorm code
    pharmacogenomic_predictions: list[dict],
) -> list[CDSCard]:
    """Generate PGx alert cards for the ``medication-prescribe`` hook.

    *pharmacogenomic_predictions* entries: ``{"gene", "phenotype", "confidence"}``.
    Phenotype values: ``"poor_metabolizer"`` or ``"ultra_rapid_metabolizer"``.
    """
    cards: list[CDSCard] = []
    ix = _DRUG_PGX_INTERACTIONS.get(medication_code)
    if ix is None:
        return cards

    gene, drug = ix["gene"], ix["drug"]
    for pred in pharmacogenomic_predictions:
        if pred.get("gene") != gene:
            continue
        pheno = pred.get("phenotype", "")
        conf = pred.get("confidence", 0.0)
        is_pm = "poor" in pheno.lower()
        is_um = "ultra" in pheno.lower() and "rapid" in pheno.lower()
        if not (is_pm or is_um):
            continue

        action = ix["pm_action"] if is_pm else ix["um_action"]
        label = "Poor Metabolizer" if is_pm else "Ultra-Rapid Metabolizer"

        # Critical for high-risk drugs with poor metabolizer, or codeine + ultra-rapid
        if (drug in _HIGH_RISK_PGX_DRUGS and is_pm) or (drug == "codeine" and is_um):
            indicator = "critical"
        else:
            indicator = "warning"

        # Drug-gene interaction card
        cards.append(CDSCard(
            summary=f"PGx Alert: {gene} {label} — {drug} interaction",
            detail=(f"### Drug-Gene Interaction: {drug.title()} / {gene}\n\n"
                    f"**Patient:** {patient_id} · **Phenotype:** {label} ({conf:.0%} confidence) · "
                    f"**RxNorm:** {medication_code}\n\n"
                    f"**Interaction:** {ix['interaction']}\n\n**Action:** {action}\n\n"
                    f"*Confirmatory genotyping recommended before definitive prescribing changes.*"),
            indicator=indicator,
            suggestions=[
                {"label": f"Switch from {drug}", "uuid": _gen_id(), "actions": [
                    {"type": "delete", "description": f"Remove draft {drug} order (PGx)",
                     "resource": {"resourceType": "MedicationRequest", "id": "{{context.draftOrders.MedicationRequest[0]}}"}}]},
                {"label": f"Order {gene} genotyping", "uuid": _gen_id(), "actions": [
                    {"type": "create", "description": f"Order {gene} pharmacogenomic panel",
                     "resource": {"resourceType": "ServiceRequest", "status": "draft", "intent": "proposal",
                                  "code": {"text": f"{gene} Genotyping Panel"}, "subject": {"reference": f"Patient/{patient_id}"}}}]},
            ],
        ))

        # Dose-adjustment card for non-critical cases
        if indicator == "warning":
            cards.append(CDSCard(
                summary=f"Dose adjustment may be needed for {drug} ({label})",
                detail=(f"### Dose Adjustment Recommendation\n\n{gene} {label} may affect {drug} metabolism.\n\n"
                        f"**Guidance:** {action}\n\nMonitor therapeutic response and adverse effects closely."),
                indicator="warning",
            ))

    logger.info("medication-prescribe for Patient/%s (RxNorm %s): %d card(s)", patient_id, medication_code, len(cards))
    return cards

# -- 2c. order-select -------------------------------------------------------

_TELOMERE_GENES = {"TERT", "TERC", "DKC1", "RTEL1", "TINF2", "NHP2", "NOP10"}

def process_order_select_hook(
    patient_id: str,
    predicted_variants: list[dict],
    confidence_threshold: float = 0.4,
) -> list[CDSCard]:
    """Recommend confirmatory genomic tests via the ``order-select`` hook.

    *predicted_variants* entries: ``{"gene", "variant", "predicted_genotype",
    "confidence", "clinical_significance"}``.
    """
    cards: list[CDSCard] = []
    actionable_sigs = {"pathogenic", "likely_pathogenic", "pharmacogenomic", "risk_factor"}
    actionable = [v for v in predicted_variants
                  if v.get("confidence", 0) >= confidence_threshold
                  and v.get("clinical_significance", "").lower() in actionable_sigs]

    if actionable:
        rows = "\n".join(f"| {v['gene']} | {v['variant']} | {v['predicted_genotype']} | {v['confidence']:.0%} | {v['clinical_significance']} |" for v in actionable)
        cards.append(CDSCard(
            summary=f"Recommend confirmatory genetic testing — {len(actionable)} variant(s) predicted",
            detail=(f"### Confirmatory Genetic Testing Recommended\n\n"
                    f"| Gene | Variant | Genotype | Confidence | Significance |\n|---|---|---|---|---|\n{rows}\n\n"
                    f"Confirmatory testing via clinical-grade sequencing recommended."),
            indicator="info",
            suggestions=[{"label": "Order targeted gene panel", "uuid": _gen_id(), "actions": [
                {"type": "create", "description": "Order targeted gene panel",
                 "resource": {"resourceType": "ServiceRequest", "status": "draft", "intent": "proposal",
                              "code": {"text": "Targeted gene panel: " + ", ".join(v["gene"] for v in actionable)},
                              "subject": {"reference": f"Patient/{patient_id}"}}}]}],
        ))

    # TL measurement if telomere-biology genes predicted
    tl_hits = [v for v in predicted_variants if v.get("gene", "").upper() in _TELOMERE_GENES and v.get("confidence", 0) >= confidence_threshold]
    if tl_hits:
        genes = ", ".join(sorted({v["gene"] for v in tl_hits}))
        cards.append(CDSCard(
            summary="Recommend telomere length measurement based on predicted variants",
            detail=(f"### Telomere Length Measurement Recommended\n\nPredicted variants in telomere biology "
                    f"genes (**{genes}**) suggest altered telomere maintenance. Clinical TL assay (qFISH/Flow-FISH) recommended."),
            indicator="info",
            suggestions=[{"label": "Order telomere length assay", "uuid": _gen_id(), "actions": [
                {"type": "create", "description": "Order qFISH telomere length assay",
                 "resource": {"resourceType": "ServiceRequest", "status": "draft", "intent": "proposal",
                              "code": {"coding": [{"system": LOINC_SYSTEM, "code": LOINC_TELOMERE_LENGTH, "display": "Mean telomere length in Blood"}]},
                              "subject": {"reference": f"Patient/{patient_id}"}}}]}],
        ))

    logger.info("order-select for Patient/%s: %d variant(s) evaluated, %d card(s)", patient_id, len(predicted_variants), len(cards))
    return cards

# ============================================================================
# 4. FHIR Subscription Resources
# ============================================================================

@dataclass
class FHIRSubscription:
    """FHIR R4 Subscription resource model."""
    resource_type: str = "Subscription"
    status: str = "active"
    reason: str = ""
    criteria: str = ""            # FHIR search criteria
    channel_type: str = "rest-hook"  # "rest-hook", "websocket", "email"
    channel_endpoint: str = ""
    channel_payload: str = "application/fhir+json"

    def to_fhir(self) -> dict[str, Any]:
        """Serialise to a FHIR R4 Subscription resource dictionary."""
        return {
            "resourceType": self.resource_type, "id": _gen_id(),
            "status": self.status, "reason": self.reason, "criteria": self.criteria,
            "channel": {"type": self.channel_type, "endpoint": self.channel_endpoint,
                        "payload": self.channel_payload, "header": ["Authorization: Bearer {{token}}"]},
        }

def create_telomere_result_subscription(patient_id: str, callback_url: str) -> dict:
    """Subscribe to new telomere results (LOINC 93592-1) for the patient."""
    sub = FHIRSubscription(
        reason=f"Notify on new telomere length results for Patient/{patient_id}",
        criteria=f"Observation?patient=Patient/{patient_id}&code={LOINC_SYSTEM}|{LOINC_TELOMERE_LENGTH}",
        channel_endpoint=callback_url,
    )
    res = sub.to_fhir()
    logger.info("Created Subscription/%s (telomere results for Patient/%s)", res["id"], patient_id)
    return res

def create_variant_classification_subscription(patient_id: str, callback_url: str) -> dict:
    """Subscribe to variant reclassification events (e.g. VUS → pathogenic)."""
    sub = FHIRSubscription(
        reason=f"Notify on variant reclassification for Patient/{patient_id}",
        criteria=f"Observation?patient=Patient/{patient_id}&code={LOINC_SYSTEM}|{LOINC_GENETIC_VARIANT}&_lastUpdated=gt{{{{now}}}}",
        channel_endpoint=callback_url,
    )
    res = sub.to_fhir()
    logger.info("Created Subscription/%s (variant reclass for Patient/%s)", res["id"], patient_id)
    return res

def create_pgx_interaction_subscription(patient_id: str, callback_url: str) -> dict:
    """Subscribe to pharmacogenomic interaction alerts for new prescriptions."""
    sub = FHIRSubscription(
        reason=f"Evaluate PGx interactions on new prescriptions for Patient/{patient_id}",
        criteria=f"MedicationRequest?patient=Patient/{patient_id}&status=active",
        channel_endpoint=callback_url,
    )
    res = sub.to_fhir()
    logger.info("Created Subscription/%s (PGx alerts for Patient/%s)", res["id"], patient_id)
    return res

# ============================================================================
# 5. Longitudinal Telomere Tracking
# ============================================================================

@dataclass
class LongitudinalTelomereRecord:
    """Single telomere measurement in a longitudinal series."""
    patient_id: str
    measurement_date: str           # ISO YYYY-MM-DD
    telomere_length_kb: float
    measurement_method: str         # "facial_estimate", "qPCR", "qFISH", "TRF"
    biological_age: int | None = None
    percentile: int | None = None

_METHOD_DISPLAY = {
    "facial_estimate": "Facial-genomic telomere estimate",
    "qPCR": "Quantitative polymerase chain reaction",
    "qFISH": "Quantitative fluorescence in situ hybridization",
    "TRF": "Terminal restriction fragment analysis",
}

def build_longitudinal_bundle(records: list[LongitudinalTelomereRecord]) -> dict:
    """Create a FHIR ``collection`` Bundle of chronological TL Observations.

    Raises ``ValueError`` if *records* is empty.
    """
    if not records:
        raise ValueError("At least one LongitudinalTelomereRecord is required.")

    sorted_recs = sorted(records, key=lambda r: r.measurement_date)
    entries: list[dict[str, Any]] = []

    for rec in sorted_recs:
        oid = _gen_id()
        components: list[dict[str, Any]] = []
        if rec.biological_age is not None:
            components.append({
                "code": {"coding": [{"system": LOINC_SYSTEM, "code": LOINC_BIOLOGICAL_AGE, "display": "Age"}], "text": "Biological Age"},
                "valueQuantity": {"value": rec.biological_age, "unit": "years", "system": "http://unitsofmeasure.org", "code": "a"},
            })
        if rec.percentile is not None:
            components.append({
                "code": {"coding": [{"system": SNOMED_SYSTEM, "code": "246205007", "display": "Quantity"}], "text": "TL percentile"},
                "valueQuantity": {"value": rec.percentile, "unit": "%ile", "system": "http://unitsofmeasure.org", "code": "{percentile}"},
            })

        obs: dict[str, Any] = {
            "resourceType": "Observation", "id": oid, "status": "final",
            "category": [{"coding": [{"system": FHIR_OBS_CAT, "code": "laboratory", "display": "Laboratory"}]}],
            "code": {"coding": [{"system": LOINC_SYSTEM, "code": LOINC_TELOMERE_LENGTH, "display": "Mean telomere length in Blood"}], "text": "Telomere Length Measurement"},
            "subject": {"reference": f"Patient/{rec.patient_id}"},
            "effectiveDateTime": rec.measurement_date,
            "valueQuantity": {"value": round(rec.telomere_length_kb, 3), "unit": "kb", "system": "http://unitsofmeasure.org", "code": "kb"},
            "method": {"text": _METHOD_DISPLAY.get(rec.measurement_method, rec.measurement_method)},
        }
        if components:
            obs["component"] = components
        entries.append({"fullUrl": f"urn:uuid:{oid}", "resource": obs})

    bundle = {"resourceType": "Bundle", "id": _gen_id(), "type": "collection", "timestamp": _now_iso(), "total": len(entries), "entry": entries}
    logger.info("Longitudinal Bundle for Patient/%s: %d observations", sorted_recs[0].patient_id, len(entries))
    return bundle

def calculate_trajectory(records: list[LongitudinalTelomereRecord]) -> dict:
    """Calculate telomere attrition trajectory via OLS regression.

    Returns slope (kb/yr), intercept, R², 95% CI, population comparison
    (~-0.025 kb/yr), and classification (accelerated/normal/decelerated).
    Raises ``ValueError`` if fewer than 2 records provided.
    """
    if len(records) < 2:
        raise ValueError("At least two records required to calculate trajectory.")

    recs = sorted(records, key=lambda r: r.measurement_date)
    base = datetime.strptime(recs[0].measurement_date, "%Y-%m-%d")
    xs = [(datetime.strptime(r.measurement_date, "%Y-%m-%d") - base).days / 365.25 for r in recs]
    ys = [r.telomere_length_kb for r in recs]
    n = len(xs)

    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x * x for x in xs)
    denom = n * sx2 - sx * sx

    if abs(denom) < 1e-12:
        slope, intercept = 0.0, sy / n
    else:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # 95 % CI for slope (z-approx)
    if n > 2 and abs(denom) > 1e-12:
        se = math.sqrt(ss_res / ((n - 2) * (sx2 - sx ** 2 / n)))
        ci_lo, ci_hi = slope - 1.96 * se, slope + 1.96 * se
    else:
        ci_lo, ci_hi = slope, slope

    pop_rate = -0.025  # kb/year population average
    classification = "accelerated" if slope < pop_rate * 1.5 else ("decelerated" if slope > pop_rate * 0.5 else "normal")
    span = xs[-1] - xs[0]

    result = {
        "slope_kb_per_year": round(slope, 5), "intercept_kb": round(intercept, 3),
        "r_squared": round(max(0.0, r_sq), 4),
        "ci_95_lower": round(ci_lo, 5), "ci_95_upper": round(ci_hi, 5),
        "population_rate_kb_per_year": pop_rate, "classification": classification,
        "n_measurements": n, "span_years": round(span, 2),
    }
    logger.info("Trajectory Patient/%s: %.4f kb/yr (R²=%.3f, %s) over %.1f yr", recs[0].patient_id, slope, r_sq, classification, span)
    return result
