"""Tests for the genetic disease risk prediction module."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from teloscopy.genomics.disease_risk import (
    BASELINE_INCIDENCE,
    BUILTIN_VARIANT_DB,
    DISCLAIMER,
    DISCLAIMER_SHORT,
    DiseasePredictor,
    DiseaseRisk,
    GeneticVariant,
    RiskProfile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_variant():
    """Return a single GeneticVariant for unit testing."""
    return GeneticVariant(
        rsid="rs429358",
        gene="APOE",
        chromosome="19",
        position=44908684,
        risk_allele="C",
        protective_allele="T",
        effect_size=1.45,
        condition="Coronary artery disease",
        category="cardiovascular",
        population_frequency=0.15,
        evidence_level="strong",
    )


@pytest.fixture
def predictor():
    """Return a DiseasePredictor with the built-in database."""
    return DiseasePredictor()


@pytest.fixture
def sample_risk_profile(predictor):
    """Return a RiskProfile produced from a realistic set of SNPs."""
    variants = {
        "rs429358": "CT",
        "rs7903146": "TT",
        "rs334": "TA",
        "rs1061170": "CC",
    }
    return predictor.predict_from_variants(variants, age=55, sex="male")


# ---------------------------------------------------------------------------
# GeneticVariant dataclass
# ---------------------------------------------------------------------------


class TestGeneticVariant:
    """Tests for GeneticVariant creation and allele_count()."""

    def test_creation(self, sample_variant):
        """Verify all fields are correctly assigned on construction."""
        assert sample_variant.rsid == "rs429358"
        assert sample_variant.gene == "APOE"
        assert sample_variant.chromosome == "19"
        assert sample_variant.position == 44908684
        assert sample_variant.risk_allele == "C"
        assert sample_variant.protective_allele == "T"
        assert sample_variant.effect_size == 1.45
        assert sample_variant.condition == "Coronary artery disease"
        assert sample_variant.category == "cardiovascular"
        assert sample_variant.population_frequency == 0.15
        assert sample_variant.evidence_level == "strong"

    def test_frozen(self, sample_variant):
        """GeneticVariant is a frozen dataclass; assignment should raise."""
        with pytest.raises(AttributeError):
            sample_variant.rsid = "rs999"

    def test_allele_count_homozygous_risk(self, sample_variant):
        """Two copies of the risk allele should return 2."""
        assert sample_variant.allele_count("CC") == 2

    def test_allele_count_heterozygous(self, sample_variant):
        """One copy of the risk allele should return 1."""
        assert sample_variant.allele_count("CT") == 1
        assert sample_variant.allele_count("TC") == 1

    def test_allele_count_homozygous_protective(self, sample_variant):
        """No risk alleles should return 0."""
        assert sample_variant.allele_count("TT") == 0

    def test_allele_count_invalid_genotype(self, sample_variant):
        """Genotypes with len != 2 should return 0."""
        assert sample_variant.allele_count("") == 0
        assert sample_variant.allele_count("C") == 0
        assert sample_variant.allele_count("CCC") == 0


# ---------------------------------------------------------------------------
# DiseaseRisk dataclass
# ---------------------------------------------------------------------------


class TestDiseaseRisk:
    """Tests for the DiseaseRisk dataclass."""

    def test_creation_minimal(self):
        """Create a DiseaseRisk with only required fields."""
        risk = DiseaseRisk(
            condition="Test Disease",
            category="test",
            lifetime_risk_pct=12.5,
            relative_risk=1.5,
            confidence=0.8,
        )
        assert risk.condition == "Test Disease"
        assert risk.category == "test"
        assert risk.lifetime_risk_pct == 12.5
        assert risk.relative_risk == 1.5
        assert risk.confidence == 0.8
        assert risk.contributing_variants == []
        assert risk.age_of_onset_range == (0, 100)
        assert risk.preventability_score == 0.5

    def test_creation_full(self):
        """Create a DiseaseRisk with all fields specified."""
        risk = DiseaseRisk(
            condition="Type 2 diabetes",
            category="diabetes",
            lifetime_risk_pct=25.0,
            relative_risk=1.8,
            confidence=0.75,
            contributing_variants=["rs7903146", "rs12255372"],
            age_of_onset_range=(30, 75),
            preventability_score=0.75,
        )
        assert len(risk.contributing_variants) == 2
        assert risk.age_of_onset_range == (30, 75)
        assert risk.preventability_score == 0.75


# ---------------------------------------------------------------------------
# RiskProfile
# ---------------------------------------------------------------------------


class TestRiskProfile:
    """Tests for the RiskProfile container."""

    def _make_risks(self):
        """Helper to create a list of DiseaseRisk objects."""
        return [
            DiseaseRisk("CAD", "cardiovascular", 30.0, 1.8, 0.9, ["rs429358"]),
            DiseaseRisk("T2D", "diabetes", 20.0, 1.5, 0.7, ["rs7903146"]),
            DiseaseRisk("AMD", "eye", 10.0, 2.0, 0.6, ["rs1061170"]),
            DiseaseRisk("Alzheimer's", "alzheimers", 40.0, 3.0, 0.85, ["rs429358"]),
            DiseaseRisk("Obesity", "metabolic", 15.0, 1.3, 0.4, ["rs9939609"]),
        ]

    def test_creation_empty(self):
        """An empty RiskProfile should have length 0."""
        rp = RiskProfile()
        assert len(rp) == 0
        assert rp.risks == []
        assert rp.metadata == {}

    def test_creation_with_risks(self):
        """RiskProfile stores risks and metadata."""
        risks = self._make_risks()
        rp = RiskProfile(risks=risks, metadata={"age": 50})
        assert len(rp) == 5
        assert rp.metadata["age"] == 50

    def test_top_risks(self):
        """top_risks() should return conditions sorted by lifetime_risk_pct descending."""
        rp = RiskProfile(risks=self._make_risks())
        top3 = rp.top_risks(n=3)
        assert len(top3) == 3
        assert top3[0].condition == "Alzheimer's"  # 40%
        assert top3[1].condition == "CAD"  # 30%
        assert top3[2].condition == "T2D"  # 20%

    def test_top_risks_more_than_available(self):
        """Requesting more than available should return all."""
        rp = RiskProfile(risks=self._make_risks())
        top = rp.top_risks(n=100)
        assert len(top) == 5

    def test_filter_by_category(self):
        """filter_by_category is case-insensitive."""
        rp = RiskProfile(risks=self._make_risks())
        cardio = rp.filter_by_category("CARDIOVASCULAR")
        assert len(cardio) == 1
        assert cardio[0].condition == "CAD"

    def test_filter_by_category_no_match(self):
        """A non-existent category should return an empty list."""
        rp = RiskProfile(risks=self._make_risks())
        result = rp.filter_by_category("nonexistent")
        assert result == []

    def test_summary_returns_dataframe(self):
        """summary() should return a pandas DataFrame with expected columns."""
        rp = RiskProfile(risks=self._make_risks())
        df = rp.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        expected_cols = {
            "condition",
            "category",
            "lifetime_risk_pct",
            "relative_risk",
            "confidence",
            "n_variants",
            "preventability_score",
            "onset_min_age",
            "onset_max_age",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_summary_empty(self):
        """summary() on an empty profile returns an empty DataFrame."""
        rp = RiskProfile()
        df = rp.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_categories_property(self):
        """categories should return sorted unique categories."""
        rp = RiskProfile(risks=self._make_risks())
        cats = rp.categories
        assert isinstance(cats, list)
        assert cats == sorted(set(r.category for r in self._make_risks()))

    def test_filter_by_confidence(self):
        """filter_by_confidence returns only risks above the threshold."""
        rp = RiskProfile(risks=self._make_risks())
        high_conf = rp.filter_by_confidence(min_confidence=0.8)
        assert all(r.confidence >= 0.8 for r in high_conf)
        assert len(high_conf) == 2  # CAD (0.9) and Alzheimer's (0.85)


# ---------------------------------------------------------------------------
# DiseasePredictor — initialisation
# ---------------------------------------------------------------------------


class TestDiseasePredictorInit:
    """Tests for DiseasePredictor construction and database loading."""

    def test_builtin_db_has_50_plus_entries(self, predictor):
        """The built-in variant database should contain at least 50 entries."""
        assert predictor.variant_count >= 50

    def test_conditions_list_nonempty(self, predictor):
        """The predictor should cover multiple conditions."""
        conds = predictor.conditions
        assert len(conds) > 10

    def test_builtin_db_module_level(self):
        """BUILTIN_VARIANT_DB is available as a module-level constant."""
        assert len(BUILTIN_VARIANT_DB) >= 50
        assert all(isinstance(v, GeneticVariant) for v in BUILTIN_VARIANT_DB)


# ---------------------------------------------------------------------------
# predict_from_variants
# ---------------------------------------------------------------------------


class TestPredictFromVariants:
    """Tests for DiseasePredictor.predict_from_variants()."""

    def test_returns_risk_profile(self, predictor):
        """Should return a RiskProfile instance."""
        variants = {"rs429358": "CT", "rs7903146": "TT"}
        profile = predictor.predict_from_variants(variants, age=50, sex="male")
        assert isinstance(profile, RiskProfile)
        assert len(profile) > 0

    def test_metadata_populated(self, predictor):
        """Metadata should contain age, sex, and variant counts."""
        variants = {"rs429358": "CC"}
        profile = predictor.predict_from_variants(variants, age=60, sex="female")
        meta = profile.metadata
        assert meta["age"] == 60
        assert meta["sex"] == "female"
        assert meta["input_variants"] == 1
        assert "disclaimer" in meta

    def test_rs429358_increases_cad_risk(self, predictor):
        """Homozygous risk allele in APOE should produce elevated CAD relative risk."""
        profile = predictor.predict_from_variants({"rs429358": "CC"}, age=50, sex="male")
        cad_risks = [r for r in profile.risks if r.condition == "Coronary artery disease"]
        assert len(cad_risks) > 0
        assert cad_risks[0].relative_risk > 1.0

    def test_multiple_variants_combined(self, predictor):
        """Multiple risk alleles for the same condition should compound risk."""
        single = predictor.predict_from_variants({"rs429358": "CT"}, age=50, sex="male")
        double = predictor.predict_from_variants(
            {"rs429358": "CT", "rs10455872": "GA"}, age=50, sex="male"
        )
        single_cad = [r for r in single.risks if r.condition == "Coronary artery disease"]
        double_cad = [r for r in double.risks if r.condition == "Coronary artery disease"]
        if single_cad and double_cad:
            assert double_cad[0].relative_risk >= single_cad[0].relative_risk

    def test_empty_variants(self, predictor):
        """Empty variant map should produce a profile with zero risks."""
        profile = predictor.predict_from_variants({}, age=40, sex="female")
        assert isinstance(profile, RiskProfile)
        assert len(profile) == 0

    def test_unknown_rsids(self, predictor):
        """Unknown rsIDs should be silently ignored."""
        profile = predictor.predict_from_variants(
            {"rs_FAKE1": "AA", "rs_FAKE2": "GG"}, age=30, sex="male"
        )
        assert len(profile) == 0

    def test_invalid_sex_raises(self, predictor):
        """Invalid sex value should raise ValueError."""
        with pytest.raises(ValueError, match="sex must be"):
            predictor.predict_from_variants({"rs429358": "CT"}, age=40, sex="other")

    def test_extreme_age_young(self, predictor):
        """A very young age should still produce valid (possibly low) risk values."""
        profile = predictor.predict_from_variants({"rs429358": "CC"}, age=5, sex="male")
        for risk in profile.risks:
            assert 0.0 <= risk.lifetime_risk_pct <= 100.0

    def test_extreme_age_old(self, predictor):
        """An elderly age may produce reduced remaining-lifetime risk for late-onset diseases."""
        profile = predictor.predict_from_variants({"rs429358": "CC"}, age=95, sex="female")
        for risk in profile.risks:
            assert 0.0 <= risk.lifetime_risk_pct <= 100.0


# ---------------------------------------------------------------------------
# predict_from_telomere_data
# ---------------------------------------------------------------------------


class TestPredictFromTelomereData:
    """Tests for DiseasePredictor.predict_from_telomere_data()."""

    def test_short_telomeres_produce_risks(self, predictor):
        """Very short telomeres should produce elevated risks."""
        risks = predictor.predict_from_telomere_data(mean_length_bp=4000.0, age=50, sex="male")
        assert isinstance(risks, list)
        assert len(risks) > 0
        assert all(isinstance(r, DiseaseRisk) for r in risks)

    def test_normal_telomeres_minimal_risk(self, predictor):
        """Age-appropriate telomere length should produce few or no extra risks."""
        # Expected at age 30: 11000 - 30*30 = 10100 bp
        risks = predictor.predict_from_telomere_data(mean_length_bp=10100.0, age=30, sex="female")
        # No shortening → all RR ≤ 1 → filtered out
        assert len(risks) == 0

    def test_long_telomeres_no_risk(self, predictor):
        """Telomeres longer than expected should produce no extra risks."""
        risks = predictor.predict_from_telomere_data(mean_length_bp=15000.0, age=40, sex="male")
        assert len(risks) == 0

    def test_short_telomere_conditions(self, predictor):
        """Conditions with known telomere modifiers should appear for short telomeres."""
        risks = predictor.predict_from_telomere_data(mean_length_bp=3000.0, age=55, sex="female")
        conditions = {r.condition for r in risks}
        # At least cardiovascular and cancer conditions should appear
        assert "Coronary artery disease" in conditions

    def test_contributing_variants_marker(self, predictor):
        """Telomere-derived risks should have 'telomere_length' as contributing variant."""
        risks = predictor.predict_from_telomere_data(mean_length_bp=4000.0, age=50, sex="male")
        for risk in risks:
            assert "telomere_length" in risk.contributing_variants


# ---------------------------------------------------------------------------
# predict_from_image_analysis
# ---------------------------------------------------------------------------


class TestPredictFromImageAnalysis:
    """Tests for DiseasePredictor.predict_from_image_analysis()."""

    def test_with_mean_intensity(self, predictor):
        """Providing mean_intensity should trigger telomere-based predictions."""
        results = predictor.predict_from_image_analysis(
            {
                "mean_intensity": 2000.0,  # proxy → 3000 bp (short)
                "age": 50,
                "sex": "male",
            }
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_high_cv_cancer_risk(self, predictor):
        """High CV should trigger genomic-instability cancer risks."""
        results = predictor.predict_from_image_analysis(
            {
                "mean_intensity": 5000.0,  # proxy → 7500 bp (normalish)
                "cv": 0.8,
                "age": 45,
                "sex": "female",
            }
        )
        conditions = {r.condition for r in results}
        # High CV should add cancer conditions
        assert "Breast cancer" in conditions or "Lung cancer" in conditions

    def test_empty_analysis_results(self, predictor):
        """No mean_intensity and low CV should produce an empty list."""
        results = predictor.predict_from_image_analysis({})
        assert results == []

    def test_cv_instability_marker(self, predictor):
        """CV-triggered risks should have 'image_cv_instability' marker."""
        results = predictor.predict_from_image_analysis(
            {
                "mean_intensity": 5000.0,
                "cv": 0.9,
                "age": 50,
                "sex": "male",
            }
        )
        cv_risks = [r for r in results if "image_cv_instability" in r.contributing_variants]
        assert len(cv_risks) > 0


# ---------------------------------------------------------------------------
# calculate_polygenic_risk
# ---------------------------------------------------------------------------


class TestCalculatePolygenicRisk:
    """Tests for DiseasePredictor.calculate_polygenic_risk()."""

    def test_known_condition_positive_score(self, predictor):
        """Risk alleles for T2D should produce a positive PRS."""
        prs = predictor.calculate_polygenic_risk(
            {"rs7903146": "TT", "rs12255372": "TT"},
            condition="Type 2 diabetes",
        )
        assert prs > 0

    def test_protective_alleles_negative_score(self, predictor):
        """Protective alleles (OR < 1) should contribute negatively to PRS."""
        # rs1801282 PPARG has OR=0.86 with risk allele G
        # Homozygous for protective allele C → 0 risk alleles
        prs = predictor.calculate_polygenic_risk(
            {"rs1801282": "CC"},
            condition="Type 2 diabetes",
        )
        assert prs == 0.0  # no risk alleles present

    def test_unknown_condition_returns_zero(self, predictor):
        """A condition not in the database should return 0.0."""
        prs = predictor.calculate_polygenic_risk(
            {"rs429358": "CC"},
            condition="Fictional Disease XYZ",
        )
        assert prs == 0.0

    def test_missing_variants_ignored(self, predictor):
        """Variants not in the genotype map should be silently skipped."""
        prs_full = predictor.calculate_polygenic_risk(
            {"rs7903146": "TT", "rs12255372": "TT"},
            condition="Type 2 diabetes",
        )
        prs_partial = predictor.calculate_polygenic_risk(
            {"rs7903146": "TT"},
            condition="Type 2 diabetes",
        )
        assert prs_full >= prs_partial

    def test_alzheimers_apoe(self, predictor):
        """Homozygous APOE ε4 (rs429358 CC) should yield a high Alzheimer's PRS."""
        prs = predictor.calculate_polygenic_risk(
            {"rs429358": "CC"},
            condition="Alzheimer's disease",
        )
        assert prs > 1.0  # log(3.68) * 2 * 1.0 ≈ 2.6


# ---------------------------------------------------------------------------
# project_risk_over_time
# ---------------------------------------------------------------------------


class TestProjectRiskOverTime:
    """Tests for DiseasePredictor.project_risk_over_time()."""

    def test_returns_projections(self, predictor, sample_risk_profile):
        """Should return a dict mapping conditions to year-by-year lists."""
        projections = predictor.project_risk_over_time(
            sample_risk_profile, current_age=55, years=20
        )
        assert isinstance(projections, dict)
        assert len(projections) > 0
        for condition, yearly in projections.items():
            assert isinstance(yearly, list)
            assert len(yearly) == 21  # 0..20
            for entry in yearly:
                assert "age" in entry
                assert "cumulative_risk_pct" in entry

    def test_cumulative_risk_increases(self, predictor, sample_risk_profile):
        """Cumulative risk should generally be non-decreasing over time."""
        projections = predictor.project_risk_over_time(
            sample_risk_profile, current_age=55, years=10
        )
        for condition, yearly in projections.items():
            risks = [e["cumulative_risk_pct"] for e in yearly]
            for i in range(1, len(risks)):
                assert risks[i] >= risks[i - 1] - 1e-9  # monotonically non-decreasing

    def test_zero_years(self, predictor, sample_risk_profile):
        """Projecting 0 years should return a single entry per condition."""
        projections = predictor.project_risk_over_time(sample_risk_profile, current_age=55, years=0)
        for condition, yearly in projections.items():
            assert len(yearly) == 1
            assert yearly[0]["age"] == 55


# ---------------------------------------------------------------------------
# get_actionable_insights
# ---------------------------------------------------------------------------


class TestGetActionableInsights:
    """Tests for DiseasePredictor.get_actionable_insights()."""

    def test_returns_recommendations(self, predictor, sample_risk_profile):
        """Should return a list of insight dicts with expected keys."""
        insights = predictor.get_actionable_insights(sample_risk_profile)
        assert isinstance(insights, list)
        assert len(insights) > 0
        for insight in insights:
            assert "condition" in insight
            assert "category" in insight
            assert "risk_level" in insight
            assert insight["risk_level"] in ("high", "moderate", "low")
            assert "relative_risk" in insight
            assert "recommendations" in insight
            assert "disclaimer" in insight

    def test_disclaimer_present(self, predictor, sample_risk_profile):
        """Every insight should carry the short disclaimer."""
        insights = predictor.get_actionable_insights(sample_risk_profile)
        for insight in insights:
            assert insight["disclaimer"] == DISCLAIMER_SHORT

    def test_empty_profile_no_insights(self, predictor):
        """An empty risk profile should produce no insights."""
        empty = RiskProfile()
        insights = predictor.get_actionable_insights(empty)
        assert insights == []

    def test_high_risk_sorted_first(self, predictor):
        """Insights should be sorted by relative_risk descending."""
        risks = [
            DiseaseRisk("Low Risk", "test", 5.0, 1.1, 0.5),
            DiseaseRisk("High Risk", "test", 50.0, 5.0, 0.9),
            DiseaseRisk("Med Risk", "test", 20.0, 2.0, 0.7),
        ]
        profile = RiskProfile(risks=risks)
        insights = predictor.get_actionable_insights(profile)
        rrs = [i["relative_risk"] for i in insights]
        assert rrs == sorted(rrs, reverse=True)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Tests for module-level constants and baseline data."""

    def test_disclaimer_nonempty(self):
        """DISCLAIMER should be a non-empty string."""
        assert isinstance(DISCLAIMER, str)
        assert len(DISCLAIMER) > 50

    def test_disclaimer_short_nonempty(self):
        """DISCLAIMER_SHORT should be a non-empty string."""
        assert isinstance(DISCLAIMER_SHORT, str)
        assert len(DISCLAIMER_SHORT) > 10

    def test_baseline_incidence_has_entries(self):
        """BASELINE_INCIDENCE should contain multiple (condition, sex) pairs."""
        assert len(BASELINE_INCIDENCE) > 20
        for key, value in BASELINE_INCIDENCE.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(value, float)
            assert value > 0
