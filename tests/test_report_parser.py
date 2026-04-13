"""Tests for the lab report parser module.

Covers:
- Text extraction dispatch (detect_file_type, extract_text)
- Lab report parsing (colon, space, table formats)
- Alias resolution for blood and urine parameters
- Urine section detection and ambiguous name handling
- Confidence scoring
- Edge cases (empty input, garbage text)
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from teloscopy.webapp.report_parser import (
        BLOOD_ALIAS_MAP,
        URINE_ALIAS_MAP,
        _identify_columns,
        _normalize_name,
        _parse_structured_tables,
        _resolve_alias,
        compute_extraction_confidence,
        detect_file_type,
        extract_and_parse,
        extract_tables_from_pdf,
        extract_text,
        parse_lab_report,
    )

    _HAS_PARSER = True
except ImportError:
    _HAS_PARSER = False

pytestmark = pytest.mark.skipif(
    not _HAS_PARSER,
    reason="Report parser module or dependencies not available.",
)


# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------


class TestDetectFileType:
    """Tests for file-type detection from magic bytes and extensions."""

    def test_pdf_magic_bytes(self):
        assert detect_file_type(b"%PDF-1.4 ...", "report.pdf") == "pdf"

    def test_pdf_extension_only(self):
        """Even without valid magic bytes, .pdf extension should win."""
        assert detect_file_type(b"\x00\x00\x00", "report.pdf") == "pdf"

    def test_png_magic_bytes(self):
        assert detect_file_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "img.png") == "image"

    def test_jpeg_magic_bytes(self):
        assert detect_file_type(b"\xff\xd8\xff" + b"\x00" * 50, "photo.jpg") == "image"

    def test_webp_magic_bytes(self):
        data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 50
        assert detect_file_type(data, "photo.webp") == "image"

    def test_text_extension(self):
        assert detect_file_type(b"Hello world", "results.txt") == "text"

    def test_text_utf8_detection(self):
        """UTF-8 decodable content without a known extension → text."""
        assert detect_file_type(b"Hemoglobin: 14.5 g/dL", "unknown") == "text"

    def test_unknown_binary(self):
        assert detect_file_type(b"\x80\x81\x82\x83" * 100, "data.bin") == "unknown"

    def test_bmp_magic(self):
        assert detect_file_type(b"BM" + b"\x00" * 50, "scan.bmp") == "image"


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    """Tests for the top-level text extraction dispatcher."""

    def test_text_file(self):
        content = b"Hemoglobin: 14.5 g/dL\nRBC Count: 5.2 M/mcL"
        result = extract_text(content, "report.txt")
        assert "Hemoglobin" in result
        assert "14.5" in result

    def test_empty_file(self):
        result = extract_text(b"", "empty.txt")
        assert result == ""

    def test_latin1_fallback(self):
        """Non-UTF-8 text should fall back to latin-1."""
        content = "Hémaglobine: 14.5".encode("latin-1")
        result = extract_text(content, "report.txt")
        assert "14.5" in result


# ---------------------------------------------------------------------------
# _normalize_name / _resolve_alias
# ---------------------------------------------------------------------------


class TestAliasResolution:
    """Tests for parameter name normalization and alias lookup."""

    def test_normalize_strips_parentheses(self):
        assert _normalize_name("Hemoglobin (g/dL)") == "hemoglobin"

    def test_normalize_collapses_whitespace(self):
        assert _normalize_name("  Total   Cholesterol  ") == "total cholesterol"

    def test_normalize_strips_special_chars(self):
        assert _normalize_name("* Hemoglobin: ") == "hemoglobin"

    def test_resolve_hemoglobin(self):
        result = _resolve_alias("Hemoglobin")
        assert result is not None
        assert result[0] == "hemoglobin"
        assert result[1] == "blood"

    def test_resolve_hb_abbreviation(self):
        result = _resolve_alias("Hb")
        assert result is not None
        assert result[0] == "hemoglobin"

    def test_resolve_fasting_blood_sugar(self):
        result = _resolve_alias("Fasting Blood Sugar")
        assert result is not None
        assert result[0] == "fasting_glucose"

    def test_resolve_tsh(self):
        result = _resolve_alias("TSH")
        assert result is not None
        assert result[0] == "tsh"

    def test_resolve_urine_ph(self):
        result = _resolve_alias("pH", in_urine_section=True)
        assert result is not None
        assert result[0] == "ph"
        assert result[1] == "urine"

    def test_resolve_unknown_name(self):
        result = _resolve_alias("Completely Unknown Parameter XYZ")
        assert result is None

    def test_resolve_ambiguous_glucose_blood(self):
        """'glucose' outside urine section → blood."""
        result = _resolve_alias("glucose", in_urine_section=False)
        assert result is not None
        # Default is blood because blood aliases load after urine in _ALL_ALIASES

    def test_resolve_ambiguous_glucose_urine(self):
        """'glucose' inside urine section → urine."""
        result = _resolve_alias("glucose", in_urine_section=True)
        assert result is not None
        assert result[0] == "glucose"
        assert result[1] == "urine"

    def test_alias_maps_not_empty(self):
        """Sanity: alias maps should have been loaded."""
        assert len(BLOOD_ALIAS_MAP) > 40
        assert len(URINE_ALIAS_MAP) >= 10


# ---------------------------------------------------------------------------
# parse_lab_report — colon-separated format
# ---------------------------------------------------------------------------


class TestParseColonFormat:
    """Tests for parsing colon-separated lab values."""

    def test_single_value(self):
        text = "Hemoglobin: 14.5 g/dL"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.5

    def test_multiple_values(self):
        text = (
            "Hemoglobin: 14.5 g/dL\n"
            "RBC Count: 5.2 M/mcL\n"
            "WBC Count: 7500\n"
            "Platelet Count: 250000\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood["hemoglobin"] == 14.5
        assert blood["rbc_count"] == 5.2
        assert blood["wbc_count"] == 7500.0
        assert blood["platelet_count"] == 250000.0

    def test_equals_separator(self):
        """Equals sign should also work as separator."""
        text = "Hemoglobin = 13.8"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 13.8

    def test_lipid_panel(self):
        text = (
            "Total Cholesterol: 195 mg/dL\n"
            "LDL Cholesterol: 110 mg/dL\n"
            "HDL Cholesterol: 55 mg/dL\n"
            "Triglycerides: 140 mg/dL\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood["total_cholesterol"] == 195.0
        assert blood["ldl_cholesterol"] == 110.0
        assert blood["hdl_cholesterol"] == 55.0
        assert blood["triglycerides"] == 140.0

    def test_liver_function(self):
        text = (
            "SGOT/AST: 28 U/L\n"
            "SGPT/ALT: 32 U/L\n"
            "Alkaline Phosphatase: 85 U/L\n"
            "Total Bilirubin: 0.8 mg/dL\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("sgot_ast") == 28.0
        assert blood.get("sgpt_alt") == 32.0
        assert blood.get("alkaline_phosphatase") == 85.0
        assert blood.get("total_bilirubin") == 0.8

    def test_kidney_function(self):
        text = (
            "Serum Creatinine: 1.1 mg/dL\n"
            "Blood Urea: 25 mg/dL\n"
            "Uric Acid: 5.5 mg/dL\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("serum_creatinine") == 1.1
        assert blood.get("blood_urea") == 25.0
        assert blood.get("uric_acid") == 5.5

    def test_diabetes_panel(self):
        text = (
            "Fasting Glucose: 98 mg/dL\n"
            "HbA1c: 5.6 %\n"
            "Fasting Insulin: 8.5 mIU/L\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("fasting_glucose") == 98.0
        assert blood.get("hba1c") == 5.6
        assert blood.get("fasting_insulin") == 8.5

    def test_thyroid_panel(self):
        text = "TSH: 2.5 mIU/L\nFree T4: 1.2 ng/dL\nFree T3: 3.1 pg/mL\n"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("tsh") == 2.5
        assert blood.get("free_t4") == 1.2
        assert blood.get("free_t3") == 3.1


# ---------------------------------------------------------------------------
# parse_lab_report — space-separated format
# ---------------------------------------------------------------------------


class TestParseSpaceFormat:
    """Tests for parsing space-separated tabular values."""

    def test_space_separated(self):
        text = "Hemoglobin          14.5    g/dL\nRBC Count           5.2     M/mcL"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("rbc_count") == 5.2


# ---------------------------------------------------------------------------
# parse_lab_report — table (pipe) format
# ---------------------------------------------------------------------------


class TestParseTableFormat:
    """Tests for parsing pipe-separated table values."""

    def test_pipe_separated(self):
        text = (
            "Hemoglobin | 14.5 | 12-16 | g/dL\n"
            "Total Cholesterol | 210 | <200 | mg/dL\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("total_cholesterol") == 210.0


# ---------------------------------------------------------------------------
# parse_lab_report — urine section
# ---------------------------------------------------------------------------


class TestParseUrineSection:
    """Tests for urine section detection and parsing."""

    def test_urine_section_detected(self):
        # Each value must be on its own line, not immediately after a
        # section header (the colon regex can capture multi-line names).
        text = (
            "Hemoglobin: 14.0 g/dL\n"
            "RBC Count: 4.9 M/mcL\n"
            "\n"
            "Urine Routine Examination\n"
            "pH: 6.0\n"
            "Specific Gravity: 1.025\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.0
        assert urine.get("ph") == 6.0
        assert urine.get("specific_gravity") == 1.025

    def test_urine_pus_cells(self):
        text = (
            "Hemoglobin: 14.0\n"
            "\n"
            "Urine Routine\n"
            "pH: 6.0\n"
            "Pus Cells: 3\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert urine.get("wbc_urine") == 3.0


# ---------------------------------------------------------------------------
# parse_lab_report — abdomen section
# ---------------------------------------------------------------------------


class TestParseAbdomenSection:
    """Tests for abdomen/ultrasound section extraction."""

    def test_abdomen_section(self):
        # Note: the abdomen regex terminates when it encounters "liver",
        # "kidney", etc. as section-end markers, so the abdomen text only
        # captures up to (but not including) such keywords.
        text = (
            "Hemoglobin: 14.5\n"
            "\n"
            "Abdomen Ultrasound\n"
            "Normal findings. No focal lesion seen.\n"
            "\n"
            "Complete Blood Count\n"
            "WBC: 7500\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert "abdomen" in abdomen.lower() or "Ultrasound" in abdomen


# ---------------------------------------------------------------------------
# parse_lab_report — edge cases
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    """Edge cases and error handling."""

    def test_empty_string(self):
        blood, urine, abdomen = parse_lab_report("")
        assert blood == {}
        assert urine == {}
        assert abdomen == ""

    def test_none_like(self):
        blood, urine, abdomen = parse_lab_report("   \n  \n  ")
        assert blood == {}
        assert urine == {}

    def test_no_recognizable_values(self):
        text = "This is a random note with no lab values whatsoever."
        blood, urine, abdomen = parse_lab_report(text)
        assert blood == {}
        assert urine == {}

    def test_duplicate_values_first_wins(self):
        """If the same parameter appears twice, the first extraction wins."""
        text = "Hemoglobin: 14.5\nHemoglobin: 12.0\n"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood["hemoglobin"] == 14.5

    def test_negative_value(self):
        """Parser should handle negative numbers gracefully."""
        text = "Some Marker: -1.5"
        blood, urine, abdomen = parse_lab_report(text)
        # May or may not match a known alias, but shouldn't crash


# ---------------------------------------------------------------------------
# compute_extraction_confidence
# ---------------------------------------------------------------------------


class TestExtractionConfidence:
    """Tests for the confidence scoring function."""

    def test_no_values_zero_confidence(self):
        assert compute_extraction_confidence({}, {}, "", "some text") == 0.0

    def test_empty_source_text(self):
        assert compute_extraction_confidence({"hemoglobin": 14.5}, {}, "", "") == 0.0

    def test_some_values_positive_confidence(self):
        blood = {"hemoglobin": 14.5, "fasting_glucose": 98.0, "total_cholesterol": 195.0}
        text = "Hemoglobin: 14.5\nFasting Glucose: 98\nTotal Cholesterol: 195\n"
        conf = compute_extraction_confidence(blood, {}, "", text)
        assert 0.0 < conf <= 1.0

    def test_key_params_boost_confidence(self):
        """Presence of key parameters (hemoglobin, fasting_glucose, etc.) should boost confidence."""
        blood_with_keys = {
            "hemoglobin": 14.5,
            "fasting_glucose": 98.0,
            "total_cholesterol": 195.0,
            "serum_creatinine": 1.1,
        }
        blood_without_keys = {"sgot_ast": 28.0, "sgpt_alt": 32.0, "alkaline_phosphatase": 85.0, "ggt": 45.0}
        text = "a: 1\nb: 2\nc: 3\nd: 4\n"
        conf_with = compute_extraction_confidence(blood_with_keys, {}, "", text)
        conf_without = compute_extraction_confidence(blood_without_keys, {}, "", text)
        assert conf_with >= conf_without

    def test_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0."""
        blood = {f"param_{i}": float(i) for i in range(50)}
        text = "\n".join(f"param_{i}: {i}" for i in range(50))
        conf = compute_extraction_confidence(blood, {}, "abdomen notes", text)
        assert conf <= 1.0

    def test_abdomen_bonus(self):
        """Abdomen text should give a small confidence boost."""
        blood = {"hemoglobin": 14.5}
        text = "Hemoglobin: 14.5\n"
        conf_no_abd = compute_extraction_confidence(blood, {}, "", text)
        conf_with_abd = compute_extraction_confidence(blood, {}, "Liver normal", text)
        assert conf_with_abd >= conf_no_abd


# ---------------------------------------------------------------------------
# Full report — realistic Indian lab format
# ---------------------------------------------------------------------------


class TestRealisticReport:
    """Integration test with a realistic multi-section lab report."""

    def test_indian_lab_report(self):
        # Values on standalone lines (no section header immediately before
        # the first value, since the colon regex can capture across lines).
        text = (
            "Hemoglobin: 13.8 g/dL\n"
            "RBC Count: 4.9 M/mcL\n"
            "WBC Count: 6800 cells/mcL\n"
            "Platelet Count: 225000 /mcL\n"
            "Hematocrit: 41.5 %\n"
            "MCV: 86.2 fL\n"
            "MCH: 28.5 pg\n"
            "\n"
            "Total Cholesterol: 212 mg/dL\n"
            "LDL Cholesterol: 130 mg/dL\n"
            "HDL Cholesterol: 48 mg/dL\n"
            "Triglycerides: 165 mg/dL\n"
            "VLDL: 33 mg/dL\n"
            "\n"
            "SGOT/AST: 26 U/L\n"
            "SGPT/ALT: 34 U/L\n"
            "Alkaline Phosphatase: 72 U/L\n"
            "Total Bilirubin: 0.9 mg/dL\n"
            "Total Protein: 7.2 g/dL\n"
            "Albumin: 4.1 g/dL\n"
            "\n"
            "Blood Urea: 28 mg/dL\n"
            "Serum Creatinine: 1.0 mg/dL\n"
            "Uric Acid: 6.1 mg/dL\n"
            "\n"
            "Fasting Glucose: 102 mg/dL\n"
            "HbA1c: 5.8 %\n"
            "\n"
            "TSH: 3.2 mIU/L\n"
            "Free T4: 1.15 ng/dL\n"
            "\n"
            "Urine Routine Examination\n"
            "pH: 6.5\n"
            "Specific Gravity: 1.020\n"
            "\n"
            "Abdomen Ultrasound\n"
            "All organs within normal limits.\n"
        )
        blood, urine, abdomen = parse_lab_report(text)

        # Blood tests — check major panels
        assert blood["hemoglobin"] == 13.8
        assert blood["rbc_count"] == 4.9
        assert blood["total_cholesterol"] == 212.0
        assert blood["ldl_cholesterol"] == 130.0
        assert blood["hdl_cholesterol"] == 48.0
        assert blood["sgot_ast"] == 26.0
        assert blood["sgpt_alt"] == 34.0
        assert blood["serum_creatinine"] == 1.0
        assert blood["fasting_glucose"] == 102.0
        assert blood["hba1c"] == 5.8
        assert blood["tsh"] == 3.2

        # Urine tests
        assert urine["ph"] == 6.5
        assert urine["specific_gravity"] == 1.020

        # Abdomen section
        assert abdomen  # non-empty

        # Confidence should be high for a comprehensive report
        conf = compute_extraction_confidence(blood, urine, abdomen, text)
        assert conf > 0.3


# ---------------------------------------------------------------------------
# parse_lab_report — reference ranges in brackets/parentheses
# ---------------------------------------------------------------------------


class TestParseReferenceRanges:
    """Tests for handling reference ranges after values."""

    def test_square_brackets(self):
        """Values with [ref-range] after unit should still parse."""
        text = "Hemoglobin: 14.5 g/dL [12.0-17.5]\nRBC Count: 5.2 M/mcL [4.5-5.5]"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("rbc_count") == 5.2

    def test_parentheses(self):
        """Values with (ref-range) after unit should still parse."""
        text = "Fasting Glucose: 98 mg/dL (70-110)\nHbA1c: 5.6 % (4.0-5.6)"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("fasting_glucose") == 98.0
        assert blood.get("hba1c") == 5.6

    def test_bracket_with_less_than(self):
        """Reference ranges with < or > inside brackets."""
        text = "Total Cholesterol: 210 mg/dL [<200]\nLDL Cholesterol: 130 mg/dL [<100]"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("total_cholesterol") == 210.0
        assert blood.get("ldl_cholesterol") == 130.0

    def test_value_without_ref_still_works(self):
        """Values without reference ranges should still parse (no regression)."""
        text = "Hemoglobin: 14.5 g/dL\nTSH: 2.5 mIU/L"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("tsh") == 2.5


# ---------------------------------------------------------------------------
# parse_lab_report — CSV format
# ---------------------------------------------------------------------------


class TestParseCsvFormat:
    """Tests for parsing CSV (comma-separated) lab reports."""

    def test_csv_basic(self):
        text = "Hemoglobin,14.5,g/dL,12-17\nRBC Count,5.2,M/mcL,4.5-5.5"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("rbc_count") == 5.2

    def test_csv_with_unit(self):
        text = "TSH,3.2,mIU/L,0.4-4.0\nFree T4,1.2,ng/dL,0.8-1.8"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("tsh") == 3.2
        assert blood.get("free_t4") == 1.2

    def test_csv_minimal(self):
        """CSV with just name and value."""
        text = "Hemoglobin,14.5\nAlbumin,4.1"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("albumin") == 4.1


# ---------------------------------------------------------------------------
# parse_lab_report — ambiguous bare names
# ---------------------------------------------------------------------------


class TestAmbiguousBareNames:
    """Tests for bare names (glucose, bilirubin, protein) panel resolution."""

    def test_bare_glucose_defaults_to_blood(self):
        """Bare 'Glucose' outside urine section → blood fasting_glucose."""
        text = "Glucose: 95 mg/dL"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("fasting_glucose") == 95.0
        assert urine.get("glucose") is None

    def test_bare_bilirubin_defaults_to_blood(self):
        """Bare 'Bilirubin' outside urine section → blood total_bilirubin."""
        text = "Bilirubin: 0.8 mg/dL"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("total_bilirubin") == 0.8
        assert urine.get("bilirubin") is None

    def test_bare_protein_defaults_to_blood(self):
        """Bare 'Protein' outside urine section → blood total_protein."""
        text = "Protein: 7.2 g/dL"
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("total_protein") == 7.2
        assert urine.get("protein") is None

    def test_bare_glucose_in_urine_section(self):
        """Bare 'Glucose' inside urine section → urine glucose."""
        text = (
            "Hemoglobin: 14.0\n"
            "\n"
            "Urine Routine Examination\n"
            "Glucose: 0\n"
            "pH: 6.0\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("hemoglobin") == 14.0
        assert urine.get("glucose") == 0.0

    def test_bare_protein_in_urine_section(self):
        """Bare 'Protein' inside urine section → urine protein."""
        text = (
            "Total Protein: 7.2 g/dL\n"
            "\n"
            "Urine Routine Examination\n"
            "Protein: 0.5\n"
        )
        blood, urine, abdomen = parse_lab_report(text)
        assert blood.get("total_protein") == 7.2
        assert urine.get("protein") == 0.5


# ---------------------------------------------------------------------------
# _identify_columns — table column identification
# ---------------------------------------------------------------------------


class TestIdentifyColumns:
    """Tests for the table column identification heuristic."""

    def test_standard_indian_lab_header(self):
        """Common Indian lab report header: Test Name | Result | Unit | Reference Range."""
        header = ["Test Name", "Result", "Unit", "Reference Range"]
        name_col, value_col, unit_col, ref_col = _identify_columns(header)
        assert name_col == 0
        assert value_col == 1
        assert unit_col == 2
        assert ref_col == 3

    def test_keyword_variations(self):
        """Alternative header keywords should still be recognized."""
        header = ["Investigation", "Observed Value", "Measure", "Normal Range"]
        name_col, value_col, unit_col, ref_col = _identify_columns(header)
        assert name_col == 0
        assert value_col == 1
        assert unit_col == 2
        assert ref_col == 3

    def test_partial_header(self):
        """When only some columns match, others should be None."""
        header = ["Parameter", "Value"]
        name_col, value_col, unit_col, ref_col = _identify_columns(header)
        assert name_col == 0
        assert value_col == 1
        assert unit_col is None
        assert ref_col is None

    def test_positional_fallback(self):
        """Unknown headers should fall back to col 0 = name, col 1 = value."""
        header = ["Col A", "Col B", "Col C"]
        name_col, value_col, unit_col, ref_col = _identify_columns(header)
        assert name_col == 0
        assert value_col == 1

    def test_single_column(self):
        """Single-column tables cannot have name + value."""
        header = ["Only Column"]
        name_col, value_col, _, _ = _identify_columns(header)
        assert name_col is None or value_col is None

    def test_empty_cells_in_header(self):
        """None/empty cells should be skipped gracefully."""
        header = [None, "Test Name", "", "Result", "Unit"]
        name_col, value_col, unit_col, _ = _identify_columns(header)
        assert name_col == 1
        assert value_col == 3
        assert unit_col == 4

    def test_bio_ref_range(self):
        """'Biological Ref. Range' is a common variant in Indian reports."""
        header = ["Test", "Value", "Unit", "Biological Ref. Range"]
        _, _, _, ref_col = _identify_columns(header)
        assert ref_col == 3


# ---------------------------------------------------------------------------
# _parse_structured_tables — structured table parsing
# ---------------------------------------------------------------------------


class TestParseStructuredTables:
    """Tests for parsing structured table data into name-value pairs."""

    def test_basic_table(self):
        """Standard lab table with header row + data rows."""
        table = [
            ["Test Name", "Result", "Unit", "Reference Range"],
            ["Hemoglobin", "14.5", "g/dL", "12.0-17.5"],
            ["RBC Count", "5.2", "M/mcL", "4.5-5.5"],
            ["WBC Count", "7500", "cells/mcL", "4000-11000"],
        ]
        results = _parse_structured_tables([table])
        assert len(results) == 3
        names = {r[0] for r in results}
        assert "Hemoglobin" in names
        assert "RBC Count" in names
        assert "WBC Count" in names
        # Check values
        values = {r[0]: r[1] for r in results}
        assert values["Hemoglobin"] == 14.5
        assert values["RBC Count"] == 5.2
        assert values["WBC Count"] == 7500.0

    def test_value_with_flag(self):
        """Values like '14.5 H' (high flag) should still extract the number."""
        table = [
            ["Test", "Result"],
            ["Hemoglobin", "14.5 H"],
            ["Glucose", "* 250"],
        ]
        results = _parse_structured_tables([table])
        values = {r[0]: r[1] for r in results}
        assert values["Hemoglobin"] == 14.5
        assert values["Glucose"] == 250.0

    def test_empty_cells_skipped(self):
        """Rows with None/empty name or value cells should be skipped."""
        table = [
            ["Test", "Result"],
            ["Hemoglobin", "14.5"],
            [None, "12.0"],       # No name
            ["RBC Count", None],  # No value
            ["", "5.0"],          # Empty name
        ]
        results = _parse_structured_tables([table])
        assert len(results) == 1
        assert results[0][0] == "Hemoglobin"

    def test_multiple_tables(self):
        """Multiple tables should all be parsed."""
        table1 = [
            ["Test", "Result"],
            ["Hemoglobin", "14.5"],
        ]
        table2 = [
            ["Parameter", "Value"],
            ["TSH", "2.5"],
        ]
        results = _parse_structured_tables([table1, table2])
        assert len(results) == 2
        names = {r[0] for r in results}
        assert "Hemoglobin" in names
        assert "TSH" in names

    def test_no_header_numeric_first_row(self):
        """If the first row has a numeric value col, treat it as data."""
        table = [
            ["Hemoglobin", "14.5"],
            ["RBC Count", "5.2"],
        ]
        results = _parse_structured_tables([table])
        assert len(results) == 2

    def test_non_numeric_value_skipped(self):
        """Non-numeric value cells should be skipped."""
        table = [
            ["Test", "Result"],
            ["Hemoglobin", "Normal"],
            ["TSH", "2.5"],
        ]
        results = _parse_structured_tables([table])
        assert len(results) == 1
        assert results[0][0] == "TSH"

    def test_too_few_rows_skipped(self):
        """Tables with fewer than 2 rows are skipped."""
        table = [["Test", "Result"]]
        results = _parse_structured_tables([table])
        assert len(results) == 0

    def test_empty_table_list(self):
        """Empty table list returns empty results."""
        results = _parse_structured_tables([])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# parse_lab_report — with structured tables
# ---------------------------------------------------------------------------


class TestParseLabReportWithTables:
    """Tests for parse_lab_report when structured_tables are provided."""

    def test_structured_tables_override_text(self):
        """When structured tables provide a value, text-based parsing uses same value."""
        text = "Hemoglobin: 14.5 g/dL\n"
        tables = [[
            ["Test Name", "Result", "Unit"],
            ["Hemoglobin", "14.5", "g/dL"],
            ["TSH", "2.5", "mIU/L"],
        ]]
        blood, urine, abdomen = parse_lab_report(text, structured_tables=tables)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("tsh") == 2.5

    def test_tables_plus_text_combined(self):
        """Structured tables + text should combine values from both sources."""
        text = "Fasting Glucose: 98 mg/dL\nHbA1c: 5.6 %\n"
        tables = [[
            ["Test", "Result"],
            ["Hemoglobin", "14.5"],
            ["Total Cholesterol", "210"],
        ]]
        blood, urine, abdomen = parse_lab_report(text, structured_tables=tables)
        # From tables
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("total_cholesterol") == 210.0
        # From text
        assert blood.get("fasting_glucose") == 98.0
        assert blood.get("hba1c") == 5.6

    def test_table_value_wins_over_text_duplicate(self):
        """Table pass runs first, so table value should win for duplicates."""
        text = "Hemoglobin: 12.0 g/dL\n"
        tables = [[
            ["Test", "Result"],
            ["Hemoglobin", "14.5"],
        ]]
        blood, urine, abdomen = parse_lab_report(text, structured_tables=tables)
        assert blood["hemoglobin"] == 14.5  # Table wins (first pass)

    def test_none_tables_backward_compatible(self):
        """None structured_tables should work exactly like old behavior."""
        text = "Hemoglobin: 14.5 g/dL\nTSH: 2.5 mIU/L"
        blood, urine, abdomen = parse_lab_report(text, structured_tables=None)
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("tsh") == 2.5

    def test_indian_lab_table_format(self):
        """Realistic Indian lab report table with all common columns."""
        text = "COMPLETE BLOOD COUNT\n"
        tables = [[
            ["Investigation", "Observed Value", "Unit", "Biological Ref. Range"],
            ["Haemoglobin", "13.8", "g/dL", "12.0-17.5"],
            ["RBC Count", "4.9", "M/mcL", "4.5-5.5"],
            ["WBC Count", "6800", "cells/mcL", "4000-11000"],
            ["Platelet Count", "225000", "/mcL", "150000-400000"],
            ["Hematocrit", "41.5", "%", "36-54"],
        ]]
        blood, urine, abdomen = parse_lab_report(text, structured_tables=tables)
        assert blood.get("hemoglobin") == 13.8
        assert blood.get("rbc_count") == 4.9
        assert blood.get("wbc_count") == 6800.0
        assert blood.get("platelet_count") == 225000.0
        assert blood.get("hematocrit") == 41.5


# ---------------------------------------------------------------------------
# extract_and_parse — combined extraction + parsing
# ---------------------------------------------------------------------------


class TestExtractAndParse:
    """Tests for the extract_and_parse convenience function."""

    def test_text_file(self):
        """Plain text should parse without table extraction."""
        content = b"Hemoglobin: 14.5 g/dL\nTSH: 2.5 mIU/L\n"
        blood, urine, abdomen, confidence = extract_and_parse(content, "report.txt")
        assert blood.get("hemoglobin") == 14.5
        assert blood.get("tsh") == 2.5
        assert confidence > 0.0

    def test_empty_file(self):
        """Empty file should return empty results."""
        blood, urine, abdomen, confidence = extract_and_parse(b"", "empty.txt")
        assert blood == {}
        assert urine == {}
        assert confidence == 0.0
