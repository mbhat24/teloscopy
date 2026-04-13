"""Lab Report Parser Module.

Extracts structured lab values from uploaded lab reports (PDF text, OCR text,
or plain text).  Supports common Indian and international lab report formats
including tabular layouts, colon-separated values, and free-text paragraphs.

This module is strictly for **research / educational** purposes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def check_extraction_support() -> dict[str, bool]:
    """Check which extraction libraries are available."""
    support = {"pdf": False, "ocr": False}
    for mod in ("fitz", "pdfplumber", "pypdf", "PyPDF2"):
        try:
            __import__(mod)
            support["pdf"] = True
            break
        except ImportError:
            continue
    try:
        __import__("pytesseract")
        __import__("PIL")
        support["ocr"] = True
    except ImportError:
        pass
    return support


# ---------------------------------------------------------------------------
# Load parameter aliases from JSON
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"


def _load_aliases() -> tuple[dict[str, str], dict[str, str]]:
    """Build reverse alias lookup dicts from parameter_aliases.json.

    Returns
    -------
    blood_alias_map : dict[str, str]
        Maps lowercase alias → BloodTestPanel field name.
    urine_alias_map : dict[str, str]
        Maps lowercase alias → UrineTestPanel field name.
    """
    with open(_DATA_DIR / "parameter_aliases.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)

    blood_map: dict[str, str] = {}
    for field_name, aliases in data.get("blood_test_aliases", {}).items():
        for alias in aliases:
            blood_map[alias.lower().strip()] = field_name

    urine_map: dict[str, str] = {}
    for field_name, aliases in data.get("urine_test_aliases", {}).items():
        for alias in aliases:
            urine_map[alias.lower().strip()] = field_name

    return blood_map, urine_map


BLOOD_ALIAS_MAP, URINE_ALIAS_MAP = _load_aliases()

# Combined map for quick lookup (blood takes priority over urine for
# overlapping names like "protein", "glucose", "bilirubin" — those are
# handled via context detection in the parser).
_ALL_ALIASES: dict[str, tuple[str, str]] = {}  # alias -> (field, "blood"|"urine")
for _alias, _field in URINE_ALIAS_MAP.items():
    _ALL_ALIASES[_alias] = (_field, "urine")
for _alias, _field in BLOOD_ALIAS_MAP.items():
    _ALL_ALIASES[_alias] = (_field, "blood")


# ---------------------------------------------------------------------------
# Regex patterns for value extraction
# ---------------------------------------------------------------------------

# Matches a numeric value (integer or float, optionally negative)
_NUM_PATTERN = r"[-+]?\d+(?:\.\d+)?"

# Reference range pattern — matches trailing "[12-17]", "(12.0-17.5)",
# "12-17", "< 200", "> 5.0" etc. that often follow units in Indian lab reports.
_REF_RANGE_SUFFIX = (
    r"(?:"
    r"[ \t]*[\[\(][\d\.\-<> /,]+[\]\)]"   # e.g. [12-17], (12.0-17.5), [<200]
    r"|[ \t]+\d+\.?\d*[ \t]*[\-–][ \t]*\d+\.?\d*"  # e.g. 12-17 or 12.0 - 17.5
    r"|[ \t]+[<>]=?[ \t]*\d+\.?\d*"        # e.g. <200 or >= 5.0
    r")*"
)

# Pattern: "Parameter Name : 14.5 g/dL" or "Parameter Name: 14.5"
# Also handles: "Parameter Name: 14.5 g/dL [12.0-17.5]" or "(12-17)"
_COLON_PATTERN = re.compile(
    r"^[ \t\-\*]*"                         # leading whitespace/bullets (no newline)
    r"(?P<name>[A-Za-z0-9 \t\-/\(\)\.\,%&']+?)"  # parameter name (no newline)
    r"[ \t]*[:=][ \t]*"                    # separator (colon or equals)
    r"(?P<value>" + _NUM_PATTERN + r")"    # numeric value
    r"(?:[ \t]*(?P<unit>[A-Za-z/%µ\.]+(?:[/ \t][A-Za-z]+)*))??"  # optional unit (lazy, no digits/brackets)
    + _REF_RANGE_SUFFIX +                  # optional trailing reference range
    r"[ \t]*$",
    re.MULTILINE,
)

# Pattern: "Parameter Name   14.5  g/dL" (whitespace separated, no colon)
_SPACE_PATTERN = re.compile(
    r"^[ \t\-\*]*"
    r"(?P<name>[A-Za-z][A-Za-z0-9 \t\-/\(\)\.\,%&']{2,40}?)"
    r"[ \t]{2,}"                            # at least 2 spaces/tabs separating
    r"(?P<value>" + _NUM_PATTERN + r")"
    r"(?:[ \t]+(?P<unit>[A-Za-z/%µ\.\-]+))?"
    + _REF_RANGE_SUFFIX +                  # optional trailing reference range
    r"",
    re.MULTILINE,
)

# Pattern: table format "Parameter | 14.5 | 13-17 | g/dL"
_TABLE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 \t\-/\(\)\.\,%&']+?)"
    r"[ \t]*\|[ \t]*"
    r"(?P<value>" + _NUM_PATTERN + r")"
    r"(?:[ \t]*\|[^|\n]*)*",               # remaining columns (ref range, unit) — stop at newline
    re.MULTILINE,
)

# Pattern: CSV format "Parameter,14.5,g/dL,12-17"
_CSV_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?P<name>[A-Za-z][A-Za-z0-9 \t\-/\(\)\.\,%&']+?)"
    r"[ \t]*,[ \t]*"
    r"(?P<value>" + _NUM_PATTERN + r")"
    r"(?:[ \t]*,[^\n]*)?",                  # remaining CSV columns (unit, ref range)
    re.MULTILINE,
)

# Pattern for detecting abdomen/ultrasound section
_ABDOMEN_SECTION_PATTERN = re.compile(
    r"(?:abdomen|abdominal|ultrasound|ultrasonography|usg|sonography)"
    r"[\s\S]*?(?=\n\s*(?:complete blood|cbc|blood test|urine|haematology|biochemistry|lipid|liver|kidney|thyroid|$))",
    re.IGNORECASE,
)

# Pattern to detect urine section headers
_URINE_SECTION_MARKERS = re.compile(
    r"(?:urine\s+(?:routine|analysis|examination|test|microscopy)|urinalysis)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text extraction from PDF
# ---------------------------------------------------------------------------


def _ocr_pdf_pages(file_bytes: bytes) -> str:
    """Render PDF pages as images and OCR them (fallback for scanned PDFs).

    Requires PyMuPDF (fitz) for page rendering and pytesseract + Pillow
    for OCR.  Returns empty string if any dependency is missing.
    """
    try:
        import fitz  # PyMuPDF for page rendering
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.debug("OCR fallback unavailable: need fitz + pytesseract + Pillow")
        return ""

    try:
        import io

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[str] = []
        for page_num, page in enumerate(doc):
            # Render page at 300 DPI for better OCR accuracy
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(image)
            if page_text.strip():
                pages.append(page_text)
            logger.debug("OCR page %d: %d chars", page_num + 1, len(page_text))
        doc.close()
        text = "\n".join(pages)
        if text.strip():
            logger.info("PDF OCR extracted via PyMuPDF + pytesseract (%d chars)", len(text))
        return text
    except Exception as exc:
        logger.warning("PDF OCR fallback failed: %s", exc)
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from a PDF file.

    Tries multiple PDF libraries in order of preference:
    1. PyMuPDF (fitz) — fastest, most reliable
    2. pdfplumber — good table extraction
    3. PyPDF2 / pypdf — basic text extraction
    4. OCR fallback — render pages as images and run pytesseract

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the PDF file.

    Returns
    -------
    str
        Extracted text from all pages.
    """
    has_pdf_lib = False

    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF

        has_pdf_lib = True
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            # sort=True reorders text blocks by visual reading order (top-to-
            # bottom, left-to-right) so multi-column layouts don't interleave.
            pages.append(page.get_text(sort=True))
        doc.close()
        text = "\n".join(pages)
        if text.strip():
            logger.info("PDF text extracted via PyMuPDF (%d chars)", len(text))
            return text
    except ImportError:
        logger.debug("PyMuPDF (fitz) not available, trying pdfplumber")
    except Exception as exc:
        logger.warning("PyMuPDF extraction failed: %s", exc)

    # Try pdfplumber
    try:
        import io

        import pdfplumber

        has_pdf_lib = True
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        text = "\n".join(pages)
        if text.strip():
            logger.info("PDF text extracted via pdfplumber (%d chars)", len(text))
            return text
    except ImportError:
        logger.debug("pdfplumber not available, trying pypdf")
    except Exception as exc:
        logger.warning("pdfplumber extraction failed: %s", exc)

    # Try pypdf / PyPDF2
    try:
        import io

        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]

        has_pdf_lib = True
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        text = "\n".join(pages)
        if text.strip():
            logger.info("PDF text extracted via pypdf (%d chars)", len(text))
            return text
    except ImportError:
        logger.debug("pypdf/PyPDF2 not available")
    except Exception as exc:
        logger.warning("pypdf extraction failed: %s", exc)

    # OCR fallback — render PDF pages as images and run pytesseract.
    # This handles scanned PDFs with no text layer.
    text = _ocr_pdf_pages(file_bytes)
    if text.strip():
        return text

    if not has_pdf_lib:
        raise RuntimeError(
            "No PDF parsing library is installed on the server. "
            "The administrator needs to install PyMuPDF, pdfplumber, or pypdf. "
            "As a workaround, you can copy-paste your lab report text into the manual entry form."
        )

    # All methods returned empty — likely a scanned PDF without OCR support
    raise RuntimeError(
        "The PDF appears to be a scanned image without a text layer. "
        "OCR libraries (pytesseract, Pillow) are not available to process it. "
        "As a workaround, you can upload the scanned page as an image file "
        "(PNG/JPEG) or manually type your lab values into the form."
    )


# ---------------------------------------------------------------------------
# Structured table extraction from PDF
# ---------------------------------------------------------------------------


def extract_tables_from_pdf(file_bytes: bytes) -> list[list[list[str | None]]]:
    """Extract structured tables from a PDF using dedicated table-detection APIs.

    Tries PyMuPDF ``page.find_tables()`` first (no extra dependencies), then
    falls back to pdfplumber ``page.extract_tables()``.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the PDF file.

    Returns
    -------
    list[list[list[str | None]]]
        A list of tables. Each table is a list of rows; each row is a list
        of cell strings (or ``None`` for empty cells).
    """
    tables: list[list[list[str | None]]] = []

    # Strategy 1: PyMuPDF find_tables (available since PyMuPDF 1.23.0)
    try:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            try:
                found = page.find_tables()
                for tbl in found.tables:
                    rows = tbl.extract()
                    if rows and len(rows) >= 2:  # need header + at least 1 data row
                        tables.append(rows)
            except Exception as exc:
                logger.debug("find_tables failed on page: %s", exc)
        doc.close()
        if tables:
            logger.info("Extracted %d tables via PyMuPDF find_tables", len(tables))
            return tables
    except (ImportError, AttributeError):
        logger.debug("PyMuPDF find_tables not available")
    except Exception as exc:
        logger.warning("PyMuPDF table extraction failed: %s", exc)

    # Strategy 2: pdfplumber extract_tables
    try:
        import io
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    page_tables = page.extract_tables()
                    for tbl in (page_tables or []):
                        if tbl and len(tbl) >= 2:
                            tables.append(tbl)
                except Exception as exc:
                    logger.debug("pdfplumber extract_tables failed on page: %s", exc)
        if tables:
            logger.info("Extracted %d tables via pdfplumber", len(tables))
            return tables
    except ImportError:
        logger.debug("pdfplumber not available for table extraction")
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed: %s", exc)

    return tables


# ---------------------------------------------------------------------------
# Parse structured table data into lab values
# ---------------------------------------------------------------------------

# Column header patterns for identifying table columns
_NAME_COL_PATTERNS = re.compile(
    r"(?:test|parameter|investigation|analyte|component|biomarker|name)",
    re.IGNORECASE,
)
_VALUE_COL_PATTERNS = re.compile(
    r"(?:result|value|observed|reading|finding|report)",
    re.IGNORECASE,
)
_UNIT_COL_PATTERNS = re.compile(r"(?:unit|measure)", re.IGNORECASE)
_REF_COL_PATTERNS = re.compile(
    r"(?:reference|ref|normal|range|biological|bio\.?\s*ref)",
    re.IGNORECASE,
)


def _identify_columns(
    header_row: list[str | None],
) -> tuple[int | None, int | None, int | None, int | None]:
    """Identify which column indices correspond to name, value, unit, reference.

    Uses both keyword matching on header text and positional heuristics.
    Indian lab reports commonly have columns ordered as:
    Test Name | Result/Value | Unit | Reference Range (| Method)

    Returns
    -------
    tuple of (name_col, value_col, unit_col, ref_col)
        Column indices, or None if not identified.
    """
    name_col = value_col = unit_col = ref_col = None
    cells = [str(c).strip() if c else "" for c in header_row]

    for i, cell in enumerate(cells):
        if not cell:
            continue
        if _NAME_COL_PATTERNS.search(cell) and name_col is None:
            name_col = i
        elif _VALUE_COL_PATTERNS.search(cell) and value_col is None:
            value_col = i
        elif _UNIT_COL_PATTERNS.search(cell) and unit_col is None:
            unit_col = i
        elif _REF_COL_PATTERNS.search(cell) and ref_col is None:
            ref_col = i

    # Positional fallback: if we couldn't identify columns by header text
    # but have at least 2 columns, assume col 0 = name, col 1 = value.
    ncols = len(cells)
    if name_col is None and ncols >= 2:
        name_col = 0
    if value_col is None and ncols >= 2:
        # Pick the first non-name column that contains a number in any data row
        value_col = 1

    return name_col, value_col, unit_col, ref_col


def _parse_structured_tables(
    tables: list[list[list[str | None]]],
    in_urine_section_fn: Any = None,
) -> list[tuple[str, float, int]]:
    """Parse structured table data into (name, value, position) tuples.

    Parameters
    ----------
    tables : list of tables (each table = list of rows, each row = list of cells)
    in_urine_section_fn : callable, optional
        Not used here (urine detection happens at the recording stage).

    Returns
    -------
    list[tuple[str, float, int]]
        List of (parameter_name, numeric_value, pseudo_position) tuples.
    """
    results: list[tuple[str, float, int]] = []
    pos = 0

    for table in tables:
        if not table or len(table) < 2:
            continue

        # Try to identify columns from the first row (header)
        name_col, value_col, unit_col, ref_col = _identify_columns(table[0])

        if name_col is None or value_col is None:
            continue

        # Check if the first row is actually a header (no numeric value in the
        # value column) or if it's a data row
        first_val = str(table[0][value_col]).strip() if table[0][value_col] else ""
        start_row = 0 if re.match(r"^[-+]?\d+(?:\.\d+)?$", first_val) else 1

        for row in table[start_row:]:
            if not row or len(row) <= max(name_col, value_col):
                continue

            name_cell = row[name_col]
            value_cell = row[value_col]

            if not name_cell or not value_cell:
                continue

            name = str(name_cell).strip()
            val_str = str(value_cell).strip()

            # Skip sub-header rows (all-caps section titles, empty values, etc.)
            if not name or not val_str:
                continue

            # Extract numeric value from the cell (handle cases like "14.5 H" or
            # "* 14.5" where flags are embedded in the value column)
            num_match = re.search(r"[-+]?\d+(?:\.\d+)?", val_str)
            if not num_match:
                continue

            try:
                value = float(num_match.group())
            except (ValueError, TypeError):
                continue

            results.append((name, value, pos))
            pos += 100  # pseudo-positions spaced apart

    logger.debug("Structured table parsing found %d name-value pairs", len(results))
    return results


# ---------------------------------------------------------------------------
# Text extraction from image (OCR)
# ---------------------------------------------------------------------------


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from an image using OCR.

    Uses pytesseract + Pillow if available.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the image file (PNG, JPEG, etc.).

    Returns
    -------
    str
        Extracted text, or empty string if OCR is unavailable.
    """
    try:
        import io

        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        if text.strip():
            logger.info("Image text extracted via pytesseract (%d chars)", len(text))
            return text
    except ImportError:
        raise RuntimeError(
            "OCR libraries (pytesseract, Pillow) are not installed on the server. "
            "Image-based lab reports cannot be processed. "
            "As a workaround, you can manually type your lab values into the form."
        )
    except Exception as exc:
        logger.warning("OCR extraction failed: %s", exc)

    return ""


# ---------------------------------------------------------------------------
# Core lab report parser
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Normalize a parameter name for alias lookup."""
    # Remove trailing units in parentheses, e.g. "Hemoglobin (g/dL)"
    name = re.sub(r"\s*\(.*?\)\s*$", "", name)
    # Remove trailing colons, dashes, asterisks
    name = name.strip(" \t\n\r:=-*#")
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name.lower().strip()


def _resolve_alias(
    name: str,
    in_urine_section: bool = False,
) -> tuple[str, str] | None:
    """Resolve a parameter name to (field_name, 'blood'|'urine').

    Parameters
    ----------
    name : str
        Raw parameter name from the report.
    in_urine_section : bool
        If True, prefer urine aliases for ambiguous names.

    Returns
    -------
    tuple[str, str] | None
        (field_name, panel_type) or None if unrecognized.
    """
    norm = _normalize_name(name)
    if not norm:
        return None

    # Direct lookup in combined map
    if norm in _ALL_ALIASES:
        field, panel = _ALL_ALIASES[norm]
        # For ambiguous names in urine sections, prefer urine
        if in_urine_section and norm in URINE_ALIAS_MAP:
            return URINE_ALIAS_MAP[norm], "urine"
        return field, panel

    # Try progressively shorter prefixes for fuzzy matching
    # e.g. "hemoglobin (hb)" -> try "hemoglobin"
    words = norm.split()
    for length in range(len(words), 0, -1):
        candidate = " ".join(words[:length])
        if candidate in _ALL_ALIASES:
            field, panel = _ALL_ALIASES[candidate]
            if in_urine_section and candidate in URINE_ALIAS_MAP:
                return URINE_ALIAS_MAP[candidate], "urine"
            return field, panel

    return None


def parse_lab_report(
    text: str,
    structured_tables: list[list[list[str | None]]] | None = None,
) -> tuple[dict[str, float], dict[str, float], str]:
    """Parse lab report text and extract structured lab values.

    Handles multiple common formats:
    - Structured tables extracted from PDF (highest confidence)
    - Colon-separated: "Hemoglobin: 14.5 g/dL"
    - Space-separated: "Hemoglobin     14.5    g/dL"
    - Table format:    "Hemoglobin | 14.5 | 13-17 | g/dL"
    - CSV format:      "Hemoglobin,14.5,g/dL,13-17"
    - Reference ranges: "Hemoglobin: 14.5 g/dL [12.0-17.5]"
    - Mixed formats within the same report

    Parameters
    ----------
    text : str
        Raw text extracted from a lab report (PDF, OCR, or manual paste).
    structured_tables : list, optional
        Pre-extracted table data from PDF table detection APIs.
        Each table is a list of rows, each row a list of cell strings.

    Returns
    -------
    blood_tests : dict[str, float]
        Extracted blood test values keyed by BloodTestPanel field names.
    urine_tests : dict[str, float]
        Extracted urine test values keyed by UrineTestPanel field names.
    abdomen_text : str
        Extracted abdomen/ultrasound section text (if found).
    """
    if not text or not text.strip():
        return {}, {}, ""

    blood_tests: dict[str, float] = {}
    urine_tests: dict[str, float] = {}
    unrecognized: list[str] = []

    # Detect urine section boundaries
    urine_section_starts: list[int] = []
    for m in _URINE_SECTION_MARKERS.finditer(text):
        urine_section_starts.append(m.start())

    # Extract abdomen section
    abdomen_text = ""
    abdomen_match = _ABDOMEN_SECTION_PATTERN.search(text)
    if abdomen_match:
        abdomen_text = abdomen_match.group(0).strip()

    def _is_in_urine_section(pos: int) -> bool:
        """Check if a position in text falls within a urine section."""
        if not urine_section_starts:
            return False
        for start in urine_section_starts:
            # Consider urine section to extend ~2000 chars from its header
            if start <= pos <= start + 2000:
                return True
        return False

    # Track which fields we've already found to avoid duplicates
    found_fields: set[str] = set()

    def _record_value(name: str, value: float, pos: int) -> bool:
        """Try to match a name-value pair to a known parameter."""
        in_urine = _is_in_urine_section(pos)
        resolved = _resolve_alias(name, in_urine_section=in_urine)
        if resolved is None:
            return False
        field, panel = resolved
        if field in found_fields:
            return True  # Already found, skip duplicate
        found_fields.add(field)
        if panel == "blood":
            blood_tests[field] = value
        else:
            urine_tests[field] = value
        return True

    # Pass 0: Structured table data (highest confidence — comes from PDF
    # table-detection APIs like PyMuPDF find_tables or pdfplumber)
    if structured_tables:
        table_pairs = _parse_structured_tables(structured_tables)
        for name, value, pos in table_pairs:
            _record_value(name, value, pos)

    # Pass 1: Table format (pipe-separated)
    for m in _TABLE_PATTERN.finditer(text):
        name = m.group("name").strip()
        try:
            value = float(m.group("value"))
        except (ValueError, TypeError):
            continue
        _record_value(name, value, m.start())

    # Pass 2: Colon-separated format
    for m in _COLON_PATTERN.finditer(text):
        name = m.group("name").strip()
        try:
            value = float(m.group("value"))
        except (ValueError, TypeError):
            continue
        _record_value(name, value, m.start())

    # Pass 3: CSV format (comma-separated)
    for m in _CSV_PATTERN.finditer(text):
        name = m.group("name").strip()
        try:
            value = float(m.group("value"))
        except (ValueError, TypeError):
            continue
        _record_value(name, value, m.start())

    # Pass 4: Space-separated format (more greedy, use last)
    for m in _SPACE_PATTERN.finditer(text):
        name = m.group("name").strip()
        try:
            value = float(m.group("value"))
        except (ValueError, TypeError):
            continue
        _record_value(name, value, m.start())

    total = len(blood_tests) + len(urine_tests)
    logger.info(
        "Parsed lab report: %d blood values, %d urine values, abdomen=%s",
        len(blood_tests),
        len(urine_tests),
        bool(abdomen_text),
    )
    return blood_tests, urine_tests, abdomen_text


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def compute_extraction_confidence(
    blood_tests: dict[str, float],
    urine_tests: dict[str, float],
    abdomen_text: str,
    source_text: str,
) -> float:
    """Compute a confidence score for the extraction quality.

    The score is based on:
    - Number of recognized parameters (more = higher confidence)
    - Ratio of extracted values to total lines with numbers
    - Presence of key critical parameters

    Returns
    -------
    float
        Confidence between 0.0 and 1.0.
    """
    if not source_text.strip():
        return 0.0

    total_extracted = len(blood_tests) + len(urine_tests)
    if total_extracted == 0:
        return 0.0

    # Count lines that look like they contain lab values
    value_lines = 0
    for line in source_text.split("\n"):
        if re.search(r"\d+\.?\d*", line) and len(line.strip()) > 5:
            value_lines += 1

    # Base confidence from extraction ratio
    if value_lines > 0:
        ratio = min(total_extracted / value_lines, 1.0)
    else:
        ratio = 0.0

    # Bonus for key parameters
    key_params = {"hemoglobin", "fasting_glucose", "total_cholesterol", "serum_creatinine"}
    key_found = sum(1 for k in key_params if k in blood_tests)
    key_bonus = key_found * 0.05

    # Bonus for volume of parameters
    volume_score = min(total_extracted / 15.0, 1.0) * 0.3

    # Combine scores
    confidence = min(ratio * 0.5 + volume_score + key_bonus + (0.05 if abdomen_text else 0.0), 1.0)

    return round(confidence, 3)


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------


def detect_file_type(file_bytes: bytes, filename: str) -> str:
    """Detect whether a file is PDF, image, or text.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename.

    Returns
    -------
    str
        One of: 'pdf', 'image', 'text', 'unknown'.
    """
    ext = Path(filename).suffix.lower() if filename else ""

    # Check magic bytes for PDF
    if file_bytes[:5] == b"%PDF-":
        return "pdf"
    if ext == ".pdf":
        return "pdf"

    # Check for common image magic bytes
    image_signatures = [
        (b"\x89PNG\r\n\x1a\n", "image"),  # PNG
        (b"\xff\xd8\xff", "image"),         # JPEG
        (b"II\x2a\x00", "image"),           # TIFF LE
        (b"MM\x00\x2a", "image"),           # TIFF BE
        (b"BM", "image"),                   # BMP
        (b"RIFF", "image"),                 # WebP
    ]
    for sig, ftype in image_signatures:
        if file_bytes[: len(sig)] == sig:
            return ftype

    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}:
        return "image"

    # Check for text content
    if ext in {".txt", ".text", ".csv", ".tsv"}:
        return "text"

    # Try to decode as text
    try:
        file_bytes[:1000].decode("utf-8")
        return "text"
    except (UnicodeDecodeError, ValueError):
        pass

    return "unknown"


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from a file based on its detected type.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename.

    Returns
    -------
    str
        Extracted text content.
    """
    file_type = detect_file_type(file_bytes, filename)

    if file_type == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type == "image":
        return extract_text_from_image(file_bytes)
    elif file_type == "text":
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except UnicodeDecodeError:
                return ""
    else:
        logger.warning("Unknown file type for '%s', attempting text decode", filename)
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return ""


def extract_and_parse(
    file_bytes: bytes,
    filename: str,
) -> tuple[dict[str, float], dict[str, float], str, float]:
    """Extract text + tables from a file and parse lab values in one step.

    This is the recommended entry point for PDF lab reports — it combines
    text extraction, structured table extraction, lab-value parsing, and
    confidence scoring into a single call.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename.

    Returns
    -------
    blood_tests : dict[str, float]
    urine_tests : dict[str, float]
    abdomen_text : str
    confidence : float
    """
    file_type = detect_file_type(file_bytes, filename)

    text = extract_text(file_bytes, filename)

    # For PDFs, also try structured table extraction
    structured_tables: list[list[list[str | None]]] | None = None
    if file_type == "pdf":
        try:
            structured_tables = extract_tables_from_pdf(file_bytes) or None
        except Exception as exc:
            logger.warning("Table extraction failed, continuing with text only: %s", exc)

    blood, urine, abdomen = parse_lab_report(text, structured_tables=structured_tables)
    confidence = compute_extraction_confidence(blood, urine, abdomen, text)
    return blood, urine, abdomen, confidence
