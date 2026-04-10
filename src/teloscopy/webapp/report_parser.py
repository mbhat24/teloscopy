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

# Pattern: "Parameter Name : 14.5 g/dL" or "Parameter Name: 14.5"
_COLON_PATTERN = re.compile(
    r"^[\s\-\*]*"                          # leading whitespace/bullets
    r"(?P<name>[A-Za-z0-9\s\-/\(\)\.\,%&']+?)"  # parameter name
    r"\s*[:=]\s*"                           # separator (colon or equals)
    r"(?P<value>" + _NUM_PATTERN + r")"     # numeric value
    r"(?:\s*(?P<unit>[A-Za-z/%µ\.\s\-\d]+))?"  # optional unit
    r"\s*$",
    re.MULTILINE,
)

# Pattern: "Parameter Name   14.5  g/dL" (whitespace separated, no colon)
_SPACE_PATTERN = re.compile(
    r"^[\s\-\*]*"
    r"(?P<name>[A-Za-z][A-Za-z0-9\s\-/\(\)\.\,%&']{2,40}?)"
    r"\s{2,}"                               # at least 2 spaces separating
    r"(?P<value>" + _NUM_PATTERN + r")"
    r"(?:\s+(?P<unit>[A-Za-z/%µ\.\-]+))?"
    r"",
    re.MULTILINE,
)

# Pattern: table format "Parameter | 14.5 | 13-17 | g/dL"
_TABLE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9\s\-/\(\)\.\,%&']+?)"
    r"\s*\|\s*"
    r"(?P<value>" + _NUM_PATTERN + r")"
    r"(?:\s*\|[^|\n]*)*",                   # remaining columns (ref range, unit) — stop at newline
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


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from a PDF file.

    Tries multiple PDF libraries in order of preference:
    1. PyMuPDF (fitz) — fastest, most reliable
    2. pdfplumber — good table extraction
    3. PyPDF2 / pypdf — basic text extraction

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the PDF file.

    Returns
    -------
    str
        Extracted text from all pages.
    """
    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
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

    logger.warning(
        "No PDF library could extract text. "
        "Install PyMuPDF (`pip install PyMuPDF`), pdfplumber, or pypdf."
    )
    return ""


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
        logger.warning(
            "pytesseract or Pillow not available for OCR. "
            "Install with: pip install pytesseract Pillow"
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


def parse_lab_report(text: str) -> tuple[dict[str, float], dict[str, float], str]:
    """Parse lab report text and extract structured lab values.

    Handles multiple common formats:
    - Colon-separated: "Hemoglobin: 14.5 g/dL"
    - Space-separated: "Hemoglobin     14.5    g/dL"
    - Table format:    "Hemoglobin | 14.5 | 13-17 | g/dL"
    - Mixed formats within the same report

    Parameters
    ----------
    text : str
        Raw text extracted from a lab report (PDF, OCR, or manual paste).

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

    # Pass 3: Space-separated format (more greedy, use last)
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
