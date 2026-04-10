"""Tests for image format detection and validation logic.

Covers:
- Magic-byte format detection (_detect_image_format)
- Extension/content mismatch handling (warning vs hard failure)
- WebP RIFF+WEBP signature tightening
- Dimension validation
- The full _validate_image_content pipeline
"""

from __future__ import annotations

import os
import struct
import sys
import zlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from teloscopy.webapp.app import (
        _detect_image_format,
        _validate_image_content,
    )

    _HAS_WEBAPP = True
except ImportError:
    _HAS_WEBAPP = False

pytestmark = pytest.mark.skipif(
    not _HAS_WEBAPP,
    reason="Webapp dependencies are not installed.",
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid/invalid byte sequences
# ---------------------------------------------------------------------------


def _make_valid_png(width: int = 64, height: int = 64) -> bytes:
    """Create a valid RGB PNG that OpenCV can decode."""

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b""
    for y in range(height):
        raw += b"\x00"  # filter: none
        for x in range(width):
            raw += bytes([x * 4 % 256, y * 4 % 256, 128])
    compressed = zlib.compress(raw)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


# Cached valid PNG for reuse across tests
_PNG_64x64 = _make_valid_png(64, 64)
_PNG_16x16 = _make_valid_png(16, 16)  # below 32px minimum

# Minimal JPEG bytes (just signature + enough for detection, not decodable)
_JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 100


# Valid WebP header: RIFF<size>WEBP  (not a valid image, but correct signature)
def _make_webp_header(payload: bytes = b"\x00" * 100) -> bytes:
    size = len(payload) + 4  # +4 for "WEBP" fourcc
    return b"RIFF" + struct.pack("<I", size) + b"WEBP" + payload


# WAV header: RIFF<size>WAVE — should NOT be detected as WebP
def _make_wav_header(payload: bytes = b"\x00" * 100) -> bytes:
    size = len(payload) + 4
    return b"RIFF" + struct.pack("<I", size) + b"WAVE" + payload


# AVI header: RIFF<size>AVI  — should NOT be detected as WebP
def _make_avi_header(payload: bytes = b"\x00" * 100) -> bytes:
    size = len(payload) + 4
    return b"RIFF" + struct.pack("<I", size) + b"AVI " + payload


_BMP_HEADER = b"BM" + b"\x00" * 100
_TIFF_LE_HEADER = b"II\x2a\x00" + b"\x00" * 100
_TIFF_BE_HEADER = b"MM\x00\x2a" + b"\x00" * 100


# ---------------------------------------------------------------------------
# _detect_image_format
# ---------------------------------------------------------------------------


class TestDetectImageFormat:
    """Unit tests for the magic-byte format detection function."""

    def test_detect_png(self):
        assert _detect_image_format(_PNG_64x64) == "png"

    def test_detect_jpeg(self):
        assert _detect_image_format(_JPEG_HEADER) == "jpeg"

    def test_detect_bmp(self):
        assert _detect_image_format(_BMP_HEADER) == "bmp"

    def test_detect_tiff_little_endian(self):
        assert _detect_image_format(_TIFF_LE_HEADER) == "tiff"

    def test_detect_tiff_big_endian(self):
        assert _detect_image_format(_TIFF_BE_HEADER) == "tiff"

    def test_detect_webp(self):
        """WebP must be RIFF????WEBP, not just RIFF prefix."""
        assert _detect_image_format(_make_webp_header()) == "webp"

    def test_wav_not_detected_as_webp(self):
        """RIFF????WAVE (WAV audio) must NOT match as webp."""
        assert _detect_image_format(_make_wav_header()) == "unknown"

    def test_avi_not_detected_as_webp(self):
        """RIFF????AVI (AVI video) must NOT match as webp."""
        assert _detect_image_format(_make_avi_header()) == "unknown"

    def test_unknown_bytes(self):
        assert _detect_image_format(b"\x00\x00\x00\x00" * 10) == "unknown"

    def test_empty_bytes(self):
        assert _detect_image_format(b"") == "unknown"

    def test_short_bytes(self):
        """Less than 4 bytes should not crash."""
        assert _detect_image_format(b"\xff\xd8") == "unknown"


# ---------------------------------------------------------------------------
# _validate_image_content — extension mismatch behaviour
# ---------------------------------------------------------------------------


class TestExtensionMismatch:
    """Extension/content format mismatch should be a warning (not a hard
    failure) when OpenCV can still decode the image."""

    def test_matched_extension_no_warnings(self):
        """Valid PNG content with .png extension → valid=True, no warnings."""
        result = _validate_image_content(_PNG_64x64, "photo.png")
        assert result.format_detected == "png"
        assert result.valid is True
        assert len(result.warnings) == 0
        assert len(result.issues) == 0

    def test_png_content_in_jpg_extension_is_warning(self):
        """Valid PNG content saved as .jpg — decodable → valid=True + warning."""
        result = _validate_image_content(_PNG_64x64, "photo.jpg")
        assert result.format_detected == "png"
        assert result.valid is True  # decodable → no hard failure
        assert any("suggests jpeg but content is png" in w for w in result.warnings)
        assert len(result.issues) == 0

    def test_webp_content_in_jpg_extension(self):
        """WebP content saved as .jpg — if not decodable, mismatch is an issue."""
        webp_bytes = _make_webp_header()
        result = _validate_image_content(webp_bytes, "photo.jpg")
        assert result.format_detected == "webp"
        # Minimal WebP is not decodable → mismatch should be in issues
        assert result.valid is False
        assert any("suggests jpeg but content is webp" in i for i in result.issues)

    def test_jpeg_content_in_png_extension(self):
        """JPEG header in .png file — mismatch handling depends on decodability."""
        result = _validate_image_content(_JPEG_HEADER, "image.png")
        assert result.format_detected == "jpeg"
        # Minimal JPEG is not decodable
        assert result.valid is False

    def test_unknown_magic_undecodable_is_hard_failure(self):
        """Unknown magic bytes + not decodable → valid=False with issue."""
        result = _validate_image_content(b"\x00" * 100, "image.png")
        assert result.valid is False
        assert len(result.issues) > 0

    def test_warnings_field_exists(self):
        """ImageValidationResponse always has a 'warnings' field."""
        result = _validate_image_content(_PNG_64x64, "test.png")
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list)


# ---------------------------------------------------------------------------
# _validate_image_content — dimension checks
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    """Dimension-related validation checks."""

    def test_undersized_png_flagged(self):
        """A 16x16 PNG is below the 32px minimum → flagged as too small."""
        result = _validate_image_content(_PNG_16x16, "tiny.png")
        assert result.format_detected == "png"
        assert any("too small" in i for i in result.issues)

    def test_valid_png_dimensions(self):
        """A 64x64 PNG passes dimension checks."""
        result = _validate_image_content(_PNG_64x64, "photo.png")
        assert result.valid is True
        assert result.width == 64
        assert result.height == 64

    def test_file_size_recorded(self):
        """file_size_bytes should match actual content length."""
        result = _validate_image_content(_PNG_64x64, "photo.png")
        assert result.file_size_bytes == len(_PNG_64x64)

    def test_format_detected_field(self):
        """format_detected should reflect the magic-byte detection."""
        result = _validate_image_content(_PNG_64x64, "photo.png")
        assert result.format_detected == "png"
