"""
Test suite for utils/parser.py

Tests cover file extension validation and text extraction from
PDF, DOCX, and TXT file objects using mock file-like objects.
"""
import pytest
import io
import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parser import allowed_file, extract_text, ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FakeFile:
    """Minimal file-like object that mimics a Werkzeug FileStorage."""

    def __init__(self, filename, content=b""):
        self.filename = filename
        self._stream = io.BytesIO(content)

    def read(self):
        return self._stream.read()

    def seek(self, pos, whence=0):
        return self._stream.seek(pos, whence)


# ---------------------------------------------------------------------------
# allowed_file
# ---------------------------------------------------------------------------
class TestAllowedFile:
    def test_accepts_pdf(self):
        assert allowed_file("resume.pdf") is True

    def test_accepts_docx(self):
        assert allowed_file("resume.docx") is True

    def test_accepts_txt(self):
        assert allowed_file("resume.txt") is True

    def test_rejects_exe(self):
        assert allowed_file("malware.exe") is False

    def test_rejects_jpg(self):
        assert allowed_file("photo.jpg") is False

    def test_rejects_no_extension(self):
        assert allowed_file("resume") is False

    def test_case_insensitive(self):
        assert allowed_file("resume.PDF") is True
        assert allowed_file("resume.Docx") is True
        assert allowed_file("resume.TXT") is True

    def test_rejects_empty_filename(self):
        assert allowed_file("") is False

    def test_accepts_dotted_filename(self):
        # e.g. "my.resume.v2.pdf" should still be accepted
        assert allowed_file("my.resume.v2.pdf") is True


# ---------------------------------------------------------------------------
# ALLOWED_EXTENSIONS constant
# ---------------------------------------------------------------------------
class TestAllowedExtensions:
    def test_contains_expected_types(self):
        assert "pdf" in ALLOWED_EXTENSIONS
        assert "docx" in ALLOWED_EXTENSIONS
        assert "txt" in ALLOWED_EXTENSIONS

    def test_no_unexpected_types(self):
        # Ensure no dangerous extensions are accidentally allowed
        for ext in ("exe", "bat", "sh", "py", "js"):
            assert ext not in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------
class TestExtractText:
    def test_unsupported_extension_raises(self):
        fake = FakeFile("resume.xyz", b"some content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(fake)

    def test_txt_extraction(self):
        content = b"John Doe\nSoftware Engineer\nPython, Flask, Docker"
        fake = FakeFile("resume.txt", content)
        result = extract_text(fake)
        assert "John Doe" in result
        assert "Software Engineer" in result
        assert "Python" in result

    def test_txt_empty_raises(self):
        fake = FakeFile("empty.txt", b"")
        with pytest.raises(ValueError, match="empty"):
            extract_text(fake)

    def test_txt_whitespace_only_raises(self):
        fake = FakeFile("spaces.txt", b"   \n\t  ")
        with pytest.raises(ValueError, match="empty"):
            extract_text(fake)

    def test_txt_unicode_content(self):
        content = "Résumé — Professional Expérience".encode("utf-8")
        fake = FakeFile("resume.txt", content)
        result = extract_text(fake)
        assert "sum" in result.lower()  # At least partial content should be readable
