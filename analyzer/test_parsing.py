"""Tests for schemas and resume text extraction.

The DOCX test builds a real .docx in memory with python-docx; the PDF test
stubs pdfplumber so no fixture file is needed.
"""

from __future__ import annotations

import io
from unittest import mock

import docx
import pytest

from analyzer import parsing
from analyzer.schemas import (
    JDRequirements,
    MatchItem,
    ParsedResume,
    ReviewResult,
    Rewrite,
    Role,
    Verdict,
)


# --- schemas ---------------------------------------------------------------


def test_parsed_resume_defaults_are_empty():
    resume = ParsedResume()
    assert resume.skills == []
    assert resume.roles == []
    assert resume.education == []
    assert resume.projects == []


def test_role_round_trips_dates_and_bullets():
    role = Role(
        title="Engineer",
        company="Acme",
        start_date="2020",
        end_date="Present",
        bullets=["Built things", "Fixed things"],
    )
    assert role.bullets == ["Built things", "Fixed things"]
    assert role.end_date == "Present"


def test_jd_requirements_fields():
    jd = JDRequirements(
        must_have=["Python"],
        nice_to_have=["Go"],
        keywords=["kubernetes"],
        min_years=5,
    )
    assert jd.must_have == ["Python"]
    assert jd.min_years == 5


def test_jd_requirements_rejects_negative_years():
    with pytest.raises(Exception):
        JDRequirements(min_years=-1)


def test_match_item_requires_id_and_status():
    item = MatchItem(id="1", status="met", evidence="led a team")
    assert item.status == "met"


def test_review_result_round_trip():
    result = ReviewResult(
        verdicts=[Verdict(index=0, supported=False, reason="adds a fact")]
    )
    assert result.verdicts[0].supported is False
    assert Rewrite(original="a", suggested="b").target == ""


# --- parsing: unsupported type ---------------------------------------------


def test_unsupported_extension_raises():
    with pytest.raises(parsing.UnsupportedFileType):
        parsing.extract_text(b"data", "resume.rtf")


def test_missing_extension_raises():
    with pytest.raises(parsing.UnsupportedFileType):
        parsing.extract_text(b"data", "resume")


# --- parsing: TXT ----------------------------------------------------------


def test_txt_extraction():
    text = parsing.extract_text(b"Hello world", "resume.txt")
    assert text == "Hello world"


def test_txt_accepts_file_like_object_and_rewinds():
    buf = io.BytesIO(b"Some resume text")
    buf.read()  # advance the cursor; extract_text should seek(0)
    assert parsing.extract_text(buf, "resume.txt") == "Some resume text"


def test_empty_txt_raises_no_text():
    with pytest.raises(parsing.NoTextExtracted):
        parsing.extract_text(b"   \n\t ", "resume.txt")


# --- parsing: DOCX ---------------------------------------------------------


def _docx_bytes(paragraphs: list[str]) -> bytes:
    document = docx.Document()
    for para in paragraphs:
        document.add_paragraph(para)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_docx_extraction():
    data = _docx_bytes(["Jane Doe", "Senior Engineer", "Python, Django"])
    text = parsing.extract_text(data, "resume.docx")
    assert "Jane Doe" in text
    assert "Django" in text


def test_corrupt_docx_raises_no_text():
    with pytest.raises(parsing.NoTextExtracted):
        parsing.extract_text(b"not a real docx", "resume.docx")


# --- parsing: PDF (pdfplumber stubbed) -------------------------------------


def test_pdf_extraction_uses_pdfplumber():
    page = mock.Mock()
    page.extract_text.return_value = "Resume PDF content"
    fake_pdf = mock.MagicMock()
    fake_pdf.pages = [page]
    fake_pdf.__enter__.return_value = fake_pdf

    with mock.patch("analyzer.parsing.pdfplumber.open", return_value=fake_pdf):
        text = parsing.extract_text(b"%PDF-1.4 fake", "resume.pdf")
    assert text == "Resume PDF content"


def test_pdf_with_no_text_raises_no_text():
    page = mock.Mock()
    page.extract_text.return_value = ""
    fake_pdf = mock.MagicMock()
    fake_pdf.pages = [page]
    fake_pdf.__enter__.return_value = fake_pdf

    with mock.patch("analyzer.parsing.pdfplumber.open", return_value=fake_pdf):
        with pytest.raises(parsing.NoTextExtracted):
            parsing.extract_text(b"%PDF-1.4 fake", "resume.pdf")
