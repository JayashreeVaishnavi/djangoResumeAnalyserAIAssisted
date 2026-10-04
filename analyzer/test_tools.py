"""Unit tests for the deterministic pipeline tools.

``semantic_search`` is exercised with a fake ``embed`` (hashed bag-of-words
vectors) so no running Ollama server is needed, matching the design's testing
approach.
"""

from __future__ import annotations

import hashlib
from datetime import date
from unittest import mock

import pytest

from analyzer import tools
from analyzer.schemas import ParsedResume, Role


# --- fake embedding (hashed bag-of-words) ----------------------------------

_DIMS = 64


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic hashed bag-of-words vectors for testing.

    Each token increments a bucket chosen by its hash, so texts that share
    words land close together under cosine similarity.
    """
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _DIMS
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % _DIMS] += 1.0
        vectors.append(vec)
    return vectors


# --- normalize_skill (Requirement 5.1) -------------------------------------


def test_normalize_skill_known_alias():
    assert tools.normalize_skill("k8s") == "kubernetes"
    assert tools.normalize_skill("K8S") == "kubernetes"


def test_normalize_skill_trims_and_collapses_whitespace():
    assert tools.normalize_skill("  Kube  ") == "kubernetes"
    assert tools.normalize_skill("ci cd") == "cicd"


def test_normalize_skill_unknown_returns_cleaned_lowercase():
    assert tools.normalize_skill("  Rust ") == "rust"


# --- years_of_experience ---------------------------------------------------


def test_years_of_experience_single_role():
    resume = ParsedResume(
        roles=[Role(start_date="Jan 2020", end_date="Jan 2023")]
    )
    assert tools.years_of_experience(resume) == pytest.approx(3.0, abs=0.1)


def test_years_of_experience_present_uses_today():
    this_year = date.today().year
    resume = ParsedResume(
        roles=[Role(start_date=str(this_year - 2), end_date="Present")]
    )
    assert tools.years_of_experience(resume) >= 1.9


def test_years_of_experience_sums_multiple_roles():
    resume = ParsedResume(
        roles=[
            Role(start_date="2018", end_date="2020"),
            Role(start_date="2020", end_date="2022"),
        ]
    )
    assert tools.years_of_experience(resume) == pytest.approx(4.0, abs=0.1)


def test_years_of_experience_skips_unparseable_and_inverted():
    resume = ParsedResume(
        roles=[
            Role(start_date="", end_date="2020"),
            Role(start_date="2022", end_date="2020"),  # inverted -> skipped
        ]
    )
    assert tools.years_of_experience(resume) == 0.0


# --- verify_evidence (Requirements 3.2, 3.3) -------------------------------


def test_verify_evidence_exact_containment_ignores_case_and_whitespace():
    resume = "Led a   team of  five Engineers to ship the product."
    assert tools.verify_evidence("led a team of FIVE engineers", resume) is True


def test_verify_evidence_accepts_high_overlap():
    resume = "Built and maintained scalable microservices in Python."
    # Nearly identical; small trailing difference keeps overlap >= 80%.
    assert tools.verify_evidence(
        "built and maintained scalable microservices in Pythonx", resume
    ) is True


def test_verify_evidence_rejects_low_overlap():
    resume = "Built scalable microservices in Python."
    assert tools.verify_evidence("Managed a budget of two million dollars", resume) is False


def test_verify_evidence_empty_is_unverified():
    assert tools.verify_evidence("", "anything") is False
    assert tools.verify_evidence("something", "") is False


# --- ats_check (Requirements 5.1, 5.2) -------------------------------------


def _full_resume() -> ParsedResume:
    return ParsedResume(
        name="Jane Doe",
        email="jane@example.com",
        phone="555-1234",
        skills=["Python", "Kubernetes", "React"],
        roles=[Role(title="Engineer", company="Acme", bullets=["Shipped things"])],
        education=["BSc Computer Science"],
    )


def test_ats_check_found_and_missing_with_alias():
    resume = _full_resume()
    # "k8s" should match "Kubernetes" via normalize_skill.
    result = tools.ats_check(resume, ["Python", "k8s", "Go"])
    assert "Python" in result["found"]
    assert "k8s" in result["found"]
    assert result["missing"] == ["Go"]


def test_ats_check_flags_missing_sections_and_contact():
    resume = ParsedResume(skills=["Python"])  # no roles/education, no contact
    result = tools.ats_check(resume, ["Python"])
    assert "experience" in result["missing_sections"]
    assert "education" in result["missing_sections"]
    assert "skills" not in result["missing_sections"]
    assert set(result["missing_contact"]) == {"email", "phone"}


def test_ats_check_complete_resume_has_no_structural_gaps():
    result = tools.ats_check(_full_resume(), ["Python"])
    assert result["missing_sections"] == []
    assert result["missing_contact"] == []


# --- chunk_text ------------------------------------------------------------


def test_chunk_text_empty_returns_empty():
    assert tools.chunk_text("") == []
    assert tools.chunk_text("   ") == []


def test_chunk_text_short_text_single_chunk():
    assert tools.chunk_text("hello world", size=500) == ["hello world"]


def test_chunk_text_splits_long_text_with_overlap():
    words = " ".join(f"word{i}" for i in range(200))
    chunks = tools.chunk_text(words, size=100, overlap=20)
    assert len(chunks) > 1
    # Every chunk should respect the size budget (allowing a single word).
    assert all(len(c) <= 100 or " " not in c for c in chunks)
    # Overlap: the end of one chunk reappears at the start of the next.
    first_tail = chunks[0].split()[-1]
    assert first_tail in chunks[1].split()


def test_chunk_text_does_not_split_words():
    chunks = tools.chunk_text("alpha beta gamma delta", size=12, overlap=0)
    for chunk in chunks:
        for word in chunk.split():
            assert word in {"alpha", "beta", "gamma", "delta"}


# --- semantic_search (fake embed) ------------------------------------------


def test_semantic_search_ranks_relevant_chunk_first():
    chunks = [
        "experienced python developer building django apps",
        "led a marketing team and managed budgets",
        "kubernetes docker and cloud infrastructure work",
    ]
    requirements = ["python django web development"]
    with mock.patch("analyzer.tools.llm.embed", side_effect=_fake_embed):
        result = tools.semantic_search(requirements, chunks, top_n=2)

    ranked = result["python django web development"]
    assert ranked[0] == "experienced python developer building django apps"
    assert len(ranked) == 2


def test_semantic_search_top_n_limits_results():
    chunks = [f"chunk about python topic {i}" for i in range(5)]
    requirements = ["python"]
    with mock.patch("analyzer.tools.llm.embed", side_effect=_fake_embed):
        result = tools.semantic_search(requirements, chunks, top_n=3)
    assert len(result["python"]) == 3


def test_semantic_search_empty_inputs():
    with mock.patch("analyzer.tools.llm.embed", side_effect=_fake_embed) as embed:
        assert tools.semantic_search([], ["a"]) == {}
        assert tools.semantic_search(["req"], []) == {"req": []}
    embed.assert_not_called()
