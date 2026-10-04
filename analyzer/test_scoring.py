"""Unit tests for the hybrid scoring module (Requirement 4).

Scoring is pure and deterministic, so these tests exercise it directly with
constructed :class:`JDRequirements` / :class:`MatchResult` instances -- no LLM
or embedding server needed.
"""

from __future__ import annotations

import pytest

from analyzer import scoring
from analyzer.schemas import JDRequirements, MatchItem, MatchResult


def _match(**status_by_id: str) -> MatchResult:
    """Build a MatchResult from ``requirement_text=status`` keyword pairs."""
    return MatchResult(
        items=[MatchItem(id=req, status=status) for req, status in status_by_id.items()]
    )


# --- weighted_coverage (Requirements 4.2, 3.3) -----------------------------


def test_coverage_all_must_have_met_is_full():
    jd = JDRequirements(must_have=["python", "django"])
    match = _match(python="met", django="met")
    assert scoring.weighted_coverage(jd, match) == pytest.approx(1.0)


def test_coverage_partial_counts_half():
    jd = JDRequirements(must_have=["python"])
    match = _match(python="partial")
    assert scoring.weighted_coverage(jd, match) == pytest.approx(0.5)


def test_coverage_missing_and_unverified_count_zero():
    jd = JDRequirements(must_have=["python", "django"])
    match = _match(python="missing", django="unverified")
    assert scoring.weighted_coverage(jd, match) == pytest.approx(0.0)


def test_coverage_nice_to_have_weighted_half():
    # One must-have (weight 1.0) met, one nice-to-have (weight 0.5) missing.
    # earned = 1.0, total weight = 1.5 -> 0.667.
    jd = JDRequirements(must_have=["python"], nice_to_have=["kubernetes"])
    match = _match(python="met", kubernetes="missing")
    assert scoring.weighted_coverage(jd, match) == pytest.approx(1.0 / 1.5)


def test_coverage_nice_to_have_met_contributes_its_weight():
    # must-have met (1.0*1.0) + nice-to-have met (0.5*1.0) = 1.5 over 1.5 -> 1.0
    jd = JDRequirements(must_have=["python"], nice_to_have=["kubernetes"])
    match = _match(python="met", kubernetes="met")
    assert scoring.weighted_coverage(jd, match) == pytest.approx(1.0)


def test_coverage_requirement_without_match_treated_missing():
    jd = JDRequirements(must_have=["python", "django"])
    match = _match(python="met")  # django has no MatchItem
    # earned 1.0 over total 2.0 -> 0.5
    assert scoring.weighted_coverage(jd, match) == pytest.approx(0.5)


def test_coverage_no_requirements_is_zero():
    assert scoring.weighted_coverage(JDRequirements(), MatchResult()) == 0.0


# --- semantic_rescale (design: (sim - 0.3)/0.5 clamped 0..1) ---------------


def test_semantic_rescale_maps_floor_to_zero():
    assert scoring.semantic_rescale([0.3]) == pytest.approx(0.0)


def test_semantic_rescale_maps_ceiling_to_one():
    assert scoring.semantic_rescale([0.8]) == pytest.approx(1.0)


def test_semantic_rescale_midpoint():
    # (0.55 - 0.3) / 0.5 = 0.5
    assert scoring.semantic_rescale([0.55]) == pytest.approx(0.5)


def test_semantic_rescale_clamps_below_floor():
    assert scoring.semantic_rescale([0.1]) == 0.0


def test_semantic_rescale_clamps_above_ceiling():
    assert scoring.semantic_rescale([0.95]) == 1.0


def test_semantic_rescale_averages_similarities():
    # mean of 0.3 and 0.8 = 0.55 -> (0.55 - 0.3)/0.5 = 0.5
    assert scoring.semantic_rescale([0.3, 0.8]) == pytest.approx(0.5)


def test_semantic_rescale_empty_is_zero():
    assert scoring.semantic_rescale([]) == 0.0


# --- ats_coverage ----------------------------------------------------------


def test_ats_coverage_fraction_found():
    assert scoring.ats_coverage(["python", "go"], ["rust"]) == pytest.approx(2 / 3)


def test_ats_coverage_all_found_is_one():
    assert scoring.ats_coverage(["python"], []) == pytest.approx(1.0)


def test_ats_coverage_none_found_is_zero():
    assert scoring.ats_coverage([], ["python"]) == 0.0


def test_ats_coverage_no_keywords_is_zero():
    assert scoring.ats_coverage([], []) == 0.0


# --- final_score (Requirement 4.1) -----------------------------------------


def test_final_score_weighted_combination():
    # 0.60*1.0 + 0.25*0.5 + 0.15*0.2 = 0.60 + 0.125 + 0.03 = 0.755
    assert scoring.final_score(1.0, 0.5, 0.2) == pytest.approx(0.755)


def test_final_score_all_one_is_one():
    assert scoring.final_score(1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_final_score_all_zero_is_zero():
    assert scoring.final_score(0.0, 0.0, 0.0) == 0.0


def test_final_score_clamps_out_of_range_inputs():
    # Inputs clamped to 0..1 first, so this is just the full-weight sum.
    assert scoring.final_score(5.0, -1.0, 2.0) == pytest.approx(0.75)


# --- score_analysis (Requirement 4.3: all components exposed) --------------


def test_score_analysis_exposes_all_components():
    jd = JDRequirements(must_have=["python"], nice_to_have=["go"])
    match = _match(python="met", go="partial")
    scores = scoring.score_analysis(
        jd=jd,
        match=match,
        top1_similarities=[0.8],
        ats_found=["python"],
        ats_missing=["rust"],
    )
    # coverage: (1.0*1.0 + 0.5*0.5) / 1.5 = 1.25/1.5 = 0.8333
    assert scores.coverage == pytest.approx(0.8333, abs=1e-3)
    assert scores.semantic == pytest.approx(1.0)
    assert scores.ats == pytest.approx(0.5)
    expected_final = scoring.final_score(1.25 / 1.5, 1.0, 0.5)
    assert scores.final == pytest.approx(round(expected_final, 4), abs=1e-3)


def test_score_analysis_as_dict_has_four_keys():
    scores = scoring.score_analysis(
        jd=JDRequirements(must_have=["python"]),
        match=_match(python="met"),
        top1_similarities=[0.8],
        ats_found=["python"],
        ats_missing=[],
    )
    assert set(scores.as_dict().keys()) == {"coverage", "semantic", "ats", "final"}
