"""Hybrid scoring for ResumeLens (Requirement 4).

The final match score blends three evidence-backed components (Requirement 4.1):

* **coverage** -- weighted requirement coverage (60%). Must-have requirements
  weigh 1.0 and nice-to-have 0.5; a ``met`` requirement counts 1.0, ``partial``
  counts 0.5 and ``missing`` / ``unverified`` count 0.0 (Requirement 4.2).
* **semantic** -- mean top-1 embedding similarity over the must-have
  requirements, rescaled ``(sim - 0.3) / 0.5`` clamped to ``0..1`` (design).
* **ats** -- ATS keyword coverage: fraction of JD keywords found in the resume.

:func:`score_analysis` ties the three together and returns a :class:`Scores`
with every component exposed alongside the final weighted score (Requirement
4.3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from analyzer.schemas import JDRequirements, MatchResult

# Component weights for the final score (Requirement 4.1).
COVERAGE_WEIGHT = 0.60
SEMANTIC_WEIGHT = 0.25
ATS_WEIGHT = 0.15

# Requirement-coverage weights (Requirement 4.2).
MUST_HAVE_WEIGHT = 1.0
NICE_TO_HAVE_WEIGHT = 0.5

# Per-status credit toward coverage (Requirement 4.2; 3.3 scores unverified as
# missing).
_STATUS_CREDIT = {
    "met": 1.0,
    "partial": 0.5,
    "missing": 0.0,
    "unverified": 0.0,
}

# Semantic-similarity rescale bounds (design: (sim - 0.3) / 0.5 clamped 0..1).
SEMANTIC_FLOOR = 0.3
SEMANTIC_SPAN = 0.5


@dataclass
class Scores:
    """The three component scores and the final weighted score.

    All values are in ``0..1``. :attr:`final` is the weighted combination; the
    individual components are kept so the UI can show them alongside it
    (Requirement 4.3).
    """

    coverage: float = 0.0
    semantic: float = 0.0
    ats: float = 0.0
    final: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Return the scores as a plain dict (for the report JSON)."""
        return asdict(self)


def _clamp01(value: float) -> float:
    """Clamp ``value`` to the inclusive range ``0..1``."""
    return max(0.0, min(1.0, value))


def weighted_coverage(
    jd: JDRequirements, match: MatchResult
) -> float:
    """Weighted requirement coverage in ``0..1`` (Requirements 4.2, 3.3).

    Each requirement contributes ``weight`` (1.0 must-have, 0.5 nice-to-have)
    to the denominator and ``weight * credit`` to the numerator, where credit
    is 1.0 for ``met``, 0.5 for ``partial`` and 0.0 for ``missing`` /
    ``unverified``. Requirements with no matching :class:`~analyzer.schemas.MatchItem`
    are treated as missing. Returns 0.0 when there are no requirements.
    """
    weights: dict[str, float] = {}
    for req in jd.must_have:
        weights[req] = MUST_HAVE_WEIGHT
    for req in jd.nice_to_have:
        # Don't let a nice-to-have downgrade a duplicate must-have.
        weights.setdefault(req, NICE_TO_HAVE_WEIGHT)

    if not weights:
        return 0.0

    status_by_id = {item.id: item.status for item in match.items}

    total_weight = 0.0
    earned = 0.0
    for req, weight in weights.items():
        total_weight += weight
        status = status_by_id.get(req, "missing")
        credit = _STATUS_CREDIT.get(status, 0.0)
        earned += weight * credit

    if total_weight == 0.0:
        return 0.0
    return _clamp01(earned / total_weight)


def semantic_rescale(top1_similarities: list[float]) -> float:
    """Mean top-1 similarity over must-haves, rescaled to ``0..1`` (design).

    Takes the best (top-1) similarity for each must-have requirement, averages
    them, then rescales with ``(mean - 0.3) / 0.5`` and clamps to ``0..1`` so
    that a raw similarity of 0.3 maps to 0.0 and 0.8 maps to 1.0. Returns 0.0
    when there are no similarities.
    """
    sims = [s for s in top1_similarities if s is not None]
    if not sims:
        return 0.0
    mean_sim = sum(sims) / len(sims)
    return _clamp01((mean_sim - SEMANTIC_FLOOR) / SEMANTIC_SPAN)


def ats_coverage(found: list[str], missing: list[str]) -> float:
    """Fraction of JD keywords found in the resume, in ``0..1``.

    ``found`` and ``missing`` come from :func:`analyzer.tools.ats_check`.
    Returns 0.0 when there are no keywords to check.
    """
    total = len(found) + len(missing)
    if total == 0:
        return 0.0
    return _clamp01(len(found) / total)


def final_score(coverage: float, semantic: float, ats: float) -> float:
    """Combine the three components into the final score (Requirement 4.1).

    ``final = 0.60*coverage + 0.25*semantic + 0.15*ats``. Each input is clamped
    to ``0..1`` first so an out-of-range component can't skew the result.
    """
    coverage = _clamp01(coverage)
    semantic = _clamp01(semantic)
    ats = _clamp01(ats)
    return _clamp01(
        COVERAGE_WEIGHT * coverage
        + SEMANTIC_WEIGHT * semantic
        + ATS_WEIGHT * ats
    )


def score_analysis(
    jd: JDRequirements,
    match: MatchResult,
    top1_similarities: list[float],
    ats_found: list[str],
    ats_missing: list[str],
) -> Scores:
    """Compute all component scores and the final weighted score (Requirement 4).

    Convenience wrapper the pipeline calls after matching and the ATS check: it
    computes :func:`weighted_coverage`, :func:`semantic_rescale` and
    :func:`ats_coverage`, combines them with :func:`final_score`, and returns a
    :class:`Scores` exposing all three components alongside the final score
    (Requirement 4.3).
    """
    coverage = weighted_coverage(jd, match)
    semantic = semantic_rescale(top1_similarities)
    ats = ats_coverage(ats_found, ats_missing)
    return Scores(
        coverage=round(coverage, 4),
        semantic=round(semantic, 4),
        ats=round(ats, 4),
        final=round(final_score(coverage, semantic, ats), 4),
    )
