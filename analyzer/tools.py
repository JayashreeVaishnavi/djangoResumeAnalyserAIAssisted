"""Deterministic tools for the ResumeLens pipeline.

These are the pure helper functions the orchestrator calls directly (small
local models are unreliable at native function calling), with every call
logged as a ``ToolCall`` by the pipeline. The functions here are side-effect
free so they are easy to test and reason about:

* ``chunk_text`` -- split resume text into overlapping chunks for retrieval.
* ``semantic_search`` -- top-N resume chunks per requirement via embedding
  similarity (Requirement 3.2; design: top 3 per requirement). Uses
  :func:`analyzer.llm.embed`.
* ``normalize_skill`` -- skill alias normaliser, e.g. ``k8s`` -> ``kubernetes``
  (Requirement 5.1).
* ``years_of_experience`` -- total years from parsed roles with dates.
* ``ats_check`` -- JD keywords found/missing (via ``normalize_skill``) plus
  missing sections and contact details (Requirements 5.1, 5.2).
* ``verify_evidence`` -- normalise whitespace/case and accept exact
  containment or >=80% longest-match overlap (Requirements 3.2, 3.3).
"""

from __future__ import annotations

import math
import re
from datetime import date

from analyzer import llm
from analyzer.schemas import ParsedResume

# Default number of resume chunks retrieved per requirement (design: top 3).
DEFAULT_TOP_N = 3

# Fraction of the longest overlapping run required to accept evidence that is
# not an exact substring (design: >=80% longest-match overlap).
EVIDENCE_OVERLAP_THRESHOLD = 0.8

# Skill aliases: map a written-as variant to its canonical skill name. Keys and
# values are compared case-insensitively (see :func:`normalize_skill`).
SKILL_ALIASES: dict[str, str] = {
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "postgres": "postgresql",
    "psql": "postgresql",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "gcp": "google cloud",
    "aws": "amazon web services",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "tf": "tensorflow",
    "k8": "kubernetes",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "dl": "deep learning",
}

# Month names (and common abbreviations) -> month number, for date parsing.
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# A resume is expected to contain these sections (Requirement 5.2).
_REQUIRED_SECTIONS = ("experience", "education", "skills")


# --- text chunking ---------------------------------------------------------


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Split ``text`` into overlapping chunks for embedding/retrieval.

    Chunks are built from whitespace-delimited tokens so words are never cut
    in half. ``size`` is the target chunk length in characters and ``overlap``
    is the number of characters of context shared between adjacent chunks.
    """
    text = (text or "").strip()
    if not text:
        return []
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size - 1))

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1 accounts for the joining space.
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > size:
            chunk = " ".join(current)
            chunks.append(chunk)
            # Start the next chunk with a tail of the previous one for overlap.
            if overlap:
                tail: list[str] = []
                tail_len = 0
                for w in reversed(current):
                    add = len(w) + (1 if tail else 0)
                    if tail_len + add > overlap:
                        break
                    tail.insert(0, w)
                    tail_len += add
                current = tail
                current_len = tail_len
            else:
                current = []
                current_len = 0
            current.append(word)
            current_len += len(word) + (1 if current_len else 0)
        else:
            current.append(word)
            current_len += extra

    if current:
        chunks.append(" ".join(current))
    return chunks


# --- embedding similarity --------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors; 0.0 if either has zero magnitude."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search(
    requirements: list[str],
    chunks: list[str],
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, list[str]]:
    """Return the top-``top_n`` resume ``chunks`` per requirement.

    Embeds the requirements and chunks with :func:`analyzer.llm.embed` and
    ranks chunks by cosine similarity (Requirement 3.2; design: top 3 per
    requirement). The result maps each requirement string to its most similar
    chunks, ordered best first.

    Tests inject a fake ``embed`` (hashed bag-of-words vectors) by patching
    ``analyzer.tools.llm.embed`` or ``analyzer.llm.embed``.
    """
    result: dict[str, list[str]] = {req: [] for req in requirements}
    if not requirements or not chunks:
        return result

    chunk_vectors = llm.embed(chunks)
    req_vectors = llm.embed(requirements)

    for req, req_vec in zip(requirements, req_vectors):
        scored = [
            (_cosine(req_vec, chunk_vec), idx)
            for idx, chunk_vec in enumerate(chunk_vectors)
        ]
        # Sort by similarity desc, then by original order for stable ties.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        result[req] = [chunks[idx] for _, idx in scored[: max(0, top_n)]]
    return result


# --- skill alias normalisation (Requirement 5.1) ---------------------------


def normalize_skill(skill: str) -> str:
    """Normalise a skill/keyword to a canonical form.

    Trims and lower-cases the input, collapses internal whitespace, then
    applies the :data:`SKILL_ALIASES` map (e.g. ``k8s`` -> ``kubernetes``).
    Unknown skills are returned in their cleaned, lower-cased form.
    """
    cleaned = re.sub(r"\s+", " ", (skill or "").strip().lower())
    return SKILL_ALIASES.get(cleaned, cleaned)


# --- years of experience ---------------------------------------------------


def _parse_date(value: str, *, is_end: bool) -> date | None:
    """Parse a loosely-formatted resume date into a :class:`date`.

    Handles "Present"/"Current"/"Now" (-> today for an end date), bare years
    ("2020"), "Mon YYYY" ("Jan 2020") and numeric "MM/YYYY" / "YYYY-MM".
    Returns ``None`` when nothing usable can be found.
    """
    text = (value or "").strip().lower()
    if not text:
        return None
    if is_end and text in {"present", "current", "now", "ongoing"}:
        return date.today()

    # Month name + year, e.g. "Jan 2020" / "September 2019".
    m = re.search(r"([a-z]{3,9})\.?\s+(\d{4})", text)
    if m:
        name = m.group(1)
        month = _MONTHS.get(name[:4]) or _MONTHS.get(name[:3])
        if month:
            return date(int(m.group(2)), month, 1)

    # Numeric MM/YYYY or MM-YYYY.
    m = re.search(r"(\d{1,2})[/\-](\d{4})", text)
    if m:
        month = min(max(int(m.group(1)), 1), 12)
        return date(int(m.group(2)), month, 1)

    # YYYY-MM.
    m = re.search(r"(\d{4})[/\-](\d{1,2})", text)
    if m:
        month = min(max(int(m.group(2)), 1), 12)
        return date(int(m.group(1)), month, 1)

    # Bare year.
    m = re.search(r"(19|20)\d{2}", text)
    if m:
        return date(int(m.group(0)), 1, 1)

    return None


def years_of_experience(resume: ParsedResume) -> float:
    """Total years of experience from the resume's roles with dates.

    Sums each role's span (start to end, with "Present" meaning today),
    rounded to one decimal place. Roles without a parseable start date are
    skipped. Overlapping roles are summed independently -- this is a simple,
    deterministic estimate used by the ATS/min-years checks.
    """
    total_days = 0
    for role in resume.roles:
        start = _parse_date(role.start_date, is_end=False)
        if start is None:
            continue
        end = _parse_date(role.end_date, is_end=True) or date.today()
        if end < start:
            continue
        total_days += (end - start).days
    return round(total_days / 365.25, 1)


# --- ATS checks (Requirements 5.1, 5.2) ------------------------------------


def _resume_haystack(resume: ParsedResume) -> str:
    """Flatten the structured resume into one lower-case search string."""
    parts: list[str] = [resume.name, resume.email, resume.phone]
    parts.extend(resume.skills)
    parts.extend(resume.education)
    parts.extend(resume.projects)
    for role in resume.roles:
        parts.extend([role.title, role.company])
        parts.extend(role.bullets)
    return " ".join(p for p in parts if p).lower()


def ats_check(resume: ParsedResume, keywords: list[str]) -> dict[str, object]:
    """Report ATS keyword coverage and structural gaps.

    Returns a dict with:

    * ``found`` / ``missing`` -- JD ``keywords`` present/absent in the resume,
      compared after :func:`normalize_skill` so aliases match (Requirement 5.1).
    * ``missing_sections`` -- any of experience/education/skills that are empty
      (Requirement 5.2).
    * ``missing_contact`` -- missing ``email`` and/or ``phone`` (Requirement 5.2).
    """
    haystack = _resume_haystack(resume)
    # Normalise each haystack token so alias keywords can match.
    haystack_norms = {normalize_skill(tok) for tok in re.split(r"[,\s/]+", haystack) if tok}

    found: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        norm = normalize_skill(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        # Match either as a whole normalised token or as a substring of the
        # (normalised) haystack, so multi-word keywords also match.
        if norm in haystack_norms or norm in normalize_skill(haystack):
            found.append(raw)
        else:
            missing.append(raw)

    missing_sections: list[str] = []
    if not resume.roles:
        missing_sections.append("experience")
    if not resume.education:
        missing_sections.append("education")
    if not resume.skills:
        missing_sections.append("skills")

    missing_contact: list[str] = []
    if not (resume.email or "").strip():
        missing_contact.append("email")
    if not (resume.phone or "").strip():
        missing_contact.append("phone")

    return {
        "found": found,
        "missing": missing,
        "missing_sections": missing_sections,
        "missing_contact": missing_contact,
    }


# --- evidence verification (Requirements 3.2, 3.3) -------------------------


def _normalize_ws(text: str) -> str:
    """Lower-case and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _longest_common_substring_len(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared by ``a`` and ``b``."""
    if not a or not b:
        return 0
    # Rolling two-row dynamic programming to keep memory linear.
    previous = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def verify_evidence(evidence: str, resume_text: str) -> bool:
    """True if ``evidence`` is backed by ``resume_text`` (Requirements 3.2/3.3).

    Both strings are normalised (whitespace collapsed, lower-cased). The
    evidence is accepted when it is contained verbatim in the resume, or when
    its longest contiguous overlap with the resume covers at least
    :data:`EVIDENCE_OVERLAP_THRESHOLD` (80%) of the evidence length. Empty
    evidence can never be verified (so it is scored as missing).
    """
    needle = _normalize_ws(evidence)
    haystack = _normalize_ws(resume_text)
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    overlap = _longest_common_substring_len(needle, haystack)
    return overlap >= EVIDENCE_OVERLAP_THRESHOLD * len(needle)
