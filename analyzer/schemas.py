"""Pydantic schemas for the ResumeLens agent pipeline.

These schemas are the contract between the agents and the rest of the app.
Every agent that talks to the LLM returns one of these models (via
``llm.chat_json``), which both constrains the model output (Ollama ``format``)
and validates it (Requirement 2.3).

* ``ParsedResume`` / ``Role`` -- structured resume (Requirement 2.1).
* ``JDRequirements`` -- structured job description (Requirement 2.2).
* ``MatchItem`` / ``MatchResult`` -- evidence-backed matching (Requirement 3).
* ``Rewrite`` / ``RewriteResult`` -- rewrite suggestions (Requirement 6.1).
* ``Verdict`` / ``ReviewResult`` -- reviewer verdicts (Requirement 6.2).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Resume (Parser agent output, Requirement 2.1) -------------------------


class Role(BaseModel):
    """A single work experience entry on the resume."""

    title: str = Field(default="", description="Job title / position held.")
    company: str = Field(default="", description="Employer or organisation.")
    start_date: str = Field(
        default="", description="Start date as written on the resume."
    )
    end_date: str = Field(
        default="",
        description="End date as written, or 'Present' for current roles.",
    )
    bullets: list[str] = Field(
        default_factory=list,
        description="Achievement / responsibility bullet points for the role.",
    )


class ParsedResume(BaseModel):
    """Structured representation of a resume (skills, roles, education, projects)."""

    name: str = Field(default="", description="Candidate name, if present.")
    email: str = Field(default="", description="Contact email, if present.")
    phone: str = Field(default="", description="Contact phone, if present.")
    skills: list[str] = Field(
        default_factory=list, description="Skills / technologies listed."
    )
    roles: list[Role] = Field(
        default_factory=list, description="Work experience entries."
    )
    education: list[str] = Field(
        default_factory=list, description="Education entries, one per line."
    )
    projects: list[str] = Field(
        default_factory=list, description="Project descriptions."
    )


# --- Job description (JD agent output, Requirement 2.2) --------------------


class JDRequirements(BaseModel):
    """Structured requirements extracted from a job description."""

    must_have: list[str] = Field(
        default_factory=list, description="Required (must-have) requirements."
    )
    nice_to_have: list[str] = Field(
        default_factory=list, description="Preferred (nice-to-have) requirements."
    )
    keywords: list[str] = Field(
        default_factory=list, description="ATS keyword terms from the JD."
    )
    min_years: int = Field(
        default=0,
        ge=0,
        description="Minimum years of experience required (0 if unspecified).",
    )


# --- Matching (Matcher agent output, Requirement 3) ------------------------


class MatchItem(BaseModel):
    """Verdict for a single requirement, with a supporting quote."""

    id: str = Field(description="Identifier of the requirement being matched.")
    status: str = Field(
        description="One of 'met', 'partial', 'missing' (or 'unverified')."
    )
    evidence: str = Field(
        default="",
        description="Verbatim quote from the resume supporting the status.",
    )


class MatchResult(BaseModel):
    """All requirement verdicts for one analysis."""

    items: list[MatchItem] = Field(
        default_factory=list, description="Per-requirement match verdicts."
    )


# --- Rewrites (Rewriter agent output, Requirement 6.1) ---------------------


class Rewrite(BaseModel):
    """A proposed rewrite of an existing resume bullet."""

    original: str = Field(description="The existing resume bullet, verbatim.")
    suggested: str = Field(description="The proposed rewritten bullet.")
    target: str = Field(
        default="",
        description="The requirement this rewrite targets.",
    )


class RewriteResult(BaseModel):
    """All rewrite proposals from one Rewriter round."""

    rewrites: list[Rewrite] = Field(
        default_factory=list, description="Proposed bullet rewrites."
    )


# --- Review (Reviewer agent output, Requirement 6.2) -----------------------


class Verdict(BaseModel):
    """Reviewer verdict on a single rewrite, keyed by its index."""

    index: int = Field(
        ge=0, description="Index of the rewrite in the RewriteResult list."
    )
    supported: bool = Field(
        description="True if the rewrite adds no facts absent from the resume."
    )
    reason: str = Field(
        default="",
        description="Explanation, especially when the rewrite is rejected.",
    )


class ReviewResult(BaseModel):
    """All reviewer verdicts for one Rewriter round."""

    verdicts: list[Verdict] = Field(
        default_factory=list, description="Per-rewrite reviewer verdicts."
    )
