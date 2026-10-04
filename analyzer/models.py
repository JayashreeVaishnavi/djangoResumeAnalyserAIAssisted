from django.db import models


class Resume(models.Model):
    """An uploaded resume plus the text extracted from it."""

    name = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to="resumes/")
    raw_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name or f"Resume #{self.pk}"


class JobDescription(models.Model):
    """A pasted job description plus structured requirements once extracted."""

    title = models.CharField(max_length=255, blank=True)
    raw_text = models.TextField()
    requirements = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or f"JobDescription #{self.pk}"


class Analysis(models.Model):
    """One analysis of a resume against a job description."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    class Stage(models.TextChoices):
        QUEUED = "queued", "Queued"
        PARSING = "parsing", "Parsing resume"
        JD = "jd", "Extracting JD requirements"
        RETRIEVAL = "retrieval", "Retrieving evidence"
        MATCHING = "matching", "Matching requirements"
        SCORING = "scoring", "Scoring"
        REWRITING = "rewriting", "Rewriting suggestions"
        DONE = "done", "Done"

    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE, related_name="analyses"
    )
    jd = models.ForeignKey(
        JobDescription, on_delete=models.CASCADE, related_name="analyses"
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    stage = models.CharField(
        max_length=20, choices=Stage.choices, default=Stage.QUEUED
    )
    match_score = models.FloatField(null=True, blank=True)
    report = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Analyses"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Analysis #{self.pk} ({self.status})"


class AgentRun(models.Model):
    """Trace of a single agent invocation: latency, attempts and token usage."""

    analysis = models.ForeignKey(
        Analysis, on_delete=models.CASCADE, related_name="agent_runs"
    )
    agent = models.CharField(max_length=50)
    attempts = models.PositiveIntegerField(default=1)
    latency_ms = models.PositiveIntegerField(default=0)
    prompt_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    output = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.agent} run for Analysis #{self.analysis_id}"


class ToolCall(models.Model):
    """Trace of a deterministic tool call made by the orchestrator."""

    analysis = models.ForeignKey(
        Analysis, on_delete=models.CASCADE, related_name="tool_calls"
    )
    agent = models.CharField(max_length=50, blank=True)
    name = models.CharField(max_length=50)
    arguments = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.name} call for Analysis #{self.analysis_id}"


class Suggestion(models.Model):
    """A rewrite suggestion plus its reviewer verdict and user decision."""

    class Status(models.TextChoices):
        PROPOSED = "proposed", "Proposed"
        ACCEPTED = "accepted", "Accepted"
        REJECTED = "rejected", "Rejected"
        BLOCKED = "blocked", "Blocked"

    analysis = models.ForeignKey(
        Analysis, on_delete=models.CASCADE, related_name="suggestions"
    )
    original = models.TextField()
    suggested = models.TextField()
    reviewer_passed = models.BooleanField(default=False)
    reviewer_note = models.TextField(blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PROPOSED
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Suggestion #{self.pk} for Analysis #{self.analysis_id}"
