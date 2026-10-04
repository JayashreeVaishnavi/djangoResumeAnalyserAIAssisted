from django.contrib import admin

from .models import (
    Resume,
    JobDescription,
    Analysis,
    AgentRun,
    ToolCall,
    Suggestion,
)


class AgentRunInline(admin.TabularInline):
    model = AgentRun
    extra = 0
    can_delete = False
    readonly_fields = (
        "agent",
        "attempts",
        "latency_ms",
        "prompt_tokens",
        "output_tokens",
        "output",
        "created_at",
    )


class ToolCallInline(admin.TabularInline):
    model = ToolCall
    extra = 0
    can_delete = False
    readonly_fields = ("agent", "name", "arguments", "result", "created_at")


class SuggestionInline(admin.TabularInline):
    model = Suggestion
    extra = 0
    readonly_fields = ("created_at",)


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "resume",
        "jd",
        "status",
        "stage",
        "match_score",
        "created_at",
    )
    list_filter = ("status", "stage")
    search_fields = ("id", "resume__name", "jd__title")
    readonly_fields = ("created_at", "updated_at")
    inlines = [AgentRunInline, ToolCallInline, SuggestionInline]


@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "file", "created_at")
    search_fields = ("name",)
    readonly_fields = ("created_at",)


@admin.register(JobDescription)
class JobDescriptionAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "created_at")
    search_fields = ("title",)
    readonly_fields = ("created_at",)


@admin.register(AgentRun)
class AgentRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "analysis",
        "agent",
        "attempts",
        "latency_ms",
        "prompt_tokens",
        "output_tokens",
        "created_at",
    )
    list_filter = ("agent",)
    search_fields = ("analysis__id", "agent")
    readonly_fields = ("created_at",)


@admin.register(ToolCall)
class ToolCallAdmin(admin.ModelAdmin):
    list_display = ("id", "analysis", "agent", "name", "created_at")
    list_filter = ("name",)
    search_fields = ("analysis__id", "name", "agent")
    readonly_fields = ("created_at",)


@admin.register(Suggestion)
class SuggestionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "analysis",
        "status",
        "reviewer_passed",
        "created_at",
    )
    list_filter = ("status", "reviewer_passed")
    search_fields = ("analysis__id",)
    readonly_fields = ("created_at",)
