# Design: ResumeLens

## Overview
Single Django project, SQLite, one background thread per analysis, Ollama for chat and embeddings.

## Pipeline
```
upload -> Analysis(pending) -> thread: run_pipeline(id)
 1 Parser agent      resume text -> ParsedResume
 2 JD agent          JD text -> JDRequirements
 3 Tools             chunk_text, embed, semantic_search (top 3 per requirement)
 4 Matcher agent     requirements + retrieved chunks -> MatchResult (+ verify_evidence guard)
 5 Tools             years_of_experience, ats_check
 6 Scoring           hybrid score
 7 Rewriter <-> Reviewer loop (max 2 rounds) -> Suggestion rows
 -> Analysis(done) with report JSON
```
Tools are called deterministically by the orchestrator (small local models are unreliable at native function calling) and every call is logged as a `ToolCall`.

## Data model
Resume(name, file, raw_text) | JobDescription(title, raw_text, requirements JSON)
Analysis(resume, jd, status, stage, match_score, report JSON, error)
AgentRun(analysis, agent, attempts, latency_ms, prompt_tokens, output_tokens, output)
ToolCall(analysis, agent, name, arguments, result)
Suggestion(analysis, original, suggested, reviewer_passed, reviewer_note, status)

## Schemas (Pydantic)
ParsedResume, Role, JDRequirements(must_have, nice_to_have, keywords, min_years),
MatchItem(id, status, evidence), MatchResult, Rewrite, RewriteResult, Verdict(index, supported, reason), ReviewResult

## Key components
- `llm.chat_json(system, user, schema)` -> (model, meta). POST /api/chat with `format` = JSON schema, temperature 0.1, retry on ValidationError.
- `llm.embed(texts)` -> list of vectors via POST /api/embed.
- `tools.verify_evidence` normalises whitespace/case and accepts exact containment or >=80% longest-match overlap.
- `scoring.final_score(coverage, semantic, ats)`; semantic = mean top-1 similarity over must-haves, rescaled (sim-0.3)/0.5 clamped to 0..1.

## Error handling
Any exception inside the pipeline sets status `failed` and stores `error`. Connection errors map to "Start Ollama and run `python manage.py check_llm`".

## Testing
pytest with fake `chat_json` (returns canned schema instances) and fake `embed` (hashed bag-of-words vectors).
Tests: tools (alias, years, verify_evidence, ats), scoring, pipeline end to end, views (upload, status, accept/reject).
