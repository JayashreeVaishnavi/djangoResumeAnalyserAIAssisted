# Requirements: ResumeLens

## Introduction
A local Django app that analyses one resume against one job description using local LLMs (Ollama), with an agent pipeline, evidence-backed scoring and human review of rewrite suggestions.

## Requirements

### 1. Upload inputs
**User story:** As a job seeker, I want to upload my resume and paste a JD, so that I can get an analysis.
1. WHEN I submit a PDF, DOCX or TXT resume and JD text THE SYSTEM SHALL store both and create an Analysis with status `pending`.
2. IF the file type is unsupported or no text can be extracted THEN THE SYSTEM SHALL show a clear error and create nothing.
3. WHEN an Analysis is created THE SYSTEM SHALL start the pipeline in the background and redirect me to its detail page.

### 2. Structured extraction
1. WHEN the pipeline runs THE Parser agent SHALL convert the resume into validated JSON (skills, roles with dates and bullets, education, projects).
2. WHEN the pipeline runs THE JD agent SHALL extract must-have requirements, nice-to-have requirements, keyword terms and minimum years.
3. IF model output fails schema validation THEN THE SYSTEM SHALL retry up to 2 times with the validation error included, then fail the Analysis with a readable message.

### 3. Evidence-backed matching
1. THE Matcher agent SHALL label each requirement `met`, `partial` or `missing` and give a quote from the resume as evidence.
2. THE SYSTEM SHALL retrieve the top 3 resume chunks per requirement via embedding similarity and give them to the Matcher.
3. IF a quote cannot be verified in the resume text THEN THE SYSTEM SHALL mark the match `unverified` and score it as missing.

### 4. Hybrid scoring
1. THE SYSTEM SHALL compute score = 60% weighted requirement coverage + 25% semantic similarity + 15% ATS keyword coverage.
2. THE SYSTEM SHALL weight must-have requirements 1.0 and nice-to-have 0.5, with `partial` counting 0.5.
3. THE SYSTEM SHALL show the three component scores alongside the final score.

### 5. ATS checks
1. THE SYSTEM SHALL report JD keywords found and missing, using a skill alias normaliser (e.g. k8s = kubernetes).
2. THE SYSTEM SHALL flag missing sections (experience, education, skills) and missing contact details.

### 6. Rewrite suggestions with review
1. THE Rewriter SHALL propose rewrites of EXISTING resume bullets targeting partial or missing requirements.
2. THE Reviewer SHALL reject any suggestion adding facts not present in the resume; rejected ones are re-sent to the Rewriter with reasons, up to 2 rounds.
3. WHEN the analysis finishes THE SYSTEM SHALL let me accept or reject each approved suggestion; blocked ones are shown separately with the reason.

### 7. Progress and traceability
1. WHILE the pipeline runs THE detail page SHALL display the current stage via polling.
2. THE SYSTEM SHALL record per agent: latency, attempts, prompt and output tokens, and output; and per tool call: arguments and result.
3. THE detail page SHALL show this trace, and Django admin SHALL expose all models.

### 8. Local-only operation
1. THE SYSTEM SHALL make no network calls except to the local Ollama server.
2. THE SYSTEM SHALL provide a `check_llm` command that verifies Ollama is reachable and both models are pulled.
3. IF Ollama is unreachable THEN THE Analysis SHALL fail with an actionable message.
