---
inclusion: always
---
# Project structure

```
config/                 settings, urls, wsgi
analyzer/
  models.py             Resume, JobDescription, Analysis, AgentRun, ToolCall, Suggestion
  schemas.py            Pydantic output schemas for each agent
  llm.py                Ollama client: chat_json (retries, token/latency capture), embed
  parsing.py            PDF / DOCX / TXT text extraction
  tools.py              chunk_text, semantic_search, normalize_skill, years_of_experience, ats_check, verify_evidence
  agents.py             parser, jd, matcher, rewriter, reviewer agents (prompts live here)
  scoring.py            hybrid score
  pipeline.py           orchestrates agents, logs traces, updates Analysis.stage
  views.py / urls.py    home (upload form), detail, status JSON, suggestion accept/reject
  templates/analyzer/   base.html, home.html, detail.html
  management/commands/check_llm.py   verifies Ollama + models are available
tests/                  mocked-LLM tests (pipeline, tools, views)
```
