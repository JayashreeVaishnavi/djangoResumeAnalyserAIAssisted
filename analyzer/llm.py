"""Local LLM client for ResumeLens.

This module is the single boundary between the app and the local Ollama
server. It exposes two helpers:

* ``chat_json(system, user, schema)`` -- a structured chat call that forces
  the model to return JSON matching a Pydantic model (via Ollama's ``format``
  parameter), validates the response and retries on validation errors with the
  error fed back into the prompt (Requirement 2.3).
* ``embed(texts)`` -- turn a list of strings into embedding vectors.

The only external endpoint contacted is the local Ollama server configured by
``settings.OLLAMA_URL`` (Requirement 8.1). Connection problems are mapped to an
actionable message (Requirement 8.3).
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
from django.conf import settings
from pydantic import BaseModel, ValidationError

# Shown whenever the local Ollama server cannot be reached. Kept as a module
# constant so views/pipeline and tests can reference the exact wording.
OLLAMA_UNREACHABLE_MESSAGE = (
    "Cannot reach the local LLM server. "
    "Start Ollama and run `python manage.py check_llm`"
)

# Maximum number of extra attempts after the first one when validation fails.
MAX_VALIDATION_RETRIES = 2

# Deterministic-ish decoding for structured extraction.
DEFAULT_TEMPERATURE = 0.1

# Generous per-request timeout; local models can be slow on first load.
REQUEST_TIMEOUT = 300


class LLMError(RuntimeError):
    """Raised for any failure talking to the local LLM server."""


class LLMConnectionError(LLMError):
    """Raised when the local Ollama server is unreachable or times out.

    The message is deliberately actionable (Requirement 8.3).
    """


def _schema_of(schema: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON schema dict for a Pydantic model class or raw schema.

    ``chat_json`` accepts either a Pydantic model class or an already-built
    JSON schema dict so it does not hard-depend on specific schema classes.
    """
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    raise TypeError(
        "schema must be a Pydantic model class or a JSON schema dict, "
        f"got {type(schema)!r}"
    )


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to the Ollama server and return the decoded response.

    Maps requests connection/timeout errors to :class:`LLMConnectionError`
    with an actionable message.
    """
    url = f"{settings.OLLAMA_URL.rstrip('/')}{path}"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise LLMConnectionError(OLLAMA_UNREACHABLE_MESSAGE) from exc
    except requests.RequestException as exc:  # pragma: no cover - defensive
        raise LLMError(f"LLM request to {path} failed: {exc}") from exc


def chat_json(
    system: str,
    user: str,
    schema: type[BaseModel] | dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Call the chat model and return ``(parsed_model, meta)``.

    The response is constrained to the given JSON ``schema`` via Ollama's
    ``format`` parameter and validated against it. On a validation error the
    call is retried up to :data:`MAX_VALIDATION_RETRIES` times, feeding the
    validation error text back into the prompt (Requirement 2.3). After the
    final failure a readable :class:`LLMError` is raised.

    ``meta`` contains: ``attempts``, ``latency_ms``, ``prompt_tokens``
    (``prompt_eval_count``) and ``output_tokens`` (``eval_count``).

    When ``schema`` is a Pydantic model class the first element is an instance
    of that model; when it is a raw JSON schema dict the first element is the
    parsed ``dict``.
    """
    json_schema = _schema_of(schema)
    is_model = isinstance(schema, type) and issubclass(schema, BaseModel)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    start = time.monotonic()
    prompt_tokens = 0
    output_tokens = 0
    last_error: ValidationError | None = None

    # First attempt plus up to MAX_VALIDATION_RETRIES retries.
    total_attempts = MAX_VALIDATION_RETRIES + 1
    for attempt in range(1, total_attempts + 1):
        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "format": json_schema,
            "stream": False,
            "options": {"temperature": DEFAULT_TEMPERATURE},
        }

        data = _post("/api/chat", payload)

        # Accumulate token usage across attempts for an honest trace.
        prompt_tokens += int(data.get("prompt_eval_count", 0) or 0)
        output_tokens += int(data.get("eval_count", 0) or 0)

        content = (data.get("message") or {}).get("content", "") or ""

        try:
            if is_model:
                parsed = schema.model_validate_json(content)
            else:
                parsed = json.loads(content)
            meta = {
                "attempts": attempt,
                "latency_ms": int((time.monotonic() - start) * 1000),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
            }
            return parsed, meta
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt > MAX_VALIDATION_RETRIES:
                break
            # Feed the validation error back so the model can self-correct.
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required "
                        "JSON schema. Fix these validation errors and return "
                        "ONLY valid JSON matching the schema:\n"
                        f"{exc}"
                    ),
                }
            )

    raise LLMError(
        "Model output failed schema validation after "
        f"{total_attempts} attempts: {last_error}"
    )


def embed(texts: list[str]) -> list[list[float]]:
    """Return an embedding vector for each string in ``texts``.

    Uses Ollama's ``/api/embed`` endpoint with the configured embedding model.
    Connection errors are mapped to an actionable message (Requirement 8.3).
    """
    if not texts:
        return []

    payload = {"model": settings.EMBED_MODEL, "input": texts}
    data = _post("/api/embed", payload)
    embeddings = data.get("embeddings")
    if embeddings is None:
        raise LLMError("Embedding response did not contain 'embeddings'")
    return embeddings
