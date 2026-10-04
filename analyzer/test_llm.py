"""Tests for the local LLM client and the ``check_llm`` command.

All tests stub the HTTP layer (``requests``) so no network calls are made and
no running Ollama server is required.
"""

from __future__ import annotations

import json
from io import StringIO
from unittest import mock

import pytest
import requests
from django.core.management import call_command
from django.core.management.base import CommandError
from pydantic import BaseModel

from analyzer import llm


class SampleSchema(BaseModel):
    name: str
    years: int


def _chat_response(content: str, prompt_tokens: int = 5, eval_tokens: int = 7):
    """Build a fake Ollama /api/chat JSON response object."""
    fake = mock.Mock()
    fake.json.return_value = {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
    }
    fake.raise_for_status.return_value = None
    return fake


# --- chat_json: happy path -------------------------------------------------

def test_chat_json_returns_model_and_meta():
    good = json.dumps({"name": "Ada", "years": 3})
    with mock.patch(
        "analyzer.llm.requests.post", return_value=_chat_response(good)
    ) as post:
        parsed, meta = llm.chat_json("sys", "usr", SampleSchema)

    assert isinstance(parsed, SampleSchema)
    assert parsed.name == "Ada"
    assert parsed.years == 3
    assert meta["attempts"] == 1
    assert meta["prompt_tokens"] == 5
    assert meta["output_tokens"] == 7
    assert meta["latency_ms"] >= 0
    # format must carry the JSON schema; temperature must be 0.1.
    _, kwargs = post.call_args
    sent = kwargs["json"]
    assert sent["format"] == SampleSchema.model_json_schema()
    assert sent["options"]["temperature"] == 0.1


def test_chat_json_accepts_raw_schema_dict():
    good = json.dumps({"name": "Ada", "years": 3})
    with mock.patch(
        "analyzer.llm.requests.post", return_value=_chat_response(good)
    ):
        parsed, meta = llm.chat_json("sys", "usr", SampleSchema.model_json_schema())
    assert parsed == {"name": "Ada", "years": 3}
    assert meta["attempts"] == 1


# --- chat_json: retry then success -----------------------------------------

def test_chat_json_retries_on_validation_error_then_succeeds():
    bad = json.dumps({"name": "Ada"})  # missing 'years'
    good = json.dumps({"name": "Ada", "years": 3})
    responses = [_chat_response(bad), _chat_response(good)]
    with mock.patch(
        "analyzer.llm.requests.post", side_effect=responses
    ) as post:
        parsed, meta = llm.chat_json("sys", "usr", SampleSchema)

    assert parsed.years == 3
    assert meta["attempts"] == 2
    # Tokens accumulate across attempts.
    assert meta["prompt_tokens"] == 10
    assert meta["output_tokens"] == 14
    # Second request should include the fed-back validation error.
    second_payload = post.call_args_list[1].kwargs["json"]
    assert any("validation" in m["content"].lower() for m in second_payload["messages"])


# --- chat_json: exhausts retries -------------------------------------------

def test_chat_json_fails_after_max_retries():
    bad = json.dumps({"name": "Ada"})
    with mock.patch(
        "analyzer.llm.requests.post",
        side_effect=[_chat_response(bad) for _ in range(5)],
    ) as post:
        with pytest.raises(llm.LLMError):
            llm.chat_json("sys", "usr", SampleSchema)
    # First attempt + 2 retries = 3 calls.
    assert post.call_count == 3


# --- chat_json: connection error -> actionable message ---------------------

def test_chat_json_maps_connection_error():
    with mock.patch(
        "analyzer.llm.requests.post", side_effect=requests.ConnectionError()
    ):
        with pytest.raises(llm.LLMConnectionError) as exc:
            llm.chat_json("sys", "usr", SampleSchema)
    assert "check_llm" in str(exc.value)


def test_chat_json_maps_timeout():
    with mock.patch(
        "analyzer.llm.requests.post", side_effect=requests.Timeout()
    ):
        with pytest.raises(llm.LLMConnectionError):
            llm.chat_json("sys", "usr", SampleSchema)


# --- embed -----------------------------------------------------------------

def test_embed_returns_vectors():
    fake = mock.Mock()
    fake.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    fake.raise_for_status.return_value = None
    with mock.patch("analyzer.llm.requests.post", return_value=fake) as post:
        vectors = llm.embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    _, kwargs = post.call_args
    assert kwargs["json"]["input"] == ["a", "b"]


def test_embed_empty_input_short_circuits():
    with mock.patch("analyzer.llm.requests.post") as post:
        assert llm.embed([]) == []
    post.assert_not_called()


def test_embed_maps_connection_error():
    with mock.patch(
        "analyzer.llm.requests.post", side_effect=requests.ConnectionError()
    ):
        with pytest.raises(llm.LLMConnectionError):
            llm.embed(["a"])


# --- check_llm command -----------------------------------------------------

def _tags_response(names):
    fake = mock.Mock()
    fake.json.return_value = {"models": [{"name": n} for n in names]}
    fake.raise_for_status.return_value = None
    return fake


def test_check_llm_success(settings):
    settings.LLM_MODEL = "llama3.1"
    settings.EMBED_MODEL = "nomic-embed-text"
    with mock.patch(
        "analyzer.management.commands.check_llm.requests.get",
        return_value=_tags_response(["llama3.1:latest", "nomic-embed-text:latest"]),
    ):
        out = StringIO()
        call_command("check_llm", stdout=out)
    assert "All required models are available." in out.getvalue()


def test_check_llm_unreachable():
    with mock.patch(
        "analyzer.management.commands.check_llm.requests.get",
        side_effect=requests.ConnectionError(),
    ):
        with pytest.raises(CommandError) as exc:
            call_command("check_llm")
    assert "check_llm" in str(exc.value)


def test_check_llm_missing_model(settings):
    settings.LLM_MODEL = "llama3.1"
    settings.EMBED_MODEL = "nomic-embed-text"
    with mock.patch(
        "analyzer.management.commands.check_llm.requests.get",
        return_value=_tags_response(["llama3.1:latest"]),
    ):
        with pytest.raises(CommandError) as exc:
            call_command("check_llm")
    assert "nomic-embed-text" in str(exc.value)
