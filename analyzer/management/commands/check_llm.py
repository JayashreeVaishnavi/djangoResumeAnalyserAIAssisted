"""``check_llm`` management command.

Verifies that the local Ollama server is reachable and that both configured
models (``LLM_MODEL`` and ``EMBED_MODEL``) are pulled (Requirement 8.2).

Usage::

    python manage.py check_llm
"""

from __future__ import annotations

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analyzer.llm import OLLAMA_UNREACHABLE_MESSAGE


class Command(BaseCommand):
    help = "Verify the local Ollama server is reachable and required models are pulled."

    def handle(self, *args, **options):
        base_url = settings.OLLAMA_URL.rstrip("/")

        # 1. Is the server reachable? /api/tags lists installed models.
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise CommandError(OLLAMA_UNREACHABLE_MESSAGE) from exc
        except requests.RequestException as exc:
            raise CommandError(f"Failed to query Ollama at {base_url}: {exc}") from exc

        self.stdout.write(self.style.SUCCESS(f"Ollama reachable at {base_url}"))

        # 2. Are both required models pulled? Match with or without a tag,
        #    since models are commonly referenced as "llama3.1" vs
        #    "llama3.1:latest".
        installed = {m.get("name", "") for m in data.get("models", [])}
        installed_bases = {name.split(":", 1)[0] for name in installed}

        required = {
            "LLM_MODEL": settings.LLM_MODEL,
            "EMBED_MODEL": settings.EMBED_MODEL,
        }

        missing = []
        for label, model in required.items():
            base = model.split(":", 1)[0]
            if model in installed or base in installed_bases:
                self.stdout.write(
                    self.style.SUCCESS(f"{label} '{model}' is pulled")
                )
            else:
                missing.append(model)
                self.stdout.write(
                    self.style.ERROR(f"{label} '{model}' is NOT pulled")
                )

        if missing:
            pulls = "\n".join(f"  ollama pull {m}" for m in missing)
            raise CommandError(
                "The following required models are not pulled:\n"
                f"{pulls}"
            )

        self.stdout.write(self.style.SUCCESS("All required models are available."))
