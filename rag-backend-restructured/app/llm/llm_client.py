"""Thin wrapper around the OpenAI-compatible SDK.

Kept as a single narrow interface (`LLMClient`) so the rest of the codebase
never imports `openai` directly — swapping providers means editing only
this file.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.core.config import get_settings


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)
        self._chat_model = settings.chat_model
        self._embedding_model = settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def extract_structured(
        self,
        messages: list[dict[str, str]],
        tool_name: str,
        tool_description: str,
        parameters_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Use tool/function calling to force a structured JSON response.

        Returns the parsed arguments dict, or None if the model chose not to
        call the tool (e.g. not enough information yet).
        """
        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": parameters_schema,
                    },
                }
            ],
            tool_choice="auto",
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return None
        call = message.tool_calls[0]
        try:
            return json.loads(call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            return None


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()
