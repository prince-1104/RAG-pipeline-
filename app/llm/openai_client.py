from __future__ import annotations

import json
from typing import Any

import httpx

from app.llm.base import LLMResult


class OpenAIClient:
    """
    Minimal OpenAI Chat Completions client for JSON generation.
    Uses OpenAI's API directly (not via OpenRouter).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        schema_description = ""
        if json_schema:
            schema_description = (
                "\n\nIMPORTANT: Output must match this exact JSON schema:\n"
                f"{json.dumps(json_schema, indent=2)}\n\n"
                "Key requirements:\n"
                "- Use 'qtype' (not 'type') for question type\n"
                "- Use 'citations' (array, not 'citation')\n"
                "- Each citation must have: chunk_id, document_id, storage_uri, page_start, page_end, excerpt\n"
                "- All required fields must be present\n"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt + schema_description},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code in (401, 403):
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", {}).get("message", resp.text[:200])
            except Exception:
                error_msg = resp.text[:200]
            raise RuntimeError(f"OpenAI API authentication failed: {error_msg}")

        if not resp.is_success:
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", {}).get("message", resp.text[:200])
            except Exception:
                error_msg = resp.text[:200]
            raise RuntimeError(f"OpenAI API error (HTTP {resp.status_code}): {error_msg}")

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"OpenAI response missing content: {e}") from e

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"OpenAI did not return valid JSON: {e}") from e

        return LLMResult(raw_text=text, parsed_json=parsed)

