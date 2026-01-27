from __future__ import annotations

import json
from typing import Any

import httpx

from app.llm.base import LLMResult
from app.llm.stub import StubLLMClient


class OpenRouterClient:
    """
    OpenRouter API client for JSON generation.
    OpenRouter provides access to multiple LLM providers (OpenAI, Anthropic, etc.)
    through a unified API.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://openrouter.ai/api/v1"

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        """
        Generate JSON response using OpenRouter API.
        Falls back to stub if OpenRouter API fails.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-org/niyukti-que-gen-Engine",  # Optional: for analytics
            "X-Title": "Niyukti Question Generation",  # Optional: for analytics
        }

        # OpenRouter uses OpenAI-compatible API format
        # Include schema in system prompt for better compliance
        schema_description = ""
        if json_schema:
            import json as json_lib
            schema_description = f"\n\nIMPORTANT: Output must match this exact JSON schema:\n{json_lib.dumps(json_schema, indent=2)}\n\nKey requirements:\n- Use 'qtype' (not 'type') for question type\n- Use 'citations' (array, not 'citation')\n- Each citation must have: chunk_id, document_id, storage_uri, page_start, page_end, excerpt\n- All required fields must be present\n"
        
        enhanced_system_prompt = system_prompt + schema_description
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},  # Force JSON output
        }

        last_exception: Exception | None = None
        last_status: int | None = None
        last_resp_text: str | None = None

        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, headers=headers, json=payload)
                last_status = resp.status_code
                last_resp_text = resp.text

                if resp.is_success:
                    data = resp.json()
                    # Extract content from OpenRouter response
                    try:
                        text = data["choices"][0]["message"]["content"]
                        parsed = json.loads(text)
                        return LLMResult(raw_text=text, parsed_json=parsed)
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Failed to parse OpenRouter response: {e}")
                        # Fall through to stub fallback below

                # Handle errors - raise exceptions so FallbackLLMClient can catch and try Gemini
                if resp.status_code == 401 or resp.status_code == 403:
                    # Authentication error - raise so fallback client can try Gemini
                    import logging
                    logger = logging.getLogger(__name__)
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", {}).get("message", resp.text[:200])
                    except Exception:
                        error_msg = resp.text[:200]
                    logger.error(f"OpenRouter API authentication failed: {error_msg}")
                    raise RuntimeError(f"OpenRouter API authentication failed: {error_msg}")

                # For other errors, raise so fallback client can try Gemini
                if not resp.is_success:
                    import logging
                    logger = logging.getLogger(__name__)
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", {}).get("message", resp.text[:200])
                    except Exception:
                        error_msg = resp.text[:200]
                    logger.warning(f"OpenRouter API error (HTTP {resp.status_code}): {error_msg}")
                    raise RuntimeError(f"OpenRouter API error (HTTP {resp.status_code}): {error_msg}")

        except httpx.HTTPStatusError as e:
            # Re-raise HTTP errors so FallbackLLMClient can catch and try Gemini
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", e.response.text[:200])
            except Exception:
                error_msg = e.response.text[:200]
            raise RuntimeError(f"OpenRouter API HTTP error ({e.response.status_code}): {error_msg}") from e

        except Exception as e:
            # Re-raise other exceptions so FallbackLLMClient can catch and try Gemini
            raise RuntimeError(f"OpenRouter API request failed: {str(e)[:200]}") from e
