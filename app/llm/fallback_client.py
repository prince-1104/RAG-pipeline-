from __future__ import annotations

from typing import Any

from app.llm.base import LLMResult
from app.llm.gemini import GeminiClient
from app.llm.openrouter_client import OpenRouterClient
from app.observability.logging import log


class FallbackLLMClient:
    """
    LLM client that tries GPT (via OpenRouter) first, falls back to Gemini if GPT fails.
    
    Requirement: GPT-4.1-mini primary, Gemini secondary.
    This ensures cost-effective primary generation with reliable fallback.
    """

    def __init__(self, *, gemini_client: GeminiClient, openrouter_client: OpenRouterClient):
        self.gemini_client = gemini_client
        self.openrouter_client = openrouter_client

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        """
        Try GPT (OpenRouter) first, fall back to Gemini if GPT fails.
        Matches requirement: GPT-4.1-mini primary, Gemini secondary.
        """
        # Try GPT (OpenRouter) first (primary)
        try:
            log.debug("llm_fallback_trying_gpt_openrouter")
            result = self.openrouter_client.generate_json(
                system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
            )
            log.info("llm_fallback_gpt_success")
            return result
        except Exception as e:
            error_msg = str(e)[:200]
            log.warning("llm_fallback_gpt_failed", error=error_msg, error_type=type(e).__name__)
            # Fall back to Gemini (secondary)
            try:
                log.info("llm_fallback_using_gemini")
                result = self.gemini_client.generate_json(
                    system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
                )
                log.info("llm_fallback_gemini_success")
                return result
            except Exception as fallback_error:
                error_msg_fallback = str(fallback_error)[:200]
                log.error("llm_fallback_both_failed", gpt_error=error_msg, gemini_error=error_msg_fallback)
                # Raise the original GPT error (more informative)
                raise RuntimeError(f"llm_generation_failed: GPT (OpenRouter) failed ({error_msg}), Gemini also failed ({error_msg_fallback})") from e
