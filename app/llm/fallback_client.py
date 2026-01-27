from __future__ import annotations

from typing import Any

from app.llm.base import LLMResult
from app.llm.gemini import GeminiClient
from app.llm.openrouter_client import OpenRouterClient
from app.observability.logging import log


class FallbackLLMClient:
    """
    LLM client that tries Gemini first, falls back to GPT-4o-mini (via OpenRouter) if Gemini fails.
    
    Requirement: Gemini primary, GPT-4o-mini secondary.
    This ensures reliable primary generation with cost-effective fallback.
    """

    def __init__(self, *, gemini_client: GeminiClient, openrouter_client: OpenRouterClient):
        self.gemini_client = gemini_client
        self.openrouter_client = openrouter_client

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        """
        Try Gemini first (primary), fall back to GPT-4o-mini (OpenRouter) if Gemini fails.
        Matches requirement: Gemini primary, GPT-4o-mini secondary.
        """
        # Try Gemini first (primary)
        try:
            log.debug("llm_fallback_trying_gemini")
            result = self.gemini_client.generate_json(
                system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
            )
            log.info("llm_fallback_gemini_success")
            return result
        except Exception as e:
            error_msg = str(e)[:200]
            log.warning("llm_fallback_gemini_failed", error=error_msg, error_type=type(e).__name__)
            # Fall back to GPT-4o-mini (OpenRouter) (secondary)
            try:
                log.info("llm_fallback_using_gpt_openrouter")
                result = self.openrouter_client.generate_json(
                    system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
                )
                log.info("llm_fallback_gpt_success")
                return result
            except Exception as fallback_error:
                error_msg_fallback = str(fallback_error)[:200]
                log.error("llm_fallback_both_failed", gemini_error=error_msg, gpt_error=error_msg_fallback)
                # Raise the original Gemini error (more informative)
                raise RuntimeError(f"llm_generation_failed: Gemini failed ({error_msg}), GPT-4o-mini (OpenRouter) also failed ({error_msg_fallback})") from e
