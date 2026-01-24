from __future__ import annotations

import concurrent.futures
from typing import Any

from app.llm.base import LLMResult
from app.llm.gemini import GeminiClient
from app.llm.openrouter_client import OpenRouterClient
from app.observability.logging import log


class RaceLLMClient:
    """
    LLM client that calls multiple providers in parallel and returns the first successful response.
    This improves latency by using whichever provider responds fastest.
    """

    def __init__(self, *, gemini_client: GeminiClient, openrouter_client: OpenRouterClient):
        self.gemini_client = gemini_client
        self.openrouter_client = openrouter_client

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        """
        Call both Gemini and OpenRouter in parallel, return the first successful response.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            gemini_future = executor.submit(
                self._call_gemini, system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
            )
            openrouter_future = executor.submit(
                self._call_openrouter, system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
            )

            # Wait for first successful response
            futures = {
                gemini_future: "gemini",
                openrouter_future: "openrouter"
            }
            errors = {}
            
            # Use as_completed to get the first result
            for future in concurrent.futures.as_completed(futures.keys()):
                provider_name = futures[future]
                try:
                    result = future.result()
                    # Log which provider won the race
                    log.info("llm_race_winner", provider=provider_name)
                    
                    # Cancel the other future (best effort - it may have already started)
                    for other_future, other_name in futures.items():
                        if other_name != provider_name:
                            other_future.cancel()
                    
                    return result
                except Exception as e:
                    errors[provider_name] = str(e)
                    log.warning("llm_race_provider_failed", provider=provider_name, error=str(e)[:200])
                    continue

            # If we get here, both providers failed
            error_summary = "; ".join([f"{p}: {e[:100]}" for p, e in errors.items()])
            raise RuntimeError(f"all_llm_providers_failed: {error_summary}")

    def _call_gemini(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        """Call Gemini client"""
        return self.gemini_client.generate_json(
            system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
        )

    def _call_openrouter(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        """Call OpenRouter client"""
        return self.openrouter_client.generate_json(
            system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
        )
