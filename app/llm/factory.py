from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.fallback_client import FallbackLLMClient
from app.llm.gemini import GeminiClient
from app.llm.openrouter_client import OpenRouterClient
from app.llm.race_client import RaceLLMClient
from app.llm.stub import StubLLMClient
from app.settings import get_settings


def get_llm_client() -> LLMClient:
    s = get_settings()
    provider = (s.llm_provider or "stub").lower()
    if provider == "stub":
        return StubLLMClient()
    if provider == "gemini":
        if not s.gemini_api_key:
            raise RuntimeError("missing_GEMINI_API_KEY")
        return GeminiClient(api_key=s.gemini_api_key, model=s.gemini_model)
    if provider == "openrouter":
        if not s.openrouter_api_key:
            raise RuntimeError("missing_OPENROUTER_API_KEY")
        return OpenRouterClient(
            api_key=s.openrouter_api_key,
            model=s.openrouter_model,
            base_url=s.openrouter_base_url
        )
    if provider == "race":
        # Race mode: use both Gemini and OpenRouter, whichever responds first
        if not s.gemini_api_key:
            raise RuntimeError("missing_GEMINI_API_KEY_for_race_mode")
        if not s.openrouter_api_key:
            raise RuntimeError("missing_OPENROUTER_API_KEY_for_race_mode")
        gemini_client = GeminiClient(api_key=s.gemini_api_key, model=s.gemini_model)
        openrouter_client = OpenRouterClient(
            api_key=s.openrouter_api_key,
            model=s.openrouter_model,
            base_url=s.openrouter_base_url
        )
        return RaceLLMClient(gemini_client=gemini_client, openrouter_client=openrouter_client)
    if provider == "fallback":
        # Fallback mode: try Gemini first, fall back to GPT-4o-mini (OpenRouter) if Gemini fails
        # Requirement: Gemini primary, GPT-4o-mini secondary
        if not s.gemini_api_key:
            raise RuntimeError("missing_GEMINI_API_KEY_for_fallback_mode")
        if not s.openrouter_api_key:
            raise RuntimeError("missing_OPENROUTER_API_KEY_for_fallback_mode")
        gemini_client = GeminiClient(api_key=s.gemini_api_key, model=s.gemini_model)
        openrouter_client = OpenRouterClient(
            api_key=s.openrouter_api_key,
            model=s.openrouter_model,
            base_url=s.openrouter_base_url
        )
        return FallbackLLMClient(gemini_client=gemini_client, openrouter_client=openrouter_client)
    raise RuntimeError(f"unsupported_llm_provider:{s.llm_provider}. Supported: stub, gemini, openrouter, race, fallback")


