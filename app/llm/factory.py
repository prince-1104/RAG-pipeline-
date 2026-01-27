from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.fallback_chain_client import FallbackChainLLMClient
from app.llm.fallback_client import FallbackLLMClient
from app.llm.gemini import GeminiClient
from app.llm.multi_race_client import MultiRaceLLMClient
from app.llm.openai_client import OpenAIClient
from app.llm.openrouter_client import OpenRouterClient
from app.llm.race_client import RaceLLMClient
from app.llm.stub import StubLLMClient
from app.settings import get_settings


def get_llm_client() -> LLMClient:
    s = get_settings()
    provider = (s.llm_provider or "stub").lower()
    if provider == "stub":
        return StubLLMClient()
    if provider in ("openai", "gpt"):
        if not s.openai_api_key:
            raise RuntimeError("missing_OPENAI_API_KEY")
        return OpenAIClient(
            api_key=s.openai_api_key,
            model=s.openai_model,
            base_url=s.openai_base_url,
            timeout_s=s.llm_timeout_s,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )
    if provider in ("openai_fallback_gemini", "openai_then_gemini"):
        # Primary: OpenAI (direct). Secondary: Gemini. No OpenRouter.
        if not s.openai_api_key:
            raise RuntimeError("missing_OPENAI_API_KEY_for_openai_fallback_gemini")
        if not s.gemini_api_key:
            raise RuntimeError("missing_GEMINI_API_KEY_for_openai_fallback_gemini")
        openai_client = OpenAIClient(
            api_key=s.openai_api_key,
            model=s.openai_model,
            base_url=s.openai_base_url,
            timeout_s=s.llm_timeout_s,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )
        gemini_client = GeminiClient(api_key=s.gemini_api_key, model=s.gemini_model)
        return FallbackChainLLMClient(
            primary=openai_client,
            secondary=gemini_client,
            primary_name="openai",
            secondary_name="gemini",
        )
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
            base_url=s.openrouter_base_url,
            timeout_s=s.llm_timeout_s,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
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
            base_url=s.openrouter_base_url,
            timeout_s=s.llm_timeout_s,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )
        return RaceLLMClient(gemini_client=gemini_client, openrouter_client=openrouter_client)
    if provider == "race_all":
        # Race all configured providers in parallel: OpenAI + Gemini + OpenRouter (any subset >= 2)
        clients: dict[str, LLMClient] = {}
        if s.openai_api_key:
            clients["openai"] = OpenAIClient(
                api_key=s.openai_api_key,
                model=s.openai_model,
                base_url=s.openai_base_url,
                timeout_s=s.llm_timeout_s,
                temperature=s.llm_temperature,
                max_tokens=s.llm_max_tokens,
            )
        if s.gemini_api_key:
            clients["gemini"] = GeminiClient(api_key=s.gemini_api_key, model=s.gemini_model)
        if s.openrouter_api_key:
            clients["openrouter"] = OpenRouterClient(
                api_key=s.openrouter_api_key,
                model=s.openrouter_model,
                base_url=s.openrouter_base_url,
                timeout_s=s.llm_timeout_s,
                temperature=s.llm_temperature,
                max_tokens=s.llm_max_tokens,
            )
        if len(clients) < 2:
            raise RuntimeError("race_all_requires_at_least_two_providers: OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY")
        return MultiRaceLLMClient(clients=clients)
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
            base_url=s.openrouter_base_url,
            timeout_s=s.llm_timeout_s,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )
        return FallbackLLMClient(gemini_client=gemini_client, openrouter_client=openrouter_client)
    raise RuntimeError(
        f"unsupported_llm_provider:{s.llm_provider}. Supported: stub, openai, gpt, openai_fallback_gemini, openai_then_gemini, gemini, openrouter, race, race_all, fallback"
    )


