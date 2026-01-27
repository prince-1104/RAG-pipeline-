from __future__ import annotations

from typing import Any

from app.llm.base import LLMClient, LLMResult
from app.observability.logging import log


class FallbackChainLLMClient:
    """
    Generic 2-step fallback: try primary, then secondary on error.
    """

    def __init__(self, *, primary: LLMClient, secondary: LLMClient, primary_name: str, secondary_name: str):
        self.primary = primary
        self.secondary = secondary
        self.primary_name = primary_name
        self.secondary_name = secondary_name

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        try:
            log.debug("llm_fallback_trying_primary", provider=self.primary_name)
            res = self.primary.generate_json(system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema)
            log.info("llm_fallback_primary_success", provider=self.primary_name)
            return res
        except Exception as e:  # noqa: BLE001
            primary_err = str(e)[:200]
            log.warning(
                "llm_fallback_primary_failed",
                provider=self.primary_name,
                error=primary_err,
                error_type=type(e).__name__,
            )
            try:
                log.info("llm_fallback_trying_secondary", provider=self.secondary_name)
                res = self.secondary.generate_json(
                    system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
                )
                log.info("llm_fallback_secondary_success", provider=self.secondary_name)
                return res
            except Exception as e2:  # noqa: BLE001
                secondary_err = str(e2)[:200]
                log.error(
                    "llm_fallback_both_failed",
                    primary_provider=self.primary_name,
                    secondary_provider=self.secondary_name,
                    primary_error=primary_err,
                    secondary_error=secondary_err,
                )
                raise RuntimeError(
                    f"llm_generation_failed: {self.primary_name} failed ({primary_err}); {self.secondary_name} failed ({secondary_err})"
                ) from e

