from __future__ import annotations

from typing import Any, Literal

import httpx

from app.llm.base import LLMClient, LLMResult
from app.observability.logging import log


def _fallback_reason(e: BaseException) -> Literal["timeout", "connection"] | None:
    """
    Return 'timeout' or 'connection' if the exception is a retriable network failure
    suitable for trying the fallback provider. Return None if we should not fall back
    (e.g. auth errors, empty/partial response, validation errors).
    """
    if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return "timeout"
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, (httpx.ConnectError, httpx.NetworkError)):
        return "connection"
    msg = (getattr(e, "message", None) or str(e)).lower()
    if "timed out" in msg or "timeout" in msg or "read operation timed out" in msg:
        return "timeout"
    if "responsehandlingexception" in type(e).__name__.lower():
        if "timed out" in msg or "timeout" in msg:
            return "timeout"
    if "connect" in msg and ("refused" in msg or "error" in msg or "failed" in msg):
        return "connection"
    return None


class FallbackChainLLMClient:
    """
    Generic 2-step fallback: try primary, then secondary only on timeout or connection errors.
    Does not fall back on empty/partial responses or validation/auth errors.
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
            reason = _fallback_reason(e)
            if reason is None:
                log.warning(
                    "llm_primary_failed_no_fallback",
                    provider=self.primary_name,
                    error=primary_err,
                    error_type=type(e).__name__,
                )
                raise
            log.warning(
                "llm_fallback_primary_failed",
                provider=self.primary_name,
                reason=reason,
                error=primary_err,
                error_type=type(e).__name__,
            )
            try:
                log.info("llm_fallback_trying_secondary", provider=self.secondary_name, reason=reason)
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

