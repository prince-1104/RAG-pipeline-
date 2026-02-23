from __future__ import annotations

import time
from dataclasses import dataclass
from secrets import token_hex
from typing import Any

import threading
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.observability.logging import log
from app.settings import get_settings


@dataclass
class AgentBillEvent:
    event_name: str
    provider: str
    model: str | None
    kind: str  # "completion" | "embedding"
    start_ns: int
    end_ns: int
    success: bool
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    attributes: dict[str, Any] | None = None


def _now_ns() -> int:
    return time.time_ns()


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if value is None:
        return {"key": key, "value": {"stringValue": ""}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _normalize_operation(kind: str) -> str:
    value = (kind or "").strip().lower()
    if value in {"completion", "chat", "generate", "generate_content", "text_completion"}:
        return "chat"
    if value in {"embedding", "embeddings"}:
        return "embeddings"
    return value or "chat"


def _normalize_provider(provider: str, model: str | None) -> tuple[str, dict[str, Any]]:
    provider_value = (provider or "").strip().lower()
    model_value = (model or "").strip()
    extra: dict[str, Any] = {}

    # OpenRouter is a gateway; infer provider from model prefix for better pricing match.
    if provider_value == "openrouter":
        if "/" in model_value:
            upstream = model_value.split("/", 1)[0].strip().lower()
            if upstream:
                extra["gen_ai.gateway.name"] = "openrouter"
                return upstream, extra
        return "openai", {"gen_ai.gateway.name": "openrouter"}

    return provider_value or "unknown", extra


def _build_payload(event: AgentBillEvent) -> dict[str, Any]:
    settings = get_settings()
    operation_name = _normalize_operation(event.kind)
    provider_name, provider_extra = _normalize_provider(event.provider, event.model)
    attributes = [
        _attr("agentbill.event_name", event.event_name),
        _attr("agentbill.is_business_event", True),
        _attr("agent.name", settings.service_name),
        _attr("gen_ai.system", provider_name),  # legacy compatibility
        _attr("gen_ai.provider.name", provider_name),
        _attr("gen_ai.request.model", event.model or ""),
        _attr("gen_ai.response.model", event.model or ""),
        _attr("gen_ai.operation", operation_name),  # legacy compatibility
        _attr("gen_ai.operation.name", operation_name),
        _attr("ai.success", event.success),
    ]
    if provider_extra:
        for k, v in provider_extra.items():
            attributes.append(_attr(k, v))
    if event.error:
        attributes.append(_attr("ai.error", event.error))
    if event.input_tokens is not None:
        attributes.append(_attr("ai.input_tokens", event.input_tokens))
        attributes.append(_attr("gen_ai.usage.input_tokens", event.input_tokens))
        attributes.append(_attr("gen_ai.input_tokens", event.input_tokens))
    if event.output_tokens is not None:
        attributes.append(_attr("ai.output_tokens", event.output_tokens))
        attributes.append(_attr("gen_ai.usage.output_tokens", event.output_tokens))
        attributes.append(_attr("gen_ai.output_tokens", event.output_tokens))
    if event.total_tokens is not None:
        attributes.append(_attr("ai.total_tokens", event.total_tokens))
        attributes.append(_attr("gen_ai.usage.total_tokens", event.total_tokens))
        attributes.append(_attr("gen_ai.total_tokens", event.total_tokens))
    if event.attributes:
        for k, v in event.attributes.items():
            attributes.append(_attr(k, v))

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attr("service.name", settings.service_name),
                        _attr("service.version", settings.env),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "agentbill.signals"},
                        "spans": [
                            {
                                "traceId": token_hex(16),
                                "spanId": token_hex(8),
                                "name": f"gen_ai.{operation_name}",
                                "kind": 3,
                                "startTimeUnixNano": str(event.start_ns),
                                "endTimeUnixNano": str(event.end_ns),
                                "status": {"code": 1 if event.success else 2},
                                "attributes": attributes,
                            }
                        ],
                    }
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Persistent HTTP client + fire-and-forget background dispatch
# ---------------------------------------------------------------------------
_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = threading.Lock()
_MISSING_API_KEY_WARNED = False
_DISABLED_WARNED = False

# Event tracking counters for diagnostics
_EVENT_COUNTERS = {
    "events_created": 0,
    "events_submitted": 0,
    "events_sent_success": 0,
    "events_sent_failed": 0,
    "events_skipped_no_key": 0,
    "events_skipped_disabled": 0,
}
_COUNTERS_LOCK = threading.Lock()


def _get_http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
                _HTTP_CLIENT = httpx.Client(
                    timeout=httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=3.0),
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                )
    return _HTTP_CLIENT


_BG_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentbill")


def _send_event_sync(payload: dict, headers: dict, endpoint: str) -> None:
    try:
        client = _get_http_client()
        resp = client.post(endpoint, headers=headers, json=payload)
        if not resp.is_success:
            with _COUNTERS_LOCK:
                _EVENT_COUNTERS["events_sent_failed"] += 1
            log.warning(
                "agentbill_track_failed",
                status_code=resp.status_code,
                body=(resp.text or "")[:200],
                endpoint=endpoint,
            )
        else:
            with _COUNTERS_LOCK:
                _EVENT_COUNTERS["events_sent_success"] += 1
    except Exception as exc:  # noqa: BLE001
        with _COUNTERS_LOCK:
            _EVENT_COUNTERS["events_sent_failed"] += 1
        log.warning(
            "agentbill_track_error",
            error=str(exc)[:200],
            error_type=type(exc).__name__,
            endpoint=endpoint,
        )


def track_agentbill_event(event: AgentBillEvent) -> None:
    global _MISSING_API_KEY_WARNED, _DISABLED_WARNED
    settings = get_settings()
    
    with _COUNTERS_LOCK:
        _EVENT_COUNTERS["events_created"] += 1
    
    # Check if AgentBill is enabled
    if not getattr(settings, "agentbill_enabled", True):
        with _COUNTERS_LOCK:
            _EVENT_COUNTERS["events_skipped_disabled"] += 1
        if not _DISABLED_WARNED:
            log.warning(
                "agentbill_disabled",
                message="AgentBill tracking is disabled. Set AGENTBILL_ENABLED=true to enable.",
            )
            _DISABLED_WARNED = True
        return
    
    # Check if API key is configured
    api_key = getattr(settings, "agentbill_api_key", None)
    if not api_key or not str(api_key).strip():
        with _COUNTERS_LOCK:
            _EVENT_COUNTERS["events_skipped_no_key"] += 1
        if not _MISSING_API_KEY_WARNED:
            log.warning(
                "agentbill_missing_api_key",
                message="AgentBill API key is not configured. Set AGENTBILL_API_KEY environment variable to enable tracking.",
            )
            _MISSING_API_KEY_WARNED = True
        return

    endpoint = getattr(
        settings, "agentbill_endpoint", "https://api.agentbill.io/functions/v1/otel-collector"
    )

    payload = _build_payload(event)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": str(api_key).strip(),
    }

    # Fire-and-forget: never block the worker/generation thread.
    try:
        with _COUNTERS_LOCK:
            _EVENT_COUNTERS["events_submitted"] += 1
        _BG_EXECUTOR.submit(_send_event_sync, payload, headers, endpoint)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "agentbill_executor_submit_failed",
            error=str(exc)[:200],
            error_type=type(exc).__name__,
        )


def start_event(*, event_name: str, provider: str, model: str | None, kind: str) -> AgentBillEvent:
    now = _now_ns()
    return AgentBillEvent(
        event_name=event_name,
        provider=provider,
        model=model,
        kind=kind,
        start_ns=now,
        end_ns=now,
        success=False,
    )


def finalize_event(
    event: AgentBillEvent,
    *,
    success: bool,
    error: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    attributes: dict[str, Any] | None = None,
) -> None:
    event.end_ns = _now_ns()
    event.success = success
    event.error = error
    event.input_tokens = input_tokens
    event.output_tokens = output_tokens
    event.total_tokens = total_tokens
    event.attributes = attributes
    track_agentbill_event(event)


def get_agentbill_status() -> dict[str, Any]:
    """
    Diagnostic function to check AgentBill configuration status.
    Returns a dictionary with configuration details (without exposing sensitive data).
    """
    settings = get_settings()
    enabled = getattr(settings, "agentbill_enabled", True)
    api_key = getattr(settings, "agentbill_api_key", None)
    endpoint = getattr(
        settings, "agentbill_endpoint", "https://api.agentbill.io/functions/v1/otel-collector"
    )
    
    status = {
        "enabled": enabled,
        "api_key_configured": bool(api_key and str(api_key).strip()),
        "endpoint": endpoint,
        "service_name": settings.service_name,
        "env": settings.env,
    }
    
    # Check executor status
    try:
        executor = _BG_EXECUTOR
        status["executor_running"] = executor is not None
        if executor:
            status["executor_shutdown"] = executor._shutdown
    except Exception:
        status["executor_running"] = False
        status["executor_error"] = "Unable to check executor status"
    
    # Check HTTP client status
    try:
        client = _HTTP_CLIENT
        status["http_client_initialized"] = client is not None
        if client:
            status["http_client_closed"] = client.is_closed
    except Exception:
        status["http_client_initialized"] = False
        status["http_client_error"] = "Unable to check HTTP client status"
    
    # Add event counters for diagnostics
    with _COUNTERS_LOCK:
        status["event_counters"] = _EVENT_COUNTERS.copy()
    
    return status
