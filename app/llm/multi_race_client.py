from __future__ import annotations

import concurrent.futures
from typing import Any

from app.llm.base import LLMClient, LLMResult
from app.observability.logging import log


class MultiRaceLLMClient:
    """
    Calls multiple LLM providers in parallel and returns the first successful response.
    Useful when parallel requests are acceptable and you want lowest latency.
    """

    def __init__(self, *, clients: dict[str, LLMClient], max_workers: int | None = None):
        if not clients:
            raise ValueError("clients_required")
        self.clients = clients
        self.max_workers = max_workers or len(clients)

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[concurrent.futures.Future[LLMResult], str] = {}
            for name, client in self.clients.items():
                futures[executor.submit(client.generate_json, system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema)] = name

            errors: dict[str, str] = {}
            for future in concurrent.futures.as_completed(futures.keys()):
                provider_name = futures[future]
                try:
                    result = future.result()
                    log.info("llm_race_winner", provider=provider_name)
                    # Best effort cancellation of the others
                    for other_future, other_name in futures.items():
                        if other_name != provider_name:
                            other_future.cancel()
                    return result
                except Exception as e:  # noqa: BLE001
                    errors[provider_name] = str(e)
                    log.warning("llm_race_provider_failed", provider=provider_name, error=str(e)[:200])
                    continue

        error_summary = "; ".join([f"{p}: {e[:100]}" for p, e in errors.items()])
        raise RuntimeError(f"all_llm_providers_failed: {error_summary}")

