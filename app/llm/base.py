from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResult:
    raw_text: str
    parsed_json: dict[str, Any]


class LLMClient(Protocol):
    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        ...


