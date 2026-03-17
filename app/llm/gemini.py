from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.llm.base import LLMResult
from app.llm.stub import StubLLMClient
from app.observability.agentbill import finalize_event, start_event
from app.observability.logging import log


def _repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues from LLM outputs.
    Handles:
    - Unescaped quotes in strings
    - Trailing commas
    - Missing closing braces/brackets
    - Control characters
    - JSON wrapped in markdown code blocks
    - Unterminated strings (truncated responses) - improved detection
    - Single quotes instead of double quotes
    - Unquoted property names
    """
    if not text:
        return text
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    # Try to extract JSON from markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    # Try to find JSON object/array boundaries if text is wrapped
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    # Fix unquoted property names (e.g., key: value -> "key": value)
    # Pattern: word character sequence followed by colon (not inside quotes)
    # Only match at the start of a line or after whitespace/comma/brace
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Fix single quotes to double quotes (but be careful with apostrophes in strings)
    # Only replace single quotes that are clearly property delimiters
    # Pattern: 'key': or 'key', or : 'value' or , 'value'
    text = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', text)  # 'key': -> "key":
    text = re.sub(r"(\s*:\s*)'([^']*)'", r'\1"\2"', text)  # : 'value' -> : "value"
    
    # Try to fix trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Improved unterminated string detection and repair
    # First pass: track string boundaries to detect unterminated strings
    result = []
    in_string = False
    escape_next = False
    string_start_idx = -1
    i = 0
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            result.append(char)
            if not in_string:
                string_start_idx = len(result) - 1
            in_string = not in_string
            if not in_string:
                string_start_idx = -1
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    text = ''.join(result)
    
    # Second pass: if we're still in a string, try to find where it should end
    # Look for patterns that suggest the string was truncated
    if in_string:
        # Search from the string content (after the opening quote) for closing patterns
        # string_start_idx points to the opening quote, so content starts at string_start_idx + 1
        if string_start_idx >= 0:
            # Search from after the opening quote
            content_start = string_start_idx + 1
            remaining_text = text[content_start:]
            
            # Pattern 1: Content, then newline, then quote (next field starting)
            match = re.search(r'([^"]*?)\n\s*"', remaining_text)
            if match:
                # Close string before the newline+quote
                content = match.group(1)
                close_pos = content_start + len(content)
                text = text[:close_pos] + '"' + text[close_pos:]
                in_string = False
            else:
                # Pattern 2: Content, then newline, then closing brace/bracket
                match = re.search(r'([^"]*?)\n\s*[}\]\]', remaining_text)
                if match:
                    content = match.group(1)
                    close_pos = content_start + len(content)
                    text = text[:close_pos] + '"' + text[close_pos:]
                    in_string = False
                else:
                    # Pattern 3: Content, then whitespace, then comma/colon/closing delimiter
                    match = re.search(r'([^"]+?)(\s+)([,:}\]\]])', remaining_text)
                    if match:
                        content = match.group(1)
                        # Only if content is substantial (more than a few chars)
                        if len(content.strip()) > 3:
                            close_pos = content_start + len(content)
                            text = text[:close_pos] + '"' + text[close_pos:]
                            in_string = False
                        else:
                            # Fallback: just close at end
                            text += '"'
                            in_string = False
                    else:
                        # Fallback: close at end
                        text += '"'
                        in_string = False
        else:
            # String started but we don't know where - just close at end
            text += '"'
    
    # Also try to close any unclosed braces/brackets at the end
    # But be smart about it - don't close if we're inside a string
    # Count braces/brackets only outside strings
    brace_depth = 0
    bracket_depth = 0
    in_string_check = False
    escape_next_check = False
    
    for char in text:
        if escape_next_check:
            escape_next_check = False
            continue
        if char == '\\':
            escape_next_check = True
            continue
        if char == '"':
            in_string_check = not in_string_check
            continue
        if not in_string_check:
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
    
    if brace_depth > 0:
        text += '}' * brace_depth
    if bracket_depth > 0:
        text += ']' * bracket_depth
    
    return text.strip()


class GeminiClient:
    """
    Minimal Gemini JSON generation client.
    Note: exact REST surface varies by Google SDK/version; we isolate it here to stay provider-agnostic.
    """

    def __init__(self, *, api_key: str, model: str, timeout_s: float = 300.0):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        # This is intentionally conservative: we request JSON-only and validate downstream.
        #
        # Gemini model naming is a common integration pitfall. Use only v1beta and
        # model names supported by Google Generative Language API. If the configured
        # model 404s, we retry a small set of safe fallbacks.
        log.info("llm_request", provider="gemini", model=self.model)

        event = start_event(
            event_name="ai_completion",
            provider="gemini",
            model=self.model,
            kind="completion",
        )
        # Use v1beta only; v1 does not support all models (e.g. gemini-1.5-flash).
        api_version = "v1beta"
        def _url(ver: str, model: str) -> str:
            return f"https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent"

        models_to_try = [self.model]
        # Valid fallbacks per Google API (no -latest suffix, no 2.5)
        for m in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
            if m not in models_to_try:
                models_to_try.append(m)

        headers = {"Content-Type": "application/json"}

        # Gemini REST compatibility varies a bit by model/version.
        # We'll try two payload variants:
        # - "json_mode": uses systemInstruction + responseMimeType=application/json
        # - "compat_mode": merges system+user into a single user message, no systemInstruction/responseMimeType
        # Note: responseSchema is not used as Gemini 2.5-flash doesn't support it reliably
        body_json_mode = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "maxOutputTokens": 16384,
                "responseMimeType": "application/json",
            },
        }
        body_compat_mode = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "maxOutputTokens": 16384,
            },
        }
        # If the model supports schema constraints, you can wire it here later;
        # we still enforce with jsonschema locally.
        params = {"key": self.api_key}
        last_resp_text: str | None = None
        last_exception: Exception | None = None
        last_status: int | None = None
        last_url: str | None = None
        error_details: list[dict[str, Any]] = []
        data: dict[str, Any] | None = None
        abort_auth = False

        with httpx.Client(timeout=self.timeout_s) as client:
            for model in models_to_try:
                if abort_auth:
                    break
                for body_name, body in [("json_mode", body_json_mode), ("compat_mode", body_compat_mode)]:
                    try:
                        url = _url(api_version, model)
                        last_url = url
                        resp = client.post(url, headers=headers, params=params, json=body)
                        last_status = resp.status_code
                        last_resp_text = resp.text

                        # Handle specific status codes
                        if resp.status_code == 404:
                            error_details.append({
                                "model": model,
                                "api_version": api_version,
                                "mode": body_name,
                                "status": 404,
                                "error": "model_not_found"
                            })
                            continue

                        if resp.status_code == 400:
                            try:
                                error_json = resp.json()
                                error_msg = error_json.get("error", {}).get("message", resp.text[:200])
                                if "schema" in error_msg.lower() or "responseSchema" in error_msg.lower():
                                    if body_name == "json_mode":
                                        continue
                            except Exception:
                                error_msg = resp.text[:200]

                            error_details.append({
                                "model": model,
                                "api_version": api_version,
                                "mode": body_name,
                                "status": 400,
                                "error": error_msg
                            })
                            continue

                        if resp.status_code == 401 or resp.status_code == 403:
                            try:
                                error_json = resp.json()
                                error_msg = error_json.get("error", {}).get("message", resp.text[:200])
                            except Exception:
                                error_msg = resp.text[:200]
                            log.error(
                                "gemini_auth_failed",
                                error=error_msg,
                                status=resp.status_code,
                                url=url,
                                model=model,
                                api_version=api_version,
                            )
                            data = None
                            abort_auth = True
                            break

                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except httpx.HTTPStatusError as e:
                        last_exception = e
                        try:
                            last_status = e.response.status_code
                            last_resp_text = e.response.text
                            if e.response.status_code not in (404, 400, 401, 403):
                                error_details.append({
                                    "model": model,
                                    "api_version": api_version,
                                    "mode": body_name,
                                    "status": e.response.status_code,
                                    "error": e.response.text[:200]
                                })
                        except Exception:
                            pass
                        continue
                    except Exception as e:
                        last_exception = e
                        error_details.append({
                            "model": model,
                            "api_version": api_version,
                            "mode": body_name,
                            "status": None,
                            "error": str(e)[:200]
                        })
                        continue

                    if data is not None:
                        break
                if data is not None or abort_auth:
                    break
            else:
                # All attempts failed. Fall back to deterministic stub so the service
                # remains usable in local/dev (Niyukti integration testing) even if
                # the configured Gemini model name is unavailable.
                error_summary = []
                if last_status:
                    error_summary.append(f"HTTP {last_status}")
                if last_url:
                    error_summary.append(f"URL: {last_url}")
                if last_resp_text:
                    try:
                        error_json = json.loads(last_resp_text)
                        if "error" in error_json:
                            error_summary.append(f"Error: {error_json['error'].get('message', 'Unknown error')}")
                    except Exception:
                        error_summary.append(f"Response: {last_resp_text[:200]}")
                
                error_msg = "; ".join(error_summary) if error_summary else str(last_exception)[:500]
                log.warning(
                    "gemini_all_models_failed_fallback_stub",
                    attempts=len(error_details),
                    last_status=last_status,
                    last_url=last_url,
                    last_error=error_msg[:500],
                )
                try:
                    log.info("llm_fallback_stub_used", provider="gemini")
                    finalize_event(
                        event,
                        success=False,
                        error="gemini_fallback_stub",
                    )
                    return StubLLMClient().generate_json(system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema)
                except Exception as stub_error:
                    log.error("llm_stub_fallback_failed", error=str(stub_error))
                    finalize_event(
                        event,
                        success=False,
                        error="gemini_stub_failed",
                    )
                    raise RuntimeError(
                        f"LLM generation failed: {error_msg}. "
                        f"Stub fallback also failed: {stub_error}"
                    )

        # Best-effort extraction of response text
        text = ""
        try:
            if data and "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                    text = candidate["content"]["parts"][0].get("text", "")
            
            # If text is still empty, try alternative extraction
            if not text and data:
                # Try to find text in other possible locations
                if "candidates" in data:
                    for cand in data["candidates"]:
                        if "text" in cand:
                            text = cand["text"]
                            break
                        if "content" in cand:
                            content = cand["content"]
                            if isinstance(content, str):
                                text = content
                                break
                            if isinstance(content, dict) and "text" in content:
                                text = content["text"]
                                break
            
            # Last resort: use raw response text
            if not text:
                text = last_resp_text or ""
                
        except Exception:  # noqa: BLE001
            # Include last response text for debugging if extraction fails
            text = last_resp_text or ""
        
        # Validate we have text
        if not text or not text.strip():
            upstream = {
                "url": last_url,
                "status": last_status,
                "resp_text_preview": (last_resp_text or "")[:1000],
                "data_keys": list(data.keys()) if data else [],
            }
            finalize_event(
                event,
                success=False,
                error="gemini_empty_response",
            )
            raise RuntimeError(f"gemini_empty_response: No text content in response; upstream={upstream}")

        # Try to parse JSON, with repair attempts
        parsed = None
        parse_error = None
        
        # First attempt: direct parse
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            parse_error = e
            try:
                repaired = _repair_json(text)
                parsed = json.loads(repaired)
                log.debug("gemini_json_repair_succeeded", original_error=str(e)[:100])
            except Exception as repair_error:
                # Repair failed, will raise original error below
                pass
        
        if parsed is None:
            # All parsing attempts failed
            upstream = {
                "url": last_url,
                "status": last_status,
                "resp_text_preview": (text or last_resp_text or "")[:1000],
            }
            error_msg = str(parse_error) if parse_error else "unknown_json_error"
            finalize_event(
                event,
                success=False,
                error="gemini_invalid_json",
            )
            raise RuntimeError(f"gemini_invalid_json_output: {error_msg}; upstream={upstream}")

        usage = None
        if isinstance(data, dict):
            usage = data.get("usageMetadata") or data.get("usage")
        input_tokens = None
        output_tokens = None
        total_tokens = None
        if isinstance(usage, dict):
            input_tokens = usage.get("promptTokenCount") or usage.get("prompt_tokens")
            output_tokens = usage.get("candidatesTokenCount") or usage.get("completion_tokens")
            total_tokens = usage.get("totalTokenCount") or usage.get("total_tokens")

        finalize_event(
            event,
            success=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        return LLMResult(raw_text=text, parsed_json=parsed)


