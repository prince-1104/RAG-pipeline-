from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.llm.base import LLMResult
from app.llm.stub import StubLLMClient


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
    """
    if not text:
        return text
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    # Try to fix trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Try to extract JSON from markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    # Try to find JSON object/array boundaries if text is wrapped
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
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

    def __init__(self, *, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_json(self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> LLMResult:
        # This is intentionally conservative: we request JSON-only and validate downstream.
        #
        # Gemini model naming is a common integration pitfall (some projects use
        # gemini-1.5-pro vs gemini-1.5-pro-latest, etc.). If the configured model
        # 404s, we retry a small set of safe fallbacks.
        def _url(api_version: str, model: str) -> str:
            return f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"

        models_to_try = [self.model]
        # Add common fallbacks (avoid duplicates) - using newer Gemini 2.x models
        for m in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]:
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
                "maxOutputTokens": 6144,  # Balanced: enough for questions, faster than 8192
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
                "maxOutputTokens": 6144,  # Balanced: enough for questions, faster than 8192
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
        
        with httpx.Client(timeout=60) as client:
            # Try both API surfaces; some projects succeed with v1 when v1beta fails
            for api_version in ["v1beta", "v1"]:
                for model in models_to_try:
                    for body_name, body in [("json_mode", body_json_mode), ("compat_mode", body_compat_mode)]:
                        try:
                            url = _url(api_version, model)
                            last_url = url
                            resp = client.post(url, headers=headers, params=params, json=body)
                            last_status = resp.status_code
                            last_resp_text = resp.text

                            # Handle specific status codes
                            if resp.status_code == 404:
                                # Model not found - try next fallback
                                error_details.append({
                                    "model": model,
                                    "api_version": api_version,
                                    "mode": body_name,
                                    "status": 404,
                                    "error": "model_not_found"
                                })
                                continue
                            
                            if resp.status_code == 400:
                                # Bad request - might be API key issue, invalid payload, or unsupported feature
                                # Try to parse error message for better diagnostics
                                try:
                                    error_json = resp.json()
                                    error_msg = error_json.get("error", {}).get("message", resp.text[:200])
                                    # If error mentions responseSchema or schema, skip json_mode for this model
                                    if "schema" in error_msg.lower() or "responseSchema" in error_msg.lower():
                                        # Skip json_mode for this model, try compat_mode instead
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
                                # Continue trying other models/modes for 400 errors
                                # (might be model-specific issue or unsupported feature)
                                continue
                            
                            if resp.status_code == 401 or resp.status_code == 403:
                                # Authentication/authorization error - likely API key issue
                                # Don't try other models, this is a key problem
                                try:
                                    error_json = resp.json()
                                    error_msg = error_json.get("error", {}).get("message", resp.text[:200])
                                except Exception:
                                    error_msg = resp.text[:200]
                                
                                # Log detailed error
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.error(
                                    f"Gemini API authentication failed: {error_msg}",
                                    extra={
                                        "status": resp.status_code,
                                        "url": url,
                                        "model": model,
                                        "api_version": api_version
                                    }
                                )
                                
                                # Still try stub fallback rather than failing completely
                                # Break out of all loops
                                data = None  # Ensure we go to fallback
                                break

                            # For other non-2xx status codes, raise to be caught below
                            resp.raise_for_status()
                            data = resp.json()
                            # Success! Break out of all loops using a flag
                            break
                        except httpx.HTTPStatusError as e:
                            last_exception = e
                            try:
                                last_status = e.response.status_code
                                last_resp_text = e.response.text
                                # Only log non-404/400 errors (already handled above)
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
                            # Keep trying other (api_version, model, body) combos
                            continue
                        except Exception as e:
                            # For other errors (network, timeout, etc.), save and try next model
                            last_exception = e
                            error_details.append({
                                "model": model,
                                "api_version": api_version,
                                "mode": body_name,
                                "status": None,
                                "error": str(e)[:200]
                            })
                            continue
                        
                        # Check if we got successful data, break out of nested loops
                        if data is not None:
                            break
                    if data is not None:
                        break
                if data is not None:
                    break
            else:
                # All attempts failed. Fall back to deterministic stub so the service
                # remains usable in local/dev (Niyukti integration testing) even if
                # the configured Gemini model name is unavailable.
                import logging
                logger = logging.getLogger(__name__)
                
                # Build comprehensive error message
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
                
                # Log all attempted configurations for debugging
                logger.warning(
                    f"All Gemini models failed ({len(error_details)} attempts), falling back to stub. "
                    f"Last error: {error_msg}",
                    extra={"attempts": error_details, "last_status": last_status, "last_url": last_url}
                )
                
                # Fallback to stub - this should always succeed
                try:
                    logger.info("Using stub LLM as fallback")
                    return StubLLMClient().generate_json(system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema)
                except Exception as stub_error:
                    # If even stub fails, raise a clear error
                    logger.error(f"Stub LLM also failed: {stub_error}")
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
            raise RuntimeError(f"gemini_empty_response: No text content in response; upstream={upstream}")

        # #region agent log
        try:
            import json
            import os
            debug_log_path = r"d:\Hopper\NMC projects\RAG-QNA\niyukti-que-gen-Engine\.cursor\debug.log"
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"gemini.py:generate_json:text_extracted","message":"Text extracted from Gemini","data":{"text_length":len(text),"text_preview":text[:300]},"timestamp":int(__import__("time").time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion

        # Try to parse JSON, with repair attempts
        parsed = None
        parse_error = None
        
        # First attempt: direct parse
        try:
            parsed = json.loads(text)
            # #region agent log
            try:
                import json
                import os
                debug_log_path = r"d:\Hopper\NMC projects\RAG-QNA\niyukti-que-gen-Engine\.cursor\debug.log"
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"gemini.py:generate_json:direct_parse_success","message":"Direct parse succeeded","data":{},"timestamp":int(__import__("time").time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
        except json.JSONDecodeError as e:
            parse_error = e
            # #region agent log
            try:
                import json
                import os
                debug_log_path = r"d:\Hopper\NMC projects\RAG-QNA\niyukti-que-gen-Engine\.cursor\debug.log"
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1,H3","location":"gemini.py:generate_json:direct_parse_failed","message":"Direct parse failed, attempting repair","data":{"error":str(e)[:200],"error_type":type(e).__name__},"timestamp":int(__import__("time").time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
            # Second attempt: try repairing common issues
            try:
                repaired = _repair_json(text)
                # #region agent log
                try:
                    import json
                    import os
                    debug_log_path = r"d:\Hopper\NMC projects\RAG-QNA\niyukti-que-gen-Engine\.cursor\debug.log"
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"gemini.py:generate_json:repair_attempt","message":"Attempting to parse repaired JSON","data":{"repaired_length":len(repaired),"repaired_preview":repaired[-200:]},"timestamp":int(__import__("time").time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
                parsed = json.loads(repaired)
                # Log that repair worked (for monitoring)
                import logging
                logger = logging.getLogger(__name__)
                logger.debug("gemini_json_repair_succeeded", original_error=str(e)[:100])
                # #region agent log
                try:
                    import json
                    import os
                    debug_log_path = r"d:\Hopper\NMC projects\RAG-QNA\niyukti-que-gen-Engine\.cursor\debug.log"
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"gemini.py:generate_json:repair_success","message":"Repair succeeded","data":{},"timestamp":int(__import__("time").time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
            except Exception as repair_error:
                # Repair failed, will raise original error below
                # #region agent log
                try:
                    import json
                    import os
                    debug_log_path = r"d:\Hopper\NMC projects\RAG-QNA\niyukti-que-gen-Engine\.cursor\debug.log"
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1,H5","location":"gemini.py:generate_json:repair_failed","message":"Repair failed","data":{"repair_error":str(repair_error)[:200]},"timestamp":int(__import__("time").time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
                pass
        
        if parsed is None:
            # All parsing attempts failed
            upstream = {
                "url": last_url,
                "status": last_status,
                "resp_text_preview": (text or last_resp_text or "")[:1000],
            }
            error_msg = str(parse_error) if parse_error else "unknown_json_error"
            raise RuntimeError(f"gemini_invalid_json_output: {error_msg}; upstream={upstream}")
        
        return LLMResult(raw_text=text, parsed_json=parsed)


