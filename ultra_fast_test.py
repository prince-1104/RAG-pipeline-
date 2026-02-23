#!/usr/bin/env python3
"""
Ultra-fast load test - simplest version matching user's example.
Sends 100 requests as fast as possible using threading.
"""

import requests
import time
import sys
from secrets import token_hex
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.settings import get_settings


def _attr(key: str, value):
    """Create OTEL attribute"""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def create_payload(seq: int):
    """Create valid OTEL span payload"""
    settings = get_settings()
    now_ns = time.time_ns()
    
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    _attr("service.name", settings.service_name),
                    _attr("service.version", settings.env),
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "agentbill.signals"},
                "spans": [{
                    "traceId": token_hex(16),
                    "spanId": token_hex(8),
                    "name": "gen_ai.chat",
                    "kind": 3,
                    "startTimeUnixNano": str(now_ns - 1000000000),
                    "endTimeUnixNano": str(now_ns),
                    "status": {"code": 1},
                    "attributes": [
                        _attr("agentbill.event_name", "ai_completion"),  # Use correct event name
                        _attr("agentbill.is_business_event", True),
                        _attr("agent.name", settings.service_name),
                        _attr("gen_ai.provider.name", "openai"),
                        _attr("gen_ai.request.model", "gpt-4o-mini"),
                        _attr("gen_ai.response.model", "gpt-4o-mini"),
                        _attr("gen_ai.operation.name", "chat"),
                        _attr("ai.success", True),
                        _attr("ai.input_tokens", 100),
                        _attr("ai.output_tokens", 250),
                        _attr("ai.total_tokens", 350),
                        _attr("test.load_test", True),
                        _attr("test.sequence", seq),
                    ],
                }]
            }]
        }]
    }


def send_one(url, headers, seq):
    """Send single request with session"""
    payload = create_payload(seq)
    session = requests.Session()
    session.headers.update(headers)
    try:
        resp = session.post(url, json=payload, timeout=10)
        return (seq, resp.status_code == 200)
    except:
        return (seq, False)
    finally:
        session.close()


# Get settings
settings = get_settings()
api_key = getattr(settings, "agentbill_api_key", None)

if not api_key:
    print("[ERROR] AGENTBILL_API_KEY not set!")
    sys.exit(1)

url = "https://api.agentbill.io/functions/v1/otel-collector"
headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

print(f"[START] Ultra-fast 100 requests test")
print(f"Time: {datetime.now().strftime('%H:%M:%S')}")

start = time.time()
success = 0
failed = 0

# Send all 100 requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_one, url, headers, i+1) for i in range(100)]
    
    for future in as_completed(futures):
        seq, ok = future.result()
        if ok:
            success += 1
        else:
            failed += 1

elapsed = time.time() - start

print(f"\n[COMPLETE] Time: {elapsed:.2f}s | Success: {success} | Failed: {failed}")
print(f"Rate: {success/elapsed*60:.1f} req/min")

if elapsed < 60:
    print(f"[SUCCESS] Completed in {elapsed:.1f}s (<60s limit)")
else:
    print(f"[WARNING] Took {elapsed:.1f}s (exceeded 60s limit)")
