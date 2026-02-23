#!/usr/bin/env python3
"""
Verify if signals are being accepted and check API responses.
"""

import requests
import time
import sys
import json
from secrets import token_hex
from pathlib import Path
from datetime import datetime

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


def create_test_payload():
    """Create a single test payload"""
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
                        _attr("agentbill.event_name", "ai_completion"),
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
                    ],
                }]
            }]
        }]
    }


def verify():
    """Verify signal acceptance"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return 1
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print("=" * 70)
    print("Signal Verification")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Endpoint: {url}")
    print("-" * 70)
    
    # Send a test signal and capture full response
    print("\n[TEST] Sending test signal...")
    payload = create_test_payload()
    
    print(f"\n[PAYLOAD] First 500 chars:")
    payload_str = json.dumps(payload, indent=2)
    print(payload_str[:500] + "...")
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        print(f"\n[RESPONSE] Status: {resp.status_code}")
        print(f"[RESPONSE] Headers: {dict(resp.headers)}")
        
        try:
            response_data = resp.json()
            print(f"\n[RESPONSE] Body:")
            print(json.dumps(response_data, indent=2))
            
            if response_data.get("queued"):
                print("\n[OK] Signal queued successfully")
                print(f"  WAL ID: {response_data.get('wal_id', 'N/A')}")
                print(f"  Message: {response_data.get('message', 'N/A')}")
            else:
                print("\n[WARNING] Response doesn't indicate queuing")
                
        except Exception as e:
            print(f"\n[RESPONSE] Text: {resp.text[:500]}")
            print(f"[ERROR] Failed to parse JSON: {e}")
            
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print("\n[FINDINGS]")
    print("1. Check the response above - does it show 'queued: True'?")
    print("2. If queued, signals may take 5-15 minutes to appear")
    print("3. Check AgentBill dashboard after waiting")
    print("\n[NOTE]")
    print("Signals are queued asynchronously. The dashboard may take")
    print("several minutes to update, especially during high load.")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(verify())
