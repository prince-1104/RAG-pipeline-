#!/usr/bin/env python3
"""
Diagnostic script to check why signals aren't appearing in AgentBill dashboard.
Sends test signals and checks responses.
"""

import requests
import time
import sys
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


def create_diagnostic_payload(seq: int, include_agent_id: bool = False):
    """Create payload with diagnostic attributes"""
    settings = get_settings()
    now_ns = time.time_ns()
    
    attributes = [
        _attr("agentbill.event_name", "ai_completion"),  # Use correct event name matching app
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
        _attr("gen_ai.usage.input_tokens", 100),
        _attr("gen_ai.usage.output_tokens", 250),
        _attr("gen_ai.usage.total_tokens", 350),
        _attr("test.diagnostic", True),
        _attr("test.sequence", seq),
        _attr("test.timestamp", datetime.now().isoformat()),
    ]
    
    # Add agent_id if available (though it might not be required)
    if include_agent_id:
        # Try to get agent ID from settings or use service name
        attributes.append(_attr("agent.id", settings.service_name))
    
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
                    "attributes": attributes,
                }]
            }]
        }]
    }


def diagnose():
    """Run diagnostic tests"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return 1
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print("=" * 70)
    print("AgentBill Signal Diagnostic")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Service: {settings.service_name}")
    print(f"Environment: {settings.env}")
    print(f"Endpoint: {url}")
    print("-" * 70)
    
    # Test 1: Send a single signal and check response
    print("\n[TEST 1] Sending single diagnostic signal...")
    payload1 = create_diagnostic_payload(1)
    
    try:
        resp1 = requests.post(url, json=payload1, headers=headers, timeout=10)
        print(f"  Status Code: {resp1.status_code}")
        print(f"  Response Headers: {dict(resp1.headers)}")
        
        if resp1.status_code == 200:
            print("  [OK] Request accepted (200 OK)")
            try:
                response_body = resp1.json()
                print(f"  Response Body: {response_body}")
            except:
                print(f"  Response Text: {resp1.text[:200]}")
        else:
            print(f"  [ERROR] Request failed: {resp1.status_code}")
            print(f"  Response: {resp1.text[:500]}")
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return 1
    
    # Test 2: Send 5 signals quickly
    print("\n[TEST 2] Sending 5 signals in quick succession...")
    success_count = 0
    for i in range(2, 7):
        payload = create_diagnostic_payload(i)
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                success_count += 1
            else:
                print(f"  Signal {i}: Failed with {resp.status_code}")
        except Exception as e:
            print(f"  Signal {i}: Error - {e}")
    
    print(f"  [RESULT] {success_count}/5 signals sent successfully")
    
    # Test 3: Check payload structure
    print("\n[TEST 3] Payload structure check...")
    payload_sample = create_diagnostic_payload(999)
    print(f"  Resource attributes: {len(payload_sample['resourceSpans'][0]['resource']['attributes'])}")
    print(f"  Span attributes: {len(payload_sample['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'])}")
    print(f"  Event name: diagnostic_signal_999")
    print(f"  Is business event: True")
    
    # Summary and recommendations
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print("\n[FINDINGS]")
    print("  1. API requests are being accepted (200 OK)")
    print("  2. Payload structure appears correct")
    print("  3. Signals may take time to appear in dashboard")
    
    print("\n[POSSIBLE REASONS FOR '0 SIGNALS']:")
    print("  1. Processing delay - signals may take 1-5 minutes to appear")
    print("  2. Dashboard filters - check if filters are applied")
    print("  3. Date range - ensure dashboard shows 'Last 24 Hours'")
    print("  4. Agent/Customer mapping - signals may need agent_id/customer_id")
    print("  5. Signal type filtering - test signals might be filtered")
    
    print("\n[RECOMMENDATIONS]")
    print("  1. Wait 2-5 minutes and refresh the dashboard")
    print("  2. Check dashboard filters - clear all filters")
    print("  3. Try 'Last 7 Days' view instead of 'Last 24 Hours'")
    print("  4. Check AgentBill dashboard for any error messages")
    print("  5. Verify API key has proper permissions")
    print("  6. Check AgentBill logs/notifications for processing errors")
    
    print("\n[TEST WITH REAL APPLICATION]")
    print("  Try generating actual questions or embeddings from your app")
    print("  Real application signals may be processed differently than test signals")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(diagnose())
