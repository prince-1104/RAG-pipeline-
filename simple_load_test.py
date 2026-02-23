#!/usr/bin/env python3
"""
Simple and fast load test script for AgentBill API.
Sends dummy signals in a loop - fastest implementation.
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


def test_100_per_minute():
    """Test 100 requests/minute - fastest version"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print(f"[START] Sending 100 requests/minute test")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    success = 0
    failed = 0
    start = time.time()
    
    for i in range(100):
        payload = create_payload(i + 1)
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                success += 1
            else:
                failed += 1
                print(f"  [{i+1}] FAILED: {resp.status_code}")
        except Exception as e:
            failed += 1
            print(f"  [{i+1}] ERROR: {str(e)[:50]}")
        
        # Progress every 10 requests
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/100] OK: {success}, Failed: {failed}, Time: {elapsed:.1f}s")
    
    elapsed = time.time() - start
    print(f"\n[COMPLETE] Time: {elapsed:.2f}s | Success: {success} | Failed: {failed}")
    print(f"Rate: {success/elapsed*60:.1f} req/min")
    
    if elapsed < 60:
        print(f"[SUCCESS] Completed in {elapsed:.1f}s (<60s)")
    else:
        print(f"[WARNING] Took {elapsed:.1f}s (exceeded 60s)")


def test_10000_per_day():
    """Test 10,000 requests/day - sends in batches"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print(f"\n[START] Sending 10,000 requests/day test")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("[INFO] This will take ~100 minutes (100 batches x 60s)")
    
    success = 0
    failed = 0
    start = time.time()
    seq = 1
    
    # Send 100 batches of 100 requests each, with 60s delay between batches
    for batch in range(100):
        batch_start = time.time()
        batch_success = 0
        
        # Send batch of 100
        for i in range(100):
            payload = create_payload(seq)
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    success += 1
                    batch_success += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
            
            seq += 1
            time.sleep(0.01)  # Small delay
        
        batch_time = time.time() - batch_start
        total_time = time.time() - start
        
        print(f"  Batch {batch+1}/100: {batch_success} OK | "
              f"Total: {success}/{seq-1} | Time: {total_time/60:.1f} min")
        
        # Wait 60s between batches (except last)
        if batch < 99:
            wait = 60 - batch_time
            if wait > 0:
                time.sleep(wait)
    
    elapsed = time.time() - start
    print(f"\n[COMPLETE] Time: {elapsed/60:.1f} min | Success: {success} | Failed: {failed}")
    print(f"Success rate: {success/10000*100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("AgentBill Simple Load Test")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "100":
            test_100_per_minute()
        elif test_type == "10000":
            test_10000_per_day()
        else:
            print(f"Unknown test: {test_type}")
            print("Usage: python simple_load_test.py [100|10000]")
    else:
        # Default: run 100/min test
        test_100_per_minute()
        
        # Ask about 10k test
        try:
            run_10k = input("\nRun 10,000/day test? (yes/no): ").strip().lower()
            if run_10k == "yes":
                test_10000_per_day()
        except KeyboardInterrupt:
            print("\nCancelled.")
