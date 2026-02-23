#!/usr/bin/env python3
"""
Fastest load test script for AgentBill API.
Sends signals as fast as possible to test rate limits.
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


def send_request(url: str, headers: dict, seq: int):
    """Send a single request with session reuse"""
    payload = create_payload(seq)
    # Use session per thread for connection pooling
    session = requests.Session()
    session.headers.update(headers)
    try:
        resp = session.post(url, json=payload, timeout=10)
        return (seq, resp.status_code == 200, resp.status_code)
    except Exception as e:
        return (seq, False, str(e)[:100])
    finally:
        session.close()


def test_100_per_minute_fast():
    """Test 100 requests/minute - fastest version with threading"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print(f"[START] Fast 100 requests/minute test")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("[INFO] Using threading for maximum speed")
    
    start = time.time()
    success = 0
    failed = 0
    
    # Use thread pool for concurrent requests - use more workers for speed
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(send_request, url, headers, i+1) for i in range(100)]
        
        completed = 0
        for future in as_completed(futures):
            seq, ok, status = future.result()
            completed += 1
            if ok:
                success += 1
            else:
                failed += 1
                if failed <= 5:  # Log first few failures
                    print(f"  [{seq}] FAILED: {status}")
            
            if completed % 10 == 0:
                elapsed = time.time() - start
                print(f"  [{completed}/100] OK: {success}, Failed: {failed}, Time: {elapsed:.1f}s")
    
    elapsed = time.time() - start
    print(f"\n[COMPLETE] Time: {elapsed:.2f}s | Success: {success} | Failed: {failed}")
    print(f"Rate: {success/elapsed*60:.1f} req/min")
    
    if elapsed < 60:
        print(f"[SUCCESS] Completed in {elapsed:.1f}s (<60s)")
    else:
        print(f"[WARNING] Took {elapsed:.1f}s (exceeded 60s)")


def test_100_per_minute_sequential():
    """Test 100 requests/minute - sequential with connection pooling"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print(f"[START] Sequential 100 requests/minute test")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("[INFO] Using session with connection pooling for better performance")
    
    success = 0
    failed = 0
    start = time.time()
    
    # Use session for connection pooling and reuse
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        # Simple loop - using session for connection reuse
        for i in range(100):
            payload = create_payload(i + 1)
            try:
                resp = session.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    success += 1
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"  [{i+1}] FAILED: {resp.status_code} - {resp.text[:100]}")
            except requests.exceptions.ConnectionError as e:
                failed += 1
                if failed <= 5:
                    print(f"  [{i+1}] CONNECTION ERROR: {str(e)}")
            except requests.exceptions.Timeout as e:
                failed += 1
                if failed <= 5:
                    print(f"  [{i+1}] TIMEOUT: {str(e)}")
            except Exception as e:
                failed += 1
                if failed <= 5:
                    error_msg = str(e)
                    print(f"  [{i+1}] ERROR: {error_msg[:100]}")
            
            # Progress every 10
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/100] OK: {success}, Failed: {failed}, Time: {elapsed:.1f}s")
    finally:
        session.close()
    
    elapsed = time.time() - start
    print(f"\n[COMPLETE] Time: {elapsed:.2f}s | Success: {success} | Failed: {failed}")
    if success > 0:
        print(f"Rate: {success/elapsed*60:.1f} req/min")
    
    if elapsed < 60 and success == 100:
        print(f"[SUCCESS] Completed in {elapsed:.1f}s (<60s)")
    elif elapsed < 60:
        print(f"[PARTIAL] Completed in {elapsed:.1f}s but {failed} requests failed")
    else:
        print(f"[WARNING] Took {elapsed:.1f}s (exceeded 60s)")
    
    if failed > 0:
        print(f"\n[NOTE] {failed} requests failed. Consider using threaded version:")
        print(f"  python fast_load_test.py  (uses threading, faster and more reliable)")


def test_10000_per_day():
    """Test 10,000 requests/day - sends in batches with proper rate limiting"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print(f"\n[START] 10,000 requests/day test")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("[INFO] Sending 100 batches of 100 requests each")
    print("[INFO] 60 second delay between batches to respect 100/min limit")
    print("[INFO] Estimated time: ~100 minutes")
    
    success = 0
    failed = 0
    start = time.time()
    seq = 1
    
    # Send 100 batches of 100 requests each
    for batch in range(100):
        batch_start = time.time()
        batch_success = 0
        batch_failed = 0
        
        # Send batch of 100 requests quickly
        for i in range(100):
            payload = create_payload(seq)
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    success += 1
                    batch_success += 1
                else:
                    failed += 1
                    batch_failed += 1
            except Exception as e:
                failed += 1
                batch_failed += 1
            
            seq += 1
        
        batch_time = time.time() - batch_start
        total_time = time.time() - start
        
        print(f"  Batch {batch+1}/100: {batch_success} OK, {batch_failed} failed | "
              f"Total: {success}/{seq-1} | Time: {total_time/60:.1f} min")
        
        # Wait 60s between batches (except last batch)
        if batch < 99:
            wait = 60 - batch_time
            if wait > 0:
                time.sleep(wait)
    
    elapsed = time.time() - start
    print(f"\n[COMPLETE] Time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
    print(f"Success: {success}/10000 | Failed: {failed}")
    print(f"Success rate: {success/10000*100:.1f}%")


if __name__ == "__main__":
    print("=" * 70)
    print("AgentBill Fast Load Test")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "100-fast":
            test_100_per_minute_fast()
        elif test_type == "100":
            print("[WARNING] Sequential version is slow and may hit connection limits.")
            print("[RECOMMEND] Use default (threaded) or '100-fast' for better performance.\n")
            test_100_per_minute_sequential()
        elif test_type == "10000":
            test_10000_per_day()
        else:
            print(f"Unknown test: {test_type}")
            print("Usage:")
            print("  python fast_load_test.py           # Threaded (fastest, recommended)")
            print("  python fast_load_test.py 100-fast   # Threaded (explicit)")
            print("  python fast_load_test.py 100        # Sequential (slow, may fail)")
            print("  python fast_load_test.py 10000      # 10k/day test")
    else:
        # Default: run fast threaded test
        print("[INFO] Running threaded version (fastest and most reliable)")
        print("[INFO] Use 'python fast_load_test.py 100' for sequential version\n")
        test_100_per_minute_fast()
