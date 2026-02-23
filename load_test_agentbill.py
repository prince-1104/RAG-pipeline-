#!/usr/bin/env python3
"""
Simple load test script for AgentBill API.
Tests rate limits: 100 requests/minute and 10,000 requests/day.
"""

import requests
import time
import sys
from datetime import datetime
from secrets import token_hex
from pathlib import Path

# Add project root to path to access settings
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.settings import get_settings


def _attr(key: str, value) -> dict:
    """Helper to create OTEL attribute"""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if value is None:
        return {"key": key, "value": {"stringValue": ""}}
    return {"key": key, "value": {"stringValue": str(value)}}


def create_dummy_payload(sequence: int) -> dict:
    """Create a valid OTEL span payload for testing"""
    settings = get_settings()
    now_ns = time.time_ns()
    
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
                                "name": "gen_ai.chat",
                                "kind": 3,
                                "startTimeUnixNano": str(now_ns - 1000000000),  # 1 second ago
                                "endTimeUnixNano": str(now_ns),
                                "status": {"code": 1},
                                "attributes": [
                                    _attr("agentbill.event_name", f"load_test_signal_{sequence}"),
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
                                    _attr("test.load_test", True),
                                    _attr("test.sequence", sequence),
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def load_test_100_per_minute(api_key: str):
    """Test 100 requests/minute rate limit"""
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print("\n" + "=" * 70)
    print("LOAD TEST: 100 requests/minute")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: 100 requests in <60 seconds")
    print("-" * 70)
    
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    for i in range(100):
        payload = create_dummy_payload(i + 1)
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                success_count += 1
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 60
                    print(f"  [{i+1:3d}/100] OK | Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} req/min")
            else:
                failed_count += 1
                print(f"  [{i+1:3d}/100] FAILED | Status: {response.status_code} | {response.text[:100]}")
        except Exception as e:
            failed_count += 1
            print(f"  [{i+1:3d}/100] ERROR | {str(e)[:100]}")
        
        # Small delay to avoid overwhelming (but still hit 100/min)
        time.sleep(0.01)
    
    elapsed_time = time.time() - start_time
    print("-" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Success: {success_count}/100")
    print(f"Failed: {failed_count}/100")
    print(f"Actual rate: {success_count / elapsed_time * 60:.1f} requests/minute")
    
    if elapsed_time < 60:
        print(f"[SUCCESS] Completed 100 requests in {elapsed_time:.1f}s (<60s limit)")
    else:
        print(f"[WARNING] Took {elapsed_time:.1f}s (exceeded 60s limit)")
    
    return success_count, failed_count, elapsed_time


def load_test_10000_per_day(api_key: str, batch_size: int = 100):
    """Test 10,000 requests/day rate limit (sends in batches)"""
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    total_requests = 10000
    print("\n" + "=" * 70)
    print("LOAD TEST: 10,000 requests/day")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: {total_requests} requests")
    print(f"Batch size: {batch_size} requests per batch")
    print("-" * 70)
    
    success_count = 0
    failed_count = 0
    start_time = time.time()
    sequence = 1
    
    # Calculate delay to stay under 100/min rate limit
    # Send batches of 100 with 60 second delays between batches
    batches = (total_requests + batch_size - 1) // batch_size
    
    for batch_num in range(batches):
        batch_start = time.time()
        batch_success = 0
        batch_failed = 0
        
        # Send batch
        for i in range(batch_size):
            if sequence > total_requests:
                break
                
            payload = create_dummy_payload(sequence)
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    success_count += 1
                    batch_success += 1
                else:
                    failed_count += 1
                    batch_failed += 1
                    if batch_failed <= 3:  # Log first few failures
                        print(f"  [Batch {batch_num+1}] Request {i+1} failed: {response.status_code}")
            except Exception as e:
                failed_count += 1
                batch_failed += 1
                if batch_failed <= 3:  # Log first few errors
                    print(f"  [Batch {batch_num+1}] Request {i+1} error: {str(e)[:100]}")
            
            sequence += 1
            time.sleep(0.01)  # Small delay within batch
        
        batch_elapsed = time.time() - batch_start
        elapsed_total = time.time() - start_time
        
        print(f"  Batch {batch_num+1}/{batches}: {batch_success} success, {batch_failed} failed | "
              f"Total: {success_count}/{sequence-1} | Elapsed: {elapsed_total/60:.1f} min")
        
        # Wait 60 seconds between batches to respect 100/min limit
        if batch_num < batches - 1 and sequence <= total_requests:
            wait_time = 60 - batch_elapsed
            if wait_time > 0:
                print(f"  Waiting {wait_time:.1f}s before next batch...")
                time.sleep(wait_time)
    
    elapsed_time = time.time() - start_time
    print("-" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"Success: {success_count}/{total_requests}")
    print(f"Failed: {failed_count}/{total_requests}")
    print(f"Success rate: {success_count/total_requests*100:.1f}%")
    
    return success_count, failed_count, elapsed_time


def main():
    """Main function"""
    print("=" * 70)
    print("AgentBill Load Test Script")
    print("=" * 70)
    
    # Get API key from settings
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key or not str(api_key).strip():
        print("\n[ERROR] AGENTBILL_API_KEY not configured!")
        print("Please set AGENTBILL_API_KEY in your .env file or environment variables.")
        return 1
    
    print(f"\nAPI Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"Endpoint: https://api.agentbill.io/functions/v1/otel-collector")
    
    # Ask user which test to run
    print("\nSelect test to run:")
    print("  1. 100 requests/minute (fast test, ~1 minute)")
    print("  2. 10,000 requests/day (slow test, ~100 minutes)")
    print("  3. Both tests")
    
    try:
        choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 0
    
    if choice == "1":
        load_test_100_per_minute(api_key)
    elif choice == "2":
        print("\n[WARNING] This will take approximately 100 minutes (100 batches x 60 seconds)")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            load_test_10000_per_day(api_key)
        else:
            print("Cancelled.")
    elif choice == "3":
        print("\nRunning both tests...")
        load_test_100_per_minute(api_key)
        print("\n[WARNING] Next test will take approximately 100 minutes")
        confirm = input("Continue with 10,000/day test? (yes/no): ").strip().lower()
        if confirm == "yes":
            load_test_10000_per_day(api_key)
        else:
            print("Skipped 10,000/day test.")
    else:
        print(f"Invalid choice: {choice}")
        return 1
    
    print("\n" + "=" * 70)
    print("Load test completed!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
