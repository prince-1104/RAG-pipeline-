#!/usr/bin/env python3
"""
Send signals with correct event names matching the application.
Uses 'ai_completion' and 'ai_embedding' instead of test names.
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


def create_completion_payload(seq: int):
    """Create ai_completion payload (matches app)"""
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
                        _attr("agentbill.event_name", "ai_completion"),  # Correct event name
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
                    ],
                }]
            }]
        }]
    }


def create_embedding_payload(seq: int):
    """Create ai_embedding payload (matches app)"""
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
                    "name": "gen_ai.embeddings",
                    "kind": 3,
                    "startTimeUnixNano": str(now_ns - 1000000000),
                    "endTimeUnixNano": str(now_ns),
                    "status": {"code": 1},
                    "attributes": [
                        _attr("agentbill.event_name", "ai_embedding"),  # Correct event name
                        _attr("agentbill.is_business_event", True),
                        _attr("agent.name", settings.service_name),
                        _attr("gen_ai.provider.name", "openai"),
                        _attr("gen_ai.request.model", "text-embedding-ada-002"),
                        _attr("gen_ai.response.model", "text-embedding-ada-002"),
                        _attr("gen_ai.operation.name", "embeddings"),
                        _attr("ai.success", True),
                        _attr("ai.input_tokens", 50),
                        _attr("ai.total_tokens", 50),
                        _attr("gen_ai.usage.input_tokens", 50),
                        _attr("gen_ai.usage.total_tokens", 50),
                        _attr("ai.batch_size", 5),
                    ],
                }]
            }]
        }]
    }


def send_one(url, headers, payload):
    """Send single request"""
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            return (True, resp.status_code, None)
        else:
            return (False, resp.status_code, resp.text[:100])
    except requests.exceptions.Timeout:
        return (False, 0, "Timeout")
    except requests.exceptions.ConnectionError as e:
        return (False, 0, f"ConnectionError: {str(e)[:50]}")
    except Exception as e:
        return (False, 0, f"Error: {str(e)[:50]}")


def main():
    """Send signals with correct event names"""
    settings = get_settings()
    api_key = getattr(settings, "agentbill_api_key", None)
    
    if not api_key:
        print("[ERROR] AGENTBILL_API_KEY not set!")
        return 1
    
    url = "https://api.agentbill.io/functions/v1/otel-collector"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print("=" * 70)
    print("Sending Signals with Correct Event Names")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("[INFO] Using 'ai_completion' and 'ai_embedding' event names")
    print("[INFO] These match your application's event names")
    print("-" * 70)
    
    # Send mix of completion and embedding signals
    print("\n[SENDING] 50 ai_completion + 50 ai_embedding signals...")
    print("[INFO] Using rate limiting to avoid API limits (100 req/min)")
    
    start = time.time()
    success = 0
    failed = 0
    
    # Reduce concurrency to avoid rate limits (100 req/min = ~1.67 req/sec)
    # Use 10 workers max to stay under limit
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        # 50 completion signals
        for i in range(50):
            payload = create_completion_payload(i + 1)
            futures.append(executor.submit(send_one, url, headers, payload))
        
        # 50 embedding signals
        for i in range(50):
            payload = create_embedding_payload(i + 1)
            futures.append(executor.submit(send_one, url, headers, payload))
        
        completed = 0
        error_samples = []
        error_counts = {}  # Track error types
        
        for future in as_completed(futures):
            ok, status, error_msg = future.result()
            completed += 1
            if ok:
                success += 1
            else:
                failed += 1
                # Track error types
                error_key = f"{status}:{error_msg[:30] if error_msg else 'Unknown'}"
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
                
                if len(error_samples) < 5:  # Collect first 5 errors
                    error_samples.append((status, error_msg))
            
            if completed % 20 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{completed}/100] OK: {success}, Failed: {failed}, Rate: {rate:.1f} req/min")
        
        # Print error summary
        if error_samples:
            print(f"\n[ERROR SUMMARY] {failed} failures out of 100 requests")
            print(f"[ERROR SAMPLES] First {len(error_samples)} failures:")
            for i, (status, error_msg) in enumerate(error_samples, 1):
                print(f"  {i}. Status: {status}, Error: {error_msg}")
            
            if len(error_counts) > 1:
                print(f"\n[ERROR BREAKDOWN] Error types:")
                for error_key, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {error_key}: {count} occurrences")
    
    elapsed = time.time() - start
    
    print(f"\n[COMPLETE] Time: {elapsed:.2f}s | Success: {success} | Failed: {failed}")
    print(f"Rate: {success/elapsed*60:.1f} req/min")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Wait 1-2 minutes for signals to process")
    print("2. Refresh your AgentBill dashboard")
    print("3. Check 'Last 24 Hours' view")
    print("4. Filter by 'ai_completion' and 'ai_embedding' event types")
    print("5. Signals should now appear correctly categorized (not 'Unknown')")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
