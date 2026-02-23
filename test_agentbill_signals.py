#!/usr/bin/env python3
"""
Test script to send pseudo AgentBill signals for diagnostics.
This script simulates LLM completion and embedding events to test AgentBill integration.
"""

import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.observability.agentbill import start_event, finalize_event, get_agentbill_status


def test_completion_event():
    """Test a completion/chat event (simulating LLM call)"""
    print("\n[TEST 1] Testing completion event...")
    
    event = start_event(
        event_name="test_ai_completion",
        provider="openai",
        model="gpt-4o-mini",
        kind="completion",
    )
    
    # Simulate some processing time
    time.sleep(0.1)
    
    # Simulate successful completion
    finalize_event(
        event,
        success=True,
        input_tokens=100,
        output_tokens=250,
        total_tokens=350,
        attributes={
            "test.run": "pseudo_script",
            "test.scenario": "completion_success",
        },
    )
    print("[OK] Completion event finalized")


def test_embedding_event():
    """Test an embedding event"""
    print("\n[TEST 2] Testing embedding event...")
    
    event = start_event(
        event_name="test_ai_embedding",
        provider="openai",
        model="text-embedding-ada-002",
        kind="embedding",
    )
    
    # Simulate some processing time
    time.sleep(0.05)
    
    # Simulate successful embedding
    finalize_event(
        event,
        success=True,
        input_tokens=50,
        output_tokens=0,  # Embeddings don't have output tokens
        total_tokens=50,
        attributes={
            "test.run": "pseudo_script",
            "test.scenario": "embedding_success",
            "ai.batch_size": 5,
        },
    )
    print("[OK] Embedding event finalized")


def test_failed_event():
    """Test a failed event (simulating API error)"""
    print("\n[TEST 3] Testing failed event...")
    
    event = start_event(
        event_name="test_ai_completion",
        provider="openai",
        model="gpt-4o-mini",
        kind="completion",
    )
    
    time.sleep(0.05)
    
    # Simulate failed completion
    finalize_event(
        event,
        success=False,
        error="test_api_error_401",
        input_tokens=100,
        output_tokens=0,
        total_tokens=100,
        attributes={
            "test.run": "pseudo_script",
            "test.scenario": "completion_failure",
        },
    )
    print("[OK] Failed event finalized")


def test_together_embedding():
    """Test a Together AI embedding event"""
    print("\n[TEST 4] Testing Together AI embedding event...")
    
    event = start_event(
        event_name="test_ai_embedding",
        provider="together",
        model="BAAI/bge-base-en-v1.5",
        kind="embedding",
    )
    
    time.sleep(0.05)
    
    finalize_event(
        event,
        success=True,
        input_tokens=75,
        total_tokens=75,
        attributes={
            "test.run": "pseudo_script",
            "test.scenario": "together_embedding",
            "ai.batch_size": 10,
        },
    )
    print("[OK] Together embedding event finalized")


def test_openrouter_event():
    """Test an OpenRouter event (gateway)"""
    print("\n[TEST 5] Testing OpenRouter event...")
    
    event = start_event(
        event_name="test_ai_completion",
        provider="openrouter",
        model="openai/gpt-4o-mini",
        kind="completion",
    )
    
    time.sleep(0.1)
    
    finalize_event(
        event,
        success=True,
        input_tokens=150,
        output_tokens=300,
        total_tokens=450,
        attributes={
            "test.run": "pseudo_script",
            "test.scenario": "openrouter_completion",
        },
    )
    print("[OK] OpenRouter event finalized")


def send_batch_signals(count=100):
    """Send a batch of pseudo signals for tracking purposes"""
    print(f"\n[SENDING] Generating {count} pseudo signals...")
    
    import random
    
    providers = ["openai", "together", "openrouter"]
    models = {
        "openai": ["gpt-4o-mini", "gpt-4o", "text-embedding-ada-002"],
        "together": ["BAAI/bge-base-en-v1.5", "togethercomputer/m2-bert-80M-8k-retrieval"],
        "openrouter": ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
    }
    kinds = ["completion", "embedding"]
    
    success_count = 0
    failed_count = 0
    
    for i in range(count):
        provider = random.choice(providers)
        model = random.choice(models[provider])
        kind = random.choice(kinds)
        success = random.random() > 0.1  # 90% success rate
        
        event = start_event(
            event_name=f"pseudo_signal_{i+1}",
            provider=provider,
            model=model,
            kind=kind,
        )
        
        # Simulate processing time
        time.sleep(random.uniform(0.01, 0.05))
        
        if success:
            input_tokens = random.randint(50, 500)
            output_tokens = random.randint(0, 1000) if kind == "completion" else 0
            total_tokens = input_tokens + output_tokens
            
            finalize_event(
                event,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                attributes={
                    "test.run": "batch_pseudo_signals",
                    "test.batch_id": f"batch_{count}",
                    "test.signal_index": i + 1,
                },
            )
            success_count += 1
        else:
            finalize_event(
                event,
                success=False,
                error=f"pseudo_error_{random.choice(['timeout', 'rate_limit', 'api_error'])}",
                input_tokens=random.randint(50, 200),
                attributes={
                    "test.run": "batch_pseudo_signals",
                    "test.batch_id": f"batch_{count}",
                    "test.signal_index": i + 1,
                },
            )
            failed_count += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{count} signals sent...")
    
    print(f"\n[COMPLETE] Sent {count} signals ({success_count} success, {failed_count} failed)")


def main():
    """Run all test scenarios"""
    print("=" * 60)
    print("AgentBill Signal Test Script")
    print("=" * 60)
    
    # Check initial status
    print("\n[INITIAL STATUS]")
    status = get_agentbill_status()
    print(f"  Enabled: {status['enabled']}")
    print(f"  API Key Configured: {status['api_key_configured']}")
    print(f"  Endpoint: {status['endpoint']}")
    print(f"  Executor Running: {status['executor_running']}")
    print(f"  HTTP Client Initialized: {status['http_client_initialized']}")
    
    initial_counters = status.get('event_counters', {})
    print(f"\n  Initial Event Counters:")
    for key, value in initial_counters.items():
        print(f"    {key}: {value}")
    
    # Send 100 pseudo signals
    print("\n" + "=" * 60)
    print("Sending 100 Pseudo Signals for Tracking...")
    print("=" * 60)
    
    try:
        send_batch_signals(count=100)
    except Exception as e:
        print(f"\n[ERROR] Error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Wait a bit for async processing (longer for 100 signals)
    print("\n[WAITING] Allowing background executor to process events...")
    print("  (This may take a few seconds for 100 signals)")
    time.sleep(5)
    
    # Check final status
    print("\n" + "=" * 60)
    print("[FINAL STATUS]")
    print("=" * 60)
    status = get_agentbill_status()
    print(f"  Enabled: {status['enabled']}")
    print(f"  API Key Configured: {status['api_key_configured']}")
    print(f"  HTTP Client Initialized: {status['http_client_initialized']}")
    
    final_counters = status.get('event_counters', {})
    print(f"\n  Final Event Counters:")
    for key, value in final_counters.items():
        initial_value = initial_counters.get(key, 0)
        delta = value - initial_value
        delta_str = f" (+{delta})" if delta > 0 else ""
        print(f"    {key}: {value}{delta_str}")
    
    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    
    events_created = final_counters.get('events_created', 0) - initial_counters.get('events_created', 0)
    events_submitted = final_counters.get('events_submitted', 0) - initial_counters.get('events_submitted', 0)
    events_success = final_counters.get('events_sent_success', 0) - initial_counters.get('events_sent_success', 0)
    events_failed = final_counters.get('events_sent_failed', 0) - initial_counters.get('events_sent_failed', 0)
    events_skipped = (
        final_counters.get('events_skipped_no_key', 0) - initial_counters.get('events_skipped_no_key', 0) +
        final_counters.get('events_skipped_disabled', 0) - initial_counters.get('events_skipped_disabled', 0)
    )
    
    print(f"  Events Created: {events_created}")
    print(f"  Events Submitted: {events_submitted}")
    print(f"  Events Sent Successfully: {events_success}")
    print(f"  Events Failed: {events_failed}")
    print(f"  Events Skipped: {events_skipped}")
    
    if events_created == 0:
        print("\n[WARNING] No events were created!")
    elif events_skipped > 0:
        print("\n[WARNING] Some events were skipped (check configuration)")
    elif events_submitted == 0:
        print("\n[WARNING] Events created but not submitted (check executor)")
    elif events_failed > 0:
        print("\n[WARNING] Some events failed to send (check logs for errors)")
    elif events_success > 0:
        print("\n[SUCCESS] Events are being sent successfully!")
    else:
        print("\n[INFO] Events were submitted but results are pending (check AgentBill dashboard)")
    
    print("\n" + "=" * 60)
    print("100 pseudo signals have been sent for tracking purposes.")
    print("Check your AgentBill dashboard to verify signals are being received.")
    print("Signals may take a few minutes to appear in the dashboard.")
    print("Also check application logs for any error messages.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
