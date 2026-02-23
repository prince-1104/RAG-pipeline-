# AgentBill Testing Guide

Quick reference for testing AgentBill signal integration.

## Test Scripts

### 1. Send Test Signals (100 signals)
```bash
python send_correct_signals.py
```
Sends 50 `ai_completion` + 50 `ai_embedding` signals with correct event names.

### 2. Fast Load Test (100 requests/minute)
```bash
python fast_load_test.py
```
Threaded version - completes in ~5-8 seconds.

### 3. Sequential Load Test
```bash
python fast_load_test.py 100
```
Sequential version - slower but more reliable.

### 4. Ultra-Fast Test
```bash
python ultra_fast_test.py
```
Simplest version - sends 100 signals quickly.

### 5. Diagnostic Test
```bash
python diagnose_signals.py
```
Checks API responses and payload structure.

### 6. Verify Signals
```bash
python verify_signals.py
```
Verifies signal acceptance and shows API responses.

## Expected Results

- **Success Rate**: 95-100% (some timeouts are normal)
- **Processing Time**: 5-15 minutes for signals to appear in dashboard
- **Dashboard**: Check "Last 24 Hours" view after waiting

## Troubleshooting

- **High failure rate**: Reduce concurrency or use sequential mode
- **Signals not appearing**: Wait 10-15 minutes, then refresh dashboard
- **"Unknown" event type**: Use `send_correct_signals.py` (uses correct event names)
