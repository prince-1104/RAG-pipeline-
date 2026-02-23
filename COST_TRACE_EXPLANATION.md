# Cost Trace Issues Explained

## Problem: "Unknown" Provider & $0.00 Cost

### What You're Seeing:
- **Provider**: `Unknown`
- **Model**: `gpt-4o-mini`
- **Traces**: 500
- **Cost**: `$0.00`
- **Agent**: `No Agent`
- **Status**: Broken chain icon (untracked)

### Why This Happens:

1. **Missing Provider Recognition**
   - AgentBill needs specific provider attributes to identify cost
   - Test scripts may be missing required attributes
   - Provider must match AgentBill's known providers (openai, anthropic, etc.)

2. **No Agent Association**
   - Signals need `agent.id` or registered agent
   - `agent.name` alone isn't enough for tracking
   - Without agent, signals show as "untracked"

3. **Cost Calculation Requires Provider**
   - AgentBill calculates cost based on provider pricing
   - Unknown provider = no pricing data = $0.00 cost
   - Model alone isn't enough without provider context

### Solution:

The real application code includes:
- `gen_ai.system` (legacy compatibility)
- `gen_ai.provider.name` 
- Proper provider normalization

Test scripts should match the real application payload structure exactly.

### Fix:

Use the actual application's AgentBill integration instead of test scripts for accurate cost tracking. Test scripts are for load testing, not cost tracking.
