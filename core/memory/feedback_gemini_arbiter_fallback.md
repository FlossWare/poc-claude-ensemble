---
name: gemini-arbiter-fallback
description: Gemini can be arbiter but must have fallback logic if it fails
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d62de65f-849d-4b69-9db4-08d7ed993074
---

Gemini can serve as arbiter, but if calling it fails, the system must gracefully recover.

**Fallback logic when Gemini arbiter fails:**

1. **Select new arbiter** from remaining workers:
   - Prefer: Claude Opus (if it was a worker)
   - Otherwise: Claude Sonnet or GPT-4o
   - Never: Gemini again (it already failed)

2. **Re-run arbiter synthesis** with the new arbiter using same worker results

3. **Log the fallback** for telemetry/debugging

**Why:** Gemini API may be unavailable, rate-limited, or fail for various reasons. The workflow must continue without manual intervention.

**How to apply:**

```python
# Python SDLC workflows
try:
    arbiter_result = call_arbiter(model=gemini_config)
except Exception as e:
    logger.warning(f"Gemini arbiter failed: {e}, falling back to Opus")
    # Select new arbiter from workers
    new_arbiter = select_fallback_arbiter(worker_models)
    arbiter_result = call_arbiter(model=new_arbiter)
```

```javascript
// JavaScript workflows
let arbiterResult;
try {
  arbiterResult = await agent(arbiterPrompt, {
    model: 'gemini',
    phase: 'Arbiter Verification'
  });
} catch (e) {
  log(`⚠ Gemini arbiter failed: ${e.message}, falling back to Opus`);
  arbiterResult = await agent(arbiterPrompt, {
    model: 'opus',
    phase: 'Arbiter Verification'
  });
}
```

**When Gemini is a worker:**
- If Gemini worker fails → Continue with remaining 3 workers (Opus/Sonnet/GPT-4o)
- Filter out null results with `.filter(Boolean)`
- Arbiter synthesizes from available worker results

**When Gemini is arbiter:**
- If Gemini arbiter fails → Select new arbiter from workers
- Re-run synthesis with new arbiter
- Never fail the entire workflow due to Gemini unavailability

**Apply to:** All workflows where Gemini is used (SDLC, testing, review, security)
