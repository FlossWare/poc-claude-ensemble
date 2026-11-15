---
name: workflow-parallel-vs-pipeline
description: Correct semantics of parallel() vs pipeline() in Workflow tool - verified by arbiter/worker pattern
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d62de65f-849d-4b69-9db4-08d7ed993074
---

**CRITICAL CORRECTION:** I previously misrepresented `pipeline()` behavior and was caught by arbiter/worker verification.

**WRONG statements I made:**
- "pipeline() allows items to flow independently without barriers" ❌
- "pipeline() saves time by eliminating wait between phases" ❌

**CORRECT semantics (verified by arbiter):**

- **`parallel()`** = Run ALL tasks **concurrently** → **eliminates waits** → use for fan-out/fan-in patterns (e.g., test 25 components simultaneously, wait for all, aggregate results)

- **`pipeline()`** = Process items **SEQUENTIALLY** (one at a time) → **ADDS waits between items** → use when items must be processed in order or to limit concurrency

**Why:** The arbiter caught this by analyzing actual workflow code:
- `parallel()` is used in ui-testing.js to run multiple agents concurrently (lines 34, 112, 179, 284)
- `pipeline()` processes items sequentially, waiting for each to complete before starting the next
- User's workflows correctly use `parallel()` with phase-based barriers for arbiter aggregation

**How to apply:** 
- When explaining Workflow patterns, verify claims against tool documentation before answering
- Use arbiter/worker verification for complex technical claims
- Never claim `pipeline()` eliminates waits - it enforces sequential ordering
- `parallel()` is for concurrent execution, `pipeline()` is for sequential processing (often combined with nested `parallel()` calls)

**Verified:** 2026-06-08 via arbiter/worker workflow analysis (accuracy score 45/100 due to pipeline() errors)
