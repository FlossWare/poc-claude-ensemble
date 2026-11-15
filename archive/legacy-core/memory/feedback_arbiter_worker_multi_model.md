---
name: arbiter-worker-multi-model
description: Arbiter/worker pattern must always use multi-model consensus for workers
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d62de65f-849d-4b69-9db4-08d7ed993074
---

When implementing arbiter/worker patterns in workflows, **ALWAYS use different AI models for workers** to get diverse perspectives and better coverage.

**Why:** Different models have different strengths and blind spots:
- Opus: Architectural issues, complex reasoning
- Sonnet: Implementation bugs, balanced analysis
- Haiku: Edge cases, fast iteration
- GPT-4o: External perspective from OpenAI
- Gemini: External perspective from Google (gracefully ignored if API unavailable)

**How to apply:**

```javascript
// ❌ WRONG: All workers same model
const workers = await parallel(
  items.map(item => () => agent(prompt))  // All inherit session model
);

// ✅ CORRECT: Multi-model consensus (Claude workflows)
const MODELS = ['opus', 'sonnet', 'haiku'];  // Gemini not yet supported in Claude Code workflows
const workers = await parallel(
  items.flatMap(item => 
    MODELS.map(model => () => 
      agent(prompt, {
        label: `${item.name}-${model}`,
        model: model
      })
    )
  )
);

// ✅ CORRECT: Multi-model consensus (Python SDLC workflows)
worker_models=[
    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    {"provider": "anthropic", "model": "claude-opus-4-7"},
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "google", "model": "gemini-1.5-pro"}  # Gracefully ignored if unavailable
]
```

**Pattern structure:**
1. **Workers**: Multiple models analyze same items independently
2. **Arbiter**: Single model (usually Opus) synthesizes all perspectives and filters false positives

**Benefits:**
- Better bug coverage (each model finds different issues)
- Diverse perspectives reduce blind spots
- Cross-validation reduces false positives
- More robust findings

**Apply to:** 
- All arbiter/worker workflows (testing, code review, security audits, bug finding, verification)
- **code-sdlc workflows**: Ensure all role configs set `worker_models` to use different AI providers (Claude, GPT, Gemini, etc.) for each worker
- Python SDLC roles: Configure `worker_models` in RoleConfig (e.g., Claude Sonnet, GPT-4o, Gemini for 3 workers)

**For code-sdlc-auto workflows:**
Each role (PM, Architect, Security, Developer, QA, DevOps, TechWriter) should use multi-model workers by default, not just multi-worker consensus with the same model.
