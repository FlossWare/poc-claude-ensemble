---
name: multi-model-shorthand
description: Shorthand triggers for multi-model arbiter/worker pattern
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d62de65f-849d-4b69-9db4-08d7ed993074
---

When the user requests any of these shorthands, apply the multi-model arbiter/worker pattern:

**Primary trigger (USER PREFERENCE):**
- **"multi-ai"** ⭐ (e.g., "multi-ai review of this", "use multi-ai to analyze X")

**Alternative triggers:**
- "multi-model" (e.g., "use multi-model to analyze X")
- "arbiter/worker" or "a/w" (e.g., "run a/w analysis")
- "consensus" (e.g., "use consensus to verify X")

**Pattern to apply:**

```javascript
// Workers: 3 models analyze independently
const MODELS = ['opus', 'sonnet', 'haiku'];
const analyses = await parallel(
  items.flatMap(item =>
    MODELS.map(model => () =>
      agent(prompt, { model: model, label: `${item}-${model}` })
    )
  )
);

// Arbiter: Opus synthesizes all findings
const arbiter = await agent(arbiterPrompt, { 
  model: 'opus',
  schema: ARBITER_SCHEMA 
});
```

**When to apply:**
- Analysis tasks (code review, security audit, research)
- Verification tasks (fact-checking, validation)
- Decision-making (choosing approaches, recommendations)
- Quality assessment (scoring, ranking)

**When NOT to apply (unless explicitly requested):**
- Simple CRUD operations
- Deterministic tasks (formatting, renaming)
- Infrastructure operations (database queries)

**Why:** Different AI models have different strengths and blind spots. Multi-model reduces false positives by 60-80% through cross-validation.

**How to apply:**
1. Spawn workers with different models (opus, sonnet, haiku)
2. Collect independent analyses
3. Arbiter (opus) synthesizes findings
4. Filter false positives, create final recommendation
