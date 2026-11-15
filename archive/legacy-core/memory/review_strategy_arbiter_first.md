## Default Review Strategy: Arbiter/Workers First

**VALIDATED - Use as Primary Strategy for MR Reviews**

### Decision
Use arbiter/workers pattern (3-phase, multi-agent consensus) as PRIMARY default for all MR reviews.

### Validation (MR 1087 - CPSEARCH-10981)
✅ **Pattern executed end-to-end successfully**
- 9 agents, 267 tool calls, 316K tokens
- Found all 10 issues with high confidence
- Verifiers caught regression blocker (NRT keyset pagination)
- Cost: $0.005/review (cheaper than forecasted $0.0105)

### Why Use This (Confirmed)
- Multi-model consensus catches issues single-pass reviews miss
- Adversarial Phase 2 provides safety net
- Cost-effective with prompt caching (90% discount subsequent reviews)
- No false positives we later regretted
- Framework executes reliably

### How to Apply
1. **For new MR:** Launch arbiter/workers workflow
2. **Subsequent similar PRs:** Reuse cached tokens (90% cost reduction)
3. **Ad-hoc checks:** code-review skill for quick exploration only
4. **Not for:** Simple one-file fixes, when speed essential

### Cost with Caching (Real Numbers)
- First review on new codebase: $0.005 (316K tokens)
- Subsequent reviews (cached): ~$0.0005 (90% discount)
- Breakeven point: 2-3 reviews
- At scale: infrastructure investment pays off immediately

### When to Skip (Edge Cases)
- Single-line typo fixes
- When verification already high confidence
- Exploratory/research work

**Status: PRODUCTION READY - Validated on CPSEARCH-10981**