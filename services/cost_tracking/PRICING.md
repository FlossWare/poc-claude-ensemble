# your organization AI API Pricing

**Last Updated:** 2026-09-27  
**Source:** Claude Ensemble GCP Billing Report (September 2026)  
**Status:** Extracted from actual usage — These are negotiated Claude Ensemble enterprise rates

---

## Known Claude Ensemble Pricing (Verified from GCP Billing)

Extracted from GCP billing report showing actual Claude Ensemble charges for Claude models:

### Claude Haiku 4.5
- **Output tokens:** $0.000005 per token
- **Input Cache Write (TTL 300s):** $0.00000125 per token
- **Input Cache Read:** $0.000000100 per token

### Claude Sonnet 4.5
- **Output tokens:** $0.000015 per token
- **Input Cache Write (TTL 300s):** $0.00000375 per token
- **Input Cache Read:** (not in billing report)

### Claude Opus 5
- **Output tokens:** $0.000025 per token
- **Input Cache Write (TTL 300s):** $0.00000625 per token
- **Input Cache Read:** $0.0000005 per token

---

## TBD Pricing (Need Confirmation)

### Google Gemini
- **Status:** Unknown — May be on separate GCP contract or different billing
- **To find:** Check GCP billing report under Gemini SKUs or contact Google account team
- **Placeholder:** Currently $0.00 in test data

### JetBrains Cursor
- **Status:** Unknown — Need to contact JetBrains for Claude Ensemble contract rates
- **Placeholder:** Currently $0.00 in test data

---

## How to Update Pricing

1. **Get Gemini rates:** Check GCP billing or contact Google account rep
2. **Get Cursor rates:** Contact JetBrains account team
3. **Update this file:** Add actual rates here
4. **Update cost logger:** Modify `cost_tracking/logger.py` pricing dict to use Claude Ensemble rates
5. **Regenerate test data:** Re-run cost data generation with new rates

---

## Current Test Data

File: `api_costs.jsonl` (13 test entries)

Using Claude Ensemble rates where known:
- **Haiku entries:** 3 calls, calculated at $0.000005/output token
- **Sonnet entries:** 2 calls, calculated at $0.000015/output token
- **Opus entries:** 4 calls, calculated at $0.000025/output token
- **Cursor entries:** 2 calls, placeholder $0.00 (awaiting rates)
- **Gemini entries:** 2 calls, placeholder $0.00 (awaiting rates)

**Total test cost (known rates only):** $0.4295 across 9 calls  
**Cursor + Gemini cost:** TBD (8 test calls awaiting rates)

---

## Notes

- Test data uses **output tokens only** for simplicity (cache pricing available if needed)
- Real usage will include input tokens, cache reads, cache writes
- This file serves as the source of truth for Claude Ensemble pricing in the toolkit
- All dashboards and cost reports reference these rates

---

## Contacts for Missing Rates

- **Google Gemini:** [Google Account Rep] — gemini-api@google.com
- **JetBrains Cursor:** [JetBrains Account Rep] — support@jetbrains.com
- **Claude Ensemble Procurement:** [Your Claude Ensemble Finance Contact] — For contract rates

