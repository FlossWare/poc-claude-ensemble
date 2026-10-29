# Where to Find Actual API Usage & Costs

**This session:** 2026-09-26 (all work was setup/testing, not real API calls)

---

## Real Usage from This Session

To find what we actually spent today:

### 1. Anthropic Claude API
**If RH has direct Anthropic account:**
- **Dashboard:** https://console.anthropic.com/account/billing/overview
- **Look for:** Usage in last 24 hours
- **Data includes:** Model (Haiku/Sonnet/Opus), tokens, cost
- **Export:** CSV available

**If RH routes through Google Cloud:**
- See "GCP Billing" section below

### 2. Google Cloud Billing (Most Likely for RH)
**You already have access** — See your GCP billing report screenshot

**Steps:**
1. Go to: https://console.cloud.google.com/billing
2. Select your RH project
3. Filter by: Last 24 hours (2026-09-26)
4. Search SKUs for: "Claude", "Gemini"
5. Download CSV for detailed breakdown

**What to look for:**
- Claude Haiku 4.5 — Input/Output/Cache tokens
- Claude Sonnet 4.5 — Input/Output/Cache tokens
- Claude Opus 5 — Input/Output/Cache tokens

### 3. Google Gemini API (if used directly)
**Dashboard:** https://aistudio.google.com/app/apikeys
- Requires separate Google account auth
- Shows API key usage
- Cost tracking varies by setup

### 4. JetBrains Cursor API
**Contact:** JetBrains account team
- Cursor usage not on public dashboards
- Requires API key from environment variables
- May be billed separately from Anthropic

---

## How to Log Real Costs

### Option A: Manual Entry
```bash
# Add a real API call to api_costs.jsonl
python3 cost_tracking/logger.py log_call \
  --model haiku \
  --input-tokens 1000 \
  --output-tokens 500 \
  --task-name "my_task"
```

### Option B: Automated from Anthropic
```bash
# If you have Anthropic API access, could query:
curl https://api.anthropic.com/v1/usage \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY" | jq .
```

### Option C: From GCP Billing Export
```bash
# GCP can export to BigQuery
# We could query daily costs and auto-log them
python3 cost_tracking/sync_from_gcp.py  # (script below)
```

---

## Template for Logging Session Work

Add this to `api_costs.jsonl` after completing work:

```json
{"timestamp": "2026-09-26T23:00:00.000000+00:00", "model": "haiku", "input_tokens": 5000, "output_tokens": 2000, "total_tokens": 7000, "cost_usd": 0.035, "task_name": "toolkit_testing", "source": "api", "metadata": {"session": "2026-09-26-claude-global-skills", "work": "verification_and_setup"}}
```

Replace:
- `input_tokens`: Actual input tokens from API response
- `output_tokens`: Actual output tokens from API response
- `cost_usd`: Calculate from RH rates or get from invoice
- `timestamp`: When the call happened
- `metadata`: What you were working on

---

## For This Specific Session (2026-09-26)

**What we did (no real API costs):**
- ✓ Set up cost tracking system
- ✓ Created test data
- ✓ Verified pricing
- ✓ Installed tools
- ✓ Tested dashboards

**Real costs to find:**
- Check GCP/Anthropic for any actual API calls
- Filter by date: 2026-09-26
- Filter by user/project: sfloess/claude-global-skills
- Look for models: Haiku, Sonnet, Opus, Gemini, Cursor

**Expected:** Likely minimal (maybe a few test calls, <$0.10)

---

## Ongoing: Automatic Cost Tracking

The cost logger is designed to:
1. **Hook into API responses** — Extract tokens + cost
2. **JSONL append-only log** — Never lose data
3. **Daily aggregation** — cost-dashboard.py reads raw log
4. **No manual entry** — Automatic when you use APIs

Once integrated, you'll have real data flowing to dashboards automatically.

---

## Contacts for Missing Data

| System | Where | Contact |
|--------|-------|---------|
| **Anthropic** | Direct API | account@anthropic.com |
| **GCP Billing** | console.cloud.google.com | Your RH GCP admin |
| **Gemini API** | aistudio.google.com | Google account team |
| **Cursor API** | JetBrains | support@jetbrains.com |

