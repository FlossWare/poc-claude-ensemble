# Backlog Architecture Audit

**Issue:** #47  
**Purpose:** Keep the Claude Ensemble backlog aligned with the implementation that actually exists.

## Why this audit exists

Claude Ensemble has accumulated several generations of tooling, services, integrations, dashboards, and documentation. Some backlog items describe earlier implementations rather than the current architecture.

The rule for future work is simple:

> **Treat an issue as a statement of intent, not as an authoritative description of the current codebase.**

Before implementing a backlog item, locate the current capability and determine whether the requested work is new functionality, an interface migration, consolidation, or documentation cleanup.

## Current capability map

| Capability | Current implementation | Backlog implication |
|---|---|---|
| Cost tracking | `cost_tracking/`, `tools/cost-dashboard.py`, `tools/performance_dashboard.py` | #2 remains valid. Multiple schemas, paths, and pricing definitions need consolidation. |
| Thompson routing | `shared/thompson_router.py`, `shared/thompson_client.py`, `shared/thompson-sampling-helper.js`, `thompson-service/`, `learning/` | Do not build another router. Future work should target contracts/integration boundaries. |
| Learning | `learning/`, `learning-service/` | Existing capability. New issues should identify missing behavior or service boundaries explicitly. |
| Compression | `compression/`, `shared/compression-bridge.js` | Existing subsystem. Extend its contract rather than creating parallel compression code. |
| Caching | `caching/`, `shared/caching-bridge.js` | Existing subsystem. Cost tracking already has cache instrumentation hooks. |
| Memory | `memory/`, `memory-service/` | Existing capability and service boundary. |
| Arbitration | `arbitration/` | Existing orchestration capability. |
| Alerting | `alert_service/` | Existing service capability. |
| Code search | `code-search-mcp/` | Existing MCP implementation. |
| PR review | `tools/code-pr-review.js`, `tools/code-pr-review-auto.js`, `tools/review.sh` | #3 is an interface/automation migration, not greenfield review logic. |
| MCP/service layer | `code-search-mcp/`, `memory-service/`, `learning-service/`, `thompson-service/`, `alert_service/` | #5-#8 should reuse existing services/contracts. |

## Detailed finding: cost tracking

The current repository has at least three cost-record assumptions:

### Legacy dashboard

`tools/cost-dashboard.py` reads:

```
~/.claude/cost_tracking/cost.log
```

and expects fields including `total_cost_usd`, `provider`, and `workflow_id`.

It also owns a separate pricing table.

### Current logger

`cost_tracking/logger.py` writes:

```
cost_tracking/api_costs.jsonl
```

with fields such as:

- `cost_usd`
- `input_tokens`
- `output_tokens`
- `task_name`
- `source`

It has its own pricing table.

### Aggregator/integration layer

`cost_tracking/aggregator.py` uses a richer `CostEntry` model containing `provider`, `workflow_id`, cache information, compression information, and `total_cost_usd`.

It defaults to the legacy `cost.log` location.

`cost_tracking/integration.py` adds routing, compression, cache, and API-call instrumentation.

### Result

This is not simply "two dashboards." It is a **data-contract problem**:

1. Multiple record schemas exist.
2. Multiple storage paths exist.
3. Pricing is duplicated.
4. Field names differ for equivalent concepts.
5. Dashboard consumers do not consistently consume the logger's current record format.

Therefore #2 should establish a canonical cost event schema and canonical persistence path first. Dashboards should become consumers of that contract.

## MCP backlog interpretation

### #3 PR review

Existing review tooling means the work should preserve the current review behavior while defining a service/MCP boundary around it.

### #5 Git History MCP
Likely new interface/capability. First inspect existing git-related tools and scripts before implementing.

### #6 Documentation MCP
Likely new interface/capability, but should reuse existing documentation-generation/release-note tooling where applicable.

### #7 Knowledge Base MCP
Should reuse memory, semantic-search, and learning components where they overlap. Avoid creating a second knowledge store without an explicit contract.

### #8 Build/CI MCP
Should expose existing build/test/CI operations rather than embedding another build system.

## Documentation drift

The repository contains status and architecture documents alongside active code, including:

- `COMPLETION_STATUS.md`
- `TOOLKIT_STATUS.md`
- `VERIFICATION_REPORT.md`
- `STAGING_TEST_REPORT.md`
- `SERVICES_GUIDE.md`
- `TOOLS_INTEGRATION_GUIDE.md`
- `SKILL_INTEGRATION_GUIDE.md`

These should be treated as candidates for periodic reconciliation against the implementation tree.

## Backlog hygiene procedure

For each future issue:

1. Locate the current implementation.
2. Identify the current contract and data path.
3. Determine whether the issue is:
   - **implemented**
   - **partially implemented**
   - **new interface**
   - **consolidation**
   - **obsolete**
   - **duplicate**
4. Update the issue with concrete repository paths.
5. Implement only the remaining delta.

## Recommended sequence

1. **#47:** establish this audit as the backlog reference.
2. **#2:** consolidate cost tracking around one canonical event contract.
3. **#3:** define the PR-review MCP/service boundary around existing review tooling.
4. **#5/#6/#7/#8:** inspect and implement each service boundary using existing capabilities.
5. Periodically reconcile the status/architecture documents with the codebase.

This deliberately avoids rebuilding capabilities that already exist merely because their older issue descriptions make them look absent.
