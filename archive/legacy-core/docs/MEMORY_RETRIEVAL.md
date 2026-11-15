# Memory retrieval follow-up contract

The execution-aware Memory retrieval added in PR #140 intentionally remains a small, deterministic retrieval layer over JSONL records. Issue #141 reviews the follow-up suggestions from Claude's review of that PR.

## Relationship priority

The current ranking is explicit and stable:

1. `parent` (100)
2. `ancestor` (90)
3. `same-request` (70)
4. `related-lineage` (60)

`same-request` remains above `related-lineage` deliberately. A same-request record is directly associated with the active request, while related-lineage is inferred from a shared lineage root. The latter is a useful fallback, but the root match alone does not establish that the two executions belong to the same immediate request flow.

This is a policy choice, not an accidental ordering. If later dogfooding demonstrates that sibling branches are consistently more useful than unrelated same-request records, the ranking can be changed as a separate policy change with retrieval evidence and regression coverage.

## Descendants and lineage

Descendant contexts are excluded before relation assignment. Empty lineages are valid input and must not cause indexing or prefix errors. A record with an empty or otherwise unusable lineage is simply unable to qualify for ancestor, descendant, or related-lineage matching; it may still qualify for `parent` or `same-request` when those relationships are independently established.

## Malformed persisted contexts

Records without a dictionary-valued `execution_context`, or with an execution context that cannot be restored by `ExecutionContext.from_dict()`, are skipped. Retrieval is fail-closed for malformed context data rather than allowing one bad persisted record to abort the query.

These records are logged at `DEBUG` because malformed records are expected to be exceptional data-quality events and retrieval is intentionally resilient. Operational monitoring can be added later if malformed-record frequency becomes actionable.

## Duplicate execution IDs

Retrieval does not deduplicate records solely by `execution_id`. JSONL records represent persisted observations, and the same execution ID may legitimately occur more than once. Matching records therefore retain their persisted order. The stable sort key is `(relationship rank descending, record index ascending)`, so equal-ranked records are deterministic.

If the system later defines one record as authoritative for an execution ID, deduplication should be introduced together with that authority contract rather than guessed by the retrieval layer.

## Scale

The current implementation reads the JSONL memory file and evaluates records in memory. Retrieval is therefore O(n) in the number of persisted records and uses memory proportional to the file size. This is acceptable for the current file-backed service and keeps retrieval deterministic. Large-store indexing or streaming retrieval should be treated as a separate scalability change once actual memory-store sizes justify it.

## Review disposition

Claude's follow-up suggestions are handled as follows:

- Related-lineage ranking: **documented as intentional**, retained at 60 for now.
- Descendant handling: **already implemented** with an immediate `continue`.
- Empty lineage assumptions: **documented and regression-tested**.
- Relation priority and stable tie-breaking: **documented and regression-tested**.
- Malformed contexts: **regression-tested**; fail-closed behavior retained.
- Duplicate execution IDs: **regression-tested and documented** as ordered observations rather than implicitly deduplicated.
- Logging level: **documented as intentional** at DEBUG.
- Large-store performance: **documented as a known O(n) boundary**, not prematurely optimized.
- Retrieval example: **documented here** rather than bloating the API docstring.
