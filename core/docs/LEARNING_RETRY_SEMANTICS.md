# Learning Service Failure and Retry Semantics

Issue #138 defines the operational contract for learning ingestion failures and retries.

## Transaction model

A process_outcome request has two durable stages:

1. **Outcome payload** is written locally as the durable retry payload.
2. **Learning completion checkpoint** is written only after downstream learning work succeeds.

The local outcome file and the checkpoint are intentionally separate. A task with an outcome file but no checkpoint is incomplete and remains eligible for retry.

## Success contract

When process_outcome returns ok=true:

- the outcome payload exists locally;
- operational Memory learning.outcome has been persisted when configured;
- Thompson has been updated when configured;
- when Thompson was updated and operational Memory is configured, the post-update thompson.state snapshot has been persisted;
- the task is marked processed.

The checkpoint is the completion marker, not the outcome file.

## Failure contract

Any required downstream failure returns ok=false with checkpoint_advanced=false and leaves the task unprocessed.

Required failure cases include:

- local outcome persistence failure;
- operational Memory learning.outcome returning False or raising;
- Thompson returning False or raising;
- Thompson circuit breaker being open;
- Thompson state retrieval failing or returning no state after a successful Thompson update when operational Memory is configured;
- Thompson state snapshot persistence returning False or raising.

A failure after the local outcome has been persisted does not delete that outcome. The persisted payload is the source of truth for the next attempt.

## Retry semantics

Retries are caller-driven. The Learning Service does not run an automatic retry loop.

On retry:

1. If the task is already checkpointed, the request is treated as a duplicate and no learning work is repeated.
2. If an outcome file exists without a checkpoint, its persisted payload is reused.
3. If the incoming payload differs from the persisted payload for the same task ID, the request is rejected as a conflict.
4. Downstream learning is attempted again using the persisted payload.
5. The checkpoint is written only after the full required sequence succeeds.

Local outcome persistence and Memory writes are idempotent by deterministic task identity. Thompson delivery is currently **at-least-once**, not exactly-once, because the Thompson API does not carry a cross-restart idempotency key and the local task lock only protects callers within one process.

## Restart recovery

A service restart does not lose an uncheckpointed outcome. On restart:

- outcome files remain durable;
- the checkpoint remains the authoritative completion marker;
- an outcome without a checkpoint is retried when the caller submits it again;
- a checkpointed task is not reprocessed.

There is no automatic startup replay queue in the current service.

## Retry limits

There is currently **no Learning Service retry-count limit**. The caller decides when and how often to retry. This avoids silently discarding durable outcomes, while leaving retry scheduling and backoff outside the Learning Service.

If bounded retries, dead-lettering, or scheduled recovery are needed later, that should be introduced as a separate contract rather than changing the meaning of the checkpoint.

## Concurrency

The service serializes ingestion for the same task ID with an in-process task lock. The daemon itself is single-threaded.

This prevents duplicate Thompson updates from concurrent callers in one process, but it is not a distributed transaction mechanism. Cross-process exactly-once Thompson delivery remains intentionally out of scope.

## Operational meaning

**Outcome file = durable work to finish.**

**Checkpoint = learning transaction completed.**

Anything between those two states is retryable work.
