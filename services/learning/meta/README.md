# Generalized meta-learning framework

One Thompson posterior per discrete decision domain. The model-selection
router and the GA are unchanged; every other decision the ensemble makes
repeatedly now has the same measured-feedback loop.

```
domains/          one module per domain (arms, normalization, ledger derivation)
learner.py        MetaLearner: select / record / posterior / status + state I/O
__main__.py       CLI: status, select, posterior, record, backfill, readiness
```

Sampling runs through `shared.thompson_client.ThompsonClient.select_arm`,
which calls `shared.thompson_policy.select` -- the same pure function the live
service and the GA replay share -- with the same policy file GA Target A
promotes. A promoted policy therefore applies to every domain, not just model
routing.

## Domains

Functional (a live consumer draws from them):

| domain | arms | evidence |
| --- | --- | --- |
| `crew_size` | 1-5 workers | review ledger: workers that actually ran, verdict survival vs. later reviews, total cost |
| `review_depth` | 1/2/3 phases | later reviews of the same subject that changed the verdict |
| `batch_size` | 25/50/100/250/500 | `ops/embedding_batch` rows + `ops/gateway_forward` ingest rows joined to bulk-ingest rows |
| `rag_scope` | `m{5,10}-k{1,2,3}-g{0,1}` | RAG injection journal: per-source retrieval yield and context size |

Instrumentation only (signal captured, nothing drawn yet):

| domain | arms | evidence |
| --- | --- | --- |
| `prompt_strategy` | `defects-v1`, `risk-first-v1`, `terse-v1` | `prompt_variant` on review worker rows |
| `chunking` | `window-v1`, `markdown-v1`, `message-v1` | `chunk_profile` on bulk-ingest rows |
| `tool_selection` | bash/read/edit/write/grep/glob/task/webfetch/other | `tools/tool_outcome_hook.py` PostToolUse rows |
| `graph_traversal` | 1/2/3/5 | `ops/graph_traverse` telemetry from the graph service |

## State

`~/.FlossWare/ledger/meta/<domain>.json`, mode 0600, atomic replace under an
flock (`POC_CLAUDE_ENSEMBLE_META_LEDGER_DIR` overrides the directory).

```json
{"schema_version": 1, "domain": "crew_size",
 "arms": {"3": {"successes": 12.0, "failures": 2.0, "calls": 14,
                "total_cost": 3.1, "total_reward": 10.5}},
 "outcome_ids": ["..."],
 "outcomes": [{"outcome_id": "...", "arm": "3", "outcome": 0.75,
               "success_delta": 0.75, "failure_delta": 0.25,
               "cost": 0.2, "context": {"task_type": "code-review"},
               "metadata": {"review_id": "..."}}]}
```

* A graded reward in `[0, 1]` is stored as fractional Beta counts, so the
  posterior mean equals the observed reward.
* `outcome_ids` makes every backfill idempotent: a row seen before is skipped,
  never double-counted.
* The raw outcome log is capped at 5,000 rows; aggregates and ids always cover
  every outcome ever recorded.

## CLI

```bash
PYTHONPATH=services python3 -m learning.meta                       # status
PYTHONPATH=services python3 -m learning.meta select crew_size --task-type code-review
PYTHONPATH=services python3 -m learning.meta posterior review_depth
PYTHONPATH=services python3 -m learning.meta backfill              # all domains
PYTHONPATH=services python3 -m learning.meta readiness             # rag_scope evidence
```

## Integrations

* `tools/crew_select.py --learn-size` draws the worker count.
* `tools/multi_model_review.py --learn-depth` draws 1/2/3 phases;
  `--thompson --learn-size` also folds the review ledger back into
  `crew_size` after the review.
* `tools/bulk_ingest.py --learn-batch` draws the run's item budget and records
  the run outcome (real runs only; a dry run is not evidence); ledger rows
  carry `chunk_profile`.
* `tools/tool_outcome_hook.py` journals tool completions (install it as a
  Claude Code `PostToolUse` hook; this repo does not modify user settings).
* The graph service writes `graph_traverse` telemetry; the ingestion preview
  advertises its `profiles` map.
* `learning.ga.targets.rag_config.fitness_signal_readiness()` reports whether
  the journal holds enough rows for Target B; promotion still requires an
  explicit usefulness rating, so the target remains a refusal stub.

## Honest limitations

* `rag_scope` rewards retrieval *yield*, not usefulness. It can inform a
  choice; it cannot justify promotion.
* Real embedding/ingest batches are far below 25, so every observed
  `batch_size` row maps to the 25 arm. The other arms are unobserved until
  traffic uses them.
* `review_depth` can only score a review once a later review of the same
  subject exists. A depth-1 review (no arbiter decision) is scored as a
  failure when it was re-reviewed, which is the only signal it leaves.
* `prompt_strategy`, `chunking`, `tool_selection`, and `graph_traversal`
  capture evidence but are not wired to a draw.
