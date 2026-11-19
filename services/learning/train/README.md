# `learning.train` — real (gradient-based) models

Three CPU-only scikit-learn models trained on the local ledgers, with a
chronological train/test split and honest generalization metrics. This is
separate from `learning.ga` (hyper-parameter search).

| model | estimator | target | label source |
|---|---|---|---|
| `quality_predictor` | `GradientBoostingClassifier` | arbiter SELECTED / usable output | review ledger |
| `retrieval_relevance` | `LogisticRegression` | did RAG context help? (`helped`, nullable) | injection journal |
| `task_classifier` | `LinearSVC` (balanced) | task type | weak keyword rules over prompts |

## CLI

```bash
PYTHONPATH=services python3 -m learning.train                 # status (read-only)
PYTHONPATH=services python3 -m learning.train train            # retrain where warranted
PYTHONPATH=services python3 -m learning.train train --force    # ignore the trigger
PYTHONPATH=services python3 -m learning.train predict task "review the auth diff"
PYTHONPATH=services python3 -m learning.train predict quality '{"model_id": "...", ...}'
```

Artifacts default to `~/.FlossWare/learning/train/*.joblib` (0600) and can be
redirected with `POC_CLAUDE_ENSEMBLE_TRAIN_DIR`. Each artifact carries a
version stamp: training-data hash, scikit-learn version, ISO timestamp,
train/test sizes, label source, and metrics.

## Retraining

`RetrainTrigger` refuses to refit until **10 new rows** have arrived and skips
entirely when the data hash is unchanged, so repeated runs are idempotent.

## Label honesty

* **Quality predictor.** The authoritative label is the arbiter's `SELECTED:`
  line. The live ledger written before 2026-11-19 contains **no** such lines,
  so with fewer than 12 authoritative rows the module falls back to a
  documented *usable-output proxy* (non-empty text, no error) and stamps
  `label_source=usable_output_proxy`. The proxy is not arbiter selection.
* **Task classifier.** The captured streams have no human task labels, so
  labels come from keyword rules (`label_source=heuristic_keyword_rules`).
  Replacing them with accepted/rejected routing outcomes is a follow-up.
* **Retrieval relevance.** The RAG hook journals features but not outcomes.
  `helped` is nullable by design; when it is null the model reports
  `insufficient labeled data` rather than inventing labels. Follow-up: derive
  `helped` from next-turn analysis (a correction or re-ask).

## Integrations

* `tools/ledger_to_thompson.py` loads `quality_predictor.joblib` when present
  and records a cost-normalized bonus for workers scored at/above
  `POC_CLAUDE_ENSEMBLE_QUALITY_BONUS_THRESHOLD` (default `0.5`), complementing
  the binary arbiter-SELECTED bonus.
* `tools/crew_select.py` consults `task_classifier.joblib` when confidence is
  at least `0.35`, then falls back to its keyword lookup. Unmapped classes
  (`other`, `implementation`) also fall through to keywords.
* `QualityPredictor.cold_start_prior` / `quality_predictor.cold_start_prior`
  give a clipped feature-based prior for an arm with no review history.

Every integration is fail-open: a missing or corrupt artifact leaves the
existing behavior unchanged.

## Tests

```bash
PYTHONPATH=services python3 -m pytest services/learning/train -q
```
