# User satisfaction signal

`tools/satisfaction_signal.py` classifies the message a user sends **after** an
assistant response and journals whether the response was helpful. It is the
missing per-interaction quality signal for Thompson, the routing predictors,
and the GA's RAG target.

* deterministic — regular expressions and keyword rules, no model call
* metadata only — no user message, response text, or retrieved context is stored
* fail-open — the hook always exits 0, never writes stdout, and never blocks

## Classification

`classify(text, previous_text=None)` returns `SATISFIED`, `CORRECTION`, or
`UNCLEAR`, a confidence in `[0, 0.95]`, the matched rule ids, and
`response_was_helpful` (`true` / `false` / `null`).

### SATISFIED rules

| id | weight | matches |
|---|---:|---|
| `thanks` | 0.92 | `thanks`, `thank you`, `thx` |
| `praise` | 0.92 | `perfect`, `great`, `excellent`, `awesome`, `amazing`, `fantastic`, `brilliant`, `wonderful`, `nice work`, `good job`, `well done`, `love it` |
| `approval` | 0.85 | `lgtm`, `looks good`, `works for me`, `that works`, `this works`, `makes sense`, `got it` |
| `affirm_opener` | 0.80 | message starts with `yes`, `yeah`, `yep`, `yup`, `sure`, `okay`, `ok`, `alright`, `correct`, `exactly`, `right`, `agreed`, `indeed`, `fine` |
| `proceed` | 0.80 | `proceed`, `continue`, `go ahead`, `carry on`, `keep going`, `move on`, `next`, `do it`, `please do`, `go for it`, `ship it`, `sounds good`, `will do`, `on it`, `roger` |
| `no_problem_guard` | 0.85 | starts with `no problem`, `no worries`, `no need`, `no rush`, `not bad`, `no complaints`; suppresses the leading-`no` correction |
| `bare_good` | 0.85 | a one-word approval: `good`, `great`, `nice`, `cool`, `awesome` |
| `negated_proceed` | 0.78 | `nah, proceed`, `no, continue`, `no, go ahead` — declining an option, not the response; suppresses the leading-`no` correction |
| `we_good` | 0.70 | `we good`, `we're good`, `all good` |
| `imperative_continuation` | 0.50 | starts with a new instruction verb: `do`, `run`, `apply`, `use`, `make`, `add`, `fix`, `update`, `include`, `remove`, `change`, `check`, `verify`, `review` — continuation, below the Thompson floor |
| `question` | 0.45 | message ends with `?` and no correction cue — weak proxy for moving to a new topic; journaled but below the Thompson floor |
| `neutral_continuation` | 0.40 | ≥ 15 characters with no cue at all — weak continuation proxy, below the Thompson floor |

### CORRECTION rules

| id | weight | matches |
|---|---:|---|
| `interrupted` | 0.95 | transcript marker `[Request interrupted by user` |
| `not_that` | 0.92 | `not that`, `that's not`, `that is not`, `not what I/we/you`, `isn't what`, `wasn't what` |
| `meant` | 0.90 | `I meant`, `I said`, `I asked`, `as I said`, `like I said`, `what I meant` |
| `negation_opener` | 0.88 | starts with `no`, `nope`, `nah`, `wrong`, `incorrect` |
| `missed` | 0.88 | `you missed`, `you forgot`, `you didn't`, `you ignored`, `missed the`, `forgot to` |
| `failure` | 0.85 | response/attempt failed: `doesn't work`, `didn't work`, `won't work`, `not working`, `still broken`, `still failing`, `it/that/this failed`, `failed again`, `didn't make it`, `the build/test/pipeline/deploy failed`, `build failure`, `broken` |
| `failure_mention` | 0.50 | a bare mention of `fail(s)`, `failed`, `failure`, `crash` — often a topic in a new instruction or question; below the Thompson floor |
| `wrong` | 0.85 | `wrong`, `incorrect`, `mistaken`, `that's a mistake` |
| `retry` | 0.82 | `try again`, `redo`, `run it again`, `one more time`, `start over` |
| `frustration` | 0.80 | `wtf`, `what the hell`, `seriously`, `come on` |
| `not_ok` | 0.80 | `not ok`, `isn't fine`, `not good` |
| `stop_interjection` | 0.85 | `stop -`, `stop,`, `stop:` mid-message: "OK stop - what I mean is ..." |
| `already_asked` | 0.80 | `already asked`, `I asked`, `asked for it/this/that` |
| `confusion` | 0.75 | `confused`, `I don't understand`, `not sure what`, `what are you building`, `why are you` |
| `restated_request` | 0.58 | ≥ 12 characters with Jaccard ≥ 0.70 content-word overlap with the previous user message — a weak re-ask (ambiguous; below the Thompson floor) |
| `wait` | 0.55 | starts with `wait`, `hold on`, `hang on`, or a bare `stop`/`stop.` |
| `actually` | 0.55 | starts with `actually`, `instead`, `rather`; weak because it often introduces a new request |

### Resolution

1. Run every rule over the whitespace-normalised message.
2. A strong correction (`≥ 0.80`) wins over any satisfied rule: `"yes, but it
   doesn't work"` is a correction.
3. Otherwise the stronger side wins; the confidence is that rule's weight,
   minus 0.10 when the other side matched within 0.15 (ambiguous).
4. Equal weak matches are `UNCLEAR`. No matches: question → `SATISFIED` 0.45,
   ≥ 15 characters → `SATISFIED` 0.40, otherwise `UNCLEAR`.

Reaction rules (`failure`, `failure_mention`, `wrong`, `retry`, `missed`,
`meant`, `not_that`, `already_asked`, `confusion`, `frustration`, `not_ok`,
`stop_interjection`, `reassurance`) search only the first 200 characters, or
the first 120 characters when the message is longer than 200. Long messages
are usually instructions or pasted logs, and words like "failed" inside them
are not user pushback.

Three additional guards keep real-world instructions from looking like
pushback:

* **Question forms.**  A trailing `?` or an interrogative opener (`what`,
  `why`, `how`, `did`, `does`, `is`, `can`, ...) marks a question.  If the
  only correction cue is a bare failure mention or `wrong`, the message is a
  status question (`SATISFIED` 0.45), not a correction.
* **Numbered/bulleted replies** (`1 ...`, `1) ...`, `- ...`) are answers or
  requirement lists; strong `failure`/`wrong` rules are skipped and only the
  weak failure mention applies.
* **Opening praise plus a later reaction word.**  If a strong satisfaction
  rule (`thanks`, `praise`, `approval`, `proceed`, `no_problem_guard`,
  `bare_good`, `we_good`) matches in the first 40 characters, a later reaction
  word is ignored unless the message contrasts it with `but`, `however`,
  `though`, `although`, `except`, `still`, `one issue`, `one problem`,
  `only issue`, or `only problem`, or the correction is unambiguous
  (`not_that`, `meant`, `missed`, `already_asked`, `not_ok`, `confusion`,
  `frustration`, `interrupted`).  `"perfect - that did work! now how do we
  test where we fail?"` is satisfaction; `"thanks, but the build failed"` is a
  correction.

`_human_text` rejects tool results, `isMeta` records, sidechains, and harness
automation: system reminders, task notifications, command wrappers, and
subagent hand-backs (`Another Claude session sent a message:` /
`<agent-message>`). Model output is never treated as user feedback.

`UNCLEAR` always has `response_was_helpful: null` and is never fed to Thompson.

## Journal

`~/.FlossWare/ledger/satisfaction/satisfaction.jsonl` (override
`POC_CLAUDE_ENSEMBLE_SATISFACTION_LEDGER`), directory `0700`, file `0600`,
append-once by `event_id`.

```json
{
  "schema_version": 1,
  "producer": "satisfaction-signal",
  "kind": "satisfaction",
  "event_id": "<sha256>",
  "ts": "2026-11-19T23:36:38.273Z",
  "session_id": "9c29386c-...",
  "classification": "SATISFIED",
  "confidence": 0.92,
  "trigger_patterns": ["praise", "thanks"],
  "response_was_helpful": true,
  "is_question": false,
  "message_chars": 15,
  "message_ts": "2026-11-19T10:00:20.000Z",
  "message_id": "<user message uuid>",
  "judged_response_message_id": "<assistant message id>",
  "model": "rits/zai-org/glm-5-3",
  "thompson_arm": "glm-5-3",
  "hook_event": "Stop",
  "rag_injection": {
    "event_id": "<injection event id>",
    "timestamp": "2026-11-19T10:00:01.000Z",
    "delta_seconds": 1.0,
    "direction": "after",
    "context_chars": 111,
    "source_count": 1,
    "truncated": false
  },
  "rag_link_status": "linked",
  "rag_anchor_ts": "2026-11-19T10:00:00.000Z"
}
```

`event_id` = SHA-256 of `satisfaction | session | user-message uuid (or text
hash) | assistant message id`. A repeated hook invocation for the same turn
produces the same id and is not appended twice (bounded 256 KiB tail scan, so
at-least-once rather than lossy).

## Which message / which response / which injection

At `Stop` of turn *N* the transcript ends with
`... assistant(N-1), user(N), assistant(N)`.

* `user(N)` is classified: it is the first thing the user said after
  `assistant(N-1)`, so it is the satisfaction signal for that response.
* The model of `assistant(N-1)` is attributed.
* The RAG injection that conditioned `assistant(N-1)` was journaled next to
  `user(N-1)`, so the link anchors on the **previous** human prompt timestamp
  and takes the nearest injection row for the same session within 60 s
  (`POC_CLAUDE_ENSEMBLE_SATISFACTION_RAG_WINDOW`). Linking to the classified
  prompt's own injection would be one turn ahead.

## Hook

Recommended registration (deliberate; this repo does not edit settings):

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command",
                     "command": "python3 /path/to/tools/satisfaction_signal.py hook",
                     "timeout": 5 } ] }
    ]
  }
}
```

`PostToolUse` also works (the event id deduplicates repeat firings), but one
`Stop` firing per turn is cleaner. The hook:

* reads only the last 1 MiB of the transcript (measured ~50 ms wall clock on a
  14 MB transcript, including interpreter start)
* skips tool results, system reminders, task notifications, meta and sidechain
  records
* skips when `stop_hook_active` is true
* performs no network I/O; a detached child feeds Thompson

## Thompson

`SATISFIED` → `success: true`; `CORRECTION` → `success: false`; one outcome per
row, `task_type: satisfaction_signal`, `cost: 0`, `tokens: 0`, request id = the
satisfaction `event_id`.

A precision gate decides which journal rows actually score a model, because a
keyword detector is not accurate enough to count every ambiguous reply:

* **Positives** need confidence ≥ 0.6
  (`POC_CLAUDE_ENSEMBLE_SATISFACTION_MIN_CONFIDENCE`).
* **Negatives** need confidence ≥ 0.8
  (`POC_CLAUDE_ENSEMBLE_SATISFACTION_CORRECTION_MIN_CONFIDENCE`), must not be a
  question, and must name a concrete problem: `not_that`, `meant`, `failure`,
  `wrong`, `retry`, `missed`, `already_asked`, `confusion`, `frustration`,
  `not_ok`, or `stop_interjection`.

A bare leading "no", a restated request, a weak `actually`/`wait`, and an
interruption stay in the journal but never count against a model.  On a
613-turn real-transcript audit this gate selected 107 positive and 7 negative
outcomes; the seven negatives were all concrete failure, wrong, retry, or intent-mismatch reports.
`UNCLEAR` rows and models that do not map to a Thompson arm are skipped.

Successful posts are recorded in
`~/.FlossWare/ledger/satisfaction/thompson-sync.jsonl`, so a later sync never
double-counts; failed posts are retried. Disable the feed with
`POC_CLAUDE_ENSEMBLE_SATISFACTION_THOMPSON=0`.

This is a deliberate reversal of the earlier decision in
`services/embeddings/turn_outcomes.py` to keep keyword pushback out of turn
`success`: that decision was about objective turn success, while this is an
explicit, confidence-gated quality observation. Keep the floor high; weak
`actually` / `wait` / restatement signals sit below it on purpose.

## RAG outcome linkage (GA Target B)

```bash
python3 tools/satisfaction_signal.py rag-outcomes
# -> ~/.FlossWare/ledger/satisfaction/rag-outcomes.jsonl
```

This joins each satisfaction row to its linked injection row and adds the
`helped` label (`true` for `SATISFIED`, `false` for `CORRECTION`; the latest
classification wins). Injection metadata is preserved. Point the existing
trainer at it:

```bash
FLOSSWARE_CE_RAG_JOURNAL=~/.FlossWare/ledger/satisfaction/rag-outcomes.jsonl \
  python3 -m learning.train --model retrieval_relevance
```

`learning/train/retrieval_relevance.py` already reads `helped`, so the linkage
unblocks its training signal. `learning/ga/targets/rag_config.py` still refuses
promotion on purpose; this tool supplies evidence, it does not change that
policy.

## CLI

```bash
# one-off classification
python3 tools/satisfaction_signal.py classify --text "thanks, perfect" --json

# pattern documentation
python3 tools/satisfaction_signal.py patterns

# feed unsent journal rows to Thompson (also spawned by the hook)
python3 tools/satisfaction_signal.py thompson-sync [--dry-run]
```

## Tests

```bash
python3 -m pytest tests/test_satisfaction_signal.py -q
```

Covers labelled-message accuracy, every documented pattern, journal schema and
file modes, append-once idempotency, bounded-tail performance, fail-open
behavior, RAG window/session/anchor selection, the derived `helped` ledger, and
Thompson positive/negative/retry/dedupe behavior.
