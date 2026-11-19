# Prompt-time RAG hook (configurable)

`memory-search-on-prompt.js` is the repo copy of the canonical
`UserPromptSubmit` context retrieval hook. It is the deployed hook at
`<profile>/hooks/memory-search-on-prompt.js` plus two changes:

1. it **reads the documented `rag.json` configuration** at prompt time instead
   of using hardcoded constants, and
2. it **journals an exposure row** for every prompt that reaches retrieval.

Nothing here is installed automatically, and this directory does not modify a
deployed profile. Installation is a deliberate copy (below).

## Files

| File | Purpose |
|---|---|
| `memory-search-on-prompt.js` | the hook (standalone, CommonJS, no dependencies) |
| `test_rag_hook.mjs` | behavior tests, `node --test` |
| `README.md` | this install and contract note |

Tests also run under the repo runner: `python -m pytest tests/test_rag_hook.py -q`.

## Configuration

The hook reads `<profile>/.flossware-claude-config/rag.json`, the file written
by `claude-config rag-config set`. Resolution order:

1. `FLOSSWARE_CE_RAG_CONFIG` — explicit file path (tests, isolated deployments)
2. `POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR/.flossware-claude-config/rag.json`
3. `CLAUDE_CONFIG_DIR/.flossware-claude-config/rag.json`
4. `<hook dir>/../.flossware-claude-config/rag.json` (the deployed layout)

The read is bounded to 64 KiB, refuses symlinks (`O_NOFOLLOW`), and never
throws. **Any problem — missing, unreadable, symlinked, oversized, malformed,
unknown section or setting, wrong type, out-of-range value — is fail-open:**
the hook writes a one-line code to stderr (`RAG configuration <code>; using
built-in defaults`) and continues with the defaults. A prompt is never blocked
by configuration, which is the deliberate difference from the older
`tools/claude-config` README behavior (that hook skipped retrieval on an
invalid file; this one still retrieves with defaults).

### Settings

Bounds are identical to `tools/claude-config/lib/rag_config.py` (the same
values the CLI validates before writing).

| Setting | Built-in default | Valid range | Effect in this hook |
|---|---:|---:|---|
| `memory.enabled` | `true` | bool | `false` skips the `POST /memory/search-hybrid` request entirely |
| `memory.max_results` | `10` | 1–10 | `top_k` on `/memory/search-hybrid`; caps the Memory section |
| `memory.timeout_ms` | `15000` | 50–10000 | timeout for the memory search and `/memory/chunk` |
| `knowledge.enabled` | `true` | bool | `false` suppresses the Knowledge section (the hybrid request still runs for transcripts) |
| `knowledge.max_results` | `3` | 1–10 | caps the Knowledge section |
| `knowledge.max_queries` | `3` | 1–3 | accepted and validated; **inert** — this hook makes one hybrid query |
| `knowledge.timeout_ms` | `15000` | 50–10000 | timeout for the hybrid request (transcripts, knowledge, intent, related sessions) |
| `graph.enabled` | `false` | bool | accepted and validated; **inert** — this hook has no graph retrieval path |
| `graph.timeout_ms` | `400` | 50–5000 | inert |
| `graph.max_roots` | `3` | 1–3 | inert |
| `graph.max_nodes` | `8` | 1–8 | inert |
| `graph.max_context_chars` | `3000` | 100–3000 | inert |

Built-in defaults are the hook's pre-configuration runtime constants, not the
CLI defaults, so a profile with **no** `rag.json` behaves exactly as before.
They are therefore outside the documented CLI range for the two timeouts
(15000 > 10000); only values *supplied in the file* are range-checked. A
partial file overrides only the keys it names.

`FLOSSWARE_MEMORY_TIMEOUT_MS` is an **operator override that wins over
`rag.json`**: when set to an integer in 50–600000 it overrides both
`memory.timeout_ms` and `knowledge.timeout_ms`; an invalid value is ignored
(with a stderr code) instead of producing a zero timeout. The range is wider
than the `rag.json` range on purpose — the live CE profile sets it to `30000`,
above the documented 10000 ceiling for `memory.timeout_ms`, and that value must
keep working. Consequence for tuning: while this variable is set in the
service environment, the two timeout knobs are not evolvable and the journal
records the effective (overridden) value with `env_timeout_override: true`.

The whole injection is still capped at 12000 characters, then truncated with
`[Context truncated]`. That budget is not configurable today.

## Exposure journal

Every prompt that reaches retrieval appends one row to
`~/.FlossWare/ledger/rag/injections.jsonl` (directory `0700`, file `0600`),
following the ledger conventions in `~/.FlossWare/ledger/README.md`:
`schema_version`, `producer`, `kind`, and an `event_id` identity key. Override
the path with `FLOSSWARE_CE_RAG_JOURNAL`; set it to `off` to disable
journaling.

```json
{
  "schema_version": 1,
  "producer": "claude-code",
  "kind": "rag_injection",
  "event_id": "<sha256 hex>",
  "timestamp": "2026-11-19T18:02:11.412Z",
  "session_id": "9c29386c-...",
  "query_hash": "0f3a1b2c",
  "prompt_chars": 42,
  "config": {
    "source": "file",
    "error": null,
    "env_timeout_override": false,
    "sha256": "<sha256 of the effective config>",
    "memory": {"enabled": true, "max_results": 5, "timeout_ms": 5000},
    "knowledge": {"enabled": true, "max_results": 3, "max_queries": 3, "timeout_ms": 5000},
    "graph": {"enabled": false, "max_nodes": 8, "max_roots": 3, "max_context_chars": 3000, "timeout_ms": 400}
  },
  "sections": {
    "memory": {"status": "ok", "results": 5, "chars": 1180},
    "knowledge": {"status": "ok", "results": 3, "chars": 640},
    "transcript": {"status": "ok", "results": 5, "chars": 900},
    "intent": {"status": "ok", "label": "code_question", "confidence": 0.83},
    "related_sessions": {"status": "ok", "results": 2}
  },
  "context_chars": 2720,
  "truncated": false,
  "source_ids": ["memory:notes.md#Deploy", "knowledge:https://example.test/doc#0", "transcript:9c29386c#0"]
}
```

- **Sections follow the request, not the heading.** `memory` describes what the
  `POST /memory/search-hybrid` payload injected — including results whose file
  name makes the formatter print them under the Knowledge heading;
  `knowledge`, `transcript`, `intent`, and `related_sessions` describe the
  hybrid `POST /search` response.
- **Metadata only.** Prompt text and retrieved text are never written. Section
  `status` is one of `ok`, `empty`, `error`, `disabled`, `absent`,
  `below_threshold`; `config.error` is a short code, never file content;
  `source_ids` are file/url/session identifiers, whitespace-collapsed and
  capped (200 chars each, 50 ids).
- **Append-once.** `event_id` = SHA-256 of
  `rag-injection | session_id | prompt | injected context | config sha256`.
  A retried or duplicated invocation produces the same id and is not written
  twice; a different prompt, session, injected context, or effective
  configuration is a new row. Duplicate detection scans a bounded 256 KiB tail
  of the journal, so beyond that window a repeat could be appended
  (at-least-once, never lossy).
- **Never blocks the prompt.** Journal failures are reported as
  `RAG injection journal unavailable: <code>` on stderr and ignored.
- A prompt with no retrievable context still gets a row (`context_chars: 0`,
  section statuses explaining why); an empty prompt does not.

This is the exposure record only. Linking an exposure to a task outcome is not
implemented here.

## Install (deliberate, into a profile)

```bash
PROFILE="${POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR:?select the CE profile}"
install -m 0700 tools/rag_hook/memory-search-on-prompt.js \
  "$PROFILE/hooks/memory-search-on-prompt.js"
```

The hook must be registered as a **synchronous** `UserPromptSubmit` command in
the profile's `settings.json`; `async: true` discards hook output and no
context is injected. The deployed profile uses:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command",
                     "command": "<profile>/hooks/memory-search-on-prompt.js",
                     "timeout": 3 } ] }
    ]
  }
}
```

Note the `timeout` is 3 seconds: a configured `*_timeout_ms` above 3000 cannot
take effect in practice because Claude Code stops waiting first.

After installing, confirm the config the hook will actually use, run one
prompt, and read the last journal row:

```bash
POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR="$PROFILE" \
  tools/claude-config/bin/claude-config rag-config show
tail -n 1 ~/.FlossWare/ledger/rag/injections.jsonl
```

`config.source` / `config.error` in that row tell you whether `rag.json` was
applied. Roll back by restoring the previous hook file from
`<profile>/.flossware-claude-config/backups/` (or re-running
`claude-config install`), which also restores the manifest SHA-256 recorded in
`<profile>/.flossware-claude-config/manifest.json`.

## Verify locally

```bash
node --test tools/rag_hook/test_rag_hook.mjs
python -m pytest tests/test_rag_hook.py -q
```

The tests spawn the real hook against local stub Memory/hybrid services and
cover: missing/malformed/oversized/symlinked/out-of-range/unknown settings,
`POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR` resolution, each live knob taking
effect, both disabled switches, the timeout knob, journal schema and file
modes, append-once idempotency, the default `$HOME` journal location, journal
disable/failure, both services down, and unchanged `hookSpecificOutput`.
