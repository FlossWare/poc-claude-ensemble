/**
 * Tests for the configurable prompt-time RAG hook.
 *
 *   node --test tools/rag_hook/test_rag_hook.mjs
 *
 * Every test spawns the real hook as a child process against a local stub of
 * the Memory and hybrid services, so configuration parsing, fail-open
 * behavior, and the exposure journal are exercised end to end.
 */

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const HOOK = path.join(path.dirname(fileURLToPath(import.meta.url)), 'memory-search-on-prompt.js');
const PROMPT = 'Explain CE prompt retrieval behavior';
const SESSION = '9c29386c-1111-2222-3333-444455556666';

const MEMORY_RESULTS = [
  { file: 'notes-alpha.md', section: 'Alpha', source: 'memory', content: 'MEMORY_ALPHA the deployment runbook lives here.' },
  { file: 'notes-beta.md', section: 'Beta', source: 'memory', content: 'MEMORY_BETA the rollback steps are documented.' },
  { file: 'notes-gamma.md', section: 'Gamma', source: 'memory', content: 'MEMORY_GAMMA the oncall rotation is weekly.' },
];
const TRANSCRIPTS = [1, 2, 3].map((index) => ({
  session_id: `aaaaaaaa-0000-0000-0000-00000000000${index}`,
  project: 'demo',
  role: 'user',
  text: `TRANSCRIPT_${index} we discussed the rollout order.`,
}));
const KNOWLEDGE = [1, 2, 3].map((index) => ({
  title: `Book ${index}`,
  url: `https://example.test/book-${index}`,
  origin: 'books',
  text: `KNOWLEDGE_${index} the runbook describes the rollout.`,
}));

const INHERITED_VARS = [
  'FLOSSWARE_CE_RAG_CONFIG',
  'FLOSSWARE_CE_RAG_JOURNAL',
  'FLOSSWARE_MEMORY_URL',
  'FLOSSWARE_HYBRID_URL',
  'FLOSSWARE_MEMORY_TIMEOUT_MS',
  'POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR',
  'CLAUDE_CONFIG_DIR',
  'CLAUDE_PROMPT',
];

function baseEnv() {
  const env = { ...process.env };
  for (const name of INHERITED_VARS) delete env[name];
  return env;
}

async function startStub(options = {}) {
  const requests = [];
  const server = http.createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => { body += chunk; });
    request.on('end', () => {
      let parsed = {};
      try { parsed = body ? JSON.parse(body) : {}; } catch { parsed = {}; }
      requests.push({ path: request.url, body: parsed });
      const send = (payload) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify(payload));
      };
      if (request.url === '/memory/search-hybrid') {
        const topK = Number.isInteger(parsed.top_k) ? parsed.top_k : 10;
        const results = (options.memoryResults ?? MEMORY_RESULTS).slice(0, topK);
        const reply = () => send({ ok: true, results });
        if (options.memoryDelayMs) setTimeout(reply, options.memoryDelayMs);
        else reply();
        return;
      }
      if (request.url === '/memory/chunk') {
        send({ ok: true, chunks: [] });
        return;
      }
      if (request.url === '/search') {
        send({
          results: options.transcripts ?? TRANSCRIPTS,
          knowledge: options.knowledge ?? KNOWLEDGE,
          intent: options.intent ?? null,
          related_sessions: options.relatedSessions ?? [],
        });
        return;
      }
      response.writeHead(404);
      response.end();
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    requests,
    close: () => {
      server.closeAllConnections?.();
      return new Promise((resolve) => server.close(resolve));
    },
  };
}

function workspace(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'ce-rag-hook-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return {
    directory,
    configPath: path.join(directory, 'rag.json'),
    journalPath: path.join(directory, 'ledger', 'rag', 'injections.jsonl'),
    writeConfig(config) {
      fs.writeFileSync(this.configPath, typeof config === 'string' ? config : JSON.stringify(config));
      return this.configPath;
    },
  };
}

function runHook({ env = {}, event = { prompt: PROMPT, session_id: SESSION }, stdin } = {}) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const child = spawn(process.execPath, [HOOK], {
      env: { ...baseEnv(), ...env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8').on('data', (chunk) => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
    child.once('error', reject);
    child.once('close', (exitCode) => {
      const output = stdout.trim();
      let json = null;
      try { json = output ? JSON.parse(output) : null; } catch { json = null; }
      resolve({
        exitCode,
        stdout: output,
        stderr,
        json,
        context: json?.hookSpecificOutput?.additionalContext ?? '',
        elapsedMs: Date.now() - started,
      });
    });
    child.stdin.end(stdin ?? JSON.stringify(event));
  });
}

function readRows(file) {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line));
}

async function withStub(t, options) {
  const stub = await startStub(options);
  t.after(() => stub.close());
  return stub;
}

// --- configuration parsing -------------------------------------------------

test('missing config falls back to built-in defaults and still injects', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.match(result.context, /MEMORY_ALPHA/);
  assert.match(result.stderr, /RAG configuration missing; using built-in defaults/);
  assert.equal(stub.requests.find((r) => r.path === '/memory/search-hybrid').body.top_k, 10);

  const rows = readRows(space.journalPath);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].config.source, 'defaults');
  assert.equal(rows[0].config.error, 'missing');
  assert.equal(rows[0].config.memory.max_results, 10);
  assert.equal(rows[0].config.knowledge.max_results, 3);
});

test('malformed config fails open to defaults without blocking the prompt', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  space.writeConfig('{"memory": {"max_results": 2');

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.match(result.context, /MEMORY_ALPHA/);
  assert.match(result.stderr, /RAG configuration malformed_json/);
  assert.equal(stub.requests.find((r) => r.path === '/memory/search-hybrid').body.top_k, 10);
  assert.equal(readRows(space.journalPath)[0].config.error, 'malformed_json');
});

test('out-of-range, unknown, and wrong-typed settings fall back to defaults', async (t) => {
  const stub = await withStub(t);
  const cases = [
    [{ memory: { max_results: 99 } }, 'out_of_range:memory.max_results'],
    [{ memory: { max_results: 0 } }, 'out_of_range:memory.max_results'],
    [{ memory: { timeout_ms: 10 } }, 'out_of_range:memory.timeout_ms'],
    [{ knowledge: { max_queries: 11 } }, 'out_of_range:knowledge.max_queries'],
    [{ memory: { top_k: 3 } }, 'unknown_setting:memory.top_k'],
    [{ retrieval: {} }, 'unknown_section:retrieval'],
    [{ memory: { enabled: 'sometimes' } }, 'invalid_type:memory.enabled'],
    [{ memory: { max_results: 2.5 } }, 'out_of_range:memory.max_results'],
  ];

  for (const [config, expected] of cases) {
    const space = workspace(t);
    space.writeConfig(config);
    const result = await runHook({
      env: {
        FLOSSWARE_CE_RAG_CONFIG: space.configPath,
        FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
        FLOSSWARE_MEMORY_URL: stub.url,
        FLOSSWARE_HYBRID_URL: stub.url,
      },
    });
    assert.equal(result.exitCode, 0, result.stderr);
    assert.match(result.context, /MEMORY_ALPHA/, JSON.stringify(config));
    const rows = readRows(space.journalPath);
    assert.equal(rows.length, 1, JSON.stringify(config));
    assert.equal(rows[0].config.error, expected, JSON.stringify(config));
    assert.equal(rows[0].config.source, 'defaults', JSON.stringify(config));
  }
  assert.equal(stub.requests.filter((r) => r.path === '/memory/search-hybrid').every((r) => r.body.top_k === 10), true);
});

test('oversized config file is refused and defaults apply', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  const padding = 'x'.repeat(70 * 1024);
  space.writeConfig(JSON.stringify({ memory: { max_results: 1 }, pad: padding }));

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(readRows(space.journalPath)[0].config.error, 'oversize');
  assert.equal(stub.requests.find((r) => r.path === '/memory/search-hybrid').body.top_k, 10);
});

test('symlinked config file is refused', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  const real = path.join(space.directory, 'real-rag.json');
  fs.writeFileSync(real, JSON.stringify({ memory: { max_results: 1 } }));
  fs.symlinkSync(real, space.configPath);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(readRows(space.journalPath)[0].config.error, 'symlink');
});

test('profile config directory is resolved from POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  const profile = path.join(space.directory, 'profile');
  fs.mkdirSync(path.join(profile, '.flossware-claude-config'), { recursive: true });
  fs.writeFileSync(
    path.join(profile, '.flossware-claude-config', 'rag.json'),
    JSON.stringify({ memory: { max_results: 1 } }),
  );

  const result = await runHook({
    env: {
      POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR: profile,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  const rows = readRows(space.journalPath);
  assert.equal(rows[0].config.source, 'file');
  assert.equal(rows[0].config.error, null);
  assert.equal(rows[0].config.memory.max_results, 1);
  assert.equal(stub.requests.find((r) => r.path === '/memory/search-hybrid').body.top_k, 1);
});

// --- documented knobs actually take effect ---------------------------------

test('memory.max_results and knowledge.max_results change what is injected', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  space.writeConfig({ memory: { max_results: 2 }, knowledge: { max_results: 1 } });

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(stub.requests.find((r) => r.path === '/memory/search-hybrid').body.top_k, 2);
  assert.match(result.context, /MEMORY_ALPHA/);
  assert.match(result.context, /MEMORY_BETA/);
  assert.doesNotMatch(result.context, /MEMORY_GAMMA/);
  assert.match(result.context, /\[K1\]/);
  assert.doesNotMatch(result.context, /\[K2\]/);

  const row = readRows(space.journalPath)[0];
  assert.equal(row.config.source, 'file');
  assert.equal(row.config.memory.max_results, 2);
  assert.equal(row.config.knowledge.max_results, 1);
  assert.equal(row.sections.memory.results, 2);
  assert.equal(row.sections.knowledge.results, 1);
});

test('memory.enabled=false skips the memory request but keeps knowledge', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  space.writeConfig({ memory: { enabled: false } });

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(stub.requests.some((r) => r.path === '/memory/search-hybrid'), false);
  assert.doesNotMatch(result.context, /MEMORY_ALPHA/);
  assert.match(result.context, /KNOWLEDGE_1/);

  const row = readRows(space.journalPath)[0];
  assert.equal(row.sections.memory.status, 'disabled');
  assert.equal(row.sections.memory.results, 0);
  assert.equal(row.sections.knowledge.status, 'ok');
});

test('knowledge.enabled=false suppresses the knowledge section only', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  space.writeConfig({ knowledge: { enabled: false } });

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.doesNotMatch(result.context, /KNOWLEDGE_1/);
  assert.match(result.context, /TRANSCRIPT_1/);
  assert.match(result.context, /MEMORY_ALPHA/);

  const row = readRows(space.journalPath)[0];
  assert.equal(row.sections.knowledge.status, 'disabled');
  assert.equal(row.sections.knowledge.results, 0);
  assert.equal(row.sections.transcript.status, 'ok');
});

test('memory.timeout_ms takes effect against a slow service', async (t) => {
  const stub = await withStub(t, { memoryDelayMs: 2000 });
  const space = workspace(t);
  space.writeConfig({ memory: { timeout_ms: 100 } });

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.ok(result.elapsedMs < 1800, `elapsed ${result.elapsedMs}ms`);
  assert.match(result.stderr, /Memory search unavailable/);
  assert.match(result.context, /KNOWLEDGE_1/);

  const row = readRows(space.journalPath)[0];
  assert.equal(row.config.memory.timeout_ms, 100);
  assert.equal(row.sections.memory.status, 'error');
});

test('valid FLOSSWARE_MEMORY_TIMEOUT_MS overrides the file; an invalid one is ignored', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  space.writeConfig({ memory: { timeout_ms: 5000 } });

  const overridden = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
      FLOSSWARE_MEMORY_TIMEOUT_MS: '900',
    },
  });
  assert.equal(overridden.exitCode, 0, overridden.stderr);
  let row = readRows(space.journalPath).at(-1);
  assert.equal(row.config.env_timeout_override, true);
  assert.equal(row.config.memory.timeout_ms, 900);
  assert.equal(row.config.knowledge.timeout_ms, 900);

  const ignored = await runHook({
    env: {
      FLOSSWARE_CE_RAG_CONFIG: space.configPath,
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
      FLOSSWARE_MEMORY_TIMEOUT_MS: 'not-a-number',
    },
  });
  assert.equal(ignored.exitCode, 0, ignored.stderr);
  assert.match(ignored.stderr, /env_out_of_range:FLOSSWARE_MEMORY_TIMEOUT_MS/);
  row = readRows(space.journalPath).at(-1);
  assert.equal(row.config.env_timeout_override, false);
  assert.equal(row.config.memory.timeout_ms, 5000);
});

test('the live profile value FLOSSWARE_MEMORY_TIMEOUT_MS=30000 is honored', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
      FLOSSWARE_MEMORY_TIMEOUT_MS: '30000',
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.doesNotMatch(result.stderr, /env_out_of_range/);
  const row = readRows(space.journalPath)[0];
  assert.equal(row.config.env_timeout_override, true);
  assert.equal(row.config.memory.timeout_ms, 30000);
  assert.equal(row.config.knowledge.timeout_ms, 30000);
});

test('memory results that look like knowledge documents still count as memory', async (t) => {
  const stub = await withStub(t, {
    memoryResults: [
      { file: 'claude-code-notes.md', section: 'Deploy', source: 'claude-code', content: 'MEMORY_DOC the runbook lives here.' },
    ],
  });
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.match(result.context, /MEMORY_DOC/);
  const row = readRows(space.journalPath)[0];
  assert.equal(row.sections.memory.status, 'ok');
  assert.equal(row.sections.memory.results, 1);
  assert.ok(row.sections.memory.chars > 0);
  assert.ok(row.source_ids.includes('memory:claude-code-notes.md#Deploy'));
});

// --- injection journal -----------------------------------------------------

test('journal row follows the ledger schema and records no text', async (t) => {
  const stub = await withStub(t, {
    intent: { label: 'code_question', confidence: 0.83 },
    relatedSessions: [{ session_id: 'bbbbbbbb-1111-2222-3333-444455556666', project: 'demo', touched: ['a.py'] }],
  });
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  const rows = readRows(space.journalPath);
  assert.equal(rows.length, 1);
  const row = rows[0];

  assert.equal(row.schema_version, 1);
  assert.equal(row.producer, 'claude-code');
  assert.equal(row.kind, 'rag_injection');
  assert.match(row.event_id, /^[0-9a-f]{64}$/);
  assert.match(row.timestamp, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  assert.equal(row.session_id, SESSION);
  assert.match(row.query_hash, /^[0-9a-f]{8}$/);
  assert.equal(row.prompt_chars, PROMPT.length);

  assert.equal(row.sections.memory.status, 'ok');
  assert.equal(row.sections.memory.results, 3);
  assert.ok(row.sections.memory.chars > 0);
  assert.equal(row.sections.knowledge.results, 3);
  assert.equal(row.sections.transcript.results, 3);
  assert.equal(row.sections.intent.status, 'ok');
  assert.equal(row.sections.intent.label, 'code_question');
  assert.equal(row.sections.intent.confidence, 0.83);
  assert.equal(row.sections.related_sessions.results, 1);
  assert.ok(row.context_chars > 0);
  assert.equal(row.truncated, false);

  assert.deepEqual(row.source_ids.slice(0, 3), [
    'memory:notes-alpha.md#Alpha',
    'memory:notes-beta.md#Beta',
    'memory:notes-gamma.md#Gamma',
  ]);
  assert.ok(row.source_ids.includes('knowledge:https://example.test/book-1#0'));
  assert.ok(row.source_ids.includes('transcript:aaaaaaaa#0'));

  const serialized = JSON.stringify(row);
  for (const forbidden of ['MEMORY_ALPHA', 'TRANSCRIPT_1', 'KNOWLEDGE_1', PROMPT, 'the deployment runbook']) {
    assert.equal(serialized.includes(forbidden), false, `journal leaked ${forbidden}`);
  }

  assert.equal(fs.statSync(space.journalPath).mode & 0o777, 0o600);
  assert.equal(fs.statSync(path.dirname(space.journalPath)).mode & 0o777, 0o700);
});

test('journal appends once per identical injection event', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);
  const env = {
    FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
    FLOSSWARE_MEMORY_URL: stub.url,
    FLOSSWARE_HYBRID_URL: stub.url,
  };

  await runHook({ env });
  await runHook({ env });
  let rows = readRows(space.journalPath);
  assert.equal(rows.length, 1, 'a retried identical injection must not duplicate');
  assert.deepEqual(new Set(rows.map((row) => row.event_id)).size, 1);

  await runHook({ env, event: { prompt: `${PROMPT} again`, session_id: SESSION } });
  rows = readRows(space.journalPath);
  assert.equal(rows.length, 2, 'a different prompt is a new exposure');

  await runHook({ env, event: { prompt: PROMPT, session_id: 'different-session' } });
  rows = readRows(space.journalPath);
  assert.equal(rows.length, 3, 'a different session is a new exposure');
});

test('the default journal location is under $HOME/.FlossWare/ledger/rag', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);

  const result = await runHook({
    env: {
      HOME: space.directory,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  const expected = path.join(space.directory, '.FlossWare', 'ledger', 'rag', 'injections.jsonl');
  assert.equal(fs.existsSync(expected), true);
  assert.equal(readRows(expected).length, 1);
});

test('journaling can be disabled and a failing journal never blocks the prompt', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);

  const disabled = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: 'off',
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });
  assert.equal(disabled.exitCode, 0, disabled.stderr);
  assert.match(disabled.context, /MEMORY_ALPHA/);
  assert.equal(fs.existsSync(space.journalPath), false);

  // A regular file where the journal directory should be makes the write fail.
  const blockedParent = path.join(space.directory, 'blocked');
  fs.writeFileSync(blockedParent, 'not a directory');
  const broken = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: path.join(blockedParent, 'injections.jsonl'),
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });
  assert.equal(broken.exitCode, 0, broken.stderr);
  assert.match(broken.context, /MEMORY_ALPHA/);
  assert.match(broken.stderr, /RAG injection journal unavailable: [A-Z]+/);
});

test('malformed payload entries and a huge intent label are tolerated', async (t) => {
  const stub = await withStub(t, {
    knowledge: [null, 'raw string', { title: 'Book ok', url: 'https://example.test/ok', text: 'KNOWLEDGE_OK the runbook.' }],
    transcripts: [null, 42, { session_id: 'cccccccc-0000-0000-0000-000000000000', role: 'user', text: 'TRANSCRIPT_OK the rollout.' }],
    intent: { label: 'x'.repeat(300), confidence: 0.9 },
  });
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.match(result.context, /KNOWLEDGE_OK/);
  assert.match(result.context, /TRANSCRIPT_OK/);
  const row = readRows(space.journalPath)[0];
  assert.equal(row.sections.knowledge.results, 1);
  assert.equal(row.sections.transcript.results, 1);
  assert.equal(row.sections.intent.label.length, 64);
});

// --- fail-open -------------------------------------------------------------

test('an unavailable memory service still returns knowledge and journals status', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: 'http://127.0.0.1:1',
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.match(result.stderr, /Memory search unavailable/);
  assert.match(result.context, /KNOWLEDGE_1/);
  assert.doesNotMatch(result.context, /MEMORY_ALPHA/);
  assert.equal(readRows(space.journalPath)[0].sections.memory.status, 'error');
});

test('both services unavailable yields no injection, exit 0, and a zero row', async (t) => {
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: 'http://127.0.0.1:1',
      FLOSSWARE_HYBRID_URL: 'http://127.0.0.1:1',
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(result.stdout, '');
  const row = readRows(space.journalPath)[0];
  assert.equal(row.context_chars, 0);
  assert.equal(row.sections.memory.status, 'error');
  assert.equal(row.sections.knowledge.status, 'error');
  assert.equal(row.sections.transcript.status, 'error');
  assert.deepEqual(row.source_ids, []);
});

test('unparseable hook input and an empty prompt are ignored without journaling', async (t) => {
  const space = workspace(t);
  const env = { FLOSSWARE_CE_RAG_JOURNAL: space.journalPath };

  const bad = await runHook({ env, stdin: 'not json at all' });
  assert.equal(bad.exitCode, 0, bad.stderr);
  assert.equal(bad.stdout, '');

  const empty = await runHook({ env, event: { prompt: '   ', session_id: SESSION } });
  assert.equal(empty.exitCode, 0, empty.stderr);
  assert.equal(empty.stdout, '');

  assert.deepEqual(readRows(space.journalPath), []);
});

test('the injected output keeps the hookSpecificOutput contract', async (t) => {
  const stub = await withStub(t);
  const space = workspace(t);

  const result = await runHook({
    env: {
      FLOSSWARE_CE_RAG_JOURNAL: space.journalPath,
      FLOSSWARE_MEMORY_URL: stub.url,
      FLOSSWARE_HYBRID_URL: stub.url,
    },
  });

  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(result.json.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
  assert.equal(typeof result.json.hookSpecificOutput.additionalContext, 'string');
  assert.equal(result.json.hookSpecificOutput.contextRetrieval.memoryCount, 3);
  assert.equal(result.json.hookSpecificOutput.contextRetrieval.knowledgeCount, 3);
  assert.equal(result.json.hookSpecificOutput.contextRetrieval.configSource, 'defaults');
  assert.match(result.json.hookSpecificOutput.contextRetrieval.injectionEventId, /^[0-9a-f]{64}$/);
});
