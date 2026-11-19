#!/usr/bin/env node

/**
 * Canonical Claude Code context retrieval hook.
 *
 * UserPromptSubmit is the single prompt-time context retrieval path.
 * Retrieval is relevance-driven: the complete user prompt is the query.
 *
 * The Memory service is shared infrastructure, but results are separated
 * into two contracts:
 *   - Memory: episodic/experience records, what happened.
 *   - Knowledge: durable Claude Code documents, what is currently known.
 *
 * The hook is non-blocking and fail-open. It deliberately does not inspect
 * magic prompt keywords such as "remember" or "recall".
 *
 * Prompt-time configuration
 * -------------------------
 * The hook reads its documented profile-scoped configuration from
 * <profile>/.flossware-claude-config/rag.json (the file written by
 * `claude-config rag-config set`).  Resolution order:
 *
 *   1. FLOSSWARE_CE_RAG_CONFIG            (explicit file path)
 *   2. POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR/.flossware-claude-config/rag.json
 *   3. CLAUDE_CONFIG_DIR/.flossware-claude-config/rag.json
 *   4. <hook dir>/../.flossware-claude-config/rag.json
 *
 * A missing, symlinked, oversized, malformed, or out-of-range file never
 * blocks a prompt: the hook reports a short error code on stderr and falls
 * back to the built-in defaults, which are the values this hook used before
 * configuration existed.  FLOSSWARE_MEMORY_TIMEOUT_MS, when set to a valid
 * value, overrides memory.timeout_ms and knowledge.timeout_ms.
 *
 * Exposure journal
 * ----------------
 * Every prompt that reaches retrieval appends one metadata-only row to
 * ~/.FlossWare/ledger/rag/injections.jsonl (mode 0600): which sources were
 * injected, how many results and characters per section, and source ids.
 * Prompt text and retrieved text are never recorded.  Rows are append-once:
 * an identical event (same session, prompt, injected context, and effective
 * configuration) is written once.  Override the path with
 * FLOSSWARE_CE_RAG_JOURNAL; set it to "off" to disable journaling.
 */

const nodeModule = (name) => (
  typeof process.getBuiltinModule === 'function' ? process.getBuiltinModule(name) : require(name)
);
const fs = nodeModule('fs');
const path = nodeModule('path');
const os = nodeModule('os');
const crypto = nodeModule('crypto');

const MEMORY_URL = process.env.FLOSSWARE_MEMORY_URL || 'http://127.0.0.1:8767';
const MEMORY_SEARCH_PATH = '/memory/search-hybrid';
const MEMORY_CHUNK_PATH = '/memory/chunk';
const MAX_CONTEXT_CHARS = 12000;
const HYBRID_URL = process.env.FLOSSWARE_HYBRID_URL || 'http://127.0.0.1:8773';
const TRANSCRIPT_RESULTS = 5;
const TRANSCRIPT_SNIPPET_CHARS = 400;
const KNOWLEDGE_SNIPPET_CHARS = 400;
const RELATED_SESSIONS = 5;
const INTENT_MIN_CONFIDENCE = 0.5;

// ---------------------------------------------------------------------------
// RAG configuration
// ---------------------------------------------------------------------------

const RAG_CONFIG_MAX_BYTES = 64 * 1024;

// Defaults preserve this hook's pre-configuration behavior exactly.  They are
// deliberately not the documented CLI defaults: with no rag.json the live
// hook must keep doing what it did before configuration existed.
const DEFAULT_RAG_CONFIG = {
  memory: { enabled: true, max_results: 10, timeout_ms: 15000 },
  knowledge: { enabled: true, max_results: 3, max_queries: 3, timeout_ms: 15000 },
  graph: { enabled: false, timeout_ms: 400, max_roots: 3, max_nodes: 8, max_context_chars: 3000 },
};

// Documented bounds, identical to tools/claude-config/lib/rag_config.py.
const RAG_RANGES = {
  'memory.max_results': [1, 10],
  'memory.timeout_ms': [50, 10000],
  'knowledge.max_results': [1, 10],
  'knowledge.max_queries': [1, 3],
  'knowledge.timeout_ms': [50, 10000],
  'graph.timeout_ms': [50, 5000],
  'graph.max_roots': [1, 3],
  'graph.max_nodes': [1, 8],
  'graph.max_context_chars': [100, 3000],
};

function ragConfigPath() {
  const explicit = process.env.FLOSSWARE_CE_RAG_CONFIG;
  if (explicit) return path.resolve(explicit);
  const profile = process.env.POC_CLAUDE_ENSEMBLE_CLAUDE_CODE_CONFIG_DIR
    || process.env.CLAUDE_CONFIG_DIR;
  const base = profile
    ? path.resolve(profile)
    : path.resolve(path.dirname(process.argv[1] || __filename), '..');
  return path.join(base, '.flossware-claude-config', 'rag.json');
}

function cloneDefaults() {
  return JSON.parse(JSON.stringify(DEFAULT_RAG_CONFIG));
}

// Merge a parsed rag.json over the defaults.  Returns { config } or { error }
// where error is a stable code, never file content.
function mergeRagConfig(supplied) {
  if (!supplied || typeof supplied !== 'object' || Array.isArray(supplied)) {
    return { error: 'not_an_object' };
  }
  const config = cloneDefaults();
  for (const section of Object.keys(supplied)) {
    if (!Object.hasOwn(config, section)) return { error: `unknown_section:${section}` };
    const values = supplied[section];
    if (!values || typeof values !== 'object' || Array.isArray(values)) {
      return { error: `invalid_section:${section}` };
    }
    for (const key of Object.keys(values)) {
      if (!Object.hasOwn(config[section], key)) return { error: `unknown_setting:${section}.${key}` };
      // Only supplied values are range-checked.  Inherited defaults are the
      // hook's pre-configuration runtime values and are intentionally kept
      // even where they sit outside the documented CLI range (timeouts).
      if (key === 'enabled') {
        if (typeof values[key] !== 'boolean') return { error: `invalid_type:${section}.enabled` };
      } else {
        const range = RAG_RANGES[`${section}.${key}`];
        const value = values[key];
        if (!range || !Number.isInteger(value) || value < range[0] || value > range[1]) {
          return { error: `out_of_range:${section}.${key}` };
        }
      }
      config[section][key] = values[key];
    }
  }
  return { config };
}

// Bounded, symlink-refusing read.  Never throws.
function readRagConfigFile(file) {
  let fd;
  try {
    fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  } catch (error) {
    if (error && error.code === 'ENOENT') return { error: 'missing' };
    if (error && (error.code === 'ELOOP' || error.code === 'EMLINK')) return { error: 'symlink' };
    return { error: 'unreadable' };
  }
  try {
    const stat = fs.fstatSync(fd);
    if (!stat.isFile()) return { error: 'not_a_file' };
    if (stat.size > RAG_CONFIG_MAX_BYTES) return { error: 'oversize' };
    const buffer = Buffer.alloc(stat.size);
    const read = buffer.length ? fs.readSync(fd, buffer, 0, buffer.length, 0) : 0;
    return { text: buffer.subarray(0, read).toString('utf8') };
  } catch {
    return { error: 'unreadable' };
  } finally {
    try { fs.closeSync(fd); } catch { /* ignore */ }
  }
}

// Operator override, deliberately outside the GA-tunable rag.json range: the
// live CE profile sets FLOSSWARE_MEMORY_TIMEOUT_MS=30000, above the documented
// 10000 ceiling for memory.timeout_ms, and that value must keep working.
const ENV_TIMEOUT_RANGE = [50, 600000];

function envTimeoutOverride(config) {
  const raw = process.env.FLOSSWARE_MEMORY_TIMEOUT_MS;
  if (raw === undefined || raw.trim() === '') return { applied: false, error: null };
  const value = Number.parseInt(raw, 10);
  if (!Number.isInteger(value) || String(value) !== raw.trim()
      || value < ENV_TIMEOUT_RANGE[0] || value > ENV_TIMEOUT_RANGE[1]) {
    return { applied: false, error: 'env_out_of_range:FLOSSWARE_MEMORY_TIMEOUT_MS' };
  }
  config.memory.timeout_ms = value;
  config.knowledge.timeout_ms = value;
  return { applied: true, error: null };
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

/**
 * Load the effective configuration.  Always returns a usable config; a
 * problem is reported as a code in `error` and defaults are used instead.
 */
function loadRagConfig() {
  const file = ragConfigPath();
  const config = cloneDefaults();
  const loaded = { config, source: 'defaults', error: null, envTimeoutOverride: false, sha256: '', path: file };

  const fileResult = readRagConfigFile(file);
  if (fileResult.error) {
    loaded.error = fileResult.error;
  } else {
    let supplied;
    try {
      supplied = JSON.parse(fileResult.text);
    } catch {
      loaded.error = 'malformed_json';
    }
    if (loaded.error === null) {
      const merged = mergeRagConfig(supplied);
      if (merged.error) {
        loaded.error = merged.error;
      } else {
        loaded.config = merged.config;
        loaded.source = 'file';
      }
    }
  }

  const override = envTimeoutOverride(loaded.config);
  loaded.envTimeoutOverride = override.applied;
  if (override.error && loaded.error === null) loaded.error = override.error;
  loaded.sha256 = crypto.createHash('sha256').update(canonicalJson(loaded.config)).digest('hex');
  return loaded;
}

// ---------------------------------------------------------------------------
// Memory / hybrid retrieval
// ---------------------------------------------------------------------------

function buildSearchUrl() {
  return new URL(MEMORY_SEARCH_PATH, MEMORY_URL.endsWith('/') ? MEMORY_URL : `${MEMORY_URL}/`).toString();
}

function buildChunkUrl() {
  return new URL(MEMORY_CHUNK_PATH, MEMORY_URL.endsWith('/') ? MEMORY_URL : `${MEMORY_URL}/`).toString();
}

// search-hybrid returns file/section/score only. Fetch each document's chunks
// once and copy the text of the matching section onto the result.
async function attachContent(payload, config) {
  if (!payload || payload.ok !== true || !Array.isArray(payload.results)) return payload;
  const chunksByFile = new Map();
  for (const result of payload.results) {
    if (!result || typeof result.file !== 'string' || typeof result.content === 'string') continue;
    if (!chunksByFile.has(result.file)) {
      chunksByFile.set(result.file, await fetchChunks(result.file, config));
    }
    const match = chunksByFile.get(result.file).find((chunk) => chunk.header === result.section);
    if (match) result.content = match.content;
  }
  return payload;
}

async function fetchChunks(name, config) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.memory.timeout_ms);
  try {
    const response = await fetch(buildChunkUrl(), {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ name }),
      signal: controller.signal,
    });
    if (!response.ok) return [];
    const payload = await response.json();
    return payload?.ok === true && Array.isArray(payload.chunks) ? payload.chunks : [];
  } catch {
    return [];
  } finally {
    clearTimeout(timer);
  }
}

function resultText(result) {
  if (!result || typeof result !== 'object') return '';
  if (typeof result.content !== 'string') return '';
  return result.content.trim();
}

function resultSource(result) {
  const file = typeof result?.file === 'string' ? result.file : '';
  const source = typeof result?.source === 'string' ? result.source : '';
  return source === 'claude-code' || file.startsWith('claude-code-') ? 'knowledge' : 'memory';
}

function dedupeResults(results) {
  const seen = new Set();
  const deduped = [];

  for (const result of results) {
    const content = resultText(result);
    if (!content) continue;

    const key = [
      resultSource(result),
      result.file || '',
      result.section || '',
      content,
    ].join('\u0000');

    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push({ ...result, content });
  }

  return deduped;
}

function extractResults(payload) {
  if (!payload || typeof payload !== 'object' || payload.ok !== true) return [];
  return Array.isArray(payload.results) ? dedupeResults(payload.results) : [];
}

function partitionResults(results, limit) {
  const memory = [];
  const knowledge = [];

  for (const result of results.slice(0, limit)) {
    (resultSource(result) === 'knowledge' ? knowledge : memory).push(result);
  }

  return { memory, knowledge };
}

function formatContext(query, payload, config) {
  const results = extractResults(payload);
  const { memory, knowledge } = partitionResults(results, config.memory.max_results);
  if (!memory.length && !knowledge.length) return { text: '', memory, knowledge };

  const lines = [
    'Relevant Claude Ensemble Context:',
    '',
  ];

  if (memory.length) {
    lines.push('### Relevant Memory (what happened)');
    for (const [index, result] of memory.entries()) {
      lines.push(`[${index + 1}] ${result.content}`);
    }
    lines.push('');
  }

  if (knowledge.length) {
    lines.push('### Relevant Knowledge (what is currently known)');
    for (const [index, result] of knowledge.entries()) {
      lines.push(`[${index + 1}] ${result.content}`);
    }
  }

  let text = lines.join('\n').trim();
  if (text.length > MAX_CONTEXT_CHARS) {
    text = text.slice(0, MAX_CONTEXT_CHARS) + '\n[Context truncated]';
  }

  return { text, memory, knowledge };
}

function contextDedupeKey(query) {
  let hash = 2166136261;
  for (let index = 0; index < query.length; index += 1) {
    hash ^= query.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

async function readHookEvent() {
  if (process.env.CLAUDE_PROMPT) {
    return { prompt: process.env.CLAUDE_PROMPT };
  }
  if (process.stdin.isTTY) return { prompt: '' };

  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  const input = Buffer.concat(chunks).toString('utf8');
  if (!input.trim()) return { prompt: '' };

  try {
    const event = JSON.parse(input);
    return {
      prompt: typeof event.prompt === 'string' ? event.prompt : '',
      sessionId: typeof event.session_id === 'string' ? event.session_id : '',
    };
  } catch {
    return { prompt: '' };
  }
}

async function searchContext(query, config) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.memory.timeout_ms);

  try {
    const response = await fetch(buildSearchUrl(), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        accept: 'application/json',
      },
      body: JSON.stringify({ query, top_k: config.memory.max_results }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Memory service returned HTTP ${response.status}`);
    }

    return await attachContent(await response.json(), config);
  } finally {
    clearTimeout(timer);
  }
}

// Hybrid search over past transcripts (keyword + vector) via the CE hybrid service.
// This single request also carries the knowledge, intent, and related-session
// sections, so its timeout is knowledge.timeout_ms.
async function transcriptSearch(query, config) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.knowledge.timeout_ms);
  try {
    const response = await fetch(`${HYBRID_URL}/search`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ query, top_k: TRANSCRIPT_RESULTS }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Hybrid search returned HTTP ${response.status}`);
    const payload = await response.json();
    // Drop non-object entries so one malformed result cannot abort the injection.
    const objects = (value) => (Array.isArray(value)
      ? value.filter((entry) => entry && typeof entry === 'object' && !Array.isArray(entry))
      : []);
    return {
      results: objects(payload.results),
      knowledge: objects(payload.knowledge),
      intent: payload.intent && typeof payload.intent === 'object' ? payload.intent : null,
      relatedSessions: objects(payload.related_sessions),
    };
  } finally {
    clearTimeout(timer);
  }
}

const STOPWORDS = new Set(['the', 'and', 'how', 'does', 'what', 'why', 'for', 'with', 'from',
  'that', 'this', 'are', 'was', 'were', 'have', 'has', 'can', 'you', 'our', 'about', 'into', 'when']);

function queryTerms(query) {
  return [...new Set((query.toLowerCase().match(/[\p{L}\p{N}_.-]{3,}/gu) || [])
    .filter((term) => !STOPWORDS.has(term)))];
}

// Extractive, query-focused compression: keep the sentences that match the query terms,
// in their original order, up to the limit. Falls back to the leading text when nothing matches.
function compressSnippet(text, terms, limit) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (normalized.length <= limit) return normalized;
  const sentences = normalized.split(/(?<=[.!?])\s+/);
  const scored = sentences.map((sentence, index) => {
    const lower = sentence.toLowerCase();
    return { index, sentence, hits: terms.filter((term) => lower.includes(term)).length };
  });
  const chosen = [];
  let used = 0;
  for (const item of [...scored].filter((s) => s.hits > 0).sort((a, b) => b.hits - a.hits || a.index - b.index)) {
    if (used + item.sentence.length + 1 > limit) continue;
    chosen.push(item);
    used += item.sentence.length + 1;
  }
  if (!chosen.length) return `${normalized.slice(0, limit)}…`;
  const ordered = chosen.sort((a, b) => a.index - b.index).map((s) => s.sentence).join(' ');
  return ordered.length > limit ? `${ordered.slice(0, limit)}…` : ordered;
}

function formatIntentContext(intent) {
  if (!intent || typeof intent.label !== 'string') return '';
  if (!Number.isFinite(intent.confidence) || intent.confidence < INTENT_MIN_CONFIDENCE) return '';
  return `### Query intent: ${intent.label} (confidence ${intent.confidence.toFixed(2)})`;
}

function formatRelatedSessionsContext(sessions) {
  if (!sessions.length) return '';
  const lines = ['### Related sessions (touched the files this question names)'];
  for (const [index, session] of sessions.slice(0, RELATED_SESSIONS).entries()) {
    const id = typeof session.session_id === 'string' ? session.session_id.slice(0, 8) : 'unknown';
    const project = session.project ? ` (${session.project})` : '';
    const touched = Array.isArray(session.touched) ? session.touched.slice(0, 3).join(', ') : '';
    lines.push(`[S${index + 1}] ${id}${project} touched: ${touched}`);
  }
  return lines.join('\n');
}

function formatKnowledgeContext(knowledge, query, limit) {
  if (!knowledge.length || limit < 1) return '';
  const terms = queryTerms(query);
  const lines = ['### Relevant Books and Documents (knowledge index)'];
  for (const [index, result] of knowledge.slice(0, limit).entries()) {
    const snippet = compressSnippet(result.text, terms, KNOWLEDGE_SNIPPET_CHARS);
    const origin = result.origin ? ` (${result.origin})` : '';
    const title = result.title || result.url || 'untitled';
    lines.push(`[K${index + 1}] ${title}${origin} ${snippet}`);
  }
  return lines.join('\n');
}

function formatTranscriptContext(results, query) {
  if (!results.length) return '';
  const terms = queryTerms(query);
  const lines = ['### Relevant Past Conversation (transcript search)'];
  for (const [index, result] of results.entries()) {
    const snippet = compressSnippet(result.text, terms, TRANSCRIPT_SNIPPET_CHARS);
    const where = result.project ? ` (${result.project}, ${result.role})` : '';
    lines.push(`[T${index + 1}]${where} ${snippet}`);
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Injection journal (metadata only, append-once)
// ---------------------------------------------------------------------------

const JOURNAL_SCHEMA_VERSION = 1;
const JOURNAL_PRODUCER = 'claude-code';
const JOURNAL_KIND = 'rag_injection';
const JOURNAL_TAIL_BYTES = 256 * 1024;
const JOURNAL_MAX_SOURCE_IDS = 50;
const JOURNAL_MAX_ID_CHARS = 200;

function ragJournalPath() {
  const configured = process.env.FLOSSWARE_CE_RAG_JOURNAL;
  if (configured !== undefined) {
    const value = configured.trim();
    if (value === '' || value === 'off' || value === '0' || value === 'false' || value === 'disabled') return '';
    return path.resolve(value);
  }
  return path.join(os.homedir(), '.FlossWare', 'ledger', 'rag', 'injections.jsonl');
}

// Ids only: collapse whitespace, drop control characters, bound the length.
function journalId(value) {
  const text = String(value ?? '').replace(/[\u0000-\u001f\u007f]+/g, ' ').replace(/\s+/g, ' ').trim();
  return text.length > JOURNAL_MAX_ID_CHARS ? text.slice(0, JOURNAL_MAX_ID_CHARS) : text;
}

// Source ids only.  Result objects carry content/text, which must never reach
// the journal, so only identifier fields are read here.
function journalSourceIds(items) {
  const ids = [];
  const push = (value) => {
    if (ids.length >= JOURNAL_MAX_SOURCE_IDS) return;
    const id = journalId(value);
    if (id) ids.push(id);
  };
  for (const result of items.memory || []) {
    const file = journalId(result?.file || 'unknown');
    const section = journalId(result?.section ?? '');
    push(section ? `memory:${file}#${section}` : `memory:${file}`);
  }
  for (const [index, result] of (items.knowledge || []).entries()) {
    const id = result?.url || result?.origin || 'unknown';
    push(`knowledge:${journalId(id)}#${index}`);
  }
  for (const [index, result] of (items.transcript || []).entries()) {
    const session = typeof result?.session_id === 'string' ? result.session_id.slice(0, 8) : 'unknown';
    push(`transcript:${journalId(session)}#${index}`);
  }
  for (const [index, session] of (items.relatedSessions || []).entries()) {
    const id = typeof session?.session_id === 'string' ? session.session_id.slice(0, 8) : 'unknown';
    push(`session:${journalId(id)}#${index}`);
  }
  return ids;
}

// A retried prompt with the same session, prompt, injected context, and
// effective configuration is the same exposure record.
function injectionEventId(sessionId, query, context, configSha) {
  return crypto.createHash('sha256')
    .update(['rag-injection', sessionId, query, context, configSha].join('\u0000'))
    .digest('hex');
}

// Bounded tail scan: the journal is append-only, so recent ids are at the end.
// Beyond the window a repeated event could be written again (at-least-once),
// which is the safe direction.
function journalHasEventId(file, eventId) {
  let fd;
  try {
    fd = fs.openSync(file, 'r');
  } catch {
    return false;
  }
  try {
    const { size } = fs.fstatSync(fd);
    const start = Math.max(0, size - JOURNAL_TAIL_BYTES);
    const buffer = Buffer.alloc(size - start);
    const read = buffer.length ? fs.readSync(fd, buffer, 0, buffer.length, start) : 0;
    const text = buffer.subarray(0, read).toString('utf8');
    for (const match of text.matchAll(/"event_id"\s*:\s*"([0-9a-f]{64})"/g)) {
      if (match[1] === eventId) return true;
    }
    return false;
  } catch {
    return false;
  } finally {
    try { fs.closeSync(fd); } catch { /* ignore */ }
  }
}

function writeInjectionRow(row) {
  const file = ragJournalPath();
  if (!file) return false;
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    if (journalHasEventId(file, row.event_id)) return false;
    const fd = fs.openSync(file, 'a', 0o600);
    try {
      fs.fchmodSync(fd, 0o600);
      fs.writeSync(fd, `${JSON.stringify(row)}\n`);
    } finally {
      fs.closeSync(fd);
    }
    return true;
  } catch (error) {
    // Code only: never echo file content or prompt text into stderr.
    process.stderr.write(`RAG injection journal unavailable: ${error && error.code ? error.code : 'error'}\n`);
    return false;
  }
}

function buildInjectionRow(event, query, loaded, sections, sourceItems, context, truncated) {
  return {
    schema_version: JOURNAL_SCHEMA_VERSION,
    producer: JOURNAL_PRODUCER,
    kind: JOURNAL_KIND,
    event_id: injectionEventId(event.sessionId || '', query, context, loaded.sha256),
    timestamp: new Date().toISOString(),
    session_id: event.sessionId || '',
    query_hash: contextDedupeKey(query),
    prompt_chars: query.length,
    config: {
      source: loaded.source,
      error: loaded.error,
      env_timeout_override: loaded.envTimeoutOverride,
      sha256: loaded.sha256,
      memory: {
        enabled: loaded.config.memory.enabled,
        max_results: loaded.config.memory.max_results,
        timeout_ms: loaded.config.memory.timeout_ms,
      },
      knowledge: {
        enabled: loaded.config.knowledge.enabled,
        max_results: loaded.config.knowledge.max_results,
        max_queries: loaded.config.knowledge.max_queries,
        timeout_ms: loaded.config.knowledge.timeout_ms,
      },
      graph: {
        enabled: loaded.config.graph.enabled,
        max_nodes: loaded.config.graph.max_nodes,
        max_roots: loaded.config.graph.max_roots,
        max_context_chars: loaded.config.graph.max_context_chars,
        timeout_ms: loaded.config.graph.timeout_ms,
      },
    },
    sections,
    context_chars: context.length,
    truncated,
    source_ids: journalSourceIds(sourceItems),
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const event = await readHookEvent();
  const query = typeof event.prompt === 'string' ? event.prompt.trim() : '';
  if (!query) return;

  const loaded = loadRagConfig();
  const config = loaded.config;
  if (loaded.error) {
    process.stderr.write(`RAG configuration ${loaded.error}; using built-in defaults\n`);
  }

  try {
    const [memoryOutcome, transcriptOutcome] = await Promise.allSettled([
      config.memory.enabled ? searchContext(query, config) : Promise.resolve(null),
      transcriptSearch(query, config),
    ]);
    if (memoryOutcome.status === 'rejected') {
      process.stderr.write(`Memory search unavailable: ${memoryOutcome.reason?.message ?? memoryOutcome.reason}\n`);
    }
    if (transcriptOutcome.status === 'rejected') {
      process.stderr.write(`Transcript search unavailable: ${transcriptOutcome.reason?.message ?? transcriptOutcome.reason}\n`);
    }
    const payload = memoryOutcome.status === 'fulfilled' ? memoryOutcome.value : null;
    const memoryFormatted = payload
      ? formatContext(query, payload, config)
      : { text: '', memory: [], knowledge: [] };
    const hits = transcriptOutcome.status === 'fulfilled'
      ? transcriptOutcome.value
      : { results: [], knowledge: [], intent: null, relatedSessions: [] };
    const knowledgeEnabled = config.knowledge.enabled;
    const knowledgeResults = knowledgeEnabled ? hits.knowledge : [];
    const intentContext = formatIntentContext(hits.intent);
    const knowledgeContext = knowledgeEnabled
      ? formatKnowledgeContext(hits.knowledge, query, config.knowledge.max_results)
      : '';
    const relatedContext = formatRelatedSessionsContext(hits.relatedSessions);
    const transcriptContext = formatTranscriptContext(hits.results, query);
    // One budget for the whole injection: memory, then intent, knowledge, related sessions, and transcript results.
    const combined = [memoryFormatted.text, intentContext, knowledgeContext, relatedContext, transcriptContext]
      .filter(Boolean).join('\n\n');
    const truncated = combined.length > MAX_CONTEXT_CHARS;
    const additionalContext = truncated
      ? `${combined.slice(0, MAX_CONTEXT_CHARS)}\n[Context truncated]`
      : combined;

    // The memory request's payload is partitioned for formatting; a result
    // whose file looks like a Claude Code document is rendered under the
    // Knowledge heading but still came from this request, so the memory
    // section counts everything the request injected.
    const memoryInjected = memoryFormatted.memory.length + memoryFormatted.knowledge.length;

    const sections = {
      memory: {
        status: !config.memory.enabled
          ? 'disabled'
          : payload && payload.ok === true
            ? (memoryInjected ? 'ok' : 'empty')
            : 'error',
        results: memoryInjected,
        chars: memoryFormatted.text.length,
      },
      knowledge: {
        status: !knowledgeEnabled
          ? 'disabled'
          : transcriptOutcome.status === 'rejected'
            ? 'error'
            : (knowledgeResults.length ? 'ok' : 'empty'),
        results: knowledgeContext ? Math.min(knowledgeResults.length, config.knowledge.max_results) : 0,
        chars: knowledgeContext.length,
      },
      transcript: {
        status: transcriptOutcome.status === 'rejected' ? 'error' : (hits.results.length ? 'ok' : 'empty'),
        results: hits.results.length,
        chars: transcriptContext.length,
      },
      intent: {
        status: intentContext ? 'ok' : (hits.intent ? 'below_threshold' : 'absent'),
        // Enum-ish label, bounded; never free text from the retrieval corpus.
        label: intentContext && typeof hits.intent.label === 'string'
          ? journalId(hits.intent.label).slice(0, 64)
          : null,
        confidence: intentContext && Number.isFinite(hits.intent.confidence) ? hits.intent.confidence : null,
      },
      related_sessions: {
        status: relatedContext ? 'ok' : 'absent',
        results: relatedContext ? Math.min(hits.relatedSessions.length, RELATED_SESSIONS) : 0,
      },
    };

    // Identifier-only view for source_ids; never the result objects themselves.
    const sourceItems = {
      memory: [...memoryFormatted.memory, ...memoryFormatted.knowledge],
      knowledge: knowledgeResults.slice(0, config.knowledge.max_results),
      transcript: hits.results.slice(0, TRANSCRIPT_RESULTS),
      relatedSessions: hits.relatedSessions.slice(0, RELATED_SESSIONS),
    };

    const row = buildInjectionRow(event, query, loaded, sections, sourceItems, additionalContext, truncated);
    writeInjectionRow(row);

    if (!additionalContext) return;

    process.stdout.write(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: 'UserPromptSubmit',
        additionalContext,
        contextRetrieval: {
          queryHash: contextDedupeKey(query),
          memoryCount: memoryFormatted.memory.length,
          knowledgeCount: sections.knowledge.results,
          configSource: loaded.source,
          injectionEventId: row.event_id,
        },
      },
    }) + '\n');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`Memory search unavailable: ${message}\n`);
  }
}

main().catch(error => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Memory search unavailable: ${message}\n`);
});
