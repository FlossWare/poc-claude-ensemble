#!/usr/bin/env python3
"""Loopback REST hybrid search service. It orchestrates other CE services over REST.

Flow for POST /search:
  1. Embedding service (8770)  POST /embeddings          embed the query (mode=query)
  2. Vector service    (8771)  POST /collections/.../query  exact nearest neighbours
  3. Lexical service   (8772)  POST /search              FTS5 BM25 candidates
  4. Reciprocal rank fusion (k=60) of the two ranked lists, done here

If a dependency fails, the response still comes back with the results from the
others and a "degraded" list naming the failed dependencies. It returns 503 only
when no candidate source succeeded.

Endpoints
  GET  /health
  POST /search   {"query": "...", "top_k": n} -> ranked results with source info
  POST /prompts  {"session_id", "prompt", "timestamp", "cwd"} -> durable ledger entry,
                 embedded and indexed in the vector store (idempotent by event ID)
"""

import hashlib
import json
import os
import re
import sys
import threading
import turn_outcomes
import time
import urllib.error
from collections import OrderedDict
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_POOL = ThreadPoolExecutor(max_workers=8)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8773
EMBEDDING_URL = os.environ.get("ENSEMBLE_EMBEDDING_URL", "http://127.0.0.1:8770")
VECTOR_URL = os.environ.get("ENSEMBLE_VECTOR_URL", "http://127.0.0.1:8771")
LEXICAL_URL = os.environ.get("ENSEMBLE_LEXICAL_URL", "http://127.0.0.1:8772")
GRAPH_URL = os.environ.get("ENSEMBLE_GRAPH_URL", "http://127.0.0.1:8766")
RERANK_URL = os.environ.get("ENSEMBLE_RERANK_URL", "http://127.0.0.1:8774")
RANKING_URL = os.environ.get("ENSEMBLE_RANKING_URL", "http://127.0.0.1:8780")
EXTRACTION_URL = os.environ.get("ENSEMBLE_EXTRACTION_URL", "http://127.0.0.1:8778")
INTENT_URL = os.environ.get("ENSEMBLE_INTENT_URL", "http://127.0.0.1:8779")
INDEX_SERVICE_URL = os.environ.get("ENSEMBLE_INDEX_URL", "http://127.0.0.1:8777")
CHUNKING_URL = os.environ.get("ENSEMBLE_CHUNKING_URL", "http://127.0.0.1:8775")
RERANK_ENABLED = os.environ.get("ENSEMBLE_RERANK_ENABLED", "1") == "1"
RERANK_TOP = 30          # cross-encoder scores only the top fused candidates
RERANK_TIMEOUT = 5       # a slow reranker must not stall a prompt; fall back to fused order
TRANSCRIPT_ROOTS = [os.path.expanduser(p) for p in os.environ.get(
    "ENSEMBLE_TRANSCRIPT_ROOTS", "~/.claude/projects:~/2026-09_16-57-09_claude/projects").split(":")]
INGEST_EMBED_BATCH = 64     # matches the embedding service's per-request cap
EMBED_CHARS = 4000          # the model reads at most 512 tokens; longer text is never embedded
INGEST_TIMEOUT = 600
INGEST_LOG = os.path.expanduser("~/.FlossWare/poc-claude-ensemble/state/logs/ingest-sessions.log")
INGEST_STATE = os.path.expanduser("~/.FlossWare/poc-claude-ensemble/state/ingest-state.json")
FILE_TOOLS = {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit"}
FILE_KEYS = ("file_path", "notebook_path", "path")
TOOL_TEXT_CHARS = 20000   # tool inputs, tool results, and thinking are clipped, as in the bulk index
COLLECTION = "transcript_text"
CANDIDATES = 200
RRF_K = 60
DEP_TIMEOUT = 30
MAX_BODY_BYTES = 65_536
MAX_QUERY_CHARS = 2_000
MAX_TOP_K = 100


class DependencyError(Exception):
    def __init__(self, name, message):
        super().__init__(f"{name}: {message}")
        self.name = name


def _call(name, url, payload=None, method="POST", timeout=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"content-type": "application/json", "accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout or DEP_TIMEOUT) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise DependencyError(name, f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise DependencyError(name, str(exc)) from exc
    if not isinstance(body, dict) or body.get("ok") is not True:
        raise DependencyError(name, f"unexpected response: {str(body)[:200]}")
    return body


def fuse(ranked_lists):
    scores, records = {}, {}
    for ranked in ranked_lists:
        for rank, record in enumerate(ranked, start=1):
            key = record["key"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            records.setdefault(key, record)
    order = sorted(scores, key=lambda k: -scores[k])
    return [dict(records[k], score=round(scores[k], 6)) for k in order]


PROMPT_LEDGER = os.path.expanduser(os.environ.get("ENSEMBLE_PROMPT_LEDGER", "~/.FlossWare/ledger/prompts/prompts.jsonl"))
MAX_PROMPT_BODY_BYTES = 262_144
MAX_PROMPT_CHARS = 65_536
_LEDGER_LOCK = threading.Lock()   # the duplicate check and the append must be one step


def store_prompt(session_id, prompt, timestamp, cwd):
    """Durably append a prompt to the ledger. Nothing is indexed here.

    The ledger is the source of truth. The event ID is derived from the session, the
    timestamp, and the text, so a retried capture is recognised as a duplicate. Prompts
    become searchable when the session's transcript is ingested at turn end (Stop), so
    the response reports neither vector nor lexical indexing.
    """
    event_id = hashlib.sha256(f"{session_id}|{timestamp}|{prompt}".encode("utf-8")).hexdigest()
    os.makedirs(os.path.dirname(PROMPT_LEDGER), exist_ok=True)
    with _LEDGER_LOCK:
        already = False
        if os.path.exists(PROMPT_LEDGER):
            with open(PROMPT_LEDGER, encoding="utf-8") as handle:
                already = any(json.loads(line).get("event_id") == event_id for line in handle if line.strip())
        if not already:
            record = {"schema_version": 1, "producer": "claude-code", "kind": "prompt",
                  "event_id": event_id, "session_id": session_id, "timestamp": timestamp,
                  "cwd": cwd, "prompt": prompt}
            fd = os.open(PROMPT_LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    return {"ok": True, "event_id": event_id, "status": "duplicate" if already else "stored",
            "vector_indexed": False, "index_error": None, "lexical_indexed": False}


FILE_QUERY = re.compile(r"(?:/[\w.\-]+)+|[\w\-]+\.[A-Za-z][A-Za-z0-9]{0,7}\b")


def _related_sessions(query):
    """Sessions that touched a file the query names, from the Graph service (REST only)."""
    entities = _call("extraction", f"{EXTRACTION_URL}/extract", {"text": query})["entities"]
    tokens = {e["text"] for e in entities if e["type"] in ("path", "file") and e["exact"]}
    if not tokens:
        return []
    files = _call("graph", f"{GRAPH_URL}/graph/query", {"type": "file", "properties": {}})["nodes"]
    matched = [n for n in files
               if any(n["properties"].get("path") == t or str(n["properties"].get("path", "")).endswith("/" + t)
                      for t in tokens)]
    sessions = {}
    for node in matched:
        walked = _call("graph", f"{GRAPH_URL}/graph/traverse",
                       {"start": node["id"], "direction": "in", "max_depth": 1, "edge_type": "touched"})
        for hit in walked["results"]:
            sn = hit["node"]
            entry = sessions.setdefault(sn["id"], {"session_id": sn["properties"].get("session_id"),
                                                   "project": sn["properties"].get("project"), "touched": []})
            entry["touched"].append(node["properties"]["path"])
    return list(sessions.values())[:10]


def _hybrid_search_uncached(query, top_k, rerank=True):
    degraded = []
    vector_list, lexical_list = [], []
    embedding = None
    # Lexical and embedding/vector legs are independent, so run the lexical call concurrently.
    lexical_future = _POOL.submit(_call, "lexical", f"{LEXICAL_URL}/search",
                                  {"query": query, "top_k": CANDIDATES})
    try:
        embedded = _call("embedding", f"{EMBEDDING_URL}/embeddings", {"texts": [query], "mode": "query"})
        embedding = embedded["embeddings"][0]
    except DependencyError as exc:
        degraded.append(exc.name)
    if embedding is not None:
        try:
            vector = _call("vector", f"{VECTOR_URL}/collections/{COLLECTION}/query",
                           {"embedding": embedding, "top_k": CANDIDATES})
            vector_list = [{"key": r["id"], "text": r["document"], "source": "vector",
                            "project": (r.get("metadata") or {}).get("project"),
                            "session_id": (r.get("metadata") or {}).get("session_id"),
                            "role": (r.get("metadata") or {}).get("role"), "is_subagent": None}
                           for r in vector["results"]]
        except DependencyError as exc:
            degraded.append(exc.name)
    try:
        lexical = lexical_future.result()
        lexical_list = [{"key": r["text_sha"], "text": r["text"], "source": "lexical",
                         "role": r["role"], "project": r["project"], "session_id": r["session_id"],
                         "is_subagent": r["is_subagent"]} for r in lexical["results"]]
    except DependencyError as exc:
        degraded.append(exc.name)
    if not vector_list and not lexical_list and degraded:
        raise DependencyError("search", f"no candidate source succeeded ({', '.join(degraded)})")
    # Source metadata (session, project, role) comes only from lexical hits. A result found
    # only by the vector side has its text but no source fields.
    records = {}
    for record in vector_list + lexical_list:
        records[record["key"]] = {**records.get(record["key"], {}), **record}
    lists = {"vector": [{"key": r["key"], "rank": i + 1} for i, r in enumerate(vector_list)],
             "lexical": [{"key": r["key"], "rank": i + 1} for i, r in enumerate(lexical_list)]}
    try:
        ranked = _call("ranking", f"{RANKING_URL}/rank", {"lists": lists, "top_k": CANDIDATES, "k": RRF_K})
        fused = [dict(records[item["key"]], score=item["score"], contributions=item["contributions"])
                 for item in ranked["results"] if item["key"] in records]
    except DependencyError as exc:
        degraded.append(exc.name)
        fused = fuse([vector_list, lexical_list])
    reranked = False
    if RERANK_ENABLED and rerank and fused:
        head = fused[:RERANK_TOP]
        try:
            scored = _call("rerank", f"{RERANK_URL}/rerank",
                           {"query": query, "documents": [r["text"][:4000] for r in head]}, timeout=RERANK_TIMEOUT)
            for record, score in zip(head, scored["scores"]):
                record["rerank_score"] = round(float(score), 4)
            head.sort(key=lambda r: -r["rerank_score"])
            fused = head + fused[RERANK_TOP:]
            reranked = True
        except DependencyError as exc:
            degraded.append(exc.name)
    results = fused[:top_k]
    related = []
    try:
        related = _related_sessions(query)
    except DependencyError as exc:
        degraded.append(exc.name)
    intent = None
    try:
        intent = _call("intent", f"{INTENT_URL}/classify", {"text": query})
        intent = {"label": intent["label"], "confidence": intent["confidence"]}
    except DependencyError as exc:
        degraded.append(exc.name)
    return {"ok": True, "results": results, "degraded": degraded, "reranked": reranked,
            "related_sessions": related, "intent": intent}


def _stable_id(prefix, *parts):
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:40]}"


def _parse_transcript(path, start=0, known_first_ts=None):
    """Extract messages and file touches from the transcript starting at a byte offset.

    Only complete lines are consumed, so a line still being written is left for the next run.
    """
    session_id = os.path.basename(path)[:-6]
    project = os.path.basename(os.path.dirname(path))
    messages, touches = [], []
    first_ts, last_ts = known_first_ts, None
    end = start
    with open(path, "rb") as handle:
        handle.seek(start)
        for raw in handle:
            if not raw.endswith(b"\n"):
                break
            end += len(raw)
            line = raw.decode("utf-8", errors="replace")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            session_id = record.get("sessionId") or session_id
            ts = record.get("timestamp") or ""
            if ts:
                first_ts = first_ts or ts
                last_ts = ts
            message = record.get("message")
            role = message.get("role") if isinstance(message, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                if isinstance(content, str) and role in ("user", "assistant") and len(content.strip()) >= 20:
                    messages.append({"uuid": record.get("uuid") or "", "role": role, "ts": ts, "text": content, "kind": "text"})
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                uuid = record.get("uuid") or ""
                btype = block.get("type")
                if btype == "text" and role in ("user", "assistant"):
                    text = block.get("text") or ""
                    if len(text.strip()) >= 20:
                        messages.append({"uuid": uuid, "role": role, "ts": ts, "text": text, "kind": "text"})
                elif btype == "thinking":
                    text = block.get("thinking") or ""
                    if text.strip():
                        messages.append({"uuid": uuid, "role": "assistant", "ts": ts,
                                         "text": text[:TOOL_TEXT_CHARS], "kind": "thinking"})
                elif btype == "tool_use":
                    name = block.get("name") or "tool"
                    args = block.get("input") or {}
                    messages.append({"uuid": uuid, "role": "assistant", "ts": ts, "kind": "tool_use",
                                     "text": f"{name} {json.dumps(args, ensure_ascii=False)}"[:TOOL_TEXT_CHARS]})
                    if name in FILE_TOOLS:
                        for key in FILE_KEYS:
                            value = args.get(key)
                            if isinstance(value, str) and value.startswith("/"):
                                touches.append({"path": value, "tool": name, "ts": ts})
                                break
                elif btype == "tool_result":
                    result = block.get("content")
                    if isinstance(result, list):
                        result = " ".join(b.get("text", "") for b in result if isinstance(b, dict))
                    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    if text and text.strip():
                        messages.append({"uuid": uuid, "role": "user", "ts": ts, "kind": "tool_result",
                                         "text": text[:TOOL_TEXT_CHARS]})
    return {"session_id": session_id, "project": project, "messages": messages, "touches": touches,
            "file": path, "first_ts": first_ts, "last_ts": last_ts or first_ts, "bytes": os.path.getsize(path),
            "end_offset": end}


def _ingest_call(name, url, payload=None):
    return _call(name, url, payload, timeout=INGEST_TIMEOUT)


def _ingest_graph(parsed):
    """Write the session, its project, and its file touches to the Graph service over REST."""
    sid = parsed["session_id"]
    session_node = _stable_id("session", sid)
    _ingest_call("graph", f"{GRAPH_URL}/graph/add-node", {
        "type": "session", "id": session_node,
        "properties": {"session_id": sid, "project": parsed["project"]}})
    project_node = _stable_id("project", parsed["project"])
    _ingest_call("graph", f"{GRAPH_URL}/graph/add-node", {
        "type": "project", "id": project_node, "properties": {"name": parsed["project"]}})
    _ingest_call("graph", f"{GRAPH_URL}/graph/add-edge", {
        "source": session_node, "target": project_node, "type": "in_project",
        "id": _stable_id("edge", sid, "in_project")})
    edges = 0
    for touch in parsed["touches"]:
        file_node = _stable_id("file", touch["path"])
        _ingest_call("graph", f"{GRAPH_URL}/graph/add-node", {
            "type": "file", "id": file_node, "properties": {"path": touch["path"]}})
        _ingest_call("graph", f"{GRAPH_URL}/graph/add-edge", {
            "source": session_node, "target": file_node, "type": "touched",
            "properties": {"tool": touch["tool"], "ts": touch["ts"]},
            "id": _stable_id("edge", sid, touch["tool"], touch["path"], touch["ts"])})
        edges += 1
    return {"session_node": session_node, "file_touch_edges": edges}


def _ingest_vectors(parsed):
    """Embed and store the session's message texts, skipping texts already stored."""
    by_id = {}
    for message in parsed["messages"]:
        if message.get("kind", "text") != "text":
            continue   # the vector store holds conversation text only
        key = hashlib.sha256(message["text"].encode("utf-8")).hexdigest()
        by_id.setdefault(key, message)
    ids = list(by_id)
    if not ids:
        return {"messages": 0, "new": 0}
    found = set(_ingest_call("vector", f"{VECTOR_URL}/collections/{COLLECTION}/existing", {"ids": ids})["found"])
    # Texts already stored (possibly from before metadata existed) get their session and project filled in.
    known = [i for i in ids if i in found]
    for start in range(0, len(known), 256):
        batch_ids = known[start:start + 256]
        _ingest_call("vector", f"{VECTOR_URL}/collections/{COLLECTION}/metadata", {
            "ids": batch_ids,
            "metadatas": [{"kind": "transcript_text", "role": by_id[i]["role"],
                           "session_id": parsed["session_id"], "project": parsed["project"],
                           "message_uuid": by_id[i]["uuid"], "timestamp": by_id[i]["ts"]} for i in batch_ids]})
    pending = [i for i in ids if i not in found]
    stored = 0
    for start in range(0, len(pending), INGEST_EMBED_BATCH):
        batch = pending[start:start + INGEST_EMBED_BATCH]
        texts = [by_id[i]["text"] for i in batch]
        embedded = _ingest_call("embedding", f"{EMBEDDING_URL}/embeddings",
                                {"texts": [t[:EMBED_CHARS] for t in texts], "mode": "document"})
        _ingest_call("vector", f"{VECTOR_URL}/collections/{COLLECTION}/upsert", {
            "ids": batch, "embeddings": embedded["embeddings"], "documents": texts,
            "metadatas": [{"kind": "transcript_text", "role": by_id[i]["role"],
                           "session_id": parsed["session_id"], "project": parsed["project"],
                           "message_uuid": by_id[i]["uuid"], "timestamp": by_id[i]["ts"]} for i in batch],
        })
        stored += len(batch)
    return {"messages": len(ids), "new": stored}


def _validate_transcript(transcript_path):
    real = os.path.realpath(os.path.expanduser(transcript_path))
    if not real.endswith(".jsonl") or not any(real.startswith(os.path.realpath(r) + os.sep) for r in TRANSCRIPT_ROOTS):
        raise ValueError("transcript_path must be a .jsonl file under an allowed transcript root")
    if not os.path.isfile(real):
        raise ValueError("transcript_path does not exist")
    return real


def _ingest_state():
    try:
        with open(INGEST_STATE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _log_ingest(record):
    os.makedirs(os.path.dirname(INGEST_LOG), exist_ok=True)
    with open(INGEST_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _run_ingest(real):
    """Background ingest. Resumes from the stored byte offset, so only new lines are processed."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    size = os.stat(real).st_size
    state = _ingest_state()
    entry = state.get(real) if isinstance(state.get(real), dict) else {}
    offset = entry.get("offset")
    if offset is not None and offset >= size:
        _log_ingest({"ok": True, "skipped": "unchanged", "transcript": os.path.basename(real), "at": stamp})
        return
    if offset is None or offset > size:
        offset, known_first = 0, None
    else:
        known_first = entry.get("first_ts")
    try:
        result = ingest_session(real, offset, known_first)
        if result.get("skipped"):
            # Index refused this revision, so its content is not published and the offset must not move.
            _log_ingest(dict(result, at=stamp, from_offset=offset))
            return
        state[real] = {"offset": result["end_offset"], "first_ts": result["first_ts"]}
        os.makedirs(os.path.dirname(INGEST_STATE), exist_ok=True)
        tmp = INGEST_STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, INGEST_STATE)
        _log_ingest(dict(result, at=stamp, from_offset=offset))
    except Exception as exc:  # recorded, not raised: this runs in a background thread
        _log_ingest({"ok": False, "transcript": os.path.basename(real), "at": stamp,
                     "error": f"{type(exc).__name__}: {exc}"})


LEXICAL_BATCH_CHARS = 4_000_000   # keeps each /documents request well under the lexical body cap
LEXICAL_BATCH_MESSAGES = 400     # the lexical service accepts at most 500 messages per document


def _ingest_lexical(parsed):
    """Send the session and its messages to the lexical service, in size-limited batches."""
    session = {"session_id": parsed["session_id"], "project": parsed["project"], "file": parsed["file"],
               "first_ts": parsed["first_ts"], "last_ts": parsed["last_ts"], "bytes": parsed["bytes"],
               "source": "orchestrator"}
    documents, batch, size = [], [], 0
    for message in parsed["messages"]:
        text = message["text"]
        if batch and (size + len(text) > LEXICAL_BATCH_CHARS or len(batch) >= LEXICAL_BATCH_MESSAGES):
            documents.append(batch)
            batch, size = [], 0
        batch.append({"uuid": message["uuid"], "role": message["role"], "ts": message["ts"],
                      "kind": message.get("kind", "text"), "text": text})
        size += len(text)
    documents.append(batch)
    inserted = already = 0
    for chunk in documents:
        result = _ingest_call("lexical", f"{LEXICAL_URL}/documents", {"session": session, "messages": chunk})
        inserted += result["inserted"]
        already += result["already_present"]
    return {"inserted": inserted, "already_present": already}


def _chunk_text(text, source_id):
    """Split one message through the Chunking service (message-v1 profile). REST only."""
    result = _ingest_call("chunking", f"{CHUNKING_URL}/chunk",
                          {"profile": "message-v1", "source_id": source_id, "media_type": "text/plain", "text": text})
    return [c["text"] for c in result["chunks"]]


def _plug_in_chunking(parsed):
    """Replace each message's text with its Chunking-service chunks; one message per chunk."""
    expanded = []
    for message in parsed["messages"]:
        for piece in _chunk_text(message["text"], parsed["file"]):
            expanded.append({**message, "text": piece})
    parsed["messages"] = expanded
    return parsed


def _index_decision(parsed, end_offset):
    """Ask Index whether this revision of the transcript may publish. Writers stop unless it may."""
    body = _ingest_call("index", f"{INDEX_SERVICE_URL}/revisions", {
        "source_id": parsed["file"], "revision_sequence": max(1, end_offset),
        "revision_id": f"offset-{end_offset}",
        "profiles": {"chunking": "message-v1@1", "tokenization": "lexical-v1@1",
                     "embedding": "nomic-embed-text-v1.5@fp32", "vector": "transcript_text@1"}})
    return body["decision"]


def _complete_lines(path, start):
    """Complete JSON lines from a byte offset, so a line still being written is left for the next run."""
    lines = []
    with open(path, "rb") as handle:
        handle.seek(start)
        for raw in handle:
            if not raw.endswith(b"\n"):
                break
            lines.append(raw.decode("utf-8", errors="replace"))
    return lines


def record_turn_outcomes(path, start):
    """Send finished-turn outcomes to Thompson. Thompson outages delay recording; they do not fail ingest."""
    retried, retry_failed = turn_outcomes.retry_pending()
    accepted, pending = turn_outcomes.record_pending(turn_outcomes.outcomes_from_lines(_complete_lines(path, start)))
    return {"accepted": accepted, "pending": pending, "retried": retried, "retry_failed": retry_failed}


def ingest_session(transcript_path, start=0, known_first_ts=None):
    """Orchestrate one session's ingestion through the owning services. Idempotent by content ID."""
    real = _validate_transcript(transcript_path)
    parsed = _parse_transcript(real, start, known_first_ts)
    decision = _index_decision(parsed, parsed["end_offset"])
    if decision not in ("accepted", "duplicate"):
        return {"ok": True, "skipped": decision, "session_id": parsed["session_id"], "end_offset": parsed["end_offset"]}
    parsed = _plug_in_chunking(parsed)
    graph = _ingest_graph(parsed)
    vectors = _ingest_vectors(parsed)
    lexical = _ingest_lexical(parsed)
    try:
        outcomes = record_turn_outcomes(real, start)
    except Exception as exc:  # outcome recording is secondary; the transcript is already indexed
        outcomes = {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "session_id": parsed["session_id"], "project": parsed["project"],
            "graph": graph, "vector": vectors,
            "lexical": lexical, "outcomes": outcomes,
            "end_offset": parsed["end_offset"], "first_ts": parsed["first_ts"]}


_SEARCH_CACHE = OrderedDict()   # (normalized query, top_k) -> (stored_at, result)
_SEARCH_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE_TTL = float(os.environ.get("ENSEMBLE_HYBRID_CACHE_TTL", "60"))
_SEARCH_CACHE_MAX = 256


def hybrid_search(query, top_k, rerank=True):
    """Cached front for the search. Only complete (non-degraded) results are cached, for a short TTL."""
    key = (" ".join(query.lower().split()), top_k, rerank)
    now = time.monotonic()
    with _SEARCH_CACHE_LOCK:
        entry = _SEARCH_CACHE.get(key)
        if entry is not None and now - entry[0] < _SEARCH_CACHE_TTL:
            _SEARCH_CACHE.move_to_end(key)
            return dict(entry[1], cached=True)
        _SEARCH_CACHE.pop(key, None)
    result = _hybrid_search_uncached(query, top_k, rerank)
    if not result["degraded"]:
        with _SEARCH_CACHE_LOCK:
            _SEARCH_CACHE[key] = (now, result)
            while len(_SEARCH_CACHE) > _SEARCH_CACHE_MAX:
                _SEARCH_CACHE.popitem(last=False)
    return dict(result, cached=False)


class HybridHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeEnsembleHybrid/1"

    def _send(self, status, payload):
        body = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message):
        self._send(status, {"ok": False, "error": {"code": code, "message": message}})

    def _store_prompt(self, body):
        session_id, prompt = body.get("session_id"), body.get("prompt")
        timestamp, cwd = body.get("timestamp"), body.get("cwd", "")
        if not isinstance(session_id, str) or not session_id:
            self._error(400, "invalid_request", "session_id must be a non-empty string")
            return
        if not isinstance(prompt, str) or not prompt.strip():
            self._error(400, "invalid_request", "prompt must be a non-empty string")
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            self._error(413, "payload_too_large", f"prompt is limited to {MAX_PROMPT_CHARS} characters")
            return
        if not isinstance(timestamp, str) or not timestamp:
            self._error(400, "invalid_request", "timestamp must be a non-empty string")
            return
        if not isinstance(cwd, str):
            self._error(400, "invalid_request", "cwd must be a string")
            return
        try:
            self._send(200, store_prompt(session_id, prompt, timestamp, cwd))
        except DependencyError as exc:
            self._error(503, "service_unavailable", str(exc))

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "service": "hybrid-search",
                             "depends_on": ["embedding", "vector", "lexical"]})
            return
        self._error(404, "not_found", "unknown path")

    def do_POST(self):
        if self.path not in {"/search", "/prompts", "/sessions/ingest"}:
            self._error(404, "not_found", "unknown path")
            return
        limit = MAX_PROMPT_BODY_BYTES if self.path == "/prompts" else MAX_BODY_BYTES
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > limit:
            self._error(400, "invalid_request", f"request body must be between 1 byte and {limit} bytes")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._error(400, "invalid_request", "body must be UTF-8 JSON")
            return
        if not isinstance(body, dict):
            self._error(400, "invalid_request", "body must be a JSON object")
            return
        if self.path == "/prompts":
            self._store_prompt(body)
            return
        if self.path == "/sessions/ingest":
            path = body.get("transcript_path")
            if not isinstance(path, str) or not path:
                self._error(400, "invalid_request", "transcript_path must be a non-empty string")
                return
            try:
                real = _validate_transcript(path)
            except ValueError as exc:
                self._error(400, "invalid_request", str(exc))
                return
            threading.Thread(target=_run_ingest, args=(real,), name="session-ingest", daemon=True).start()
            self._send(202, {"ok": True, "status": "accepted", "transcript": os.path.basename(real)})
            return
        query, top_k = body.get("query"), body.get("top_k", 10)
        rerank = body.get("rerank", True)
        if not isinstance(rerank, bool):
            self._error(400, "invalid_request", "rerank must be true or false")
            return
        if not isinstance(query, str) or not query.strip():
            self._error(400, "invalid_request", "query must be a non-empty string")
            return
        if len(query) > MAX_QUERY_CHARS:
            self._error(413, "payload_too_large", f"query is limited to {MAX_QUERY_CHARS} characters")
            return
        if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            self._error(400, "invalid_request", f"top_k must be an integer from 1 to {MAX_TOP_K}")
            return
        try:
            self._send(200, hybrid_search(query, top_k, rerank))
        except DependencyError as exc:
            self._error(503, "service_unavailable", str(exc))

    def log_message(self, fmt, *args):
        return


def create_server(host, port):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("hybrid search only accepts loopback binds")
    return ThreadingHTTPServer((host, port), HybridHandler)


def main():
    host = os.environ.get("ENSEMBLE_HYBRID_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ENSEMBLE_HYBRID_PORT", str(DEFAULT_PORT)))
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
