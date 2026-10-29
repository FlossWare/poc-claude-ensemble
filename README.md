# poc-claude-ensemble

Proof-of-concept services for Claude Ensemble, the local REST services behind prompt-time memory, transcript search, and ingestion. This repository is the baseline the next Ensemble work branches from. The previous implementation is archived at FlossWare-Archives/claude-ensemble-legacy.

## Services

Each service is a standard-library HTTP server bound to loopback, with its own tests.

| Service | Port | Contract |
|---|---|---|
| Embedding (`services/embeddings`) | 8770 | nomic-embed-text-v1.5 vectors, full precision |
| Vector (`services/embeddings`) | 8771 | ChromaDB-backed exact search |
| Lexical (`services/embeddings`) | 8772 | SQLite FTS5 keyword search, incremental writes |
| Hybrid (`services/embeddings`) | 8773 | Search orchestration and transcript ingestion |
| Reranking (`services/embeddings`) | 8774 | Local cross-encoder |
| Chunking (`services/chunking`) | 8775 | Versioned, deterministic chunk profiles |
| Tokenization (`services/tokenization`) | 8776 | Versioned analyzers |
| Index (`services/index`) | 8777 | Source manifest: revision order and tombstones |
| Extraction (`services/extraction`) | 8778 | Exact identifiers, with uncertainty flagged |
| Intent (`services/intent`) | 8779 | Query intent with evidence |
| Ranking (`services/ranking`) | 8780 | Explainable reciprocal rank fusion |
| Knowledge (`services/knowledge`) | 8781 | Sourced records; revisions checked against Index |

## Running the tests

From the repository root, run the whole suite with pytest:

    python3 -m pytest services -q

Each service directory also runs on its own, for example `cd services/chunking && python3 -m unittest -v test_chunking_service`. The embeddings tests load the real nomic and cross-encoder models from `~/.FlossWare/models` and skip only when those files are absent. Tests that cross a service boundary start the real services in-process on ephemeral loopback ports.

## Status

The services run from `shared/` in the working deployment and are copied here as they stand.

- Keyword analysis: index text and queries go through the Tokenization service (`lexical-v1`). Existing databases are upgraded with `services/embeddings/migrate_lexical_tokenization.py`, run with the lexical service stopped.
- Sourced documents: `services/embeddings/build_knowledge_chunks.py` stores each document as a Knowledge record, chunks it through Chunking, and writes vectors to `knowledge_chunks`. Lexical indexing covers transcript sessions only.
- Not yet built: Thompson outcome recording from transcripts, and worker/arbiter quality monitoring.
