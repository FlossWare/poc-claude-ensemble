"""ChromaDB vector store for local embeddings (embedded, persistent, no server).

Collections use cosine distance. Each record stores the text it was embedded
from, so search results are self-contained. Collection metadata records the
embedding model and dimensions, so a mismatched model can be detected.
"""

import os

import chromadb

from embed_lib import DIMENSIONS, MODEL_ID

STORE_DIR = os.path.expanduser(os.environ.get("ENSEMBLE_VECTOR_STORE", "~/.FlossWare/data/chromadb/1.5.9"))


def client():
    os.makedirs(STORE_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=STORE_DIR)


def collection(name):
    col = client().get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine", "model": MODEL_ID, "dimensions": DIMENSIONS},
    )
    stored = col.metadata or {}
    if stored.get("model") != MODEL_ID or stored.get("dimensions") != DIMENSIONS:
        raise RuntimeError(f"collection {name} was built with {stored}, not {MODEL_ID}")
    return col
