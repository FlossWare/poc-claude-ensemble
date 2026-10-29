"""Local CPU embeddings with nomic-embed-text-v1.5 (ONNX, no network access).

Documents are embedded with the "search_document: " prefix and queries with
"search_query: ". Vectors are mean-pooled over real tokens and L2-normalized,
so cosine similarity is a dot product.
"""

import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.path.expanduser("~/.FlossWare/models/nomic-embed-text-v1.5")
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DIMENSIONS = 768
MAX_TOKENS = 512
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder:
    def __init__(self, variant="model.onnx", threads=None, cpu_mem_arena=False):
        self.tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or os.cpu_count()
        options.enable_cpu_mem_arena = cpu_mem_arena
        self.session = ort.InferenceSession(
            os.path.join(MODEL_DIR, "onnx", variant),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _encode_raw(self, texts):
        encodings = self.tokenizer.encode_batch(texts)
        width = max(len(e.ids) for e in encodings)
        ids = np.zeros((len(texts), width), dtype=np.int64)
        mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, e in enumerate(encodings):
            ids[row, : len(e.ids)] = e.ids
            mask[row, : len(e.ids)] = e.attention_mask
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def embed_documents(self, texts):
        return self._encode_raw([DOC_PREFIX + t for t in texts])

    def embed_queries(self, texts):
        return self._encode_raw([QUERY_PREFIX + t for t in texts])
