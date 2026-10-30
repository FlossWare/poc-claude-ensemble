"""Local Nomic embeddings with selectable ONNX Runtime/OpenVINO execution.

Documents use the search_document prefix and queries use search_query. Both
backends share tokenization, masked mean pooling, and L2 normalization so the
REST contract and vector dimensions remain unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

MODEL_DIR = os.path.expanduser(os.environ.get(
    "ENSEMBLE_EMBEDDING_MODEL_DIR",
    "~/.FlossWare/models/embeddings/nomic-ai/nomic-embed-text-v1.5/"
    "e9b6763023c676ca8431644204f50c2b100d9aab",
))
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DIMENSIONS = 768
MAX_TOKENS = 512
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BACKENDS = {"onnxruntime-cpu", "openvino-cpu", "openvino-gpu"}


class Embedder:
    """Encode Nomic text with an explicitly selected local inference backend.

    CPU ONNX Runtime FP32 remains the default. OpenVINO GPU is opt-in and uses
    the pinned FP16 graph. A requested GPU backend fails closed if OpenVINO or
    a GPU device is unavailable; it never silently falls back to CPU.
    """

    def __init__(
        self,
        variant: str | None = None,
        threads: int | None = None,
        cpu_mem_arena: bool = False,
        backend: str | None = None,
        performance_hint: str | None = None,
    ):
        self.backend = backend or os.environ.get(
            "ENSEMBLE_EMBED_BACKEND", "onnxruntime-cpu"
        ).strip().lower()
        if self.backend not in BACKENDS:
            raise ValueError(
                f"unsupported embedding backend {self.backend!r}; "
                f"choose one of {', '.join(sorted(BACKENDS))}"
            )

        default_variant = (
            "model_fp16.onnx" if self.backend == "openvino-gpu" else "model.onnx"
        )
        self.model_file = variant or default_variant
        model_path = Path(MODEL_DIR) / "onnx" / self.model_file
        if not model_path.is_file():
            raise FileNotFoundError(f"embedding model graph not found: {model_path}")

        self.tokenizer = Tokenizer.from_file(str(Path(MODEL_DIR) / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.input_names: set[str]
        self.runtime = ""
        self.device = ""
        self.precision = "FP32"

        if self.backend == "onnxruntime-cpu":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads or os.cpu_count() or 1
            options.enable_cpu_mem_arena = cpu_mem_arena
            self.session = ort.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
            self.input_names = {item.name for item in self.session.get_inputs()}
            self.runtime = f"onnxruntime/{ort.__version__}"
            self.device = self.session.get_providers()[0]
            self.output_port = None
        else:
            try:
                import openvino as ov
            except ImportError as exc:
                raise RuntimeError(
                    f"{self.backend} requires the optional 'openvino' package"
                ) from exc

            core = ov.Core()
            available = list(core.available_devices)
            if self.backend == "openvino-gpu":
                gpu_devices = [name for name in available if name.upper().startswith("GPU")]
                if not gpu_devices:
                    raise RuntimeError(
                        "OpenVINO GPU was requested but no GPU device is available; "
                        f"available devices: {available}"
                    )
                device = gpu_devices[0]
                self.precision = "FP16"
            else:
                if "CPU" not in available:
                    raise RuntimeError(
                        f"OpenVINO CPU was requested but CPU is unavailable: {available}"
                    )
                device = "CPU"

            hint = (performance_hint or os.environ.get(
                "ENSEMBLE_OPENVINO_PERFORMANCE_HINT", "THROUGHPUT"
            )).strip().upper()
            if hint not in {"LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
                raise ValueError(f"unsupported OpenVINO performance hint: {hint!r}")
            self.session = core.compile_model(
                str(model_path), device, {"PERFORMANCE_HINT": hint}
            )
            self.input_names = {str(port.get_any_name()) for port in self.session.inputs}
            self.output_port = self.session.output(0)
            self.runtime = f"openvino/{ov.__version__}"
            self.device = device

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        width = max(len(item.ids) for item in encodings)
        ids = np.zeros((len(texts), width), dtype=np.int64)
        mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, item in enumerate(encodings):
            ids[row, :len(item.ids)] = item.ids
            mask[row, :len(item.ids)] = item.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        if self.backend == "onnxruntime-cpu":
            hidden = self.session.run(None, feed)[0]
        else:
            result = self.session(feed)
            hidden = result[self.output_port]

        hidden = np.asarray(hidden, dtype=np.float32)
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(
            weights.sum(axis=1), 1e-9
        )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode_raw([DOC_PREFIX + text for text in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode_raw([QUERY_PREFIX + text for text in texts])
