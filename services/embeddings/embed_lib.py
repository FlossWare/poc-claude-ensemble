"""Nomic embeddings with OpenVINO GPU acceleration and ONNX Runtime CPU fallback.

Documents use the "search_document: " prefix and queries use "search_query: ".
Vectors are mean-pooled over real tokens and L2-normalized, so cosine similarity
is a dot product. OpenVINO is optional; ONNX Runtime remains the CPU fallback.
"""

import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ~/.FlossWare/models/<kind>/<model-name>/<revision>/ ; the revision is the Hugging Face commit.
MODEL_DIR = os.path.expanduser(os.environ.get(
    "ENSEMBLE_EMBEDDING_MODEL_DIR",
    "~/.FlossWare/models/embeddings/nomic-ai/nomic-embed-text-v1.5/e9b6763023c676ca8431644204f50c2b100d9aab",
))
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_NAME = "nomic-embed-text-v1.5"
DIMENSIONS = 768
MAX_TOKENS = 512
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder:
    """Embedding model with explicit runtime selection and observable execution metadata.

    ENSEMBLE_EMBED_RUNTIME:
        auto          Prefer OpenVINO GPU when available; otherwise ONNX Runtime CPU.
        openvino      Require OpenVINO; AUTO device prefers GPU, then CPU.
        onnxruntime   Require ONNX Runtime CPU.

    ENSEMBLE_EMBED_DEVICE:
        AUTO, GPU, or CPU. GPU requires OpenVINO.

    ENSEMBLE_EMBED_PRECISION:
        AUTO, FP16, or FP32. AUTO selects FP16 for GPU and FP32 for CPU.
    """

    def __init__(self, variant=None, threads=None, cpu_mem_arena=False,
                 runtime=None, device=None, precision=None):
        self.tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)

        self.requested_runtime = (runtime or os.environ.get("ENSEMBLE_EMBED_RUNTIME", "auto")).strip().lower()
        self.requested_device = (device or os.environ.get("ENSEMBLE_EMBED_DEVICE", "AUTO")).strip().upper()
        self.requested_precision = (precision or os.environ.get("ENSEMBLE_EMBED_PRECISION", "AUTO")).strip().upper()
        if self.requested_runtime not in {"auto", "openvino", "onnxruntime"}:
            raise ValueError("ENSEMBLE_EMBED_RUNTIME must be auto, openvino, or onnxruntime")
        if self.requested_device not in {"AUTO", "CPU", "GPU"}:
            raise ValueError("ENSEMBLE_EMBED_DEVICE must be AUTO, CPU, or GPU")
        if self.requested_precision not in {"AUTO", "FP16", "FP32"}:
            raise ValueError("ENSEMBLE_EMBED_PRECISION must be AUTO, FP16, or FP32")

        self.session = None
        self.compiled_model = None
        self.fallback_reason = None

        if self.requested_runtime == "onnxruntime":
            if self.requested_device == "GPU":
                raise ValueError("GPU execution requires ENSEMBLE_EMBED_RUNTIME=openvino")
            if self.requested_precision == "FP16":
                raise ValueError("FP16 execution requires the OpenVINO runtime")
            self._init_onnxruntime(threads, cpu_mem_arena)
        elif self.requested_runtime == "openvino":
            self._init_openvino(variant=variant, threads=threads)
        elif self.requested_device == "CPU":
            if self.requested_precision == "FP16":
                raise ValueError("FP16 execution requires the OpenVINO GPU")
            self._init_onnxruntime(threads, cpu_mem_arena)
        else:
            self._init_auto(variant=variant, threads=threads, cpu_mem_arena=cpu_mem_arena)

        self.profile = f"{MODEL_NAME}@{self.precision.lower()}"
        self.info = {
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "variant": self.variant,
            "profile": self.profile,
            "fallback_reason": self.fallback_reason,
        }

    def _init_auto(self, variant, threads, cpu_mem_arena):
        """Use GPU FP16 when available; otherwise preserve the established CPU FP32 path."""
        try:
            import openvino as ov

            core = ov.Core()
            if "GPU" not in core.available_devices:
                if self.requested_device == "GPU" or self.requested_precision == "FP16":
                    raise RuntimeError("OpenVINO GPU device is not available")
                self._init_onnxruntime(threads, cpu_mem_arena)
                return
            self._compile_openvino(core, "GPU", variant, threads)
        except Exception as exc:
            # AUTO is deliberately conservative: only implicit GPU selection may fall back.
            if self.requested_device == "GPU" or self.requested_precision == "FP16":
                raise
            self.fallback_reason = f"OpenVINO GPU unavailable; using ONNX Runtime CPU: {exc}"
            self._init_onnxruntime(threads, cpu_mem_arena)

    def _init_openvino(self, variant, threads):
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO was requested but is not installed in this Python environment") from exc

        core = ov.Core()
        available = list(core.available_devices)
        if self.requested_device == "AUTO":
            device = "GPU" if "GPU" in available else "CPU"
        else:
            device = self.requested_device
        if device not in available:
            raise RuntimeError(f"OpenVINO device {device!r} is unavailable; available devices: {available}")
        self._compile_openvino(core, device, variant, threads)

    def _compile_openvino(self, core, device, variant, threads):
        precision = self.requested_precision
        if precision == "AUTO":
            precision = "FP16" if device == "GPU" else "FP32"
        selected_variant = variant or ("model_fp16.onnx" if precision == "FP16" else "model.onnx")
        model_path = os.path.join(MODEL_DIR, "onnx", selected_variant)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"embedding model variant not found: {model_path}")

        model = core.read_model(model_path)
        config = {"INFERENCE_PRECISION_HINT": "f16" if precision == "FP16" else "f32"}
        self.compiled_model = core.compile_model(model, device, config)
        self.input_names = {port.get_any_name() for port in self.compiled_model.inputs}
        self.backend = "openvino"
        self.device = device
        self.precision = precision
        self.variant = selected_variant

    def _init_onnxruntime(self, threads, cpu_mem_arena):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or int(os.environ.get("ENSEMBLE_EMBED_THREADS", os.cpu_count()))
        options.enable_cpu_mem_arena = cpu_mem_arena
        self.variant = "model.onnx"
        self.session = ort.InferenceSession(
            os.path.join(MODEL_DIR, "onnx", self.variant),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.backend = "onnxruntime"
        self.device = "CPU"
        self.precision = "FP32"

    def _infer(self, feed):
        if self.compiled_model is not None:
            result = self.compiled_model(feed)
            return result[self.compiled_model.output(0)]
        return self.session.run(None, feed)[0]

    def _encode_raw(self, texts):
        encodings = self.tokenizer.encode_batch(texts)
        width = max(len(e.ids) for e in encodings)
        ids = np.zeros((len(texts), width), dtype=np.int64)
        mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            ids[row, : len(encoding.ids)] = encoding.ids
            mask[row, : len(encoding.ids)] = encoding.attention_mask
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._infer(feed)
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def embed_documents(self, texts):
        return self._encode_raw([DOC_PREFIX + text for text in texts])

    def embed_queries(self, texts):
        return self._encode_raw([QUERY_PREFIX + text for text in texts])
