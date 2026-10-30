"""Unit tests for embedding runtime selection and OpenVINO execution."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import embed_lib


class FakeEncoding:
    ids = [1, 2]
    attention_mask = [1, 1]


class FakeTokenizer:
    def enable_truncation(self, _maximum):
        pass

    def encode_batch(self, texts):
        return [FakeEncoding() for _ in texts]


class FakePort:
    def __init__(self, name):
        self.name = name

    def get_any_name(self):
        return self.name


class FakeCompiledModel:
    inputs = [FakePort("input_ids"), FakePort("attention_mask"), FakePort("token_type_ids")]

    def output(self, _index):
        return "last_hidden_state"

    def __call__(self, feed):
        batch, width = feed["input_ids"].shape
        hidden = np.ones((batch, width, embed_lib.DIMENSIONS), dtype=np.float32)
        return {"last_hidden_state": hidden}


class FakeCore:
    available_devices = ["CPU", "GPU"]

    def __init__(self):
        self.compiled = None

    def read_model(self, path):
        return path

    def compile_model(self, model, device, config):
        self.compiled = (model, device, config)
        return FakeCompiledModel()


class FakeOrtSession:
    def get_inputs(self):
        return [types.SimpleNamespace(name=name)
                for name in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, _outputs, feed):
        batch, width = feed["input_ids"].shape
        return [np.ones((batch, width, embed_lib.DIMENSIONS), dtype=np.float32)]


class EmbedderRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.root.name, "onnx"))
        for name in ("model.onnx", "model_fp16.onnx"):
            with open(os.path.join(self.root.name, "onnx", name), "wb") as handle:
                handle.write(b"test model")
        self.model_dir = patch.object(embed_lib, "MODEL_DIR", self.root.name)
        self.model_dir.start()
        self.tokenizer = patch.object(embed_lib.Tokenizer, "from_file", return_value=FakeTokenizer())
        self.tokenizer.start()

    def tearDown(self):
        patch.stopall()
        self.root.cleanup()

    def test_openvino_gpu_auto_precision_selects_fp16_model(self):
        core = FakeCore()
        fake_openvino = types.SimpleNamespace(Core=lambda: core)
        with patch.dict(sys.modules, {"openvino": fake_openvino}):
            model = embed_lib.Embedder(runtime="openvino", device="GPU", precision="AUTO")

        self.assertEqual(model.backend, "openvino")
        self.assertEqual((model.device, model.precision, model.variant),
                         ("GPU", "FP16", "model_fp16.onnx"))
        self.assertEqual(model.profile, "nomic-embed-text-v1.5@fp16")
        compiled_path, compiled_device, config = core.compiled
        self.assertTrue(compiled_path.endswith(os.path.join("onnx", "model_fp16.onnx")))
        self.assertEqual(compiled_device, "GPU")
        self.assertEqual(config["INFERENCE_PRECISION_HINT"], "f16")

        vectors = model.embed_documents(["a small test"])
        self.assertEqual(vectors.shape, (1, embed_lib.DIMENSIONS))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    def test_auto_runtime_falls_back_to_onnxruntime_cpu_without_openvino(self):
        with patch.dict(sys.modules, {"openvino": None}):
            with patch.object(embed_lib.ort, "InferenceSession", return_value=FakeOrtSession()):
                model = embed_lib.Embedder(runtime="auto", device="AUTO", precision="AUTO")

        self.assertEqual((model.backend, model.device, model.precision),
                         ("onnxruntime", "CPU", "FP32"))
        self.assertEqual(model.profile, "nomic-embed-text-v1.5@fp32")
        self.assertIn("OpenVINO GPU unavailable", model.fallback_reason)
        self.assertEqual(model.embed_documents(["fallback test"]).shape,
                         (1, embed_lib.DIMENSIONS))

    def test_explicit_gpu_request_fails_when_gpu_is_unavailable(self):
        core = types.SimpleNamespace(available_devices=["CPU"])
        fake_openvino = types.SimpleNamespace(Core=lambda: core)
        with patch.dict(sys.modules, {"openvino": fake_openvino}):
            with self.assertRaisesRegex(RuntimeError, "GPU.*not available"):
                embed_lib.Embedder(runtime="openvino", device="GPU", precision="AUTO")


if __name__ == "__main__":
    unittest.main()
