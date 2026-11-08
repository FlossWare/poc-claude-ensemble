import json
import socket
import unittest
from unittest.mock import patch

import ingestion_service as ingestion


class IngestionSourceTests(unittest.TestCase):
    def test_inline_text_preview_is_non_mutating(self):
        result = ingestion.preview({"title": "Notes", "text": "A small source document."})
        self.assertTrue(result["ok"])
        self.assertFalse(result["persisted"])
        self.assertEqual(result["source"]["origin"], "inline-text")
        self.assertEqual(result["preview"], "A small source document.")
        self.assertIn("chunking", result["planned_stages"])

    def test_html_extractor_omits_script_and_style_content(self):
        text, media_type = ingestion._extract(
            b"<html><body><h1>Visible</h1><script>secret()</script><p>Text</p></body></html>",
            "text/html",
            "page.html",
        )
        self.assertEqual(media_type, "text/html")
        self.assertIn("Visible", text)
        self.assertIn("Text", text)
        self.assertNotIn("secret", text)

    def test_private_and_loopback_resolution_is_rejected(self):
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
        with patch.object(ingestion.socket, "getaddrinfo", return_value=answers):
            with self.assertRaises(ingestion.IngestionError) as caught:
                ingestion._global_address("localhost")
        self.assertEqual(caught.exception.code, "unsafe_uri")

    def test_unsupported_uri_scheme_is_rejected_before_fetch(self):
        with self.assertRaises(ingestion.IngestionError) as caught:
            ingestion.resolve_source({"uri": "s3://example.org/document.pdf"})
        self.assertEqual(caught.exception.code, "unsupported_scheme")

    def test_ftp_connector_requires_explicit_opt_in(self):
        with patch.dict("os.environ", {"POC_CLAUDE_ENSEMBLE_INGESTION_ENABLE_FTP": ""}):
            with self.assertRaises(ingestion.IngestionError) as caught:
                ingestion.resolve_source({"uri": "ftp://example.org/report.txt"})
        self.assertEqual(caught.exception.code, "transport_disabled")

    def test_opted_in_ftp_fetcher_uses_anonymous_login_and_extracts_text(self):
        class FakeFTP:
            def __init__(self, pinned_host):
                self.pinned_host = pinned_host
                self.connected = None
                self.logged_in = False
                self.passive = False
                self.closed = False

            def connect(self, host, port, timeout):
                self.connected = (host, port, timeout)

            def login(self):
                self.logged_in = True

            def set_pasv(self, enabled):
                self.passive = enabled

            def retrbinary(self, command, callback, blocksize):
                self.command = command
                self.blocksize = blocksize
                callback(b"FTP document content")

            def close(self):
                self.closed = True

        clients = []

        def make_client(address):
            client = FakeFTP(address)
            clients.append(client)
            return client

        with patch.dict("os.environ", {"POC_CLAUDE_ENSEMBLE_INGESTION_ENABLE_FTP": "true"}):
            with patch.object(ingestion, "_global_address", return_value="93.184.216.34"):
                with patch.object(ingestion, "_PinnedFTP", side_effect=make_client):
                    source = ingestion.resolve_source({"uri": "ftp://example.org/report.txt"})

        self.assertEqual(source["text"], "FTP document content")
        self.assertEqual(source["media_type"], "text/plain")
        self.assertEqual(clients[0].connected[0], "93.184.216.34")
        self.assertTrue(clients[0].logged_in)
        self.assertTrue(clients[0].passive)
        self.assertTrue(clients[0].closed)

    def test_upload_rejects_invalid_base64(self):
        with self.assertRaises(ingestion.IngestionError):
            ingestion.resolve_source({"filename": "notes.txt", "content_base64": "not base64!"})

    def test_ingest_requires_a_valid_preview_token(self):
        with self.assertRaises(ingestion.IngestionError) as caught:
            ingestion.ingest({})
        self.assertEqual(caught.exception.code, "preview_required")

    def test_preview_token_is_single_use_after_success(self):
        preview_result = ingestion.preview({"text": "single-use source"})
        preview_id = preview_result["preview_id"]
        def reject_knowledge(_base, path, _payload, timeout=60):
            if path == "/records":
                return {"ok": True, "decision": "rejected"}
            return {"ok": True}

        with patch.object(ingestion, "_post_json", side_effect=reject_knowledge):
            result = ingestion.ingest({"preview_id": preview_id})
        self.assertEqual(result["status"], "rejected")
        with self.assertRaises(ingestion.IngestionError):
            ingestion.ingest({"preview_id": preview_id})

    def test_ingest_uses_existing_knowledge_chunk_embedding_and_indexes(self):
        calls = []

        def fake_post(base, path, payload, timeout=60):
            calls.append((base, path, payload))
            if path == "/records":
                return {"ok": True, "decision": "accepted"}
            if path == "/chunk":
                return {"ok": True, "chunks": [
                    {"chunk_id": "chunk-1", "index": 0, "text": "First chunk"},
                    {"chunk_id": "chunk-2", "index": 1, "text": "Second chunk"},
                ]}
            if path == "/embeddings":
                return {"ok": True, "embeddings": [[0.1, 0.2], [0.3, 0.4]]}
            return {"ok": True}

        preview_result = ingestion.preview({"title": "Test", "text": "A test source."})
        with patch.object(ingestion, "resolve_source", side_effect=AssertionError("ingest must use the reviewed preview")):
            with patch.object(ingestion, "_post_json", side_effect=fake_post):
                result = ingestion.ingest({"preview_id": preview_result["preview_id"]})

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["vectors"], 2)
        paths = [path for _, path, _ in calls]
        self.assertIn("/records", paths)
        self.assertIn("/chunk", paths)
        self.assertIn("/embeddings", paths)
        self.assertIn("/knowledge/documents", paths)
        self.assertIn("/collections/knowledge_chunks/upsert", paths)
        self.assertIn("/collections/knowledge_chunks/delete-superseded", paths)
        self.assertIn("/memory/append-once", paths)


if __name__ == "__main__":
    unittest.main()
