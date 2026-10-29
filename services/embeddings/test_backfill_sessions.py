"""Backfill selection and the record_outcomes switch in the hybrid ingest."""
import json
import os
import tempfile
import time
import unittest
from unittest import mock

import backfill_sessions as bf
import hybrid_service as hs


def make_transcript(root, project, name, lines=3):
    directory = os.path.join(root, project)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(lines):
            handle.write(json.dumps({"uuid": f"{name}-{index}"}) + "\n")
    return os.path.realpath(path)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_unprocessed_transcripts_are_selected_oldest_first(self):
        old = make_transcript(self.root, "p", "old.jsonl")
        time.sleep(0.01)
        new = make_transcript(self.root, "p", "new.jsonl")
        os.utime(old, (1_000_000, 1_000_000))
        self.assertEqual(bf.pending(self.root, {}), [old, new])

    def test_fully_ingested_transcripts_are_skipped(self):
        path = make_transcript(self.root, "p", "done.jsonl")
        state = {path: {"offset": os.path.getsize(path)}}
        self.assertEqual(bf.pending(self.root, state), [])

    def test_partially_ingested_transcript_is_selected(self):
        path = make_transcript(self.root, "p", "partial.jsonl")
        state = {path: {"offset": 10}}
        self.assertEqual(bf.pending(self.root, state), [path])

    def test_subagent_transcripts_are_not_selected(self):
        make_transcript(self.root, os.path.join("p", "sub"), "journal.jsonl")
        self.assertEqual(bf.pending(self.root, {}), [])


class OutcomeSwitchTests(unittest.TestCase):
    """ingest_session must not touch Thompson when record_outcomes is false."""

    def stub_ingest(self, path):
        patches = [
            mock.patch.object(hs, "_validate_transcript", return_value=path),
            mock.patch.object(hs, "_index_decision", return_value="accepted"),
            mock.patch.object(hs, "_plug_in_chunking", side_effect=lambda parsed: parsed),
            mock.patch.object(hs, "_ingest_graph", return_value={}),
            mock.patch.object(hs, "_ingest_vectors", return_value={}),
            mock.patch.object(hs, "_ingest_lexical", return_value={}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_outcomes_are_skipped_when_switched_off(self):
        path = make_transcript(tempfile.mkdtemp(), "p", "s.jsonl")
        self.stub_ingest(path)
        with mock.patch.object(hs, "record_turn_outcomes") as recorder:
            result = hs.ingest_session(path, record_outcomes=False)
        recorder.assert_not_called()
        self.assertEqual(result["outcomes"], {"skipped": "record_outcomes is false"})

    def test_outcomes_are_recorded_by_default(self):
        path = make_transcript(tempfile.mkdtemp(), "p", "s.jsonl")
        self.stub_ingest(path)
        with mock.patch.object(hs, "record_turn_outcomes", return_value={"accepted": 0}) as recorder:
            hs.ingest_session(path)
        recorder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
