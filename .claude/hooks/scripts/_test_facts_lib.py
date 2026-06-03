#!/usr/bin/env python3
"""Tests for _facts_lib.py — session facts + Knowledge Archive (ADR-079 Increment 4).

_facts_lib is the TOP layer, depending on all four lower modules (_core_lib,
_capture_lib, _validation_lib, _snapshot_lib). The TestCrossModuleDepsImported
class is the inc3-lesson regression guard: it ensures the cross-module helpers
the facts functions call (several inside try/except, the exact inc3 failure mode)
are actually importable, so a dropped import cannot silently swallow output.

Run: python3 -m pytest _test_facts_lib.py -v
  or: python3 _test_facts_lib.py
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import _facts_lib as facts


class TestCrossModuleDepsImported(unittest.TestCase):
    """Regression (ADR-079 inc4, inc3 bug class): every lower-module helper the
    facts functions call must be bound in _facts_lib, or a swallowed NameError
    silently drops Knowledge-Archive fields."""

    def test_all_cross_module_helpers_bound(self):
        for name in (
            # _core_lib
            "atomic_write", "sot_paths", "estimate_tokens",
            "_DIAG_EVIDENCE_RE", "_DIAG_SELECTED_RE",
            # _capture_lib
            "capture_git_state", "detect_conversation_phase",
            "detect_phase_transitions", "detect_ulw_mode", "extract_completion_state",
            # _validation_lib
            "parse_review_verdict",
            # _snapshot_lib
            "_extract_decisions",
        ):
            self.assertTrue(hasattr(facts, name),
                            f"_facts_lib must import {name} from its source module")


class TestExtractPathTags(unittest.TestCase):
    def test_snake_and_camel_and_extension(self):
        tags = facts.extract_path_tags(["src/user_auth.py", "lib/AuthService.py"])
        self.assertIn("user", tags)
        self.assertIn("auth", tags)
        self.assertIn("service", tags)
        self.assertEqual(tags, sorted(tags))  # sorted unique

    def test_empty_and_skip(self):
        self.assertEqual(facts.extract_path_tags([]), [])
        self.assertEqual(facts.extract_path_tags([""]), [])


class TestValidateSessionFacts(unittest.TestCase):
    def test_fills_required_defaults(self):
        out = facts._validate_session_facts({"session_id": "s1"})
        # RLM-critical keys must all be present after validation
        for key in ("session_id", "timestamp", "modified_files", "tools_used",
                    "final_status", "tags", "diagnosis_patterns"):
            self.assertIn(key, out)
        self.assertEqual(out["session_id"], "s1")


class TestPatternExtractors(unittest.TestCase):
    def test_classify_error_patterns_runs(self):
        entries = [
            {"type": "tool_use", "tool_name": "Edit", "tool_use_id": "e1",
             "file_path": "/x.py", "timestamp": "t1"},
            {"type": "tool_result", "tool_use_id": "e1",
             "is_error": True, "content": "Error: String to replace not found", "timestamp": "t2"},
        ]
        result = facts._classify_error_patterns(entries)
        self.assertIsInstance(result, (list, dict))

    def test_extract_success_patterns_runs(self):
        entries = [
            {"type": "tool_use", "tool_name": "Edit", "tool_use_id": "e1",
             "file_path": "/x.py", "timestamp": "t1"},
            {"type": "tool_result", "tool_use_id": "e1", "is_error": False,
             "content": "ok", "timestamp": "t2"},
            {"type": "tool_use", "tool_name": "Bash", "tool_use_id": "b1",
             "command": "pytest", "timestamp": "t3"},
            {"type": "tool_result", "tool_use_id": "b1", "is_error": False,
             "content": "passed", "timestamp": "t4"},
        ]
        result = facts._extract_success_patterns(entries)
        self.assertIsInstance(result, (list, dict))


class TestExtractSessionFacts(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.proj)

    def test_end_to_end_returns_dict_no_nameerror(self):
        # Exercises the cross-module calls (capture_git_state, detect_*,
        # extract_completion_state, estimate_tokens, _extract_decisions).
        # A missing cross-module import would raise NameError here.
        entries = [
            {"type": "user_message", "content": "implement the parser", "timestamp": "t1"},
            {"type": "assistant_text", "content": "Working on it", "timestamp": "t2"},
            {"type": "tool_use", "tool_name": "Write", "tool_use_id": "w1",
             "file_path": str(self.proj / "parser.py"), "timestamp": "t3"},
            {"type": "tool_result", "tool_use_id": "w1", "is_error": False,
             "content": "ok", "timestamp": "t4"},
        ]
        out = facts.extract_session_facts("sess-1", "Stop", str(self.proj), entries)
        self.assertIsInstance(out, dict)
        # RLM-critical keys present (via _validate_session_facts)
        self.assertIn("session_id", out)
        self.assertIn("tags", out)
        self.assertIn("tools_used", out)


class TestKnowledgeIndexIO(unittest.TestCase):
    def setUp(self):
        self.snap = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.snap)

    def test_replace_or_append_and_dedup(self):
        ki = self.snap / "knowledge-index.jsonl"
        facts.replace_or_append_session_facts(str(ki), {"session_id": "s1", "tags": ["a"]})
        facts.replace_or_append_session_facts(str(ki), {"session_id": "s1", "tags": ["b"]})  # dedup
        facts.replace_or_append_session_facts(str(ki), {"session_id": "s2", "tags": ["c"]})
        lines = [json.loads(ln) for ln in ki.read_text(encoding="utf-8").splitlines() if ln.strip()]
        ids = [e.get("session_id") for e in lines]
        self.assertEqual(ids.count("s1"), 1, "same session_id must be deduped")
        self.assertIn("s2", ids)

    def test_cleanup_no_dir_safe(self):
        facts.cleanup_knowledge_index(str(self.snap / "nonexistent"))
        facts.cleanup_session_archives(str(self.snap / "nonexistent"))


if __name__ == "__main__":
    unittest.main()
