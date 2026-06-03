#!/usr/bin/env python3
"""Tests for _capture_lib.py — session capture primitives (ADR-077 Increment 2).

Closes the coverage gap flagged by the inc2 adversarial verification: the 16
capture functions had zero direct unit tests. Exercises transcript parsing,
SOT/git/completion-state capture, ULW detection/compliance, and phase
classification through realistic fixtures.

Run: python3 -m pytest _test_capture_lib.py -v
  or: python3 _test_capture_lib.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import _capture_lib as cap


def _write_jsonl(path, objs):
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")


class TestParseTranscript(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_missing_or_empty_path_returns_empty(self):
        self.assertEqual(cap.parse_transcript(str(self.tmp / "nope.jsonl")), [])
        self.assertEqual(cap.parse_transcript(""), [])

    def test_parses_each_entry_type(self):
        p = self.tmp / "t.jsonl"
        _write_jsonl(p, [
            {"type": "user", "timestamp": "t1",
             "message": {"content": "please build the feature"}},
            {"type": "assistant", "timestamp": "t2", "message": {"content": [
                {"type": "text", "text": "Working on it"},
                {"type": "tool_use", "id": "tu1", "name": "Write",
                 "input": {"file_path": "/tmp/x.py", "content": "a\nb\nc"}},
            ]}},
            {"type": "user", "timestamp": "t3", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "ok done"},
            ]}},
            {"type": "progress", "timestamp": "t4"},  # must be skipped
        ])
        entries = cap.parse_transcript(str(p))
        types = [e["type"] for e in entries]
        self.assertEqual(types, ["user_message", "assistant_text", "tool_use", "tool_result"])
        tu = next(e for e in entries if e["type"] == "tool_use")
        self.assertEqual(tu["tool_name"], "Write")
        self.assertEqual(tu["file_path"], "/tmp/x.py")
        self.assertEqual(tu["line_count"], 3)
        self.assertEqual(tu["tool_use_id"], "tu1")

    def test_malformed_line_skipped(self):
        p = self.tmp / "bad.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write("{not valid json}\n")
            f.write(json.dumps({"type": "user", "timestamp": "t1",
                                "message": {"content": "hello"}}) + "\n")
        entries = cap.parse_transcript(str(p))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "user_message")

    def test_local_command_user_message_filtered(self):
        p = self.tmp / "lc.jsonl"
        _write_jsonl(p, [
            {"type": "user", "timestamp": "t1",
             "message": {"content": "<local-command-stdout>x</local-command-stdout>"}},
        ])
        self.assertEqual(cap.parse_transcript(str(p)), [])


class TestToolSummaries(unittest.TestCase):
    def test_tool_use_summary_per_tool(self):
        self.assertIn("Write → /a/b.py", cap._extract_tool_use_summary(
            "Write", {"file_path": "/a/b.py", "content": "x\ny"}))
        self.assertIn("Edit → /a/b.py", cap._extract_tool_use_summary(
            "Edit", {"file_path": "/a/b.py", "old_string": "o", "new_string": "n"}))
        self.assertIn("Bash:", cap._extract_tool_use_summary(
            "Bash", {"command": "ls -la"}))
        self.assertEqual(cap._extract_tool_use_summary(
            "Read", {"file_path": "/r.py"}), "Read → /r.py")
        # Unknown tool → generic JSON dump
        self.assertIn("MysteryTool", cap._extract_tool_use_summary(
            "MysteryTool", {"k": "v"}))

    def test_tool_result_summary_error_gets_larger_limit(self):
        # Error content preserved up to ERROR_RESULT_CHARS (> NORMAL_RESULT_CHARS)
        big_err = "Traceback (most recent call last): " + ("e" * 2500)
        big_ok = "all good " + ("o" * 2500)
        err_summary = cap._extract_tool_result_summary(big_err)
        ok_summary = cap._extract_tool_result_summary(big_ok)
        self.assertGreater(len(err_summary), len(ok_summary))

    def test_tool_result_summary_list_content(self):
        out = cap._extract_tool_result_summary(
            [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}])
        self.assertIn("line1", out)
        self.assertIn("line2", out)
        self.assertEqual(cap._extract_tool_result_summary(123), "")


class TestSotReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / ".claude").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write_sot(self, text):
        (self.tmp / ".claude" / "state.yaml").write_text(text, encoding="utf-8")

    def test_capture_sot_present_and_absent(self):
        self.assertIsNone(cap.capture_sot(str(self.tmp)))
        self._write_sot("workflow:\n  name: wf\n")
        result = cap.capture_sot(str(self.tmp))
        self.assertIsNotNone(result)
        self.assertIn("workflow", result["content"])
        self.assertTrue(result["path"].endswith("state.yaml"))

    def test_read_autopilot_state_enabled(self):
        self._write_sot(
            "workflow:\n"
            "  name: test-wf\n"
            "  status: running\n"
            "  current_step: 3\n"
            "  autopilot:\n"
            "    enabled: true\n"
            "    activated_at: '2026-06-03'\n"
            "    auto_approved_steps: [1, 2]\n"
            "  outputs:\n"
            "    step-1: out1.md\n"
        )
        st = cap.read_autopilot_state(str(self.tmp))
        self.assertIsNotNone(st)
        self.assertTrue(st["enabled"])
        self.assertEqual(st["current_step"], 3)
        self.assertEqual(st["outputs"].get("step-1"), "out1.md")

    def test_read_autopilot_state_disabled_or_missing(self):
        self.assertIsNone(cap.read_autopilot_state(str(self.tmp)))
        self._write_sot("workflow:\n  autopilot:\n    enabled: false\n")
        self.assertIsNone(cap.read_autopilot_state(str(self.tmp)))

    def test_read_active_team_state(self):
        self.assertIsNone(cap.read_active_team_state(str(self.tmp)))
        self._write_sot(
            "workflow:\n"
            "  active_team:\n"
            "    name: team-1\n"
            "    status: partial\n"
            "    tasks_completed: [task-1]\n"
            "    tasks_pending: [task-2]\n"
        )
        st = cap.read_active_team_state(str(self.tmp))
        self.assertIsNotNone(st)
        self.assertEqual(st["name"], "team-1")
        self.assertEqual(st["status"], "partial")
        self.assertIn("task-1", st["tasks_completed"])


class TestUlw(unittest.TestCase):
    def test_detect_ulw_mode_word_boundary(self):
        active = cap.detect_ulw_mode([
            {"type": "user_message", "content": "do this in ulw mode", "timestamp": "t"}])
        self.assertIsNotNone(active)
        self.assertTrue(active["active"])
        self.assertEqual(active["message_index"], 0)
        # No false positive on substrings
        self.assertIsNone(cap.detect_ulw_mode([
            {"type": "user_message", "content": "use ulwrapper lib", "timestamp": "t"}]))
        self.assertIsNone(cap.detect_ulw_mode([
            {"type": "user_message", "content": "normal request", "timestamp": "t"}]))

    def test_check_ulw_compliance_flags_no_decomposition(self):
        entries = [{"type": "user_message", "content": "go ulw", "timestamp": "t0"}]
        for i in range(5):
            entries.append({"type": "tool_use", "tool_name": "Read",
                            "content": "r", "timestamp": f"t{i+1}"})
        comp = cap.check_ulw_compliance(entries)
        self.assertIsNotNone(comp)
        self.assertTrue(comp["active"])
        self.assertEqual(comp["total_tool_uses"], 5)
        self.assertEqual(comp["task_creates"], 0)
        self.assertTrue(any("ULW_NO_DECOMPOSITION" in w for w in comp["warnings"]))

    def test_check_ulw_compliance_none_when_inactive(self):
        self.assertIsNone(cap.check_ulw_compliance([
            {"type": "user_message", "content": "no keyword here", "timestamp": "t"}]))

    def test_extract_file_from_nearby_tool_use(self):
        entries = [
            {"type": "tool_use", "tool_name": "Edit", "file_path": "/x.py"},
            {"type": "assistant_text", "content": "..."},
            {"type": "tool_result", "is_error": True, "content": "error"},
        ]
        self.assertEqual(cap._extract_file_from_nearby_tool_use(entries, 2), "/x.py")
        self.assertIsNone(cap._extract_file_from_nearby_tool_use(
            [{"type": "tool_result", "content": "e"}], 0))


class TestCaptureGitState(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _git(self, *args):
        subprocess.run(["git", *args], cwd=self.tmp, capture_output=True, text=True, check=True)

    def test_git_state_keys_and_status(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        self._git("init")
        self._git("config", "user.email", "t@t.com")
        self._git("config", "user.name", "t")
        (self.tmp / "f.txt").write_text("v1\n", encoding="utf-8")
        self._git("add", "f.txt")
        self._git("commit", "-m", "init")
        (self.tmp / "f.txt").write_text("v2\n", encoding="utf-8")  # uncommitted change
        result = cap.capture_git_state(str(self.tmp))
        self.assertEqual(set(result.keys()),
                         {"status", "diff_stat", "diff_content", "recent_commits"})
        self.assertIn("f.txt", result["status"])
        self.assertIn("init", result["recent_commits"])

    def test_git_state_non_repo_returns_empty_strings(self):
        result = cap.capture_git_state(str(self.tmp))  # not a git repo
        self.assertEqual(result["status"], "")
        self.assertEqual(set(result.keys()),
                         {"status", "diff_stat", "diff_content", "recent_commits"})


class TestExtractCompletionState(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_success_fail_counts_via_tool_use_id(self):
        existing = self.tmp / "made.py"
        existing.write_text("x\n", encoding="utf-8")
        entries = [
            {"type": "tool_use", "tool_name": "Write", "tool_use_id": "w1",
             "file_path": str(existing), "timestamp": "t1"},
            {"type": "tool_result", "tool_use_id": "w1", "content": "ok", "is_error": False},
            {"type": "tool_use", "tool_name": "Bash", "tool_use_id": "b1", "timestamp": "t2"},
            {"type": "tool_result", "tool_use_id": "b1", "content": "command failed", "is_error": True},
        ]
        st = cap.extract_completion_state(entries, str(self.tmp))
        self.assertEqual(st["write_success"], 1)
        self.assertEqual(st["bash_fail"], 1)
        self.assertEqual(st["total_tool_calls"], 2)
        self.assertEqual(st["tool_counts"].get("Write"), 1)
        verified = {v["path"]: v["exists"] for v in st["file_verification"]}
        self.assertTrue(verified.get(str(existing)))


class TestPhaseClassification(unittest.TestCase):
    def _tu(self, name):
        return {"type": "tool_use", "tool_name": name}

    def test_detect_conversation_phase(self):
        self.assertEqual(cap.detect_conversation_phase([]), "unknown")
        research = [self._tu("Read"), self._tu("Grep"), self._tu("Glob"), self._tu("Read")]
        self.assertEqual(cap.detect_conversation_phase(research), "research")
        impl = [self._tu("Edit"), self._tu("Write"), self._tu("Bash"), self._tu("Edit")]
        self.assertEqual(cap.detect_conversation_phase(impl), "implementation")
        planning = [self._tu("AskUserQuestion"), self._tu("ExitPlanMode")]
        self.assertEqual(cap.detect_conversation_phase(planning), "planning")

    def test_detect_phase_transitions_small_list_single_phase(self):
        out = cap.detect_phase_transitions([self._tu("Read"), self._tu("Grep")])
        self.assertEqual(len(out), 1)
        phase, start, end = out[0]
        self.assertEqual(phase, "research")
        self.assertEqual((start, end), (0, 2))


if __name__ == "__main__":
    unittest.main()
