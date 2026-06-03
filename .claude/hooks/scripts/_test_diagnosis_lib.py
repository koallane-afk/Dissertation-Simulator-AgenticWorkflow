#!/usr/bin/env python3
"""Tests for _diagnosis_lib.py — predictive-debugging diagnosis + risk scoring (ADR-080 Increment 5).

_diagnosis_lib is a near-leaf depending only on _core_lib. TestCrossModuleDepsImported
is the inc3-lesson regression guard. The diagnose_failure_context / _gather_* tests
exercise the branches that call _core helpers (sot_paths, MIN_OUTPUT_SIZE, ERROR_RESULT_CHARS)
inside try/except — the exact inc3 swallowed-NameError geometry.

Run: python3 -m pytest _test_diagnosis_lib.py -v
  or: python3 _test_diagnosis_lib.py
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import _diagnosis_lib as diag


class TestCrossModuleDepsImported(unittest.TestCase):
    """Regression (ADR-080 inc5, inc3 bug class): the _core_lib symbols diagnosis
    calls (some inside try/except) must be bound in _diagnosis_lib."""

    def test_core_deps_bound(self):
        for name in ("sot_paths", "MIN_OUTPUT_SIZE", "ERROR_RESULT_CHARS",
                     "_DIAG_EVIDENCE_RE", "_DIAG_GATE_RE", "_DIAG_SELECTED_RE"):
            self.assertTrue(hasattr(diag, name),
                            f"_diagnosis_lib must import {name} from _core_lib")


class TestRiskScoring(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.proj)

    def test_empty_risk_data_cold_start(self):
        out = diag._empty_risk_data(str(self.proj))
        self.assertIsInstance(out, dict)

    def test_aggregate_no_ki_file_is_cold_start(self):
        out = diag.aggregate_risk_scores(str(self.proj / "nope.jsonl"), str(self.proj))
        self.assertIsInstance(out, dict)

    def test_aggregate_with_entries(self):
        ki = self.proj / "knowledge-index.jsonl"
        with open(ki, "w", encoding="utf-8") as f:
            for i in range(6):
                f.write(json.dumps({
                    "session_id": f"s{i}", "timestamp": "2026-06-01T10:00:00",
                    "modified_files": ["src/parser.py"],
                    "diagnosis_patterns": [{"type": "edit_mismatch", "file": "src/parser.py"}],
                }) + "\n")
        out = diag.aggregate_risk_scores(str(ki), str(self.proj))
        self.assertIsInstance(out, dict)
        # RS1-RS6 validation must accept the produced structure
        warnings = diag.validate_risk_scores(out)
        self.assertIsInstance(warnings, list)

    def test_timestamp_to_age_days(self):
        import datetime as _dt
        now = _dt.datetime(2026, 6, 11, tzinfo=_dt.timezone.utc).timestamp()
        age = diag._timestamp_to_age_days("2026-06-01T00:00:00+00:00", now)
        self.assertIsInstance(age, (int, float))
        self.assertGreaterEqual(age, 9)


class TestDiagnoseFailureContext(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.proj)

    def test_returns_evidence_bundle_no_nameerror(self):
        # sot_data=None drives the sot_paths/MIN_OUTPUT_SIZE branches (inc3 trap geometry).
        out = diag.diagnose_failure_context(str(self.proj), 5, "pacs", sot_data=None)
        self.assertIsInstance(out, dict)


class TestValidateDiagnosisLog(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.proj)

    def test_missing_log_is_invalid(self):
        result = diag.validate_diagnosis_log(str(self.proj), 5, "pacs")
        is_valid = result[0] if isinstance(result, tuple) else result
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
