#!/usr/bin/env python3
"""
Context Preservation System — Shared Library (_context_lib.py)

All hook scripts share this module for:
- Transcript JSONL parsing with deterministic extraction rules
- Structured MD snapshot generation (facts only, no heuristic inference)
- SOT state capture
- Atomic file writes with locking
- Token estimation (multi-signal)
- Dedup guard

Architecture:
  RLM Pattern: Snapshots are external memory objects (files on disk).
  P1 Compliance: Code handles deterministic extraction only.
                  Semantic interpretation is Claude's responsibility.
  SOT Compliance: Read-only access to SOT; writes only to context-snapshots/.
  Quality First: 100% accurate structured data, zero heuristic inference.
"""

import json
import os
import re
import sys
import time
import fcntl
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure sibling modules (_core_lib, _validation_lib) resolve even when this
# module is loaded by file path without the scripts dir on sys.path (ADR-076).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- ADR-076 Increment 1: re-export shim ---------------------------------
# Foundation primitives and quality-gate validators were extracted to
# _core_lib.py and _validation_lib.py. These star-imports preserve
# _context_lib's public interface so the ~40 importers stay unchanged.
from _core_lib import *  # noqa: F401,F403
from _validation_lib import *  # noqa: F401,F403
from _snapshot_lib import *  # noqa: F401,F403
from _facts_lib import *  # noqa: F401,F403
from _diagnosis_lib import *  # noqa: F401,F403
from _facts_lib import (  # noqa: F401 — underscore not covered by import *
    _extract_hypothesis_graveyard,
    _extract_thesis_continuity,
)
from _snapshot_lib import (  # noqa: F401 — underscore not covered by import *
    _extract_decisions,
    _remove_section,
)
from _capture_lib import *  # noqa: F401,F403
# Underscore-prefixed symbols imported by external scripts are not covered
# by `import *`, so re-export them explicitly:
from _validation_lib import (  # noqa: F401
    _DKS_REF_RE,
    _find_translation_files_for_step,
    _TRACE_MARKER_RE,
)
from _core_lib import (  # noqa: F401 — underscore symbols not covered by import *
    _truncate,
    _DIAG_GATE_RE,
    _DIAG_SELECTED_RE,
    _DIAG_EVIDENCE_RE,
)


def append_with_lock(filepath, content):
    """Append content with file locking (fcntl.flock)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(content)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_work_log(snapshot_dir):
    """Load work log entries from JSONL."""
    log_path = os.path.join(snapshot_dir, "work_log.jsonl")
    entries = []
    if not os.path.exists(log_path):
        return entries

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    return entries


# =============================================================================
# Dedup Guard
# =============================================================================

def should_skip_save(snapshot_dir, trigger=None):
    """Check if a save was done within dedup window.

    SessionEnd is exempt: /clear is an explicit user action,
    so the save must always happen regardless of dedup window.
    Stop hook uses wider window (30s) to reduce noise.
    """
    if trigger in ("sessionend",):
        return False
    latest_path = os.path.join(snapshot_dir, "latest.md")
    if os.path.exists(latest_path):
        age = time.time() - os.path.getmtime(latest_path)
        # Stop hook uses wider window (30s) to reduce noise
        window = STOP_DEDUP_WINDOW_SECONDS if trigger == "stop" else DEDUP_WINDOW_SECONDS
        if age < window:
            return True
    return False


def read_stdin_json():
    """Read and parse JSON from stdin (hook input)."""
    try:
        raw = sys.stdin.read()
        if raw.strip():
            return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        pass
    return {}


# ---------------------------------------------------------------------------
# Thesis state summary — shared by save_context.py & generate_context_summary.py
# ---------------------------------------------------------------------------

def get_thesis_state_summary(project_dir):
    """Read thesis SOT(s) and return a brief state summary for snapshots.

    P1 compliant: deterministic file reads only, no AI judgment.
    Non-blocking: returns empty string on any error.
    Read-only: reads session.json — never modifies it.

    Returns markdown string with step, status, gates, and HITL info.
    """
    try:
        thesis_root = os.path.join(project_dir, "thesis-output")
        if not os.path.isdir(thesis_root):
            return ""

        summaries = []
        for proj_name in sorted(os.listdir(thesis_root)):
            sot_path = os.path.join(thesis_root, proj_name, "session.json")
            if not os.path.isfile(sot_path):
                continue
            with open(sot_path, "r", encoding="utf-8") as f:
                sot = json.load(f)

            step = sot.get("current_step", 0)
            total = sot.get("total_steps", "?")
            status = sot.get("status", "unknown")
            rtype = sot.get("research_type", "undecided")

            # Gate summary
            gates = sot.get("gates", {})
            gate_str = ", ".join(
                f"{k}:{v.get('status', v) if isinstance(v, dict) else v}"
                for k, v in gates.items()
            ) if gates else "none"

            # HITL summary
            hitl = sot.get("hitl_checkpoints", {})
            completed_hitl = [
                k for k, v in hitl.items()
                if (v.get("status") if isinstance(v, dict) else v) == "completed"
            ]
            hitl_str = ", ".join(completed_hitl) if completed_hitl else "none"

            summaries.append(
                f"  - **{proj_name}**: step {step}/{total}, "
                f"status={status}, type={rtype}\n"
                f"    - Gates: {gate_str}\n"
                f"    - HITL completed: {hitl_str}"
            )

        if not summaries:
            return ""

        return (
            "\n\n## Thesis Workflow State\n\n"
            + "\n".join(summaries)
            + "\n"
        )
    except Exception:
        return ""
