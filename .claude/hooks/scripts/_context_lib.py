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

# --- Predictive Debugging: Risk Score Constants (P1 — module-level) ---
# Used by aggregate_risk_scores() and validate_risk_scores()
# Weights per error type: higher = more indicative of fragile code
# D-7: Keys MUST match ERROR_TAXONOMY in _classify_error_patterns() (in _facts_lib.py, ADR-079)
#      + "unknown" for unclassified errors. Mismatch → fallback weight 0.7 applied.
_RISK_WEIGHTS = {
    "edit_mismatch": 2.0,   # File structure instability (frequent edit failures)
    "dependency": 2.5,       # High ripple effect
    "type_error": 1.5,       # Type complexity
    "syntax": 1.0,           # Repetitive — complex file indicator
    "value_error": 1.0,
    "git_error": 1.0,
    "timeout": 0.5,          # Often environmental, not code
    "file_not_found": 0.5,   # Usually one-time
    "permission": 0.5,
    "connection": 0.3,       # Network — may not be code issue
    "memory": 0.3,
    "command_not_found": 0.3,
    "unknown": 0.7,          # ~30% of errors — ignoring loses significant data
}
# Recency decay: (max_days, weight_multiplier)
# More recent errors are more relevant to current code state
_RECENCY_DECAY_DAYS = [
    (30, 1.0),              # 0-30 days: full weight
    (90, 0.5),              # 31-90 days: half weight
    (float("inf"), 0.25),   # 91+ days: quarter weight
]
# Minimum risk score to trigger PreToolUse warning
_RISK_SCORE_THRESHOLD = 3.0
# Minimum sessions in knowledge-index before activation (cold start guard)
_RISK_MIN_SESSIONS = 5

# Used by Abductive Diagnosis functions — diagnosis-logs/ parsing
# Captures the FULL heading line (including H-ID) so AD9 can extract H[1-4] from it.
# E.g., "## H1: Upstream data quality issue" → captures "H1: Upstream data quality issue"
_DIAG_HYPOTHESIS_RE = re.compile(
    r"^#+\s*((?:H\d|Hypothesis)\b.+)", re.MULTILINE | re.IGNORECASE,
)
_DIAG_SOURCE_STEP_RE = re.compile(
    r"\(source:\s*Step\s+(\d+)\)", re.IGNORECASE,
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


# =============================================================================
# Predictive Debugging: Risk Score Aggregation (P1 — Deterministic)
# =============================================================================

def aggregate_risk_scores(ki_path, project_dir):
    """Aggregate per-file risk scores from knowledge-index.jsonl.

    P1 Compliance: All operations are deterministic arithmetic.
    No semantic inference — pure counting, weighting, and decay.

    Called by: restore_context.py at SessionStart (once per session).
    Output: dict suitable for JSON serialization to risk-scores.json.

    Data flow:
      knowledge-index.jsonl → read entries → extract error_patterns
      → per-file error counting → weight application → recency decay
      → resolution rate calculation → validate → return
    """
    # Read all knowledge-index entries
    entries = []
    if not ki_path or not os.path.exists(ki_path):
        return _empty_risk_data(project_dir)

    try:
        with open(ki_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return _empty_risk_data(project_dir)

    if len(entries) < _RISK_MIN_SESSIONS:
        return _empty_risk_data(project_dir, data_sessions=len(entries))

    # Per-file error accumulation
    # Key: relative path → {error_types: {type: count}, total_weighted: float,
    #                        resolved_count: int, total_count: int, last_error_date: str}
    file_risks = {}
    now_ts = time.time()

    for entry in entries:
        error_patterns = entry.get("error_patterns", [])
        if not isinstance(error_patterns, list):
            continue

        # Parse entry timestamp for recency decay
        entry_ts = entry.get("timestamp", "")
        entry_age_days = _timestamp_to_age_days(entry_ts, now_ts)

        # Determine recency weight
        recency_weight = _RECENCY_DECAY_DAYS[-1][1]  # default: oldest bracket
        for max_days, weight in _RECENCY_DECAY_DAYS:
            if entry_age_days <= max_days:
                recency_weight = weight
                break

        # Get modified files from this session (for file↔error association)
        modified_files = entry.get("modified_files", [])

        for ep in error_patterns:
            if not isinstance(ep, dict):
                continue
            error_type = ep.get("type", "unknown")
            error_file = ep.get("file", "")
            resolution = ep.get("resolution")
            has_resolution = isinstance(resolution, dict) and bool(resolution)

            # Determine which file(s) to attribute the error to
            # Priority: error-specific file > session modified files
            target_files = []
            if error_file:
                target_files = [error_file]
            else:
                # No specific file — attribute to all modified files
                target_files = [os.path.basename(f) for f in modified_files[:5]]

            for tf in target_files:
                # Normalize to relative path
                rel_path = _normalize_to_relative(tf, project_dir, modified_files)
                if not rel_path:
                    continue

                if rel_path not in file_risks:
                    file_risks[rel_path] = {
                        "error_types": {},
                        "total_weighted": 0.0,
                        "resolved_count": 0,
                        "total_count": 0,
                        "last_error_date": "",
                    }

                fr = file_risks[rel_path]
                # Apply type weight × recency weight
                type_weight = _RISK_WEIGHTS.get(error_type, 0.7)
                weighted_score = type_weight * recency_weight
                fr["total_weighted"] += weighted_score
                fr["error_types"][error_type] = fr["error_types"].get(error_type, 0) + 1
                fr["total_count"] += 1
                if has_resolution:
                    fr["resolved_count"] += 1

                # Track most recent error date
                entry_date = entry_ts[:10] if len(entry_ts) >= 10 else ""
                if entry_date > fr["last_error_date"]:
                    fr["last_error_date"] = entry_date

    # P1-FIX: Merge entries with same basename but different paths
    # (bare names like "_context_lib.py" vs relative ".claude/hooks/scripts/_context_lib.py")
    # Keep the longest (most specific) path as canonical key, sum scores.
    basename_groups = {}
    for rel_path, fr in file_risks.items():
        bname = os.path.basename(rel_path)
        if bname not in basename_groups:
            basename_groups[bname] = []
        basename_groups[bname].append((rel_path, fr))

    merged_risks = {}
    for bname, group in basename_groups.items():
        if len(group) == 1:
            merged_risks[group[0][0]] = group[0][1]
        else:
            # Pick longest path as canonical (most specific)
            canonical_path = max(group, key=lambda x: len(x[0]))[0]
            merged = {
                "error_types": {},
                "total_weighted": 0.0,
                "resolved_count": 0,
                "total_count": 0,
                "last_error_date": "",
            }
            for _, fr in group:
                merged["total_weighted"] += fr["total_weighted"]
                merged["total_count"] += fr["total_count"]
                merged["resolved_count"] += fr["resolved_count"]
                for etype, cnt in fr["error_types"].items():
                    merged["error_types"][etype] = (
                        merged["error_types"].get(etype, 0) + cnt
                    )
                if fr["last_error_date"] > merged["last_error_date"]:
                    merged["last_error_date"] = fr["last_error_date"]
            merged_risks[canonical_path] = merged

    # Build output
    files_output = {}
    for rel_path, fr in merged_risks.items():
        resolution_rate = (
            fr["resolved_count"] / fr["total_count"]
            if fr["total_count"] > 0
            else 0.0
        )
        files_output[rel_path] = {
            "risk_score": round(fr["total_weighted"], 2),
            "error_count": fr["total_count"],
            "error_types": fr["error_types"],
            "last_error_session": fr["last_error_date"],
            "resolution_rate": round(resolution_rate, 2),
        }

    # Sort by risk_score descending for top_risk_files
    sorted_files = sorted(
        files_output.keys(),
        key=lambda k: files_output[k]["risk_score"],
        reverse=True,
    )
    top_risk = [
        f for f in sorted_files[:10]
        if files_output[f]["risk_score"] >= _RISK_SCORE_THRESHOLD
    ]

    risk_data = {
        "generated_at": datetime.now().isoformat(),
        "data_sessions": len(entries),
        "project_dir": project_dir,
        "risk_threshold": _RISK_SCORE_THRESHOLD,
        "files": files_output,
        "top_risk_files": top_risk,
    }

    # P1: Self-validation before return
    validation_warnings = validate_risk_scores(risk_data)
    if validation_warnings:
        risk_data["_validation_warnings"] = validation_warnings

    return risk_data


def validate_risk_scores(risk_data):
    """P1 Risk Score Validation (RS1-RS6).

    Deterministic schema enforcement for risk-scores.json.
    Follows the same pattern as validate_sot_schema (S1-S8),
    validate_review_output (R1-R5), validate_translation_output (T1-T7).

    Returns: list of warning strings (empty = all checks pass).
    """
    warnings = []

    if not isinstance(risk_data, dict):
        warnings.append("RS1 FAIL: risk_data is not a dict")
        return warnings

    # RS1: Required top-level keys
    required_keys = {"generated_at", "data_sessions", "files", "top_risk_files", "risk_threshold"}
    missing = required_keys - set(risk_data.keys())
    if missing:
        warnings.append(f"RS1 FAIL: Missing required keys: {missing}")

    # RS2: data_sessions is int >= 0
    ds = risk_data.get("data_sessions")
    if not isinstance(ds, int) or ds < 0:
        warnings.append(f"RS2 FAIL: data_sessions must be int >= 0, got {ds!r}")

    # RS3-RS5: Per-file validation
    files = risk_data.get("files", {})
    if not isinstance(files, dict):
        warnings.append("RS3 FAIL: files must be a dict")
    else:
        for fpath, fdata in files.items():
            if not isinstance(fdata, dict):
                warnings.append(f"RS3 FAIL: files[{fpath!r}] is not a dict")
                continue

            # RS3: risk_score is numeric >= 0
            score = fdata.get("risk_score")
            if not isinstance(score, (int, float)) or score < 0:
                warnings.append(
                    f"RS3 FAIL: files[{fpath!r}].risk_score must be "
                    f"numeric >= 0, got {score!r}"
                )

            # RS4: error_count >= sum(error_types.values())
            ec = fdata.get("error_count", 0)
            et = fdata.get("error_types", {})
            if isinstance(et, dict) and isinstance(ec, int):
                type_sum = sum(
                    v for v in et.values() if isinstance(v, (int, float))
                )
                if ec < type_sum:
                    warnings.append(
                        f"RS4 FAIL: files[{fpath!r}].error_count ({ec}) < "
                        f"sum(error_types) ({type_sum})"
                    )

            # RS5: resolution_rate is float, 0.0 <= rate <= 1.0
            rr = fdata.get("resolution_rate")
            if rr is not None:
                if not isinstance(rr, (int, float)) or rr < 0.0 or rr > 1.0:
                    warnings.append(
                        f"RS5 FAIL: files[{fpath!r}].resolution_rate must be "
                        f"0.0-1.0, got {rr!r}"
                    )

    # RS6: top_risk_files entries exist in files dict and sorted by risk_score desc
    top = risk_data.get("top_risk_files", [])
    if isinstance(top, list) and isinstance(files, dict):
        for tf in top:
            if tf not in files:
                warnings.append(
                    f"RS6 FAIL: top_risk_files entry {tf!r} not found in files"
                )
        # Check sort order
        scores = [
            files.get(tf, {}).get("risk_score", 0)
            for tf in top if tf in files
        ]
        if scores != sorted(scores, reverse=True):
            warnings.append(
                "RS6 FAIL: top_risk_files not sorted by risk_score desc"
            )

    return warnings


def _empty_risk_data(project_dir, data_sessions=0):
    """Return empty risk data structure (cold start / no data)."""
    return {
        "generated_at": datetime.now().isoformat(),
        "data_sessions": data_sessions,
        "project_dir": project_dir,
        "risk_threshold": _RISK_SCORE_THRESHOLD,
        "files": {},
        "top_risk_files": [],
    }


def _timestamp_to_age_days(ts_str, now_ts):
    """Convert ISO timestamp string to age in days.

    P1 Compliance: Deterministic datetime parsing.
    Returns float (days). Returns 365.0 on parse failure (conservative decay).
    """
    if not ts_str:
        return 365.0
    try:
        # Handle both "2026-02-20T15:30:00" and "2026-02-20T153000" formats
        dt = datetime.fromisoformat(
            ts_str.replace("Z", "+00:00") if ts_str.endswith("Z") else ts_str
        )
        age_seconds = now_ts - dt.timestamp()
        return max(0.0, age_seconds / 86400.0)
    except (ValueError, TypeError, OSError):
        return 365.0  # Conservative: treat unparseable as old


def _normalize_to_relative(filename, project_dir, modified_files):
    """Normalize a filename to project-relative path.

    Strategy:
      1. If filename is a bare name, find full path in modified_files
      2. If filename is absolute and under project_dir, make relative
      3. Otherwise return as-is (best effort)

    P1 Compliance: Deterministic string operations only.
    Returns: relative path string, or empty string on failure.
    """
    if not filename:
        return ""

    # Case 1: Bare filename (no path separator) — find in modified_files
    if not os.path.isabs(filename) and os.sep not in filename:
        for mf in modified_files:
            if os.path.basename(mf) == filename:
                if os.path.isabs(mf) and project_dir:
                    try:
                        return os.path.relpath(mf, project_dir)
                    except ValueError:
                        return mf
                return mf
        return filename  # Return bare filename as-is (best effort)

    # Case 2: Absolute path — make relative to project
    if os.path.isabs(filename) and project_dir:
        try:
            return os.path.relpath(filename, project_dir)
        except ValueError:
            return filename

    return filename


# =============================================================================
# Abductive Diagnosis Layer (P1: Deterministic Pre/Post Analysis)
# =============================================================================
# Inserts a 3-step diagnosis (P1 pre-evidence → LLM judgment → P1 post-validation)
# between quality gate FAIL and retry. Existing 4-layer QA is NOT modified.
# SOT Compliance: Read-only access to SOT, verification-logs/, pacs-logs/,
#                 review-logs/, diagnosis-logs/.
# P1 Compliance: All evidence gathering and validation is deterministic.


def diagnose_failure_context(project_dir, step, gate, sot_data=None):
    """Pre-analysis: Gather deterministic evidence bundle for a failed quality gate.

    Called by Orchestrator AFTER a gate FAIL and BEFORE retry.
    Returns a dict with structured evidence for LLM-based hypothesis selection.

    Args:
        project_dir: Project root path.
        step: Step number that failed.
        gate: One of 'verification', 'pacs', 'review'.
        sot_data: Optional pre-loaded SOT dict (avoids re-reading).

    Returns:
        dict with keys: step, gate, retry_history, upstream_evidence,
                        hypothesis_priority, fast_path, raw_evidence.
    """
    retry_history = _gather_retry_history(project_dir, step, gate)
    upstream_evidence = _gather_upstream_evidence(project_dir, step, sot_data)
    hypothesis_priority = _determine_hypothesis_priority(
        retry_history, upstream_evidence, gate
    )
    fast_path = _check_fast_path_eligibility(
        project_dir, step, gate, retry_history, sot_data=sot_data
    )
    raw_evidence = _gather_raw_evidence(project_dir, step, gate)

    return {
        "step": step,
        "gate": gate,
        "retry_history": retry_history,
        "upstream_evidence": upstream_evidence,
        "hypothesis_priority": hypothesis_priority,
        "fast_path": fast_path,
        "raw_evidence": raw_evidence,
    }


def _gather_retry_history(project_dir, step, gate):
    """Read retry counter and previous diagnosis logs for this step+gate.

    Returns:
        dict with keys: retries_used (int), max_retries (int),
                        previous_diagnoses (list of dicts).
    """
    # D-7: Retry limit constants must match validate_retry_budget.py
    # DEFAULT_MAX_RETRIES and ULW_MAX_RETRIES. Change both files together.
    _DEFAULT_MAX_RETRIES = 10
    _ULW_MAX_RETRIES = 15

    result = {
        "retries_used": 0,
        "max_retries": _DEFAULT_MAX_RETRIES,
        "previous_diagnoses": [],
    }

    # Read retry counter
    counter_dir = os.path.join(project_dir, f"{gate}-logs")
    counter_file = os.path.join(counter_dir, f".step-{step}-retry-count")
    if os.path.exists(counter_file):
        try:
            with open(counter_file, "r", encoding="utf-8") as f:
                result["retries_used"] = int(f.read().strip() or "0")
        except (ValueError, OSError):
            pass

    # Detect ULW mode for max_retries adjustment (10 → 15)
    # D-7: ULW detection pattern must match validate_retry_budget.py _ULW_SNAPSHOT_RE
    # and restore_context.py — all use "ULW 상태" section header presence.
    snapshot_path = os.path.join(
        project_dir, ".claude", "context-snapshots", "latest.md"
    )
    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                content = f.read(8000)  # First 8KB only
            if re.search(r"ULW 상태|Ultrawork Mode State", content):
                result["max_retries"] = _ULW_MAX_RETRIES
        except OSError:
            pass

    # Gather previous diagnosis logs
    diag_dir = os.path.join(project_dir, "diagnosis-logs")
    if os.path.isdir(diag_dir):
        try:
            for fname in sorted(os.listdir(diag_dir)):
                if fname.startswith(f"step-{step}-{gate}-") and fname.endswith(".md"):
                    fpath = os.path.join(diag_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read()
                        selected = _DIAG_SELECTED_RE.search(content)
                        result["previous_diagnoses"].append({
                            "file": fname,
                            "selected_hypothesis": selected.group(1).strip() if selected else "unknown",
                        })
                    except OSError:
                        pass
        except OSError:
            pass

    return result


def _gather_upstream_evidence(project_dir, step, sot_data=None):
    """Collect evidence from upstream step outputs referenced by SOT.

    Returns:
        dict with keys: upstream_outputs (list of {step, path, exists, size}),
                        sot_current_step (int), sot_status (str).
    """
    result = {
        "upstream_outputs": [],
        "sot_current_step": step,
        "sot_status": "unknown",
    }

    # Load SOT if not provided
    if sot_data is None:
        sot_data = {}
        try:
            import yaml
            for sp in sot_paths(project_dir):
                if os.path.exists(sp):
                    with open(sp, "r", encoding="utf-8") as f:
                        sot_data = yaml.safe_load(f) or {}
                    break
        except Exception:
            pass

    result["sot_current_step"] = sot_data.get("current_step", step)
    result["sot_status"] = sot_data.get("workflow_status", "unknown")

    # Gather upstream outputs (steps 1..step-1)
    # Guard: YAML `outputs: null` returns None, not {}
    outputs = sot_data.get("outputs") or {}
    for prev_step in range(1, step):
        key = f"step-{prev_step}"
        path_raw = outputs.get(key, "")
        if not path_raw:
            continue
        full_path = os.path.join(project_dir, path_raw)
        result["upstream_outputs"].append({
            "step": prev_step,
            "path": path_raw,
            "exists": os.path.exists(full_path),
            "size": os.path.getsize(full_path) if os.path.exists(full_path) else 0,
        })

    return result


def _determine_hypothesis_priority(retry_history, upstream_evidence, gate):
    """Rule-based hypothesis prioritization based on available evidence.

    Four hypothesis categories (H1, H2, H3, H4):
        H1: Upstream data quality (missing/thin upstream outputs)
        H2: Current step execution gap (most common)
        H3: Criteria interpretation error (rare)
        H4: Capability gap — missing tool, script, or infrastructure

    Returns:
        list of dicts with keys: id (str), label (str), priority (int 1-3),
                                  reason (str).
    """
    hypotheses = []

    # H1: Upstream data quality — check if any upstream output is missing/thin
    thin_upstreams = []
    for uo in upstream_evidence.get("upstream_outputs", []):
        if not uo.get("exists"):
            thin_upstreams.append(f"step-{uo['step']} missing")
        elif uo.get("size", 0) < MIN_OUTPUT_SIZE:
            thin_upstreams.append(f"step-{uo['step']} thin ({uo['size']}B)")

    h1_priority = 1 if thin_upstreams else 3
    hypotheses.append({
        "id": "H1",
        "label": "Upstream data quality issue",
        "priority": h1_priority,
        "reason": "; ".join(thin_upstreams) if thin_upstreams else "All upstream outputs present and adequate",
    })

    # H2: Current step execution gap — most common, default high priority
    prev_diag = retry_history.get("previous_diagnoses", [])
    h2_priority = 2 if prev_diag else 1
    # If previous diagnosis already selected H2, lower priority (try different hypothesis)
    if prev_diag and any(
        d.get("selected_hypothesis", "").startswith("H2") or
        "execution" in d.get("selected_hypothesis", "").lower()
        for d in prev_diag
    ):
        h2_priority = 2

    hypotheses.append({
        "id": "H2",
        "label": "Current step execution gap",
        "priority": h2_priority,
        "reason": f"{len(prev_diag)} previous diagnosis(es)" if prev_diag else "First attempt",
    })

    # H3: Criteria interpretation error — higher priority for review gate
    # Also elevated for verification gate when V1d failures suggest hallucinated evidence
    h3_priority = 3
    if gate == "review":
        h3_priority = 2
    elif gate == "verification":
        # Check if raw evidence contains V1d failures (hallucinated evidence pattern)
        v1d_hint = any(
            "V1d" in d.get("selected_hypothesis", "") or
            "evidence" in d.get("selected_hypothesis", "").lower()
            for d in prev_diag
        )
        if v1d_hint:
            h3_priority = 1  # Hallucination pattern → criteria re-examination critical
    hypotheses.append({
        "id": "H3",
        "label": "Criteria interpretation error",
        "priority": h3_priority,
        "reason": (
            "Review gate benefits from criteria re-examination" if gate == "review"
            else "Verification gate with evidence quality issues" if h3_priority == 1
            else "Low prior probability"
        ),
    })

    # H4: Capability gap — missing tool, script, or infrastructure
    # Elevated priority when: (a) repeated retries with same H2, (b) error
    # patterns suggest missing commands/tools. OpenAI harness pattern:
    # "build the missing capability rather than retrying manually."
    h4_priority = 3  # default: low
    h2_repeats = sum(
        1 for d in prev_diag
        if d.get("selected_hypothesis", "").startswith("H2")
    )
    if h2_repeats >= 2:
        # Two H2 attempts failed → likely not an execution gap but a missing capability
        h4_priority = 1
    hypotheses.append({
        "id": "H4",
        "label": "Capability gap — missing tool, script, or infrastructure",
        "priority": h4_priority,
        "reason": (
            f"H2 selected {h2_repeats} times without resolution — "
            "consider building missing capability"
            if h2_repeats >= 2
            else "Low prior probability — check after H2 exhausted"
        ),
    })

    # Sort by priority (1 = highest)
    hypotheses.sort(key=lambda h: h["priority"])
    return hypotheses


def _check_fast_path_eligibility(project_dir, step, gate, retry_history,
                                 sot_data=None):
    """Deterministic fast-path checks (FP1-FP3) that skip LLM diagnosis.

    FP1: Missing output file — diagnosis is trivially 'file not generated'.
    FP2: Empty/near-empty output — diagnosis is 'incomplete generation'.
    FP3: Identical retry — same hypothesis selected twice without change.

    Args:
        sot_data: Optional pre-loaded SOT dict (avoids redundant I/O).

    Returns:
        dict with keys: eligible (bool), reason (str), fp_id (str or None).
    """
    result = {"eligible": False, "reason": "", "fp_id": None}

    # FP1: Missing output file for current step
    try:
        if sot_data is None:
            import yaml
            sot_data = {}
            for sp in sot_paths(project_dir):
                if os.path.exists(sp):
                    with open(sp, "r", encoding="utf-8") as f:
                        sot_data = yaml.safe_load(f) or {}
                    break
        # Guard: YAML `outputs: null` returns None, not {}
        outputs = sot_data.get("outputs") or {}
        step_key = f"step-{step}"
        output_path_raw = outputs.get(step_key, "")
        if output_path_raw:
            full_path = os.path.join(project_dir, output_path_raw)
            if not os.path.exists(full_path):
                result["eligible"] = True
                result["reason"] = f"FP1: Output file missing — {output_path_raw}"
                result["fp_id"] = "FP1"
                return result
            # FP2: Empty/near-empty output
            fsize = os.path.getsize(full_path)
            if fsize < MIN_OUTPUT_SIZE:
                result["eligible"] = True
                result["reason"] = f"FP2: Output too small ({fsize}B < {MIN_OUTPUT_SIZE}B)"
                result["fp_id"] = "FP2"
                return result
    except Exception:
        pass

    # FP3: Identical retry — same hypothesis selected in 2+ previous diagnoses
    prev_diag = retry_history.get("previous_diagnoses", [])
    if len(prev_diag) >= 2:
        selected = [d.get("selected_hypothesis", "") for d in prev_diag[-2:]]
        if selected[0] and selected[0] == selected[1]:
            result["eligible"] = True
            result["reason"] = f"FP3: Same hypothesis '{selected[0]}' selected twice — escalate"
            result["fp_id"] = "FP3"
            return result

    return result


def _gather_raw_evidence(project_dir, step, gate):
    """Bundle raw log content for the failing gate.

    Returns:
        dict with keys: gate_log_path (str), gate_log_excerpt (str),
                        pacs_log_excerpt (str or None).
    """
    result = {
        "gate_log_path": "",
        "gate_log_excerpt": "",
        "pacs_log_excerpt": None,
    }

    # Determine log path based on gate type
    if gate == "verification":
        log_path = os.path.join(
            project_dir, "verification-logs", f"step-{step}-verify.md"
        )
    elif gate == "pacs":
        log_path = os.path.join(
            project_dir, "pacs-logs", f"step-{step}-pacs.md"
        )
    elif gate == "review":
        log_path = os.path.join(
            project_dir, "review-logs", f"step-{step}-review.md"
        )
    else:
        return result

    result["gate_log_path"] = log_path
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                # Read only first ERROR_RESULT_CHARS to avoid OOM on large logs
                result["gate_log_excerpt"] = f.read(ERROR_RESULT_CHARS)
        except OSError:
            pass

    # Always include pacs log if available (even for non-pacs gates)
    if gate != "pacs":
        pacs_path = os.path.join(
            project_dir, "pacs-logs", f"step-{step}-pacs.md"
        )
        if os.path.exists(pacs_path):
            try:
                with open(pacs_path, "r", encoding="utf-8") as f:
                    result["pacs_log_excerpt"] = f.read(ERROR_RESULT_CHARS)
            except OSError:
                pass

    # VE cross-check: include hallucination evidence if available
    # (validate_criteria_evidence.py writes JSON to stdout, not to disk,
    #  but diagnosis can still benefit from V1d/V1e warnings in verification log)
    if gate == "verification":
        result["hallucination_check_hint"] = (
            "Run: python3 .claude/hooks/scripts/validate_criteria_evidence.py "
            f"--step {step} --project-dir {project_dir} --auto-detect"
        )

    return result


def validate_diagnosis_log(project_dir, step, gate):
    """P1 Post-validation: Verify diagnosis log structural integrity (AD1-AD10).

    Called after LLM writes the diagnosis log. All checks are deterministic.

    Args:
        project_dir: Project root path.
        step: Step number.
        gate: One of 'verification', 'pacs', 'review'.

    Returns:
        tuple(is_valid: bool, warnings: list[str])

    Checks:
        AD1: Diagnosis log file exists in diagnosis-logs/
        AD2: Minimum file size (≥ 100 bytes)
        AD3: Gate field matches expected gate
        AD4: Selected hypothesis present (H1/H2/H3/H4)
        AD5: Evidence section present (≥ 1 evidence item)
        AD6: Action plan section present
        AD7: No forward step references (source: Step N where N > step)
        AD8: Hypothesis count ≥ 2 (must consider alternatives)
        AD9: Selected hypothesis is one of the listed hypotheses
        AD10: Previous diagnosis referenced (if retry > 0)
    """
    warnings = []

    # AD1: File exists
    diag_dir = os.path.join(project_dir, "diagnosis-logs")
    # Find the latest diagnosis log for this step+gate
    diag_path = None
    if os.path.isdir(diag_dir):
        candidates = sorted([
            f for f in os.listdir(diag_dir)
            if f.startswith(f"step-{step}-{gate}-") and f.endswith(".md")
        ])
        if candidates:
            diag_path = os.path.join(diag_dir, candidates[-1])

    if not diag_path or not os.path.exists(diag_path):
        warnings.append(
            f"AD1 FAIL: No diagnosis log found for step-{step} gate={gate} "
            f"in diagnosis-logs/"
        )
        return False, warnings

    # Read content
    try:
        with open(diag_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        warnings.append(f"AD1 FAIL: Cannot read diagnosis log — {e}")
        return False, warnings

    # AD2: Minimum size
    if len(content) < 100:
        warnings.append(
            f"AD2 FAIL: Diagnosis log too small ({len(content)}B < 100B)"
        )

    # AD3: Gate field matches
    gate_match = _DIAG_GATE_RE.search(content)
    if gate_match:
        found_gate = gate_match.group(1).lower()
        if found_gate != gate.lower():
            warnings.append(
                f"AD3 FAIL: Gate mismatch — expected '{gate}', found '{found_gate}'"
            )
    else:
        warnings.append("AD3 FAIL: No Gate field found in diagnosis log")

    # AD4: Selected hypothesis present
    selected_match = _DIAG_SELECTED_RE.search(content)
    if not selected_match:
        warnings.append("AD4 FAIL: No selected hypothesis found")

    # AD5: Evidence items (≥ 1)
    evidence_items = _DIAG_EVIDENCE_RE.findall(content)
    if len(evidence_items) < 1:
        warnings.append(
            f"AD5 FAIL: Insufficient evidence items ({len(evidence_items)} < 1)"
        )

    # AD6: Action plan section
    action_plan_re = re.compile(
        r"^#+\s*(?:Action\s*Plan|Recommended\s*Action|Next\s*Steps?)\b",
        re.MULTILINE | re.IGNORECASE,
    )
    if not action_plan_re.search(content):
        warnings.append("AD6 FAIL: No Action Plan section found")

    # AD7: No forward step references
    source_refs = _DIAG_SOURCE_STEP_RE.findall(content)
    for ref_step_str in source_refs:
        ref_step = int(ref_step_str)
        if ref_step > step:
            warnings.append(
                f"AD7 FAIL: Forward reference to Step {ref_step} (current: {step})"
            )

    # AD8: Hypothesis count ≥ 2
    hypotheses_found = _DIAG_HYPOTHESIS_RE.findall(content)
    if len(hypotheses_found) < 2:
        warnings.append(
            f"AD8 FAIL: Insufficient hypotheses ({len(hypotheses_found)} < 2)"
        )

    # AD9: Selected hypothesis is one of the listed ones
    # Extract H-IDs only from hypothesis headings (not from arbitrary body text)
    if selected_match and hypotheses_found:
        listed_h_ids = set()
        for h_text in hypotheses_found:
            h_id_match = re.search(r"\bH[1-4]\b", h_text)
            if h_id_match:
                listed_h_ids.add(h_id_match.group())
        selected_h_id = re.search(
            r"\bH[1-4]\b", selected_match.group(1).strip()
        )
        if selected_h_id and selected_h_id.group() not in listed_h_ids:
            warnings.append(
                f"AD9 FAIL: Selected hypothesis '{selected_h_id.group()}' "
                f"not found among listed hypotheses {listed_h_ids}"
            )

    # AD10: Previous diagnosis referenced (if retry > 0)
    retry_history = _gather_retry_history(project_dir, step, gate)
    if retry_history["retries_used"] > 0 and retry_history["previous_diagnoses"]:
        prev_ref_re = re.compile(
            r"(?:previous|prior|earlier)\s+(?:diagnosis|attempt|retry)",
            re.IGNORECASE,
        )
        if not prev_ref_re.search(content):
            warnings.append(
                "AD10 WARNING: No reference to previous diagnosis "
                f"(retry #{retry_history['retries_used']})"
            )

    # Determine overall validity (any FAIL → invalid)
    is_valid = not any("FAIL" in w for w in warnings)
    return is_valid, warnings


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
