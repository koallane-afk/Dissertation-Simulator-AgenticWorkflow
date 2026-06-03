#!/usr/bin/env python3
"""
Workflow Observability Tool — query_workflow.py

Single entry point for querying workflow execution state.
NOT a Hook — manually invoked by Orchestrator or user during workflow execution.

Usage:
    python3 .claude/hooks/scripts/query_workflow.py --project-dir . --dashboard
    python3 .claude/hooks/scripts/query_workflow.py --project-dir . --weakest-step
    python3 .claude/hooks/scripts/query_workflow.py --project-dir . --retry-summary
    python3 .claude/hooks/scripts/query_workflow.py --project-dir . --blocked
    python3 .claude/hooks/scripts/query_workflow.py --project-dir . --error-trends
    python3 .claude/hooks/scripts/query_workflow.py --project-dir . --dialogue
    python3 .claude/hooks/scripts/query_workflow.py --project-dir . --pccs

Output: JSON to stdout

Modes:
    --dashboard      Overview: current step, pACS history, pending validations
    --weakest-step   Step with lowest pACS score and its weak dimension
    --retry-summary  Retry budget usage across all gates
    --blocked        Identify what's blocking progress (failed gates, missing files)
    --error-trends   Cross-session error pattern aggregation from knowledge-index.jsonl
    --dialogue       Adversarial Dialogue status — current round, domain, file inventory
    --pccs           pCCS per-claim confidence — calibration delta, step history, decisions

P1 Compliance: All data extraction is deterministic (regex + file-system checks).
    SOT schema validated via validate_sot_schema() before any query.
    PyYAML required for YAML parsing — setup_init.py guarantees availability.
    pACS score extraction uses min()-formula-first strategy (same as _context_lib.py).
    All file read errors reported as typed errors in JSON output.
SOT Compliance: Read-only — no file writes.
"""

import argparse
import glob
import json
import os
import re
import sys

# Add script directory to path for shared library import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _context_lib import validate_sot_schema

# ---------------------------------------------------------------------------
# SOT discovery — extends _core_lib.py:SOT_FILENAMES with thesis SOT (session.json)
# ---------------------------------------------------------------------------
_SOT_FILENAMES = ("state.yaml", "state.yml", "state.json", "session.json")

# ---------------------------------------------------------------------------
# pACS score extraction — context-aware, min()-formula-first strategy
# Same logic as _context_lib.py verify_pacs_arithmetic() to avoid hallucination.
# ---------------------------------------------------------------------------
# D-7: Must match _context_lib.py:_PACS_WITH_MIN_RE / _PACS_SIMPLE_RE
_PACS_MIN_FORMULA_RE = re.compile(
    r"pACS\s*=\s*min\s*\([^)]+\)\s*=\s*(\d{1,3})", re.IGNORECASE
)
_PACS_SIMPLE_RE = re.compile(
    r"pACS\s*=\s*(\d{1,3})\b", re.IGNORECASE
)
_WEAK_DIM_RE = re.compile(
    r"^[#*\s]*weak(?:est)?\s*dimension\s*[=:]\s*([FCL])",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_pacs_score(content):
    """Extract pACS score from log content. Returns int or None.

    D-7: Patterns aligned with _context_lib.py:_PACS_WITH_MIN_RE/_PACS_SIMPLE_RE.
    Strategy:
      1. Prefer explicit min() formula (e.g., "pACS = min(F,C,L) = 75")
      2. Fallback to "pACS = N" if exactly 1 match (= separator only, no colon)
      3. Return None if ambiguous (0 or multiple matches without min formula)
    """
    min_match = _PACS_MIN_FORMULA_RE.search(content)
    if min_match:
        score = int(min_match.group(1))
        # Range validation: pACS must be 0-100 (D-7: matches verify_pacs_arithmetic)
        return score if 0 <= score <= 100 else None
    simple_matches = _PACS_SIMPLE_RE.findall(content)
    if len(simple_matches) == 1:
        score = int(simple_matches[0])
        return score if 0 <= score <= 100 else None
    return None  # Ambiguous — refuse to guess


def _extract_weak_dimension(content):
    """Extract weak dimension from log content. Returns 'F', 'C', 'L', or None.

    Only matches explicit 'Weak dimension: X' patterns at heading level
    to avoid false positives from body text.
    """
    m = _WEAK_DIM_RE.search(content)
    # .upper() normalizes case — IGNORECASE makes [FCL] match lowercase,
    # but downstream S7c expects uppercase ("F","C","L") only
    return m.group(1).upper() if m else None


def _read_file_safe(path):
    """Read file content with typed error reporting. Returns (content, error_dict).

    P1 Compliance: Never returns silent None — always either content or error.
    """
    if not os.path.isfile(path):
        return None, {"type": "file_not_found", "path": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), None
    except PermissionError:
        return None, {"type": "permission_denied", "path": path}
    except UnicodeDecodeError:
        return None, {"type": "encoding_error", "path": path}
    except OSError as e:
        return None, {"type": "os_error", "path": path, "detail": str(e)}


def _find_sot(project_dir):
    """Find and parse the SOT file. Returns (data, path, error).

    P1 Compliance:
      - PyYAML required (no fallback that truncates nested values).
      - Parse errors reported as typed errors, never silently swallowed.
    """
    for name in _SOT_FILENAMES:
        p = os.path.join(project_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            if name.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return None, p, f"SOT {name} parsed but is not a dict (got {type(data).__name__})"
                return data, p, None
            else:
                import yaml
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data is None:
                    data = {}
                if not isinstance(data, dict):
                    return None, p, f"SOT {name} parsed but is not a dict (got {type(data).__name__})"
                return data, p, None
        except ImportError:
            return None, p, "PyYAML not installed — run: pip install pyyaml (setup_init.py should have caught this)"
        except json.JSONDecodeError as e:
            return None, p, f"SOT {name} JSON parse error: {e}"
        except Exception as e:
            return None, p, f"SOT {name} parse error: {type(e).__name__}: {e}"
    return None, None, "No SOT file found (state.yaml/state.yml/state.json)"


# ---------------------------------------------------------------------------
# Mode: --dashboard
# ---------------------------------------------------------------------------
def _dashboard(project_dir, sot):
    """Overview: current step, pACS history, pending validations."""
    current_step = sot.get("current_step", 0)
    total_steps = sot.get("total_steps", "?")
    workflow_status = sot.get("workflow_status") or sot.get("status", "unknown")

    # Collect pACS history with typed errors
    pacs_history = {}
    pacs_errors = []
    pacs_dir = os.path.join(project_dir, "pacs-logs")
    if os.path.isdir(pacs_dir):
        for f in sorted(glob.glob(os.path.join(pacs_dir, "step-*-pacs.md"))):
            m = re.search(r"step-(\d+)-pacs\.md$", f)
            if not m:
                continue
            step_num = int(m.group(1))
            content, err = _read_file_safe(f)
            if err:
                pacs_errors.append(err)
                continue
            score = _extract_pacs_score(content)
            if score is not None:
                pacs_history[f"step-{step_num}"] = score

    # Check pending validations
    pending = []
    outputs = sot.get("outputs", {})
    if isinstance(outputs, dict):
        for key in outputs:
            m = re.match(r"step-(\d+)$", str(key))
            if not m:
                continue
            s = int(m.group(1))
            vlog = os.path.join(project_dir, "verification-logs", f"step-{s}-verify.md")
            if not os.path.isfile(vlog):
                pending.append(f"step-{s}: verification log missing")
            plog = os.path.join(project_dir, "pacs-logs", f"step-{s}-pacs.md")
            if not os.path.isfile(plog):
                pending.append(f"step-{s}: pACS log missing")

    # Autopilot status
    autopilot = sot.get("autopilot", {})
    autopilot_enabled = autopilot.get("enabled", False) if isinstance(autopilot, dict) else False

    result = {
        "mode": "dashboard",
        "current_step": current_step,
        "total_steps": total_steps,
        "workflow_status": workflow_status,
        "autopilot_enabled": autopilot_enabled,
        "pacs_history": pacs_history,
        "pending_validations": pending,
        "completed_steps": current_step - 1 if isinstance(current_step, int) and current_step > 0 else 0,
    }

    # Thesis SOT extensions (session.json specific fields)
    if "gates" in sot:
        gates = sot["gates"]
        if isinstance(gates, dict):
            result["gates"] = {
                k: v.get("status", v) if isinstance(v, dict) else v
                for k, v in gates.items()
            }
    if "hitl_checkpoints" in sot:
        hitl = sot["hitl_checkpoints"]
        if isinstance(hitl, dict):
            result["hitl_checkpoints"] = {
                k: v.get("status", v) if isinstance(v, dict) else v
                for k, v in hitl.items()
            }
    if "research_type" in sot:
        result["research_type"] = sot["research_type"]
    if "fallback_history" in sot:
        fb = sot["fallback_history"]
        if isinstance(fb, list):
            result["fallback_events"] = len(fb)
    if pacs_errors:
        result["pacs_read_errors"] = pacs_errors
    return result


# ---------------------------------------------------------------------------
# Mode: --weakest-step
# ---------------------------------------------------------------------------
def _weakest_step(project_dir, sot):
    """Step with lowest pACS score and its weak dimension."""
    pacs_dir = os.path.join(project_dir, "pacs-logs")
    if not os.path.isdir(pacs_dir):
        return {"mode": "weakest_step", "found": False, "reason": "No pacs-logs directory"}

    weakest = None
    weakest_score = 101
    read_errors = []

    for f in sorted(glob.glob(os.path.join(pacs_dir, "step-*-pacs.md"))):
        m = re.search(r"step-(\d+)-pacs\.md$", f)
        if not m:
            continue
        step_num = int(m.group(1))
        content, err = _read_file_safe(f)
        if err:
            read_errors.append(err)
            continue
        score = _extract_pacs_score(content)
        if score is None:
            continue
        weak_dim = _extract_weak_dimension(content)
        if score < weakest_score:
            weakest_score = score
            weakest = {
                "step": step_num,
                "pacs_score": score,
                "weak_dimension": weak_dim,
                "zone": "RED" if score < 50 else "YELLOW" if score < 70 else "GREEN",
                "file": f,
            }

    result = {"mode": "weakest_step"}
    if weakest:
        result.update({"found": True, **weakest})
    else:
        result.update({"found": False, "reason": "No pACS logs found"})
    if read_errors:
        result["read_errors"] = read_errors
    return result


# ---------------------------------------------------------------------------
# Mode: --retry-summary
# ---------------------------------------------------------------------------
def _retry_summary(project_dir, sot):
    """Retry budget usage across all gates."""
    retries = []
    for gate_dir in ("verification-logs", "pacs-logs", "review-logs"):
        gate = gate_dir.replace("-logs", "")
        full_dir = os.path.join(project_dir, gate_dir)
        if not os.path.isdir(full_dir):
            continue
        for f in sorted(glob.glob(os.path.join(full_dir, ".step-*-retry-count"))):
            m = re.search(r"\.step-(\d+)-retry-count$", f)
            if not m:
                continue
            step_num = int(m.group(1))
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    count = int(fh.read().strip())
            except (ValueError, OSError):
                count = -1  # -1 signals unreadable counter
            retries.append({
                "step": step_num,
                "gate": gate,
                "retries_used": count,
            })

    return {
        "mode": "retry_summary",
        "total_retries": sum(r["retries_used"] for r in retries if r["retries_used"] > 0),
        "entries": retries,
    }


# ---------------------------------------------------------------------------
# Mode: --blocked
# ---------------------------------------------------------------------------
def _blocked(project_dir, sot):
    """Identify what's blocking progress."""
    blockers = []
    current_step = sot.get("current_step", 0)
    if not isinstance(current_step, int):
        return {"mode": "blocked", "blockers": [{"type": "schema_error", "detail": "current_step is not an integer"}]}

    outputs = sot.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}

    # Check if current step output exists
    step_key = f"step-{current_step}"
    if step_key not in outputs:
        blockers.append({
            "type": "missing_output",
            "step": current_step,
            "detail": f"SOT outputs.{step_key} not recorded",
        })
    else:
        output_path = outputs[step_key]
        if isinstance(output_path, str):
            # Prevent absolute path bypass — SOT paths must be relative
            if os.path.isabs(output_path):
                blockers.append({
                    "type": "absolute_path",
                    "step": current_step,
                    "detail": f"SOT output path is absolute (must be relative): {output_path}",
                })
            elif not os.path.isfile(os.path.join(project_dir, output_path)):
                blockers.append({
                    "type": "missing_file",
                    "step": current_step,
                    "detail": f"Output file not found: {output_path}",
                })

    # Check for failed verification
    vlog = os.path.join(project_dir, "verification-logs", f"step-{current_step}-verify.md")
    content, err = _read_file_safe(vlog)
    if content is not None:
        if re.search(r"\bFAIL\b", content) and not re.search(r"Overall.*PASS", content):
            blockers.append({
                "type": "verification_fail",
                "step": current_step,
                "detail": "Verification log contains FAIL — criteria not met",
            })
    elif err and err["type"] != "file_not_found":
        blockers.append({
            "type": "verification_read_error",
            "step": current_step,
            "detail": f"Cannot read verification log: {err['type']}",
        })

    # Check for RED pACS
    plog = os.path.join(project_dir, "pacs-logs", f"step-{current_step}-pacs.md")
    content, err = _read_file_safe(plog)
    if content is not None:
        score = _extract_pacs_score(content)
        if score is not None and score < 50:
            blockers.append({
                "type": "pacs_red",
                "step": current_step,
                "detail": f"pACS = {score} (RED zone, < 50)",
            })
    elif err and err["type"] != "file_not_found":
        blockers.append({
            "type": "pacs_read_error",
            "step": current_step,
            "detail": f"Cannot read pACS log: {err['type']}",
        })

    # Check for review FAIL
    rlog = os.path.join(project_dir, "review-logs", f"step-{current_step}-review.md")
    content, err = _read_file_safe(rlog)
    if content is not None:
        if re.search(r"Verdict\s*:\s*FAIL", content, re.IGNORECASE):
            blockers.append({
                "type": "review_fail",
                "step": current_step,
                "detail": "Adversarial Review verdict is FAIL",
            })
    elif err and err["type"] != "file_not_found":
        blockers.append({
            "type": "review_read_error",
            "step": current_step,
            "detail": f"Cannot read review log: {err['type']}",
        })

    # Thesis-specific blockers: check gates and HITL checkpoints
    gates = sot.get("gates", {})
    if isinstance(gates, dict):
        for gate_name, gate_val in gates.items():
            status = gate_val.get("status", gate_val) if isinstance(gate_val, dict) else gate_val
            if status == "fail":
                blockers.append({
                    "type": "gate_fail",
                    "gate": gate_name,
                    "detail": f"Cross-validation gate {gate_name} failed",
                })

    hitl = sot.get("hitl_checkpoints", {})
    if isinstance(hitl, dict):
        for hitl_name, hitl_val in hitl.items():
            status = hitl_val.get("status", hitl_val) if isinstance(hitl_val, dict) else hitl_val
            if status == "blocked":
                blockers.append({
                    "type": "hitl_blocked",
                    "checkpoint": hitl_name,
                    "detail": f"HITL checkpoint {hitl_name} is blocked — requires human input",
                })

    return {
        "mode": "blocked",
        "current_step": current_step,
        "blockers": blockers,
        "is_blocked": len(blockers) > 0,
    }


# ---------------------------------------------------------------------------
# Error Trends (P1 — deterministic aggregation from knowledge-index.jsonl)
# ---------------------------------------------------------------------------
def _error_trends(project_dir):
    """Aggregate error_patterns across sessions for cross-session trend analysis.

    Reads knowledge-index.jsonl, extracts error_patterns from each session entry,
    and produces frequency-ranked aggregation with deterministic sort order
    (frequency desc → alphabetical asc for tie-breaking).

    P1: No LLM inference — pure JSON parsing + counting.
    SOT: Read-only — reads knowledge-index.jsonl only.
    """
    ki_path = os.path.join(
        project_dir, ".claude", "context-snapshots", "knowledge-index.jsonl"
    )
    if not os.path.isfile(ki_path):
        return {
            "mode": "error_trends",
            "error": "knowledge-index.jsonl not found",
            "ki_path": ki_path,
        }

    totals = {}  # error_type -> count
    resolved = {}  # error_type -> resolved_count
    sessions_with_errors = 0
    total_sessions = 0
    recent_examples = {}  # error_type -> most recent session_id

    try:
        with open(ki_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_sessions += 1
                error_patterns = entry.get("error_patterns")
                if not error_patterns:
                    continue

                session_id = entry.get("session_id", "unknown")
                has_errors = False

                # Handle both list-of-dicts and dict formats
                if isinstance(error_patterns, list):
                    for ep in error_patterns:
                        if not isinstance(ep, dict):
                            continue
                        error_type = ep.get("type", "unknown")
                        has_errors = True
                        totals[error_type] = totals.get(error_type, 0) + 1
                        if ep.get("resolution"):
                            resolved[error_type] = resolved.get(error_type, 0) + 1
                        recent_examples[error_type] = session_id
                elif isinstance(error_patterns, dict):
                    for error_type, details in error_patterns.items():
                        has_errors = True
                        if isinstance(details, int):
                            count = details
                            res_count = 0
                        elif isinstance(details, dict):
                            count = details.get("count", 1)
                            res_count = 1 if details.get("resolution") else 0
                        else:
                            continue
                        totals[error_type] = totals.get(error_type, 0) + count
                        resolved[error_type] = resolved.get(error_type, 0) + res_count
                        recent_examples[error_type] = session_id

                if has_errors:
                    sessions_with_errors += 1

    except OSError as e:
        return {
            "mode": "error_trends",
            "error": f"Failed to read knowledge-index: {e}",
            "ki_path": ki_path,
        }

    # Deterministic sort: frequency desc → alphabetical asc
    sorted_trends = sorted(
        totals.items(), key=lambda x: (-x[1], x[0])
    )

    trends = []
    for error_type, count in sorted_trends:
        res_count = resolved.get(error_type, 0)
        trends.append({
            "error_type": error_type,
            "total_occurrences": count,
            "resolved_count": res_count,
            "resolution_rate": round(res_count / count, 2) if count > 0 else 0,
            "last_session": recent_examples.get(error_type),
        })

    return {
        "mode": "error_trends",
        "total_sessions": total_sessions,
        "sessions_with_errors": sessions_with_errors,
        "error_rate": (
            round(sessions_with_errors / total_sessions, 2)
            if total_sessions > 0 else 0
        ),
        "trends": trends,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dialogue Status — Adversarial Dialogue observability
# ---------------------------------------------------------------------------

_DIALOGUE_OUTCOME_RE = re.compile(
    r"Outcome\s*:\s*(consensus|escalated)", re.IGNORECASE
)
_DIALOGUE_ROUNDS_RE = re.compile(r"Rounds\s+Used\s*:\s*(\d+)", re.IGNORECASE)


def _dialogue_status(project_dir, sot):
    """Query Adversarial Dialogue state for all steps.

    P1 Compliance: Reads dialogue-logs/ + session.json.dialogue_state.
    Returns: mode, current round, domain, file inventory per step.
    """
    dialogue_dir = os.path.join(project_dir, "dialogue-logs")
    result = {
        "mode": "dialogue",
        "dialogue_dir": dialogue_dir,
        "dialogue_dir_exists": os.path.exists(dialogue_dir),
        "steps": {},
        "in_progress": None,
        "summary": {"completed": 0, "in_progress": 0, "escalated": 0},
    }

    # Read in-progress state from session.json.dialogue_state
    session_json = os.path.join(project_dir, "session.json")
    if os.path.exists(session_json):
        try:
            with open(session_json, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            ds = session_data.get("dialogue_state")
            if isinstance(ds, dict) and ds.get("status") == "in_progress":
                result["in_progress"] = {
                    "step": ds.get("step"),
                    "rounds_used": ds.get("rounds_used", 0),
                    "max_rounds": ds.get("max_rounds"),
                    "domain": ds.get("domain"),
                    "status": ds.get("status"),
                    "round_history": ds.get("round_history", []),
                }
                result["summary"]["in_progress"] += 1
        except Exception as exc:
            result["session_json_error"] = str(exc)

    # Scan dialogue-logs/ for completed/escalated dialogue files
    if os.path.exists(dialogue_dir):
        # Collect step numbers from summary files
        summary_pattern = re.compile(r"step-(\d+)-summary\.md$")
        round_pattern = re.compile(r"step-(\d+)-r(\d+)-(fc|rv|cr)\.md$")

        for fname in sorted(os.listdir(dialogue_dir)):
            sm = summary_pattern.match(fname)
            if sm:
                step_num = int(sm.group(1))
                fpath = os.path.join(dialogue_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read(1000)
                    outcome_m = _DIALOGUE_OUTCOME_RE.search(content)
                    rounds_m = _DIALOGUE_ROUNDS_RE.search(content)
                    outcome = outcome_m.group(1) if outcome_m else "unknown"
                    rounds = int(rounds_m.group(1)) if rounds_m else None
                    step_key = f"step-{step_num}"
                    if step_key not in result["steps"]:
                        result["steps"][step_key] = {}
                    result["steps"][step_key].update({
                        "outcome": outcome,
                        "rounds_used": rounds,
                        "summary_file": fname,
                    })
                    if outcome == "consensus":
                        result["summary"]["completed"] += 1
                    elif outcome == "escalated":
                        result["summary"]["escalated"] += 1
                except Exception:
                    pass

            rm = round_pattern.match(fname)
            if rm:
                step_num = int(rm.group(1))
                round_num = int(rm.group(2))
                critic = rm.group(3)
                step_key = f"step-{step_num}"
                if step_key not in result["steps"]:
                    result["steps"][step_key] = {}
                rounds_key = f"round_{round_num}_files"
                if rounds_key not in result["steps"][step_key]:
                    result["steps"][step_key][rounds_key] = []
                result["steps"][step_key][rounds_key].append(fname)

    return result


def _pccs_status(project_dir, sot):
    """Query pCCS (per-claim confidence score) state across steps.

    P1 Compliance: Reads session.json.pccs block + pccs report files.
    Returns: calibration delta, history per step, mean scores, decision actions.
    """
    result = {
        "mode": "pccs",
        "pccs_active": False,
        "cal_delta": 0.0,
        "total_cal_samples": 0,
        "last_step": None,
        "history": {},
        "step_count": 0,
        "mean_pccs_across_steps": None,
        "action_counts": {"proceed": 0, "rewrite_claims": 0, "rewrite_step": 0},
    }

    pccs = sot.get("pccs")
    if not isinstance(pccs, dict):
        return result

    result["pccs_active"] = True
    result["cal_delta"] = pccs.get("cal_delta", 0.0)
    result["total_cal_samples"] = pccs.get("total_cal_samples", 0)
    result["last_step"] = pccs.get("last_step")

    history = pccs.get("history")
    if isinstance(history, dict) and history:
        all_means = []
        for key in sorted(history.keys()):
            entry = history[key]
            if isinstance(entry, dict):
                mean = entry.get("mean_pccs")
                action = entry.get("action", "unknown")
                result["history"][key] = {
                    "mean_pccs": mean,
                    "green": entry.get("green", 0),
                    "yellow": entry.get("yellow", 0),
                    "red": entry.get("red", 0),
                    "total": entry.get("total_claims", 0),
                    "action": action,
                }
                if isinstance(mean, (int, float)):
                    all_means.append(mean)
                if action in result["action_counts"]:
                    result["action_counts"][action] += 1

        result["step_count"] = len(history)
        if all_means:
            result["mean_pccs_across_steps"] = round(
                sum(all_means) / len(all_means), 1
            )

    return result


# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Workflow Observability Tool")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--dashboard", action="store_true", help="Overview dashboard")
    parser.add_argument("--weakest-step", action="store_true", help="Weakest pACS step")
    parser.add_argument("--retry-summary", action="store_true", help="Retry budget usage")
    parser.add_argument("--blocked", action="store_true", help="Progress blockers")
    parser.add_argument("--error-trends", action="store_true", help="Cross-session error trends")
    parser.add_argument("--dialogue", action="store_true",
                        help="Adversarial Dialogue status — current round, domain, file inventory")
    parser.add_argument("--pccs", action="store_true",
                        help="pCCS per-claim confidence score state — calibration, history, decisions")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)

    # --error-trends reads knowledge-index.jsonl, not SOT
    if args.error_trends:
        result = _error_trends(project_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)

    sot, sot_path, sot_error = _find_sot(project_dir)

    if sot is None:
        print(json.dumps({
            "error": sot_error or "No SOT file found",
            "sot_path": sot_path,
            "project_dir": project_dir,
        }, indent=2))
        sys.exit(1)

    # P1: Validate SOT schema before any query
    schema_warnings = validate_sot_schema(sot)

    if args.dashboard:
        result = _dashboard(project_dir, sot)
    elif args.weakest_step:
        result = _weakest_step(project_dir, sot)
    elif args.retry_summary:
        result = _retry_summary(project_dir, sot)
    elif args.blocked:
        result = _blocked(project_dir, sot)
    elif args.dialogue:
        result = _dialogue_status(project_dir, sot)
    elif args.pccs:
        result = _pccs_status(project_dir, sot)
    else:
        result = _dashboard(project_dir, sot)  # default

    result["sot_path"] = sot_path
    if schema_warnings:
        result["sot_schema_warnings"] = schema_warnings

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e), "error_type": type(e).__name__}, indent=2))
        sys.exit(1)
