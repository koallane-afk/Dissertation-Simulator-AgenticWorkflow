#!/usr/bin/env python3
"""Quality-gate P1 validators extracted from _context_lib.py per ADR-076 (Increment 1).

Covers pACS, traceability, review, translation, verification, domain-knowledge,
workflow, and L0 anti-skip validation. Depends only on _core_lib.
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

# Ensure _core_lib resolves even when this module is loaded by file path
# without the scripts dir on sys.path (ADR-076).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _core_lib import (  # noqa: E402
    MIN_OUTPUT_SIZE,
    sot_paths,
)


_REVIEW_REQUIRED_SECTIONS = [
    re.compile(r"^#+\s*Pre-mortem", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^#+\s*Issues\s+Found", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^#+\s*Independent\s+pACS", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^#+\s*Verdict", re.MULTILINE | re.IGNORECASE),
]


_REVIEW_VERDICT_RE = re.compile(
    r"^#+\s*Verdict\s*:\s*[*_~`]*\s*(PASS|FAIL)(?:\b|[*_~`])",
    re.MULTILINE | re.IGNORECASE,
)


_REVIEW_ISSUE_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|", re.MULTILINE
)


_REVIEW_CRITICAL_RE = re.compile(r"\bCritical\b", re.IGNORECASE)


_REVIEW_WARNING_RE = re.compile(r"\bWarning\b", re.IGNORECASE)


_REVIEW_SUGGESTION_RE = re.compile(r"\bSuggestion\b", re.IGNORECASE)


_REVIEW_PACS_DIM_RE = re.compile(
    r"^\|\s*([FCL])\s*\|\s*(\d{1,3})\s*\|", re.MULTILINE
)


_REVIEW_PACS_FINAL_RE = re.compile(
    r"Reviewer\s+pACS\s*=.*?=\s*(\d{1,3})", re.IGNORECASE
)


_REVIEW_GENERATOR_PACS_RE = re.compile(
    r"Generator\s+pACS\s*=\s*(\d{1,3})", re.IGNORECASE
)


def validate_review_output(project_dir, step_number):
    """Anti-Skip Guard for Adversarial Review outputs.

    P1 Compliance: All 5 checks are deterministic (filesystem + regex).
    SOT Compliance: Read-only access to review-logs/.

    Checks:
      R1: review-logs/step-{N}-review.md exists
      R2: File size >= MIN_OUTPUT_SIZE (100 bytes)
      R3: All 4 required sections present (Pre-mortem, Issues, pACS, Verdict)
      R4: Verdict is explicitly PASS or FAIL
      R5: Issues table has >= 1 data row (rubber-stamp prevention)

    Args:
        project_dir: Project root directory path
        step_number: Step number (int) to validate

    Returns:
        tuple: (is_valid: bool, verdict: str|None, issues_count: int,
                warnings: list[str])
        - is_valid: True only if all R1-R5 pass
        - verdict: "PASS" or "FAIL" or None if not extractable
        - issues_count: Number of issue rows found
        - warnings: List of human-readable failure reasons
    """
    warnings = []
    review_path = os.path.join(
        project_dir, "review-logs", f"step-{step_number}-review.md"
    )

    # R1: File existence
    if not os.path.exists(review_path):
        warnings.append(
            f"R1 FAIL: review-logs/step-{step_number}-review.md not found"
        )
        return (False, None, 0, warnings)

    # R2: Minimum size
    try:
        size = os.path.getsize(review_path)
    except OSError:
        warnings.append(f"R2 FAIL: Cannot read file size: {review_path}")
        return (False, None, 0, warnings)

    if size < MIN_OUTPUT_SIZE:
        warnings.append(
            f"R2 FAIL: Review too small ({size} bytes, min {MIN_OUTPUT_SIZE})"
        )
        return (False, None, 0, warnings)

    # Read content for R3-R5
    try:
        with open(review_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        warnings.append(f"R2 FAIL: Cannot read file: {e}")
        return (False, None, 0, warnings)

    # R3: Required sections
    for i, pattern in enumerate(_REVIEW_REQUIRED_SECTIONS):
        section_names = ["Pre-mortem", "Issues Found", "Independent pACS", "Verdict"]
        if not pattern.search(content):
            warnings.append(f"R3 FAIL: Missing required section: {section_names[i]}")

    # R4: Verdict extraction
    verdict_match = _REVIEW_VERDICT_RE.search(content)
    verdict = verdict_match.group(1).upper() if verdict_match else None
    if verdict is None:
        warnings.append("R4 FAIL: No explicit PASS/FAIL verdict found")

    # R5: Issues table rows (rubber-stamp prevention)
    issue_rows = _REVIEW_ISSUE_ROW_RE.findall(content)
    issues_count = len(issue_rows)
    if issues_count < 1:
        warnings.append("R5 FAIL: No issues found in table (minimum 1 required)")

    is_valid = len(warnings) == 0
    return (is_valid, verdict, issues_count, warnings)


def parse_review_verdict(review_path):
    """Extract PASS/FAIL verdict and issue severity counts from review report.

    P1 Compliance: Regex-based extraction only, no LLM interpretation.
    Useful for Orchestrator to make deterministic proceed/rework decisions.

    Args:
        review_path: Absolute path to review-logs/step-N-review.md

    Returns:
        dict with keys:
        - verdict: "PASS" | "FAIL" | None
        - critical_count: int (number of Critical issues)
        - warning_count: int (number of Warning issues)
        - suggestion_count: int (number of Suggestion issues)
        - reviewer_pacs: int | None (reviewer's pACS score)
        - pacs_dimensions: dict | None ({"F": int, "C": int, "L": int})
    """
    result = {
        "verdict": None,
        "critical_count": 0,
        "warning_count": 0,
        "suggestion_count": 0,
        "reviewer_pacs": None,
        "pacs_dimensions": None,
    }

    if not os.path.exists(review_path):
        return result

    try:
        with open(review_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError):
        return result

    # Verdict
    verdict_match = _REVIEW_VERDICT_RE.search(content)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).upper()

    # Issue severity counts from table rows
    issue_rows = _REVIEW_ISSUE_ROW_RE.findall(content)
    # Each row is like "| 1 | Critical | file:line | Problem | Fix |"
    # We need to check each full row line for severity
    for row_start in _REVIEW_ISSUE_ROW_RE.finditer(content):
        # Get the full line containing this row
        line_start = content.rfind("\n", 0, row_start.start()) + 1
        line_end = content.find("\n", row_start.start())
        if line_end == -1:
            line_end = len(content)
        line = content[line_start:line_end]

        if _REVIEW_CRITICAL_RE.search(line):
            result["critical_count"] += 1
        elif _REVIEW_WARNING_RE.search(line):
            result["warning_count"] += 1
        elif _REVIEW_SUGGESTION_RE.search(line):
            result["suggestion_count"] += 1

    # Reviewer pACS dimensions
    dims = {}
    for dim_match in _REVIEW_PACS_DIM_RE.finditer(content):
        dim_name = dim_match.group(1).upper()
        dim_score = int(dim_match.group(2))
        if 0 <= dim_score <= 100:
            dims[dim_name] = dim_score

    if len(dims) == 3 and all(k in dims for k in ("F", "C", "L")):
        result["pacs_dimensions"] = dims

    # Reviewer pACS final score
    pacs_match = _REVIEW_PACS_FINAL_RE.search(content)
    if pacs_match:
        score = int(pacs_match.group(1))
        if 0 <= score <= 100:
            result["reviewer_pacs"] = score
    elif result["pacs_dimensions"]:
        # Fallback: calculate from dimensions (min of F, C, L)
        result["reviewer_pacs"] = min(result["pacs_dimensions"].values())

    return result


def verify_verdict_consistency(verdict, critical_count, warning_count=0):
    """H3: P1 cross-check — verdict must be logically consistent with issues.

    Rules (deterministic):
      - PASS + critical_count >= 1 → INCONSISTENCY (Critical issues require FAIL)
      - FAIL + critical_count == 0 AND warning_count == 0 → SUSPICIOUS (justify)

    Returns: list of warning strings (empty = consistent).
    """
    warnings = []
    if verdict is None:
        return warnings

    v = verdict.upper()
    if v == "PASS" and critical_count >= 1:
        warnings.append(
            f"VERDICT_INCONSISTENCY: Verdict=PASS but {critical_count} Critical "
            f"issue(s) found. Critical issues require FAIL verdict."
        )
    if v == "FAIL" and critical_count == 0 and warning_count == 0:
        warnings.append(
            "VERDICT_SUSPICIOUS: Verdict=FAIL but no Critical/Warning issues found. "
            "Justify failure or correct verdict."
        )
    return warnings


def calculate_pacs_delta(project_dir, step_number):
    """Calculate |generator_pACS - reviewer_pACS| for reconciliation detection.

    P1 Compliance: Pure arithmetic — no LLM interpretation.
    Reads from pacs-logs/step-N-pacs.md (generator) and
    review-logs/step-N-review.md (reviewer).

    A delta >= 15 indicates potential miscalibration and may require
    reconciliation (either generator inflated or reviewer too harsh).

    Args:
        project_dir: Project root directory path
        step_number: Step number (int)

    Returns:
        dict with keys:
        - generator_score: int | None
        - reviewer_score: int | None
        - delta: int | None (absolute difference, None if either score missing)
        - needs_reconciliation: bool (True if delta >= 15)
    """
    result = {
        "generator_score": None,
        "reviewer_score": None,
        "delta": None,
        "needs_reconciliation": False,
    }

    # Extract generator pACS from pacs-logs/step-N-pacs.md
    generator_path = os.path.join(
        project_dir, "pacs-logs", f"step-{step_number}-pacs.md"
    )
    if os.path.exists(generator_path):
        try:
            with open(generator_path, "r", encoding="utf-8") as f:
                gen_content = f.read()
            # Pattern: "pACS = min(F, C, L) = 85" or "pACS = 85"
            gen_match = re.search(
                r"pACS\s*=.*?=\s*(\d{1,3})|pACS\s*=\s*(\d{1,3})",
                gen_content, re.IGNORECASE
            )
            if gen_match:
                score_str = gen_match.group(1) or gen_match.group(2)
                score = int(score_str)
                if 0 <= score <= 100:
                    result["generator_score"] = score
        except (IOError, UnicodeDecodeError, ValueError):
            pass

    # Extract reviewer pACS from review report
    review_path = os.path.join(
        project_dir, "review-logs", f"step-{step_number}-review.md"
    )
    review_data = parse_review_verdict(review_path)
    result["reviewer_score"] = review_data.get("reviewer_pacs")

    # Calculate delta
    if result["generator_score"] is not None and result["reviewer_score"] is not None:
        result["delta"] = abs(result["generator_score"] - result["reviewer_score"])
        result["needs_reconciliation"] = result["delta"] >= 15

    return result


def _read_sot_outputs(project_dir):
    """Read SOT file and return outputs dict. Read-only.

    P1 Compliance: Deterministic file read + parse.
    SOT Compliance: Read-only access.

    Returns:
        dict: outputs section from SOT, or {} on any failure.
    """
    for sot_file in sot_paths(project_dir):
        if not os.path.exists(sot_file):
            continue
        try:
            with open(sot_file, "r", encoding="utf-8") as f:
                content = f.read()
            if sot_file.endswith(".json"):
                import json
                data = json.loads(content)
            else:
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except ImportError:
                    continue
            if isinstance(data, dict):
                outputs = data.get("outputs", {})
                return outputs if isinstance(outputs, dict) else {}
        except Exception:
            continue
    return {}


def _find_translation_files_for_step(project_dir, step_number):
    """Discover translation files for a step via 3-tier fallback.

    Tier 1: SOT outputs.step-{N}-ko (explicit ko path from SOT)
    Tier 2: translations/ directory (legacy compatibility)
    Tier 3: Sibling *.ko.md next to SOT outputs.step-{N} (same-dir convention
             per translator.md: "output file must be in the same directory
             as the English original")

    P1 Compliance: Filesystem operations only.
    SOT Compliance: Read-only access via _read_sot_outputs().

    Returns:
        list: Existing translation file paths (deduplicated by realpath).
    """
    found = []
    seen = set()

    def _add(path):
        rp = os.path.realpath(path)
        if rp not in seen and os.path.exists(path):
            seen.add(rp)
            found.append(path)

    # --- Tier 1 & 3: Read SOT once ---
    outputs = _read_sot_outputs(project_dir)
    if outputs:
        # Tier 1: Explicit ko path from SOT outputs
        ko_key = f"step-{step_number}-ko"
        ko_val = outputs.get(ko_key)
        if ko_val:
            ko_path = (
                ko_val if os.path.isabs(ko_val)
                else os.path.join(project_dir, ko_val)
            )
            _add(ko_path)

        # Tier 3: Sibling .ko.md next to original output file
        orig_key = f"step-{step_number}"
        orig_val = outputs.get(orig_key)
        if orig_val:
            orig_path = (
                orig_val if os.path.isabs(orig_val)
                else os.path.join(project_dir, orig_val)
            )
            base, ext = os.path.splitext(orig_path)
            sibling = f"{base}.ko{ext}" if ext else f"{orig_path}.ko.md"
            _add(sibling)

    # --- Tier 2: translations/ directory (legacy) ---
    translations_dir = os.path.join(project_dir, "translations")
    if os.path.isdir(translations_dir):
        prefix = f"step-{step_number}"
        try:
            for fname in os.listdir(translations_dir):
                if fname.startswith(prefix) and fname.endswith(".ko.md"):
                    _add(os.path.join(translations_dir, fname))
        except OSError:
            pass

    return found


def validate_review_sequence(project_dir, step_number):
    """Verify that review PASS preceded translation start.

    P1 Compliance: File timestamp comparison — deterministic.
    Prevents translating flawed output (review FAIL → no translation).

    Translation file discovery uses 3-tier fallback:
      Tier 1: SOT outputs.step-{N}-ko (explicit path)
      Tier 2: translations/ directory (legacy)
      Tier 3: Sibling *.ko.md next to original (translator.md convention)

    Checks:
      1. review-logs/step-{N}-review.md exists with PASS verdict
      2. If translation (*.ko.md) exists for this step, review file must be older

    Args:
        project_dir: Project root directory path
        step_number: Step number (int)

    Returns:
        tuple: (is_valid: bool, warning: str | None)
        - is_valid: True if sequence is correct or no translation exists
        - warning: Human-readable issue description, None if valid
    """
    review_path = os.path.join(
        project_dir, "review-logs", f"step-{step_number}-review.md"
    )

    # 3-tier translation file discovery
    translation_files = _find_translation_files_for_step(project_dir, step_number)

    # No translation files → sequence is trivially valid
    if not translation_files:
        return (True, None)

    # Translation exists but no review → violation
    if not os.path.exists(review_path):
        return (
            False,
            f"Step {step_number}: Translation exists but no review report found. "
            f"Review must PASS before translation.",
        )

    # Check review verdict
    review_data = parse_review_verdict(review_path)
    if review_data["verdict"] != "PASS":
        return (
            False,
            f"Step {step_number}: Translation exists but review verdict is "
            f"{review_data['verdict'] or 'UNKNOWN'} (must be PASS).",
        )

    # Timestamp check: review must be older than (or same as) translation
    try:
        review_mtime = os.path.getmtime(review_path)
    except OSError:
        return (
            False,
            f"Step {step_number}: Cannot read review file timestamp.",
        )

    for tf in translation_files:
        try:
            trans_mtime = os.path.getmtime(tf)
            if trans_mtime < review_mtime:
                return (
                    False,
                    f"Step {step_number}: Translation {os.path.basename(tf)} "
                    f"(mtime {trans_mtime:.0f}) is older than review "
                    f"(mtime {review_mtime:.0f}). Translation may precede review.",
                )
        except OSError:
            continue

    return (True, None)


def validate_file_coverage(review_path, files_to_check):
    """CR6: Verify all declared files appear in the code review report.

    P1 Compliance: String presence check — deterministic.
    SOT Compliance: Read-only access to review file.

    Args:
        review_path: Path to the review report file
        files_to_check: List of filenames (basename only) that must appear in report

    Returns:
        tuple: (is_valid: bool, missing_files: list[str], warnings: list[str])
    """
    if not files_to_check:
        return (True, [], [])

    if not os.path.exists(review_path):
        return (
            False,
            list(files_to_check),
            [f"CR6 FAIL: Review file not found: {review_path}"],
        )

    try:
        with open(review_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return (False, list(files_to_check), [f"CR6 FAIL: Cannot read review file: {e}"])

    missing = []
    for filename in files_to_check:
        # Check for basename presence (case-sensitive for code files)
        basename = os.path.basename(filename.strip())
        if basename and basename not in content:
            missing.append(basename)

    if missing:
        return (
            False,
            missing,
            [f"CR6 FAIL: Files not mentioned in review: {', '.join(missing)}"],
        )

    return (True, [], [])


_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


_CODE_BLOCK_FENCE_RE = re.compile(r"^```", re.MULTILINE)


_PACS_DIM_UNIVERSAL_RE = re.compile(
    r"^\|\s*([A-Z][a-z]?)\s*(?:\([^)]*\))?\s*\|\s*(\d{1,3})\s*\|",
    re.MULTILINE,
)


_PACS_WITH_MIN_RE = re.compile(
    r"pACS\s*=\s*min\s*\([^)]+\)\s*=\s*(\d{1,3})",
    re.IGNORECASE,
)


_PACS_SIMPLE_RE = re.compile(
    r"pACS\s*=\s*(\d{1,3})\b",
    re.IGNORECASE,
)


_VERIFY_CRITERION_CHECKLIST_RE = re.compile(
    r"^[-*]\s*\[?\s*[xX✅❌ ]?\s*\]?\s*(.+?)[:：]\s*(PASS|FAIL)",
    re.MULTILINE | re.IGNORECASE,
)


_VERIFY_CRITERION_TABLE_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(PASS|FAIL)\s*\|",
    re.MULTILINE | re.IGNORECASE,
)


_VERIFY_CRITERION_TABLE_EVIDENCE_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(PASS|FAIL)[^|]*\|\s*([^|]*?)\s*\|",
    re.MULTILINE | re.IGNORECASE,
)


_VERIFY_COMPOUND_CRITERION_RE = re.compile(
    r"\s+(?:\+|and\s|및\s|그리고\s|&\s)",
    re.IGNORECASE,
)


_VERIFY_OVERALL_RE = re.compile(
    r"(?:Overall|Total|종합|최종)\s*(?:Result|결과|Verdict|판정)?\s*[:：]\s*(PASS|FAIL)",
    re.IGNORECASE,
)


_VERIFY_TABLE_HEADER_WORDS = frozenset({
    "criterion", "criteria", "check", "기준", "항목",
    "dimension", "result", "evidence", "---",
})


def validate_translation_output(project_dir, step_number):
    """Anti-Skip Guard for Translation outputs (T1-T7).

    P1 Compliance: All 7 checks are deterministic (filesystem + regex).
    SOT Compliance: Read-only access via _read_sot_outputs().

    Checks:
      T1: Translation file exists (3-tier discovery via _find_translation_files_for_step)
      T2: Translation file size >= MIN_OUTPUT_SIZE (100 bytes)
      T3: English source file exists (from SOT outputs)
      T4: Translation file has .ko.md extension
      T5: Translation content is non-empty (not just whitespace)
      T6: Structural completeness — heading count EN ≈ KO (±20% tolerance)
      T7: Code block preservation — code block fence count EN == KO

    Args:
        project_dir: Project root directory path
        step_number: Step number (int) to validate

    Returns:
        tuple: (is_valid: bool, warnings: list[str])
        - is_valid: True only if all T1-T7 pass
        - warnings: List of human-readable failure reasons
    """
    warnings = []

    # --- T1: Translation file existence (3-tier discovery) ---
    translation_files = _find_translation_files_for_step(project_dir, step_number)
    if not translation_files:
        warnings.append(
            f"T1 FAIL: No translation file found for step {step_number} "
            f"(checked SOT ko key, translations/ dir, sibling .ko.md)"
        )
        return (False, warnings)

    # Use first discovered translation file as primary
    ko_path = translation_files[0]

    # --- T2: Minimum size ---
    try:
        ko_size = os.path.getsize(ko_path)
    except OSError:
        warnings.append(f"T2 FAIL: Cannot read file size: {ko_path}")
        return (False, warnings)

    if ko_size < MIN_OUTPUT_SIZE:
        warnings.append(
            f"T2 FAIL: Translation too small ({ko_size} bytes, min {MIN_OUTPUT_SIZE})"
        )

    # --- T3: English source file existence ---
    outputs = _read_sot_outputs(project_dir)
    en_path = None
    if outputs:
        en_val = outputs.get(f"step-{step_number}")
        if en_val:
            en_path = (
                en_val if os.path.isabs(en_val)
                else os.path.join(project_dir, en_val)
            )

    if en_path is None:
        warnings.append(
            f"T3 FAIL: No English source path in SOT outputs.step-{step_number}"
        )
    elif not os.path.exists(en_path):
        warnings.append(f"T3 FAIL: English source not found: {en_path}")

    # --- T4: .ko.md extension ---
    ko_basename = os.path.basename(ko_path)
    if not ko_basename.endswith(".ko.md"):
        warnings.append(
            f"T4 FAIL: Translation filename '{ko_basename}' does not end with .ko.md"
        )

    # --- T5-T7: Content-based checks (require reading files) ---
    ko_content = None
    try:
        with open(ko_path, "r", encoding="utf-8") as f:
            ko_content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        warnings.append(f"T5 FAIL: Cannot read translation file: {e}")

    if ko_content is not None:
        # T5: Non-empty content
        if not ko_content.strip():
            warnings.append("T5 FAIL: Translation file contains only whitespace")

        # T6 & T7 require English source content
        en_content = None
        if en_path and os.path.exists(en_path):
            try:
                with open(en_path, "r", encoding="utf-8") as f:
                    en_content = f.read()
            except (IOError, UnicodeDecodeError) as e:
                warnings.append(
                    f"T6/T7 SKIP: English source exists but unreadable: {e}"
                )

        if en_content is not None and ko_content.strip():
            # T6: Structural completeness
            t6_valid, t6_msg = _check_structural_completeness(en_content, ko_content)
            if not t6_valid:
                warnings.append(t6_msg)

            # T7: Code block preservation
            t7_valid, t7_msg = _check_code_block_preservation(en_content, ko_content)
            if not t7_valid:
                warnings.append(t7_msg)

    is_valid = len(warnings) == 0
    return (is_valid, warnings)


def _check_structural_completeness(en_content, ko_content):
    """T6: Heading count comparison between EN and KO documents.

    P1 Compliance: Regex counting — deterministic.

    Tolerance: KO headings within ±20% of EN count (minimum ±1).
    Minor structural adjustments by translator are acceptable;
    major omissions are not.

    Args:
        en_content: English document content (str)
        ko_content: Korean document content (str)

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    en_headings = len(_HEADING_RE.findall(en_content))
    ko_headings = len(_HEADING_RE.findall(ko_content))

    if en_headings == 0:
        return (True, "T6 SKIP: No headings in English source")

    # ±20% tolerance (minimum ±1 for small documents)
    tolerance = max(1, int(en_headings * 0.2))
    diff = abs(en_headings - ko_headings)

    if diff > tolerance:
        return (
            False,
            f"T6 FAIL: Heading count mismatch — EN={en_headings}, KO={ko_headings} "
            f"(tolerance ±{tolerance})",
        )

    return (True, f"T6 PASS: EN={en_headings}, KO={ko_headings}")


def _check_code_block_preservation(en_content, ko_content):
    """T7: Code block fence count must match exactly between EN and KO.

    P1 Compliance: Regex counting — deterministic.
    Per translator.md: "Code blocks are NEVER translated — Keep all code."
    Triple-backtick fences must be preserved 1:1.

    Args:
        en_content: English document content (str)
        ko_content: Korean document content (str)

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    en_fences = len(_CODE_BLOCK_FENCE_RE.findall(en_content))
    ko_fences = len(_CODE_BLOCK_FENCE_RE.findall(ko_content))

    if en_fences == 0:
        return (True, "T7 SKIP: No code blocks in English source")

    if en_fences != ko_fences:
        return (
            False,
            f"T7 FAIL: Code block fence count mismatch — EN={en_fences}, KO={ko_fences} "
            f"(must be exact match)",
        )

    return (True, f"T7 PASS: {en_fences} code fences preserved")


def check_glossary_freshness(project_dir, step_number):
    """T8: Verify glossary was updated during/after translation.

    P1 Compliance: File timestamp comparison — deterministic.
    Per translator.md protocol: Step 5 (Update Glossary) → Step 6 (Write Output).

    Checks:
      - translations/glossary.yaml exists
      - glossary.yaml was modified within 1 hour of translation file

    Args:
        project_dir: Project root directory path
        step_number: Step number (int) to validate

    Returns:
        tuple: (is_valid: bool, warning: str | None)
        - is_valid: True if glossary is fresh or no translation exists
        - warning: Human-readable issue description, None if valid
    """
    glossary_path = os.path.join(project_dir, "translations", "glossary.yaml")

    if not os.path.exists(glossary_path):
        return (
            False,
            "T8 FAIL: translations/glossary.yaml not found — "
            "translator must create/update glossary (Step 5)",
        )

    # Find translation files for timestamp comparison
    translation_files = _find_translation_files_for_step(project_dir, step_number)
    if not translation_files:
        return (True, None)  # No translation → T8 trivially valid

    try:
        glossary_mtime = os.path.getmtime(glossary_path)
    except OSError:
        return (False, "T8 FAIL: Cannot read glossary.yaml timestamp")

    ko_path = translation_files[0]
    try:
        ko_mtime = os.path.getmtime(ko_path)
    except OSError:
        return (False, f"T8 FAIL: Cannot read translation file timestamp: {ko_path}")

    # Tolerance: glossary should be modified within 1 hour of translation
    # (translator protocol: Step 5 → Step 6, typically within same session)
    staleness = ko_mtime - glossary_mtime
    if staleness > 3600:
        return (
            False,
            f"T8 FAIL: Glossary is stale — modified {staleness:.0f}s before "
            f"translation (max 3600s). Translator may have skipped Step 5.",
        )

    return (True, None)


def verify_pacs_arithmetic(pacs_log_path):
    """T9: Universal pACS arithmetic verification.

    P1 Compliance: Regex + arithmetic — deterministic.
    Applies to ALL pACS log types (general, translation, reviewer).

    Verifies that the reported pACS score equals min(dimension scores).
    This catches AI hallucination where dimension scores are stated but
    min() is calculated incorrectly.

    Strategy:
      1. Prefer explicit min() formula match (e.g., "pACS = min(F,C,L) = 75")
      2. Fallback to simple "pACS = N" if unambiguous (exactly 1 match)
      3. Skip if ambiguous (multiple simple matches without min formula)

    Supports dimension naming patterns:
      - General: F, C, L (Faithfulness, Completeness, Logic)
      - Translation: Ft, Ct, Nt (Fidelity, Completeness, Naturalness)
      - Any single/two-letter uppercase dimension codes

    Ambiguity guard: If the same dimension letter appears with different
    scores (e.g., generator and reviewer tables in same file), verification
    is skipped to avoid false alarms.

    Args:
        pacs_log_path: Absolute path to any pACS log file

    Returns:
        tuple: (is_valid: bool, warning: str | None)
        - is_valid: True if arithmetic is correct or cannot be verified
        - warning: Human-readable issue description, None if valid
    """
    if not os.path.exists(pacs_log_path):
        return (True, None)  # No file → nothing to verify

    try:
        with open(pacs_log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError):
        return (True, None)  # Unreadable → graceful skip

    # H7: Normalize non-table pACS dimension formats before extraction
    # Converts "F (Faithfulness): 85" or "F: 85" to "| F | 85 |" for regex matching
    content = re.sub(
        r'^([FCL][a-z]?)\s*(?:\([^)]*\))?\s*:\s*(\d{1,3})\s*$',
        r'| \1 | \2 |',
        content, flags=re.MULTILINE,
    )

    # --- Extract dimension scores ---
    dims = {}
    seen_dim_scores = {}  # dim -> list of scores (for ambiguity detection)
    for match in _PACS_DIM_UNIVERSAL_RE.finditer(content):
        dim_name = match.group(1)
        dim_score = int(match.group(2))
        if 0 <= dim_score <= 100:
            if dim_name in seen_dim_scores:
                seen_dim_scores[dim_name].append(dim_score)
            else:
                seen_dim_scores[dim_name] = [dim_score]
            dims[dim_name] = dim_score

    if len(dims) < 2:
        return (True, None)  # Not enough dimensions → skip

    # Ambiguity guard: same dimension with different scores → skip
    for scores in seen_dim_scores.values():
        if len(set(scores)) > 1:
            return (True, None)

    # --- Extract reported final score ---
    # Strategy: prefer explicit min() formula over simple "pACS = N"
    min_match = _PACS_WITH_MIN_RE.search(content)
    if min_match:
        reported_score = int(min_match.group(1))
    else:
        # Fallback: simple "pACS = N" (no min formula)
        simple_matches = _PACS_SIMPLE_RE.findall(content)
        if len(simple_matches) == 1:
            reported_score = int(simple_matches[0])
        else:
            return (True, None)  # 0 or multiple → ambiguous → skip

    # --- Verify arithmetic ---
    expected_score = min(dims.values())
    if reported_score != expected_score:
        dim_str = ", ".join(f"{k}={v}" for k, v in sorted(dims.items()))
        return (
            False,
            f"T9 FAIL: pACS arithmetic error in {os.path.basename(pacs_log_path)} — "
            f"reported {reported_score} but min({dim_str}) = {expected_score}",
        )

    return (True, None)


def _verify_pre_mortem_substance(content):
    """H5: PA4a/PA4b — Verify Pre-mortem section has substantive content.

    P1 Compliance: Deterministic regex + line counting.
    SOT Compliance: Read-only — operates on content string.

    Checks:
      PA4a: Pre-mortem section has ≥ 3 substantive lines (not headers/blank)
      PA4b: No generic dismissal patterns (e.g. "nothing wrong", "all correct")

    Returns: list of warning strings.
    """
    warnings = []

    # Extract pre-mortem section content
    pm_match = re.search(
        r"(?:Pre-mortem|사전 부검|pre.mortem|프리모템|what could go wrong)"
        r".*?\n(.*?)(?=^#+\s|\Z)",
        content, re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if not pm_match:
        return warnings  # PA4 handles missing section

    pm_text = pm_match.group(1).strip()

    # PA4a: Minimum substantive lines (≥ 3 non-empty, non-header lines)
    lines = [
        line for line in pm_text.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) < 3:
        warnings.append(
            f"PA4a WARN: Pre-mortem has {len(lines)} substantive line(s), "
            f"expected ≥ 3. Provide specific risk analysis."
        )

    # PA4b: Generic dismissal pattern detection
    _GENERIC_DISMISSALS = [
        r"nothing.{0,15}(?:wrong|bad|incorrect)",
        r"no\s+(?:issue|problem|risk|concern)s?(?:\s+(?:found|detected|identified))?",
        r"everything.{0,10}(?:fine|good|perfect|ok|correct)",
        r"all.{0,10}(?:correct|good|fine|proper)",
        r"no\s+(?:significant|major|critical)\s+(?:issue|problem|risk)",
    ]
    for pat in _GENERIC_DISMISSALS:
        match = re.search(pat, pm_text, re.IGNORECASE)
        if match:
            warnings.append(
                f"PA4b WARN: Pre-mortem contains generic dismissal: "
                f"'{match.group()[:50]}'. Provide specific, actionable risks."
            )
            break  # One warning is sufficient

    return warnings


def validate_pacs_output(project_dir, step_number, pacs_type="general"):
    """PA1-PA7: pACS log structural integrity + arithmetic + RED threshold.

    P1 Compliance: All validation is deterministic (regex + arithmetic).
    SOT Compliance: Read-only — no file writes.

    Checks:
      PA1: pACS log file exists
      PA2: Minimum file size (≥ 50 bytes — pACS logs are concise)
      PA3: Dimension scores present (F/C/L or Ft/Ct/Nt, each 0-100)
      PA4: Pre-mortem section present (mandatory before scoring)
      PA5: pACS = min(dimensions) arithmetic correctness (delegates to verify_pacs_arithmetic)
      PA7: RED threshold — pACS < 50 blocks step advancement (FAIL)

    Optional:
      PA6: Color zone validation — score vs declared zone (RED/YELLOW/GREEN)

    Args:
        project_dir: Absolute path to project root
        step_number: Workflow step number
        pacs_type: "general" | "translation" | "review"
                   Determines expected file name pattern

    Returns:
        tuple: (is_valid: bool, warnings: list[str])
    """
    warnings = []

    # Determine file path based on type
    if pacs_type == "translation":
        pacs_filename = f"step-{step_number}-translation-pacs.md"
    elif pacs_type == "review":
        pacs_filename = f"step-{step_number}-review-pacs.md"
    else:
        pacs_filename = f"step-{step_number}-pacs.md"

    pacs_path = os.path.join(project_dir, "pacs-logs", pacs_filename)

    # PA1: File exists
    if not os.path.exists(pacs_path):
        return (False, [f"PA1 FAIL: pACS log not found: {pacs_filename}"])

    try:
        with open(pacs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return (False, [f"PA1 FAIL: Cannot read {pacs_filename}: {e}"])

    # PA2: Minimum size
    if len(content.strip()) < 50:
        warnings.append(f"PA2 FAIL: {pacs_filename} too small ({len(content)} bytes, min 50)")

    # PA3: Dimension scores present (0-100 range)
    dims_found = {}
    for match in _PACS_DIM_UNIVERSAL_RE.finditer(content):
        dim_name = match.group(1)
        dim_score = int(match.group(2))
        if 0 <= dim_score <= 100:
            dims_found[dim_name] = dim_score

    if len(dims_found) < 3:
        warnings.append(
            f"PA3 FAIL: Expected ≥ 3 dimension scores, found {len(dims_found)}: "
            f"{', '.join(f'{k}={v}' for k, v in dims_found.items()) or 'none'}"
        )
    else:
        # PA6 (optional): Color zone validation
        reported_pacs = None
        min_match = _PACS_WITH_MIN_RE.search(content)
        if min_match:
            reported_pacs = int(min_match.group(1))
        else:
            simple_matches = _PACS_SIMPLE_RE.findall(content)
            if len(simple_matches) == 1:
                reported_pacs = int(simple_matches[0])

        if reported_pacs is not None:
            # PA7: RED threshold — score < 50 blocks step advancement
            if reported_pacs < 50:
                warnings.append(
                    f"PA7 FAIL: pACS={reported_pacs} (RED zone, < 50) — "
                    f"rework required before step advancement"
                )

            # PA6 (optional): Check zone consistency
            content_upper = content.upper()
            if reported_pacs < 50 and "GREEN" in content_upper:
                warnings.append(
                    f"PA6 WARN: pACS={reported_pacs} (RED zone) but GREEN declared"
                )
            elif reported_pacs >= 70 and "RED" in content_upper:
                warnings.append(
                    f"PA6 WARN: pACS={reported_pacs} (GREEN zone) but RED declared"
                )

    # PA4: Pre-mortem section present
    _pre_mortem_patterns = [
        "pre-mortem", "Pre-mortem", "Pre-Mortem", "PRE-MORTEM",
        "사전 부검", "pre mortem", "프리모템",
        "what could go wrong", "약점", "weakness", "risk",
    ]
    has_pre_mortem = any(p in content for p in _pre_mortem_patterns)
    if not has_pre_mortem:
        warnings.append(
            "PA4 FAIL: Pre-mortem section not found — mandatory before pACS scoring"
        )
    else:
        # H5: PA4a/PA4b — Pre-mortem substance validation (P1 deterministic)
        pm_warnings = _verify_pre_mortem_substance(content)
        warnings.extend(pm_warnings)

    # PA5: Arithmetic correctness (delegates to verify_pacs_arithmetic)
    arith_valid, arith_warning = verify_pacs_arithmetic(pacs_path)
    if not arith_valid and arith_warning:
        warnings.append(arith_warning)

    # Determine overall validity
    has_fail = any("FAIL" in w for w in warnings)
    return (not has_fail, warnings)


def validate_step_output(project_dir, step_number, sot_data=None):
    """L0 Anti-Skip Guard: Validate step output file exists and meets minimum size.

    P1 Compliance: Deterministic file system checks only.
    SOT Compliance: Read-only — no file writes.

    Called by Orchestrator before advancing current_step.
    This is the code implementation of L0 Anti-Skip Guard,
    previously only a design-level checklist item.

    Checks:
      L0a: Output file exists (path from SOT outputs.step-N)
      L0b: File size ≥ MIN_OUTPUT_SIZE (100 bytes)
      L0c: File is not all whitespace

    Args:
        project_dir: Absolute path to project root
        step_number: Workflow step number to validate
        sot_data: Parsed SOT dict or read_autopilot_state() result (optional).
                  If None, reads SOT from disk.
                  Supports three data shapes:
                    1. read_autopilot_state() result: {"outputs": {"step-1": "path"}, ...}
                    2. Raw SOT (AGENTS.md schema): {"workflow": {"outputs": {"step-1": "path"}}}
                    3. Raw SOT (flat schema): {"outputs": {"step-1": "path"}}

    Returns:
        tuple: (is_valid: bool, warnings: list[str])
    """
    warnings = []

    # Load SOT if not provided
    if sot_data is None:
        for sot_path in sot_paths(project_dir):
            if os.path.exists(sot_path):
                try:
                    import yaml
                    with open(sot_path, "r", encoding="utf-8") as f:
                        sot_data = yaml.safe_load(f) or {}
                    break
                except Exception:
                    pass
        if sot_data is None:
            return (False, ["L0 FAIL: SOT file not found — cannot determine output path"])

    # FIX-R2: Extract outputs — handle both flat and nested SOT schemas
    # Shape 1 (read_autopilot_state / flat): {"outputs": {"step-1": "path"}}
    # Shape 2 (raw YAML nested): {"workflow": {"outputs": {"step-1": "path"}}}
    outputs = sot_data.get("outputs", {})
    if not outputs and isinstance(sot_data.get("workflow"), dict):
        outputs = sot_data["workflow"].get("outputs", {})
    step_key = f"step-{step_number}"
    output_path_raw = outputs.get(step_key)

    if not output_path_raw:
        return (False, [f"L0a FAIL: No output path in SOT outputs.{step_key}"])

    # Resolve relative path
    output_path = os.path.join(project_dir, output_path_raw)

    # L0a: File exists
    if not os.path.exists(output_path):
        return (False, [f"L0a FAIL: Output file not found: {output_path_raw}"])

    # L0b: Minimum size
    try:
        file_size = os.path.getsize(output_path)
    except OSError as e:
        return (False, [f"L0a FAIL: Cannot stat output file: {e}"])

    if file_size < MIN_OUTPUT_SIZE:
        warnings.append(
            f"L0b FAIL: Output file too small ({file_size} bytes, min {MIN_OUTPUT_SIZE}): "
            f"{output_path_raw}"
        )

    # L0c: Not all whitespace
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read(MIN_OUTPUT_SIZE + 10)
        if not content.strip():
            warnings.append(f"L0c FAIL: Output file is empty/whitespace-only: {output_path_raw}")
    except (IOError, UnicodeDecodeError):
        pass  # Binary files are OK (e.g., images)

    has_fail = any("FAIL" in w for w in warnings)
    return (not has_fail, warnings)


def validate_verification_log(project_dir, step_number):
    """V1: Verification log structural integrity (V1a-V1e).

    P1 Compliance: Filesystem + regex — deterministic.
    SOT Compliance: Read-only access to verification-logs/.

    Checks:
      V1a: verification-logs/step-{N}-verify.md exists + size >= MIN_OUTPUT_SIZE
      V1b: Each criterion has explicit PASS/FAIL marking (checklist or table)
      V1c: Logical consistency — if any criterion is FAIL, overall must be FAIL
      V1d: Evidence quality — each criterion's evidence >= 20 chars (EVP)
      V1e: Compound criterion detection — warn if criteria bundle multiple actions (EVP)

    Args:
        project_dir: Project root directory path
        step_number: Step number (int) to validate

    Returns:
        tuple: (is_valid: bool, warnings: list[str])
        - is_valid: True only if V1a-V1e pass (V1e is WARNING, does not fail)
    """
    warnings = []
    verify_path = os.path.join(
        project_dir, "verification-logs", f"step-{step_number}-verify.md"
    )

    # V1a: File existence + minimum size
    if not os.path.exists(verify_path):
        warnings.append(
            f"V1a FAIL: verification-logs/step-{step_number}-verify.md not found"
        )
        return (False, warnings)

    try:
        size = os.path.getsize(verify_path)
    except OSError:
        warnings.append(f"V1a FAIL: Cannot read file size: {verify_path}")
        return (False, warnings)

    if size < MIN_OUTPUT_SIZE:
        warnings.append(
            f"V1a FAIL: Verification log too small "
            f"({size} bytes, min {MIN_OUTPUT_SIZE})"
        )

    # Read content for V1b-V1c
    try:
        with open(verify_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        warnings.append(f"V1a FAIL: Cannot read file: {e}")
        return (False, warnings)

    # V1b: Extract per-criterion PASS/FAIL results
    criteria = []
    # Try checklist format: "- [x] Criterion: PASS"
    for match in _VERIFY_CRITERION_CHECKLIST_RE.finditer(content):
        criteria.append({
            "name": match.group(1).strip(),
            "result": match.group(2).upper(),
        })
    # Also try table format: "| Criterion | PASS |"
    for match in _VERIFY_CRITERION_TABLE_RE.finditer(content):
        name = match.group(1).strip()
        # Skip table header and separator rows
        if name.lower().rstrip("-") in _VERIFY_TABLE_HEADER_WORDS:
            continue
        if name.startswith("-"):
            continue
        criteria.append({
            "name": name,
            "result": match.group(2).upper(),
        })

    if not criteria:
        warnings.append(
            "V1b FAIL: No per-criterion PASS/FAIL results found "
            "(expected checklist or table format)"
        )

    # V1c: Logical consistency — any FAIL criterion → overall must be FAIL
    has_individual_fail = any(c["result"] == "FAIL" for c in criteria)
    overall_match = _VERIFY_OVERALL_RE.search(content)
    if overall_match:
        overall_result = overall_match.group(1).upper()
        if has_individual_fail and overall_result == "PASS":
            failed_names = [c["name"] for c in criteria if c["result"] == "FAIL"]
            warnings.append(
                f"V1c FAIL: Logical inconsistency — overall is PASS but "
                f"these criteria are FAIL: {', '.join(failed_names)}"
            )
    elif criteria:
        warnings.append(
            "V1c FAIL: No overall PASS/FAIL result found in verification log"
        )

    # V1d: Evidence quality — each criterion must have substantive evidence (≥ 20 chars)
    # Extract evidence from table format (3rd column after PASS/FAIL)
    evidence_map = {}
    for match in _VERIFY_CRITERION_TABLE_EVIDENCE_RE.finditer(content):
        name = match.group(1).strip()
        if name.lower().rstrip("-") in _VERIFY_TABLE_HEADER_WORDS:
            continue
        if name.startswith("-"):
            continue
        evidence = match.group(3).strip()
        # Unescape pipe characters for accurate length measurement
        evidence = evidence.replace("\\|", "|")
        evidence_map[name] = evidence

    _MIN_EVIDENCE_LEN = 20
    for c in criteria:
        cname = c["name"]
        ev = evidence_map.get(cname, "")
        if ev and len(ev) < _MIN_EVIDENCE_LEN:
            warnings.append(
                f"V1d FAIL: Criterion '{cname}' has insufficient evidence "
                f"({len(ev)} chars, min {_MIN_EVIDENCE_LEN}): \"{ev}\""
            )

    # V1e: Compound criterion detection — warn if a single criterion bundles
    # multiple independent verifiable actions (conjunction pattern)
    for c in criteria:
        if _VERIFY_COMPOUND_CRITERION_RE.search(c["name"]):
            warnings.append(
                f"V1e WARNING: Criterion '{c['name']}' may bundle multiple "
                f"actions — consider splitting into atomic criteria (EVP-1)"
            )

    # V1e is WARNING-only (does not invalidate); V1a-V1d are FAIL
    is_valid = not any("FAIL" in w for w in warnings)
    return (is_valid, warnings)


def generate_verification_log(step_number, criteria_results, overall=None):
    """Generate a V1a-V1e compliant verification log (deterministic).

    P1 Compliance: Pure string formatting — no LLM interpretation.
    Called by Orchestrator at E5.3 to produce verification-logs/step-N-verify.md.
    V1d Note: Each evidence string should be >= 20 chars to pass V1d quality check.

    Args:
        step_number: Step number (int).
        criteria_results: List of dicts, each with keys:
            - "criterion": str (e.g., "L0: Output exists and non-empty")
            - "result": "PASS" or "FAIL"
            - "evidence": str (e.g., "wave-results/wave-1/literature-search.md, 4523 bytes")
        overall: "PASS" or "FAIL" or None (auto-derived: FAIL if any criterion FAIL).

    Returns:
        str: Complete markdown content for verification-logs/step-N-verify.md.
    """
    if not criteria_results:
        criteria_results = []

    # Auto-derive overall from criteria
    if overall is None:
        has_fail = any(
            c.get("result", "").upper() == "FAIL" for c in criteria_results
        )
        overall = "FAIL" if has_fail else "PASS"

    lines = [
        f"# Verification Log — Step {step_number}",
        "",
        "## Criteria Results",
        "",
        "| Criterion | Result | Evidence |",
        "|-----------|--------|----------|",
    ]

    for c in criteria_results:
        criterion = c.get("criterion", "Unknown")
        result = c.get("result", "UNKNOWN").upper()
        evidence = c.get("evidence", "—")
        # Escape pipe characters in values to prevent table breakage
        criterion = criterion.replace("|", "\\|")
        evidence = evidence.replace("|", "\\|")
        lines.append(f"| {criterion} | {result} | {evidence} |")

    lines.append("")
    lines.append(f"## Overall Result: {overall}")
    lines.append("")

    return "\n".join(lines)


_WORKFLOW_INHERITED_DNA_RE = re.compile(
    r"^##\s+Inherited DNA", re.MULTILINE
)


_WORKFLOW_INHERITED_TABLE_RE = re.compile(
    r"Inherited Patterns[^\n]*\n(?:\s*\n)?"  # header line + optional blank
    r"(\|[^\n]+\n)"                           # table header row
    r"(\|[-| :]+\n)"                          # separator row
    r"((?:\|[^\n]+\n)*)",                     # data rows
    re.MULTILINE,
)


_WORKFLOW_CONSTITUTIONAL_RE = re.compile(
    r"Constitutional Principles", re.IGNORECASE
)


_WORKFLOW_CAP_RE = re.compile(
    r"CAP-[1-4]|코딩\s*기준점|Coding\s*Anchor\s*Points", re.IGNORECASE
)


_WORKFLOW_CT_VERIFICATION_RE = re.compile(
    r"교차\s*단계\s*추적성|cross[- ]?step\s*traceability|trace:step-",
    re.IGNORECASE,
)


_WORKFLOW_CT_POSTPROCESS_RE = re.compile(
    r"validate_traceability", re.IGNORECASE,
)


_WORKFLOW_DKS_VERIFICATION_RE = re.compile(
    r"domain[- ]?knowledge|도메인\s*지식\s*구조|\[dks:|domain-knowledge\.yaml",
    re.IGNORECASE,
)


_WORKFLOW_DKS_POSTPROCESS_RE = re.compile(
    r"validate_domain_knowledge", re.IGNORECASE,
)


_WORKFLOW_ENGLISH_FIRST_RE = re.compile(
    r"English[- ]?First", re.IGNORECASE,
)


def validate_workflow_md(workflow_path):
    """W1-W9: Generated workflow.md structural integrity for DNA inheritance.

    P1 Compliance: All validation is deterministic (regex + string checks).
    SOT Compliance: Read-only — no file writes.

    Checks:
      W1: Workflow file exists and is readable
      W2: Minimum file size (≥ 500 bytes — workflow files are substantial)
      W3: '## Inherited DNA' header present
      W4: Inherited Patterns table present (≥ 3 data rows)
      W5: Constitutional Principles section present
      W6: Coding Anchor Points (CAP) reference present
      W7: If Verification mentions cross-step traceability, validate_traceability
          post-processing must be present (Verification-Validator consistency)
      W8: If workflow references domain knowledge, validate_domain_knowledge
          post-processing must be present (Verification-Validator consistency)
      W9: English-First Execution pattern present in Inherited DNA section

    Args:
        workflow_path: Absolute path to generated workflow.md

    Returns:
        tuple: (is_valid: bool, warnings: list[str])
    """
    warnings = []

    # W1: File exists
    if not os.path.exists(workflow_path):
        return (False, [f"W1 FAIL: Workflow file not found: {workflow_path}"])

    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return (False, [f"W1 FAIL: Cannot read workflow: {e}"])

    # W2: Minimum size
    if len(content.strip()) < 500:
        warnings.append(
            f"W2 FAIL: Workflow too small ({len(content)} bytes, min 500)"
        )

    # W3: Inherited DNA header
    if not _WORKFLOW_INHERITED_DNA_RE.search(content):
        warnings.append(
            "W3 FAIL: '## Inherited DNA' section not found in workflow"
        )

    # W4: Inherited Patterns table (≥ 3 data rows)
    table_match = _WORKFLOW_INHERITED_TABLE_RE.search(content)
    if table_match:
        data_rows_text = table_match.group(3)
        data_rows = [
            line for line in data_rows_text.split("\n")
            if line.strip().startswith("|")
        ]
        if len(data_rows) < 3:
            warnings.append(
                f"W4 FAIL: Inherited Patterns table has {len(data_rows)} "
                f"data rows, expected ≥ 3"
            )
    else:
        warnings.append("W4 FAIL: Inherited Patterns table not found")

    # W5: Constitutional Principles
    if not _WORKFLOW_CONSTITUTIONAL_RE.search(content):
        warnings.append(
            "W5 FAIL: Constitutional Principles section not found"
        )

    # W6: Coding Anchor Points (CAP) reference
    if not _WORKFLOW_CAP_RE.search(content):
        warnings.append(
            "W6 FAIL: Coding Anchor Points (CAP) reference not found"
        )

    # W7: Verification-Validator consistency — Cross-Step Traceability
    # If any Verification criteria mentions traceability, the workflow must
    # include validate_traceability post-processing to enforce P1 validation.
    if _WORKFLOW_CT_VERIFICATION_RE.search(content):
        if not _WORKFLOW_CT_POSTPROCESS_RE.search(content):
            warnings.append(
                "W7 FAIL: Workflow references cross-step traceability in "
                "Verification criteria but has no validate_traceability.py "
                "Post-processing command"
            )

    # W8: Verification-Validator consistency — Domain Knowledge Structure
    # If the workflow references domain knowledge (DKS), the workflow must
    # include validate_domain_knowledge post-processing.
    if _WORKFLOW_DKS_VERIFICATION_RE.search(content):
        if not _WORKFLOW_DKS_POSTPROCESS_RE.search(content):
            warnings.append(
                "W8 FAIL: Workflow references domain knowledge structure but "
                "has no validate_domain_knowledge.py Post-processing command"
            )

    # W9: English-First Execution DNA — MANDATORY presence
    # English-First is an expression of Quality Absolutism (absolute criterion 1)
    # applied to language choice. All child workflows must inherit this pattern.
    if not _WORKFLOW_ENGLISH_FIRST_RE.search(content):
        warnings.append(
            "W9 FAIL: English-First Execution pattern not found in workflow. "
            "Add 'English-First Execution' row to Inherited Patterns table"
        )

    has_fail = any("FAIL" in w for w in warnings)
    return (not has_fail, warnings)


_TRACE_MARKER_RE = re.compile(
    r'\[trace:step-(\d+):([a-z0-9_-]+)(?::([a-z0-9_-]+))?\]',
    re.IGNORECASE,
)


_HEADING_SLUG_RE = re.compile(r'^#+\s+(.+)$', re.MULTILINE)


_MIN_TRACE_MARKERS = 3


def validate_cross_step_traceability(project_dir, step_number, sot_data=None):
    """CT1-CT5: Cross-step traceability structural integrity.

    P1 Compliance: Filesystem + regex — deterministic.
    SOT Compliance: Read-only — reads SOT outputs for path resolution.

    Validates that a step's output contains trace markers referencing
    previous steps, enabling horizontal (cross-step) verification.

    Checks:
      CT1: Trace markers exist in output (>= 1)
      CT2: Referenced step outputs exist on disk (SOT outputs.step-N path)
      CT3: Section IDs resolve to headings in source files (Warning only)
      CT4: Minimum trace marker density (>= MIN_TRACE_MARKERS)
      CT5: No forward references (step-N where N >= current step)

    Args:
        project_dir: Absolute path to project root
        step_number: Current step number being validated
        sot_data: Parsed SOT dict (optional). If None, reads from disk.

    Returns:
        tuple: (is_valid: bool, warnings: list[str])
    """
    warnings = []
    trace_count = 0
    verified_count = 0

    # Load SOT if not provided
    if sot_data is None:
        for sp in sot_paths(project_dir):
            if os.path.exists(sp):
                try:
                    import yaml
                    with open(sp, "r", encoding="utf-8") as f:
                        sot_data = yaml.safe_load(f) or {}
                    break
                except Exception:
                    pass
        if sot_data is None:
            return (False, ["CT FAIL: SOT file not found — cannot resolve output paths"])

    # Extract outputs from SOT (handle nested and flat schemas)
    outputs = sot_data.get("outputs", {})
    if not outputs and isinstance(sot_data.get("workflow"), dict):
        outputs = sot_data["workflow"].get("outputs", {})

    # Step 1 has no previous steps — traceability N/A
    if step_number <= 1:
        return (True, ["CT SKIP: Step 1 has no previous steps — traceability N/A"])

    # Get current step output path
    step_key = f"step-{step_number}"
    output_path_raw = outputs.get(step_key)
    if not output_path_raw:
        return (False, [f"CT FAIL: No output path in SOT outputs.{step_key}"])

    output_path = os.path.join(project_dir, output_path_raw)
    if not os.path.exists(output_path):
        return (False, [f"CT FAIL: Output file not found: {output_path_raw}"])

    # Read output content
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return (False, [f"CT FAIL: Cannot read output file: {e}"])

    # Extract all trace markers
    markers = _TRACE_MARKER_RE.findall(content)
    trace_count = len(markers)

    # CT1: At least one trace marker must exist
    if trace_count == 0:
        warnings.append("CT1 FAIL: No [trace:step-N:...] markers found in output")
        return (False, warnings)

    # CT4: Minimum density
    if trace_count < _MIN_TRACE_MARKERS:
        warnings.append(
            f"CT4 FAIL: Only {trace_count} trace markers found, "
            f"minimum {_MIN_TRACE_MARKERS} required"
        )

    # Validate each marker
    for ref_step_str, section_id, locator in markers:
        ref_step = int(ref_step_str)

        # CT5: No forward references
        if ref_step >= step_number:
            warnings.append(
                f"CT5 FAIL: Forward reference [trace:step-{ref_step}:...] "
                f"in step {step_number} — must reference earlier steps only"
            )
            continue

        # CT2: Referenced step output exists
        ref_step_key = f"step-{ref_step}"
        ref_output_raw = outputs.get(ref_step_key)
        if not ref_output_raw:
            warnings.append(
                f"CT2 FAIL: Referenced step-{ref_step} has no output in SOT"
            )
            continue

        ref_output_path = os.path.join(project_dir, ref_output_raw)
        if not os.path.exists(ref_output_path):
            warnings.append(
                f"CT2 FAIL: Referenced output file not found: {ref_output_raw}"
            )
            continue

        # CT3: Section ID resolution (Warning only, not FAIL)
        try:
            with open(ref_output_path, "r", encoding="utf-8") as f:
                ref_content = f.read()
            headings = _HEADING_SLUG_RE.findall(ref_content)
            slugified = [_slugify_heading(h) for h in headings]
            if section_id.lower() not in slugified:
                warnings.append(
                    f"CT3 WARNING: Section '{section_id}' not resolved in "
                    f"step-{ref_step} headings (may be a sub-section or ID)"
                )
            else:
                verified_count += 1
        except (IOError, UnicodeDecodeError):
            verified_count += 1  # Can't read = trust the reference

    # Append metadata as info (not FAIL)
    warnings.append(
        f"CT INFO: trace_count={trace_count}, verified_count={verified_count}"
    )

    has_fail = any("FAIL" in w for w in warnings)
    return (not has_fail, warnings)


def _slugify_heading(heading_text):
    """Convert a markdown heading to a slug for section-id matching.

    Matches the trace marker convention: lowercase, alphanumeric + hyphens.
    Strips markdown artifacts: links [text](url) → text, bold/italic markers,
    backticks, and other non-alphanumeric characters.
    """
    slug = heading_text.strip()
    # Remove markdown links: [text](url) → text
    slug = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', slug)
    # Remove inline code backticks
    slug = re.sub(r'`([^`]*)`', r'\1', slug)
    slug = slug.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s]+', '-', slug)
    slug = slug.strip('-')
    return slug


_DKS_REF_RE = re.compile(r'\[dks:([a-z0-9_-]+)\]', re.IGNORECASE)


_DKS_ID_RE = re.compile(r'^[a-z][a-z0-9_-]*$')


def validate_domain_knowledge(project_dir, check_output_step=None, sot_data=None):
    """DK1-DK7: Domain Knowledge Structure structural integrity.

    P1 Compliance: Filesystem + YAML parse + regex — deterministic.
    SOT Compliance: Read-only — no file writes.

    Validates domain-knowledge.yaml schema and optionally cross-references
    with step output DKS markers.

    Checks:
      DK1: File exists and YAML is valid
      DK2: metadata contains required keys (domain, schema_version)
      DK3: entities structure (id unique + slug format, type string, attributes dict)
      DK4: relations referential integrity (subject/object -> entities.id, confidence valid)
      DK5: constraints structure (id, description, check present)
      DK6: (--check-output) Output DKS markers resolve to entity/relation IDs
      DK7: (--check-output) Constraint non-violation (best-effort numeric check)

    Args:
        project_dir: Absolute path to project root
        check_output_step: Step number to cross-check DKS markers (optional)
        sot_data: Parsed SOT dict (optional). If None, reads from disk.

    Returns:
        tuple: (is_valid: bool, warnings: list[str])
    """
    warnings = []

    dk_path = os.path.join(project_dir, "domain-knowledge.yaml")

    # DK1: File exists and YAML is valid
    if not os.path.exists(dk_path):
        return (False, ["DK1 FAIL: domain-knowledge.yaml not found"])

    try:
        import yaml
        with open(dk_path, "r", encoding="utf-8") as f:
            dk_data = yaml.safe_load(f)
        if not isinstance(dk_data, dict):
            return (False, ["DK1 FAIL: domain-knowledge.yaml is not a valid YAML mapping"])
    except Exception as e:
        return (False, [f"DK1 FAIL: Cannot parse domain-knowledge.yaml: {e}"])

    # DK2: metadata required keys
    metadata = dk_data.get("metadata", {})
    if not isinstance(metadata, dict):
        warnings.append("DK2 FAIL: 'metadata' must be a mapping")
    else:
        for key in ("domain", "schema_version"):
            if key not in metadata:
                warnings.append(f"DK2 FAIL: metadata.{key} is missing")

    # DK3: entities structure
    entities = dk_data.get("entities", [])
    entity_ids = set()
    if not isinstance(entities, list):
        warnings.append("DK3 FAIL: 'entities' must be a list")
        entities = []

    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            warnings.append(f"DK3 FAIL: entities[{i}] is not a mapping")
            continue
        eid = entity.get("id")
        if not eid:
            warnings.append(f"DK3 FAIL: entities[{i}] missing 'id'")
        elif not _DKS_ID_RE.match(str(eid)):
            warnings.append(
                f"DK3 FAIL: entities[{i}].id '{eid}' is not valid slug format "
                f"(lowercase letter start, alphanumeric + hyphens)"
            )
        else:
            if eid in entity_ids:
                warnings.append(f"DK3 FAIL: Duplicate entity id '{eid}'")
            entity_ids.add(eid)

        if not isinstance(entity.get("type"), str):
            warnings.append(f"DK3 FAIL: entities[{i}].type must be a string")
        if not isinstance(entity.get("attributes", {}), dict):
            warnings.append(f"DK3 FAIL: entities[{i}].attributes must be a mapping")

    # DK4: relations referential integrity
    relations = dk_data.get("relations", [])
    relation_ids = set()
    valid_confidences = {"high", "medium", "low"}
    if not isinstance(relations, list):
        if relations is not None:
            warnings.append("DK4 FAIL: 'relations' must be a list")
        relations = []

    for i, rel in enumerate(relations):
        if not isinstance(rel, dict):
            warnings.append(f"DK4 FAIL: relations[{i}] is not a mapping")
            continue
        rid = rel.get("id")
        if rid:
            if rid in relation_ids:
                warnings.append(f"DK4 FAIL: Duplicate relation id '{rid}'")
            relation_ids.add(rid)

        subj = rel.get("subject")
        obj = rel.get("object")
        if subj and subj not in entity_ids:
            warnings.append(
                f"DK4 FAIL: relations[{i}].subject '{subj}' not found in entities"
            )
        if obj and obj not in entity_ids:
            warnings.append(
                f"DK4 FAIL: relations[{i}].object '{obj}' not found in entities"
            )
        conf = rel.get("confidence")
        if conf and str(conf).lower() not in valid_confidences:
            warnings.append(
                f"DK4 FAIL: relations[{i}].confidence '{conf}' must be "
                f"high|medium|low"
            )

    # DK5: constraints structure
    constraints = dk_data.get("constraints", [])
    if not isinstance(constraints, list):
        if constraints is not None:
            warnings.append("DK5 FAIL: 'constraints' must be a list")
        constraints = []

    for i, con in enumerate(constraints):
        if not isinstance(con, dict):
            warnings.append(f"DK5 FAIL: constraints[{i}] is not a mapping")
            continue
        for key in ("id", "description", "check"):
            if key not in con:
                warnings.append(f"DK5 FAIL: constraints[{i}] missing '{key}'")

    # DK6 + DK7: Output cross-check (optional)
    if check_output_step is not None:
        _validate_dks_output_refs(
            project_dir, check_output_step, entity_ids, relation_ids,
            constraints, dk_data, sot_data, warnings,
        )

    # Summary info
    warnings.append(
        f"DK INFO: entity_count={len(entity_ids)}, "
        f"relation_count={len(relation_ids)}, "
        f"constraint_count={len(constraints)}"
    )

    has_fail = any("FAIL" in w for w in warnings)
    return (not has_fail, warnings)


def _validate_dks_output_refs(
    project_dir, step_number, entity_ids, relation_ids,
    constraints, dk_data, sot_data, warnings,
):
    """DK6-DK7: Cross-check DKS references in step output.

    DK6: All [dks:xxx] markers resolve to entity or relation IDs.
    DK7: Best-effort numeric constraint validation.
    """
    # Load SOT if not provided
    if sot_data is None:
        for sp in sot_paths(project_dir):
            if os.path.exists(sp):
                try:
                    import yaml
                    with open(sp, "r", encoding="utf-8") as f:
                        sot_data = yaml.safe_load(f) or {}
                    break
                except Exception:
                    pass
        if sot_data is None:
            warnings.append("DK6 SKIP: SOT file not found — cannot resolve output path")
            return

    # Extract outputs from SOT
    outputs = sot_data.get("outputs", {})
    if not outputs and isinstance(sot_data.get("workflow"), dict):
        outputs = sot_data["workflow"].get("outputs", {})

    step_key = f"step-{step_number}"
    output_path_raw = outputs.get(step_key)
    if not output_path_raw:
        warnings.append(f"DK6 SKIP: No output path in SOT outputs.{step_key}")
        return

    output_path = os.path.join(project_dir, output_path_raw)
    if not os.path.exists(output_path):
        warnings.append(f"DK6 SKIP: Output file not found: {output_path_raw}")
        return

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        warnings.append(f"DK6 SKIP: Cannot read output: {e}")
        return

    # DK6: Resolve DKS markers
    all_ids = entity_ids | relation_ids
    dks_refs = _DKS_REF_RE.findall(content)
    unresolved = []
    for ref_id in dks_refs:
        if ref_id.lower() not in {eid.lower() for eid in all_ids}:
            unresolved.append(ref_id)

    if unresolved:
        warnings.append(
            f"DK6 FAIL: Unresolved DKS references: {', '.join(unresolved)}"
        )

    # DK7: Best-effort constraint validation (numeric sum checks)
    entities_list = dk_data.get("entities", []) if dk_data else []
    for con in constraints:
        check_str = str(con.get("check", ""))
        sum_match = re.match(
            r'sum\((\w+)\)\s*<=\s*(\d+(?:\.\d+)?)', check_str
        )
        if sum_match:
            field_name = sum_match.group(1)
            max_val = float(sum_match.group(2))
            total = 0.0
            found_any = False
            for entity in entities_list:
                attrs = entity.get("attributes", {})
                if field_name in attrs:
                    try:
                        val_str = str(attrs[field_name]).replace("%", "").replace("$", "")
                        total += float(val_str)
                        found_any = True
                    except (ValueError, TypeError):
                        pass
            if found_any and total > max_val:
                warnings.append(
                    f"DK7 FAIL: Constraint '{con.get('id', '?')}' violated: "
                    f"sum({field_name})={total} > {max_val}"
                )
