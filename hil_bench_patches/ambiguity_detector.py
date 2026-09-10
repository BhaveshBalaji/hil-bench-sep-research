"""
ambiguity_detector.py
---------------------
Pre-run LLM classifier that reads a task's problem statement and database
name, identifies business ambiguities that cannot be resolved from schema
alone, and pre-populates the per-task ledger before the agent starts.

This solves FM1 (silent omission): even if the agent never lists ambiguities
itself, the ledger is already populated and the gateway will block submission
until each critical ambiguity is resolved.

Usage (called by the harness before each task):
    from hil_bench.ambiguity_detector import detect_and_populate

    detect_and_populate(
        task_id="public_sql_18",
        problem_statement="Find all senior account owners...",
        database_name="financial",
        ledger_base_dir="/scratch/.../ledgers",
        model="openrouter/meta-llama/llama-3.3-70b-instruct",
        api_key="sk-or-v1-...",
    )

The function is idempotent: if the ledger already exists and has ambiguities,
it skips re-detection to avoid overwriting agent-discovered entries.

Environment variables (used when not passed explicitly):
    LEDGER_BASE_PATH          — base directory for ledger files
    LITELLM_API_KEY           — API key for the detector LLM
    DETECTOR_MODEL            — model string (default: openrouter/meta-llama/llama-3.3-70b-instruct)
    DETECTOR_MAX_AMBIGUITIES  — max ambiguities to detect per task (default: 8)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL        = "openrouter/meta-llama/llama-3.3-70b-instruct"
DEFAULT_MAX_AMBIG    = 8
DEFAULT_TEMPERATURE  = 0.1   # low temperature for consistent structured output


# ── Detector prompt ───────────────────────────────────────────────────────────

DETECTOR_PROMPT = """You are an expert SQL analyst reviewing a database task for ambiguities.

DATABASE: {database_name}
TASK: {problem_statement}

Your job is to identify ALL business ambiguities in this task — terms, conditions, metrics, or thresholds that:
1. Cannot be resolved by inspecting table/column names alone
2. Require a human business expert to define (e.g. "What counts as senior?", "What period is 'recent'?")
3. Could lead to different SQL queries depending on the interpretation

DO NOT include:
- Questions answerable by schema inspection (e.g. "which column stores age?")
- Standard SQL concepts or functions
- Ambiguities about database structure

For each ambiguity, provide:
- id: short alphanumeric ID (A1, A2, ...)
- description: one sentence describing what is ambiguous
- type: one of business_definition | threshold | metric | time_period | other
- critical: true if the query result would change depending on interpretation, false otherwise

Respond with ONLY a JSON array. No explanation. No markdown. Example:
[
  {{"id": "A1", "description": "What age threshold defines a 'senior' account owner?", "type": "threshold", "critical": true}},
  {{"id": "A2", "description": "What time period defines the 'crisis period'?", "type": "time_period", "critical": true}}
]

If there are no ambiguities, respond with an empty array: []
Limit to at most {max_ambiguities} ambiguities. Prioritize critical ones."""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_detector_llm(
    prompt: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
) -> str:
    """Call the LLM and return the raw response text."""
    try:
        import litellm
        kwargs: dict = {
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens":  1024,
            "api_key":     api_key,
        }
        if base_url:
            kwargs["api_base"] = base_url
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Detector LLM call failed: {e}")
        raise


def _parse_ambiguities(raw: str) -> list[dict]:
    """Parse LLM JSON output, handling markdown fences if present."""
    # Strip markdown code fences
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    # Find first [ ... ] block
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        parsed = json.loads(raw[start:end + 1])
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


def _build_entries(raw_ambiguities: list[dict]) -> list:
    """Convert raw dicts to AmbiguityEntry objects, skipping malformed entries."""
    from hil_bench.ambiguity_ledger import AmbiguityEntry, AmbiguityStatus

    entries = []
    seen_ids: set[str] = set()

    for i, item in enumerate(raw_ambiguities):
        if not isinstance(item, dict):
            continue

        # Generate ID if missing
        aid = str(item.get("id", f"A{i+1}")).strip()
        if not aid:
            aid = f"A{i+1}"

        # Deduplicate
        if aid in seen_ids:
            aid = f"{aid}_{i}"
        seen_ids.add(aid)

        description = str(item.get("description", "")).strip()
        if not description:
            continue  # skip entries with no description

        atype    = str(item.get("type", "business_definition")).strip()
        critical = bool(item.get("critical", True))

        entries.append(AmbiguityEntry(
            id=aid,
            description=description,
            type=atype,
            critical=critical,
            status=AmbiguityStatus.UNRESOLVED,
            source="detector",
        ))

    return entries


# ── Public API ────────────────────────────────────────────────────────────────

def detect_ambiguities(
    problem_statement: str,
    database_name: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_ambiguities: int = DEFAULT_MAX_AMBIG,
) -> list:
    """
    Run the detector LLM and return a list of AmbiguityEntry objects.

    Does not read or write the ledger — that's handled by detect_and_populate.
    """
    model   = model   or os.environ.get("DETECTOR_MODEL", DEFAULT_MODEL)
    api_key = api_key or os.environ.get("DETECTOR_API_KEY") or os.environ.get("LITELLM_API_KEY", "")
    base_url = base_url or os.environ.get("DETECTOR_BASE_URL")
    max_ambiguities = int(os.environ.get("DETECTOR_MAX_AMBIGUITIES", max_ambiguities))

    if not problem_statement.strip():
        logger.warning("Detector called with empty problem statement — skipping")
        return []

    prompt = DETECTOR_PROMPT.format(
        database_name=database_name or "unknown",
        problem_statement=problem_statement.strip(),
        max_ambiguities=max_ambiguities,
    )

    try:
        raw  = _call_detector_llm(prompt, model, api_key, base_url)
        data = _parse_ambiguities(raw)
        entries = _build_entries(data[:max_ambiguities])
        logger.info(
            f"Detector found {len(entries)} ambiguities for '{database_name}': "
            + ", ".join(e.id for e in entries)
        )
        return entries
    except Exception as e:
        logger.error(f"Detector failed for task: {e}")
        raise RuntimeError("Ambiguity detector call failed") from e


def detect_and_populate(
    task_id: str,
    problem_statement: str,
    database_name: str,
    ledger_base_dir: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_ambiguities: int = DEFAULT_MAX_AMBIG,
    overwrite: bool = False,
) -> "Ledger":  # type: ignore[name-defined]
    """
    Run detection and save results to the ledger.

    Idempotent by default: if ledger already has ambiguities, skips
    re-detection (pass overwrite=True to force re-run).

    Returns the final Ledger object.
    """
    from hil_bench.ambiguity_ledger import (
        Ledger, load_or_create_ledger, save_ledger
    )

    base_dir = ledger_base_dir or os.environ.get("LEDGER_BASE_PATH")
    if not base_dir:
        raise RuntimeError(
            "LEDGER_BASE_PATH env var must be set (or pass ledger_base_dir)"
        )

    # Load or create ledger
    ledger = load_or_create_ledger(base_dir, task_id)
    ledger.problem_statement = problem_statement
    ledger.database_name     = database_name

    # Skip if already populated and not overwriting
    if ledger.ambiguities and not overwrite:
        logger.info(
            f"Ledger for {task_id} already has {len(ledger.ambiguities)} "
            "ambiguities — skipping detection"
        )
        save_ledger(base_dir, ledger)
        return ledger

    # Run detection
    entries = detect_ambiguities(
        problem_statement=problem_statement,
        database_name=database_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_ambiguities=max_ambiguities,
    )

    for entry in entries:
        ledger.add(entry)

    save_ledger(base_dir, ledger)
    logger.info(
        f"Ledger saved for {task_id}: {len(ledger.ambiguities)} ambiguities, "
        f"{len(ledger.unresolved_critical())} critical unresolved"
    )
    return ledger


def detect_batch(
    instances: list[dict],
    ledger_base_dir: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    overwrite: bool = False,
    workers: int = 4,
) -> dict[str, int]:
    """
    Run detection for a batch of SQL instances in parallel.

    instances: list of dicts with keys: id (or problem_statement.id),
               problem_statement (str or dict), env.database_name
    Returns: dict mapping task_id -> number of ambiguities detected
    """
    import concurrent.futures

    results: dict[str, int] = {}

    def _process(inst: dict) -> tuple[str, int]:
        # Extract task ID
        task_id = inst.get("_original_instance_id") or inst.get("instance_id")
        if not task_id:
            ps = inst.get("problem_statement", {})
            if isinstance(ps, dict):
                task_id = ps.get("id", "unknown")
            else:
                task_id = "unknown"

        # Extract problem statement text
        ps = inst.get("problem_statement", {})
        if isinstance(ps, dict):
            problem_text = ps.get("problem_statement", "") or ps.get("text", "")
        else:
            problem_text = str(ps)

        # Extract database name
        env = inst.get("env", {})
        db_name = ""
        if isinstance(env, dict):
            db_name = env.get("database_name", "")
            if not db_name:
                # Try to infer from db path
                db_path = env.get("base_db_path", "")
                if db_path:
                    db_name = Path(db_path).stem

        try:
            ledger = detect_and_populate(
                task_id=task_id,
                problem_statement=problem_text,
                database_name=db_name,
                ledger_base_dir=ledger_base_dir,
                model=model,
                api_key=api_key,
                overwrite=overwrite,
            )
            return task_id, len(ledger.ambiguities)
        except Exception as e:
            logger.error(f"Detection failed for {task_id}: {e}")
            raise RuntimeError(f"Detection failed for {task_id}") from e

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, inst): inst for inst in instances}
        for future in concurrent.futures.as_completed(futures):
            try:
                task_id, count = future.result()
                results[task_id] = count
            except Exception as e:
                raise RuntimeError("Ambiguity batch detection failed") from e

    return results


# Allow running as a script for testing
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print("Usage: python -m hil_bench.ambiguity_detector <task_id> <database_name> <problem_statement>")
        sys.exit(1)

    task_id_arg   = sys.argv[1]
    db_name_arg   = sys.argv[2]
    problem_arg   = " ".join(sys.argv[3:])
    base_dir_arg  = os.environ.get("LEDGER_BASE_PATH", "/tmp/ledgers")

    ledger = detect_and_populate(
        task_id=task_id_arg,
        problem_statement=problem_arg,
        database_name=db_name_arg,
        ledger_base_dir=base_dir_arg,
    )

    print(json.dumps(ledger.to_dict(), indent=2))
