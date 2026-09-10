"""
ambiguity_ledger.py
--------------------
Data model and file I/O for the per-task ambiguity ledger.

The ledger tracks every ambiguity detected in a task and its resolution status.
It is stored as a JSON file at:

    LEDGER_BASE_PATH / <original_task_id> / ledger.json

One ledger file per task per pass (pass number is part of TASK_INSTANCE_ID but
the ledger path uses only the original task ID so the detector can pre-populate
it once and the agent reads it regardless of pass number).

State machine per ambiguity:
    UNRESOLVED → ASKED → RESOLVED   (oracle returned a real answer)
                 ASKED → UNRESOLVED  (oracle said "irrelevant question")
                 ASKED → FAILED      (oracle said "can't answer")

Invariant enforced by gateway_submit_sql:
    submit allowed  ⟺  |{a : a.critical and a.status != RESOLVED}| = 0
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Status enum ──────────────────────────────────────────────────────────────

class AmbiguityStatus(str, Enum):
    UNRESOLVED = "unresolved"
    ASKED      = "asked"
    RESOLVED   = "resolved"
    FAILED     = "failed"   # oracle said "can't answer"


# ── Ambiguity entry ───────────────────────────────────────────────────────────

@dataclass
class AmbiguityEntry:
    id: str                              # e.g. "A1", "A2"
    description: str                     # natural language description
    type: str = "business_definition"    # business_definition | threshold | metric | other
    critical: bool = True                # if True, blocks submission when unresolved
    status: AmbiguityStatus = AmbiguityStatus.UNRESOLVED
    question_asked: Optional[str] = None # last question the agent asked
    resolution: Optional[str] = None    # oracle's response when resolved
    resolution_source: Optional[str] = None  # "human" | "business_info"
    evidence: list[str] = field(default_factory=list)  # supporting evidence strings
    source: str = "detector"            # "detector" | "agent"
    business_info_checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AmbiguityEntry:
        d = dict(d)
        d["status"] = AmbiguityStatus(d.get("status", "unresolved"))
        return cls(**d)


# ── Ledger ────────────────────────────────────────────────────────────────────

@dataclass
class Ledger:
    task_id: str
    problem_statement: str = ""
    database_name: str = ""
    ambiguities: list[AmbiguityEntry] = field(default_factory=list)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, ambiguity_id: str) -> Optional[AmbiguityEntry]:
        for a in self.ambiguities:
            if a.id == ambiguity_id:
                return a
        return None

    def unresolved_critical(self) -> list[AmbiguityEntry]:
        return [
            a for a in self.ambiguities
            if a.critical and a.status != AmbiguityStatus.RESOLVED
        ]

    def is_submit_allowed(self) -> bool:
        return len(self.unresolved_critical()) == 0

    def summary(self) -> str:
        total    = len(self.ambiguities)
        resolved = sum(1 for a in self.ambiguities if a.status == AmbiguityStatus.RESOLVED)
        unresolved = sum(
            1 for a in self.ambiguities
            if a.critical and a.status != AmbiguityStatus.RESOLVED
        )
        return (
            f"{resolved}/{total} ambiguities resolved. "
            f"{unresolved} critical unresolved."
        )

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(self, entry: AmbiguityEntry) -> None:
        """Add a new ambiguity. No-op if ID already exists."""
        if self.get(entry.id) is None:
            self.ambiguities.append(entry)

    def mark_asked(self, ambiguity_id: str, question: str) -> bool:
        a = self.get(ambiguity_id)
        if a is None:
            return False
        a.status = AmbiguityStatus.ASKED
        a.question_asked = question
        return True

    def record_business_info_check(
        self,
        ambiguity_id: str,
        search_string: str,
        result: str,
        found: bool,
    ) -> bool:
        a = self.get(ambiguity_id)
        if a is None:
            return False
        a.business_info_checks.append({
            "search_string": search_string,
            "result": result,
            "found": found,
        })
        return True

    def has_business_info_check(self, ambiguity_id: str) -> bool:
        a = self.get(ambiguity_id)
        return bool(a and a.business_info_checks)

    def business_info_supports(self, ambiguity_id: str, evidence: str) -> bool:
        """Return whether cited evidence occurs in a recorded KB response."""
        a = self.get(ambiguity_id)
        normalized = evidence.strip().casefold()
        if a is None or len(normalized) < 8:
            return False
        return any(
            check.get("found", False)
            and normalized in str(check.get("result", "")).casefold()
            for check in a.business_info_checks
        )

    def mark_resolved(
        self,
        ambiguity_id: str,
        resolution: str,
        source: str = "human",
    ) -> bool:
        a = self.get(ambiguity_id)
        if a is None:
            return False
        a.status = AmbiguityStatus.RESOLVED
        a.resolution = resolution
        a.resolution_source = source
        a.evidence.append(resolution)
        return True

    def mark_failed(self, ambiguity_id: str) -> bool:
        """Oracle said can't answer — reset to unresolved for retry."""
        a = self.get(ambiguity_id)
        if a is None:
            return False
        a.status = AmbiguityStatus.FAILED
        return True

    def mark_rejected(self, ambiguity_id: str) -> bool:
        """Oracle said irrelevant question — back to unresolved."""
        a = self.get(ambiguity_id)
        if a is None:
            return False
        a.status = AmbiguityStatus.UNRESOLVED
        return True

    def next_unresolved_id(self) -> Optional[str]:
        """Return the first critical unresolved ambiguity ID, or None."""
        for a in self.ambiguities:
            if a.critical and a.status == AmbiguityStatus.UNRESOLVED:
                return a.id
        return None

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "problem_statement": self.problem_statement,
            "database_name": self.database_name,
            "ambiguities": [a.to_dict() for a in self.ambiguities],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Ledger:
        ambiguities = [AmbiguityEntry.from_dict(a) for a in d.get("ambiguities", [])]
        return cls(
            task_id=d["task_id"],
            problem_statement=d.get("problem_statement", ""),
            database_name=d.get("database_name", ""),
            ambiguities=ambiguities,
        )


# ── File I/O ──────────────────────────────────────────────────────────────────

_file_locks: dict[str, threading.Lock] = {}
_file_locks_lock = threading.Lock()


def _get_lock(path: str) -> threading.Lock:
    with _file_locks_lock:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


def get_ledger_path(base_dir: str, task_id: str) -> Path:
    """
    Compute the ledger file path for a task.

    Strips model/mode/pass suffixes from compound task IDs so the detector
    and agent always share the same ledger file regardless of pass number.
    """
    # Strip __model__mode__pass_N suffixes — keep only the original task ID
    original_id = task_id.split("__")[0]
    return Path(base_dir) / original_id / "ledger.json"


def load_ledger(base_dir: str, task_id: str) -> Optional[Ledger]:
    """Load ledger from disk. Returns None if file doesn't exist."""
    path = get_ledger_path(base_dir, task_id)
    lock = _get_lock(str(path))
    with lock:
        if not path.exists():
            return None
        try:
            return Ledger.from_dict(json.loads(path.read_text()))
        except Exception:
            return None


def save_ledger(base_dir: str, ledger: Ledger) -> None:
    """Save ledger to disk atomically (write to tmp then rename)."""
    path = get_ledger_path(base_dir, ledger.task_id)
    lock = _get_lock(str(path))
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger.to_dict(), indent=2))
        tmp.replace(path)


def load_or_create_ledger(base_dir: str, task_id: str) -> Ledger:
    """Load existing ledger or create an empty one."""
    existing = load_ledger(base_dir, task_id)
    if existing is not None:
        return existing
    original_id = task_id.split("__")[0]
    return Ledger(task_id=original_id)


def get_ledger_base_dir() -> Optional[str]:
    """Get LEDGER_BASE_PATH from environment. Returns None if not set."""
    return os.environ.get("LEDGER_BASE_PATH")
