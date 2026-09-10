"""
cant_answer_audit.py
--------------------
Shows per-pass cant-answer counts and rates for one or more conditions.
Helps diagnose whether rate limiting is affecting evaluation quality.

Usage:
    python cant_answer_audit.py \
        --logs results/llama33_70b/baseline/*/ask_human_logs.json "Baseline" \
        --logs results/llama33_70b/enum_only/*/ask_human_logs.json "Enum-Only" \
        --logs results/llama33_70b/sep/*/ask_human_logs.json "SEP"

    # Or pass directly:
    python cant_answer_audit.py \
        --logs path/to/ask_human_logs.json "Label"
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

CANT_ANSWER = "can't answer"
IRRELEVANT = "irrelevant question"


def load_logs(path_pattern: str) -> dict:
    """Load ask_human_logs.json, supporting glob patterns."""
    matches = sorted(Path(".").glob(path_pattern))
    if not matches:
        # Try as literal path
        p = Path(path_pattern)
        if p.exists():
            matches = [p]
    if not matches:
        print(f"  [WARN] No file found for: {path_pattern}", file=sys.stderr)
        return {}
    # Use the most recent match
    path = matches[-1]
    with open(path) as f:
        return json.load(f)


def audit(logs: dict, label: str, total_tasks: int = 100):
    # Group by pass
    by_pass = defaultdict(dict)
    for iid, log in logs.items():
        p = int(iid.split("__pass_")[-1]) if "__pass_" in iid else 1
        orig = iid.split("__")[0]
        by_pass[p][orig] = log

    print(f"\n{'='*60}")
    print(f"CANT-ANSWER AUDIT — {label}")
    print(f"{'='*60}")

    total_cant = 0
    total_q = 0

    for p in sorted(by_pass.keys()):
        tasks = by_pass[p]
        all_questions = [
            q for log in tasks.values()
            for q in log.get("questions", [])
        ]
        n_q = len(all_questions)
        n_cant = sum(
            1 for q in all_questions
            if CANT_ANSWER in q.get("response", "").lower()
        )
        n_irrel = sum(
            1 for q in all_questions
            if IRRELEVANT in q.get("response", "").lower()
        )
        n_blocker = sum(
            1 for q in all_questions
            if q.get("blocker_name") is not None
        )

        # Tasks with at least one cant-answer
        tasks_with_cant = sum(
            1 for log in tasks.values()
            if any(
                CANT_ANSWER in q.get("response", "").lower()
                for q in log.get("questions", [])
            )
        )

        pct = (n_cant / n_q * 100) if n_q > 0 else 0
        total_cant += n_cant
        total_q += n_q

        flag = " ⚠️ " if n_cant > 10 else "    "
        print(
            f"  Pass {p}:{flag}"
            f"cant_answer={n_cant:3d} / {n_q:4d} total_q ({pct:5.1f}%)  "
            f"irrelevant={n_irrel:3d}  "
            f"blocker_hits={n_blocker:3d}  "
            f"tasks_affected={tasks_with_cant:3d}/{total_tasks}"
        )

    overall_pct = (total_cant / total_q * 100) if total_q > 0 else 0
    print(f"  {'─'*55}")
    print(
        f"  TOTAL: cant_answer={total_cant} / {total_q} "
        f"({overall_pct:.1f}%) across {len(by_pass)} passes"
    )
    if overall_pct > 5:
        print(
            f"  ⚠️  {overall_pct:.1f}% cant-answer rate — "
            f"results likely contaminated by rate limiting"
        )
    else:
        print(f"  ✅ cant-answer rate is low — results are reliable")


def main():
    parser = argparse.ArgumentParser(description="Audit cant-answer rates per pass")
    parser.add_argument(
        "--logs",
        nargs=2,
        action="append",
        metavar=("PATH", "LABEL"),
        required=True,
        help="Path to ask_human_logs.json and a label. Repeat for multiple conditions.",
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=100,
        help="Total number of tasks in the evaluation (default: 100)",
    )
    args = parser.parse_args()

    for path_str, label in args.logs:
        logs = load_logs(path_str)
        if logs:
            audit(logs, label, args.total_tasks)
        else:
            print(f"\n[SKIP] No logs found for {label} at {path_str}")

    print()


if __name__ == "__main__":
    main()
