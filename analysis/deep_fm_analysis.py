"""
deep_fm_analysis.py  —  Phase 1: Structural Failure Mode Analysis
-----------------------------------------------------------------
Richer FM2 and FM3 breakdown using only log structure (no LLM calls).

FM1  Silent omission          — task has zero questions
FM2  Abandonment breakdown    — what happens after an irrelevant question
  FM2a  Full stop             — got rejection, asked nothing more in the task
  FM2b  Continued             — got rejection, asked at least one more question
    FM2b-rephrase (Phase 2)   — next question is same topic (needs LLM)
    FM2b-switch  (Phase 2)    — next question is different topic (needs LLM)
  FM2c  Continued but failed  — continued after rejection, zero blockers discovered
  Recovery rate               — among FM2b tasks, fraction that discovered ≥1 blocker
FM3  Rejection cause heuristics
  MULTI_TOPIC                 — question contains connecting keywords between clauses
  SCHEMA_QUESTION             — mentions table/column/schema exploration keywords
  TOO_VAGUE                   — very short or overly generic
  ASSUMPTION_CONFIRM          — embeds assumption and asks to confirm
  OTHER                       — none of the above patterns matched

Also tracks: rejection-then-continue pairs saved for Phase 2 LLM classification.

Usage:
    python deep_fm_analysis.py \\
        --logs path/to/baseline/ask_human_logs.json "Baseline" \\
        --logs path/to/sep/ask_human_logs.json "SEP" \\
        --total-tasks 100 \\
        --save-phase2 phase2_pairs.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Response constants ────────────────────────────────────────────────────────
IRRELEVANT  = "irrelevant question"
CANT_ANSWER = "can't answer"

# ── FM3 heuristic patterns ────────────────────────────────────────────────────

# Schema-question signals — asks about table/column structure rather than business rules
_SCHEMA_PATTERNS = re.compile(
    r"\b(table|column|field|schema|database|which column|what column|"
    r"what table|which table|where is|stored in|contain|exist in|"
    r"get_table|get_column|select|join|foreign key)\b",
    re.IGNORECASE,
)

# Multi-topic signals — two question clauses joined by connectors
_MULTI_TOPIC_PATTERNS = re.compile(
    r"\b(and|also|additionally|as well as|furthermore|moreover)\b.{5,}\?",
    re.IGNORECASE,
)

# Assumption-confirmation — embeds a guess and asks for confirmation
_ASSUMPTION_PATTERNS = re.compile(
    r"\b(assuming|i assume|is it correct|is that right|is this correct|"
    r"would that be|should i assume|can i assume|does .{1,30} mean)\b",
    re.IGNORECASE,
)

# Too-vague signals — very short questions or generic "what does X mean" patterns
def _is_too_vague(q: str) -> bool:
    words = q.split()
    if len(words) < 6:
        return True
    # Simple definition requests that just parrot back a term
    if re.match(r"^(what|how) (does|is|do) ['\"]?.{1,30}['\"]? (mean|defined|defined as)\??$",
                q.strip(), re.IGNORECASE):
        return True
    return False


def _classify_fm3(question: str) -> str:
    """Heuristic FM3 rejection cause classification."""
    q = question.strip()
    if _ASSUMPTION_PATTERNS.search(q):
        return "ASSUMPTION_CONFIRM"
    if _MULTI_TOPIC_PATTERNS.search(q):
        return "MULTI_TOPIC"
    if _SCHEMA_PATTERNS.search(q):
        return "SCHEMA_QUESTION"
    if _is_too_vague(q):
        return "TOO_VAGUE"
    return "OTHER"


# ── Core per-pass analysis ────────────────────────────────────────────────────

def analyze_pass(
    tasks: dict[str, Any],
    total_tasks: int,
) -> dict:
    """
    Analyse one pass worth of task logs.

    tasks: dict mapping instance_id -> log dict
           log dict has keys: questions (list), blockers (dict)
    """
    # ── Basic counts ──────────────────────────────────────────────────────────
    tasks_in_logs   = len(tasks)
    fm1             = total_tasks - tasks_in_logs   # never asked

    all_questions = [
        q for log in tasks.values()
        for q in log.get("questions", [])
    ]
    total_q = len(all_questions)

    irrelevant_q = sum(
        1 for q in all_questions
        if IRRELEVANT in q.get("response", "").lower()
    )
    cant_answer_q = sum(
        1 for q in all_questions
        if CANT_ANSWER in q.get("response", "").lower()
    )
    blocker_hits = sum(
        1 for q in all_questions
        if q.get("blocker_name") is not None
    )

    # ── FM2 breakdown ─────────────────────────────────────────────────────────
    # For every task that received at least one rejection, classify behaviour.

    fm2a = 0          # got rejection → no more questions at all
    fm2b = 0          # got rejection → continued asking
    fm2c = 0          # got rejection → continued → still discovered nothing

    # Pairs for Phase 2: (task_id, rejected_q, next_q)
    phase2_pairs: list[dict] = []

    for task_id, log in tasks.items():
        qs = log.get("questions", [])
        if not qs:
            continue

        discovered = any(v for v in log.get("blockers", {}).values())

        # Find every position where the response was irrelevant OR cant-answer
        rejection_indices = [
            i for i, q in enumerate(qs)
            if IRRELEVANT in q.get("response", "").lower()
            or CANT_ANSWER in q.get("response", "").lower()
        ]

        # For FM3 cause classification we only want irrelevant (not cant-answer)
        irrelevant_indices = [
            i for i, q in enumerate(qs)
            if IRRELEVANT in q.get("response", "").lower()
        ]

        if not rejection_indices:
            continue  # no rejections in this task

        # FM2a: last rejection is the very last question → full stop
        # FM2b: at least one rejection has a subsequent question
        last_rejection_idx = max(rejection_indices)
        has_question_after_last_rejection = last_rejection_idx < len(qs) - 1

        if not has_question_after_last_rejection:
            # Stopped after rejection — only count as abandonment if nothing was discovered
            if not discovered:
                fm2a += 1
        else:
            fm2b += 1
            if not discovered:
                fm2c += 1

        # Collect ALL (rejected_q, next_q) pairs regardless of FM2a/FM2b
        # Every rejection→next instance captures one rephrase-or-switch decision
        # Only use irrelevant_indices — cant-answer pairs aren't rephrase candidates
        for idx in irrelevant_indices:
            if idx + 1 < len(qs):
                phase2_pairs.append({
                    "task_id":      task_id,
                    "rejected":     qs[idx].get("question", ""),
                    "next":         qs[idx + 1].get("question", ""),
                    "next_blocker": qs[idx + 1].get("blocker_name"),
                    "fm2_type":     "FM2b" if has_question_after_last_rejection else "FM2a",
                })

    # Recovery rate: among FM2b tasks, fraction that discovered ≥1 blocker
    fm2b_recovered = fm2b - fm2c
    recovery_rate  = fm2b_recovered / fm2b if fm2b > 0 else 0.0

    # ── FM3 heuristic classification ──────────────────────────────────────────
    fm3_causes: dict[str, int] = defaultdict(int)
    for q in all_questions:
        if IRRELEVANT in q.get("response", "").lower():
            cause = _classify_fm3(q.get("question", ""))
            fm3_causes[cause] += 1

    # ── Blocker discovery ─────────────────────────────────────────────────────
    discovered_any = sum(
        1 for log in tasks.values()
        if any(v for v in log.get("blockers", {}).values())
    )
    total_discovered = sum(
        sum(1 for v in log.get("blockers", {}).values() if v)
        for log in tasks.values()
    )

    q_per_blocker = (
        total_q / total_discovered if total_discovered > 0 else float("inf")
    )

    return {
        "total_tasks":        total_tasks,
        "tasks_in_logs":      tasks_in_logs,
        # FM1
        "fm1":                fm1,
        "fm1_pct":            fm1 / total_tasks * 100,
        # FM2 breakdown
        "tasks_with_rejection": len([
            t for t in tasks.values()
            if any(IRRELEVANT in q.get("response","").lower()
                   for q in t.get("questions",[]))
        ]),        "fm2a":               fm2a,
        "fm2a_pct":           fm2a / total_tasks * 100,
        "fm2b":               fm2b,
        "fm2b_pct":           fm2b / total_tasks * 100,
        "fm2c":               fm2c,
        "fm2c_pct":           fm2c / total_tasks * 100,
        "fm2b_recovered":     fm2b_recovered,
        "recovery_rate":      recovery_rate * 100,
        # FM3
        "total_q":            total_q,
        "irrelevant_q":       irrelevant_q,
        "cant_answer_q":      cant_answer_q,
        "fm3_pct":            irrelevant_q / max(total_q, 1) * 100,
        "fm3_causes":         dict(fm3_causes),
        # blocker discovery
        "discovered_any":     discovered_any,
        "discovered_any_pct": discovered_any / total_tasks * 100,
        "total_discovered":   total_discovered,
        "q_per_blocker":      q_per_blocker,
        "blocker_hits":       blocker_hits,
        # phase 2 data
        "phase2_pairs":       phase2_pairs,
    }


# ── Aggregation across passes ─────────────────────────────────────────────────

def aggregate_passes(pass_results: list[dict]) -> dict:
    """Average scalar metrics across passes, merge phase2 pairs."""
    n = len(pass_results)
    if n == 0:
        return {}

    scalar_keys = [
        "fm1", "fm1_pct",
        "tasks_with_rejection",
        "fm2a", "fm2a_pct",
        "fm2b", "fm2b_pct",
        "fm2c", "fm2c_pct",
        "fm2b_recovered", "recovery_rate",
        "total_q", "irrelevant_q", "cant_answer_q",
        "fm3_pct",
        "discovered_any", "discovered_any_pct",
        "total_discovered", "q_per_blocker", "blocker_hits",
    ]

    agg: dict[str, Any] = {}
    for k in scalar_keys:
        vals = [r[k] for r in pass_results if r[k] != float("inf")]
        agg[k] = sum(vals) / len(vals) if vals else 0.0

    # FM3 causes — sum across passes then normalise
    combined_causes: dict[str, int] = defaultdict(int)
    for r in pass_results:
        for cause, cnt in r.get("fm3_causes", {}).items():
            combined_causes[cause] += cnt
    agg["fm3_causes"] = dict(combined_causes)

    # Per-pass values for variance display
    agg["per_pass"] = pass_results

    # Merge phase2 pairs
    agg["phase2_pairs"] = [
        p for r in pass_results for p in r.get("phase2_pairs", [])
    ]

    return agg


# ── Pretty printing ───────────────────────────────────────────────────────────

def _pct(val: float) -> str:
    return f"{val:.1f}%"

def _n(val: float) -> str:
    return f"{val:.1f}"


def print_report(label: str, agg: dict, total_tasks: int) -> None:
    pp = agg["per_pass"]
    n  = len(pp)

    print(f"\n{'='*65}")
    print(f"DEEP FM ANALYSIS — {label}  ({n} passes, {total_tasks} tasks each)")
    print(f"{'='*65}")

    # ── FM1 ───────────────────────────────────────────────────────────────────
    print(f"\n── FM1  Silent Omission (never asked) ──────────────────────")
    print(f"  Avg tasks silent:  {_pct(agg['fm1_pct'])}  "
          f"({_n(agg['fm1'])}/{total_tasks})")
    for i, r in enumerate(pp, 1):
        print(f"    Pass {i}: {_pct(r['fm1_pct'])}")

    # ── FM2 breakdown ─────────────────────────────────────────────────────────
    print(f"\n── FM2  Abandonment Breakdown ───────────────────────────────")
    print(f"  Tasks with any rejection:  "
          f"{_n(agg['tasks_with_rejection'])}/{total_tasks}")
    print()
    print(f"  FM2a  Full stop after rejection (asked nothing more):")
    print(f"        {_pct(agg['fm2a_pct'])}  ({_n(agg['fm2a'])} tasks avg)")
    for i, r in enumerate(pp, 1):
        print(f"          Pass {i}: {_pct(r['fm2a_pct'])}")

    print()
    print(f"  FM2b  Continued asking after rejection:")
    print(f"        {_pct(agg['fm2b_pct'])}  ({_n(agg['fm2b'])} tasks avg)")
    for i, r in enumerate(pp, 1):
        print(f"          Pass {i}: {_pct(r['fm2b_pct'])}")

    print()
    print(f"  FM2c  Continued but discovered zero blockers:")
    print(f"        {_pct(agg['fm2c_pct'])}  ({_n(agg['fm2c'])} tasks avg)")
    for i, r in enumerate(pp, 1):
        print(f"          Pass {i}: {_pct(r['fm2c_pct'])}")

    print()
    print(f"  Rejection recovery rate (FM2b tasks that found ≥1 blocker):")
    print(f"        {_pct(agg['recovery_rate'])}")
    for i, r in enumerate(pp, 1):
        print(f"          Pass {i}: {_pct(r['recovery_rate'])}")

    print(f"\n  Phase 2 pairs saved: {len(agg['phase2_pairs'])} "
          f"(rejection → next-question pairs)")

    # ── FM3 breakdown ─────────────────────────────────────────────────────────
    print(f"\n── FM3  Rejection Cause Heuristics ─────────────────────────")
    print(f"  Total questions:     {_n(agg['total_q'])} avg/pass")
    print(f"  Rejected (irrel.):   {_n(agg['irrelevant_q'])} avg/pass  "
          f"({_pct(agg['fm3_pct'])})")
    print(f"  Can't-answer:        {_n(agg['cant_answer_q'])} avg/pass")

    causes = agg["fm3_causes"]
    total_rejected = sum(causes.values())
    if total_rejected > 0:
        print(f"\n  Rejection cause breakdown (all passes combined):")
        for cause in ["MULTI_TOPIC", "SCHEMA_QUESTION", "TOO_VAGUE",
                      "ASSUMPTION_CONFIRM", "OTHER"]:
            cnt  = causes.get(cause, 0)
            pct  = cnt / total_rejected * 100
            bar  = "█" * int(pct / 5)
            print(f"    {cause:<22} {cnt:4d}  ({pct:5.1f}%)  {bar}")

    # ── Blocker discovery ─────────────────────────────────────────────────────
    print(f"\n── Blocker Discovery ────────────────────────────────────────")
    print(f"  Tasks discovering ≥1 blocker:  {_pct(agg['discovered_any_pct'])}")
    print(f"  Total blockers discovered:     {_n(agg['total_discovered'])} avg/pass")
    q_pb = agg['q_per_blocker']
    print(f"  Questions per discovered blocker: "
          f"{'∞' if q_pb > 999 else _n(q_pb)}")


def print_comparison(label_a: str, agg_a: dict,
                     label_b: str, agg_b: dict) -> None:
    """Side-by-side delta table."""

    def delta(a: float, b: float) -> str:
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"

    rows = [
        ("FM1  Silent omission (%)",         "fm1_pct"),
        ("FM2a Full stop after reject (%)",  "fm2a_pct"),
        ("FM2b Continued after reject (%)",  "fm2b_pct"),
        ("FM2c Continued, 0 blockers (%)",   "fm2c_pct"),
        ("Rejection recovery rate (%)",      "recovery_rate"),
        ("FM3  Rejection rate (%)",          "fm3_pct"),
        ("Tasks discovering ≥1 blocker (%)","discovered_any_pct"),
        ("Avg questions per pass",           "total_q"),
        ("Questions per discovered blocker", "q_per_blocker"),
        ("Cant-answer per pass",             "cant_answer_q"),
    ]

    w = 38
    print(f"\n{'='*72}")
    print(f"COMPARISON: {label_a}  vs  {label_b}")
    print(f"{'='*72}")
    print(f"  {'Metric':<{w}}  {label_a:>10}  {label_b:>10}  {'Δ':>8}")
    print(f"  {'─'*68}")
    for name, key in rows:
        va = agg_a.get(key, 0.0)
        vb = agg_b.get(key, 0.0)
        if va == float("inf"): va = 0.0
        if vb == float("inf"): vb = 0.0
        print(f"  {name:<{w}}  {va:>10.1f}  {vb:>10.1f}  {delta(va,vb):>8}")

    # FM3 cause comparison
    causes_a = agg_a.get("fm3_causes", {})
    causes_b = agg_b.get("fm3_causes", {})
    tot_a    = max(sum(causes_a.values()), 1)
    tot_b    = max(sum(causes_b.values()), 1)
    print(f"\n  FM3 rejection cause breakdown (% of all rejections):")
    print(f"  {'Cause':<22}  {label_a:>10}  {label_b:>10}  {'Δ':>8}")
    print(f"  {'─'*56}")
    for cause in ["MULTI_TOPIC", "SCHEMA_QUESTION", "TOO_VAGUE",
                  "ASSUMPTION_CONFIRM", "OTHER"]:
        pa = causes_a.get(cause, 0) / tot_a * 100
        pb = causes_b.get(cause, 0) / tot_b * 100
        print(f"  {cause:<22}  {pa:>10.1f}  {pb:>10.1f}  {delta(pa,pb):>8}")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_logs(path_str: str) -> dict:
    matches = sorted(Path(".").glob(path_str))
    if not matches:
        p = Path(path_str)
        if p.exists():
            matches = [p]
    if not matches:
        print(f"[WARN] No file found: {path_str}", file=sys.stderr)
        return {}
    path = matches[-1]
    print(f"  Loading: {path}", file=sys.stderr)
    with open(path) as f:
        return json.load(f)


def split_by_pass(logs: dict) -> dict[int, dict]:
    """Group log entries by pass number."""
    by_pass: dict[int, dict] = defaultdict(dict)
    for iid, log in logs.items():
        p = int(iid.split("__pass_")[-1]) if "__pass_" in iid else 1
        orig = iid.split("__")[0]
        by_pass[p][orig] = log
    return by_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 deep failure-mode analysis (no LLM required)"
    )
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
    )
    parser.add_argument(
        "--save-phase2",
        type=str,
        default=None,
        help="If set, save Phase 2 rejection→next pairs to this JSON file.",
    )
    args = parser.parse_args()

    results: list[tuple[str, dict]] = []

    for path_str, label in args.logs:
        print(f"\nLoading {label} ...", file=sys.stderr)
        logs = load_logs(path_str)
        if not logs:
            print(f"[SKIP] {label}", file=sys.stderr)
            continue

        by_pass = split_by_pass(logs)
        pass_results = []
        for p in sorted(by_pass.keys()):
            r = analyze_pass(by_pass[p], args.total_tasks)
            r["pass"] = p
            pass_results.append(r)

        agg = aggregate_passes(pass_results)
        print_report(label, agg, args.total_tasks)
        results.append((label, agg))

    # Comparison table (first two conditions)
    if len(results) >= 2:
        print_comparison(results[0][0], results[0][1],
                         results[1][0], results[1][1])

    # Save Phase 2 pairs
    if args.save_phase2 and results:
        all_pairs: dict[str, list] = {}
        for label, agg in results:
            all_pairs[label] = agg.get("phase2_pairs", [])
        with open(args.save_phase2, "w") as f:
            json.dump(all_pairs, f, indent=2)
        print(f"\nPhase 2 pairs saved to: {args.save_phase2}")

    print()


if __name__ == "__main__":
    main()
