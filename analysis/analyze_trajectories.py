"""
analyze_trajectories.py
-----------------------
Quantifies failure modes from ask_human_logs.json.

KEY INSIGHT: ask_human_logs.json only contains entries for tasks where
ask_human was called at least once. Tasks where the agent NEVER called
ask_human are absent from the logs — these are the true FM1 (silent omission).

FM1: Silent omission — agent never called ask_human (absent from logs)
FM2: Abandonment — last question got "irrelevant question", no blocker found
FM3: Combined/vague questions — question got "irrelevant question" response

Usage:
    /scratch/bbalaji2/hil-bench/.venv/bin/python analyze_trajectories.py \
        --ask-logs path/to/ask_human_logs.json \
        --label "Baseline" \
        --total-tasks 100 \
        --compare-logs path/to/other_logs.json \
        --compare-label "Relaxed V3"
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

IRRELEVANT = "irrelevant question"
CANT_ANSWER = "can't answer"


def analyze(logs, label, total_tasks=100):
    # Group by pass number
    by_pass = defaultdict(dict)
    for iid, log in logs.items():
        p = int(iid.split("__pass_")[-1]) if "__pass_" in iid else 1
        orig = iid.split("__")[0]
        by_pass[p][orig] = log

    pass_results = {}
    for p, tasks in sorted(by_pass.items()):
        tasks_in_logs = len(tasks)

        # FM1: tasks NOT in logs never called ask_human at all
        fm1 = total_tasks - tasks_in_logs

        # Tasks that asked at least once (all tasks in logs by definition)
        tasks_asked = tasks_in_logs

        # Total questions and question-level metrics
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

        # FM2: last question irrelevant, no blocker discovered
        fm2 = 0
        for log in tasks.values():
            qs = log.get("questions", [])
            if not qs:
                continue
            last = qs[-1].get("response", "").lower()
            discovered = any(v for v in log.get("blockers", {}).values())
            if IRRELEVANT in last and not discovered:
                fm2 += 1

        # FM3: % of questions that got irrelevant
        fm3_pct = irrelevant_q / max(total_q, 1) * 100

        # Blocker discovery
        discovered_any = sum(
            1 for log in tasks.values()
            if any(v for v in log.get("blockers", {}).values())
        )

        # Total blockers discovered across all tasks in this pass
        total_blockers_discovered = sum(
            sum(1 for v in log.get("blockers", {}).values() if v)
            for log in tasks.values()
        )

        # Interaction cost metrics
        # Questions per discovered blocker — how many questions does it
        # take to discover each blocker? Lower = more efficient.
        q_per_blocker = (
            total_q / total_blockers_discovered
            if total_blockers_discovered > 0 else float("inf")
        )

        # Rejected questions per task (among tasks that asked at all)
        # Shows whether the agent wastes oracle calls
        rejected_per_asking_task = (
            irrelevant_q / tasks_in_logs
            if tasks_in_logs > 0 else 0.0
        )

        # Questions per asking task (among tasks that asked at all)
        q_per_asking_task = (
            total_q / tasks_in_logs
            if tasks_in_logs > 0 else 0.0
        )

        pass_results[p] = {
            "total_tasks": total_tasks,
            "tasks_in_logs": tasks_in_logs,
            "fm1": fm1,
            "fm1_pct": fm1 / total_tasks * 100,
            "tasks_asked": tasks_asked,
            "tasks_asked_pct": tasks_asked / total_tasks * 100,
            "total_q": total_q,
            "irrelevant_q": irrelevant_q,
            "cant_answer_q": cant_answer_q,
            "fm2": fm2,
            "fm2_pct": fm2 / total_tasks * 100,
            "fm3_pct": fm3_pct,
            "discovered_any": discovered_any,
            "discovered_any_pct": discovered_any / total_tasks * 100,
            # Interaction-cost metrics
            "total_blockers_discovered": total_blockers_discovered,
            "q_per_blocker": q_per_blocker,
            "rejected_per_asking_task": rejected_per_asking_task,
            "q_per_asking_task": q_per_asking_task,
        }

    passes = list(pass_results.keys())
    n = len(passes)

    def avg(k):
        vals = [pass_results[p][k] for p in passes
                if pass_results[p][k] != float("inf")]
        return sum(vals) / len(vals) if vals else float("inf")

    return {
        "label": label,
        "passes": passes,
        "per_pass": pass_results,
        "avg": {k: avg(k) for k in [
            "fm1_pct", "tasks_asked_pct", "total_q",
            "fm2_pct", "fm3_pct", "discovered_any_pct",
            "fm1", "tasks_asked", "irrelevant_q", "cant_answer_q",
            "total_blockers_discovered", "q_per_blocker",
            "rejected_per_asking_task", "q_per_asking_task",
        ]},
    }


def print_result(r):
    a = r["avg"]
    print(f"\n{'='*65}")
    print(f"FAILURE MODE ANALYSIS — {r['label']}")
    print(f"{'='*65}")
    print(f"Passes: {r['passes']}")
    print()
    print(f"Coverage (avg across passes):")
    print(f"  Tasks that asked ≥1 question: {a['tasks_asked']:.0f}/100 = {a['tasks_asked_pct']:.1f}%")
    print(f"  Avg questions per pass:        {a['total_q']:.0f}")
    print(f"  Tasks discovering ≥1 blocker:  {a['discovered_any_pct']:.1f}%")
    print()
    print(f"Failure Modes (avg across {len(r['passes'])} passes):")
    print(f"  FM1 Silent omission (never asked):             {a['fm1']:.0f}/100 = {a['fm1_pct']:.1f}%")
    print(f"  FM2 Abandoned after rejection (no discovery):  {a['fm2_pct']:.1f}%")
    print(f"  FM3 Questions rejected (irrelevant response):  {a['fm3_pct']:.1f}% of all Qs")
    print()
    print(f"Interaction Cost (avg across passes):")
    q_per_b = a['q_per_blocker']
    q_per_b_str = f"{q_per_b:.2f}" if q_per_b != float("inf") else "∞ (no blockers discovered)"
    print(f"  Questions per discovered blocker:  {q_per_b_str}")
    print(f"  Questions per asking task:         {a['q_per_asking_task']:.2f}")
    print(f"  Rejected questions per asking task:{a['rejected_per_asking_task']:.2f}")
    print(f"  Total blockers discovered:         {a['total_blockers_discovered']:.0f}")
    print()
    for p, pr in sorted(r["per_pass"].items()):
        ca = f"  ⚠️ {pr['cant_answer_q']} cant-answer" if pr["cant_answer_q"] else ""
        q_b = f"{pr['q_per_blocker']:.2f}" if pr["q_per_blocker"] != float("inf") else "∞"
        print(f"  Pass {p}: FM1={pr['fm1_pct']:.0f}% "
              f"FM2={pr['fm2_pct']:.0f}% "
              f"FM3={pr['fm3_pct']:.0f}% "
              f"asked={pr['tasks_asked']}/100 "
              f"Qs={pr['total_q']} "
              f"Q/blocker={q_b}{ca}")


def print_comparison(r1, r2):
    a1, a2 = r1["avg"], r2["avg"]
    print(f"\n{'='*65}")
    print(f"COMPARISON: {r1['label']}  vs  {r2['label']}")
    print(f"{'='*65}")
    rows = [
        ("Tasks asked ≥1 question (%)",        "tasks_asked_pct"),
        ("Avg questions per pass",              "total_q"),
        ("Tasks discovering ≥1 blocker (%)",   "discovered_any_pct"),
        ("FM1 Silent omission (%)",             "fm1_pct"),
        ("FM2 Abandoned after rejection (%)",   "fm2_pct"),
        ("FM3 Rejected questions (%)",          "fm3_pct"),
        ("Questions per discovered blocker",    "q_per_blocker"),
        ("Questions per asking task",           "q_per_asking_task"),
        ("Rejected questions per asking task",  "rejected_per_asking_task"),
    ]
    print(f"{'Metric':<42} {r1['label']:>10} {r2['label']:>10} {'Δ':>8}")
    print("─"*65)
    for name, k in rows:
        v1, v2 = a1[k], a2[k]
        if v1 == float("inf") or v2 == float("inf"):
            print(f"{name:<42} {'∞':>10} {'∞':>10} {'N/A':>8}")
            continue
        d = v2 - v1
        sign = "+" if d > 0 else ""
        print(f"{name:<42} {v1:>10.2f} {v2:>10.2f} {sign}{d:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-logs", required=True)
    ap.add_argument("--label", default="Condition 1")
    ap.add_argument("--total-tasks", type=int, default=100)
    ap.add_argument("--compare-logs", default=None)
    ap.add_argument("--compare-label", default="Condition 2")
    ap.add_argument("--compare-total-tasks", type=int, default=None)
    args = ap.parse_args()

    p1 = Path(args.ask_logs)
    if not p1.exists():
        print(f"Not found: {p1}", file=sys.stderr)
        sys.exit(1)

    logs1 = json.loads(p1.read_text())
    print(f"Loaded {len(logs1)} entries from {p1.name}")
    r1 = analyze(logs1, args.label, args.total_tasks)
    print_result(r1)

    if args.compare_logs:
        p2 = Path(args.compare_logs)
        if not p2.exists():
            print(f"Not found: {p2}", file=sys.stderr)
            sys.exit(1)
        logs2 = json.loads(p2.read_text())
        total2 = args.compare_total_tasks or args.total_tasks
        print(f"Loaded {len(logs2)} entries from {p2.name}")
        r2 = analyze(logs2, args.compare_label, total2)
        print_result(r2)
        print_comparison(r1, r2)


if __name__ == "__main__":
    main()
