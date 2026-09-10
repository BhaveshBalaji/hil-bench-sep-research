"""
recompute_metrics.py
--------------------
Recomputes SQL pass@1 and ask_f1 metrics for an existing results directory
without re-running any agent tasks. Useful when metrics computation failed
after evaluation completed successfully.

Usage:
    python recompute_metrics.py <results_dir> <instances_file> [--ask-logs <path>]

Example:
    python recompute_metrics.py \
        results/sql_100_ask_human/openai_Qwen3-32B_ask_human_20260729_153650 \
        data/sql_100_instances/instances.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def compute_ask_f1_from_logs(logs: dict, pass_suffix: str) -> dict:
    """Compute precision, recall, ask_f1 for one pass from ask_human_logs.json."""
    pass_logs = {
        k: v for k, v in logs.items()
        if pass_suffix in k
    }
    if not pass_logs:
        return {}

    total_questions = 0
    relevant_questions = 0
    total_blockers = 0
    discovered_blockers = 0

    for inst_id, log in pass_logs.items():
        questions = log.get("questions", [])
        total_questions += len(questions)
        relevant_questions += sum(
            1 for q in questions if q.get("blocker_name") is not None
        )
        total_blockers += log.get("n_blockers", 0)
        discovered_blockers += sum(
            1 for hit in log.get("blockers", {}).values() if hit
        )

    precision = relevant_questions / total_questions if total_questions > 0 else 0.0
    recall = discovered_blockers / total_blockers if total_blockers > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "ask_f1": f1,
        "n_questions": total_questions,
        "n_blockers_present": total_blockers,
        "n_blockers_discovered": discovered_blockers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recompute SQL metrics for an existing results directory."
    )
    parser.add_argument("results_dir", help="Path to the run results directory.")
    parser.add_argument("instances_file", help="Path to instances.json used for this run.")
    parser.add_argument(
        "--ask-logs", default=None,
        help="Path to ask_human_logs.json (default: <results_dir>/ask_human_logs.json)."
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    instances_file = Path(args.instances_file)

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)
    if not instances_file.exists():
        print(f"❌ Instances file not found: {instances_file}", file=sys.stderr)
        sys.exit(1)

    # Load ask_human logs if available
    ask_logs_path = Path(args.ask_logs) if args.ask_logs else results_dir / "ask_human_logs.json"
    ask_logs = {}
    if ask_logs_path.exists():
        ask_logs = json.loads(ask_logs_path.read_text())
        print(f"📋 Loaded ask_human logs: {len(ask_logs)} instances")
    else:
        print("⚠️  No ask_human_logs.json found — ask_f1 will not be computed")

    # Find all model/mode/pass directories
    from hil_bench.utils.calculate_sql_pass_at_1 import (
        calculate_sql_pass_at_1,
        print_sql_pass_at_1_summary,
    )

    all_pass_results = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in {"public_metrics", "metadata"}:
            continue
        for mode_dir in sorted(model_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            # Find pass directories
            pass_dirs = sorted(
                [d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("pass_")],
                key=lambda p: int(p.name.split("_")[1])
            )
            if not pass_dirs:
                # Single pass — mode_dir is the pass dir
                pass_dirs = [mode_dir]

            for pass_dir in pass_dirs:
                preds_file = pass_dir / "preds.json"
                if not preds_file.exists():
                    print(f"⚠️  No preds.json in {pass_dir} — skipping")
                    continue

                pass_num = int(pass_dir.name.split("_")[1]) if pass_dir.name.startswith("pass_") else 1
                print(f"\n🧮 Computing metrics for {model_dir.name}/{mode_dir.name}/pass_{pass_num} ...")

                try:
                    metrics = calculate_sql_pass_at_1(
                        trajectory_dir=str(pass_dir),
                        tasks_dir=str(instances_file.parent),
                        instances_file=str(instances_file),
                        max_workers=4,
                    )
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    metrics = {}

                pass_at_1 = metrics.get("pass_at_1_rate", 0.0)
                n_correct = metrics.get("pass_at_1_count", 0)
                n_total = metrics.get("total_instances", 0)
                print(f"  ✅ pass@1 = {pass_at_1:.1f}% ({n_correct}/{n_total})")

                # Save metrics.json
                metrics_out = pass_dir / "metrics.json"
                metrics_out.write_text(json.dumps(metrics, indent=2))
                print(f"  💾 Saved metrics.json")

                # Compute ask_f1 from logs if available
                pass_suffix = f"pass_{pass_num}"
                ask_metrics = compute_ask_f1_from_logs(ask_logs, pass_suffix)
                if ask_metrics:
                    ask_f1_pct = ask_metrics["ask_f1"] * 100
                    print(f"  📊 ask_f1 = {ask_f1_pct:.1f}%  "
                          f"(precision={ask_metrics['precision']:.3f}, "
                          f"recall={ask_metrics['recall']:.3f}, "
                          f"questions={ask_metrics['n_questions']}, "
                          f"blockers_discovered={ask_metrics['n_blockers_discovered']}/"
                          f"{ask_metrics['n_blockers_present']})")

                all_pass_results.append({
                    "model": model_dir.name,
                    "mode": mode_dir.name,
                    "pass_num": pass_num,
                    "pass_at_1": pass_at_1,
                    "n_correct": n_correct,
                    "n_total": n_total,
                    **{f"ask_{k}": v for k, v in ask_metrics.items()},
                })

    if not all_pass_results:
        print("\n❌ No valid pass directories found.", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print("\n" + "="*70)
    print("📈 Summary")
    print("="*70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_pass_results:
        grouped[(r["model"], r["mode"])].append(r)

    for (model, mode), passes in sorted(grouped.items()):
        pass_at_1_values = [r["pass_at_1"] for r in passes]
        mean_p1 = sum(pass_at_1_values) / len(pass_at_1_values)
        std_p1 = math.sqrt(
            sum((x - mean_p1) ** 2 for x in pass_at_1_values) / max(len(pass_at_1_values) - 1, 1)
        )
        individual = ", ".join(f"{v:.1f}%" for v in pass_at_1_values)
        print(f"\n  {model} / {mode}:")
        print(f"    pass@1 = {mean_p1:.1f}% ± {std_p1:.1f}% [{individual}]")

        ask_f1_values = [r.get("ask_ask_f1", 0) * 100 for r in passes if "ask_ask_f1" in r]
        if ask_f1_values:
            mean_f1 = sum(ask_f1_values) / len(ask_f1_values)
            std_f1 = math.sqrt(
                sum((x - mean_f1) ** 2 for x in ask_f1_values) / max(len(ask_f1_values) - 1, 1)
            )
            print(f"    ask_f1 = {mean_f1:.2f}% ± {std_f1:.2f}%")

        # True pass@3
        from collections import defaultdict as dd2
        task_passes = dd2(list)
        for r in passes:
            pass_dir_path = results_dir / model / mode / f"pass_{r['pass_num']}"
            preds_file = pass_dir_path / "preds.json"
            if preds_file.exists():
                preds = json.loads(preds_file.read_text())
                metrics_file = pass_dir_path / "metrics.json"
                if metrics_file.exists():
                    m = json.loads(metrics_file.read_text())
                    resolved = set(m.get("resolved_instances", []))
                    for inst_id in preds:
                        orig = inst_id.split("__")[0]
                        task_passes[orig].append(inst_id in resolved)

        if task_passes:
            pass_at_k = sum(1 for p in task_passes.values() if any(p))
            total_tasks = len(task_passes)
            print(f"    pass@{len(passes)} (any pass correct) = {pass_at_k}/{total_tasks} = {pass_at_k/total_tasks*100:.1f}%")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
