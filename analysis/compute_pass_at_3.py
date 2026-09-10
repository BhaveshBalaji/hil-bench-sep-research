#!/usr/bin/env python3
"""
Compute pass@3 for each experimental condition.

pass@3 = fraction of tasks where at least one pass out of 3 produced a correct submission.
pass@1 (mean) = average per-pass success rate across the 3 passes.

Usage:
  python compute_pass_at_3.py --results_dir /scratch/bbalaji2/hil-bench/results

The script looks for subdirectories matching the pattern:
  <results_dir>/<condition>_pass<N>/consolidated_results.json
  or
  <results_dir>/<condition>_pass<N>/results.json

Adjust the path patterns below if your layout differs.
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path


def load_results(results_file: Path) -> dict:
    """Load a results JSON file. Returns dict mapping instance_id -> resolved (bool)."""
    with open(results_file) as f:
        data = json.load(f)

    task_results = {}

    # Handle list format (list of result dicts)
    if isinstance(data, list):
        for item in data:
            # Try common field names
            task_id = (item.get("instance_id")
                       or item.get("task_id")
                       or item.get("id"))
            resolved = (item.get("resolved")
                        or item.get("correct")
                        or item.get("pass")
                        or False)
            if task_id is not None:
                task_results[str(task_id)] = bool(resolved)

    # Handle dict format (dict mapping id -> result)
    elif isinstance(data, dict):
        # Could be {"results": [...]} or {"instance_id": {...}, ...}
        if "results" in data:
            return load_results_from_list(data["results"])
        for task_id, val in data.items():
            if isinstance(val, dict):
                resolved = (val.get("resolved")
                            or val.get("correct")
                            or val.get("pass")
                            or False)
                task_results[str(task_id)] = bool(resolved)
            elif isinstance(val, bool):
                task_results[str(task_id)] = val

    return task_results


def load_results_from_list(items):
    task_results = {}
    for item in items:
        task_id = (item.get("instance_id")
                   or item.get("task_id")
                   or item.get("id"))
        resolved = (item.get("resolved")
                    or item.get("correct")
                    or item.get("pass")
                    or False)
        if task_id is not None:
            task_results[str(task_id)] = bool(resolved)
    return task_results


def find_result_files(results_dir: Path):
    """
    Walk results_dir and find all result files.
    Returns dict: condition_name -> list of (pass_number, filepath)
    """
    condition_passes = defaultdict(list)

    for path in sorted(results_dir.rglob("*.json")):
        name = path.name
        if name not in ("consolidated_results.json", "results.json",
                        "eval_results.json", "output.json"):
            continue

        # Try to infer condition and pass number from directory name
        # Expected patterns:
        #   baseline_pass1, sep_pass2, ablation_b_pass3, etc.
        dir_name = path.parent.name
        parts = dir_name.rsplit("_pass", 1)
        if len(parts) == 2 and parts[1].isdigit():
            condition = parts[0]
            pass_num = int(parts[1])
        else:
            # Try parent's parent
            dir_name2 = path.parent.parent.name
            parts2 = dir_name2.rsplit("_pass", 1)
            if len(parts2) == 2 and parts2[1].isdigit():
                condition = parts2[0]
                pass_num = int(parts2[1])
            else:
                condition = dir_name
                pass_num = len(condition_passes[condition]) + 1

        condition_passes[condition].append((pass_num, path))

    return condition_passes


def compute_metrics(condition_passes: dict):
    print(f"\n{'Condition':<30} {'Pass@1 (mean)':<18} {'Pass@3':<12} {'Tasks solved (any pass)'}")
    print("-" * 80)

    for condition in sorted(condition_passes.keys()):
        passes = sorted(condition_passes[condition], key=lambda x: x[0])
        if not passes:
            continue

        # Load per-pass results
        per_pass_results = []
        for pass_num, filepath in passes:
            try:
                results = load_results(filepath)
                per_pass_results.append(results)
                pass1 = sum(results.values()) / len(results) * 100 if results else 0
                print(f"  [{condition} pass{pass_num}] "
                      f"{len(results)} tasks, {sum(results.values())} correct "
                      f"({pass1:.1f}%)")
            except Exception as e:
                print(f"  Warning: could not load {filepath}: {e}")

        if not per_pass_results:
            continue

        # pass@1 mean: average success rate across passes
        pass1_rates = []
        for results in per_pass_results:
            if results:
                pass1_rates.append(sum(results.values()) / len(results) * 100)
        pass1_mean = sum(pass1_rates) / len(pass1_rates) if pass1_rates else 0

        # pass@3: union of tasks solved in any pass
        all_task_ids = set()
        for results in per_pass_results:
            all_task_ids.update(results.keys())

        solved_in_any_pass = set()
        for task_id in all_task_ids:
            if any(results.get(task_id, False) for results in per_pass_results):
                solved_in_any_pass.add(task_id)

        pass3 = len(solved_in_any_pass) / len(all_task_ids) * 100 if all_task_ids else 0

        print(f"\n{'':>2}{condition:<28} {pass1_mean:<18.1f} {pass3:<12.1f} "
              f"{len(solved_in_any_pass)}/{len(all_task_ids)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Compute pass@1 and pass@3 from results")
    parser.add_argument("--results_dir", type=Path,
                        default=Path("/scratch/bbalaji2/hil-bench/results"),
                        help="Root directory containing result subdirectories")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file details")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        print("Please provide --results_dir pointing to your results folder.")
        return

    print(f"Scanning: {args.results_dir}")
    condition_passes = find_result_files(args.results_dir)

    if not condition_passes:
        print("No result files found. Check that your results directory contains")
        print("subdirectories named like: baseline_pass1, sep_pass2, etc.")
        print("with consolidated_results.json or results.json inside.")
        return

    compute_metrics(condition_passes)


if __name__ == "__main__":
    main()
