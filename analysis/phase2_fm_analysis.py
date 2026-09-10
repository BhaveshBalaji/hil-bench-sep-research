"""
phase2_fm_analysis.py  —  Phase 2: LLM Semantic Classification
---------------------------------------------------------------
Takes phase2_pairs.json produced by deep_fm_analysis.py and classifies
each (rejected_question, next_question) pair as:

  REPHRASE     — next question asks about the same ambiguity as the rejected
                 one, just worded differently. Agent is persisting.
  TOPIC_SWITCH — next question asks about a completely different ambiguity.
                 Agent gave up on the rejected topic and moved on.

This splits FM2b into:
  FM2b-rephrase   : retry persistence (SEP's intended behavior)
  FM2b-switch     : partial abandonment (gave up on that blocker thread)

Also runs FM3 cause classification using the LLM — validates/corrects the
heuristic classification from Phase 1.

FM3 LLM classes:
  MULTI_TOPIC          — question combines two or more distinct ambiguities
  SCHEMA_QUESTION      — asks about table/column structure, not business rules
  TOO_VAGUE            — overly generic, no specific ambiguity targeted
  ASSUMPTION_CONFIRM   — embeds the agent's own assumption and asks to confirm
  OTHER                — rejected for another reason

Usage:
    # Minimal — just FM2 rephrase classification:
    python phase2_fm_analysis.py \\
        --pairs phase2_pairs.json \\
        --api-key sk-or-v1-xxx \\
        --model openrouter/meta-llama/llama-3.3-70b-instruct

    # Also run FM3 LLM classification on ask_human_logs:
    python phase2_fm_analysis.py \\
        --pairs phase2_pairs.json \\
        --fm3-logs results/.../ask_human_logs.json "Baseline" \\
        --fm3-logs results/.../ask_human_logs.json "SEP" \\
        --api-key sk-or-v1-xxx \\
        --model openrouter/meta-llama/llama-3.3-70b-instruct \\
        --save-results phase2_results.json

Environment variable alternative to --api-key:
    export LITELLM_API_KEY=sk-or-v1-xxx
    python phase2_fm_analysis.py --pairs phase2_pairs.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import litellm
    litellm.set_verbose = False
    import logging
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
except ImportError:
    print("[ERROR] litellm not installed. Run: pip install litellm", file=sys.stderr)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
IRRELEVANT  = "irrelevant question"
CANT_ANSWER = "can't answer"

DEFAULT_MODEL    = "openrouter/meta-llama/llama-3.3-70b-instruct"
RETRY_DELAY_S    = 2.0   # seconds between API calls to avoid rate limits
MAX_RETRIES      = 3


# ── LLM call wrapper ──────────────────────────────────────────────────────────

def call_llm(prompt: str, api_key: str, model: str,
             base_url: str | None = None) -> str:
    """Single LLM call with retries. Returns the response text."""
    kwargs: dict = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens":  256,
        "api_key":     api_key,
    }
    if base_url:
        kwargs["api_base"] = base_url

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_DELAY_S * attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] {e} — waiting {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    return ""


# ── FM2 rephrase classifier ───────────────────────────────────────────────────

FM2_PROMPT = """You are a precise semantic classifier for a research study.

Two questions were asked by an AI agent during a database task:

QUESTION A (was rejected as "irrelevant question"):
"{rejected}"

QUESTION B (asked immediately after the rejection):
"{next}"

Your task: determine whether Question B is asking about the SAME ambiguity as Question A, just rephrased — or whether it is asking about a COMPLETELY DIFFERENT topic.

REPHRASE: Question B targets the same underlying ambiguity as Question A. The core topic, business term, or uncertainty being asked about is the same, even if the wording is different.

TOPIC_SWITCH: Question B asks about a different business term, metric, threshold, or ambiguity. The agent has abandoned the topic of Question A and moved on.

Rules:
- Focus only on the semantic topic being asked about, not surface wording
- If in doubt between REPHRASE and TOPIC_SWITCH, choose TOPIC_SWITCH
- Respond with exactly one word: either REPHRASE or TOPIC_SWITCH

Classification:"""


def classify_fm2_pair(rejected: str, next_q: str,
                      api_key: str, model: str,
                      base_url: str | None = None) -> str:
    """Returns 'REPHRASE' or 'TOPIC_SWITCH'."""
    prompt = FM2_PROMPT.format(rejected=rejected, next=next_q)
    try:
        result = call_llm(prompt, api_key, model, base_url).upper()
        if "REPHRASE" in result:
            return "REPHRASE"
        if "TOPIC_SWITCH" in result or "SWITCH" in result:
            return "TOPIC_SWITCH"
        return "UNKNOWN"
    except Exception as e:
        print(f"  [FM2 classify error] {e}", file=sys.stderr)
        return "UNKNOWN"


# ── FM3 rejection cause classifier ───────────────────────────────────────────

FM3_PROMPT = """You are a precise classifier for a research study on AI agent behavior.

An AI agent asked the following question to a human oracle, and it was rejected as "irrelevant question":

REJECTED QUESTION:
"{question}"

Classify the PRIMARY reason this question was likely rejected. Choose exactly one:

MULTI_TOPIC        — The question combines two or more distinct ambiguities or topics in a single question (e.g., "What does X mean and how is Y calculated?")

SCHEMA_QUESTION    — The question asks about database structure, table contents, or column definitions rather than about business rules, thresholds, or domain-specific definitions (e.g., "Which column contains the unemployment rate?")

TOO_VAGUE          — The question is too generic or broad, lacking specific intent (e.g., "What does 'high-quality' mean?" without showing any analysis of the specific business context)

ASSUMPTION_CONFIRM — The question embeds the agent's own assumption and asks for confirmation rather than openly asking for the correct answer (e.g., "I'm assuming X means Y, is that correct?")

OTHER              — None of the above clearly apply

Rules:
- Choose the single most dominant reason
- If the question seems fine but was still rejected, use OTHER
- Respond with exactly one of: MULTI_TOPIC, SCHEMA_QUESTION, TOO_VAGUE, ASSUMPTION_CONFIRM, OTHER

Classification:"""


def classify_fm3_question(question: str,
                          api_key: str, model: str,
                          base_url: str | None = None) -> str:
    """Returns one of the FM3 cause labels."""
    valid = {"MULTI_TOPIC", "SCHEMA_QUESTION", "TOO_VAGUE",
             "ASSUMPTION_CONFIRM", "OTHER"}
    prompt = FM3_PROMPT.format(question=question)
    try:
        result = call_llm(prompt, api_key, model, base_url).upper().strip()
        for label in valid:
            if label in result:
                return label
        return "OTHER"
    except Exception as e:
        print(f"  [FM3 classify error] {e}", file=sys.stderr)
        return "UNKNOWN"


# ── FM2 phase 2 runner ────────────────────────────────────────────────────────

def run_fm2_classification(
    pairs_by_label: dict[str, list],
    api_key: str,
    model: str,
    base_url: str | None = None,
) -> dict[str, dict]:
    """
    Classify all FM2b pairs per label.
    Returns dict: label -> {rephrase: int, topic_switch: int, unknown: int,
                            total: int, rephrase_pct: float, pairs: list}
    """
    results: dict[str, dict] = {}

    for label, pairs in pairs_by_label.items():
        print(f"\nFM2 Phase 2 — classifying {len(pairs)} pairs for [{label}]")
        counts: dict[str, int] = defaultdict(int)
        annotated: list[dict] = []

        for i, pair in enumerate(pairs, 1):
            rejected   = pair.get("rejected", "")
            next_q     = pair.get("next", "")
            task_id    = pair.get("task_id", "?")
            next_blocker = pair.get("next_blocker")
            fm2_type   = pair.get("fm2_type", "unknown")

            if not rejected or not next_q:
                counts["UNKNOWN"] += 1
                continue

            classification = classify_fm2_pair(
                rejected, next_q, api_key, model, base_url
            )
            counts[classification] += 1
            annotated.append({**pair, "classification": classification})

            indicator = "✅" if classification == "REPHRASE" else "🔀"
            print(f"  [{i:2d}/{len(pairs)}] {task_id[:30]:<30} "
                  f"[{fm2_type}] {indicator} {classification}"
                  + (f"  [blocker: {next_blocker}]" if next_blocker else ""))

            time.sleep(RETRY_DELAY_S)

        total = len(pairs)
        rephrase     = counts.get("REPHRASE", 0)
        topic_switch = counts.get("TOPIC_SWITCH", 0)
        unknown      = counts.get("UNKNOWN", 0)

        # Split by FM2a vs FM2b
        fm2a_pairs = [p for p in annotated if p.get("fm2_type") == "FM2a"]
        fm2b_pairs = [p for p in annotated if p.get("fm2_type") == "FM2b"]

        def split_counts(subset):
            r  = sum(1 for p in subset if p.get("classification") == "REPHRASE")
            s  = sum(1 for p in subset if p.get("classification") == "TOPIC_SWITCH")
            n  = len(subset)
            # Successful = next question hit a blocker
            r_success = sum(1 for p in subset
                           if p.get("classification") == "REPHRASE"
                           and p.get("next_blocker") is not None)
            s_success = sum(1 for p in subset
                           if p.get("classification") == "TOPIC_SWITCH"
                           and p.get("next_blocker") is not None)
            return r, s, n, r_success, s_success

        fm2a_r, fm2a_s, fm2a_n, fm2a_rs, fm2a_ss = split_counts(fm2a_pairs)
        fm2b_r, fm2b_s, fm2b_n, fm2b_rs, fm2b_ss = split_counts(fm2b_pairs)

        # Overall success counts
        all_r_success = sum(1 for p in annotated
                           if p.get("classification") == "REPHRASE"
                           and p.get("next_blocker") is not None)
        all_s_success = sum(1 for p in annotated
                           if p.get("classification") == "TOPIC_SWITCH"
                           and p.get("next_blocker") is not None)

        results[label] = {
            "total":         total,
            "rephrase":      rephrase,
            "topic_switch":  topic_switch,
            "unknown":       unknown,
            "rephrase_pct":  rephrase / max(total, 1) * 100,
            "switch_pct":    topic_switch / max(total, 1) * 100,
            # Successful = next question hit a registered blocker
            "rephrase_success":     all_r_success,
            "rephrase_success_pct": all_r_success / max(rephrase, 1) * 100,
            "switch_success":       all_s_success,
            "switch_success_pct":   all_s_success / max(topic_switch, 1) * 100,
            # FM2a breakdown
            "fm2a_total":            fm2a_n,
            "fm2a_rephrase":         fm2a_r,
            "fm2a_switch":           fm2a_s,
            "fm2a_rephrase_pct":     fm2a_r / max(fm2a_n, 1) * 100,
            "fm2a_switch_pct":       fm2a_s / max(fm2a_n, 1) * 100,
            "fm2a_rephrase_success": fm2a_rs,
            "fm2a_rephrase_success_pct": fm2a_rs / max(fm2a_r, 1) * 100,
            "fm2a_switch_success":   fm2a_ss,
            "fm2a_switch_success_pct": fm2a_ss / max(fm2a_s, 1) * 100,
            # FM2b breakdown
            "fm2b_total":            fm2b_n,
            "fm2b_rephrase":         fm2b_r,
            "fm2b_switch":           fm2b_s,
            "fm2b_rephrase_pct":     fm2b_r / max(fm2b_n, 1) * 100,
            "fm2b_switch_pct":       fm2b_s / max(fm2b_n, 1) * 100,
            "fm2b_rephrase_success": fm2b_rs,
            "fm2b_rephrase_success_pct": fm2b_rs / max(fm2b_r, 1) * 100,
            "fm2b_switch_success":   fm2b_ss,
            "fm2b_switch_success_pct": fm2b_ss / max(fm2b_s, 1) * 100,
            "pairs":         annotated,
        }

    return results


# ── FM3 phase 2 runner ────────────────────────────────────────────────────────

def load_logs(path_str: str) -> dict:
    matches = sorted(Path(".").glob(path_str))
    if not matches:
        p = Path(path_str)
        if p.exists():
            matches = [p]
    if not matches:
        return {}
    with open(matches[-1]) as f:
        return json.load(f)


def run_fm3_classification(
    logs_by_label: list[tuple[str, str]],
    api_key: str,
    model: str,
    base_url: str | None = None,
) -> dict[str, dict]:
    """
    For each rejected question in the logs, classify FM3 cause using LLM.
    Returns dict: label -> {causes: {cause: count}, total_rejected: int, pairs: list}
    """
    results: dict[str, dict] = {}

    for path_str, label in logs_by_label:
        logs = load_logs(path_str)
        if not logs:
            print(f"[SKIP] No logs found for {label}", file=sys.stderr)
            continue

        # Collect all rejected questions across all passes
        rejected_questions: list[dict] = []
        for iid, log in logs.items():
            for q in log.get("questions", []):
                if IRRELEVANT in q.get("response", "").lower():
                    rejected_questions.append({
                        "task_id":  iid,
                        "question": q.get("question", ""),
                    })

        print(f"\nFM3 Phase 2 — classifying {len(rejected_questions)} "
              f"rejected questions for [{label}]")

        causes: dict[str, int] = defaultdict(int)
        annotated: list[dict] = []

        for i, item in enumerate(rejected_questions, 1):
            q       = item["question"]
            task_id = item["task_id"]

            if not q.strip():
                causes["OTHER"] += 1
                continue

            cause = classify_fm3_question(q, api_key, model, base_url)
            causes[cause] += 1
            annotated.append({**item, "cause": cause})

            if i % 10 == 0 or i == len(rejected_questions):
                print(f"  {i}/{len(rejected_questions)} classified...",
                      file=sys.stderr)

            time.sleep(RETRY_DELAY_S)

        results[label] = {
            "total_rejected": len(rejected_questions),
            "causes":         dict(causes),
            "pairs":          annotated,
        }

    return results


# ── Report printing ───────────────────────────────────────────────────────────

def print_fm2_report(fm2_results: dict) -> None:
    print(f"\n{'='*65}")
    print("PHASE 2 — FM2 REPHRASE vs TOPIC_SWITCH CLASSIFICATION")
    print(f"{'='*65}")

    labels = list(fm2_results.keys())

    for label, r in fm2_results.items():
        print(f"\n  {label}:")
        print(f"    Total pairs:        {r['total']}")
        print(f"    REPHRASE:           {r['rephrase']:3d}  ({r['rephrase_pct']:.1f}%)  "
              f"→ hit blocker: {r['rephrase_success']} ({r['rephrase_success_pct']:.1f}%)")
        print(f"    TOPIC_SWITCH:       {r['topic_switch']:3d}  ({r['switch_pct']:.1f}%)  "
              f"→ hit blocker: {r['switch_success']} ({r['switch_success_pct']:.1f}%)")
        if r.get('unknown', 0) > 0:
            print(f"    UNKNOWN:            {r['unknown']:3d}")

        # FM2a split
        if r.get('fm2a_total', 0) > 0:
            print(f"\n    FM2a (mid-conv rejections, task ultimately stopped):")
            print(f"      n={r['fm2a_total']}  "
                  f"REPHRASE={r['fm2a_rephrase']} ({r['fm2a_rephrase_pct']:.1f}%)"
                  f" hit={r['fm2a_rephrase_success']} ({r['fm2a_rephrase_success_pct']:.1f}%)  "
                  f"SWITCH={r['fm2a_switch']} ({r['fm2a_switch_pct']:.1f}%)"
                  f" hit={r['fm2a_switch_success']} ({r['fm2a_switch_success_pct']:.1f}%)")

        # FM2b split
        if r.get('fm2b_total', 0) > 0:
            print(f"\n    FM2b (rejections in tasks that continued past last rejection):")
            print(f"      n={r['fm2b_total']}  "
                  f"REPHRASE={r['fm2b_rephrase']} ({r['fm2b_rephrase_pct']:.1f}%)"
                  f" hit={r['fm2b_rephrase_success']} ({r['fm2b_rephrase_success_pct']:.1f}%)  "
                  f"SWITCH={r['fm2b_switch']} ({r['fm2b_switch_pct']:.1f}%)"
                  f" hit={r['fm2b_switch_success']} ({r['fm2b_switch_success_pct']:.1f}%)")

    # Comparison if two conditions
    if len(labels) == 2:
        a, b = labels
        ra, rb = fm2_results[a], fm2_results[b]
        print(f"\n  {'─'*62}")
        print(f"  {'':38}  {a:>8}  {b:>8}  {'Δ':>6}")
        print(f"  {'─'*62}")

        def row(name, key):
            va = ra.get(key, 0.0)
            vb = rb.get(key, 0.0)
            d  = vb - va
            sign = "+" if d >= 0 else ""
            print(f"  {name:<38}  {va:>8.1f}  {vb:>8.1f}  {sign}{d:>5.1f}")

        row("REPHRASE — all pairs (%)",       "rephrase_pct")
        row("TOPIC_SWITCH — all pairs (%)",   "switch_pct")
        row("REPHRASE hit blocker (%)",        "rephrase_success_pct")
        row("TOPIC_SWITCH hit blocker (%)",    "switch_success_pct")
        print(f"  {'─'*62}")
        row("REPHRASE — FM2a pairs (%)",      "fm2a_rephrase_pct")
        row("REPHRASE FM2a hit blocker (%)",  "fm2a_rephrase_success_pct")
        row("TOPIC_SWITCH — FM2a pairs (%)",  "fm2a_switch_pct")
        print(f"  {'─'*62}")
        row("REPHRASE — FM2b pairs (%)",      "fm2b_rephrase_pct")
        row("REPHRASE FM2b hit blocker (%)",  "fm2b_rephrase_success_pct")
        row("TOPIC_SWITCH — FM2b pairs (%)",  "fm2b_switch_pct")


def print_fm3_report(fm3_results: dict) -> None:
    print(f"\n{'='*65}")
    print("PHASE 2 — FM3 REJECTION CAUSE (LLM CLASSIFICATION)")
    print(f"{'='*65}")

    labels = list(fm3_results.keys())
    causes_order = ["MULTI_TOPIC", "SCHEMA_QUESTION", "TOO_VAGUE",
                    "ASSUMPTION_CONFIRM", "OTHER", "UNKNOWN"]

    for label, r in fm3_results.items():
        total = max(r["total_rejected"], 1)
        print(f"\n  {label}  ({r['total_rejected']} rejected questions):")
        for cause in causes_order:
            cnt = r["causes"].get(cause, 0)
            pct = cnt / total * 100
            bar = "█" * int(pct / 5)
            print(f"    {cause:<22} {cnt:4d}  ({pct:5.1f}%)  {bar}")

    # Comparison
    if len(labels) == 2:
        a, b = labels
        ra, rb = fm3_results[a], fm3_results[b]
        ta = max(ra["total_rejected"], 1)
        tb = max(rb["total_rejected"], 1)
        print(f"\n  {'─'*60}")
        print(f"  {'Cause':<22}  {a:>10}  {b:>10}  {'Δ':>8}")
        print(f"  {'─'*60}")
        for cause in causes_order:
            pa = ra["causes"].get(cause, 0) / ta * 100
            pb = rb["causes"].get(cause, 0) / tb * 100
            d  = pb - pa
            sign = "+" if d >= 0 else ""
            print(f"  {cause:<22}  {pa:>10.1f}  {pb:>10.1f}  {sign}{d:>7.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: LLM semantic classification of FM2 and FM3"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to phase2_pairs.json produced by deep_fm_analysis.py",
    )
    parser.add_argument(
        "--fm3-logs",
        nargs=2,
        action="append",
        metavar=("PATH", "LABEL"),
        default=[],
        help="ask_human_logs.json path and label for FM3 LLM classification. "
             "Repeat for multiple conditions.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key. Falls back to LITELLM_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LiteLLM model string (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional API base URL override.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save full annotated results to this JSON file.",
    )
    parser.add_argument(
        "--skip-fm2",
        action="store_true",
        help="Skip FM2 rephrase classification (only run FM3).",
    )
    parser.add_argument(
        "--skip-fm3",
        action="store_true",
        help="Skip FM3 LLM classification (only run FM2).",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("LITELLM_API_KEY", "")
    if not api_key:
        print("[ERROR] No API key. Set --api-key or LITELLM_API_KEY env var.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Retry delay: {RETRY_DELAY_S}s between calls")

    all_results: dict = {}

    # ── FM2 classification ────────────────────────────────────────────────────
    fm2_results: dict = {}
    if not args.skip_fm2:
        with open(args.pairs) as f:
            pairs_by_label: dict = json.load(f)

        total_pairs = sum(len(v) for v in pairs_by_label.values())
        est_cost    = total_pairs * 0.0005
        print(f"\nFM2 pairs to classify: {total_pairs} "
              f"(estimated cost: ~${est_cost:.3f})")

        fm2_results = run_fm2_classification(
            pairs_by_label, api_key, args.model, args.base_url
        )
        print_fm2_report(fm2_results)
        all_results["fm2"] = fm2_results

    # ── FM3 classification ────────────────────────────────────────────────────
    fm3_results: dict = {}
    if not args.skip_fm3 and args.fm3_logs:
        total_fm3 = 0
        for path_str, label in args.fm3_logs:
            logs = load_logs(path_str)
            cnt  = sum(
                1 for log in logs.values()
                for q in log.get("questions", [])
                if IRRELEVANT in q.get("response", "").lower()
            )
            total_fm3 += cnt
            print(f"  {label}: {cnt} rejected questions for FM3 classification")

        est_cost = total_fm3 * 0.0005
        print(f"\nFM3 questions to classify: {total_fm3} "
              f"(estimated cost: ~${est_cost:.3f})")

        confirm = input("Proceed with FM3 LLM classification? [y/N] ").strip().lower()
        if confirm == "y":
            fm3_results = run_fm3_classification(
                args.fm3_logs, api_key, args.model, args.base_url
            )
            print_fm3_report(fm3_results)
            all_results["fm3"] = fm3_results
        else:
            print("FM3 classification skipped.")
    elif not args.skip_fm3 and not args.fm3_logs:
        print("\nNo --fm3-logs provided. Skipping FM3 LLM classification.")
        print("Re-run with --fm3-logs to classify FM3 rejection causes.")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save_results and all_results:
        # Remove full pair lists for cleaner summary file
        summary = {}
        for task, res in all_results.items():
            summary[task] = {}
            for label, data in res.items():
                summary[task][label] = {
                    k: v for k, v in data.items() if k != "pairs"
                }
        with open(args.save_results, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")

    print()


if __name__ == "__main__":
    main()
