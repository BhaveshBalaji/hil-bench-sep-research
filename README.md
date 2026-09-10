# HIL-Bench SEP Research

Research code for studying help-seeking behavior in LLM agents on the [HIL-Bench](https://github.com/hilbenchauthors/hil-bench) SQL benchmark.

---

## Overview

LLM agents working on ambiguous SQL tasks systematically fail to ask clarifying questions — even when those questions are necessary to produce a correct query. This project diagnoses why this happens, proposes a targeted fix, and introduces a new architecture that addresses the residual failure.

**Three contributions:**

1. **Failure mode taxonomy** — trajectory-level analysis of 300 baseline runs identifies three root causes: silent omission (FM1, 55% of tasks), rejection abandonment (FM2, 19%), and schema-framed questions (FM3, 53% of rejections). LLM-judged Phase 2 analysis further decomposes FM2 into full-stop abandonment vs. continued engagement with rephrase/topic-switch classification.

2. **Structured Elicitation Prompting (SEP)** — a three-constraint prompt intervention that raises Ask-F1 from 26.3% to 48.4% (+22 pp) without any model training or fine-tuning. Ablation study identifies the resolution path constraint as the load-bearing element.

3. **Agent1 + persistent checklist** — addresses attention drift, where the agent identifies 4-5 ambiguities in early turns but forgets to ask about them as the context fills with tool observations. An external checklist tool writes ambiguities to a persistent file so they survive context window dilution.

---

## Results

| Condition | Ask-F1 | Tasks asking ≥1 Q | Tasks discovering ≥1 blocker |
|---|---|---|---|
| Baseline (authors') | 26.3% ± 2.9 | 45% | 26% |
| SEP (ours) | **48.4% ± 1.7** | 92.7% | 77% |
| Agent1 + checklist | in progress | — | — |

Cross-model validation on Llama-3.3-70B confirms the same failure patterns generalize beyond Qwen3-32B.

**Failure mode analysis (baseline vs SEP):**

| Metric | Baseline | SEP | Δ |
|---|---|---|---|
| FM1 Silent omission | 55.0% | 7.3% | −47.7 pp |
| FM2a Full stop after rejection | 19.0% | 15.7% | −3.3 pp |
| FM2b Continued after rejection | 0.3% | 13.3% | +13.0 pp |
| FM3 Rejection rate | 38.0% | 24.4% | −13.7 pp |
| Tasks discovering ≥1 blocker | 26.0% | 77.0% | +51.0 pp |

---

## Repository structure

```
configs/
  sql/                      Agent config YAMLs for each experimental condition
    ask_sql_config_qwen3_32b.yaml     SEP prompt (main result)
    agent1_clarify_qwen3_32b.yaml     Agent1 + checklist (progressive discovery)
    ablation_{A,B,C,D}_*.yaml         Ablation variants (one constraint removed each)
    baseline_llama33_70b.yaml         Cross-model baseline (Llama-3.3-70B)
    sep_llama33_70b.yaml              Cross-model SEP (Llama-3.3-70B)
  litellm/                  LiteLLM model registry files

tools/
  checklist/                Persistent ambiguity tracking tool for Agent1
    bin/checklist_add       Add a new ambiguity to the checklist
    bin/checklist_done      Mark an ambiguity as resolved
    bin/checklist_status    Show all items with status
    bin/checklist_pending   Show only unresolved items (call before submit_sql)
    config.yaml             SWE-agent tool declaration
    install.sh              chmod setup script

prompts/
  SEP.txt                       Full SEP system prompt
  ablation_{a,b,c,d}_prompt.txt Ablation variants
  authors_prompt_baseline.txt   Authors' original baseline prompt (for reference)

analysis/
  analyze_trajectories.py   Phase 1 failure mode analysis (FM1/FM2/FM3)
  deep_fm_analysis.py       Extended FM2 behavioral breakdown (FM2a/FM2b/FM2c)
  phase2_fm_analysis.py     LLM-judged FM2 rephrase/switch + FM3 cause classification
  cant_answer_audit.py      Per-pass can't-answer rate audit
  generate_fm_figure.py     Matplotlib failure mode distribution figure
  compute_pass_at_3.py      Pass@3 metric computation
  recompute_metrics.py      Recompute Ask-F1 from raw trajectories

hil_bench_patches/          Modified HIL-Bench files (drop into hil-bench repo)
  swe.py                    Orchestrator: blockers-file fix, skip_warmup
  server_utils.py           300s health check timeout for business_info server
  ask_human_server.py       --blockers-file support (fixes ARG_MAX on 100 tasks)
  business_info_server.py   pysqlite3 swap at import time
  ambiguity_ledger.py       Ledger data model for Agent1
  ambiguity_detector.py     Pre-run LLM ambiguity classifier
  ambiguity_gateway_server.py  Gateway server (deterministic submission enforcement)

scripts/
  run_agent1_pilot.sh       SLURM job script — Agent1 pilot run (15 tasks)

paper/
  main2.tex                 Full paper draft
  references.bib
  figures/fm_distribution.pdf   Failure mode pie chart figure
```

---

## Setup

Clone [HIL-Bench](https://github.com/hilbenchauthors/hil-bench) and apply patches:

```bash
git clone https://github.com/hilbenchauthors/hil-bench
cd hil-bench

# Apply patched core files
cp path/to/this/repo/hil_bench_patches/swe.py              hil_bench/scripts/swe.py
cp path/to/this/repo/hil_bench_patches/server_utils.py     hil_bench/utils/server_utils.py
cp path/to/this/repo/hil_bench_patches/ask_human_server.py hil_bench/ask_human_server.py
cp path/to/this/repo/hil_bench_patches/business_info_server.py hil_bench/business_info_server.py

# Copy Agent1 components
cp path/to/this/repo/hil_bench_patches/ambiguity_ledger.py      hil_bench/
cp path/to/this/repo/hil_bench_patches/ambiguity_detector.py    hil_bench/
cp path/to/this/repo/hil_bench_patches/ambiguity_gateway_server.py hil_bench/

# Copy configs and tools
cp -r path/to/this/repo/configs/sql/*.yaml configs/
cp -r path/to/this/repo/configs/litellm/*.json configs/
cp -r path/to/this/repo/tools/checklist SWE-agent/tools/checklist
cp path/to/this/repo/config_mappings.yaml .
```

Install dependencies and run SEP:

```bash
python3.11 -m venv .venv && source .venv/bin/activate && uv sync

.venv/bin/hil sql data/sql_100_instances/instances.json \
    --model openai/Qwen3-32B \
    --passes 3 \
    --ask-human \
    --config-mapping config_mappings.yaml \
    --judge-config judge_config.yaml \
    --output-dir results/sep
```

---

## Analysis

```bash
# Phase 1 — structural failure mode breakdown
python analysis/deep_fm_analysis.py \
    --logs "results/baseline/*/ask_human_logs.json" "Baseline" \
    --logs "results/sep/*/ask_human_logs.json" "SEP" \
    --total-tasks 100 \
    --save-phase2 phase2_pairs.json

# Phase 2 — LLM-judged FM2 rephrase/switch + FM3 cause classification
python analysis/phase2_fm_analysis.py \
    --pairs phase2_pairs.json \
    --model openrouter/meta-llama/llama-3.3-70b-instruct \
    --save-results phase2_results.json

# Audit can't-answer rate per pass
python analysis/cant_answer_audit.py \
    --logs "results/baseline/*/ask_human_logs.json" "Baseline" \
    --logs "results/sep/*/ask_human_logs.json" "SEP"

# Generate failure mode figure
python analysis/generate_fm_figure.py
```

---

## Checklist tool

The `tools/checklist` tool gives the agent persistent external memory for tracking ambiguities. It writes to a JSON file at `LEDGER_BASE_PATH/<task_id>/checklist.json`, which survives across all turns regardless of context window length.

| Command | Purpose |
|---|---|
| `checklist_add "description"` | Register a new ambiguity, returns ID |
| `checklist_done A1` | Mark A1 resolved after ask_human or get_business_info |
| `checklist_status` | Show all items |
| `checklist_pending` | Show only unresolved — call before submit_sql |

---

## Citation

Built on top of HIL-Bench:

```bibtex
@article{hilbench2025,
  title  = {HIL-Bench: Benchmarking Human-in-the-Loop SQL Agents},
  author = {HIL-Bench Authors},
  year   = {2025}
}
```

---

## License

This repository contains original research contributions only. The underlying HIL-Bench framework is © the HIL-Bench authors and is not redistributed here.
