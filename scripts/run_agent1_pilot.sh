#!/bin/bash
#SBATCH --job-name=hil_agent1_pilot
#SBATCH --account=class_cse59827694spring2026
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=hil_agent1_pilot_%j.out
#SBATCH --error=hil_agent1_pilot_%j.err

# Agent 1 pilot — 15 tasks x 3 passes
# Model: openai/Qwen3-32B (remapped to agent1_clarify_qwen3_32b.yaml)
# Metrics: Ask-F1, precision, recall

############################################
# Paths and Environment
############################################

export SIF="/data/sse/gaudi/containers/vllm-gaudi.sif"
export HF_TOKEN="YOUR_HF_TOKEN"

export HIL_DIR="/scratch/$USER/hil-bench"
export HF_CACHE="$HIL_DIR/hf_cache"
export HLOG="$HIL_DIR/habana_logs/agent1_pilot"

mkdir -p "$HF_CACHE" "$HLOG"

echo "HIL-Bench base: $HIL_DIR"
echo "Agent1 pilot: 15 tasks x 3 passes"
echo "Starting at $(date)"
hl-smi

############################################
# Kill old ports
############################################

fuser -k 8196/tcp || true

############################################
# Start vLLM server (Qwen3-32B)
############################################

apptainer exec --cleanenv \
  --bind "$HF_CACHE:/hf_cache" \
  --bind "$HLOG:/var/log/habana_logs" \
  --env HF_HOME=/hf_cache \
  --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
  --env TRANSFORMERS_CACHE=/hf_cache/hub \
  --env XDG_CACHE_HOME=/hf_cache \
  --env HABANA_VISIBLE_DEVICES=0,1 \
  --env HF_TOKEN="$HF_TOKEN" \
  --env HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  "$SIF" /bin/bash --noprofile --norc -c "
    vllm serve Qwen/Qwen3-32B \
      --device hpu --host 0.0.0.0 --port 8196 \
      --api-key local --served-model-name Qwen3-32B \
      --tensor-parallel-size 2 \
      --max-model-len 32768 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
  " > "$HLOG/server.log" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"
sleep 60

############################################
# Wait for vLLM server
############################################

echo "Waiting for vLLM server..."
while true; do
  RESPONSE=$(curl -s http://127.0.0.1:8196/v1/models -H "Authorization: Bearer local")
  if echo "$RESPONSE" | grep -q "Qwen3-32B"; then
    echo "Server ready."
    break
  fi
  echo "Still waiting..."
  sleep 30
done

############################################
# Python environment
############################################

cd "$HIL_DIR" || exit 1

if [ -f "$HIL_DIR/.env" ]; then
    set -a; source "$HIL_DIR/.env"; set +a
    echo "Loaded .env"
fi

if [ ! -f ".venv/bin/activate" ]; then
    python3.11 -m venv .venv
    source .venv/bin/activate
    uv sync
else
    source .venv/bin/activate
fi

SITE_PACKAGES=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/sitecustomize.py" << 'PYEOF'
try:
    import pysqlite3 as _pysqlite3
    import sys as _sys
    _sys.modules["sqlite3"] = _pysqlite3
except ImportError:
    pass
PYEOF

export OPENAI_API_KEY="local"
export LEDGER_BASE_PATH="$HIL_DIR/data/checklists/agent1_pilot"
mkdir -p "$LEDGER_BASE_PATH"
echo "LEDGER_BASE_PATH: $LEDGER_BASE_PATH"

DATA_PATH="data/micro_batch_instances/instances.json"

############################################
# Remap openai/Qwen3-32B -> agent1 config
############################################

MAPPING="config_mappings.yaml"
ORIGINAL_MAPPING="config_mappings.yaml.bak"
cp "$MAPPING" "$ORIGINAL_MAPPING"

sed -i "s|openai/Qwen3-32B: configs/sql/ask_sql_config_qwen3_32b.yaml|openai/Qwen3-32B: configs/sql/agent1_clarify_qwen3_32b.yaml|g" "$MAPPING"

echo "Config mapping after swap:"
grep "Qwen3-32B:" "$MAPPING" | grep -v "gateway\|agent1\|Qwen2" | head -2

############################################
# Run: Agent1 (15 tasks x 3 passes)
############################################

echo "=========================================="
echo "Agent1 run at $(date)"
echo "=========================================="

.venv/bin/hil sql "$DATA_PATH" \
    --model openai/Qwen3-32B \
    --passes 3 \
    --ask-human \
    --config-mapping "$MAPPING" \
    --judge-config judge_config.yaml \
    --output-dir "results/agent1_pilot/agent1"

echo "Agent1 complete at $(date)"

############################################
# Restore original mapping
############################################

cp "$ORIGINAL_MAPPING" "$MAPPING"
rm -f "$ORIGINAL_MAPPING"
echo "config_mappings.yaml restored."

############################################
# Cleanup
############################################

kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "All done at $(date)."
