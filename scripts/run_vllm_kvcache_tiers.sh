#!/usr/bin/env bash
set -u

ROOT_DIR="$(pwd)"
BASE_RESULTS="$ROOT_DIR/vllm_model_benchmarking_results"
MODEL_BASE=/mnt/datasets/huggingface/hub
NVME_KV_DIR="/mnt/nvme1n1p1/lmcache_kv"

TIERS=(nvme)
KV_OFF_GB=80
LMCACHE_PORT=5555
LMCACHE_L1_GB=20
NVME_L2_ADAPTER='{"type":"fs","base_path":"/mnt/nvme1n1p1/lmcache_kv"}'
LOAD_WAIT=180
NUM_PROMPTS=2048
CONCURRENCY=128

if [ -z "${SHAREGPT_DATASET_PATH:-}" ] || [ ! -f "$SHAREGPT_DATASET_PATH" ]; then
  echo "ERROR: SHAREGPT_DATASET_PATH not set or missing" >&2
  exit 1
fi

resolve_model_dir() {
  local model_root="$1" file
  file=$(find -L "$model_root/snapshots" -maxdepth 2 \( -name config.json -o -name params.json \) 2>/dev/null | head -n 1)
  if [ -z "$file" ]; then
    file=$(find -L "$model_root" \( -name config.json -o -name params.json \) 2>/dev/null | head -n 1)
  fi
  if [ -n "$file" ]; then dirname "$file"; else echo "$model_root"; fi
}

wait_for_port() {
  local port="$1" tries="${2:-60}" i
  for i in $(seq 1 "$tries"); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then exec 3>&- 3<&-; return 0; fi
    sleep 2
  done
  return 1
}

LMCACHE_SERVER_PID=""
start_lmcache_server() {
  local rdir="$1"
  mkdir -p "$NVME_KV_DIR"
  local slog="$rdir/lmcache_server.log"
  echo "Starting lmcache server -> $slog"
  lmcache server \
    --host localhost --port "$LMCACHE_PORT" \
    --l1-size-gb "$LMCACHE_L1_GB" \
    --eviction-policy LRU \
    --l2-adapter "$NVME_L2_ADAPTER" \
    > "$slog" 2>&1 &
  LMCACHE_SERVER_PID=$!
  if ! wait_for_port "$LMCACHE_PORT" 90; then
    echo "ERROR: lmcache server did not open :$LMCACHE_PORT" >&2
    return 1
  fi
}

stop_lmcache_server() {
  if [ -n "$LMCACHE_SERVER_PID" ]; then
    kill "$LMCACHE_SERVER_PID" 2>/dev/null
    wait "$LMCACHE_SERVER_PID" 2>/dev/null
    LMCACHE_SERVER_PID=""
  fi
  pkill -f "lmcache server" 2>/dev/null
  sleep 3
}

configure_tier() {
  KV_OFF_ARGS=""
  case "$1" in
    nvme)
      if ! python -c "import lmcache" 2>/dev/null; then
        echo "lmcache not installed" >&2
        return 1
      fi
      start_lmcache_server "$2" || return 1
      KV_OFF_ARGS="--kv-offloading-backend lmcache --kv-offloading-size ${KV_OFF_GB}"
      ;;
    *)
      return 1
      ;;
  esac
  return 0
}

MODELS=(
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "meta-llama/Llama-2-7b-hf"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "meta-llama/Meta-Llama-3.1-70B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
)

for KV_TIER in "${TIERS[@]}"; do
  RESULTS_DIR="$BASE_RESULTS/$KV_TIER"
  mkdir -p "$RESULTS_DIR"
  LOG_FILE="$RESULTS_DIR/vllm_model_comparison.log"
  : > "$LOG_FILE"

  if ! configure_tier "$KV_TIER" "$RESULTS_DIR"; then
    echo "Skipping $KV_TIER" | tee -a "$LOG_FILE"
    continue
  fi

  echo "########## KV TIER: $KV_TIER ##########" | tee -a "$LOG_FILE"

  for MODEL_NAME in "${MODELS[@]}"; do
    case "$MODEL_NAME" in
      *"70B"*|*"72B"*|*"8x22B"*|*"8x7B"*) TP=4 ;;
      *) TP=2 ;;
    esac

    MODEL_ROOT="${MODEL_BASE}/models--${MODEL_NAME//\//--}"
    LOCAL_MODEL="$(resolve_model_dir "$MODEL_ROOT")"
    TAG="${KV_TIER}_$(echo "$MODEL_NAME" | tr '/:' '_')_${TP}tp_${CONCURRENCY}c"

    echo "=== [$KV_TIER] $MODEL_NAME | TP=$TP | C=$CONCURRENCY | MODEL=$LOCAL_MODEL ===" | tee -a "$LOG_FILE"

    if [ ! -f "$LOCAL_MODEL/config.json" ] && [ ! -f "$LOCAL_MODEL/params.json" ]; then
      echo "!!! SKIP: config missing under $MODEL_ROOT" | tee -a "$LOG_FILE"
      continue
    fi

    nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
      -o "$RESULTS_DIR/$TAG" \
      bash -lc '
        MODEL="$1"; TP="$2"; C="$3"; LOG_FILE="$4"
        SHAREGPT_DATASET_PATH="$5"; KV_OFF_ARGS="$6"; LOAD_WAIT="$7"; NUM_PROMPTS="$8"

        vllm serve "$MODEL" \
          --dtype bfloat16 \
          --tensor-parallel-size "$TP" \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager \
          $KV_OFF_ARGS &
        SERVER_PID=$!

        sleep "$LOAD_WAIT"

        {
          echo "=== vLLM benchmark: $MODEL ==="
          echo "TP=$TP"; echo "Concurrency=$C"
          vllm bench serve \
            --backend vllm \
            --model "$MODEL" \
            --dataset-name sharegpt \
            --dataset-path "$SHAREGPT_DATASET_PATH" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$C" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,90,99
        } 2>&1 | tee -a "$LOG_FILE"

        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        sleep 30
      ' _ "$LOCAL_MODEL" "$TP" "$CONCURRENCY" "$LOG_FILE" \
          "$SHAREGPT_DATASET_PATH" "$KV_OFF_ARGS" "$LOAD_WAIT" "$NUM_PROMPTS"

    pkill -f "vllm serve" 2>/dev/null
    sleep 10
  done

  stop_lmcache_server
done

echo "All tiers complete"
