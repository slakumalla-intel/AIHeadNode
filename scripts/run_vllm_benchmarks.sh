#!/usr/bin/env bash
set -u

ROOT_DIR="$(pwd)"
RESULTS_DIR="$ROOT_DIR/vllm_model_benchmarking_results"
mkdir -p "$RESULTS_DIR"

MODEL_BASE=/mnt/datasets/huggingface/hub
LOG_FILE="$RESULTS_DIR/vllm_model_comparison.log"
: > "$LOG_FILE"

if [ ! -f "$SHAREGPT_DATASET_PATH" ]; then
  echo "ERROR: SHAREGPT_DATASET_PATH not set or file missing: '$SHAREGPT_DATASET_PATH'" >&2
  exit 1
fi
echo "Using ShareGPT dataset: $SHAREGPT_DATASET_PATH" | tee -a "$LOG_FILE"

resolve_model_dir() {
  local model_root="$1"
  local file
  file=$(find -L "$model_root/snapshots" -maxdepth 2 \
           \( -name config.json -o -name params.json \) 2>/dev/null | head -n 1)
  if [ -z "$file" ]; then
    file=$(find -L "$model_root" \
             \( -name config.json -o -name params.json \) 2>/dev/null | head -n 1)
  fi
  if [ -n "$file" ]; then
    dirname "$file"
  else
    echo "$model_root"
  fi
}

for MODEL_NAME in \
  "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  "meta-llama/Llama-2-7b-hf" \
  "Qwen/Qwen2.5-7B-Instruct" \
  "Qwen/Qwen2.5-14B-Instruct" \
  "meta-llama/Meta-Llama-3.1-70B-Instruct" \
  "meta-llama/Llama-3.3-70B-Instruct" \
  "Qwen/Qwen2.5-72B-Instruct" \
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
  "mistralai/Mixtral-8x7B-Instruct-v0.1" \
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
do

  case "$MODEL_NAME" in
    *"70B"*|*"72B"*|*"8x22B"*|*"8x7B"*)
      TP=4
      ;;
    *)
      TP=2
      ;;
  esac

  for C in 128; do
    MODEL_ROOT="${MODEL_BASE}/models--${MODEL_NAME//\//--}"
    LOCAL_MODEL="$(resolve_model_dir "$MODEL_ROOT")"
    TAG="$(echo "$MODEL_NAME" | tr '/:' '_')_${TP}tp_${C}c"

    echo "=== $MODEL_NAME | TP=$TP | concurrency=$C | MODEL=$LOCAL_MODEL ===" | tee -a "$LOG_FILE"

    if [ ! -f "$LOCAL_MODEL/config.json" ] && [ ! -f "$LOCAL_MODEL/params.json" ]; then
      echo "!!! SKIP: no config.json/params.json found under $MODEL_ROOT" | tee -a "$LOG_FILE"
      continue
    fi

    nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
      -o "$RESULTS_DIR/$TAG" \
      bash -lc '
        MODEL="$1"
        TP="$2"
        C="$3"
        LOG_FILE="$4"
        SHAREGPT_DATASET_PATH="$5"

        vllm serve "$MODEL" \
          --dtype bfloat16 \
          --tensor-parallel-size "$TP" \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.90 \
          --enforce-eager &
        SERVER_PID=$!

        sleep 180

        {
          echo "=== vLLM benchmark: $MODEL ==="
          echo "TP=$TP"
          echo "Concurrency=$C"
          echo "Timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

          vllm bench serve \
            --backend vllm \
            --model "$MODEL" \
            --dataset-name sharegpt \
            --dataset-path "$SHAREGPT_DATASET_PATH" \
            --num-prompts 2048 \
            --max-concurrency "$C" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,90,99
        } 2>&1 | tee -a "$LOG_FILE"

        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
      ' _ "$LOCAL_MODEL" "$TP" "$C" "$LOG_FILE" "$SHAREGPT_DATASET_PATH"

    pkill -f "vllm serve" 2>/dev/null
    sleep 10
  done
done
