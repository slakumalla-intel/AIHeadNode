#!/usr/bin/env bash
set -euo pipefail

# Concurrent two-model multi-user benchmark on four H100 GPUs.
# Llama uses GPUs 0,1 and Mixtral uses GPUs 2,3; each endpoint uses TP=2.
# A smaller fallback is used only if its requested FP8 model is unavailable
# locally or fails the vLLM readiness check.

VENV="${VENV:-/mnt/nvme1n1p1/venvs/.venv-vllm-updated}"
MODEL_BASE="${MODEL_BASE:-/mnt/datasets/huggingface/hub}"
DATASET="${SHAREGPT_DATASET_PATH:-/mnt/datasets_backup/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/vllm_fp8_concurrent_$(date +%Y%m%d_%H%M%S)}"

HOST="127.0.0.1"
LLAMA_PORT="${LLAMA_PORT:-8000}"
QWEN_PORT="${QWEN_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
CONCURRENCY_PER_MODEL="${CONCURRENCY_PER_MODEL:-32}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1200}"

LLAMA_REQUESTED="meta-llama/Llama-3.3-70B-Instruct"
LLAMA_FALLBACK="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_REQUESTED="Qwen/Qwen2.5-72B-Instruct"
QWEN_FALLBACK="Qwen/Qwen2.5-14B-Instruct"

LLAMA_PID=""
QWEN_PID=""

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

resolve_model_dir() {
  local model_id="$1"
  local model_root="$MODEL_BASE/models--${model_id//\//--}"
  local config_file=""

  if [ -d "$model_root/snapshots" ]; then
    config_file="$(find -L "$model_root/snapshots" -maxdepth 2 -name config.json -print 2>/dev/null | head -n 1)"
  fi
  if [ -z "$config_file" ] && [ -d "$model_root" ]; then
    config_file="$(find -L "$model_root" -maxdepth 3 -name config.json -print 2>/dev/null | head -n 1)"
  fi
  [ -n "$config_file" ] && dirname "$config_file"
}

stop_server_group() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  sleep 3
  kill -KILL -- "-$pid" 2>/dev/null || true
}

cleanup() {
  stop_server_group "$LLAMA_PID"
  stop_server_group "$QWEN_PID"
  pkill -TERM -f 'vllm.entrypoints|EngineCore|Worker_TP' 2>/dev/null || true
  sleep 3
  pkill -KILL -f 'vllm.entrypoints|EngineCore|Worker_TP' 2>/dev/null || true
}

wait_for_vllm() {
  local port="$1" pid="$2" serve_log="$3"
  local attempts=$((READY_TIMEOUT_SEC / 5))
  local attempt

  for attempt in $(seq 1 "$attempts"); do
    if curl -fsS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "vLLM on port $port exited before readiness."
      tail -n 160 "$serve_log" || true
      return 1
    fi
    sleep 5
  done

  log "Timed out after ${READY_TIMEOUT_SEC}s waiting for port $port."
  tail -n 160 "$serve_log" || true
  return 1
}

start_server() {
  local gpu_ids="$1" port="$2" model_dir="$3" quantization="$4" serve_log="$5"
  local command=(
    env "CUDA_VISIBLE_DEVICES=$gpu_ids"
    vllm serve "$model_dir"
    --host "$HOST"
    --port "$port"
    --dtype bfloat16
    --tensor-parallel-size 2
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enforce-eager
  )

  if [ "$quantization" = "fp8" ]; then
    command+=(--quantization fp8)
  fi

  printf '%q ' "${command[@]}" > "${serve_log%.log}_command.txt"
  printf '\n' >> "${serve_log%.log}_command.txt"
  setsid "${command[@]}" > "$serve_log" 2>&1 &
  printf '%s\n' "$!"
}

start_endpoint_with_fallback() {
  local label="$1" gpu_ids="$2" port="$3" requested="$4" fallback="$5"
  local case_dir="$RESULTS_DIR/$label"
  local requested_dir="" fallback_dir="" selected_dir="" selected_model="" mode="" pid=""

  mkdir -p "$case_dir"
  requested_dir="$(resolve_model_dir "$requested" || true)"

  if [ -n "$requested_dir" ]; then
    log "Starting requested FP8 model $requested on GPUs $gpu_ids, port $port."
    pid="$(start_server "$gpu_ids" "$port" "$requested_dir" fp8 "$case_dir/vllm_serve_requested.log")"
    if [ "$label" = "llama" ]; then LLAMA_PID="$pid"; else QWEN_PID="$pid"; fi
    if wait_for_vllm "$port" "$pid" "$case_dir/vllm_serve_requested.log"; then
      selected_dir="$requested_dir"
      selected_model="$requested"
      mode="requested_fp8"
    else
      stop_server_group "$pid"
    fi
  fi

  if [ -z "$selected_dir" ]; then
    fallback_dir="$(resolve_model_dir "$fallback" || true)"
    if [ -z "$fallback_dir" ]; then
      log "No runnable requested or fallback snapshot for $label."
      printf 'status=skipped\nrequested_model=%s\nfallback_model=%s\n' "$requested" "$fallback" > "$case_dir/status.txt"
      return 1
    fi
    log "Starting BF16 fallback $fallback on GPUs $gpu_ids, port $port."
    pid="$(start_server "$gpu_ids" "$port" "$fallback_dir" bf16 "$case_dir/vllm_serve_fallback.log")"
    if [ "$label" = "llama" ]; then LLAMA_PID="$pid"; else QWEN_PID="$pid"; fi
    wait_for_vllm "$port" "$pid" "$case_dir/vllm_serve_fallback.log"
    selected_dir="$fallback_dir"
    selected_model="$fallback"
    mode="fallback_bf16"
  fi

  {
    echo "requested_model=$requested"
    echo "selected_model=$selected_model"
    echo "selected_snapshot=$selected_dir"
    echo "mode=$mode"
    echo "gpu_ids=$gpu_ids"
    echo "port=$port"
    echo "tensor_parallel_size=2"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "num_prompts=$NUM_PROMPTS"
    echo "max_concurrency=$CONCURRENCY_PER_MODEL"
  } > "$case_dir/config.txt"
}

run_benchmark() {
  local label="$1"
  local port="$2"
  local case_dir="$RESULTS_DIR/$label"
  local model_dir
  model_dir="$(awk -F= '$1 == "selected_snapshot" { print $2 }' "$case_dir/config.txt")"

  vllm bench serve \
    --backend vllm \
    --base-url "http://${HOST}:${port}" \
    --model "$model_dir" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY_PER_MODEL" \
    --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    2>&1 | tee "$case_dir/benchmark.log"
}

trap cleanup EXIT INT TERM

[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found: $VENV" >&2; exit 1; }
[ -f "$DATASET" ] || { echo "ERROR: dataset not found: $DATASET" >&2; exit 1; }

source "$VENV/bin/activate"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

cleanup
mkdir -p "$RESULTS_DIR"
log "Results directory: $RESULTS_DIR"
log "Concurrent topology: Llama GPUs=0,1 port=$LLAMA_PORT; Qwen GPUs=2,3 port=$QWEN_PORT."
log "Each endpoint: TP=2, concurrency=$CONCURRENCY_PER_MODEL, prompts=$NUM_PROMPTS."

start_endpoint_with_fallback llama "0,1" "$LLAMA_PORT" "$LLAMA_REQUESTED" "$LLAMA_FALLBACK"
start_endpoint_with_fallback qwen "2,3" "$QWEN_PORT" "$QWEN_REQUESTED" "$QWEN_FALLBACK"

log "Both endpoints are ready. Starting both multi-user benchmarks concurrently."
run_benchmark llama "$LLAMA_PORT" &
LLAMA_BENCH_PID=$!
run_benchmark qwen "$QWEN_PORT" &
QWEN_BENCH_PID=$!

wait "$LLAMA_BENCH_PID" || true
wait "$QWEN_BENCH_PID" || true

for label in llama qwen; do
  case_dir="$RESULTS_DIR/$label"
  if [ -f "$case_dir/benchmark.log" ] && grep -qE 'Successful requests:[[:space:]]*[1-9]' "$case_dir/benchmark.log"; then
    echo "status=passed" >> "$case_dir/status.txt"
  else
    echo "status=failed" >> "$case_dir/status.txt"
  fi
done

log "Concurrent test complete. Results: $RESULTS_DIR"
find "$RESULTS_DIR" -name status.txt -print -exec cat {} \;
