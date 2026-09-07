#!/usr/bin/env bash
set -uo pipefail

# -----------------------------------------------------------------------------
# Multi-model vLLM KV-cache tier benchmark: GDDR/HBM vs DDR vs NVMe.
# Requires vLLM 0.28.0 and LMCache from .venv-vllm-updated.
# -----------------------------------------------------------------------------

VENV="/mnt/nvme1n1p1/venvs/.venv-vllm-updated"
MODEL_BASE="/mnt/datasets/huggingface/hub"
DATASET="/mnt/datasets_backup/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"

RESULTS_DIR="${RESULTS_DIR:-$PWD/vllm_kv_multimodel_results_$(date +%Y%m%d_%H%M%S)}"
LMCACHE_ROOT="/mnt/nvme1n1p1/lmcache_kv_multimodel"

HOST="127.0.0.1"
VLLM_PORT=8000
LMCACHE_PORT=5555

NUM_PROMPTS="${NUM_PROMPTS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"

GDDR_CONCURRENCY="${GDDR_CONCURRENCY:-128}"
DDR_CONCURRENCY="${DDR_CONCURRENCY:-128}"
NVME_CONCURRENCY="${NVME_CONCURRENCY:-128}"

DDR_L1_GB="${DDR_L1_GB:-80}"
NVME_L1_GB="${NVME_L1_GB:-20}"

# Override with: TIERS="gddr ddr" ./run_vllm_kv_multimodel_bench.sh
TIERS="${TIERS:-gddr ddr nvme}"

# Regex filter applied to the model label.
# Example: MODEL_FILTER='llama31_8b|qwen25_7b'
MODEL_FILTER="${MODEL_FILTER:-.*}"

VLLM_PID=""
LMCACHE_PID=""

source "$VENV/bin/activate"
export SHAREGPT_DATASET_PATH="$DATASET"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: required file not found: $1" >&2
    exit 1
  fi
}

log() {
  echo "[$(date -Is)] $*"
}

cleanup_services() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi

  if [ -n "$LMCACHE_PID" ] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
    kill -TERM "$LMCACHE_PID" 2>/dev/null || true
    wait "$LMCACHE_PID" 2>/dev/null || true
  fi

  pkill -TERM -f "vllm.entrypoints" 2>/dev/null || true
  pkill -TERM -f "EngineCore" 2>/dev/null || true
  pkill -TERM -f "Worker_TP" 2>/dev/null || true
  pkill -TERM -f "lmcache server" 2>/dev/null || true
  sleep 3

  pkill -KILL -f "vllm.entrypoints" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  pkill -KILL -f "Worker_TP" 2>/dev/null || true
  pkill -KILL -f "lmcache server" 2>/dev/null || true

  VLLM_PID=""
  LMCACHE_PID=""
  sleep 3
}

wait_for_vllm() {
  local pid="$1"
  local log_file="$2"
  local attempt

  for attempt in $(seq 1 300); do
    if curl -fsS "http://${HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: vLLM exited before becoming ready." >&2
      tail -n 160 "$log_file" >&2 || true
      return 1
    fi

    sleep 2
  done

  echo "ERROR: timed out waiting for vLLM on port ${VLLM_PORT}." >&2
  tail -n 160 "$log_file" >&2 || true
  return 1
}

wait_for_lmcache() {
  local pid="$1"
  local log_file="$2"
  local attempt

  for attempt in $(seq 1 120); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${LMCACHE_PORT}") 2>/dev/null; then
      exec 3>&- 3<&- || true
      return 0
    fi

    if grep -q "LMCache ZMQ cache server is running" "$log_file" 2>/dev/null; then
      return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: LMCache exited before becoming ready." >&2
      tail -n 160 "$log_file" >&2 || true
      return 1
    fi

    sleep 2
  done

  echo "ERROR: timed out waiting for LMCache on port ${LMCACHE_PORT}." >&2
  tail -n 160 "$log_file" >&2 || true
  return 1
}

resolve_model_dir() {
  local model_cache_name="$1"
  local model_root="${MODEL_BASE}/models--${model_cache_name//\//--}"
  local config_file=""

  if [ -d "$model_root/snapshots" ]; then
    config_file="$(find -L "$model_root/snapshots" -maxdepth 2 \
      \( -name config.json -o -name params.json \) -print 2>/dev/null | head -n 1)"
  fi

  if [ -z "$config_file" ] && [ -d "$model_root" ]; then
    config_file="$(find -L "$model_root" -maxdepth 3 \
      \( -name config.json -o -name params.json \) -print 2>/dev/null | head -n 1)"
  fi

  if [ -n "$config_file" ]; then
    dirname "$config_file"
  fi
}

start_lmcache() {
  local tier="$1"
  local l1_gb="$2"
  local case_dir="$3"
  local l2_adapter="${4:-}"
  local lmcache_log="$case_dir/lmcache_server.log"

  if [ -n "$l2_adapter" ]; then
    lmcache server \
      --host localhost \
      --port "$LMCACHE_PORT" \
      --l1-size-gb "$l1_gb" \
      --eviction-policy LRU \
      --l2-adapter "$l2_adapter" \
      > "$lmcache_log" 2>&1 &
  else
    lmcache server \
      --host localhost \
      --port "$LMCACHE_PORT" \
      --l1-size-gb "$l1_gb" \
      --eviction-policy LRU \
      > "$lmcache_log" 2>&1 &
  fi

  LMCACHE_PID=$!
  wait_for_lmcache "$LMCACHE_PID" "$lmcache_log"
  log "LMCache ready: tier=$tier pid=$LMCACHE_PID"
}

start_vllm() {
  local model_dir="$1"
  local tp="$2"
  local context_len="$3"
  local transfer_config="$4"
  local case_dir="$5"
  local serve_log="$case_dir/vllm_serve.log"

  local serve_command=(
    vllm serve "$model_dir"
    --host "$HOST"
    --port "$VLLM_PORT"
    --dtype bfloat16
    --tensor-parallel-size "$tp"
    --max-model-len "$context_len"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enforce-eager
  )

  if [ -n "$transfer_config" ]; then
    serve_command+=(
      --kv-transfer-config "$transfer_config"
    )
  fi

  printf '%q ' "${serve_command[@]}" > "$case_dir/serve_command.txt"
  printf '\n' >> "$case_dir/serve_command.txt"

  "${serve_command[@]}" > "$serve_log" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm "$VLLM_PID" "$serve_log"
  log "vLLM ready: pid=$VLLM_PID model=$(basename "$model_dir")"
}

run_benchmark() {
  local model_dir="$1"
  local concurrency="$2"
  local case_dir="$3"
  local bench_log="$case_dir/benchmark.log"

  vllm bench serve \
    --backend vllm \
    --base-url "http://${HOST}:${VLLM_PORT}" \
    --model "$model_dir" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$concurrency" \
    --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    2>&1 | tee "$bench_log"
}

metric_value() {
  local file="$1"
  local label="$2"

  grep -F "$label" "$file" 2>/dev/null | tail -n 1 | \
    sed -E 's/.*:[[:space:]]*//' | awk '{print $1}'
}

write_csv_row() {
  local tier="$1"
  local label="$2"
  local model_id="$3"
  local tp="$4"
  local context_len="$5"
  local concurrency="$6"
  local case_dir="$7"
  local bench_log="$case_dir/benchmark.log"
  local status="$8"

  local success="NA"
  local failed="NA"
  local req_s="NA"
  local out_tok_s="NA"
  local ttft_p50="NA"
  local tpot_p50="NA"
  local itl_p50="NA"
  local e2el_p50="NA"

  if [ -f "$bench_log" ]; then
    success="$(metric_value "$bench_log" "Successful requests" || true)"
    failed="$(metric_value "$bench_log" "Failed requests" || true)"
    req_s="$(metric_value "$bench_log" "Request throughput" || true)"
    out_tok_s="$(metric_value "$bench_log" "Output token throughput" || true)"
    ttft_p50="$(metric_value "$bench_log" "P50 TTFT" || true)"
    tpot_p50="$(metric_value "$bench_log" "P50 TPOT" || true)"
    itl_p50="$(metric_value "$bench_log" "P50 ITL" || true)"
    e2el_p50="$(metric_value "$bench_log" "P50 E2EL" || true)"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$tier" "$label" "$model_id" "$tp" "$context_len" "$concurrency" "$status" \
    "${success:-NA}" "${failed:-NA}" "${req_s:-NA}" "${out_tok_s:-NA}" \
    "${ttft_p50:-NA}" "${tpot_p50:-NA}" "${itl_p50:-NA}" \
    >> "$RESULTS_DIR/comparison.csv"
}

run_case() {
  local tier="$1"
  local label="$2"
  local model_id="$3"
  local tp="$4"
  local context_len="$5"
  local model_dir="$6"
  local concurrency=""
  local transfer_config=""
  local l2_adapter=""
  local cache_path=""
  local case_dir="$RESULTS_DIR/$tier/$label"
  local status="failed"

  mkdir -p "$case_dir"
  cleanup_services

  case "$tier" in
    gddr)
      concurrency="$GDDR_CONCURRENCY"
      ;;
    ddr)
      concurrency="$DDR_CONCURRENCY"
      transfer_config='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
      ;;
    nvme)
      concurrency="$NVME_CONCURRENCY"
      transfer_config='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
      cache_path="$LMCACHE_ROOT/nvme/$label"
      rm -rf "$cache_path"
      mkdir -p "$cache_path"
      l2_adapter="{\"type\":\"fs\",\"base_path\":\"$cache_path\"}"
      ;;
    *)
      echo "ERROR: unknown tier $tier" >&2
      return 1
      ;;
  esac

  {
    echo "tier=$tier"
    echo "label=$label"
    echo "model_id=$model_id"
    echo "model_dir=$model_dir"
    echo "tensor_parallel_size=$tp"
    echo "max_model_len=$context_len"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "num_prompts=$NUM_PROMPTS"
    echo "max_concurrency=$concurrency"
    echo "timestamp=$(date -Is)"
  } > "$case_dir/config.txt"

  log "Starting $tier / $label (TP=$tp, context=$context_len, concurrency=$concurrency)"

  if [ "$tier" = "ddr" ]; then
    start_lmcache "$tier" "$DDR_L1_GB" "$case_dir" || return 1
  elif [ "$tier" = "nvme" ]; then
    start_lmcache "$tier" "$NVME_L1_GB" "$case_dir" "$l2_adapter" || return 1
  fi

  start_vllm "$model_dir" "$tp" "$context_len" "$transfer_config" "$case_dir" || return 1

  if run_benchmark "$model_dir" "$concurrency" "$case_dir"; then
    if grep -qE "Successful requests:[[:space:]]*[1-9]" "$case_dir/benchmark.log"; then
      status="passed"
    else
      status="failed"
    fi
  fi

  if [ "$tier" = "nvme" ]; then
    {
      echo "cache_path=$cache_path"
      du -sh "$cache_path" 2>/dev/null || true
      find "$cache_path" -maxdepth 1 -type f 2>/dev/null | wc -l | \
        awk '{print "nvme_cache_file_count=" $1}'
    } > "$case_dir/nvme_usage.txt"
  fi

  echo "$status" > "$case_dir/status.txt"
  write_csv_row "$tier" "$label" "$model_id" "$tp" "$context_len" \
    "$concurrency" "$case_dir" "$status"

  cleanup_services
  log "Completed $tier / $label: $status"
  [ "$status" = "passed" ]
}

# label|HuggingFace repository id|TP|context length
MODELS=(
  "llama31_8b|meta-llama/Meta-Llama-3.1-8B-Instruct|2|8192"
  "llama2_7b|meta-llama/Llama-2-7b-hf|2|4096"
  "llama31_70b|meta-llama/Meta-Llama-3.1-70B-Instruct|4|8192"
  "llama33_70b|meta-llama/Llama-3.3-70B-Instruct|4|8192"
  "qwen25_7b|Qwen/Qwen2.5-7B-Instruct|2|8192"
  "qwen25_14b|Qwen/Qwen2.5-14B-Instruct|2|8192"
  "qwen25_72b|Qwen/Qwen2.5-72B-Instruct|4|8192"
  "deepseek_qwen_7b|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|2|8192"
  "deepseek_llama_70b|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|4|8192"
  "mixtral_8x7b|mistralai/Mixtral-8x7B-Instruct-v0.1|4|8192"
  "mixtral_8x22b|mistralai/Mixtral-8x22B-Instruct-v0.1|4|8192"
)

trap cleanup_services EXIT INT TERM

require_file "$DATASET"
mkdir -p "$RESULTS_DIR" "$LMCACHE_ROOT"

cat > "$RESULTS_DIR/comparison.csv" <<'CSV'
tier,model_label,model_id,tp,max_model_len,max_concurrency,status,successful_requests,failed_requests,request_throughput_req_s,output_token_throughput_tok_s,p50_ttft_ms,p50_tpot_ms,p50_itl_ms,p50_e2el_ms
CSV

log "Results: $RESULTS_DIR"
log "Models selected by regex: $MODEL_FILTER"
log "Tiers: $TIERS"

for tier in $TIERS; do
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r label model_id tp context_len <<< "$entry"

    if ! [[ "$label" =~ $MODEL_FILTER ]]; then
      continue
    fi

    model_dir="$(resolve_model_dir "$model_id")"
    if [ -z "$model_dir" ] || { [ ! -f "$model_dir/config.json" ] && [ ! -f "$model_dir/params.json" ]; }; then
      log "SKIP $label: local HuggingFace snapshot not found for $model_id"
      continue
    fi

    if ! run_case "$tier" "$label" "$model_id" "$tp" "$context_len" "$model_dir"; then
      log "FAILED $tier / $label; continuing to the next case."
    fi
  done
done

log "All selected cases complete."
log "Comparison CSV: $RESULTS_DIR/comparison.csv"
column -s, -t < "$RESULTS_DIR/comparison.csv" 2>/dev/null || cat "$RESULTS_DIR/comparison.csv"
