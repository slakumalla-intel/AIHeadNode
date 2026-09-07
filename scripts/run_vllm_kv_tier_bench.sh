#!/usr/bin/env bash
set -euo pipefail

VENV="/mnt/nvme1n1p1/venvs/.venv-vllm-updated"
MODEL="/mnt/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATASET="/mnt/datasets_backup/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"

RESULTS_DIR="${RESULTS_DIR:-$PWD/vllm_kv_tier_results_$(date +%Y%m%d_%H%M%S)}"
LM_CACHE_ROOT="/mnt/nvme1n1p1/lmcache_kv_tier_bench"
HOST="127.0.0.1"
VLLM_PORT=8000
LMCACHE_PORT=5555

TP=2
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.60
NUM_PROMPTS=2048

# Per-tier client concurrency. NVMe is kept lower because the LMCache
# fs L2 path is prone to a native segfault at very high concurrency.
GDDR_CONCURRENCY=128
DDR_CONCURRENCY=128
NVME_CONCURRENCY=32

# Host-DRAM cache for the DDR tier. No L2 adapter is configured.
DDR_L1_GB=80

# Small DRAM L1 intentionally forces overflow into the NVMe L2 tier.
NVME_L1_GB=20

VLLM_PID=""
LMCACHE_PID=""

source "$VENV/bin/activate"
export SHAREGPT_DATASET_PATH="$DATASET"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

require_file() {
  [ -f "$1" ] || { echo "ERROR: missing file: $1" >&2; exit 1; }
}

stop_tier_services() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi

  if [ -n "$LMCACHE_PID" ] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
    kill "$LMCACHE_PID" 2>/dev/null || true
    wait "$LMCACHE_PID" 2>/dev/null || true
  fi

  # vLLM spawns worker subprocesses, so remove any left behind by a failed run.
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  pkill -f "EngineCore" 2>/dev/null || true
  pkill -f "Worker_TP" 2>/dev/null || true
  pkill -f "lmcache server" 2>/dev/null || true

  VLLM_PID=""
  LMCACHE_PID=""
  sleep 5
}

wait_for_http() {
  local url="$1"
  local pid="$2"
  local log_file="$3"
  local attempt

  for attempt in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: service exited before becoming ready." >&2
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done

  echo "ERROR: timed out waiting for $url" >&2
  tail -n 120 "$log_file" >&2 || true
  return 1
}

wait_for_lmcache() {
  local pid="$1"
  local log_file="$2"
  local attempt

  for attempt in $(seq 1 120); do
    # The ZMQ server listening on 5555 is the real readiness signal.
    if (exec 3<>"/dev/tcp/127.0.0.1/${LMCACHE_PORT}") 2>/dev/null; then
      exec 3>&- 3<&- || true
      return 0
    fi
    if grep -qE "cache server is running" "$log_file" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: lmcache server exited before becoming ready." >&2
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done

  echo "ERROR: timed out waiting for lmcache on :${LMCACHE_PORT}" >&2
  tail -n 120 "$log_file" >&2 || true
  return 1
}

start_lmcache() {
  local tier="$1"
  local tier_dir="$2"
  local l1_gb="$3"
  local l2_adapter="${4:-}"
  local log_file="$tier_dir/lmcache_server.log"

  mkdir -p "$LM_CACHE_ROOT"

  if [ -n "$l2_adapter" ]; then
    lmcache server \
      --host localhost \
      --port "$LMCACHE_PORT" \
      --l1-size-gb "$l1_gb" \
      --eviction-policy LRU \
      --l2-adapter "$l2_adapter" \
      > "$log_file" 2>&1 &
  else
    lmcache server \
      --host localhost \
      --port "$LMCACHE_PORT" \
      --l1-size-gb "$l1_gb" \
      --eviction-policy LRU \
      > "$log_file" 2>&1 &
  fi

  LMCACHE_PID=$!
  wait_for_lmcache "$LMCACHE_PID" "$log_file"
  echo "LMCache ready for $tier: PID=$LMCACHE_PID"
}

start_vllm() {
  local tier="$1"
  local tier_dir="$2"
  local transfer_config="${3:-}"
  local log_file="$tier_dir/vllm_serve.log"

  local serve_args=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$VLLM_PORT"
    --dtype bfloat16
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enforce-eager
  )

  if [ -n "$transfer_config" ]; then
    serve_args+=(--kv-transfer-config "$transfer_config")
  fi

  "${serve_args[@]}" > "$log_file" 2>&1 &
  VLLM_PID=$!

  wait_for_http "http://${HOST}:${VLLM_PORT}/health" "$VLLM_PID" "$log_file"
  echo "vLLM ready for $tier: PID=$VLLM_PID"
}

run_benchmark() {
  local tier="$1"
  local tier_dir="$2"
  local concurrency="$3"
  local bench_log="$tier_dir/benchmark.log"

  echo "Running tier=$tier prompts=$NUM_PROMPTS concurrency=$concurrency" | tee "$bench_log"

  vllm bench serve \
    --backend vllm \
    --base-url "http://${HOST}:${VLLM_PORT}" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$concurrency" \
    --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    2>&1 | tee -a "$bench_log"
}

run_tier() {
  local tier="$1"
  local tier_dir="$RESULTS_DIR/$tier"
  local transfer_config=""
  local l2_adapter=""
  local concurrency=""

  mkdir -p "$tier_dir"
  stop_tier_services

  echo "============================================================" | tee "$tier_dir/run.log"
  echo "Starting tier: $tier" | tee -a "$tier_dir/run.log"
  echo "Started: $(date -Is)" | tee -a "$tier_dir/run.log"
  echo "============================================================" | tee -a "$tier_dir/run.log"

  case "$tier" in
    gddr)
      concurrency="$GDDR_CONCURRENCY"
      echo "KV tier: GPU HBM baseline (no LMCache server)" | tee -a "$tier_dir/run.log"
      ;;
    ddr)
      concurrency="$DDR_CONCURRENCY"
      echo "KV tier: host DRAM LMCache L1=${DDR_L1_GB} GiB" | tee -a "$tier_dir/run.log"
      start_lmcache "$tier" "$tier_dir" "$DDR_L1_GB"
      transfer_config='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
      ;;
    nvme)
      concurrency="$NVME_CONCURRENCY"
      echo "KV tier: LMCache L1=${NVME_L1_GB} GiB + NVMe fs L2" | tee -a "$tier_dir/run.log"
      rm -rf "$LM_CACHE_ROOT/nvme" 2>/dev/null || true
      mkdir -p "$LM_CACHE_ROOT/nvme"
      l2_adapter="{\"type\":\"fs\",\"base_path\":\"$LM_CACHE_ROOT/nvme\"}"
      start_lmcache "$tier" "$tier_dir" "$NVME_L1_GB" "$l2_adapter"
      transfer_config='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
      ;;
    *)
      echo "ERROR: unknown tier '$tier'" >&2
      return 1
      ;;
  esac

  start_vllm "$tier" "$tier_dir" "$transfer_config"
  run_benchmark "$tier" "$tier_dir" "$concurrency"

  if [ "$tier" = "nvme" ]; then
    du -sh "$LM_CACHE_ROOT/nvme" | tee "$tier_dir/nvme_usage.txt"
    find "$LM_CACHE_ROOT/nvme" -maxdepth 1 -type f | wc -l | \
      awk '{print "NVMe cache files: " $1}' | tee -a "$tier_dir/nvme_usage.txt"
  fi

  stop_tier_services
  echo "Completed tier: $tier at $(date -Is)" | tee -a "$tier_dir/run.log"
}

extract_metric() {
  # $1 = benchmark.log path, $2 = label to match
  local log_file="$1"
  local label="$2"
  grep -E "$label" "$log_file" 2>/dev/null | tail -n 1 \
    | grep -oE '[0-9]+(\.[0-9]+)?' | tail -n 1
}

write_summary() {
  local summary="$RESULTS_DIR/comparison_summary.txt"
  {
    echo "KV Cache Tier Comparison"
    echo "Model: $MODEL"
    echo "Prompts: $NUM_PROMPTS"
    echo "Generated: $(date -Is)"
    echo ""
    printf "%-6s %-8s %-14s %-14s %-14s %-14s %-14s\n" \
      "tier" "conc" "req/s" "out_tok/s" "TTFT_p50_ms" "TPOT_p50_ms" "E2EL_p50_ms"
    printf "%-6s %-8s %-14s %-14s %-14s %-14s %-14s\n" \
      "----" "----" "-----" "---------" "-----------" "-----------" "-----------"

    for tier in gddr ddr nvme; do
      local log_file="$RESULTS_DIR/$tier/benchmark.log"
      [ -f "$log_file" ] || continue
      local conc req_s out_s ttft tpot e2el
      conc=$(grep -oE "concurrency=[0-9]+" "$log_file" | tail -n1 | cut -d= -f2)
      req_s=$(extract_metric "$log_file" "Request throughput")
      out_s=$(extract_metric "$log_file" "Output token throughput")
      ttft=$(extract_metric "$log_file" "P50 TTFT")
      tpot=$(extract_metric "$log_file" "P50 TPOT")
      e2el=$(extract_metric "$log_file" "P50 E2EL")
      printf "%-6s %-8s %-14s %-14s %-14s %-14s %-14s\n" \
        "$tier" "${conc:-NA}" "${req_s:-NA}" "${out_s:-NA}" \
        "${ttft:-NA}" "${tpot:-NA}" "${e2el:-NA}"
    done
  } | tee "$summary"
}

trap stop_tier_services EXIT INT TERM

require_file "$MODEL/config.json"
require_file "$DATASET"
mkdir -p "$RESULTS_DIR"

echo "Results directory: $RESULTS_DIR"
echo "Model: $MODEL"
echo "Dataset: $DATASET"

for tier in gddr ddr nvme; do
  run_tier "$tier"
done

write_summary
echo "All tiers complete: $RESULTS_DIR"
