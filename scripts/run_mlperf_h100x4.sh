#!/usr/bin/env bash
# =============================================================================
# run_mlperf_h100x4.sh
#
# Orchestrates the full MLPerf inference and training benchmark suite
# (AI_HOSTCPU_ML_TEST_ID_001–021) on a system with 4× H100 SXM5 GPUs.
#
# Usage:
#   bash scripts/run_mlperf_h100x4.sh [OPTIONS]
#
# Options:
#   --data-dir  DIR    Dataset root directory     (default: <repo>/datasets)
#   --model-dir DIR    Model root directory        (default: <repo>/datasets)
#   --output-dir DIR   Results / KPI output dir    (default: artifacts/mlperf_results)
#   --scenario  SCN    LoadGen scenario             (default: Offline)
#   --only IDs         Comma-separated test IDs     e.g. --only 001,006,014
#   --dry-run          Print commands without executing
#   --no-vllm-server   Skip launching vLLM serve steps (use an already-running server)
#   --vllm-port PORT   Port vLLM server listens on  (default: 8000)
#   --vllm-host HOST   Host to target for benchmarking (default: localhost)
#   --tp-size N        Tensor parallel size          (default: 4)
#   --num-gpus N       Number of GPUs for torchrun   (default: 4)
#   --timeout-sec N    Per-command timeout (seconds) (default: 7200)
#   --kpi-file FILE    KPI JSONL append target        (default: artifacts/mlperf_suite_kpis.jsonl)
#
# Environment:
#   HF_TOKEN          HuggingFace access token for gated models
#   MLPERF_INFERENCE_DIR  Path to MLCommons inference repo (default: /root/workloads/mlcommons/inference)
#   MLPERF_TRAINING_DIR   Path to MLCommons training repo  (default: /root/workloads/mlcommons/training)
#
# KPI output:
#   Results are appended to --kpi-file in JSONL format and also available
#   in ${OUTPUT_DIR}/<case_id>/summary.json after each run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DATASET_ROOT="${REPO_ROOT}/datasets"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR="${MLPERF_DATA_DIR:-${DEFAULT_DATASET_ROOT}}"
MODEL_DIR="${MLPERF_MODEL_DIR:-${DEFAULT_DATASET_ROOT}}"
OUTPUT_DIR="artifacts/mlperf_results"
SCENARIO="Offline"
ONLY_IDS=""
DRY_RUN=false
NO_VLLM_SERVER=false
VLLM_PORT=8000
VLLM_HOST="localhost"
TP_SIZE=4
NUM_GPUS=4
TIMEOUT_SEC=7200
KPI_FILE="artifacts/mlperf_suite_kpis.jsonl"
HF_TOKEN="${HF_TOKEN:-}"

INFERENCE_DIR="${MLPERF_INFERENCE_DIR:-/root/workloads/mlcommons/inference}"
TRAINING_DIR="${MLPERF_TRAINING_DIR:-/root/workloads/mlcommons/training}"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)        DATA_DIR="$2";          shift 2 ;;
    --model-dir)       MODEL_DIR="$2";         shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2";        shift 2 ;;
    --scenario)        SCENARIO="$2";          shift 2 ;;
    --only)            ONLY_IDS="$2";          shift 2 ;;
    --dry-run)         DRY_RUN=true;           shift   ;;
    --no-vllm-server)  NO_VLLM_SERVER=true;    shift   ;;
    --vllm-port)       VLLM_PORT="$2";         shift 2 ;;
    --vllm-host)       VLLM_HOST="$2";         shift 2 ;;
    --tp-size)         TP_SIZE="$2";           shift 2 ;;
    --num-gpus)        NUM_GPUS="$2";          shift 2 ;;
    --timeout-sec)     TIMEOUT_SEC="$2";       shift 2 ;;
    --kpi-file)        KPI_FILE="$2";          shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1   ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Global variable to track the current vLLM server PID
CURRENT_VLLM_PID=""

cleanup_all() {
  local exit_code=$?
  log "Cleanup: Terminating all child processes and vLLM workers..."
  
  # Kill the tracked vLLM server process if running
  if [[ -n "$CURRENT_VLLM_PID" ]] && kill -0 "$CURRENT_VLLM_PID" 2>/dev/null; then
    kill "$CURRENT_VLLM_PID" 2>/dev/null || true
  fi
  
  # Aggressively kill all vLLM-related processes
  kill_vllm "$VLLM_PORT"
  
  # Kill any remaining vLLM processes
  pkill -9 -f "VLLM::Worker" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
  pkill -9 -f "vllm.benchmarks.benchmark_serving" 2>/dev/null || true
  pkill -9 -f "bench latency" 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null || true
  
  # Release GPU memory via nvidia-smi if available
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "Cleanup: Releasing GPU memory..."
    nvidia-smi -i 0,1,2,3 -pm 1 2>/dev/null || true
  fi
  
  exit "$exit_code"
}

# Register cleanup handler for normal exit and signals
trap cleanup_all EXIT INT TERM

run() {
  if [[ "$DRY_RUN" == "true" ]]; then echo "[DRY-RUN] $*"
  else echo "[RUN] $*"; "$@"; fi
}

should_run() {
  local tid="$1"
  [[ -z "$ONLY_IDS" ]] && return 0
  local norm
  norm=$(printf "%03d" "${tid##*_}" 2>/dev/null || echo "$tid")
  echo "$ONLY_IDS" | tr ',' '\n' | grep -qiE "(AI_HOSTCPU_ML_TEST_ID_0*${norm#0*}|0*${norm#0*})"
}

log()   { echo "$(date '+%Y-%m-%d %H:%M:%S')  [INFO]   $*"; }
warn()  { echo "$(date '+%Y-%m-%d %H:%M:%S')  [WARN]   $*" >&2; }
error() { echo "$(date '+%Y-%m-%d %H:%M:%S')  [ERROR]  $*" >&2; }

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

record_result() {
  local case_id="$1" status="$2" note="${3:-}"
  local ts; ts=$(date '+%Y-%m-%dT%H:%M:%S')
  mkdir -p "$(dirname "$KPI_FILE")"
  echo "{\"timestamp\":\"${ts}\",\"case_id\":\"${case_id}\",\"metrics\":{\"mode\":\"execution_script\",\"status\":\"${status}\",\"note\":\"${note}\"}}" \
    >> "$KPI_FILE"
  case "$status" in
    passed)  ((PASS_COUNT++)); log  "  PASS  ${case_id}  ${note}" ;;
    failed)  ((FAIL_COUNT++)); error "  FAIL  ${case_id}  ${note}" ;;
    skipped) ((SKIP_COUNT++)); warn  "  SKIP  ${case_id}  ${note}" ;;
  esac
}

wait_for_vllm() {
  local host="$1" port="$2" max_wait="${3:-120}"
  local elapsed=0
  log "Waiting for vLLM server at http://${host}:${port}/health ..."
  while ! curl -sf "http://${host}:${port}/health" &>/dev/null; do
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed >= max_wait )); then
      error "vLLM server did not become ready within ${max_wait}s"
      return 1
    fi
  done
  log "vLLM server ready (waited ${elapsed}s)"
}

kill_vllm() {
  local port="$1"
  local pid
  
  # Kill the main vLLM server by port
  pid=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [[ -n "$pid" ]]; then
    log "Stopping vLLM server (PID ${pid})..."
    # Kill process group to catch all subprocesses
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      sleep 2
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  
  # Kill all vLLM worker processes (TP0, TP1, etc.)
  local worker_pids
  worker_pids=$(pgrep -f "VLLM::Worker" 2>/dev/null || true)
  if [[ -n "$worker_pids" ]]; then
    log "Terminating vLLM worker processes..."
    while read -r wpid; do
      if [[ -n "$wpid" ]]; then
        kill -TERM "$wpid" 2>/dev/null || true
      fi
    done <<< "$worker_pids"
    sleep 2
    worker_pids=$(pgrep -f "VLLM::Worker" 2>/dev/null || true)
    while read -r wpid; do
      if [[ -n "$wpid" ]]; then
        kill -KILL "$wpid" 2>/dev/null || true
      fi
    done <<< "$worker_pids"
  fi
  
  # Kill any vllm.entrypoints processes
  local entry_pids
  entry_pids=$(pgrep -f "vllm.entrypoints" 2>/dev/null || true)
  if [[ -n "$entry_pids" ]]; then
    log "Terminating vLLM entrypoint processes..."
    while read -r epid; do
      if [[ -n "$epid" ]]; then
        kill -TERM "$epid" 2>/dev/null || true
      fi
    done <<< "$entry_pids"
    sleep 1
    epids=$(pgrep -f "vllm.entrypoints" 2>/dev/null || true)
    while read -r epid; do
      if [[ -n "$epid" ]]; then
        kill -KILL "$epid" 2>/dev/null || true
      fi
    done <<< "$entry_pids"
  fi
}

run_with_timeout() {
  local timeout_s="$1"; shift
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN timeout=${timeout_s}s] $*"
    return 0
  fi
  timeout "$timeout_s" "$@"
}

run mkdir -p "$OUTPUT_DIR"
run mkdir -p "$(dirname "$KPI_FILE")"

log "========================================================="
log "  MLPerf H100×4 Benchmark Runner"
log "  Data dir   : $DATA_DIR"
log "  Model dir  : $MODEL_DIR"
log "  Output dir : $OUTPUT_DIR"
log "  Scenario   : $SCENARIO"
log "  GPUs       : $NUM_GPUS  |  TP : $TP_SIZE"
log "  KPI file   : $KPI_FILE"
log "========================================================="

# =============================================================================
# Inline KPI extractor  (bash; full extraction is done by the Python test suite)
# =============================================================================
extract_vllm_throughput() {
  # Finds the last "tokens/s" value in a log file or stdin
  grep -oP 'Throughput:\s*[\d.]+\s*requests/s,\s*\K[\d.]+(?=\s*(?:output\s+)?tokens/s)' \
    2>/dev/null | tail -1 || echo "N/A"
}

extract_loadgen_qps() {
  grep -oP 'samples_per_second\s*[=:]\s*\K[\d.]+' 2>/dev/null | tail -1 || echo "N/A"
}

extract_training_throughput() {
  grep -oiP '[\d.]+\s*(?=\s+samples?/s(?:ec)?)' 2>/dev/null | tail -1 || echo "N/A"
}

# =============================================================================
# LLM Inference helpers  (IDs 001–005)
# =============================================================================

run_vllm_benchmark() {
  local case_id="$1" model="$2" extra_serve_args="$3" extra_bench_args="$4"
  local out_dir="${OUTPUT_DIR}/${case_id}"
  local log_serve="${out_dir}/vllm_serve.log"
  local log_bench="${out_dir}/vllm_bench.log"
  run mkdir -p "$out_dir"

  if [[ "$NO_VLLM_SERVER" == "false" ]]; then
    # Kill any existing server on the target port
    kill_vllm "$VLLM_PORT" || true
    log "${case_id}: Starting vLLM server for ${model}..."
    # shellcheck disable=SC2086
    run nohup vllm serve "$model" \
      --tensor-parallel-size "$TP_SIZE" \
      --gpu-memory-utilization 0.90 \
      --port "$VLLM_PORT" \
      $extra_serve_args \
      >"$log_serve" 2>&1 &
    CURRENT_VLLM_PID=$!
    wait_for_vllm "$VLLM_HOST" "$VLLM_PORT" 300
  fi

  log "${case_id}: Running vLLM benchmark_serving..."
  # shellcheck disable=SC2086
  if run_with_timeout "$TIMEOUT_SEC" python3 -m vllm.benchmarks.benchmark_serving \
      --backend vllm \
      --model "$model" \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --percentiles 50,90,99 \
      $extra_bench_args \
      >"$log_bench" 2>&1; then
    local tps; tps=$(extract_vllm_throughput <"$log_bench")
    record_result "$case_id" "passed" "throughput_tok_per_sec=${tps}"
    log "${case_id}: Throughput = ${tps} tokens/s (full KPIs in ${log_bench})"
  else
    record_result "$case_id" "failed" "benchmark_serving exited non-zero; see ${log_bench}"
  fi

  if [[ "$NO_VLLM_SERVER" == "false" ]]; then
    log "${case_id}: Cleanup - stopping vLLM server..."
    kill_vllm "$VLLM_PORT" || true
    sleep 2
    # Force cleanup of any remaining workers
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
    CURRENT_VLLM_PID=""
  fi
}

# =============================================================================
# LoadGen inference helper  (IDs 006–013)
# =============================================================================

run_loadgen_benchmark() {
  local case_id="$1" run_script="$2"
  local out_dir="${OUTPUT_DIR}/${case_id}"
  local log_file="${out_dir}/loadgen.log"
  run mkdir -p "$out_dir"
  log "${case_id}: Running LoadGen benchmark..."
  if run_with_timeout "$TIMEOUT_SEC" bash -c "$run_script" >"$log_file" 2>&1; then
    local qps; qps=$(extract_loadgen_qps <"$log_file")
    record_result "$case_id" "passed" "samples_per_sec=${qps}"
    log "${case_id}: Throughput = ${qps} samples/s (full results in ${log_file})"
  else
    record_result "$case_id" "failed" "see ${log_file}"
  fi
}

# =============================================================================
# Training helper  (IDs 014–021)
# =============================================================================

run_training_job() {
  local case_id="$1" cmd="$2"
  local out_dir="${OUTPUT_DIR}/${case_id}"
  local log_file="${out_dir}/train.log"
  run mkdir -p "$out_dir"
  log "${case_id}: Launching training job..."
  # shellcheck disable=SC2086
  if run_with_timeout "$TIMEOUT_SEC" bash -c "$cmd" >"$log_file" 2>&1; then
    local tps; tps=$(extract_training_throughput <"$log_file")
    record_result "$case_id" "passed" "samples_per_sec=${tps}"
    log "${case_id}: Training throughput = ${tps} samples/s (log: ${log_file})"
  else
    record_result "$case_id" "failed" "see ${log_file}"
  fi
}

# =============================================================================
# ID 001 – Llama 3.1 405B  (text generation / long context)
# =============================================================================
if should_run "001"; then
  run_vllm_benchmark \
    "AI_HOSTCPU_ML_TEST_ID_001" \
    "meta-llama/Llama-3.1-405B-Instruct" \
    "--quantization fp8 --max-model-len 131072 --enforce-eager --trust-remote-code" \
    "--dataset-name random --random-input-len 8192 --random-output-len 2048 --num-prompts 200"
fi

# =============================================================================
# ID 002 – DeepSeek-R1  (reasoning)
# =============================================================================
if should_run "002"; then
  run_vllm_benchmark \
    "AI_HOSTCPU_ML_TEST_ID_002" \
    "deepseek-ai/DeepSeek-R1" \
    "--dtype bfloat16 --max-model-len 32768 --trust-remote-code" \
    "--dataset-name random --random-input-len 2048 --random-output-len 4096 --num-prompts 200"
fi

# =============================================================================
# ID 003 – Mixtral 8×7B  (MoE text generation)
# =============================================================================
if should_run "003"; then
  SHAREGPT="${DATA_DIR}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
  BENCH_ARGS="--dataset-name sharegpt --num-prompts 1000"
  [[ -f "$SHAREGPT" ]] && BENCH_ARGS="$BENCH_ARGS --dataset-path ${SHAREGPT}"
  run_vllm_benchmark \
    "AI_HOSTCPU_ML_TEST_ID_003" \
    "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    "--max-model-len 32768" \
    "$BENCH_ARGS"
fi

# =============================================================================
# ID 004 – Llama 3.1 8B  (summarization)
# =============================================================================
if should_run "004"; then
  SHAREGPT="${DATA_DIR}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
  BENCH_ARGS="--dataset-name sharegpt --num-prompts 1000"
  [[ -f "$SHAREGPT" ]] && BENCH_ARGS="$BENCH_ARGS --dataset-path ${SHAREGPT}"
  run_vllm_benchmark \
    "AI_HOSTCPU_ML_TEST_ID_004" \
    "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    "--max-model-len 8192" \
    "$BENCH_ARGS"
fi

# =============================================================================
# ID 005 – Llama 2 70B  (Q&A)
# =============================================================================
if should_run "005"; then
  SHAREGPT="${DATA_DIR}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
  BENCH_ARGS="--dataset-name sharegpt --num-prompts 1000"
  [[ -f "$SHAREGPT" ]] && BENCH_ARGS="$BENCH_ARGS --dataset-path ${SHAREGPT}"
  run_vllm_benchmark \
    "AI_HOSTCPU_ML_TEST_ID_005" \
    "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    "--dtype bfloat16 --max-model-len 4096" \
    "$BENCH_ARGS"
fi

# =============================================================================
# ID 006 – ResNet-50 v1.5
# =============================================================================
if should_run "006"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_006" "
    python3 ${INFERENCE_DIR}/vision/classification_and_detection/run_local.sh \
      pytorch resnet50 gpu \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/imagenet \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_006 \
      --count 50000 && \
    python3 ${INFERENCE_DIR}/vision/classification_and_detection/tools/accuracy-imagenet.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_006/mlperf_log_accuracy.json \
      --imagenet-val-file ${DATA_DIR}/imagenet/val_map.txt
  "
fi

# =============================================================================
# ID 007 – RetinaNet
# =============================================================================
if should_run "007"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_007" "
    python3 ${INFERENCE_DIR}/vision/classification_and_detection/run_local.sh \
      pytorch retinanet gpu \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/openimages \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_007 \
      --count 24781 && \
    python3 ${INFERENCE_DIR}/vision/classification_and_detection/tools/accuracy-openimages.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_007/mlperf_log_accuracy.json \
      --openimages-dir ${DATA_DIR}/openimages
  "
fi

# =============================================================================
# ID 008 – 3D U-Net KiTS19
# =============================================================================
if should_run "008"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_008" "
    python3 ${INFERENCE_DIR}/medical_imaging/3d-unet-kits19/run_local.sh \
      pytorch gpu \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/kits19 \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_008 \
      --count 42 && \
    python3 ${INFERENCE_DIR}/medical_imaging/3d-unet-kits19/accuracy-kits.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_008/mlperf_log_accuracy.json \
      --label-dir ${DATA_DIR}/kits19
  "
fi

# =============================================================================
# ID 009 – RNN-T
# =============================================================================
if should_run "009"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_009" "
    python3 ${INFERENCE_DIR}/speech_recognition/rnnt/run_local.sh \
      pytorch rnnt gpu \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/librispeech \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_009 \
      --count 2513 && \
    python3 ${INFERENCE_DIR}/speech_recognition/rnnt/accuracy.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_009/mlperf_log_accuracy.json \
      --dataset-path ${DATA_DIR}/librispeech \
      --manifest ${DATA_DIR}/librispeech/librispeech-dev-clean-wav.json
  "
fi

# =============================================================================
# ID 010 – Whisper
# =============================================================================
if should_run "010"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_010" "
    python3 ${INFERENCE_DIR}/speech_recognition/whisper/run_local.sh \
      pytorch gpu \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/librispeech \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_010 \
      --model openai/whisper-large-v2 \
      --count 2620 && \
    python3 ${INFERENCE_DIR}/speech_recognition/whisper/accuracy.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_010/mlperf_log_accuracy.json \
      --dataset-path ${DATA_DIR}/librispeech
  "
fi

# =============================================================================
# ID 011 – DLRM / DLRM-DCNv2
# =============================================================================
if should_run "011"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_011" "
    python3 ${INFERENCE_DIR}/recommendation/dlrm_v2/pytorch/run.py \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/criteo \
      --model-path ${MODEL_DIR}/dlrm/model_weights \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_011 \
      --num-samples 204800 --count 204800 --num-processes ${NUM_GPUS} && \
    python3 ${INFERENCE_DIR}/recommendation/dlrm_v2/pytorch/accuracy.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_011/mlperf_log_accuracy.json \
      --day-23-file ${DATA_DIR}/criteo/day_23
  "
fi

# =============================================================================
# ID 012 – SDXL 1.0
# =============================================================================
if should_run "012"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_012" "
    python3 ${INFERENCE_DIR}/text_to_image/run_local.sh \
      pytorch gpu \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/coco-sdxl \
      --model-path ${MODEL_DIR}/sdxl \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_012 \
      --count 5000 --num-inf-steps 20 && \
    python3 ${INFERENCE_DIR}/text_to_image/tools/accuracy-clip.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_012/mlperf_log_accuracy.json \
      --dataset-dir ${DATA_DIR}/coco-sdxl
  "
fi

# =============================================================================
# ID 013 – RGAT
# =============================================================================
if should_run "013"; then
  run_loadgen_benchmark "AI_HOSTCPU_ML_TEST_ID_013" "
    python3 ${INFERENCE_DIR}/graph/R-GAT/run_local.sh \
      pytorch gpu \
      --scenario ${SCENARIO} \
      --dataset-path ${DATA_DIR}/igb-het \
      --model-path ${MODEL_DIR}/rgat \
      --output ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_013 \
      --count 9035 && \
    python3 ${INFERENCE_DIR}/graph/R-GAT/accuracy.py \
      --mlperf-accuracy-file ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_013/mlperf_log_accuracy.json
  "
fi

# =============================================================================
# ID 014 – GPT-J summarization training
# =============================================================================
if should_run "014"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_014" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/language_model/gpt-j/run_training.py \
      --model-name EleutherAI/gpt-j-6b \
      --dataset-path ${DATA_DIR}/cnn_dailymail \
      --output-dir ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_014 \
      --target-rouge1 42.9245 \
      --per-device-batch-size 8 \
      --gradient-accumulation-steps 4 \
      --fp16 \
      --log-interval 100
  "
fi

# =============================================================================
# ID 015 – BERT-Large SQuAD
# =============================================================================
if should_run "015"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_015" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/language_model/pytorch/bert/run_squad.py \
      --bert_model bert-large-uncased \
      --do_train \
      --train_file ${DATA_DIR}/squad/train-v1.1.json \
      --predict_file ${DATA_DIR}/squad/dev-v1.1.json \
      --train_batch_size 48 \
      --learning_rate 3e-5 \
      --num_train_epochs 2.0 \
      --max_seq_length 384 \
      --doc_stride 128 \
      --output_dir ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_015 \
      --fp16
  "
fi

# =============================================================================
# ID 016 – 3D U-Net training
# =============================================================================
if should_run "016"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_016" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/image_segmentation/pytorch/train.py \
      --data-dir ${DATA_DIR}/kits19 \
      --epochs 4000 \
      --batch-size 2 \
      --optimizer adam \
      --learning-rate 0.0002 \
      --amp \
      --log-every 50
  "
fi

# =============================================================================
# ID 017 – Stable Diffusion v2 training
# =============================================================================
if should_run "017"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_017" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/stable_diffusion/main.py \
      --base ${TRAINING_DIR}/stable_diffusion/configs/train_01x04x08.yaml \
      --actual_resume ${MODEL_DIR}/sd/v2-1_512-ema-pruned.ckpt \
      --data_dir ${DATA_DIR}/laion400m \
      --log-every-n-steps 100
  "
fi

# =============================================================================
# ID 018 – BERT-Large NLP pre-training (phase 1 + 2)
# =============================================================================
if should_run "018"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_018" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/language_model/pytorch/bert/run_pretraining.py \
      --bert_model bert-large-uncased --do_train \
      --train_file ${DATA_DIR}/bert/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --output_dir ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_018/phase1 \
      --max_seq_length 128 --train_batch_size 512 \
      --max_steps 7038 --learning_rate 6e-3 --warmup_proportion 0.2843 --fp16 && \
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/language_model/pytorch/bert/run_pretraining.py \
      --bert_model bert-large-uncased --do_train \
      --train_file ${DATA_DIR}/bert/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
      --output_dir ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_018/phase2 \
      --max_seq_length 512 --train_batch_size 128 \
      --phase1_end_step 7038 --max_steps 107538 --learning_rate 4e-3 --fp16
  "
fi

# =============================================================================
# ID 019 – GPT-3 175B pre-training
# =============================================================================
if should_run "019"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_019" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/large_language_model/megatron-lm/pretrain_gpt.py \
      --tensor-model-parallel-size ${TP_SIZE} \
      --pipeline-model-parallel-size 1 \
      --num-layers 96 \
      --hidden-size 12288 \
      --ffn-hidden-size 49152 \
      --num-attention-heads 96 \
      --seq-length 2048 \
      --global-batch-size 1536 \
      --train-iters 450 \
      --lr-decay-iters 430 \
      --data-path ${DATA_DIR}/c4/c4_text_document \
      --vocab-file ${DATA_DIR}/gpt2-vocab.json \
      --merge-file ${DATA_DIR}/gpt2-merges.txt \
      --split 949,50,1 \
      --distributed-backend nccl \
      --lr 6.0e-5 --min-lr 6.0e-6 \
      --lr-decay-style cosine \
      --weight-decay 0.1 \
      --clip-grad 1.0 \
      --bf16 \
      --log-interval 10
  "
fi

# =============================================================================
# ID 020 – Llama 2 70B LoRA fine-tuning
# =============================================================================
if should_run "020"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_020" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/llm_finetune/run_lora_finetune.py \
      --model_name_or_path meta-llama/Meta-Llama-3.1-70B-Instruct \
      --dataset_name scrolls \
      --dataset_config govreport \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --learning_rate 4e-4 \
      --bf16 \
      --output_dir ${OUTPUT_DIR}/AI_HOSTCPU_ML_TEST_ID_020 \
      --logging_steps 50
  "
fi

# =============================================================================
# ID 021 – DLRMv2 training
# =============================================================================
if should_run "021"; then
  run_training_job "AI_HOSTCPU_ML_TEST_ID_021" "
    torchrun --nproc_per_node=${NUM_GPUS} \
      ${TRAINING_DIR}/recommendation/dlrm_v2/pytorch/dlrm_s_pytorch.py \
      --arch-sparse-feature-size=128 \
      --arch-mlp-bot=13-512-256-128 \
      --arch-mlp-top=1024-1024-512-256-1 \
      --data-generation=dataset \
      --data-set=terabyte \
      --raw-data-file=${DATA_DIR}/criteo/day_0 \
      --processed-data-file=${DATA_DIR}/criteo/terabyte_processed.npz \
      --loss-function=bce \
      --round-targets=True \
      --learning-rate=1.0 \
      --mini-batch-size=2048 \
      --print-freq=2048 --print-time \
      --test-mini-batch-size=16384 --test-num-workers=16 \
      --memory-map \
      --mlperf-logging \
      --mlperf-auc-threshold=0.8025 \
      --mlperf-bin-loader \
      --mlperf-bin-shuffle \
      --num-batches=5000
  "
fi

# =============================================================================
# Final summary
# =============================================================================
log ""
log "========================================================"
log "  Run complete"
log "  PASSED : ${PASS_COUNT}"
log "  FAILED : ${FAIL_COUNT}"
log "  SKIPPED: ${SKIP_COUNT}"
log "  KPI file: ${KPI_FILE}"
log "  Results : ${OUTPUT_DIR}/"
log "========================================================"

if (( FAIL_COUNT > 0 )); then
  error "${FAIL_COUNT} test case(s) failed."
  exit 1
fi

exit 0
