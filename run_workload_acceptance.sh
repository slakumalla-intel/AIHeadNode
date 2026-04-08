#!/usr/bin/env bash
set -euo pipefail

INCLUDE_LARGE_MODELS=0
SKIP_LARGE_MODELS=0
MIN_DURATION_SEC="300"
KPI_FILE="artifacts/node_validation_kpis.jsonl"
KV_CACHE_CONTEXT_LENGTH="8192"
KV_CACHE_BATCH_SIZE="4"
KV_CACHE_NUM_PROMPTS="10"
YOLO_BATCH_SIZE="128"
YOLO_IMG_SIZE="640"
EXTRA_PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-large-models)
      INCLUDE_LARGE_MODELS=1
      shift
      ;;
    --skip-large-models)
      SKIP_LARGE_MODELS=1
      shift
      ;;
    --min-duration-sec)
      MIN_DURATION_SEC="$2"
      shift 2
      ;;
    --kpi-file)
      KPI_FILE="$2"
      shift 2
      ;;
    --kv-context-length)
      KV_CACHE_CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --kv-batch-size)
      KV_CACHE_BATCH_SIZE="$2"
      shift 2
      ;;
    --kv-num-prompts)
      KV_CACHE_NUM_PROMPTS="$2"
      shift 2
      ;;
    --yolo-batch-size)
      YOLO_BATCH_SIZE="$2"
      shift 2
      ;;
    --yolo-img-size)
      YOLO_IMG_SIZE="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_PYTEST_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

set_default_env() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    export "$name=$value"
  fi
}

if [[ "$SKIP_LARGE_MODELS" -eq 1 ]]; then
  export HF_ENABLE_LARGE_MODELS="0"
else
  export HF_ENABLE_LARGE_MODELS="1"
fi
if [[ "$INCLUDE_LARGE_MODELS" -eq 1 ]]; then
  export HF_ENABLE_LARGE_MODELS="1"
fi

export HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC="$MIN_DURATION_SEC"
export NODE_VALIDATION_KPI_FILE="$KPI_FILE"

set_default_env HF_LLAMA_7B_MODEL "meta-llama/Llama-2-7b-hf"
set_default_env HF_LLAMA_70B_MODEL "meta-llama/Llama-2-70b-hf"
set_default_env HF_QWEN_70B_MODEL "Qwen/Qwen-72B"
set_default_env HF_DEEPSEEK_R1_QUANT_MODEL "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
set_default_env HF_INFERENCE_MODEL "meta-llama/Llama-2-7b-hf"
set_default_env HF_TRAINING_MODEL "bert-base-uncased"

set_default_env VLLM_KV_OFFLOAD_MODEL "meta-llama/Llama-2-7b-hf"
export VLLM_KV_CONTEXT_LENGTH="$KV_CACHE_CONTEXT_LENGTH"
export VLLM_KV_BATCH_SIZE="$KV_CACHE_BATCH_SIZE"
export VLLM_KV_NUM_PROMPTS="$KV_CACHE_NUM_PROMPTS"
set_default_env VLLM_KV_MAX_TOKENS "512"
set_default_env VLLM_GPU_MEMORY_UTIL "0.7"
set_default_env VLLM_CPU_OFFLOAD_GB "128"

set_default_env RDMA_SMOKE_ITERS "1000"
set_default_env RDMA_USE_CUDA_DEVICE "0"
set_default_env RDMA_4KB_MAX_AVG_LAT_US "300.0"
set_default_env RDMA_SOAK_LOOPS "1000"
set_default_env SOAK_MAX_P95_US "20000"

set_default_env YOLO_DETECT_MODEL "yolov8x.pt"
export YOLO_DETECT_BATCH="$YOLO_BATCH_SIZE"
export YOLO_DETECT_IMGSZ="$YOLO_IMG_SIZE"
set_default_env YOLO_DETECT_SOURCE "https://ultralytics.com/images/bus.jpg"
set_default_env YOLO_DETECT_TIMEOUT_SEC "240"
set_default_env YOLO_CPU_SAMPLE_INTERVAL_SEC "0.5"
set_default_env YOLO_AGGREGATE_FPS_TARGET "100"

set_default_env DIFFUSERS_NUM_IMAGES "50"
set_default_env DIFFUSERS_HEIGHT "512"
set_default_env DIFFUSERS_WIDTH "512"
set_default_env DIFFUSERS_BATCH_SIZE "4"

set_default_env HF_HOME "$HOME/.cache/huggingface"
set_default_env HF_TOKEN ""
set_default_env HUGGINGFACE_HUB_CACHE "$HF_HOME/hub"
set_default_env HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD "1"

set_default_env TORCH_HOME "$HOME/.cache/torch"
set_default_env CUDA_VISIBLE_DEVICES "0,1,2,3"
set_default_env CUDA_DEVICE_ORDER "PCI_BUS_ID"
set_default_env PYTEST_TIMEOUT "600"
export PYTHONUNBUFFERED="1"

echo "========================================================================"
echo "Workload Acceptance Test Configuration (Linux)"
echo "========================================================================"
echo "HF_ENABLE_LARGE_MODELS=$HF_ENABLE_LARGE_MODELS"
echo "HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=$HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC"
echo "NODE_VALIDATION_KPI_FILE=$NODE_VALIDATION_KPI_FILE"
echo "VLLM_KV_CONTEXT_LENGTH=$VLLM_KV_CONTEXT_LENGTH"
echo "YOLO_DETECT_BATCH=$YOLO_DETECT_BATCH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================================================"

pytest -m workload_acceptance -vv "${EXTRA_PYTEST_ARGS[@]}"
