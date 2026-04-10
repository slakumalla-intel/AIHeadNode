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
set_default_env VLLM_BASELINE_GPU_MEMORY_UTIL "0.9"
set_default_env VLLM_GPU_MEMORY_UTIL "0.7"
set_default_env VLLM_CPU_OFFLOAD_GB "128"

# TC-013: Small LLM training (AI_HOSTCPU_FUNC_TEST_ID_013)
set_default_env LLM_TRAIN_SMALL_MODEL "google/gemma-3-270m-it"
set_default_env LLM_TRAIN_SMALL_BATCH "2"
set_default_env LLM_TRAIN_SMALL_SEQ_LEN "512"
set_default_env LLM_TRAIN_SMALL_STEPS "50"
set_default_env LLM_TRAIN_SMALL_WARMUP "5"
set_default_env LLM_TRAIN_SMALL_MAX_SYNC_RATIO "0.15"
set_default_env LLM_TRAIN_SMALL_TIMEOUT_SEC "3600"
set_default_env LLM_TRAIN_SMALL_MASTER_PORT "29601"

# TC-014: Large LLM training (AI_HOSTCPU_FUNC_TEST_ID_014)
set_default_env LLM_TRAIN_LARGE_MODEL "${HF_LLAMA_7B_MODEL:-meta-llama/Meta-Llama-3-8B}"
set_default_env LLM_TRAIN_LARGE_BATCH "1"
set_default_env LLM_TRAIN_LARGE_SEQ_LEN "512"
set_default_env LLM_TRAIN_LARGE_STEPS "20"
set_default_env LLM_TRAIN_LARGE_WARMUP "3"
set_default_env LLM_TRAIN_LARGE_MAX_SYNC_RATIO "0.10"
set_default_env LLM_TRAIN_LARGE_MIN_SCALING_EFF "70.0"
set_default_env LLM_TRAIN_LARGE_USE_DEEPSPEED "1"
set_default_env LLM_TRAIN_LARGE_TIMEOUT_SEC "3600"
set_default_env LLM_TRAIN_LARGE_MASTER_PORT "29602"

set_default_env RDMA_SMOKE_ITERS "1000"
set_default_env RDMA_USE_CUDA_DEVICE "0"
set_default_env RDMA_4KB_MAX_AVG_LAT_US "300.0"
set_default_env RDMA_SOAK_LOOPS "1000"
set_default_env SOAK_MAX_P95_US "20000"

# TC-047 / TC-048: multi-tenant inference -------------------------------------
set_default_env MULTITENANT_ENABLE_LARGE_MODELS "0"
set_default_env MULTITENANT_MODEL_A             "gpt2"
set_default_env MULTITENANT_MODEL_B             "gpt2"
set_default_env MULTITENANT_N_REQUESTS          "20"
set_default_env MULTITENANT_MAX_NEW_TOKENS      "32"
set_default_env MULTITENANT_GPU_MEM_UTIL        "0.85"
set_default_env MULTITENANT_MAX_MODEL_LEN       "2048"
set_default_env MULTITENANT_KV_DTYPE_B          ""
set_default_env MULTITENANT_KV_CPU_OFFLOAD_GB   "8.0"
set_default_env MULTITENANT_MAX_P99_P50_RATIO   "5.0"
set_default_env MULTITENANT_KV_OFFLOAD_MAX_P99_RATIO "3.0"
set_default_env MULTITENANT_TIMEOUT_SEC         "3600"

# TC-049: DeepSeek-R1-70B KV DRAM offload ------------------------------------
set_default_env DEEPSEEK_KV_OFFLOAD_MODEL       "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
set_default_env DEEPSEEK_TP_SIZE                "4"
set_default_env DEEPSEEK_MAX_MODEL_LEN          "16384"
set_default_env DEEPSEEK_CPU_OFFLOAD_GB         "8.0"
set_default_env DEEPSEEK_GPU_MEM_UTIL           "0.85"
set_default_env DEEPSEEK_N_REQUESTS             "10"
set_default_env DEEPSEEK_MAX_NEW_TOKENS         "64"
set_default_env DEEPSEEK_MIN_TPS                "1.0"
set_default_env DEEPSEEK_MAX_GPU_MEM_PCT        "95.0"

# TC-050: PCIe BW with 2 MB hugepages (reuses NODE_PCIE_BW_BYTES / NODE_PCIE_BW_ITERS)

# TC-051: GPU write performance limiters -------------------------------------
set_default_env GPU_WRITE_ITERS                 "10"
set_default_env GPU_WRITE_MIN_PEAK_GBS          "167.0"
set_default_env GPU_WRITE_HBM_BOUND_RATIO       "2.0"

# TC-052: P2P multi-stream bandwidth -----------------------------------------
set_default_env P2P_MULTI_STREAM_BYTES          "268435456"
set_default_env P2P_MULTI_STREAM_ITERS          "5"
set_default_env P2P_MULTI_STREAM_MIN_SCALING    "1.5"

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
echo "VLLM_BASELINE_GPU_MEMORY_UTIL=${VLLM_BASELINE_GPU_MEMORY_UTIL:-}"
echo "VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL:-}"
echo "YOLO_DETECT_BATCH=$YOLO_DETECT_BATCH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================================================"

pytest -m workload_acceptance -vv "${EXTRA_PYTEST_ARGS[@]}"
