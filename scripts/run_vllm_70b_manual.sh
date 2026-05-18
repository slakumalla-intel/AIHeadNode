#!/usr/bin/env bash
set -euo pipefail

# Manual 70B vLLM run helper: setup -> serve -> bench -> cleanup
# Usage:
#   bash scripts/run_vllm_70b_manual.sh start
#   bash scripts/run_vllm_70b_manual.sh bench
#   bash scripts/run_vllm_70b_manual.sh cleanup
#   bash scripts/run_vllm_70b_manual.sh all

MODE="${1:-all}"

export PATH=/opt/venvs/torch-cu128/bin:${PATH}
hash -r

export HF_HOME=${HF_HOME:-/mnt/datasets/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/datasets/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/datasets/huggingface/hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

MODEL_DIR=${MODEL_DIR:-$(ls -td /mnt/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/* | head -1)}
TOK_DIR=${TOK_DIR:-${MODEL_DIR}}
SERVE_LOG=${SERVE_LOG:-/tmp/vllm_serve.log}

BENCH_CONCURRENCY=${BENCH_CONCURRENCY:-64}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-100}
BENCH_INPUT_LEN=${BENCH_INPUT_LEN:-2048}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-256}

show_info() {
  echo "MODEL_DIR=${MODEL_DIR}"
  ls -lh "${MODEL_DIR}/config.json"
}

start_serve() {
  echo "Starting vLLM serve; log=${SERVE_LOG}"
  nohup /opt/venvs/torch-cu128/bin/vllm serve "${MODEL_DIR}" \
    --served-model-name meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.60 \
    --port 8000 \
    > "${SERVE_LOG}" 2>&1 &
  echo "Started. Follow logs: tail -f ${SERVE_LOG}"
}

run_bench() {
  /opt/venvs/torch-cu128/bin/vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tokenizer "${TOK_DIR}" \
    --dataset-name random \
    --random-input-len "${BENCH_INPUT_LEN}" \
    --random-output-len "${BENCH_OUTPUT_LEN}" \
    --max-concurrency "${BENCH_CONCURRENCY}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" \
    --metric-percentiles 50,90,99
}

cleanup_vllm() {
  ps -eo pid,ppid,stat,comm,args | grep -Ei "VLLM::|vllm serve|EngineCore|Worker_TP" | grep -v grep || true
  pids=$(ps -eo pid,args | grep -Ei "VLLM::|vllm serve|EngineCore|Worker_TP" | grep -v grep | awk '{print $1}')

  if [[ -z "${pids}" ]]; then
    echo "No matching vLLM processes found."
    return 0
  fi

  echo "Killing PIDs: ${pids}"
  kill -TERM ${pids} 2>/dev/null || true
  sleep 5
  kill -9 ${pids} 2>/dev/null || true
}

case "${MODE}" in
  start)
    show_info
    start_serve
    ;;
  bench)
    run_bench
    ;;
  cleanup)
    cleanup_vllm
    ;;
  all)
    show_info
    start_serve
    echo "Waiting for server readiness..."
    until curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null; do
      sleep 5
      echo "Still waiting for /health ..."
    done
    echo "Server is healthy. Running benchmark."
    run_bench
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash scripts/run_vllm_70b_manual.sh {start|bench|cleanup|all}"
    exit 1
    ;;
esac
