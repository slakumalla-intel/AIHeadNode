#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
TP="${TP:-4}"
DTYPE="${DTYPE:-bfloat16}"
GPU_UTIL="${GPU_UTIL:-0.90}"

INPUT_LENS="${INPUT_LENS:-128 512 2048 8192}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
OUTPUT_LEN="${OUTPUT_LEN:-1}"

WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
RUN_PERF="${RUN_PERF:-1}"

OUTDIR="${OUTDIR:-vllm_4h100_prefill_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

CSV="$OUTDIR/summary.csv"
LOG="$OUTDIR/run.log"

# ============================================================================
# Cleanup handler to ensure all vLLM processes are terminated
# ============================================================================
cleanup() {
  local exit_code=$?
  echo "[INFO] Cleanup: Terminating vLLM worker processes..." | tee -a "$LOG"
  
  # Kill all vLLM worker processes
  pkill -9 -f "VLLM::Worker" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints.cli.main" 2>/dev/null || true
  pkill -9 -f "bench latency" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
  
  # Release GPU memory
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] Cleanup: Releasing GPU memory..." | tee -a "$LOG"
    nvidia-smi -i 0,1,2,3 -pm 1 2>/dev/null || true
  fi
  
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "model,input_len,output_len,batch_size,tp,dtype,status,latency_json,perf_log" > "$CSV"

echo "==================================================" | tee -a "$LOG"
echo "vLLM 4xH100 Prompt Ingestion / Prefill Validation" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "MODEL      : $MODEL" | tee -a "$LOG"
echo "TP         : $TP" | tee -a "$LOG"
echo "DTYPE      : $DTYPE" | tee -a "$LOG"
echo "GPU_UTIL   : $GPU_UTIL" | tee -a "$LOG"
echo "INPUT_LENS : $INPUT_LENS" | tee -a "$LOG"
echo "BATCHES    : $BATCH_SIZES" | tee -a "$LOG"
echo "OUTPUT_LEN : $OUTPUT_LEN" | tee -a "$LOG"
echo "OUTDIR     : $OUTDIR" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

echo "[INFO] Checking GPUs..." | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

GPU_COUNT=$(nvidia-smi -L | wc -l)
if [[ "$GPU_COUNT" -lt 4 ]]; then
  echo "[ERROR] Expected 4 GPUs, found $GPU_COUNT" | tee -a "$LOG"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

echo "[INFO] Checking vLLM..." | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import vllm
print("vLLM version:", vllm.__version__)
PY

for LEN in $INPUT_LENS; do
  for BS in $BATCH_SIZES; do
    CASE="len${LEN}_bs${BS}_out${OUTPUT_LEN}_tp${TP}"
    JSON_OUT="$OUTDIR/${CASE}.json"
    PERF_OUT="$OUTDIR/${CASE}.perf.txt"

    CMD=(
      python3 -m vllm.entrypoints.cli.main bench latency
      --model "$MODEL"
      --tensor-parallel-size "$TP"
      --dtype "$DTYPE"
      --gpu-memory-utilization "$GPU_UTIL"
      --input-len "$LEN"
      --output-len "$OUTPUT_LEN"
      --batch-size "$BS"
      --num-iters-warmup "$WARMUP"
      --num-iters "$ITERS"
      --output-json "$JSON_OUT"
      --disable-log-stats
    )

    echo "" | tee -a "$LOG"
    echo "--------------------------------------------------" | tee -a "$LOG"
    echo "[RUN] $CASE" | tee -a "$LOG"
    echo "[CMD] ${CMD[*]}" | tee -a "$LOG"

    STATUS="PASS"

    if [[ "$RUN_PERF" == "1" ]]; then
      if command -v perf >/dev/null 2>&1; then
        set +e
        perf stat -d -r 3 -o "$PERF_OUT" "${CMD[@]}" 2>&1 | tee -a "$LOG"
        RC=${PIPESTATUS[0]}
        set -e
      else
        echo "[WARN] perf not found; running without perf" | tee -a "$LOG"
        set +e
        "${CMD[@]}" 2>&1 | tee -a "$LOG"
        RC=${PIPESTATUS[0]}
        set -e
        PERF_OUT="NA"
      fi
    else
      set +e
      "${CMD[@]}" 2>&1 | tee -a "$LOG"
      RC=${PIPESTATUS[0]}
      set -e
      PERF_OUT="NA"
    fi

    if [[ "$RC" -ne 0 ]]; then
      STATUS="FAIL"
      echo "[FAIL] $CASE rc=$RC" | tee -a "$LOG"
    else
      echo "[PASS] $CASE" | tee -a "$LOG"
    fi

    echo "$MODEL,$LEN,$OUTPUT_LEN,$BS,$TP,$DTYPE,$STATUS,$JSON_OUT,$PERF_OUT" >> "$CSV"

    nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits > "$OUTDIR/${CASE}.nvidia_smi.csv" || true

    # Cleanup worker processes after each case to prevent GPU memory accumulation
    echo "[INFO] Post-case cleanup - killing vLLM workers..." | tee -a "$LOG"
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.cli.main" 2>/dev/null || true
    sleep 1
  done
done

echo "" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "Completed." | tee -a "$LOG"
echo "Summary CSV: $CSV" | tee -a "$LOG"
echo "Logs       : $LOG" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
