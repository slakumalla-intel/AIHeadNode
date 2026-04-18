#!/usr/bin/env bash
set -euo pipefail

# Parallel soak workload runner:
#   1) stress-ng CPU/memory pressure
#   2) fio disk pressure on root mount (or custom path)
#   3) Llama 70B training command loop
#
# Defaults target the user's described test node:
#   - 224 CPU cores
#   - 371 GB memory
#   - 1.7 TB root disk mounted on /

DURATION_HOURS="12"
CPU_CORES="224"
TOTAL_MEM_GB="371"
DISK_PATH="/"
FIO_SIZE_GB="600"
FIO_NUMJOBS="8"
FIO_IODEPTH="64"
FIO_BS="1M"
SNAPSHOT_INTERVAL_SEC="60"
LOG_DIR="artifacts/soak_$(date +%Y%m%d_%H%M%S)"

# Set this from shell before running, or pass with --llama-cmd.
LLAMA_TRAIN_CMD="${LLAMA_TRAIN_CMD:-}"

# Optional but useful defaults for MLPerf-style llama 70B LoRA fine-tuning.
ML_NUM_GPUS="${ML_NUM_GPUS:-4}"
MLPERF_TRAINING_DIR="${MLPERF_TRAINING_DIR:-/root/workloads/mlcommons/training}"
MLPERF_OUTPUT_DIR="${MLPERF_OUTPUT_DIR:-artifacts/mlperf}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --duration-hours <N>      Soak duration in hours (default: 12)
  --cpu-cores <N>           stress-ng CPU workers (default: 224)
  --total-mem-gb <N>        Total host memory in GB (default: 371)
  --disk-path <PATH>        Mount/path to place fio test file (default: /)
  --fio-size-gb <N>         fio file size in GB (default: 600)
  --fio-numjobs <N>         fio numjobs (default: 8)
  --fio-iodepth <N>         fio iodepth (default: 64)
  --fio-bs <SIZE>           fio block size (default: 1M)
  --snapshot-interval-sec <N>
                            Health snapshot interval in seconds (default: 60)
  --log-dir <PATH>          Log directory (default: artifacts/soak_<timestamp>)
  --llama-cmd <CMD>         Llama training command (overrides LLAMA_TRAIN_CMD env)
  -h, --help                Show help

Example:
  LLAMA_TRAIN_CMD='torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/llm_finetune/run_lora_finetune.py --model_name_or_path meta-llama/Llama-2-70b-hf --dataset_name scrolls --dataset_config govreport --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --num_train_epochs 1 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --learning_rate 4e-4 --bf16 --output_dir artifacts/mlperf/llama2-lora --logging_steps 50' \
  ./scripts/run_parallel_soak_12h.sh --duration-hours 12
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration-hours)
      DURATION_HOURS="$2"; shift 2 ;;
    --cpu-cores)
      CPU_CORES="$2"; shift 2 ;;
    --total-mem-gb)
      TOTAL_MEM_GB="$2"; shift 2 ;;
    --disk-path)
      DISK_PATH="$2"; shift 2 ;;
    --fio-size-gb)
      FIO_SIZE_GB="$2"; shift 2 ;;
    --fio-numjobs)
      FIO_NUMJOBS="$2"; shift 2 ;;
    --fio-iodepth)
      FIO_IODEPTH="$2"; shift 2 ;;
    --fio-bs)
      FIO_BS="$2"; shift 2 ;;
    --snapshot-interval-sec)
      SNAPSHOT_INTERVAL_SEC="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --llama-cmd)
      LLAMA_TRAIN_CMD="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$LLAMA_TRAIN_CMD" ]]; then
  echo "ERROR: Llama training command is required. Set LLAMA_TRAIN_CMD or pass --llama-cmd." >&2
  echo "Hint: Use the MLPerf llama2-lora command from tests/test_mlperf_suite.py." >&2
  exit 2
fi

for c in stress-ng fio timeout; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $c" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR"
mkdir -p "$MLPERF_OUTPUT_DIR"

DURATION_SEC=$((DURATION_HOURS * 3600))
END_TS=$(( $(date +%s) + DURATION_SEC ))

# Keep stress memory pressure below total system memory to avoid immediate OOM kill.
STRESS_MEM_GB=$(( TOTAL_MEM_GB * 75 / 100 ))
if (( STRESS_MEM_GB < 32 )); then
  STRESS_MEM_GB=32
fi

FIO_FILE="$DISK_PATH/fio_soak_$(date +%Y%m%d_%H%M%S).dat"
PIDS=()

cleanup() {
  echo "[INFO] Cleanup invoked. Stopping child workloads..."
  for p in "${PIDS[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
  wait || true
}
trap cleanup INT TERM

start_stressng() {
  echo "[INFO] Starting stress-ng workload..."
  stress-ng \
    --cpu "$CPU_CORES" \
    --cpu-method all \
    --vm 8 \
    --vm-bytes "${STRESS_MEM_GB}G" \
    --vm-keep \
    --timeout "${DURATION_SEC}s" \
    --metrics-brief \
    > "$LOG_DIR/stressng.log" 2>&1 &
  PIDS+=("$!")
}

start_fio() {
  echo "[INFO] Starting fio workload on $FIO_FILE ..."
  fio \
    --name=soak_rw \
    --filename="$FIO_FILE" \
    --size="${FIO_SIZE_GB}G" \
    --rw=randrw \
    --rwmixread=70 \
    --bs="$FIO_BS" \
    --iodepth="$FIO_IODEPTH" \
    --numjobs="$FIO_NUMJOBS" \
    --direct=1 \
    --time_based=1 \
    --runtime="$DURATION_SEC" \
    --group_reporting \
    > "$LOG_DIR/fio.log" 2>&1 &
  PIDS+=("$!")
}

start_llama_loop() {
  echo "[INFO] Starting Llama training loop workload..."
  (
    set +e
    run_count=0
    fail_count=0
    while [[ $(date +%s) -lt "$END_TS" ]]; do
      remain=$(( END_TS - $(date +%s) ))
      run_count=$((run_count + 1))
      echo "[INFO] Llama run #$run_count (remaining ${remain}s)"

      timeout "$remain" bash -lc "$LLAMA_TRAIN_CMD"
      rc=$?
      if [[ $rc -ne 0 && $rc -ne 124 ]]; then
        fail_count=$((fail_count + 1))
        echo "[WARN] Llama run #$run_count exited rc=$rc"
      fi
    done
    echo "[INFO] Llama loop finished: runs=$run_count failed=$fail_count"
  ) > "$LOG_DIR/llama_training.log" 2>&1 &
  PIDS+=("$!")
}

start_health_monitor() {
  echo "[INFO] Starting health snapshot monitor (interval=${SNAPSHOT_INTERVAL_SEC}s)..."
  (
    set +e
    timeline_log="$LOG_DIR/health_timeline.log"
    while [[ $(date +%s) -lt "$END_TS" ]]; do
      ts_iso="$(date -Iseconds)"
      {
        echo "===== SNAPSHOT $ts_iso ====="
        echo "-- uptime --"
        uptime || true

        echo "-- memory --"
        free -h || true

        echo "-- disk --"
        df -h "$DISK_PATH" || true

        if command -v nvidia-smi >/dev/null 2>&1; then
          echo "-- nvidia-smi --"
          nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.graphics,pstate --format=csv,noheader,nounits || true
        else
          echo "-- nvidia-smi --"
          echo "nvidia-smi not found"
        fi

        if command -v iostat >/dev/null 2>&1; then
          echo "-- iostat --"
          iostat -xz 1 1 || true
        else
          echo "-- iostat --"
          echo "iostat not found"
        fi

        if command -v mpstat >/dev/null 2>&1; then
          echo "-- mpstat --"
          mpstat -P ALL 1 1 || true
        else
          echo "-- mpstat --"
          echo "mpstat not found"
        fi
        echo
      } >> "$timeline_log" 2>&1

      sleep "$SNAPSHOT_INTERVAL_SEC"
    done
  ) &
  PIDS+=("$!")
}

cat <<EOF
[INFO] Parallel soak configuration
  duration_hours       : $DURATION_HOURS
  cpu_cores            : $CPU_CORES
  total_mem_gb         : $TOTAL_MEM_GB
  stress_mem_gb        : $STRESS_MEM_GB
  disk_path            : $DISK_PATH
  fio_size_gb          : $FIO_SIZE_GB
  fio_numjobs          : $FIO_NUMJOBS
  fio_iodepth          : $FIO_IODEPTH
  fio_bs               : $FIO_BS
  snapshot_interval_sec: $SNAPSHOT_INTERVAL_SEC
  log_dir              : $LOG_DIR
  fio_file             : $FIO_FILE
  ml_num_gpus          : $ML_NUM_GPUS
  mlperf_training_dir  : $MLPERF_TRAINING_DIR
  mlperf_output_dir    : $MLPERF_OUTPUT_DIR
EOF

start_stressng
start_fio
start_llama_loop
start_health_monitor

echo "[INFO] Workloads launched. Waiting for completion..."

status=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    status=1
  fi
done

rm -f "$FIO_FILE" || true

if [[ $status -eq 0 ]]; then
  echo "[INFO] Soak run completed. Logs in: $LOG_DIR"
else
  echo "[ERROR] One or more workloads exited with non-zero status. Check logs in: $LOG_DIR" >&2
  exit 1
fi
