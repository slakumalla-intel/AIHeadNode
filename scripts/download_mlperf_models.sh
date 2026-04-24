#!/usr/bin/env bash
# =============================================================================
# download_mlperf_models.sh
#
# Downloads all model checkpoints required for AI_HOSTCPU_ML_TEST_ID_001–021
# on a 4×H100 SXM5 system.
#
# Usage:
#   bash scripts/download_mlperf_models.sh [OPTIONS]
#
# Options:
#   --model-dir DIR   Root directory for model storage  (default: <repo>/datasets)
#   --hf-token TOKEN  HuggingFace access token           (or set HF_TOKEN env var)
#   --skip-large      Skip models that require gated/paid access (405B, DeepSeek-R1)
#   --dry-run         Print commands but do not execute them
#   --only IDs        Comma-separated list of test IDs to download models for
#                     e.g. --only 001,004,006
#
# Prerequisites:
#   huggingface_hub >= 0.23  (pip install -U huggingface_hub)
#   git-lfs                  (for ONNX / binary files)
#   aria2c (optional)        large-file parallel download
#
# Notes:
#   - Llama family models require accepting the Meta licence on huggingface.co
#     and passing a valid --hf-token / HF_TOKEN.
#   - DeepSeek-R1 671B requires ~680 GB disk in BF16; use --skip-large to omit.
#   - GPT-3 175B training does NOT require a pre-downloaded checkpoint; the
#     Megatron-LM script initialises weights from scratch.
#   - ONNX model files for inference benchmarks (ResNet-50, RetinaNet, 3D U-Net,
#     RNN-T, DLRM) are fetched from the MLCommons GCP bucket.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DATASET_ROOT="${REPO_ROOT}/datasets"

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags)
# ---------------------------------------------------------------------------
MODEL_DIR="${MLPERF_MODEL_DIR:-${DEFAULT_DATASET_ROOT}}"
HF_TOKEN="${HF_TOKEN:-}"
SKIP_LARGE=false
DRY_RUN=false
ONLY_IDS=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)  MODEL_DIR="$2";  shift 2 ;;
    --hf-token)   HF_TOKEN="$2";   shift 2 ;;
    --skip-large) SKIP_LARGE=true; shift   ;;
    --dry-run)    DRY_RUN=true;    shift   ;;
    --only)       ONLY_IDS="$2";   shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $*"
  else
    echo "[RUN] $*"
    "$@"
  fi
}

dir_has_content() {
  local path="$1"
  [[ -d "$path" ]] && find "$path" -mindepth 1 -print -quit | grep -q .
}

file_is_present() {
  local path="$1"
  [[ -s "$path" ]]
}

should_run() {
  local test_id="$1"
  if [[ -z "$ONLY_IDS" ]]; then return 0; fi
  # Accepts "001", "AI_HOSTCPU_ML_TEST_ID_001", or a short number
  local normalized
  normalized=$(printf "%03d" "${test_id##*_}" 2>/dev/null || echo "$test_id")
  if echo "$ONLY_IDS" | tr ',' '\n' | grep -qiE "(AI_HOSTCPU_ML_TEST_ID_0*${normalized#0*}|0*${normalized#0*})"; then
    return 0
  fi
  return 1
}

hf_download() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local extra_args=("$@")

  if dir_has_content "$local_dir"; then
    log "[SKIP] ${repo_id} already present at ${local_dir}"
    return 0
  fi

  run mkdir -p "$local_dir"

  if [[ -z "$HF_TOKEN" ]]; then
    run huggingface-cli download "$repo_id" \
      --local-dir "$local_dir" \
      --local-dir-use-symlinks False \
      "${extra_args[@]+"${extra_args[@]}"}"
  else
    run huggingface-cli download "$repo_id" \
      --token "$HF_TOKEN" \
      --local-dir "$local_dir" \
      --local-dir-use-symlinks False \
      "${extra_args[@]+"${extra_args[@]}"}"
  fi
}

fetch_file() {
  local url="$1"
  local dest="$2"

  if file_is_present "$dest"; then
    log "[SKIP] $(basename "$dest") already present at ${dest}"
    return 0
  fi

  run mkdir -p "$(dirname "$dest")"
  run wget -q --show-progress -c -O "$dest" "$url"
}

log()  { echo "$(date '+%Y-%m-%d %H:%M:%S')  [INFO]  $*"; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S')  [WARN]  $*" >&2; }

# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------
log "Checking prerequisites..."
command -v huggingface-cli &>/dev/null || {
  warn "huggingface-cli not found. Installing huggingface_hub..."
  run pip install -U "huggingface_hub[cli]"
}
command -v git-lfs &>/dev/null || warn "git-lfs not found – large binary files may fail to download"

mkdir -p "$MODEL_DIR"
log "Model root: $MODEL_DIR"

# =============================================================================
# Inference models
# =============================================================================

# ---------------------------------------------------------------------------
# ID 001 – Llama 3.1 405B  (text generation / long context)
# ---------------------------------------------------------------------------
if should_run "001" && [[ "$SKIP_LARGE" == "false" ]]; then
  log "[001] Downloading Llama 3.1 405B-Instruct (~800 GB in BF16, FP8 shards preferred)..."
  hf_download "meta-llama/Llama-3.1-405B-Instruct" \
    "${MODEL_DIR}/llama31-405b" \
    --include "*.safetensors" "*.json" "*.model" "tokenizer*"
else
  log "[001] Skipping Llama 3.1 405B (--skip-large or not selected)"
fi

# ---------------------------------------------------------------------------
# ID 002 – DeepSeek-R1  (reasoning)
# ---------------------------------------------------------------------------
if should_run "002" && [[ "$SKIP_LARGE" == "false" ]]; then
  log "[002] Downloading DeepSeek-R1 671B (BF16; ~680 GB)..."
  hf_download "deepseek-ai/DeepSeek-R1" \
    "${MODEL_DIR}/deepseek-r1" \
    --include "*.safetensors" "*.json" "tokenizer*"
else
  log "[002] Skipping DeepSeek-R1 (--skip-large or not selected)"
fi

# ---------------------------------------------------------------------------
# ID 003 – Mixtral 8×7B  (MoE text generation)
# ---------------------------------------------------------------------------
if should_run "003"; then
  log "[003] Downloading Mixtral-8x7B-Instruct-v0.1..."
  hf_download "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    "${MODEL_DIR}/mixtral-8x7b"
fi

# ---------------------------------------------------------------------------
# ID 004 – Llama 3.1 8B  (summarization)
# ---------------------------------------------------------------------------
if should_run "004"; then
  log "[004] Downloading Meta-Llama-3.1-8B-Instruct..."
  hf_download "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    "${MODEL_DIR}/llama31-8b"
fi

# ---------------------------------------------------------------------------
# ID 005 – Llama 3.1 70B  (Q&A)
# ---------------------------------------------------------------------------
if should_run "005"; then
  log "[005] Downloading Meta-Llama-3.1-70B-Instruct..."
  hf_download "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    "${MODEL_DIR}/llama31-70b"
fi

# ---------------------------------------------------------------------------
# ID 006 – ResNet-50 v1.5  (ONNX from MLCommons GCP bucket)
# ---------------------------------------------------------------------------
if should_run "006"; then
  log "[006] Downloading ResNet-50 v1.5 ONNX..."
  fetch_file \
    "https://zenodo.org/record/4735647/files/resnet50_v1.onnx" \
    "${MODEL_DIR}/resnet50/resnet50_v1.onnx"
fi

# ---------------------------------------------------------------------------
# ID 007 – RetinaNet  (ONNX model)
# ---------------------------------------------------------------------------
if should_run "007"; then
  log "[007] Downloading RetinaNet ResNeXt-50 ONNX..."
  fetch_file \
    "https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx" \
    "${MODEL_DIR}/retinanet/retinanet_resnext50_32x4d.onnx"
fi

# ---------------------------------------------------------------------------
# ID 008 – 3D U-Net KiTS19  (ONNX model)
# ---------------------------------------------------------------------------
if should_run "008"; then
  log "[008] Downloading 3D U-Net KiTS19 ONNX..."
  fetch_file \
    "https://zenodo.org/record/5597155/files/3dunet_kits19_128x128x128.onnx" \
    "${MODEL_DIR}/3d-unet/3dunet_kits19.onnx"
fi

# ---------------------------------------------------------------------------
# ID 009 – RNN-T  (PyTorch checkpoint)
# ---------------------------------------------------------------------------
if should_run "009"; then
  log "[009] Downloading RNN-T PyTorch checkpoint..."
  fetch_file \
    "https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt" \
    "${MODEL_DIR}/rnnt/rnnt.pt"
fi

# ---------------------------------------------------------------------------
# ID 010 – Whisper large-v2  (from HuggingFace Hub)
# ---------------------------------------------------------------------------
if should_run "010"; then
  log "[010] Downloading openai/whisper-large-v2..."
  hf_download "openai/whisper-large-v2" \
    "${MODEL_DIR}/whisper-large-v2"
fi

# ---------------------------------------------------------------------------
# ID 011 – DLRM-DCNv2  (weights from MLCommons team)
# ---------------------------------------------------------------------------
if should_run "011"; then
  log "[011] Downloading DLRM-DCNv2 model weights..."
  # Primary: MLCommons-hosted weights (~40 GB)
  fetch_file \
    "https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download" \
    "${MODEL_DIR}/dlrm/model_weights/dlrm_dcn_v2.pt" || \
    warn "[011] DLRM weight download failed – download manually from https://cloud.mlcommons.org"
fi

# ---------------------------------------------------------------------------
# ID 012 – Stable Diffusion XL 1.0
# ---------------------------------------------------------------------------
if should_run "012"; then
  log "[012] Downloading stabilityai/stable-diffusion-xl-base-1.0..."
  hf_download "stabilityai/stable-diffusion-xl-base-1.0" \
    "${MODEL_DIR}/sdxl"
fi

# ---------------------------------------------------------------------------
# ID 013 – RGAT  (checkpoint from MLCommons)
# ---------------------------------------------------------------------------
if should_run "013"; then
  log "[013] Downloading RGAT graph model checkpoint..."
  fetch_file \
    "https://cloud.mlcommons.org/index.php/s/graph_rgat_checkpoint/download" \
    "${MODEL_DIR}/rgat/rgat.pt" || \
    warn "[013] RGAT checkpoint download failed – see https://github.com/mlcommons/inference/tree/master/graph/R-GAT"
fi

# =============================================================================
# Training models / checkpoints
# =============================================================================

# ---------------------------------------------------------------------------
# ID 014 – GPT-J 6B  (base checkpoint for fine-tuning)
# ---------------------------------------------------------------------------
if should_run "014"; then
  log "[014] Downloading EleutherAI/gpt-j-6b..."
  hf_download "EleutherAI/gpt-j-6b" \
    "${MODEL_DIR}/gpt-j-6b"
fi

# ---------------------------------------------------------------------------
# ID 015 – BERT-Large  (SQuAD fine-tune starting point)
# ---------------------------------------------------------------------------
if should_run "015"; then
  log "[015] Downloading bert-large-uncased (base for SQuAD training)..."
  hf_download "bert-large-uncased" \
    "${MODEL_DIR}/bert-large-uncased"
fi

# ---------------------------------------------------------------------------
# ID 016 – 3D U-Net training  (uses same ONNX/checkpoint as ID 008)
# ---------------------------------------------------------------------------
if should_run "016"; then
  log "[016] 3D U-Net training initialises from scratch via nnU-Net – no separate download needed."
fi

# ---------------------------------------------------------------------------
# ID 017 – Stable Diffusion v2.1  (SD training base checkpoint)
# ---------------------------------------------------------------------------
if should_run "017"; then
  log "[017] Downloading Stable Diffusion v2.1 base checkpoint..."
  fetch_file \
    "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt" \
    "${MODEL_DIR}/sd/v2-1_512-ema-pruned.ckpt"
fi

# ---------------------------------------------------------------------------
# ID 018 – BERT-Large NLP pre-training  (no external checkpoint; trains from scratch)
# ---------------------------------------------------------------------------
if should_run "018"; then
  log "[018] BERT-Large pre-training starts from random weights – no checkpoint needed."
fi

# ---------------------------------------------------------------------------
# ID 019 – GPT-3 175B  (Megatron initialises weights; need vocabulary files)
# ---------------------------------------------------------------------------
if should_run "019"; then
  log "[019] Downloading GPT-2 vocabulary + merge files for Megatron-LM..."
  fetch_file \
    "https://huggingface.co/gpt2/resolve/main/vocab.json" \
    "${MODEL_DIR}/gpt3-vocab/gpt2-vocab.json"
  fetch_file \
    "https://huggingface.co/gpt2/resolve/main/merges.txt" \
    "${MODEL_DIR}/gpt3-vocab/gpt2-merges.txt"
  # Also symlink to top-level data dir expected by the training config
  DATA_DIR="${MLPERF_DATA_DIR:-${DEFAULT_DATASET_ROOT}}"
  run ln -sfn "${MODEL_DIR}/gpt3-vocab/gpt2-vocab.json" "${DATA_DIR}/gpt2-vocab.json" 2>/dev/null || true
  run ln -sfn "${MODEL_DIR}/gpt3-vocab/gpt2-merges.txt" "${DATA_DIR}/gpt2-merges.txt" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# ID 020 – Llama 3.1 70B  (same as ID 005 – reuse download)
# ---------------------------------------------------------------------------
if should_run "020"; then
  if dir_has_content "${MODEL_DIR}/llama31-70b" || dir_has_content "${MODEL_DIR}/llama31-70b-base"; then
    log "[020] Llama 3.1 70B already at ${MODEL_DIR}/llama31-70b – skipping re-download"
  else
    log "[020] Downloading Meta-Llama-3.1-70B-Instruct for LoRA fine-tuning..."
    hf_download "meta-llama/Meta-Llama-3.1-70B-Instruct" \
      "${MODEL_DIR}/llama31-70b-base"
  fi
fi

# ---------------------------------------------------------------------------
# ID 021 – DLRMv2 training  (no pretrained checkpoint; trains from scratch)
# ---------------------------------------------------------------------------
if should_run "021"; then
  log "[021] DLRMv2 training initialises weights from scratch – no checkpoint download needed."
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "=== Model download complete ==="
log "Model directory: $MODEL_DIR"
log ""
log "Disk usage:"
if [[ "$DRY_RUN" == "false" ]]; then
  du -sh "${MODEL_DIR}/"* 2>/dev/null | sort -h || true
fi
log ""
log "Next step: run scripts/download_mlperf_datasets.sh"
