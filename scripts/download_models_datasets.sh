#!/usr/bin/env bash
# =============================================================================
# download_models_datasets.sh
#
# Pre-downloads all HuggingFace models and public datasets required by
# node-validation, performance-suite, and MLPerf tests into /mnt/datasets.
#
# Models / datasets larger than ~400 GB are skipped (see SKIP_LARGE below).
#
# Usage:
#   export HF_TOKEN=<your_token>        # required for gated models
#   bash scripts/download_models_datasets.sh
#
# Override storage root:
#   DATASETS_ROOT=/mnt/datasets bash scripts/download_models_datasets.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS_ROOT="${DATASETS_ROOT:-/mnt/datasets}"
HF_HOME="${DATASETS_ROOT}/huggingface"
YOLO_DIR="${DATASETS_ROOT}/yolo"
SHAREGPT_DIR="${DATASETS_ROOT}/sharegpt"

# Token is read from env; never hard-code credentials in source.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN is not set. Export it before running this script."
    echo "        export HF_TOKEN=<your_huggingface_token>"
    exit 1
fi

export HF_TOKEN
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# Free space check (require at least 600 GB)
AVAIL_GB=$(df -BG "${DATASETS_ROOT}" 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || echo 9999)
if (( AVAIL_GB < 600 )); then
    echo "[WARN] Only ${AVAIL_GB} GB free on ${DATASETS_ROOT}. Some downloads may fail."
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] OK  $*"; }
skip() { echo "[$(date '+%H:%M:%S')] SKIP $*"; }

pip_install_once() {
    python3 -c "import huggingface_hub" 2>/dev/null || \
        pip install -q --upgrade huggingface_hub
    python3 -c "import ultralytics" 2>/dev/null || \
        pip install -q ultralytics
}

# Download a model snapshot to HF cache; skip if already present.
hf_download() {
    local repo_id="$1"
    local cache_dir="${HUGGINGFACE_HUB_CACHE}/models--$(echo "${repo_id}" | tr '/' '--')"

    if [[ -d "${cache_dir}/snapshots" ]]; then
        ok "Already cached: ${repo_id}"
        return 0
    fi

    log "Downloading: ${repo_id} ..."
    python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${repo_id}",
    token=os.environ["HF_TOKEN"],
    cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
    ignore_patterns=["*.bin", "*.pt"],   # prefer .safetensors; remove if you need .pt
)
# Second pass: include weights (safetensors or fallback to pytorch .bin)
snapshot_download(
    repo_id="${repo_id}",
    token=os.environ["HF_TOKEN"],
    cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
)
PYEOF
    ok "Done: ${repo_id}"
}

# Download a public dataset to HF cache.
hf_dataset() {
    local dataset_id="$1"
    local cache_dir="${HUGGINGFACE_HUB_CACHE}/datasets--$(echo "${dataset_id}" | tr '/' '--')"

    if [[ -d "${cache_dir}" ]]; then
        ok "Already cached dataset: ${dataset_id}"
        return 0
    fi

    log "Downloading dataset: ${dataset_id} ..."
    python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${dataset_id}",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
    cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
)
PYEOF
    ok "Done dataset: ${dataset_id}"
}

# ---------------------------------------------------------------------------
# Ensure storage root exists
# ---------------------------------------------------------------------------
mkdir -p "${HF_HOME}/hub" "${YOLO_DIR}" "${SHAREGPT_DIR}"
log "Storage root : ${DATASETS_ROOT}"
log "HF cache     : ${HF_HOME}/hub"

# ---------------------------------------------------------------------------
# Install Python helpers
# ---------------------------------------------------------------------------
pip_install_once

# ===========================================================================
# SECTION 1: HuggingFace Models
# Size guide (approx):
#   7-8B  models  ~  15-16 GB
#   70B   models  ~ 130-145 GB
#   72B   models  ~ 140 GB
#   Mixtral 8x22B ~ 280 GB  (MoE, under 400 GB limit)
# Models >400 GB are explicitly skipped.
# ===========================================================================
log "=== Downloading HuggingFace models ==="

# --- Small / base models (always download) ----------------------------------
hf_download "openai-community/gpt2"              # ~500 MB
hf_download "bert-base-uncased"                   # ~440 MB
hf_download "stabilityai/sd-turbo"                # ~3 GB  (Diffusers)
hf_download "google/gemma-2-2b"                   # ~5 GB  (small LLM training)

# --- Medium quantized models ------------------------------------------------
hf_download "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"    # ~15 GB  (quant workload)

# --- 8B instruction models --------------------------------------------------
hf_download "meta-llama/Meta-Llama-3.1-8B-Instruct"      # ~16 GB
hf_download "meta-llama/Meta-Llama-3-8B"                  # ~16 GB  (HF_LLAMA_7B default)

# --- 70B instruction models (~130-145 GB each, all under 400 GB limit) ------
hf_download "meta-llama/Meta-Llama-3.1-70B-Instruct"      # ~140 GB
hf_download "meta-llama/Llama-3.3-70B-Instruct"           # ~140 GB
hf_download "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"   # ~140 GB

# --- 72B model --------------------------------------------------------------
hf_download "Qwen/Qwen2.5-72B-Instruct"                   # ~144 GB

# --- Qwen-72B (legacy name used by some test runs) -------------------------
hf_download "Qwen/Qwen-72B"                               # ~144 GB

# --- Mixtral (MoE, ~280 GB – still under 400 GB limit) --------------------
hf_download "mistralai/Mixtral-8x22B-Instruct-v0.1"       # ~280 GB

# SKIP: DeepSeek-R1 (671B, FP16 ~1.3 TB) – exceeds 400 GB limit
skip "deepseek-ai/DeepSeek-R1 (671B) – exceeds 400 GB limit"

# --- MLPerf model checkpoints -----------------------------------------------
hf_download "openai/whisper-large-v3"           # ~3 GB  (speech)
hf_download "facebook/opt-66b"                  # ~130 GB (DLRM proxy)
hf_download "EleutherAI/gpt-j-6b"              # ~24 GB  (GPT-J MLPerf training)

# ===========================================================================
# SECTION 2: YOLO weights (via ultralytics auto-download)
# ===========================================================================
log "=== Downloading YOLO weights ==="
python3 - <<PYEOF
import os, sys
os.environ["ULTRALYTICS_DIR"] = "${YOLO_DIR}"
from pathlib import Path
try:
    from ultralytics import YOLO
    for model_name in ["yolov8x.pt", "yolov8n.pt"]:
        dest = Path("${YOLO_DIR}") / model_name
        if dest.exists():
            print(f"  Already present: {model_name}")
        else:
            print(f"  Fetching {model_name} ...")
            m = YOLO(model_name)   # downloads to cwd by default
            import shutil, pathlib
            src = pathlib.Path(model_name)
            if src.exists():
                shutil.move(str(src), str(dest))
            print(f"  OK {model_name}")
except Exception as exc:
    print(f"  [WARN] YOLO download failed: {exc}", file=sys.stderr)
PYEOF

# ===========================================================================
# SECTION 3: ShareGPT dataset (used by MLPerf LLM inference vLLM runner)
# ===========================================================================
log "=== Downloading ShareGPT dataset ==="
SHAREGPT_FILE="${SHAREGPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ -f "${SHAREGPT_FILE}" ]]; then
    ok "Already present: ShareGPT_V3_unfiltered_cleaned_split.json"
else
    log "Fetching ShareGPT_V3_unfiltered_cleaned_split.json ..."
    python3 - <<PYEOF
import os, urllib.request, pathlib
url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
dest = pathlib.Path("${SHAREGPT_FILE}")
dest.parent.mkdir(parents=True, exist_ok=True)
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as fh:
    while chunk := resp.read(1024 * 1024):
        fh.write(chunk)
print(f"  OK  {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
PYEOF
fi

# ===========================================================================
# SECTION 4: Emit an env file for test execution
# ===========================================================================
ENV_FILE="${DATASETS_ROOT}/test_env.sh"
log "=== Writing env file: ${ENV_FILE} ==="
cat > "${ENV_FILE}" <<ENVEOF
# Auto-generated by scripts/download_models_datasets.sh
# Source this before running pytest: source ${DATASETS_ROOT}/test_env.sh

export DATASETS_ROOT="${DATASETS_ROOT}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${DATASETS_ROOT}/torch"
export SHAREGPT_DATASET_PATH="${SHAREGPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"

# HuggingFace model IDs → all resolved from local cache
export HF_INFERENCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TRAINING_MODEL="bert-base-uncased"
export HF_DEEPSEEK_R1_QUANT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export HF_LLAMA_7B_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_LLAMA_70B_MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
export HF_QWEN_70B_MODEL="Qwen/Qwen2.5-72B-Instruct"
export HF_LLAMA_33_70B_MODEL="meta-llama/Llama-3.3-70B-Instruct"
export HF_MIXTRAL_8X22B_MODEL="mistralai/Mixtral-8x22B-Instruct-v0.1"
export HF_DEEPSEEK_R1_70B_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# Diffusers / YOLO
export DIFFUSERS_MODEL_ID="stabilityai/sd-turbo"
export YOLO_DETECT_MODEL="${YOLO_DIR}/yolov8x.pt"

# MLPerf dataset/model root
export MLPERF_DATA_DIR="${DATASETS_ROOT}"
export MLPERF_MODEL_DIR="${DATASETS_ROOT}"

# vLLM KV offload model
export NODE_VLLM_KV_OFFLOAD_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
ENVEOF

ok "Env file written."

# ===========================================================================
log "=== All downloads complete ==="
log "Disk usage: $(du -sh "${DATASETS_ROOT}" 2>/dev/null | cut -f1) used in ${DATASETS_ROOT}"
log ""
log "Before running tests, source the env file:"
log "  source ${DATASETS_ROOT}/test_env.sh"
log "  python -m pytest tests/ ..."
