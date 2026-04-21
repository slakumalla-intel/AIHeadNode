#!/usr/bin/env bash
# =============================================================================
# download_llama31_70b.sh
#
# Downloads Meta-Llama-3.1-70B-Instruct from HuggingFace Hub into a local
# cache directory and prints the env-var export to use with the test suite.
#
# Usage:
#   bash scripts/download_llama31_70b.sh [OPTIONS]
#
# Options:
#   --model-dir DIR     Local directory to save the model  (default: /models/llama)
#   --hf-token TOKEN    HuggingFace access token           (or set HF_TOKEN)
#   --base              Download base (non-instruct) variant instead
#   --dry-run           Print commands but do not execute them
#
# Prerequisites:
#   pip install -U "huggingface_hub[cli]"
#
# Notes:
#   - Llama 3.1 is a gated model. You MUST accept the licence at:
#       https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
#     and pass a valid HF_TOKEN.
#   - FP16 shards are ~140 GB. Ensure enough disk space before downloading.
#   - After download, set this env var before running the test suite:
#       export HF_LLAMA_70B_MODEL=/models/llama/Meta-Llama-3.1-70B-Instruct
#       export HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_DIR="${LLAMA_MODEL_DIR:-/models/llama}"
HF_TOKEN="${HF_TOKEN:-}"
USE_BASE=false
DRY_RUN=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)  MODEL_DIR="$2";  shift 2 ;;
    --hf-token)   HF_TOKEN="$2";   shift 2 ;;
    --base)       USE_BASE=true;   shift   ;;
    --dry-run)    DRY_RUN=true;    shift   ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve model repo and local path
# ---------------------------------------------------------------------------
if $USE_BASE; then
  REPO_ID="meta-llama/Meta-Llama-3.1-70B"
else
  REPO_ID="meta-llama/Meta-Llama-3.1-70B-Instruct"
fi

MODEL_NAME="${REPO_ID##*/}"          # Meta-Llama-3.1-70B-Instruct
LOCAL_DIR="${MODEL_DIR}/${MODEL_NAME}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

run() {
  if $DRY_RUN; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ -z "$HF_TOKEN" ]]; then
  error "HF_TOKEN is not set. Pass --hf-token or export HF_TOKEN before running.
  Get your token at: https://huggingface.co/settings/tokens
  Accept the model licence at: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct"
fi

command -v huggingface-cli &>/dev/null || {
  info "huggingface-cli not found — installing huggingface_hub[cli] ..."
  run pip install -U "huggingface_hub[cli]"
}

# Check available disk space (need ~145 GB for FP16 shards)
REQUIRED_GB=145
if command -v df &>/dev/null; then
  AVAIL_GB=$(df -BG "$MODEL_DIR" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}' || echo 0)
  if [[ "${AVAIL_GB:-0}" -lt "$REQUIRED_GB" ]]; then
    warn "Available disk space on $MODEL_DIR appears to be ${AVAIL_GB}GB — at least ${REQUIRED_GB}GB recommended for FP16 weights."
  fi
fi

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
info "Repo    : $REPO_ID"
info "Local   : $LOCAL_DIR"
info "Starting download — this will take a while (~140 GB) ..."

mkdir -p "$LOCAL_DIR"

run huggingface-cli download \
  "$REPO_ID" \
  --token "$HF_TOKEN" \
  --local-dir "$LOCAL_DIR" \
  --local-dir-use-symlinks False

info "Download complete: $LOCAL_DIR"

# ---------------------------------------------------------------------------
# Verify key files
# ---------------------------------------------------------------------------
if ! $DRY_RUN; then
  EXPECTED_FILES=(
    "config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "generation_config.json"
  )
  MISSING=0
  for f in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "$LOCAL_DIR/$f" ]]; then
      warn "Expected file missing: $LOCAL_DIR/$f"
      MISSING=$((MISSING + 1))
    fi
  done

  SHARD_COUNT=$(find "$LOCAL_DIR" -name "model-*.safetensors" 2>/dev/null | wc -l)
  info "Safetensor shards found: $SHARD_COUNT (expected ~30)"

  if [[ $MISSING -gt 0 ]]; then
    warn "$MISSING expected config files are missing — download may be incomplete."
  fi
fi

# ---------------------------------------------------------------------------
# Print env-var exports for the test suite
# ---------------------------------------------------------------------------
cat <<EOF

=====================================================================
  Model downloaded to: $LOCAL_DIR
=====================================================================
  Run the tests with these env vars to use the local model and
  avoid any further remote downloads:

    export HF_LLAMA_70B_MODEL="$LOCAL_DIR"
    export HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0
    export HF_ENABLE_LARGE_MODELS=1
    export HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=60

  Then run the validation suite:

    bash run_all_validation_tests.sh 2>&1 | tee validation_run.log

  Or run just the 70B workload test:

    HF_LLAMA_70B_MODEL="$LOCAL_DIR" \\
    HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0 \\
    pytest -vv tests/test_node_validation.py \\
           -k "llama_70b" --log-cli-level=INFO

=====================================================================
EOF
