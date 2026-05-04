"""MLPerf inference and training benchmark suite for 4x H100 SXM5 validation.

Test IDs: AI_HOSTCPU_ML_TEST_ID_001 – AI_HOSTCPU_ML_TEST_ID_021

Category structure:
  001-005  ML Perf / LLAMA                            (inference: text gen, reasoning, MoE, summarization, Q&A)
  006-013  ML Perf / Vision · Speech · Recs · Graph   (inference: image, speech, recommender, diffusion, graph)
  014-021  ML Perf / MLPerf Training (Datacenter)     (training: GPT-J, BERT, UNet, SD, GPT-3, Llama2-LoRA, DLRM)

Execution control (environment variables):
  ML_EXECUTE_CASES=1           – run benchmark commands  (default: dry-run catalog only)
  ML_CASE_IDS=001,005          – comma-separated IDs or short numbers to run a subset
    ML_SKIP_LARGE_CASES=1        – skip cases 001/002 unless explicitly disabled
  ML_CMD_TIMEOUT_SEC=3600      – per-command timeout seconds (default 3600)
  ML_SUITE_KPI_FILE            – JSONL output path (default: artifacts/mlperf_suite_kpis.jsonl)

GPU / path configuration (4×H100 SXM5 defaults):
  ML_NUM_GPUS=4
  ML_TENSOR_PARALLEL=4
    MLPERF_DATA_DIR=<repo>/datasets      – root dataset/model mount
    MLPERF_MODEL_DIR=<repo>/datasets     – root dataset/model mount
  MLPERF_OUTPUT_DIR=artifacts/mlperf_results
  MLPERF_INFERENCE_DIR=/root/workloads/mlcommons/inference
  MLPERF_TRAINING_DIR=/root/workloads/mlcommons/training
  MLPERF_LOADGEN_SCENARIO=Offline      – Offline | Server | SingleStream

KPIs are written to ML_SUITE_KPI_FILE (JSONL) via _record_kpi() for every case
whether in catalog or execution mode.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer /mnt/datasets when the download script has pre-populated it.
_DATASETS_MOUNT = Path(os.environ.get("DATASETS_ROOT", "/mnt/datasets"))
DEFAULT_DATASET_ROOT = _DATASETS_MOUNT if _DATASETS_MOUNT.exists() else REPO_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

def _configure_mlperf_env_defaults() -> None:
    os.environ.setdefault("ML_EXECUTE_CASES", "1")
    os.environ.setdefault("ML_SKIP_LARGE_CASES", "1")
    os.environ.setdefault("ML_CMD_TIMEOUT_SEC", "3600")
    os.environ.setdefault("ML_SUITE_KPI_FILE", "artifacts/mlperf_suite_kpis.jsonl")
    _detected_gpus = str(max(
        len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()])
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() else 0,
        1,
    ))
    os.environ.setdefault("ML_NUM_GPUS", _detected_gpus)
    os.environ.setdefault("ML_TENSOR_PARALLEL", _detected_gpus)
    os.environ.setdefault("MLPERF_DATA_DIR", str(DEFAULT_DATASET_ROOT))
    os.environ.setdefault("MLPERF_MODEL_DIR", str(DEFAULT_DATASET_ROOT))
    os.environ.setdefault("MLPERF_OUTPUT_DIR", "artifacts/mlperf_results")
    os.environ.setdefault("MLPERF_LOADGEN_SCENARIO", "Offline")
    os.environ.setdefault("MLPERF_INFERENCE_DIR", "/root/workloads/mlcommons/inference")
    os.environ.setdefault("MLPERF_TRAINING_DIR", "/root/workloads/mlcommons/training")
    os.environ.setdefault("ML_VLLM_GPU_MEMORY_UTIL", "0.90")
    os.environ.setdefault("ML_AUTO_INSTALL", "1" if sys.platform.startswith("linux") else "0")
    os.environ.setdefault("ML_AUTO_INSTALL_TIMEOUT_SEC", "600")


_configure_mlperf_env_defaults()


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLPerfCase:
    case_id: str
    title: str
    category: str           # always "ML Perf"
    sub_category: str       # "LLAMA" | "Vision / Speech / Recs / Graph" | "MLPerf Training (Datacenter) model list"
    description: str        # matches TCD Description column verbatim
    kpi_keys: List[str]     # KPI metric keys expected in extracted output
    prerequisites: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    expected: str = ""
    planned_only: bool = False


# ---------------------------------------------------------------------------
# KPI output helpers
# ---------------------------------------------------------------------------

def _kpi_output_path() -> Path:
    return Path(os.environ.get("ML_SUITE_KPI_FILE", "artifacts/mlperf_suite_kpis.jsonl"))


def _record_kpi(case_id: str, metrics: Dict) -> None:
    """Append a single KPI record to the JSONL output file."""
    out_path = _kpi_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case_id": case_id,
        "metrics": metrics,
    }
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
    logger.info("KPI %s: %s", case_id, json.dumps(metrics, default=str, sort_keys=True))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _visible_gpu_count() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        return max(0, len([x for x in raw.split(",") if x.strip()]))
    return int(os.environ.get("ML_NUM_GPUS", "4"))


def _tensor_parallel_size() -> int:
    return int(os.environ.get("ML_TENSOR_PARALLEL", str(_visible_gpu_count())))


def _selected_case_ids() -> Optional[set]:
    raw = os.environ.get("ML_CASE_IDS", "").strip()
    if not raw:
        return None

    def _normalize(token: str) -> str:
        cleaned = token.strip().strip('"').strip("'")
        if not cleaned:
            return ""
        m = re.fullmatch(r"AI_HOSTCPU_ML_TEST_ID_(\d{1,3})", cleaned, flags=re.IGNORECASE)
        if m:
            return f"AI_HOSTCPU_ML_TEST_ID_{int(m.group(1)):03d}"
        m = re.fullmatch(r"\d{1,3}", cleaned)
        if m:
            return f"AI_HOSTCPU_ML_TEST_ID_{int(cleaned):03d}"
        return cleaned

    normalized = {_normalize(x) for x in raw.split(",") if x.strip()}
    normalized.discard("")
    return normalized


def _skip_large_cases() -> bool:
    return _env_flag("ML_SKIP_LARGE_CASES", "1")


def _is_selected(case_id: str) -> bool:
    if _skip_large_cases() and case_id in {
        "AI_HOSTCPU_ML_TEST_ID_001",
        "AI_HOSTCPU_ML_TEST_ID_002",
    }:
        return False
    selected = _selected_case_ids()
    return selected is None or case_id in selected


def _binary_exists(token: str) -> bool:
    candidate = "python" if token == "python3" else token
    if "/" in candidate or "\\" in candidate:
        p = Path(candidate)
        return p.exists() and os.access(p, os.X_OK)
    return shutil.which(candidate) is not None


# ---------------------------------------------------------------------------
# Command preparation
# ---------------------------------------------------------------------------

def _prepare_command(command: str) -> str:
    """Expand env-var placeholders and adjust runtime parameters."""
    cmd = command.strip()

    # Expand env-var style placeholders
    substitutions = {
        "${ML_TENSOR_PARALLEL}": str(_tensor_parallel_size()),
        "${ML_NUM_GPUS}": str(_visible_gpu_count()),
        "${MLPERF_DATA_DIR}": os.environ.get("MLPERF_DATA_DIR", str(DEFAULT_DATASET_ROOT)),
        "${MLPERF_MODEL_DIR}": os.environ.get("MLPERF_MODEL_DIR", str(DEFAULT_DATASET_ROOT)),
        "${MLPERF_OUTPUT_DIR}": os.environ.get("MLPERF_OUTPUT_DIR", "artifacts/mlperf_results"),
        "${MLPERF_INFERENCE_DIR}": os.environ.get("MLPERF_INFERENCE_DIR", "/root/workloads/mlcommons/inference"),
        "${MLPERF_TRAINING_DIR}": os.environ.get("MLPERF_TRAINING_DIR", "/root/workloads/mlcommons/training"),
        "${MLPERF_LOADGEN_SCENARIO}": os.environ.get("MLPERF_LOADGEN_SCENARIO", "Offline"),
    }
    for placeholder, value in substitutions.items():
        cmd = cmd.replace(placeholder, value)

    # Use active Python interpreter for python3 commands
    if cmd.startswith("python3 ") or cmd == "python3":
        cmd = sys.executable + cmd[7:]

    # Clamp torchrun --nproc_per_node to available GPU count
    cmd = re.sub(
        r"(torchrun\s+--nproc_per_node=)\d+",
        lambda m: f"{m.group(1)}{_visible_gpu_count()}",
        cmd,
    )

    # Clamp vllm --tensor-parallel-size to available GPU count
    def _clamp_tp(m: re.Match) -> str:
        requested = int(m.group(2))
        clamped = min(requested, _visible_gpu_count())
        return f"{m.group(1)}{clamped}"

    cmd = re.sub(r"(--tensor-parallel-size\s+)(\d+)", _clamp_tp, cmd)

    return cmd


def _run_command(command: str, timeout_sec: int) -> Dict:
    """Execute a single shell command and return structured result including stdout/stderr tails."""
    start = time.perf_counter()
    prepared = _prepare_command(command)
    logger.info("Executing MLPerf command: %s", prepared)
    try:
        result = subprocess.run(
            prepared,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "command": command,
            "executed_command": prepared,
            "returncode": 124,
            "elapsed_sec": round(elapsed, 3),
            "stdout_tail": (exc.stdout or "")[-6000:],
            "stderr_tail": (exc.stderr or "")[-6000:],
        }
    elapsed = time.perf_counter() - start
    return {
        "command": command,
        "executed_command": prepared,
        "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": (result.stdout or "")[-6000:],
        "stderr_tail": (result.stderr or "")[-6000:],
    }


# ---------------------------------------------------------------------------
# KPI Parsers
# ---------------------------------------------------------------------------

def _extract_vllm_kpis(stdout: str, stderr: str) -> Dict[str, float]:
    """Parse vLLM benchmark_serving console output into structured KPIs.

    Captures:
      throughput_requests_per_sec, throughput_tok_per_sec,
      ttft_p{50|90|99}_ms, itl_p{50|90|99}_ms,
      e2e_lat_p{50|90|99}_ms
    """
    kpis: Dict[str, float] = {}
    combined = stdout + "\n" + stderr

    # Throughput line: "Throughput: 12.34 requests/s, 5678.9 output tokens/s"
    m = re.search(
        r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*(?:output\s+)?tokens/s",
        combined,
    )
    if m:
        kpis["throughput_requests_per_sec"] = float(m.group(1))
        kpis["throughput_tok_per_sec"] = float(m.group(2))

    for pct in (50, 90, 99):
        # Time-to-first-token
        m = re.search(
            rf"[Pp]{pct}\s+TTFT[^\d]+([\d.]+)\s*ms", combined, re.IGNORECASE
        )
        if m:
            kpis[f"ttft_p{pct}_ms"] = float(m.group(1))

        # Inter-token latency
        m = re.search(
            rf"[Pp]{pct}\s+ITL[^\d]+([\d.]+)\s*ms", combined, re.IGNORECASE
        )
        if m:
            kpis[f"itl_p{pct}_ms"] = float(m.group(1))

        # End-to-end latency
        m = re.search(
            rf"[Pp]{pct}\s+E2E\s*(?:[Ll]atency)?[^\d]+([\d.]+)\s*ms",
            combined,
            re.IGNORECASE,
        )
        if m:
            kpis[f"e2e_lat_p{pct}_ms"] = float(m.group(1))

    return kpis


def _extract_loadgen_kpis(stdout: str, stderr: str) -> Dict[str, float]:
    """Parse MLCommons LoadGen result_summary.txt style output into KPIs.

    Captures:
      samples_per_sec, p99_latency_ms, mean_latency_ms,
      accuracy / mAP / auc / wer / rouge / dice / f1,
      result_valid (1.0 = VALID, 0.0 = INVALID)
    """
    kpis: Dict[str, float] = {}
    combined = stdout + "\n" + stderr

    # Throughput
    for pat in [
        r"samples_per_second\s*[=:]\s*([\d.]+)",
        r"Throughput\s*[=:]\s*([\d.]+)\s*qps",
        r"\bQPS\s*[=:]\s*([\d.]+)",
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            kpis["samples_per_sec"] = float(m.group(1))
            break

    # P99 latency (LoadGen reports in nanoseconds)
    m = re.search(r"99th\s+percentile\s+latency\s*\(ns\)\s*[=:]\s*(\d+)", combined, re.IGNORECASE)
    if m:
        kpis["p99_latency_ms"] = round(float(m.group(1)) / 1e6, 3)

    # Mean latency
    m = re.search(r"mean\s+latency\s*\(ns\)\s*[=:]\s*(\d+)", combined, re.IGNORECASE)
    if m:
        kpis["mean_latency_ms"] = round(float(m.group(1)) / 1e6, 3)

    # Task-specific accuracy metrics (capture whichever appears)
    for metric_name, pat in [
        ("accuracy", r"\baccuracy\s*[=:\s]+([\d.]+)\s*%?"),
        ("mAP",      r"\bmAP\s*[=:\s]+([\d.]+)"),
        ("auc",      r"\bAUC\s*[=:\s]+([\d.]+)"),
        ("wer",      r"\bWER\s*[=:\s]+([\d.]+)\s*%?"),
        ("rouge",    r"\bROUGE[_\-]?[12Lf]?\s*[=:\s]+([\d.]+)"),
        ("dice",     r"\bDice\s*[=:\s]+([\d.]+)"),
        ("clip_score", r"\bCLIP[_\s]?[Ss]core\s*[=:\s]+([\d.]+)"),
        ("fid",      r"\bFID\s*[=:\s]+([\d.]+)"),
        ("f1",       r"\bF1\s*[=:\s]+([\d.]+)"),
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            kpis[metric_name.lower()] = float(m.group(1))

    # LoadGen validity flag
    m = re.search(r"result_validity\s*[=:\s]+(\w+)", combined, re.IGNORECASE)
    if m:
        kpis["result_valid"] = 1.0 if m.group(1).upper() in {"VALID", "TRUE", "PASS"} else 0.0

    return kpis


def _extract_training_kpis(stdout: str, stderr: str) -> Dict[str, float]:
    """Parse training job stdout/stderr into structured KPIs.

    Captures:
      samples_per_sec, time_to_train_min, elapsed_sec,
      final_loss, gpu_util_pct,
      target_* (bleu / rouge / f1 / accuracy / auc)
    """
    kpis: Dict[str, float] = {}
    combined = stdout + "\n" + stderr

    # Training throughput
    for pat in [
        r"(?:throughput|samples/s(?:ec)?|seqs/s(?:ec)?|seq[s]?/sec)[^\d]+([\d.]+)",
        r"([\d.]+)\s+(?:samples|sequences|seqs|tokens)\s*/\s*(?:sec|s)\b",
        r"it/s[=:\s]+([\d.]+)",
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            kpis["samples_per_sec"] = float(m.group(1))
            break

    # Final loss (last occurrence wins)
    for pat in [
        r"(?:train_loss|final_loss|val_loss|loss)\s*[=:[\s]+([\d.]+(?:\.\d+)?)",
        r"\"loss\":\s*([\d.]+)",
    ]:
        for m in re.finditer(pat, combined, re.IGNORECASE):
            try:
                kpis["final_loss"] = float(m.group(1))
            except ValueError:
                pass

    # time_to_train in minutes (MLPerf training benchmark emits this key)
    m = re.search(r"time_to_train\s*[=:[\s]+([\d.]+)", combined, re.IGNORECASE)
    if m:
        kpis["time_to_train_min"] = float(m.group(1))

    # Total elapsed time in seconds (fallback)
    m = re.search(r"total\s+(?:training\s+)?time[^\d]+([\d.]+)\s*s(?:ec(?:onds?)?)?", combined, re.IGNORECASE)
    if m:
        kpis["elapsed_sec"] = float(m.group(1))

    # GPU utilization
    m = re.search(r"gpu\s*util(?:ization)?\s*[=:[\s]+([\d.]+)\s*%?", combined, re.IGNORECASE)
    if m:
        kpis["gpu_util_pct"] = float(m.group(1))

    # Target quality metrics
    for metric_name, pat in [
        ("bleu",    r"\bbleu[_\-]?\d*\s*[=:\s]+([\d.]+)"),
        ("rouge1",  r"\brouge[_\-]?1\s*[=:\s]+([\d.]+)"),
        ("rouge2",  r"\brouge[_\-]?2\s*[=:\s]+([\d.]+)"),
        ("f1",      r"\bF1\s*[=:\s]+([\d.]+)"),
        ("accuracy",r"\baccuracy\s*[=:\s]+([\d.]+)\s*%?"),
        ("auc",     r"\bAUC\s*[=:\s]+([\d.]+)"),
        ("fid",     r"\bFID\s*[=:\s]+([\d.]+)"),
    ]:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            kpis[f"target_{metric_name}"] = float(m.group(1))

    return kpis


def _select_kpi_extractor(sub_category: str):
    if sub_category == "LLAMA":
        return _extract_vllm_kpis
    if sub_category == "MLPerf Training (Datacenter) model list":
        return _extract_training_kpis
    return _extract_loadgen_kpis


# ---------------------------------------------------------------------------
# Test case catalogue
# ---------------------------------------------------------------------------

MLPERF_CASES: List[MLPerfCase] = [

    # =========================================================================
    # Subcategory: LLAMA  (IDs 001–005)  — MLPerf Inference – LLM workloads
    # =========================================================================

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_001",
        title="Llama 3.1 405B – text generation / long-context inference",
        category="ML Perf",
        sub_category="LLAMA",
        description="Llama 3.1 405B (text generation / long context style workload)",
        kpi_keys=[
            "throughput_tok_per_sec",
            "ttft_p50_ms", "ttft_p99_ms",
            "itl_p50_ms", "itl_p99_ms",
            "e2e_lat_p99_ms",
        ],
        prerequisites=[
            "vLLM >=0.5 installed with FP8 support",
            "meta-llama/Llama-3.1-405B-Instruct model downloaded",
            "4x H100 SXM5 (320 GB total HBM3) with NVLink 4",
            "HuggingFace token set (HF_TOKEN env var)",
        ],
        commands=[
            # Step 1: Start vLLM server (non-blocking; run in background in the real workflow)
            "vllm serve meta-llama/Llama-3.1-405B-Instruct "
            "--quantization fp8 "
            "--tensor-parallel-size ${ML_TENSOR_PARALLEL} "
            "--max-model-len 131072 "
            "--gpu-memory-utilization 0.90 "
            "--trust-remote-code "
            "--enforce-eager "
            "--port 8000",
            # Step 2: Benchmark
            "python3 -m vllm.benchmarks.benchmark_serving "
            "--backend vllm "
            "--model meta-llama/Llama-3.1-405B-Instruct "
            "--dataset-name random "
            "--random-input-len 8192 "
            "--random-output-len 2048 "
            "--num-prompts 200 "
            "--percentiles 50,90,99 "
            "--port 8000",
        ],
        expected=(
            "Offline throughput >500 output tokens/sec on 4×H100 FP8; "
            "P99 TTFT <30 s for 8k-in/2k-out; no OOM"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_002",
        title="DeepSeek-R1 – reasoning inference",
        category="ML Perf",
        sub_category="LLAMA",
        description="DeepSeek-R1 (reasoning)",
        kpi_keys=[
            "throughput_tok_per_sec",
            "ttft_p50_ms", "ttft_p99_ms",
            "itl_p50_ms", "itl_p99_ms",
            "e2e_lat_p99_ms",
        ],
        prerequisites=[
            "vLLM >=0.5 installed",
            "deepseek-ai/DeepSeek-R1 model downloaded (671B MoE; ~350 GB in BF16)",
            "4x H100 SXM5 + --trust-remote-code",
        ],
        commands=[
            "vllm serve deepseek-ai/DeepSeek-R1 "
            "--tensor-parallel-size ${ML_TENSOR_PARALLEL} "
            "--dtype bfloat16 "
            "--max-model-len 32768 "
            "--gpu-memory-utilization 0.90 "
            "--trust-remote-code "
            "--port 8000",
            "python3 -m vllm.benchmarks.benchmark_serving "
            "--backend vllm "
            "--model deepseek-ai/DeepSeek-R1 "
            "--dataset-name random "
            "--random-input-len 2048 "
            "--random-output-len 4096 "
            "--num-prompts 200 "
            "--percentiles 50,90,99 "
            "--port 8000",
        ],
        expected=(
            "Output throughput >1500 tokens/sec on 4×H100 BF16; "
            "reasoning chain (4k output) P99 latency <120 s"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_003",
        title="Mixtral 8×7B – MoE text generation",
        category="ML Perf",
        sub_category="LLAMA",
        description="Mixtral 8×7B (MoE text generation)",
        kpi_keys=[
            "throughput_tok_per_sec",
            "ttft_p50_ms", "ttft_p99_ms",
            "itl_p50_ms", "itl_p99_ms",
            "e2e_lat_p99_ms",
        ],
        prerequisites=[
            "vLLM >=0.5 installed",
            "mistralai/Mixtral-8x7B-Instruct-v0.1 model downloaded",
            f"ShareGPT dataset at ${{MLPERF_DATA_DIR}}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json (also: {os.environ.get('SHAREGPT_DATASET_PATH', '')})",
        ],
        commands=[
            "vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 "
            "--tensor-parallel-size ${ML_TENSOR_PARALLEL} "
            "--max-model-len 32768 "
            "--gpu-memory-utilization 0.90 "
            "--port 8000",
            "python3 -m vllm.benchmarks.benchmark_serving "
            "--backend vllm "
            "--model mistralai/Mixtral-8x7B-Instruct-v0.1 "
            "--dataset-name sharegpt "
            "--dataset-path ${MLPERF_DATA_DIR}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json "
            "--num-prompts 1000 "
            "--percentiles 50,90,99 "
            "--port 8000",
        ],
        expected=(
            "MoE throughput >5000 tokens/sec on 4×H100; "
            "P99 TTFT <1 s; MoE routing overhead <10% vs Llama-8B dense"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_004",
        title="Llama 3.1 8B – summarization inference",
        category="ML Perf",
        sub_category="LLAMA",
        description="Llama 3.1 8B (summarization)",
        kpi_keys=[
            "throughput_tok_per_sec",
            "ttft_p50_ms", "ttft_p99_ms",
            "itl_p50_ms", "itl_p99_ms",
            "e2e_lat_p99_ms",
        ],
        prerequisites=[
            "vLLM >=0.5 installed",
            "meta-llama/Meta-Llama-3.1-8B-Instruct model downloaded",
            "ShareGPT dataset at ${MLPERF_DATA_DIR}/sharegpt/",
        ],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct "
            "--tensor-parallel-size ${ML_TENSOR_PARALLEL} "
            "--max-model-len 8192 "
            "--gpu-memory-utilization 0.90 "
            "--port 8000",
            "python3 -m vllm.benchmarks.benchmark_serving "
            "--backend vllm "
            "--model meta-llama/Meta-Llama-3.1-8B-Instruct "
            "--dataset-name sharegpt "
            "--dataset-path ${MLPERF_DATA_DIR}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json "
            "--num-prompts 1000 "
            "--percentiles 50,90,99 "
            "--port 8000",
        ],
        expected=(
            "Offline throughput >12000 tokens/sec on 4×H100; "
            "P99 TTFT <500 ms; P99 ITL <50 ms"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_005",
        title="Llama 2 70B – Q&A inference",
        category="ML Perf",
        sub_category="LLAMA",
        description="Llama 2 70B (Q&A)",
        kpi_keys=[
            "throughput_tok_per_sec",
            "ttft_p50_ms", "ttft_p99_ms",
            "itl_p50_ms", "itl_p99_ms",
            "e2e_lat_p99_ms",
        ],
        prerequisites=[
            "vLLM >=0.5 installed",
            "meta-llama/Llama-2-70b-chat-hf model downloaded",
            "4x H100 SXM5",
        ],
        commands=[
            "vllm serve meta-llama/Llama-2-70b-chat-hf "
            "--tensor-parallel-size ${ML_TENSOR_PARALLEL} "
            "--dtype bfloat16 "
            "--max-model-len 4096 "
            "--gpu-memory-utilization 0.90 "
            "--port 8000",
            "python3 -m vllm.benchmarks.benchmark_serving "
            "--backend vllm "
            "--model meta-llama/Llama-2-70b-chat-hf "
            "--dataset-name sharegpt "
            "--dataset-path ${MLPERF_DATA_DIR}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json "
            "--num-prompts 1000 "
            "--percentiles 50,90,99 "
            "--port 8000",
        ],
        expected=(
            "Offline throughput >3000 tokens/sec on 4×H100 BF16; "
            "P99 TTFT <2 s; no OOM on 4×80 GB"
        ),
    ),

    # =========================================================================
    # Subcategory: Vision / Speech / Recs / Graph  (IDs 006–013)
    # =========================================================================

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_006",
        title="ResNet-50 v1.5 – image classification",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="ResNet-50 v1.5 (image classification)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "accuracy", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "ImageNet-2012 validation set (50 000 images) at ${MLPERF_DATA_DIR}/imagenet",
            "ResNet-50 ONNX weights at ${MLPERF_MODEL_DIR}/resnet50/resnet50_v1.onnx",
            "onnxruntime-gpu or TensorRT backend",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/vision/classification_and_detection/run_local.sh "
            "pytorch resnet50 gpu "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/imagenet "
            "--output ${MLPERF_OUTPUT_DIR}/resnet50 "
            "--count 50000",
            "python3 ${MLPERF_INFERENCE_DIR}/vision/classification_and_detection/tools/accuracy-imagenet.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/resnet50/mlperf_log_accuracy.json "
            "--imagenet-val-file ${MLPERF_DATA_DIR}/imagenet/val_map.txt",
        ],
        expected=(
            "Offline throughput >100000 samples/sec on 4×H100 FP16; "
            "top-1 accuracy >=76.46% (MLPerf v4.1 target)"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_007",
        title="RetinaNet – object detection",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="RetinaNet (object detection)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "mAP", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "OpenImages v6 validation subset (24 781 images) at ${MLPERF_DATA_DIR}/openimages",
            "RetinaNet ResNeXt-50 ONNX weights at ${MLPERF_MODEL_DIR}/retinanet/",
            "onnxruntime-gpu or TensorRT backend",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/vision/classification_and_detection/run_local.sh "
            "pytorch retinanet gpu "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/openimages "
            "--output ${MLPERF_OUTPUT_DIR}/retinanet "
            "--count 24781",
            "python3 ${MLPERF_INFERENCE_DIR}/vision/classification_and_detection/tools/accuracy-openimages.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/retinanet/mlperf_log_accuracy.json "
            "--openimages-dir ${MLPERF_DATA_DIR}/openimages",
        ],
        expected=(
            "Offline throughput >5000 samples/sec on 4×H100; "
            "mAP >=37.55 (MLPerf v4.1 target)"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_008",
        title="3D U-Net KiTS19 – medical image segmentation",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="3D U-Net (KiTS) (medical segmentation)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "dice", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "KiTS19 dataset (42 validation volumes) at ${MLPERF_DATA_DIR}/kits19",
            "3D U-Net ONNX weights at ${MLPERF_MODEL_DIR}/3d-unet/",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/medical_imaging/3d-unet-kits19/run_local.sh "
            "pytorch gpu "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/kits19 "
            "--output ${MLPERF_OUTPUT_DIR}/3dunet "
            "--count 42",
            "python3 ${MLPERF_INFERENCE_DIR}/medical_imaging/3d-unet-kits19/accuracy-kits.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/3dunet/mlperf_log_accuracy.json "
            "--label-dir ${MLPERF_DATA_DIR}/kits19",
        ],
        expected=(
            "Offline throughput >7 samples/sec on 4×H100; "
            "mean Dice score >=0.908 (MLPerf v4.1 target)"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_009",
        title="RNN-T – automatic speech recognition",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="RNN-T (speech-to-text)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "wer", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "LibriSpeech test-clean (~5.4 h, 2 513 utterances) at ${MLPERF_DATA_DIR}/librispeech",
            "RNN-T PyTorch checkpoint at ${MLPERF_MODEL_DIR}/rnnt/rnnt.pt",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/speech_recognition/rnnt/run_local.sh "
            "pytorch rnnt gpu "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/librispeech "
            "--output ${MLPERF_OUTPUT_DIR}/rnnt "
            "--count 2513",
            "python3 ${MLPERF_INFERENCE_DIR}/speech_recognition/rnnt/accuracy.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/rnnt/mlperf_log_accuracy.json "
            "--dataset-path ${MLPERF_DATA_DIR}/librispeech "
            "--manifest ${MLPERF_DATA_DIR}/librispeech/librispeech-dev-clean-wav.json",
        ],
        expected=(
            "Offline throughput >12000 utterances/sec on 4×H100; "
            "WER <=7.452431 (MLPerf v4.1 target)"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_010",
        title="Whisper – automatic speech recognition",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="Whisper (speech-to-text)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "wer", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "openai/whisper-large-v2 model downloaded",
            "LibriSpeech test-clean at ${MLPERF_DATA_DIR}/librispeech",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/speech_recognition/whisper/run_local.sh "
            "pytorch gpu "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/librispeech "
            "--output ${MLPERF_OUTPUT_DIR}/whisper "
            "--model openai/whisper-large-v2 "
            "--count 2620",
            "python3 ${MLPERF_INFERENCE_DIR}/speech_recognition/whisper/accuracy.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/whisper/mlperf_log_accuracy.json "
            "--dataset-path ${MLPERF_DATA_DIR}/librispeech",
        ],
        expected=(
            "Offline throughput >150 utterances/sec on 4×H100; "
            "WER <3.0% on LibriSpeech test-clean"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_011",
        title="DLRM / DLRM-DCNv2 – recommender system",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="DLRM / DLRM-DCNv2 (recommender)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "auc", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "Criteo Terabyte Day 23 dataset at ${MLPERF_DATA_DIR}/criteo",
            "DLRM-DCNv2 model weights at ${MLPERF_MODEL_DIR}/dlrm/model_weights",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/recommendation/dlrm_v2/pytorch/run.py "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/criteo "
            "--model-path ${MLPERF_MODEL_DIR}/dlrm/model_weights "
            "--output ${MLPERF_OUTPUT_DIR}/dlrm "
            "--num-samples 204800 "
            "--count 204800 "
            "--num-processes 4",
            "python3 ${MLPERF_INFERENCE_DIR}/recommendation/dlrm_v2/pytorch/accuracy.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/dlrm/mlperf_log_accuracy.json "
            "--day-23-file ${MLPERF_DATA_DIR}/criteo/day_23",
        ],
        expected=(
            "Offline throughput >60 M samples/sec on 4×H100; "
            "AUC >=0.8025 (MLPerf v4.1 target)"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_012",
        title="SDXL 1.0 – text-to-image generation",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="SDXL 1.0 (text-to-image generation)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "fid", "clip_score", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "stabilityai/stable-diffusion-xl-base-1.0 weights at ${MLPERF_MODEL_DIR}/sdxl",
            "COCO 2014 5k evaluation captions at ${MLPERF_DATA_DIR}/coco-sdxl",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/text_to_image/run_local.sh "
            "pytorch gpu "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/coco-sdxl "
            "--model-path ${MLPERF_MODEL_DIR}/sdxl "
            "--output ${MLPERF_OUTPUT_DIR}/sdxl "
            "--count 5000 "
            "--num-inf-steps 20",
            "python3 ${MLPERF_INFERENCE_DIR}/text_to_image/tools/accuracy-clip.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/sdxl/mlperf_log_accuracy.json "
            "--dataset-dir ${MLPERF_DATA_DIR}/coco-sdxl",
        ],
        expected=(
            "Offline throughput >15 images/sec on 4×H100 (20 DDIM steps); "
            "FID <=23.01; CLIP score >=31.68 (MLPerf v4.1 targets)"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_013",
        title="RGAT – graph node classification",
        category="ML Perf",
        sub_category="Vision / Speech / Recs / Graph",
        description="RGAT (graph node classification)",
        kpi_keys=["samples_per_sec", "p99_latency_ms", "accuracy", "result_valid"],
        prerequisites=[
            "MLCommons inference repo cloned at ${MLPERF_INFERENCE_DIR}",
            "IGB-Het dataset at ${MLPERF_DATA_DIR}/igb-het",
            "RGAT checkpoint at ${MLPERF_MODEL_DIR}/rgat",
            "PyTorch Geometric (PyG) installed",
        ],
        commands=[
            "python3 ${MLPERF_INFERENCE_DIR}/graph/R-GAT/run_local.sh "
            "pytorch gpu "
            "--scenario ${MLPERF_LOADGEN_SCENARIO} "
            "--dataset-path ${MLPERF_DATA_DIR}/igb-het "
            "--model-path ${MLPERF_MODEL_DIR}/rgat "
            "--output ${MLPERF_OUTPUT_DIR}/rgat "
            "--count 9035",
            "python3 ${MLPERF_INFERENCE_DIR}/graph/R-GAT/accuracy.py "
            "--mlperf-accuracy-file ${MLPERF_OUTPUT_DIR}/rgat/mlperf_log_accuracy.json",
        ],
        expected=(
            "Offline throughput >4000 samples/sec on 4×H100; "
            "accuracy >=0.7286 (MLPerf v4.1 target)"
        ),
    ),

    # =========================================================================
    # Subcategory: MLPerf Training (Datacenter) model list  (IDs 014–021)
    # =========================================================================

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_014",
        title="GPT-J 6B – summarization training",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="GPT-J (summarization; listed in suite)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss", "target_rouge1"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "EleutherAI/gpt-j-6b base checkpoint downloaded",
            "CNN/DailyMail dataset at ${MLPERF_DATA_DIR}/cnn_dailymail",
        ],
        commands=[
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/language_model/gpt-j/run_training.py "
            "--model-name EleutherAI/gpt-j-6b "
            "--dataset-path ${MLPERF_DATA_DIR}/cnn_dailymail "
            "--output-dir ${MLPERF_OUTPUT_DIR}/gpt-j "
            "--target-rouge1 42.9245 "
            "--per-device-batch-size 8 "
            "--gradient-accumulation-steps 4 "
            "--fp16 "
            "--log-interval 100",
        ],
        expected=(
            "Train to ROUGE-1 >=42.9245 in <120 min on 4×H100; "
            "throughput >500 samples/sec"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_015",
        title="BERT-Large – SQuAD-style language processing",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="BERT-Large (SQuAD-style language processing; listed in suite)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss", "target_f1"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "SQuAD v1.1 dataset at ${MLPERF_DATA_DIR}/squad",
            "bert-large-uncased pre-trained checkpoint",
        ],
        commands=[
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/language_model/pytorch/bert/run_squad.py "
            "--bert_model bert-large-uncased "
            "--do_train "
            "--train_file ${MLPERF_DATA_DIR}/squad/train-v1.1.json "
            "--predict_file ${MLPERF_DATA_DIR}/squad/dev-v1.1.json "
            "--train_batch_size 48 "
            "--learning_rate 3e-5 "
            "--num_train_epochs 2.0 "
            "--max_seq_length 384 "
            "--doc_stride 128 "
            "--output_dir ${MLPERF_OUTPUT_DIR}/bert-squad "
            "--fp16",
        ],
        expected=(
            "F1 >=90.874 on SQuAD v1.1 dev set; "
            "throughput >2000 samples/sec on 4×H100"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_016",
        title="3D U-Net – medical image segmentation training",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="3D U-Net (segmentation)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss", "target_dice"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "KiTS19 dataset at ${MLPERF_DATA_DIR}/kits19 (nnU-Net pre-processed)",
        ],
        commands=[
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/image_segmentation/pytorch/train.py "
            "--data-dir ${MLPERF_DATA_DIR}/kits19 "
            "--epochs 4000 "
            "--batch-size 2 "
            "--optimizer adam "
            "--learning-rate 0.0002 "
            "--amp "
            "--log-every 50",
        ],
        expected=(
            "Mean Dice >=0.908 on KiTS19 validation; "
            "typical wall-time <8 h on 4×H100 with AMP"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_017",
        title="Stable Diffusion v2 – image generation training",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="Stable Diffusion v2 (image generation)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss", "target_fid"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "LAION-400M webdataset subset at ${MLPERF_DATA_DIR}/laion400m",
            "SD v2.1 base checkpoint at ${MLPERF_MODEL_DIR}/sd/v2-1_512-ema-pruned.ckpt",
        ],
        commands=[
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/stable_diffusion/main.py "
            "--base ${MLPERF_TRAINING_DIR}/stable_diffusion/configs/train_01x04x08.yaml "
            "--actual_resume ${MLPERF_MODEL_DIR}/sd/v2-1_512-ema-pruned.ckpt "
            "--data_dir ${MLPERF_DATA_DIR}/laion400m "
            "--log-every-n-steps 100",
        ],
        expected=(
            "FID <=23.0 on COCO 2014 5k subset after fine-tuning; "
            "throughput >80 images/sec on 4×H100"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_018",
        title="BERT-Large NLP – pre-training phase 1 + 2",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="BERT-Large (NLP)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "Wikipedia + BooksCorpus pre-tokenised HDF5 at ${MLPERF_DATA_DIR}/bert/",
            "Phase-1 (seq 128) and phase-2 (seq 512) HDF5 shards prepared",
        ],
        commands=[
            # Phase 1: seq-len 128
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/language_model/pytorch/bert/run_pretraining.py "
            "--bert_model bert-large-uncased "
            "--do_train "
            "--train_file ${MLPERF_DATA_DIR}/bert/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus "
            "--output_dir ${MLPERF_OUTPUT_DIR}/bert-phase1 "
            "--max_seq_length 128 "
            "--train_batch_size 512 "
            "--max_steps 7038 "
            "--learning_rate 6e-3 "
            "--warmup_proportion 0.2843 "
            "--fp16",
            # Phase 2: seq-len 512
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/language_model/pytorch/bert/run_pretraining.py "
            "--bert_model bert-large-uncased "
            "--do_train "
            "--train_file ${MLPERF_DATA_DIR}/bert/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus "
            "--output_dir ${MLPERF_OUTPUT_DIR}/bert-phase2 "
            "--max_seq_length 512 "
            "--train_batch_size 128 "
            "--phase1_end_step 7038 "
            "--max_steps 107538 "
            "--learning_rate 4e-3 "
            "--fp16",
        ],
        expected=(
            "Masked-LM accuracy >=0.720 after phase-2; "
            "throughput >2000 seq/sec (phase-1) on 4×H100"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_019",
        title="GPT-3 175B – LLM pre-training",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="GPT-3 (175B) (LLM training)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "Megatron-LM installed and built",
            "C4 dataset (text documents) at ${MLPERF_DATA_DIR}/c4",
            "GPT-2 vocabulary + merge files at ${MLPERF_DATA_DIR}/",
            "4×H100 SXM5 minimum (full 175B run is multi-node; single-node exercises tensor-parallel path)",
        ],
        commands=[
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/large_language_model/megatron-lm/pretrain_gpt.py "
            "--tensor-model-parallel-size ${ML_TENSOR_PARALLEL} "
            "--pipeline-model-parallel-size 1 "
            "--num-layers 96 "
            "--hidden-size 12288 "
            "--ffn-hidden-size 49152 "
            "--num-attention-heads 96 "
            "--seq-length 2048 "
            "--global-batch-size 1536 "
            "--train-iters 450 "
            "--lr-decay-iters 430 "
            "--data-path ${MLPERF_DATA_DIR}/c4/c4_text_document "
            "--vocab-file ${MLPERF_DATA_DIR}/gpt2-vocab.json "
            "--merge-file ${MLPERF_DATA_DIR}/gpt2-merges.txt "
            "--split 949,50,1 "
            "--distributed-backend nccl "
            "--lr 6.0e-5 "
            "--min-lr 6.0e-6 "
            "--lr-decay-style cosine "
            "--weight-decay 0.1 "
            "--clip-grad 1.0 "
            "--bf16 "
            "--log-interval 10",
        ],
        expected=(
            "Throughput >200 tokens/GPU/sec on H100 SXM5 BF16 (TP4); "
            "note: full 175B convergence requires multi-node"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_020",
        title="Llama 2 70B LoRA – LLM fine-tuning",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="Llama 2 70B LoRA (LLM fine-tuning)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss", "target_rouge2"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "meta-llama/Llama-2-70b-hf base model downloaded",
            "SCROLLS dataset at ${MLPERF_DATA_DIR}/scrolls",
            "peft and bitsandbytes Python packages installed",
        ],
        commands=[
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/llm_finetune/run_lora_finetune.py "
            "--model_name_or_path meta-llama/Llama-2-70b-hf "
            "--dataset_name scrolls "
            "--dataset_config govreport "
            "--lora_r 16 "
            "--lora_alpha 32 "
            "--lora_dropout 0.05 "
            "--num_train_epochs 1 "
            "--per_device_train_batch_size 2 "
            "--gradient_accumulation_steps 4 "
            "--learning_rate 4e-4 "
            "--bf16 "
            "--output_dir ${MLPERF_OUTPUT_DIR}/llama2-lora "
            "--logging_steps 50",
        ],
        expected=(
            "ROUGE-2 >=15.1 target; "
            "throughput >30 samples/sec on 4×H100 BF16 with LoRA"
        ),
    ),

    MLPerfCase(
        case_id="AI_HOSTCPU_ML_TEST_ID_021",
        title="DLRMv2 – recommendation system training",
        category="ML Perf",
        sub_category="MLPerf Training (Datacenter) model list",
        description="DLRMv2 (recommendation)",
        kpi_keys=["samples_per_sec", "time_to_train_min", "final_loss", "target_auc"],
        prerequisites=[
            "MLCommons training repo cloned at ${MLPERF_TRAINING_DIR}",
            "Criteo Terabyte dataset (~1 TB) at ${MLPERF_DATA_DIR}/criteo",
            "Criteo terabyte_processed.npz pre-processed file",
            "FBGeMM or torch.nn.EmbeddingBag (CUDA) available",
        ],
        commands=[
            "torchrun --nproc_per_node=${ML_NUM_GPUS} "
            "${MLPERF_TRAINING_DIR}/recommendation/dlrm_v2/pytorch/dlrm_s_pytorch.py "
            "--arch-sparse-feature-size=128 "
            "--arch-mlp-bot=13-512-256-128 "
            "--arch-mlp-top=1024-1024-512-256-1 "
            "--data-generation=dataset "
            "--data-set=terabyte "
            "--raw-data-file=${MLPERF_DATA_DIR}/criteo/day_0 "
            "--processed-data-file=${MLPERF_DATA_DIR}/criteo/terabyte_processed.npz "
            "--loss-function=bce "
            "--round-targets=True "
            "--learning-rate=1.0 "
            "--mini-batch-size=2048 "
            "--print-freq=2048 "
            "--print-time "
            "--test-mini-batch-size=16384 "
            "--test-num-workers=16 "
            "--memory-map "
            "--mlperf-logging "
            "--mlperf-auc-threshold=0.8025 "
            "--mlperf-bin-loader "
            "--mlperf-bin-shuffle "
            "--num-batches=5000",
        ],
        expected=(
            "AUC >=0.8025 on Criteo Day 23 holdout; "
            "throughput >2 M samples/sec on 4×H100"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.node_validation
@pytest.mark.workload_acceptance
@pytest.mark.slow
@pytest.mark.mlperf
@pytest.mark.parametrize("case", MLPERF_CASES, ids=[c.case_id for c in MLPERF_CASES])
def test_mlperf_case_suite(case: MLPerfCase) -> None:
    """MLPerf catalog + optional execution harness for 4×H100 SXM5 validation.

    Dry-run (default): validates metadata shape and writes a catalog KPI entry.
    Execution mode (ML_EXECUTE_CASES=1): runs benchmark commands, extracts KPIs,
    and fails the test if any command exits non-zero.
    """
    if not _is_selected(case.case_id):
        pytest.skip("Filtered by ML_CASE_IDS")

    # ------------------------------------------------------------------
    # Metadata validation (always runs, even in dry-run mode)
    # ------------------------------------------------------------------
    assert case.case_id.startswith("AI_HOSTCPU_ML_TEST_ID_"), (
        f"case_id must follow AI_HOSTCPU_ML_TEST_ID_NNN format: {case.case_id}"
    )
    assert case.title.strip(), f"{case.case_id}: title must not be empty"
    assert case.category == "ML Perf", f"{case.case_id}: category must be 'ML Perf'"
    assert case.sub_category.strip(), f"{case.case_id}: sub_category must not be empty"
    assert case.description.strip(), f"{case.case_id}: description must not be empty"
    assert case.kpi_keys, f"{case.case_id}: kpi_keys must declare at least one expected metric"

    _record_kpi(
        case.case_id,
        {
            "mode": "catalog",
            "title": case.title,
            "category": case.category,
            "sub_category": case.sub_category,
            "description": case.description,
            "kpi_keys": case.kpi_keys,
            "expected": case.expected,
            "prerequisites": case.prerequisites,
            "commands": case.commands,
        },
    )

    if case.planned_only:
        pytest.skip(f"{case.case_id}: planned placeholder – not yet implemented")

    execute = _env_flag("ML_EXECUTE_CASES", "0")
    if not execute:
        pytest.skip(
            f"{case.case_id}: catalog validated. Set ML_EXECUTE_CASES=1 to run benchmark commands"
        )

    # ------------------------------------------------------------------
    # Execution mode: run commands and collect results
    # ------------------------------------------------------------------
    timeout_sec = int(os.environ.get("ML_CMD_TIMEOUT_SEC", "3600"))
    command_results: List[Dict] = []

    for command in case.commands:
        result = _run_command(command, timeout_sec=timeout_sec)
        command_results.append(result)

        if result["returncode"] not in {0, 124}:
            _record_kpi(
                case.case_id,
                {
                    "mode": "execution",
                    "status": "failed",
                    "results": command_results,
                },
            )
            pytest.fail(
                f"{case.case_id}: command failed (rc={result['returncode']})\n"
                f"executed_command: {result['executed_command']}\n"
                f"stderr_tail:\n{result['stderr_tail']}\n"
                f"stdout_tail:\n{result['stdout_tail']}"
            )

    # ------------------------------------------------------------------
    # KPI extraction
    # ------------------------------------------------------------------
    merged_stdout = "\n".join(r["stdout_tail"] for r in command_results)
    merged_stderr = "\n".join(r["stderr_tail"] for r in command_results)

    extractor = _select_kpi_extractor(case.sub_category)
    extracted: Dict[str, float] = extractor(merged_stdout, merged_stderr)

    total_elapsed = sum(r["elapsed_sec"] for r in command_results)

    aggregated: Dict = {
        "mode": "execution",
        "status": "passed",
        "elapsed_sec": round(total_elapsed, 3),
        **extracted,
    }

    # Warn (but do not fail) when expected KPI keys were absent in captured output
    missing_kpis = [k for k in case.kpi_keys if k not in extracted]
    if missing_kpis:
        logger.warning(
            "%s: expected KPI keys not found in command output: %s",
            case.case_id,
            missing_kpis,
        )
        aggregated["missing_kpi_keys"] = missing_kpis

    _record_kpi(case.case_id, aggregated)
