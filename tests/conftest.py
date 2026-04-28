"""
Shared pytest fixtures and configuration for the AI Head Node test suite.

Hardware target:
  160-core CPU  (2 sockets × 80 cores, SMT-2)
  4× NVIDIA H100 SXM5 (80 GB HBM3 each, NVLink 4 full-mesh)
"""

import os
import sys
import json
import re
import signal
import platform
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("ai_headnode_tests")

_REPORT_PATH = Path(__file__).resolve().parents[1] / "TESTCASE_EXECUTION_FINAL_REPORT.md"
_REPORT_ALIAS_PATHS = [
    Path(__file__).resolve().parents[1] / "NODE_VALIDATION_REPORT.md",
]
_VALIDATION_LOG_PATH = Path(__file__).resolve().parents[1] / "validation_run.log"
_FINALIZED_MAP_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "reports" / "finalized_test_to_testcaseid_map.csv"

_VLLM_PROCESS_PATTERNS = (
    "enginecore",
    "vllm.benchmarks.benchmark_serving",
    "vllm.entrypoints.openai.api_server",
)


def _vllm_cleanup_enabled() -> bool:
    return os.environ.get("VLLM_STRICT_CLEANUP", "1") == "1"


def _is_vllm_related_item(item: pytest.Item) -> bool:
    marker_names = {marker.name for marker in item.iter_markers()}
    file_name = Path(str(item.fspath)).name.lower()
    nodeid_lower = item.nodeid.lower()

    if "vllm_prefill" in marker_names:
        return True
    if file_name in {
        "test_vllm_prefill_validation.py",
        "test_performance_suite.py",
        "test_mlperf_suite.py",
        "test_node_validation.py",
    }:
        return True
    return "vllm" in nodeid_lower or "benchmark_serving" in nodeid_lower


def _find_stale_vllm_processes() -> List[Tuple[int, str]]:
    if not sys.platform.startswith("linux"):
        return []

    current_pid = os.getpid()
    stale: List[Tuple[int, str]] = []

    if _psutil is not None:
        for proc in _psutil.process_iter(["pid", "cmdline"]):
            try:
                pid = int(proc.info.get("pid", -1))
                if pid <= 0 or pid == current_pid:
                    continue
                cmdline = proc.info.get("cmdline") or []
                cmd = " ".join(cmdline).strip()
                cmd_lower = cmd.lower()
                if any(pattern in cmd_lower for pattern in _VLLM_PROCESS_PATTERNS):
                    stale.append((pid, cmd or "<unknown>"))
            except Exception:
                continue
        return stale

    # Fallback path if psutil is unavailable.
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            pid = int(parts[0])
            if pid == current_pid:
                continue
            cmd = parts[1] if len(parts) > 1 else ""
            cmd_lower = cmd.lower()
            if any(pattern in cmd_lower for pattern in _VLLM_PROCESS_PATTERNS):
                stale.append((pid, cmd or "<unknown>"))
    except Exception:
        return []

    return stale


def _terminate_stale_vllm_processes(tag: str, nodeid: str) -> None:
    if not _vllm_cleanup_enabled():
        return

    stale = _find_stale_vllm_processes()
    if not stale:
        return

    logger.warning(
        "[%s] Found %d stale vLLM process(es) before cleanup for %s",
        tag,
        len(stale),
        nodeid,
    )

    for pid, cmd in stale:
        try:
            logger.warning("[%s] Terminating stale vLLM PID=%s cmd=%s", tag, pid, cmd[:220])
            if _psutil is not None:
                proc = _psutil.Process(pid)
                proc.terminate()
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            continue

    deadline = time.time() + 5.0
    while time.time() < deadline:
        remaining = _find_stale_vllm_processes()
        if not remaining:
            return
        time.sleep(0.2)

    # Escalate to hard kill when process did not terminate gracefully.
    for pid, _cmd in _find_stale_vllm_processes():
        try:
            if _psutil is not None:
                _psutil.Process(pid).kill()
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            continue


def _release_torch_cuda_memory() -> None:
    if _torch is None:
        return
    try:
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
            if hasattr(_torch.cuda, "ipc_collect"):
                _torch.cuda.ipc_collect()
    except Exception:
        pass


def _detect_nvidia_gpu_count() -> int:
    """Best-effort NVIDIA GPU count without importing torch."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        rows = [line.strip() for line in out.splitlines() if line.strip()]
        return len(rows)
    except Exception:
        return 0


def _sanitize_cuda_visible_devices(value: str, gpu_count: int) -> str:
    """Keep only valid integer indices in range [0, gpu_count)."""
    if gpu_count <= 0:
        return value
    valid: List[str] = []
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        if token.isdigit() and int(token) < gpu_count:
            valid.append(token)
    if not valid:
        return ",".join(str(i) for i in range(gpu_count))
    return ",".join(valid)


def _recommended_cuda_visible_devices() -> str:
    gpu_count = _detect_nvidia_gpu_count()
    if gpu_count <= 0:
        return "0"
    return ",".join(str(i) for i in range(gpu_count))


# ---------------------------------------------------------------------------
# Environment Variable Defaults (pytest_configure hook)
# ---------------------------------------------------------------------------
def _configure_default_env() -> None:
    """Set environment variable defaults for all workload tests."""
    
    # =========================================================================
    # HuggingFace & Large Model Configuration
    # =========================================================================
    os.environ.setdefault("HF_ENABLE_LARGE_MODELS", "1")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD", "1")
    
    # Model IDs
    os.environ.setdefault("HF_LLAMA_7B_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    os.environ.setdefault("HF_LLAMA_70B_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
    os.environ.setdefault("HF_QWEN_70B_MODEL", "Qwen/Qwen-72B")
    os.environ.setdefault("HF_DEEPSEEK_R1_QUANT_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    os.environ.setdefault("HF_INFERENCE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    os.environ.setdefault("HF_TRAINING_MODEL", "bert-base-uncased")
    os.environ.setdefault("HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC", "300")
    
    # =========================================================================
    # KV Cache Offload Configuration
    # =========================================================================
    # Use NODE_VLLM_* names so vLLM does not treat these as its own env vars.
    # Migrate legacy VLLM_* names for backward compatibility, then remove them
    # from the environment to avoid "Unknown vLLM environment variable" warnings.
    kv_env_map = {
        "VLLM_KV_OFFLOAD_MODEL": "NODE_VLLM_KV_OFFLOAD_MODEL",
        "VLLM_KV_CONTEXT_LENGTH": "NODE_VLLM_KV_CONTEXT_LENGTH",
        "VLLM_KV_BATCH_SIZE": "NODE_VLLM_KV_BATCH_SIZE",
        "VLLM_KV_NUM_PROMPTS": "NODE_VLLM_KV_NUM_PROMPTS",
        "VLLM_KV_MAX_TOKENS": "NODE_VLLM_KV_MAX_TOKENS",
        "VLLM_BASELINE_GPU_MEMORY_UTIL": "NODE_VLLM_BASELINE_GPU_MEMORY_UTIL",
        "VLLM_GPU_MEMORY_UTIL": "NODE_VLLM_GPU_MEMORY_UTIL",
        "VLLM_CPU_OFFLOAD_GB": "NODE_VLLM_CPU_OFFLOAD_GB",
    }
    for legacy_name, new_name in kv_env_map.items():
        legacy_value = os.environ.get(legacy_name)
        if legacy_value and new_name not in os.environ:
            os.environ[new_name] = legacy_value
        os.environ.pop(legacy_name, None)

    os.environ.setdefault("NODE_VLLM_KV_OFFLOAD_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    os.environ.setdefault("NODE_VLLM_KV_CONTEXT_LENGTH", "8192")
    os.environ.setdefault("NODE_VLLM_KV_BATCH_SIZE", "4")
    os.environ.setdefault("NODE_VLLM_KV_NUM_PROMPTS", "10")
    os.environ.setdefault("NODE_VLLM_KV_MAX_TOKENS", "512")
    os.environ.setdefault("NODE_VLLM_BASELINE_GPU_MEMORY_UTIL", "0.9")
    os.environ.setdefault("NODE_VLLM_GPU_MEMORY_UTIL", "0.7")
    os.environ.setdefault("NODE_VLLM_CPU_OFFLOAD_GB", "128")

    # =========================================================================
    # RDMA / MMIO / DMA Validation Configuration
    # =========================================================================
    os.environ.setdefault("RDMA_SMOKE_ITERS", "1000")
    os.environ.setdefault("RDMA_USE_CUDA_DEVICE", "0")
    os.environ.setdefault("RDMA_4KB_MAX_AVG_LAT_US", "300.0")
    os.environ.setdefault("RDMA_SOAK_LOOPS", "1000")
    os.environ.setdefault("SOAK_MAX_P95_US", "20000")
    
    # =========================================================================
    # Vision Workload Configuration (YOLO)
    # =========================================================================
    os.environ.setdefault("YOLO_DETECT_MODEL", "yolov8x.pt")
    os.environ.setdefault("YOLO_DETECT_BATCH", "128")
    os.environ.setdefault("YOLO_DETECT_IMGSZ", "640")
    os.environ.setdefault("YOLO_DETECT_SOURCE", "https://ultralytics.com/images/bus.jpg")
    os.environ.setdefault("YOLO_DETECT_TIMEOUT_SEC", "240")
    os.environ.setdefault("YOLO_IMAGES_PER_GPU", "256")
    os.environ.setdefault("YOLO_HALF_PRECISION", "1")
    os.environ.setdefault("YOLO_GPU_COUNT", "4")
    os.environ.setdefault("YOLO_CPU_SAMPLE_INTERVAL_SEC", "0.5")
    os.environ.setdefault("YOLO_AGGREGATE_FPS_TARGET", "200")
    os.environ.setdefault("YOLO_CPU_UTIL_TARGET", "50")
    
    # =========================================================================
    # Vision Workload Configuration (Diffusers)
    # =========================================================================
    os.environ.setdefault("DIFFUSERS_MODEL_ID", "stabilityai/sd-turbo")
    os.environ.setdefault("DIFFUSERS_NUM_IMAGES", "32")
    os.environ.setdefault("DIFFUSERS_IMAGES_PER_GPU", os.environ.get("DIFFUSERS_NUM_IMAGES", "32"))
    os.environ.setdefault("DIFFUSERS_HEIGHT", "512")
    os.environ.setdefault("DIFFUSERS_WIDTH", "512")
    os.environ.setdefault("DIFFUSERS_STEPS", "1")
    os.environ.setdefault("DIFFUSERS_GUIDANCE_SCALE", "0.0")
    os.environ.setdefault("DIFFUSERS_WARMUP", "2")
    os.environ.setdefault("DIFFUSERS_GPU_COUNT", "4")
    os.environ.setdefault("DIFFUSERS_MASTER_PORT", "29501")
    os.environ.setdefault("DIFFUSERS_GENERATE_TIMEOUT_SEC", "600")
    os.environ.setdefault("DIFFUSERS_TARGET_IMG_PER_SEC", "10.0")
    
    # =========================================================================
    # PyTorch & CUDA Configuration
    # =========================================================================
    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
    gpu_count = _detect_nvidia_gpu_count()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = _sanitize_cuda_visible_devices(
            os.environ["CUDA_VISIBLE_DEVICES"], gpu_count
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _recommended_cuda_visible_devices()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    
    # =========================================================================
    # KPI & Test Configuration
    # =========================================================================
    os.environ.setdefault("NODE_VALIDATION_KPI_FILE", "artifacts/node_validation_kpis.jsonl")
    os.environ.setdefault("AGENTIC_KPI_FILE", "artifacts/agentic_ai_kpis.jsonl")
    os.environ.setdefault("PIPELINE_KPI_FILE", "artifacts/ai_pipeline_kpis.jsonl")
    os.environ.setdefault("HEALTH_KPI_FILE", "artifacts/system_health_kpis.jsonl")
    os.environ.setdefault("ML_SUITE_KPI_FILE", "artifacts/mlperf_suite_kpis.jsonl")
    os.environ.setdefault("NVBANDWIDTH_BIN", "/root/workloads/nvbandwidth")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTEST_TIMEOUT", "600")
    os.environ.setdefault("NODE_WORKLOAD_AUTO_INSTALL", "1" if sys.platform.startswith("linux") else "0")
    os.environ.setdefault("NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC", "1800")

    # =========================================================================
    # Performance Suite Configuration
    # =========================================================================
    os.environ.setdefault("PERF_CMD_TIMEOUT_SEC", "1800")
    os.environ.setdefault("PERF_SUITE_KPI_FILE", "artifacts/performance_suite_kpis.jsonl")
    os.environ.setdefault("VLLM_PREFILL_KPI_FILE", "artifacts/vllm_prefill_kpis.jsonl")
    os.environ.setdefault("PERF_NVBANDWIDTH_BIN", os.environ.get("NVBANDWIDTH_BIN", "/root/workloads/nvbandwidth"))
    os.environ.setdefault("PERF_NCCL_TESTS_DIR", "/root/workloads/nccl-tests/build")
    os.environ.setdefault("PERF_VLLM_GPU_MEMORY_UTILIZATION", "0.70")
    os.environ.setdefault("PERF_VLLM_FP8_DTYPE_FALLBACK", "bfloat16")
    os.environ.setdefault("PERF_VLLM_BENCHMARK_MODULE", "vllm.benchmarks.benchmark_serving")
    os.environ.setdefault("PERF_AUTO_INSTALL", "1" if sys.platform.startswith("linux") else "0")
    os.environ.setdefault("PERF_AUTO_INSTALL_TIMEOUT_SEC", "1800")

    # Enable execution-mode suites by default so test runs do not require
    # manual EXECUTE flag exports unless explicitly opting into catalog mode.
    os.environ.setdefault("ML_EXECUTE_CASES", "1")
    os.environ.setdefault("PERF_EXECUTE_CASES", "1")
    os.environ.setdefault("PIPELINE_EXECUTE", "1")
    os.environ.setdefault("AGENTIC_EXECUTE", "1")
    
    logger.info(
        "Environment variables configured:\n"
        "  HF_ENABLE_LARGE_MODELS=%s\n"
        "  NODE_VLLM_KV_CONTEXT_LENGTH=%s\n"
        "  YOLO_DETECT_BATCH=%s\n"
        "  CUDA_VISIBLE_DEVICES=%s\n"
        "  NVBANDWIDTH_BIN=%s\n"
        "  NODE_WORKLOAD_AUTO_INSTALL=%s\n"
        "  NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=%s\n"
        "  PERF_CMD_TIMEOUT_SEC=%s\n"
        "  PERF_AUTO_INSTALL=%s\n"
        "  PERF_NCCL_TESTS_DIR=%s\n"
        "  PERF_VLLM_GPU_MEMORY_UTILIZATION=%s\n"
        "  ML_EXECUTE_CASES=%s\n"
        "  PERF_EXECUTE_CASES=%s\n"
        "  PIPELINE_EXECUTE=%s\n"
        "  AGENTIC_EXECUTE=%s",
        os.environ.get("HF_ENABLE_LARGE_MODELS"),
        os.environ.get("NODE_VLLM_KV_CONTEXT_LENGTH"),
        os.environ.get("YOLO_DETECT_BATCH"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("NVBANDWIDTH_BIN"),
        os.environ.get("NODE_WORKLOAD_AUTO_INSTALL"),
        os.environ.get("NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC"),
        os.environ.get("PERF_CMD_TIMEOUT_SEC"),
        os.environ.get("PERF_AUTO_INSTALL"),
        os.environ.get("PERF_NCCL_TESTS_DIR"),
        os.environ.get("PERF_VLLM_GPU_MEMORY_UTILIZATION"),
        os.environ.get("ML_EXECUTE_CASES"),
        os.environ.get("PERF_EXECUTE_CASES"),
        os.environ.get("PIPELINE_EXECUTE"),
        os.environ.get("AGENTIC_EXECUTE"),
    )


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _color_badge(text: str, color: str) -> str:
    """Return colored HTML badge for markdown reports."""
    safe = _md_escape(text)
    return f"<span style=\"color:{color};font-weight:700\">{safe}</span>"


def _purpose_from_item(item: pytest.Item) -> str:
    doc = getattr(item.obj, "__doc__", None)
    if not doc:
        return item.name
    # Keep report purpose concise by using only the first docstring paragraph.
    paragraph = doc.strip().split("\n\n", 1)[0]
    return " ".join(paragraph.splitlines()).strip()


def _report_section_for_item(item: pytest.Item) -> str:
    marker_names = {marker.name for marker in item.iter_markers()}
    nodeid_lower = item.nodeid.lower()

    if "mlperf" in marker_names or "mlperf" in nodeid_lower:
        return "MLPerf"
    if "agentic_ai" in marker_names or Path(str(item.fspath)).name == "test_agentic_ai_validation.py":
        return "Agentic AI Validation"
    if Path(str(item.fspath)).name == "test_performance_suite.py":
        return "Performance"
    if "vllm_prefill" in marker_names or Path(str(item.fspath)).name == "test_vllm_prefill_validation.py":
        return "vLLM Prefill"
    if "workload_acceptance" in marker_names:
        return "Workload Acceptance"
    if "bottleneck" in marker_names:
        return "Bottleneck"
    if "node_validation" in marker_names:
        return "Node Validation"
    if (
        "ai_pipeline" in marker_names
        or "data_loading" in marker_names
        or "model_orch" in marker_names
        or "storage_net_io" in marker_names
        or "inference_kpis" in marker_names
        or Path(str(item.fspath)).name == "test_ai_pipeline_kpis.py"
    ):
        return "AI Pipeline KPIs"
    if (
        "system_health" in marker_names
        or "thermal_stability" in marker_names
        or "ecc_link_health" in marker_names
        or "energy_efficiency" in marker_names
        or "quant_correctness" in marker_names
        or "ddp_correctness" in marker_names
        or "compile_correctness" in marker_names
        or "stream_correctness" in marker_names
        or Path(str(item.fspath)).name == "test_system_health_and_reliability.py"
    ):
        return "System Health & Reliability"
    return "Other"


def _report_domain_for_item(item: pytest.Item) -> str:
    # Keep reporting simple: all tests are grouped under a single functional domain.
    return "Functional Validation"


def _display_test_name(nodeid: str) -> str:
    """Return a concise testcase label (method name, preserving param suffix)."""
    label = nodeid.strip()
    if "::" in label:
        label = label.split("::")[-1]
    return label


def _status_result_analysis(report: pytest.TestReport) -> Tuple[str, str, str]:
    if report.outcome == "passed":
        return "PASS", f"PASS ({report.duration:.3f}s)", "Executed successfully; acceptance criteria met."

    if report.outcome == "skipped":
        reason = "Skipped"
        if hasattr(report.longrepr, "reprcrash") and report.longrepr.reprcrash:
            reason = report.longrepr.reprcrash.message
        else:
            lr = str(report.longrepr)
            if "Skipped:" in lr:
                reason = lr.split("Skipped:", 1)[-1].strip()
        return "SKIP", f"SKIPPED: {reason}", "Skipped by design: environment, tools, or prerequisites unavailable."

    # Failed
    fail_msg = "Assertion or runtime failure"
    if hasattr(report.longrepr, "reprcrash") and report.longrepr.reprcrash:
        fail_msg = report.longrepr.reprcrash.message
    return "FAIL", f"FAILED: {fail_msg}", "Investigate thresholds, platform readiness, and benchmark runtime conditions."


def _kpi_output_path() -> Path:
    return Path(os.environ.get("NODE_VALIDATION_KPI_FILE", "artifacts/node_validation_kpis.jsonl"))


def _normalize_test_key(value: str) -> str:
    key = value.strip()
    if "::" in key:
        key = key.split("::")[-1]
    if "[" in key:
        key = key.split("[", 1)[0]
    return key


def _parameterized_test_key(value: str) -> Optional[str]:
    raw = value.strip()
    if "[" not in raw or "]" not in raw:
        return None
    return raw.split("[", 1)[1].rsplit("]", 1)[0].strip() or None


def _kpi_paths() -> List[Path]:
    paths = [
        Path(os.environ.get("NODE_VALIDATION_KPI_FILE", "artifacts/node_validation_kpis.jsonl")),
        Path(os.environ.get("AGENTIC_KPI_FILE", "artifacts/agentic_ai_kpis.jsonl")),
        Path(os.environ.get("PERF_SUITE_KPI_FILE", "artifacts/performance_suite_kpis.jsonl")),
        Path(os.environ.get("VLLM_PREFILL_KPI_FILE", "artifacts/vllm_prefill_kpis.jsonl")),
        Path(os.environ.get("PIPELINE_KPI_FILE", "artifacts/ai_pipeline_kpis.jsonl")),
        Path(os.environ.get("HEALTH_KPI_FILE", "artifacts/system_health_kpis.jsonl")),
    ]
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _load_latest_kpis() -> Dict[str, Dict[str, Any]]:
    """Load latest KPI payload per test name or performance case ID from JSONL."""
    out: Dict[str, Dict[str, Any]] = {}
    for kpi_path in _kpi_paths():
        if not kpi_path.exists():
            continue

        try:
            with kpi_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    test_name = str(payload.get("test") or payload.get("case_id") or "").strip()
                    if not test_name:
                        continue
                    metrics = payload.get("metrics", {})
                    if not isinstance(metrics, dict):
                        metrics = {"value": metrics}
                    record = {
                        "timestamp": payload.get("timestamp", ""),
                        "metrics": metrics,
                    }
                    out[test_name] = record
                    out[_normalize_test_key(test_name)] = record
        except Exception as exc:
            logger.warning("Unable to parse KPI JSONL %s: %s", kpi_path, exc)
    return out


def _lookup_kpi_metrics(nodeid: str, latest_kpis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [nodeid, _normalize_test_key(nodeid)]
    param_key = _parameterized_test_key(nodeid)
    if param_key:
        candidates.insert(1, param_key)
    for key in candidates:
        metrics = latest_kpis.get(key, {}).get("metrics", {})
        if metrics:
            return metrics
    return {}


def _extract_reason(result: str) -> str:
    if result.startswith("FAILED:"):
        return result.split("FAILED:", 1)[-1].strip()
    if result.startswith("SKIPPED:"):
        return result.split("SKIPPED:", 1)[-1].strip()
    if result.startswith("PASS"):
        return "Acceptance criteria satisfied"
    return result


def _compact_metrics(metrics: Dict[str, Any], max_items: int = 5) -> str:
    if not metrics:
        return "-"
    items = []
    for i, (k, v) in enumerate(metrics.items()):
        if i >= max_items:
            items.append("...")
            break
        if isinstance(v, float):
            items.append(f"{k}={v:.3f}")
        else:
            items.append(f"{k}={v}")
    return "; ".join(items)


def _pick_top_kpis(latest_kpis: Dict[str, Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Pick a compact set of high-value KPI numbers for quick report scanning."""
    flattened: List[Dict[str, Any]] = []
    for test_key, payload in latest_kpis.items():
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                flattened.append(
                    {"test": test_key, "metric": metric_name, "value": float(value)}
                )

    if not flattened:
        return []

    priorities: List[Tuple[str, bool]] = [
        ("throughput_tokens_per_sec", True),
        ("samples_per_sec", True),
        ("aggregate_fps", True),
        ("bandwidth_gbps", True),
        ("p95_latency_ms", False),
        ("mean_latency_ms", False),
        ("page_faults_per_sec", False),
        ("cpu_mean_util_pct", False),
        ("gpu_mean_util_pct", True),
    ]

    selected: List[Dict[str, Any]] = []
    used = set()

    for key_hint, high_is_good in priorities:
        matches = [x for x in flattened if key_hint in x["metric"] and (x["test"], x["metric"]) not in used]
        if not matches:
            continue
        best = max(matches, key=lambda x: x["value"]) if high_is_good else min(matches, key=lambda x: x["value"])
        selected.append(best)
        used.add((best["test"], best["metric"]))
        if len(selected) >= limit:
            return selected

    for x in sorted(flattened, key=lambda v: abs(v["value"]), reverse=True):
        key = (x["test"], x["metric"])
        if key in used:
            continue
        selected.append(x)
        used.add(key)
        if len(selected) >= limit:
            break

    return selected


def _status_sort_key(status: str) -> int:
    order = {"FAIL": 0, "SKIP": 1, "PASS": 2}
    return order.get(status, 9)


def _render_status(status: str) -> str:
    if status == "PASS":
        return _color_badge("PASS", "#1b8a3b")
    if status == "FAIL":
        return _color_badge("FAIL", "#cc1f1a")
    if status == "SKIP":
        return _color_badge("SKIP", "#d97706")
    return _md_escape(status)


def _render_result_text(result: str) -> str:
    if result.startswith("PASS"):
        return _color_badge(result, "#1b8a3b")
    if result.startswith("FAILED"):
        return _color_badge(result, "#cc1f1a")
    if result.startswith("SKIPPED"):
        return _color_badge(result, "#d97706")
    return _md_escape(result)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)


def _truncate_line(text: str, max_chars: int = 170) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _compact_outcome_lines(status: str, result: str, analysis: str, fail_reason: Optional[str] = None) -> str:
    lines: List[str] = []

    if status == "FAIL":
        lines.append(fail_reason or _extract_reason(result) or "Failure detected")
        lines.append(analysis or "Investigate runtime conditions and thresholds.")
    elif status == "SKIP":
        lines.append(_extract_reason(result) or "Skipped")
        lines.append(analysis or "Skipped by design or missing prerequisites.")
    else:
        lines.append("Executed successfully")
        if analysis:
            lines.append(analysis)

    compact = [f"<div class=\"cell-wrap\">{_md_escape(_truncate_line(line))}</div>" for line in lines[:3] if line]
    return "".join(compact) if compact else "<div class=\"cell-wrap\">-</div>"


def _parse_validation_log(log_path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "kpi_by_test": {},
        "failure_reason_by_test": {},
        "kpi_rows": [],
    }
    if not log_path.exists():
        return payload

    test_line_re = re.compile(r"^(tests/[^\s]+::[^\s]+)")
    kpi_line_re = re.compile(
        r"ai_headnode_tests\s+(?P<metric>.+?)\s+measured=\s*(?P<measured>[-+]?\d+(?:\.\d+)?)\s+"
        r"theoretical=\s*(?P<theoretical>[-+]?\d+(?:\.\d+)?)\s+efficiency=\s*(?P<eff>[-+]?\d+(?:\.\d+)?)%\s+"
        r"threshold=\s*(?P<thresh>[-+]?\d+(?:\.\d+)?)%"
    )
    fail_summary_re = re.compile(r"^FAILED\s+(?P<test>tests/[^\s]+::[^\s]+)\s+-\s+(?P<reason>.+)$")

    current_test: Optional[str] = None
    current_norm: Optional[str] = None
    kpi_rows: List[Dict[str, Any]] = []
    kpi_by_test: Dict[str, str] = {}
    fail_by_test: Dict[str, str] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = _strip_ansi(raw).strip()
            if not line:
                continue

            test_match = test_line_re.match(line)
            if test_match:
                current_test = test_match.group(1)
                current_norm = _normalize_test_key(current_test)
                continue

            kpi_match = kpi_line_re.search(line)
            if kpi_match:
                metric = kpi_match.group("metric").strip()
                measured = float(kpi_match.group("measured"))
                theoretical = float(kpi_match.group("theoretical"))
                efficiency = float(kpi_match.group("eff"))
                threshold = float(kpi_match.group("thresh"))

                summary = (
                    f"{metric}: measured={measured:.3f}, theoretical={theoretical:.3f}, "
                    f"eff={efficiency:.2f}%, threshold={threshold:.2f}%"
                )

                if current_test:
                    kpi_by_test[current_test] = summary
                if current_norm:
                    kpi_by_test[current_norm] = summary

                kpi_rows.append(
                    {
                        "test": current_test or "-",
                        "metric": metric,
                        "measured": measured,
                        "theoretical": theoretical,
                        "efficiency": efficiency,
                        "threshold": threshold,
                    }
                )
                continue

            fail_match = fail_summary_re.match(line)
            if fail_match:
                test_id = fail_match.group("test").strip()
                reason = _truncate_line(fail_match.group("reason").strip(), max_chars=220)
                fail_by_test[test_id] = reason
                fail_by_test[_normalize_test_key(test_id)] = reason

    payload["kpi_rows"] = kpi_rows
    payload["kpi_by_test"] = kpi_by_test
    payload["failure_reason_by_test"] = fail_by_test
    return payload


def _lookup_log_kpi(nodeid: str, log_payload: Dict[str, Any]) -> str:
    kpis = log_payload.get("kpi_by_test", {})
    for key in (nodeid, _normalize_test_key(nodeid)):
        if key in kpis:
            return str(kpis[key])
    return "-"


def _preferred_order(keys: List[str], preferred: List[str]) -> List[str]:
    ordered: List[str] = []
    for name in preferred:
        if name in keys:
            ordered.append(name)
    for name in sorted(keys):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _load_testcase_id_map() -> Dict[str, str]:
    """Load selector -> testcase ID mapping from the finalized mapping CSV."""
    if not _FINALIZED_MAP_PATH.exists():
        return {}

    try:
        import csv

        mapping: Dict[str, str] = {}
        with _FINALIZED_MAP_PATH.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                selector = (row.get("TestSelector") or "").strip()
                testcase_id = (row.get("FinalPrimaryTestCaseID") or "").strip()
                if selector and testcase_id:
                    mapping[selector] = testcase_id
        return mapping
    except Exception as exc:
        logger.warning("Failed to load testcase ID map from %s: %s", _FINALIZED_MAP_PATH, exc)
        return {}


def _nodeid_lookup_candidates(nodeid: str) -> List[str]:
    """Return candidate selector keys for mapping lookup, including de-parameterized nodeid."""
    candidates: List[str] = []
    if nodeid:
        candidates.append(nodeid)

    # Convert parametrized nodeids like test_fn[param] -> test_fn for selector matching.
    parts = nodeid.split("::")
    if parts:
        last = parts[-1]
        if "[" in last and last.endswith("]"):
            parts[-1] = last.split("[", 1)[0]
            candidates.append("::".join(parts))

    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for key in candidates:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _testcase_id_for_nodeid(nodeid: str, id_map: Dict[str, str]) -> str:
    for key in _nodeid_lookup_candidates(nodeid):
        if key in id_map:
            return id_map[key]
    return "-"


def _write_final_report(config: pytest.Config) -> None:
    """Write compact, reportable markdown summaries for the current pytest session."""
    rows: Dict[str, Dict[str, str]] = getattr(config, "_overall_report_rows", {})
    if not rows:
        return

    latest_kpis = _load_latest_kpis()
    log_payload = _parse_validation_log(_VALIDATION_LOG_PATH)
    testcase_id_map: Dict[str, str] = getattr(config, "_testcase_id_map", {})

    statuses = [v["status"] for v in rows.values()]
    passed = statuses.count("PASS")
    failed = statuses.count("FAIL")
    skipped = statuses.count("SKIP")
    total = len(statuses)
    executed = passed + failed

    failed_rows = [(nodeid, row) for nodeid, row in sorted(rows.items()) if row.get("status") == "FAIL"]
    skipped_rows = [(nodeid, row) for nodeid, row in sorted(rows.items()) if row.get("status") == "SKIP"]

    preferred_section_order = [
        "Node Validation",
        "Bottleneck",
        "Performance",
        "Workload Acceptance",
        "MLPerf",
        "Agentic AI Validation",
        "AI Pipeline KPIs",
        "System Health & Reliability",
        "Other",
    ]
    preferred_domain_order = ["Functional Validation"]

    section_rows: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    for nodeid, row in rows.items():
        section = row.get("section", "Other")
        section_rows.setdefault(section, []).append((nodeid, row))

    section_order = _preferred_order(list(section_rows.keys()), preferred_section_order)
    section_summaries: List[Tuple[str, int, int, int, int, float]] = []
    for section in section_order:
        entries = section_rows[section]
        section_statuses = [row.get("status", "") for _, row in entries]
        section_pass = section_statuses.count("PASS")
        section_fail = section_statuses.count("FAIL")
        section_skip = section_statuses.count("SKIP")
        section_exec = section_pass + section_fail
        pass_rate = (100.0 * section_pass / section_exec) if section_exec else 0.0
        section_summaries.append(
            (
                section,
                len(entries),
                section_pass,
                section_fail,
                section_skip,
                pass_rate,
            )
        )

    domain_rows: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    for nodeid, row in rows.items():
        domain = row.get("domain", "Other")
        domain_rows.setdefault(domain, []).append((nodeid, row))

    domain_order = _preferred_order(list(domain_rows.keys()), preferred_domain_order)
    domain_summaries: List[Tuple[str, int, int, int, int, float]] = []
    for domain in domain_order:
        entries = domain_rows[domain]
        domain_statuses = [row.get("status", "") for _, row in entries]
        domain_pass = domain_statuses.count("PASS")
        domain_fail = domain_statuses.count("FAIL")
        domain_skip = domain_statuses.count("SKIP")
        domain_exec = domain_pass + domain_fail
        pass_rate = (100.0 * domain_pass / domain_exec) if domain_exec else 0.0
        domain_summaries.append(
            (
                domain,
                len(entries),
                domain_pass,
                domain_fail,
                domain_skip,
                pass_rate,
            )
        )

    highlights = [
        f"- {_color_badge('INFO', '#1f6feb')}: Total testcases captured: {total}",
        f"- {_color_badge('INFO', '#1f6feb')}: Tests executed (pass + fail): {executed}",
        f"- {_color_badge('PASS', '#1b8a3b')}: {passed}",
        f"- {_color_badge('FAIL', '#cc1f1a')}: {failed}",
        f"- {_color_badge('SKIP', '#d97706')}: {skipped}",
    ]
    if failed:
        highlights.append(f"- {_color_badge('FAIL', '#cc1f1a')}: {failed} testcase failures detected in this run.")
    else:
        highlights.append(f"- {_color_badge('PASS', '#1b8a3b')}: No testcase failures detected in this run.")

    report_title = "# AI Head Node Overall Test Execution Report"

    lines = [
        report_title,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Pytest command line: {' '.join(config.invocation_params.args)}",
        f"KPI source files: {', '.join(str(path) for path in _kpi_paths())}",
        "",
        "<style>",
        ".report-scroll { overflow-x: auto; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px; }",
        ".report-table { border-collapse: collapse; width: 100%; min-width: 1100px; table-layout: auto; font-size: 12px; }",
        ".report-table th, .report-table td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }",
        ".report-table th { background: #f6f8fa; position: sticky; top: 0; z-index: 1; }",
        ".cell-wrap { white-space: normal; word-break: break-word; max-width: 420px; margin: 0 0 3px 0; }",
        ".summary-table { border-collapse: collapse; width: 100%; max-width: 720px; }",
        ".summary-table th, .summary-table td { border: 1px solid #d0d7de; padding: 6px 8px; }",
        ".summary-table th { background: #f6f8fa; text-align: left; }",
        "</style>",
        "",
        "## Summary",
        "",
        *highlights,
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Testcases captured | {total} |",
        f"| Tests executed | {executed} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| Skipped | {skipped} |",
        "",
    ]

    lines.extend([
        "## Domain Summary",
        "",
        "| Domain | Testcases | Passed | Failed | Skipped | Pass Rate (Exec) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for domain, domain_total, domain_passed, domain_failed, domain_skipped, domain_rate in domain_summaries:
        lines.append(
            f"| {_md_escape(domain)} | {domain_total} | {domain_passed} | {domain_failed} | {domain_skipped} | {domain_rate:.1f}% |"
        )

    lines.extend([
        "",
        "## Section Summary",
        "",
        "| Section | Testcases | Passed | Failed | Skipped | Pass Rate (Exec) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for section, section_total, section_passed, section_failed, section_skipped, section_rate in section_summaries:
        lines.append(
            f"| {_md_escape(section)} | {section_total} | {section_passed} | {section_failed} | {section_skipped} | {section_rate:.1f}% |"
        )

    lines.extend([
        "",
        "## Failures and Skips",
        "",
    ])

    issue_rows = failed_rows + skipped_rows
    if issue_rows:
        lines.extend([
            "| Domain | Section | TestCaseID | Test | Status | Outcome (2-3 lines) | KPI Snapshot |",
            "|---|---|---|---|---|---|---|",
        ])
        for nodeid, row in issue_rows[:40]:
            log_fail_reason = log_payload.get("failure_reason_by_test", {}).get(nodeid) or log_payload.get("failure_reason_by_test", {}).get(_normalize_test_key(nodeid))
            kpi_metrics = _lookup_kpi_metrics(nodeid, latest_kpis)
            kpi_text = _compact_metrics(kpi_metrics, max_items=4) if kpi_metrics else _lookup_log_kpi(nodeid, log_payload)
            testcase_id = row.get("testcase_id", _testcase_id_for_nodeid(nodeid, testcase_id_map))
            lines.append(
                f"| {_md_escape(row.get('domain', 'Functional Validation'))} | {_md_escape(row.get('section', 'Other'))} | {_md_escape(testcase_id)} | {_md_escape(_display_test_name(nodeid))} | {_render_status(row.get('status', ''))} | {_compact_outcome_lines(row.get('status', ''), row.get('result', ''), row.get('analysis', ''), log_fail_reason)} | <div class=\"cell-wrap\">{_md_escape(_truncate_line(kpi_text))}</div> |"
            )
    else:
        lines.append(f"- {_color_badge('PASS', '#1b8a3b')}: No failures or skips detected.")

    top_kpis = _pick_top_kpis(latest_kpis, limit=6)
    log_kpis = sorted(
        log_payload.get("kpi_rows", []),
        key=lambda x: abs(x.get("efficiency", 0.0) - x.get("threshold", 0.0)),
        reverse=True,
    )[:6]
    lines.extend([
        "",
        "## KPI Highlights",
        "",
    ])
    if top_kpis or log_kpis:
        lines.extend([
            "| Source | Test | KPI | Value |",
            "|---|---|---|---|",
        ])
        for item in top_kpis:
            value_str = f"{item['value']:.3f}"
            lines.append(
                f"| JSONL | {_md_escape(_normalize_test_key(str(item['test'])))} | {_md_escape(item['metric'])} | {_md_escape(value_str)} |"
            )
        for item in log_kpis:
            value_str = f"eff={item.get('efficiency', 0.0):.2f}% (thr={item.get('threshold', 0.0):.2f}%)"
            lines.append(
                f"| validation_run.log | {_md_escape(_normalize_test_key(str(item.get('test', '-'))))} | {_md_escape(str(item.get('metric', '-')))} | {_md_escape(value_str)} |"
            )
    else:
        lines.append("- No KPI highlights available for this run.")

    lines.extend([
        "",
        "## Domain Details",
        "",
    ])

    for domain in domain_order:
        domain_entries = domain_rows.get(domain, [])
        lines.extend([
            "",
            f"### {domain}",
            "",
        ])

        if not domain_entries:
            lines.append("- No tests captured for this domain.")
            continue

        lines.extend([
            "<div class=\"report-scroll\">",
            "<table class=\"report-table\">",
            "<thead><tr><th style=\"min-width: 180px;\">Section</th><th style=\"min-width: 230px;\">TestCaseID</th><th style=\"min-width: 280px;\">Test</th><th style=\"min-width: 90px;\">Status</th><th style=\"min-width: 320px;\">Outcome (2-3 lines)</th><th style=\"min-width: 340px;\">Success / Failure KPI Snapshot</th></tr></thead>",
            "<tbody>",
        ])

        sorted_entries = sorted(
            domain_entries,
            key=lambda item: (_status_sort_key(item[1].get("status", "")), item[1].get("section", "Other"), item[0]),
        )
        for nodeid, row in sorted_entries:
            kpi_metrics = _lookup_kpi_metrics(nodeid, latest_kpis)
            kpi_text = _compact_metrics(kpi_metrics, max_items=4) if kpi_metrics else _lookup_log_kpi(nodeid, log_payload)
            fail_reason = log_payload.get("failure_reason_by_test", {}).get(nodeid) or log_payload.get("failure_reason_by_test", {}).get(_normalize_test_key(nodeid))
            testcase_id = row.get("testcase_id", _testcase_id_for_nodeid(nodeid, testcase_id_map))
            lines.append(
                "<tr>"
                f"<td><div class=\"cell-wrap\">{_md_escape(row.get('section', 'Other'))}</div></td>"
                f"<td><div class=\"cell-wrap\">{_md_escape(testcase_id)}</div></td>"
                f"<td><div class=\"cell-wrap\">{_md_escape(_display_test_name(nodeid))}</div></td>"
                f"<td>{_render_status(row['status'])}</td>"
                f"<td>{_compact_outcome_lines(row.get('status', ''), row.get('result', ''), row.get('analysis', ''), fail_reason)}</td>"
                f"<td><div class=\"cell-wrap\">{_md_escape(_truncate_line(kpi_text, max_chars=220))}</div></td>"
                "</tr>"
            )

        lines.extend([
            "</tbody>",
            "</table>",
            "</div>",
        ])

    report_body = "\n".join(lines) + "\n"
    output_paths = [_REPORT_PATH] + _REPORT_ALIAS_PATHS
    for output_path in output_paths:
        output_path.write_text(report_body, encoding="utf-8")
        logger.info("Wrote testcase report: %s", output_path)

# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


_numpy = _try_import("numpy")
_psutil = _try_import("psutil")
_torch = _try_import("torch")
_pynvml = _try_import("pynvml")
_multiprocessing = _try_import("multiprocessing")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    _configure_default_env()
    config.addinivalue_line("markers", "cpu: tests that exercise CPU compute")
    config.addinivalue_line("markers", "memory: tests that exercise the memory subsystem")
    config.addinivalue_line("markers", "gpu: tests that require NVIDIA GPU(s)")
    config.addinivalue_line("markers", "nvlink: tests that require NVLink fabric")
    config.addinivalue_line("markers", "pcie: tests that measure PCIe host↔device bandwidth")
    config.addinivalue_line("markers", "bottleneck: bottleneck / roofline analysis tests")
    config.addinivalue_line("markers", "slow: tests that take more than 30 s to complete")
    config.addinivalue_line("markers", "node_validation: node validation and bring-up tests")
    config.addinivalue_line("markers", "gpu_enumeration: GPU enumeration and topology tests")
    config.addinivalue_line("markers", "pcie_validation: PCIe bandwidth validation tests")
    config.addinivalue_line("markers", "memory_validation: memory type validation tests")
    config.addinivalue_line("markers", "p2p_validation: GPU P2P bandwidth tests")
    config.addinivalue_line("markers", "sync_overhead: CPU-GPU sync overhead tests")
    config.addinivalue_line("markers", "power_validation: power and thermal tests")
    config.addinivalue_line("markers", "nccl_validation: NCCL and inter-GPU communication tests")
    config.addinivalue_line("markers", "workload_acceptance: workload acceptance tests")
    config.addinivalue_line("markers", "rdma_validation: RDMA/MMIO/FENCE/DMA functional validation tests")
    config.addinivalue_line("markers", "mlperf: MLPerf benchmark and compliance tests")
    config.addinivalue_line("markers", "ai_pipeline: all AI pipeline KPI tests (data loading, model orch, storage, inference)")
    config.addinivalue_line("markers", "data_loading: data loading and preprocessing KPI tests")
    config.addinivalue_line("markers", "model_orch: model orchestration KPI tests (load time, checkpoint, KV cache, TP)")
    config.addinivalue_line("markers", "storage_net_io: storage and networking I/O KPI tests")
    config.addinivalue_line("markers", "inference_kpis: inference pipeline KPI tests (TTFT, ITL, batching, GPU util)")
    config.addinivalue_line("markers", "system_health: all system health and reliability tests")
    config.addinivalue_line("markers", "thermal_stability: GPU thermal limits and clock throttle detection")
    config.addinivalue_line("markers", "ecc_link_health: GPU ECC error counts and PCIe link health")
    config.addinivalue_line("markers", "energy_efficiency: GFLOPS/W energy efficiency measurements")
    config.addinivalue_line("markers", "quant_correctness: FP8/INT8/BF16 quantisation numerical correctness")
    config.addinivalue_line("markers", "ddp_correctness: DDP/NCCL gradient synchronisation correctness")
    config.addinivalue_line("markers", "compile_correctness: torch.compile correctness and speedup validation")
    config.addinivalue_line("markers", "stream_correctness: CUDA stream ordering and event synchronisation correctness")
    setattr(config, "_overall_report_rows", {})
    setattr(config, "_testcase_id_map", _load_testcase_id_map())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    rows: Dict[str, Dict[str, str]] = getattr(item.config, "_overall_report_rows")
    testcase_id_map: Dict[str, str] = getattr(item.config, "_testcase_id_map", {})
    nodeid = item.nodeid

    # Prefer call-phase results. For setup-skip/fail without call, capture setup phase.
    if report.when == "call" or (report.when == "setup" and report.outcome in {"failed", "skipped"}):
        status, result, analysis = _status_result_analysis(report)
        rows[nodeid] = {
            "domain": _report_domain_for_item(item),
            "section": _report_section_for_item(item),
            "purpose": _purpose_from_item(item),
            "result": result,
            "status": status,
            "analysis": analysis,
            "testcase_id": _testcase_id_for_nodeid(nodeid, testcase_id_map),
        }


def pytest_runtest_setup(item: pytest.Item) -> None:
    if _is_vllm_related_item(item):
        _terminate_stale_vllm_processes("pre-test", item.nodeid)


def pytest_runtest_teardown(item: pytest.Item, nextitem: Optional[pytest.Item]) -> None:
    if _is_vllm_related_item(item):
        _release_torch_cuda_memory()
        _terminate_stale_vllm_processes("post-test", item.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _write_final_report(session.config)

    # Best-effort cleanup: terminate any child processes left by workload tests.
    if _multiprocessing is not None:
        try:
            children = _multiprocessing.active_children()
            for proc in children:
                try:
                    proc.terminate()
                except Exception:
                    pass
            for proc in children:
                try:
                    proc.join(timeout=3)
                except Exception:
                    pass
            if children:
                logger.info("Cleaned up %d child process(es) at pytest session end", len(children))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def system_info() -> dict:
    """Collect and log basic system information once per test session."""
    info: dict = {}

    # CPU / OS
    info["platform"] = platform.platform()
    info["python_version"] = sys.version
    info["cpu_count_logical"] = os.cpu_count()

    if _psutil:
        info["cpu_count_physical"] = _psutil.cpu_count(logical=False)
        vm = _psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / 1024 ** 3, 2)
    else:
        info["cpu_count_physical"] = None
        info["total_ram_gb"] = None

    # GPU
    if _torch and _torch.cuda.is_available():
        info["gpu_count"] = _torch.cuda.device_count()
        info["gpu_names"] = [
            _torch.cuda.get_device_name(i) for i in range(_torch.cuda.device_count())
        ]
        info["cuda_version"] = _torch.version.cuda
    else:
        info["gpu_count"] = 0
        info["gpu_names"] = []
        info["cuda_version"] = None

    logger.info("System info: %s", info)
    return info


@pytest.fixture(scope="session")
def available_gpus(system_info) -> List[int]:
    """Return list of available GPU device indices."""
    return list(range(system_info.get("gpu_count", 0)))


@pytest.fixture(scope="session")
def numpy_available() -> bool:
    return _numpy is not None


@pytest.fixture(scope="session")
def torch_available() -> bool:
    return _torch is not None and _torch.cuda.is_available()


@pytest.fixture(scope="session")
def nvml_handle_list():
    """Return list of NVML device handles, or empty list if pynvml is absent."""
    if _pynvml is None:
        return []
    try:
        _pynvml.nvmlInit()
        count = _pynvml.nvmlDeviceGetCount()
        handles = [_pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        return handles
    except Exception as exc:
        logger.warning("NVML init failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Function-scoped helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cpu_core_count() -> int:
    """Physical CPU core count detected at runtime."""
    if _psutil:
        return _psutil.cpu_count(logical=False) or 1
    return os.cpu_count() or 1


@pytest.fixture
def wall_timer():
    """
    Simple context-manager timer that records elapsed wall-clock seconds.

    Usage::

        def test_something(wall_timer):
            with wall_timer() as t:
                do_work()
            assert t.elapsed < 5.0
    """

    class _Timer:
        elapsed: float = 0.0

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self._start

    class _Factory:
        def __call__(self) -> _Timer:
            return _Timer()

    return _Factory()


@pytest.fixture
def torch_device_cuda():
    """Return a CUDA torch.device for GPU index 0, skip if unavailable."""
    if _torch is None or not _torch.cuda.is_available():
        pytest.skip("CUDA / PyTorch not available")
    return _torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Utility helpers (shared across test modules)
# ---------------------------------------------------------------------------

def run_subprocess_benchmark(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    """
    Run an external benchmark command and return (returncode, stdout, stderr).
    """
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def assert_efficiency(
    measured: float,
    theoretical: float,
    threshold: float,
    label: str,
) -> None:
    """
    Assert that *measured* is at least *threshold* × *theoretical*.

    Prints a structured efficiency report regardless of pass/fail.
    """
    if theoretical <= 0:
        pytest.skip(f"Theoretical value for '{label}' is zero or negative")
    efficiency = measured / theoretical
    logger.info(
        "%-50s  measured=%10.3f  theoretical=%10.3f  efficiency=%6.2f%%  threshold=%6.0f%%",
        label,
        measured,
        theoretical,
        efficiency * 100,
        threshold * 100,
    )
    assert efficiency >= threshold, (
        f"{label}: efficiency {efficiency:.2%} is below threshold {threshold:.2%} "
        f"(measured={measured:.3f}, theoretical={theoretical:.3f})"
    )
