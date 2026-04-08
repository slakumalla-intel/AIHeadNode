"""
Shared pytest fixtures and configuration for the AI Head Node test suite.

Hardware target:
  160-core CPU  (2 sockets × 80 cores, SMT-2)
  4× NVIDIA H100 SXM5 (80 GB HBM3 each, NVLink 4 full-mesh)
"""

import os
import sys
import json
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
_NODE_VALIDATION_REPORT_PATH = Path(__file__).resolve().parents[1] / "NODE_VALIDATION_REPORT.md"


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
    os.environ.setdefault("HF_LLAMA_7B_MODEL", "meta-llama/Llama-2-7b-hf")
    os.environ.setdefault("HF_LLAMA_70B_MODEL", "meta-llama/Llama-2-70b-hf")
    os.environ.setdefault("HF_QWEN_70B_MODEL", "Qwen/Qwen-72B")
    os.environ.setdefault("HF_DEEPSEEK_R1_QUANT_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    os.environ.setdefault("HF_INFERENCE_MODEL", "meta-llama/Llama-2-7b-hf")
    os.environ.setdefault("HF_TRAINING_MODEL", "bert-base-uncased")
    os.environ.setdefault("HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC", "300")
    
    # =========================================================================
    # KV Cache Offload Configuration
    # =========================================================================
    os.environ.setdefault("VLLM_KV_OFFLOAD_MODEL", "meta-llama/Llama-2-7b-hf")
    os.environ.setdefault("VLLM_KV_CONTEXT_LENGTH", "8192")
    os.environ.setdefault("VLLM_KV_BATCH_SIZE", "4")
    os.environ.setdefault("VLLM_KV_NUM_PROMPTS", "10")
    os.environ.setdefault("VLLM_KV_MAX_TOKENS", "512")
    os.environ.setdefault("VLLM_GPU_MEMORY_UTIL", "0.7")
    os.environ.setdefault("VLLM_CPU_OFFLOAD_GB", "128")

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
    os.environ.setdefault("YOLO_CPU_SAMPLE_INTERVAL_SEC", "0.5")
    os.environ.setdefault("YOLO_AGGREGATE_FPS_TARGET", "100")
    
    # =========================================================================
    # Vision Workload Configuration (Diffusers)
    # =========================================================================
    os.environ.setdefault("DIFFUSERS_NUM_IMAGES", "50")
    os.environ.setdefault("DIFFUSERS_HEIGHT", "512")
    os.environ.setdefault("DIFFUSERS_WIDTH", "512")
    os.environ.setdefault("DIFFUSERS_BATCH_SIZE", "4")
    
    # =========================================================================
    # PyTorch & CUDA Configuration
    # =========================================================================
    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    
    # =========================================================================
    # KPI & Test Configuration
    # =========================================================================
    os.environ.setdefault("NODE_VALIDATION_KPI_FILE", "artifacts/node_validation_kpis.jsonl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTEST_TIMEOUT", "600")
    
    logger.info(
        "Environment variables configured:\n"
        "  HF_ENABLE_LARGE_MODELS=%s\n"
        "  VLLM_KV_CONTEXT_LENGTH=%s\n"
        "  YOLO_DETECT_BATCH=%s\n"
        "  CUDA_VISIBLE_DEVICES=%s",
        os.environ.get("HF_ENABLE_LARGE_MODELS"),
        os.environ.get("VLLM_KV_CONTEXT_LENGTH"),
        os.environ.get("YOLO_DETECT_BATCH"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
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
    return " ".join(doc.strip().splitlines()).strip()


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


def _load_latest_kpis() -> Dict[str, Dict[str, Any]]:
    """Load latest KPI payload per test name from JSONL, if available."""
    out: Dict[str, Dict[str, Any]] = {}
    kpi_path = _kpi_output_path()
    if not kpi_path.exists():
        return out

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
                test_name = str(payload.get("test", "")).strip()
                if not test_name:
                    continue
                metrics = payload.get("metrics", {})
                if not isinstance(metrics, dict):
                    metrics = {"value": metrics}
                out[_normalize_test_key(test_name)] = {
                    "timestamp": payload.get("timestamp", ""),
                    "metrics": metrics,
                }
    except Exception as exc:
        logger.warning("Unable to parse KPI JSONL %s: %s", kpi_path, exc)
    return out


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


def _render_status(status: str) -> str:
    if status == "PASS":
        return _color_badge("PASS", "#1b8a3b")
    if status == "FAIL":
        return _color_badge("FAIL", "#cc1f1a")
    if status == "SKIP":
        return _color_badge("SKIP", "#1f6feb")
    return _md_escape(status)


def _render_result_text(result: str) -> str:
    if result.startswith("PASS"):
        return _color_badge(result, "#1b8a3b")
    if result.startswith("FAILED"):
        return _color_badge(result, "#cc1f1a")
    if result.startswith("SKIPPED"):
        return _color_badge(result, "#1f6feb")
    return _md_escape(result)


def _write_final_report(config: pytest.Config, is_node_validation: bool = False) -> None:
    """Write final report to appropriate file based on test category."""
    report_dict_attr = "_node_validation_report_rows" if is_node_validation else "_exec_report_rows"
    rows: Dict[str, Dict[str, str]] = getattr(config, report_dict_attr, {})
    report_path = _NODE_VALIDATION_REPORT_PATH if is_node_validation else _REPORT_PATH

    if not rows:
        return

    latest_kpis = _load_latest_kpis()

    statuses = [v["status"] for v in rows.values()]
    passed = statuses.count("PASS")
    failed = statuses.count("FAIL")
    skipped = statuses.count("SKIP")
    total = len(statuses)

    skipped_rows = [
        (nodeid, row)
        for nodeid, row in sorted(rows.items())
        if row.get("status") == "SKIP"
    ]
    failed_rows = [
        (nodeid, row)
        for nodeid, row in sorted(rows.items())
        if row.get("status") == "FAIL"
    ]
    passed_rows = [
        (nodeid, row)
        for nodeid, row in sorted(rows.items())
        if row.get("status") == "PASS"
    ]

    highlights = [
        f"- {_color_badge('INFO', '#1f6feb')}: Total testcases captured: {total}",
        f"- {_color_badge('PASS', '#1b8a3b')}: {passed}",
        f"- {_color_badge('FAIL', '#cc1f1a')}: {failed}",
        f"- {_color_badge('SKIP', '#1f6feb')}: {skipped}",
    ]
    if failed:
        highlights.append(
            f"- {_color_badge('FAIL', '#cc1f1a')}: Attention required - one or more testcase failures were detected."
        )
    else:
        highlights.append(
            f"- {_color_badge('PASS', '#1b8a3b')}: No testcase failures detected in this run."
        )

    report_title = (
        "# AI Head Node — Node Validation Report"
        if is_node_validation
        else "# AI Head Node Testcase Execution Final Report"
    )

    lines = [
        report_title,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Pytest command line: {' '.join(config.invocation_params.args)}",
        f"KPI source file: {_kpi_output_path()}",
        "",
        "<style>",
        ".report-scroll { overflow-x: auto; resize: horizontal; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px; }",
        ".report-table { border-collapse: collapse; width: 100%; min-width: 1400px; table-layout: auto; font-size: 12px; }",
        ".report-table th, .report-table td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }",
        ".report-table th { background: #f6f8fa; position: sticky; top: 0; z-index: 1; }",
        ".cell-wrap { white-space: normal; word-break: break-word; max-width: 540px; }",
        "</style>",
        "",
        "## Highlights",
        "",
        *highlights,
        "",
    ]

    top_kpis = _pick_top_kpis(latest_kpis, limit=5)
    lines.extend([
        "## Top 5 KPIs",
        "",
    ])
    if top_kpis:
        lines.extend([
            "| KPI | Value | Test |",
            "|---|---|---|",
        ])
        for item in top_kpis:
            lines.append(
                f"| {_md_escape(item['metric'])} | {_md_escape(f'{item['value']:.3f}')} | {_md_escape(item['test'])} |"
            )
    else:
        lines.append("- No numeric KPI values available for this run.")

    lines.extend([
        "",
        "## Issues Summary",
        "",
    ])

    if failed_rows:
        lines.extend([
            "| Type | Test | Reason |",
            "|---|---|---|",
        ])
        for nodeid, row in failed_rows[:20]:
            lines.append(
                f"| {_color_badge('FAIL', '#cc1f1a')} | {_md_escape(nodeid)} | {_md_escape(_extract_reason(row.get('result', '')))} |"
            )
    else:
        lines.append(f"- {_color_badge('PASS', '#1b8a3b')}: No failures detected.")

    if skipped_rows:
        lines.extend([
            "",
            "### Skipped Cases",
            "",
            "| Type | Test | Reason |",
            "|---|---|---|",
        ])
        for nodeid, row in skipped_rows[:20]:
            lines.append(
                f"| {_color_badge('SKIP', '#1f6feb')} | {_md_escape(nodeid)} | {_md_escape(_extract_reason(row.get('result', '')))} |"
            )

    lines.extend([
        "",
        "## Success Summary",
        "",
        f"- {_color_badge('PASS', '#1b8a3b')}: Successful testcases: {passed}",
    ])

    if passed_rows:
        lines.extend([
            "",
            "| Test | Success Case |",
            "|---|---|",
        ])
        for nodeid, row in passed_rows[:10]:
            lines.append(
                f"| {_md_escape(nodeid)} | {_md_escape(_extract_reason(row.get('result', '')))} |"
            )

    lines.extend([
        "",
        "## KPI Summary",
        "",
    ])

    if latest_kpis:
        lines.extend([
            "| Test Key | KPI Metrics |",
            "|---|---|",
        ])
        for test_key in sorted(latest_kpis.keys()):
            metrics = latest_kpis[test_key].get("metrics", {})
            lines.append(
                f"| {_md_escape(test_key)} | {_md_escape(_compact_metrics(metrics, max_items=8))} |"
            )
    else:
        lines.append("- No KPI records found for this run. Ensure workload tests called KPI recorder.")

    lines.extend([
        "",
        "## Execution Status Table",
        "",
        "<div class=\"report-scroll\">",
        "<table class=\"report-table\">",
        "<thead><tr><th style=\"min-width: 380px;\">Test</th><th style=\"min-width: 360px;\">Purpose</th><th style=\"min-width: 360px;\">Result</th><th style=\"min-width: 100px;\">Status</th><th style=\"min-width: 300px;\">Analysis</th><th style=\"min-width: 380px;\">KPI</th><th style=\"min-width: 320px;\">Reason / Success Case / Issue</th></tr></thead>",
        "<tbody>",
    ])

    for nodeid in sorted(rows):
        row = rows[nodeid]
        test_key = _normalize_test_key(nodeid)
        kpi_metrics = latest_kpis.get(test_key, {}).get("metrics", {})
        reason = _extract_reason(row.get("result", ""))
        lines.append(
            "<tr>"
            f"<td><div class=\"cell-wrap\">{_md_escape(nodeid)}</div></td>"
            f"<td><div class=\"cell-wrap\">{_md_escape(row['purpose'])}</div></td>"
            f"<td><div class=\"cell-wrap\">{_render_result_text(row['result'])}</div></td>"
            f"<td>{_render_status(row['status'])}</td>"
            f"<td><div class=\"cell-wrap\">{_md_escape(row['analysis'])}</div></td>"
            f"<td><div class=\"cell-wrap\">{_md_escape(_compact_metrics(kpi_metrics, max_items=6))}</div></td>"
            f"<td><div class=\"cell-wrap\">{_md_escape(reason)}</div></td>"
            "</tr>"
        )

    lines.extend([
        "</tbody>",
        "</table>",
        "</div>",
    ])

    if skipped_rows:
        lines.extend([
            "",
            "## Skipped Tests",
            "",
            "| Test | Skip Reason |",
            "|---|---|",
        ])
        for nodeid, row in skipped_rows:
            reason = row.get("result", "")
            if reason.startswith("SKIPPED:"):
                reason = reason.split("SKIPPED:", 1)[-1].strip()
            lines.append(f"| {_md_escape(nodeid)} | {_md_escape(reason)} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote final testcase report: %s", report_path)

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
    setattr(config, "_exec_report_rows", {})
    setattr(config, "_node_validation_report_rows", {})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    # Determine if this is a node_validation test
    is_node_validation = "node_validation" in [marker.name for marker in item.iter_markers()]

    # Choose appropriate report rows dictionary
    report_dict_attr = "_node_validation_report_rows" if is_node_validation else "_exec_report_rows"
    rows: Dict[str, Dict[str, str]] = getattr(item.config, report_dict_attr)
    nodeid = item.nodeid

    # Prefer call-phase results. For setup-skip/fail without call, capture setup phase.
    if report.when == "call" or (report.when == "setup" and report.outcome in {"failed", "skipped"}):
        status, result, analysis = _status_result_analysis(report)
        rows[nodeid] = {
            "purpose": _purpose_from_item(item),
            "result": result,
            "status": status,
            "analysis": analysis,
        }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Write both reports
    _write_final_report(session.config, is_node_validation=False)
    _write_final_report(session.config, is_node_validation=True)

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
