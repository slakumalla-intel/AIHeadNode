"""
Shared pytest fixtures and configuration for the AI Head Node test suite.

Hardware target:
  160-core CPU  (2 sockets × 80 cores, SMT-2)
  4× NVIDIA H100 SXM5 (80 GB HBM3 each, NVLink 4 full-mesh)
"""

import os
import sys
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


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _purpose_from_item(item: pytest.Item) -> str:
    doc = getattr(item.obj, "__doc__", None)
    if not doc:
        return item.name
    return " ".join(doc.strip().splitlines()).strip()


def _status_result_analysis(report: pytest.TestReport) -> Tuple[str, str, str]:
    if report.outcome == "passed":
        return "PASS", f"PASS ({report.duration:.3f}s)", "Executed successfully."

    if report.outcome == "skipped":
        reason = "Skipped"
        if hasattr(report.longrepr, "reprcrash") and report.longrepr.reprcrash:
            reason = report.longrepr.reprcrash.message
        else:
            lr = str(report.longrepr)
            if "Skipped:" in lr:
                reason = lr.split("Skipped:", 1)[-1].strip()
        return "SKIP", f"SKIPPED: {reason}", "Execution skipped due to environment or prerequisites."

    # Failed
    fail_msg = "Assertion or runtime failure"
    if hasattr(report.longrepr, "reprcrash") and report.longrepr.reprcrash:
        fail_msg = report.longrepr.reprcrash.message
    return "FAIL", f"FAILED: {fail_msg}", "Check assertion thresholds and benchmark environment consistency."


def _write_final_report(config: pytest.Config) -> None:
    rows: Dict[str, Dict[str, str]] = getattr(config, "_exec_report_rows", {})
    if not rows:
        return

    statuses = [v["status"] for v in rows.values()]
    passed = statuses.count("PASS")
    failed = statuses.count("FAIL")
    skipped = statuses.count("SKIP")
    total = len(statuses)

    highlights = [
        f"- Total testcases captured: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Skipped: {skipped}",
    ]
    if failed:
        highlights.append("- Attention required: One or more testcase failures were detected.")
    else:
        highlights.append("- No testcase failures detected in this run.")

    lines = [
        "# AI Head Node Testcase Execution Final Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Pytest command line: {' '.join(config.invocation_params.args)}",
        "",
        "## Highlights",
        "",
        *highlights,
        "",
        "## Execution Status Table",
        "",
        "| Test | Purpose | Result | Status | Analysis |",
        "|---|---|---|---|---|",
    ]

    for nodeid in sorted(rows):
        row = rows[nodeid]
        lines.append(
            "| "
            f"{_md_escape(nodeid)} | "
            f"{_md_escape(row['purpose'])} | "
            f"{_md_escape(row['result'])} | "
            f"{_md_escape(row['status'])} | "
            f"{_md_escape(row['analysis'])} |"
        )

    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote final testcase report: %s", _REPORT_PATH)

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
    config.addinivalue_line("markers", "cpu: tests that exercise CPU compute")
    config.addinivalue_line("markers", "memory: tests that exercise the memory subsystem")
    config.addinivalue_line("markers", "gpu: tests that require NVIDIA GPU(s)")
    config.addinivalue_line("markers", "nvlink: tests that require NVLink fabric")
    config.addinivalue_line("markers", "pcie: tests that measure PCIe host↔device bandwidth")
    config.addinivalue_line("markers", "bottleneck: bottleneck / roofline analysis tests")
    config.addinivalue_line("markers", "slow: tests that take more than 30 s to complete")
    setattr(config, "_exec_report_rows", {})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    rows: Dict[str, Dict[str, str]] = getattr(item.config, "_exec_report_rows")
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
    _write_final_report(session.config)


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
