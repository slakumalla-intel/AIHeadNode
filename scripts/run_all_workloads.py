#!/usr/bin/env python3
"""Run all AIHeadNode workload domains with default environment settings.

This runner executes each domain as an isolated pytest invocation so failures in one
area do not hide status from other areas. It writes per-domain JUnit XML + log files,
and a consolidated status report in JSON and Markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class DomainSpec:
    key: str
    description: str
    pytest_args: List[str]


DEFAULT_ENV: Dict[str, str] = {
    # Core execution controls
    "ML_EXECUTE_CASES": "1",
    "PERF_EXECUTE_CASES": "1",
    "PIPELINE_EXECUTE": "1",
    "AGENTIC_EXECUTE": "1",
    "HEALTH_EXECUTE": "1",
    # GPU / CUDA defaults
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    # HuggingFace / model defaults (prefer local datasets root)
    "DATASETS_ROOT": "/mnt/datasets",
    "HF_HOME": "/mnt/datasets/huggingface",
    "HF_HUB_CACHE": "/mnt/datasets/huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "/mnt/datasets/huggingface/hub",
    "TRANSFORMERS_CACHE": "/mnt/datasets/huggingface/hub",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_ENABLE_LARGE_MODELS": "1",
    "HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD": "0",
    "HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC": "300",
    "HF_INFERENCE_MODEL": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "HF_TRAINING_MODEL": "bert-base-uncased",
    "NODE_VLLM_KV_OFFLOAD_MODEL": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "MLPERF_DATA_DIR": "/mnt/datasets",
    "MLPERF_MODEL_DIR": "/mnt/datasets",
    # Node workload and install behavior
    "NODE_WORKLOAD_AUTO_INSTALL": "1",
    "NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC": "1800",
    # Tooling binaries / external workload paths
    "NVBANDWIDTH_BIN": "/root/workloads/nvbandwidth",
    "PERF_NVBANDWIDTH_BIN": "/root/workloads/nvbandwidth",
    "PERF_NCCL_TESTS_DIR": "/root/workloads/nccl-tests/build",
    "PERF_VLLM_BENCH_NUM_PROMPTS_CAP": "100",
    # Timeout defaults
    "PYTEST_TIMEOUT": "1800",
    "PERF_CMD_TIMEOUT_SEC": "1800",
    "ML_CMD_TIMEOUT_SEC": "3600",
    # KPI artifact defaults
    "NODE_VALIDATION_KPI_FILE": "artifacts/node_validation_kpis.jsonl",
    "PERF_SUITE_KPI_FILE": "artifacts/performance_suite_kpis.jsonl",
    "ML_SUITE_KPI_FILE": "artifacts/mlperf_suite_kpis.jsonl",
    "VLLM_PREFILL_KPI_FILE": "artifacts/vllm_prefill_kpis.jsonl",
    "PIPELINE_KPI_FILE": "artifacts/ai_pipeline_kpis.jsonl",
    "HEALTH_KPI_FILE": "artifacts/system_health_kpis.jsonl",
    "AGENTIC_KPI_FILE": "artifacts/agentic_ai_kpis.jsonl",
}


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
    if gpu_count <= 0:
        return value
    valid: List[str] = []
    for token in (part.strip() for part in value.split(",")):
        if token.isdigit() and int(token) < gpu_count:
            valid.append(token)
    if not valid:
        return ",".join(str(i) for i in range(gpu_count))
    return ",".join(valid)


def _default_cuda_visible_devices() -> str:
    gpu_count = _detect_nvidia_gpu_count()
    if gpu_count <= 0:
        return "0"
    return ",".join(str(i) for i in range(gpu_count))


DOMAINS: "OrderedDict[str, DomainSpec]" = OrderedDict(
    # Domain ordering is significant:
    #   1) CPU-only / read-only diagnostics first (system_config, cpu, memory) --
    #      cheap, always-safe, give us a baseline before any CUDA context exists.
    #   2) GPU diagnostics that only do short CUDA work next (bottleneck, gpu,
    #      interconnect, system_health) -- they need clean GPUs but don't risk
    #      wedging the driver.
    #   3) CPU<->GPU integration and lightweight serving (cpu_headnode,
    #      agentic_ai, ai_pipeline, vllm_prefill) -- modest risk.
    #   4) Heavy / serving / cache-offload workloads LAST (node_validation,
    #      performance_suite, mlperf_suite). These are the suites that have
    #      historically left vLLM EngineCore workers in D-state and wedged the
    #      CUDA driver. Putting them last means the cheap suites above always
    #      produce valid data even if the heavies crash the driver.
    [
        # --- Phase 1: CPU-only / safe diagnostics ---
        (
            "system_config",
            DomainSpec(
                key="system_config",
                description="System config report validation",
                pytest_args=["tests/test_system_config_report.py"],
            ),
        ),
        (
            "cpu",
            DomainSpec(
                key="cpu",
                description="CPU validation",
                pytest_args=["tests/test_cpu_validation.py"],
            ),
        ),
        (
            "memory",
            DomainSpec(
                key="memory",
                description="Memory subsystem validation",
                pytest_args=["tests/test_memory_validation.py"],
            ),
        ),
        # --- Phase 2: short CUDA diagnostics (need clean GPUs, don't wedge driver) ---
        (
            "bottleneck",
            DomainSpec(
                key="bottleneck",
                description="Roofline and bottleneck analysis",
                pytest_args=["tests/test_bottleneck_analysis.py"],
            ),
        ),
        (
            "gpu",
            DomainSpec(
                key="gpu",
                description="GPU validation",
                pytest_args=["tests/test_gpu_validation.py"],
            ),
        ),
        (
            "interconnect",
            DomainSpec(
                key="interconnect",
                description="PCIe / NVLink / P2P interconnect validation",
                pytest_args=["tests/test_interconnect_validation.py"],
            ),
        ),
        (
            "system_health",
            DomainSpec(
                key="system_health",
                description="System health and reliability",
                pytest_args=["tests/test_system_health_and_reliability.py"],
            ),
        ),
        # --- Phase 3: CPU<->GPU integration / lightweight serving ---
        (
            "cpu_headnode",
            DomainSpec(
                key="cpu_headnode",
                description="CPU<->GPU integration tests for head-node serving scenarios",
                pytest_args=["tests/test_cpu_headnode.py"],
            ),
        ),
        (
            "agentic_ai",
            DomainSpec(
                key="agentic_ai",
                description="Agentic AI KPI validation",
                pytest_args=["tests/test_agentic_ai_validation.py"],
            ),
        ),
        (
            "ai_pipeline",
            DomainSpec(
                key="ai_pipeline",
                description="AI pipeline KPI validation",
                pytest_args=["tests/test_ai_pipeline_kpis.py"],
            ),
        ),
        (
            "vllm_prefill",
            DomainSpec(
                key="vllm_prefill",
                description="vLLM prefill validation",
                pytest_args=["tests/test_vllm_prefill_validation.py"],
            ),
        ),
        # --- Phase 4: heavy / KV-offload / long-running (may wedge driver) ---
        (
            "node_validation",
            DomainSpec(
                key="node_validation",
                description="Node bring-up and workload acceptance (heavy; runs late "
                            "because KV-cache-offload tests can wedge the CUDA driver)",
                pytest_args=["tests/test_node_validation.py"],
            ),
        ),
        (
            "performance_suite",
            DomainSpec(
                key="performance_suite",
                description="Performance suite catalog / execution",
                pytest_args=["tests/test_performance_suite.py", "-k", "test_performance_case_suite"],
            ),
        ),
        (
            "mlperf_suite",
            DomainSpec(
                key="mlperf_suite",
                description="MLPerf suite catalog / execution",
                pytest_args=["tests/test_mlperf_suite.py", "-k", "test_mlperf_case_suite"],
            ),
        ),
    ]
)


# Domains in phase 2+ that need a working CUDA context. If the GPU health gate
# detects a wedged driver, the runner will mark remaining members of this set
# BLOCKED (skipped with explanation) instead of running them and producing dozens
# of identical, misleading cudaErrorDevicesUnavailable failures.
_CUDA_DEPENDENT_DOMAINS: frozenset = frozenset({
    "bottleneck",
    "gpu",
    "interconnect",
    "system_health",
    "cpu_headnode",
    "agentic_ai",
    "ai_pipeline",
    "vllm_prefill",
    "node_validation",
    "performance_suite",
    "mlperf_suite",
})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coalesce_message(testcase: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        child = testcase.find(tag)
        if child is None:
            continue
        msg = (child.attrib.get("message") or "").strip()
        if not msg:
            text = (child.text or "").strip()
            msg = text.splitlines()[0].strip() if text else tag
        return msg[:260]
    return ""


def _parse_junit(xml_path: Path) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "duration_sec": 0.0,
        "failed_tests": [],
    }
    if not xml_path.exists():
        payload["errors"] = 1
        payload["failed_tests"] = [
            {
                "nodeid": "pytest-invocation",
                "message": f"Missing JUnit report: {xml_path}",
                "kind": "error",
            }
        ]
        return payload

    root = ET.parse(xml_path).getroot()
    suites: List[ET.Element]
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = root.findall("testsuite")

    failed_tests: List[Dict[str, str]] = []
    tests = failures = errors = skipped = 0
    duration = 0.0

    for suite in suites:
        tests += int(float(suite.attrib.get("tests", "0")))
        failures += int(float(suite.attrib.get("failures", "0")))
        errors += int(float(suite.attrib.get("errors", "0")))
        skipped += int(float(suite.attrib.get("skipped", "0")))
        duration += float(suite.attrib.get("time", "0") or 0)

        for testcase in suite.findall("testcase"):
            classname = testcase.attrib.get("classname", "")
            name = testcase.attrib.get("name", "")
            nodeid = f"{classname}::{name}" if classname else name
            if testcase.find("failure") is not None:
                failed_tests.append({"nodeid": nodeid, "message": _coalesce_message(testcase), "kind": "failure"})
            elif testcase.find("error") is not None:
                failed_tests.append({"nodeid": nodeid, "message": _coalesce_message(testcase), "kind": "error"})

    payload.update(
        {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "duration_sec": round(duration, 3),
            "failed_tests": failed_tests[:50],
        }
    )
    return payload


def _domain_status(result: Dict[str, object], return_code: int) -> str:
    if int(result.get("failures", 0)) > 0 or int(result.get("errors", 0)) > 0:
        return "FAIL"
    if return_code != 0:
        return "ERROR"
    return "PASS"


def _write_markdown(report: Dict[str, object], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Workload Execution Status Report")
    lines.append("")
    lines.append(f"Generated (UTC): {report['generated_at_utc']}")
    lines.append(f"Host: {report['host']}")
    lines.append("")
    summary = report["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Domains run: {summary['domains_run']}")
    lines.append(f"- Passed: {summary['passed_domains']}")
    lines.append(f"- Failed: {summary['failed_domains']}")
    lines.append(f"- Total tests: {summary['tests']}")
    lines.append(f"- Failures: {summary['failures']}")
    lines.append(f"- Errors: {summary['errors']}")
    lines.append(f"- Skipped: {summary['skipped']}")
    lines.append("")

    lines.append("## Domain Status")
    lines.append("")
    lines.append("| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for row in report["domains"]:
        lines.append(
            "| {domain} | {status} | {tests} | {failures} | {errors} | {skipped} | {duration_sec} | {junit_xml} | {log_file} |".format(
                **row
            )
        )

    failing = [d for d in report["domains"] if d["status"] != "PASS"]
    lines.append("")
    lines.append("## Failure Details")
    lines.append("")
    if not failing:
        lines.append("All domains passed.")
    else:
        for domain in failing:
            lines.append(f"### {domain['domain']}")
            lines.append("")
            failed_tests = domain.get("failed_tests", [])
            if not failed_tests:
                lines.append("- Domain failed without testcase-level failure details. Check log and XML artifacts.")
            else:
                for item in failed_tests:
                    lines.append(
                        f"- [{item['kind']}] {item['nodeid']}: {item['message']}"
                    )
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ensure_artifact_dirs(root: Path) -> Dict[str, Path]:
    artifacts = root / "artifacts"
    reports = artifacts / "reports"
    junit_dir = reports / "junit"
    logs_dir = reports / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    junit_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "artifacts": artifacts,
        "reports": reports,
        "junit": junit_dir,
        "logs": logs_dir,
    }


def _select_domains(only: Optional[str], skip: Optional[str] = None) -> List[DomainSpec]:
    if only:
        requested = [x.strip() for x in only.split(",") if x.strip()]
        unknown = [x for x in requested if x not in DOMAINS]
        if unknown:
            known = ", ".join(DOMAINS.keys())
            raise SystemExit(f"Unknown domain(s): {', '.join(unknown)}. Known domains: {known}")
        selected = [DOMAINS[k] for k in requested]
    else:
        selected = list(DOMAINS.values())

    if skip:
        skip_keys = {x.strip() for x in skip.split(",") if x.strip()}
        unknown_skip = [k for k in skip_keys if k not in DOMAINS]
        if unknown_skip:
            known = ", ".join(DOMAINS.keys())
            raise SystemExit(
                f"Unknown --skip domain(s): {', '.join(unknown_skip)}. Known domains: {known}"
            )
        selected = [d for d in selected if d.key not in skip_keys]
        if not selected:
            raise SystemExit(
                f"--skip removed every selected domain (skip={sorted(skip_keys)}). "
                "Nothing to run."
            )
    return selected


def _run_domain(
    root: Path,
    domain: DomainSpec,
    env: Dict[str, str],
    junit_dir: Path,
    logs_dir: Path,
    extra_pytest_args: List[str],
) -> Dict[str, object]:
    junit_xml = junit_dir / f"{domain.key}.xml"
    log_file = logs_dir / f"{domain.key}.log"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "--junitxml",
        str(junit_xml),
        "--log-file",
        str(log_file),
        "--log-file-level",
        "INFO",
        *domain.pytest_args,
        *extra_pytest_args,
    ]

    started = _timestamp()
    result = subprocess.run(cmd, cwd=str(root), env=env, check=False)
    junit_stats = _parse_junit(junit_xml)
    status = _domain_status(junit_stats, result.returncode)

    return {
        "domain": domain.key,
        "description": domain.description,
        "status": status,
        "return_code": result.returncode,
        "started_at_utc": started,
        "finished_at_utc": _timestamp(),
        "tests": int(junit_stats.get("tests", 0)),
        "failures": int(junit_stats.get("failures", 0)),
        "errors": int(junit_stats.get("errors", 0)),
        "skipped": int(junit_stats.get("skipped", 0)),
        "duration_sec": float(junit_stats.get("duration_sec", 0.0)),
        "failed_tests": list(junit_stats.get("failed_tests", [])),
        "junit_xml": str(junit_xml.relative_to(root)),
        "log_file": str(log_file.relative_to(root)),
        "command": " ".join(shlex.quote(x) for x in cmd),
    }


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    for key, value in DEFAULT_ENV.items():
        env.setdefault(key, value)

    gpu_count = _detect_nvidia_gpu_count()
    if "CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = _sanitize_cuda_visible_devices(
            env["CUDA_VISIBLE_DEVICES"], gpu_count
        )
    else:
        env["CUDA_VISIBLE_DEVICES"] = _default_cuda_visible_devices()
    return env


# ---------------------------------------------------------------------------
# GPU health gate: probe between domains to detect a wedged CUDA driver and
# stop running CUDA-dependent suites in a known-broken state.
# ---------------------------------------------------------------------------
_GPU_PROBE_SCRIPT = (
    "import sys\n"
    "try:\n"
    "    import torch\n"
    "    if not torch.cuda.is_available():\n"
    "        print('no_cuda'); sys.exit(2)\n"
    "    n = torch.cuda.device_count()\n"
    "    if n == 0:\n"
    "        print('zero_devices'); sys.exit(2)\n"
    "    for i in range(n):\n"
    "        torch.cuda.set_device(i)\n"
    "        x = torch.ones(64, device=f'cuda:{i}')\n"
    "        y = (x + x).sum().item()\n"
    "        if y != 128.0:\n"
    "            print(f'bad_result_gpu_{i}'); sys.exit(3)\n"
    "    print(f'ok n={n}')\n"
    "    sys.exit(0)\n"
    "except Exception as e:\n"
    "    print(f'probe_failed: {type(e).__name__}: {e}')\n"
    "    sys.exit(4)\n"
)


def _probe_gpu_health(env: Dict[str, str], timeout_sec: float = 30.0) -> Dict[str, object]:
    """Run a short CUDA probe in a subprocess. Returns dict with ok/reason/output.

    Subprocess isolation is critical: if the driver is wedged in the parent
    process's CUDA context, a probe in the parent would itself hang.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _GPU_PROBE_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout", "stdout": "", "stderr": "probe timed out"}
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": out,
        "stderr": err,
        "reason": out if not proc.returncode == 0 else "ok",
    }


def _attempt_gpu_recovery(env: Dict[str, str]) -> Dict[str, object]:
    """Best-effort recovery: kill leftover python/vllm procs, then try gpu-reset.

    Returns a dict describing what was tried. We never raise; the caller decides
    whether to keep running based on the next probe.
    """
    actions: List[str] = []

    # 1) Reap any obvious orphan vllm/python workers holding GPU memory.
    try:
        r = subprocess.run(
            ["pkill", "-9", "-f", "vllm|EngineCore|test_node_validation"],
            capture_output=True, text=True, timeout=10,
        )
        actions.append(f"pkill rc={r.returncode}")
    except Exception as e:
        actions.append(f"pkill_failed: {type(e).__name__}")

    # 2) Try a userspace GPU reset. Requires no compute clients attached;
    #    skips silently if the driver refuses.
    try:
        r = subprocess.run(
            ["nvidia-smi", "--gpu-reset"],
            capture_output=True, text=True, timeout=60,
        )
        actions.append(f"gpu-reset rc={r.returncode}")
        if r.stdout:
            actions.append("stdout=" + r.stdout.strip().splitlines()[0][:120])
    except FileNotFoundError:
        actions.append("nvidia-smi_not_found")
    except subprocess.TimeoutExpired:
        actions.append("gpu-reset_timeout")
    except Exception as e:
        actions.append(f"gpu-reset_failed: {type(e).__name__}")

    return {"actions": actions}


def _make_blocked_row(domain: "DomainSpec", reason: str) -> Dict[str, object]:
    """Synthesize a domain-result row representing 'did not run due to broken GPUs'."""
    return {
        "domain": domain.key,
        "description": domain.description,
        "status": "BLOCKED",
        "return_code": -1,
        "duration_sec": 0.0,
        "tests": 0,
        "passed": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "first_failure": None,
        "junit": None,
        "log": None,
        "command": "",
        "blocked_reason": reason,
    }


def _clean_artifacts(junit_dir: Path, logs_dir: Path) -> int:
    """Remove stale per-domain JUnit XML and log files. Returns count removed."""
    removed = 0
    for d in (junit_dir, logs_dir):
        if not d.exists():
            continue
        for p in d.glob("*.xml"):
            try:
                p.unlink(); removed += 1
            except OSError:
                pass
        for p in d.glob("*.log"):
            try:
                p.unlink(); removed += 1
            except OSError:
                pass
    return removed


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all workload domains and emit a status/failure report."
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated domain keys to run. Default: run all domains.",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated domain keys to exclude. Applied after --only. "
             "Useful for 'run everything except the slow heavyweights', e.g. "
             "--skip mlperf_suite,performance_suite",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failing domain.",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Additional args appended to every pytest invocation.",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/workload_status_report.json",
        help="Output path for consolidated JSON report.",
    )
    parser.add_argument(
        "--report-md",
        default="artifacts/workload_status_report.md",
        help="Output path for consolidated Markdown report.",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List known domain keys and exit.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the selected domains this many times sequentially. Per-iteration "
             "reports are written to artifacts/loop/run_<i>/ and an aggregated "
             "cross-run summary to artifacts/loop/aggregate.{json,md}.",
    )
    parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="Wipe artifacts/reports/{junit,logs}/*.xml,*.log before running so "
             "the consolidated report cannot include stale data from a previous run.",
    )
    parser.add_argument(
        "--no-gpu-gate",
        action="store_true",
        help="Disable the inter-domain GPU health probe. Default behavior probes "
             "CUDA between domains and marks remaining CUDA-dependent suites BLOCKED "
             "if the driver is wedged (avoiding dozens of fake 'cudaErrorDevicesUnavailable' "
             "failures). Use --no-gpu-gate to restore the legacy unconditional run.",
    )
    parser.add_argument(
        "--gpu-probe-timeout",
        type=float,
        default=30.0,
        help="Per-probe timeout (seconds) for the inter-domain GPU health check.",
    )
    return parser.parse_args(argv)


def _run_once(
    *,
    root: Path,
    args: argparse.Namespace,
    domains: List,
    env: Dict[str, str],
    extra_pytest_args: List[str],
    junit_dir: Path,
    logs_dir: Path,
    report_json: Path,
    report_md: Path,
) -> Dict[str, object]:
    """Execute one full pass over the selected domains and write per-run reports.

    Returns the consolidated report dict for cross-run aggregation.
    """
    domain_results: List[Dict[str, object]] = []
    gpus_wedged = False
    wedged_reason = ""
    gate_enabled = not getattr(args, "no_gpu_gate", False)
    probe_timeout = float(getattr(args, "gpu_probe_timeout", 30.0))

    for domain in domains:
        # If the GPU driver was previously detected wedged and this domain
        # needs CUDA, skip it with a clear BLOCKED status instead of running
        # it and producing identical cudaErrorDevicesUnavailable failures.
        if gate_enabled and gpus_wedged and domain.key in _CUDA_DEPENDENT_DOMAINS:
            reason = (
                f"GPU health gate: CUDA driver wedged after a prior domain "
                f"({wedged_reason}). Skipped to avoid misleading failure cascade. "
                f"Run 'nvidia-smi --gpu-reset' or reboot, then rerun."
            )
            print(f"[skip] {domain.key}: BLOCKED — {reason}")
            domain_results.append(_make_blocked_row(domain, reason))
            continue

        print(f"[run] {domain.key}: {domain.description}")
        row = _run_domain(
            root=root,
            domain=domain,
            env=env,
            junit_dir=junit_dir,
            logs_dir=logs_dir,
            extra_pytest_args=extra_pytest_args,
        )
        domain_results.append(row)
        print(
            f"[done] {domain.key}: status={row['status']} tests={row['tests']} "
            f"failures={row['failures']} errors={row['errors']} skipped={row['skipped']}"
        )

        # After this domain, if it's one that might wedge the driver (or simply
        # the next domain needs CUDA), probe GPU health. Cheap and informative.
        if gate_enabled and not gpus_wedged:
            probe = _probe_gpu_health(env, timeout_sec=probe_timeout)
            if not probe["ok"]:
                wedged_reason = (
                    f"after domain '{domain.key}' "
                    f"(probe rc={probe.get('returncode')}, "
                    f"reason={probe.get('reason')!r})"
                )
                print(f"[gate] GPU probe FAILED {wedged_reason}")
                if probe.get("stdout"):
                    print(f"[gate] probe stdout: {probe['stdout'][:200]}")
                if probe.get("stderr"):
                    print(f"[gate] probe stderr: {probe['stderr'][:200]}")
                # Attempt a single best-effort recovery.
                rec = _attempt_gpu_recovery(env)
                print(f"[gate] recovery attempted: {rec['actions']}")
                # Re-probe once after recovery.
                probe2 = _probe_gpu_health(env, timeout_sec=probe_timeout)
                if probe2["ok"]:
                    print("[gate] GPUs recovered, continuing.")
                else:
                    gpus_wedged = True
                    print("[gate] GPUs still wedged; remaining CUDA domains will be BLOCKED.")

        if args.stop_on_failure and row["status"] != "PASS":
            break

    summary = {
        "domains_run": len(domain_results),
        "passed_domains": sum(1 for d in domain_results if d["status"] == "PASS"),
        "failed_domains": sum(1 for d in domain_results if d["status"] not in ("PASS", "BLOCKED")),
        "blocked_domains": sum(1 for d in domain_results if d["status"] == "BLOCKED"),
        "tests": sum(int(d["tests"]) for d in domain_results),
        "failures": sum(int(d["failures"]) for d in domain_results),
        "errors": sum(int(d["errors"]) for d in domain_results),
        "skipped": sum(int(d["skipped"]) for d in domain_results),
    }

    report = {
        "generated_at_utc": _timestamp(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "domains": domain_results,
        "summary": summary,
        "defaults_applied": dict(DEFAULT_ENV),
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, report_md)

    print(f"[report] JSON: {report_json.relative_to(root)}")
    print(f"[report] Markdown: {report_md.relative_to(root)}")
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_domains:
        for key, spec in DOMAINS.items():
            print(f"{key:18s} {spec.description}")
        return 0

    if args.repeat < 1:
        print(f"[error] --repeat must be >= 1, got {args.repeat}", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    dirs = _ensure_artifact_dirs(root)

    if getattr(args, "clean_artifacts", False):
        removed = _clean_artifacts(dirs["junit"], dirs["logs"])
        print(f"[clean] removed {removed} stale artifact file(s) from "
              f"{dirs['junit'].relative_to(root)} and {dirs['logs'].relative_to(root)}")

    domains = _select_domains(args.only, args.skip)
    env = _build_env()
    extra_pytest_args = shlex.split(args.pytest_args) if args.pytest_args else []

    # Single-run mode: preserve the original artifact layout exactly.
    if args.repeat == 1:
        report = _run_once(
            root=root,
            args=args,
            domains=domains,
            env=env,
            extra_pytest_args=extra_pytest_args,
            junit_dir=dirs["junit"],
            logs_dir=dirs["logs"],
            report_json=root / args.report_json,
            report_md=root / args.report_md,
        )
        return 1 if report["summary"]["failed_domains"] > 0 else 0

    # Repeat mode: per-iteration artifacts under artifacts/loop/run_<i>/
    loop_root = root / "artifacts" / "loop"
    loop_root.mkdir(parents=True, exist_ok=True)
    iteration_reports: List[Dict[str, object]] = []
    print(f"[loop] running {args.repeat} iterations across {len(domains)} domain(s)")
    started_at = _timestamp()
    for i in range(1, args.repeat + 1):
        iter_dir = loop_root / f"run_{i:04d}"
        iter_junit = iter_dir / "junit"
        iter_logs = iter_dir / "logs"
        iter_junit.mkdir(parents=True, exist_ok=True)
        iter_logs.mkdir(parents=True, exist_ok=True)
        print(f"\n===== loop iteration {i} / {args.repeat} =====")
        report = _run_once(
            root=root,
            args=args,
            domains=domains,
            env=env,
            extra_pytest_args=extra_pytest_args,
            junit_dir=iter_junit,
            logs_dir=iter_logs,
            report_json=iter_dir / "status.json",
            report_md=iter_dir / "status.md",
        )
        iteration_reports.append({
            "iteration": i,
            "generated_at_utc": report["generated_at_utc"],
            "summary": report["summary"],
            "per_domain": [
                {
                    "key": d["key"],
                    "status": d["status"],
                    "tests": d["tests"],
                    "failures": d["failures"],
                    "errors": d["errors"],
                    "skipped": d["skipped"],
                }
                for d in report["domains"]
            ],
        })

    # Aggregate across all iterations
    domain_stats: Dict[str, Dict[str, int]] = {}
    for ir in iteration_reports:
        for d in ir["per_domain"]:
            key = d["key"]
            slot = domain_stats.setdefault(key, {
                "runs": 0, "pass": 0, "fail": 0,
                "tests_total": 0, "failures_total": 0,
                "errors_total": 0, "skipped_total": 0,
            })
            slot["runs"] += 1
            if d["status"] == "PASS":
                slot["pass"] += 1
            else:
                slot["fail"] += 1
            slot["tests_total"] += int(d["tests"])
            slot["failures_total"] += int(d["failures"])
            slot["errors_total"] += int(d["errors"])
            slot["skipped_total"] += int(d["skipped"])

    full_pass_runs = sum(1 for ir in iteration_reports if ir["summary"]["failed_domains"] == 0)
    aggregate = {
        "started_at_utc": started_at,
        "finished_at_utc": _timestamp(),
        "host": socket.gethostname(),
        "iterations": args.repeat,
        "only": args.only or "<all>",
        "skip": args.skip or "<none>",
        "full_pass_runs": full_pass_runs,
        "any_failure_runs": args.repeat - full_pass_runs,
        "per_domain": domain_stats,
        "iteration_reports": iteration_reports,
    }
    agg_json = loop_root / "aggregate.json"
    agg_md = loop_root / "aggregate.md"
    agg_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    md_lines = [
        f"# Aggregated loop report ({args.repeat} iterations)",
        "",
        f"- host: `{aggregate['host']}`",
        f"- started: `{aggregate['started_at_utc']}`",
        f"- finished: `{aggregate['finished_at_utc']}`",
        f"- only: `{aggregate['only']}`",
        f"- full-pass runs: **{full_pass_runs} / {args.repeat}**",
        f"- runs with any failure: **{aggregate['any_failure_runs']} / {args.repeat}**",
        "",
        "## Per-domain reliability",
        "",
        "| domain | pass/runs | tests (sum) | failures (sum) | errors (sum) | skipped (sum) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(domain_stats):
        s = domain_stats[key]
        md_lines.append(
            f"| {key} | {s['pass']}/{s['runs']} | {s['tests_total']} | "
            f"{s['failures_total']} | {s['errors_total']} | {s['skipped_total']} |"
        )
    agg_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n[loop] aggregate JSON: {agg_json.relative_to(root)}")
    print(f"[loop] aggregate MD:   {agg_md.relative_to(root)}")
    print(f"[loop] full-pass runs: {full_pass_runs} / {args.repeat}")
    return 0 if full_pass_runs == args.repeat else 1


if __name__ == "__main__":
    raise SystemExit(main())
