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
    # Timeout defaults
    "PYTEST_TIMEOUT": "600",
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
    [
        (
            "bottleneck",
            DomainSpec(
                key="bottleneck",
                description="Roofline and bottleneck analysis",
                pytest_args=["tests/test_bottleneck_analysis.py"],
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
            "node_validation",
            DomainSpec(
                key="node_validation",
                description="Node bring-up and workload acceptance",
                pytest_args=["tests/test_node_validation.py"],
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
            "system_health",
            DomainSpec(
                key="system_health",
                description="System health and reliability",
                pytest_args=["tests/test_system_health_and_reliability.py"],
            ),
        ),
        (
            "system_config",
            DomainSpec(
                key="system_config",
                description="System config report validation",
                pytest_args=["tests/test_system_config_report.py"],
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


def _select_domains(only: Optional[str]) -> List[DomainSpec]:
    if not only:
        return list(DOMAINS.values())

    requested = [x.strip() for x in only.split(",") if x.strip()]
    unknown = [x for x in requested if x not in DOMAINS]
    if unknown:
        known = ", ".join(DOMAINS.keys())
        raise SystemExit(f"Unknown domain(s): {', '.join(unknown)}. Known domains: {known}")
    return [DOMAINS[k] for k in requested]


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
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_domains:
        for key, spec in DOMAINS.items():
            print(f"{key:18s} {spec.description}")
        return 0

    root = Path(__file__).resolve().parents[1]
    dirs = _ensure_artifact_dirs(root)

    domains = _select_domains(args.only)
    env = _build_env()
    extra_pytest_args = shlex.split(args.pytest_args) if args.pytest_args else []

    domain_results: List[Dict[str, object]] = []
    for domain in domains:
        print(f"[run] {domain.key}: {domain.description}")
        row = _run_domain(
            root=root,
            domain=domain,
            env=env,
            junit_dir=dirs["junit"],
            logs_dir=dirs["logs"],
            extra_pytest_args=extra_pytest_args,
        )
        domain_results.append(row)
        print(
            f"[done] {domain.key}: status={row['status']} tests={row['tests']} "
            f"failures={row['failures']} errors={row['errors']} skipped={row['skipped']}"
        )
        if args.stop_on_failure and row["status"] != "PASS":
            break

    summary = {
        "domains_run": len(domain_results),
        "passed_domains": sum(1 for d in domain_results if d["status"] == "PASS"),
        "failed_domains": sum(1 for d in domain_results if d["status"] != "PASS"),
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

    report_json = root / args.report_json
    report_md = root / args.report_md
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, report_md)

    print(f"[report] JSON: {report_json.relative_to(root)}")
    print(f"[report] Markdown: {report_md.relative_to(root)}")

    return 1 if summary["failed_domains"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
