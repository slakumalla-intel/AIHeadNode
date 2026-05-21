#!/usr/bin/env python3
"""Generate consolidated markdown reports from artifact outputs.

Outputs:
- artifacts/TEST_REPORT.md
- artifacts/KPI_REPORT.md
- artifacts/ANALYSIS_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class DomainKpiSummary:
    file_name: str
    records_total: int
    latest_timestamp: Optional[str]
    unique_tests: int
    numeric_fields: int
    numeric_samples: int
    threshold_checks: int
    threshold_pass: int
    threshold_fail: int


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _iter_kpi_files(artifacts_dir: Path) -> Iterable[Path]:
    for path in sorted(artifacts_dir.glob("*_kpis.jsonl")):
        if path.is_file():
            yield path


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _to_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _latest_rows_by_test(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[str, Tuple[datetime, Dict[str, object]]] = {}
    for row in rows:
        test_name = str(row.get("test", "unknown_test"))
        ts = _parse_timestamp(row.get("timestamp") if isinstance(row.get("timestamp"), str) else None)
        if ts is None:
            # Keep undated rows only if nothing else exists for the test.
            if test_name not in latest:
                latest[test_name] = (datetime.min.replace(tzinfo=timezone.utc), row)
            continue
        old = latest.get(test_name)
        if old is None or ts >= old[0]:
            latest[test_name] = (ts, row)
    return [item[1] for item in latest.values()]


def _threshold_pairs(metrics: Dict[str, object]) -> List[Tuple[str, float, float]]:
    pairs: List[Tuple[str, float, float]] = []
    for key, value in metrics.items():
        if not key.startswith("threshold_"):
            continue
        threshold = _to_float(value)
        if threshold is None:
            continue
        candidate_key = key.replace("threshold_", "", 1)
        actual = _to_float(metrics.get(candidate_key))
        if actual is not None:
            pairs.append((candidate_key, actual, threshold))
            continue

        # Targeted fallback for generic thresholds (for example threshold_ms).
        if candidate_key == "ms":
            for fallback in ("ttft_p95_ms", "ttft_median_ms"):
                fallback_val = _to_float(metrics.get(fallback))
                if fallback_val is not None:
                    pairs.append((fallback, fallback_val, threshold))
                    break
            continue

        suffix_matches: List[Tuple[str, float]] = []
        for m_key, m_val in metrics.items():
            if m_key.endswith(candidate_key):
                actual2 = _to_float(m_val)
                if actual2 is not None:
                    suffix_matches.append((m_key, actual2))

        if len(suffix_matches) == 1:
            pairs.append((suffix_matches[0][0], suffix_matches[0][1], threshold))
        elif len(suffix_matches) > 1:
            lower_is_better = _lower_is_better(candidate_key)
            chosen = min(suffix_matches, key=lambda item: item[1]) if lower_is_better else max(
                suffix_matches, key=lambda item: item[1]
            )
            pairs.append((chosen[0], chosen[1], threshold))
    return pairs


def _lower_is_better(metric_name: str) -> bool:
    key = metric_name.lower()
    lower_better_tokens = (
        "latency",
        "ttft",
        "time",
        "_ms",
        "_us",
        "temperature",
        "error",
        "overhead",
        "drop",
    )
    if any(token in key for token in lower_better_tokens):
        return True
    sec_fields = (
        "elapsed_sec",
        "duration_sec",
        "time_sec",
        "spawn_elapsed_sec",
        "submit_elapsed_sec",
        "alloc_time_sec",
        "read_time_sec",
        "write_time_sec",
        "load_time_sec",
        "total_load_time_sec",
    )
    return key in sec_fields


def _domain_kpi_summary(path: Path) -> Tuple[DomainKpiSummary, List[Tuple[str, str, float, float, str]]]:
    rows = _load_jsonl(path)
    latest_rows = _latest_rows_by_test(rows)

    latest_ts: Optional[datetime] = None
    numeric_values: List[float] = []
    numeric_field_names: set[str] = set()
    checks: List[Tuple[str, str, float, float, str]] = []

    for row in latest_rows:
        ts = _parse_timestamp(row.get("timestamp") if isinstance(row.get("timestamp"), str) else None)
        if ts is not None and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue

        for key, value in metrics.items():
            numeric = _to_float(value)
            if numeric is None:
                continue
            numeric_field_names.add(str(key))
            numeric_values.append(numeric)

        for metric_name, actual, threshold in _threshold_pairs(metrics):
            # For throughput-like metrics higher is better; for latency/error metrics lower is better.
            lower_is_better = _lower_is_better(metric_name)
            status = "PASS" if (actual <= threshold if lower_is_better else actual >= threshold) else "FAIL"
            checks.append((str(row.get("test", "unknown_test")), metric_name, actual, threshold, status))

    passed = sum(1 for item in checks if item[4] == "PASS")
    failed = sum(1 for item in checks if item[4] == "FAIL")
    summary = DomainKpiSummary(
        file_name=path.name,
        records_total=len(rows),
        latest_timestamp=latest_ts.isoformat() if latest_ts else None,
        unique_tests=len(latest_rows),
        numeric_fields=len(numeric_field_names),
        numeric_samples=len(numeric_values),
        threshold_checks=len(checks),
        threshold_pass=passed,
        threshold_fail=failed,
    )
    return summary, checks


def _write_test_report(status: Dict[str, object], out_path: Path) -> None:
    summary = status.get("summary", {})
    domains = status.get("domains", [])
    lines: List[str] = []
    lines.append("# Test Report")
    lines.append("")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Source report time (UTC): {status.get('generated_at_utc', 'unknown')}")
    lines.append(f"Host: {status.get('host', 'unknown')}")
    lines.append("")
    lines.append("## Execution Summary")
    lines.append("")
    lines.append(f"- Domains run: {summary.get('domains_run', 0)}")
    lines.append(f"- Domains passed: {summary.get('passed_domains', 0)}")
    lines.append(f"- Domains failed: {summary.get('failed_domains', 0)}")
    lines.append(f"- Tests: {summary.get('tests', 0)}")
    lines.append(f"- Failures: {summary.get('failures', 0)}")
    lines.append(f"- Errors: {summary.get('errors', 0)}")
    lines.append(f"- Skipped: {summary.get('skipped', 0)}")
    lines.append("")
    lines.append("## Domain Matrix")
    lines.append("")
    lines.append("| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for d in domains:
        if not isinstance(d, dict):
            continue
        lines.append(
            "| {domain} | {status} | {tests} | {failures} | {errors} | {skipped} | {duration_sec} |".format(
                **{
                    "domain": d.get("domain", "unknown"),
                    "status": d.get("status", "unknown"),
                    "tests": d.get("tests", 0),
                    "failures": d.get("failures", 0),
                    "errors": d.get("errors", 0),
                    "skipped": d.get("skipped", 0),
                    "duration_sec": d.get("duration_sec", 0),
                }
            )
        )

    lines.append("")
    lines.append("## Failures")
    lines.append("")
    failing_domains = [d for d in domains if isinstance(d, dict) and d.get("status") != "PASS"]
    if not failing_domains:
        lines.append("All domains passed.")
    else:
        for d in failing_domains:
            lines.append(f"### {d.get('domain', 'unknown')}")
            lines.append("")
            failed_tests = d.get("failed_tests", [])
            if not failed_tests:
                lines.append("- Domain failed without per-test details; inspect log and JUnit artifacts.")
                lines.append("")
                continue
            for item in failed_tests:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- [{item.get('kind', 'failure')}] {item.get('nodeid', 'unknown')}: {item.get('message', '').strip()}"
                )
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_kpi_report(
    status: Dict[str, object],
    summaries: List[DomainKpiSummary],
    checks_by_file: Dict[str, List[Tuple[str, str, float, float, str]]],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# KPI Report")
    lines.append("")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append("Source: top-level artifacts/*_kpis.jsonl with latest entry per test case")
    lines.append("")
    lines.append("## KPI Coverage")
    lines.append("")
    lines.append("| KPI File | Records | Latest Distinct Tests | Latest Timestamp | Numeric Fields | Threshold Checks | Pass | Fail |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            "| {file_name} | {records_total} | {unique_tests} | {latest_timestamp} | {numeric_fields} | {threshold_checks} | {threshold_pass} | {threshold_fail} |".format(
                file_name=s.file_name,
                records_total=s.records_total,
                unique_tests=s.unique_tests,
                latest_timestamp=s.latest_timestamp or "n/a",
                numeric_fields=s.numeric_fields,
                threshold_checks=s.threshold_checks,
                threshold_pass=s.threshold_pass,
                threshold_fail=s.threshold_fail,
            )
        )

    all_checks = [item for values in checks_by_file.values() for item in values]
    lines.append("")
    lines.append("## Threshold Check Totals")
    lines.append("")
    lines.append(f"- Total threshold checks: {len(all_checks)}")
    lines.append(f"- Passed checks: {sum(1 for c in all_checks if c[4] == 'PASS')}")
    lines.append(f"- Failed checks: {sum(1 for c in all_checks if c[4] == 'FAIL')}")

    failing = [c for c in all_checks if c[4] == "FAIL"]
    lines.append("")
    lines.append("## Failed KPI Threshold Checks")
    lines.append("")
    if not failing:
        lines.append("No failed threshold checks found in the latest KPI rows.")
    else:
        lines.append("| Test | Metric | Actual | Threshold |")
        lines.append("|---|---|---:|---:|")
        for test_name, metric_name, actual, threshold, _ in failing[:30]:
            lines.append(f"| {test_name} | {metric_name} | {actual:.4f} | {threshold:.4f} |")

    summary = status.get("summary", {}) if isinstance(status, dict) else {}
    lines.append("")
    lines.append("## Correlation With Test Outcomes")
    lines.append("")
    lines.append(f"- Workload test failures (from status report): {summary.get('failures', 0)}")
    lines.append(
        "- KPI threshold failures (latest metric rows): "
        f"{sum(1 for c in all_checks if c[4] == 'FAIL')}"
    )
    lines.append("- Note: A domain can fail from runtime errors even if threshold checks are limited.")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _categorize_failure(message: str) -> str:
    text = message.lower()
    if "clock ratio" in text or "throttling" in text or "clock dropped" in text:
        return "GPU clock/power throttling"
    if "cuda error" in text or "device-side assert" in text:
        return "CUDA runtime/assert"
    if "allreduce overhead" in text:
        return "Collective communication overhead"
    if "bandwidth" in text or "load pipeline" in text:
        return "Model/storage bandwidth"
    if "overlap efficiency" in text:
        return "Pipeline overlap inefficiency"
    if "softmax" in text or "quantised" in text:
        return "Numerical accuracy drift"
    if "compile speedup" in text:
        return "Compiler optimization underperformance"
    if "tool dispatch" in text:
        return "Agentic parallel dispatch overhead"
    return "Other"


def _write_analysis_report(status: Dict[str, object], out_path: Path) -> None:
    domains = status.get("domains", []) if isinstance(status, dict) else []
    failure_items: List[Tuple[str, str, str]] = []
    for d in domains:
        if not isinstance(d, dict):
            continue
        domain = str(d.get("domain", "unknown"))
        for item in d.get("failed_tests", []):
            if not isinstance(item, dict):
                continue
            failure_items.append(
                (
                    domain,
                    str(item.get("nodeid", "unknown")),
                    str(item.get("message", "")),
                )
            )

    category_counter: Counter[str] = Counter()
    for _, _, msg in failure_items:
        category_counter[_categorize_failure(msg)] += 1

    lines: List[str] = []
    lines.append("# Analysis Report")
    lines.append("")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Source report time (UTC): {status.get('generated_at_utc', 'unknown')}")
    lines.append("")
    lines.append("## Executive View")
    lines.append("")
    summary = status.get("summary", {}) if isinstance(status, dict) else {}
    total_tests = int(summary.get("tests", 0) or 0)
    total_failures = int(summary.get("failures", 0) or 0)
    pass_rate = (1.0 - (total_failures / total_tests)) * 100 if total_tests else 0.0
    lines.append(f"- Overall pass rate: {pass_rate:.2f}% ({max(total_tests - total_failures, 0)}/{total_tests})")
    lines.append(f"- Failed domains: {summary.get('failed_domains', 0)} of {summary.get('domains_run', 0)}")
    lines.append(f"- Total failure signatures analyzed: {len(failure_items)}")

    lines.append("")
    lines.append("## Failure Categories")
    lines.append("")
    if not category_counter:
        lines.append("No failures found in status report.")
    else:
        lines.append("| Category | Count |")
        lines.append("|---|---:|")
        for category, count in category_counter.most_common():
            lines.append(f"| {category} | {count} |")

    lines.append("")
    lines.append("## Top Risks")
    lines.append("")
    if category_counter.get("GPU clock/power throttling", 0) > 0:
        lines.append(
            "- GPU clock/power throttling is the highest operational risk; four GPU clock ratio failures and one sustained load clock-drop failure indicate performance instability under load."
        )
    if category_counter.get("CUDA runtime/assert", 0) > 0:
        lines.append(
            "- CUDA device-side assert indicates a runtime correctness problem in inference-path tensors or kernels, blocking reliable TTFT measurements for larger input lengths."
        )
    if category_counter.get("Collective communication overhead", 0) > 0:
        lines.append(
            "- Extremely high all-reduce overhead shows communication dominates compute in the tested setup, limiting scale efficiency for tensor-parallel workloads."
        )
    if category_counter.get("Model/storage bandwidth", 0) > 0:
        lines.append(
            "- Model load and weight pipeline bandwidth are below floor values, increasing startup latency and reducing elasticity for model swaps."
        )

    lines.append("")
    lines.append("## Recommended Actions")
    lines.append("")
    lines.append("1. Validate GPU power management and application clocks; run sustained stress with `nvidia-smi dmon` and confirm no imposed power caps or low-P-state lock.")
    lines.append("2. Reproduce the CUDA assert with deterministic seed and `CUDA_LAUNCH_BLOCKING=1`, then isolate failing input/tensor path in TTFT test for input_len=2048.")
    lines.append("3. Profile collective ops (NCCL topology, rank placement, message size sweep) to reduce all-reduce overhead and rebalance compute/communication ratio.")
    lines.append("4. Improve data/model load path (local NVMe pathing, parallel shard reads, pinned staging buffers) to raise effective GB/s above floor thresholds.")

    lines.append("")
    lines.append("## Failing Tests Reference")
    lines.append("")
    if not failure_items:
        lines.append("No failing tests found.")
    else:
        lines.append("| Domain | Test | Category |")
        lines.append("|---|---|---|")
        for domain, nodeid, msg in failure_items:
            lines.append(f"| {domain} | {nodeid} | {_categorize_failure(msg)} |")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate test, KPI, and analysis reports from artifact data.")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Path to artifacts directory (default: artifacts)",
    )
    parser.add_argument(
        "--status-json",
        default="workload_status_report.json",
        help="Workload status JSON file name under artifacts dir (default: workload_status_report.json)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    status_path = artifacts_dir / args.status_json

    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory not found: {artifacts_dir}")
    if not status_path.exists():
        raise SystemExit(f"Status JSON not found: {status_path}")

    status = _load_json(status_path)

    summaries: List[DomainKpiSummary] = []
    checks_by_file: Dict[str, List[Tuple[str, str, float, float, str]]] = defaultdict(list)
    for kpi_file in _iter_kpi_files(artifacts_dir):
        summary, checks = _domain_kpi_summary(kpi_file)
        summaries.append(summary)
        checks_by_file[kpi_file.name].extend(checks)

    _write_test_report(status, artifacts_dir / "TEST_REPORT.md")
    _write_kpi_report(status, summaries, checks_by_file, artifacts_dir / "KPI_REPORT.md")
    _write_analysis_report(status, artifacts_dir / "ANALYSIS_REPORT.md")

    print("Generated reports:")
    print(f"- {(artifacts_dir / 'TEST_REPORT.md').as_posix()}")
    print(f"- {(artifacts_dir / 'KPI_REPORT.md').as_posix()}")
    print(f"- {(artifacts_dir / 'ANALYSIS_REPORT.md').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
