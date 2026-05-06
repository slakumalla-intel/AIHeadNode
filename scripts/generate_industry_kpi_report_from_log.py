#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "alltests.log"
REPORT_FILE = ROOT / "artifacts" / "reports" / "industry_standard_kpi_report.md"


RESULT_RE = re.compile(r"^(tests/[^:]+)::.*?\s+(PASSED|FAILED|SKIPPED)\b")
TEST_PATH_RE = re.compile(r"^(tests/[^:]+)::")
COLLECTED_RE = re.compile(r"collecting \.\.\. collected (\d+) items")
GENERATED_RE = re.compile(r"generated xml file: .*/([^/]+\.xml) -")
SUMMARY_RE = re.compile(r"=+ .* in .* =+")
PAIR_RE = re.compile(r"(\d+)\s+(failed|passed|skipped|warnings)")
MEASURED_RE = re.compile(
    r"ai_headnode_tests\s+(.+?)\s+measured=\s*([0-9.]+)\s+theoretical=\s*([0-9.]+)\s+efficiency=\s*([0-9.]+)%\s+threshold=\s*([0-9.]+)%"
)
JSON_KPI_RE = re.compile(r"KPI (?:\[([^\]]+)\]|([A-Za-z0-9_]+)):\s*(\{.*\})")
LOGGER_DOMAIN_RE = re.compile(r"tests\.test_([a-zA-Z0-9_]+)")


def domain_from_test_path(test_path: str) -> str:
    stem = Path(test_path).stem
    if stem.startswith("test_"):
        stem = stem[5:]
    mapping = {
        "ai_pipeline_kpis": "ai_pipeline",
        "bottleneck_analysis": "bottleneck",
        "cpu_validation": "cpu",
        "gpu_validation": "gpu",
        "interconnect_validation": "interconnect",
        "memory_validation": "memory",
        "mlperf_suite": "mlperf_suite",
        "node_validation": "node_validation",
        "performance_suite": "performance_suite",
        "system_config_report": "system_config",
        "system_health_and_reliability": "system_health",
        "vllm_prefill_validation": "vllm_prefill",
        "agentic_ai_validation": "agentic_ai",
    }
    return mapping.get(stem, stem)


def parse_sessions(lines: list[str]) -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    def finalize() -> None:
        nonlocal current
        if not current:
            return
        collected = int(current.get("collected", 0) or 0)
        counts = current.get("counts", {}) or {}
        passed = int(counts.get("passed", 0))
        failed = int(counts.get("failed", 0))
        skipped = int(counts.get("skipped", 0))
        if collected:
            sessions.append(
                {
                    "suite": current.get("suite") or f"{current.get('domain', 'unknown')}.xml",
                    "domain": current.get("domain") or str(current.get("suite", "unknown")).replace(".xml", ""),
                    "collected": collected,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "executed": passed + failed,
                    "pass_rate": (passed / (passed + failed) * 100.0) if (passed + failed) else 0.0,
                }
            )
        current = None

    for line in lines:
        collected_match = COLLECTED_RE.search(line)
        if collected_match:
            finalize()
            current = {"collected": int(collected_match.group(1)), "counts": {}}
            continue

        if current is None:
            continue

        result_match = RESULT_RE.match(line)
        if result_match:
            current.setdefault("domain", domain_from_test_path(result_match.group(1)))
            continue

        generated_match = GENERATED_RE.search(line)
        if generated_match:
            current["suite"] = generated_match.group(1)
            current.setdefault("domain", generated_match.group(1).replace(".xml", ""))
            continue

        if SUMMARY_RE.match(line):
            pairs = {name: int(value) for value, name in PAIR_RE.findall(line)}
            if pairs:
                current["counts"] = pairs

    finalize()
    return sessions


def parse_measured_kpis(lines: list[str], sessions: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current_domain = "unknown"
    seen: set[tuple[object, ...]] = set()

    for line in lines:
        test_path_match = TEST_PATH_RE.match(line)
        if test_path_match:
            current_domain = domain_from_test_path(test_path_match.group(1))

        logger_match = LOGGER_DOMAIN_RE.search(line)
        if logger_match:
            current_domain = domain_from_test_path(f"tests/test_{logger_match.group(1)}.py")

        result_match = RESULT_RE.match(line)
        if result_match:
            current_domain = domain_from_test_path(result_match.group(1))
            continue

        metric_match = MEASURED_RE.search(line)
        if not metric_match:
            continue

        metric_name, measured, theoretical, efficiency, threshold = metric_match.groups()
        key = (current_domain, metric_name, measured, theoretical, threshold)
        if key in seen:
            continue
        seen.add(key)
        eff = float(efficiency)
        thr = float(threshold)
        records.append(
            {
                "domain": current_domain,
                "metric": metric_name.strip(),
                "measured": float(measured),
                "theoretical": float(theoretical),
                "efficiency": eff,
                "threshold": thr,
                "status": "PASS" if eff >= thr else "FAIL",
            }
        )

    return records


def extract_threshold_records(lines: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    current_domain = "unknown"

    def test_label(test_name: str, data: dict[str, object]) -> str:
        parts = [test_name]
        model_id = data.get("model_id") or data.get("model")
        input_len = data.get("input_len")
        workload = data.get("workload")
        if isinstance(model_id, str) and model_id:
            parts.append(model_id)
        if workload not in (None, ""):
            parts.append(f"workload={workload}")
        if isinstance(input_len, (int, float)):
            parts.append(f"input_len={int(input_len)}")
        return " | ".join(parts)

    def add_record(test_name: str, actual_metric: str, actual_value: float, threshold_metric: str, threshold_value: float, op: str, data: dict[str, object]) -> None:
        label = test_label(test_name, data)
        key = (label, actual_metric, threshold_metric)
        if key in seen:
            return
        seen.add(key)
        if op == ">=":
            status = "PASS" if actual_value >= threshold_value else "FAIL"
            gap = actual_value - threshold_value
        else:
            status = "PASS" if actual_value <= threshold_value else "FAIL"
            gap = threshold_value - actual_value
        records.append(
            {
                "domain": current_domain,
                "test": label,
                "actual_metric": actual_metric,
                "actual_value": actual_value,
                "threshold_metric": threshold_metric,
                "threshold_value": threshold_value,
                "rule": op,
                "gap": gap,
                "status": status,
            }
        )

    for line in lines:
        test_path_match = TEST_PATH_RE.match(line)
        if test_path_match:
            current_domain = domain_from_test_path(test_path_match.group(1))

        logger_match = LOGGER_DOMAIN_RE.search(line)
        if logger_match:
            current_domain = domain_from_test_path(f"tests/test_{logger_match.group(1)}.py")

        result_match = RESULT_RE.match(line)
        if result_match:
            current_domain = domain_from_test_path(result_match.group(1))
            continue

        kpi_match = JSON_KPI_RE.search(line)
        if not kpi_match:
            continue

        test_name = (kpi_match.group(1) or kpi_match.group(2) or "unknown").strip()
        payload = kpi_match.group(3)
        try:
            data = json.loads(payload)
        except Exception:
            continue

        mappings = [
            ("overlap_efficiency_pct", "threshold_overlap_pct", ">="),
            ("tokens_per_sec", "threshold_tokens_per_sec", ">="),
            ("total_tokens_per_sec", "threshold_tokens_per_sec", ">="),
            ("prefill_throughput_tokens_per_sec", "threshold_tokens_per_sec", ">="),
            ("samples_per_sec", "threshold_samples_per_sec", ">="),
            ("load_bandwidth_gbs", "threshold_gbs", ">="),
            ("end_to_end_gbs", "threshold_gbs", ">="),
            ("read_bandwidth_gbs", "threshold_gbs", ">="),
            ("write_bandwidth_gbs", "threshold_gbs", ">="),
            ("throughput_gbps", "threshold_gbps", ">="),
            ("iops", "threshold_iops", ">="),
            ("blocks_per_sec", "threshold_blocks_per_sec", ">="),
            ("restore_bandwidth_gbs", "threshold_restore_gbs", ">="),
            ("save_bandwidth_gbs", "threshold_save_gbs", ">="),
            ("assemble_us_per_batch", "threshold_us_per_batch", "<="),
            ("comm_overhead_pct", "threshold_comm_overhead_pct", "<="),
            ("ttft_p95_ms", "threshold_ms", "<="),
            ("mean_sync_pct", "threshold_sync_pct", "<="),
            ("scaling_ratio", "threshold_scaling_ratio", ">="),
            ("min_gbs", "min_threshold_gbs", ">="),
            ("unidir_min_gbs", "min_threshold_gbs", ">="),
            ("throughput_img_per_sec", "throughput_target_img_per_sec", ">="),
            ("aggregate_fps", "aggregate_fps_target", ">="),
        ]

        for actual_metric, threshold_metric, op in mappings:
            actual_value = data.get(actual_metric)
            threshold_value = data.get(threshold_metric)
            if isinstance(actual_value, (int, float)) and isinstance(threshold_value, (int, float)):
                add_record(test_name, actual_metric, float(actual_value), threshold_metric, float(threshold_value), op, data)

    return records


def parse_workload_kpis(lines: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    current_domain = "unknown"

    interesting_tests = (
        "vllm",
        "hf_causal_lm",
        "time_to_first_token",
        "training",
        "inference",
        "batch_throughput",
    )
    metric_order = [
        "ttft_p95_ms",
        "ttft_median_ms",
        "prefill_throughput_tokens_per_sec",
        "throughput_tokens_per_sec",
        "mean_latency_ms",
        "p95_latency_ms",
        "samples_per_sec",
        "total_samples_per_sec",
        "mean_tflops_per_gpu",
        "completion_time_sec",
        "runtime_sec",
    ]

    for line in lines:
        test_path_match = TEST_PATH_RE.match(line)
        if test_path_match:
            current_domain = domain_from_test_path(test_path_match.group(1))

        logger_match = LOGGER_DOMAIN_RE.search(line)
        if logger_match:
            current_domain = domain_from_test_path(f"tests/test_{logger_match.group(1)}.py")

        result_match = RESULT_RE.match(line)
        if result_match:
            current_domain = domain_from_test_path(result_match.group(1))
            continue

        kpi_match = JSON_KPI_RE.search(line)
        if not kpi_match:
            continue

        test_name = (kpi_match.group(1) or kpi_match.group(2) or "unknown").strip()
        payload = kpi_match.group(3)
        try:
            data = json.loads(payload)
        except Exception:
            continue

        model = str(data.get("model") or data.get("model_id") or data.get("component") or "")
        if not model and not any(token in test_name.lower() for token in interesting_tests):
            continue

        metrics: list[str] = []
        for name in metric_order:
            value = data.get(name)
            if isinstance(value, (int, float)):
                metrics.append(f"{name}={float(value):.3f}")

        status = str(data.get("status") or "INFO").upper()
        note = str(data.get("note") or "")
        workload = str(data.get("workload") or "")
        label = test_name
        input_len = data.get("input_len")
        if isinstance(input_len, (int, float)):
            label = f"{test_name}[input_len={int(input_len)}]"

        key = (current_domain, label, model)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "domain": current_domain,
                "test": label,
                "model": model or "-",
                "workload": workload or "-",
                "metrics": ", ".join(metrics) if metrics else "-",
                "status": status,
                "note": note or "-",
            }
        )

    return records


def render_report(
    sessions: list[dict[str, object]],
    measured_kpis: list[dict[str, object]],
    threshold_kpis: list[dict[str, object]],
    workload_kpis: list[dict[str, object]],
) -> str:
    total = sum(int(s["collected"]) for s in sessions)
    passed = sum(int(s["passed"]) for s in sessions)
    failed = sum(int(s["failed"]) for s in sessions)
    skipped = sum(int(s["skipped"]) for s in sessions)
    executed = passed + failed
    pass_rate = (passed / executed * 100.0) if executed else 0.0

    top_failing_suites = sorted(sessions, key=lambda s: int(s["failed"]), reverse=True)
    top_failing_suites = [s for s in top_failing_suites if int(s["failed"]) > 0][:5]
    failing_perf = [r for r in measured_kpis if r["status"] == "FAIL"]
    failing_threshold = [r for r in threshold_kpis if r["status"] == "FAIL"]

    lines: list[str] = []
    lines.append("# Industry Standard KPI Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Source: alltests.log")
    lines.append("Scope: Fresh execution-only report built directly from the current log output")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Total tests: {total}")
    lines.append(f"- Executed tests: {executed}")
    lines.append(f"- Passed: {passed}")
    lines.append(f"- Failed: {failed}")
    lines.append(f"- Skipped: {skipped}")
    lines.append(f"- Pass rate: {pass_rate:.1f}%")
    lines.append(f"- Performance KPIs parsed: {len(measured_kpis)}")
    lines.append(f"- Threshold-based workload KPIs parsed: {len(threshold_kpis)}")
    lines.append(f"- Model/workload KPI payloads parsed: {len(workload_kpis)}")
    lines.append("")
    lines.append("## Domain Scorecard")
    lines.append("")
    lines.append("| Domain | Tests | Pass | Fail | Skip | Pass Rate | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for session in sorted(sessions, key=lambda s: str(s["domain"])):
        suite_pass_rate = (int(session["passed"]) / max(int(session["executed"]), 1) * 100.0) if int(session["executed"]) else 0.0
        status = "PASS" if int(session["failed"]) == 0 else "AT_RISK"
        lines.append(
            f"| {session['domain']} | {session['collected']} | {session['passed']} | {session['failed']} | {session['skipped']} | {suite_pass_rate:.1f}% | {status} |"
        )

    lines.append("")
    lines.append("## Critical Findings")
    lines.append("")
    if top_failing_suites:
        for idx, session in enumerate(top_failing_suites, start=1):
            lines.append(f"{idx}. {session['domain']}: {session['failed']} failing tests out of {session['collected']}")
    else:
        lines.append("- No failing suites detected.")

    lines.append("")
    lines.append("## Performance KPI Scorecard")
    lines.append("")
    lines.append("| Domain | KPI | Measured | Theoretical | Efficiency | Threshold | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for record in measured_kpis:
        lines.append(
            f"| {record['domain']} | {record['metric']} | {record['measured']:.3f} | {record['theoretical']:.3f} | {record['efficiency']:.2f}% | {record['threshold']:.2f}% | {record['status']} |"
        )

    lines.append("")
    lines.append("## Workload KPI Threshold Scorecard")
    lines.append("")
    lines.append("| Domain | Test | Actual | Threshold | Rule | Gap | Status |")
    lines.append("|---|---|---:|---:|---|---:|---|")
    for record in threshold_kpis:
        lines.append(
            f"| {record['domain']} | {record['test']} | {record['actual_value']:.3f} | {record['threshold_value']:.3f} | {record['rule']} | {record['gap']:.3f} | {record['status']} |"
        )

    lines.append("")
    lines.append("## Model And Workload KPI Coverage")
    lines.append("")
    lines.append("| Domain | Test | Model/Component | Workload | Key Metrics | Status | Note |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in workload_kpis:
        lines.append(
            f"| {record['domain']} | {record['test']} | {record['model']} | {record['workload']} | {record['metrics']} | {record['status']} | {record['note']} |"
        )

    lines.append("")
    lines.append("## Priority KPI Failures")
    lines.append("")
    if failing_perf or failing_threshold:
        lines.append("### Performance KPI Failures")
        lines.append("")
        if failing_perf:
            for record in failing_perf[:15]:
                lines.append(
                    f"- {record['domain']}: {record['metric']} at {record['efficiency']:.2f}% vs threshold {record['threshold']:.2f}%"
                )
        else:
            lines.append("- No performance KPI failures.")

        lines.append("")
        lines.append("### Workload KPI Failures")
        lines.append("")
        if failing_threshold:
            for record in failing_threshold[:15]:
                lines.append(
                    f"- {record['domain']}: {record['test']} ({record['actual_metric']} {record['actual_value']:.3f} vs {record['rule']} {record['threshold_value']:.3f})"
                )
        else:
            lines.append("- No workload KPI failures.")
    else:
        lines.append("- No KPI failures detected.")

    lines.append("")
    lines.append("## Source Notes")
    lines.append("")
    lines.append("- This report is generated directly from alltests.log only.")
    lines.append("- It does not depend on pre-existing executive or KPI reports.")
    lines.append("- KPI status is derived from either logged efficiency thresholds or explicit threshold fields present in logged KPI payloads.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    sessions = parse_sessions(lines)
    measured_kpis = parse_measured_kpis(lines, sessions)
    threshold_kpis = extract_threshold_records(lines)
    workload_kpis = parse_workload_kpis(lines)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(render_report(sessions, measured_kpis, threshold_kpis, workload_kpis), encoding="utf-8")
    print(f"WROTE={REPORT_FILE.relative_to(ROOT)}")
    print(f"SESSIONS={len(sessions)}")
    print(f"MEASURED_KPIS={len(measured_kpis)}")
    print(f"THRESHOLD_KPIS={len(threshold_kpis)}")
    print(f"WORKLOAD_KPIS={len(workload_kpis)}")


if __name__ == "__main__":
    main()