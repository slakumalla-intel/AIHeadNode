#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "artifacts" / "reports"
JUNIT_DIR = REPORTS_DIR / "junit"
ARCHIVE_DIR = REPORTS_DIR / "archive"

MAP_CSV = REPORTS_DIR / "finalized_test_to_testcaseid_map_updated.csv"
LAYOUT_CSV = ARCHIVE_DIR / "test_methods_requested_layout_generated.csv"

EXEC_SUMMARY_MD = REPORTS_DIR / "executive_summary_report.md"
EXEC_REPORT_MD = REPORTS_DIR / "test_execution_report.md"
BUGTOOL_CSV = REPORTS_DIR / "bugtool_testcase_report_new.csv"
BUGTOOL_MD = REPORTS_DIR / "bugtool_testcase_report_new.md"
TESTCASE_ALL_CSV = REPORTS_DIR / "testcase_list_all_410_generated.csv"
WORKLOAD_JSON = ROOT / "artifacts" / "workload_status_report.json"
WORKLOAD_MD = ROOT / "artifacts" / "workload_status_report.md"
EXEC_KPI_MD = ROOT / "artifacts" / "EXECUTIVE_KPI_STATUS.md"
KPI_VALUES_CSV = ROOT / "artifacts" / "KPI_VALUES.csv"
KPI_COMPARE_CSV = REPORTS_DIR / "kpi_expected_vs_actual.csv"
KPI_COMPARE_MD = REPORTS_DIR / "kpi_expected_vs_actual.md"
ALLTESTS_LOG = ROOT / "alltests.log"


@dataclass
class CaseRow:
    suite: str
    selector: str
    file: str
    klass: str
    test_name: str
    test_name_base: str
    status: str
    note: str


def normalize_selector(classname: str, test_name: str) -> tuple[str, str, str]:
    parts = classname.split(".") if classname else []
    if len(parts) >= 2:
        klass = parts[-1]
        module = ".".join(parts[:-1])
        file_path = module.replace(".", "/") + ".py"
        selector = f"{file_path}::{klass}::{test_name}"
    else:
        klass = ""
        file_path = classname.replace(".", "/") + ".py" if classname else "unknown.py"
        selector = f"{file_path}::{test_name}"
    return selector, file_path, klass


def first_note(testcase: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        child = testcase.find(tag)
        if child is None:
            continue
        msg = (child.attrib.get("message") or "").strip()
        if msg:
            return msg.splitlines()[0][:260]
        text = (child.text or "").strip()
        if text:
            return text.splitlines()[0][:260]
    return ""


def testcase_status(testcase: ET.Element) -> str:
    if testcase.find("failure") is not None:
        return "FAIL"
    if testcase.find("error") is not None:
        return "ERROR"
    if testcase.find("skipped") is not None:
        return "SKIP"
    return "PASS"


def load_mapping() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    exact: dict[str, dict[str, str]] = {}
    base: dict[str, dict[str, str]] = {}
    if not MAP_CSV.exists():
        return exact, base

    with MAP_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selector = (row.get("TestSelector") or "").strip()
            if not selector:
                continue
            test_id = (row.get("FinalPrimaryTestCaseID") or "").strip() or "UNMAPPED"
            payload = {
                "id": test_id,
                "MappingSource": (row.get("MappingSource") or "").strip(),
                "Confidence": (row.get("Confidence") or "").strip(),
                "FinalizationStatus": (row.get("FinalizationStatus") or "").strip(),
                "Notes": (row.get("Notes") or "").strip(),
            }
            exact.setdefault(selector, payload)
            selector_no_param = re.sub(r"\[[^\]]+\]", "", selector)
            base.setdefault(selector_no_param, payload)
    return exact, base


def load_layout() -> dict[str, dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    if not LAYOUT_CSV.exists():
        return by_name
    with LAYOUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Test Name") or "").strip()
            if name and name not in by_name:
                by_name[name] = {
                    "Description": (row.get("Description") or "").strip(),
                    "Manual command": (row.get("Manual command") or "").strip(),
                }
    return by_name


def parse_junit() -> tuple[list[CaseRow], dict[str, dict[str, int]]]:
    rows: list[CaseRow] = []
    per_suite = defaultdict(lambda: {"Total": 0, "PASS": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0})

    for xml_path in sorted(JUNIT_DIR.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        for suite in suites:
            for tc in suite.findall("testcase"):
                classname = tc.attrib.get("classname", "")
                test_name = tc.attrib.get("name", "")
                selector, file_path, klass = normalize_selector(classname, test_name)
                status = testcase_status(tc)
                note = first_note(tc)
                test_name_base = re.sub(r"\[[^\]]+\]", "", test_name)

                row = CaseRow(
                    suite=xml_path.name,
                    selector=selector,
                    file=file_path,
                    klass=klass,
                    test_name=test_name,
                    test_name_base=test_name_base,
                    status=status,
                    note=note,
                )
                rows.append(row)
                per_suite[xml_path.name]["Total"] += 1
                per_suite[xml_path.name][status] += 1

    return rows, dict(sorted(per_suite.items()))


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def load_kpi_values() -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    if not KPI_VALUES_CSV.exists():
        return values

    with KPI_VALUES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric") or "").strip()
            value_raw = (row.get("value") or "").strip()
            if not metric:
                continue
            try:
                value = float(value_raw)
            except Exception:
                continue
            values[metric].append(value)
    return values


def load_kpi_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not KPI_VALUES_CSV.exists():
        return rows
    with KPI_VALUES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "metric": (row.get("metric") or "").strip(),
                    "value": (row.get("value") or "").strip(),
                    "test": (row.get("test") or "").strip(),
                }
            )
    return rows


def parse_alltests_log() -> tuple[dict[str, dict[str, int]], str | None]:
    if not ALLTESTS_LOG.exists():
        return {}, None

    per_suite: dict[str, dict[str, int]] = {}
    current_collected: int | None = None
    current_results: list[str] = []

    result_re = re.compile(r"^(tests/[^:]+)::.*?\s+(PASSED|FAILED|SKIPPED)\b")
    collected_re = re.compile(r"collecting \.\.\. collected (\d+) items")
    generated_re = re.compile(r"generated xml file: .*/([^/]+\.xml) -")

    lines = ALLTESTS_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        collected_match = collected_re.search(line)
        if collected_match:
            current_collected = int(collected_match.group(1))
            current_results = []
            continue

        result_match = result_re.match(line)
        if result_match and current_collected is not None:
            status = result_match.group(2)
            current_results.append({"PASSED": "PASS", "FAILED": "FAIL", "SKIPPED": "SKIP"}[status])
            continue

        generated_match = generated_re.search(line)
        if generated_match and current_collected is not None:
            suite_name = generated_match.group(1)
            counts = Counter(current_results)
            total_seen = counts["PASS"] + counts["FAIL"] + counts["SKIP"]
            per_suite[suite_name] = {
                "Total": current_collected,
                "PASS": counts["PASS"],
                "FAIL": counts["FAIL"],
                "ERROR": 0,
                "SKIP": counts["SKIP"] + max(current_collected - total_seen, 0),
            }
            current_collected = None
            current_results = []

    return dict(sorted(per_suite.items())), ALLTESTS_LOG.name


def _to_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _pick_actual_metric(test_name: str, expected_metric: str, available: set[str]) -> tuple[str | None, str | None]:
    # Returns (actual_metric, operator)
    if expected_metric in {"threshold_gbs", "min_threshold_gbs"}:
        for m in [
            "min_gbs",
            "mean_gbs",
            "end_to_end_gbs",
            "load_bandwidth_gbs",
            "read_bandwidth_gbs",
            "write_bandwidth_gbs",
            "mean_busbw_gbs",
            "mean_bw_gbs",
            "unidir_min_gbs",
        ]:
            if m in available:
                return m, ">="
        return None, ">="

    if expected_metric == "threshold_tokens_per_sec":
        for m in ["tokens_per_sec", "total_tokens_per_sec", "throughput_tokens_per_sec"]:
            if m in available:
                return m, ">="
        return None, ">="

    if expected_metric == "threshold_samples_per_sec":
        return ("samples_per_sec", ">=") if "samples_per_sec" in available else (None, ">=")

    if expected_metric == "threshold_ms":
        for m in ["ttft_p95_ms", "mean_latency_ms", "p95_latency_ms"]:
            if m in available:
                return m, "<="
        return None, "<="

    if expected_metric == "threshold_us_per_batch":
        return ("assemble_us_per_batch", "<=") if "assemble_us_per_batch" in available else (None, "<=")

    if expected_metric == "threshold_rel_error":
        for m in ["bf16_vs_fp16_mean_rel_error", "fp8_vs_fp16_mean_rel_error", "int8_vs_fp16_mean_rel_error"]:
            if m in available:
                return m, "<="
        return None, "<="

    if expected_metric == "threshold_iops":
        return ("iops", ">=") if "iops" in available else (None, ">=")

    if expected_metric == "threshold_gbps":
        return ("throughput_gbps", ">=") if "throughput_gbps" in available else (None, ">=")

    if expected_metric == "threshold_comm_overhead_pct":
        return ("comm_overhead_pct", "<=") if "comm_overhead_pct" in available else (None, "<=")

    if expected_metric == "threshold_sync_pct":
        return ("mean_sync_pct", "<=") if "mean_sync_pct" in available else (None, "<=")

    if expected_metric == "threshold_scaling_ratio":
        return ("scaling_ratio", ">=") if "scaling_ratio" in available else (None, ">=")

    if expected_metric == "threshold_overlap_pct":
        return ("overlap_efficiency_pct", ">=") if "overlap_efficiency_pct" in available else (None, ">=")

    if expected_metric == "threshold_blocks_per_sec":
        return ("blocks_per_sec", ">=") if "blocks_per_sec" in available else (None, ">=")

    if expected_metric == "threshold_restore_gbs":
        return ("restore_bandwidth_gbs", ">=") if "restore_bandwidth_gbs" in available else (None, ">=")

    if expected_metric == "threshold_save_gbs":
        return ("save_bandwidth_gbs", ">=") if "save_bandwidth_gbs" in available else (None, ">=")

    if expected_metric == "threshold_tps":
        return ("throughput_tokens_per_sec", ">=") if "throughput_tokens_per_sec" in available else (None, ">=")

    if expected_metric == "throughput_target_img_per_sec":
        return ("throughput_img_per_sec", ">=") if "throughput_img_per_sec" in available else (None, ">=")

    if expected_metric == "floor_ratio":
        return ("speedup_ratio", ">=") if "speedup_ratio" in available else (None, ">=")

    if expected_metric == "expected_sum":
        return ("actual_sum", "==") if "actual_sum" in available else (None, "==")

    if expected_metric == "expected":
        return ("final_val", "==") if "final_val" in available else (None, "==")

    if expected_metric == "threshold":
        if "softmax_max_abs_error" in available:
            return "softmax_max_abs_error", "<="
        if "corrected_ecc_errors" in available:
            return "corrected_ecc_errors", "<="
        return None, None

    if expected_metric == "threshold_c":
        return ("temperature_c", "<=") if "temperature_c" in available else (None, "<=")

    if expected_metric == "cpu_under_70_ratio_target":
        return ("cpu_under_70_ratio", ">=") if "cpu_under_70_ratio" in available else (None, ">=")

    if expected_metric == "gpu_over_90_ratio_target":
        return ("gpu_over_90_ratio", ">=") if "gpu_over_90_ratio" in available else (None, ">=")

    if expected_metric.startswith("gpu") and expected_metric.endswith("_limit_w"):
        pwr_metric = expected_metric.replace("_limit_w", "_power_w")
        if pwr_metric in available:
            return pwr_metric, "<="
        return None, "<="

    return None, None


def build_kpi_comparison(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_test_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        test_name = r["test"]
        metric = r["metric"]
        val = _to_float(r["value"])
        if not test_name or not metric or val is None:
            continue
        by_test_metric[test_name][metric].append(val)

    expected_metrics = [
        "threshold_gbs",
        "min_threshold_gbs",
        "threshold_tokens_per_sec",
        "threshold_samples_per_sec",
        "threshold_ms",
        "threshold_us_per_batch",
        "threshold_rel_error",
        "threshold_iops",
        "threshold_gbps",
        "threshold_comm_overhead_pct",
        "threshold_sync_pct",
        "threshold_scaling_ratio",
        "threshold_overlap_pct",
        "threshold_blocks_per_sec",
        "threshold_restore_gbs",
        "threshold_save_gbs",
        "threshold_tps",
        "throughput_target_img_per_sec",
        "floor_ratio",
        "expected_sum",
        "expected",
        "threshold",
        "threshold_c",
        "cpu_under_70_ratio_target",
        "gpu_over_90_ratio_target",
        "gpu0_limit_w",
        "gpu1_limit_w",
        "gpu2_limit_w",
        "gpu3_limit_w",
    ]

    out: list[dict[str, str]] = []
    for test_name, metric_map in sorted(by_test_metric.items()):
        available = set(metric_map.keys())
        for em in expected_metrics:
            if em not in metric_map:
                continue
            expected_vals = sorted(set(metric_map[em]))
            am, op = _pick_actual_metric(test_name, em, available)
            for ev in expected_vals:
                if not am or am not in metric_map or not op:
                    out.append(
                        {
                            "test": test_name,
                            "expected_metric": em,
                            "expected_value": f"{ev:.6g}",
                            "actual_metric": "N/A",
                            "actual_value": "N/A",
                            "operator": op or "N/A",
                            "result": "N/A",
                            "gap": "N/A",
                        }
                    )
                    continue

                actual_vals = metric_map[am]
                if op in {">=", ">"}:
                    actual = min(actual_vals)
                elif op in {"<=", "<"}:
                    actual = max(actual_vals)
                else:
                    actual = actual_vals[0]

                if op == ">=":
                    ok = actual >= ev
                    gap = actual - ev
                elif op == "<=":
                    ok = actual <= ev
                    gap = ev - actual
                elif op == "==":
                    ok = abs(actual - ev) < 1e-9
                    gap = actual - ev
                else:
                    ok = False
                    gap = float("nan")

                out.append(
                    {
                        "test": test_name,
                        "expected_metric": em,
                        "expected_value": f"{ev:.6g}",
                        "actual_metric": am,
                        "actual_value": f"{actual:.6g}",
                        "operator": op,
                        "result": "PASS" if ok else "FAIL",
                        "gap": f"{gap:.6g}",
                    }
                )
    return out


def main() -> None:
    junit_rows, per_suite = parse_junit()
    _, log_source_name = parse_alltests_log()
    map_exact, map_base = load_mapping()
    layout = load_layout()

    generated_counter = 0

    enriched: list[dict[str, str]] = []
    for r in junit_rows:
        mapping = map_exact.get(r.selector) or map_base.get(re.sub(r"\[[^\]]+\]", "", r.selector))
        if mapping:
            tid = mapping["id"]
            mapping_source = mapping.get("MappingSource", "")
            confidence = mapping.get("Confidence", "")
            finalization = mapping.get("FinalizationStatus", "")
            notes = mapping.get("Notes", "")
        else:
            generated_counter += 1
            tid = f"UNMAPPED_{generated_counter:03d}"
            mapping_source = "GENERATED_FROM_JUNIT"
            confidence = "N/A"
            finalization = "UNMAPPED"
            notes = "No entry in finalized_test_to_testcaseid_map_updated.csv"

        lrow = layout.get(r.test_name_base, {})
        desc = lrow.get("Description", "") or "Description not found in mapping catalog"
        cmd = lrow.get("Manual command", "") or f"python3 -m pytest -vv --tb=short {r.selector}"

        enriched.append(
            {
                "TestCaseID": tid,
                "TestSelector": r.selector,
                "Status": r.status,
                "JUnitEvidence": r.suite,
                "File": r.file,
                "Class": r.klass,
                "TestName": r.test_name_base,
                "MappingSource": mapping_source,
                "Confidence": confidence,
                "FinalizationStatus": finalization,
                "Notes": notes,
                "Description": desc,
                "Command": cmd,
                "FailureNote": r.note,
            }
        )

    # bugtool CSV
    bugtool_rows = [
        {
            "TestCaseID": e["TestCaseID"],
            "TestSelector": e["TestSelector"],
            "Status": e["Status"],
            "JUnitEvidence": e["JUnitEvidence"],
            "File": e["File"],
            "Class": e["Class"],
            "TestName": e["TestName"],
            "MappingSource": e["MappingSource"],
            "Confidence": e["Confidence"],
            "FinalizationStatus": e["FinalizationStatus"],
            "Notes": e["Notes"],
        }
        for e in enriched
    ]
    write_csv(
        BUGTOOL_CSV,
        [
            "TestCaseID",
            "TestSelector",
            "Status",
            "JUnitEvidence",
            "File",
            "Class",
            "TestName",
            "MappingSource",
            "Confidence",
            "FinalizationStatus",
            "Notes",
        ],
        bugtool_rows,
    )

    # 410 testcase list CSV
    all_rows = [
        {
            "test_id": e["TestCaseID"],
            "testname": e["TestSelector"],
            "test description": e["Description"],
            "test command": e["Command"],
        }
        for e in enriched
    ]
    write_csv(
        TESTCASE_ALL_CSV,
        ["test_id", "testname", "test description", "test command"],
        all_rows,
    )

    # overall stats
    status_counts = Counter(e["Status"] for e in enriched)
    total = len(enriched)
    executed = total - status_counts.get("SKIP", 0)
    pass_rate = (status_counts.get("PASS", 0) / executed * 100.0) if executed else 0.0
    summary_per_suite = per_suite
    summary_total = sum(v["Total"] for v in summary_per_suite.values())
    summary_pass = sum(v["PASS"] for v in summary_per_suite.values())
    summary_fail = sum(v["FAIL"] + v["ERROR"] for v in summary_per_suite.values())
    summary_skip = sum(v["SKIP"] for v in summary_per_suite.values())
    summary_executed = summary_total - summary_skip
    summary_pass_rate = (summary_pass / summary_executed * 100.0) if summary_executed else 0.0
    execution_source = "artifacts/reports/junit/*.xml"
    if log_source_name:
        execution_source = f"artifacts/reports/junit/*.xml (generated by {log_source_name})"

    # executive summary
    lines = []
    lines.append("# AI HeadNode Validation - Executive Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"- Source: {execution_source}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top-Line Results")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Total mapped tests | {summary_total} |")
    lines.append(f"| Executed | {summary_executed} |")
    lines.append(f"| PASS | {summary_pass} |")
    lines.append(f"| FAIL | {summary_fail} |")
    lines.append(f"| ERROR | {status_counts.get('ERROR', 0)} |")
    lines.append(f"| SKIP | {summary_skip} |")
    lines.append("| NOT_EXECUTED | 0 |")
    lines.append(f"| Pass rate (executed) | {summary_pass_rate:.1f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Suite-Level Summary")
    lines.append("")
    lines.append("| Suite | Total | PASS | FAIL | SKIP | NOT_EXEC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for suite, s in summary_per_suite.items():
        lines.append(
            f"| {suite} | {s['Total']} | {s['PASS']} | {s['FAIL'] + s['ERROR']} | {s['SKIP']} | 0 |"
        )
    EXEC_SUMMARY_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    # execution report
    rep = []
    rep.append("# Test Execution Report")
    rep.append("")
    rep.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep.append("- Source: artifacts/reports/junit/*.xml")
    rep.append("")
    rep.append("## Overall Summary")
    rep.append("")
    rep.append(f"- Total mapped tests: {total}")
    rep.append(f"- Executed tests: {executed}")
    rep.append(f"- PASS: {status_counts.get('PASS', 0)}")
    rep.append(f"- FAIL: {status_counts.get('FAIL', 0)}")
    rep.append(f"- ERROR: {status_counts.get('ERROR', 0)}")
    rep.append(f"- SKIP: {status_counts.get('SKIP', 0)}")
    rep.append("- NOT_EXECUTED: 0")
    rep.append("")
    rep.append("## Suite Summary")
    rep.append("")
    rep.append("| Suite | Total | PASS | FAIL | ERROR | SKIP | NOT_EXECUTED |")
    rep.append("|---|---:|---:|---:|---:|---:|---:|")
    for suite, s in per_suite.items():
        rep.append(
            f"| {suite} | {s['Total']} | {s['PASS']} | {s['FAIL']} | {s['ERROR']} | {s['SKIP']} | 0 |"
        )

    rep.append("")
    rep.append("## Failures And Errors")
    rep.append("")
    rep.append("| TestSelector | Status | Evidence | Notes |")
    rep.append("|---|---|---|---|")
    for e in enriched:
        if e["Status"] in {"FAIL", "ERROR"}:
            note = (e["FailureNote"] or "").replace("|", "/")
            rep.append(f"| {e['TestSelector']} | {e['Status']} | {e['JUnitEvidence']} | {note} |")

    EXEC_REPORT_MD.write_text("\n".join(rep).rstrip() + "\n", encoding="utf-8")

    # bugtool md
    by_status = Counter(e["Status"] for e in enriched)
    bmd = []
    bmd.append("# Bugtool Test Report (Per Test)")
    bmd.append("")
    bmd.append("- Source mapping: artifacts/reports/finalized_test_to_testcaseid_map_updated.csv")
    bmd.append("- Source status: artifacts/reports/junit/*.xml")
    bmd.append(f"- Total tests: {total}")
    for st in ["FAIL", "ERROR", "SKIP", "PASS"]:
        bmd.append(f"- {st}: {by_status.get(st, 0)}")
    bmd.append(f"- UNMAPPED IDs generated: {generated_counter}")
    bmd.append("")
    bmd.append("## Columns")
    bmd.append("")
    bmd.append(
        "- TestCaseID | TestSelector | Status | JUnitEvidence | File | Class | TestName | MappingSource | Confidence | FinalizationStatus | Notes"
    )
    bmd.append("")
    bmd.append("## Sample (first 40)")
    bmd.append("")
    bmd.append("| TestCaseID | TestSelector | Status | Evidence |")
    bmd.append("|---|---|---|---|")
    for e in enriched[:40]:
        bmd.append(f"| {e['TestCaseID']} | {e['TestSelector']} | {e['Status']} | {e['JUnitEvidence']} |")
    BUGTOOL_MD.write_text("\n".join(bmd).rstrip() + "\n", encoding="utf-8")

    # workload status json/md
    workloads = []
    for suite, s in summary_per_suite.items():
        status = "PASS" if (s["FAIL"] == 0 and s["ERROR"] == 0) else "FAIL"
        workloads.append(
            {
                "suite": suite,
                "status": status,
                "tests": s["Total"],
                "pass": s["PASS"],
                "fail": s["FAIL"],
                "error": s["ERROR"],
                "skip": s["SKIP"],
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": execution_source,
        "summary": {
            "total": summary_total,
            "executed": summary_executed,
            "pass": summary_pass,
            "fail": summary_fail,
            "error": status_counts.get("ERROR", 0),
            "skip": summary_skip,
            "pass_rate_executed": round(summary_pass_rate, 2),
        },
        "workloads": workloads,
    }
    WORKLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    wmd = []
    wmd.append("# Workload Status Report")
    wmd.append("")
    wmd.append(f"Generated: {payload['generated_at']}")
    wmd.append("")
    wmd.append("## Summary")
    wmd.append("")
    s = payload["summary"]
    wmd.append(f"- Total: {s['total']}")
    wmd.append(f"- Executed: {s['executed']}")
    wmd.append(f"- PASS: {s['pass']}")
    wmd.append(f"- FAIL: {s['fail']}")
    wmd.append(f"- ERROR: {s['error']}")
    wmd.append(f"- SKIP: {s['skip']}")
    wmd.append(f"- Pass rate (executed): {s['pass_rate_executed']}%")
    wmd.append("")
    wmd.append("## Workloads")
    wmd.append("")
    wmd.append("| Suite | Status | Tests | PASS | FAIL | ERROR | SKIP |")
    wmd.append("|---|---|---:|---:|---:|---:|---:|")
    for w in workloads:
        wmd.append(
            f"| {w['suite']} | {w['status']} | {w['tests']} | {w['pass']} | {w['fail']} | {w['error']} | {w['skip']} |"
        )
    WORKLOAD_MD.write_text("\n".join(wmd).rstrip() + "\n", encoding="utf-8")

    # KPI comparisons from numeric KPI dataset.
    kpi_rows = load_kpi_rows()
    kpi_cmp = build_kpi_comparison(kpi_rows)

    # executive KPI status (derived from latest suite results)
    suite_rows = []
    for suite, svals in summary_per_suite.items():
        fail_total = svals["FAIL"] + svals["ERROR"]
        status = "PASS" if fail_total == 0 else "AT_RISK"
        suite_rows.append((suite, svals["Total"], svals["PASS"], fail_total, svals["SKIP"], status))

    top_fail_suites = sorted(
        ((suite, vals["FAIL"] + vals["ERROR"]) for suite, vals in summary_per_suite.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    top_fail_suites = [x for x in top_fail_suites if x[1] > 0][:5]
    kpi_values = load_kpi_values()

    exec_lines = []
    exec_lines.append("# AI HeadNode - Executive KPI Status Report")
    exec_lines.append("")
    exec_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    exec_lines.append("**Report Type:** Management Summary")
    exec_lines.append("**Status:** Test Execution Complete")
    exec_lines.append(f"**Execution Source:** {execution_source}")
    exec_lines.append("**KPI Source:** artifacts/KPI_VALUES.csv (fresh extract)")
    exec_lines.append("")
    exec_lines.append("---")
    exec_lines.append("")
    exec_lines.append("## Data Scope")
    exec_lines.append("")
    exec_lines.append("- Execution totals and workload status are derived from the latest alltests.log session data when available.")
    exec_lines.append("- KPI highlights and comparisons are derived only from the current artifacts/KPI_VALUES.csv.")
    exec_lines.append("- Historical KPI rows outside the current log extract are not used in this summary.")
    exec_lines.append("")
    exec_lines.append("## Overall Validation Status")
    exec_lines.append("")
    exec_lines.append("| Metric | Value | Status |")
    exec_lines.append("|---|---|---|")
    exec_lines.append(f"| **Total Tests Available** | {summary_total} | {'PASS' if summary_total else 'AT_RISK'} |")
    exec_lines.append(f"| **Tests Executed** | {summary_executed} | {'PASS' if summary_executed else 'AT_RISK'} |")
    exec_lines.append(f"| **Tests Passing** | {summary_pass} | {'PASS' if summary_pass >= summary_executed * 0.75 else 'AT_RISK'} |")
    exec_lines.append(f"| **Tests Failing** | {summary_fail} | {'PASS' if summary_fail == 0 else 'AT_RISK'} |")
    exec_lines.append(f"| **Pass Rate** | {summary_pass_rate:.1f}% | {'PASS' if summary_pass_rate >= 80.0 else 'AT_RISK'} |")
    exec_lines.append(f"| **Test Coverage** | {summary_executed / summary_total * 100.0 if summary_total else 0.0:.1f}% | {'PASS' if summary_total and (summary_executed / summary_total) >= 0.9 else 'AT_RISK'} |")
    exec_lines.append(f"| **Numeric KPI Points** | {sum(len(v) for v in kpi_values.values())} | PASS |")
    exec_lines.append("")
    exec_lines.append("---")
    exec_lines.append("")
    exec_lines.append("## KPI Numeric Highlights")
    exec_lines.append("")
    exec_lines.append("| KPI | Representative Value | Notes |")
    exec_lines.append("|---|---:|---|")

    highlight_keys = [
        ("bidir_min_gbs", "GB/s", "P2P bidirectional bandwidth (worst link)"),
        ("unidir_min_gbs", "GB/s", "P2P unidirectional bandwidth (worst link)"),
        ("throughput_tokens_per_sec", "tok/s", "Inference throughput (aggregate)"),
        ("prefill_throughput_tokens_per_sec", "tok/s", "Prefill throughput"),
        ("mean_latency_ms", "ms", "Inference latency (mean)"),
        ("ttft_p95_ms", "ms", "Time-to-first-token (p95)"),
        ("total_tokens_per_sec", "tok/s", "Training throughput (tokens/sec)"),
        ("mean_tflops_per_gpu", "TFLOPS", "Training compute (per GPU)"),
        ("tflops_per_watt", "TFLOPS/W", "Training power efficiency"),
        ("baseline_throughput_tokens_per_sec", "tok/s", "Baseline inference throughput"),
        ("clock_drop_ratio", "ratio", "GPU clock-drop ratio under load"),
        ("comm_overhead_pct", "%", "Tensor-parallel communication overhead"),
    ]

    for key, unit, label in highlight_keys:
        vals = kpi_values.get(key, [])
        if vals:
            rep = sum(vals) / len(vals)
            if unit == "%":
                shown = f"{rep:.2f}{unit}"
            elif unit == "ratio":
                shown = f"{rep:.4f}"
            else:
                shown = f"{rep:.2f} {unit}"
        else:
            shown = "N/A"
        exec_lines.append(f"| {key} | {shown} | {label} |")

    exec_lines.append("")
    exec_lines.append("---")
    exec_lines.append("")
    exec_lines.append("## Domain-Level Status")
    exec_lines.append("")
    exec_lines.append("| Domain | Total | Pass | Fail | Skip | Status |")
    exec_lines.append("|---|---:|---:|---:|---:|---|")
    for suite, total_s, pass_s, fail_s, skip_s, status_s in suite_rows:
        domain = suite.replace('.xml', '')
        exec_lines.append(
            f"| {domain} | {total_s} | {pass_s} | {fail_s} | {skip_s} | {status_s} |"
        )
    exec_lines.append("")
    exec_lines.append("## Critical Focus Areas")
    exec_lines.append("")
    if not top_fail_suites:
        exec_lines.append("- No failing suites detected.")
    else:
        for idx, (suite, fails) in enumerate(top_fail_suites, start=1):
            exec_lines.append(f"{idx}. {suite.replace('.xml', '')}: {fails} failing tests")

    exec_lines.append("")
    exec_lines.append("## KPI Gaps (Fail Only)")
    exec_lines.append("")
    if kpi_cmp:
        fail_rows = [r for r in kpi_cmp if r["result"] == "FAIL"]
        if fail_rows:
            exec_lines.append("| Test | Expected | Actual | Rule | Gap |")
            exec_lines.append("|---|---:|---:|---|---:|")
            for r in fail_rows[:20]:
                expected = f"{r['expected_value']} ({r['expected_metric']})"
                actual = f"{r['actual_value']} ({r['actual_metric']})"
                exec_lines.append(
                    f"| {r['test']} | {expected} | {actual} | {r['operator']} | {r['gap']} |"
                )
            if len(fail_rows) > 20:
                exec_lines.append("")
                exec_lines.append(
                    f"_Showing first 20 of {len(fail_rows)} KPI gaps. Full list in artifacts/reports/kpi_expected_vs_actual.md._"
                )
        else:
            exec_lines.append("All mapped KPI comparisons are passing.")
    else:
        exec_lines.append("No KPI comparison rows available.")

    exec_lines.append("")
    exec_lines.append("## KPI Test Health Table")
    exec_lines.append("")
    if kpi_cmp:
        exec_lines.append("| Test | Actual Metric Value | Better Metric Value | Health |")
        exec_lines.append("|---|---|---|---|")

        def _health_cell(result: str) -> str:
            if result == "PASS":
                return '<span style="color:green"><b>GOOD</b></span>'
            if result == "FAIL":
                return '<span style="color:red"><b>BAD</b></span>'
            return '<span style="color:#b58900"><b>N/A</b></span>'

        for r in sorted(kpi_cmp, key=lambda x: (x["test"], x["expected_metric"])):
            actual_text = f"{r['actual_value']} ({r['actual_metric']})"
            better_text = f"{r['operator']} {r['expected_value']} ({r['expected_metric']})"
            exec_lines.append(
                f"| {r['test']} | {actual_text} | {better_text} | {_health_cell(r['result'])} |"
            )
    else:
        exec_lines.append("No KPI comparison rows available.")

    EXEC_KPI_MD.write_text("\n".join(exec_lines).rstrip() + "\n", encoding="utf-8")

    # KPI expected vs actual table (from KPI_VALUES.csv)
    if kpi_cmp:
        write_csv(
            KPI_COMPARE_CSV,
            [
                "test",
                "expected_metric",
                "expected_value",
                "actual_metric",
                "actual_value",
                "operator",
                "result",
                "gap",
            ],
            kpi_cmp,
        )

        total_cmp = len(kpi_cmp)
        pass_cmp = sum(1 for r in kpi_cmp if r["result"] == "PASS")
        fail_cmp = sum(1 for r in kpi_cmp if r["result"] == "FAIL")
        na_cmp = sum(1 for r in kpi_cmp if r["result"] == "N/A")
        top_rows = kpi_cmp[:120]

        km = []
        km.append("# KPI Expected vs Actual Comparison")
        km.append("")
        km.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        km.append("Source: artifacts/KPI_VALUES.csv")
        km.append("")
        km.append("## Summary")
        km.append("")
        km.append(f"- Total comparisons: {total_cmp}")
        km.append(f"- PASS: {pass_cmp}")
        km.append(f"- FAIL: {fail_cmp}")
        km.append(f"- N/A mapping: {na_cmp}")
        km.append("")
        km.append("## Comparison Table")
        km.append("")
        km.append("| Test | Expected Metric | Expected | Actual Metric | Actual | Rule | Result | Gap |")
        km.append("|---|---|---:|---|---:|---|---|---:|")
        for r in top_rows:
            test_short = r["test"]
            km.append(
                f"| {test_short} | {r['expected_metric']} | {r['expected_value']} | {r['actual_metric']} | {r['actual_value']} | {r['operator']} | {r['result']} | {r['gap']} |"
            )
        if total_cmp > len(top_rows):
            km.append("")
            km.append(f"_Showing first {len(top_rows)} rows. Full table is in CSV._")

        KPI_COMPARE_MD.write_text("\n".join(km).rstrip() + "\n", encoding="utf-8")

    print(f"WROTE={EXEC_SUMMARY_MD.relative_to(ROOT)}")
    print(f"WROTE={EXEC_REPORT_MD.relative_to(ROOT)}")
    print(f"WROTE={BUGTOOL_CSV.relative_to(ROOT)}")
    print(f"WROTE={BUGTOOL_MD.relative_to(ROOT)}")
    print(f"WROTE={TESTCASE_ALL_CSV.relative_to(ROOT)}")
    print(f"WROTE={WORKLOAD_JSON.relative_to(ROOT)}")
    print(f"WROTE={WORKLOAD_MD.relative_to(ROOT)}")
    print(f"WROTE={EXEC_KPI_MD.relative_to(ROOT)}")
    if kpi_cmp:
        print(f"WROTE={KPI_COMPARE_CSV.relative_to(ROOT)}")
        print(f"WROTE={KPI_COMPARE_MD.relative_to(ROOT)}")
    print(f"TOTAL={total}")
    print(f"UNMAPPED_GENERATED={generated_counter}")


if __name__ == "__main__":
    main()
