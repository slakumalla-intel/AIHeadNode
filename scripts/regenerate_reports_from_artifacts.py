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


def main() -> None:
    junit_rows, per_suite = parse_junit()
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

    # executive summary
    lines = []
    lines.append("# AI HeadNode Validation - Executive Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("- Source: artifacts/reports/junit/*.xml")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top-Line Results")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Total mapped tests | {total} |")
    lines.append(f"| Executed | {executed} |")
    lines.append(f"| PASS | {status_counts.get('PASS', 0)} |")
    lines.append(f"| FAIL | {status_counts.get('FAIL', 0)} |")
    lines.append(f"| ERROR | {status_counts.get('ERROR', 0)} |")
    lines.append(f"| SKIP | {status_counts.get('SKIP', 0)} |")
    lines.append("| NOT_EXECUTED | 0 |")
    lines.append(f"| Pass rate (executed) | {pass_rate:.1f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Suite-Level Summary")
    lines.append("")
    lines.append("| Suite | Total | PASS | FAIL | SKIP | NOT_EXEC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for suite, s in per_suite.items():
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
    for suite, s in per_suite.items():
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
        "source": "artifacts/reports/junit/*.xml",
        "summary": {
            "total": total,
            "executed": executed,
            "pass": status_counts.get("PASS", 0),
            "fail": status_counts.get("FAIL", 0),
            "error": status_counts.get("ERROR", 0),
            "skip": status_counts.get("SKIP", 0),
            "pass_rate_executed": round(pass_rate, 2),
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

    # executive KPI status (derived from latest suite results)
    suite_rows = []
    for suite, svals in per_suite.items():
        fail_total = svals["FAIL"] + svals["ERROR"]
        status = "PASS" if fail_total == 0 else "AT_RISK"
        suite_rows.append((suite, svals["Total"], svals["PASS"], fail_total, svals["SKIP"], status))

    top_fail_suites = sorted(
        ((suite, vals["FAIL"] + vals["ERROR"]) for suite, vals in per_suite.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    top_fail_suites = [x for x in top_fail_suites if x[1] > 0][:5]

    exec_lines = []
    exec_lines.append("# AI HeadNode - Executive KPI Status Report")
    exec_lines.append("")
    exec_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    exec_lines.append("**Report Type:** Management Summary")
    exec_lines.append("**Status:** Test Execution Complete")
    exec_lines.append("")
    exec_lines.append("---")
    exec_lines.append("")
    exec_lines.append("## Overall Validation Status")
    exec_lines.append("")
    exec_lines.append("| Metric | Value | Status |")
    exec_lines.append("|---|---|---|")
    exec_lines.append(f"| **Total Tests Available** | {total} | {'PASS' if total else 'AT_RISK'} |")
    exec_lines.append(f"| **Tests Executed** | {executed} | {'PASS' if executed else 'AT_RISK'} |")
    exec_lines.append(f"| **Tests Passing** | {status_counts.get('PASS', 0)} | {'PASS' if status_counts.get('PASS', 0) >= executed * 0.75 else 'AT_RISK'} |")
    exec_lines.append(f"| **Tests Failing** | {status_counts.get('FAIL', 0)} | {'PASS' if status_counts.get('FAIL', 0) == 0 else 'AT_RISK'} |")
    exec_lines.append(f"| **Pass Rate** | {pass_rate:.1f}% | {'PASS' if pass_rate >= 80.0 else 'AT_RISK'} |")
    exec_lines.append(f"| **Test Coverage** | {executed / total * 100.0 if total else 0.0:.1f}% | {'PASS' if total and (executed / total) >= 0.9 else 'AT_RISK'} |")
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

    EXEC_KPI_MD.write_text("\n".join(exec_lines).rstrip() + "\n", encoding="utf-8")

    print(f"WROTE={EXEC_SUMMARY_MD.relative_to(ROOT)}")
    print(f"WROTE={EXEC_REPORT_MD.relative_to(ROOT)}")
    print(f"WROTE={BUGTOOL_CSV.relative_to(ROOT)}")
    print(f"WROTE={BUGTOOL_MD.relative_to(ROOT)}")
    print(f"WROTE={TESTCASE_ALL_CSV.relative_to(ROOT)}")
    print(f"WROTE={WORKLOAD_JSON.relative_to(ROOT)}")
    print(f"WROTE={WORKLOAD_MD.relative_to(ROOT)}")
    print(f"WROTE={EXEC_KPI_MD.relative_to(ROOT)}")
    print(f"TOTAL={total}")
    print(f"UNMAPPED_GENERATED={generated_counter}")


if __name__ == "__main__":
    main()
