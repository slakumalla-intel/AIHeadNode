#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_industry_kpi_report_from_log import (
    extract_threshold_records,
    parse_measured_kpis,
    parse_sessions,
    parse_workload_kpis,
)


LEFT_LOG = ROOT / "alltests.log"
RIGHT_LOG = ROOT / "gnr_alltests.log"
REPORT_FILE = ROOT / "artifacts" / "reports" / "alltests_vs_gnr_alltests_comparison.md"
CSV_FILE = ROOT / "artifacts" / "reports" / "alltests_vs_gnr_alltests_comparison.csv"

DMR_LABEL = "DMR"
GNR_LABEL = "GNR"


def _escape_cell(value: object) -> str:
    return html.escape(str(value), quote=False).replace("|", "&#124;")


def _fmt_num(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _pct_delta(left: float, right: float) -> float:
    if left == 0.0:
        return 0.0 if right == 0.0 else math.inf
    return abs(right - left) / abs(left) * 100.0


def _signed_delta(left: float, right: float) -> float:
    return right - left


def _status_from_values(dmr: float, gnr: float, higher_is_better: bool) -> str:
    if dmr == gnr:
        return '<span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span>'

    if higher_is_better:
        if dmr < gnr:
            return '<span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span>'
        if gnr != 0 and ((dmr - gnr) / abs(gnr) * 100.0) > 5.0:
            return '<span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span>'
        return '<span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span>'

    if dmr > gnr:
        return '<span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span>'
    if gnr != 0 and ((gnr - dmr) / abs(gnr) * 100.0) > 5.0:
        return '<span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span>'
    return '<span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span>'


def _log_summary(name: str, sessions: list[dict[str, object]], perf: list[dict[str, object]], thresh: list[dict[str, object]], workload: list[dict[str, object]]) -> dict[str, object]:
    collected = sum(int(s.get("collected", 0) or 0) for s in sessions)
    passed = sum(int(s.get("passed", 0) or 0) for s in sessions)
    failed = sum(int(s.get("failed", 0) or 0) for s in sessions)
    skipped = sum(int(s.get("skipped", 0) or 0) for s in sessions)
    executed = passed + failed
    pass_rate = (passed / executed * 100.0) if executed else 0.0
    return {
        "name": name,
        "sessions": len(sessions),
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "executed": executed,
        "pass_rate": pass_rate,
        "perf_count": len(perf),
        "threshold_count": len(thresh),
        "workload_count": len(workload),
    }


def _build_perf_rows(left_rows: list[dict[str, object]], right_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    left_map = {(str(r["domain"]), str(r["metric"])): r for r in left_rows}
    right_map = {(str(r["domain"]), str(r["metric"])): r for r in right_rows}
    rows: list[dict[str, object]] = []
    for key in sorted(left_map.keys() & right_map.keys()):
        left = left_map[key]
        right = right_map[key]
        left_eff = float(left["efficiency"])
        right_eff = float(right["efficiency"])
        delta = _signed_delta(left_eff, right_eff)
        delta_pct = _pct_delta(left_eff, right_eff)
        verdict = _status_from_values(left_eff, right_eff, higher_is_better=True)
        rows.append(
            {
                "category": "Performance",
                "domain": key[0],
                "key": key[1],
                "left_value": float(left.get("measured", 0.0)),
                "right_value": float(right.get("measured", 0.0)),
                "left_eff": left_eff,
                "right_eff": right_eff,
                "delta": delta,
                "delta_pct": delta_pct,
                "verdict": verdict,
            }
        )
    return rows


def _build_threshold_rows(left_rows: list[dict[str, object]], right_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def make_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
        return (
            str(row["domain"]),
            str(row["test"]),
            str(row["actual_metric"]),
            str(row["threshold_metric"]),
            str(row["rule"]),
        )

    left_map = {make_key(r): r for r in left_rows}
    right_map = {make_key(r): r for r in right_rows}
    rows: list[dict[str, object]] = []
    for key in sorted(left_map.keys() & right_map.keys()):
        left = left_map[key]
        right = right_map[key]
        left_actual = float(left["actual_value"])
        right_actual = float(right["actual_value"])
        left_threshold = float(left["threshold_value"])
        right_threshold = float(right["threshold_value"])
        delta = _signed_delta(left_actual, right_actual)
        delta_pct = _pct_delta(left_actual, right_actual)
        verdict = _status_from_values(left_actual, right_actual, higher_is_better=(key[4] != "<="))
        rows.append(
            {
                "category": "Threshold",
                "domain": key[0],
                "key": key[1],
                "metric": key[2],
                "threshold_metric": key[3],
                "rule": key[4],
                "left_value": left_actual,
                "right_value": right_actual,
                "left_threshold": left_threshold,
                "right_threshold": right_threshold,
                "delta": delta,
                "delta_pct": delta_pct,
                "verdict": verdict,
            }
        )
    return rows


def _render_table(headers: list[str], rows: list[dict[str, object]], row_builder) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append(row_builder(row))
    return lines


def _perf_row(row: dict[str, object]) -> str:
    return (
        f"| {_escape_cell(row['domain'])} | {_escape_cell(row['key'])} | "
        f"{_fmt_num(row['left_value'])} | {_fmt_num(row['right_value'])} | "
        f"{_fmt_num(row['left_eff'])}% | {_fmt_num(row['right_eff'])}% | "
        f"{row['verdict']} |"
    )


def _threshold_row(row: dict[str, object]) -> str:
    return (
        f"| {_escape_cell(row['domain'])} | {_escape_cell(row['key'])} | {_escape_cell(row['metric'])} | {_escape_cell(row['rule'])} | "
        f"{_fmt_num(row['left_value'])} | {_fmt_num(row['right_value'])} | "
        f"{_fmt_num(row['left_threshold'])} | {_fmt_num(row['right_threshold'])} | "
        f"{row['verdict']} |"
    )


def _write_csv(perf_rows: list[dict[str, object]], threshold_rows: list[dict[str, object]]) -> None:
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    target_csv = CSV_FILE
    try:
        target_csv.open("w", newline="", encoding="utf-8").close()
    except PermissionError:
        target_csv = CSV_FILE.with_name(CSV_FILE.stem + "_latest.csv")

    with target_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "domain",
                "key",
                "metric",
                "rule",
                "left_value",
                "right_value",
                "left_threshold",
                "right_threshold",
                "delta",
                "delta_pct",
                "verdict",
            ],
        )
        writer.writeheader()
        for row in perf_rows:
            writer.writerow(
                {
                    "category": row["category"],
                    "domain": row["domain"],
                    "key": row["key"],
                    "metric": "efficiency",
                    "rule": "higher is better",
                    "left_value": row["left_value"],
                    "right_value": row["right_value"],
                    "left_threshold": "",
                    "right_threshold": "",
                    "delta": row["delta"],
                    "delta_pct": row["delta_pct"],
                    "verdict": row["verdict"],
                }
            )
        for row in threshold_rows:
            writer.writerow(
                {
                    "category": row["category"],
                    "domain": row["domain"],
                    "key": row["key"],
                    "metric": row["metric"],
                    "rule": row["rule"],
                    "left_value": row["left_value"],
                    "right_value": row["right_value"],
                    "left_threshold": row["left_threshold"],
                    "right_threshold": row["right_threshold"],
                    "delta": row["delta"],
                    "delta_pct": row["delta_pct"],
                    "verdict": row["verdict"],
                }
            )
    print(f"WROTE={target_csv.relative_to(ROOT)}")


def _render_report(left_summary: dict[str, object], right_summary: dict[str, object], perf_rows: list[dict[str, object]], threshold_rows: list[dict[str, object]]) -> str:
    # Filter rows by verdict status
    perf_red = [r for r in perf_rows if "RED" in str(r["verdict"])]
    perf_matching = [r for r in perf_rows if "MATCHING" in str(r["verdict"])]
    perf_exceeds = [r for r in perf_rows if "EXCEEDS" in str(r["verdict"])]
    threshold_red = [r for r in threshold_rows if "RED" in str(r["verdict"])]
    threshold_matching = [r for r in threshold_rows if "MATCHING" in str(r["verdict"])]
    threshold_exceeds = [r for r in threshold_rows if "EXCEEDS" in str(r["verdict"])]

    perf_red_count = len(perf_red)
    perf_matching_count = len(perf_matching)
    perf_exceeds_count = len(perf_exceeds)
    threshold_red_count = len(threshold_red)
    threshold_matching_count = len(threshold_matching)
    threshold_exceeds_count = len(threshold_exceeds)

    lines: list[str] = []
    lines.append("# Log Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sources: {_escape_cell(LEFT_LOG.name)} vs {_escape_cell(RIGHT_LOG.name)}")
    lines.append("Scope: side-by-side comparison of common pytest session data and KPI records")
    lines.append("")
    lines.append("## Execution Summary")
    lines.append("")
    lines.append("| Source | Sessions | Collected | Executed | Passed | Failed | Skipped | Pass Rate | Performance KPIs | Threshold KPIs | Workload Payloads |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for summary in (left_summary, right_summary):
        lines.append(
            f"| {_escape_cell(summary['name'])} | {summary['sessions']} | {summary['collected']} | {summary['executed']} | {summary['passed']} | {summary['failed']} | {summary['skipped']} | {summary['pass_rate']:.1f}% | {summary['perf_count']} | {summary['threshold_count']} | {summary['workload_count']} |"
        )
    lines.append("")
    lines.append("## Comparison Summary")
    lines.append("")
    lines.append("| Category | RED | MATCHING (light green) | EXCEEDS (dark green) | Common Rows |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| Performance | {perf_red_count} | {perf_matching_count} | {perf_exceeds_count} | {len(perf_rows)} |")
    lines.append(f"| Threshold | {threshold_red_count} | {threshold_matching_count} | {threshold_exceeds_count} | {len(threshold_rows)} |")
    lines.append("")

    # RED Results Section
    lines.append("## RED Results (DMR Underperformed)")
    lines.append("")
    if perf_red or threshold_red:
        if perf_red:
            lines.append("### Performance Metrics (RED)")
            lines.append("")
            lines.append(f"| Domain | KPI | {DMR_LABEL} measured | {GNR_LABEL} measured | {DMR_LABEL} eff | {GNR_LABEL} eff | Match |")
            lines.append("|---|---|---:|---:|---:|---:|---|")
            for row in perf_red:
                lines.append(_perf_row(row))
            lines.append("")
        if threshold_red:
            lines.append("### Threshold Metrics (RED)")
            lines.append("")
            lines.append(f"| Domain | Test | Metric | Rule | {DMR_LABEL} measured | {GNR_LABEL} measured | {DMR_LABEL} threshold | {GNR_LABEL} threshold | Match |")
            lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
            for row in threshold_red:
                lines.append(_threshold_row(row))
            lines.append("")
    else:
        lines.append("*No RED results found.*")
        lines.append("")

    # MATCHING Results Section
    lines.append("## MATCHING Results (Within 5% Parity)")
    lines.append("")
    if perf_matching or threshold_matching:
        if perf_matching:
            lines.append("### Performance Metrics (MATCHING)")
            lines.append("")
            lines.append(f"| Domain | KPI | {DMR_LABEL} measured | {GNR_LABEL} measured | {DMR_LABEL} eff | {GNR_LABEL} eff | Match |")
            lines.append("|---|---|---:|---:|---:|---:|---|")
            for row in perf_matching:
                lines.append(_perf_row(row))
            lines.append("")
        if threshold_matching:
            lines.append("### Threshold Metrics (MATCHING)")
            lines.append("")
            lines.append(f"| Domain | Test | Metric | Rule | {DMR_LABEL} measured | {GNR_LABEL} measured | {DMR_LABEL} threshold | {GNR_LABEL} threshold | Match |")
            lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
            for row in threshold_matching:
                lines.append(_threshold_row(row))
            lines.append("")
    else:
        lines.append("*No MATCHING results found.*")
        lines.append("")

    # EXCEEDS Results Section
    lines.append("## EXCEEDS Results (DMR Outperformed by >5%)")
    lines.append("")
    if perf_exceeds or threshold_exceeds:
        if perf_exceeds:
            lines.append("### Performance Metrics (EXCEEDS)")
            lines.append("")
            lines.append(f"| Domain | KPI | {DMR_LABEL} measured | {GNR_LABEL} measured | {DMR_LABEL} eff | {GNR_LABEL} eff | Match |")
            lines.append("|---|---|---:|---:|---:|---:|---|")
            for row in perf_exceeds:
                lines.append(_perf_row(row))
            lines.append("")
        if threshold_exceeds:
            lines.append("### Threshold Metrics (EXCEEDS)")
            lines.append("")
            lines.append(f"| Domain | Test | Metric | Rule | {DMR_LABEL} measured | {GNR_LABEL} measured | {DMR_LABEL} threshold | {GNR_LABEL} threshold | Match |")
            lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
            for row in threshold_exceeds:
                lines.append(_threshold_row(row))
            lines.append("")
    else:
        lines.append("*No EXCEEDS results found.*")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(f"- RED = {DMR_LABEL} is worse than {GNR_LABEL} for that metric/rule.")
    lines.append(f"- MATCHING (light green) = {DMR_LABEL} is approximately at parity with {GNR_LABEL} (≤5% difference).")
    lines.append(f"- EXCEEDS (dark green) = {DMR_LABEL} performs better than {GNR_LABEL} by more than 5%.")
    lines.append("- The CSV companion contains the full comparable row set for downstream filtering.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    left_lines = LEFT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    right_lines = RIGHT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()

    left_sessions = parse_sessions(left_lines)
    right_sessions = parse_sessions(right_lines)
    left_perf = parse_measured_kpis(left_lines, left_sessions)
    right_perf = parse_measured_kpis(right_lines, right_sessions)
    left_threshold = extract_threshold_records(left_lines)
    right_threshold = extract_threshold_records(right_lines)
    left_workload = parse_workload_kpis(left_lines)
    right_workload = parse_workload_kpis(right_lines)

    left_summary = _log_summary(LEFT_LOG.name, left_sessions, left_perf, left_threshold, left_workload)
    right_summary = _log_summary(RIGHT_LOG.name, right_sessions, right_perf, right_threshold, right_workload)
    perf_rows = _build_perf_rows(left_perf, right_perf)
    threshold_rows = _build_threshold_rows(left_threshold, right_threshold)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(_render_report(left_summary, right_summary, perf_rows, threshold_rows), encoding="utf-8")
    _write_csv(perf_rows, threshold_rows)

    print(f"WROTE={REPORT_FILE.relative_to(ROOT)}")
    print(f"LEFT_PERF={len(left_perf)} RIGHT_PERF={len(right_perf)} COMMON_PERF={len(perf_rows)}")
    print(f"LEFT_THRESHOLD={len(left_threshold)} RIGHT_THRESHOLD={len(right_threshold)} COMMON_THRESHOLD={len(threshold_rows)}")


if __name__ == "__main__":
    main()