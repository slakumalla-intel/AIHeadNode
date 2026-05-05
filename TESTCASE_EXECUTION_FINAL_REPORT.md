# AI HeadNode Consolidated Validation Report

- Updated: 2026-05-05
- Source of truth: artifacts/reports/executive_summary_report.md
- Detailed execution data: artifacts/reports/test_execution_report.md
- Testcase-level evidence: artifacts/reports/bugtool_testcase_report_new.csv

## Top-Line Results

| Metric | Count |
|---|---:|
| Total tests available for execution | 410 |
| Executed | 375 |
| PASS | 303 |
| FAIL | 72 |
| SKIP | 35 |
| NOT_EXECUTED | 0 |
| Pass rate (executed) | 80.8% |

## Suite-Level Summary

| Suite | Total | PASS | FAIL | ERROR | SKIP | NOT_EXECUTED |
|---|---:|---:|---:|---:|---:|---:|
| agentic_ai.xml | 9 | 8 | 1 | 0 | 0 | 0 |
| ai_pipeline.xml | 38 | 20 | 17 | 0 | 1 | 0 |
| bottleneck.xml | 25 | 24 | 1 | 0 | 0 | 0 |
| cpu.xml | 33 | 31 | 1 | 0 | 1 | 0 |
| gpu.xml | 53 | 44 | 9 | 0 | 0 | 0 |
| interconnect.xml | 77 | 72 | 1 | 0 | 4 | 0 |
| memory.xml | 28 | 21 | 7 | 0 | 0 | 0 |
| mlperf_suite.xml | 21 | 0 | 19 | 0 | 2 | 0 |
| node_validation.xml | 51 | 36 | 7 | 0 | 8 | 0 |
| performance_suite.xml | 25 | 9 | 1 | 0 | 15 | 0 |
| system_config.xml | 8 | 8 | 0 | 0 | 0 | 0 |
| system_health.xml | 39 | 28 | 8 | 0 | 3 | 0 |
| vllm_prefill.xml | 3 | 2 | 0 | 0 | 1 | 0 |

## Notes

- This file is intentionally concise and executive-oriented.
- For per-test failure details, use artifacts/reports/test_execution_report.md.
- For bugtool import and testcase evidence, use artifacts/reports/bugtool_testcase_report_new.csv.

## Notes

- This file is intentionally concise and executive-oriented.
- For per-test failure details, use artifacts/reports/test_execution_report.md.
- For bugtool import and testcase evidence, use artifacts/reports/bugtool_testcase_report_new.csv.
