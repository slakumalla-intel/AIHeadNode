# AI HeadNode Consolidated Validation Report

- Updated: 2026-04-27
- Source of truth: artifacts/reports/executive_summary_report.md
- Detailed execution data: artifacts/reports/test_execution_report.md
- Testcase-level evidence: artifacts/reports/bugtool_testcase_report_new.csv

## Top-Line Results

| Metric | Count |
|---|---:|
| Total mapped tests | 285 |
| Executed | 246 |
| PASS | 152 |
| FAIL | 62 |
| SKIP | 32 |
| NOT_EXECUTED | 39 |
| Pass rate (executed) | 61.8% |

## Suite-Level Summary

| Suite | Total | PASS | FAIL | SKIP | NOT_EXEC |
|---|---:|---:|---:|---:|---:|
| agentic_ai.xml | 9 | 8 | 1 | 0 | 0 |
| ai_pipeline.xml | 20 | 7 | 13 | 0 | 0 |
| bottleneck.xml | 20 | 16 | 4 | 0 | 0 |
| cpu.xml | 26 | 21 | 4 | 0 | 0 |
| gpu.xml | 21 | 19 | 2 | 0 | 0 |
| interconnect.xml | 7 | 2 | 0 | 4 | 0 |
| memory.xml | 20 | 16 | 4 | 0 | 0 |
| mlperf_suite.xml | 21 | 0 | 19 | 2 | 0 |
| node_validation.xml | 47 | 32 | 9 | 6 | 0 |
| performance_suite.xml | 25 | 10 | 0 | 15 | 0 |
| system_config.xml | 8 | 8 | 0 | 0 | 0 |
| system_health.xml | 19 | 12 | 4 | 3 | 0 |
| vllm_prefill.xml | 3 | 0 | 0 | 1 | 0 |
| not yet executed | 39 | - | - | - | 39 |

## Key Blockers

1. BLK-1 (P1): MLPerf datasets/models missing or timeout-driven failures (16+ impacted cases).
2. BLK-2 (P0): CUDA device-side assert cascade in ai_pipeline (8 failing tests).
3. BLK-3 (P0): GPU compute efficiency below threshold due to throttle/power constraints.
4. BLK-4 (P2): vLLM prefill timeout (>600s).
5. BLK-5 (P0): GPU power throttling and clock drop under load.
6. BLK-6 (P1): PCIe/NVLink topology and bandwidth issues.
7. BLK-7 (P2): RAM below 1 TiB target (1007 GB observed).

## Recommended Action Priority

| Priority | Action | Expected Recovery |
|---|---|---|
| P0 | Resolve BLK-5 GPU power throttling | Improves BLK-3 and BLK-4 outcomes |
| P0 | Fix BLK-2 CUDA assert root failure | Clears major ai_pipeline cascade |
| P1 | Resolve BLK-1 MLPerf dataset/model readiness | Restores mlperf suite execution |
| P1 | Resolve BLK-6 PCIe/NVLink issues | Recovers interconnect and P2P failures |
| P2 | Tune BLK-4 timeout after power fix | Stabilizes vLLM prefill run |
| P2 | Validate BLK-7 DIMM population | Restores memory capacity target |

## Notes

- This file is intentionally concise and executive-oriented.
- For per-test failure details, use artifacts/reports/test_execution_report.md.
- For bugtool import and testcase evidence, use artifacts/reports/bugtool_testcase_report_new.csv.
