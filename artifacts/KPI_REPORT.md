# KPI Report

Generated (UTC): 2026-05-21T02:05:09.357242+00:00
Source: top-level artifacts/*_kpis.jsonl with latest entry per test case

## KPI Coverage

| KPI File | Records | Latest Distinct Tests | Latest Timestamp | Numeric Fields | Threshold Checks | Pass | Fail |
|---|---:|---:|---|---:|---:|---:|---:|
| agentic_ai_kpis.jsonl | 48 | 8 | 2026-05-21T01:37:27 | 27 | 0 | 0 | 0 |
| ai_pipeline_kpis.jsonl | 94 | 16 | 2026-05-21T01:38:02 | 74 | 17 | 16 | 1 |
| bottleneck_kpis.jsonl | 9 | 1 | 2026-05-20T23:25:14 | 0 | 0 | 0 | 0 |
| cpu_validation_kpis.jsonl | 175 | 25 | 2026-05-20T23:29:30 | 49 | 14 | 14 | 0 |
| gpu_validation_kpis.jsonl | 71 | 24 | 2026-05-20T23:30:06 | 9 | 0 | 0 | 0 |
| interconnect_kpis.jsonl | 42 | 14 | 2026-05-20T23:30:27 | 5 | 0 | 0 | 0 |
| memory_validation_kpis.jsonl | 7 | 1 | 2026-05-20T23:29:41 | 2 | 0 | 0 | 0 |
| mlperf_suite_kpis.jsonl | 158 | 1 | 2026-05-20T13:55:25 | 0 | 0 | 0 | 0 |
| node_validation_kpis.jsonl | 180 | 41 | 2026-05-21T00:54:32 | 161 | 10 | 10 | 0 |
| performance_suite_kpis.jsonl | 404 | 1 | 2026-05-21T01:22:57 | 0 | 0 | 0 | 0 |
| system_health_kpis.jsonl | 92 | 36 | 2026-05-21T01:38:55+00:00 | 60 | 7 | 7 | 0 |

## Threshold Check Totals

- Total threshold checks: 48
- Passed checks: 47
- Failed checks: 1

## Failed KPI Threshold Checks

| Test | Metric | Actual | Threshold |
|---|---|---:|---:|
| test_tensor_parallel_allreduce_overhead_fraction | comm_overhead_pct | 98.6500 | 20.0000 |

## Correlation With Test Outcomes

- Workload test failures (from status report): 13
- KPI threshold failures (latest metric rows): 1
- Note: A domain can fail from runtime errors even if threshold checks are limited.
