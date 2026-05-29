# Workload Execution Status Report

Generated (UTC): 2026-05-29T06:14:22.635998+00:00
Host: sc00901168s0070

## Summary

- Domains run: 14
- Passed: 2
- Failed: 1
- Total tests: 69
- Failures: 2
- Errors: 0
- Skipped: 6

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log | KPI Table |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| system_config | PASS | 8 | 0 | 0 | 5 | 2.166 | artifacts/reports/logs/full_logs/system_config_20260529_061137_933755.xml | artifacts/reports/logs/full_logs/system_config_20260529_061137_933755.log | - |
| cpu | FAIL | 33 | 2 | 0 | 1 | 134.695 | artifacts/reports/logs/full_logs/cpu_20260529_061137_933755.xml | artifacts/reports/logs/full_logs/cpu_20260529_061137_933755.log | - |
| memory | PASS | 28 | 0 | 0 | 0 | 23.048 | artifacts/reports/logs/full_logs/memory_20260529_061137_933755.xml | artifacts/reports/logs/full_logs/memory_20260529_061137_933755.log | - |
| bottleneck | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| gpu | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| interconnect | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| system_health | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| cpu_headnode | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| agentic_ai | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| ai_pipeline | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| vllm_prefill | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| node_validation | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| performance_suite | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |
| mlperf_suite | BLOCKED | 0 | 0 | 0 | 0 | 0.0 | - | - | - |

## Failure Details

### cpu

- [failure] tests.test_cpu_validation.TestMemoryBandwidth::test_single_thread_stream_bandwidth: AssertionError: Single-thread memory BW 7.32 GB/s < 30 GB/s
assert 7.316549947344709 >= 30.0
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[16]: AssertionError: Parallel speedup at 16 cores: efficiency 79.04% is below threshold 85.00% (measured=580.860, theoretical=734.897)
assert 0.7903953773550404 >= 0.85

### bottleneck

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### gpu

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### interconnect

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### system_health

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### cpu_headnode

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### agentic_ai

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### ai_pipeline

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### vllm_prefill

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### node_validation

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### performance_suite

- Domain failed without testcase-level failure details. Check log and XML artifacts.

### mlperf_suite

- Domain failed without testcase-level failure details. Check log and XML artifacts.
