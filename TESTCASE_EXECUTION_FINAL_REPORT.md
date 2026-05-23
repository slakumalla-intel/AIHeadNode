# AI Head Node Overall Test Execution Report

Generated: 2026-05-23 05:33:25
Pytest command line: -v --tb=short --junitxml /root/workloads/mpudhuch/AIHeadNode/artifacts/reports/junit/mlperf_suite.xml --log-file /root/workloads/mpudhuch/AIHeadNode/artifacts/reports/logs/mlperf_suite.log --log-file-level INFO tests/test_mlperf_suite.py -k test_mlperf_case_suite
KPI source files: artifacts/node_validation_kpis.jsonl, artifacts/agentic_ai_kpis.jsonl, artifacts/performance_suite_kpis.jsonl, artifacts/vllm_prefill_kpis.jsonl, artifacts/ai_pipeline_kpis.jsonl, artifacts/system_health_kpis.jsonl

<style>
.report-scroll { overflow-x: auto; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px; }
.report-table { border-collapse: collapse; width: 100%; min-width: 1100px; table-layout: auto; font-size: 12px; }
.report-table th, .report-table td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }
.report-table th { background: #f6f8fa; position: sticky; top: 0; z-index: 1; }
.cell-wrap { white-space: normal; word-break: break-word; max-width: 420px; margin: 0 0 3px 0; }
.summary-table { border-collapse: collapse; width: 100%; max-width: 720px; }
.summary-table th, .summary-table td { border: 1px solid #d0d7de; padding: 6px 8px; }
.summary-table th { background: #f6f8fa; text-align: left; }
</style>

## Summary

- <span style="color:#1f6feb;font-weight:700">INFO</span>: Total testcases captured: 21
- <span style="color:#1f6feb;font-weight:700">INFO</span>: Tests executed (pass + fail): 1
- <span style="color:#1b8a3b;font-weight:700">PASS</span>: 0
- <span style="color:#cc1f1a;font-weight:700">FAIL</span>: 1
- <span style="color:#d97706;font-weight:700">SKIP</span>: 20
- <span style="color:#cc1f1a;font-weight:700">FAIL</span>: 1 testcase failures detected in this run.

| Metric | Count |
|---|---:|
| Testcases captured | 21 |
| Tests executed | 1 |
| Passed | 0 |
| Failed | 1 |
| Skipped | 20 |

## Domain Summary

| Domain | Testcases | Passed | Failed | Skipped | Pass Rate (Exec) |
|---|---:|---:|---:|---:|---:|
| Functional Validation | 21 | 0 | 1 | 20 | 0.0% |

## Section Summary

| Section | Testcases | Passed | Failed | Skipped | Pass Rate (Exec) |
|---|---:|---:|---:|---:|---:|
| MLPerf | 21 | 0 | 1 | 20 | 0.0% |

## Failures and Skips

| Domain | Section | TestCaseID | Test | Status | Outcome (2-3 lines) | KPI Snapshot |
|---|---|---|---|---|---|---|
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004] | <span style="color:#cc1f1a;font-weight:700">FAIL</span> | <div class="cell-wrap">Failed: Timeout (>1800.0s) from pytest-timeout.</div><div class="cell-wrap">Investigate thresholds, platform readiness, and benchmark runtime conditions.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_001] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">Filtered by ML_CASE_IDS')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_002] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">Filtered by ML_CASE_IDS')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_003: Required HuggingFace model snapshot not present in local cache (HF offline mode active)')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_005: Required HuggingFace model snapshot not present in local cache (HF offline mode active)')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_006: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_007: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_008: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_009: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_010: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_011: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_012: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_013: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_014: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_015: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_016: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_017: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_018: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_019: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_020: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |
| Functional Validation | MLPerf | - | test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021] | <span style="color:#d97706;font-weight:700">SKIP</span> | <div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_021: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div> | <div class="cell-wrap">-</div> |

## KPI Highlights

| Source | Test | KPI | Value |
|---|---|---|---|
| JSONL | test_time_to_first_token | prefill_throughput_tokens_per_sec | 85530.100 |
| JSONL | test_dataloader_cpu_to_pinned_throughput | samples_per_sec | 48118.800 |
| JSONL | test_vision_object_detection_yolo_fps | aggregate_fps_target | 200.000 |
| JSONL | test_looped_rdma_mmio_dma_stress_soak | bandwidth_gbps | 0.546 |
| JSONL | TC-AGENT-08: Latency percentiles | p95_latency_ms | 75.917 |
| JSONL | test_hf_causal_lm_inference_latency_enterprise_workloads_llama_7b | mean_latency_ms | 544.440 |

## Domain Details


### Functional Validation

<div class="report-scroll">
<table class="report-table">
<thead><tr><th style="min-width: 180px;">Section</th><th style="min-width: 230px;">TestCaseID</th><th style="min-width: 280px;">Test</th><th style="min-width: 90px;">Status</th><th style="min-width: 320px;">Outcome (2-3 lines)</th><th style="min-width: 340px;">Success / Failure KPI Snapshot</th></tr></thead>
<tbody>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]</div></td><td><span style="color:#cc1f1a;font-weight:700">FAIL</span></td><td><div class="cell-wrap">Failed: Timeout (>1800.0s) from pytest-timeout.</div><div class="cell-wrap">Investigate thresholds, platform readiness, and benchmark runtime conditions.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_001]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">Filtered by ML_CASE_IDS')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_002]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">Filtered by ML_CASE_IDS')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_003: Required HuggingFace model snapshot not present in local cache (HF offline mode active)')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_005: Required HuggingFace model snapshot not present in local cache (HF offline mode active)')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_006: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_007: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_008: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_009: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_010: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_011: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_012: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_013: Required MLPerf benchmark binary not found on PATH')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_014: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_015: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_016: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_017: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_018: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_019: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_020: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
<tr><td><div class="cell-wrap">MLPerf</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021]</div></td><td><span style="color:#d97706;font-weight:700">SKIP</span></td><td><div class="cell-wrap">AI_HOSTCPU_ML_TEST_ID_021: Required MLPerf dataset or config file not staged on this node')</div><div class="cell-wrap">Skipped by design: environment, tools, or prerequisites unavailable.</div></td><td><div class="cell-wrap">-</div></td></tr>
</tbody>
</table>
</div>
