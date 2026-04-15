# AI Head Node Overall Test Execution Report

Generated: 2026-04-15 04:44:59
Pytest command line: -m bottleneck -vv tests/test_bottleneck_analysis.py
KPI source files: artifacts/node_validation_kpis.jsonl, artifacts/performance_suite_kpis.jsonl

<style>
.report-scroll { overflow-x: auto; resize: horizontal; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px; }
.report-table { border-collapse: collapse; width: 100%; min-width: 1400px; table-layout: auto; font-size: 12px; }
.report-table th, .report-table td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }
.report-table th { background: #f6f8fa; position: sticky; top: 0; z-index: 1; }
.cell-wrap { white-space: normal; word-break: break-word; max-width: 540px; }
.summary-table { border-collapse: collapse; width: 100%; max-width: 720px; }
.summary-table th, .summary-table td { border: 1px solid #d0d7de; padding: 6px 8px; }
.summary-table th { background: #f6f8fa; text-align: left; }
</style>

## Summary

- <span style="color:#1f6feb;font-weight:700">INFO</span>: Total testcases captured: 25
- <span style="color:#1f6feb;font-weight:700">INFO</span>: Tests executed (pass + fail): 25
- <span style="color:#1b8a3b;font-weight:700">PASS</span>: 21
- <span style="color:#cc1f1a;font-weight:700">FAIL</span>: 4
- <span style="color:#d97706;font-weight:700">SKIP</span>: 0
- <span style="color:#cc1f1a;font-weight:700">FAIL</span>: Attention required - one or more testcase failures were detected.

| Metric | Count |
|---|---:|
| Testcases captured | 25 |
| Tests executed | 25 |
| Passed | 21 |
| Failed | 4 |
| Skipped | 0 |

## Domain Summary

| Domain | Testcases | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Functional Validation | 25 | 21 | 4 | 0 |
## Brief Issues

| Domain | Section | Test | Status | Issue |
|---|---|---|---|---|
| Functional Validation | Bottleneck | test_cpu_fp32_efficiency_report | <span style="color:#cc1f1a;font-weight:700">FAIL</span> | AssertionError: CPU FP32 (1 process, BLAS threads=224): efficiency 3.25% is below threshold 70.00% (measured=0.884, theoretical=27.238) assert 0.032464112676673144 >= 0.7 |
| Functional Validation | Bottleneck | test_aggregate_4gpu_fp16_tflops | <span style="color:#cc1f1a;font-weight:700">FAIL</span> | AssertionError: 4-GPU aggregate FP16 TFLOPS: efficiency 22.85% is below threshold 75.00% (measured=1808.734, theoretical=7916.000) assert 0.22849090879967332 >= 0.75 |
| Functional Validation | Bottleneck | test_per_gpu_fp16_efficiency | <span style="color:#cc1f1a;font-weight:700">FAIL</span> | AssertionError: GPU 0 FP16 Tensor Core: efficiency 23.65% is below threshold 75.00% (measured=468.105, theoretical=1979.000) assert 0.23653628371996754 >= 0.75 |
| Functional Validation | Bottleneck | test_hbm_bandwidth_per_gpu_vs_theoretical | <span style="color:#cc1f1a;font-weight:700">FAIL</span> | AssertionError: GPU 0 HBM3 bandwidth: efficiency 55.22% is below threshold 80.00% (measured=1.850, theoretical=3.350) assert 0.5521712460068665 >= 0.8 |

## Section Summary

| Section | Testcases | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Workload Acceptance | 0 | 0 | 0 | 0 |
| Node Validation | 0 | 0 | 0 | 0 |
| Bottleneck | 25 | 21 | 4 | 0 |
| Performance | 0 | 0 | 0 | 0 |
| MLPerf | 0 | 0 | 0 | 0 |
| Other | 0 | 0 | 0 | 0 |

## Top 5 KPIs

- No numeric KPI values available for this run.

## KPI Summary

- No KPI records found for this run. Ensure workload tests called KPI recorder.

## Workload Acceptance

- No tests captured in this section for the current run.

## Node Validation

- No tests captured in this section for the current run.

## Bottleneck

<div class="report-scroll">
<table class="report-table">
<thead><tr><th style="min-width: 380px;">Test</th><th style="min-width: 360px;">Purpose</th><th style="min-width: 100px;">Status</th><th style="min-width: 360px;">Result</th><th style="min-width: 300px;">Analysis</th><th style="min-width: 380px;">KPI</th><th style="min-width: 320px;">Reason / Success Case / Issue</th></tr></thead>
<tbody>
<tr><td><div class="cell-wrap">test_cpu_fp32_efficiency_report</div></td><td><div class="cell-wrap">TC-BN-01: CPU FP32 TFLOPS via one process + threaded BLAS vs CPU theoretical.</div></td><td><span style="color:#cc1f1a;font-weight:700">FAIL</span></td><td><div class="cell-wrap"><span style="color:#cc1f1a;font-weight:700">FAILED: AssertionError: CPU FP32 (1 process, BLAS threads=224): efficiency 3.25% is below threshold 70.00% (measured=0.884, theoretical=27.238) assert 0.032464112676673144 >= 0.7</span></div></td><td><div class="cell-wrap">Investigate thresholds, platform readiness, and benchmark runtime conditions.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">AssertionError: CPU FP32 (1 process, BLAS threads=224): efficiency 3.25% is below threshold 70.00% (measured=0.884, theoretical=27.238) assert 0.032464112676673144 >= 0.7</div></td></tr>
<tr><td><div class="cell-wrap">test_aggregate_4gpu_fp16_tflops</div></td><td><div class="cell-wrap">TC-BN-03b: Aggregate 4-GPU FP16 TFLOPS ≥ 75 % of 4 × 1979 TFLOPS.</div></td><td><span style="color:#cc1f1a;font-weight:700">FAIL</span></td><td><div class="cell-wrap"><span style="color:#cc1f1a;font-weight:700">FAILED: AssertionError: 4-GPU aggregate FP16 TFLOPS: efficiency 22.85% is below threshold 75.00% (measured=1808.734, theoretical=7916.000) assert 0.22849090879967332 >= 0.75</span></div></td><td><div class="cell-wrap">Investigate thresholds, platform readiness, and benchmark runtime conditions.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">AssertionError: 4-GPU aggregate FP16 TFLOPS: efficiency 22.85% is below threshold 75.00% (measured=1808.734, theoretical=7916.000) assert 0.22849090879967332 >= 0.75</div></td></tr>
<tr><td><div class="cell-wrap">test_per_gpu_fp16_efficiency</div></td><td><div class="cell-wrap">TC-BN-03a: FP16 efficiency per GPU ≥ 75 % of 1979 TFLOPS theoretical.</div></td><td><span style="color:#cc1f1a;font-weight:700">FAIL</span></td><td><div class="cell-wrap"><span style="color:#cc1f1a;font-weight:700">FAILED: AssertionError: GPU 0 FP16 Tensor Core: efficiency 23.65% is below threshold 75.00% (measured=468.105, theoretical=1979.000) assert 0.23653628371996754 >= 0.75</span></div></td><td><div class="cell-wrap">Investigate thresholds, platform readiness, and benchmark runtime conditions.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">AssertionError: GPU 0 FP16 Tensor Core: efficiency 23.65% is below threshold 75.00% (measured=468.105, theoretical=1979.000) assert 0.23653628371996754 >= 0.75</div></td></tr>
<tr><td><div class="cell-wrap">test_hbm_bandwidth_per_gpu_vs_theoretical</div></td><td><div class="cell-wrap">TC-BN-04a: HBM3 BW per GPU ≥ 80 % of theoretical 3.35 TB/s.</div></td><td><span style="color:#cc1f1a;font-weight:700">FAIL</span></td><td><div class="cell-wrap"><span style="color:#cc1f1a;font-weight:700">FAILED: AssertionError: GPU 0 HBM3 bandwidth: efficiency 55.22% is below threshold 80.00% (measured=1.850, theoretical=3.350) assert 0.5521712460068665 >= 0.8</span></div></td><td><div class="cell-wrap">Investigate thresholds, platform readiness, and benchmark runtime conditions.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">AssertionError: GPU 0 HBM3 bandwidth: efficiency 55.22% is below threshold 80.00% (measured=1.850, theoretical=3.350) assert 0.5521712460068665 >= 0.8</div></td></tr>
<tr><td><div class="cell-wrap">test_data_preprocessing_not_bottleneck</div></td><td><div class="cell-wrap">TC-BN-08: CPU data-prep time must be < GPU compute time for same data volume.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (1.109s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_cpu_dram_bandwidth_measured</div></td><td><div class="cell-wrap">TC-BN-02b: Measured CPU DRAM BW ≥ 75 % of theoretical peak.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (5.142s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_cpu_dram_bandwidth_vs_hbm</div></td><td><div class="cell-wrap">TC-BN-02: CPU DRAM BW must be << HBM3 BW (confirms memory bottleneck topology).</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.001s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_hbm_is_limiting_for_memory_bound_kernels</div></td><td><div class="cell-wrap">TC-BN-04b: FP16 GEMM at low arithmetic intensity must be HBM-limited (not compute).</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.057s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_scaling_efficiency_2_vs_1_gpu</div></td><td><div class="cell-wrap">TC-BN-09a: 2-GPU throughput ≥ 80 % of 2× single-GPU throughput.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.032s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_scaling_efficiency_4_vs_1_gpu</div></td><td><div class="cell-wrap">TC-BN-09b: N-GPU throughput ≥ 80 % of N× single-GPU throughput.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.034s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_nvlink_bw_vs_theoretical</div></td><td><div class="cell-wrap">TC-BN-06a: Inter-GPU P2P BW must meet threshold for active fabric.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.048s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_nvlink_faster_than_pcie_bottleneck</div></td><td><div class="cell-wrap">TC-BN-06b: Validate expected relationship between inter-GPU fabric and host PCIe BW.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_pcie_bottleneck_ratio_vs_hbm</div></td><td><div class="cell-wrap">TC-BN-05b: PCIe H2D BW must be << HBM BW (confirms I/O bottleneck).</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_pcie_h2d_bandwidth_vs_theoretical</div></td><td><div class="cell-wrap">TC-BN-05a: H2D PCIe BW per GPU ≥ 65 % of 63 GB/s theoretical.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (2.253s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_compute_bound_regime</div></td><td><div class="cell-wrap">TC-BN-07c: A kernel with AI=1000 FLOP/byte must be compute-bound.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_knee_fp16</div></td><td><div class="cell-wrap">TC-BN-07: Roofline knee (AI threshold) for FP16 must be ~590 FLOP/byte.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.001s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_knee_per_precision[BF16-1979.0-BF16 Tensor Core]</div></td><td><div class="cell-wrap">TC-BN-07d: Report roofline knee (FLOP/byte) for each precision.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_knee_per_precision[FP16-1979.0-FP16 Tensor Core]</div></td><td><div class="cell-wrap">TC-BN-07d: Report roofline knee (FLOP/byte) for each precision.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_knee_per_precision[FP32-67.0-FP32 CUDA Core]</div></td><td><div class="cell-wrap">TC-BN-07d: Report roofline knee (FLOP/byte) for each precision.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_knee_per_precision[FP64-34.0-FP64 Tensor Core]</div></td><td><div class="cell-wrap">TC-BN-07d: Report roofline knee (FLOP/byte) for each precision.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_knee_per_precision[INT8-3958.0-INT8 Tensor Core]</div></td><td><div class="cell-wrap">TC-BN-07d: Report roofline knee (FLOP/byte) for each precision.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_roofline_memory_bound_regime</div></td><td><div class="cell-wrap">TC-BN-07b: A kernel with AI=10 FLOP/byte must be memory-bound on H100.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_end_to_end_bottleneck_report</div></td><td><div class="cell-wrap">TC-BN-10c: Run all key benchmarks and produce a consolidated bottleneck report.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.767s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_known_bottlenecks_summary</div></td><td><div class="cell-wrap">TC-BN-10b: Log known theoretical bottlenecks for the platform.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.000s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
<tr><td><div class="cell-wrap">test_theoretical_platform_summary</div></td><td><div class="cell-wrap">TC-BN-10: Log full theoretical platform specification summary.</div></td><td><span style="color:#1b8a3b;font-weight:700">PASS</span></td><td><div class="cell-wrap"><span style="color:#1b8a3b;font-weight:700">PASS (0.001s)</span></div></td><td><div class="cell-wrap">Executed successfully; acceptance criteria met.</div></td><td><div class="cell-wrap">-</div></td><td><div class="cell-wrap">Acceptance criteria satisfied</div></td></tr>
</tbody>
</table>
</div>

## Performance

- No tests captured in this section for the current run.

## MLPerf

- No MLPerf tests captured in this run. This section is reserved for upcoming MLPerf coverage.

## Other

- No tests captured in this section for the current run.
