# AI Head Node Testcase Execution Final Report

Date: 2026-04-06
Scope: Bottleneck and interconnect validation updates (CPU FP32 methodology, PCIe and multistream measurements)

## Highlights

- CPU FP32 bottleneck method was corrected to one-process threaded BLAS to avoid process oversubscription artifacts.
- CPU FP32 comparison was changed to CPU-only TFLOPS vs CPU theoretical TFLOPS (no GPU comparison for this testcase).
- Interconnect validation now supports PCIe-only systems for GPU P2P and no longer assumes NVLink is always present.
- Bidirectional PCIe bandwidth now uses simultaneous H2D and D2H stream capture in one timed window.
- Multistream sweep tests were added for bidirectional, H2D, and D2H with adaptive size selection based on available CUDA memory.
- Latency micro-sweep and steady-state reporting (best/median/p95) were added to improve measurement quality.

## Execution Status Table

| Test | Purpose | Result | Status | Analysis |
|---|---|---|---|---|
| tests/test_bottleneck_analysis.py::TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report | Measure CPU FP32 throughput and compare against CPU theoretical peak fairly | Previous run showed extremely low efficiency with multi-process method (artifact-prone). Method updated to one process plus threaded BLAS. Re-run pending after fix. | Updated, Re-run Required | Earlier failure was likely due to process/thread oversubscription and tiny GEMM setup bias. Current method is more representative for full-chip BLAS throughput. |
| tests/test_bottleneck_analysis.py::TestCPUComputeBottleneck::test_cpu_fp32_base_vs_turbo_headroom | CPU-only theoretical consistency check (base vs turbo) | Not yet captured in shared logs | Pending Execution | Expected to pass unless CPU spec metadata is inconsistent. |
| tests/test_interconnect_validation.py::TestD2HBandwidth::test_d2h_bandwidth_per_gpu[0] | Validate D2H PCIe bandwidth on GPU 0 | Earlier observed failure around 8 GB/s before D2H measurement fix | Updated, Re-run Required | D2H measurement was changed to preallocated host buffer, warmup, and synchronized copy timing. Re-run needed to confirm corrected value. |
| tests/test_interconnect_validation.py::TestD2HBandwidth::test_d2h_bandwidth_per_gpu[1] | Validate D2H PCIe bandwidth on GPU 1 | Earlier observed failure around 8.29 GB/s before D2H measurement fix | Updated, Re-run Required | Same root cause and same fix path as GPU 0; requires new execution evidence. |
| tests/test_interconnect_validation.py::TestBidirectionalPCIeBandwidth::test_bidir_pcie_bandwidth_per_gpu[*] | Validate true bidirectional PCIe via simultaneous streams | Logic updated from separate sum to simultaneous capture | Updated, Re-run Required | This is now aligned with expected methodology for concurrent H2D and D2H behavior. |
| tests/test_interconnect_validation.py::TestMultiStreamParallelSweep::test_multistream_parallel_bidir_sweep[*] | Capture bidirectional multistream BW across sweep sizes | New testcase added | Added, Pending Execution | Reports size-wise captured throughput and is suitable for saturation trend analysis. |
| tests/test_interconnect_validation.py::TestMultiStreamH2DSweep::test_multistream_h2d_sweep[*] | Capture H2D multistream BW across sweep sizes | New testcase added | Added, Pending Execution | Supports adaptive sweep sizing based on free CUDA memory to reduce OOM risk. |
| tests/test_interconnect_validation.py::TestMultiStreamD2HSweep::test_multistream_d2h_sweep[*] | Capture D2H multistream BW across sweep sizes | New testcase added | Added, Pending Execution | Complements H2D sweep and enables directional comparison under identical sweep policy. |
| tests/test_interconnect_validation.py::TestMicroTransferLatency::test_h2d_latency_sweep[*] | Measure H2D micro-transfer latency trend | New testcase added | Added, Pending Execution | Useful to explain lower apparent GB/s at small transfer sizes where fixed overhead dominates. |
| tests/test_interconnect_validation.py::TestMicroTransferLatency::test_d2h_latency_sweep[*] | Measure D2H micro-transfer latency trend | New testcase added | Added, Pending Execution | Provides latency distribution and outlier visibility for D2H path. |

## Overall Assessment

- The test framework has been substantially improved to produce more realistic and topology-aware measurements.
- Most key changes are implementation-complete, but final validation requires re-running the updated tests on the target node.
- Any prior performance conclusions from the older CPU FP32 or D2H methods should be treated as provisional until re-execution data is collected.

## Recommended Next Run Set

1. tests/test_bottleneck_analysis.py::TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report
2. tests/test_bottleneck_analysis.py::TestCPUComputeBottleneck::test_cpu_fp32_base_vs_turbo_headroom
3. tests/test_interconnect_validation.py::TestD2HBandwidth::test_d2h_bandwidth_per_gpu[0]
4. tests/test_interconnect_validation.py::TestD2HBandwidth::test_d2h_bandwidth_per_gpu[1]
5. tests/test_interconnect_validation.py::TestBidirectionalPCIeBandwidth::test_bidir_pcie_bandwidth_per_gpu[0]
6. tests/test_interconnect_validation.py::TestMultiStreamParallelSweep::test_multistream_parallel_bidir_sweep[0]
7. tests/test_interconnect_validation.py::TestMultiStreamH2DSweep::test_multistream_h2d_sweep[0]
8. tests/test_interconnect_validation.py::TestMultiStreamD2HSweep::test_multistream_d2h_sweep[0]
9. tests/test_interconnect_validation.py::TestMicroTransferLatency::test_h2d_latency_sweep[0]
10. tests/test_interconnect_validation.py::TestMicroTransferLatency::test_d2h_latency_sweep[0]
