# Workload Execution Status Report

Generated (UTC): 2026-05-23T05:33:25.879687+00:00
Host: sc00901168s0075

## Summary

- Domains run: 13
- Passed: 7
- Failed: 6
- Total tests: 396
- Failures: 16
- Errors: 0
- Skipped: 53

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| cpu | PASS | 33 | 0 | 0 | 1 | 268.122 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 2 | 0 | 0 | 27.129 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | FAIL | 53 | 2 | 0 | 0 | 18.516 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | PASS | 77 | 0 | 0 | 4 | 104.742 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 3 | 0 | 11 | 1574.543 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| cpu_headnode | PASS | 10 | 0 | 0 | 2 | 40.287 | artifacts/reports/junit/cpu_headnode.xml | artifacts/reports/logs/cpu_headnode.log |
| agentic_ai | PASS | 9 | 0 | 0 | 0 | 18.2 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | PASS | 38 | 0 | 0 | 4 | 151.112 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 7 | 0 | 3 | 43.782 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 3.748 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | PASS | 3 | 0 | 0 | 1 | 657.76 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | FAIL | 26 | 1 | 0 | 7 | 1549.848 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 1 | 0 | 20 | 1839.62 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### memory

- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 13.23 GB/s < 15 GB/s
assert 13.23311603926585 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 13.39 GB/s < 15 GB/s
assert 13.38512099253258 >= 15.0

### gpu

- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_host_to_device_bandwidth[0]: AssertionError: PCIe H2D bandwidth GPU 0: efficiency 38.10% is below threshold 65.00% (measured=24.006, theoretical=63.000)
assert 0.38104147042384023 >= 0.65
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[1]: AssertionError: PCIe D2H bandwidth GPU 1: efficiency 37.99% is below threshold 65.00% (measured=23.937, theoretical=63.000)
assert 0.3799487783236618 >= 0.65

### node_validation

- [failure] tests.test_node_validation.TestPCIeBandwidthDirect::test_h2d_bandwidth_all_gpus: Failed: H2D PCIe bandwidth validation failed.
threshold=40.95 GB/s, boundary=1.00 GB/s, min=39.74 GB/s, mean=51.98 GB/s
Per-GPU failures:
GPU 0: 39.74 GB/s < 40.95 GB/s
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: AssertionError: YOLO aggregate FPS too low: 92.8 < target 200.0
assert 92.8198194671177 >= 200.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload: AssertionError: Offload latency increase too high: 702.1% (baseline 5015.41ms -> offload 40228.46ms)
assert 702.0971807380655 <= 150

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0]: AssertionError: GPU 0 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1]: AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 46.2% during sustained load (initial 1755 MHz → min 945 MHz, max_temp 59 °C). Thermal or power-limit throttling detected.
assert 0.46153846153846156 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_aggregate_4gpu_fp16_tflops_per_watt: AssertionError: 4-GPU aggregate efficiency 0.821 TFLOPS/W < 1.0 floor
assert 0.8205215640156254 >= 1.0
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.811× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8108262610541674 >= 1.05

### performance_suite

- [failure] tests.test_performance_suite::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]: Failed: Command failed for AI_HOSTCPU_PERF_TEST_ID_011: genai-perf profile --endpoint-type chat --service-kind openai --url http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
executed_command:
genai-perf profile --endpoint-type chat --service-ki

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: Timeout (>1800.0s) from pytest-timeout.
