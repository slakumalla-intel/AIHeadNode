# Workload Execution Status Report

Generated (UTC): 2026-05-21T01:49:53.502642+00:00
Host: sc00901168s0075

## Summary

- Domains run: 5
- Passed: 2
- Failed: 3
- Total tests: 97
- Failures: 13
- Errors: 0
- Skipped: 16

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| agentic_ai | FAIL | 9 | 1 | 0 | 0 | 18.029 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | FAIL | 38 | 5 | 0 | 12 | 36.67 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 7 | 0 | 3 | 45.54 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 3.794 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | PASS | 3 | 0 | 0 | 1 | 648.589 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |

## Failure Details

### agentic_ai

- [failure] tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability: AssertionError: Tool dispatch scales poorly: 16 tools take 6.8× time of 1 tool
assert 6.794190516873131 < 5.0

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: AssertionError: Async prefetch overlap efficiency 25.0% < 60.0%
assert 24.981896126754208 >= 60.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: AssertionError: Model load bandwidth 0.40 GB/s < 1.0 GB/s
assert 0.4001328022972373 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction: AssertionError: AllReduce overhead 98.7% > 20.0% of compute
assert 98.65338073057603 <= 20.0
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: AssertionError: Weight load pipeline 0.85 GB/s < 1.0 GB/s
assert 0.8517238650420215 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0]: AssertionError: GPU 0 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1]: AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 46.2% during sustained load (initial 1755 MHz → min 945 MHz, max_temp 58 °C). Thermal or power-limit throttling detected.
assert 0.46153846153846156 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03
assert 0.0025634765625 < 0.001
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.808× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8077488450400229 >= 1.05
