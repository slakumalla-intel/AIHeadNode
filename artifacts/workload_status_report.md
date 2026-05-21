# Workload Execution Status Report

Generated (UTC): 2026-05-21T16:26:32.889532+00:00
Host: sc00901168s0075

## Summary

- Domains run: 13
- Passed: 4
- Failed: 9
- Total tests: 411
- Failures: 29
- Errors: 0
- Skipped: 56

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| bottleneck | FAIL | 25 | 2 | 0 | 0 | 40.879 | artifacts/reports/junit/bottleneck.xml | artifacts/reports/logs/bottleneck.log |
| cpu | PASS | 33 | 0 | 0 | 1 | 268.612 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 2 | 0 | 0 | 25.801 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | FAIL | 53 | 1 | 0 | 0 | 19.584 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | PASS | 77 | 0 | 0 | 4 | 91.782 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 7 | 0 | 10 | 1609.162 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| agentic_ai | FAIL | 9 | 1 | 0 | 0 | 19.476 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | FAIL | 38 | 6 | 0 | 12 | 34.246 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 6 | 0 | 3 | 45.196 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 3.724 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | PASS | 3 | 0 | 0 | 1 | 628.526 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | FAIL | 26 | 1 | 0 | 7 | 1539.001 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 3 | 0 | 18 | 1845.74 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### bottleneck

- [failure] tests.test_bottleneck_analysis.TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report: AssertionError: CPU FP32 (1 processes × 32 threads): efficiency 3.78% is below threshold 5.00% (measured=1.030, theoretical=27.238)
assert 0.03779821688081017 >= 0.05
- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency: AssertionError: GPU 2 FP16 Tensor Core: efficiency 58.01% is below threshold 60.00% (measured=438.556, theoretical=756.000)
assert 0.5801000358561131 >= 0.6

### memory

- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 12.23 GB/s < 15 GB/s
assert 12.230952997548338 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 12.82 GB/s < 15 GB/s
assert 12.81515025556338 >= 15.0

### gpu

- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[0]: AssertionError: FP32 GFLOPS GPU 0: efficiency 35.86% is below threshold 52.00% (measured=24025.277, theoretical=67000.000)
assert 0.3585862257467484 >= 0.52

### node_validation

- [failure] tests.test_node_validation.TestPCIeBandwidthDirect::test_d2h_bandwidth_all_gpus: Failed: D2H PCIe bandwidth validation failed.
threshold=40.95 GB/s, boundary=1.00 GB/s, min=39.08 GB/s, mean=51.37 GB/s
Per-GPU failures:
GPU 0: 39.08 GB/s < 40.95 GB/s
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary: AssertionError: 1 P2P bandwidth values below 40.0 GB/s: ['17.3']
assert not [17.26]
- [failure] tests.test_node_validation.TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability: AssertionError: Power variance 59.1 W is too high (expected < 50 W)
assert np.float64(59.06382419038794) < 50
- [failure] tests.test_node_validation.TestNCCLAndNVLinkValidation::test_nccl_p2p_perf: AssertionError: NCCL sendrecv busbw 24.1 GB/s too low
assert 24.12 >= 30.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill: AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%, target_ratio=0.50, shared_gpu=True)
assert 0.0 >= 0.5
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: AssertionError: YOLO aggregate FPS too low: 92.9 < target 200.0
assert 92.92775966709132 >= 200.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload: AssertionError: Offload throughput degradation too high: 87.5% (baseline 339.4 -> offload 42.4)
assert 87.50405345445897 <= 60

### agentic_ai

- [failure] tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability: AssertionError: Tool dispatch scales poorly: 16 tools take 6.6× time of 1 tool
assert 6.566299469604243 < 5.0

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling: AssertionError: 4-thread tokenizer scaling 1.63× < 2.0× single-thread
assert 1.6318026356755186 >= 2.0
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: AssertionError: Async prefetch overlap efficiency 25.3% < 60.0%
assert 25.268130746022198 >= 60.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: AssertionError: Model load bandwidth 0.43 GB/s < 1.0 GB/s
assert 0.4295125697207292 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction: AssertionError: AllReduce overhead 98.6% > 20.0% of compute
assert 98.57780799286192 <= 20.0
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: AssertionError: Weight load pipeline 0.87 GB/s < 1.0 GB/s
assert 0.8693625258957309 >= 1.0
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
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 45.3% during sustained load (initial 1755 MHz → min 960 MHz, max_temp 59 °C). Thermal or power-limit throttling detected.
assert 0.452991452991453 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.806× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8061902872568043 >= 1.05

### performance_suite

- [failure] tests.test_performance_suite::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]: Failed: Command failed for AI_HOSTCPU_PERF_TEST_ID_011: genai-perf profile --endpoint-type chat --service-kind openai --url http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
executed_command:
genai-perf profile --endpoint-type chat --service-ki

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]: Failed: AI_HOSTCPU_ML_TEST_ID_003: command failed (rc=1)
executed_command: vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=135802) Traceback (m
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: Timeout (>1800.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]: Failed: AI_HOSTCPU_ML_TEST_ID_005: command failed (rc=1)
executed_command: vllm serve meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=138286) Tr
