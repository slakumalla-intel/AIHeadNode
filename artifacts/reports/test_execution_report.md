# Test Execution Report

- Generated: 2026-05-05 14:12:48
- Source: artifacts/reports/junit/*.xml

## Overall Summary

- Total mapped tests: 410
- Executed tests: 375
- PASS: 303
- FAIL: 72
- ERROR: 0
- SKIP: 35
- NOT_EXECUTED: 0

## Suite Summary

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

## Failures And Errors

| TestSelector | Status | Evidence | Notes |
|---|---|---|---|
| tests/test_agentic_ai_validation.py::TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability | FAIL | agentic_ai.xml | AssertionError: Tool dispatch scales poorly: 16 tools take 7.8× time of 1 tool |
| tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling | FAIL | ai_pipeline.xml | AssertionError: 4-thread tokenizer scaling 1.48× < 2.0× single-thread |
| tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency | FAIL | ai_pipeline.xml | AssertionError: Async prefetch overlap efficiency 25.3% < 60.0% |
| tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_model_load_time_to_gpu | FAIL | ai_pipeline.xml | AssertionError: Model load bandwidth 0.20 GB/s < 1.0 GB/s |
| tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction | FAIL | ai_pipeline.xml | AssertionError: AllReduce overhead 98.6% > 20.0% of compute |
| tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth | FAIL | ai_pipeline.xml | AssertionError: Weight load pipeline 0.86 GB/s < 1.0 GB/s |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_time_to_first_token[2048] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_inter_token_latency | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_prefill_vs_decode_throughput_ratio | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput[1] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput[4] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput[8] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput[16] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_gpu_utilisation_during_inference | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_hbm_bandwidth_utilisation_during_decode | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_latency_throughput_tradeoff_curve | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_inference_pipeline_latency_breakdown | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_multi_gpu_inference_throughput_scaling | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered |
| tests/test_bottleneck_analysis.py::TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency | FAIL | bottleneck.xml | AssertionError: GPU 2 FP16 Tensor Core: efficiency 36.51% is below threshold 60.00% (measured=276.047, theoretical=756.000) |
| tests/test_cpu_validation.py::TestCacheCoherency::test_false_sharing_detection | FAIL | cpu.xml | AssertionError: Expected padded median (0.290s) to be within 10% of unpadded median (0.253s) |
| tests/test_gpu_validation.py::TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[2] | FAIL | gpu.xml | AssertionError: FP16 TFLOPS GPU 2: efficiency 55.83% is below threshold 60.00% (measured=422.109, theoretical=756.000) |
| tests/test_gpu_validation.py::TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[3] | FAIL | gpu.xml | AssertionError: FP16 TFLOPS GPU 3: efficiency 44.71% is below threshold 75.00% (measured=442.197, theoretical=989.000) |
| tests/test_gpu_validation.py::TestBF16Throughput::test_bf16_tflops_per_gpu[3] | FAIL | gpu.xml | AssertionError: BF16 TFLOPS GPU 3: efficiency 49.41% is below threshold 75.00% (measured=488.632, theoretical=989.000) |
| tests/test_gpu_validation.py::TestINT8Throughput::test_int8_tops_per_gpu[0] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 0: efficiency 13.82% is below threshold 75.00% (measured=547.033, theoretical=3958.000) |
| tests/test_gpu_validation.py::TestINT8Throughput::test_int8_tops_per_gpu[1] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 1: efficiency 13.88% is below threshold 75.00% (measured=549.562, theoretical=3958.000) |
| tests/test_gpu_validation.py::TestINT8Throughput::test_int8_tops_per_gpu[2] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 2: efficiency 12.71% is below threshold 75.00% (measured=502.965, theoretical=3958.000) |
| tests/test_gpu_validation.py::TestINT8Throughput::test_int8_tops_per_gpu[3] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 3: efficiency 13.74% is below threshold 75.00% (measured=543.937, theoretical=3958.000) |
| tests/test_gpu_validation.py::TestGPUECC::test_gpu_ecc_enabled | FAIL | gpu.xml | AssertionError: GPU 0: ECC is not enabled (mode=[1, 1]) |
| tests/test_gpu_validation.py::TestGPUEnumerationPCIeTopology::test_gpu_pci_bus_numbers_are_unique | FAIL | gpu.xml | AssertionError: GPU 2: PCIe bus number 0x61 collides with another GPU |
| tests/test_interconnect_validation.py::TestPCIeTopology::test_pcie_gen5_sysfs | FAIL | interconnect.xml | AssertionError: No NVIDIA GPU PCIe Gen 5 (32 GT/s) links found. Observed: ['32.0 GT/s PCIe'] |
| tests/test_memory_validation.py::TestRAMCapacity::test_total_ram_at_least_1tb | FAIL | memory.xml | AssertionError: Total RAM 1007 GB < 1 TB |
| tests/test_memory_validation.py::TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs | FAIL | memory.xml | AssertionError: STREAM Scale 11.90 GB/s < 15 GB/s |
| tests/test_memory_validation.py::TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs | FAIL | memory.xml | AssertionError: STREAM Triad 11.83 GB/s < 15 GB/s |
| tests/test_memory_validation.py::TestMemoryLatency::test_latency_at_working_set[32-4.0] | FAIL | memory.xml | AssertionError: Working-set 32 KB: latency 82.10 ns > 4.0 ns |
| tests/test_memory_validation.py::TestMemoryLatency::test_latency_at_working_set[512-8.0] | FAIL | memory.xml | AssertionError: Working-set 512 KB: latency 85.00 ns > 8.0 ns |
| tests/test_memory_validation.py::TestMemoryLatency::test_latency_at_working_set[16384-30.0] | FAIL | memory.xml | AssertionError: Working-set 16384 KB: latency 184.97 ns > 30.0 ns |
| tests/test_memory_validation.py::TestMemoryLatency::test_latency_at_working_set[524288-120.0] | FAIL | memory.xml | AssertionError: Working-set 524288 KB: latency 222.69 ns > 120.0 ns |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_003: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_004: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_005: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_006: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_007: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_008: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_009: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_010: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_011: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_012: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_013: command failed (rc=2) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_014: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_015: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_016: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_017: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_018: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_019: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_020: command failed (rc=1) |
| tests.py::test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_021: command failed (rc=1) |
| tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback | FAIL | node_validation.xml | Failed: P2P bandwidth fallback validation failed. |
| tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_multi_stream_bandwidth | FAIL | node_validation.xml | Failed: P2P multi-stream stability validation failed. |
| tests/test_node_validation.py::TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability | FAIL | node_validation.xml | AssertionError: Power variance 57.9 W is too high (expected < 50 W) |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_p2p_perf | FAIL | node_validation.xml | AssertionError: NCCL sendrecv busbw 25.1 GB/s too low |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_70b-meta-llama/Meta-Llama-3.1-70B-Instruct-16-12000.0-True] | FAIL | node_validation.xml | OSError: You are trying to access a gated repo. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps | FAIL | node_validation.xml | AssertionError: YOLO aggregate FPS too low: 95.0 < target 200.0 |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu | FAIL | node_validation.xml | Failed: TC-013 worker failures: [{'rank': 3, 'status': 'error', 'error': 'Model load failed: You are trying to access a gated repo.\nMake sure to have access to it at https://huggingface.co/google/gemma-3-270m-it.\n401 Client Error. (Request ID: Root=1-69fa4d6 |
| tests.py::test_performance_suite::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_001] | FAIL | performance_suite.xml | OSError: You are trying to access a gated repo. |
| tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0] | FAIL | system_health.xml | AssertionError: GPU 0 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz) |
| tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1] | FAIL | system_health.xml | AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz) |
| tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2] | FAIL | system_health.xml | AssertionError: GPU 2 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz) |
| tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3] | FAIL | system_health.xml | AssertionError: GPU 3 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz) |
| tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle | FAIL | system_health.xml | AssertionError: SM clock dropped 45.3% during sustained load (initial 1755 MHz → min 960 MHz, max_temp 60 °C). Thermal or power-limit throttling detected. |
| tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_power_limit_headroom | FAIL | system_health.xml | AssertionError: GPU(s) exceeding 90 % power limit at rest: GPU0 99% |
| tests/test_system_health_and_reliability.py::TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16 | FAIL | system_health.xml | AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03 |
| tests/test_system_health_and_reliability.py::TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager | FAIL | system_health.xml | AssertionError: torch.compile speedup 0.809× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration. |
