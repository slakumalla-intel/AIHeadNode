# Workload Execution Status Report

Generated (UTC): 2026-04-28T04:42:12.181620+00:00
Host: sc00901168s0075

## Summary

- Domains run: 13
- Passed: 2
- Failed: 11
- Total tests: 410
- Failures: 98
- Errors: 0
- Skipped: 33

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| bottleneck | FAIL | 25 | 4 | 0 | 0 | 10.051 | artifacts/reports/junit/bottleneck.xml | artifacts/reports/logs/bottleneck.log |
| cpu | FAIL | 33 | 9 | 0 | 1 | 232.45 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 8 | 0 | 0 | 17.36 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | FAIL | 53 | 20 | 0 | 0 | 6.141 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | FAIL | 77 | 1 | 0 | 4 | 67.352 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 9 | 0 | 7 | 2577.366 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| agentic_ai | FAIL | 9 | 1 | 0 | 0 | 14.103 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | FAIL | 38 | 18 | 0 | 0 | 68.638 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 8 | 0 | 3 | 34.454 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 0.841 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | FAIL | 3 | 1 | 0 | 1 | 1130.979 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | PASS | 25 | 0 | 0 | 15 | 208.218 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 19 | 0 | 2 | 1889.065 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### bottleneck

- [failure] tests.test_bottleneck_analysis.TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report: AssertionError: CPU FP32 (1 process, BLAS threads=224): efficiency 7.05% is below threshold 70.00% (measured=1.919, theoretical=27.238)
assert 0.07045469366717601 >= 0.7
- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency: AssertionError: GPU 0 FP16 Tensor Core: efficiency 31.59% is below threshold 75.00% (measured=477.978, theoretical=1513.000)
assert 0.31591398244437896 >= 0.75
- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_aggregate_4gpu_fp16_tflops: AssertionError: 4-GPU aggregate FP16 TFLOPS: efficiency 29.91% is below threshold 75.00% (measured=1809.936, theoretical=6052.000)
assert 0.2990641025347952 >= 0.75
- [failure] tests.test_bottleneck_analysis.TestPCIeBottleneck::test_pcie_bottleneck_ratio_vs_hbm: AssertionError: HBM/PCIe ratio 31.7× < 40× – PCIe may not be the ingestion bottleneck
assert 31.746031746031747 >= 40.0

### cpu

- [failure] tests.test_cpu_validation.TestSingleCoreFP32::test_single_core_gflops_above_floor: AttributeError: 'CPUSpecs' object has no attribute 'avx512_fp32_flops_per_cycle_per_core'. Did you mean: 'simd_fp32_flops_per_cycle_per_core'?
- [failure] tests.test_cpu_validation.TestAllCoreFP32Scaling::test_all_core_fp32_gflops: AssertionError: All-core FP32 GFLOPS (BLAS): efficiency 26.05% is below threshold 70.00% (measured=7095.134, theoretical=27238.400)
assert 0.2604827907403243 >= 0.7
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[4]: AssertionError: Parallel speedup at 4 cores: efficiency 6.74% is below threshold 85.00% (measured=72.243, theoretical=1072.159)
assert 0.06738127724876494 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[8]: AssertionError: Parallel speedup at 8 cores: efficiency 0.26% is below threshold 85.00% (measured=6.095, theoretical=2342.638)
assert 0.002601853902865249 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[16]: AssertionError: Parallel speedup at 16 cores: efficiency 0.12% is below threshold 85.00% (measured=5.609, theoretical=4838.845)
assert 0.001159159670273666 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[32]: AssertionError: Parallel speedup at 32 cores: efficiency 0.06% is below threshold 85.00% (measured=5.081, theoretical=9075.926)
assert 0.0005597988124748729 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[64]: AssertionError: Parallel speedup at 64 cores: efficiency 0.03% is below threshold 85.00% (measured=5.313, theoretical=18064.635)
assert 0.0002941245472917233 >= 0.85
- [failure] tests.test_cpu_validation.TestIntegerThroughput::test_integer_vector_ops_throughput: AssertionError: Concurrent INT64 vector add throughput 23.71 GOPS < 56.00 GOPS (224 threads × 0.25 GOPS/thread floor)
assert 23.709933761795686 >= 56.0
- [failure] tests.test_cpu_validation.TestIntegerThroughput::test_integer_multiply_throughput: AssertionError: Concurrent INT64 multiply throughput 36.93 GOPS < 56.00 GOPS (224 threads × 0.25 GOPS/thread floor)
assert 36.931165837541656 >= 56.0

### memory

- [failure] tests.test_memory_validation.TestRAMCapacity::test_total_ram_at_least_1tb: AssertionError: Total RAM 1007 GB < 1 TB
assert 1006.9956436157227 >= 1024.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 11.28 GB/s < 15 GB/s
assert 11.280456482108326 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 11.88 GB/s < 15 GB/s
assert 11.87695927351219 >= 15.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[32-4.0]: AssertionError: Working-set 32 KB: latency 83.29 ns > 4.0 ns
assert 83.29272466189863 <= 4.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[512-8.0]: AssertionError: Working-set 512 KB: latency 87.26 ns > 8.0 ns
assert 87.26184081916944 <= 8.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[16384-30.0]: AssertionError: Working-set 16384 KB: latency 194.29 ns > 30.0 ns
assert 194.28579000077661 <= 30.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[524288-120.0]: AssertionError: Working-set 524288 KB: latency 235.67 ns > 120.0 ns
assert 235.66586000015377 <= 120.0
- [failure] tests.test_memory_validation.TestSustainedBandwidth::test_bandwidth_does_not_degrade: AssertionError: BW run 2 (39.43 GB/s) degraded > 15 % vs first run (47.49 GB/s)
assert 0.16978049229886918 <= 0.15

### gpu

- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[0]: AssertionError: FP32 GFLOPS GPU 0: efficiency 36.07% is below threshold 70.00% (measured=24168.578, theoretical=67000.000)
assert 0.36072503853902255 >= 0.7
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[1]: AssertionError: FP32 GFLOPS GPU 1: efficiency 54.87% is below threshold 70.00% (measured=36761.214, theoretical=67000.000)
assert 0.5486748406149198 >= 0.7
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[2]: AssertionError: FP32 GFLOPS GPU 2: efficiency 53.19% is below threshold 70.00% (measured=35636.882, theoretical=67000.000)
assert 0.5318937582298633 >= 0.7
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[3]: AssertionError: FP32 GFLOPS GPU 3: efficiency 68.78% is below threshold 70.00% (measured=46080.061, theoretical=67000.000)
assert 0.6877620997363648 >= 0.7
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[0]: AssertionError: FP16 TFLOPS GPU 0: efficiency 31.58% is below threshold 75.00% (measured=477.814, theoretical=1513.000)
assert 0.31580569083420135 >= 0.75
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[1]: AssertionError: FP16 TFLOPS GPU 1: efficiency 31.59% is below threshold 75.00% (measured=477.994, theoretical=1513.000)
assert 0.31592476349969084 >= 0.75
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[2]: AssertionError: FP16 TFLOPS GPU 2: efficiency 28.10% is below threshold 75.00% (measured=425.163, theoretical=1513.000)
assert 0.28100690708920706 >= 0.75
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[3]: AssertionError: FP16 TFLOPS GPU 3: efficiency 29.30% is below threshold 75.00% (measured=443.349, theoretical=1513.000)
assert 0.2930264347583894 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[0]: AssertionError: BF16 TFLOPS GPU 0: efficiency 25.95% is below threshold 75.00% (measured=513.460, theoretical=1979.000)
assert 0.2594543154601441 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[1]: AssertionError: BF16 TFLOPS GPU 1: efficiency 25.90% is below threshold 75.00% (measured=512.632, theoretical=1979.000)
assert 0.25903562500712995 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[2]: AssertionError: BF16 TFLOPS GPU 2: efficiency 23.41% is below threshold 75.00% (measured=463.366, theoretical=1979.000)
assert 0.23414131608715041 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[3]: AssertionError: BF16 TFLOPS GPU 3: efficiency 24.25% is below threshold 75.00% (measured=479.871, theoretical=1979.000)
assert 0.242481471890651 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[0]: AssertionError: INT8-proxy TOPS GPU 0: efficiency 13.86% is below threshold 75.00% (measured=548.543, theoretical=3958.000)
assert 0.13859088218519344 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[1]: AssertionError: INT8-proxy TOPS GPU 1: efficiency 13.93% is below threshold 75.00% (measured=551.272, theoretical=3958.000)
assert 0.13928038304458404 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[2]: AssertionError: INT8-proxy TOPS GPU 2: efficiency 12.73% is below threshold 75.00% (measured=503.757, theoretical=3958.000)
assert 0.12727569244148984 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[3]: AssertionError: INT8-proxy TOPS GPU 3: efficiency 13.54% is below threshold 75.00% (measured=536.108, theoretical=3958.000)
assert 0.13544923129144987 >= 0.75
- [failure] tests.test_gpu_validation.TestGPUECC::test_gpu_ecc_enabled: AssertionError: GPU 0: ECC is not enabled (mode=[1, 1])
assert [1, 1] == 1
 +  where 1 = pynvml.NVML_FEATURE_ENABLED
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_gpu_pci_bus_numbers_are_unique: AssertionError: GPU 2: PCIe bus number 0x61 collides with another GPU
assert 97 not in [97, 177]
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[2]: AssertionError: PCIe D2H bandwidth GPU 2: efficiency 38.14% is below threshold 65.00% (measured=24.028, theoretical=63.000)
assert 0.38139346362324594 >= 0.65
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[3]: AssertionError: PCIe D2H bandwidth GPU 3: efficiency 39.11% is below threshold 65.00% (measured=24.640, theoretical=63.000)
assert 0.3911168913237322 >= 0.65

### interconnect

- [failure] tests.test_interconnect_validation.TestPCIeTopology::test_pcie_gen5_sysfs: AssertionError: No PCIe Gen 5 (32 GT/s) links found in sysfs
assert False

### node_validation

- [failure] tests.test_node_validation.TestPCIeBandwidthDirect::test_d2h_bandwidth_all_gpus: Failed: D2H PCIe bandwidth validation failed.
threshold=40.95 GB/s, boundary=1.00 GB/s, min=39.33 GB/s, mean=51.78 GB/s
Per-GPU failures:
GPU 0: 39.33 GB/s < 40.95 GB/s
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary: AssertionError: 3 P2P bandwidth values below 40.0 GB/s: ['17.3', '38.3', '39.8']
assert not [17.27, 39.81, 38.27]
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback: Failed: P2P bandwidth fallback validation failed.
threshold=40.0 GB/s, measured_pairs=12, inaccessible_pairs=0
Failures:
GPU1->GPU2: 33.6 GB/s < 40.0 GB/s
GPU1->GPU3: 33.5 GB/s < 40.0 GB/s
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_multi_stream_bandwidth: Failed: P2P multi-stream stability validation failed.
tested_pairs=12, required_min_stability_ratio=0.85, assertion_size=1GiB, observed_min_scaling_ratio=0.789, observed_avg_scaling_ratio=0.964, required_min_directional_symmetry=0.75, observed_min_directional_
- [failure] tests.test_node_validation.TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability: AssertionError: Power variance 57.6 W is too high (expected < 50 W)
assert np.float64(57.64407180539518) < 50
- [failure] tests.test_node_validation.TestNCCLAndNVLinkValidation::test_nccl_p2p_perf: AssertionError: NCCL sendrecv busbw 25.1 GB/s too low
assert 25.07 >= 30.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill: AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%, target_ratio=0.50, shared_gpu=True)
assert 0.0 >= 0.5
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload: AssertionError: Offload throughput degradation too high: 93.1% (baseline 658.1 -> offload 45.1)
assert 93.14828946216544 <= 60
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu: AssertionError: TC-013: gradient sync ratio 21.3% exceeds threshold 15% — communication bottleneck detected; check NVLink status and NCCL topology
assert 0.213325 <= 0.15

### agentic_ai

- [failure] tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability: AssertionError: Tool dispatch scales poorly: 16 tools take 7.1× time of 1 tool
assert 7.050985136536467 < 5.0

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling: AssertionError: 4-thread tokenizer scaling 1.51× < 2.0× single-thread
assert 1.5091894807137098 >= 2.0
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: AssertionError: Async prefetch overlap efficiency 25.0% < 60.0%
assert 24.987182701528184 >= 60.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: AssertionError: Model load bandwidth 0.19 GB/s < 1.0 GB/s
assert 0.18914551734399657 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction: AssertionError: AllReduce overhead 98.6% > 20.0% of compute
assert 98.62941541171739 <= 20.0
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: AssertionError: Weight load pipeline 0.91 GB/s < 1.0 GB/s
assert 0.9062301473010916 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_inter_token_latency: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_prefill_vs_decode_throughput_ratio: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[1]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[4]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[8]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[16]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_gpu_utilisation_during_inference: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_hbm_bandwidth_utilisation_during_decode: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_continuous_batching_throughput_vllm: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_latency_throughput_tradeoff_curve: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_inference_pipeline_latency_breakdown: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_multi_gpu_inference_throughput_scaling: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0]: AssertionError: GPU 0 clock ratio 20.51% below 80%; throttling suspected (current 360 MHz, max 1755 MHz)
assert 0.20512820512820512 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1]: AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 47.0% during sustained load (initial 1755 MHz → min 930 MHz, max_temp 60 °C). Thermal or power-limit throttling detected.
assert 0.4700854700854701 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_power_limit_headroom: AssertionError: GPU(s) exceeding 90 % power limit at rest: GPU0 99%
assert not [(0, {'power_w': 345.936, 'limit_w': 350.0, 'ratio': 0.9883885714285714})]
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03
assert 0.0025634765625 < 0.001
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.811× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8108152349229382 >= 1.05

### vllm_prefill

- [failure] tests.test_vllm_prefill_validation.TestVLLMPrefillValidation::test_vllm_4h100_prefill_validation: Failed: Timeout (>600.0s) from pytest-timeout.

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]: Failed: Timeout (>600.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: Timeout (>600.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]: Failed: Timeout (>600.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006]: Failed: AI_HOSTCPU_ML_TEST_ID_006: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch resnet50 gpu --scenario Offline --dataset-path /root/workloads/AIHeadNode/da
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007]: Failed: AI_HOSTCPU_ML_TEST_ID_007: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch retinanet gpu --scenario Offline --dataset-path /root/workloads/AIHeadNode/d
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008]: Failed: AI_HOSTCPU_ML_TEST_ID_008: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/medical_imaging/3d-unet-kits19/run_local.sh pytorch gpu --scenario Offline --dataset-path /root/workloads/AIHeadNode/datasets/kits19 
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009]: Failed: AI_HOSTCPU_ML_TEST_ID_009: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/speech_recognition/rnnt/run_local.sh pytorch rnnt gpu --scenario Offline --dataset-path /root/workloads/AIHeadNode/datasets/librispee
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010]: Failed: AI_HOSTCPU_ML_TEST_ID_010: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/speech_recognition/whisper/run_local.sh pytorch gpu --scenario Offline --dataset-path /root/workloads/AIHeadNode/datasets/librispeech
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011]: Failed: AI_HOSTCPU_ML_TEST_ID_011: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/recommendation/dlrm_v2/pytorch/run.py --scenario Offline --dataset-path /root/workloads/AIHeadNode/datasets/criteo --model-path /root
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012]: Failed: AI_HOSTCPU_ML_TEST_ID_012: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/text_to_image/run_local.sh pytorch gpu --scenario Offline --dataset-path /root/workloads/AIHeadNode/datasets/coco-sdxl --model-path /
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013]: Failed: AI_HOSTCPU_ML_TEST_ID_013: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/graph/R-GAT/run_local.sh pytorch gpu --scenario Offline --dataset-path /root/workloads/AIHeadNode/datasets/igb-het --model-path /root
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014]: Failed: AI_HOSTCPU_ML_TEST_ID_014: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/gpt-j/run_training.py --model-name EleutherAI/gpt-j-6b --dataset-path /root/workloads/AIHeadNode/datasets/c
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015]: Failed: AI_HOSTCPU_ML_TEST_ID_015: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_squad.py --bert_model bert-large-uncased --do_train --train_file /root/workloads/AIHeadNod
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016]: Failed: AI_HOSTCPU_ML_TEST_ID_016: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/image_segmentation/pytorch/train.py --data-dir /root/workloads/AIHeadNode/datasets/kits19 --epochs 4000 --batch-size 2 --o
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017]: Failed: AI_HOSTCPU_ML_TEST_ID_017: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/stable_diffusion/main.py --base /root/workloads/mlcommons/training/stable_diffusion/configs/train_01x04x08.yaml --actual_r
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018]: Failed: AI_HOSTCPU_ML_TEST_ID_018: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_pretraining.py --bert_model bert-large-uncased --do_train --train_file /root/workloads/AIH
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019]: Failed: AI_HOSTCPU_ML_TEST_ID_019: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/large_language_model/megatron-lm/pretrain_gpt.py --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --num-lay
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020]: Failed: AI_HOSTCPU_ML_TEST_ID_020: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/llm_finetune/run_lora_finetune.py --model_name_or_path meta-llama/Llama-2-70b-hf --dataset_name scrolls --dataset_config g
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021]: Failed: AI_HOSTCPU_ML_TEST_ID_021: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/recommendation/dlrm_v2/pytorch/dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot=13-512-256-128 --arch-mlp-t
