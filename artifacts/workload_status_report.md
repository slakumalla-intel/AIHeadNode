# Workload Execution Status Report

Generated (UTC): 2026-04-27T17:07:57.616000+00:00
Host: sc00901168s0075

## Summary

- Domains run: 13
- Passed: 2
- Failed: 11
- Total tests: 410
- Failures: 107
- Errors: 0
- Skipped: 35

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| bottleneck | FAIL | 25 | 5 | 0 | 0 | 9.899 | artifacts/reports/junit/bottleneck.xml | artifacts/reports/logs/bottleneck.log |
| cpu | FAIL | 33 | 11 | 0 | 1 | 268.736 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 8 | 0 | 0 | 15.259 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | FAIL | 53 | 19 | 0 | 0 | 6.616 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | FAIL | 77 | 3 | 0 | 4 | 72.995 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 15 | 0 | 9 | 1240.779 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| agentic_ai | FAIL | 9 | 1 | 0 | 0 | 14.245 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | FAIL | 38 | 18 | 0 | 0 | 55.789 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 7 | 0 | 3 | 34.512 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 0.714 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | FAIL | 3 | 1 | 0 | 1 | 1110.551 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | PASS | 25 | 0 | 0 | 15 | 248.181 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 19 | 0 | 2 | 759.975 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### bottleneck

- [failure] tests.test_bottleneck_analysis.TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report: AssertionError: CPU FP32 (1 process, BLAS threads=224): efficiency 7.03% is below threshold 70.00% (measured=1.915, theoretical=27.238)
assert 0.07029843576423664 >= 0.7
- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency: AssertionError: GPU 0 FP16 Tensor Core: efficiency 31.62% is below threshold 75.00% (measured=478.387, theoretical=1513.000)
assert 0.31618443124915063 >= 0.75
- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_aggregate_4gpu_fp16_tflops: AssertionError: 4-GPU aggregate FP16 TFLOPS: efficiency 29.80% is below threshold 75.00% (measured=1803.664, theoretical=6052.000)
assert 0.29802782333568895 >= 0.75
- [failure] tests.test_bottleneck_analysis.TestHBMBandwidthBottleneck::test_hbm_bandwidth_per_gpu_vs_theoretical: AssertionError: GPU 2 HBM3 bandwidth: efficiency 9.16% is below threshold 80.00% (measured=0.183, theoretical=2.000)
assert 0.0915859993385234 >= 0.8
- [failure] tests.test_bottleneck_analysis.TestPCIeBottleneck::test_pcie_bottleneck_ratio_vs_hbm: AssertionError: HBM/PCIe ratio 31.7× < 40× – PCIe may not be the ingestion bottleneck
assert 31.746031746031747 >= 40.0

### cpu

- [failure] tests.test_cpu_validation.TestSingleCoreFP32::test_single_core_gflops_above_floor: AttributeError: 'CPUSpecs' object has no attribute 'avx512_fp32_flops_per_cycle_per_core'. Did you mean: 'simd_fp32_flops_per_cycle_per_core'?
- [failure] tests.test_cpu_validation.TestAllCoreFP32Scaling::test_all_core_fp32_gflops: AssertionError: All-core FP32 GFLOPS (BLAS): efficiency 26.75% is below threshold 70.00% (measured=7287.016, theoretical=27238.400)
assert 0.26752730753896187 >= 0.7
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[2]: AssertionError: Parallel speedup at 2 cores: efficiency 83.40% is below threshold 85.00% (measured=424.439, theoretical=508.891)
assert 0.8340484327116333 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[4]: AssertionError: Parallel speedup at 4 cores: efficiency 14.29% is below threshold 85.00% (measured=160.452, theoretical=1122.967)
assert 0.14288246938137358 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[8]: AssertionError: Parallel speedup at 8 cores: efficiency 0.31% is below threshold 85.00% (measured=6.389, theoretical=2031.079)
assert 0.0031458064110784253 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[16]: AssertionError: Parallel speedup at 16 cores: efficiency 0.11% is below threshold 85.00% (measured=4.872, theoretical=4482.323)
assert 0.0010869388334792756 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[32]: AssertionError: Parallel speedup at 32 cores: efficiency 0.05% is below threshold 85.00% (measured=4.463, theoretical=9000.100)
assert 0.0004958489628036974 >= 0.85
- [failure] tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[64]: AssertionError: Parallel speedup at 64 cores: efficiency 0.03% is below threshold 85.00% (measured=4.724, theoretical=18197.715)
assert 0.0002595676794268458 >= 0.85
- [failure] tests.test_cpu_validation.TestIntegerThroughput::test_integer_vector_ops_throughput: AssertionError: Concurrent INT64 vector add throughput 21.35 GOPS < 56.00 GOPS (224 threads × 0.25 GOPS/thread floor)
assert 21.354101254305142 >= 56.0
- [failure] tests.test_cpu_validation.TestIntegerThroughput::test_integer_multiply_throughput: AssertionError: Concurrent INT64 multiply throughput 39.35 GOPS < 56.00 GOPS (224 threads × 0.25 GOPS/thread floor)
assert 39.34511729619078 >= 56.0
- [failure] tests.test_cpu_validation.TestCacheCoherency::test_false_sharing_detection: AssertionError: Expected padded (0.888s) < unpadded (0.862s)
assert 0.8884903129946906 < 0.8623696630238555

### memory

- [failure] tests.test_memory_validation.TestRAMCapacity::test_total_ram_at_least_1tb: AssertionError: Total RAM 1007 GB < 1 TB
assert 1006.9956436157227 >= 1024.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 12.31 GB/s < 15 GB/s
assert 12.314018330940955 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 12.15 GB/s < 15 GB/s
assert 12.147721856935542 >= 15.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[32-4.0]: AssertionError: Working-set 32 KB: latency 83.60 ns > 4.0 ns
assert 83.60131431572881 <= 4.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[512-8.0]: AssertionError: Working-set 512 KB: latency 90.44 ns > 8.0 ns
assert 90.44337456742824 <= 8.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[16384-30.0]: AssertionError: Working-set 16384 KB: latency 195.22 ns > 30.0 ns
assert 195.2218500082381 <= 30.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[524288-120.0]: AssertionError: Working-set 524288 KB: latency 233.94 ns > 120.0 ns
assert 233.9418200426735 <= 120.0
- [failure] tests.test_memory_validation.TestSustainedBandwidth::test_bandwidth_does_not_degrade: AssertionError: BW run 2 (12.12 GB/s) degraded > 15 % vs first run (47.53 GB/s)
assert 0.7450064389859917 <= 0.15

### gpu

- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[0]: AssertionError: FP32 GFLOPS GPU 0: efficiency 55.52% is below threshold 70.00% (measured=37197.708, theoretical=67000.000)
assert 0.555189674981243 >= 0.7
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[1]: AssertionError: FP32 GFLOPS GPU 1: efficiency 54.73% is below threshold 70.00% (measured=36671.047, theoretical=67000.000)
assert 0.5473290536180478 >= 0.7
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[2]: AssertionError: FP32 GFLOPS GPU 2: efficiency 52.13% is below threshold 70.00% (measured=34924.735, theoretical=67000.000)
assert 0.521264695085823 >= 0.7
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[3]: AssertionError: FP32 GFLOPS GPU 3: efficiency 68.65% is below threshold 70.00% (measured=45997.801, theoretical=67000.000)
assert 0.6865343487513104 >= 0.7
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[0]: AssertionError: FP16 TFLOPS GPU 0: efficiency 31.66% is below threshold 75.00% (measured=478.966, theoretical=1513.000)
assert 0.3165671841580488 >= 0.75
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[1]: AssertionError: FP16 TFLOPS GPU 1: efficiency 31.72% is below threshold 75.00% (measured=479.998, theoretical=1513.000)
assert 0.31724912784080156 >= 0.75
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[2]: AssertionError: FP16 TFLOPS GPU 2: efficiency 28.08% is below threshold 75.00% (measured=424.841, theoretical=1513.000)
assert 0.2807939085819351 >= 0.75
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[3]: AssertionError: FP16 TFLOPS GPU 3: efficiency 29.07% is below threshold 75.00% (measured=439.832, theoretical=1513.000)
assert 0.2907018526618181 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[0]: AssertionError: BF16 TFLOPS GPU 0: efficiency 25.91% is below threshold 75.00% (measured=512.696, theoretical=1979.000)
assert 0.2590680805251704 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[1]: AssertionError: BF16 TFLOPS GPU 1: efficiency 25.92% is below threshold 75.00% (measured=512.877, theoretical=1979.000)
assert 0.2591597891728717 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[2]: AssertionError: BF16 TFLOPS GPU 2: efficiency 23.33% is below threshold 75.00% (measured=461.739, theoretical=1979.000)
assert 0.2333194946517316 >= 0.75
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[3]: AssertionError: BF16 TFLOPS GPU 3: efficiency 24.33% is below threshold 75.00% (measured=481.402, theoretical=1979.000)
assert 0.2432550310045876 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[0]: AssertionError: INT8-proxy TOPS GPU 0: efficiency 13.84% is below threshold 75.00% (measured=547.654, theoretical=3958.000)
assert 0.13836633977546975 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[1]: AssertionError: INT8-proxy TOPS GPU 1: efficiency 13.88% is below threshold 75.00% (measured=549.173, theoretical=3958.000)
assert 0.13875005126896786 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[2]: AssertionError: INT8-proxy TOPS GPU 2: efficiency 12.74% is below threshold 75.00% (measured=504.417, theoretical=3958.000)
assert 0.12744237137166187 >= 0.75
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[3]: AssertionError: INT8-proxy TOPS GPU 3: efficiency 13.58% is below threshold 75.00% (measured=537.586, theoretical=3958.000)
assert 0.13582259679696718 >= 0.75
- [failure] tests.test_gpu_validation.TestGPUECC::test_gpu_ecc_enabled: AssertionError: GPU 0: ECC is not enabled (mode=[1, 1])
assert [1, 1] == 1
 +  where 1 = pynvml.NVML_FEATURE_ENABLED
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_gpu_pci_bus_numbers_are_unique: AssertionError: GPU 2: PCIe bus number 0x61 collides with another GPU
assert 97 not in [97, 177]
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[2]: AssertionError: PCIe D2H bandwidth GPU 2: efficiency 37.82% is below threshold 65.00% (measured=23.829, theoretical=63.000)
assert 0.3782341608687158 >= 0.65

### interconnect

- [failure] tests.test_interconnect_validation.TestPCIeTopology::test_pcie_gen5_sysfs: AssertionError: No PCIe Gen 5 (32 GT/s) links found in sysfs
assert False
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[0-1]: AssertionError: P2P UVA copy GPU 0→1 produced incorrect values
assert tensor(False)
 +  where tensor(False) = <built-in method all of Tensor object at 0x7f14d82934d0>()
 +    where <built-in method all of Tensor object at 0x7f14d82934d0> = tensor([0., 0., 0., 
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[1-0]: AssertionError: P2P UVA copy GPU 1→0 produced incorrect values
assert tensor(False)
 +  where tensor(False) = <built-in method all of Tensor object at 0x7f14d83025d0>()
 +    where <built-in method all of Tensor object at 0x7f14d83025d0> = tensor([0., 0., 0., 

### node_validation

- [failure] tests.test_node_validation.TestPCIeBandwidthDirect::test_d2h_bandwidth_all_gpus: Failed: D2H PCIe bandwidth validation failed.
threshold=40.95 GB/s, boundary=1.00 GB/s, min=40.28 GB/s, mean=51.64 GB/s
Per-GPU failures:
GPU 3: 40.28 GB/s < 40.95 GB/s
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary: AssertionError: 2 P2P bandwidth values below 40.0 GB/s: ['17.2', '17.2']
assert not [17.25, 17.23]
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_multi_stream_bandwidth: Failed: P2P multi-stream stability validation failed.
tested_pairs=12, required_min_stability_ratio=0.85, assertion_size=1GiB, observed_min_scaling_ratio=0.794, observed_avg_scaling_ratio=1.041, required_min_directional_symmetry=0.75, observed_min_directional_
- [failure] tests.test_node_validation.TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability: AssertionError: Power variance 57.4 W is too high (expected < 50 W)
assert np.float64(57.391626369413764) < 50
- [failure] tests.test_node_validation.TestNCCLAndNVLinkValidation::test_nccl_p2p_perf: AssertionError: NCCL sendrecv busbw 25.1 GB/s too low
assert 25.05 >= 30.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[deepseek_r1_quantized-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-32-3000.0-True]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_7b-meta-llama/Meta-Llama-3.1-8B-Instruct-32-3000.0-False]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_batch_throughput: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill: AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%, target_ratio=0.50, shared_gpu=True)
assert 0.0 >= 0.5
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: AssertionError: YOLO aggregate FPS too low: 92.9 < target 200.0
assert 92.94495106042463 >= 200.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_encoder_training_throughput: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_inference_throughput: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_one_dma_read_4kb_validation: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_looped_rdma_mmio_dma_stress_soak: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu: AssertionError: TC-013: gradient sync ratio 21.4% exceeds threshold 15% — communication bottleneck detected; check NVLink status and NCCL topology
assert 0.214175 <= 0.15

### agentic_ai

- [failure] tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability: AssertionError: Tool dispatch scales poorly: 16 tools take 7.4× time of 1 tool
assert 7.4356755629691405 < 5.0

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling: AssertionError: 4-thread tokenizer scaling 0.53× < 2.0× single-thread
assert 0.5254029624596774 >= 2.0
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: AssertionError: Async prefetch overlap efficiency 25.2% < 60.0%
assert 25.157076046922278 >= 60.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: AssertionError: Model load bandwidth 0.16 GB/s < 1.0 GB/s
assert 0.16018335238275 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction: AssertionError: AllReduce overhead 98.7% > 20.0% of compute
assert 98.68103591038746 <= 20.0
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: AssertionError: Weight load pipeline 0.84 GB/s < 1.0 GB/s
assert 0.8412086480859542 >= 1.0
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

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1]: AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 46.2% during sustained load (initial 1755 MHz → min 945 MHz, max_temp 57 °C). Thermal or power-limit throttling detected.
assert 0.46153846153846156 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_power_limit_headroom: AssertionError: GPU(s) exceeding 90 % power limit at rest: GPU0 96%
assert not [(0, {'power_w': 337.146, 'limit_w': 350.0, 'ratio': 0.9632742857142857})]
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03
assert 0.0025634765625 < 0.001
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.812× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8121578871726854 >= 1.05

### vllm_prefill

- [failure] tests.test_vllm_prefill_validation.TestVLLMPrefillValidation::test_vllm_4h100_prefill_validation: Failed: Timeout (>600.0s) from pytest-timeout.

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]: Failed: Timeout (>600.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: AI_HOSTCPU_ML_TEST_ID_004: command failed (rc=1)
executed_command: vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
568)                    ^^^^^^^^^^^
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]: Failed: AI_HOSTCPU_ML_TEST_ID_005: command failed (rc=1)
executed_command: vllm serve meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
328)                    ^
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
