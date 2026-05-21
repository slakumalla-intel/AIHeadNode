# Workload Execution Status Report

Generated (UTC): 2026-05-21T06:11:07.586230+00:00
Host: sc00901168s0075

## Summary

- Domains run: 13
- Passed: 2
- Failed: 11
- Total tests: 411
- Failures: 93
- Errors: 0
- Skipped: 63

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| bottleneck | FAIL | 25 | 3 | 0 | 0 | 40.969 | artifacts/reports/junit/bottleneck.xml | artifacts/reports/logs/bottleneck.log |
| cpu | PASS | 33 | 0 | 0 | 1 | 280.374 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 8 | 0 | 0 | 28.26 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | FAIL | 53 | 2 | 0 | 0 | 18.006 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | FAIL | 77 | 1 | 0 | 4 | 100.604 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 20 | 0 | 15 | 218.838 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| agentic_ai | FAIL | 9 | 1 | 0 | 0 | 17.332 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | FAIL | 38 | 12 | 0 | 16 | 30.846 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 20 | 0 | 9 | 12.075 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | FAIL | 8 | 6 | 0 | 0 | 3.467 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | PASS | 3 | 0 | 0 | 1 | 642.82 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | FAIL | 26 | 1 | 0 | 15 | 358.118 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 19 | 0 | 2 | 42.472 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### bottleneck

- [failure] tests.test_bottleneck_analysis.TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report: AssertionError: CPU FP32 (1 processes × 32 threads): efficiency 3.78% is below threshold 5.00% (measured=1.030, theoretical=27.238)
assert 0.03782912600018136 >= 0.05
- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency: AssertionError: GPU 2 FP16 Tensor Core: efficiency 58.35% is below threshold 60.00% (measured=441.124, theoretical=756.000)
assert 0.5834979163867962 >= 0.6
- [failure] tests.test_bottleneck_analysis.TestMultiGPUScalingBottleneck::test_scaling_efficiency_2_vs_1_gpu: AssertionError: 2-GPU FP16 scaling efficiency: efficiency 37.54% is below threshold 80.00% (measured=359.835, theoretical=958.654)
assert 0.37535391723720046 >= 0.8

### memory

- [failure] tests.test_memory_validation.TestRAMCapacity::test_total_ram_at_least_1tb: AssertionError: Total RAM 1007 GB < 1 TB
assert 1006.9953422546387 >= 1024.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 12.31 GB/s < 15 GB/s
assert 12.312918601543629 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 13.27 GB/s < 15 GB/s
assert 13.274808501090751 >= 15.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[32-4.0]: AssertionError: Working-set 32 KB: latency 79.98 ns > 4.0 ns
assert 79.97558615358002 <= 4.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[512-8.0]: AssertionError: Working-set 512 KB: latency 79.49 ns > 8.0 ns
assert 79.48526004053491 <= 8.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[16384-30.0]: AssertionError: Working-set 16384 KB: latency 129.05 ns > 30.0 ns
assert 129.05498999316478 <= 30.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[524288-120.0]: AssertionError: Working-set 524288 KB: latency 216.30 ns > 120.0 ns
assert 216.3036699857912 <= 120.0
- [failure] tests.test_memory_validation.TestCacheHierarchyBandwidth::test_bandwidth_at_working_set[L3-cache-8.0-20.0]: AssertionError: L3-cache: measured 19.4 GB/s < expected 20.0 GB/s
assert 19.39079454327372 >= 20.0

### gpu

- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[0]: AssertionError: FP32 GFLOPS GPU 0: efficiency 32.80% is below threshold 52.00% (measured=21976.288, theoretical=67000.000)
assert 0.32800430353811677 >= 0.52
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[3]: AssertionError: PCIe D2H bandwidth GPU 3: efficiency 38.00% is below threshold 65.00% (measured=23.938, theoretical=63.000)
assert 0.3799609248945428 >= 0.65

### interconnect

- [failure] tests.test_interconnect_validation.TestPCIeTopology::test_pcie_gen5_sysfs: AssertionError: No NVIDIA GPU PCIe Gen 5 (32 GT/s) links found. Observed: ['32.0 GT/s PCIe']
assert False

### node_validation

- [failure] tests.test_node_validation.TestPCIeBandwidthDirect::test_h2d_bandwidth_all_gpus: Failed: H2D PCIe bandwidth validation failed.
threshold=40.95 GB/s, boundary=1.00 GB/s, min=0.00 GB/s, mean=0.00 GB/s
Per-GPU failures:
GPU 0: 0.0 GB/s (measurement failed; check CUDA/P2P path and logs)
GPU 1: 0.0 GB/s (measurement failed; check CUDA/P2P path 
- [failure] tests.test_node_validation.TestPCIeBandwidthDirect::test_d2h_bandwidth_all_gpus: Failed: D2H PCIe bandwidth validation failed.
threshold=40.95 GB/s, boundary=1.00 GB/s, min=0.00 GB/s, mean=0.00 GB/s
Per-GPU failures:
GPU 0: 0.0 GB/s (measurement failed; check CUDA/P2P path and logs)
GPU 1: 0.0 GB/s (measurement failed; check CUDA/P2P path 
- [failure] tests.test_node_validation.TestNVBandwidthMemcpy::test_pcie_bw_p2p_latency_hugepages_2mb: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_multi_stream_bandwidth: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestCPUGPULaunchAndSyncOverhead::test_kernel_launch_latency_stats: AssertionError: Kernel launch measurement failed
assert {}
- [failure] tests.test_node_validation.TestCPUGPULaunchAndSyncOverhead::test_synchronize_overhead_stats: AssertionError: Synchronize measurement failed
assert {}
- [failure] tests.test_node_validation.TestCPUGPUPowerAndThrottling::test_gpu_write_performance_limiters: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[deepseek_r1_quantized-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-32-3000.0-True]: Failed: Insufficient free GPU memory for quantized enterprise workload deepseek-ai/DeepSeek-R1-Distill-Qwen-7B. Require 4 GPU(s) with at least 6.0GiB headroom. Offending devices: gpu0 free=-1.0GiB reserve=6.0GiB, gpu1 free=-1.0GiB reserve=6.0GiB, gpu2 free=-1.
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_7b-meta-llama/Meta-Llama-3.1-8B-Instruct-32-3000.0-False]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_70b-meta-llama/Meta-Llama-3.1-70B-Instruct-16-12000.0-True]: Failed: Insufficient free GPU memory for quantized enterprise workload meta-llama/Meta-Llama-3.1-70B-Instruct. Require 4 GPU(s) with at least 6.0GiB headroom. Offending devices: gpu0 free=-1.0GiB reserve=6.0GiB, gpu1 free=-1.0GiB reserve=6.0GiB, gpu2 free=-1.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[qwen_70b-Qwen/Qwen2.5-72B-Instruct-16-12000.0-True]: Failed: Insufficient free GPU memory for quantized enterprise workload Qwen/Qwen2.5-72B-Instruct. Require 4 GPU(s) with at least 6.0GiB headroom. Offending devices: gpu0 free=-1.0GiB reserve=6.0GiB, gpu1 free=-1.0GiB reserve=6.0GiB, gpu2 free=-1.0GiB reserve=6
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_batch_throughput: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: Failed: YOLO worker failure: [{'gpu': 0, 'status': 'error', 'error': 'CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTo
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_image_generation_diffusers_throughput: Failed: Diffusers worker failure: [{'status': 'error', 'rank': 1, 'error': 'CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug t
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_encoder_training_throughput: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_one_dma_read_4kb_validation: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_looped_rdma_mmio_dma_stress_soak: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu: Failed: TC-013 worker failures: [{'rank': 1, 'status': 'error', 'error': 'CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to 

### agentic_ai

- [failure] tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability: AssertionError: Tool dispatch scales poorly: 16 tools take 6.1× time of 1 tool
assert 6.108586243709013 < 5.0

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling: AssertionError: 4-thread tokenizer scaling 1.66× < 2.0× single-thread
assert 1.6562835316740567 >= 2.0
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput[4]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_checkpoint_save_restore_throughput: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_kv_cache_memory_allocation_efficiency: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_batch_assembly_gpu_latency[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_batch_assembly_gpu_latency[4]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_batch_assembly_gpu_latency[16]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_aggregate_4gpu_fp16_tflops_per_watt: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_int8_matmul_error_vs_fp16: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_fp8_e4m3_matmul_error_vs_fp16: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_bf16_vs_fp16_matmul_equivalence: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestDDPGradientSynchronizationCorrectness::test_ddp_gradient_sync_matches_single_gpu: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestDDPGradientSynchronizationCorrectness::test_multi_gpu_parameter_consistency_after_allreduce: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_output_matches_eager: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_event_wait_enforces_ordering: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_multi_stream_no_race_on_shared_tensor: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_cuda_event_elapsed_time_positive: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_concurrent_stream_throughput_vs_sequential: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call

### system_config

- [failure] tests.test_system_config_report.TestGPUConfigReport::test_collect_and_print_gpu_config: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_config_report.TestGPUConfigReport::test_gpu_model_names_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_config_report.TestGPUConfigReport::test_total_gpu_memory_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_config_report.TestPCIeTopologyReport::test_pcie_topology_present_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_config_report.TestPCIeTopologyReport::test_pcie_links_healthy_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call
- [failure] tests.test_system_config_report.TestCombinedConfigReport::test_generate_and_save_combined_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=2, num_gpus=2

CUDA call

### performance_suite

- [failure] tests.test_performance_suite::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]: Failed: Command failed for AI_HOSTCPU_PERF_TEST_ID_011: genai-perf profile --endpoint-type chat --service-kind openai --url http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
executed_command:
genai-perf profile --endpoint-type chat --service-ki

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]: Failed: AI_HOSTCPU_ML_TEST_ID_003: command failed (rc=1)
executed_command: vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=198491) Traceback (m
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: AI_HOSTCPU_ML_TEST_ID_004: command failed (rc=1)
executed_command: vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
envs/torch-cu128/lib64/python3.12/s
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]: Failed: AI_HOSTCPU_ML_TEST_ID_005: command failed (rc=1)
executed_command: vllm serve meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=199048) Tr
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006]: Failed: AI_HOSTCPU_ML_TEST_ID_006: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch resnet50 gpu --scenario Offline --dataset-path /mnt/datasets/imagenet --output artifa
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007]: Failed: AI_HOSTCPU_ML_TEST_ID_007: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch retinanet gpu --scenario Offline --dataset-path /mnt/datasets/openimages --output art
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008]: Failed: AI_HOSTCPU_ML_TEST_ID_008: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/medical_imaging/3d-unet-kits19/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/kits19 --output artifacts/mlperf_resul
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009]: Failed: AI_HOSTCPU_ML_TEST_ID_009: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/speech_recognition/rnnt/run_local.sh pytorch rnnt gpu --scenario Offline --dataset-path /mnt/datasets/librispeech --output artifacts/mlperf_re
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010]: Failed: AI_HOSTCPU_ML_TEST_ID_010: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/speech_recognition/whisper/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/librispeech --output artifacts/mlperf_resu
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011]: Failed: AI_HOSTCPU_ML_TEST_ID_011: command failed (rc=2)
executed_command: /opt/venvs/torch-cu128/bin/python /root/workloads/mlcommons/inference/recommendation/dlrm_v2/pytorch/run.py --scenario Offline --dataset-path /mnt/datasets/criteo --model-path /mnt/data
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012]: Failed: AI_HOSTCPU_ML_TEST_ID_012: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/text_to_image/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/coco-sdxl --model-path /mnt/datasets/sdxl --output arti
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013]: Failed: AI_HOSTCPU_ML_TEST_ID_013: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/graph/R-GAT/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/igb-het --model-path /mnt/datasets/rgat --output artifact
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014]: Failed: AI_HOSTCPU_ML_TEST_ID_014: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/gpt-j/run_training.py --model-name EleutherAI/gpt-j-6b --dataset-path /mnt/datasets/cnn_dailymail --output-
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015]: Failed: AI_HOSTCPU_ML_TEST_ID_015: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_squad.py --bert_model bert-large-uncased --do_train --train_file /mnt/datasets/squad/train
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016]: Failed: AI_HOSTCPU_ML_TEST_ID_016: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/image_segmentation/pytorch/train.py --data-dir /mnt/datasets/kits19 --epochs 4000 --batch-size 2 --optimizer adam --learni
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017]: Failed: AI_HOSTCPU_ML_TEST_ID_017: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/stable_diffusion/main.py --base /root/workloads/mlcommons/training/stable_diffusion/configs/train_01x04x08.yaml --actual_r
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018]: Failed: AI_HOSTCPU_ML_TEST_ID_018: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_pretraining.py --bert_model bert-large-uncased --do_train --train_file /mnt/datasets/bert/
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019]: Failed: AI_HOSTCPU_ML_TEST_ID_019: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/large_language_model/megatron-lm/pretrain_gpt.py --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --num-lay
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020]: Failed: AI_HOSTCPU_ML_TEST_ID_020: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/llm_finetune/run_lora_finetune.py --model_name_or_path meta-llama/Llama-2-70b-hf --dataset_name scrolls --dataset_config g
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021]: Failed: AI_HOSTCPU_ML_TEST_ID_021: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/recommendation/dlrm_v2/pytorch/dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot=13-512-256-128 --arch-mlp-t
