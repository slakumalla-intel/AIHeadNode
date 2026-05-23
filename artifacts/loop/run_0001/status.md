# Workload Execution Status Report

Generated (UTC): 2026-05-23T00:03:29.823496+00:00
Host: sc00901168s0075

## Summary

- Domains run: 9
- Passed: 2
- Failed: 7
- Total tests: 336
- Failures: 164
- Errors: 0
- Skipped: 59

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| cpu | PASS | 33 | 0 | 0 | 1 | 235.479 | artifacts/loop/run_0001/junit/cpu.xml | artifacts/loop/run_0001/logs/cpu.log |
| memory | FAIL | 28 | 3 | 0 | 0 | 26.634 | artifacts/loop/run_0001/junit/memory.xml | artifacts/loop/run_0001/logs/memory.log |
| gpu | FAIL | 53 | 42 | 0 | 6 | 17.398 | artifacts/loop/run_0001/junit/gpu.xml | artifacts/loop/run_0001/logs/gpu.log |
| interconnect | FAIL | 77 | 63 | 0 | 12 | 27.669 | artifacts/loop/run_0001/junit/interconnect.xml | artifacts/loop/run_0001/logs/interconnect.log |
| node_validation | FAIL | 51 | 18 | 0 | 17 | 255.543 | artifacts/loop/run_0001/junit/node_validation.xml | artifacts/loop/run_0001/logs/node_validation.log |
| agentic_ai | PASS | 9 | 0 | 0 | 0 | 17.438 | artifacts/loop/run_0001/junit/agentic_ai.xml | artifacts/loop/run_0001/logs/agentic_ai.log |
| ai_pipeline | FAIL | 38 | 10 | 0 | 17 | 32.046 | artifacts/loop/run_0001/junit/ai_pipeline.xml | artifacts/loop/run_0001/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 22 | 0 | 6 | 11.655 | artifacts/loop/run_0001/junit/system_health.xml | artifacts/loop/run_0001/logs/system_health.log |
| system_config | FAIL | 8 | 6 | 0 | 0 | 3.769 | artifacts/loop/run_0001/junit/system_config.xml | artifacts/loop/run_0001/logs/system_config.log |

## Failure Details

### memory

- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 13.78 GB/s < 15 GB/s
assert 13.777981853952554 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 13.66 GB/s < 15 GB/s
assert 13.664780190801295 >= 15.0
- [failure] tests.test_memory_validation.TestCacheHierarchyBandwidth::test_bandwidth_at_working_set[L3-cache-8.0-18.0]: AssertionError: L3-cache: measured 17.4 GB/s < expected 18.0 GB/s
assert 17.351036462331077 >= 18.0

### gpu

- [failure] tests.test_gpu_validation.TestGPUDiscovery::test_all_gpus_initialise: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUModel::test_gpu_model_is_h100: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUModel::test_compute_capability_90: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestHBMCapacity::test_hbm_capacity_per_gpu: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestHBMCapacity::test_total_gpu_memory: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestHBMBandwidth::test_hbm_bandwidth_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestHBMBandwidth::test_hbm_bandwidth_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestHBMBandwidth::test_hbm_bandwidth_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestHBMBandwidth::test_hbm_bandwidth_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP64Throughput::test_fp64_tflops_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP64Throughput::test_fp64_tflops_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP64Throughput::test_fp64_tflops_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestFP64Throughput::test_fp64_tflops_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestMultiGPUCompute::test_all_gpus_concurrent_fp16_tflops: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUUtilisation::test_gpu_utilisation_near_100_percent: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUTemperature::test_gpu_temperature_in_range: pynvml.NVMLError_ResetRequired: GPU requires restart
- [failure] tests.test_gpu_validation.TestGPUPowerDraw::test_gpu_power_within_tdp_at_idle: pynvml.NVMLError_NotSupported: Not Supported
- [failure] tests.test_gpu_validation.TestCUDAStreamConcurrency::test_concurrent_streams_faster_than_serial: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_host_to_device_bandwidth[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_host_to_device_bandwidth[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_host_to_device_bandwidth[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_host_to_device_bandwidth[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_device_to_host_bandwidth[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call

### interconnect

- [failure] tests.test_interconnect_validation.TestH2DBandwidth::test_h2d_bandwidth_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestH2DBandwidth::test_h2d_bandwidth_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestH2DBandwidth::test_h2d_bandwidth_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestH2DBandwidth::test_h2d_bandwidth_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestD2HBandwidth::test_d2h_bandwidth_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestD2HBandwidth::test_d2h_bandwidth_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestD2HBandwidth::test_d2h_bandwidth_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestD2HBandwidth::test_d2h_bandwidth_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestBidirectionalPCIeBandwidth::test_bidir_pcie_bandwidth_per_gpu[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestBidirectionalPCIeBandwidth::test_bidir_pcie_bandwidth_per_gpu[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestBidirectionalPCIeBandwidth::test_bidir_pcie_bandwidth_per_gpu[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestBidirectionalPCIeBandwidth::test_bidir_pcie_bandwidth_per_gpu[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestAllGPUH2D::test_concurrent_h2d_all_gpus: AssertionError: Concurrent H2D BW all 4 GPUs: efficiency 0.00% is below threshold 65.00% (measured=0.000, theoretical=252.000)
assert 0.0 >= 0.65
- [failure] tests.test_interconnect_validation.TestP2PBandwidth::test_p2p_bandwidth_pair[0-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestP2PBandwidth::test_p2p_bandwidth_pair[0-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestP2PBandwidth::test_p2p_bandwidth_pair[0-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestP2PBandwidth::test_p2p_bandwidth_pair[1-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestP2PBandwidth::test_p2p_bandwidth_pair[1-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestP2PBandwidth::test_p2p_bandwidth_pair[2-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[0-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[0-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[0-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[1-0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[1-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[1-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[2-0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[2-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[2-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[3-0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[3-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestUVAMemoryAccess::test_uva_p2p_read_write[3-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestCPUGPUPipelineOverlap::test_compute_transfer_overlap[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestCPUGPUPipelineOverlap::test_compute_transfer_overlap[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestCPUGPUPipelineOverlap::test_compute_transfer_overlap[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestCPUGPUPipelineOverlap::test_compute_transfer_overlap[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[64-0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[64-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[64-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[64-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[256-0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[256-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[256-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[256-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[512-0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[512-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[512-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[512-3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[1024-0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[1024-1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_interconnect_validation.TestDMAConsistency::test_large_h2d_data_integrity[1024-2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call

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
- [failure] tests.test_node_validation.TestNVBandwidthMemcpy::test_pcie_bw_p2p_latency_hugepages_2mb: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary: AssertionError: 1 P2P bandwidth values below 40.0 GB/s: ['18.2']
assert not [18.18]
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_multi_stream_bandwidth: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestCPUGPULaunchAndSyncOverhead::test_kernel_launch_latency_stats: AssertionError: Kernel launch measurement failed
assert {}
- [failure] tests.test_node_validation.TestCPUGPULaunchAndSyncOverhead::test_synchronize_overhead_stats: AssertionError: Synchronize measurement failed
assert {}
- [failure] tests.test_node_validation.TestCPUGPUPowerAndThrottling::test_gpu_write_performance_limiters: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_7b-meta-llama/Meta-Llama-3.1-8B-Instruct-32-3000.0-False]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_batch_throughput: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: Failed: YOLO worker failure: [{'gpu': 1, 'status': 'error', 'error': 'CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTo
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_image_generation_diffusers_throughput: Failed: Diffusers worker failure: [{'status': 'error', 'rank': 3, 'error': 'CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug t
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_encoder_training_throughput: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_one_dma_read_4kb_validation: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_looped_rdma_mmio_dma_stress_soak: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu: Failed: TC-013 worker failures: [{'rank': 3, 'status': 'error', 'error': 'CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to 

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput[4]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_checkpoint_save_restore_throughput: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_kv_cache_memory_allocation_efficiency: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_batch_assembly_gpu_latency[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_batch_assembly_gpu_latency[4]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_batch_assembly_gpu_latency[16]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_temperature_within_safe_limits[1]: pynvml.NVMLError_ResetRequired: GPU requires restart
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0]: AssertionError: GPU 0 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[0]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[1]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[2]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_fp16_gflops_per_watt[3]: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestEnergyEfficiency::test_aggregate_4gpu_fp16_tflops_per_watt: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_int8_matmul_error_vs_fp16: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_fp8_e4m3_matmul_error_vs_fp16: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_bf16_vs_fp16_matmul_equivalence: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestDDPGradientSynchronizationCorrectness::test_ddp_gradient_sync_matches_single_gpu: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestDDPGradientSynchronizationCorrectness::test_multi_gpu_parameter_consistency_after_allreduce: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_output_matches_eager: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_event_wait_enforces_ordering: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_multi_stream_no_race_on_shared_tensor: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_cuda_event_elapsed_time_positive: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_health_and_reliability.TestCUDAStreamOrderingCorrectness::test_concurrent_stream_throughput_vs_sequential: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call

### system_config

- [failure] tests.test_system_config_report.TestGPUConfigReport::test_collect_and_print_gpu_config: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_config_report.TestGPUConfigReport::test_gpu_model_names_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_config_report.TestGPUConfigReport::test_total_gpu_memory_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_config_report.TestPCIeTopologyReport::test_pcie_topology_present_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_config_report.TestPCIeTopologyReport::test_pcie_links_healthy_in_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
- [failure] tests.test_system_config_report.TestCombinedConfigReport::test_generate_and_save_combined_report: torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=3, num_gpus=3

CUDA call
