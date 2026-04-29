# Test Execution Report

- Generated: 2026-04-28 20:13:44 -0700
- Source: artifacts/reports/junit/*.xml

## Overall Summary

- Total mapped tests: 410
- Executed tests: 377
- PASS: 278
- FAIL: 99
- ERROR: 0
- SKIP: 33
- NOT_EXECUTED: 0

## Suite Summary

| Suite | Total | PASS | FAIL | ERROR | SKIP | NOT_EXECUTED |
|---|---:|---:|---:|---:|---:|---:|
| agentic_ai.xml | 9 | 8 | 1 | 0 | 0 | 0 |
| ai_pipeline.xml | 38 | 20 | 18 | 0 | 0 | 0 |
| bottleneck.xml | 25 | 21 | 4 | 0 | 0 | 0 |
| cpu.xml | 33 | 22 | 10 | 0 | 1 | 0 |
| gpu.xml | 53 | 34 | 19 | 0 | 0 | 0 |
| interconnect.xml | 77 | 71 | 2 | 0 | 4 | 0 |
| memory.xml | 28 | 20 | 8 | 0 | 0 | 0 |
| mlperf_suite.xml | 21 | 0 | 19 | 0 | 2 | 0 |
| node_validation.xml | 51 | 35 | 9 | 0 | 7 | 0 |
| performance_suite.xml | 25 | 10 | 0 | 0 | 15 | 0 |
| system_config.xml | 8 | 8 | 0 | 0 | 0 | 0 |
| system_health.xml | 39 | 28 | 8 | 0 | 3 | 0 |
| vllm_prefill.xml | 3 | 1 | 1 | 0 | 1 | 0 |

## Failures And Errors

| TestSelector | Status | Evidence | Notes |
|---|---|---|---|
| tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability | FAIL | agentic_ai.xml | AssertionError: Tool dispatch scales poorly: 16 tools take 7.5× time of 1 tool
assert 7.503821899484043 < 5.0 |
| tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling | FAIL | ai_pipeline.xml | AssertionError: 4-thread tokenizer scaling 1.41× < 2.0× single-thread
assert 1.4116155470997793 >= 2.0 |
| tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency | FAIL | ai_pipeline.xml | AssertionError: Async prefetch overlap efficiency 25.3% < 60.0%
assert 25.276616688414226 >= 60.0 |
| tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu | FAIL | ai_pipeline.xml | AssertionError: Model load bandwidth 0.28 GB/s < 1.0 GB/s
assert 0.2803094930637924 >= 1.0 |
| tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction | FAIL | ai_pipeline.xml | AssertionError: AllReduce overhead 98.7% > 20.0% of compute
assert 98.6813844844272 <= 20.0 |
| tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth | FAIL | ai_pipeline.xml | AssertionError: Weight load pipeline 0.90 GB/s < 1.0 GB/s
assert 0.899260756225 >= 1.0 |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_inter_token_latency | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_prefill_vs_decode_throughput_ratio | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[1] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[4] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[8] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[16] | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_gpu_utilisation_during_inference | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_hbm_bandwidth_utilisation_during_decode | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_continuous_batching_throughput_vllm | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_latency_throughput_tradeoff_curve | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_inference_pipeline_latency_breakdown | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_multi_gpu_inference_throughput_scaling | FAIL | ai_pipeline.xml | torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be a |
| tests.test_bottleneck_analysis.TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report | FAIL | bottleneck.xml | AssertionError: CPU FP32 (1 process, BLAS threads=224): efficiency 7.12% is below threshold 70.00% (measured=1.940, theoretical=27.238)
assert 0.07123810428453564 >= 0.7 |
| tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency | FAIL | bottleneck.xml | AssertionError: GPU 0 FP16 Tensor Core: efficiency 31.55% is below threshold 75.00% (measured=477.345, theoretical=1513.000)
assert 0.3154956695894 >= 0.75 |
| tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_aggregate_4gpu_fp16_tflops | FAIL | bottleneck.xml | AssertionError: 4-GPU aggregate FP16 TFLOPS: efficiency 29.90% is below threshold 75.00% (measured=1809.681, theoretical=6052.000)
assert 0.2990220273473491 >= 0.75 |
| tests.test_bottleneck_analysis.TestPCIeBottleneck::test_pcie_bottleneck_ratio_vs_hbm | FAIL | bottleneck.xml | AssertionError: HBM/PCIe ratio 31.7× < 40× – PCIe may not be the ingestion bottleneck
assert 31.746031746031747 >= 40.0 |
| tests.test_cpu_validation.TestSingleCoreFP32::test_single_core_gflops_above_floor | FAIL | cpu.xml | AttributeError: 'CPUSpecs' object has no attribute 'avx512_fp32_flops_per_cycle_per_core'. Did you mean: 'simd_fp32_flops_per_cycle_per_core'? |
| tests.test_cpu_validation.TestAllCoreFP32Scaling::test_all_core_fp32_gflops | FAIL | cpu.xml | AssertionError: All-core FP32 GFLOPS (BLAS): efficiency 25.80% is below threshold 70.00% (measured=7028.528, theoretical=27238.400)
assert 0.258037475450912 >= 0.7 |
| tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[4] | FAIL | cpu.xml | AssertionError: Parallel speedup at 4 cores: efficiency 27.91% is below threshold 85.00% (measured=285.266, theoretical=1022.248)
assert 0.27905769461164864 >= 0.85 |
| tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[8] | FAIL | cpu.xml | AssertionError: Parallel speedup at 8 cores: efficiency 0.28% is below threshold 85.00% (measured=6.052, theoretical=2192.089)
assert 0.0027607357427014375 >= 0.85 |
| tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[16] | FAIL | cpu.xml | AssertionError: Parallel speedup at 16 cores: efficiency 0.76% is below threshold 85.00% (measured=4.644, theoretical=614.420)
assert 0.007558764827011876 >= 0.85 |
| tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[32] | FAIL | cpu.xml | AssertionError: Parallel speedup at 32 cores: efficiency 0.05% is below threshold 85.00% (measured=4.480, theoretical=8707.313)
assert 0.000514498289784649 >= 0.85 |
| tests.test_cpu_validation.TestParallelSpeedup::test_parallel_speedup_at_n_cores[64] | FAIL | cpu.xml | AssertionError: Parallel speedup at 64 cores: efficiency 0.02% is below threshold 85.00% (measured=4.467, theoretical=18280.312)
assert 0.0002443367377898264 >= 0.85 |
| tests.test_cpu_validation.TestIntegerThroughput::test_integer_vector_ops_throughput | FAIL | cpu.xml | AssertionError: Concurrent INT64 vector add throughput 17.75 GOPS < 56.00 GOPS (224 threads × 0.25 GOPS/thread floor)
assert 17.754507680559854 >= 56.0 |
| tests.test_cpu_validation.TestIntegerThroughput::test_integer_multiply_throughput | FAIL | cpu.xml | AssertionError: Concurrent INT64 multiply throughput 46.19 GOPS < 56.00 GOPS (224 threads × 0.25 GOPS/thread floor)
assert 46.19014561618231 >= 56.0 |
| tests.test_cpu_validation.TestSMTThroughput::test_smt_provides_throughput_gain | FAIL | cpu.xml | AssertionError: SMT run (5.6 GFLOPS) not faster than physical-only run (5.8 GFLOPS)
assert 5.643508040402071 > 5.827205667684056 |
| tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[0] | FAIL | gpu.xml | AssertionError: FP32 GFLOPS GPU 0: efficiency 39.82% is below threshold 70.00% (measured=26682.629, theoretical=67000.000)
assert 0.39824819434211944 >= 0.7 |
| tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[1] | FAIL | gpu.xml | AssertionError: FP32 GFLOPS GPU 1: efficiency 55.19% is below threshold 70.00% (measured=36979.140, theoretical=67000.000)
assert 0.5519274631829902 >= 0.7 |
| tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[2] | FAIL | gpu.xml | AssertionError: FP32 GFLOPS GPU 2: efficiency 52.96% is below threshold 70.00% (measured=35485.605, theoretical=67000.000)
assert 0.5296358992441652 >= 0.7 |
| tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[3] | FAIL | gpu.xml | AssertionError: FP32 GFLOPS GPU 3: efficiency 68.81% is below threshold 70.00% (measured=46105.069, theoretical=67000.000)
assert 0.6881353515537373 >= 0.7 |
| tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[0] | FAIL | gpu.xml | AssertionError: FP16 TFLOPS GPU 0: efficiency 31.60% is below threshold 75.00% (measured=478.077, theoretical=1513.000)
assert 0.31597932541290413 >= 0.75 |
| tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[1] | FAIL | gpu.xml | AssertionError: FP16 TFLOPS GPU 1: efficiency 31.46% is below threshold 75.00% (measured=476.036, theoretical=1513.000)
assert 0.3146305751378209 >= 0.75 |
| tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[2] | FAIL | gpu.xml | AssertionError: FP16 TFLOPS GPU 2: efficiency 28.37% is below threshold 75.00% (measured=429.287, theoretical=1513.000)
assert 0.2837324745822251 >= 0.75 |
| tests.test_gpu_validation.TestFP16ThroughputGPU::test_fp16_tflops_per_gpu[3] | FAIL | gpu.xml | AssertionError: FP16 TFLOPS GPU 3: efficiency 29.23% is below threshold 75.00% (measured=442.254, theoretical=1513.000)
assert 0.2923026360686367 >= 0.75 |
| tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[0] | FAIL | gpu.xml | AssertionError: BF16 TFLOPS GPU 0: efficiency 25.91% is below threshold 75.00% (measured=512.683, theoretical=1979.000)
assert 0.2590614847041806 >= 0.75 |
| tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[1] | FAIL | gpu.xml | AssertionError: BF16 TFLOPS GPU 1: efficiency 25.86% is below threshold 75.00% (measured=511.764, theoretical=1979.000)
assert 0.25859739908194807 >= 0.75 |
| tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[2] | FAIL | gpu.xml | AssertionError: BF16 TFLOPS GPU 2: efficiency 23.49% is below threshold 75.00% (measured=464.873, theoretical=1979.000)
assert 0.23490274390623062 >= 0.75 |
| tests.test_gpu_validation.TestBF16Throughput::test_bf16_tflops_per_gpu[3] | FAIL | gpu.xml | AssertionError: BF16 TFLOPS GPU 3: efficiency 24.38% is below threshold 75.00% (measured=482.555, theoretical=1979.000)
assert 0.2438378020096712 >= 0.75 |
| tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[0] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 0: efficiency 13.86% is below threshold 75.00% (measured=548.540, theoretical=3958.000)
assert 0.1385900801983928 >= 0.75 |
| tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[1] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 1: efficiency 13.90% is below threshold 75.00% (measured=550.218, theoretical=3958.000)
assert 0.1390142313707581 >= 0.75 |
| tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[2] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 2: efficiency 12.73% is below threshold 75.00% (measured=503.761, theoretical=3958.000)
assert 0.1272767422092992 >= 0.75 |
| tests.test_gpu_validation.TestINT8Throughput::test_int8_tops_per_gpu[3] | FAIL | gpu.xml | AssertionError: INT8-proxy TOPS GPU 3: efficiency 13.69% is below threshold 75.00% (measured=541.832, theoretical=3958.000)
assert 0.13689541726492968 >= 0.75 |
| tests.test_gpu_validation.TestGPUECC::test_gpu_ecc_enabled | FAIL | gpu.xml | AssertionError: GPU 0: ECC is not enabled (mode=[1, 1])
assert [1, 1] == 1
 +  where 1 = pynvml.NVML_FEATURE_ENABLED |
| tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_gpu_pci_bus_numbers_are_unique | FAIL | gpu.xml | AssertionError: GPU 2: PCIe bus number 0x61 collides with another GPU
assert 97 not in [97, 177] |
| tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_host_to_device_bandwidth[0] | FAIL | gpu.xml | AssertionError: PCIe H2D bandwidth GPU 0: efficiency 37.71% is below threshold 65.00% (measured=23.756, theoretical=63.000)
assert 0.37708500670934236 >= 0.65 |
| tests.test_interconnect_validation.TestPCIeTopology::test_pcie_gen5_sysfs | FAIL | interconnect.xml | AssertionError: No PCIe Gen 5 (32 GT/s) links found in sysfs
assert False |
| tests.test_interconnect_validation.TestCPUGPUPipelineOverlap::test_compute_transfer_overlap[2] | FAIL | interconnect.xml | AssertionError: GPU 2: overlapped (0.112s) not faster than serial (0.063s)
assert 0.112324768997496 < 0.06313897999643814 |
| tests.test_memory_validation.TestRAMCapacity::test_total_ram_at_least_1tb | FAIL | memory.xml | AssertionError: Total RAM 1007 GB < 1 TB
assert 1006.9956436157227 >= 1024.0 |
| tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs | FAIL | memory.xml | AssertionError: STREAM Scale 12.20 GB/s < 15 GB/s
assert 12.204042875095416 >= 15.0 |
| tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs | FAIL | memory.xml | AssertionError: STREAM Triad 11.46 GB/s < 15 GB/s
assert 11.456160813855783 >= 15.0 |
| tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[32-4.0] | FAIL | memory.xml | AssertionError: Working-set 32 KB: latency 89.21 ns > 4.0 ns
assert 89.21459837551993 <= 4.0 |
| tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[512-8.0] | FAIL | memory.xml | AssertionError: Working-set 512 KB: latency 83.13 ns > 8.0 ns
assert 83.12736499149764 <= 8.0 |
| tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[16384-30.0] | FAIL | memory.xml | AssertionError: Working-set 16384 KB: latency 195.21 ns > 30.0 ns
assert 195.21187990903854 <= 30.0 |
| tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[524288-120.0] | FAIL | memory.xml | AssertionError: Working-set 524288 KB: latency 226.34 ns > 120.0 ns
assert 226.3421899988316 <= 120.0 |
| tests.test_memory_validation.TestCacheHierarchyBandwidth::test_bandwidth_at_working_set[L3-cache-8.0-20.0] | FAIL | memory.xml | AssertionError: L3-cache: measured 17.2 GB/s < expected 20.0 GB/s
assert 17.195346603123824 >= 20.0 |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003] | FAIL | mlperf_suite.xml | Failed: Timeout (>600.0s) from pytest-timeout. |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004] | FAIL | mlperf_suite.xml | Failed: Timeout (>600.0s) from pytest-timeout. |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005] | FAIL | mlperf_suite.xml | Failed: Timeout (>600.0s) from pytest-timeout. |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_006: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch resnet50 gpu --scenario Offline --da |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_007: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch retinanet gpu --scenario Offline --d |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_008: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/medical_imaging/3d-unet-kits19/run_local.sh pytorch gpu --scenario Offline --dataset-path /ro |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_009: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/speech_recognition/rnnt/run_local.sh pytorch rnnt gpu --scenario Offline --dataset-path /root |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_010: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/speech_recognition/whisper/run_local.sh pytorch gpu --scenario Offline --dataset-path /root/w |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_011: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/recommendation/dlrm_v2/pytorch/run.py --scenario Offline --dataset-path /root/workloads/AIHea |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_012: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/text_to_image/run_local.sh pytorch gpu --scenario Offline --dataset-path /root/workloads/AIHe |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_013: command failed (rc=2)
executed_command: /usr/bin/python /root/workloads/mlcommons/inference/graph/R-GAT/run_local.sh pytorch gpu --scenario Offline --dataset-path /root/workloads/AIHead |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_014: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/gpt-j/run_training.py --model-name EleutherAI/gpt-j-6b --dataset-pa |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_015: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_squad.py --bert_model bert-large-uncased --do_trai |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_016: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/image_segmentation/pytorch/train.py --data-dir /root/workloads/AIHeadNode/datasets |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_017: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/stable_diffusion/main.py --base /root/workloads/mlcommons/training/stable_diffusio |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_018: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_pretraining.py --bert_model bert-large-uncased --d |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_019: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/large_language_model/megatron-lm/pretrain_gpt.py --tensor-model-parallel-size 4 -- |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_020: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/llm_finetune/run_lora_finetune.py --model_name_or_path meta-llama/Llama-2-70b-hf - |
| tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021] | FAIL | mlperf_suite.xml | Failed: AI_HOSTCPU_ML_TEST_ID_021: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/recommendation/dlrm_v2/pytorch/dlrm_s_pytorch.py --arch-sparse-feature-size=128 -- |
| tests.test_node_validation.TestNVBandwidthMemcpy::test_host_to_device_bidirectional_memcpy_ce | FAIL | node_validation.xml | AssertionError: host_to_device_bidirectional_memcpy_ce: min BW 37.3 GB/s below threshold 40.0 GB/s
assert 37.32 >= 40.0 |
| tests.test_node_validation.TestNVBandwidthMemcpy::test_device_to_host_bidirectional_memcpy_ce | FAIL | node_validation.xml | AssertionError: device_to_host_bidirectional_memcpy_ce: min BW 36.5 GB/s below threshold 40.0 GB/s
assert 36.5 >= 40.0 |
| tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary | FAIL | node_validation.xml | AssertionError: 1 P2P bandwidth values below 40.0 GB/s: ['17.3']
assert not [17.26] |
| tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback | FAIL | node_validation.xml | Failed: P2P bandwidth fallback validation failed.
threshold=40.0 GB/s, measured_pairs=12, inaccessible_pairs=0
Failures:
GPU1->GPU0: 37.9 GB/s < 40.0 GB/s
GPU3->GPU0: 36.9 GB/s < 40.0 GB/s |
| tests.test_node_validation.TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability | FAIL | node_validation.xml | AssertionError: Power variance 57.9 W is too high (expected < 50 W)
assert np.float64(57.8951628472031) < 50 |
| tests.test_node_validation.TestNCCLAndNVLinkValidation::test_nccl_p2p_perf | FAIL | node_validation.xml | AssertionError: NCCL sendrecv busbw 25.1 GB/s too low
assert 25.13 >= 30.0 |
| tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill | FAIL | node_validation.xml | AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%, target_ratio=0.50, shared_gpu=True)
assert 0.0 >= 0.5 |
| tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload | FAIL | node_validation.xml | AssertionError: Offload throughput degradation too high: 93.2% (baseline 663.5 -> offload 45.1)
assert 93.20332579358124 <= 60 |
| tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu | FAIL | node_validation.xml | AssertionError: TC-013: gradient sync ratio 21.1% exceeds threshold 15% — communication bottleneck detected; check NVLink status and NCCL topology
assert 0.2113 <= 0.15 |
| tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0] | FAIL | system_health.xml | AssertionError: GPU 0 clock ratio 21.37% below 80%; throttling suspected (current 375 MHz, max 1755 MHz)
assert 0.21367521367521367 >= 0.8 |
| tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1] | FAIL | system_health.xml | AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8 |
| tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2] | FAIL | system_health.xml | AssertionError: GPU 2 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8 |
| tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3] | FAIL | system_health.xml | AssertionError: GPU 3 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8 |
| tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle | FAIL | system_health.xml | AssertionError: SM clock dropped 49.6% during sustained load (initial 1755 MHz → min 885 MHz, max_temp 60 °C). Thermal or power-limit throttling detected.
assert 0.49572649572649574 <= 0.1 |
| tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_power_limit_headroom | FAIL | system_health.xml | AssertionError: GPU(s) exceeding 90 % power limit at rest: GPU0 98%
assert not [(0, {'power_w': 343.825, 'limit_w': 350.0, 'ratio': 0.9823571428571428})] |
| tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16 | FAIL | system_health.xml | AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03
assert 0.0025634765625 < 0.001 |
| tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager | FAIL | system_health.xml | AssertionError: torch.compile speedup 0.808× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8081042771166228 >= 1.05 |
| tests.test_vllm_prefill_validation.TestVLLMPrefillValidation::test_vllm_4h100_prefill_validation | FAIL | vllm_prefill.xml | Failed: Timeout (>600.0s) from pytest-timeout. |
