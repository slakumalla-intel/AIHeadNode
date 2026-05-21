# Analysis Report

Generated (UTC): 2026-05-21T02:05:09.359372+00:00
Source report time (UTC): 2026-05-21T01:49:53.502642+00:00

## Executive View

- Overall pass rate: 86.60% (84/97)
- Failed domains: 3 of 5
- Total failure signatures analyzed: 13

## Failure Categories

| Category | Count |
|---|---:|
| GPU clock/power throttling | 5 |
| Model/storage bandwidth | 2 |
| Agentic parallel dispatch overhead | 1 |
| Pipeline overlap inefficiency | 1 |
| Collective communication overhead | 1 |
| CUDA runtime/assert | 1 |
| Numerical accuracy drift | 1 |
| Compiler optimization underperformance | 1 |

## Top Risks

- GPU clock/power throttling is the highest operational risk; four GPU clock ratio failures and one sustained load clock-drop failure indicate performance instability under load.
- CUDA device-side assert indicates a runtime correctness problem in inference-path tensors or kernels, blocking reliable TTFT measurements for larger input lengths.
- Extremely high all-reduce overhead shows communication dominates compute in the tested setup, limiting scale efficiency for tensor-parallel workloads.
- Model load and weight pipeline bandwidth are below floor values, increasing startup latency and reducing elasticity for model swaps.

## Recommended Actions

1. Validate GPU power management and application clocks; run sustained stress with `nvidia-smi dmon` and confirm no imposed power caps or low-P-state lock.
2. Reproduce the CUDA assert with deterministic seed and `CUDA_LAUNCH_BLOCKING=1`, then isolate failing input/tensor path in TTFT test for input_len=2048.
3. Profile collective ops (NCCL topology, rank placement, message size sweep) to reduce all-reduce overhead and rebalance compute/communication ratio.
4. Improve data/model load path (local NVMe pathing, parallel shard reads, pinned staging buffers) to raise effective GB/s above floor thresholds.

## Failing Tests Reference

| Domain | Test | Category |
|---|---|---|
| agentic_ai | tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability | Agentic parallel dispatch overhead |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency | Pipeline overlap inefficiency |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu | Model/storage bandwidth |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction | Collective communication overhead |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth | Model/storage bandwidth |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048] | CUDA runtime/assert |
| system_health | tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0] | GPU clock/power throttling |
| system_health | tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1] | GPU clock/power throttling |
| system_health | tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2] | GPU clock/power throttling |
| system_health | tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3] | GPU clock/power throttling |
| system_health | tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle | GPU clock/power throttling |
| system_health | tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16 | Numerical accuracy drift |
| system_health | tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager | Compiler optimization underperformance |
