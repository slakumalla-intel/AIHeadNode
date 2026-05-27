# Analysis Report

Generated (UTC): 2026-05-27T05:03:01.632683+00:00
Source report time (UTC): 2026-05-27T04:38:50.814131+00:00

## Executive View

- Overall pass rate: 84.21% (32/38)
- Failed domains: 1 of 1
- Total failure signatures analyzed: 6

## Failure Categories

| Category | Count |
|---|---:|
| Model/storage bandwidth | 2 |
| Other | 1 |
| Pipeline overlap inefficiency | 1 |
| Collective communication overhead | 1 |
| CUDA runtime/assert | 1 |

## Top Risks

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
| ai_pipeline | tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling | Other |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency | Pipeline overlap inefficiency |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu | Model/storage bandwidth |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction | Collective communication overhead |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth | Model/storage bandwidth |
| ai_pipeline | tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048] | CUDA runtime/assert |
