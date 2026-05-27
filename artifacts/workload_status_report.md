# Workload Execution Status Report

Generated (UTC): 2026-05-27T04:38:50.814131+00:00
Host: sc00901168s0075

## Summary

- Domains run: 1
- Passed: 0
- Failed: 1
- Total tests: 38
- Failures: 6
- Errors: 0
- Skipped: 12

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| ai_pipeline | FAIL | 38 | 6 | 0 | 12 | 35.702 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |

## Failure Details

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling: AssertionError: 4-thread tokenizer scaling 0.91× < 2.0× single-thread
assert 0.9138791827999527 >= 2.0
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: AssertionError: Async prefetch overlap efficiency 25.0% < 60.0%
assert 25.02408713555639 >= 60.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: AssertionError: Model load bandwidth 0.30 GB/s < 1.0 GB/s
assert 0.30320926109528845 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction: AssertionError: AllReduce overhead 98.7% > 20.0% of compute
assert 98.71310505800027 <= 20.0
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: AssertionError: Weight load pipeline 0.89 GB/s < 1.0 GB/s
assert 0.8872573157490499 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API
