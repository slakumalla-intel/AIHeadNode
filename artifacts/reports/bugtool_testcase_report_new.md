# Bugtool Test Report (Per Test)

- Source mapping: artifacts/reports/finalized_test_to_testcaseid_map.csv
- Source status: artifacts/reports/test_report_pass_fail_by_testcaseid.csv
- Total tests: 239
- FAIL: 62
- NOT_EXECUTED: 2
- PASS: 158
- PASS_WITH_SKIPS: 1
- SKIP: 16

## Columns

- TestCaseID | TestSelector | Status | JUnitEvidence | File | Class | TestName | MappingSource | Confidence | FinalizationStatus | Notes

## Sample (first 40)

| TestCaseID | TestSelector | Status | Evidence |
|---|---|---|---|
| AI_HOSTCPU_AGENTIC_TEST_ID_001 | tests/test_agentic_ai_validation.py::TestAgentStateManagement::test_agent_state_serialization_overhead | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_002 | tests/test_agentic_ai_validation.py::TestCPUGPUSynchronization::test_cpu_gpu_sync_latency | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_003 | tests/test_agentic_ai_validation.py::TestCPUMemoryDuringToolExecution::test_memory_usage_scaling_with_tools | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_004 | tests/test_agentic_ai_validation.py::TestLatencyPercentiles::test_agent_latency_percentiles | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_005 | tests/test_agentic_ai_validation.py::TestMultiTurnConversation::test_multi_turn_session_latency | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_006 | tests/test_agentic_ai_validation.py::TestReasoningPhaseLatency::test_reasoning_latency_by_token_count | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_007 | tests/test_agentic_ai_validation.py::TestToolExecutionParallelization::test_sequential_vs_parallel_tool_execution | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_008 | tests/test_agentic_ai_validation.py::TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability | FAIL | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_009 | tests/test_agentic_ai_validation.py::TestToolInvocationOverhead::test_single_tool_dispatch_latency | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_010 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_011 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_cpu_preprocessing_pipeline_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_012 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_013 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_014 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_015 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_016 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_continuous_batching_throughput_vllm | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_017 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_gpu_utilisation_during_inference | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_018 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_hbm_bandwidth_utilisation_during_decode | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_019 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_inference_pipeline_latency_breakdown | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_020 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_inter_token_latency | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_021 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_latency_throughput_tradeoff_curve | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_022 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_multi_gpu_inference_throughput_scaling | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_023 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_prefill_vs_decode_throughput_ratio | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_024 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_batch_assembly_gpu_latency | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_025 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_checkpoint_save_restore_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_026 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_kv_cache_memory_allocation_efficiency | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_027 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_model_load_time_to_gpu | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_028 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_029 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_030 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_storage_random_4k_read_iops | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_031 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_tcp_loopback_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_032 | tests/test_bottleneck_analysis.py::TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report | FAIL | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_033 | tests/test_bottleneck_analysis.py::TestCPUGPUPipelineBottleneck::test_data_preprocessing_not_bottleneck | PASS | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_034 | tests/test_bottleneck_analysis.py::TestCPUMemoryBandwidthBottleneck::test_cpu_dram_bandwidth_measured | PASS | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_035 | tests/test_bottleneck_analysis.py::TestCPUMemoryBandwidthBottleneck::test_cpu_dram_bandwidth_vs_hbm | PASS | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_036 | tests/test_bottleneck_analysis.py::TestGPUComputeBottleneck::test_aggregate_4gpu_fp16_tflops | FAIL | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_037 | tests/test_bottleneck_analysis.py::TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency | FAIL | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_038 | tests/test_bottleneck_analysis.py::TestHBMBandwidthBottleneck::test_hbm_bandwidth_per_gpu_vs_theoretical | FAIL | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_039 | tests/test_bottleneck_analysis.py::TestHBMBandwidthBottleneck::test_hbm_is_limiting_for_memory_bound_kernels | PASS | bottleneck.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_040 | tests/test_bottleneck_analysis.py::TestMultiGPUScalingBottleneck::test_scaling_efficiency_2_vs_1_gpu | PASS | bottleneck.xml |
