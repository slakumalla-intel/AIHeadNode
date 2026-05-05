# Bugtool Test Report (Per Test)

- Source mapping: artifacts/reports/finalized_test_to_testcaseid_map_updated.csv
- Source status: artifacts/reports/junit/*.xml
- Total tests: 410
- FAIL: 72
- ERROR: 0
- SKIP: 35
- PASS: 303
- UNMAPPED IDs generated: 46

## Columns

- TestCaseID | TestSelector | Status | JUnitEvidence | File | Class | TestName | MappingSource | Confidence | FinalizationStatus | Notes

## Sample (first 40)

| TestCaseID | TestSelector | Status | Evidence |
|---|---|---|---|
| AI_HOSTCPU_AGENTIC_TEST_ID_009 | tests/test_agentic_ai_validation.py::TestToolInvocationOverhead::test_single_tool_dispatch_latency | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_008 | tests/test_agentic_ai_validation.py::TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability | FAIL | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_006 | tests/test_agentic_ai_validation.py::TestReasoningPhaseLatency::test_reasoning_latency_by_token_count | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_002 | tests/test_agentic_ai_validation.py::TestCPUGPUSynchronization::test_cpu_gpu_sync_latency | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_007 | tests/test_agentic_ai_validation.py::TestToolExecutionParallelization::test_sequential_vs_parallel_tool_execution | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_005 | tests/test_agentic_ai_validation.py::TestMultiTurnConversation::test_multi_turn_session_latency | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_003 | tests/test_agentic_ai_validation.py::TestCPUMemoryDuringToolExecution::test_memory_usage_scaling_with_tools | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_001 | tests/test_agentic_ai_validation.py::TestAgentStateManagement::test_agent_state_serialization_overhead | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_004 | tests/test_agentic_ai_validation.py::TestLatencyPercentiles::test_agent_latency_percentiles | PASS | agentic_ai.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_014 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[32] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_014 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[128] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_014 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[512] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_014 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[2048] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_013 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_012 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput[0] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_012 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput[2] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_012 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_dataloader_cpu_to_pinned_throughput[4] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_010 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_011 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_cpu_preprocessing_pipeline_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_027 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_model_load_time_to_gpu | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_025 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_checkpoint_save_restore_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_026 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_kv_cache_memory_allocation_efficiency | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_028 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_024 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_batch_assembly_gpu_latency[1] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_024 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_batch_assembly_gpu_latency[4] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_024 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_batch_assembly_gpu_latency[16] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_FUNC_TEST_ID_026 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_storage_sequential_read_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_FUNC_TEST_ID_026 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_storage_sequential_write_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_030 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_storage_random_4k_read_iops | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_031 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_tcp_loopback_throughput | PASS | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_029 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_FUNC_TEST_ID_037 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_time_to_first_token[32] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_FUNC_TEST_ID_037 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_time_to_first_token[128] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_FUNC_TEST_ID_037 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_time_to_first_token[512] | PASS | ai_pipeline.xml |
| AI_HOSTCPU_FUNC_TEST_ID_037 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_time_to_first_token[2048] | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_020 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_inter_token_latency | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_023 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_prefill_vs_decode_throughput_ratio | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_015 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput[1] | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_015 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput[4] | FAIL | ai_pipeline.xml |
| AI_HOSTCPU_AGENTIC_TEST_ID_015 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_batched_inference_throughput[8] | FAIL | ai_pipeline.xml |
