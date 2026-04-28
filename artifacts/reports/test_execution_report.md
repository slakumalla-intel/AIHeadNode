# Test Execution Report

- Generated: 2026-04-27 22:41:51 -07:00
- Source: artifacts/reports/bugtool_testcase_report_new.csv

## Overall Summary

- Total mapped tests: 285
- Executed tests: 246
- PASS: 152
- FAIL: 62
- ERROR: 0
- SKIP: 32
- NOT_EXECUTED: 39

## Suite Summary

| Suite | Total | PASS | FAIL | ERROR | SKIP | NOT_EXECUTED |
|---|---:|---:|---:|---:|---:|---:|
| none | 39 | 0 | 0 | 0 | 0 | 39 |
| agentic_ai.xml | 9 | 8 | 0 | 0 | 0 | 0 |
| ai_pipeline.xml | 20 | 7 | 13 | 0 | 0 | 0 |
| bottleneck.xml | 20 | 16 | 4 | 0 | 0 | 0 |
| cpu.xml | 26 | 21 | 4 | 0 | 0 | 0 |
| gpu.xml | 21 | 19 | 2 | 0 | 0 | 0 |
| interconnect.xml | 7 | 2 | 0 | 0 | 4 | 0 |
| memory.xml | 20 | 16 | 4 | 0 | 0 | 0 |
| mlperf_suite.xml | 21 | 0 | 19 | 0 | 2 | 0 |
| node_validation.xml | 47 | 32 | 9 | 0 | 6 | 0 |
| performance_suite.xml | 25 | 10 | 0 | 0 | 15 | 0 |
| system_config.xml | 8 | 8 | 0 | 0 | 0 | 0 |
| system_health.xml | 19 | 12 | 4 | 0 | 3 | 0 |
| vllm_prefill.xml | 3 | 0 | 0 | 0 | 0 | 0 |

## Failures And Errors

| TestCaseID | TestSelector | Status | Evidence | Notes |
|---|---|---|---|---|
| AI_HOSTCPU_AGENTIC_TEST_ID_008 | tests/test_agentic_ai_validation.py::TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability | FAIL | agentic_ai.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_010 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_013 | tests/test_ai_pipeline_kpis.py::TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_016 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_continuous_batching_throughput_vllm | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_017 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_gpu_utilisation_during_inference | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_018 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_hbm_bandwidth_utilisation_during_decode | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_019 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_inference_pipeline_latency_breakdown | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_020 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_inter_token_latency | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_021 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_latency_throughput_tradeoff_curve | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_022 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_multi_gpu_inference_throughput_scaling | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_023 | tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_prefill_vs_decode_throughput_ratio | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_027 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_model_load_time_to_gpu | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_028 | tests/test_ai_pipeline_kpis.py::TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_029 | tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth | FAIL | ai_pipeline.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_032 | tests/test_bottleneck_analysis.py::TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report | FAIL | bottleneck.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_036 | tests/test_bottleneck_analysis.py::TestGPUComputeBottleneck::test_aggregate_4gpu_fp16_tflops | FAIL | bottleneck.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_037 | tests/test_bottleneck_analysis.py::TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency | FAIL | bottleneck.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_044 | tests/test_bottleneck_analysis.py::TestPCIeBottleneck::test_pcie_bottleneck_ratio_vs_hbm | FAIL | bottleneck.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_053 | tests/test_cpu_validation.py::TestAllCoreFP32Scaling::test_all_core_fp32_gflops | FAIL | cpu.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_064 | tests/test_cpu_validation.py::TestIntegerThroughput::test_integer_multiply_throughput | FAIL | cpu.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_065 | tests/test_cpu_validation.py::TestIntegerThroughput::test_integer_vector_ops_throughput | FAIL | cpu.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_076 | tests/test_cpu_validation.py::TestSingleCoreFP32::test_single_core_gflops_above_floor | FAIL | cpu.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_088 | tests/test_gpu_validation.py::TestGPUECC::test_gpu_ecc_enabled | FAIL | gpu.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_092 | tests/test_gpu_validation.py::TestGPUEnumerationPCIeTopology::test_gpu_pci_bus_numbers_are_unique | FAIL | gpu.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_126 | tests/test_interconnect_validation.py::TestPCIeTopology::test_pcie_gen5_sysfs | FAIL | interconnect.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_143 | tests/test_memory_validation.py::TestRAMCapacity::test_total_ram_at_least_1tb | FAIL | memory.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_147 | tests/test_memory_validation.py::TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs | FAIL | memory.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_148 | tests/test_memory_validation.py::TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs | FAIL | memory.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_150 | tests/test_memory_validation.py::TestSustainedBandwidth::test_bandwidth_does_not_degrade | FAIL | memory.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_ML_TEST_ID_003 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003] | FAIL | mlperf_suite.xml | MLPerf suite case 3 |
| AI_HOSTCPU_ML_TEST_ID_004 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004] | FAIL | mlperf_suite.xml | MLPerf suite case 4 |
| AI_HOSTCPU_ML_TEST_ID_005 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005] | FAIL | mlperf_suite.xml | MLPerf suite case 5 |
| AI_HOSTCPU_ML_TEST_ID_006 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006] | FAIL | mlperf_suite.xml | MLPerf suite case 6 |
| AI_HOSTCPU_ML_TEST_ID_007 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007] | FAIL | mlperf_suite.xml | MLPerf suite case 7 |
| AI_HOSTCPU_ML_TEST_ID_008 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008] | FAIL | mlperf_suite.xml | MLPerf suite case 8 |
| AI_HOSTCPU_ML_TEST_ID_009 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009] | FAIL | mlperf_suite.xml | MLPerf suite case 9 |
| AI_HOSTCPU_ML_TEST_ID_010 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010] | FAIL | mlperf_suite.xml | MLPerf suite case 10 |
| AI_HOSTCPU_ML_TEST_ID_011 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011] | FAIL | mlperf_suite.xml | MLPerf suite case 11 |
| AI_HOSTCPU_ML_TEST_ID_012 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012] | FAIL | mlperf_suite.xml | MLPerf suite case 12 |
| AI_HOSTCPU_ML_TEST_ID_013 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013] | FAIL | mlperf_suite.xml | MLPerf suite case 13 |
| AI_HOSTCPU_ML_TEST_ID_014 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014] | FAIL | mlperf_suite.xml | MLPerf suite case 14 |
| AI_HOSTCPU_ML_TEST_ID_015 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015] | FAIL | mlperf_suite.xml | MLPerf suite case 15 |
| AI_HOSTCPU_ML_TEST_ID_016 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016] | FAIL | mlperf_suite.xml | MLPerf suite case 16 |
| AI_HOSTCPU_ML_TEST_ID_017 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017] | FAIL | mlperf_suite.xml | MLPerf suite case 17 |
| AI_HOSTCPU_ML_TEST_ID_018 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018] | FAIL | mlperf_suite.xml | MLPerf suite case 18 |
| AI_HOSTCPU_ML_TEST_ID_019 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019] | FAIL | mlperf_suite.xml | MLPerf suite case 19 |
| AI_HOSTCPU_ML_TEST_ID_020 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020] | FAIL | mlperf_suite.xml | MLPerf suite case 20 |
| AI_HOSTCPU_ML_TEST_ID_021 | tests/test_mlperf_suite.py::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021] | FAIL | mlperf_suite.xml | MLPerf suite case 21 |
| AI_HOSTCPU_FUNC_TEST_ID_007 | tests/test_node_validation.py::TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_005 | tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_005 | tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_052 | tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_multi_stream_bandwidth | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_008 | tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_p2p_perf | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_003 | tests/test_node_validation.py::TestPCIeBandwidthDirect::test_d2h_bandwidth_all_gpus | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_013 | tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_012 | tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_010 | tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill | FAIL | node_validation.xml | - |
| AI_HOSTCPU_FUNC_TEST_ID_027 | tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_power_limit_headroom | FAIL | system_health.xml | - |
| AI_HOSTCPU_AGENTIC_TEST_ID_181 | tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle | FAIL | system_health.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_185 | tests/test_system_health_and_reliability.py::TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16 | FAIL | system_health.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_187 | tests/test_system_health_and_reliability.py::TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager | FAIL | system_health.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
| AI_HOSTCPU_AGENTIC_TEST_ID_188 | tests/test_vllm_prefill_validation.py::TestVLLMPrefillValidation::test_vllm_4h100_prefill_validation | FAIL | vllm_prefill.xml | Assigned via AI_HOSTCPU_AGENTIC_TEST_ID catalog logic |
