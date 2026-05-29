# Workload Execution Status Report

Generated (UTC): 2026-05-29T11:19:59.347864+00:00
Host: sc00901168s0070

## Summary

- Domains run: 14
- Passed: 8
- Failed: 6
- Total tests: 421
- Failures: 44
- Errors: 0
- Skipped: 98

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log | KPI Table |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| system_config | PASS | 8 | 0 | 0 | 0 | 4.211 | artifacts/reports/logs/full_logs/system_config_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/system_config_20260529_110624_831109.log | - |
| cpu | PASS | 33 | 0 | 0 | 1 | 227.514 | artifacts/reports/logs/full_logs/cpu_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/cpu_20260529_110624_831109.log | - |
| memory | PASS | 28 | 0 | 0 | 0 | 21.095 | artifacts/reports/logs/full_logs/memory_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/memory_20260529_110624_831109.log | - |
| bottleneck | FAIL | 25 | 1 | 0 | 0 | 41.363 | artifacts/reports/logs/full_logs/bottleneck_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/bottleneck_20260529_110624_831109.log | - |
| gpu | FAIL | 53 | 2 | 0 | 0 | 15.299 | artifacts/reports/logs/full_logs/gpu_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/gpu_20260529_110624_831109.log | - |
| interconnect | PASS | 77 | 0 | 0 | 16 | 91.041 | artifacts/reports/logs/full_logs/interconnect_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/interconnect_20260529_110624_831109.log | - |
| system_health | FAIL | 39 | 7 | 0 | 3 | 41.662 | artifacts/reports/logs/full_logs/system_health_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/system_health_20260529_110624_831109.log | artifacts/reports/logs/full_logs/system_health_20260529_110624_831109_kpi_table.md |
| cpu_headnode | PASS | 10 | 0 | 0 | 3 | 40.453 | artifacts/reports/logs/full_logs/cpu_headnode_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/cpu_headnode_20260529_110624_831109.log | - |
| agentic_ai | PASS | 9 | 0 | 0 | 0 | 18.133 | artifacts/reports/logs/full_logs/agentic_ai_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/agentic_ai_20260529_110624_831109.log | artifacts/reports/logs/full_logs/agentic_ai_20260529_110624_831109_kpi_table.md |
| ai_pipeline | FAIL | 38 | 23 | 0 | 2 | 22.65 | artifacts/reports/logs/full_logs/ai_pipeline_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/ai_pipeline_20260529_110624_831109.log | artifacts/reports/logs/full_logs/ai_pipeline_20260529_110624_831109_kpi_table.md |
| vllm_prefill | PASS | 3 | 0 | 0 | 3 | 2.637 | artifacts/reports/logs/full_logs/vllm_prefill_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/vllm_prefill_20260529_110624_831109.log | - |
| node_validation | FAIL | 51 | 10 | 0 | 24 | 111.603 | artifacts/reports/logs/full_logs/node_validation_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/node_validation_20260529_110624_831109.log | artifacts/reports/logs/full_logs/node_validation_20260529_110624_831109_kpi_table.md |
| performance_suite | FAIL | 26 | 1 | 0 | 25 | 73.046 | artifacts/reports/logs/full_logs/performance_suite_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/performance_suite_20260529_110624_831109.log | artifacts/reports/logs/full_logs/performance_suite_20260529_110624_831109_kpi_table.md |
| mlperf_suite | PASS | 21 | 0 | 0 | 21 | 17.677 | artifacts/reports/logs/full_logs/mlperf_suite_20260529_110624_831109.xml | artifacts/reports/logs/full_logs/mlperf_suite_20260529_110624_831109.log | - |

## Failure Details

### bottleneck

- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency: AssertionError: GPU 2 FP16 Tensor Core: efficiency 57.64% is below threshold 60.00% (measured=435.748, theoretical=756.000)
assert 0.576386167139651 >= 0.6

### gpu

- [failure] tests.test_gpu_validation.TestFP32Throughput::test_fp32_gflops_per_gpu[2]: AssertionError: FP32 GFLOPS GPU 2: efficiency 42.49% is below threshold 65.00% (measured=28469.662, theoretical=67000.000)
assert 0.42492032955201925 >= 0.65
- [failure] tests.test_gpu_validation.TestGPUEnumerationPCIeTopology::test_pcie_host_to_device_bandwidth[1]: AssertionError: PCIe H2D bandwidth GPU 1: efficiency 37.18% is below threshold 65.00% (measured=23.426, theoretical=63.000)
assert 0.37184246100701396 >= 0.65

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0]: AssertionError: GPU 0 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1]: AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 46.2% during sustained load (initial 1755 MHz → min 945 MHz, max_temp 58 °C). Thermal or power-limit throttling detected.
assert 0.46153846153846156 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03
assert 0.0025634765625 < 0.001
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.808× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8080083039376981 >= 1.05

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[32]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[128]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[512]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_throughput_by_seq_len[2048]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_cpu_preprocessing_pipeline_throughput: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: AssertionError: Weight load pipeline 0.88 GB/s < 1.0 GB/s
assert 0.8830925332512203 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[32]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[128]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[512]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_inter_token_latency: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_prefill_vs_decode_throughput_ratio: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[1]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[4]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[8]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_batched_inference_throughput[16]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_gpu_utilisation_during_inference: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_hbm_bandwidth_utilisation_during_decode: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_latency_throughput_tradeoff_curve: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_inference_pipeline_latency_breakdown: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_multi_gpu_inference_throughput_scaling: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli

### node_validation

- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[deepseek_r1_quantized-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-32-3000.0-True]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_7b-meta-llama/Meta-Llama-3.1-8B-Instruct-32-3000.0-False]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_70b-meta-llama/Meta-Llama-3.1-70B-Instruct-16-12000.0-True]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[qwen_70b-Qwen/Qwen2.5-72B-Instruct-16-12000.0-True]: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_batch_throughput: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: AssertionError: YOLO aggregate FPS too low: 92.1 < target 200.0
assert 92.09325219865188 >= 200.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_image_generation_diffusers_throughput: Failed: Diffusers worker failure: [{'status': 'error', 'rank': 3, 'error': 'stabilityai/sd-turbo does not appear to have a file named model_index.json.'}, {'status': 'error', 'rank': 2, 'error': 'stabilityai/sd-turbo does not appear to have a file named model_
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_encoder_training_throughput: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offli
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_large_multigpu: deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)

### performance_suite

- [failure] tests.test_performance_suite::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]: Failed: Command failed for AI_HOSTCPU_PERF_TEST_ID_011: genai-perf profile --endpoint-type chat --service-kind openai --url http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
executed_command:
genai-perf profile --endpoint-type chat --service-ki
