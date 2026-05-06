# Industry Standard KPI Report

Generated: 2026-05-05 19:02:41
Source: alltests.log
Scope: Fresh execution-only report built directly from the current log output

## Executive Summary

- Total tests: 410
- Executed tests: 375
- Passed: 303
- Failed: 72
- Skipped: 35
- Pass rate: 80.8%
- Performance KPIs parsed: 78
- Threshold-based workload KPIs parsed: 24
- Model/workload KPI payloads parsed: 14

## Domain Scorecard

| Domain | Tests | Pass | Fail | Skip | Pass Rate | Status |
|---|---:|---:|---:|---:|---:|---|
| agentic_ai | 9 | 8 | 1 | 0 | 88.9% | AT_RISK |
| ai_pipeline | 38 | 20 | 17 | 1 | 54.1% | AT_RISK |
| bottleneck | 25 | 24 | 1 | 0 | 96.0% | AT_RISK |
| cpu | 33 | 31 | 1 | 1 | 96.9% | AT_RISK |
| gpu | 53 | 44 | 9 | 0 | 83.0% | AT_RISK |
| interconnect | 77 | 72 | 1 | 4 | 98.6% | AT_RISK |
| memory | 28 | 21 | 7 | 0 | 75.0% | AT_RISK |
| mlperf_suite | 21 | 0 | 19 | 2 | 0.0% | AT_RISK |
| node_validation | 51 | 36 | 7 | 8 | 83.7% | AT_RISK |
| performance_suite | 25 | 9 | 1 | 15 | 90.0% | AT_RISK |
| system_config | 8 | 8 | 0 | 0 | 100.0% | PASS |
| system_health | 39 | 28 | 8 | 3 | 77.8% | AT_RISK |
| vllm_prefill | 3 | 2 | 0 | 1 | 100.0% | PASS |

## Critical Findings

1. mlperf_suite: 19 failing tests out of 21
2. ai_pipeline: 17 failing tests out of 38
3. gpu: 9 failing tests out of 53
4. system_health: 8 failing tests out of 39
5. memory: 7 failing tests out of 28

## Performance KPI Scorecard

| Domain | KPI | Measured | Theoretical | Efficiency | Threshold | Status |
|---|---|---:|---:|---:|---:|---|
| bottleneck | CPU FP32 (1 processes × 32 threads) | 1.957 | 27.238 | 7.18% | 5.00% | PASS |
| bottleneck | CPU FP32 (4 processes × 52 threads) | 2.674 | 27.238 | 9.82% | 5.00% | PASS |
| bottleneck | CPU DRAM BW (multi-thread STREAM Copy) | 1477.639 | 307.200 | 481.00% | 75.00% | PASS |
| bottleneck | GPU 0 FP16 Tensor Core | 472.943 | 756.000 | 62.56% | 60.00% | PASS |
| bottleneck | GPU 1 FP16 Tensor Core | 478.443 | 756.000 | 63.29% | 60.00% | PASS |
| bottleneck | GPU 2 FP16 Tensor Core | 276.047 | 756.000 | 36.51% | 60.00% | FAIL |
| bottleneck | 4-GPU aggregate FP16 TFLOPS | 1829.814 | 3024.000 | 60.51% | 60.00% | PASS |
| bottleneck | GPU 0 HBM3 bandwidth | 1.853 | 2.000 | 92.65% | 80.00% | PASS |
| bottleneck | GPU 1 HBM3 bandwidth | 1.853 | 2.000 | 92.67% | 80.00% | PASS |
| bottleneck | GPU 2 HBM3 bandwidth | 1.855 | 2.000 | 92.75% | 80.00% | PASS |
| bottleneck | GPU 3 HBM3 bandwidth | 3.557 | 2.000 | 177.83% | 80.00% | PASS |
| bottleneck | GPU 0 H2D PCIe bandwidth | 53.736 | 63.000 | 85.30% | 65.00% | PASS |
| bottleneck | GPU 1 H2D PCIe bandwidth | 55.611 | 63.000 | 88.27% | 65.00% | PASS |
| bottleneck | GPU 2 H2D PCIe bandwidth | 56.044 | 63.000 | 88.96% | 65.00% | PASS |
| bottleneck | GPU 3 H2D PCIe bandwidth | 55.933 | 63.000 | 88.78% | 65.00% | PASS |
| bottleneck | PCIe P2P P2P BW GPU 0->1 | 51.811 | 63.000 | 82.24% | 50.00% | PASS |
| bottleneck | 2-GPU FP16 scaling efficiency | 905.199 | 845.736 | 107.03% | 80.00% | PASS |
| bottleneck | 4-GPU FP16 scaling efficiency | 1799.163 | 1770.729 | 101.61% | 80.00% | PASS |
| cpu | Single-core FP32 GFLOPS | 945.980 | 121.600 | 777.94% | 35.00% | PASS |
| cpu | All-core FP32 GFLOPS (BLAS) | 9268.680 | 27238.400 | 34.03% | 20.00% | PASS |
| cpu | FP32 scaling efficiency (8 cores) | 310.911 | 293.647 | 105.88% | 85.00% | PASS |
| cpu | Aggregate DRAM bandwidth (multi-thread) | 2168.793 | 307.200 | 705.99% | 75.00% | PASS |
| cpu | Parallel speedup at 1 cores | 38.972 | 38.879 | 100.24% | 85.00% | PASS |
| cpu | Parallel speedup at 2 cores | 74.317 | 77.018 | 96.49% | 85.00% | PASS |
| cpu | Parallel speedup at 4 cores | 142.294 | 135.493 | 105.02% | 85.00% | PASS |
| cpu | Parallel speedup at 8 cores | 328.812 | 290.315 | 113.26% | 85.00% | PASS |
| cpu | Parallel speedup at 16 cores | 660.719 | 676.533 | 97.66% | 85.00% | PASS |
| cpu | Parallel speedup at 32 cores | 1216.118 | 1337.293 | 90.94% | 85.00% | PASS |
| cpu | Parallel speedup at 64 cores | 2391.389 | 2615.431 | 91.43% | 85.00% | PASS |
| memory | Aggregate STREAM Triad BW | 816.562 | 307.200 | 265.81% | 75.00% | PASS |
| gpu | HBM3 bandwidth GPU 0 | 1.852 | 2.000 | 92.61% | 80.00% | PASS |
| gpu | HBM3 bandwidth GPU 1 | 1.854 | 2.000 | 92.72% | 80.00% | PASS |
| gpu | HBM3 bandwidth GPU 2 | 1.852 | 2.000 | 92.60% | 80.00% | PASS |
| gpu | HBM3 bandwidth GPU 3 | 3.569 | 3.900 | 91.51% | 80.00% | PASS |
| gpu | FP32 GFLOPS GPU 0 | 36990.378 | 67000.000 | 55.21% | 52.00% | PASS |
| gpu | FP32 GFLOPS GPU 1 | 37349.718 | 67000.000 | 55.75% | 52.00% | PASS |
| gpu | FP32 GFLOPS GPU 2 | 35253.555 | 67000.000 | 52.62% | 52.00% | PASS |
| gpu | FP32 GFLOPS GPU 3 | 45932.750 | 67000.000 | 68.56% | 65.00% | PASS |
| gpu | FP16 TFLOPS GPU 0 | 475.592 | 756.000 | 62.91% | 60.00% | PASS |
| gpu | FP16 TFLOPS GPU 1 | 477.256 | 756.000 | 63.13% | 60.00% | PASS |
| gpu | FP16 TFLOPS GPU 2 | 422.109 | 756.000 | 55.83% | 60.00% | FAIL |
| gpu | FP16 TFLOPS GPU 3 | 442.197 | 989.000 | 44.71% | 75.00% | FAIL |
| gpu | BF16 TFLOPS GPU 0 | 510.777 | 756.000 | 67.56% | 60.00% | PASS |
| gpu | BF16 TFLOPS GPU 1 | 510.991 | 756.000 | 67.59% | 60.00% | PASS |
| gpu | BF16 TFLOPS GPU 2 | 457.892 | 756.000 | 60.57% | 60.00% | PASS |
| gpu | BF16 TFLOPS GPU 3 | 488.632 | 989.000 | 49.41% | 75.00% | FAIL |
| gpu | INT8-proxy TOPS GPU 0 | 547.033 | 3958.000 | 13.82% | 75.00% | FAIL |
| gpu | INT8-proxy TOPS GPU 1 | 549.562 | 3958.000 | 13.88% | 75.00% | FAIL |
| gpu | INT8-proxy TOPS GPU 2 | 502.965 | 3958.000 | 12.71% | 75.00% | FAIL |
| gpu | INT8-proxy TOPS GPU 3 | 543.937 | 3958.000 | 13.74% | 75.00% | FAIL |
| gpu | FP64 TFLOPS GPU 0 | 35.898 | 34.000 | 105.58% | 70.00% | PASS |
| gpu | FP64 TFLOPS GPU 1 | 36.014 | 34.000 | 105.92% | 70.00% | PASS |
| gpu | FP64 TFLOPS GPU 2 | 32.100 | 34.000 | 94.41% | 70.00% | PASS |
| gpu | FP64 TFLOPS GPU 3 | 41.582 | 34.000 | 122.30% | 70.00% | PASS |
| gpu | Multi-GPU FP16 aggregate (4 GPUs) | 1807.010 | 1901.348 | 95.04% | 80.00% | PASS |
| gpu | PCIe H2D bandwidth GPU 0 | 55.657 | 63.000 | 88.34% | 65.00% | PASS |
| gpu | PCIe H2D bandwidth GPU 1 | 55.885 | 63.000 | 88.71% | 65.00% | PASS |
| gpu | PCIe H2D bandwidth GPU 2 | 56.057 | 63.000 | 88.98% | 65.00% | PASS |
| gpu | PCIe H2D bandwidth GPU 3 | 56.020 | 63.000 | 88.92% | 65.00% | PASS |
| gpu | PCIe D2H bandwidth GPU 0 | 55.241 | 63.000 | 87.68% | 65.00% | PASS |
| gpu | PCIe D2H bandwidth GPU 1 | 55.264 | 63.000 | 87.72% | 65.00% | PASS |
| gpu | PCIe D2H bandwidth GPU 2 | 55.270 | 63.000 | 87.73% | 65.00% | PASS |
| gpu | PCIe D2H bandwidth GPU 3 | 57.045 | 63.000 | 90.55% | 65.00% | PASS |
| interconnect | H2D PCIe BW GPU 0 | 56.097 | 63.000 | 89.04% | 65.00% | PASS |
| interconnect | H2D PCIe BW GPU 1 | 56.062 | 63.000 | 88.99% | 65.00% | PASS |
| interconnect | H2D PCIe BW GPU 2 | 56.106 | 63.000 | 89.06% | 65.00% | PASS |
| interconnect | H2D PCIe BW GPU 3 | 56.053 | 63.000 | 88.97% | 65.00% | PASS |
| interconnect | D2H PCIe BW GPU 0 | 55.300 | 63.000 | 87.78% | 65.00% | PASS |
| interconnect | D2H PCIe BW GPU 1 | 55.243 | 63.000 | 87.69% | 65.00% | PASS |
| interconnect | D2H PCIe BW GPU 2 | 55.308 | 63.000 | 87.79% | 65.00% | PASS |
| interconnect | D2H PCIe BW GPU 3 | 57.081 | 63.000 | 90.61% | 65.00% | PASS |
| interconnect | Concurrent H2D BW all 4 GPUs | 206.813 | 252.000 | 82.07% | 65.00% | PASS |
| interconnect | PCIe P2P P2P BW GPU 0->1 | 52.053 | 63.000 | 82.62% | 50.00% | PASS |
| interconnect | PCIe P2P P2P BW GPU 0->2 | 46.134 | 63.000 | 73.23% | 50.00% | PASS |
| interconnect | PCIe P2P P2P BW GPU 0->3 | 44.194 | 63.000 | 70.15% | 50.00% | PASS |
| interconnect | PCIe P2P P2P BW GPU 1->2 | 44.609 | 63.000 | 70.81% | 50.00% | PASS |
| interconnect | PCIe P2P P2P BW GPU 1->3 | 44.512 | 63.000 | 70.65% | 50.00% | PASS |
| interconnect | PCIe P2P P2P BW GPU 2->3 | 51.706 | 63.000 | 82.07% | 50.00% | PASS |

## Workload KPI Threshold Scorecard

| Domain | Test | Actual | Threshold | Rule | Gap | Status |
|---|---|---:|---:|---|---:|---|
| node_validation | test_p2p_bandwidth_latency_binary | 37.060 | 40.000 | >= | -2.940 | FAIL |
| node_validation | test_p2p_bandwidth_pytorch_fallback | 34.170 | 40.000 | >= | -5.830 | FAIL |
| node_validation | test_vision_object_detection_yolo_fps | yolov8x.pt | 95.010 | 200.000 | >= | -104.990 | FAIL |
| node_validation | test_vision_image_generation_diffusers_throughput | stabilityai/sd-turbo | 46.849 | 10.000 | >= | 36.849 | PASS |
| node_validation | test_llm_training_large_multigpu | meta-llama/Meta-Llama-3.1-8B-Instruct | 0.000 | 20.000 | <= | 20.000 | PASS |
| ai_pipeline | test_tokenizer_throughput_by_seq_len | 474000.700 | 50000.000 | >= | 424000.700 | PASS |
| ai_pipeline | test_tokenizer_multi_thread_scaling | 1.476 | 2.000 | >= | -0.524 | FAIL |
| ai_pipeline | test_dataloader_cpu_to_pinned_throughput | 277820.300 | 1000.000 | >= | 276820.300 | PASS |
| ai_pipeline | test_async_prefetch_gpu_overlap_efficiency | 25.260 | 60.000 | >= | -34.740 | FAIL |
| ai_pipeline | test_cpu_preprocessing_pipeline_throughput | 654788.000 | 200000.000 | >= | 454788.000 | PASS |
| ai_pipeline | test_model_load_time_to_gpu | openai-community/gpt2 | 0.199 | 1.000 | >= | -0.801 | FAIL |
| ai_pipeline | test_checkpoint_save_restore_throughput | 4.433 | 1.000 | >= | 3.433 | PASS |
| ai_pipeline | test_checkpoint_save_restore_throughput | 1.606 | 0.500 | >= | 1.106 | PASS |
| ai_pipeline | test_kv_cache_memory_allocation_efficiency | 120540.000 | 10000.000 | >= | 110540.000 | PASS |
| ai_pipeline | test_tensor_parallel_allreduce_overhead_fraction | 98.560 | 20.000 | <= | -78.560 | FAIL |
| ai_pipeline | test_batch_assembly_gpu_latency | 18.107 | 500.000 | <= | 481.893 | PASS |
| ai_pipeline | test_storage_sequential_read_throughput | 1.568 | 1.000 | >= | 0.568 | PASS |
| ai_pipeline | test_storage_sequential_write_throughput | 1.257 | 0.500 | >= | 0.757 | PASS |
| ai_pipeline | test_storage_random_4k_read_iops | 12473.000 | 5000.000 | >= | 7473.000 | PASS |
| ai_pipeline | test_tcp_loopback_throughput | 101.573 | 5.000 | >= | 96.573 | PASS |
| ai_pipeline | test_model_weight_load_pipeline_bandwidth | 0.858 | 1.000 | >= | -0.142 | FAIL |
| ai_pipeline | test_time_to_first_token | openai-community/gpt2 | input_len=32 | 6.030 | 200.000 | <= | 193.970 | PASS |
| ai_pipeline | test_time_to_first_token | openai-community/gpt2 | input_len=128 | 5.870 | 200.000 | <= | 194.130 | PASS |
| ai_pipeline | test_time_to_first_token | openai-community/gpt2 | input_len=512 | 5.910 | 500.000 | <= | 494.090 | PASS |

## Model And Workload KPI Coverage

| Domain | Test | Model/Component | Workload | Key Metrics | Status | Note |
|---|---|---|---|---|---|---|
| node_validation | test_hf_causal_lm_inference_latency | meta-llama/Meta-Llama-3.1-8B-Instruct | - | throughput_tokens_per_sec=61.370, mean_latency_ms=521.440, p95_latency_ms=642.830, completion_time_sec=327.790, runtime_sec=300.360 | INFO | - |
| node_validation | test_hf_causal_lm_inference_latency_enterprise_workloads_deepseek_r1_quantized | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | - | throughput_tokens_per_sec=29.860, mean_latency_ms=1071.830, p95_latency_ms=1212.680, completion_time_sec=347.710, runtime_sec=300.110 | INFO | - |
| node_validation | test_hf_causal_lm_inference_latency_enterprise_workloads_llama_7b | meta-llama/Meta-Llama-3.1-8B-Instruct | - | throughput_tokens_per_sec=61.610, mean_latency_ms=519.420, p95_latency_ms=626.760, completion_time_sec=306.850, runtime_sec=300.230 | INFO | - |
| node_validation | test_hf_causal_lm_inference_latency_enterprise_workloads_qwen_70b | Qwen/Qwen2.5-72B-Instruct | - | throughput_tokens_per_sec=10.290, mean_latency_ms=1555.420, p95_latency_ms=1730.470, completion_time_sec=391.020, runtime_sec=300.200 | INFO | - |
| node_validation | test_hf_causal_lm_batch_throughput | meta-llama/Meta-Llama-3.1-8B-Instruct | - | throughput_tokens_per_sec=511.960, completion_time_sec=26.600 | INFO | - |
| node_validation | test_vision_object_detection_yolo_fps | yolov8x.pt | - | - | INFO | - |
| node_validation | test_vision_image_generation_diffusers_throughput | stabilityai/sd-turbo | - | - | INFO | - |
| node_validation | test_hf_encoder_training_throughput | bert-base-uncased | - | samples_per_sec=347.100, completion_time_sec=1.540 | INFO | - |
| node_validation | test_llm_training_large_multigpu | meta-llama/Meta-Llama-3.1-8B-Instruct | - | total_samples_per_sec=3.080, mean_tflops_per_gpu=18.968 | INFO | - |
| node_validation | test_vllm_training_kpi_status | vllm | - | - | NOT_APPLICABLE | vLLM is inference-serving runtime; use PyTorch/DeepSpeed for training KPIs |
| ai_pipeline | test_model_load_time_to_gpu | openai-community/gpt2 | - | - | INFO | - |
| ai_pipeline | test_time_to_first_token[input_len=32] | openai-community/gpt2 | - | ttft_p95_ms=6.030, ttft_median_ms=6.010, prefill_throughput_tokens_per_sec=5325.900 | INFO | - |
| ai_pipeline | test_time_to_first_token[input_len=128] | openai-community/gpt2 | - | ttft_p95_ms=5.870, ttft_median_ms=5.800, prefill_throughput_tokens_per_sec=22081.700 | INFO | - |
| ai_pipeline | test_time_to_first_token[input_len=512] | openai-community/gpt2 | - | ttft_p95_ms=5.910, ttft_median_ms=5.850, prefill_throughput_tokens_per_sec=87541.300 | INFO | - |

## Priority KPI Failures

### Performance KPI Failures

- bottleneck: GPU 2 FP16 Tensor Core at 36.51% vs threshold 60.00%
- gpu: FP16 TFLOPS GPU 2 at 55.83% vs threshold 60.00%
- gpu: FP16 TFLOPS GPU 3 at 44.71% vs threshold 75.00%
- gpu: BF16 TFLOPS GPU 3 at 49.41% vs threshold 75.00%
- gpu: INT8-proxy TOPS GPU 0 at 13.82% vs threshold 75.00%
- gpu: INT8-proxy TOPS GPU 1 at 13.88% vs threshold 75.00%
- gpu: INT8-proxy TOPS GPU 2 at 12.71% vs threshold 75.00%
- gpu: INT8-proxy TOPS GPU 3 at 13.74% vs threshold 75.00%

### Workload KPI Failures

- node_validation: test_p2p_bandwidth_latency_binary (unidir_min_gbs 37.060 vs >= 40.000)
- node_validation: test_p2p_bandwidth_pytorch_fallback (min_gbs 34.170 vs >= 40.000)
- node_validation: test_vision_object_detection_yolo_fps | yolov8x.pt (aggregate_fps 95.010 vs >= 200.000)
- ai_pipeline: test_tokenizer_multi_thread_scaling (scaling_ratio 1.476 vs >= 2.000)
- ai_pipeline: test_async_prefetch_gpu_overlap_efficiency (overlap_efficiency_pct 25.260 vs >= 60.000)
- ai_pipeline: test_model_load_time_to_gpu | openai-community/gpt2 (load_bandwidth_gbs 0.199 vs >= 1.000)
- ai_pipeline: test_tensor_parallel_allreduce_overhead_fraction (comm_overhead_pct 98.560 vs <= 20.000)
- ai_pipeline: test_model_weight_load_pipeline_bandwidth (end_to_end_gbs 0.858 vs >= 1.000)

## Source Notes

- This report is generated directly from alltests.log only.
- It does not depend on pre-existing executive or KPI reports.
- KPI status is derived from either logged efficiency thresholds or explicit threshold fields present in logged KPI payloads.
