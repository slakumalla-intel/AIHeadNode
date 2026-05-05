# AI HeadNode - Executive KPI Status Report

**Generated:** 2026-05-05 15:27:06
**Report Type:** Management Summary
**Status:** Test Execution Complete

---

## Overall Validation Status

| Metric | Value | Status |
|---|---|---|
| **Total Tests Available** | 410 | PASS |
| **Tests Executed** | 375 | PASS |
| **Tests Passing** | 303 | PASS |
| **Tests Failing** | 72 | AT_RISK |
| **Pass Rate** | 80.8% | PASS |
| **Test Coverage** | 91.5% | PASS |
| **Numeric KPI Points** | 584 | PASS |

---

## KPI Numeric Highlights

| KPI | Representative Value | Notes |
|---|---:|---|
| bidir_min_gbs | 17.26 GB/s | P2P bidirectional bandwidth (worst link) |
| unidir_min_gbs | 9.93 GB/s | P2P unidirectional bandwidth (worst link) |
| throughput_tokens_per_sec | 3146.86 tok/s | Inference throughput |
| mean_latency_ms | 879.44 ms | Inference latency |
| clock_drop_ratio | 0.4957 | GPU clock-drop ratio under load |
| comm_overhead_pct | 98.68% | Tensor-parallel communication overhead |

---

## Domain-Level Status

| Domain | Total | Pass | Fail | Skip | Status |
|---|---:|---:|---:|---:|---|
| agentic_ai | 9 | 8 | 1 | 0 | AT_RISK |
| ai_pipeline | 38 | 20 | 17 | 1 | AT_RISK |
| bottleneck | 25 | 24 | 1 | 0 | AT_RISK |
| cpu | 33 | 31 | 1 | 1 | AT_RISK |
| gpu | 53 | 44 | 9 | 0 | AT_RISK |
| interconnect | 77 | 72 | 1 | 4 | AT_RISK |
| memory | 28 | 21 | 7 | 0 | AT_RISK |
| mlperf_suite | 21 | 0 | 19 | 2 | AT_RISK |
| node_validation | 51 | 36 | 7 | 8 | AT_RISK |
| performance_suite | 25 | 9 | 1 | 15 | AT_RISK |
| system_config | 8 | 8 | 0 | 0 | PASS |
| system_health | 39 | 28 | 8 | 3 | AT_RISK |
| vllm_prefill | 3 | 2 | 0 | 1 | PASS |

## Critical Focus Areas

1. mlperf_suite: 19 failing tests
2. ai_pipeline: 17 failing tests
3. gpu: 9 failing tests
4. system_health: 8 failing tests
5. memory: 7 failing tests

## KPI Gaps (Fail Only)

| Test | Expected | Actual | Rule | Gap |
|---|---:|---:|---|---:|
| test_async_prefetch_gpu_overlap_efficiency | 60 (threshold_overlap_pct) | 25.28 (overlap_efficiency_pct) | >= | -34.72 |
| test_compile_speedup_over_eager | 1.05 (floor_ratio) | 0.8081 (speedup_ratio) | >= | -0.2419 |
| test_llm_training_small_multigpu | 15 (threshold_sync_pct) | 21.13 (mean_sync_pct) | <= | -6.13 |
| test_model_load_time_to_gpu | 1 (threshold_gbs) | 0.28 (load_bandwidth_gbs) | >= | -0.72 |
| test_model_weight_load_pipeline_bandwidth | 1 (threshold_gbs) | 0.899 (end_to_end_gbs) | >= | -0.101 |
| test_p2p_bandwidth_latency_binary | 40 (min_threshold_gbs) | 9.93 (unidir_min_gbs) | >= | -30.07 |
| test_p2p_bandwidth_pytorch_fallback | 40 (min_threshold_gbs) | 36.92 (min_gbs) | >= | -3.08 |
| test_quantised_softmax_output_matches_fp16 | 0.001 (threshold) | 0.00256348 (softmax_max_abs_error) | <= | -0.00156348 |
| test_tensor_parallel_allreduce_overhead_fraction | 20 (threshold_comm_overhead_pct) | 98.68 (comm_overhead_pct) | <= | -78.68 |
| test_tokenizer_multi_thread_scaling | 2 (threshold_scaling_ratio) | 1.412 (scaling_ratio) | >= | -0.588 |
| test_vllm_data_pipeline_saturation_inference_prefill | 0.5 (gpu_over_90_ratio_target) | 0 (gpu_over_90_ratio) | >= | -0.5 |

## KPI Test Health Table

| Test | Actual Metric Value | Better Metric Value | Health |
|---|---|---|---|
| test_async_prefetch_gpu_overlap_efficiency | 25.28 (overlap_efficiency_pct) | >= 60 (threshold_overlap_pct) | <span style="color:red"><b>BAD</b></span> |
| test_batch_assembly_gpu_latency | 17.744 (assemble_us_per_batch) | <= 500 (threshold_us_per_batch) | <span style="color:green"><b>GOOD</b></span> |
| test_bf16_vs_fp16_matmul_equivalence | 0.002869 (bf16_vs_fp16_mean_rel_error) | <= 0.1 (threshold_rel_error) | <span style="color:green"><b>GOOD</b></span> |
| test_checkpoint_save_restore_throughput | 4.158 (restore_bandwidth_gbs) | >= 1 (threshold_restore_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_checkpoint_save_restore_throughput | 1.637 (save_bandwidth_gbs) | >= 0.5 (threshold_save_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_compile_speedup_over_eager | 0.8081 (speedup_ratio) | >= 1.05 (floor_ratio) | <span style="color:red"><b>BAD</b></span> |
| test_continuous_batching_throughput_vllm | 21061 (throughput_tokens_per_sec) | >= 200 (threshold_tps) | <span style="color:green"><b>GOOD</b></span> |
| test_cpu_preprocessing_pipeline_throughput | 525912 (total_tokens_per_sec) | >= 200000 (threshold_tokens_per_sec) | <span style="color:green"><b>GOOD</b></span> |
| test_d2h_bandwidth_all_gpus | 54.93 (min_gbs) | >= 40.95 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_dataloader_cpu_to_pinned_throughput | 26989.9 (samples_per_sec) | >= 1000 (threshold_samples_per_sec) | <span style="color:green"><b>GOOD</b></span> |
| test_event_wait_enforces_ordering | 1.67772e+07 (actual_sum) | == 1.67772e+07 (expected_sum) | <span style="color:green"><b>GOOD</b></span> |
| test_fp8_e4m3_matmul_error_vs_fp16 | 0.037445 (fp8_vs_fp16_mean_rel_error) | <= 0.15 (threshold_rel_error) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_power_limit_headroom | 343.825 (gpu0_power_w) | <= 350 (gpu0_limit_w) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_power_limit_headroom | 48.925 (gpu1_power_w) | <= 350 (gpu1_limit_w) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_power_limit_headroom | 51.955 (gpu2_power_w) | <= 310 (gpu2_limit_w) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_power_limit_headroom | 61.892 (gpu3_power_w) | <= 400 (gpu3_limit_w) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_temperature_within_safe_limits[0] | 38 (temperature_c) | <= 83 (threshold_c) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_temperature_within_safe_limits[1] | 35 (temperature_c) | <= 83 (threshold_c) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_temperature_within_safe_limits[2] | 36 (temperature_c) | <= 83 (threshold_c) | <span style="color:green"><b>GOOD</b></span> |
| test_gpu_temperature_within_safe_limits[3] | 35 (temperature_c) | <= 83 (threshold_c) | <span style="color:green"><b>GOOD</b></span> |
| test_h2d_bandwidth_all_gpus | 56.02 (min_gbs) | >= 40.95 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_hbm_ecc_corrected_errors_below_threshold[0] | 0 (corrected_ecc_errors) | <= 100 (threshold) | <span style="color:green"><b>GOOD</b></span> |
| test_hbm_ecc_corrected_errors_below_threshold[1] | 0 (corrected_ecc_errors) | <= 100 (threshold) | <span style="color:green"><b>GOOD</b></span> |
| test_hbm_ecc_corrected_errors_below_threshold[2] | 0 (corrected_ecc_errors) | <= 100 (threshold) | <span style="color:green"><b>GOOD</b></span> |
| test_hbm_ecc_corrected_errors_below_threshold[3] | 0 (corrected_ecc_errors) | <= 100 (threshold) | <span style="color:green"><b>GOOD</b></span> |
| test_int8_matmul_error_vs_fp16 | 0.015343 (int8_vs_fp16_mean_rel_error) | <= 0.05 (threshold_rel_error) | <span style="color:green"><b>GOOD</b></span> |
| test_kv_cache_memory_allocation_efficiency | 120039 (blocks_per_sec) | >= 10000 (threshold_blocks_per_sec) | <span style="color:green"><b>GOOD</b></span> |
| test_llm_training_large_multigpu | 0 (mean_sync_pct) | <= 10 (threshold_sync_pct) | <span style="color:green"><b>GOOD</b></span> |
| test_llm_training_small_multigpu | 21.13 (mean_sync_pct) | <= 15 (threshold_sync_pct) | <span style="color:red"><b>BAD</b></span> |
| test_model_load_time_to_gpu | 0.28 (load_bandwidth_gbs) | >= 1 (threshold_gbs) | <span style="color:red"><b>BAD</b></span> |
| test_model_weight_load_pipeline_bandwidth | 0.899 (end_to_end_gbs) | >= 1 (threshold_gbs) | <span style="color:red"><b>BAD</b></span> |
| test_multi_stream_no_race_on_shared_tensor | 4 (final_val) | == 4 (expected) | <span style="color:green"><b>GOOD</b></span> |
| test_nccl_all_reduce_perf | 31.31 (mean_busbw_gbs) | >= 30 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_nvbandwidth_device_to_device_memcpy_read_ce | 47.36 (mean_bw_gbs) | >= 30 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_nvbandwidth_device_to_host_bidirectional_memcpy_ce | 46.55 (mean_bw_gbs) | >= 40 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_nvbandwidth_device_to_host_memcpy_ce | 55.87 (mean_bw_gbs) | >= 40 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_nvbandwidth_device_to_host_memcpy_sm | 52.8 (mean_bw_gbs) | >= 30 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_nvbandwidth_host_to_device_bidirectional_memcpy_ce | 47.78 (mean_bw_gbs) | >= 40 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_nvbandwidth_host_to_device_memcpy_ce | 55.89 (mean_bw_gbs) | >= 40 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_nvbandwidth_host_to_device_memcpy_sm | 53.39 (mean_bw_gbs) | >= 30 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_p2p_bandwidth_latency_binary | 9.93 (unidir_min_gbs) | >= 40 (min_threshold_gbs) | <span style="color:red"><b>BAD</b></span> |
| test_p2p_bandwidth_pytorch_fallback | 36.92 (min_gbs) | >= 40 (min_threshold_gbs) | <span style="color:red"><b>BAD</b></span> |
| test_quantised_softmax_output_matches_fp16 | 0.00256348 (softmax_max_abs_error) | <= 0.001 (threshold) | <span style="color:red"><b>BAD</b></span> |
| test_storage_random_4k_read_iops | 12553 (iops) | >= 5000 (threshold_iops) | <span style="color:green"><b>GOOD</b></span> |
| test_storage_sequential_read_throughput | 1.59 (read_bandwidth_gbs) | >= 1 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_storage_sequential_write_throughput | 1.296 (write_bandwidth_gbs) | >= 0.5 (threshold_gbs) | <span style="color:green"><b>GOOD</b></span> |
| test_tcp_loopback_throughput | 101.568 (throughput_gbps) | >= 5 (threshold_gbps) | <span style="color:green"><b>GOOD</b></span> |
| test_tensor_parallel_allreduce_overhead_fraction | 98.68 (comm_overhead_pct) | <= 20 (threshold_comm_overhead_pct) | <span style="color:red"><b>BAD</b></span> |
| test_time_to_first_token | 22.37 (ttft_p95_ms) | <= 200 (threshold_ms) | <span style="color:green"><b>GOOD</b></span> |
| test_time_to_first_token | 22.37 (ttft_p95_ms) | <= 500 (threshold_ms) | <span style="color:green"><b>GOOD</b></span> |
| test_tokenizer_multi_thread_scaling | 1.412 (scaling_ratio) | >= 2 (threshold_scaling_ratio) | <span style="color:red"><b>BAD</b></span> |
| test_tokenizer_throughput_by_seq_len | 441895 (tokens_per_sec) | >= 20000 (threshold_tokens_per_sec) | <span style="color:green"><b>GOOD</b></span> |
| test_tokenizer_throughput_by_seq_len | 441895 (tokens_per_sec) | >= 50000 (threshold_tokens_per_sec) | <span style="color:green"><b>GOOD</b></span> |
| test_vision_image_generation_diffusers_throughput | 29.001 (throughput_img_per_sec) | >= 10 (throughput_target_img_per_sec) | <span style="color:green"><b>GOOD</b></span> |
| test_vllm_data_pipeline_saturation_inference_prefill | 1 (cpu_under_70_ratio) | >= 0.8 (cpu_under_70_ratio_target) | <span style="color:green"><b>GOOD</b></span> |
| test_vllm_data_pipeline_saturation_inference_prefill | 0 (gpu_over_90_ratio) | >= 0.5 (gpu_over_90_ratio_target) | <span style="color:red"><b>BAD</b></span> |
