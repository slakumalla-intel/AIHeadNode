# KPI Expected vs Actual Comparison

Generated: 2026-05-05 14:51:10
Source: artifacts/KPI_VALUES.csv

## Summary

- Total comparisons: 56
- PASS: 45
- FAIL: 11
- N/A mapping: 0

## Comparison Table

| Test | Expected Metric | Expected | Actual Metric | Actual | Rule | Result | Gap |
|---|---|---:|---|---:|---|---|---:|
| test_async_prefetch_gpu_overlap_efficiency | threshold_overlap_pct | 60 | overlap_efficiency_pct | 25.28 | >= | FAIL | -34.72 |
| test_batch_assembly_gpu_latency | threshold_us_per_batch | 500 | assemble_us_per_batch | 17.744 | <= | PASS | 482.256 |
| test_bf16_vs_fp16_matmul_equivalence | threshold_rel_error | 0.1 | bf16_vs_fp16_mean_rel_error | 0.002869 | <= | PASS | 0.097131 |
| test_checkpoint_save_restore_throughput | threshold_restore_gbs | 1 | restore_bandwidth_gbs | 4.158 | >= | PASS | 3.158 |
| test_checkpoint_save_restore_throughput | threshold_save_gbs | 0.5 | save_bandwidth_gbs | 1.637 | >= | PASS | 1.137 |
| test_compile_speedup_over_eager | floor_ratio | 1.05 | speedup_ratio | 0.8081 | >= | FAIL | -0.2419 |
| test_continuous_batching_throughput_vllm | threshold_tps | 200 | throughput_tokens_per_sec | 21061 | >= | PASS | 20861 |
| test_cpu_preprocessing_pipeline_throughput | threshold_tokens_per_sec | 200000 | total_tokens_per_sec | 525912 | >= | PASS | 325912 |
| test_d2h_bandwidth_all_gpus | threshold_gbs | 40.95 | min_gbs | 54.93 | >= | PASS | 13.98 |
| test_dataloader_cpu_to_pinned_throughput | threshold_samples_per_sec | 1000 | samples_per_sec | 26989.9 | >= | PASS | 25989.9 |
| test_event_wait_enforces_ordering | expected_sum | 1.67772e+07 | actual_sum | 1.67772e+07 | == | PASS | 0 |
| test_fp8_e4m3_matmul_error_vs_fp16 | threshold_rel_error | 0.15 | fp8_vs_fp16_mean_rel_error | 0.037445 | <= | PASS | 0.112555 |
| test_gpu_power_limit_headroom | gpu0_limit_w | 350 | gpu0_power_w | 343.825 | <= | PASS | 6.175 |
| test_gpu_power_limit_headroom | gpu1_limit_w | 350 | gpu1_power_w | 48.925 | <= | PASS | 301.075 |
| test_gpu_power_limit_headroom | gpu2_limit_w | 310 | gpu2_power_w | 51.955 | <= | PASS | 258.045 |
| test_gpu_power_limit_headroom | gpu3_limit_w | 400 | gpu3_power_w | 61.892 | <= | PASS | 338.108 |
| test_gpu_temperature_within_safe_limits[0] | threshold_c | 83 | temperature_c | 38 | <= | PASS | 45 |
| test_gpu_temperature_within_safe_limits[1] | threshold_c | 83 | temperature_c | 35 | <= | PASS | 48 |
| test_gpu_temperature_within_safe_limits[2] | threshold_c | 83 | temperature_c | 36 | <= | PASS | 47 |
| test_gpu_temperature_within_safe_limits[3] | threshold_c | 83 | temperature_c | 35 | <= | PASS | 48 |
| test_h2d_bandwidth_all_gpus | threshold_gbs | 40.95 | min_gbs | 56.02 | >= | PASS | 15.07 |
| test_hbm_ecc_corrected_errors_below_threshold[0] | threshold | 100 | corrected_ecc_errors | 0 | <= | PASS | 100 |
| test_hbm_ecc_corrected_errors_below_threshold[1] | threshold | 100 | corrected_ecc_errors | 0 | <= | PASS | 100 |
| test_hbm_ecc_corrected_errors_below_threshold[2] | threshold | 100 | corrected_ecc_errors | 0 | <= | PASS | 100 |
| test_hbm_ecc_corrected_errors_below_threshold[3] | threshold | 100 | corrected_ecc_errors | 0 | <= | PASS | 100 |
| test_int8_matmul_error_vs_fp16 | threshold_rel_error | 0.05 | int8_vs_fp16_mean_rel_error | 0.015343 | <= | PASS | 0.034657 |
| test_kv_cache_memory_allocation_efficiency | threshold_blocks_per_sec | 10000 | blocks_per_sec | 120039 | >= | PASS | 110039 |
| test_llm_training_large_multigpu | threshold_sync_pct | 10 | mean_sync_pct | 0 | <= | PASS | 10 |
| test_llm_training_small_multigpu | threshold_sync_pct | 15 | mean_sync_pct | 21.13 | <= | FAIL | -6.13 |
| test_model_load_time_to_gpu | threshold_gbs | 1 | load_bandwidth_gbs | 0.28 | >= | FAIL | -0.72 |
| test_model_weight_load_pipeline_bandwidth | threshold_gbs | 1 | end_to_end_gbs | 0.899 | >= | FAIL | -0.101 |
| test_multi_stream_no_race_on_shared_tensor | expected | 4 | final_val | 4 | == | PASS | 0 |
| test_nccl_all_reduce_perf | threshold_gbs | 30 | mean_busbw_gbs | 31.31 | >= | PASS | 1.31 |
| test_nvbandwidth_device_to_device_memcpy_read_ce | threshold_gbs | 30 | mean_bw_gbs | 47.36 | >= | PASS | 17.36 |
| test_nvbandwidth_device_to_host_bidirectional_memcpy_ce | threshold_gbs | 40 | mean_bw_gbs | 46.55 | >= | PASS | 6.55 |
| test_nvbandwidth_device_to_host_memcpy_ce | threshold_gbs | 40 | mean_bw_gbs | 55.87 | >= | PASS | 15.87 |
| test_nvbandwidth_device_to_host_memcpy_sm | threshold_gbs | 30 | mean_bw_gbs | 52.8 | >= | PASS | 22.8 |
| test_nvbandwidth_host_to_device_bidirectional_memcpy_ce | threshold_gbs | 40 | mean_bw_gbs | 47.78 | >= | PASS | 7.78 |
| test_nvbandwidth_host_to_device_memcpy_ce | threshold_gbs | 40 | mean_bw_gbs | 55.89 | >= | PASS | 15.89 |
| test_nvbandwidth_host_to_device_memcpy_sm | threshold_gbs | 30 | mean_bw_gbs | 53.39 | >= | PASS | 23.39 |
| test_p2p_bandwidth_latency_binary | min_threshold_gbs | 40 | unidir_min_gbs | 9.93 | >= | FAIL | -30.07 |
| test_p2p_bandwidth_pytorch_fallback | min_threshold_gbs | 40 | min_gbs | 36.92 | >= | FAIL | -3.08 |
| test_quantised_softmax_output_matches_fp16 | threshold | 0.001 | softmax_max_abs_error | 0.00256348 | <= | FAIL | -0.00156348 |
| test_storage_random_4k_read_iops | threshold_iops | 5000 | iops | 12553 | >= | PASS | 7553 |
| test_storage_sequential_read_throughput | threshold_gbs | 1 | read_bandwidth_gbs | 1.59 | >= | PASS | 0.59 |
| test_storage_sequential_write_throughput | threshold_gbs | 0.5 | write_bandwidth_gbs | 1.296 | >= | PASS | 0.796 |
| test_tcp_loopback_throughput | threshold_gbps | 5 | throughput_gbps | 101.568 | >= | PASS | 96.568 |
| test_tensor_parallel_allreduce_overhead_fraction | threshold_comm_overhead_pct | 20 | comm_overhead_pct | 98.68 | <= | FAIL | -78.68 |
| test_time_to_first_token | threshold_ms | 200 | ttft_p95_ms | 22.37 | <= | PASS | 177.63 |
| test_time_to_first_token | threshold_ms | 500 | ttft_p95_ms | 22.37 | <= | PASS | 477.63 |
| test_tokenizer_multi_thread_scaling | threshold_scaling_ratio | 2 | scaling_ratio | 1.412 | >= | FAIL | -0.588 |
| test_tokenizer_throughput_by_seq_len | threshold_tokens_per_sec | 20000 | tokens_per_sec | 441895 | >= | PASS | 421895 |
| test_tokenizer_throughput_by_seq_len | threshold_tokens_per_sec | 50000 | tokens_per_sec | 441895 | >= | PASS | 391895 |
| test_vision_image_generation_diffusers_throughput | throughput_target_img_per_sec | 10 | throughput_img_per_sec | 29.001 | >= | PASS | 19.001 |
| test_vllm_data_pipeline_saturation_inference_prefill | cpu_under_70_ratio_target | 0.8 | cpu_under_70_ratio | 1 | >= | PASS | 0.2 |
| test_vllm_data_pipeline_saturation_inference_prefill | gpu_over_90_ratio_target | 0.5 | gpu_over_90_ratio | 0 | >= | FAIL | -0.5 |
