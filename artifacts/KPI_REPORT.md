# KPI Report

Generated (UTC): 2026-05-27T05:03:01.630952+00:00
Source: top-level artifacts/*_kpis.jsonl with latest entry per test case

## KPI Coverage

| KPI File | Records | Latest Distinct Tests | Latest Timestamp | Numeric Fields | Threshold Checks | Pass | Fail |
|---|---:|---:|---|---:|---:|---:|---:|
| agentic_ai_kpis.jsonl | 56 | 8 | 2026-05-21T02:48:59 | 27 | 0 | 0 | 0 |
| ai_pipeline_kpis.jsonl | 219 | 16 | 2026-05-27T04:38:44 | 74 | 14 | 10 | 4 |
| bottleneck_kpis.jsonl | 10 | 1 | 2026-05-21T02:14:34 | 0 | 0 | 0 | 0 |
| cpu_validation_kpis.jsonl | 200 | 25 | 2026-05-21T02:19:07 | 49 | 14 | 14 | 0 |
| gpu_validation_kpis.jsonl | 95 | 24 | 2026-05-21T02:19:42 | 9 | 0 | 0 | 0 |
| interconnect_kpis.jsonl | 56 | 14 | 2026-05-21T02:20:01 | 5 | 0 | 0 | 0 |
| memory_validation_kpis.jsonl | 8 | 1 | 2026-05-21T02:19:16 | 2 | 0 | 0 | 0 |
| mlperf_suite_kpis.jsonl | 195 | 1 | 2026-05-21T03:57:55 | 0 | 0 | 0 | 0 |
| node_validation_kpis.jsonl | 220 | 41 | 2026-05-21T02:48:33 | 161 | 10 | 10 | 0 |
| performance_suite_kpis.jsonl | 508 | 1 | 2026-05-26T21:51:26 | 0 | 0 | 0 | 0 |
| system_health_kpis.jsonl | 128 | 36 | 2026-05-21T02:50:22+00:00 | 60 | 7 | 7 | 0 |

## Threshold Check Totals

- Total threshold checks: 45
- Passed checks: 41
- Failed checks: 4

## Good Data

| Test | Metric | Actual | Threshold |
|---|---|---:|---:|
| test_tokenizer_throughput_by_seq_len | tokens_per_sec | 1372823.9000 | 20000.0000 |
| test_dataloader_cpu_to_pinned_throughput | samples_per_sec | 41384.6000 | 1000.0000 |
| test_cpu_preprocessing_pipeline_throughput | total_tokens_per_sec | 700388.0000 | 200000.0000 |
| test_storage_sequential_read_throughput | read_bandwidth_gbs | 1.4170 | 1.0000 |
| test_storage_sequential_write_throughput | write_bandwidth_gbs | 1.1820 | 0.5000 |
| test_storage_random_4k_read_iops | iops | 11080.0000 | 5000.0000 |
| test_tcp_loopback_throughput | throughput_gbps | 85.1070 | 5.0000 |
| test_kv_cache_memory_allocation_efficiency | blocks_per_sec | 126799.0000 | 10000.0000 |
| test_batch_assembly_gpu_latency | assemble_us_per_batch | 20.0970 | 500.0000 |
| test_time_to_first_token | ttft_p95_ms | 5.8600 | 500.0000 |
| TC-CPU-05a_single_thread_stream_bw | bandwidth_gbs | 41.5500 | 30.0000 |
| TC-CPU-07_parallel_speedup_n1 | efficiency_pct | 102.1200 | 85.0000 |
| TC-CPU-07_parallel_speedup_n2 | efficiency_pct | 100.0900 | 85.0000 |
| TC-CPU-07_parallel_speedup_n4 | efficiency_pct | 92.4700 | 85.0000 |
| TC-CPU-07_parallel_speedup_n8 | efficiency_pct | 102.6300 | 85.0000 |
| TC-CPU-07_parallel_speedup_n16 | efficiency_pct | 103.0100 | 85.0000 |
| TC-CPU-07_parallel_speedup_n32 | efficiency_pct | 94.7900 | 85.0000 |
| TC-CPU-07_parallel_speedup_n64 | efficiency_pct | 98.5800 | 85.0000 |
| TC-CPU-09a_int64_add_throughput | total_gops | 6.6590 | 1.6000 |
| TC-CPU-09b_int64_multiply_throughput | total_gops | 15.1080 | 1.6000 |
| TC-CPU-10a_predictable_branch | elapsed_sec | 0.2192 | 0.5000 |
| TC-CPU-10b_unpredictable_branch | elapsed_sec | 0.1821 | 5.0000 |
| TC-CPU-11a_threadpool_spawn_latency | spawn_elapsed_sec | 0.0252 | 2.0000 |
| TC-CPU-11b_threadpool_reuse_overhead | submit_elapsed_sec | 0.0154 | 1.0000 |
| test_h2d_bandwidth_all_gpus | mean_gbs | 56.1900 | 40.9500 |
| test_d2h_bandwidth_all_gpus | mean_gbs | 55.2700 | 40.9500 |
| test_nvbandwidth_host_to_device_memcpy_ce | mean_bw_gbs | 55.9100 | 40.0000 |
| test_nvbandwidth_device_to_host_memcpy_ce | mean_bw_gbs | 55.7900 | 40.0000 |
| test_nvbandwidth_host_to_device_bidirectional_memcpy_ce | mean_bw_gbs | 51.3400 | 40.0000 |
| test_nvbandwidth_device_to_host_bidirectional_memcpy_ce | mean_bw_gbs | 49.8700 | 40.0000 |

## Bad Data

| Test | Metric | Actual | Threshold |
|---|---|---:|---:|
| test_tokenizer_multi_thread_scaling | scaling_ratio | 0.9140 | 2.0000 |
| test_model_load_time_to_gpu | load_bandwidth_gbs | 0.3030 | 1.0000 |
| test_tensor_parallel_allreduce_overhead_fraction | comm_overhead_pct | 98.7100 | 20.0000 |
| test_model_weight_load_pipeline_bandwidth | end_to_end_gbs | 0.8870 | 1.0000 |

## Correlation With Test Outcomes

- Workload test failures (from status report): 6
- KPI threshold failures (latest metric rows): 4
- Note: A domain can fail from runtime errors even if threshold checks are limited.
