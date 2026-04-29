# Performance Metrics Summary Table

Extracted: April 28, 2026
Source: artifacts/KPI_VALUES.csv (622 numeric data points)

---

## 1. Functional Performance

| Metric | Value | Unit | Test |
|--------|-------|------|------|
| bf16_vs_fp16_mean_rel_error | 0.0029 | relative error | test_bf16_vs_fp16_matmul_equivalence |
| int8_vs_fp16_mean_rel_error | 0.0153 | relative error | test_int8_matmul_error_vs_fp16 |
| fp8_vs_fp16_mean_rel_error | 0.0374 | relative error | test_fp8_e4m3_matmul_error_vs_fp16 |
| softmax_max_abs_error | 0.0026 | absolute error | test_quantised_softmax_output_matches_fp16 |
| threshold_rel_error (fp8) | 0.1500 | tolerance | test_fp8_e4m3_matmul_error_vs_fp16 |
| corrected_ecc_errors | 0.0000 | count | test_hbm_ecc_corrected_errors_below_threshold[0] |
| failing_pair_count | 0.0000 | count | test_p2p_multi_stream_bandwidth |

Total functional metrics: 13

---

## 2. Inference Performance

| Metric | Value | Unit | Test |
|--------|-------|------|------|
| tokens_per_sec | 1,540,823.5 | tokens/sec | test_tokenizer_throughput_by_seq_len |
| threads_4_tokens_per_sec | 1,265,423.9 | tokens/sec | test_tokenizer_multi_thread_scaling |
| threads_1_tokens_per_sec | 896,436.6 | tokens/sec | test_tokenizer_multi_thread_scaling |
| total_tokens_per_sec | 525,912.0 | tokens/sec | test_cpu_preprocessing_pipeline_throughput |
| prefill_throughput_tokens_per_sec | 93,292.9 | tokens/sec | test_time_to_first_token |
| offload_p95_latency_ms | 37,878.0599 | ms | test_predictable_memory_orchestration_kv_cache_offload |
| offload_p99_latency_ms | 37,882.6876 | ms | test_predictable_memory_orchestration_kv_cache_offload |

Total inference metrics: 103

---

## 3. NVBandwidth Performance

| Metric | Value | Unit | Test |
|--------|-------|------|------|
| busbw_peak_gbs | 60,000,000.0 | GB/s | unknown |
| busbw_mean_gbs | 13,838,535.733 | GB/s | unknown |
| peak_write_bw_gbs | 2,264.8 | GB/s | test_gpu_write_performance_limiters |
| mean_bw_gbs | 55.89 | GB/s | test_nvbandwidth_host_to_device_memcpy_ce |

Total NVBandwidth metrics: 46

---

## 4. P2P Performance

| Metric | Value | Unit | Test |
|--------|-------|------|------|
| allreduce_time_ms | 11.6155 | ms | test_tensor_parallel_allreduce_overhead_fraction |

Total P2P metrics: 1

---

## 5. Training Performance

| Metric | Value | Unit | Test |
|--------|-------|------|------|
| compiled_time_s | 0.0082 | sec | test_compile_speedup_over_eager |

Total training metrics: 1

---

## Summary

| Category | Metric Count |
|----------|--------------|
| Functional | 13 |
| Inference | 103 |
| NVBandwidth | 46 |
| P2P | 1 |
| Training | 1 |
| Total categorized metrics | 164 |
| Total numeric KPI data points | 622 |
