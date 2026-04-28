# Performance Metrics Summary Table

**Extracted:** April 28, 2026  
**Source:** KPI_VALUES.csv (19,466 data points)

---

## 1. FUNCTIONAL PERFORMANCE

| Metric | Value | Unit | Test |
|--------|-------|------|------|
| bf16_vs_fp16_mean_rel_error | 0.0029 | relative error | test_bf16_vs_fp16_matmul_equivalence |
| int8_vs_fp16_mean_rel_error | 0.0153 | relative error | test_int8_matmul_error_vs_fp16 |
| fp8_vs_fp16_mean_rel_error | 0.0374 | relative error | test_fp8_e4m3_matmul_error_vs_fp16 |
| softmax_max_abs_error | 0.0026 | absolute error | test_quantised_softmax_output_matches_fp16 |
| threshold_rel_error (int8) | 0.0500 | tolerance | test_int8_matmul_error_vs_fp16 |
| threshold_rel_error (bf16) | 0.1000 | tolerance | test_bf16_vs_fp16_matmul_equivalence |
| threshold_rel_error (fp8) | 0.1500 | tolerance | test_fp8_e4m3_matmul_error_vs_fp16 |
| corrected_ecc_errors | 0.0000 | count | test_hbm_ecc_corrected_errors_below_threshold |
| **Total Functional Metrics** | **20** | | |

---

## 2. INFERENCE PERFORMANCE

| Metric | Min Value | Max Value | Unit | Test |
|--------|-----------|-----------|------|------|
| **Throughput (Tokens/sec)** | | | | |
| tokens_per_sec (tokenizer) | 732,163 | 1,406,331 | tokens/sec | test_tokenizer_throughput_by_seq_len |
| threads_1_tokens_per_sec | 825,214 | 871,522 | tokens/sec | test_tokenizer_multi_thread_scaling |
| threads_4_tokens_per_sec | 1,251,906 | 1,315,292 | tokens/sec | test_tokenizer_multi_thread_scaling |
| total_tokens_per_sec | 661,633 | 661,633 | tokens/sec | test_cpu_preprocessing_pipeline_throughput |
| prefill_throughput_tokens_per_sec | 94,728 | 94,728 | tokens/sec | test_vllm_prefill_performance |
| **Latency Metrics** | | | | |
| p95_latency_ms | 88.737 | 88.737 | ms | test_agentic_ai_latency |
| mean_latency_ms | 475.28 | 475.28 | ms | test_ai_pipeline_latency |
| avg_latency_ms | 57.11 | 59.12 | ms | test_batch_assembly |
| **Total Inference Metrics** | **361** | | | |

---

## 3. NVBandwidth PERFORMANCE

| Metric | Min Value | Max Value | Unit | Test |
|--------|-----------|-----------|------|------|
| **HBM (GPU Memory)** | | | | |
| hbm_bandwidth_gbps | 1,663 | 1,663 | GB/s | GPU Memory Bandwidth (on-GPU) |
| **PCIe Bandwidth (P2P/Host Communication)** | | | | |
| h2d_bandwidth_gbps | 21.89 | 21.89 | GB/s | Host-to-Device (PCIe) |
| d2h_bandwidth_gbps | 31.78 | 31.78 | GB/s | Device-to-Host (PCIe) |
| p2p_bandwidth_gbps | 17.3 | 17.3 | GB/s | GPU-to-GPU P2P (PCIe) |
| **NCCL Collective Communications (over PCIe)** | | | | |
| nccl_p2p_busbw_gbps | 25.1 | 622.72 | GB/s | NCCL P2P bus bandwidth (PCIe-based) |
| busbw_mean_gbs | 622.72 | 14,622,216 | GB/s | NCCL collective bus bandwidth |
| busbw_peak_gbs | 3,446 | 60,000,000 | GB/s | NCCL peak bandwidth |
| busbw_samples | 30 | 30 | samples | NCCL measurements |
| **Total NVBandwidth Metrics** | **238** | | | |

---

## 4. P2P PERFORMANCE (Peer-to-Peer / Collective Communications)

| Metric | Min Value | Max Value | Unit | Test |
|--------|-----------|-----------|------|------|
| **AllReduce Latency** | | | | |
| allreduce_time_ms | 10.72 | 11.58 | ms | test_tensor_parallel_allreduce_overhead_fraction |
| nccl_allreduce_time_us | 10.5 | 10.5 | μs | test_nccl_allreduce |
| **AllGather Latency** | | | | |
| nccl_allgather_time_us | 14.5 | 14.5 | μs | test_nccl_allgather |
| **P2P Direct Latency/Bandwidth** | | | | |
| p2p_latency_us | 15.2 | 15.2 | μs | test_p2p_latency |
| nccl_p2p_perf_us | 92.3 | 92.3 | μs | test_nccl_p2p_performance |
| p2p_bandwidth_gbps | 17.3 | 17.3 | GB/s | test_p2p_bandwidth |
| nccl_p2p_busbw_gbps | 25.1 | 622.72 | GB/s | test_nccl_p2p_busbandwidth |
| **Total P2P Metrics** | **3** (unique) | | | |

---

## 5. TRAINING PERFORMANCE

| Metric | Value | Unit | Test |
|--------|-------|------|------|
| **Compilation Speedup** | | | |
| compile_speedup_ratio | 1.049 | ratio (speedup) | test_compile_speedup_over_eager |
| compiled_time_s | 0.0082 | seconds | torch.compile optimization |
| **Memory Efficiency** | | | |
| alloc_time_sec | 0.0072 - 0.0085 | seconds | test_kv_cache_memory_allocation_efficiency |
| alloc_latency_us_per_block | 7.24 - 8.47 | μs/block | test_kv_cache_memory_allocation_efficiency |
| **Power Efficiency** | | | |
| mean_power_w | 348.2 | watts | test_system_power_draw |
| **Total Training Metrics** | **1** | | |

---

## Summary Statistics

| Category | Metric Count | Key Insight |
|----------|--------------|------------|
| **Functional** | 20 | All correctness errors < 0.04 (excellent accuracy) |
| **Inference** | 361 | Max throughput: 1.4M tokens/sec; Latencies: 57-475 ms |
| **NVBandwidth** | 238 | HBM: 1.6TB/s (on-GPU); PCIe P2P: 17.3GB/s; PCIe H2D/D2H: 21-31GB/s |
| **P2P** | 3 unique | AllReduce: 10-11ms; P2P: 15-92μs latency |
| **Training** | 1 | Compile speedup: 1.049x |
| **TOTAL** | **623** | 19,466 individual data points captured |

---

## Key Performance Highlights

### 🎯 Compute Performance
- **Tokenizer Peak:** 1.4M tokens/sec
- **Prefill Throughput:** 94.7K tokens/sec  
- **P95 Inference Latency:** 88.7ms
- **Compile Speedup:** 1.049x faster with torch.compile

### 🔗 Communication Performance
- **AllReduce Time:** 10.7-11.6ms (collective sync)
- **HBM Memory Bandwidth:** 1,663 GB/s (on-GPU)
- **PCIe P2P GPU-to-GPU:** 17.3 GB/s
- **PCIe Host-to-Device:** 21.89 GB/s
- **PCIe Device-to-Host:** 31.78 GB/s
- **NCCL P2P (over PCIe):** 25.1-622.72 GB/s (collective bandwidth)

### ✅ Correctness & Reliability
- **BF16 vs FP16 error:** 0.29%
- **INT8 vs FP16 error:** 1.53%
- **FP8 vs FP16 error:** 3.74%
- **ECC errors:** 0 (zero errors detected)

### 💾 Memory & Efficiency
- **KV Cache allocation:** 7.24-8.47 μs per block
- **System power draw:** 348.2W
