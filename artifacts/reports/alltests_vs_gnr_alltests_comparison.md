# Log Comparison Report

Generated: 2026-05-11 14:09:27
Sources: alltests.log vs gnr_alltests.log
Scope: side-by-side comparison of common pytest session data and KPI records

## Execution Summary

| Source | Sessions | Collected | Executed | Passed | Failed | Skipped | Pass Rate | Performance KPIs | Threshold KPIs | Workload Payloads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alltests.log | 13 | 410 | 375 | 303 | 72 | 35 | 80.8% | 78 | 24 | 14 |
| gnr_alltests.log | 13 | 410 | 355 | 297 | 58 | 55 | 83.7% | 76 | 24 | 14 |

## Comparison Summary

| Category | RED | MATCHING (light green) | EXCEEDS (dark green) | Common Rows |
|---|---:|---:|---:|---:|
| Performance | 35 | 22 | 19 | 76 |
| Threshold | 6 | 5 | 13 | 24 |

## RED Results (DMR Underperformed)

### Performance Metrics (RED)

| Domain | KPI | DMR measured | GNR measured | DMR eff | GNR eff | Match |
|---|---|---:|---:|---:|---:|---|
| bottleneck | 4-GPU FP16 scaling efficiency | 1799.163 | 1850.456 | 101.610% | 102.230% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| bottleneck | 4-GPU aggregate FP16 TFLOPS | 1829.814 | 1872.165 | 60.510% | 61.910% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| bottleneck | CPU FP32 (1 processes × 32 threads) | 1.957 | 5.051 | 7.180% | 64.900% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| bottleneck | CPU FP32 (4 processes × 52 threads) | 2.674 | 3.832 | 9.820% | 49.240% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| bottleneck | GPU 0 H2D PCIe bandwidth | 53.736 | 55.543 | 85.300% | 88.160% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| bottleneck | GPU 0 HBM3 bandwidth | 1.853 | 1.854 | 92.650% | 92.720% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| bottleneck | GPU 1 HBM3 bandwidth | 1.853 | 1.857 | 92.670% | 92.830% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| bottleneck | GPU 2 HBM3 bandwidth | 1.855 | 3.561 | 92.750% | 178.040% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| cpu | All-core FP32 GFLOPS (BLAS) | 9268.680 | 6577.199 | 34.030% | 84.510% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| cpu | FP32 scaling efficiency (8 cores) | 310.911 | 805.778 | 105.880% | 112.190% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| cpu | Parallel speedup at 16 cores | 660.719 | 1590.432 | 97.660% | 105.960% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| cpu | Parallel speedup at 2 cores | 74.317 | 194.843 | 96.490% | 100.240% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| cpu | Parallel speedup at 32 cores | 1216.118 | 3183.825 | 90.940% | 106.940% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| cpu | Parallel speedup at 4 cores | 142.294 | 469.648 | 105.020% | 119.810% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| cpu | Parallel speedup at 64 cores | 2391.389 | 6254.490 | 91.430% | 108.350% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | BF16 TFLOPS GPU 1 | 510.991 | 514.807 | 67.590% | 68.100% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | BF16 TFLOPS GPU 3 | 488.632 | 514.521 | 49.410% | 68.060% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | FP16 TFLOPS GPU 1 | 477.256 | 484.101 | 63.130% | 64.030% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | FP16 TFLOPS GPU 3 | 442.197 | 480.478 | 44.710% | 63.560% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | FP32 GFLOPS GPU 2 | 35253.555 | 45853.752 | 52.620% | 68.440% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | FP64 TFLOPS GPU 1 | 36.014 | 36.351 | 105.920% | 106.910% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | FP64 TFLOPS GPU 2 | 32.100 | 41.468 | 94.410% | 121.960% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | HBM3 bandwidth GPU 0 | 1.852 | 1.855 | 92.610% | 92.770% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | HBM3 bandwidth GPU 1 | 1.854 | 1.856 | 92.720% | 92.790% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | HBM3 bandwidth GPU 3 | 3.569 | 1.858 | 91.510% | 92.900% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | INT8-proxy TOPS GPU 1 | 549.562 | 553.357 | 13.880% | 13.980% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | INT8-proxy TOPS GPU 2 | 502.965 | 543.851 | 12.710% | 13.740% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | INT8-proxy TOPS GPU 3 | 543.937 | 550.946 | 13.740% | 13.920% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | Multi-GPU FP16 aggregate (4 GPUs) | 1807.010 | 1851.051 | 95.040% | 103.090% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | PCIe D2H bandwidth GPU 0 | 55.241 | 55.432 | 87.680% | 87.990% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | PCIe D2H bandwidth GPU 1 | 55.264 | 55.455 | 87.720% | 88.020% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| gpu | PCIe D2H bandwidth GPU 2 | 55.270 | 57.272 | 87.730% | 90.910% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| interconnect | D2H PCIe BW GPU 0 | 55.300 | 55.521 | 87.780% | 88.130% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| interconnect | D2H PCIe BW GPU 1 | 55.243 | 55.522 | 87.690% | 88.130% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| interconnect | D2H PCIe BW GPU 2 | 55.308 | 57.352 | 87.790% | 91.030% | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |

### Threshold Metrics (RED)

| Domain | Test | Metric | Rule | DMR measured | GNR measured | DMR threshold | GNR threshold | Match |
|---|---|---|---|---:|---:|---:|---:|---|
| ai_pipeline | test_async_prefetch_gpu_overlap_efficiency | overlap_efficiency_pct | &gt;= | 25.260 | 27.160 | 60.000 | 60.000 | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| ai_pipeline | test_model_load_time_to_gpu &#124; openai-community/gpt2 | load_bandwidth_gbs | &gt;= | 0.199 | 0.325 | 1.000 | 1.000 | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| ai_pipeline | test_tokenizer_multi_thread_scaling | scaling_ratio | &gt;= | 1.476 | 1.499 | 2.000 | 2.000 | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| ai_pipeline | test_tokenizer_throughput_by_seq_len | tokens_per_sec | &gt;= | 474000.700 | 553865.400 | 50000.000 | 50000.000 | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| node_validation | test_p2p_bandwidth_latency_binary | unidir_min_gbs | &gt;= | 37.060 | 38.410 | 40.000 | 40.000 | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |
| node_validation | test_p2p_bandwidth_pytorch_fallback | min_gbs | &gt;= | 34.170 | 40.920 | 40.000 | 40.000 | <span style="color:#c62828;background:#ffebee;font-weight:700;padding:1px 6px;border-radius:4px;">RED</span> |

## MATCHING Results (Within 5% Parity)

### Performance Metrics (MATCHING)

| Domain | KPI | DMR measured | GNR measured | DMR eff | GNR eff | Match |
|---|---|---:|---:|---:|---:|---|
| bottleneck | GPU 0 FP16 Tensor Core | 472.943 | 450.861 | 62.560% | 59.640% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| bottleneck | GPU 1 H2D PCIe bandwidth | 55.611 | 55.491 | 88.270% | 88.080% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| bottleneck | GPU 2 H2D PCIe bandwidth | 56.044 | 55.517 | 88.960% | 88.120% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| bottleneck | GPU 3 H2D PCIe bandwidth | 55.933 | 55.509 | 88.780% | 88.110% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| cpu | Parallel speedup at 1 cores | 38.972 | 98.029 | 100.240% | 100.230% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| cpu | Parallel speedup at 8 cores | 328.812 | 860.651 | 113.260% | 109.420% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | BF16 TFLOPS GPU 0 | 510.777 | 489.973 | 67.560% | 64.810% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | FP16 TFLOPS GPU 0 | 475.592 | 454.605 | 62.910% | 60.130% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | FP32 GFLOPS GPU 0 | 36990.378 | 35239.177 | 55.210% | 52.600% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | FP32 GFLOPS GPU 1 | 37349.718 | 37022.075 | 55.750% | 55.260% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | HBM3 bandwidth GPU 2 | 1.852 | 3.557 | 92.600% | 91.200% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | INT8-proxy TOPS GPU 0 | 547.033 | 522.486 | 13.820% | 13.200% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | PCIe D2H bandwidth GPU 3 | 57.045 | 55.460 | 90.550% | 88.030% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | PCIe H2D bandwidth GPU 0 | 55.657 | 55.133 | 88.340% | 87.510% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | PCIe H2D bandwidth GPU 1 | 55.885 | 54.844 | 88.710% | 87.050% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | PCIe H2D bandwidth GPU 2 | 56.057 | 54.705 | 88.980% | 86.830% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| gpu | PCIe H2D bandwidth GPU 3 | 56.020 | 54.635 | 88.920% | 86.720% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| interconnect | D2H PCIe BW GPU 3 | 57.081 | 55.531 | 90.610% | 88.150% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| interconnect | H2D PCIe BW GPU 0 | 56.097 | 55.568 | 89.040% | 88.200% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| interconnect | H2D PCIe BW GPU 1 | 56.062 | 55.624 | 88.990% | 88.290% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| interconnect | H2D PCIe BW GPU 2 | 56.106 | 55.633 | 89.060% | 88.310% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| interconnect | H2D PCIe BW GPU 3 | 56.053 | 55.632 | 88.970% | 88.300% | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |

### Threshold Metrics (MATCHING)

| Domain | Test | Metric | Rule | DMR measured | GNR measured | DMR threshold | GNR threshold | Match |
|---|---|---|---|---:|---:|---:|---:|---|
| ai_pipeline | test_kv_cache_memory_allocation_efficiency | blocks_per_sec | &gt;= | 120540.000 | 118980.000 | 10000.000 | 10000.000 | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| ai_pipeline | test_model_weight_load_pipeline_bandwidth | end_to_end_gbs | &gt;= | 0.858 | 0.827 | 1.000 | 1.000 | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| ai_pipeline | test_storage_sequential_read_throughput | read_bandwidth_gbs | &gt;= | 1.568 | 1.518 | 1.000 | 1.000 | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| ai_pipeline | test_tensor_parallel_allreduce_overhead_fraction | comm_overhead_pct | &lt;= | 98.560 | 98.740 | 20.000 | 20.000 | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |
| node_validation | test_llm_training_large_multigpu &#124; meta-llama/Meta-Llama-3.1-8B-Instruct | mean_sync_pct | &lt;= | 0.000 | 0.000 | 20.000 | 20.000 | <span style="color:#2e7d32;background:#e8f5e9;font-weight:700;padding:1px 6px;border-radius:4px;">MATCHING</span> |

## EXCEEDS Results (DMR Outperformed by >5%)

### Performance Metrics (EXCEEDS)

| Domain | KPI | DMR measured | GNR measured | DMR eff | GNR eff | Match |
|---|---|---:|---:|---:|---:|---|
| bottleneck | 2-GPU FP16 scaling efficiency | 905.199 | 931.498 | 107.030% | 101.720% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| bottleneck | CPU DRAM BW (multi-thread STREAM Copy) | 1477.639 | 184.145 | 481.000% | 59.940% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| bottleneck | GPU 3 HBM3 bandwidth | 3.557 | 1.856 | 177.830% | 92.810% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| bottleneck | PCIe P2P P2P BW GPU 0-&gt;1 | 51.811 | 41.473 | 82.240% | 65.830% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| cpu | Aggregate DRAM bandwidth (multi-thread) | 2168.793 | 190.481 | 705.990% | 62.010% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| cpu | Single-core FP32 GFLOPS | 945.980 | 854.345 | 777.940% | 702.590% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| gpu | BF16 TFLOPS GPU 2 | 457.892 | 477.123 | 60.570% | 48.240% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| gpu | FP16 TFLOPS GPU 2 | 422.109 | 441.107 | 55.830% | 44.600% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| gpu | FP32 GFLOPS GPU 3 | 45932.750 | 37043.657 | 68.560% | 55.290% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| gpu | FP64 TFLOPS GPU 0 | 35.898 | 33.661 | 105.580% | 99.000% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| gpu | FP64 TFLOPS GPU 3 | 41.582 | 35.652 | 122.300% | 104.860% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| interconnect | Concurrent H2D BW all 4 GPUs | 206.813 | 188.955 | 82.070% | 74.980% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| interconnect | PCIe P2P P2P BW GPU 0-&gt;1 | 52.053 | 41.807 | 82.620% | 66.360% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| interconnect | PCIe P2P P2P BW GPU 0-&gt;2 | 46.134 | 41.615 | 73.230% | 66.050% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| interconnect | PCIe P2P P2P BW GPU 0-&gt;3 | 44.194 | 40.920 | 70.150% | 64.950% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| interconnect | PCIe P2P P2P BW GPU 1-&gt;2 | 44.609 | 41.720 | 70.810% | 66.220% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| interconnect | PCIe P2P P2P BW GPU 1-&gt;3 | 44.512 | 41.077 | 70.650% | 65.200% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| interconnect | PCIe P2P P2P BW GPU 2-&gt;3 | 51.706 | 41.008 | 82.070% | 65.090% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| memory | Aggregate STREAM Triad BW | 816.562 | 126.463 | 265.810% | 41.170% | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |

### Threshold Metrics (EXCEEDS)

| Domain | Test | Metric | Rule | DMR measured | GNR measured | DMR threshold | GNR threshold | Match |
|---|---|---|---|---:|---:|---:|---:|---|
| ai_pipeline | test_batch_assembly_gpu_latency | assemble_us_per_batch | &lt;= | 18.107 | 22.533 | 500.000 | 500.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_checkpoint_save_restore_throughput | restore_bandwidth_gbs | &gt;= | 4.433 | 3.312 | 1.000 | 1.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_checkpoint_save_restore_throughput | save_bandwidth_gbs | &gt;= | 1.606 | 0.861 | 0.500 | 0.500 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_cpu_preprocessing_pipeline_throughput | total_tokens_per_sec | &gt;= | 654788.000 | 557824.000 | 200000.000 | 200000.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_dataloader_cpu_to_pinned_throughput | samples_per_sec | &gt;= | 277820.300 | 218413.900 | 1000.000 | 1000.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_storage_random_4k_read_iops | iops | &gt;= | 12473.000 | 10912.000 | 5000.000 | 5000.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_storage_sequential_write_throughput | write_bandwidth_gbs | &gt;= | 1.257 | 1.115 | 0.500 | 0.500 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_tcp_loopback_throughput | throughput_gbps | &gt;= | 101.573 | 49.882 | 5.000 | 5.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_time_to_first_token &#124; openai-community/gpt2 &#124; input_len=128 | ttft_p95_ms | &lt;= | 5.870 | 7.460 | 200.000 | 200.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_time_to_first_token &#124; openai-community/gpt2 &#124; input_len=32 | ttft_p95_ms | &lt;= | 6.030 | 7.530 | 200.000 | 200.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| ai_pipeline | test_time_to_first_token &#124; openai-community/gpt2 &#124; input_len=512 | ttft_p95_ms | &lt;= | 5.910 | 7.530 | 500.000 | 500.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| node_validation | test_vision_image_generation_diffusers_throughput &#124; stabilityai/sd-turbo | throughput_img_per_sec | &gt;= | 46.849 | 39.904 | 10.000 | 10.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |
| node_validation | test_vision_object_detection_yolo_fps &#124; yolov8x.pt | aggregate_fps | &gt;= | 95.010 | 88.940 | 200.000 | 200.000 | <span style="color:#1b5e20;background:#c8e6c9;font-weight:700;padding:1px 6px;border-radius:4px;">EXCEEDS</span> |

## Notes

- RED = DMR is worse than GNR for that metric/rule.
- MATCHING (light green) = DMR is approximately at parity with GNR (≤5% difference).
- EXCEEDS (dark green) = DMR performs better than GNR by more than 5%.
- The CSV companion contains the full comparable row set for downstream filtering.
