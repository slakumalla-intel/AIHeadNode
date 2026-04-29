# KPI Inventory by Validation Domain

Generated: 2026-04-28
Source: artifacts/*_kpis.jsonl files

## Summary

Total domains: 7
Total KPI records: 184

## Agentic Ai

**Test Count:** 8

**Metrics Captured:** 31

### Key Metrics

- avg_dispatch_latency_ms
- avg_latency_ms
- avg_reasoning_latency_ms
- avg_serialization_ms
- avg_sync_latency_us
- avg_tool_latency_ms
- device
- iterations
- max_dispatch_latency_ms
- max_latency_ms
- max_sync_latency_us
- memory_baseline_mb
- memory_deltas_by_tool_count
- min_dispatch_latency_ms
- min_latency_ms
- min_sync_latency_us
- num_tools
- num_tools_per_turn
- num_turns
- p50_latency_ms
- p95_latency_ms
- p99_latency_ms
- parallel_time_s
- per_turn_latencies_ms
- reasoning_latencies_by_tokens
- ... and 6 more

### Sample Tests

- TC-AGENT-01a: Single tool dispatch
- TC-AGENT-02: Reasoning phase latency
- TC-AGENT-03: CPU-GPU sync overhead

## Ai Pipeline

**Test Count:** 26

**Metrics Captured:** 83

### Key Metrics

- alloc_latency_us_per_block
- alloc_time_sec
- allreduce_time_ms
- assemble_us_per_batch
- avg_latency_us
- batch_size
- batches_per_sec
- block_bytes
- block_size_bytes
- block_size_tokens
- blocks_per_sec
- checkpoint_size_gb
- comm_overhead_pct
- compute_only_sec
- compute_time_ms
- elapsed_sec
- end_to_end_gbs
- file_size_gb
- head_dim
- hidden_dim
- input_len
- iops
- iterations
- load_bandwidth_gbs
- load_time_sec
- ... and 58 more

### Sample Tests

- test_tokenizer_throughput_by_seq_len
- test_tokenizer_throughput_by_seq_len
- test_tokenizer_throughput_by_seq_len

## Mlperf Suite

**Test Count:** 35

**Metrics Captured:** 11

### Key Metrics

- category
- commands
- description
- expected
- kpi_keys
- mode
- prerequisites
- results
- status
- sub_category
- title

### Sample Tests

- unknown
- unknown
- unknown

## Node Validation

**Test Count:** 43

**Metrics Captured:** 173

### Key Metrics

- assertion_size_bytes
- assertion_size_label
- attempted_pairs
- available
- avg_latency_us
- bandwidth_gbps
- baseline_active_compute_processes
- baseline_mean_latency_ms
- baseline_p50_latency_ms
- baseline_p95_latency_ms
- baseline_p99_latency_ms
- baseline_page_faults_per_sec
- baseline_throughput_tokens_per_sec
- batch_size
- benchmark_mode
- bidir_mean_gbs
- bidir_min_gbs
- borderline_gpus
- boundary_gbs
- bw_by_size_gbs
- bw_results
- command_reg_read_1
- command_reg_read_2
- completion_status
- completion_time_sec
- ... and 148 more

### Sample Tests

- test_gpu_enumeration
- test_pcie_topology_discovery
- test_topology_consistency

## Performance Suite

**Test Count:** 35

**Metrics Captured:** 14

### Key Metrics

- busbw_mean_gbs
- busbw_peak_gbs
- busbw_samples
- commands
- expected
- mode
- planned_only
- results
- status
- subsystem
- throughput_tokens_per_sec
- title
- tokenizer
- workload

### Sample Tests

- unknown
- unknown
- unknown

## System Health

**Test Count:** 36

**Metrics Captured:** 61

### Key Metrics

- actual_sum
- bf16_vs_fp16_mean_rel_error
- clock_drop_ratio
- clock_ratio
- compiled_time_s
- concurrent_time_s
- concurrent_vs_sequential_ratio
- correct
- corrected_ecc_errors
- current_clock_mhz
- device_idx
- duration_sec
- eager_time_s
- expected
- expected_sum
- final_val
- floor_ratio
- fp16_tflops
- fp8_vs_fp16_mean_rel_error
- gemm_elapsed_ms
- gpu0_limit_w
- gpu0_power_w
- gpu0_ratio
- gpu0_retired_pages_dbe
- gpu1_limit_w
- ... and 36 more

### Sample Tests

- test_gpu_temperature_within_safe_limits[0]
- test_gpu_temperature_within_safe_limits[1]
- test_gpu_temperature_within_safe_limits[2]

## Vllm Prefill

**Test Count:** 1

**Metrics Captured:** 4

### Key Metrics

- dtype
- model
- num_test_cases
- tensor_parallel_size

### Sample Tests

- test_vllm_prefill_quick

