# AI Head Node — Node Validation Report

## Run Context

- **Generated:** 2026-04-14 16:29:56
- **Pytest command line:** `-vv tests/test_performance_suite.py -k test_performance_case_suite`
- **KPI source files:** `artifacts/node_validation_kpis.jsonl`, `artifacts/performance_suite_kpis.jsonl`

> Add environment details here when available (recommended): host identifier, OS/kernel, GPU model/count, driver/CUDA, container image, repo commit SHA, and key env vars (e.g., `CUDA_VISIBLE_DEVICES`, `NCCL_*`).

---

## Summary

| Metric | Value |
|---|---:|
| Total testcases captured | 25 |
| PASS | 5 |
| FAIL | 14 |
| SKIP | 6 |
| Overall status | **FAIL** (one or more testcase failures were detected) |

### Top 5 KPIs

| KPI | Value | Test |
|---|---:|---|
| baseline_throughput_tokens_per_sec | 683.720 | test_predictable_memory_orchestration_kv_cache_offload |
| samples_per_sec | 305.710 | test_hf_encoder_training_throughput |
| aggregate_fps_target | 200.000 | test_vision_object_detection_yolo_fps |
| bandwidth_gbps | 0.595 | test_looped_rdma_mmio_dma_stress_soak |
| p95_latency_ms | 536.420 | test_hf_causal_lm_inference_latency_enterprise_workloads_llama_7b |

---

## Failures (grouped)

### Timeouts

| Test | Reason |
|---|---|
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_003]` | Timeout (>600.0s) from pytest-timeout | 

### Command failed

| Test | Reason |
|---|---|
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_004]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_004 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_005]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_005 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_006]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_006 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_007]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_007 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_008]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_008 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_009]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_009 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_016]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_016 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_017]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_017 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_018]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_018 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_021]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_021 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_022]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_022 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_023]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_023 |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_025]` | Command failed for AI_HOSTCPU_PERF_TEST_ID_025 |

---

## Passes

| Test | Note |
|---|---|
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_001]` | Acceptance criteria satisfied |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_002]` | Acceptance criteria satisfied |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_019]` | Acceptance criteria satisfied |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_020]` | Acceptance criteria satisfied |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_024]` | Acceptance criteria satisfied |

---

## Skips

| Test | Skip reason |
|---|---|
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_010]` | Missing required binaries: ['trtllm-bench'] |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]` | Missing required binaries: ['genai-perf'] |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_012]` | Planned placeholder case; details not provided yet |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_013]` | Planned placeholder case; details not provided yet |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_014]` | Planned placeholder case; details not provided yet |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_015]` | Planned placeholder case; details not provided yet |

---

## KPIs (collapsed)

<details>
<summary>KPI Summary (raw)</summary>

```text
AI_HOSTCPU_PERF_TEST_ID_001 | tokenizer=meta-llama/Llama-3.1-8B; throughput_tokens_per_sec={'len_128': {'threads_1': 930527.59, 'threads_2': 930829.18, 'threads_4': 936086.76, 'threads_8': 93439[...]AI_HOSTCPU_PERF_TEST_ID_002 | mode=execution; status=passed; results=[{'command': 'nvbandwidth -t host_to_device_memcpy_ce -b 64 -i 50', 'executed_command': '/root/workloads/nvbandwidth/nvbandwi[...]AI_HOSTCPU_PERF_TEST_ID_003 | mode=catalog; title=vLLM Serve + Load FP8 8B; subsystem=Inference FP8; workload=Long context 8k, multi-user serving; expected=Stable serving with ITL and tail laten[...]AI_HOSTCPU_PERF_TEST_ID_004 | mode=execution; status=failed; results=[{'command': 'vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 8192', 'e[...]AI_HOSTCPU_PERF_TEST_ID_005 | mode=execution; status=failed; results=[{'command': 'vllm serve meta-llama/Llama-3.1-70B-Instruct --quantization fp8 --tensor-parallel-size 4 --max-model-len 32768 [...]AI_HOSTCPU_PERF_TEST_ID_006 | mode=execution; status=failed; results=[{'command': 'vllm serve meta-llama/Llama-3.1-70B-Instruct --dtype bfloat16 --tensor-parallel-size 4 --max-model-len 16384', [...]AI_HOSTCPU_PERF_TEST_ID_007 | mode=execution; status=failed; results=[{'command': 'python3 -m vllm.benchmarks.benchmark_serving --backend vllm --dataset sharegpt --request-rate poisson --max-con[...]AI_HOSTCPU_PERF_TEST_ID_008 | mode=execution; status=failed; results=[{'command': 'vllm serve mistralai/Mixtral-8x22B-v0.1 --tensor-parallel-size 4 --quantization fp8 --max-model-len 32768 --dis[...]AI_HOSTCPU_PERF_TEST_ID_009 | mode=execution; status=failed; results=[{'command': 'python3 -m vllm.benchmarks.benchmark_serving --backend vllm --dataset sharegpt --request-rate burst --burst-len[...]AI_HOSTCPU_PERF_TEST_ID_010 | mode=catalog; title=trtllm-bench offline throughput; subsystem=Text generation offline benchmark; workload=Fixed synthetic dataset throughput run; expected=Stable t[...]AI_HOSTCPU_PERF_TEST_ID_011 | mode=catalog; title=GenAI-Perf end-to-end HTTP benchmark; subsystem=E2E serving + networking; workload=Dual-server concurrent load isolation; expected=Both server i[...]AI_HOSTCPU_PERF_TEST_ID_012 | mode=catalog; title=Reserved performance case; subsystem=TBD; workload=TBD; expected=; planned_only=True; commands=[]AI_HOSTCPU_PERF_TEST_ID_013 | mode=catalog; title=Reserved performance case; subsystem=TBD; workload=TBD; expected=; planned_only=True; commands=[]AI_HOSTCPU_PERF_TEST_ID_014 | mode=catalog; title=Reserved performance case; subsystem=TBD; workload=TBD; expected=; planned_only=True; commands=[]AI_HOSTCPU_PERF_TEST_ID_015 | mode=catalog; title=Reserved performance case; subsystem=TBD; workload=TBD; expected=; planned_only=True; commands=[]AI_HOSTCPU_PERF_TEST_ID_016 | mode=execution; status=failed; results=[{'command': 'vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8 --dtype float8 --max-model-len 16384 --enable-chunk[...]AI_HOSTCPU_PERF_TEST_ID_017 | mode=execution; status=failed; results=[{'command': 'vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8 --dtype float8 --max-model-len 16384 --enable-chunk[...]AI_HOSTCPU_PERF_TEST_ID_018 | mode=execution; status=failed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 1K -e 4G -f 2 -n 1[...]AI_HOSTCPU_PERF_TEST_ID_019 | mode=execution; status=passed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 8 -e 256K -f 2 -n [...]AI_HOSTCPU_PERF_TEST_ID_020 | mode=execution; status=passed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 256K -e 256M -f 2 [...]AI_HOSTCPU_PERF_TEST_ID_021 | mode=execution; status=failed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/reduce_scatter_perf -g 1 -b 1K -e 4G -f 2 [...]AI_HOSTCPU_PERF_TEST_ID_022 | mode=execution; status=failed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_gather_perf -g 1 -b 1K -e 4G -f 2 -n [...]AI_HOSTCPU_PERF_TEST_ID_023 | mode=execution; status=failed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/alltoall_perf -g 1 -b 1K -e 4G -f 2 -n 10[...]AI_HOSTCPU_PERF_TEST_ID_024 | mode=execution; status=passed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 -x NCCL_P2P_DISABLE=1 ./build/all_reduce_perf -g [...]AI_HOSTCPU_PERF_TEST_ID_025 | mode=execution; status=failed; results=[{'command': 'mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 8M -e 4G -f 2 -n [...]... (node_validation_kpis.jsonl and performance_suite_kpis.jsonl content continues)
```

</details>

---

## Detailed tables

> Note: The section below preserves the existing HTML/CSS for wide tables. If you prefer pure Markdown tables for better diffs, we can convert this next.

<style>
.report-scroll { overflow-x: auto; resize: horizontal; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px; }
.report-table { border-collapse: collapse; width: 100%; min-width: 1400px; table-layout: auto; font-size: 12px; }
.report-table th, .report-table td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }
.report-table th { background: #f6f8fa; position: sticky; top: 0; z-index: 1; }
.cell-wrap { white-space: normal; word-break: break-word; max-width: 540px; }
</style>

### Execution Status Table

<div class="report-scroll">
<table class="report-table">
<thead><tr><th style="min-width: 380px;">Test</th><th style="min-width: 360px;">Purpose</th><th style="min-width: 360px;">Result</th><th style="min-width: 100px;">Status</th><th style="min-width:[...]<tbody>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_001]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_002]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_003]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_004]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_005]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_006]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_007]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_008]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_009]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_010]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_012]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_013]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_014]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_015]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_016]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_017]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_018]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_019]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_020]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_021]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_022]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_023]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_024]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
<tr><td><div class="cell-wrap">tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_025]</div></td><td><div class="cell-wrap">Catalog + optional execution harness [...]</td></tr>
</tbody>
</table>
</div>

### Skipped Tests

| Test | Skip Reason |
|---|---|
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_010]` | Missing required binaries: ['trtllm-bench']") |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]` | Missing required binaries: ['genai-perf']") |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_012]` | Planned placeholder case; details not provided yet') |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_013]` | Planned placeholder case; details not provided yet') |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_014]` | Planned placeholder case; details not provided yet') |
| `tests/test_performance_suite.py::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_015]` | Planned placeholder case; details not provided yet') |