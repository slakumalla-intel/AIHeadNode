# AIHeadNode

Linux-first validation suite for AI head-node bring-up, performance qualification,
and workload acceptance on multi-GPU systems (for example 4x H100).

## Scope

Main Linux execution paths in this repo:


## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional if PyTorch is not already present in your image:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

System packages often needed for native builds:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# RHEL / Rocky / CentOS
sudo dnf install -y python3-devel gcc gcc-c++ make
```

## Core Validation Procedure

### 1) Run All Validation Stages

```bash
chmod +x run_all_validation_tests.sh
./run_all_validation_tests.sh
```

Stages executed in order:

1. Bottleneck tests
2. Node validation tests
3. Workload acceptance
4. Performance suite
5. MLPerf suite

### 2) Run Workload Acceptance Only

```bash
chmod +x run_workload_acceptance.sh
./run_workload_acceptance.sh
```

### 3) Save Console Logs to File

Persist pytest logs with timestamps and logger names:

```bash
pytest -m workload_acceptance -vv --log-file=node_validation.log --log-file-level=INFO
```

Capture full console output (stdout + stderr) as well:

```bash
pytest -m workload_acceptance -vv 2>&1 | tee node_validation_full.log
```

## Useful Runner Options

For staged validation:

```bash
# Catalog-only mode for both performance and MLPerf suites
./run_all_validation_tests.sh --performance-catalog-only
./run_all_validation_tests.sh --performance-case-ids 001,002,018
./run_all_validation_tests.sh --stop-on-failure
./run_all_validation_tests.sh --skip-consolidated-report
./run_all_validation_tests.sh -- -s --log-cli-level=INFO
```

By default, the staged runner executes a final consolidated pytest pass to refresh
`TESTCASE_EXECUTION_FINAL_REPORT.md` with a single unified report snapshot across
the full `tests/` directory.

For workload acceptance:

```bash
./run_workload_acceptance.sh --min-duration-sec 600
./run_workload_acceptance.sh --kv-context-length 16384 --kv-batch-size 8
./run_workload_acceptance.sh --skip-large-models
```

## Direct Pytest Commands

```bash
pytest -m bottleneck -vv tests/test_bottleneck_analysis.py
pytest -m node_validation -vv tests/test_node_validation.py
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite
```

Other useful selectors:

```bash
pytest -m workload_acceptance -vv
pytest -m rdma_validation -vv tests/test_node_validation.py
pytest -m "not slow" -vv
```

## Environment Defaults (Workload Path)

Default behavior for large-model workload tests:


Common overrides:

```bash
export NODE_WORKLOAD_AUTO_INSTALL=0
export NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=1800
export HF_ENABLE_LARGE_MODELS=0
export HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0
export HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=300
```

## Performance Suite Controls

`tests/test_performance_suite.py` uses these controls (execution is enabled by default):

```bash
# Optional: force catalog-only mode
# export PERF_EXECUTE_CASES=0
export PERF_CASE_IDS=001,002,018
export PERF_CMD_TIMEOUT_SEC=1800
export PERF_SUITE_KPI_FILE=artifacts/performance_suite_kpis.jsonl
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
```

## MLPerf Workflow (4x H100)

Download models:

```bash
chmod +x scripts/download_mlperf_models.sh
scripts/download_mlperf_models.sh --model-dir /models/mlperf
```

Download datasets:

```bash
chmod +x scripts/download_mlperf_datasets.sh
scripts/download_mlperf_datasets.sh --data-dir /data/mlperf
```

Run MLPerf orchestration script:

```bash
chmod +x scripts/run_mlperf_h100x4.sh
scripts/run_mlperf_h100x4.sh --data-dir /data/mlperf --model-dir /models/mlperf --num-gpus 4 --tp-size 4
```

Or run the MLPerf pytest suite directly:

```bash
# Optional: force catalog-only mode
# export ML_EXECUTE_CASES=0
export ML_CASE_IDS=001,006,014
export ML_CMD_TIMEOUT_SEC=3600
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite
```

## Artifacts and Reports

Main generated outputs:


Node KPI output path can be overridden with:

```bash
export NODE_VALIDATION_KPI_FILE=artifacts/node_validation_kpis.jsonl
```

## Reference

# AIHeadNode Validation Suite

A **vendor-neutral** Linux-first validation and competitive benchmarking suite
for AI head nodes built from any combination of server-class CPUs and AI
accelerators. The suite covers CPU, GPU, memory, interconnect, agentic AI
workloads, and system health.

Tested hardware configurations:

| CPU stack | GPU / Accelerator | Interconnect |
|---|---|---|
| Intel Xeon Platinum (SPR / EMR) | NVIDIA H100 SXM5 / PCIe | NVLink 4.0 |
| Intel Xeon Platinum (SPR / EMR) | NVIDIA H200 SXM5 | NVLink 4.0 |
| AMD EPYC Genoa / Bergamo | AMD Instinct MI300X | xGMI 3.0 |
| AMD EPYC Genoa | NVIDIA A100 SXM4 / PCIe | NVLink 3.0 |
| Any x86-64 server | Any CUDA ≥ 7.0 or ROCm device | PCIe Gen 5 |

All thresholds are expressed as **percentages of detected theoretical peak**
(not absolute vendor-specific numbers), so pass/fail comparisons are valid
across platforms.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Competitive Analysis KPIs](#competitive-analysis-kpis)
3. [Test Modules](#test-modules)
4. [Hardware Env Vars](#hardware-env-vars)
5. [Running the Suite](#running-the-suite)
6. [Agentic AI Validation](#agentic-ai-validation)
7. [MLPerf Workflow](#mlperf-workflow)
8. [Artifacts & Reports](#artifacts--reports)
9. [Environment Reference](#environment-reference)

---

## Quick Start

```bash
# Clone and prepare
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If PyTorch is not already in your image (CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For AMD ROCm:
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# System packages (Ubuntu/Debian)
sudo apt-get install -y python3-dev build-essential
```

Run the full staged pipeline:

```bash
chmod +x run_all_validation_tests.sh
./run_all_validation_tests.sh
```

Stages executed in order: bottleneck → node validation → workload acceptance →
performance suite → MLPerf suite.

---

## Competitive Analysis KPIs

The following KPIs are captured and written to structured JSONL artifact files.
They are percentage-of-peak or latency-based so values can be compared fairly
across Intel + NVIDIA and AMD + AMD configurations.

### GPU Compute KPIs

| KPI | Unit | Threshold | Artifact file |
|---|---|---|---|
| FP16 matrix TFLOPS | TFLOPS | ≥ 75 % of detected peak | `gpu_validation_kpis.jsonl` |
| BF16 matrix TFLOPS | TFLOPS | ≥ 75 % of detected peak | `gpu_validation_kpis.jsonl` |
| INT8 matrix TOPS | TOPS | ≥ 70 % of detected peak | `gpu_validation_kpis.jsonl` |
| FP32 TFLOPS | TFLOPS | ≥ 70 % of detected peak | `gpu_validation_kpis.jsonl` |
| FP64 TFLOPS | TFLOPS | ≥ 70 % of detected peak | `gpu_validation_kpis.jsonl` |
| Multi-GPU scaling efficiency | % | ≥ 90 % linear | `gpu_validation_kpis.jsonl` |
| Energy efficiency | TFLOPS / Watt | logged (higher is better) | `gpu_validation_kpis.jsonl` |

> **Reference peaks** — detected at runtime from pynvml:
> H100 SXM5 FP16 = 1979 TFLOPS, MI300X FP16 = 1307 TFLOPS, A100 FP16 = 312 TFLOPS.

### GPU Memory KPIs

| KPI | Unit | Threshold | Artifact file |
|---|---|---|---|
| HBM / VRAM capacity per GPU | GB | ≥ 98 % of spec | `gpu_validation_kpis.jsonl` |
| Memory bandwidth per GPU | TB/s | ≥ 80 % of detected peak | `gpu_validation_kpis.jsonl` |

> **Reference peaks** — H100 SXM5 = 3.35 TB/s, MI300X = 5.3 TB/s, A100 = 2.0 TB/s.

### CPU Compute KPIs

| KPI | Unit | Threshold | Artifact file |
|---|---|---|---|
| Single-core SIMD FP32 (AVX-512 or AVX2) | GFLOPS | ≥ 70 % of theoretical | `cpu_validation_kpis.jsonl` |
| All-core SIMD FP32 aggregate | GFLOPS | ≥ 70 % of theoretical | `cpu_validation_kpis.jsonl` |
| BLAS DGEMM throughput | GFLOPS | ≥ 70 % of theoretical | `cpu_validation_kpis.jsonl` |
| CPU utilization under load | % | ≥ 90 % | `cpu_validation_kpis.jsonl` |
| CPU frequency scaling | GHz | ≥ 90 % of base | `cpu_validation_kpis.jsonl` |
| Memory latency (random access) | ns | ≤ 150 ns | `cpu_validation_kpis.jsonl` |
| NUMA node balance | % | ≤ 10 % imbalance | `cpu_validation_kpis.jsonl` |

### Interconnect KPIs

| KPI | Unit | Threshold | Artifact file |
|---|---|---|---|
| CPU → GPU PCIe host-to-device bandwidth | GB/s | ≥ 80 % of Gen 5 peak (63 GB/s) | `interconnect_kpis.jsonl` |
| CPU → GPU PCIe device-to-host bandwidth | GB/s | ≥ 80 % of Gen 5 peak | `interconnect_kpis.jsonl` |
| GPU-to-GPU fabric bandwidth (NVLink / xGMI) | GB/s | ≥ 80 % of link spec | `interconnect_kpis.jsonl` |
| InfiniBand / RDMA bandwidth | GB/s | ≥ 90 % of link speed | `interconnect_kpis.jsonl` |

### LLM Serving / Inference KPIs

| KPI | Unit | Competitive target | Artifact file |
|---|---|---|---|
| Time to First Token (TTFT) | ms | ≤ 500 ms (enterprise) | `vllm_prefill_kpis.jsonl` |
| Inter-Token Latency (ITL) | ms | ≤ 30 ms / token | `vllm_prefill_kpis.jsonl` |
| Tokens per second (TPS) throughput | tok/s | ≥ 1 000 tok/s | `vllm_prefill_kpis.jsonl` |
| KV cache prefill efficiency | % | ≥ 75 % of HBM bandwidth | `vllm_prefill_kpis.jsonl` |

### Agentic AI KPIs

| KPI | Unit | Target | Artifact file |
|---|---|---|---|
| Tool dispatch overhead | ms | ≤ 5 ms | `agentic_ai_kpis.jsonl` |
| CPU → GPU synchronization latency | µs | ≤ 500 µs | `agentic_ai_kpis.jsonl` |
| Tool parallelization speedup | ×N | ≥ 0.7 × N | `agentic_ai_kpis.jsonl` |
| Agent turn P99 latency | ms | logged | `agentic_ai_kpis.jsonl` |
| Agent state serialization throughput | MB/s | ≥ 100 MB/s | `agentic_ai_kpis.jsonl` |

---

## Test Modules

| File | TC range | Focus area | KPI artifact |
|---|---|---|---|
| `tests/test_cpu_validation.py` | TC-CPU-01 – TC-CPU-25 | CPU topology, SIMD compute, NUMA, industrial KPIs | `cpu_validation_kpis.jsonl` |
| `tests/test_gpu_validation.py` | TC-GPU-01 – TC-GPU-15 | GPU compute (FP16/BF16/INT8/FP32/FP64), memory, power, ECC | `gpu_validation_kpis.jsonl` |
| `tests/test_memory_validation.py` | TC-MEM-* | System DRAM bandwidth, latency, ECC | `memory_validation_kpis.jsonl` |
| `tests/test_interconnect_validation.py` | TC-IC-* | PCIe H2D/D2H, NVLink / xGMI, RDMA | `interconnect_kpis.jsonl` |
| `tests/test_ai_pipeline_kpis.py` | TC-PIPE-* | End-to-end AI pipeline throughput and latency | `ai_pipeline_kpis.jsonl` |
| `tests/test_vllm_prefill_validation.py` | TC-VLLM-* | vLLM TTFT, ITL, TPS, KV cache efficiency | `vllm_prefill_kpis.jsonl` |
| `tests/test_agentic_ai_validation.py` | TC-AGENT-01 – TC-AGENT-08 | Agentic workload: tool dispatch, CPU+GPU sync, multi-turn | `agentic_ai_kpis.jsonl` |
| `tests/test_node_validation.py` | TC-NODE-* | System bring-up, RDMA, service health | `node_validation_kpis.jsonl` |
| `tests/test_bottleneck_analysis.py` | TC-BOTTLENECK-* | Resource bottleneck detection | — |
| `tests/test_performance_suite.py` | TC-PERF-* | Performance catalog execution | `performance_suite_kpis.jsonl` |
| `tests/test_mlperf_suite.py` | TC-MLPERF-* | MLPerf Inference suite orchestration | `mlperf_suite_kpis.jsonl` |
| `tests/test_system_health_and_reliability.py` | TC-SHR-* | System health, thermals, error logs | — |
| `tests/test_system_config_report.py` | TC-CFG-* | Configuration report generation | — |

---

## Hardware Env Vars

All GPU and CPU specification detection is automatic via `pynvml` / `/proc` /
`sysfs`. These env vars let you **override** detected values or tune thresholds
when running against hardware without pynvml support or custom OEM configurations.

### GPU overrides

```bash
# GPU memory bandwidth in TB/s (auto-detected; override only if needed)
export GPU_MEMORY_BANDWIDTH_TBS=5.3     # MI300X example

# GPU HBM/VRAM capacity in GB (auto-detected from pynvml)
export GPU_MEMORY_CAPACITY_GB=192       # MI300X example

# GPU FP16 theoretical peak in TFLOPS
export GPU_FP16_TFLOPS=1307             # MI300X example

# GPU TDP in Watts (auto-detected from pynvml power limit)
export GPU_TDP_WATTS=750                # MI300X example

# GPU maximum operating temperature in °C (auto-detected from pynvml shutdown limit × 0.90)
export GPU_MAX_TEMP_C=90                # MI300X example

# Minimum compute capability major version for AI-grade assertion (default: 7)
export GPU_MIN_CC_MAJOR=7
```

### CPU overrides

```bash
# SIMD FP32 FLOPS per cycle per core
# 64 = Intel AVX-512 (FMA)  |  16 = AMD AVX2 (FMA)  |  32 = AVX-512 without FMA
export CPU_SIMD_FP32_FLOPS_PER_CYCLE=16    # AMD EPYC example
```

---

## Running the Suite

### Full staged pipeline

```bash
./run_all_validation_tests.sh
# Options:
./run_all_validation_tests.sh --performance-catalog-only
./run_all_validation_tests.sh --performance-case-ids 001,002,018
./run_all_validation_tests.sh --stop-on-failure
./run_all_validation_tests.sh -- -s --log-cli-level=INFO
```

### Workload acceptance only

```bash
./run_workload_acceptance.sh
./run_workload_acceptance.sh --min-duration-sec 600
./run_workload_acceptance.sh --kv-context-length 16384 --kv-batch-size 8
./run_workload_acceptance.sh --skip-large-models
```

### Individual test modules

```bash
pytest -vv tests/test_cpu_validation.py        # CPU KPIs
pytest -vv tests/test_gpu_validation.py        # GPU KPIs
pytest -vv tests/test_memory_validation.py     # Memory KPIs
pytest -vv tests/test_interconnect_validation.py
pytest -vv tests/test_agentic_ai_validation.py
pytest -m "not slow" -vv                       # skip long-running tests
pytest -m bottleneck -vv tests/test_bottleneck_analysis.py
pytest -m node_validation -vv tests/test_node_validation.py
```

### Persist logs

```bash
pytest -vv --log-file=validation.log --log-file-level=INFO 2>&1 | tee validation_full.log
```

---

## Agentic AI Validation

`tests/test_agentic_ai_validation.py` measures CPU+GPU co-operation for agentic
workloads (LLM-driven tool use, multi-turn reasoning, parallel tool execution).

```bash
# Enable live GPU token generation (requires PyTorch + CUDA/ROCm)
export AGENTIC_EXECUTE=1
export AGENTIC_MODEL_ID=facebook/opt-125m    # or any HuggingFace model
export AGENTIC_NUM_TOOLS=8
export AGENTIC_NUM_TURNS=5
pytest -vv tests/test_agentic_ai_validation.py
```

KPIs written to `artifacts/agentic_ai_kpis.jsonl`. Full documentation:
[`AGENTIC_AI_VALIDATION.md`](AGENTIC_AI_VALIDATION.md).

---

## MLPerf Workflow

```bash
# Download models and datasets
scripts/download_mlperf_models.sh --model-dir /models/mlperf
scripts/download_mlperf_datasets.sh --data-dir /data/mlperf

# Run on N GPUs (vendor-agnostic; script detects GPU count)
scripts/run_mlperf_h100x4.sh \
	--data-dir /data/mlperf --model-dir /models/mlperf \
	--num-gpus 4 --tp-size 4

# Or via pytest
# Optional: force catalog-only mode
# export ML_EXECUTE_CASES=0
export ML_CASE_IDS=001,006,014
export ML_CMD_TIMEOUT_SEC=3600
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite
```

---

## Artifacts & Reports

All structured KPI data is written as newline-delimited JSON (JSONL) to the
`artifacts/` directory. Each record has `{timestamp, test, metrics}` fields
suitable for import into any analytics pipeline for competitive comparison.

| File | Contents |
|---|---|
| `TESTCASE_EXECUTION_FINAL_REPORT.md` | Unified pytest pass/fail report |
| `artifacts/cpu_validation_kpis.jsonl` | All CPU KPI measurements |
| `artifacts/gpu_validation_kpis.jsonl` | All GPU KPI measurements |
| `artifacts/memory_validation_kpis.jsonl` | DRAM bandwidth and latency |
| `artifacts/interconnect_kpis.jsonl` | PCIe, NVLink / xGMI, RDMA bandwidth |
| `artifacts/vllm_prefill_kpis.jsonl` | TTFT, ITL, TPS |
| `artifacts/agentic_ai_kpis.jsonl` | Tool dispatch, CPU+GPU sync latency |
| `artifacts/ai_pipeline_kpis.jsonl` | End-to-end pipeline throughput |
| `artifacts/node_validation_kpis.jsonl` | Node bring-up and health checks |
| `artifacts/performance_suite_kpis.jsonl` | Catalog performance cases |
| `artifacts/mlperf_suite_kpis.jsonl` | MLPerf Inference results |

Override the node KPI output path:

```bash
export NODE_VALIDATION_KPI_FILE=artifacts/node_validation_kpis.jsonl
```

---

## Environment Reference

| Variable | Default | Purpose |
|---|---|---|
| `GPU_TDP_WATTS` | auto | GPU TDP override (Watts) |
| `GPU_MAX_TEMP_C` | auto | GPU max operating temp override (°C) |
| `GPU_MEMORY_BANDWIDTH_TBS` | auto | GPU memory bandwidth override (TB/s) |
| `GPU_MEMORY_CAPACITY_GB` | auto | GPU VRAM capacity override (GB) |
| `GPU_FP16_TFLOPS` | auto | GPU FP16 peak override (TFLOPS) |
| `GPU_MIN_CC_MAJOR` | `7` | Minimum CUDA compute capability major version |
| `CPU_SIMD_FP32_FLOPS_PER_CYCLE` | `64` (AVX-512) / `16` (AVX2) | CPU SIMD width override |
| `HF_ENABLE_LARGE_MODELS` | `1` | Enable >7B parameter model tests |
| `HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD` | `1` | Allow HuggingFace Hub downloads |
| `NODE_WORKLOAD_AUTO_INSTALL` | `1` | Auto-install missing Python packages |
| `NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC` | `900` | pip install timeout |
| `HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC` | `300` | Min soak duration for latency tests |
| `PERF_EXECUTE_CASES` | `1` | Enable performance case execution (`0` = catalog-only mode) |
| `PERF_CASE_IDS` | all | Comma-separated performance case IDs |
| `PERF_CMD_TIMEOUT_SEC` | `1800` | Per-case timeout |
| `ML_EXECUTE_CASES` | `1` | Enable MLPerf case execution (`0` = catalog-only mode) |
| `ML_CASE_IDS` | all | Comma-separated MLPerf case IDs |
| `ML_CMD_TIMEOUT_SEC` | `3600` | Per-MLPerf-case timeout |
| `PIPELINE_EXECUTE` | `1` | Enable AI pipeline execution-gated tests |
| `AGENTIC_EXECUTE` | `1` | Enable live GPU inference in agentic tests |
| `AGENTIC_MODEL_ID` | `facebook/opt-125m` | HuggingFace model for agentic tests |

> Linux is the supported execution target for this validation suite.
