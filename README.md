# AIHeadNode Validation Suite

A **vendor-neutral** Linux-first validation and competitive benchmarking suite for AI head nodes built from any combination of server-class CPUs and AI accelerators. The suite covers CPU, GPU, memory, interconnect, agentic AI workloads, and system health.

**Tested hardware:**

| CPU stack | GPU / Accelerator | Interconnect |
|---|---|---|
| Intel Xeon Platinum (SPR / EMR) | NVIDIA H100 SXM5 / PCIe | NVLink 4.0 |
| Intel Xeon Platinum (SPR / EMR) | NVIDIA H200 SXM5 | NVLink 4.0 |
| AMD EPYC Genoa / Bergamo | AMD Instinct MI300X | xGMI 3.0 |
| AMD EPYC Genoa | NVIDIA A100 SXM4 / PCIe | NVLink 3.0 |
| Any x86-64 server | Any CUDA ≥ 7.0 or ROCm device | PCIe Gen 5 |

All thresholds are expressed as **percentages of detected theoretical peak** (not absolute vendor-specific numbers), so pass/fail comparisons are valid across platforms.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running Tests](#running-tests)
3. [Test Modules](#test-modules)
4. [Competitive Analysis KPIs](#competitive-analysis-kpis)
5. [Environment Variables](#environment-variables)
6. [Specialized Workflows](#specialized-workflows)
7. [Artifacts & Reports](#artifacts--reports)

---

## Quick Start

### Setup (Linux)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch (CUDA 12.x; skip if already present)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or for AMD ROCm:
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# Install system packages (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# Or for RHEL/Rocky/CentOS:
sudo dnf install -y python3-devel gcc gcc-c++ make
```

### Run All Tests (Recommended)

```bash
python scripts/run_all_workloads.py
```

Outputs:
- Per-domain logs: `artifacts/reports/logs/*.log`
- Per-domain JUnit XML: `artifacts/reports/junit/*.xml`
- Consolidated JSON status: `artifacts/workload_status_report.json`
- Consolidated Markdown report: `artifacts/workload_status_report.md`

---

## Running Tests

### 1. Run All Domains

```bash
python scripts/run_all_workloads.py
```

### 2. Run Specific Domains

```bash
# Single domain
python scripts/run_all_workloads.py --only gpu

# Multiple domains
python scripts/run_all_workloads.py --only gpu,interconnect,node_validation
```

### 3. List Available Domains

```bash
python scripts/run_all_workloads.py --list-domains
```

Output shows all 13 domains: `bottleneck`, `cpu`, `memory`, `gpu`, `interconnect`, `node_validation`, `agentic_ai`, `ai_pipeline`, `system_health`, `system_config`, `vllm_prefill`, `performance_suite`, `mlperf_suite`.

### 4. Useful Runner Options

```bash
# Stop on first failure
python scripts/run_all_workloads.py --stop-on-failure

# Pass extra pytest arguments
python scripts/run_all_workloads.py --pytest-args "-s --maxfail=1"

# Combine options
python scripts/run_all_workloads.py --only bottleneck,cpu --pytest-args "-v"
```

### 5. Direct Pytest (Power Users)

```bash
# Individual test module
pytest -vv tests/test_cpu_validation.py
pytest -vv tests/test_gpu_validation.py

# By marker
pytest -m bottleneck -vv tests/test_bottleneck_analysis.py
pytest -m node_validation -vv tests/test_node_validation.py

# By test name
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite

# Skip slow tests
pytest -m "not slow" -vv

# With logging
pytest -vv --log-file=validation.log --log-file-level=INFO 2>&1 | tee validation_full.log
```

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

## Competitive Analysis KPIs

The following KPIs are captured and written to structured JSONL artifact files. They are percentage-of-peak or latency-based so values can be compared fairly across Intel + NVIDIA and AMD + AMD configurations.

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

> **Reference peaks** — detected at runtime from pynvml: H100 SXM5 FP16 = 1979 TFLOPS, MI300X FP16 = 1307 TFLOPS, A100 FP16 = 312 TFLOPS.

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

## Environment Variables

Default environment values are applied automatically in two places:
- `tests/conftest.py` — for direct pytest runs
- `scripts/run_all_workloads.py` — for orchestrated multi-domain runs

### Execution Flags (Default: enabled)

```bash
ML_EXECUTE_CASES=1              # Enable MLPerf suite execution
PERF_EXECUTE_CASES=1            # Enable performance suite execution
PIPELINE_EXECUTE=1              # Enable AI pipeline execution
AGENTIC_EXECUTE=1               # Enable agentic AI workload execution
HEALTH_EXECUTE=1                # Enable system health checks
```

### KPI Artifact Paths

```bash
NODE_VALIDATION_KPI_FILE=artifacts/node_validation_kpis.jsonl
PIPELINE_KPI_FILE=artifacts/ai_pipeline_kpis.jsonl
AGENTIC_KPI_FILE=artifacts/agentic_ai_kpis.jsonl
HEALTH_KPI_FILE=artifacts/system_health_kpis.jsonl
PERF_SUITE_KPI_FILE=artifacts/performance_suite_kpis.jsonl
ML_SUITE_KPI_FILE=artifacts/mlperf_suite_kpis.jsonl
VLLM_PREFILL_KPI_FILE=artifacts/vllm_prefill_kpis.jsonl
```

### GPU / CUDA Defaults

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### Model & Workload Defaults

```bash
HF_ENABLE_LARGE_MODELS=1                               # Enable large model downloads
HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1                 # Allow remote model fetches
HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=300             # Enterprise latency baseline
HF_INFERENCE_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_TRAINING_MODEL=bert-base-uncased
NODE_WORKLOAD_AUTO_INSTALL=1                           # Auto-install node workloads
NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=1800            # Installation timeout (30 min)
```

### Common Overrides

```bash
# Disable large model support
export HF_ENABLE_LARGE_MODELS=0
export HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0

# Shorten workload installation timeout (for rapid testing)
export NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=600

# Skip workload installation (if already cached)
export NODE_WORKLOAD_AUTO_INSTALL=0
```

### GPU Hardware Overrides

Use these only if auto-detection fails or for custom OEM configurations:

```bash
# GPU memory bandwidth in TB/s
export GPU_MEMORY_BANDWIDTH_TBS=5.3     # MI300X example

# GPU HBM/VRAM capacity in GB
export GPU_MEMORY_CAPACITY_GB=192       # MI300X example

# GPU FP16 theoretical peak in TFLOPS
export GPU_FP16_TFLOPS=1307             # MI300X example

# GPU TDP in Watts
export GPU_TDP_WATTS=750                # MI300X example

# GPU max safe operating temperature (°C)
export GPU_MAX_TEMP_C=90                # MI300X example

# Minimum compute capability major version (default: 7)
export GPU_MIN_CC_MAJOR=7
```

### CPU Hardware Overrides

```bash
# SIMD FP32 FLOPS per cycle per core
# 64 = Intel AVX-512 (FMA)  |  16 = AMD AVX2 (FMA)  |  32 = AVX-512 without FMA
export CPU_SIMD_FP32_FLOPS_PER_CYCLE=16    # AMD EPYC example
```

---

## Specialized Workflows

### Performance Suite

Run the performance benchmark catalog:

```bash
# Run default case set
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite

# Or with orchestrator
python scripts/run_all_workloads.py --only performance_suite

# Custom case IDs and timeout
export PERF_CASE_IDS=001,002,018
export PERF_CMD_TIMEOUT_SEC=1800
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite

# Catalog-only mode (no case execution)
export PERF_EXECUTE_CASES=0
pytest -vv tests/test_performance_suite.py
```

### MLPerf Inference Suite (4x H100)

Download models and datasets:

```bash
chmod +x scripts/download_mlperf_models.sh
scripts/download_mlperf_models.sh --model-dir /models/mlperf

chmod +x scripts/download_mlperf_datasets.sh
scripts/download_mlperf_datasets.sh --data-dir /data/mlperf
```

Run via orchestration script:

```bash
chmod +x scripts/run_mlperf_h100x4.sh
scripts/run_mlperf_h100x4.sh \
  --data-dir /data/mlperf \
  --model-dir /models/mlperf \
  --num-gpus 4 \
  --tp-size 4
```

Or run pytest suite directly:

```bash
# Custom case IDs and timeout
export ML_CASE_IDS=001,006,014
export ML_CMD_TIMEOUT_SEC=3600
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite

# Catalog-only mode
export ML_EXECUTE_CASES=0
pytest -vv tests/test_mlperf_suite.py

# Or use orchestrator
python scripts/run_all_workloads.py --only mlperf_suite
```

### Agentic AI Validation

Measure CPU+GPU co-operation for LLM-driven tool use and multi-turn reasoning:

```bash
# Enable live GPU token generation (requires PyTorch + CUDA/ROCm)
export AGENTIC_EXECUTE=1
export AGENTIC_MODEL_ID=facebook/opt-125m    # or any HuggingFace model
export AGENTIC_NUM_TOOLS=8
export AGENTIC_NUM_TURNS=5

# Run test module
pytest -vv tests/test_agentic_ai_validation.py

# Or use orchestrator
python scripts/run_all_workloads.py --only agentic_ai
```

See [AGENTIC_AI_VALIDATION.md](AGENTIC_AI_VALIDATION.md) for detailed documentation.

---

## Artifacts & Reports

### Output Locations

When you run `python scripts/run_all_workloads.py`, outputs are written to:

- **Per-domain logs**: `artifacts/reports/logs/{domain}.log`
- **Per-domain JUnit XML**: `artifacts/reports/junit/{domain}.xml`
- **Consolidated JSON status**: `artifacts/workload_status_report.json`
- **Consolidated Markdown report**: `artifacts/workload_status_report.md`
- **Domain-specific KPI files**: `artifacts/{domain}_kpis.jsonl`

### Exit Code

- **0** — All domains passed
- **Non-zero** — One or more domains failed (see Markdown report for details)

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
