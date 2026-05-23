# AIHeadNode Validation Suite

A **vendor-neutral** Linux-first validation and competitive benchmarking suite for AI head nodes built from any combination of server-class CPUs and AI accelerators. The suite covers CPU, GPU, memory, interconnect, agentic AI workloads, and system health.

**Tested Hardware (DMR):**

| CPU Stack | GPU / Accelerator | Interconnect | Configuration |
|---|---|---|---|
| Intel Xeon Platinum (SPR) 160-core | 4x NVIDIA H100 SXM5 (80 GB HBM3 each) | NVLink 4.0 full-mesh | DMR validation platform |

**H100 SXM5 Specifications (Single GPU):**
- **FP16/BF16 Peak**: 1,979 TFLOPS
- **FP32 Peak**: 989.5 TFLOPS  
- **FP64 Peak**: 494.75 TFLOPS
- **INT8 Peak**: 3,959 TOPS
- **HBM3 Bandwidth**: 3.35 TB/s = **3,350 GB/s**
- **HBM3 Capacity**: 80 GB
- **NVLink 4.0 Bandwidth** (per link pair): 900 GB/s bidirectional
- **PCIe 5.0 ×16 Bandwidth** (H2D/D2H): 63 GB/s unidirectional
- **TDP**: 700 W
- **Compute Capability**: 9.0 (Ada)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pre-Downloading Models & Datasets](#pre-downloading-models--datasets)
3. [Running Tests](#running-tests)
4. [Test Modules](#test-modules)
5. [Competitive Analysis KPIs](#competitive-analysis-kpis)
6. [Environment Variables](#environment-variables)
7. [Specialized Workflows](#specialized-workflows)
8. [Artifacts & Reports](#artifacts--reports)

---

## Quick Start

### Setup (Linux)

```bash
# From the repo root on the Linux SUT

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

### Linux SUT Preflight (Required for full workload execution)

Several tests expect vendor tools and workload binaries to be available on the SUT.

```bash
# GPU visibility
nvidia-smi

# Expected defaults used by tests/conftest.py and scripts/run_all_workloads.py
test -x /root/workloads/nvbandwidth
test -d /root/workloads/nccl-tests/build
```

If your binaries are in different locations, override them before running tests:

```bash
export NVBANDWIDTH_BIN=/opt/nvbandwidth/nvbandwidth
export PERF_NVBANDWIDTH_BIN=/opt/nvbandwidth/nvbandwidth
export PERF_NCCL_TESTS_DIR=/opt/nccl-tests/build
```

> **GPU count is auto-detected.** `CUDA_VISIBLE_DEVICES`, `YOLO_GPU_COUNT`,
> `DIFFUSERS_GPU_COUNT`, `PERF_NUM_GPUS`, `ML_NUM_GPUS`, and `ML_TENSOR_PARALLEL`
> are all derived at runtime from `nvidia-smi` / `CUDA_VISIBLE_DEVICES`.
> No hardcoded 4-GPU assumption remains in any test file.

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

## Pre-Downloading Models & Datasets

All HuggingFace models and public datasets required by every test suite can be
pre-downloaded to a dedicated storage volume (recommended: at least 1 TB) before
running any tests. The download script uses `HF_TOKEN` from the environment so
**the token is never written into any source file**.

### Storage Layout

```
/mnt/datasets/          ← DATASETS_ROOT (override with $DATASETS_ROOT)
  huggingface/
    hub/                ← HF_HOME / HUGGINGFACE_HUB_CACHE
      models--meta-llama--Meta-Llama-3.1-8B-Instruct/
      models--Qwen--Qwen2.5-72B-Instruct/
      ...               (all models listed below)
  yolo/
    yolov8x.pt
    yolov8n.pt
  sharegpt/
    ShareGPT_V3_unfiltered_cleaned_split.json
  torch/               ← TORCH_HOME
  test_env.sh          ← generated; source before pytest
```

### Running the Download Script

```bash
export HF_TOKEN=<your_huggingface_token>

# Optional: change storage root (default /mnt/datasets)
export DATASETS_ROOT=/mnt/datasets

bash scripts/download_models_datasets.sh
```

Models downloaded (all individually under 400 GB):

| Model | Approx. Size |
|---|---|
| openai-community/gpt2 | 500 MB |
| bert-base-uncased | 440 MB |
| stabilityai/sd-turbo | 3 GB |
| openai/whisper-large-v3 | 3 GB |
| google/gemma-2-2b | 5 GB |
| EleutherAI/gpt-j-6b | 24 GB |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 15 GB |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 16 GB |
| meta-llama/Meta-Llama-3-8B | 16 GB |
| meta-llama/Meta-Llama-3.1-70B-Instruct | 140 GB |
| meta-llama/Llama-3.3-70B-Instruct | 140 GB |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 140 GB |
| Qwen/Qwen2.5-72B-Instruct | 144 GB |
| Qwen/Qwen-72B | 144 GB |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | 280 GB |
| YOLO weights (yolov8x.pt + yolov8n.pt) | 280 MB |
| ShareGPT dataset | 300 MB |
| **deepseek-ai/DeepSeek-R1 (671B)** | **SKIPPED — exceeds 400 GB** |

### Source the Generated Environment File

After the script finishes it writes `/mnt/datasets/test_env.sh` containing all
required environment variable exports. Source it before every pytest run:

```bash
source /mnt/datasets/test_env.sh
pytest -vv tests/
```

Alternatively, set `DATASETS_ROOT` manually and `tests/conftest.py` will derive
all paths automatically:

```bash
export DATASETS_ROOT=/mnt/datasets
export HF_TOKEN=<your_token>
pytest -vv tests/
```

---

## Running Tests

### 1. Run All Domains

```bash
python scripts/run_all_workloads.py
```

### 1a. Linux SUT smoke before full run

```bash
python scripts/run_all_workloads.py --only cpu,gpu,interconnect,node_validation
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

Output shows all 14 domains: `bottleneck`, `cpu`, `memory`, `gpu`, `interconnect`, `node_validation`, `cpu_headnode`, `agentic_ai`, `ai_pipeline`, `system_health`, `system_config`, `vllm_prefill`, `performance_suite`, `mlperf_suite`.

### 4. Useful Runner Options

```bash
# Stop on first failure
python scripts/run_all_workloads.py --stop-on-failure

# Pass extra pytest arguments
python scripts/run_all_workloads.py --pytest-args "-s --maxfail=1"

# Combine options
python scripts/run_all_workloads.py --only bottleneck,cpu --pytest-args "-v"

# Exclude specific domains (applied after --only; "everything except X")
python scripts/run_all_workloads.py --skip mlperf_suite,performance_suite

# --only and --skip compose: run all GPU-side domains except the slow ones
python scripts/run_all_workloads.py \
    --only gpu,interconnect,node_validation,ai_pipeline,vllm_prefill \
    --skip vllm_prefill

# Stress / reliability loop: run the same selection N times.
# Each iteration writes artifacts/loop/run_<NNNN>/{status.json,status.md,junit/,logs/}
# plus a roll-up at artifacts/loop/aggregate.{json,md}.
python scripts/run_all_workloads.py --only cpu_headnode,node_validation --repeat 50
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

## Test Modules

| File | TC range | Focus area | KPI artifact |
|---|---|---|---|
| `tests/test_cpu_validation.py` | TC-CPU-01 – TC-CPU-25 | CPU topology, SIMD compute, NUMA, industrial KPIs | `cpu_validation_kpis.jsonl` |
| `tests/test_cpu_headnode.py` | TC-CPUH-01 – TC-CPUH-10 | CPU↔GPU integration: NUMA pinning, pinned memory, tokenizer/dataloader scaling, IRQ affinity, governor, sync latency, NVMe model-cache I/O, asyncio scaling | `cpu_headnode_kpis.jsonl` |
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
| `tests/test_functional_test_plan_mapping.py` | AI_HOSTCPU_FUNC_TEST_ID_001–054 | Functional test-plan ID catalog mapping test selectors to plan IDs | — |
| `tests/test_agentic_test_plan_mapping.py` | AI_HOSTCPU_AGENTIC_TEST_ID_### | Agentic test-plan ID catalog for tests not covered by FUNC/PERF/ML catalogs | — |

---

## Competitive Analysis KPIs

The following KPIs are captured and written to structured JSONL artifact files. Values include both the H100 theoretical baseline and the corresponding absolute threshold.

### GPU Compute KPIs

| KPI | Theoretical (H100) | Threshold | Actual Threshold | Artifact |
|---|---|---|---|---|
| FP16 matrix TFLOPS | 1,979 TFLOPS | ≥ 75 % | ≥ 1,484 TFLOPS | `gpu_validation_kpis.jsonl` |
| BF16 matrix TFLOPS | 1,979 TFLOPS | ≥ 75 % | ≥ 1,484 TFLOPS | `gpu_validation_kpis.jsonl` |
| INT8 matrix TOPS | 3,959 TOPS | ≥ 70 % | ≥ 2,771 TOPS | `gpu_validation_kpis.jsonl` |
| FP32 TFLOPS | 989.5 TFLOPS | ≥ 70 % | ≥ 692.6 TFLOPS | `gpu_validation_kpis.jsonl` |
| FP64 TFLOPS | 494.75 TFLOPS | ≥ 70 % | ≥ 346.3 TFLOPS | `gpu_validation_kpis.jsonl` |
| Multi-GPU scaling efficiency (4x) | 7,916 TFLOPS aggregate | ≥ 90 % linear | ≥ 7,124 TFLOPS | `gpu_validation_kpis.jsonl` |
| Energy efficiency | ~2.8 TFLOPS/W (FP16) | logged (higher is better) | — | `gpu_validation_kpis.jsonl` |

### GPU Memory KPIs

| KPI | Theoretical (H100) | Threshold | Actual Threshold | Artifact |
|---|---|---|---|---|
| HBM3 capacity per GPU | 80 GB | ≥ 98 % of spec | ≥ 78.4 GB | `gpu_validation_kpis.jsonl` |
| HBM3 memory bandwidth per GPU | 3,350 GB/s (3.35 TB/s) | ≥ 80 % | ≥ 2,680 GB/s (2.68 TB/s) | `gpu_validation_kpis.jsonl` |
| Aggregate 4x GPU bandwidth | 13,400 GB/s (13.4 TB/s) | ≥ 80 % | ≥ 10,720 GB/s (10.72 TB/s) | `gpu_validation_kpis.jsonl` |

### CPU Compute KPIs

| KPI | Theoretical (SPR 160-core @ 2.7 GHz) | Threshold | Actual Threshold | Artifact |
|---|---|---|---|---|
| Single-core SIMD FP32 (AVX-512 FMA) | 172.8 GFLOPS/core | ≥ 70 % | ≥ 121 GFLOPS | `cpu_validation_kpis.jsonl` |
| All-core SIMD FP32 aggregate (160 cores) | 27.6 TFLOPS | ≥ 70 % | ≥ 19.3 TFLOPS | `cpu_validation_kpis.jsonl` |
| BLAS DGEMM throughput (all-core) | ~20 TFLOPS (BLAS-optimized) | ≥ 70 % | ≥ 14 TFLOPS | `cpu_validation_kpis.jsonl` |
| CPU utilization under load | 100 % of available threads | ≥ 90 % | ≥ 90 % | `cpu_validation_kpis.jsonl` |
| CPU frequency scaling (Turbo boost) | 3.8 GHz (base 2.7 GHz) | ≥ 90 % of base | ≥ 2.43 GHz | `cpu_validation_kpis.jsonl` |
| Memory latency (random access, L3 miss) | ~40-80 ns | ≤ 150 ns | ≤ 150 ns | `cpu_validation_kpis.jsonl` |
| NUMA node balance (2 sockets) | N/A | ≤ 10 % imbalance | ≤ 10 % | `cpu_validation_kpis.jsonl` |

### Interconnect KPIs

| KPI | Theoretical | Threshold | Actual Threshold | Artifact |
|---|---|---|---|---|
| CPU → GPU PCIe 5.0 x16 H2D bandwidth (per GPU) | 63 GB/s | ≥ 80 % | ≥ 50.4 GB/s | `interconnect_kpis.jsonl` |
| GPU → CPU PCIe 5.0 x16 D2H bandwidth (per GPU) | 63 GB/s | ≥ 80 % | ≥ 50.4 GB/s | `interconnect_kpis.jsonl` |
| GPU-to-GPU NVLink 4.0 bandwidth (per link pair) | 900 GB/s bidirectional | ≥ 80 % | ≥ 720 GB/s | `interconnect_kpis.jsonl` |
| Aggregate 4x GPU all-to-all NVLink | 5,400 GB/s (6 pairs x 900 GB/s) | ≥ 75 % | ≥ 4,050 GB/s | `interconnect_kpis.jsonl` |

### LLM Serving / Inference KPIs

| KPI | Theoretical (4x H100) | Enterprise Target | Actual Threshold | Artifact |
|---|---|---|---|---|
| Time to First Token (TTFT) | N/A | ≤ 500 ms (prefill latency) | ≤ 500 ms | `vllm_prefill_kpis.jsonl` |
| Inter-Token Latency (ITL) | N/A | ≤ 30 ms / token (decode rate) | ≤ 30 ms | `vllm_prefill_kpis.jsonl` |
| Tokens per second (aggregate TPS) | ~1,500-2,000 TPS (vLLM batch mode) | ≥ 1,000 tok/s | ≥ 1,000 tok/s | `vllm_prefill_kpis.jsonl` |
| KV cache prefill efficiency | ~13.4 TB/s (HBM bandwidth) | ≥ 75 % = 10.05 TB/s | ≥ 10.05 TB/s | `vllm_prefill_kpis.jsonl` |

### Agentic AI KPIs

| KPI | Theoretical (H100) | Target | Notes | Artifact |
|---|---|---|---|---|
| Tool dispatch overhead | ~1-2 ms (kernel launch) | ≤ 5 ms | PyTorch overhead + agent scheduling | `agentic_ai_kpis.jsonl` |
| CPU → GPU synchronization latency | ~100-300 us (typical) | ≤ 500 us | Host-to-device fence/sync overhead | `agentic_ai_kpis.jsonl` |
| Tool parallelization speedup (8 tools) | ≥ 6x (with NVLink) | ≥ 5.6x (0.7 x 8 tools) | Linear = 8x; 70% efficiency target | `agentic_ai_kpis.jsonl` |
| Agent turn P99 latency (full round-trip) | ~500-1500 ms | logged | Prefill + token generation + tool call | `agentic_ai_kpis.jsonl` |
| Agent state serialization throughput | ~500-1000 MB/s (PCIe/NVLink) | ≥ 100 MB/s | Actor/state checkpointing speed | `agentic_ai_kpis.jsonl` |

### Understanding Theoretical vs Measured / Threshold Numbers

**Theoretical Peak**: The maximum achievable performance under ideal conditions (single benchmark, no contention, full compute utilization). Used as the reference baseline.
- Examples:
  - H100 FP16: 1,979 TFLOPS (both tensors fully loaded, no pipeline bubbles)
  - HBM bandwidth: 3,350 GB/s (sustained, no bank conflicts)
  - PCIe 5.0 x16: 63 GB/s (saturated link, optimal packet sizes)

**Measured / Threshold**: The actual performance target or typical run-time observation. Usually expressed as a percentage of theoretical to account for:
- Kernel launch overhead
- Pipeline bubbles and synchronization overhead
- Memory bank conflicts (for bandwidth)
- Contention from other workloads

**Examples:**
- **75% of H100 FP16 peak** = 75% x 1,979 TFLOPS = **1,484 TFLOPS minimum threshold** (test passes if measured ≥ 1,484)
- **80% of HBM bandwidth** = 80% x 3,350 GB/s = **2,680 GB/s minimum threshold** (account for bank conflicts, limited vectorization)
- **80% of PCIe peak** = 80% x 63 GB/s = **50.4 GB/s minimum** (bus contention, protocol overhead)

All thresholds in this document are percentages of theoretical peak, converted to absolute values for this DMR + 4x H100 configuration.

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

### Derived Efficiency KPIs

The suite now auto-derives two cross-suite value metrics when enough source
fields are present in a KPI payload:

- `performance_per_watt`
- `performance_per_dollar`

Formulas:

- `performance_per_watt = performance_rate / power_watts`
- `performance_per_dollar = (performance_rate * 3600) / cost_per_hour_usd`

Where `performance_rate` is inferred from the first available metric in this
priority set:

- `throughput_tokens_per_sec`
- `tokens_per_sec`
- `samples_per_sec`
- `requests_per_sec`
- `throughput_qps`
- `qps`
- `aggregate_fps`
- `throughput_fps`
- `tflops`
- `gflops`

Power source resolution order:

1. Existing KPI payload fields such as `mean_power_watts`, `avg_power_watts`,
  `power_watts`, `gpu_power_watts`, `mean_w`, or `power_w`.
2. Fallback to `GPU_TDP_WATTS * gpu_count` when explicit power metrics are not
  present.

Cost source resolution order:

1. Existing KPI payload fields such as `cost_per_hour_usd`,
  `hourly_cost_usd`, `node_cost_per_hour_usd`, or `total_cost_per_hour_usd`.
2. Environment fallback values in order:
  `KPI_COST_PER_HOUR_USD`, `NODE_COST_PER_HOUR_USD`,
  or `GPU_COST_PER_HOUR_USD * gpu_count`.

Notes:

- If the required denominator is not available (or <= 0), that derived metric
  is omitted instead of writing a zero.
- Existing explicit fields are preserved; derivation only fills missing values.

### Efficiency KPI Environment Example

Use one of the following patterns before running pytest.

Linux/macOS:

```bash
export GPU_TDP_WATTS=700
export NODE_COST_PER_HOUR_USD=18.50

# Optional alternatives (use one pricing model):
# export KPI_COST_PER_HOUR_USD=18.50
# export GPU_COST_PER_HOUR_USD=4.60
```

Windows PowerShell:

```powershell
$env:GPU_TDP_WATTS = "700"
$env:NODE_COST_PER_HOUR_USD = "18.50"

# Optional alternatives (use one pricing model):
# $env:KPI_COST_PER_HOUR_USD = "18.50"
# $env:GPU_COST_PER_HOUR_USD = "4.60"
```

### Storage & Cache Defaults

All paths default to `/mnt/datasets` when that directory exists. Override with:

```bash
DATASETS_ROOT=/mnt/datasets          # root for all models, datasets, weights
HF_HOME=${DATASETS_ROOT}/huggingface
HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
TORCH_HOME=${DATASETS_ROOT}/torch
YOLO_DETECT_MODEL=${DATASETS_ROOT}/yolo/yolov8x.pt   # auto if file exists
SHAREGPT_DATASET_PATH=${DATASETS_ROOT}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
```

When `/mnt/datasets` is absent the suite falls back to `~/.cache/huggingface`
(standard HuggingFace defaults).

### GPU / CUDA Defaults

```bash
# CUDA_VISIBLE_DEVICES is auto-detected from nvidia-smi at startup.
# No value is hardcoded. Override explicitly if needed:
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### Model & Workload Defaults

```bash
HF_ENABLE_LARGE_MODELS=1                               # Enable large model downloads
HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1                 # Allow remote model fetches
HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=300             # Enterprise latency baseline
HF_INFERENCE_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_TRAINING_MODEL=bert-base-uncased
HF_LLAMA_7B_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_LLAMA_70B_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
HF_QWEN_70B_MODEL=Qwen/Qwen2.5-72B-Instruct
HF_DEEPSEEK_R1_QUANT_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
HF_LLAMA_33_70B_MODEL=meta-llama/Llama-3.3-70B-Instruct
HF_MIXTRAL_8X22B_MODEL=mistralai/Mixtral-8x22B-Instruct-v0.1
HF_DEEPSEEK_R1_70B_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
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

### MLPerf Inference Suite

Pre-download all required models and datasets (one-time setup):

```bash
export HF_TOKEN=<your_token>
export DATASETS_ROOT=/mnt/datasets   # 1 TB+ recommended
bash scripts/download_models_datasets.sh
```

Then run via pytest:

```bash
# Source the generated environment file first
source /mnt/datasets/test_env.sh

# Custom case IDs and timeout
export ML_CASE_IDS=001,006,014
export ML_CMD_TIMEOUT_SEC=3600
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite

# Catalog-only mode (no execution)
export ML_EXECUTE_CASES=0
pytest -vv tests/test_mlperf_suite.py

# Or use orchestrator
python scripts/run_all_workloads.py --only mlperf_suite
```

> `ML_NUM_GPUS` and `ML_TENSOR_PARALLEL` are auto-detected from
> `CUDA_VISIBLE_DEVICES` / `nvidia-smi`. Override explicitly if needed:
> `export ML_NUM_GPUS=4 ML_TENSOR_PARALLEL=4`

Legacy per-suite download scripts (`download_mlperf_models.sh`,
`download_mlperf_datasets.sh`, `run_mlperf_h100x4.sh`) are still present for
MLPerf-specific datasets not covered by the general download script.

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

### CPU Head-Node Integration (cpu_headnode)

Validates the CPU side of CPU↔GPU integration in real head-node serving
scenarios (RAG front-ends, vLLM serving, dataloader-fed training). Catches
the "GPUs at 30% util in production while CPU looks fine" class of bugs.

```bash
# Run the whole CPU head-node suite
python scripts/run_all_workloads.py --only cpu_headnode

# Or run a single test directly
pytest -vv tests/test_cpu_headnode.py::test_numa_local_vs_remote_h2d_bandwidth

# Quick smoke (skip slow ones)
pytest -m "cpu_headnode and not slow" -vv tests/test_cpu_headnode.py
```

Tests included (all env-tunable, skip cleanly when prereqs are absent):

| ID | Test | What it asserts |
|---|---|---|
| TC-CPUH-01 | `test_numa_local_vs_remote_h2d_bandwidth` | Per-GPU remote/local H2D ratio ≥ 0.60 (catches CPU worker mispinning) |
| TC-CPUH-02 | `test_pinned_host_memory_allocation_scaling` | Pinned `cudaHostAlloc` succeeds + ≥ 10 GB/s H2D at 256/1024/4096 MB (catches `RLIMIT_MEMLOCK` / hugepage misconfig) |
| TC-CPUH-03 | `test_cpu_memory_bandwidth_under_gpu_load` | Combined CPU triad + multi-GPU H2D ≥ 70% of sum-of-idle (catches DRAM/PCIe contention) |
| TC-CPUH-04 | `test_tokenizer_throughput_scaling` | HF tokenizer ≥ 30k tok/s single-thread + ≥ 50% linear scaling (catches the #1 serving bottleneck) |
| TC-CPUH-05 | `test_dataloader_throughput_scaling` | PyTorch `DataLoader(num_workers>0)` ≥ 1.5x of `num_workers=0` (catches fork/pin_memory regressions) |
| TC-CPUH-06 | `test_cpu_governor_and_cstate_policy` | Every CPU uses `performance` governor (catches default `powersave` killing TTFT) |
| TC-CPUH-07 | `test_irq_affinity_distribution` | mlx5 / nvme / nvidia IRQs span ≥ 2 CPUs (catches IRQs pinned to CPU 0) |
| TC-CPUH-08 | `test_cuda_sync_primitive_latency` | Kernel-launch + sync median ≤ 25 µs per GPU + outliers ≤ 3x (catches degraded PCIe link) |
| TC-CPUH-09 | `test_nvme_model_cache_read_throughput` | Sequential read from HF cache ≥ 1.5 GB/s (catches cache on slow disk / NFS) |
| TC-CPUH-10 | `test_python_async_concurrency_scaling` | asyncio p99 at high concurrency ≤ 8x single-task p99 (catches GIL / event-loop ceiling) |

Key environment overrides (all optional — defaults are sized for an H100
head-node, increase or relax per SKU):

```bash
# Topology / memory
export CPU_HEADNODE_NUMA_H2D_MIN_RATIO=0.60       # remote/local H2D floor
export CPU_HEADNODE_PINNED_SIZES_MB=256,1024,4096 # pinned alloc tiers
export CPU_HEADNODE_PINNED_MIN_H2D_GBS=10.0
export CPU_HEADNODE_TRIAD_MIN_COMBINED_RATIO=0.40 # contention floor (PCIe-saturated)

# Tokenizer / dataloader
export CPU_HEADNODE_TOK_MODEL=openai-community/gpt2
export CPU_HEADNODE_TOK_THREADS=1,4,8
export CPU_HEADNODE_TOK_MIN_SINGLE_TPS=30000
export CPU_HEADNODE_TOK_MIN_SCALING_RATIO=0.20
export CPU_HEADNODE_DL_WORKERS=0,4,8
export CPU_HEADNODE_DL_MIN_SPEEDUP=1.20

# Sysfs / IRQ policy (set WARN_ONLY=1 to downgrade to log warnings)
export CPU_HEADNODE_ALLOWED_GOVERNORS=performance
export CPU_HEADNODE_GOVERNOR_WARN_ONLY=0
export CPU_HEADNODE_IRQ_DRIVERS=mlx5,nvme,nvidia
export CPU_HEADNODE_IRQ_MIN_UNIQUE_CPUS=2
export CPU_HEADNODE_IRQ_WARN_ONLY=0

# Sync / I/O / async
export CPU_HEADNODE_SYNC_MAX_MEDIAN_US=25.0
export CPU_HEADNODE_NVME_CACHE_DIR=$HF_HOME          # falls back to HF_HOME / cwd
export CPU_HEADNODE_NVME_MIN_READ_GBS=1.5
export CPU_HEADNODE_ASYNC_CONCURRENCY=1,16,64
export CPU_HEADNODE_ASYNC_MAX_P99_RATIO=50.0

# KPI output path
export CPU_HEADNODE_KPI_FILE=artifacts/cpu_headnode_kpis.jsonl
```

KPIs written to `artifacts/cpu_headnode_kpis.jsonl`.

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
# One-time pre-download (uses /mnt/datasets by default)
export HF_TOKEN=<your_token>
bash scripts/download_models_datasets.sh

# Source generated env and run
source /mnt/datasets/test_env.sh
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
| `artifacts/cpu_headnode_kpis.jsonl` | CPU↔GPU integration KPIs (NUMA H2D, pinned memory, tokenizer/dataloader scaling, sync latency, NVMe, asyncio) |
| `artifacts/gpu_validation_kpis.jsonl` | All GPU KPI measurements |
| `artifacts/memory_validation_kpis.jsonl` | DRAM bandwidth and latency |
| `artifacts/interconnect_kpis.jsonl` | PCIe, NVLink / xGMI, RDMA bandwidth |
| `artifacts/vllm_prefill_kpis.jsonl` | TTFT, ITL, TPS |
| `artifacts/agentic_ai_kpis.jsonl` | Tool dispatch, CPU+GPU sync latency |
| `artifacts/ai_pipeline_kpis.jsonl` | End-to-end pipeline throughput |
| `artifacts/node_validation_kpis.jsonl` | Node bring-up and health checks |
| `artifacts/performance_suite_kpis.jsonl` | Catalog performance cases |
| `artifacts/mlperf_suite_kpis.jsonl` | MLPerf Inference results |
| `artifacts/bottleneck_kpis.jsonl` | Bottleneck / roofline analysis results |

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
| `DATASETS_ROOT` | `/mnt/datasets` | Root directory for all pre-downloaded models and datasets |
| `HF_HOME` | `$DATASETS_ROOT/huggingface` | HuggingFace home (overrides `~/.cache/huggingface`) |
| `HUGGINGFACE_HUB_CACHE` | `$HF_HOME/hub` | HuggingFace model cache |
| `TORCH_HOME` | `$DATASETS_ROOT/torch` | PyTorch model cache |
| `YOLO_DETECT_MODEL` | `$DATASETS_ROOT/yolo/yolov8x.pt` (if exists) | YOLO weights path |
| `SHAREGPT_DATASET_PATH` | `$DATASETS_ROOT/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json` | ShareGPT dataset |
| `HF_TOKEN` | _(required for gated models)_ | HuggingFace access token |
| `HF_ENABLE_LARGE_MODELS` | `1` | Enable >7B parameter model tests |
| `HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD` | `1` | Allow HuggingFace Hub downloads |
| `NODE_WORKLOAD_AUTO_INSTALL` | `1` | Auto-install missing Python packages |
| `NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC` | `1800` | pip install timeout |
| `HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC` | `300` | Min soak duration for latency tests |
| `PERF_EXECUTE_CASES` | `1` | Enable performance case execution (`0` = catalog-only mode) |
| `PERF_CASE_IDS` | all | Comma-separated performance case IDs |
| `PERF_CMD_TIMEOUT_SEC` | `1800` | Per-case timeout |
| `PERF_NUM_GPUS` | auto-detected | GPU count for NCCL `mpirun -np` |
| `PERF_CUDA_VISIBLE_DEVICES` | auto-detected | CUDA mask for performance benchmarks |
| `PERF_REQUIRE_4GPUS` | `0` | Set to `1` to hard-require 4 GPUs for perf suite |
| `ML_EXECUTE_CASES` | `1` | Enable MLPerf case execution (`0` = catalog-only mode) |
| `ML_CASE_IDS` | all | Comma-separated MLPerf case IDs |
| `ML_CMD_TIMEOUT_SEC` | `3600` | Per-MLPerf-case timeout |
| `ML_NUM_GPUS` | auto-detected | GPU count for MLPerf runs |
| `ML_TENSOR_PARALLEL` | auto-detected | Tensor-parallel degree for vLLM/MLPerf |
| `PIPELINE_EXECUTE` | `1` | Enable AI pipeline execution-gated tests |
| `AGENTIC_EXECUTE` | `1` | Enable live GPU inference in agentic tests |
| `AGENTIC_MODEL_ID` | `facebook/opt-125m` | HuggingFace model for agentic tests |
| `WORKLOADS_ROOT` | `/root/workloads` | Root directory for pre-built binary workloads (nvbandwidth, nccl-tests) |
| `CPU_KPI_FILE` | `artifacts/cpu_validation_kpis.jsonl` | CPU KPI output path |
| `CPU_HEADNODE_KPI_FILE` | `artifacts/cpu_headnode_kpis.jsonl` | CPU head-node integration KPI output path |
| `CPU_HEADNODE_NUMA_H2D_MIN_RATIO` | `0.60` | Per-GPU remote-NUMA / local-NUMA H2D bandwidth floor |
| `CPU_HEADNODE_PINNED_SIZES_MB` | `256,1024,4096` | Pinned host-memory allocation sizes (MB) to validate |
| `CPU_HEADNODE_PINNED_MIN_H2D_GBS` | `10.0` | Minimum H2D BW required from pinned buffers (GB/s) |
| `CPU_HEADNODE_TRIAD_MIN_COMBINED_RATIO` | `0.40` | Min (loaded CPU triad + loaded H2D) / (idle sum); low because idle H2D is sequential per-GPU while loaded is concurrent |
| `CPU_HEADNODE_TOK_MODEL` | `openai-community/gpt2` | HuggingFace tokenizer used for scaling test |
| `CPU_HEADNODE_TOK_MIN_SINGLE_TPS` | `30000` | Minimum single-thread tokenizer throughput (tokens/sec) |
| `CPU_HEADNODE_TOK_MIN_SCALING_RATIO` | `0.20` | Min tokenizer thread-scaling efficiency vs linear (per-prompt encode is GIL-bound) |
| `CPU_HEADNODE_DL_MIN_SPEEDUP` | `1.20` | Min `DataLoader(num_workers>0)` speedup over `num_workers=0` (synthetic dataset; raise for I/O-heavy workloads) |
| `CPU_HEADNODE_ALLOWED_GOVERNORS` | `performance` | Comma-separated allowed cpufreq governors |
| `CPU_HEADNODE_GOVERNOR_WARN_ONLY` | `0` | Set `1` to downgrade governor mismatch to a log warning |
| `CPU_HEADNODE_IRQ_DRIVERS` | `mlx5,nvme,nvidia` | Drivers whose IRQ affinity is validated |
| `CPU_HEADNODE_IRQ_MIN_UNIQUE_CPUS` | `2` | Minimum distinct CPUs in IRQ effective affinity per driver |
| `CPU_HEADNODE_SYNC_MAX_MEDIAN_US` | `25.0` | Max acceptable per-GPU CUDA kernel-launch + sync median latency (µs) |
| `CPU_HEADNODE_NVME_MIN_READ_GBS` | `1.5` | Minimum sequential read throughput from model-cache directory (GB/s) |
| `CPU_HEADNODE_ASYNC_MAX_P99_RATIO` | `50.0` | Max ratio of high-concurrency p99 vs single-task p99 (CPU-bound tasks queue on the loop) |
| `GPU_KPI_FILE` | `artifacts/gpu_validation_kpis.jsonl` | GPU KPI output path |
| `MEMORY_KPI_FILE` | `artifacts/memory_validation_kpis.jsonl` | Memory KPI output path |
| `INTERCONNECT_KPI_FILE` | `artifacts/interconnect_kpis.jsonl` | Interconnect KPI output path |
| `BOTTLENECK_KPI_FILE` | `artifacts/bottleneck_kpis.jsonl` | Bottleneck analysis KPI output path |

> Linux is the supported execution target for this validation suite.
