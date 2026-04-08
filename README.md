# AIHeadNode

Advanced Python test suite for validating an AI Head Node — a high-performance compute
platform based on a **160-core CPU + 4× NVIDIA H100 SXM5 GPU** configuration.

---

## Platform Specifications

| Component | Specification |
|-----------|--------------|
| CPU | 2× sockets, 80 physical cores each (160 total), SMT-2 (320 threads) |
| CPU clock | 1.9 GHz base / 3.9 GHz max turbo |
| CPU FP32 peak | ~19.5 TFLOPS (base) / ~40 TFLOPS (turbo) via AVX-512 |
| System RAM | 2 TB DDR5-4800, 16 memory channels (8 per socket) |
| RAM bandwidth | ~614 GB/s theoretical peak |
| L3 cache | 640 MB total (320 MB per socket) |
| GPUs | 4× NVIDIA H100 SXM5 |
| GPU FP16 | 1979 TFLOPS per GPU (dense), 7916 TFLOPS aggregate |
| GPU BF16 | 1979 TFLOPS per GPU (dense) |
| GPU FP32 | 67 TFLOPS per GPU |
| GPU FP64 | 34 TFLOPS per GPU |
| GPU INT8 | 3958 TOPS per GPU (dense) |
| GPU FP8 | 3958 TFLOPS per GPU (dense) |
| HBM3 memory | 80 GB per GPU (320 GB total) |
| HBM3 bandwidth | 3.35 TB/s per GPU (13.4 TB/s aggregate) |
| GPU interconnect | NVLink 4.0 full-mesh, 900 GB/s per GPU (18 links × 50 GB/s) |
| Host interface | PCIe 5.0 ×16 per GPU, ~63 GB/s unidirectional |

---

## Test Suite Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures, markers, utilities
├── theoretical_specs.py           # Theoretical platform specs & acceptance thresholds
├── test_cpu_validation.py         # 160-core CPU advanced validation (15 test classes)
├── test_memory_validation.py      # Memory subsystem tests (12 test classes)
├── test_gpu_validation.py         # 4× H100 GPU validation (15 test classes)
├── test_interconnect_validation.py# PCIe / NVLink interconnect tests (12 test classes)
└── test_bottleneck_analysis.py    # Roofline + bottleneck analysis (10 test classes)
```

### Test Categories (pytest markers)

| Marker | Description |
|--------|-------------|
| `cpu` | CPU compute and topology tests |
| `memory` | DRAM / cache bandwidth and latency tests |
| `gpu` | NVIDIA H100 compute and memory tests |
| `nvlink` | NVLink 4.0 GPU-to-GPU bandwidth tests |
| `pcie` | PCIe 5.0 host↔device bandwidth tests |
| `bottleneck` | Roofline analysis and bottleneck identification |
| `workload_acceptance` | End-to-end ML workload validation (inference, training, vision) |
| `rdma_validation` | RDMA/MMIO/FENCE/DMA functional validation tests |
| `node_validation` | High-level node health and performance acceptance tests |
| `slow` | Tests that take > 30 s |

---

## Known Platform Bottlenecks

| Bottleneck | Details | Ratio |
|-----------|---------|-------|
| **PCIe H2D ingestion** | 63 GB/s PCIe vs 3350 GB/s HBM3 | 53× bottleneck |
| **CPU DRAM bandwidth** | ~614 GB/s DRAM vs 13,400 GB/s aggregate HBM3 | 22× bottleneck |
| **CPU compute vs GPU** | ~40 TFLOPS CPU FP32 vs 268 TFLOPS GPU FP32 | 7× GPU advantage |
| **Memory-bound kernels** | FP16 roofline knee ≈ 590 FLOP/byte | HBM-limited below knee |
| **NVLink in all-reduce** | 900 GB/s NVLink vs 3350 GB/s HBM3 per GPU | 3.7× HBM advantage |

---

## Installation

```bash
pip install -r requirements.txt

# For GPU tests (requires CUDA 12+):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-ml-py
```

### System Prerequisites (for vLLM tests)

vLLM can compile native helpers at runtime and requires Python C headers
(`Python.h`) and a working compiler toolchain.

```bash
# RHEL / Rocky / CentOS
sudo dnf install -y python3-devel gcc gcc-c++ make

# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y python3-dev build-essential
```

---

## Running the Tests

```bash
# Run all tests (CPU-only tests skip gracefully without GPU)
pytest

# Run only CPU validation tests
pytest -m cpu

# Run only GPU validation tests (requires H100 hardware)
pytest -m gpu

# Run bottleneck analysis and roofline tests
pytest -m bottleneck

# Run all except slow tests
pytest -m "not slow"

# Run a specific test file
pytest tests/test_cpu_validation.py -v

# Run with verbose output and live logging
pytest -v --log-cli-level=INFO

# Run only fast tests with a concise summary
pytest -m "not slow" --tb=line -q
```

---

## Workload Acceptance Tests

The suite includes comprehensive end-to-end ML workload validation:

- **LLM Inference Latency** (`test_hf_causal_lm_inference_latency_enterprise_workloads`):
  Measures sustained inference throughput and latency for enterprise models (Llama 7B/70B, Qwen 72B, DeepSeek-R1).

- **vLLM Data-Pipeline Saturation** (`test_vllm_data_pipeline_saturation_inference_prefill`):
  Validates GPU/CPU utilization during vLLM prefill workload with monitoring of utilization metrics.

- **KV Cache Offload** (`test_predictable_memory_orchestration_kv_cache_offload`):
  Tests graceful KV cache offload to host DDR under high-context workloads.
  Measures throughput degradation, latency impact, and page faults.
  Captures PCIe bandwidth utilization for memory orchestration.

- **Vision Workloads**:
  - YOLO object detection (`test_vision_object_detection_yolo_fps`): FPS and CPU utilization.
  - Diffusers image generation (`test_vision_image_generation_diffusers_throughput`): Image throughput.

- **HuggingFace Training** (`test_hf_encoder_training_throughput`):
  Baseline transformer training throughput on H100 hardware.

### Running Workload Acceptance Tests

```bash
# Run all workload acceptance tests with Linux Bash script (recommended)
./run_workload_acceptance.sh

# Run with custom configuration
./run_workload_acceptance.sh --kv-context-length 16384 --kv-batch-size 8 --min-duration-sec 600

# Skip large models (faster iteration)
./run_workload_acceptance.sh --skip-large-models

# Run via pytest directly
pytest -m workload_acceptance -vv --log-cli-level=INFO

# Run only RDMA/MMIO/FENCE/DMA validation tests
pytest -m rdma_validation -vv tests/test_node_validation.py
```

---

## Workload Environment Variables

All workload tests support configuration via environment variables with sensible defaults.
See [`.env.example`](.env.example) for a comprehensive reference.

### Key Configuration Variables

#### HuggingFace & Large Models

```bash
HF_ENABLE_LARGE_MODELS=1              # Enable 70B+ model workloads
HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1 # Auto-download from hub
HF_LLAMA_7B_MODEL=meta-llama/Llama-2-7b-hf
HF_LLAMA_70B_MODEL=meta-llama/Llama-2-70b-hf
HF_QWEN_70B_MODEL=Qwen/Qwen-72B
HF_DEEPSEEK_R1_QUANT_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

#### KV Cache Offload

```bash
VLLM_KV_OFFLOAD_MODEL=meta-llama/Llama-2-7b-hf
VLLM_KV_CONTEXT_LENGTH=8192           # Context length (4K-32K)
VLLM_KV_BATCH_SIZE=4                  # Batch size
VLLM_KV_NUM_PROMPTS=10                # Number of prompts
VLLM_GPU_MEMORY_UTIL=0.7              # Lower = forces earlier offload
VLLM_CPU_OFFLOAD_GB=128               # Host DDR offload size
```

#### Vision Workloads (YOLO)

```bash
YOLO_DETECT_MODEL=yolov8x.pt          # Model variant (n=nano, x=large)
YOLO_DETECT_BATCH=128                 # Inference batch size
YOLO_DETECT_IMGSZ=640                 # Image size for inference
```

#### PyTorch & CUDA

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3          # GPU device IDs
CUDA_DEVICE_ORDER=PCI_BUS_ID          # Consistent ordering
TORCH_HOME=~/.cache/torch             # PyTorch cache
```

### Configuration Methods (in precedence order)

1. **Environment variables** (highest priority) — set before running tests:
   ```bash
   export VLLM_KV_CONTEXT_LENGTH=16384
   pytest -m workload_acceptance
   ```

2. **Bash script arguments** (default-friendly):
  ```bash
  ./run_workload_acceptance.sh --kv-context-length 16384
   ```

3. **`.env` file** — copy `.env.example` to `.env` and customize:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   pytest -m workload_acceptance
   ```

4. **conftest.py defaults** (lowest priority) — built-in sensible defaults

### Example Workflows

#### Test KV cache offload with large context

```bash
export VLLM_KV_OFFLOAD_MODEL=meta-llama/Llama-2-70b-hf
export VLLM_KV_CONTEXT_LENGTH=32768
export VLLM_KV_BATCH_SIZE=8
pytest -m workload_acceptance tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload -vv
```

#### Quick YOLO validation with small model

```bash
export YOLO_DETECT_MODEL=yolov8n.pt
export YOLO_AGGREGATE_FPS_TARGET=50
pytest -m workload_acceptance tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps -vv
```

#### Enterprise workloads with long inference duration

```bash
./run_workload_acceptance.sh --min-duration-sec 900
```

---

## Node Validation Report

After each pytest session, the framework writes a node validation report to:

- `NODE_VALIDATION_REPORT.md` (for tests marked `node_validation`)
- `TESTCASE_EXECUTION_FINAL_REPORT.md` (other test categories)

The report includes:

- **Highlights**: total, pass, fail, skip counts with color badges.
- **Top 5 KPIs**: compact high-value metrics (throughput/FPS/latency/bandwidth).
- **Issues Summary**: explicit failure reasons and skipped reasons.
- **Success Summary**: successful cases with concise outcomes.
- **KPI Summary**: latest KPI metrics by test key loaded from JSONL.
- **Execution Status Table**: colorized, scrollable, and horizontally resizable table with:
  - test name
  - purpose
  - result
  - status
  - analysis
  - KPI snapshot
  - reason/success/issue detail

KPI source file is controlled by:

```bash
NODE_VALIDATION_KPI_FILE=artifacts/node_validation_kpis.jsonl
```

If KPI file is missing for a run, report generation continues and shows a clear note.

### Generate Node Validation Report

```bash
# Node validation only
pytest -m node_validation -vv

# Workload acceptance only
pytest -m workload_acceptance -vv
```

---

## Test Coverage Summary

| File | Test Classes | Total Tests |
|------|-------------|-------------|
| `test_cpu_validation.py` | 15 | ~35 |
| `test_memory_validation.py` | 12 | ~30 |
| `test_gpu_validation.py` | 15 | ~35 |
| `test_interconnect_validation.py` | 12 | ~30 |
| `test_bottleneck_analysis.py` | 10 | ~25 |
| `test_node_validation.py` | 1 | ~20 (workload acceptance) |
| **Total** | **65** | **~175** |
