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

## Test Coverage Summary

| File | Test Classes | Total Tests |
|------|-------------|-------------|
| `test_cpu_validation.py` | 15 | ~35 |
| `test_memory_validation.py` | 12 | ~30 |
| `test_gpu_validation.py` | 15 | ~35 |
| `test_interconnect_validation.py` | 12 | ~30 |
| `test_bottleneck_analysis.py` | 10 | ~25 |
| **Total** | **64** | **~155** |
