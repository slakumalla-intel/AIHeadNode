"""
GPU validation tests for the AI Head Node — vendor-neutral.

Designed to run against any CUDA-compatible AI accelerator:
  NVIDIA H100, H200, A100, RTX 4090 — or AMD MI300X / MI250X via ROCm.

GPU model, memory capacity, memory bandwidth, and TDP are all detected at
runtime from pynvml (or env-var overrides). Thresholds are expressed as
percentages of the detected theoretical peak so that comparisons across
vendors are valid.

Test categories
---------------
TC-GPU-01  GPU discovery and count
TC-GPU-02  GPU model and runtime version (report, not assert vendor)
TC-GPU-03  Accelerator memory capacity per GPU
TC-GPU-04  Accelerator memory bandwidth per GPU
TC-GPU-05  FP32 compute throughput (SGEMM)
TC-GPU-06  FP16 Tensor / Matrix Core throughput
TC-GPU-07  BF16 Tensor / Matrix Core throughput
TC-GPU-08  INT8 Tensor / Matrix Core throughput
TC-GPU-09  FP64 compute throughput
TC-GPU-10  Multi-GPU aggregate compute
TC-GPU-11  GPU utilisation under GEMM load
TC-GPU-12  GPU temperature within safe operating range
TC-GPU-13  GPU power draw within TDP limits
TC-GPU-14  ECC uncorrected error count
TC-GPU-15  CUDA stream concurrency
"""

import logging
import os
import time
from typing import List

import pytest

logger = logging.getLogger("gpu_validation")

from tests.theoretical_specs import H100_SPECS, GPU_SPECS, SYSTEM_SPECS, ACCEPTANCE_THRESHOLDS
from tests.conftest import assert_efficiency

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None          # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    pynvml = None         # type: ignore[assignment]
    _HAS_NVML = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_GPU_COUNT = SYSTEM_SPECS.gpu_count
# AI-grade GPU minimum compute capability for matrix-core operations:
#   NVIDIA Volta = 7.0, Turing = 7.5, Ampere = 8.0, Hopper = 9.0
#   Minimum bar for TC/matrix-core capable devices:
MIN_COMPUTE_CAPABILITY_MAJOR = int(os.environ.get("GPU_MIN_CC_MAJOR", "7"))
# GPU TDP and thermal limits come from runtime detection in GPU_SPECS
GPU_TDP_W      = GPU_SPECS.tdp_watts
MAX_GPU_TEMP_C = GPU_SPECS.max_temp_c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_gpu():
    if not _HAS_TORCH:
        pytest.skip("CUDA / PyTorch not available")


def _run_sgemm_gflops(device_idx: int, m: int = 4096, k: int = 4096, n: int = 4096,
                      n_iters: int = 20) -> float:
    """Return peak FP32 GFLOPS for a single GPU using cuBLAS SGEMM."""
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)
    # warm-up
    torch.mm(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    flops = 2 * m * k * n * n_iters
    return flops / elapsed / 1e9


def _run_hgemm_tflops(device_idx: int, m: int = 8192, k: int = 8192, n: int = 8192,
                      n_iters: int = 10) -> float:
    """Return peak FP16 Tensor Core TFLOPS using torch.mm with float16."""
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    torch.mm(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    flops = 2 * m * k * n * n_iters
    return flops / elapsed / 1e12


def _run_bf16_tflops(device_idx: int, m: int = 8192, k: int = 8192, n: int = 8192,
                     n_iters: int = 10) -> float:
    """Return peak BF16 TFLOPS."""
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    torch.mm(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    flops = 2 * m * k * n * n_iters
    return flops / elapsed / 1e12


def _run_fp64_tflops(device_idx: int, m: int = 4096, k: int = 4096, n: int = 4096,
                     n_iters: int = 5) -> float:
    """Return peak FP64 TFLOPS."""
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    a = torch.randn(m, k, device=device, dtype=torch.float64)
    b = torch.randn(k, n, device=device, dtype=torch.float64)
    torch.mm(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    flops = 2 * m * k * n * n_iters
    return flops / elapsed / 1e12


def _hbm_bandwidth_tbs(device_idx: int, n_bytes: int = 4 * 1024 ** 3) -> float:
    """
    Measure HBM3 read bandwidth via a large device-to-device memory copy.
    Returns bandwidth in TB/s.
    """
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    n = n_bytes // 2   # float16 elements
    src = torch.ones(n, device=device, dtype=torch.float16)
    dst = torch.empty(n, device=device, dtype=torch.float16)
    dst.copy_(src)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    dst.copy_(src)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    bytes_moved = n_bytes * 2  # read + write
    return bytes_moved / elapsed / 1e12


def _pcie_h2d_bandwidth_gbs(device_idx: int, n_bytes: int = 512 * 1024 * 1024,
                             n_iters: int = 5) -> float:
    """Measure PCIe host-to-device transfer bandwidth in GB/s (pinned memory)."""
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    host_buf = torch.ones(n_bytes // 4, dtype=torch.float32).pin_memory()
    dev_buf  = torch.empty(n_bytes // 4, device=device, dtype=torch.float32)
    dev_buf.copy_(host_buf)                 # warm-up
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dev_buf.copy_(host_buf)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return (n_bytes * n_iters) / elapsed / 1e9


def _pcie_d2h_bandwidth_gbs(device_idx: int, n_bytes: int = 512 * 1024 * 1024,
                             n_iters: int = 5) -> float:
    """Measure PCIe device-to-host transfer bandwidth in GB/s (pinned memory)."""
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    dev_buf  = torch.ones(n_bytes // 4, device=device, dtype=torch.float32)
    host_buf = torch.empty(n_bytes // 4, dtype=torch.float32).pin_memory()
    host_buf.copy_(dev_buf)                 # warm-up
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        host_buf.copy_(dev_buf)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return (n_bytes * n_iters) / elapsed / 1e9


# ---------------------------------------------------------------------------
# TC-GPU-01  GPU discovery
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUDiscovery:

    def test_gpu_count_equals_four(self):
        """TC-GPU-01a: System must expose exactly 4 CUDA GPUs."""
        _skip_if_no_gpu()
        count = torch.cuda.device_count()
        assert count == EXPECTED_GPU_COUNT, (
            f"Expected {EXPECTED_GPU_COUNT} GPUs, detected {count}"
        )

    def test_all_gpus_initialise(self):
        """TC-GPU-01b: All 4 GPUs must initialise without error."""
        _skip_if_no_gpu()
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            _ = torch.zeros(1, device=f"cuda:{i}")


# ---------------------------------------------------------------------------
# TC-GPU-02  GPU model and runtime version
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUModel:

    def test_gpu_model_reported(self):
        """TC-GPU-02a: Log GPU model name for every device (no vendor assertion).

        Competitive note: records exact model strings (H100, MI300X, A100, etc.)
        for use in KPI comparison tables without failing non-H100 hardware.
        """
        _skip_if_no_gpu()
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "GPU %d: %s | CC %d.%d | VRAM %.1f GB",
                i, name, props.major, props.minor,
                props.total_memory / 1024 ** 3,
            )
            # Simply verify the name is non-empty (never assert a specific vendor)
            assert name, f"GPU {i} returned empty model name"

    def test_cuda_version_12_or_above(self):
        """TC-GPU-02b: CUDA / ROCm runtime version must be ≥ 12.0."""
        _skip_if_no_gpu()
        version_str = torch.version.cuda or torch.version.hip or "0.0"
        major = int(version_str.split(".")[0])
        assert major >= 12, f"CUDA/ROCm version {version_str} < 12.0"

    def test_compute_capability_ai_grade(self):
        """TC-GPU-02c: All GPUs must have compute capability ≥ 7.0 (AI-grade).

        Covers Volta (7.0), Turing (7.5), Ampere (8.0), Hopper (9.0),
        and AMD ROCm emulated capabilities.
        Set GPU_MIN_CC_MAJOR env var to raise or lower the threshold.
        """
        _skip_if_no_gpu()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            major = props.major
            assert major >= MIN_COMPUTE_CAPABILITY_MAJOR, (
                f"GPU {i}: compute capability {major}.{props.minor} "
                f"< minimum AI-grade {MIN_COMPUTE_CAPABILITY_MAJOR}.0"
            )


# ---------------------------------------------------------------------------
# TC-GPU-03  Accelerator memory capacity
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUMemoryCapacity:

    def test_memory_capacity_per_gpu(self):
        """TC-GPU-03: Each GPU must report ≥ 98 % of its detected theoretical VRAM.

        Uses the runtime-detected `GPU_SPECS.memory_capacity_gb` so the test
        passes correctly on H100 (80 GB), MI300X (192 GB), A100 (40/80 GB), etc.
        """
        _skip_if_no_gpu()
        theoretical_gb = GPU_SPECS.memory_capacity_gb
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            capacity_gb = props.total_memory / 1024 ** 3
            floor_gb = theoretical_gb * 0.98 if theoretical_gb > 0 else 8.0
            assert capacity_gb >= floor_gb, (
                f"GPU {i}: {capacity_gb:.2f} GB < expected floor {floor_gb:.2f} GB"
            )
            logger.info("GPU %d VRAM: %.2f GB (theoretical %.2f GB, %.1f%%)",
                        i, capacity_gb, theoretical_gb,
                        100 * capacity_gb / theoretical_gb if theoretical_gb else 0)

    def test_total_gpu_memory_scales_with_count(self):
        """TC-GPU-03b: Total GPU memory across all GPUs must match N × per-GPU spec."""
        _skip_if_no_gpu()
        n_gpus = torch.cuda.device_count()
        total_gb = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(n_gpus)
        ) / 1024 ** 3
        theoretical_gb = GPU_SPECS.memory_capacity_gb
        if theoretical_gb > 0:
            expected_total_gb = n_gpus * theoretical_gb * 0.98
            assert total_gb >= expected_total_gb, (
                f"Total GPU memory {total_gb:.1f} GB < {expected_total_gb:.1f} GB"
            )
        logger.info("Total GPU VRAM: %.2f GB across %d GPU(s)", total_gb, n_gpus)


# ---------------------------------------------------------------------------
# TC-GPU-04  Accelerator memory bandwidth
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestGPUMemoryBandwidth:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_memory_bandwidth_per_gpu(self, device_idx):
        """TC-GPU-04: GPU memory BW ≥ 80 % of detected theoretical peak (TB/s).

        Uses `GPU_SPECS.memory_bandwidth_tbs` which is detected at runtime from
        pynvml so it reflects H100 SXM5 (3.35 TB/s), H100 PCIe (2.0 TB/s),
        MI300X (5.3 TB/s), etc. correctly.
        """
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        if GPU_SPECS.memory_bandwidth_tbs <= 0:
            pytest.skip("Memory bandwidth spec not detected")
        bw_tbs = _hbm_bandwidth_tbs(device_idx, n_bytes=4 * 1024 ** 3)
        assert_efficiency(
            bw_tbs, GPU_SPECS.memory_bandwidth_tbs,
            ACCEPTANCE_THRESHOLDS["gpu_hbm_bandwidth_efficiency"],
            f"GPU {device_idx} memory bandwidth"
        )


# ---------------------------------------------------------------------------
# TC-GPU-05  FP32 throughput
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestFP32Throughput:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_fp32_gflops_per_gpu(self, device_idx):
        """TC-GPU-05: FP32 GFLOPS ≥ 70 % of detected theoretical GPU FP32 peak.

        Uses runtime-detected `GPU_SPECS.fp32_tflops` so the threshold scales
        correctly across H100 (67 TFLOPS), A100 (19.5 TFLOPS), MI300X (163 TFLOPS).
        """
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        gflops = _run_sgemm_gflops(device_idx, m=4096, k=4096, n=4096, n_iters=20)
        theoretical_gflops = GPU_SPECS.fp32_tflops * 1000
        assert_efficiency(
            gflops, theoretical_gflops,
            ACCEPTANCE_THRESHOLDS["gpu_fp32_efficiency"],
            f"FP32 GFLOPS GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-GPU-06  FP16 Tensor Core throughput
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestFP16ThroughputGPU:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_fp16_tflops_per_gpu(self, device_idx):
        """TC-GPU-06: FP16 matrix TFLOPS ≥ 75 % of detected theoretical peak.

        H100 SXM5 = 1979 TFLOPS, MI300X = 1307 TFLOPS, A100 = 312 TFLOPS.
        """
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        tflops = _run_hgemm_tflops(device_idx, m=8192, k=8192, n=8192, n_iters=10)
        assert_efficiency(
            tflops, GPU_SPECS.fp16_tflops,
            ACCEPTANCE_THRESHOLDS["gpu_fp16_efficiency"],
            f"FP16 TFLOPS GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-GPU-07  BF16 Tensor Core throughput
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestBF16Throughput:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_bf16_tflops_per_gpu(self, device_idx):
        """TC-GPU-07: BF16 matrix TFLOPS ≥ 75 % of detected theoretical peak."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        tflops = _run_bf16_tflops(device_idx, m=8192, k=8192, n=8192, n_iters=10)
        assert_efficiency(
            tflops, GPU_SPECS.bf16_tflops,
            ACCEPTANCE_THRESHOLDS["gpu_bf16_efficiency"],
            f"BF16 TFLOPS GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-GPU-08  INT8 throughput (via PyTorch int8 GEMM)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestINT8Throughput:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_int8_tops_per_gpu(self, device_idx):
        """TC-GPU-08: INT8 throughput per GPU ≥ 75 % of theoretical 3958 TOPS."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        device = torch.device(f"cuda:{device_idx}")
        m, k, n, n_iters = 8192, 8192, 8192, 5
        a = torch.randint(-128, 127, (m, k), device=device, dtype=torch.int8)
        b = torch.randint(-128, 127, (k, n), device=device, dtype=torch.int8)
        # PyTorch does not support int8 mm directly on all platforms; use bmm via float
        a_fp = a.to(torch.float16)
        b_fp = b.to(torch.float16)
        torch.mm(a_fp, b_fp)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            c = torch.mm(a_fp, b_fp)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        ops = 2 * m * k * n * n_iters
        tops = ops / elapsed / 1e12
        assert_efficiency(
            tops, GPU_SPECS.int8_tops,
            ACCEPTANCE_THRESHOLDS["gpu_int8_efficiency"],
            f"INT8-proxy TOPS GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-GPU-09  FP64 throughput
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestFP64Throughput:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_fp64_tflops_per_gpu(self, device_idx):
        """TC-GPU-09: FP64 TFLOPS per GPU ≥ 70 % of theoretical 34 TFLOPS."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        tflops = _run_fp64_tflops(device_idx, m=4096, k=4096, n=4096, n_iters=5)
        assert_efficiency(
            tflops, GPU_SPECS.fp64_tflops,
            ACCEPTANCE_THRESHOLDS["gpu_fp32_efficiency"],
            f"FP64 TFLOPS GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-GPU-10  Multi-GPU aggregate compute
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestMultiGPUCompute:

    def test_all_gpus_concurrent_fp16_tflops(self):
        """TC-GPU-10: Aggregate FP16 TFLOPS across 4 concurrent GPUs ≥ 80 % of 4× single."""
        _skip_if_no_gpu()
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need at least 2 GPUs for multi-GPU test")

        import threading
        results: List[float] = [0.0] * n_gpus

        def _worker(idx: int):
            results[idx] = _run_hgemm_tflops(idx, m=8192, k=8192, n=8192, n_iters=5)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Single-GPU baseline (sequential)
        single_gpu_tflops = _run_hgemm_tflops(0, m=8192, k=8192, n=8192, n_iters=5)
        ideal_aggregate = single_gpu_tflops * n_gpus
        aggregate = sum(results)

        assert_efficiency(
            aggregate, ideal_aggregate,
            ACCEPTANCE_THRESHOLDS["multi_gpu_scaling_efficiency"],
            f"Multi-GPU FP16 aggregate ({n_gpus} GPUs)"
        )


# ---------------------------------------------------------------------------
# TC-GPU-11  GPU utilisation under load
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestGPUUtilisation:

    def test_gpu_utilisation_near_100_percent(self):
        """TC-GPU-11: GPU 0 utilisation must be ≥ 90 % during GEMM load."""
        _skip_if_no_gpu()
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        device = torch.device("cuda:0")
        m, k, n = 8192, 8192, 8192
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Launch GEMM in background thread
        running = [True]
        def _gemm_loop():
            while running[0]:
                torch.mm(a, b)
                torch.cuda.synchronize(device)

        import threading
        t = threading.Thread(target=_gemm_loop)
        t.start()
        time.sleep(1.0)   # let it ramp

        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        running[0] = False
        t.join()
        assert util >= 90, f"GPU utilisation {util}% < 90 % during GEMM"


# ---------------------------------------------------------------------------
# TC-GPU-12  GPU temperature
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUTemperature:

    def test_gpu_temperature_in_range(self):
        """TC-GPU-12: All GPU temperatures must be < detected max operating temp.

        Threshold is detected via pynvml shutdown limit (or GPU_MAX_TEMP_C env var),
        so it works for H100 (83 °C), MI300X (90 °C), A100 (83 °C), etc.
        """
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            assert temp < MAX_GPU_TEMP_C, (
                f"GPU {i} temperature {temp} \u00b0C \u2265 limit {MAX_GPU_TEMP_C} \u00b0C"
            )


# ---------------------------------------------------------------------------
# TC-GPU-13  GPU power draw
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUPowerDraw:

    def test_gpu_power_within_tdp_at_idle(self):
        """TC-GPU-13: All GPU power draws must be < detected TDP at idle.

        TDP is detected via pynvml power limit query (or GPU_TDP_WATTS env var),
        so it works for H100 SXM5 (700 W), H100 PCIe (350 W), MI300X (750 W), etc.
        """
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
            assert power_w < GPU_TDP_W, (
                f"GPU {i} power draw {power_w:.0f} W exceeds TDP {GPU_TDP_W:.0f} W"
            )


# ---------------------------------------------------------------------------
# TC-GPU-14  ECC error count
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUECC:

    def test_gpu_ecc_enabled(self):
        """TC-GPU-14a: ECC must be enabled on all GPUs."""
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                mode = pynvml.nvmlDeviceGetEccMode(handle)
                # mode is a tuple: (current, pending); current must be FEATURE_ENABLED (1)
                current_mode = mode[0] if isinstance(mode, tuple) else mode
                assert current_mode == pynvml.NVML_FEATURE_ENABLED, (
                    f"GPU {i}: ECC is not enabled (mode={current_mode})"
                )
            except pynvml.NVMLError as e:
                pytest.skip(f"NVML ECC query failed: {e}")

    def test_no_uncorrected_gpu_errors(self):
        """TC-GPU-14b: All GPUs must have zero uncorrected ECC errors."""
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                errors = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_AGGREGATE_ECC,
                )
                assert errors == 0, (
                    f"GPU {i}: {errors} uncorrected ECC errors detected"
                )
            except pynvml.NVMLError as e:
                pytest.skip(f"NVML ECC error query failed: {e}")


# ---------------------------------------------------------------------------
# TC-GPU-15  CUDA stream concurrency
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestCUDAStreamConcurrency:

    def test_concurrent_streams_faster_than_serial(self):
        """TC-GPU-15: Two concurrent CUDA streams must complete faster than 2× serial."""
        _skip_if_no_gpu()
        device = torch.device("cuda:0")
        m, k, n, n_iters = 4096, 4096, 4096, 5
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)

        # Serial timing
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_iters * 2):
            torch.mm(a, b)
        torch.cuda.synchronize(device)
        serial_time = time.perf_counter() - t0

        # Concurrent streams
        s1 = torch.cuda.Stream(device=device)
        s2 = torch.cuda.Stream(device=device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.cuda.stream(s1):
            for _ in range(n_iters):
                torch.mm(a, b)
        with torch.cuda.stream(s2):
            for _ in range(n_iters):
                torch.mm(a, b)
        torch.cuda.synchronize(device)
        concurrent_time = time.perf_counter() - t0

        assert concurrent_time < serial_time, (
            f"Concurrent ({concurrent_time:.3f}s) not faster than serial ({serial_time:.3f}s)"
        )


# ---------------------------------------------------------------------------
# TC-GPU-16  GPU enumeration and PCIe topology
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.pcie
class TestGPUEnumerationPCIeTopology:
    """Verify correct GPU enumeration and that PCIe links meet the Gen-5 ×16 spec."""

    def test_gpu_enumeration_count_consistent(self):
        """TC-GPU-16a: NVML and CUDA must report identical GPU counts (= 4)."""
        _skip_if_no_gpu()
        cuda_count = torch.cuda.device_count()
        if _HAS_NVML:
            nvml_count = pynvml.nvmlDeviceGetCount()
            assert cuda_count == nvml_count, (
                f"CUDA reports {cuda_count} GPUs but NVML reports {nvml_count}"
            )
        assert cuda_count == EXPECTED_GPU_COUNT, (
            f"Expected {EXPECTED_GPU_COUNT} GPUs, detected {cuda_count}"
        )

    def test_each_gpu_has_unique_pci_bus_id(self):
        """TC-GPU-16b: Each GPU must have a unique, non-empty PCI bus ID string."""
        _skip_if_no_gpu()
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        seen: List[str] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle  = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            raw_id  = pci_info.busId
            bus_id  = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            assert bus_id, f"GPU {i}: PCI bus ID is empty"
            assert bus_id not in seen, (
                f"GPU {i}: duplicate PCI bus ID '{bus_id}' already seen on another GPU"
            )
            seen.append(bus_id)

    def test_pcie_max_link_generation_is_gen5(self):
        """TC-GPU-16c: Maximum supported PCIe generation must be ≥ Gen 5 on every GPU."""
        _skip_if_no_gpu()
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                max_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                assert max_gen >= H100_SPECS.pcie_gen, (
                    f"GPU {i}: max PCIe Gen {max_gen} < expected Gen {H100_SPECS.pcie_gen}"
                )
            except pynvml.NVMLError as exc:
                pytest.skip(f"GPU {i}: PCIe gen query not supported – {exc}")

    def test_pcie_max_link_width_is_x16(self):
        """TC-GPU-16d: Maximum supported PCIe link width must be ≥ ×16 on every GPU."""
        _skip_if_no_gpu()
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                max_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                assert max_width >= H100_SPECS.pcie_lanes, (
                    f"GPU {i}: max PCIe width ×{max_width} < expected ×{H100_SPECS.pcie_lanes}"
                )
            except pynvml.NVMLError as exc:
                pytest.skip(f"GPU {i}: PCIe width query not supported – {exc}")

    def test_pcie_current_link_gen_not_degraded(self):
        """TC-GPU-16e: Current PCIe gen must equal the advertised max (no downgrade)."""
        _skip_if_no_gpu()
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                curr_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                max_gen  = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                assert curr_gen >= max_gen, (
                    f"GPU {i}: PCIe link gen degraded – current Gen {curr_gen} < max Gen {max_gen}"
                )
            except pynvml.NVMLError as exc:
                pytest.skip(f"GPU {i}: PCIe gen query not supported – {exc}")

    def test_pcie_current_link_width_not_degraded(self):
        """TC-GPU-16f: Current PCIe width must equal the advertised max (no downgrade)."""
        _skip_if_no_gpu()
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                curr_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                max_width  = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                assert curr_width >= max_width, (
                    f"GPU {i}: PCIe link width degraded – ×{curr_width} < max ×{max_width}"
                )
            except pynvml.NVMLError as exc:
                pytest.skip(f"GPU {i}: PCIe width query not supported – {exc}")

    def test_gpu_pci_bus_numbers_are_unique(self):
        """TC-GPU-16g: Each GPU must reside on a distinct PCIe bus number."""
        _skip_if_no_gpu()
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        bus_numbers: List[int] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle   = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            bus      = pci_info.bus
            assert bus not in bus_numbers, (
                f"GPU {i}: PCIe bus number {bus:#04x} collides with another GPU"
            )
            bus_numbers.append(bus)

    @pytest.mark.slow
    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_pcie_host_to_device_bandwidth(self, device_idx):
        """TC-GPU-16h: H2D PCIe BW per GPU must be ≥ 65 % of theoretical 63 GB/s."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        bw_gbs = _pcie_h2d_bandwidth_gbs(device_idx)
        assert_efficiency(
            bw_gbs, H100_SPECS.pcie_unidirectional_bw_gbs,
            ACCEPTANCE_THRESHOLDS["pcie_h2d_bandwidth_efficiency"],
            f"PCIe H2D bandwidth GPU {device_idx}",
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_pcie_device_to_host_bandwidth(self, device_idx):
        """TC-GPU-16i: D2H PCIe BW per GPU must be ≥ 65 % of theoretical 63 GB/s."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        bw_gbs = _pcie_d2h_bandwidth_gbs(device_idx)
        assert_efficiency(
            bw_gbs, H100_SPECS.pcie_unidirectional_bw_gbs,
            ACCEPTANCE_THRESHOLDS["pcie_d2h_bandwidth_efficiency"],
            f"PCIe D2H bandwidth GPU {device_idx}",
        )
