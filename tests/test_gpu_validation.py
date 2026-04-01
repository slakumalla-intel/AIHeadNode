"""
GPU (4× NVIDIA H100 SXM5) validation tests for the AI Head Node.

Test categories
---------------
TC-GPU-01  GPU discovery and count
TC-GPU-02  GPU model and driver version
TC-GPU-03  HBM3 memory capacity per GPU
TC-GPU-04  HBM3 memory bandwidth (per GPU)
TC-GPU-05  FP32 compute throughput (SGEMM)
TC-GPU-06  FP16 Tensor Core throughput (HGEMM)
TC-GPU-07  BF16 Tensor Core throughput
TC-GPU-08  INT8 Tensor Core throughput
TC-GPU-09  FP64 compute throughput
TC-GPU-10  Multi-GPU aggregate compute (all 4 GPUs simultaneously)
TC-GPU-11  GPU utilisation stays near 100 % under GEMM load
TC-GPU-12  GPU temperature within operating range during load
TC-GPU-13  GPU power draw within TDP limits
TC-GPU-14  GPU ECC uncorrected error count
TC-GPU-15  CUDA stream concurrency (overlapping kernels)
"""

import time
from typing import List

import pytest

from tests.theoretical_specs import H100_SPECS, SYSTEM_SPECS, ACCEPTANCE_THRESHOLDS
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

EXPECTED_GPU_COUNT = SYSTEM_SPECS.gpu_count   # 4
H100_KEYWORDS      = {"H100", "H200"}         # acceptable GPU model identifiers
MAX_GPU_TEMP_C     = 83.0                      # H100 max operating temp (°C)
H100_TDP_W         = 700.0                     # H100 SXM5 TDP in watts


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
# TC-GPU-02  GPU model and driver
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUModel:

    def test_gpu_model_is_h100(self):
        """TC-GPU-02a: All GPUs must be NVIDIA H100 (or H200) class."""
        _skip_if_no_gpu()
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            assert any(kw in name for kw in H100_KEYWORDS), (
                f"GPU {i} is '{name}', expected an H100 or H200 device"
            )

    def test_cuda_version_12_or_above(self):
        """TC-GPU-02b: CUDA runtime version must be ≥ 12.0."""
        _skip_if_no_gpu()
        version_str = torch.version.cuda or "0.0"
        major = int(version_str.split(".")[0])
        assert major >= 12, f"CUDA version {version_str} < 12.0"

    def test_compute_capability_90(self):
        """TC-GPU-02c: H100 compute capability must be 9.0."""
        _skip_if_no_gpu()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            major, minor = props.major, props.minor
            assert (major, minor) == (9, 0), (
                f"GPU {i}: compute capability {major}.{minor}, expected 9.0"
            )


# ---------------------------------------------------------------------------
# TC-GPU-03  HBM3 memory capacity
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestHBMCapacity:

    def test_hbm_capacity_per_gpu(self):
        """TC-GPU-03: Each GPU must have ≥ 79 GB HBM3 (≥ 98.75 % of 80 GB spec)."""
        _skip_if_no_gpu()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            capacity_gb = props.total_memory / 1024 ** 3
            assert capacity_gb >= H100_SPECS.hbm3_capacity_gb * 0.9875, (
                f"GPU {i}: {capacity_gb:.2f} GB < {H100_SPECS.hbm3_capacity_gb * 0.9875:.2f} GB"
            )

    def test_total_gpu_memory(self):
        """TC-GPU-03b: Total GPU memory across 4 GPUs must be ≥ 315 GB."""
        _skip_if_no_gpu()
        total_gb = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ) / 1024 ** 3
        assert total_gb >= 315.0, f"Total GPU memory {total_gb:.1f} GB < 315 GB"


# ---------------------------------------------------------------------------
# TC-GPU-04  HBM3 bandwidth
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestHBMBandwidth:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_hbm_bandwidth_per_gpu(self, device_idx):
        """TC-GPU-04: HBM3 BW per GPU must be ≥ 80 % of theoretical 3.35 TB/s."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        bw_tbs = _hbm_bandwidth_tbs(device_idx, n_bytes=4 * 1024 ** 3)
        assert_efficiency(
            bw_tbs, H100_SPECS.hbm3_bandwidth_tbs,
            ACCEPTANCE_THRESHOLDS["gpu_hbm_bandwidth_efficiency"],
            f"HBM3 bandwidth GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-GPU-05  FP32 throughput
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
class TestFP32Throughput:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_fp32_gflops_per_gpu(self, device_idx):
        """TC-GPU-05: FP32 GFLOPS per GPU ≥ 70 % of theoretical 67 TFLOPS."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        gflops = _run_sgemm_gflops(device_idx, m=4096, k=4096, n=4096, n_iters=20)
        theoretical_gflops = H100_SPECS.fp32_tflops * 1000
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
        """TC-GPU-06: FP16 TFLOPS per GPU ≥ 75 % of theoretical 1979 TFLOPS."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        tflops = _run_hgemm_tflops(device_idx, m=8192, k=8192, n=8192, n_iters=10)
        assert_efficiency(
            tflops, H100_SPECS.fp16_tflops,
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
        """TC-GPU-07: BF16 TFLOPS per GPU ≥ 75 % of theoretical 1979 TFLOPS."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        tflops = _run_bf16_tflops(device_idx, m=8192, k=8192, n=8192, n_iters=10)
        assert_efficiency(
            tflops, H100_SPECS.bf16_tflops,
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
        # Compare against INT8 theoretical; FP16 proxy understimates INT8 but
        # validates Tensor Core utilisation at comparable matrix size.
        assert_efficiency(
            tops, H100_SPECS.int8_tops,
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
            tflops, H100_SPECS.fp64_tflops,
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
        """TC-GPU-12: All GPU temperatures must be < 83 °C at idle/warm-up."""
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            assert temp < MAX_GPU_TEMP_C, (
                f"GPU {i} temperature {temp} °C ≥ {MAX_GPU_TEMP_C} °C"
            )


# ---------------------------------------------------------------------------
# TC-GPU-13  GPU power draw
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUPowerDraw:

    def test_gpu_power_within_tdp_at_idle(self):
        """TC-GPU-13: All GPU power draws must be < TDP (700 W) at idle."""
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
            assert power_w < H100_TDP_W, (
                f"GPU {i} power draw {power_w:.0f} W exceeds TDP {H100_TDP_W:.0f} W"
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
