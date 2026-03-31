"""
Bottleneck analysis test suite for the AI Head Node.

Compares measured performance against the theoretical peak specifications
for a 160-core CPU + 4× NVIDIA H100 SXM5 platform and identifies
performance bottlenecks using roofline analysis.

Test categories
---------------
TC-BN-01  CPU compute bottleneck identification
TC-BN-02  CPU memory bandwidth bottleneck
TC-BN-03  GPU compute bottleneck (per GPU and aggregate)
TC-BN-04  HBM3 memory bandwidth bottleneck
TC-BN-05  PCIe host↔device data-movement bottleneck
TC-BN-06  NVLink GPU↔GPU bandwidth bottleneck
TC-BN-07  Roofline model analysis per precision (FP16 / BF16 / FP32 / FP64)
TC-BN-08  CPU-GPU pipeline bottleneck (data-prep vs compute)
TC-BN-09  Multi-GPU scaling efficiency bottleneck
TC-BN-10  System-level throughput summary report
"""

import math
import os
import time
import threading
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest

from tests.theoretical_specs import (
    CPU_SPECS,
    H100_SPECS,
    SYSTEM_SPECS,
    ACCEPTANCE_THRESHOLDS,
)
from tests.conftest import assert_efficiency

logger = logging.getLogger("bottleneck_analysis")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
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
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    label: str
    measured: float
    theoretical: float
    unit: str
    efficiency: float = 0.0
    bottleneck_flag: bool = False

    def __post_init__(self):
        if self.theoretical > 0:
            self.efficiency = self.measured / self.theoretical
            self.bottleneck_flag = self.efficiency < 0.70


def _format_report(results: List[BenchmarkResult]) -> str:
    lines = [
        "",
        "=" * 90,
        f"{'METRIC':<45} {'MEASURED':>12} {'THEORETICAL':>14} {'EFF%':>8} {'STATUS':>12}",
        "=" * 90,
    ]
    for r in results:
        status = "⚠ BOTTLENECK" if r.bottleneck_flag else "  OK"
        lines.append(
            f"{r.label:<45} {r.measured:>10.2f} {r.unit:>2}  "
            f"{r.theoretical:>12.2f} {r.unit:>2}  "
            f"{r.efficiency * 100:>6.1f}%  {status:>12}"
        )
    lines.append("=" * 90)
    bottlenecks = [r for r in results if r.bottleneck_flag]
    if bottlenecks:
        lines.append(f"\n⚠  {len(bottlenecks)} BOTTLENECK(S) DETECTED:")
        for r in bottlenecks:
            lines.append(
                f"   • {r.label}: {r.efficiency * 100:.1f}% efficiency "
                f"({r.measured:.2f} / {r.theoretical:.2f} {r.unit})"
            )
    else:
        lines.append("\n✅  No bottlenecks detected (all metrics ≥ 70 % of theoretical).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Low-level measurement helpers (duplicate-free, standalone)
# ---------------------------------------------------------------------------

def _cpu_blas_gflops(n_iters: int = 100, size: int = 1024) -> float:
    if not _HAS_NUMPY:
        return 0.0
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    _ = np.dot(a, b)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = np.dot(a, b)
    elapsed = time.perf_counter() - t0
    return 2 * size ** 3 * n_iters / elapsed / 1e9


def _dram_bw_gbs(n_bytes: int = 512 * 1024 * 1024) -> float:
    if not _HAS_NUMPY:
        return 0.0
    n = n_bytes // 4
    a = np.ones(n, dtype=np.float32)
    b = np.empty(n, dtype=np.float32)
    np.copyto(b, a)
    t0 = time.perf_counter()
    np.copyto(b, a)
    return (2 * n * 4) / (time.perf_counter() - t0) / 1e9


def _gpu_fp16_tflops(device_idx: int, m: int = 8192, k: int = 8192, n: int = 8192,
                     n_iters: int = 10) -> float:
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    torch.mm(a, b); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.mm(a, b)
    torch.cuda.synchronize(device)
    return 2 * m * k * n * n_iters / (time.perf_counter() - t0) / 1e12


def _gpu_fp32_gflops(device_idx: int, m: int = 4096, k: int = 4096, n: int = 4096,
                     n_iters: int = 20) -> float:
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)
    torch.mm(a, b); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.mm(a, b)
    torch.cuda.synchronize(device)
    return 2 * m * k * n * n_iters / (time.perf_counter() - t0) / 1e9


def _hbm_bw_tbs(device_idx: int, n_bytes: int = 4 * 1024 ** 3) -> float:
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    n = n_bytes // 2
    src = torch.ones(n, device=device, dtype=torch.float16)
    dst = torch.empty(n, device=device, dtype=torch.float16)
    dst.copy_(src); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    dst.copy_(src); torch.cuda.synchronize(device)
    return (n_bytes * 2) / (time.perf_counter() - t0) / 1e12


def _h2d_bw_gbs(device_idx: int, n_bytes: int = 1 * 1024 ** 3) -> float:
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    host = torch.ones(n_bytes // 4, dtype=torch.float32).pin_memory()
    host.to(device); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    dev = host.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    return n_bytes / (time.perf_counter() - t0) / 1e9


def _cpu_blas_gflops_worker(args: tuple) -> float:
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    sz, iters = args
    return _cpu_blas_gflops(n_iters=iters, size=sz)


def _p2p_bw_gbs(src: int, dst: int, n_bytes: int = 1 * 1024 ** 3) -> float:
    if not _HAS_TORCH:
        return 0.0
    s = torch.device(f"cuda:{src}")
    d = torch.device(f"cuda:{dst}")
    t = torch.ones(n_bytes // 4, device=s, dtype=torch.float32)
    t.to(d); torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    r = t.to(d); torch.cuda.synchronize(d)
    return n_bytes / (time.perf_counter() - t0) / 1e9


# ---------------------------------------------------------------------------
# TC-BN-01  CPU compute bottleneck
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.cpu
class TestCPUComputeBottleneck:

    def test_cpu_fp32_efficiency_report(self):
        """TC-BN-01: Measure and report CPU FP32 GFLOPS vs theoretical peak."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")

        from concurrent.futures import ProcessPoolExecutor

        n_workers = min(CPU_SPECS.total_physical_cores, os.cpu_count() or 1)
        args_list = [(256, 50)] * n_workers
        # Use the module-level _cpu_blas_gflops_worker (picklable) via a wrapper
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            per_core = list(ex.map(_cpu_blas_gflops_worker, args_list))

        measured_gflops = sum(per_core)
        theoretical_gflops = CPU_SPECS.peak_fp32_tflops_base * 1000
        # Scale theoretical by available cores ratio
        scaled_theoretical = theoretical_gflops * n_workers / CPU_SPECS.total_physical_cores

        result = BenchmarkResult(
            label="CPU FP32 (AVX-512 BLAS, all cores)",
            measured=measured_gflops,
            theoretical=scaled_theoretical,
            unit="GFLOPS",
        )
        logger.info(_format_report([result]))
        assert_efficiency(
            measured_gflops, scaled_theoretical,
            ACCEPTANCE_THRESHOLDS["cpu_avx512_fp32_efficiency"],
            result.label,
        )

    def test_cpu_not_bottleneck_vs_gpu_fp32(self):
        """TC-BN-01b: CPU FP32 peak must be < GPU FP32 peak (GPU should dominate)."""
        cpu_tflops = CPU_SPECS.peak_fp32_tflops_turbo
        gpu_total_tflops = SYSTEM_SPECS.total_fp32_tflops_gpu
        assert cpu_tflops < gpu_total_tflops, (
            f"CPU peak ({cpu_tflops:.1f} TFLOPS) ≥ GPU aggregate ({gpu_total_tflops:.1f} TFLOPS)"
        )
        ratio = gpu_total_tflops / cpu_tflops
        logger.info(
            "GPU FP32 / CPU FP32 ratio = %.1f× (%.1f TFLOPS GPU vs %.1f TFLOPS CPU)",
            ratio, gpu_total_tflops, cpu_tflops,
        )
        assert ratio >= 5.0, (
            f"GPU/CPU compute ratio {ratio:.1f}× is below expected 5× "
            "– CPU may be the bottleneck for FP32 workloads"
        )


# ---------------------------------------------------------------------------
# TC-BN-02  CPU memory bandwidth bottleneck
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.memory
class TestCPUMemoryBandwidthBottleneck:

    def test_cpu_dram_bandwidth_vs_hbm(self):
        """TC-BN-02: CPU DRAM BW must be << HBM3 BW (confirms memory bottleneck topology)."""
        cpu_bw = CPU_SPECS.peak_memory_bandwidth_gbs   # ~614 GB/s
        gpu_hbm_total = SYSTEM_SPECS.total_hbm_bandwidth_tbs * 1000  # TB/s → GB/s

        ratio = gpu_hbm_total / cpu_bw
        logger.info(
            "HBM3 aggregate / CPU DRAM ratio = %.1f× (%.0f GB/s HBM vs %.0f GB/s DRAM)",
            ratio, gpu_hbm_total, cpu_bw,
        )
        # For AI training: GPU HBM should be >> CPU DRAM
        assert ratio >= 10.0, (
            f"HBM/DRAM ratio {ratio:.1f}× < 10× expected – "
            "CPU memory BW may bottleneck data preprocessing"
        )

    @pytest.mark.slow
    def test_cpu_dram_bandwidth_measured(self):
        """TC-BN-02b: Measured CPU DRAM BW ≥ 75 % of theoretical peak."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        results: List[float] = []

        def _worker():
            results.append(_dram_bw_gbs(n_bytes=256 * 1024 * 1024))

        n_threads = min(CPU_SPECS.total_physical_cores, os.cpu_count() or 1)
        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_bw = sum(results)
        result = BenchmarkResult(
            label="CPU DRAM BW (multi-thread STREAM Copy)",
            measured=total_bw,
            theoretical=CPU_SPECS.peak_memory_bandwidth_gbs,
            unit="GB/s",
        )
        logger.info(_format_report([result]))
        assert_efficiency(
            total_bw,
            CPU_SPECS.peak_memory_bandwidth_gbs,
            ACCEPTANCE_THRESHOLDS["cpu_memory_bandwidth_efficiency"],
            result.label,
        )


# ---------------------------------------------------------------------------
# TC-BN-03  GPU compute bottleneck
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.gpu
@pytest.mark.slow
class TestGPUComputeBottleneck:

    def test_per_gpu_fp16_efficiency(self):
        """TC-BN-03a: FP16 efficiency per GPU ≥ 75 % of 1979 TFLOPS theoretical."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        results = []
        for i in range(torch.cuda.device_count()):
            tflops = _gpu_fp16_tflops(i, m=8192, k=8192, n=8192, n_iters=10)
            results.append(
                BenchmarkResult(
                    label=f"GPU {i} FP16 Tensor Core",
                    measured=tflops,
                    theoretical=H100_SPECS.fp16_tflops,
                    unit="TFLOPS",
                )
            )
        logger.info(_format_report(results))
        for r in results:
            assert_efficiency(
                r.measured, r.theoretical,
                ACCEPTANCE_THRESHOLDS["gpu_fp16_efficiency"],
                r.label,
            )

    def test_aggregate_4gpu_fp16_tflops(self):
        """TC-BN-03b: Aggregate 4-GPU FP16 TFLOPS ≥ 75 % of 4 × 1979 TFLOPS."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        n_gpus = torch.cuda.device_count()
        aggregate = [0.0]
        lock = threading.Lock()

        def _worker(idx: int):
            t = _gpu_fp16_tflops(idx, m=8192, k=8192, n=8192, n_iters=5)
            with lock:
                aggregate[0] += t

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = BenchmarkResult(
            label=f"4-GPU aggregate FP16 TFLOPS",
            measured=aggregate[0],
            theoretical=SYSTEM_SPECS.total_fp16_tflops_gpu,
            unit="TFLOPS",
        )
        logger.info(_format_report([result]))
        assert_efficiency(
            aggregate[0],
            SYSTEM_SPECS.total_fp16_tflops_gpu,
            ACCEPTANCE_THRESHOLDS["gpu_fp16_efficiency"],
            result.label,
        )


# ---------------------------------------------------------------------------
# TC-BN-04  HBM3 memory bandwidth bottleneck
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.gpu
@pytest.mark.slow
class TestHBMBandwidthBottleneck:

    def test_hbm_bandwidth_per_gpu_vs_theoretical(self):
        """TC-BN-04a: HBM3 BW per GPU ≥ 80 % of theoretical 3.35 TB/s."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        results = []
        for i in range(torch.cuda.device_count()):
            bw = _hbm_bw_tbs(i, n_bytes=4 * 1024 ** 3)
            results.append(
                BenchmarkResult(
                    label=f"GPU {i} HBM3 bandwidth",
                    measured=bw,
                    theoretical=H100_SPECS.hbm3_bandwidth_tbs,
                    unit="TB/s",
                )
            )
        logger.info(_format_report(results))
        for r in results:
            assert_efficiency(
                r.measured, r.theoretical,
                ACCEPTANCE_THRESHOLDS["gpu_hbm_bandwidth_efficiency"],
                r.label,
            )

    def test_hbm_is_limiting_for_memory_bound_kernels(self):
        """TC-BN-04b: FP16 GEMM at low arithmetic intensity must be HBM-limited (not compute)."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        # Very small GEMM → memory-bound (AI < roofline knee)
        small_tflops = _gpu_fp16_tflops(0, m=64, k=64, n=64, n_iters=1000)
        large_tflops = _gpu_fp16_tflops(0, m=8192, k=8192, n=8192, n_iters=10)
        # Small GEMM must be dominated by memory BW → much lower effective TFLOPS
        assert small_tflops < large_tflops * 0.5, (
            f"Small GEMM ({small_tflops:.1f} TFLOPS) unexpectedly close to "
            f"large GEMM ({large_tflops:.1f} TFLOPS) – HBM bottleneck not visible"
        )


# ---------------------------------------------------------------------------
# TC-BN-05  PCIe bottleneck
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.pcie
@pytest.mark.slow
class TestPCIeBottleneck:

    def test_pcie_h2d_bandwidth_vs_theoretical(self):
        """TC-BN-05a: H2D PCIe BW per GPU ≥ 65 % of 63 GB/s theoretical."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        results = []
        for i in range(torch.cuda.device_count()):
            bw = _h2d_bw_gbs(i, n_bytes=1 * 1024 ** 3)
            results.append(
                BenchmarkResult(
                    label=f"GPU {i} H2D PCIe bandwidth",
                    measured=bw,
                    theoretical=H100_SPECS.pcie_unidirectional_bw_gbs,
                    unit="GB/s",
                )
            )
        logger.info(_format_report(results))
        for r in results:
            assert_efficiency(
                r.measured, r.theoretical,
                ACCEPTANCE_THRESHOLDS["pcie_h2d_bandwidth_efficiency"],
                r.label,
            )

    def test_pcie_bottleneck_ratio_vs_hbm(self):
        """TC-BN-05b: PCIe H2D BW must be << HBM BW (confirms I/O bottleneck)."""
        # Ratio: per-GPU HBM BW vs per-GPU PCIe H2D BW
        hbm_gbs = H100_SPECS.hbm3_bandwidth_tbs * 1000   # TB/s → GB/s
        pcie_gbs = H100_SPECS.pcie_unidirectional_bw_gbs

        ratio = hbm_gbs / pcie_gbs
        logger.info(
            "HBM/PCIe BW ratio = %.1f× (%.0f GB/s HBM vs %.0f GB/s PCIe H2D)",
            ratio, hbm_gbs, pcie_gbs,
        )
        # PCIe is always the data-ingestion bottleneck on this platform
        assert ratio >= 40.0, (
            f"HBM/PCIe ratio {ratio:.1f}× < 40× – PCIe may not be the ingestion bottleneck"
        )


# ---------------------------------------------------------------------------
# TC-BN-06  NVLink bottleneck
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.nvlink
@pytest.mark.slow
class TestNVLinkBottleneck:

    def test_nvlink_bw_vs_theoretical(self):
        """TC-BN-06a: NVLink P2P BW between GPU 0→1 ≥ 70 % of theoretical 900 GB/s."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")
        bw = _p2p_bw_gbs(0, 1, n_bytes=1 * 1024 ** 3)
        result = BenchmarkResult(
            label="NVLink P2P BW GPU 0→1",
            measured=bw,
            theoretical=H100_SPECS.nvlink_total_bw_gbs,
            unit="GB/s",
        )
        logger.info(_format_report([result]))
        assert_efficiency(
            bw, H100_SPECS.nvlink_total_bw_gbs,
            ACCEPTANCE_THRESHOLDS["nvlink_bandwidth_efficiency"],
            result.label,
        )

    def test_nvlink_faster_than_pcie_bottleneck(self):
        """TC-BN-06b: NVLink must not be the bandwidth bottleneck vs PCIe."""
        # NVLink BW >> PCIe BW means the bottleneck is PCIe (host ingestion),
        # not NVLink (GPU-to-GPU).
        nvlink_bw = H100_SPECS.nvlink_total_bw_gbs           # 900 GB/s
        pcie_bw = H100_SPECS.pcie_unidirectional_bw_gbs      # 63 GB/s
        assert nvlink_bw > pcie_bw, (
            "NVLink BW should exceed PCIe BW – topology misconfigured?"
        )


# ---------------------------------------------------------------------------
# TC-BN-07  Roofline model analysis
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
class TestRooflineAnalysis:

    def test_roofline_knee_fp16(self):
        """TC-BN-07: Roofline knee (AI threshold) for FP16 must be ~590 FLOP/byte."""
        ai_threshold = SYSTEM_SPECS.arithmetic_intensity_roofline
        # H100: 1979 TFLOPS FP16 / 3.35 TB/s HBM = ~590 FLOP/byte
        assert 400 <= ai_threshold <= 800, (
            f"Roofline AI threshold {ai_threshold:.1f} FLOP/byte out of expected range [400, 800]"
        )
        logger.info("FP16 roofline knee: %.1f FLOP/byte", ai_threshold)

    def test_roofline_memory_bound_regime(self):
        """TC-BN-07b: A kernel with AI=10 FLOP/byte must be memory-bound on H100."""
        # Expected BW-limited throughput at AI=10:
        # 10 FLOP/byte × 3.35 TB/s HBM = 33.5 TFLOPS  (<< 1979 TFLOPS compute peak)
        ai = 10.0  # FLOP/byte
        bw_limited_tflops = ai * H100_SPECS.hbm3_bandwidth_tbs
        assert bw_limited_tflops < H100_SPECS.fp16_tflops, (
            f"At AI={ai}, BW-limited throughput ({bw_limited_tflops:.1f} TFLOPS) "
            f"should be below compute peak ({H100_SPECS.fp16_tflops:.1f} TFLOPS)"
        )

    def test_roofline_compute_bound_regime(self):
        """TC-BN-07c: A kernel with AI=1000 FLOP/byte must be compute-bound."""
        ai = 1000.0
        bw_limited_tflops = ai * H100_SPECS.hbm3_bandwidth_tbs
        assert bw_limited_tflops >= H100_SPECS.fp16_tflops, (
            f"At AI={ai}, BW-limited throughput ({bw_limited_tflops:.1f} TFLOPS) "
            f"should exceed compute peak – kernel is compute-bound"
        )

    @pytest.mark.parametrize("precision,theoretical_tflops,ai_label", [
        ("FP16",  H100_SPECS.fp16_tflops,  "FP16 Tensor Core"),
        ("BF16",  H100_SPECS.bf16_tflops,  "BF16 Tensor Core"),
        ("FP32",  H100_SPECS.fp32_tflops,  "FP32 CUDA Core"),
        ("FP64",  H100_SPECS.fp64_tflops,  "FP64 Tensor Core"),
        ("INT8",  H100_SPECS.int8_tops,    "INT8 Tensor Core"),
    ])
    def test_roofline_knee_per_precision(self, precision, theoretical_tflops, ai_label):
        """TC-BN-07d: Report roofline knee (FLOP/byte) for each precision."""
        knee = theoretical_tflops * 1e12 / (H100_SPECS.hbm3_bandwidth_tbs * 1e12)
        logger.info(
            "Roofline knee %-20s: %8.1f FLOP/byte  (peak=%7.1f TFLOPS, HBM=%.2f TB/s)",
            ai_label, knee, theoretical_tflops, H100_SPECS.hbm3_bandwidth_tbs,
        )
        assert knee > 0, f"Invalid roofline knee for {precision}"


# ---------------------------------------------------------------------------
# TC-BN-08  CPU-GPU pipeline bottleneck
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.slow
class TestCPUGPUPipelineBottleneck:

    def test_data_preprocessing_not_bottleneck(self):
        """TC-BN-08: CPU data-prep time must be < GPU compute time for same data volume."""
        if not _HAS_NUMPY or not _HAS_TORCH:
            pytest.skip("numpy/torch not available")

        # CPU side: prepare a float32 batch (simulate data normalisation)
        n = 8192 * 8192
        cpu_array = np.random.rand(n).astype(np.float32)

        t0 = time.perf_counter()
        cpu_array = (cpu_array - cpu_array.mean()) / (cpu_array.std() + 1e-8)
        cpu_time = time.perf_counter() - t0

        # GPU side: run GEMM on equivalent data volume
        device = torch.device("cuda:0")
        m = 8192
        a = torch.randn(m, m, device=device, dtype=torch.float16)
        b = torch.randn(m, m, device=device, dtype=torch.float16)
        torch.mm(a, b); torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        c = torch.mm(a, b)
        torch.cuda.synchronize(device)
        gpu_time = time.perf_counter() - t0

        logger.info(
            "CPU data-prep: %.4f s  |  GPU GEMM: %.4f s  |  ratio: %.1f×",
            cpu_time, gpu_time, cpu_time / max(gpu_time, 1e-9),
        )
        # GPU GEMM should be faster than CPU data prep (GPU is the compute engine)
        assert gpu_time <= cpu_time, (
            f"GPU GEMM ({gpu_time:.4f}s) is SLOWER than CPU data-prep ({cpu_time:.4f}s) "
            "– possible GPU underutilisation or thermal throttling"
        )


# ---------------------------------------------------------------------------
# TC-BN-09  Multi-GPU scaling efficiency
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
@pytest.mark.gpu
@pytest.mark.slow
class TestMultiGPUScalingBottleneck:

    def test_scaling_efficiency_2_vs_1_gpu(self):
        """TC-BN-09a: 2-GPU throughput ≥ 80 % of 2× single-GPU throughput."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        if torch.cuda.device_count() < 2:
            pytest.skip("Need ≥ 2 GPUs")
        single = _gpu_fp16_tflops(0, m=8192, k=8192, n=8192, n_iters=5)
        results = [0.0, 0.0]

        def _w(i):
            results[i] = _gpu_fp16_tflops(i, m=8192, k=8192, n=8192, n_iters=5)

        ts = [threading.Thread(target=_w, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        two_gpu_total = sum(results)
        result = BenchmarkResult(
            label="2-GPU FP16 scaling efficiency",
            measured=two_gpu_total,
            theoretical=single * 2,
            unit="TFLOPS",
        )
        logger.info(_format_report([result]))
        assert_efficiency(
            two_gpu_total, single * 2,
            ACCEPTANCE_THRESHOLDS["multi_gpu_scaling_efficiency"],
            result.label,
        )

    def test_scaling_efficiency_4_vs_1_gpu(self):
        """TC-BN-09b: 4-GPU throughput ≥ 80 % of 4× single-GPU throughput."""
        if not _HAS_TORCH:
            pytest.skip("CUDA / PyTorch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 4:
            pytest.skip("Need 4 GPUs")
        single = _gpu_fp16_tflops(0, m=8192, k=8192, n=8192, n_iters=5)
        results = [0.0] * n_gpus

        def _w(i):
            results[i] = _gpu_fp16_tflops(i, m=8192, k=8192, n=8192, n_iters=5)

        ts = [threading.Thread(target=_w, args=(i,)) for i in range(n_gpus)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        total = sum(results)
        result = BenchmarkResult(
            label=f"{n_gpus}-GPU FP16 scaling efficiency",
            measured=total,
            theoretical=single * n_gpus,
            unit="TFLOPS",
        )
        logger.info(_format_report([result]))
        assert_efficiency(
            total, single * n_gpus,
            ACCEPTANCE_THRESHOLDS["multi_gpu_scaling_efficiency"],
            result.label,
        )


# ---------------------------------------------------------------------------
# TC-BN-10  System-level summary report
# ---------------------------------------------------------------------------
@pytest.mark.bottleneck
class TestSystemLevelSummary:

    def test_theoretical_platform_summary(self):
        """TC-BN-10: Log full theoretical platform specification summary."""
        specs = {
            "CPU cores (physical)": f"{CPU_SPECS.total_physical_cores}",
            "CPU threads (logical)": f"{CPU_SPECS.total_logical_threads}",
            "CPU FP32 peak (base, TFLOPS)": f"{CPU_SPECS.peak_fp32_tflops_base:.1f}",
            "CPU FP32 peak (turbo, TFLOPS)": f"{CPU_SPECS.peak_fp32_tflops_turbo:.1f}",
            "CPU DRAM BW peak (GB/s)": f"{CPU_SPECS.peak_memory_bandwidth_gbs:.1f}",
            "CPU L3 cache total (MB)": f"{CPU_SPECS.l3_cache_mb_total:.0f}",
            "GPU count": f"{SYSTEM_SPECS.gpu_count}",
            "GPU model": "NVIDIA H100 SXM5",
            "GPU FP32 peak/GPU (TFLOPS)": f"{H100_SPECS.fp32_tflops:.1f}",
            "GPU FP16 peak/GPU (TFLOPS)": f"{H100_SPECS.fp16_tflops:.1f}",
            "GPU BF16 peak/GPU (TFLOPS)": f"{H100_SPECS.bf16_tflops:.1f}",
            "GPU INT8 peak/GPU (TOPS)": f"{H100_SPECS.int8_tops:.1f}",
            "GPU FP8  peak/GPU (TFLOPS)": f"{H100_SPECS.fp8_tflops:.1f}",
            "GPU FP64 peak/GPU (TFLOPS)": f"{H100_SPECS.fp64_tflops:.1f}",
            "HBM3 capacity/GPU (GB)": f"{H100_SPECS.hbm3_capacity_gb:.0f}",
            "HBM3 BW/GPU (TB/s)": f"{H100_SPECS.hbm3_bandwidth_tbs:.2f}",
            "Total HBM3 BW (TB/s)": f"{SYSTEM_SPECS.total_hbm_bandwidth_tbs:.2f}",
            "NVLink total BW/GPU (GB/s)": f"{H100_SPECS.nvlink_total_bw_gbs:.0f}",
            "PCIe 5.0 H2D BW/GPU (GB/s)": f"{H100_SPECS.pcie_unidirectional_bw_gbs:.0f}",
            "Aggregate GPU FP16 (TFLOPS)": f"{SYSTEM_SPECS.total_fp16_tflops_gpu:.0f}",
            "Roofline knee FP16 (FLOP/byte)": f"{SYSTEM_SPECS.arithmetic_intensity_roofline:.1f}",
        }
        lines = ["\n", "=" * 70, "  AI HEAD NODE — THEORETICAL PLATFORM SPECIFICATIONS", "=" * 70]
        for k, v in specs.items():
            lines.append(f"  {k:<45} {v:>12}")
        lines.append("=" * 70)
        logger.info("\n".join(lines))
        # Always passes — pure reporting test
        assert True

    def test_known_bottlenecks_summary(self):
        """TC-BN-10b: Log known theoretical bottlenecks for the platform."""
        bottlenecks = [
            {
                "subsystem": "Host PCIe (data ingestion)",
                "bottleneck": "PCIe 5.0 ×16 = 63 GB/s per GPU << HBM3 = 3350 GB/s",
                "ratio": f"{H100_SPECS.hbm3_bandwidth_tbs * 1000 / H100_SPECS.pcie_unidirectional_bw_gbs:.0f}× HBM/PCIe",
                "mitigation": "Use in-GPU preprocessing, NVLink-based data redistribution, or PCIe switches.",
            },
            {
                "subsystem": "CPU DRAM bandwidth",
                "bottleneck": f"CPU DRAM peak = {CPU_SPECS.peak_memory_bandwidth_gbs:.0f} GB/s "
                              f"<< HBM3 total = {SYSTEM_SPECS.total_hbm_bandwidth_tbs * 1000:.0f} GB/s",
                "ratio": f"{SYSTEM_SPECS.total_hbm_bandwidth_tbs * 1000 / CPU_SPECS.peak_memory_bandwidth_gbs:.0f}× HBM/DRAM",
                "mitigation": "Pre-process data on GPU, use NUMA-local allocation, or async prefetch.",
            },
            {
                "subsystem": "CPU FP32 vs GPU FP32",
                "bottleneck": f"CPU FP32 turbo = {CPU_SPECS.peak_fp32_tflops_turbo:.0f} TFLOPS "
                              f"<< GPU total FP32 = {SYSTEM_SPECS.total_fp32_tflops_gpu:.0f} TFLOPS",
                "ratio": f"{SYSTEM_SPECS.total_fp32_tflops_gpu / CPU_SPECS.peak_fp32_tflops_turbo:.0f}× GPU/CPU",
                "mitigation": "Offload all FP32 workloads to GPU; CPU handles orchestration only.",
            },
            {
                "subsystem": "NVLink fabric (GPU-to-GPU)",
                "bottleneck": f"NVLink total BW = {SYSTEM_SPECS.nvlink_fabric_bw_gbs:.0f} GB/s — "
                              "not a bottleneck vs HBM but limits all-reduce in training",
                "ratio": f"{SYSTEM_SPECS.nvlink_fabric_bw_gbs / (H100_SPECS.hbm3_bandwidth_tbs * 1000):.2f}× NVLink/HBM",
                "mitigation": "Overlap all-reduce with compute; use ring-all-reduce or NCCL overlap.",
            },
            {
                "subsystem": "Memory-bound kernels (low AI)",
                "bottleneck": f"Any kernel with AI < {SYSTEM_SPECS.arithmetic_intensity_roofline:.0f} FLOP/byte "
                              "is HBM-bandwidth-limited on H100",
                "ratio": f"Roofline knee = {SYSTEM_SPECS.arithmetic_intensity_roofline:.0f} FLOP/byte",
                "mitigation": "Fuse kernels to increase arithmetic intensity; use Flash Attention.",
            },
        ]

        lines = ["\n", "=" * 90, "  KNOWN BOTTLENECKS — 160-core CPU + 4× H100 SXM5 PLATFORM", "=" * 90]
        for bn in bottlenecks:
            lines.append(f"\n  ⚠  {bn['subsystem']}")
            lines.append(f"     Bottleneck : {bn['bottleneck']}")
            lines.append(f"     Ratio      : {bn['ratio']}")
            lines.append(f"     Mitigation : {bn['mitigation']}")
        lines.append("\n" + "=" * 90)
        logger.info("\n".join(lines))
        # Always passes — pure reporting test
        assert True

    @pytest.mark.slow
    def test_end_to_end_bottleneck_report(self):
        """TC-BN-10c: Run all key benchmarks and produce a consolidated bottleneck report."""
        results: List[BenchmarkResult] = []

        # CPU DRAM BW
        if _HAS_NUMPY:
            dram_bw = _dram_bw_gbs(n_bytes=512 * 1024 * 1024)
            results.append(BenchmarkResult(
                "CPU DRAM BW (single-thread STREAM)", dram_bw,
                CPU_SPECS.peak_memory_bandwidth_gbs, "GB/s",
            ))

        # GPU FP16 (GPU 0 only for speed)
        if _HAS_TORCH:
            fp16 = _gpu_fp16_tflops(0, m=8192, k=8192, n=8192, n_iters=5)
            results.append(BenchmarkResult(
                "GPU 0 FP16 Tensor Core", fp16, H100_SPECS.fp16_tflops, "TFLOPS",
            ))

            # GPU FP32
            fp32_gflops = _gpu_fp32_gflops(0, m=4096, k=4096, n=4096, n_iters=10)
            results.append(BenchmarkResult(
                "GPU 0 FP32 CUDA Core", fp32_gflops / 1000,
                H100_SPECS.fp32_tflops, "TFLOPS",
            ))

            # HBM BW
            hbm = _hbm_bw_tbs(0, n_bytes=4 * 1024 ** 3)
            results.append(BenchmarkResult(
                "GPU 0 HBM3 BW", hbm, H100_SPECS.hbm3_bandwidth_tbs, "TB/s",
            ))

            # PCIe H2D
            h2d = _h2d_bw_gbs(0, n_bytes=1 * 1024 ** 3)
            results.append(BenchmarkResult(
                "GPU 0 PCIe H2D BW", h2d, H100_SPECS.pcie_unidirectional_bw_gbs, "GB/s",
            ))

            # P2P (NVLink)
            if torch.cuda.device_count() >= 2:
                p2p = _p2p_bw_gbs(0, 1, n_bytes=1 * 1024 ** 3)
                results.append(BenchmarkResult(
                    "NVLink P2P BW (GPU 0→1)", p2p, H100_SPECS.nvlink_total_bw_gbs, "GB/s",
                ))

        if results:
            logger.info(_format_report(results))

        # Test always passes – the report is the output
        assert True
