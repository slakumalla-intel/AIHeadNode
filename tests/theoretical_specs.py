"""
Theoretical performance specifications for the AI Head Node platform.

Runtime-detected topology values (core count, memory size, storage size, GPU count)
are used where possible to avoid hardcoded configuration assumptions.

Bandwidth and throughput figures remain theoretical peaks from vendor datasheets.
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import Dict

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _detect_cpu_counts() -> tuple[int, int]:
    """Return detected (physical_cores, logical_threads) with safe fallbacks."""
    logical = os.cpu_count() or 1
    physical = logical
    if _HAS_PSUTIL:
        physical = psutil.cpu_count(logical=False) or logical
        logical = psutil.cpu_count(logical=True) or logical
    return max(1, physical), max(1, logical)


def _detect_sockets() -> int:
    """Infer socket count from Linux NUMA node directories; fallback to 1."""
    numa_root = "/sys/devices/system/node"
    if os.path.isdir(numa_root):
        nodes = [
            d for d in os.listdir(numa_root)
            if d.startswith("node") and d[4:].isdigit()
        ]
        if nodes:
            return max(1, len(nodes))
    return 1


def _detect_total_ram_gb() -> float:
    """Return installed RAM (GB) or 0.0 when unavailable."""
    if _HAS_PSUTIL:
        return round(psutil.virtual_memory().total / 1024 ** 3, 2)
    return 0.0


def _detect_total_storage_gb() -> float:
    """Return total filesystem size (GB) for root volume."""
    try:
        root = os.path.abspath(os.sep)
        return round(shutil.disk_usage(root).total / 1024 ** 3, 2)
    except OSError:
        return 0.0


def _detect_gpu_count() -> int:
    """Return number of detected CUDA GPUs; 0 when CUDA is unavailable."""
    if _HAS_TORCH:
        return torch.cuda.device_count()
    return 0


def _detect_nvlink_available() -> bool:
    """Check if NVLink is available on at least one GPU via NVML."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                # Try to query NVLink link state; if it succeeds, NVLink is supported
                state = pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                return True  # At least one link is queryable
            except pynvml.NVMLError:
                # Link not available; continue to next GPU
                continue
        return False
    except Exception:
        return False

# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPUSpecs:
    """Theoretical specifications for the host CPU subsystem."""

    # Topology
    sockets: int = 2
    cores_per_socket: int = 80
    total_physical_cores: int = 160
    threads_per_core: int = 2
    total_logical_threads: int = 320

    # Clock
    base_clock_ghz: float = 1.9
    max_turbo_ghz: float = 3.9

    # AVX-512 FLOPS: 2 FMA units × 16 FP32 per cycle × 2 ops/FMA
    avx512_fp32_flops_per_cycle_per_core: int = 64   # 2×FMA×16×2
    avx512_fp64_flops_per_cycle_per_core: int = 32

    # Peak FP32 throughput (TFLOPS) at base clock
    @property
    def peak_fp32_tflops_base(self) -> float:
        return (
            self.total_physical_cores
            * self.avx512_fp32_flops_per_cycle_per_core
            * self.base_clock_ghz
            * 1e9
            / 1e12
        )

    # Peak FP32 throughput (TFLOPS) at max turbo
    @property
    def peak_fp32_tflops_turbo(self) -> float:
        return (
            self.total_physical_cores
            * self.avx512_fp32_flops_per_cycle_per_core
            * self.max_turbo_ghz
            * 1e9
            / 1e12
        )

    # L3 cache
    l3_cache_mb_per_socket: float = 320.0
    l3_cache_mb_total: float = 640.0

    # Memory
    memory_channels_per_socket: int = 8
    memory_speed_mts: int = 4800          # DDR5-4800
    memory_width_bytes_per_channel: int = 8
    @property
    def peak_memory_bandwidth_gbs(self) -> float:
        """Peak aggregate memory bandwidth (GB/s) across both sockets."""
        return (
            self.sockets
            * self.memory_channels_per_socket
            * self.memory_speed_mts
            * 1e6        # MT/s → transfers/s
            * self.memory_width_bytes_per_channel
            / 1e9        # → GB/s
        )


_DETECTED_PHYSICAL_CORES, _DETECTED_LOGICAL_THREADS = _detect_cpu_counts()
_DETECTED_SOCKETS = _detect_sockets()
_DETECTED_THREADS_PER_CORE = max(1, _DETECTED_LOGICAL_THREADS // _DETECTED_PHYSICAL_CORES)
_DETECTED_CORES_PER_SOCKET = max(1, _DETECTED_PHYSICAL_CORES // _DETECTED_SOCKETS)

CPU_SPECS = CPUSpecs(
    sockets=_DETECTED_SOCKETS,
    cores_per_socket=_DETECTED_CORES_PER_SOCKET,
    total_physical_cores=_DETECTED_PHYSICAL_CORES,
    threads_per_core=_DETECTED_THREADS_PER_CORE,
    total_logical_threads=_DETECTED_LOGICAL_THREADS,
)


# ---------------------------------------------------------------------------
# GPU — NVIDIA H100 SXM5
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class H100Specs:
    """Theoretical specifications for a single NVIDIA H100 SXM5 GPU."""

    # Compute throughput (per GPU)
    fp64_tflops: float = 34.0          # Tensor Core FP64
    fp32_tflops: float = 67.0          # non-Tensor Core FP32
    tf32_tflops: float = 989.0         # Tensor Core TF32
    fp16_tflops: float = 1979.0        # Tensor Core FP16 (dense)
    bf16_tflops: float = 1979.0        # Tensor Core BF16 (dense)
    int8_tops: float = 3958.0          # Tensor Core INT8 (dense)
    fp8_tflops: float = 3958.0         # Tensor Core FP8 (dense)

    # Memory
    hbm3_capacity_gb: float = 80.0
    hbm3_bandwidth_tbs: float = 3.35   # HBM3 bandwidth in TB/s

    # NVLink 4.0 bandwidth per GPU (bidirectional, full-mesh with 3 peers)
    # NOTE: Set to 0 when NVLink is not available (PCIe-only fabric)
    nvlink_bw_per_link_gbs: float = 50.0     # 50 GB/s per unidirectional NVLink port
    nvlink_links_per_gpu: int = 18
    has_nvlink: bool = True               # Set to False for PCIe-only systems
    
    @property
    def nvlink_total_bw_gbs(self) -> float:
        """Total unidirectional NVLink bandwidth leaving this GPU (GB/s).
        Returns 0 if NVLink is not available on this system.
        """
        if not self.has_nvlink:
            return 0.0
        return self.nvlink_links_per_gpu * self.nvlink_bw_per_link_gbs

    # PCIe 5.0 ×16 host interface
    pcie_gen: int = 5
    pcie_lanes: int = 16
    pcie_unidirectional_bw_gbs: float = 63.0    # ~63 GB/s unidirectional per ×16
    
    @property
    def interconnect_type(self) -> str:
        """Return the primary GPU-to-GPU interconnect type available."""
        if self.has_nvlink:
            return "NVLink 4.0"
        return "PCIe Gen-5 (no NVLink)"


_NVLINK_AVAILABLE = _detect_nvlink_available()

H100_SPECS = H100Specs(has_nvlink=_NVLINK_AVAILABLE)


# ---------------------------------------------------------------------------
# System-level aggregates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemSpecs:
    """Aggregate theoretical specifications for the full AI Head Node."""

    gpu_count: int = 4
    total_memory_gb: float = 0.0
    total_storage_gb: float = 0.0
    has_nvlink: bool = False              # Set to False for PCIe-only systems
    cpu: CPUSpecs = field(default_factory=CPUSpecs)
    gpu: H100Specs = field(default_factory=H100Specs)

    # --- Compute ---
    @property
    def total_fp32_tflops_gpu(self) -> float:
        return self.gpu_count * self.gpu.fp32_tflops

    @property
    def total_fp16_tflops_gpu(self) -> float:
        return self.gpu_count * self.gpu.fp16_tflops

    @property
    def total_bf16_tflops_gpu(self) -> float:
        return self.gpu_count * self.gpu.bf16_tflops

    @property
    def total_int8_tops_gpu(self) -> float:
        return self.gpu_count * self.gpu.int8_tops

    @property
    def total_fp8_tflops_gpu(self) -> float:
        return self.gpu_count * self.gpu.fp8_tflops

    # --- Memory ---
    @property
    def total_gpu_memory_gb(self) -> float:
        return self.gpu_count * self.gpu.hbm3_capacity_gb

    @property
    def total_hbm_bandwidth_tbs(self) -> float:
        return self.gpu_count * self.gpu.hbm3_bandwidth_tbs

    # --- GPU-to-GPU bandwidth (NVLink or PCIe) ---
    @property
    def nvlink_fabric_bw_gbs(self) -> float:
        """Total aggregate NVLink bandwidth across all GPUs (unidirectional).
        Returns 0 if NVLink is not available on this system.
        """
        if not self.has_nvlink:
            return 0.0
        return self.gpu_count * self.gpu.nvlink_total_bw_gbs
    
    @property
    def fabric_topology(self) -> str:
        """Return the primary GPU-to-GPU connectivity topology."""
        if self.has_nvlink:
            return f"NVLink 4.0 full-mesh ({self.gpu_count} GPUs)"
        return f"PCIe Gen-5 fabric ({self.gpu_count} GPUs, no NVLink)"

    # --- CPU↔GPU PCIe ---
    @property
    def pcie_host_bw_gbs(self) -> float:
        """Total unidirectional PCIe host-to-device bandwidth across all GPUs."""
        return self.gpu_count * self.gpu.pcie_unidirectional_bw_gbs

    # --- Roofline thresholds ---
    @property
    def arithmetic_intensity_roofline(self) -> float:
        """
        Arithmetic intensity threshold (FLOP/byte) at which FP16 GPU compute
        becomes the bottleneck rather than HBM bandwidth (roofline knee).
        """
        return (self.gpu.fp16_tflops * 1e12) / (self.gpu.hbm3_bandwidth_tbs * 1e12)


SYSTEM_SPECS = SystemSpecs(
    gpu_count=_detect_gpu_count(),
    total_memory_gb=_detect_total_ram_gb(),
    total_storage_gb=_detect_total_storage_gb(),
    has_nvlink=_NVLINK_AVAILABLE,
    cpu=CPU_SPECS,
    gpu=H100_SPECS,
)


# ---------------------------------------------------------------------------
# Acceptance thresholds (fraction of theoretical peak considered "passing")
# ---------------------------------------------------------------------------

ACCEPTANCE_THRESHOLDS: Dict[str, float] = {
    # CPU
    "cpu_avx512_fp32_efficiency": 0.70,   # ≥ 70 % of AVX-512 FP32 peak
    "cpu_memory_bandwidth_efficiency": 0.75,  # ≥ 75 % of DRAM BW peak
    "cpu_l3_bandwidth_efficiency": 0.70,
    "cpu_core_scaling_efficiency": 0.85,  # parallel speedup vs. ideal linear
    "cpu_numa_efficiency": 0.80,          # cross-NUMA vs. local BW ratio

    # GPU compute
    "gpu_fp32_efficiency": 0.70,
    "gpu_fp16_efficiency": 0.75,
    "gpu_bf16_efficiency": 0.75,
    "gpu_int8_efficiency": 0.75,
    "gpu_fp8_efficiency": 0.75,
    "gpu_tf32_efficiency": 0.75,

    # GPU memory
    "gpu_hbm_bandwidth_efficiency": 0.80,

    # Interconnect
    "nvlink_bandwidth_efficiency": 0.70,
    "pcie_h2d_bandwidth_efficiency": 0.65,
    "pcie_d2h_bandwidth_efficiency": 0.65,

    # System-level
    "multi_gpu_scaling_efficiency": 0.80,   # 4-GPU vs. 4× single-GPU throughput
    "cpu_gpu_overlap_efficiency": 0.75,     # pipeline overlap utilisation
}
