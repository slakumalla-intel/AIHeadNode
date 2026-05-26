"""
Performance specifications for the AI Head Node platform.

Designed to be vendor-neutral and architecture-portable for competitive KPI comparison
across Intel + NVIDIA, AMD (EPYC + MI300X), and other CPU+GPU combinations.

Runtime-detected topology values (core count, memory size, storage size, GPU count)
are used wherever possible; only percentage-based efficiency thresholds are hardcoded.
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
    _HAS_TORCH = True
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
        try:
            count = int(torch.cuda.device_count())
            if count <= 0:
                return 0
            # Force lightweight init so deferred failures are handled here.
            torch.cuda.get_device_properties(0)
            return count
        except Exception:
            return 0
    return 0


def _detect_nvlink_available() -> bool:
    """Check if NVLink is active (state == 1) on at least one link of any GPU.

    Previously this function returned True whenever the NVML call succeeded,
    even on GPUs whose NVLinks were physically present but disabled by the
    platform (e.g. H100 PCIe SKUs that report 0 active links). That produced
    misleading bandwidth expectations downstream. We now check the actual link
    state and only return True when at least one link reports active.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        # Most modern NVIDIA GPUs expose up to 18 NVLinks; probe a reasonable
        # upper bound and stop at the first inactive index when an error is
        # raised (the link index is beyond the supported range for that GPU).
        max_links_to_probe = 18
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for link in range(max_links_to_probe):
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                except pynvml.NVMLError:
                    break
                if int(state) == 1:
                    return True
        return False
    except Exception:
        return False


def _detect_gpu_memory_capacity_gb() -> float:
    """Detect per-GPU HBM/VRAM capacity in GB via torch; fallback 0."""
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        except Exception:
            pass
    return 0.0


def _detect_gpu_memory_bandwidth_tbs() -> float:
    """
    Probe GPU memory bandwidth via pynvml BAR1 clock or fall back to
    a conservative estimate derived from the detected memory capacity.
    Returns value in TB/s (0.0 if unknown).
    """
    # Try pynvml for accurate peak bandwidth
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Known BW by approximate capacity (GB):
        #   H100 SXM5 80 GB → 3.35 TB/s  |  H100 PCIe 80 GB → 2.0 TB/s
        #   H100 NVL  94 GB → 3.9 TB/s
        #   A100  SXM 80 GB → 2.0 TB/s   |  MI300X 192 GB   → 5.3 TB/s
        #   A100  PCIe 40 GB→ 1.555 TB/s |  MI250X  128 GB  → 3.2 TB/s
        cap_gb = mem_info.total / 1024 ** 3
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        name_upper = name.upper()
        if "H100" in name_upper:
            if "NVL" in name_upper or cap_gb >= 93:
                return 3.9
            if "SXM" in name_upper:
                return 3.35
            return 2.0
        if "H200" in name_upper:
            return 4.8
        if "A100" in name_upper and cap_gb >= 79:
            return 2.0
        if "A100" in name_upper:
            return 1.555
        if "MI300X" in name_upper:
            return 5.3
        if "MI250X" in name_upper:
            return 3.2
        if "MI210" in name_upper:
            return 1.638
        if "RTX" in name_upper and "4090" in name_upper:
            return 1.008
        # Conservative fallback: 0.8 TB/s per 16 GB of VRAM (rough heuristic)
        return max(0.4, round(cap_gb / 16 * 0.8, 3))
    except Exception:
        pass
    return 0.0


def _detect_gpu_fp16_tflops() -> float:
    """
    Return the theoretical FP16 Tensor Core peak TFLOPS for the first GPU.
    Falls back to 0 if detection fails.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        n = name.upper()
        # Use dense Tensor Core peaks for utilization calculations.
        if "H100" in n and "SXM" in n:  return 989.0
        if "H100" in n:                  return 756.0
        if "H200" in n:                  return 989.0
        if "A100" in n and "80" in n:    return 312.0
        if "A100" in n:                  return 312.0
        if "MI300X" in n:                return 1307.0
        if "MI250X" in n:                return 383.0
        if "RTX 4090" in n:              return 165.2
    except Exception:
        pass
    return 0.0


def _detect_gpu_tdp_watts() -> float:
    """Return GPU TDP in watts via pynvml power limit query, or env override."""
    # Allow operator to override via env for non-NVML GPUs (AMD ROCm, etc.)
    env_tdp = os.environ.get("GPU_TDP_WATTS")
    if env_tdp:
        try:
            return float(env_tdp)
        except ValueError:
            pass
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        return round(limit_mw / 1000.0, 1)
    except Exception:
        pass
    return 700.0  # Safe fallback (H100 SXM5 TDP)


def _detect_gpu_max_temp_c() -> float:
    """Return safe max GPU operating temperature from pynvml or env override."""
    env_temp = os.environ.get("GPU_MAX_TEMP_C")
    if env_temp:
        try:
            return float(env_temp)
        except ValueError:
            pass
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        threshold = pynvml.nvmlDeviceGetTemperatureThreshold(
            handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
        )
        # Return 90% of shutdown threshold as operating limit
        return round(threshold * 0.90, 1)
    except Exception:
        pass
    return 85.0  # Conservative default across all AI-grade GPUs

# ---------------------------------------------------------------------------
# CPU  (vendor-neutral — all values are runtime-detected)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPUSpecs:
    """
    Host CPU specifications, fully runtime-detected.

    Works for Intel Xeon, AMD EPYC, or any other multi-core x86-64 CPU.
    SIMD width is probed at runtime; no ISA-specific names are hardcoded.
    """

    # Topology (detected)
    sockets: int = 2
    cores_per_socket: int = 80
    total_physical_cores: int = 160
    threads_per_core: int = 2
    total_logical_threads: int = 320

    # Clock (detected or conservative defaults)
    base_clock_ghz: float = 1.9
    max_turbo_ghz: float = 3.9

    # SIMD FP32 FLOPS per cycle per core.
    # Intel Xeon w/ AVX-512 FMA: 64 (2×FMA16×2)
    # AMD EPYC w/ AVX-512 FMA:   64 (same ISA)
    # Intel/AMD w/ AVX2 only:    16 (2×FMA8×1)
    # Set via CPU_SIMD_FP32_FLOPS_PER_CYCLE env var to override.
    simd_fp32_flops_per_cycle_per_core: int = 64
    simd_fp64_flops_per_cycle_per_core: int = 32

    # Peak FP32 throughput (TFLOPS) at base clock
    @property
    def peak_fp32_tflops_base(self) -> float:
        return (
            self.total_physical_cores
            * self.simd_fp32_flops_per_cycle_per_core
            * self.base_clock_ghz
            * 1e9
            / 1e12
        )

    # Peak FP32 throughput (TFLOPS) at max turbo
    @property
    def peak_fp32_tflops_turbo(self) -> float:
        return (
            self.total_physical_cores
            * self.simd_fp32_flops_per_cycle_per_core
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

# Allow operator to override SIMD width for non-AVX-512 CPUs (e.g. AVX2-only)
_SIMD_FP32_WIDTH = int(os.environ.get("CPU_SIMD_FP32_FLOPS_PER_CYCLE", "64"))

CPU_SPECS = CPUSpecs(
    sockets=_DETECTED_SOCKETS,
    cores_per_socket=_DETECTED_CORES_PER_SOCKET,
    total_physical_cores=_DETECTED_PHYSICAL_CORES,
    threads_per_core=_DETECTED_THREADS_PER_CORE,
    total_logical_threads=_DETECTED_LOGICAL_THREADS,
    simd_fp32_flops_per_cycle_per_core=_SIMD_FP32_WIDTH,
    simd_fp64_flops_per_cycle_per_core=_SIMD_FP32_WIDTH // 2,
)


# ---------------------------------------------------------------------------
# GPU — vendor-neutral, runtime-detected
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUSpecs:
    """
    Vendor-neutral GPU specifications.

    Defaults match NVIDIA H100 SXM5. Values are overridden at import time
    by runtime detection (_detect_gpu_*) so that the same test suite runs
    correctly against NVIDIA H100/A100/H200, AMD MI300X/MI250X, and future
    accelerators without code changes.

    To test against AMD or a custom GPU:
      export GPU_TDP_WATTS=750
      export GPU_MAX_TEMP_C=90
    """

    # Compute throughput (per GPU)
    fp64_tflops: float = 34.0          # Tensor Core FP64
    fp32_tflops: float = 67.0          # non-Tensor Core FP32
    tf32_tflops: float = 989.0         # Tensor Core TF32
    fp16_tflops: float = 989.0         # Tensor Core FP16 (dense)
    bf16_tflops: float = 989.0         # Tensor Core BF16 (dense)
    int8_tops: float = 3958.0          # Tensor Core INT8 (dense)
    fp8_tflops: float = 3958.0         # Tensor Core FP8 (dense)

    # Accelerator memory (HBM or GDDR)
    memory_capacity_gb: float = 80.0        # per-GPU memory capacity
    memory_bandwidth_tbs: float = 2.0      # per-GPU peak memory BW (TB/s), conservative PCIe fallback

    # NOTE: previously the fp64/fp32/tf32/fp16/bf16/int8/fp8 fields were
    # declared twice in this dataclass (once above and once here). Python
    # silently kept the second declaration, masking the duplication; the
    # canonical declarations now live above and the duplicates are removed.

    # Power and thermal
    tdp_watts: float = 700.0
    max_temp_c: float = 85.0

    # GPU-to-GPU fabric (NVLink or xGMI/Infinity Fabric for AMD)
    has_high_bw_fabric: bool = True         # False = PCIe-only
    fabric_bw_per_link_gbs: float = 50.0
    fabric_links_per_gpu: int = 18

    @property
    def fabric_total_bw_gbs(self) -> float:
        """Total unidirectional GPU-to-GPU fabric BW (NVLink / xGMI) in GB/s."""
        if not self.has_high_bw_fabric:
            return 0.0
        return self.fabric_links_per_gpu * self.fabric_bw_per_link_gbs

    # PCIe host interface (used for CPU↔GPU transfers regardless of fabric type)
    pcie_gen: int = 5
    pcie_lanes: int = 16
    pcie_unidirectional_bw_gbs: float = 63.0

    @property
    def interconnect_label(self) -> str:
        """Human-readable label for GPU-to-GPU interconnect."""
        if self.has_high_bw_fabric:
            return "NVLink / xGMI / Infinity Fabric"
        return "PCIe (no dedicated GPU fabric)"

    @property
    def interconnect_type(self) -> str:
        """Legacy alias kept for older tests and reports."""
        return self.interconnect_label

    @property
    def nvlink_total_bw_gbs(self) -> float:
        """Legacy alias for per-GPU high-bandwidth fabric total BW."""
        return self.fabric_total_bw_gbs

    # Legacy aliases kept for backwards compatibility
    @property
    def hbm3_capacity_gb(self) -> float:
        return self.memory_capacity_gb

    @property
    def hbm3_bandwidth_tbs(self) -> float:
        return self.memory_bandwidth_tbs


_NVLINK_AVAILABLE = _detect_nvlink_available()
_DETECTED_GPU_MEM_GB = _detect_gpu_memory_capacity_gb()
_DETECTED_GPU_BW_TBS = _detect_gpu_memory_bandwidth_tbs()
_DETECTED_GPU_FP16   = _detect_gpu_fp16_tflops()
_DETECTED_GPU_TDP    = _detect_gpu_tdp_watts()
_DETECTED_GPU_TEMP   = _detect_gpu_max_temp_c()

H100_SPECS = GPUSpecs(
    has_high_bw_fabric=_NVLINK_AVAILABLE,
    memory_capacity_gb=_DETECTED_GPU_MEM_GB if _DETECTED_GPU_MEM_GB > 0 else 80.0,
    memory_bandwidth_tbs=_DETECTED_GPU_BW_TBS if _DETECTED_GPU_BW_TBS > 0 else 2.0,
    fp16_tflops=(
        _DETECTED_GPU_FP16
        if _DETECTED_GPU_FP16 > 0
        else (756.0 if (_DETECTED_GPU_BW_TBS > 0 and _DETECTED_GPU_BW_TBS <= 2.1) else 989.0)
    ),
    tdp_watts=_DETECTED_GPU_TDP,
    max_temp_c=_DETECTED_GPU_TEMP,
)

# Canonical alias for new code
GPU_SPECS = H100_SPECS


# ---------------------------------------------------------------------------
# System-level aggregates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemSpecs:
    """Aggregate specifications for the full AI Head Node (vendor-neutral)."""

    gpu_count: int = 4
    total_memory_gb: float = 0.0
    total_storage_gb: float = 0.0
    has_nvlink: bool = False
    cpu: CPUSpecs = field(default_factory=CPUSpecs)
    gpu: GPUSpecs = field(default_factory=GPUSpecs)

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

    # --- GPU-to-GPU bandwidth (NVLink, xGMI/Infinity Fabric, or PCIe) ---
    @property
    def nvlink_fabric_bw_gbs(self) -> float:
        """Total aggregate GPU fabric BW across all GPUs (unidirectional, GB/s)."""
        if not self.has_nvlink:
            return 0.0
        return self.gpu_count * self.gpu.fabric_total_bw_gbs

    @property
    def fabric_topology(self) -> str:
        """Human-readable description of GPU-to-GPU connectivity."""
        if self.has_nvlink:
            return f"{self.gpu.interconnect_label} full-mesh ({self.gpu_count} GPUs)"
        return f"PCIe fabric ({self.gpu_count} GPUs, no dedicated GPU fabric)"

    # --- CPU↔GPU PCIe ---
    @property
    def pcie_host_bw_gbs(self) -> float:
        """Total unidirectional PCIe host-to-device bandwidth across all GPUs."""
        return self.gpu_count * self.gpu.pcie_unidirectional_bw_gbs

    # --- Roofline thresholds ---
    @property
    def arithmetic_intensity_roofline(self) -> float:
        """
        Arithmetic intensity (FLOP/byte) at the roofline knee: above this
        value the workload is compute-bound; below it is memory-bandwidth-bound.
        Works for any GPU with detected FP16 TFLOPS and memory bandwidth.
        """
        bw_bytes = self.gpu.memory_bandwidth_tbs * 1e12
        if bw_bytes <= 0:
            return 0.0
        return (self.gpu.fp16_tflops * 1e12) / bw_bytes


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
    # CPU — vendor-neutral, measured as % of detected theoretical peak
    "cpu_simd_fp32_efficiency": 0.70,     # ≥ 70 % of SIMD FP32 peak (AVX-512 or AVX2)
    "cpu_avx512_fp32_efficiency": 0.70,  # legacy alias — same threshold
    "cpu_memory_bandwidth_efficiency": 0.75,  # ≥ 75 % of DRAM BW peak (DDR4/DDR5/HBM)
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
    "p2p_pcie_bandwidth_efficiency": 0.50,
    "p2p_multi_stream_stability_ratio": 0.85,
    "p2p_multi_stream_directional_symmetry": 0.75,
    "pcie_h2d_bandwidth_efficiency": 0.65,
    "pcie_d2h_bandwidth_efficiency": 0.65,
    "pcie_bidir_bandwidth_efficiency": 0.70,

    # System-level
    "multi_gpu_scaling_efficiency": 0.80,   # 4-GPU vs. 4× single-GPU throughput
    "cpu_gpu_overlap_efficiency": 0.75,     # pipeline overlap utilisation
}
