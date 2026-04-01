"""
Theoretical performance specifications for the AI Head Node platform.

Platform configuration:
  - CPU: 160 physical cores (e.g. 2× Intel Xeon Platinum 8592+ @ 1.9 GHz base / 3.9 GHz turbo)
  - RAM: 2 TB DDR5-4800 (16-channel per socket, 8 DIMMs per channel)
  - GPUs: 4× NVIDIA H100 SXM5 (80 GB HBM3 each)
  - GPU interconnect: NVLink 4.0 full-mesh between all 4 H100s
  - CPU↔GPU: PCIe 5.0 ×16 per GPU pair (via PCIe switch)

All bandwidth and throughput figures are theoretical peaks from vendor datasheets.
"""

from dataclasses import dataclass, field
from typing import Dict

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


CPU_SPECS = CPUSpecs()


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
    nvlink_bw_per_link_gbs: float = 50.0     # 50 GB/s per unidirectional NVLink port
    nvlink_links_per_gpu: int = 18
    @property
    def nvlink_total_bw_gbs(self) -> float:
        """Total unidirectional NVLink bandwidth leaving this GPU (GB/s)."""
        return self.nvlink_links_per_gpu * self.nvlink_bw_per_link_gbs

    # PCIe 5.0 ×16 host interface
    pcie_gen: int = 5
    pcie_lanes: int = 16
    pcie_unidirectional_bw_gbs: float = 63.0    # ~63 GB/s unidirectional per ×16


H100_SPECS = H100Specs()


# ---------------------------------------------------------------------------
# System-level aggregates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemSpecs:
    """Aggregate theoretical specifications for the full AI Head Node."""

    gpu_count: int = 4
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

    # --- GPU-to-GPU bandwidth (NVLink, full-mesh) ---
    @property
    def nvlink_fabric_bw_gbs(self) -> float:
        """Total aggregate NVLink bandwidth across all GPUs (unidirectional)."""
        return self.gpu_count * self.gpu.nvlink_total_bw_gbs

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


SYSTEM_SPECS = SystemSpecs()


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
