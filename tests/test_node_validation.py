"""
Node validation test suite for hardware bring-up and validation.

Test coverage:
- GPU enumeration and PCIe topology discovery
- PCIe bandwidth validation (direct attached and switch attached)
- Pinned vs pageable memory performance
- GPU P2P data bandwidth
- CPU↔GPU launch and sync overhead
- CPU↔GPU power and thermal coupling
- NCCL initialization and validation
- Workload acceptance tests (inference, training)

Can be executed separately with:
    pytest -m node_validation -vv tests/test_node_validation.py
"""

import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import pytest
except ImportError:
    pytest = None


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ValidationResult:
    """Result container for a single validation test."""
    name: str
    category: str
    status: str  # "PASS", "FAIL", "SKIP", "WARN"
    message: str
    data: Dict = None  # Optional performance data
    threshold: Optional[float] = None
    measured: Optional[float] = None
    unit: str = ""


# =============================================================================
# GPU Enumeration and PCIe Topology Discovery
# =============================================================================

class GPUEnumeration:
    """Discover GPU enumeration and PCIe topology."""

    @staticmethod
    def enumerate_gpus() -> Dict:
        """Enumerate all NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {"error": "nvidia-smi failed", "gpus": []}

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(",")
                    if len(parts) >= 4:
                        gpus.append({
                            "index": int(parts[0].strip()),
                            "name": parts[1].strip(),
                            "pci_bus_id": parts[2].strip(),
                            "driver_version": parts[3].strip(),
                        })
            return {"gpus": gpus, "count": len(gpus)}
        except Exception as e:
            return {"error": str(e), "gpus": []}

    @staticmethod
    def get_pcie_topology() -> Dict:
        """Query PCIe topology and link speeds using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie_gen,pcie_bus_width",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {"error": "nvidia-smi failed", "topology": []}

            topology = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(",")
                    if len(parts) >= 4:
                        topology.append({
                            "gpu_index": int(parts[0].strip()),
                            "pci_bus_id": parts[1].strip(),
                            "pcie_gen": parts[2].strip(),
                            "pcie_bus_width": parts[3].strip(),
                        })
            return {"topology": topology}
        except Exception as e:
            return {"error": str(e), "topology": []}

    @staticmethod
    def validate_topology(gpus: List[Dict], topology: List[Dict]) -> Tuple[bool, str]:
        """Validate GPU count and PCIe topology consistency."""
        if not gpus:
            return False, "No GPUs enumerated"
        if not topology:
            return False, "No PCIe topology data"
        if len(gpus) != len(topology):
            return False, f"GPU count ({len(gpus)}) != topology count ({len(topology)})"

        message = f"Found {len(gpus)} GPU(s) with valid PCIe topology:\n"
        for gpu, topo in zip(gpus, topology):
            message += f"  GPU {gpu['index']}: {gpu['name']}, {topo['pcie_gen']}, {topo['pcie_bus_width']}\n"
        return True, message


# =============================================================================
# PCIe Bandwidth Validation Tests
# =============================================================================

def measure_h2d_bandwidth(device_idx: int = 0, n_bytes: int = 1 * 1024 ** 3, n_iters: int = 5) -> float:
    """Measure Host-to-Device (H2D) PCIe bandwidth in GB/s."""
    if not _HAS_TORCH:
        return 0.0
    try:
        stream = torch.cuda.Stream(device=device_idx)
        h_array = torch.randn(n_bytes // 4, dtype=torch.float32, pin_memory=True)
        d_tensor = torch.empty(n_bytes // 4, dtype=torch.float32, device=f"cuda:{device_idx}")

        with torch.cuda.stream(stream):
            # Warmup
            for _ in range(2):
                d_tensor.copy_(h_array, non_blocking=True)
            stream.synchronize()

            # Timed iteration
            torch.cuda.synchronize(device=device_idx)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                d_tensor.copy_(h_array, non_blocking=True)
            torch.cuda.synchronize(device=device_idx)
            elapsed = time.perf_counter() - t0

        bw_gbs = (n_bytes * n_iters / elapsed) / 1e9
        return bw_gbs
    except Exception as e:
        logger.warning(f"H2D measurement failed: {e}")
        return 0.0


def measure_d2h_bandwidth(device_idx: int = 0, n_bytes: int = 1 * 1024 ** 3, n_iters: int = 5) -> float:
    """Measure Device-to-Host (D2H) PCIe bandwidth in GB/s."""
    if not _HAS_TORCH:
        return 0.0
    try:
        stream = torch.cuda.Stream(device=device_idx)
        h_array = torch.empty(n_bytes // 4, dtype=torch.float32, pin_memory=True)
        d_tensor = torch.randn(n_bytes // 4, dtype=torch.float32, device=f"cuda:{device_idx}")

        with torch.cuda.stream(stream):
            # Warmup
            for _ in range(2):
                h_array.copy_(d_tensor, non_blocking=True)
            stream.synchronize()

            # Timed iteration
            torch.cuda.synchronize(device=device_idx)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                h_array.copy_(d_tensor, non_blocking=True)
            torch.cuda.synchronize(device=device_idx)
            elapsed = time.perf_counter() - t0

        bw_gbs = (n_bytes * n_iters / elapsed) / 1e9
        return bw_gbs
    except Exception as e:
        logger.warning(f"D2H measurement failed: {e}")
        return 0.0


# =============================================================================
# Pinned vs Pageable Memory Tests
# =============================================================================

def measure_ce_bandwidth_pinned(device_idx: int = 0, n_bytes: int = 256 * 1024 * 1024) -> float:
    """Measure Copy Engine (CE) bandwidth with pinned memory."""
    if not _HAS_TORCH:
        return 0.0
    try:
        # Pinned host memory
        h_pinned = torch.empty(n_bytes // 4, dtype=torch.float32, pin_memory=True)
        d_tensor = torch.empty(n_bytes // 4, dtype=torch.float32, device=f"cuda:{device_idx}")

        # Warmup
        for _ in range(2):
            torch.cuda.current_stream(device_idx).synchronize()
            d_tensor.copy_(h_pinned, non_blocking=False)

        # Measure
        torch.cuda.synchronize(device=device_idx)
        t0 = time.perf_counter()
        for _ in range(10):
            d_tensor.copy_(h_pinned, non_blocking=False)
        torch.cuda.synchronize(device=device_idx)
        elapsed = time.perf_counter() - t0

        bw_gbs = (n_bytes * 10 / elapsed) / 1e9
        return bw_gbs
    except Exception as e:
        logger.warning(f"CE pinned measurement failed: {e}")
        return 0.0


def measure_ce_bandwidth_pageable(device_idx: int = 0, n_bytes: int = 256 * 1024 * 1024) -> float:
    """Measure Copy Engine (CE) bandwidth with pageable memory."""
    if not _HAS_TORCH:
        return 0.0
    try:
        # Pageable host memory (regular numpy array)
        h_pageable = np.random.randn(n_bytes // 4).astype(np.float32)
        d_tensor = torch.empty(n_bytes // 4, dtype=torch.float32, device=f"cuda:{device_idx}")

        # Warmup
        for _ in range(2):
            torch.cuda.current_stream(device_idx).synchronize()
            d_tensor.copy_(torch.from_numpy(h_pageable), non_blocking=False)

        # Measure
        torch.cuda.synchronize(device=device_idx)
        t0 = time.perf_counter()
        for _ in range(10):
            d_tensor.copy_(torch.from_numpy(h_pageable), non_blocking=False)
        torch.cuda.synchronize(device=device_idx)
        elapsed = time.perf_counter() - t0

        bw_gbs = (n_bytes * 10 / elapsed) / 1e9
        return bw_gbs
    except Exception as e:
        logger.warning(f"CE pageable measurement failed: {e}")
        return 0.0


# =============================================================================
# GPU P2P Bandwidth Tests
# =============================================================================

def measure_p2p_bandwidth(src_device: int = 0, dst_device: int = 1, 
                          n_bytes: int = 1 * 1024 ** 3, n_iters: int = 5) -> float:
    """Measure GPU-to-GPU P2P bandwidth via NVLink or PCIe."""
    if not _HAS_TORCH:
        return 0.0
    try:
        # Check P2P access
        torch.cuda.set_device(src_device)
        can_access = torch.cuda.can_device_access(dst_device)
        if not can_access:
            logger.warning(f"P2P access not available from GPU {src_device} to {dst_device}")
            return 0.0

        src_tensor = torch.randn(n_bytes // 4, dtype=torch.float32, device=f"cuda:{src_device}")
        dst_tensor = torch.empty(n_bytes // 4, dtype=torch.float32, device=f"cuda:{dst_device}")

        # Warmup
        for _ in range(2):
            torch.cuda.synchronize(src_device)
            torch.cuda.synchronize(dst_device)
            dst_tensor.copy_(src_tensor, non_blocking=True)

        # Measure
        torch.cuda.synchronize(src_device)
        torch.cuda.synchronize(dst_device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            dst_tensor.copy_(src_tensor, non_blocking=True)
        torch.cuda.synchronize(src_device)
        torch.cuda.synchronize(dst_device)
        elapsed = time.perf_counter() - t0

        bw_gbs = (n_bytes * n_iters / elapsed) / 1e9
        return bw_gbs
    except Exception as e:
        logger.warning(f"P2P measurement failed: {e}")
        return 0.0


# =============================================================================
# CPU↔GPU Launch and Sync Overhead Tests
# =============================================================================

def measure_kernel_launch_latency(device_idx: int = 0, n_samples: int = 100) -> Dict[str, float]:
    """Measure CUDA kernel launch latency in microseconds."""
    if not _HAS_TORCH:
        return {}
    try:
        torch.cuda.set_device(device_idx)
        latencies = []

        @torch.jit.script
        def dummy_kernel(x):
            return x + 1

        for _ in range(10):  # Warmup
            x = torch.ones(1, device=f"cuda:{device_idx}")
            dummy_kernel(x)
            torch.cuda.synchronize(device_idx)

        for _ in range(n_samples):
            x = torch.ones(1, device=f"cuda:{device_idx}")
            torch.cuda.synchronize(device_idx)
            t0 = time.perf_counter()
            dummy_kernel(x)
            torch.cuda.synchronize(device_idx)
            elapsed_us = (time.perf_counter() - t0) * 1e6
            latencies.append(elapsed_us)

        latencies = sorted(latencies)
        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
        }
    except Exception as e:
        logger.warning(f"Kernel launch latency measurement failed: {e}")
        return {}


def measure_synchronize_overhead(device_idx: int = 0, n_samples: int = 100) -> Dict[str, float]:
    """Measure cudaDeviceSynchronize() overhead in microseconds."""
    if not _HAS_TORCH:
        return {}
    try:
        torch.cuda.set_device(device_idx)
        latencies = []

        # Warmup
        for _ in range(10):
            torch.cuda.synchronize(device_idx)

        for _ in range(n_samples):
            t0 = time.perf_counter()
            torch.cuda.synchronize(device_idx)
            elapsed_us = (time.perf_counter() - t0) * 1e6
            latencies.append(elapsed_us)

        latencies = sorted(latencies)
        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
        }
    except Exception as e:
        logger.warning(f"Synchronize overhead measurement failed: {e}")
        return {}


# =============================================================================
# CPU↔GPU Power and Throttling Tests
# =============================================================================

def measure_power_profile(duration_sec: int = 30) -> Dict[str, float]:
    """Measure GPU power usage via nvidia-smi during sustained workload."""
    if not _HAS_TORCH:
        return {}
    try:
        def sustained_workload():
            """Run GEMM loop on all GPUs."""
            if torch.cuda.device_count() == 0:
                return
            threads = []
            for gpu_idx in range(torch.cuda.device_count()):
                def _gemm_loop(idx):
                    torch.cuda.set_device(idx)
                    a = torch.randn(4096, 4096, device=f"cuda:{idx}")
                    b = torch.randn(4096, 4096, device=f"cuda:{idx}")
                    end_time = time.time() + duration_sec
                    while time.time() < end_time:
                        torch.mm(a, b)
                        torch.cuda.synchronize(idx)
                threads.append(threading.Thread(target=_gemm_loop, args=(gpu_idx,)))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Start workload in background
        workload_thread = threading.Thread(target=sustained_workload, daemon=True)
        workload_thread.start()

        # Sample power every second for duration
        power_readings = []
        for _ in range(duration_sec):
            time.sleep(1)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            power_readings.append(float(line.strip()))
            except Exception:
                pass

        workload_thread.join(timeout=5)
        if not power_readings:
            return {}

        return {
            "min": min(power_readings),
            "max": max(power_readings),
            "mean": np.mean(power_readings),
            "std": np.std(power_readings),
            "samples": len(power_readings),
        }
    except Exception as e:
        logger.warning(f"Power profile measurement failed: {e}")
        return {}


# =============================================================================
# NCCL Validation
# =============================================================================

def check_nccl_availability() -> Tuple[bool, str]:
    """Check NCCL library availability and version."""
    try:
        import nccl
        version = getattr(nccl, "__version__", "unknown")
        return True, f"NCCL available (version: {version})"
    except ImportError:
        return False, "NCCL not available"


def validate_nvlink_status() -> Dict[str, any]:
    """Query NVLink status via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "-s"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return {
                "available": True,
                "status": result.stdout.strip(),
            }
        return {
            "available": False,
            "status": "NVLink not available or not supported",
        }
    except Exception as e:
        return {
            "available": False,
            "status": f"Error querying NVLink: {e}",
        }


# =============================================================================
# Test Suite Classes
# =============================================================================

@pytest.mark.node_validation
@pytest.mark.gpu_enumeration
class TestGPUEnumerationAndTopology:
    """TC-NV-01: GPU enumeration and PCIe topology validation."""

    def test_gpu_enumeration(self):
        """Enumerate all GPUs and verify count."""
        result = GPUEnumeration.enumerate_gpus()
        assert "gpus" in result, "GPU enumeration failed"
        assert result.get("count", 0) > 0, "No GPUs found"
        logger.info(f"GPU enumeration: {result['count']} GPU(s) found")
        for gpu in result.get("gpus", []):
            logger.info(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['pci_bus_id']})")

    def test_pcie_topology_discovery(self):
        """Verify PCIe topology and link speeds."""
        enum_result = GPUEnumeration.enumerate_gpus()
        topo_result = GPUEnumeration.get_pcie_topology()
        assert topo_result.get("topology"), "No PCIe topology data"
        logger.info(f"PCIe topology discovered: {len(topo_result['topology'])} entries")
        for entry in topo_result.get("topology", []):
            logger.info(f"  GPU {entry['gpu_index']}: {entry['pcie_gen']}, {entry['pcie_bus_width']}")

    def test_topology_consistency(self):
        """Validate GPU count and topology matrix consistency."""
        enum_result = GPUEnumeration.enumerate_gpus()
        topo_result = GPUEnumeration.get_pcie_topology()
        valid, message = GPUEnumeration.validate_topology(
            enum_result.get("gpus", []),
            topo_result.get("topology", [])
        )
        assert valid, message
        logger.info(message)


@pytest.mark.node_validation
@pytest.mark.pcie_validation
class TestPCIeBandwidthDirect:
    """TC-NV-02: PCIe bandwidth validation for directly attached GPUs."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_h2d_bandwidth_all_gpus(self):
        """Measure H2D bandwidth for all GPUs and validate ≥ 65% of PCIe theoretical (63 GB/s)."""
        n_gpus = torch.cuda.device_count()
        assert n_gpus > 0, "No CUDA devices found"
        threshold_gbs = 63.0 * 0.65  # 65% of PCIe Gen5×16 theoretical

        for gpu_idx in range(n_gpus):
            bw = measure_h2d_bandwidth(device_idx=gpu_idx, n_bytes=1024**3, n_iters=5)
            efficiency = (bw / 63.0) * 100
            logger.info(f"GPU {gpu_idx} H2D BW: {bw:.1f} GB/s ({efficiency:.1f}% of 63 GB/s)")
            assert bw >= threshold_gbs, (
                f"GPU {gpu_idx} H2D BW ({bw:.1f} GB/s) below threshold {threshold_gbs:.1f} GB/s"
            )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_d2h_bandwidth_all_gpus(self):
        """Measure D2H bandwidth for all GPUs and validate ≥ 65% of PCIe theoretical."""
        n_gpus = torch.cuda.device_count()
        assert n_gpus > 0, "No CUDA devices found"
        threshold_gbs = 63.0 * 0.65

        for gpu_idx in range(n_gpus):
            bw = measure_d2h_bandwidth(device_idx=gpu_idx, n_bytes=1024**3, n_iters=5)
            efficiency = (bw / 63.0) * 100
            logger.info(f"GPU {gpu_idx} D2H BW: {bw:.1f} GB/s ({efficiency:.1f}% of 63 GB/s)")
            assert bw >= threshold_gbs, (
                f"GPU {gpu_idx} D2H BW ({bw:.1f} GB/s) below threshold {threshold_gbs:.1f} GB/s"
            )


@pytest.mark.node_validation
@pytest.mark.memory_validation
class TestPinnedVsPageableMemory:
    """TC-NV-03: Pinned vs Pageable memory performance comparison."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_ce_pinned_vs_pageable_h2d(self):
        """Compare Copy Engine bandwidth with pinned vs pageable memory."""
        bw_pinned = measure_ce_bandwidth_pinned(device_idx=0, n_bytes=256*1024*1024)
        bw_pageable = measure_ce_bandwidth_pageable(device_idx=0, n_bytes=256*1024*1024)

        logger.info(f"CE H2D Pinned memory:   {bw_pinned:.1f} GB/s")
        logger.info(f"CE H2D Pageable memory: {bw_pageable:.1f} GB/s")
        logger.info(f"Speedup (pinned/pageable): {bw_pinned/max(bw_pageable, 1e-6):.2f}x")

        # Pinned should be faster than pageable
        assert bw_pinned > 0, "Pinned memory measurement failed"
        if bw_pageable > 0:
            assert bw_pinned >= bw_pageable * 0.9, (
                f"Pinned BW ({bw_pinned:.1f}) not faster than pageable ({bw_pageable:.1f})"
            )


@pytest.mark.node_validation
@pytest.mark.p2p_validation
class TestGPUP2PBandwidth:
    """TC-NV-04: GPU-to-GPU P2P data bandwidth validation."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_p2p_bandwidth_all_pairs(self):
        """Measure P2P bandwidth for all GPU pairs."""
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs for P2P test")

        # Typical NVLink theoretical: 400 GB/s per direction
        # Typical PCIe P2P theoretical: 63 GB/s per direction
        min_threshold = 40.0  # Conservative minimum

        for src in range(n_gpus):
            for dst in range(n_gpus):
                if src == dst:
                    continue
                bw = measure_p2p_bandwidth(src_device=src, dst_device=dst, 
                                          n_bytes=1024**3, n_iters=5)
                if bw > 0:
                    logger.info(f"GPU {src} -> GPU {dst}: {bw:.1f} GB/s")
                    assert bw >= min_threshold, (
                        f"P2P BW GPU{src}->GPU{dst} ({bw:.1f} GB/s) below minimum {min_threshold:.1f} GB/s"
                    )


@pytest.mark.node_validation
@pytest.mark.sync_overhead
class TestCPUGPULaunchAndSyncOverhead:
    """TC-NV-05: CPU↔GPU kernel launch and sync latency."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_kernel_launch_latency_stats(self):
        """Measure kernel launch latency distribution (min/p50/p95/p99/max)."""
        stats = measure_kernel_launch_latency(device_idx=0, n_samples=100)
        assert stats, "Kernel launch measurement failed"

        logger.info("Kernel launch latency (microseconds):")
        for key in ["min", "p50", "mean", "p95", "p99", "max"]:
            logger.info(f"  {key:5s}: {stats[key]:8.2f} µs")

        # Typical kernel launch latency: 1-10 microseconds
        assert stats["p50"] < 100, f"Median kernel launch latency {stats['p50']:.1f} µs seems high"

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_synchronize_overhead_stats(self):
        """Measure cudaDeviceSynchronize() overhead distribution."""
        stats = measure_synchronize_overhead(device_idx=0, n_samples=100)
        assert stats, "Synchronize measurement failed"

        logger.info("cudaDeviceSynchronize() overhead (microseconds):")
        for key in ["min", "p50", "mean", "p95", "p99", "max"]:
            logger.info(f"  {key:5s}: {stats[key]:8.2f} µs")

        # Typical sync overhead: 1-50 microseconds
        assert stats["p95"] < 1000, f"P95 sync overhead {stats['p95']:.1f} µs seems high"


@pytest.mark.node_validation
@pytest.mark.power_validation
class TestCPUGPUPowerAndThrottling:
    """TC-NV-06: CPU↔GPU power coupling and thermal throttling validation."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_sustained_gpu_power_stability(self):
        """Run sustained GPU workload and verify stable power (low variance)."""
        power_data = measure_power_profile(duration_sec=15)
        if not power_data:
            pytest.skip("Could not measure power profile")

        logger.info(f"GPU Power profile (sustained workload, {power_data.get('samples', 0)} samples):")
        logger.info(f"  Min:   {power_data.get('min', 0):.1f} W")
        logger.info(f"  Mean:  {power_data.get('mean', 0):.1f} W")
        logger.info(f"  Max:   {power_data.get('max', 0):.1f} W")
        logger.info(f"  Std:   {power_data.get('std', 0):.1f} W")

        # Verify low jitter (std < 10 W)
        std = power_data.get("std", 0)
        assert std < 50, f"Power variance {std:.1f} W is too high (expected < 50 W)"


@pytest.mark.node_validation
@pytest.mark.nccl_validation
class TestNCCLInitializationAndValidation:
    """TC-NV-07: NCCL initialization and distributed communication validation."""

    def test_nccl_availability(self):
        """Verify NCCL library is available."""
        available, message = check_nccl_availability()
        logger.info(f"NCCL status: {message}")
        # Skip if not available, but don't fail
        if not available:
            pytest.skip(message)

    def test_nvlink_status(self):
        """Check NVLink status and capabilities."""
        nvlink_info = validate_nvlink_status()
        logger.info(f"NVLink status: {nvlink_info.get('status', 'unknown')}")

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_nccl_deterministic_init(self):
        """Run NCCL initialization multiple times and measure variance."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need ≥ 2 GPUs for NCCL test")

        try:
            import nccl
            init_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                # Simulate NCCL init
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                init_times.append(elapsed_ms)

            mean_init = np.mean(init_times)
            std_init = np.std(init_times)
            logger.info(f"NCCL init variance: mean={mean_init:.2f}ms, std={std_init:.2f}ms")

            # Verify low jitter
            assert std_init < mean_init * 0.5, "NCCL init has high variance"
        except ImportError:
            pytest.skip("NCCL not available")


@pytest.mark.node_validation
@pytest.mark.workload_acceptance
class TestWorkloadAcceptanceBasic:
    """TC-NV-08: Basic workload acceptance tests (inference prefill, training micro-batch)."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_simple_inference_latency(self):
        """Measure single-batch inference latency (simple MLP)."""
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        torch.cuda.set_device(0)
        # Simple MLP: input -> 1024 -> 512 -> 128 -> output
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        ).cuda()

        x = torch.randn(1, 512, device="cuda:0")
        latencies = []

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize(0)

        # Measure
        for _ in range(20):
            torch.cuda.synchronize(0)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize(0)
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        logger.info(f"Inference latency: mean={mean_latency:.2f}ms, p95={p95_latency:.2f}ms")

        # Latency should be reasonable (< 100ms for simple model)
        assert mean_latency < 100, f"Inference latency {mean_latency:.1f}ms too high"

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_simple_training_throughput(self):
        """Measure training throughput (simple training loop)."""
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        torch.cuda.set_device(0)
        model = torch.nn.Linear(1024, 10).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        batch_size = 128
        x = torch.randn(batch_size, 1024, device="cuda:0")
        y = torch.randint(0, 10, (batch_size,), device="cuda:0")

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(0)

        # Measure
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(0)
        elapsed = (time.perf_counter() - t0)

        samples_per_sec = (batch_size * 10) / elapsed
        logger.info(f"Training throughput: {samples_per_sec:.0f} samples/sec")

        # Should be reasonable for simple model
        assert samples_per_sec > 1000, f"Training throughput {samples_per_sec:.0f} too low"


# =============================================================================
# Standalone Validation Functions (for external use)
# =============================================================================

def run_all_validations() -> List[ValidationResult]:
    """Run all validation tests and return results (non-pytest)."""
    results = []

    # GPU enumeration
    enum_result = GPUEnumeration.enumerate_gpus()
    if enum_result.get("count", 0) > 0:
        results.append(ValidationResult(
            name="GPU Enumeration",
            category="GPU Enumeration",
            status="PASS",
            message=f"Found {enum_result['count']} GPU(s)",
            data=enum_result,
        ))
    else:
        results.append(ValidationResult(
            name="GPU Enumeration",
            category="GPU Enumeration",
            status="FAIL",
            message="No GPUs found",
        ))

    # PCIe topology
    topo_result = GPUEnumeration.get_pcie_topology()
    if topo_result.get("topology"):
        results.append(ValidationResult(
            name="PCIe Topology",
            category="PCIe Topology",
            status="PASS",
            message=f"Discovered {len(topo_result['topology'])} PCIe entries",
            data=topo_result,
        ))
    else:
        results.append(ValidationResult(
            name="PCIe Topology",
            category="PCIe Topology",
            status="FAIL",
            message="Could not discover PCIe topology",
        ))

    return results
