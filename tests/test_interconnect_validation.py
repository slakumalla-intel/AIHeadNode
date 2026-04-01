"""
CPU↔GPU and GPU↔GPU interconnect validation tests for the AI Head Node.

Test categories
---------------
TC-IC-01  PCIe topology – all 4 GPUs on PCIe 5.0 ×16
TC-IC-02  Host-to-device (H2D) bandwidth per GPU
TC-IC-03  Device-to-host (D2H) bandwidth per GPU
TC-IC-04  Bidirectional PCIe bandwidth per GPU
TC-IC-05  H2D bandwidth across all 4 GPUs simultaneously
TC-IC-06  NVLink detection and topology
TC-IC-07  GPU-to-GPU (P2P) copy bandwidth via NVLink
TC-IC-08  NVLink all-to-all bandwidth (4-GPU ring)
TC-IC-09  PCIe vs NVLink bandwidth comparison
TC-IC-10  Peer-to-peer memory access (UVA)
TC-IC-11  CPU↔GPU pipeline overlap (compute + transfer concurrency)
TC-IC-12  DMA transfer alignment and large-copy consistency
"""

import time
import threading
from typing import List, Tuple

import pytest

from tests.theoretical_specs import H100_SPECS, SYSTEM_SPECS, ACCEPTANCE_THRESHOLDS
from tests.conftest import assert_efficiency

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
THEORETICAL_H2D_BW = H100_SPECS.pcie_unidirectional_bw_gbs   # 63 GB/s
THEORETICAL_NVLINK_BW = H100_SPECS.nvlink_total_bw_gbs        # 900 GB/s per GPU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_gpu():
    if not _HAS_TORCH:
        pytest.skip("CUDA / PyTorch not available")


def _h2d_bandwidth_gbs(device_idx: int, n_bytes: int = 1 * 1024 ** 3,
                        n_runs: int = 3, pinned: bool = True) -> float:
    """
    Measure host-to-device PCIe bandwidth.
    Returns best GB/s over n_runs.
    """
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    host_tensor = (
        torch.ones(n_bytes // 4, dtype=torch.float32).pin_memory()
        if pinned
        else torch.ones(n_bytes // 4, dtype=torch.float32)
    )
    # warm-up
    dev_tensor = host_tensor.to(device, non_blocking=False)
    torch.cuda.synchronize(device)

    best_bw = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        dev_tensor = host_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        bw = n_bytes / elapsed / 1e9
        best_bw = max(best_bw, bw)
    return best_bw


def _d2h_bandwidth_gbs(device_idx: int, n_bytes: int = 1 * 1024 ** 3,
                        n_runs: int = 3, pinned: bool = True) -> float:
    """Measure device-to-host PCIe bandwidth. Returns best GB/s over n_runs."""
    if not _HAS_TORCH:
        return 0.0
    device = torch.device(f"cuda:{device_idx}")
    dev_tensor = torch.ones(n_bytes // 4, device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)

    best_bw = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        host_tensor = dev_tensor.to("cpu")
        elapsed = time.perf_counter() - t0
        bw = n_bytes / elapsed / 1e9
        best_bw = max(best_bw, bw)
    return best_bw


def _p2p_bandwidth_gbs(src_idx: int, dst_idx: int,
                        n_bytes: int = 1 * 1024 ** 3,
                        n_runs: int = 3) -> float:
    """Measure GPU-to-GPU copy bandwidth. Returns best GB/s."""
    if not _HAS_TORCH:
        return 0.0
    src = torch.device(f"cuda:{src_idx}")
    dst = torch.device(f"cuda:{dst_idx}")
    src_tensor = torch.ones(n_bytes // 4, device=src, dtype=torch.float32)
    torch.cuda.synchronize(src)

    best_bw = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        dst_tensor = src_tensor.to(dst)
        torch.cuda.synchronize(dst)
        elapsed = time.perf_counter() - t0
        bw = n_bytes / elapsed / 1e9
        best_bw = max(best_bw, bw)
    return best_bw


# ---------------------------------------------------------------------------
# TC-IC-01  PCIe topology
# ---------------------------------------------------------------------------
@pytest.mark.pcie
class TestPCIeTopology:

    def test_pcie_gen5_sysfs(self):
        """TC-IC-01: PCI root complex speed files must indicate PCIe Gen 5."""
        import glob as _glob
        speed_files = _glob.glob("/sys/bus/pci/devices/*/current_link_speed")
        if not speed_files:
            pytest.skip("PCIe sysfs speed files not accessible")
        gen5_found = False
        for sf in speed_files:
            try:
                with open(sf) as f:
                    speed = f.read().strip()
                if "32 GT/s" in speed:   # PCIe Gen 5 = 32 GT/s
                    gen5_found = True
                    break
            except OSError:
                continue
        assert gen5_found, "No PCIe Gen 5 (32 GT/s) links found in sysfs"

    def test_gpu_pcie_devices_visible(self):
        """TC-IC-01b: At least 4 NVIDIA PCI devices must be visible."""
        import glob as _glob
        # NVIDIA vendor ID 10de
        pci_devices = _glob.glob("/sys/bus/pci/devices/*/vendor")
        nvidia_count = 0
        for vf in pci_devices:
            try:
                with open(vf) as f:
                    vendor = f.read().strip()
                if vendor == "0x10de":
                    nvidia_count += 1
            except OSError:
                continue
        if nvidia_count == 0:
            pytest.skip("No NVIDIA PCI devices found in sysfs (may lack permissions)")
        assert nvidia_count >= EXPECTED_GPU_COUNT, (
            f"Only {nvidia_count} NVIDIA PCI devices, expected ≥ {EXPECTED_GPU_COUNT}"
        )


# ---------------------------------------------------------------------------
# TC-IC-02  H2D bandwidth per GPU
# ---------------------------------------------------------------------------
@pytest.mark.pcie
@pytest.mark.slow
class TestH2DBandwidth:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_h2d_bandwidth_per_gpu(self, device_idx):
        """TC-IC-02: H2D bandwidth per GPU ≥ 65 % of PCIe 5.0 ×16 theoretical (63 GB/s)."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        bw = _h2d_bandwidth_gbs(device_idx, n_bytes=1 * 1024 ** 3, n_runs=3)
        assert_efficiency(
            bw, THEORETICAL_H2D_BW,
            ACCEPTANCE_THRESHOLDS["pcie_h2d_bandwidth_efficiency"],
            f"H2D PCIe BW GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-IC-03  D2H bandwidth per GPU
# ---------------------------------------------------------------------------
@pytest.mark.pcie
@pytest.mark.slow
class TestD2HBandwidth:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_d2h_bandwidth_per_gpu(self, device_idx):
        """TC-IC-03: D2H bandwidth per GPU ≥ 65 % of theoretical (63 GB/s)."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        bw = _d2h_bandwidth_gbs(device_idx, n_bytes=1 * 1024 ** 3, n_runs=3)
        assert_efficiency(
            bw, THEORETICAL_H2D_BW,
            ACCEPTANCE_THRESHOLDS["pcie_d2h_bandwidth_efficiency"],
            f"D2H PCIe BW GPU {device_idx}"
        )


# ---------------------------------------------------------------------------
# TC-IC-04  Bidirectional PCIe bandwidth – all GPUs
# ---------------------------------------------------------------------------
@pytest.mark.pcie
@pytest.mark.slow
class TestBidirectionalPCIeBandwidth:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_bidir_pcie_bandwidth_per_gpu(self, device_idx):
        """TC-IC-04: Simultaneous H2D+D2H PCIe BW per GPU must exceed 100 GB/s total."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        results: List[float] = [0.0, 0.0]

        def _h2d():
            results[0] = _h2d_bandwidth_gbs(
                device_idx, n_bytes=512 * 1024 ** 2, n_runs=2
            )

        def _d2h():
            results[1] = _d2h_bandwidth_gbs(
                device_idx, n_bytes=512 * 1024 ** 2, n_runs=2
            )

        t0 = threading.Thread(target=_h2d)
        t1 = threading.Thread(target=_d2h)
        t0.start(); t1.start()
        t0.join(); t1.join()

        bidir_bw = results[0] + results[1]
        assert bidir_bw >= 100.0, (
            f"GPU {device_idx}: bidirectional PCIe BW {bidir_bw:.2f} GB/s < 100 GB/s"
        )


# ---------------------------------------------------------------------------
# TC-IC-05  H2D across all 4 GPUs simultaneously
# ---------------------------------------------------------------------------
@pytest.mark.pcie
@pytest.mark.slow
class TestAllGPUH2D:

    def test_concurrent_h2d_all_gpus(self):
        """TC-IC-05: Simultaneous H2D on all 4 GPUs must total ≥ 65 % of 4× PCIe BW."""
        _skip_if_no_gpu()
        n_gpus = torch.cuda.device_count()
        results: List[float] = [0.0] * n_gpus

        def _worker(idx: int):
            results[idx] = _h2d_bandwidth_gbs(idx, n_bytes=512 * 1024 ** 2, n_runs=2)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_bw = sum(results)
        theoretical = THEORETICAL_H2D_BW * n_gpus
        assert_efficiency(
            total_bw, theoretical,
            ACCEPTANCE_THRESHOLDS["pcie_h2d_bandwidth_efficiency"],
            f"Concurrent H2D BW all {n_gpus} GPUs"
        )


# ---------------------------------------------------------------------------
# TC-IC-06  NVLink detection
# ---------------------------------------------------------------------------
@pytest.mark.nvlink
class TestNVLinkDetection:

    def test_p2p_access_enabled(self):
        """TC-IC-06a: Peer-to-peer access must be enabled between adjacent GPUs."""
        _skip_if_no_gpu()
        if not SYSTEM_SPECS.has_nvlink:
            pytest.skip("NVLink not available on this system (PCIe-only fabric)")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs for P2P test")
        for src in range(n_gpus):
            for dst in range(n_gpus):
                if src != dst:
                    can_access = torch.cuda.can_device_access_peer(src, dst)
                    assert can_access, (
                        f"GPU {src} cannot access GPU {dst} via P2P"
                    )

    def test_nvlink_link_count_per_gpu(self):
        """TC-IC-06b: Each H100 must report NVLink links via NVML (if available)."""
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        if not SYSTEM_SPECS.has_nvlink:
            pytest.skip("NVLink not available on this system")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            active_links = 0
            for link_id in range(H100_SPECS.nvlink_links_per_gpu):
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(handle, link_id)
                    if state == pynvml.NVML_FEATURE_ENABLED:
                        active_links += 1
                except pynvml.NVMLError:
                    break
            assert active_links == H100_SPECS.nvlink_links_per_gpu, (
                f"GPU {i}: {active_links} NVLink links active, "
                f"expected {H100_SPECS.nvlink_links_per_gpu}"
            )


# ---------------------------------------------------------------------------
# TC-IC-07  GPU-to-GPU (P2P) bandwidth
# ---------------------------------------------------------------------------
@pytest.mark.nvlink
@pytest.mark.slow
class TestP2PBandwidth:

    @pytest.mark.parametrize("src,dst", [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    def test_p2p_bandwidth_pair(self, src, dst):
        """TC-IC-07: P2P BW between GPU pair ≥ 70 % of available fabric bandwidth."""
        _skip_if_no_gpu()
        if not SYSTEM_SPECS.has_nvlink:
            pytest.skip("NVLink not available on this system (using PCIe fabric instead)")
        n_gpus = torch.cuda.device_count()
        if src >= n_gpus or dst >= n_gpus:
            pytest.skip(f"GPU {src} or {dst} not available")
        bw = _p2p_bandwidth_gbs(src, dst, n_bytes=1 * 1024 ** 3, n_runs=3)
        assert_efficiency(
            bw, H100_SPECS.nvlink_total_bw_gbs,
            ACCEPTANCE_THRESHOLDS["nvlink_bandwidth_efficiency"],
            f"NVLink P2P BW GPU {src}→{dst}"
        )


# ---------------------------------------------------------------------------
# TC-IC-08  NVLink all-to-all bandwidth
# ---------------------------------------------------------------------------
@pytest.mark.nvlink
@pytest.mark.slow
class TestNVLinkAllToAll:

    def test_all_to_all_aggregate_bandwidth(self):
        """TC-IC-08: All-to-all P2P aggregate BW ≥ 70 % of theoretical fabric BW."""
        _skip_if_no_gpu()
        if not SYSTEM_SPECS.has_nvlink:
            pytest.skip("NVLink not available on this system")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")

        results: List[float] = []

        def _worker(src: int, dst: int):
            results.append(_p2p_bandwidth_gbs(src, dst, n_bytes=512 * 1024 ** 2, n_runs=2))

        threads = []
        for src in range(n_gpus):
            for dst in range(n_gpus):
                if src != dst:
                    threads.append(threading.Thread(target=_worker, args=(src, dst)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_bw = sum(results)
        theoretical = SYSTEM_SPECS.nvlink_fabric_bw_gbs
        assert_efficiency(
            total_bw, theoretical,
            ACCEPTANCE_THRESHOLDS["nvlink_bandwidth_efficiency"],
            "All-to-all P2P aggregate BW"
        )


# ---------------------------------------------------------------------------
# TC-IC-09  PCIe vs NVLink bandwidth comparison
# ---------------------------------------------------------------------------
@pytest.mark.nvlink
@pytest.mark.pcie
@pytest.mark.slow
class TestPCIeVsNVLink:

    def test_nvlink_faster_than_pcie(self):
        """TC-IC-09: Compare P2P fabric vs PCIe host bandwidth."""
        _skip_if_no_gpu()
        if not SYSTEM_SPECS.has_nvlink:
            pytest.skip("NVLink not available – using PCIe-only fabric")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")
        pcie_bw = _h2d_bandwidth_gbs(0, n_bytes=512 * 1024 ** 2, n_runs=2)
        nvlink_bw = _p2p_bandwidth_gbs(0, 1, n_bytes=512 * 1024 ** 2, n_runs=2)
        assert nvlink_bw >= 5 * pcie_bw, (
            f"NVLink BW {nvlink_bw:.2f} GB/s < 5× PCIe BW {pcie_bw:.2f} GB/s"
        )


# ---------------------------------------------------------------------------
# TC-IC-10  Unified Virtual Addressing (UVA) / P2P memory access – all pairs
# ---------------------------------------------------------------------------

# Build every ordered (src, dst) pair at collection time so parametrize
# only generates pairs that are logically valid (src ≠ dst).
_ALL_P2P_PAIRS: List[Tuple[int, int]] = [
    (s, d)
    for s in range(EXPECTED_GPU_COUNT)
    for d in range(EXPECTED_GPU_COUNT)
    if s != d
]


@pytest.mark.nvlink
class TestUVAMemoryAccess:

    @pytest.mark.parametrize("src_idx,dst_idx", _ALL_P2P_PAIRS)
    def test_uva_p2p_read_write(self, src_idx: int, dst_idx: int):
        """TC-IC-10: Every GPU pair must support direct UVA P2P write with data integrity."""
        _skip_if_no_gpu()
        n_gpus = torch.cuda.device_count()
        if src_idx >= n_gpus or dst_idx >= n_gpus:
            pytest.skip(f"GPU {src_idx} or {dst_idx} not available")
        if not torch.cuda.can_device_access_peer(src_idx, dst_idx):
            pytest.skip(
                f"P2P access not available between GPU {src_idx} and {dst_idx}"
            )

        src_dev = torch.device(f"cuda:{src_idx}")
        dst_dev = torch.device(f"cuda:{dst_idx}")
        n = 1024 * 1024  # 1 M float32

        src = torch.ones(n, device=src_dev, dtype=torch.float32)
        dst = torch.zeros(n, device=dst_dev, dtype=torch.float32)

        dst.copy_(src)
        torch.cuda.synchronize(dst_dev)

        result = dst.to("cpu")
        assert (result == 1.0).all(), (
            f"P2P UVA copy GPU {src_idx}→{dst_idx} produced incorrect values"
        )


# ---------------------------------------------------------------------------
# TC-IC-11  CPU↔GPU pipeline overlap – all GPUs
# ---------------------------------------------------------------------------
@pytest.mark.pcie
@pytest.mark.slow
class TestCPUGPUPipelineOverlap:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    def test_compute_transfer_overlap(self, device_idx: int):
        """TC-IC-11: Overlapping compute + H2D transfer must be faster than serial execution."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        device = torch.device(f"cuda:{device_idx}")
        m = 4096
        a = torch.randn(m, m, device=device, dtype=torch.float16)
        b = torch.randn(m, m, device=device, dtype=torch.float16)
        host_buf = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.float32).pin_memory()

        # Serial: compute then transfer
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(5):
            c = torch.mm(a, b)
            torch.cuda.synchronize(device)
        for _ in range(5):
            dev = host_buf.to(device, non_blocking=False)
            torch.cuda.synchronize(device)
        serial_time = time.perf_counter() - t0

        # Overlapped: compute on stream 0, transfer on stream 1
        compute_stream = torch.cuda.Stream(device=device)
        transfer_stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(5):
            with torch.cuda.stream(compute_stream):
                c = torch.mm(a, b)
            with torch.cuda.stream(transfer_stream):
                dev = host_buf.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        overlap_time = time.perf_counter() - t0

        assert overlap_time < serial_time, (
            f"GPU {device_idx}: overlapped ({overlap_time:.3f}s) not faster "
            f"than serial ({serial_time:.3f}s)"
        )


# ---------------------------------------------------------------------------
# TC-IC-12  DMA large-copy consistency – all GPUs
# ---------------------------------------------------------------------------
@pytest.mark.pcie
class TestDMAConsistency:

    @pytest.mark.parametrize("device_idx", list(range(EXPECTED_GPU_COUNT)))
    @pytest.mark.parametrize("transfer_mb", [64, 256, 512, 1024])
    def test_large_h2d_data_integrity(self, device_idx: int, transfer_mb: int):
        """TC-IC-12: Large H2D DMA transfers on every GPU must preserve data integrity."""
        _skip_if_no_gpu()
        if device_idx >= torch.cuda.device_count():
            pytest.skip(f"GPU {device_idx} not available")
        device = torch.device(f"cuda:{device_idx}")
        n = transfer_mb * 1024 * 1024 // 4  # float32 elements
        host = torch.arange(n, dtype=torch.float32)
        dev = host.to(device)
        result = dev.to("cpu")
        assert torch.equal(host, result), (
            f"GPU {device_idx}: {transfer_mb} MB H2D transfer produced data mismatch"
        )
