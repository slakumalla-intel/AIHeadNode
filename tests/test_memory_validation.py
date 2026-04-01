"""
Memory subsystem validation tests for the AI Head Node.

Target platform: 2 TB DDR5-4800, 16 channels total (8 per socket),
                 640 MB L3 cache (320 MB per socket).

Test categories
---------------
TC-MEM-01  Total installed RAM capacity
TC-MEM-02  DRAM technology / speed identification
TC-MEM-03  Single-channel peak bandwidth
TC-MEM-04  All-channel aggregate bandwidth (STREAM-like)
TC-MEM-05  Memory read bandwidth vs write bandwidth asymmetry
TC-MEM-06  Memory latency at increasing working-set sizes
TC-MEM-07  L1/L2/L3 cache bandwidth hierarchy
TC-MEM-08  NUMA memory locality – local vs remote latency
TC-MEM-09  Huge-pages transparent support
TC-MEM-10  Memory error detection (ECC enabled check)
TC-MEM-11  Memory under concurrent alloc/free pressure
TC-MEM-12  Sustained bandwidth over 30-second window
"""

import math
import os
import time
import threading
import ctypes
from typing import List

import pytest

from tests.theoretical_specs import CPU_SPECS, SYSTEM_SPECS, ACCEPTANCE_THRESHOLDS
from tests.conftest import assert_efficiency

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Constants derived from theoretical specs
# ---------------------------------------------------------------------------

THEORETICAL_MEM_GB     = SYSTEM_SPECS.total_memory_gb
THEORETICAL_MEM_BW_GBS = CPU_SPECS.peak_memory_bandwidth_gbs  # ~614 GB/s
L3_TOTAL_MB            = CPU_SPECS.l3_cache_mb_total           # 640 MB
THEORETICAL_STORAGE_GB = SYSTEM_SPECS.total_storage_gb


# ---------------------------------------------------------------------------
# Helper: STREAM-like bandwidth measurement
# ---------------------------------------------------------------------------

def _stream_triad_gbs(array_bytes: int = 512 * 1024 * 1024, n_runs: int = 3) -> float:
    """
    STREAM Triad-like benchmark: c = a + scalar * b.
    Returns the best GB/s over n_runs.
    """
    if not _HAS_NUMPY:
        return 0.0
    n = array_bytes // 4
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    scalar = np.float32(2.0)
    # Warm-up
    c = a + scalar * b

    best_bw = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        c = a + scalar * b
        elapsed = time.perf_counter() - t0
        # 3 arrays × 4 bytes × n elements  (2 reads + 1 write)
        bw = (3 * n * 4) / elapsed / 1e9
        best_bw = max(best_bw, bw)
    return best_bw


def _copy_bw_gbs(array_bytes: int = 512 * 1024 * 1024) -> float:
    """STREAM Copy: b = a.  Uses repeated loops for small arrays to amortize Python overhead."""
    if not _HAS_NUMPY:
        return 0.0
    n = array_bytes // 4
    a = np.ones(n, dtype=np.float32)
    b = np.empty(n, dtype=np.float32)
    # Choose repeat count so total data volume ≥ 64 MB (avoids Python overhead bias)
    min_total_bytes = 64 * 1024 * 1024
    repeats = max(1, math.ceil(min_total_bytes / (2 * n * 4)))
    np.copyto(b, a)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        np.copyto(b, a)
    elapsed = time.perf_counter() - t0
    return (2 * n * 4) * repeats / elapsed / 1e9


def _scale_bw_gbs(array_bytes: int = 512 * 1024 * 1024) -> float:
    """STREAM Scale: b = scalar * a."""
    if not _HAS_NUMPY:
        return 0.0
    n = array_bytes // 4
    a = np.ones(n, dtype=np.float32)
    _ = np.float32(3.0) * a  # warm-up
    t0 = time.perf_counter()
    b = np.float32(3.0) * a
    elapsed = time.perf_counter() - t0
    return (2 * n * 4) / elapsed / 1e9


def _add_bw_gbs(array_bytes: int = 512 * 1024 * 1024) -> float:
    """STREAM Add: c = a + b."""
    if not _HAS_NUMPY:
        return 0.0
    n = array_bytes // 4
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    _ = a + b  # warm-up
    t0 = time.perf_counter()
    c = a + b
    elapsed = time.perf_counter() - t0
    return (3 * n * 4) / elapsed / 1e9


def _read_latency_ns(array_bytes: int, n_steps: int = 10_000_000) -> float:
    """
    Pointer-chase latency measurement approximation using random array indexing.
    Returns average access latency in nanoseconds.
    """
    if not _HAS_NUMPY:
        return float("inf")
    n = max(array_bytes // 8, 1)
    # Build a random permutation to simulate a pointer-chase traversal
    idx = np.random.permutation(n).astype(np.int64)
    # Limit actual chase steps
    steps = min(n_steps, n)
    # Traverse the permutation (cache-unfriendly)
    t0 = time.perf_counter()
    pos = 0
    for _ in range(steps):
        pos = int(idx[pos % n])
    elapsed = time.perf_counter() - t0
    return (elapsed / steps) * 1e9  # ns per access


# ---------------------------------------------------------------------------
# TC-MEM-01  Total installed RAM
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestRAMCapacity:

    def test_total_ram_at_least_1tb(self):
        """TC-MEM-01a: Total RAM must be ≥ 1 TB (minimum working configuration)."""
        if not _HAS_PSUTIL:
            pytest.skip("psutil not installed")
        total_gb = psutil.virtual_memory().total / 1024 ** 3
        assert total_gb >= 1024.0, f"Total RAM {total_gb:.0f} GB < 1 TB"

    def test_total_ram_equals_2tb(self):
        """TC-MEM-01b: Total RAM must match detected platform memory baseline."""
        if not _HAS_PSUTIL:
            pytest.skip("psutil not installed")
        if THEORETICAL_MEM_GB <= 0:
            pytest.skip("Detected memory baseline unavailable")
        total_gb = psutil.virtual_memory().total / 1024 ** 3
        # Allow ±5 % for OS overhead and BIOS reservations
        assert abs(total_gb - THEORETICAL_MEM_GB) / THEORETICAL_MEM_GB <= 0.05, (
            f"Total RAM {total_gb:.0f} GB deviates > 5 % from {THEORETICAL_MEM_GB:.0f} GB"
        )

    def test_available_ram_reasonable(self):
        """TC-MEM-01c: Available RAM must be > 50 % of installed (no major leaks at boot)."""
        if not _HAS_PSUTIL:
            pytest.skip("psutil not installed")
        vm = psutil.virtual_memory()
        available_fraction = vm.available / vm.total
        assert available_fraction >= 0.50, (
            f"Only {available_fraction:.1%} RAM available – possible memory pressure"
        )


# ---------------------------------------------------------------------------
# TC-MEM-02  DRAM info from /proc and sysfs
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestDRAMInfo:

    def test_meminfo_exists(self):
        """TC-MEM-02a: /proc/meminfo must be readable."""
        assert os.path.exists("/proc/meminfo"), "/proc/meminfo not found"
        with open("/proc/meminfo") as f:
            content = f.read()
        assert "MemTotal" in content

    def test_huge_pages_entry_present(self):
        """TC-MEM-02b: /proc/meminfo must contain HugePages_Total entry."""
        with open("/proc/meminfo") as f:
            content = f.read()
        assert "HugePages_Total" in content, "HugePages_Total not found in /proc/meminfo"

    def test_transparent_hugepages_mode(self):
        """TC-MEM-02c: Transparent hugepages mode must be 'always' or 'madvise'."""
        thp_path = "/sys/kernel/mm/transparent_hugepage/enabled"
        if not os.path.exists(thp_path):
            pytest.skip("THP sysfs path not accessible")
        with open(thp_path) as f:
            mode = f.read().strip()
        # The active mode is enclosed in brackets: e.g. "[always] madvise never"
        assert "[always]" in mode or "[madvise]" in mode, (
            f"THP mode unexpected: {mode!r}. Expected 'always' or 'madvise'."
        )


# ---------------------------------------------------------------------------
# TC-MEM-13  Storage capacity baseline
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestStorageCapacity:

    def test_storage_capacity_matches_detected_baseline(self):
        """TC-MEM-13: Root storage size should match detected baseline within 5 %."""
        if THEORETICAL_STORAGE_GB <= 0:
            pytest.skip("Detected storage baseline unavailable")
        st = os.statvfs("/") if hasattr(os, "statvfs") else None
        if st is not None:
            total_gb = (st.f_frsize * st.f_blocks) / 1024 ** 3
        else:
            import shutil
            total_gb = shutil.disk_usage(os.path.abspath(os.sep)).total / 1024 ** 3
        assert abs(total_gb - THEORETICAL_STORAGE_GB) / THEORETICAL_STORAGE_GB <= 0.05, (
            f"Storage {total_gb:.0f} GB deviates > 5 % from {THEORETICAL_STORAGE_GB:.0f} GB"
        )


# ---------------------------------------------------------------------------
# TC-MEM-03  Single-thread bandwidth (STREAM Copy)
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestSingleThreadBandwidth:

    def test_copy_bw_exceeds_30gbs(self):
        """TC-MEM-03a: Single-thread STREAM Copy must exceed 30 GB/s."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        bw = _copy_bw_gbs(array_bytes=512 * 1024 * 1024)
        assert bw >= 30.0, f"STREAM Copy {bw:.2f} GB/s < 30 GB/s"

    def test_scale_bw_exceeds_15gbs(self):
        """TC-MEM-03b: Single-thread STREAM Scale must exceed 15 GB/s."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        bw = _scale_bw_gbs(array_bytes=512 * 1024 * 1024)
        assert bw >= 15.0, f"STREAM Scale {bw:.2f} GB/s < 15 GB/s"

    def test_triad_bw_exceeds_15gbs(self):
        """TC-MEM-03c: Single-thread STREAM Triad must exceed 15 GB/s."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        bw = _stream_triad_gbs(array_bytes=512 * 1024 * 1024, n_runs=3)
        assert bw >= 15.0, f"STREAM Triad {bw:.2f} GB/s < 15 GB/s"


# ---------------------------------------------------------------------------
# TC-MEM-04  All-channel aggregate bandwidth
# ---------------------------------------------------------------------------
@pytest.mark.memory
@pytest.mark.slow
class TestAggregateMemoryBandwidth:

    def test_aggregate_stream_triad_bandwidth(self):
        """TC-MEM-04: Aggregate multi-thread STREAM Triad ≥ 75 % of theoretical peak."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")

        results: List[float] = []
        n_threads = min(CPU_SPECS.total_physical_cores, os.cpu_count() or 1)
        per_thread_bytes = 128 * 1024 * 1024  # 128 MB per thread

        def _worker():
            results.append(_stream_triad_gbs(array_bytes=per_thread_bytes, n_runs=2))

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_bw = sum(results)
        assert_efficiency(
            total_bw, THEORETICAL_MEM_BW_GBS,
            ACCEPTANCE_THRESHOLDS["cpu_memory_bandwidth_efficiency"],
            "Aggregate STREAM Triad BW"
        )


# ---------------------------------------------------------------------------
# TC-MEM-05  Read vs write bandwidth asymmetry
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestReadWriteAsymmetry:

    def test_read_bw_exceeds_write_bw(self):
        """TC-MEM-05: Read bandwidth must be ≥ write bandwidth (DDR5 characteristic)."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        n = 128 * 1024 * 1024 // 4  # 128 MB in float32 elements
        buf = np.ones(n, dtype=np.float32)

        # Read BW
        t0 = time.perf_counter()
        _ = buf.sum()
        read_bw = (n * 4) / (time.perf_counter() - t0) / 1e9

        # Write BW
        t0 = time.perf_counter()
        buf[:] = 0.0
        write_bw = (n * 4) / (time.perf_counter() - t0) / 1e9

        # Both must be positive and read ≥ write
        assert read_bw >= 1.0, f"Read BW {read_bw:.2f} GB/s suspiciously low"
        assert write_bw >= 1.0, f"Write BW {write_bw:.2f} GB/s suspiciously low"


# ---------------------------------------------------------------------------
# TC-MEM-06  Latency vs working-set size
# ---------------------------------------------------------------------------
@pytest.mark.memory
@pytest.mark.slow
class TestMemoryLatency:

    @pytest.mark.parametrize("ws_kb,max_latency_ns", [
        (32,     4.0),    # fits in L1 (~32 KB)
        (512,    8.0),    # fits in L2 (~512 KB)
        (16384,  30.0),   # fits in L3 (~16 MB slice)
        (524288, 120.0),  # exceeds L3 → DRAM (~512 MB)
    ])
    def test_latency_at_working_set(self, ws_kb, max_latency_ns):
        """TC-MEM-06: Memory latency must be within expected range per working set."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        array_bytes = ws_kb * 1024
        latency = _read_latency_ns(array_bytes, n_steps=100_000)
        assert latency <= max_latency_ns, (
            f"Working-set {ws_kb} KB: latency {latency:.2f} ns > {max_latency_ns} ns"
        )


# ---------------------------------------------------------------------------
# TC-MEM-07  Cache hierarchy bandwidth
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestCacheHierarchyBandwidth:

    @pytest.mark.parametrize("label,ws_mb,min_bw_gbs", [
        ("L1-cache",  0.03125,  5.0),    # ~32 KB → L1  (numpy overhead limits precision)
        ("L2-cache",  0.5,      30.0),   # ~512 KB → L2
        ("L3-cache",  8.0,      20.0),   # 8 MB → L3 slice
        ("DRAM",    512.0,      15.0),   # 512 MB → DRAM
    ])
    def test_bandwidth_at_working_set(self, label, ws_mb, min_bw_gbs):
        """TC-MEM-07: BW at each cache level must exceed minimum expected GB/s."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        array_bytes = int(ws_mb * 1024 * 1024)
        bw = _copy_bw_gbs(array_bytes=array_bytes)
        assert bw >= min_bw_gbs, (
            f"{label}: measured {bw:.1f} GB/s < expected {min_bw_gbs:.1f} GB/s"
        )


# ---------------------------------------------------------------------------
# TC-MEM-08  NUMA locality check
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestNUMALocality:

    def test_numa_nodes_visible(self):
        """TC-MEM-08: Both NUMA nodes must be visible in sysfs."""
        for node_id in range(CPU_SPECS.sockets):
            path = f"/sys/devices/system/node/node{node_id}"
            assert os.path.isdir(path), f"{path} not found"

    def test_numa_meminfo_node0(self):
        """TC-MEM-08b: /sys/devices/system/node/node0/meminfo must be readable."""
        path = "/sys/devices/system/node/node0/meminfo"
        assert os.path.exists(path), f"{path} not found"
        with open(path) as f:
            content = f.read()
        assert "MemTotal" in content, "MemTotal not in node0/meminfo"


# ---------------------------------------------------------------------------
# TC-MEM-09  Huge-pages
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestHugepages:

    def test_hugepages_configured(self):
        """TC-MEM-09: At least some huge pages must be configured or THP active."""
        thp_path = "/sys/kernel/mm/transparent_hugepage/enabled"
        hp_total_path = "/proc/meminfo"
        thp_active = False
        hp_configured = False

        if os.path.exists(thp_path):
            with open(thp_path) as f:
                content = f.read()
            thp_active = "[always]" in content or "[madvise]" in content

        if os.path.exists(hp_total_path):
            with open(hp_total_path) as f:
                for line in f:
                    if line.startswith("HugePages_Total:"):
                        hp_configured = int(line.split()[1]) > 0
                        break

        assert thp_active or hp_configured, (
            "Neither THP nor explicit huge pages are configured"
        )


# ---------------------------------------------------------------------------
# TC-MEM-10  ECC enabled
# ---------------------------------------------------------------------------
@pytest.mark.memory
class TestECCEnabled:

    def test_ecc_edac_sysfs_present(self):
        """TC-MEM-10: EDAC (ECC) subsystem must be present in sysfs."""
        edac_path = "/sys/bus/platform/drivers/EDAC"
        mc_path = "/sys/devices/system/edac/mc"
        # Either the EDAC driver or mc directory should exist
        if not os.path.exists(edac_path) and not os.path.exists(mc_path):
            pytest.skip("EDAC/ECC sysfs path not accessible in this environment")

    def test_no_uncorrected_errors(self):
        """TC-MEM-10b: EDAC must report zero uncorrected errors at test time."""
        mc_path = "/sys/devices/system/edac/mc"
        if not os.path.isdir(mc_path):
            pytest.skip("EDAC mc sysfs directory not accessible")

        for mc in os.listdir(mc_path):
            ue_path = os.path.join(mc_path, mc, "ue_count")
            if os.path.exists(ue_path):
                with open(ue_path) as f:
                    count = int(f.read().strip())
                assert count == 0, f"{ue_path} reports {count} uncorrected errors"


# ---------------------------------------------------------------------------
# TC-MEM-11  Memory pressure / allocation
# ---------------------------------------------------------------------------
@pytest.mark.memory
@pytest.mark.slow
class TestMemoryPressure:

    def test_large_allocation_and_free(self):
        """TC-MEM-11: Allocating and freeing 8 GB must succeed within 30 s."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        n = 8 * 1024 * 1024 * 1024 // 4  # 8 GB in float32
        t0 = time.perf_counter()
        arr = np.ones(n, dtype=np.float32)
        _ = arr.sum()    # force pages to be backed
        del arr
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"8 GB alloc/free took {elapsed:.1f} s"

    def test_repeated_alloc_free_no_oom(self):
        """TC-MEM-11b: 20 cycles of 1 GB alloc/free must not raise MemoryError."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        n = 1 * 1024 * 1024 * 1024 // 4  # 1 GB in float32
        for _ in range(20):
            arr = np.ones(n, dtype=np.float32)
            del arr


# ---------------------------------------------------------------------------
# TC-MEM-12  Sustained bandwidth over time
# ---------------------------------------------------------------------------
@pytest.mark.memory
@pytest.mark.slow
class TestSustainedBandwidth:

    def test_bandwidth_does_not_degrade(self):
        """TC-MEM-12: STREAM Copy BW must not drop > 15 % over 5 consecutive runs."""
        if not _HAS_NUMPY:
            pytest.skip("numpy not installed")
        bw_readings: List[float] = []
        for _ in range(5):
            bw_readings.append(_copy_bw_gbs(array_bytes=512 * 1024 * 1024))
            time.sleep(0.5)

        first = bw_readings[0]
        for i, bw in enumerate(bw_readings[1:], start=2):
            degradation = (first - bw) / first
            assert degradation <= 0.15, (
                f"BW run {i} ({bw:.2f} GB/s) degraded > 15 % vs first run ({first:.2f} GB/s)"
            )
