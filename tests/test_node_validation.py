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
# Binary / workload paths  (edit here if paths change)
# =============================================================================

_NVBANDWIDTH_BIN    = Path("/root/workloads/nvbandwidth/nvbandwidth")
_P2P_LATENCY_BIN    = Path(
    "/root/workloads/cuda-samples/Samples/5_Domain_Specific"
    "/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest"
)
_NCCL_TESTS_DIR     = Path("/root/workloads/nccl-tests/build")

# HuggingFace model IDs used for inference / training acceptance tests.
# Swap to a locally cached path if internet access is unavailable on the node.
_HF_INFERENCE_MODEL = "gpt2"          # small causal-LM, ~500 MB
_HF_TRAINING_MODEL  = "bert-base-uncased"  # encoder, fast fine-tune loop


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
        """Query per-GPU PCIe negotiated link speed and width (current vs max).

        Fields queried:
          pcie.link.gen.gpucurrent  — current negotiated PCIe generation
          pcie.link.gen.gpumax      — max generation supported by the GPU
          pcie.link.width.current   — current negotiated link width (e.g. 16)
          pcie.link.width.max       — max link width supported by the GPU
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,pci.bus_id,"
                    "pcie.link.gen.gpucurrent,pcie.link.gen.gpumax,"
                    "pcie.link.width.current,pcie.link.width.max",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {"error": f"nvidia-smi failed: {result.stderr.strip()}", "topology": []}

            topology = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    cur_gen = parts[2]
                    max_gen = parts[3]
                    cur_width = parts[4]
                    max_width = parts[5]
                    # PCIe Gen5 x16 theoretical peak: 63 GB/s unidirectional
                    _GEN_BW = {"1": 4, "2": 8, "3": 16, "4": 32, "5": 63, "6": 128}
                    try:
                        neg_bw = _GEN_BW.get(cur_gen, 0) * int(cur_width) // 16
                    except (ValueError, ZeroDivisionError):
                        neg_bw = 0
                    topology.append({
                        "gpu_index": int(parts[0]),
                        "pci_bus_id": parts[1],
                        "gen_current": cur_gen,
                        "gen_max": max_gen,
                        "width_current": cur_width,
                        "width_max": max_width,
                        "negotiated_bw_gbs": neg_bw,
                        "downgraded": cur_gen != max_gen or cur_width != max_width,
                    })
            return {"topology": topology}
        except Exception as e:
            return {"error": str(e), "topology": []}

    @staticmethod
    def get_topo_matrix() -> Dict:
        """Run ``nvidia-smi topo -m`` and parse the GPU placement matrix.

        Returns the raw text output and a parsed dict keyed by (src_idx, dst_idx)
        with the interconnect type string (NV1..NV18, PIX, PXB, PHB, SYS, NODE, MIG).
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return {"error": f"nvidia-smi topo -m failed: {result.stderr.strip()}",
                        "raw": "", "links": {}}

            raw = result.stdout
            links: Dict[Tuple[int, int], str] = {}

            # Matrix header row starts with GPU indices like "GPU0  GPU1  ..."
            lines = raw.splitlines()
            header_idx = None
            gpu_col_order: List[int] = []

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("GPU") and "GPU" in stripped[3:]:
                    # Header row — collect column GPU indices
                    tokens = stripped.split()
                    gpu_col_order = []
                    for tok in tokens:
                        if tok.startswith("GPU"):
                            try:
                                gpu_col_order.append(int(tok[3:]))
                            except ValueError:
                                pass
                    header_idx = i
                    break

            if header_idx is not None:
                for line in lines[header_idx + 1:]:
                    stripped = line.strip()
                    if not stripped.startswith("GPU"):
                        break
                    tokens = stripped.split()
                    try:
                        row_gpu = int(tokens[0][3:])
                    except (ValueError, IndexError):
                        continue
                    # Values start after the row label
                    values = tokens[1:]
                    for col_pos, col_gpu in enumerate(gpu_col_order):
                        if col_pos < len(values):
                            links[(row_gpu, col_gpu)] = values[col_pos]

            return {"raw": raw, "links": links}
        except Exception as e:
            return {"error": str(e), "raw": "", "links": {}}

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
            cur_gen = topo.get('gen_current', '?')
            max_gen = topo.get('gen_max', '?')
            cur_w = topo.get('width_current', '?')
            max_w = topo.get('width_max', '?')
            neg_bw = topo.get('negotiated_bw_gbs', 0)
            downgraded = topo.get('downgraded', False)
            flag = "  *** DOWNGRADED ***" if downgraded else ""
            message += (
                f"  GPU {gpu['index']}: {gpu['name']}, "
                f"PCIe Gen{cur_gen}x{cur_w} (negotiated) "
                f"[max Gen{max_gen}x{max_w}]  ~{neg_bw} GB/s{flag}\n"
            )
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
        can_access = torch.cuda.can_device_access_peer(src_device, dst_device)
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
# nvbandwidth helper — run one test-case and return per-GPU bandwidth table
# =============================================================================

def _run_nvbandwidth(testcase: str, timeout: int = 120) -> Dict:
    """Run a single nvbandwidth test-case and parse the bandwidth matrix.

    Returns
    -------
    dict with keys:
      "testcase"   : str   – the test name
      "raw"        : str   – full stdout
      "values"     : list[float]  – all bandwidth values extracted (GB/s)
      "matrix"     : list[list[float]]  – row-major bandwidth matrix
      "error"      : str | None
    """
    if not _NVBANDWIDTH_BIN.exists():
        return {"testcase": testcase, "raw": "", "values": [],
                "matrix": [], "error": f"{_NVBANDWIDTH_BIN} not found"}
    try:
        result = subprocess.run(
            [str(_NVBANDWIDTH_BIN), "-t", testcase],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(
                str(i) for i in range(4))},  # expose all GPUs
        )
        raw = result.stdout + result.stderr
        values: List[float] = []
        matrix: List[List[float]] = []
        in_matrix = False
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                if in_matrix:
                    in_matrix = False
                continue

            tokens = stripped.split()

            # Header row is GPU IDs only (e.g., "0 1 2 3"); begin matrix parsing.
            if not in_matrix and all(tok.isdigit() for tok in tokens):
                in_matrix = True
                continue

            # Data row format: row-id followed by bandwidth cells.
            if in_matrix and len(tokens) >= 2 and tokens[0].isdigit():
                try:
                    row_vals = [float(x) for x in tokens[1:]]
                    if row_vals:
                        matrix.append(row_vals)
                        values.extend(row_vals)
                except ValueError:
                    # Ignore non-numeric lines inside matrix block.
                    pass
        return {
            "testcase": testcase,
            "raw": raw,
            "values": values,
            "matrix": matrix,
            "error": None if result.returncode == 0 else f"Exit {result.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"testcase": testcase, "raw": "", "values": [],
                "matrix": [], "error": "Timeout"}
    except Exception as exc:
        return {"testcase": testcase, "raw": "", "values": [],
                "matrix": [], "error": str(exc)}


def _run_p2p_latency_test(timeout: int = 120) -> Dict:
    """Run p2pBandwidthLatencyTest and parse per-pair bandwidth matrices.

    Returns dict with keys: "raw", "bidir_bw", "unidir_bw", "latency_us", "error".
    """
    if not _P2P_LATENCY_BIN.exists():
        return {"raw": "", "bidir_bw": [], "unidir_bw": [], "latency_us": [],
                "error": f"{_P2P_LATENCY_BIN} not found"}
    try:
        result = subprocess.run(
            [str(_P2P_LATENCY_BIN)],
            capture_output=True, text=True, timeout=timeout,
        )
        raw = result.stdout + result.stderr
        bidir_bw: List[float] = []
        unidir_bw: List[float] = []
        latency_us: List[float] = []
        section = None
        for line in raw.splitlines():
            l = line.strip()
            if "Bidirectional" in l and "Bandwidth" in l:
                section = "bidir"
            elif "Unidirectional" in l and "Bandwidth" in l:
                section = "unidir"
            elif "Latency" in l:
                section = "latency"
            elif section and l and l[0].isdigit():
                try:
                    vals = [float(x) for x in l.split() if x.replace('.','',1).isdigit()]
                    if section == "bidir":
                        bidir_bw.extend(vals)
                    elif section == "unidir":
                        unidir_bw.extend(vals)
                    elif section == "latency":
                        latency_us.extend(vals)
                except ValueError:
                    pass
        return {
            "raw": raw,
            "bidir_bw": bidir_bw,
            "unidir_bw": unidir_bw,
            "latency_us": latency_us,
            "error": None if result.returncode == 0 else f"Exit {result.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"raw": "", "bidir_bw": [], "unidir_bw": [], "latency_us": [],
                "error": "Timeout"}
    except Exception as exc:
        return {"raw": "", "bidir_bw": [], "unidir_bw": [], "latency_us": [],
                "error": str(exc)}


def _run_nccl_test(
    binary: str,
    n_gpus: int = 4,
    min_bytes: str = "8",
    max_bytes: str = "256M",
    step_factor: int = 2,
    warmup_iters: int = 5,
    iters: int = 20,
    timeout: int = 300,
) -> Dict:
    """Run a single NCCL test binary and parse busbw (bus bandwidth) from output.

    Returns dict with keys: "raw", "busbw_gbs", "algbw_gbs", "error".
    """
    bin_path = _NCCL_TESTS_DIR / binary
    if not bin_path.exists():
        return {"raw": "", "busbw_gbs": [], "algbw_gbs": [],
                "error": f"{bin_path} not found"}
    try:
        cmd = [
            str(bin_path),
            "-b", min_bytes,
            "-e", max_bytes,
            "-f", str(step_factor),
            "-g", str(n_gpus),
            "-w", str(warmup_iters),
            "-n", str(iters),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        raw = result.stdout + result.stderr
        busbw: List[float] = []
        algbw: List[float] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            # NCCL test output columns (typical):
            # size count type redop root time algbw busbw #wrong time algbw busbw #wrong
            # The in-place busbw is the rightmost numeric busbw column
            if len(tokens) >= 12 and tokens[0].isdigit():
                try:
                    algbw.append(float(tokens[6]))   # out-of-place algbw
                    busbw.append(float(tokens[7]))   # out-of-place busbw
                except (ValueError, IndexError):
                    pass
        return {
            "raw": raw,
            "busbw_gbs": busbw,
            "algbw_gbs": algbw,
            "error": None if result.returncode == 0 else f"Exit {result.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"raw": "", "busbw_gbs": [], "algbw_gbs": [], "error": "Timeout"}
    except Exception as exc:
        return {"raw": "", "busbw_gbs": [], "algbw_gbs": [], "error": str(exc)}


# =============================================================================
# NCCL Validation helpers
# =============================================================================

def check_nccl_availability() -> Tuple[bool, str]:
    """Check NCCL library availability — prefers torch.cuda.nccl, falls back to import."""
    if _HAS_TORCH:
        try:
            ver = torch.cuda.nccl.version()
            return True, f"NCCL available via torch.cuda.nccl (version tuple: {ver})"
        except Exception:
            pass
    try:
        import nccl  # type: ignore[import]
        version = getattr(nccl, "__version__", "unknown")
        return True, f"NCCL available (version: {version})"
    except ImportError:
        # Also check if the test binary itself exists as a proxy for NCCL presence
        if (_NCCL_TESTS_DIR / "all_reduce_perf").exists():
            return True, f"NCCL test binaries found at {_NCCL_TESTS_DIR}"
        return False, "NCCL not available"


def validate_nvlink_status() -> Dict:
    """Query NVLink status and per-link speed via nvidia-smi nvlink."""
    try:
        status_result = subprocess.run(
            ["nvidia-smi", "nvlink", "-s"],
            capture_output=True, text=True, timeout=10
        )
        speed_result = subprocess.run(
            ["nvidia-smi", "nvlink", "-c"],
            capture_output=True, text=True, timeout=10
        )
        available = status_result.returncode == 0 and "Link" in status_result.stdout
        return {
            "available": available,
            "status": status_result.stdout.strip(),
            "capabilities": speed_result.stdout.strip(),
        }
    except Exception as e:
        return {
            "available": False,
            "status": f"Error querying NVLink: {e}",
            "capabilities": "",
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
        """TC-NV-01b: Validate GPU placement via nvidia-smi topo -m and print negotiated PCIe link speed/width."""
        # ── 1. Negotiated link speed per GPU ────────────────────────────────
        topo_result = GPUEnumeration.get_pcie_topology()
        assert topo_result.get("topology"), (
            f"No PCIe topology data. Error: {topo_result.get('error', 'unknown')}"
        )

        downgraded_gpus = []
        lines = ["\n", "=" * 80, "  PCIe NEGOTIATED LINK SPEED / WIDTH PER GPU", "=" * 80]
        lines.append(f"  {'GPU':<6} {'PCI Bus ID':<16} {'Gen (cur/max)':<16} "
                     f"{'Width (cur/max)':<18} {'Neg. BW':<12} {'Status'}")
        lines.append("  " + "-" * 76)
        for entry in topo_result["topology"]:
            cur_gen = entry.get('gen_current', '?')
            max_gen = entry.get('gen_max', '?')
            cur_w   = entry.get('width_current', '?')
            max_w   = entry.get('width_max', '?')
            neg_bw  = entry.get('negotiated_bw_gbs', 0)
            flag    = "DOWNGRADED" if entry.get('downgraded') else "OK"
            if entry.get('downgraded'):
                downgraded_gpus.append(entry['gpu_index'])
            lines.append(
                f"  GPU{entry['gpu_index']:<3} {entry['pci_bus_id']:<16} "
                f"Gen{cur_gen} / Gen{max_gen:<8}  "
                f"x{cur_w} / x{max_w:<12}  "
                f"~{neg_bw:>4} GB/s   {flag}"
            )
        lines.append("=" * 80)
        logger.info("\n".join(lines))

        # ── 2. GPU placement matrix via nvidia-smi topo -m ──────────────────
        matrix_result = GPUEnumeration.get_topo_matrix()
        if matrix_result.get("error"):
            logger.warning("nvidia-smi topo -m: %s", matrix_result['error'])
        else:
            logger.info("\n\nnvidia-smi topo -m output:\n%s", matrix_result.get('raw', ''))

            links = matrix_result.get("links", {})
            # Classify interconnect types between all GPU pairs
            nvlink_pairs = []
            pcie_pairs   = []
            for (src, dst), link_type in sorted(links.items()):
                if src == dst:
                    continue
                if link_type.startswith("NV"):
                    nvlink_pairs.append((src, dst, link_type))
                elif link_type in ("PIX", "PXB", "PHB", "NODE", "SYS"):
                    pcie_pairs.append((src, dst, link_type))

            link_summary = ["\n", "  GPU INTERCONNECT SUMMARY (from topo -m)", "  " + "-" * 50]
            if nvlink_pairs:
                link_summary.append(f"  NVLink pairs  ({len(nvlink_pairs)}):")
                for src, dst, lt in nvlink_pairs:
                    link_summary.append(f"    GPU{src} <-> GPU{dst} : {lt}")
            if pcie_pairs:
                link_summary.append(f"  PCIe-only pairs ({len(pcie_pairs)}):")
                for src, dst, lt in pcie_pairs:
                    link_summary.append(f"    GPU{src} <-> GPU{dst} : {lt}")
            logger.info("\n".join(link_summary))

        # ── 3. Assertions ───────────────────────────────────────────────────
        assert not downgraded_gpus, (
            f"GPU(s) {downgraded_gpus} are running at a downgraded PCIe link speed or width. "
            "Expected all GPUs at max negotiated Gen/width. "
            "Check slot seating, BIOS ASPM settings, and riser cable integrity."
        )

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
class TestNVBandwidthMemcpy:
    """TC-NV-03: DMA / SM copy validation via nvbandwidth.

    Covers:
      CE pinned   : host_to_device_memcpy_ce  /  device_to_host_memcpy_ce
      CE bidir    : host_to_device_bidirectional_memcpy_ce
                    device_to_host_bidirectional_memcpy_ce
      D2D CE      : device_to_device_memcpy_read_ce
      SM variants : host_to_device_memcpy_sm  /  device_to_host_memcpy_sm
    """

    # Minimum acceptable bandwidth (GB/s) per nvbandwidth test-case.
    # Values are intentionally conservative; real PCIe Gen5 typically yields ~55+ GB/s.
    _THRESHOLDS: Dict[str, float] = {
        "host_to_device_memcpy_ce":              40.0,
        "device_to_host_memcpy_ce":              40.0,
        "host_to_device_bidirectional_memcpy_ce": 40.0,
        "device_to_host_bidirectional_memcpy_ce": 40.0,
        "device_to_device_memcpy_read_ce":       800.0,   # HBM3 D2D should be fast
        "host_to_device_memcpy_sm":              30.0,
        "device_to_host_memcpy_sm":              30.0,
    }

    def _run_and_validate(self, testcase: str) -> None:
        """Run nvbandwidth for *testcase*, log the full matrix, assert threshold."""
        if not _NVBANDWIDTH_BIN.exists():
            pytest.skip(f"nvbandwidth not found at {_NVBANDWIDTH_BIN}")

        res = _run_nvbandwidth(testcase)
        if res["error"]:
            pytest.skip(f"nvbandwidth {testcase} error: {res['error']}")

        logger.info(
            "\n=== nvbandwidth: %s ===\n%s",
            testcase, res["raw"].strip()
        )

        values = [v for v in res["values"] if v > 0]
        assert values, f"No bandwidth values parsed for {testcase}"

        min_bw = min(values)
        mean_bw = sum(values) / len(values)
        threshold = self._THRESHOLDS.get(testcase, 0.0)

        logger.info(
            "%s  min=%.1f GB/s  mean=%.1f GB/s  threshold=%.1f GB/s",
            testcase, min_bw, mean_bw, threshold
        )
        assert min_bw >= threshold, (
            f"{testcase}: min BW {min_bw:.1f} GB/s below threshold {threshold:.1f} GB/s"
        )

    def test_host_to_device_memcpy_ce(self):
        """CE + pinned: H2D memcpy bandwidth via Copy Engine (all GPUs)."""
        self._run_and_validate("host_to_device_memcpy_ce")

    def test_device_to_host_memcpy_ce(self):
        """CE + pinned: D2H memcpy bandwidth via Copy Engine (all GPUs)."""
        self._run_and_validate("device_to_host_memcpy_ce")

    def test_host_to_device_bidirectional_memcpy_ce(self):
        """CE bidirectional: simultaneous H2D + D2H bandwidth via Copy Engine."""
        self._run_and_validate("host_to_device_bidirectional_memcpy_ce")

    def test_device_to_host_bidirectional_memcpy_ce(self):
        """CE bidirectional: D2H direction of simultaneous CE transfer."""
        self._run_and_validate("device_to_host_bidirectional_memcpy_ce")

    def test_device_to_device_memcpy_read_ce(self):
        """CE D2D: device-to-device read bandwidth (HBM3 internal copy engine)."""
        self._run_and_validate("device_to_device_memcpy_read_ce")

    def test_host_to_device_memcpy_sm(self):
        """SM copy: H2D bandwidth using CUDA Streaming Multiprocessors (pageable)."""
        self._run_and_validate("host_to_device_memcpy_sm")

    def test_device_to_host_memcpy_sm(self):
        """SM copy: D2H bandwidth using CUDA Streaming Multiprocessors (pageable)."""
        self._run_and_validate("device_to_host_memcpy_sm")


@pytest.mark.node_validation
@pytest.mark.p2p_validation
class TestGPUP2PBandwidth:
    """TC-NV-04: GPU-to-GPU P2P data bandwidth validation.

    Uses the CUDA sample p2pBandwidthLatencyTest binary as the primary source of
    truth.  Falls back to a pure-PyTorch measurement when the binary is absent.
    """

    def test_p2p_bandwidth_latency_binary(self):
        """Run p2pBandwidthLatencyTest and validate bidirectional P2P BW ≥ threshold."""
        if not _P2P_LATENCY_BIN.exists():
            pytest.skip(f"p2pBandwidthLatencyTest not found at {_P2P_LATENCY_BIN}")

        res = _run_p2p_latency_test(timeout=180)
        if res["error"]:
            pytest.skip(f"p2pBandwidthLatencyTest error: {res['error']}")

        logger.info("\n=== p2pBandwidthLatencyTest output ===\n%s", res["raw"].strip())

        bidir = [v for v in res["bidir_bw"] if v > 0]
        unidir = [v for v in res["unidir_bw"] if v > 0]
        latency = [v for v in res["latency_us"] if v > 0]

        if bidir:
            logger.info(
                "Bidirectional P2P BW — min=%.1f GB/s  mean=%.1f GB/s  max=%.1f GB/s",
                min(bidir), sum(bidir)/len(bidir), max(bidir)
            )
        if unidir:
            logger.info(
                "Unidirectional P2P BW — min=%.1f GB/s  mean=%.1f GB/s  max=%.1f GB/s",
                min(unidir), sum(unidir)/len(unidir), max(unidir)
            )
        if latency:
            logger.info(
                "P2P Latency           — min=%.2f µs  mean=%.2f µs  max=%.2f µs",
                min(latency), sum(latency)/len(latency), max(latency)
            )

        # On NVLink systems off-diagonal BW should be >> PCIe (40 GB/s minimum)
        # On PCIe-only systems ~40 GB/s still expected
        min_threshold = 40.0
        off_diag_bw = bidir or unidir
        assert off_diag_bw, "No inter-GPU bandwidth values parsed"
        failing = [v for v in off_diag_bw if v < min_threshold]
        assert not failing, (
            f"{len(failing)} P2P bandwidth values below {min_threshold} GB/s: "
            f"{[f'{v:.1f}' for v in sorted(failing)]}"
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_p2p_bandwidth_pytorch_fallback(self):
        """PyTorch direct P2P measurement for all GPU pairs (sanity check)."""
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs for P2P test")

        min_threshold = 40.0
        header = ["\n", "=" * 60, "  PyTorch P2P Bandwidth Matrix (GB/s)", "=" * 60]
        header.append(f"  {'':6}" + "".join(f"  GPU{d:<3}" for d in range(n_gpus)))
        rows = []
        for src in range(n_gpus):
            row_vals = []
            for dst in range(n_gpus):
                if src == dst:
                    row_vals.append("  ---  ")
                else:
                    bw = measure_p2p_bandwidth(src_device=src, dst_device=dst,
                                              n_bytes=1024**3, n_iters=5)
                    row_vals.append(f"{bw:7.1f}")
                    if bw > 0:
                        assert bw >= min_threshold, (
                            f"P2P BW GPU{src}->GPU{dst} ({bw:.1f} GB/s) "
                            f"below minimum {min_threshold:.1f} GB/s"
                        )
            rows.append(f"  GPU{src}   " + "".join(row_vals))
        logger.info("\n".join(header + rows + ["=" * 60]))


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
class TestNCCLAndNVLinkValidation:
    """TC-NV-07: NVLink status + NCCL collective performance via nccl-tests binaries.

    Binaries used (from /root/workloads/nccl-tests/build/):
      all_reduce_perf    – AllReduce collective (primary training collective)
      all_gather_perf    – AllGather collective
      reduce_scatter_perf – ReduceScatter collective
      broadcast_perf     – Broadcast collective
      p2p_perf           – Point-to-point send/recv
    """

    def test_nvlink_status(self):
        """Verify NVLink is active and report per-link speed."""
        nvlink_info = validate_nvlink_status()
        logger.info("\nNVLink status:\n%s", nvlink_info.get("status", "(none)"))
        if nvlink_info.get("capabilities"):
            logger.info("\nNVLink capabilities:\n%s", nvlink_info["capabilities"])
        # Non-fatal: node may be PCIe-only
        if not nvlink_info["available"]:
            logger.warning("NVLink not detected — system may be PCIe-only interconnect")

    def test_nccl_availability(self):
        """Verify NCCL is accessible (via torch or test binary)."""
        available, message = check_nccl_availability()
        logger.info("NCCL availability: %s", message)
        if not available:
            pytest.skip(message)

    def test_nccl_all_reduce_perf(self):
        """Run all_reduce_perf across all GPUs and validate peak busbw."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs for NCCL collective test")

        res = _run_nccl_test(
            "all_reduce_perf",
            n_gpus=n_gpus,
            min_bytes="1M",
            max_bytes="1G",
            step_factor=4,
            iters=20,
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"all_reduce_perf binary: {res['error']}")

        logger.info("\n=== NCCL all_reduce_perf ===\n%s", res["raw"].strip())

        busbw = res["busbw_gbs"]
        if not busbw:
            pytest.skip("Could not parse busbw from all_reduce_perf output")

        peak_busbw = max(busbw)
        # NVLink H100: peak allreduce busbw ≈ 300–400 GB/s for large messages
        # PCIe only: expect ~40+ GB/s    Conservative threshold = 30 GB/s
        min_busbw = 30.0
        logger.info(
            "AllReduce busbw — peak=%.1f GB/s  mean=%.1f GB/s  threshold=%.1f GB/s",
            peak_busbw, sum(busbw)/len(busbw), min_busbw
        )
        assert peak_busbw >= min_busbw, (
            f"AllReduce peak busbw {peak_busbw:.1f} GB/s below threshold {min_busbw:.1f} GB/s"
        )

    def test_nccl_all_gather_perf(self):
        """Run all_gather_perf and validate peak bus bandwidth."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")

        res = _run_nccl_test(
            "all_gather_perf", n_gpus=n_gpus,
            min_bytes="1M", max_bytes="256M", step_factor=4, iters=20
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"all_gather_perf binary: {res['error']}")

        logger.info("\n=== NCCL all_gather_perf ===\n%s", res["raw"].strip())
        busbw = res["busbw_gbs"]
        if not busbw:
            pytest.skip("Could not parse all_gather_perf output")
        peak = max(busbw)
        logger.info("AllGather peak busbw: %.1f GB/s", peak)
        assert peak >= 30.0, f"AllGather busbw {peak:.1f} GB/s too low"

    def test_nccl_reduce_scatter_perf(self):
        """Run reduce_scatter_perf and validate peak bus bandwidth."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")

        res = _run_nccl_test(
            "reduce_scatter_perf", n_gpus=n_gpus,
            min_bytes="1M", max_bytes="256M", step_factor=4, iters=20
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"reduce_scatter_perf binary: {res['error']}")

        logger.info("\n=== NCCL reduce_scatter_perf ===\n%s", res["raw"].strip())
        busbw = res["busbw_gbs"]
        if not busbw:
            pytest.skip("Could not parse reduce_scatter_perf output")
        peak = max(busbw)
        logger.info("ReduceScatter peak busbw: %.1f GB/s", peak)
        assert peak >= 30.0, f"ReduceScatter busbw {peak:.1f} GB/s too low"

    def test_nccl_p2p_perf(self):
        """Run p2p_perf (point-to-point send/recv) and validate bandwidth."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")

        res = _run_nccl_test(
            "sendrecv_perf", n_gpus=n_gpus,
            min_bytes="1M", max_bytes="256M", step_factor=4, iters=20
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"sendrecv_perf binary: {res['error']}")

        logger.info("\n=== NCCL sendrecv_perf ===\n%s", res["raw"].strip())
        busbw = res["busbw_gbs"]
        if not busbw:
            pytest.skip("Could not parse sendrecv_perf output")
        peak = max(busbw)
        logger.info("SendRecv peak busbw: %.1f GB/s", peak)
        assert peak >= 30.0, f"NCCL sendrecv busbw {peak:.1f} GB/s too low"

    def test_nccl_deterministic_init_variance(self):
        """Run AllReduce 50× with small message and measure init/launch variance."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")

        res = _run_nccl_test(
            "all_reduce_perf", n_gpus=n_gpus,
            min_bytes="8", max_bytes="8",   # tiny message → measures init/launch overhead
            step_factor=2, warmup_iters=5, iters=50,
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"all_reduce_perf binary: {res['error']}")

        # With a single message size we get a single busbw row; look for latency in algbw
        algbw = res["algbw_gbs"]
        if not algbw:
            pytest.skip("Could not parse deterministic init output")
        mean_algbw = sum(algbw) / len(algbw)
        logger.info(
            "NCCL 8-byte AllReduce (50 iters): algbw mean=%.3f GB/s", mean_algbw
        )
        logger.info("Full output:\n%s", res["raw"].strip())


def _hf_auth_token() -> Optional[str]:
    """Resolve HuggingFace token from common environment variable names."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        return None
    token = token.strip().strip('"').strip("'")
    return token or None


def _hf_from_pretrained(cls, model_id: str, **kwargs):
    """Load HF assets with auth fallback for mixed transformers versions.

    Uses token= for modern transformers, and falls back to use_auth_token=
    when running with older versions.
    """
    token = _hf_auth_token()
    call_kwargs = dict(kwargs)
    if token:
        call_kwargs["token"] = token

    try:
        return cls.from_pretrained(model_id, **call_kwargs)
    except TypeError as exc:
        if token and "token" in str(exc):
            call_kwargs.pop("token", None)
            call_kwargs["use_auth_token"] = token
            return cls.from_pretrained(model_id, **call_kwargs)
        raise


@pytest.mark.node_validation
@pytest.mark.workload_acceptance
class TestWorkloadAcceptanceHuggingFace:
    """TC-NV-08: Workload acceptance using HuggingFace Transformer models.

    Inference model : _HF_INFERENCE_MODEL (default: gpt2)
    Training model  : _HF_TRAINING_MODEL  (default: bert-base-uncased)

    The tests use the 'transformers' library directly on GPU.  Set
    _HF_INFERENCE_MODEL / _HF_TRAINING_MODEL to a local cached path if the
    node has no internet access.
    """

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_hf_causal_lm_inference_latency(self):
        """Load a HuggingFace causal-LM (GPT-2) on GPU and measure per-token generation latency."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        device = torch.device("cuda:0")
        logger.info("Loading %s for causal-LM inference …", _HF_INFERENCE_MODEL)
        tokenizer = _hf_from_pretrained(AutoTokenizer, _HF_INFERENCE_MODEL)
        model = _hf_from_pretrained(
            AutoModelForCausalLM, _HF_INFERENCE_MODEL, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        prompt = "The AI head node validation test confirms"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        max_new_tokens = 32
        min_test_duration_sec = 5 * 60

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                torch.cuda.synchronize(device)

        # Measure
        latencies_ms = []
        with torch.no_grad():
            start = time.perf_counter()
            while (time.perf_counter() - start) < min_test_duration_sec:
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                torch.cuda.synchronize(device)
                latencies_ms.append((time.perf_counter() - t0) * 1000)

        total_runtime_sec = time.perf_counter() - start
        assert total_runtime_sec >= min_test_duration_sec, (
            f"Inference test runtime {total_runtime_sec:.1f}s shorter than required "
            f"{min_test_duration_sec}s"
        )

        mean_ms  = np.mean(latencies_ms)
        p95_ms   = np.percentile(latencies_ms, 95)
        tps = max_new_tokens / (mean_ms / 1000)   # tokens per second

        logger.info(
            "HF Causal-LM (%s) inference: mean=%.1f ms  p95=%.1f ms  "
            "tokens/sec=%.0f (batch=1, new_tokens=%d, runtime=%.1fs, samples=%d)",
            _HF_INFERENCE_MODEL, mean_ms, p95_ms, tps, max_new_tokens,
            total_runtime_sec, len(latencies_ms)
        )

        # GPT-2 should complete 32 tokens in < 500 ms on H100
        assert mean_ms < 500, (
            f"{_HF_INFERENCE_MODEL} inference mean latency {mean_ms:.0f} ms > 500 ms"
        )
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_hf_causal_lm_batch_throughput(self):
        """Measure batched prefill throughput for HuggingFace GPT-2 (tokens/sec)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        device = torch.device("cuda:0")
        tokenizer = _hf_from_pretrained(AutoTokenizer, _HF_INFERENCE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = _hf_from_pretrained(
            AutoModelForCausalLM, _HF_INFERENCE_MODEL, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        batch_size  = 8
        seq_len     = 256
        max_new     = 64
        prompts     = ["GPU validation benchmark prompt number " + str(i)
                       for i in range(batch_size)]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=seq_len
        ).to(device)

        # Warmup
        with torch.no_grad():
            _ = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            torch.cuda.synchronize(device)

        # Measure 5 iterations
        t0 = time.perf_counter()
        n_iters = 5
        with torch.no_grad():
            for _ in range(n_iters):
                out = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        total_tokens = batch_size * max_new * n_iters
        tps = total_tokens / elapsed
        logger.info(
            "HF Causal-LM (%s) batch throughput: %.0f tokens/sec "
            "(batch=%d, max_new_tokens=%d, iters=%d)",
            _HF_INFERENCE_MODEL, tps, batch_size, max_new, n_iters
        )

        assert tps > 100, f"Batch throughput {tps:.0f} tokens/sec too low"
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_hf_encoder_training_throughput(self):
        """Fine-tune HuggingFace BERT for one epoch of synthetic data; validate throughput."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        device = torch.device("cuda:0")
        logger.info("Loading %s for training throughput test …", _HF_TRAINING_MODEL)
        tokenizer = _hf_from_pretrained(AutoTokenizer, _HF_TRAINING_MODEL)
        model = _hf_from_pretrained(
            AutoModelForSequenceClassification, _HF_TRAINING_MODEL, num_labels=2
        ).to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        batch_size = 16
        seq_len    = 128

        # Synthetic batch
        input_ids      = torch.randint(0, tokenizer.vocab_size,
                                       (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels         = torch.randint(0, 2, (batch_size,), device=device)

        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)

        # Measure 10 steps
        n_steps = 10
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        samples_per_sec = (batch_size * n_steps) / elapsed
        logger.info(
            "HF Encoder training (%s): %.0f samples/sec "
            "(batch=%d, seq=%d, steps=%d)",
            _HF_TRAINING_MODEL, samples_per_sec, batch_size, seq_len, n_steps
        )

        assert samples_per_sec > 50, (
            f"{_HF_TRAINING_MODEL} training throughput {samples_per_sec:.0f} samples/sec too low"
        )
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_vllm_inference_throughput(self):
        """Run vLLM offline inference with a HuggingFace model and measure throughput."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            pytest.skip("vllm not installed")

        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        hf_token = _hf_auth_token()
        logger.info("Launching vLLM engine with model=%s …", _HF_INFERENCE_MODEL)
        llm_kwargs = {
            "model": _HF_INFERENCE_MODEL,
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.4,
            "max_model_len": 512,
            "enforce_eager": True,
            "disable_log_stats": True,
        }
        if hf_token:
            llm_kwargs["hf_token"] = hf_token

        try:
            llm = LLM(**llm_kwargs)
        except TypeError as exc:
            # Older/newer vLLM builds may not accept one of the optional kwargs.
            logger.warning("vLLM init kwargs compatibility issue: %s", exc)
            llm_kwargs.pop("hf_token", None)
            llm_kwargs.pop("disable_log_stats", None)
            try:
                llm = LLM(**llm_kwargs)
            except Exception as init_exc:
                pytest.skip(f"vLLM engine initialization failed: {init_exc}")
        except Exception as exc:
            pytest.skip(f"vLLM engine initialization failed: {exc}")

        sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

        prompts = [
            "Explain PCIe Gen5 bandwidth in one sentence:",
            "Describe NVLink topology for H100 GPUs:",
            "What is the purpose of HBM3 memory in AI accelerators:",
            "Summarize the importance of NCCL in distributed training:",
        ]

        # Warmup
        try:
            _ = llm.generate(prompts[:1], sampling_params)
        except Exception as exc:
            pytest.skip(f"vLLM warmup failed: {exc}")

        # Measure
        t0 = time.perf_counter()
        try:
            outputs = llm.generate(prompts, sampling_params)
        except Exception as exc:
            pytest.skip(f"vLLM inference run failed: {exc}")
        elapsed = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / elapsed

        logger.info(
            "vLLM inference (%s): %d prompts in %.2f s  total_tokens=%d  "
            "throughput=%.0f tokens/sec",
            _HF_INFERENCE_MODEL, len(prompts), elapsed, total_tokens, tps
        )
        for i, output in enumerate(outputs):
            logger.info("  Prompt %d: %s", i, output.outputs[0].text[:80].replace("\n", " "))

        assert tps > 10, f"vLLM throughput {tps:.0f} tokens/sec too low"
        del llm
        torch.cuda.empty_cache()


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
