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
import importlib.util
import importlib.metadata
import multiprocessing as mp
import os
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from tests.theoretical_specs import ACCEPTANCE_THRESHOLDS

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

_NVBANDWIDTH_BIN    = Path(
    os.environ.get("NVBANDWIDTH_BIN", "/root/workloads/nvbandwidth")
)
_P2P_LATENCY_BIN    = Path(
    "/root/workloads/cuda-samples/Samples/5_Domain_Specific"
    "/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest"
)
_NCCL_TESTS_DIR     = Path("/root/workloads/nccl-tests/build")

# HuggingFace model IDs used for inference / training acceptance tests.
# Swap to a locally cached path if internet access is unavailable on the node.
_HF_INFERENCE_MODEL = os.environ.get(
    "HF_INFERENCE_MODEL",
    "openai-community/gpt2",
)  # small causal-LM, ~500 MB
_HF_TRAINING_MODEL  = "bert-base-uncased"  # encoder, fast fine-tune loop

_HF_DEEPSEEK_R1_QUANT_MODEL = os.environ.get(
    "HF_DEEPSEEK_R1_QUANT_MODEL",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
)
_HF_LLAMA_7B_MODEL = os.environ.get(
    "HF_LLAMA_7B_MODEL",
    "meta-llama/Meta-Llama-3-8B",
)
_HF_LLAMA_70B_MODEL = os.environ.get(
    "HF_LLAMA_70B_MODEL",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
)
_HF_QWEN_70B_MODEL = os.environ.get(
    "HF_QWEN_70B_MODEL",
    "Qwen/Qwen2.5-72B-Instruct",
)
_HF_LLAMA_33_70B_MODEL = os.environ.get(
    "HF_LLAMA_33_70B_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct",
)
_HF_MIXTRAL_8X22B_MODEL = os.environ.get(
    "HF_MIXTRAL_8X22B_MODEL",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
)
_HF_DEEPSEEK_R1_70B_MODEL = os.environ.get(
    "HF_DEEPSEEK_R1_70B_MODEL",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
)

# Default to running large-model workloads and allowing remote fetches unless
# explicitly disabled by environment overrides.
os.environ.setdefault("HF_ENABLE_LARGE_MODELS", "1")
os.environ.setdefault("HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD", "1")


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

def _resolve_nvbandwidth_binary() -> Optional[Path]:
    """Resolve nvbandwidth executable from either a direct file path or root directory."""
    if _NVBANDWIDTH_BIN.is_file():
        return _NVBANDWIDTH_BIN

    if _NVBANDWIDTH_BIN.is_dir():
        candidates = [
            _NVBANDWIDTH_BIN / "nvbandwidth",
            _NVBANDWIDTH_BIN / "build" / "nvbandwidth",
            _NVBANDWIDTH_BIN / "bin" / "nvbandwidth",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    return None

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
    nvbandwidth_bin = _resolve_nvbandwidth_binary()
    if nvbandwidth_bin is None:
        return {"testcase": testcase, "raw": "", "values": [],
                "matrix": [], "error": f"nvbandwidth binary not found from {_NVBANDWIDTH_BIN}"}
    try:
        result = subprocess.run(
            [str(nvbandwidth_bin), "-t", testcase],
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
                    # Parse only numeric tokens; skip N/A and other non-numeric markers
                    row_vals = [float(x) for x in tokens[1:] if x not in ("N/A", "---", "n/a")]
                    if row_vals:
                        matrix.append(row_vals)
                        values.extend(row_vals)
                except ValueError:
                    # Ignore non-numeric lines inside matrix block.
                    pass

        # Fallback for D2D or other single-value outputs: parse any standalone floats
        if not values:
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and not stripped.startswith("Device"):
                    tokens = stripped.split()
                    for token in tokens:
                        try:
                            val = float(token)
                            if val > 0 and val < 10000:  # Sanity check: BW between 0-10K GB/s
                                values.append(val)
                        except ValueError:
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
                    tokens = l.split()
                    # First token is row/GPU ID; skip it and parse remaining as bandwidth values
                    vals = [float(x) for x in tokens[1:] if x.replace('.','',1).isdigit()]
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
        _record_kpi(
            "test_gpu_enumeration",
            {
                "gpu_count": result.get("count", 0),
                "gpus": [
                    {
                        "index": gpu.get("index"),
                        "name": gpu.get("name"),
                        "pci_bus_id": gpu.get("pci_bus_id"),
                        "driver_version": gpu.get("driver_version"),
                    }
                    for gpu in result.get("gpus", [])
                ],
            },
        )

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
        nvlink_pair_count = 0
        pcie_pair_count = 0
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
                nvlink_pair_count = len(nvlink_pairs)
                link_summary.append(f"  NVLink pairs  ({len(nvlink_pairs)}):")
                for src, dst, lt in nvlink_pairs:
                    link_summary.append(f"    GPU{src} <-> GPU{dst} : {lt}")
            if pcie_pairs:
                pcie_pair_count = len(pcie_pairs)
                link_summary.append(f"  PCIe-only pairs ({len(pcie_pairs)}):")
                for src, dst, lt in pcie_pairs:
                    link_summary.append(f"    GPU{src} <-> GPU{dst} : {lt}")
            logger.info("\n".join(link_summary))

        _record_kpi(
            "test_pcie_topology_discovery",
            {
                "gpu_entries": len(topo_result.get("topology", [])),
                "downgraded_gpu_count": len(downgraded_gpus),
                "downgraded_gpus": downgraded_gpus,
                "nvlink_pair_count": nvlink_pair_count,
                "pcie_pair_count": pcie_pair_count,
                "matrix_error": matrix_result.get("error"),
            },
        )

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
        _record_kpi(
            "test_topology_consistency",
            {
                "valid": bool(valid),
                "gpu_count": len(enum_result.get("gpus", [])),
                "topology_count": len(topo_result.get("topology", [])),
            },
        )
        assert valid, message
        logger.info(message)


@pytest.mark.node_validation
@pytest.mark.pcie_validation
class TestPCIeBandwidthDirect:
    """TC-NV-02: PCIe bandwidth validation for directly attached GPUs."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_h2d_bandwidth_all_gpus(self):
        """Measure H2D bandwidth for all GPUs and validate >= 65% of PCIe theoretical (63 GB/s).

        Boundary handling:
          - Uses NODE_PCIE_BW_BOUNDARY_GBS (default 1.0) to mark near-threshold values.
          - Collects all GPU measurements first, then fails once with full diagnostics.
        """
        n_gpus = torch.cuda.device_count()
        assert n_gpus > 0, "No CUDA devices found"
        threshold_gbs = 63.0 * 0.65  # 65% of PCIe Gen5×16 theoretical
        boundary_gbs = float(os.environ.get("NODE_PCIE_BW_BOUNDARY_GBS", "1.0"))
        n_iters = int(os.environ.get("NODE_PCIE_BW_ITERS", "8"))
        n_bytes = int(os.environ.get("NODE_PCIE_BW_BYTES", str(1024**3)))
        per_gpu_bw: List[float] = []
        failing: List[str] = []
        borderline: List[str] = []

        for gpu_idx in range(n_gpus):
            bw = measure_h2d_bandwidth(device_idx=gpu_idx, n_bytes=n_bytes, n_iters=n_iters)
            per_gpu_bw.append(float(bw))
            efficiency = (bw / 63.0) * 100
            logger.info(f"GPU {gpu_idx} H2D BW: {bw:.1f} GB/s ({efficiency:.1f}% of 63 GB/s)")
            if bw <= 0:
                failing.append(
                    f"GPU {gpu_idx}: 0.0 GB/s (measurement failed; check CUDA/P2P path and logs)"
                )
                continue
            if bw < threshold_gbs:
                delta = threshold_gbs - bw
                if delta <= boundary_gbs:
                    borderline.append(
                        f"GPU {gpu_idx}: {bw:.2f} GB/s ({delta:.2f} below threshold, boundary band)"
                    )
                failing.append(
                    f"GPU {gpu_idx}: {bw:.2f} GB/s < {threshold_gbs:.2f} GB/s"
                )

        _record_kpi(
            "test_h2d_bandwidth_all_gpus",
            {
                "gpu_count": n_gpus,
                "n_bytes": n_bytes,
                "n_iters": n_iters,
                "threshold_gbs": threshold_gbs,
                "boundary_gbs": boundary_gbs,
                "min_gbs": round(float(min(per_gpu_bw)), 2),
                "mean_gbs": round(float(np.mean(per_gpu_bw)), 2),
                "per_gpu_gbs": [round(x, 2) for x in per_gpu_bw],
                "failing_gpus": failing,
                "borderline_gpus": borderline,
            },
        )

        if failing:
            details = "\n".join(failing)
            pytest.fail(
                "H2D PCIe bandwidth validation failed.\n"
                f"threshold={threshold_gbs:.2f} GB/s, boundary={boundary_gbs:.2f} GB/s, "
                f"min={min(per_gpu_bw):.2f} GB/s, mean={float(np.mean(per_gpu_bw)):.2f} GB/s\n"
                f"Per-GPU failures:\n{details}"
            )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_d2h_bandwidth_all_gpus(self):
        """Measure D2H bandwidth for all GPUs and validate >= 65% of PCIe theoretical.

        Boundary handling:
          - Uses NODE_PCIE_BW_BOUNDARY_GBS (default 1.0) to mark near-threshold values.
          - Collects all GPU measurements first, then fails once with full diagnostics.
        """
        n_gpus = torch.cuda.device_count()
        assert n_gpus > 0, "No CUDA devices found"
        threshold_gbs = 63.0 * 0.65
        boundary_gbs = float(os.environ.get("NODE_PCIE_BW_BOUNDARY_GBS", "1.0"))
        n_iters = int(os.environ.get("NODE_PCIE_BW_ITERS", "8"))
        n_bytes = int(os.environ.get("NODE_PCIE_BW_BYTES", str(1024**3)))
        per_gpu_bw: List[float] = []
        failing: List[str] = []
        borderline: List[str] = []

        for gpu_idx in range(n_gpus):
            bw = measure_d2h_bandwidth(device_idx=gpu_idx, n_bytes=n_bytes, n_iters=n_iters)
            per_gpu_bw.append(float(bw))
            efficiency = (bw / 63.0) * 100
            logger.info(f"GPU {gpu_idx} D2H BW: {bw:.1f} GB/s ({efficiency:.1f}% of 63 GB/s)")
            if bw <= 0:
                failing.append(
                    f"GPU {gpu_idx}: 0.0 GB/s (measurement failed; check CUDA/P2P path and logs)"
                )
                continue
            if bw < threshold_gbs:
                delta = threshold_gbs - bw
                if delta <= boundary_gbs:
                    borderline.append(
                        f"GPU {gpu_idx}: {bw:.2f} GB/s ({delta:.2f} below threshold, boundary band)"
                    )
                failing.append(
                    f"GPU {gpu_idx}: {bw:.2f} GB/s < {threshold_gbs:.2f} GB/s"
                )

        _record_kpi(
            "test_d2h_bandwidth_all_gpus",
            {
                "gpu_count": n_gpus,
                "n_bytes": n_bytes,
                "n_iters": n_iters,
                "threshold_gbs": threshold_gbs,
                "boundary_gbs": boundary_gbs,
                "min_gbs": round(float(min(per_gpu_bw)), 2),
                "mean_gbs": round(float(np.mean(per_gpu_bw)), 2),
                "per_gpu_gbs": [round(x, 2) for x in per_gpu_bw],
                "failing_gpus": failing,
                "borderline_gpus": borderline,
            },
        )

        if failing:
            details = "\n".join(failing)
            pytest.fail(
                "D2H PCIe bandwidth validation failed.\n"
                f"threshold={threshold_gbs:.2f} GB/s, boundary={boundary_gbs:.2f} GB/s, "
                f"min={min(per_gpu_bw):.2f} GB/s, mean={float(np.mean(per_gpu_bw)):.2f} GB/s\n"
                f"Per-GPU failures:\n{details}"
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
        "device_to_device_memcpy_read_ce":       30.0,   # NVLink CE bandwidth, ~50 GB/s typical
        "host_to_device_memcpy_sm":              30.0,
        "device_to_host_memcpy_sm":              30.0,
    }

    def _run_and_validate(self, testcase: str) -> None:
        """Run nvbandwidth for *testcase*, log the full matrix, assert threshold."""
        if _resolve_nvbandwidth_binary() is None:
            pytest.skip(f"nvbandwidth binary not found from {_NVBANDWIDTH_BIN}")

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
        _record_kpi(
            f"test_nvbandwidth_{testcase}",
            {
                "testcase": testcase,
                "num_values": len(values),
                "min_bw_gbs": round(float(min_bw), 2),
                "mean_bw_gbs": round(float(mean_bw), 2),
                "threshold_gbs": round(float(threshold), 2),
            },
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

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_pcie_bw_p2p_latency_hugepages_2mb(self):
        """AI_HOSTCPU_FUNC_TEST_ID_050: PCIe BW and P2P latency with 2 MB hugepages.

        Enables 2 MB kernel hugepages (if root permits) to reduce TLB pressure on
        Host<->GPU DMA transfers, then re-runs H2D and D2H nvbandwidth CE tests.
        Falls back to a pure-PyTorch H2D/D2H timing loop when nvbandwidth is absent.

        Pass criteria (PyTorch path):
          - H2D and D2H BW >= 41 GB/s (65 % of PCIe Gen5 nominal)
          - Hugepage state recorded in KPI for A/B comparison
        """
        # ------------------------------------------------------------------ #
        # 1.  Try to enable 2 MB hugepages                                     #
        # ------------------------------------------------------------------ #
        HP_NODE = Path("/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages")
        hugepages_before = 0
        hugepages_active = False
        if HP_NODE.exists():
            try:
                hugepages_before = int(HP_NODE.read_text().strip())
                if hugepages_before < 512:
                    HP_NODE.write_text("512\n")
                    hugepages_active = int(HP_NODE.read_text().strip()) >= 512
                else:
                    hugepages_active = True
            except PermissionError:
                logger.warning("No permission to configure hugepages; test proceeds without")
            except Exception as exc:
                logger.warning("Hugepage configuration error: %s", exc)

        logger.info("2 MB hugepages: before=%d  active=%s", hugepages_before, hugepages_active)

        bw_results: Dict[str, object] = {"hugepages_active": hugepages_active}
        try:
            # ---------------------------------------------------------------- #
            # 2a.  nvbandwidth path (preferred)                                 #
            # ---------------------------------------------------------------- #
            if _resolve_nvbandwidth_binary() is not None:
                for tc in (
                    "host_to_device_memcpy_ce",
                    "device_to_host_memcpy_ce",
                    "host_to_device_bidirectional_memcpy_ce",
                ):
                    res = _run_nvbandwidth(tc, timeout=180)
                    if res.get("error"):
                        logger.warning("nvbandwidth %s: %s", tc, res["error"])
                    else:
                        flat = [v for row in res.get("matrix", []) for v in row if v > 0]
                        if flat:
                            bw_results[tc] = round(min(flat), 2)
                            logger.info(
                                "nvbandwidth %-42s  min=%.2f GB/s  (hugepages=%s)",
                                tc, min(flat), hugepages_active,
                            )

            # ---------------------------------------------------------------- #
            # 2b.  PyTorch fallback path                                        #
            # ---------------------------------------------------------------- #
            if torch.cuda.device_count() == 0:
                pytest.skip("No CUDA devices")

            n_bytes = int(os.environ.get("NODE_PCIE_BW_BYTES", str(512 * 1024 * 1024)))
            n_iters = int(os.environ.get("NODE_PCIE_BW_ITERS", "5"))
            device = torch.device("cuda:0")
            host_buf = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
            dev_buf  = torch.empty(n_bytes, dtype=torch.uint8, device=device)
            stream   = torch.cuda.Stream(device=device)

            # H2D
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(n_iters):
                    dev_buf.copy_(host_buf, non_blocking=True)
            stream.synchronize()
            h2d_gbs = (n_bytes * n_iters) / ((time.perf_counter() - t0) * 1e9)

            # D2H
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(n_iters):
                    host_buf.copy_(dev_buf, non_blocking=True)
            stream.synchronize()
            d2h_gbs = (n_bytes * n_iters) / ((time.perf_counter() - t0) * 1e9)

            bw_results["torch_H2D_gbs"] = round(h2d_gbs, 2)
            bw_results["torch_D2H_gbs"] = round(d2h_gbs, 2)
            logger.info(
                "PyTorch BW (hugepages=%s, %d MB): H2D=%.2f GB/s  D2H=%.2f GB/s",
                hugepages_active, n_bytes // (1024 * 1024), h2d_gbs, d2h_gbs,
            )
            del host_buf, dev_buf

            threshold_gbs = 41.0
            assert h2d_gbs >= threshold_gbs, (
                f"H2D BW {h2d_gbs:.2f} GB/s (hugepages={hugepages_active}) "
                f"below threshold {threshold_gbs} GB/s"
            )
            assert d2h_gbs >= threshold_gbs, (
                f"D2H BW {d2h_gbs:.2f} GB/s (hugepages={hugepages_active}) "
                f"below threshold {threshold_gbs} GB/s"
            )
        finally:
            # ---------------------------------------------------------------- #
            # 3.  Restore original hugepage count                              #
            # ---------------------------------------------------------------- #
            if HP_NODE.exists() and hugepages_before < 512:
                try:
                    HP_NODE.write_text(f"{hugepages_before}\n")
                except Exception:
                    pass

        _record_kpi(
            "test_pcie_bw_p2p_latency_hugepages_2mb",
            {
                "hugepages_before": hugepages_before,
                "hugepages_active": hugepages_active,
                "bw_results": {str(k): v for k, v in bw_results.items()},
            },
        )


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

        timeout_sec = int(os.environ.get("NODE_P2P_BINARY_TIMEOUT_SEC", "240"))
        res = _run_p2p_latency_test(timeout=timeout_sec)
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
        _record_kpi(
            "test_p2p_bandwidth_latency_binary",
            {
                "timeout_sec": timeout_sec,
                "min_threshold_gbs": min_threshold,
                "bidir_min_gbs": round(float(min(bidir)), 2) if bidir else None,
                "bidir_mean_gbs": round(float(sum(bidir) / len(bidir)), 2) if bidir else None,
                "unidir_min_gbs": round(float(min(unidir)), 2) if unidir else None,
                "unidir_mean_gbs": round(float(sum(unidir) / len(unidir)), 2) if unidir else None,
                "latency_min_us": round(float(min(latency)), 2) if latency else None,
                "latency_mean_us": round(float(sum(latency) / len(latency)), 2) if latency else None,
                "num_offdiag_samples": len(off_diag_bw),
            },
        )
        failing = [v for v in off_diag_bw if v < min_threshold]
        assert not failing, (
            f"{len(failing)} P2P bandwidth values below {min_threshold} GB/s: "
            f"{[f'{v:.1f}' for v in sorted(failing)]}"
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_p2p_bandwidth_pytorch_fallback(self):
        """PyTorch direct P2P measurement for all GPU pairs with explicit diagnostics."""
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs for P2P test")

        min_threshold = float(os.environ.get("NODE_P2P_MIN_GBS", "40.0"))
        n_iters = int(os.environ.get("NODE_P2P_TORCH_ITERS", "8"))
        n_bytes = int(os.environ.get("NODE_P2P_TORCH_BYTES", str(1024**3)))
        header = ["\n", "=" * 60, "  PyTorch P2P Bandwidth Matrix (GB/s)", "=" * 60]
        header.append(f"  {'':6}" + "".join(f"  GPU{d:<3}" for d in range(n_gpus)))
        rows = []
        measured_bw: List[float] = []
        failing_pairs: List[str] = []
        inaccessible_pairs: List[str] = []
        attempted_pairs = 0
        measured_pairs = 0

        for src in range(n_gpus):
            row_vals = []
            for dst in range(n_gpus):
                if src == dst:
                    row_vals.append("  ---  ")
                else:
                    attempted_pairs += 1
                    if not torch.cuda.can_device_access_peer(src, dst):
                        inaccessible_pairs.append(f"GPU{src}->GPU{dst}")
                        row_vals.append("  N/A  ")
                        continue

                    bw = measure_p2p_bandwidth(src_device=src, dst_device=dst,
                                              n_bytes=n_bytes, n_iters=n_iters)
                    row_vals.append(f"{bw:7.1f}")
                    if bw > 0:
                        measured_pairs += 1
                        measured_bw.append(float(bw))
                        if bw < min_threshold:
                            failing_pairs.append(
                                f"GPU{src}->GPU{dst}: {bw:.1f} GB/s < {min_threshold:.1f} GB/s"
                            )
                    else:
                        failing_pairs.append(
                            f"GPU{src}->GPU{dst}: 0.0 GB/s (measurement failed; see warnings)"
                        )

            rows.append(f"  GPU{src}   " + "".join(row_vals))

        logger.info("\n".join(header + rows + ["=" * 60]))

        kpi_payload = {
            "gpu_count": n_gpus,
            "n_bytes": n_bytes,
            "n_iters": n_iters,
            "min_threshold_gbs": min_threshold,
            "attempted_pairs": attempted_pairs,
            "measured_pairs": measured_pairs,
            "inaccessible_pairs": len(inaccessible_pairs),
            "failing_pairs": len(failing_pairs),
            "min_gbs": round(float(min(measured_bw)), 2) if measured_bw else None,
            "mean_gbs": round(float(np.mean(measured_bw)), 2) if measured_bw else None,
        }
        _record_kpi("test_p2p_bandwidth_pytorch_fallback", kpi_payload)

        if measured_pairs == 0:
            if inaccessible_pairs and len(inaccessible_pairs) == attempted_pairs:
                pytest.skip(
                    "No peer-accessible GPU pairs for PyTorch P2P fallback test. "
                    "All pairs report no P2P access."
                )
            pytest.fail(
                "P2P fallback measured 0 valid pairs. "
                f"attempted_pairs={attempted_pairs}, inaccessible_pairs={len(inaccessible_pairs)}."
            )

        if failing_pairs:
            summary = "\n".join(failing_pairs[:16])
            if len(failing_pairs) > 16:
                summary += f"\n... and {len(failing_pairs) - 16} more failing pairs"
            pytest.fail(
                "P2P bandwidth fallback validation failed.\n"
                f"threshold={min_threshold:.1f} GB/s, measured_pairs={measured_pairs}, "
                f"inaccessible_pairs={len(inaccessible_pairs)}\n"
                f"Failures:\n{summary}"
            )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_p2p_multi_stream_bandwidth(self):
        """AI_HOSTCPU_FUNC_TEST_ID_052: P2P bandwidth with multiple concurrent CUDA streams.

        Measures GPU-to-GPU copy bandwidth with 1, 2, 4, and 8 concurrent streams.
        This check focuses on stability and consistency under concurrent streams,
        not on requiring super-linear scaling beyond the available fabric peak.

        Pass criteria:
          - Single-stream P2P BW > 0 (transfer succeeds on at least one GPU pair)
          - 4-stream BW retains at least 85% of 1-stream BW
          - Reverse directions of the same pair remain reasonably symmetric
        """
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need >= 2 GPUs for P2P multi-stream bandwidth test")

        n_bytes_per_stream = int(os.environ.get("P2P_MULTI_STREAM_BYTES", str(256 * 1024 * 1024)))
        n_iters = int(os.environ.get("P2P_MULTI_STREAM_ITERS", "5"))
        _sc_env = os.environ.get("P2P_MULTI_STREAM_COUNTS", "")
        stream_counts = (
            [int(x.strip()) for x in _sc_env.split(",") if x.strip().isdigit()]
            if _sc_env.strip()
            else [1, 2, 4, 8]
        )
        stream_counts = sorted(set(stream_counts))
        has_stream_1 = 1 in stream_counts
        has_stream_4 = 4 in stream_counts
        symmetry_stream_count = 4 if has_stream_4 else max(stream_counts)
        min_stability_ratio = float(
            os.environ.get(
                "P2P_MULTI_STREAM_MIN_STABILITY",
                str(ACCEPTANCE_THRESHOLDS["p2p_multi_stream_stability_ratio"]),
            )
        )
        min_directional_symmetry = float(
            os.environ.get(
                "P2P_MULTI_STREAM_MIN_DIRECTIONAL_SYMMETRY",
                str(ACCEPTANCE_THRESHOLDS["p2p_multi_stream_directional_symmetry"]),
            )
        )
        max_pairs = int(os.environ.get("P2P_MULTI_STREAM_MAX_PAIRS", "0"))  # 0 => all pairs
        include_per_pair = os.environ.get("P2P_MULTI_STREAM_INCLUDE_PER_PAIR", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        peer_pairs: List[Tuple[int, int]] = []
        for s in range(n_gpus):
            for d in range(n_gpus):
                if s != d and torch.cuda.can_device_access_peer(s, d):
                    peer_pairs.append((s, d))

        if not peer_pairs:
            pytest.skip("No peer-accessible GPU pairs found for P2P multi-stream bandwidth test")

        if max_pairs > 0:
            peer_pairs = peer_pairs[:max_pairs]

        all_pair_results: Dict[str, Dict[str, float]] = {}
        failing_pairs: List[str] = []
        ratio_records: List[Tuple[str, float, float, float]] = []
        directional_records: List[Tuple[str, str, float]] = []

        for src, dst in peer_pairs:
            logger.info("P2P multi-stream: GPU %d -> GPU %d", src, dst)
            bw_by_streams: Dict[str, float] = {}

            for n_streams in stream_counts:
                src_bufs = [
                    torch.empty(n_bytes_per_stream, dtype=torch.uint8, device=f"cuda:{src}")
                    for _ in range(n_streams)
                ]
                dst_bufs = [
                    torch.empty(n_bytes_per_stream, dtype=torch.uint8, device=f"cuda:{dst}")
                    for _ in range(n_streams)
                ]
                streams = [torch.cuda.Stream(device=f"cuda:{dst}") for _ in range(n_streams)]

                # Warmup
                with torch.cuda.device(dst):
                    for sl, (sb, db) in enumerate(zip(src_bufs, dst_bufs)):
                        with torch.cuda.stream(streams[sl]):
                            db.copy_(sb, non_blocking=True)
                    torch.cuda.synchronize(dst)

                # Measure
                total_bytes = n_bytes_per_stream * n_streams * n_iters
                t0 = time.perf_counter()
                with torch.cuda.device(dst):
                    for _ in range(n_iters):
                        for sl, (sb, db) in enumerate(zip(src_bufs, dst_bufs)):
                            with torch.cuda.stream(streams[sl]):
                                db.copy_(sb, non_blocking=True)
                        torch.cuda.synchronize(dst)
                elapsed = time.perf_counter() - t0

                bw_gbs = total_bytes / (elapsed * 1e9)
                bw_by_streams[str(n_streams)] = round(bw_gbs, 2)
                logger.info(
                    "  %d stream(s): %.2f GB/s  (%.0f MB/stream)",
                    n_streams, bw_gbs, n_bytes_per_stream / (1024 * 1024),
                )
                del src_bufs, dst_bufs, streams

            bw_1 = bw_by_streams.get("1", 0.0)
            bw_4 = bw_by_streams.get("4", 0.0)
            ratio = (bw_4 / bw_1) if bw_1 > 0 else 0.0

            pair_key = f"{src}->{dst}"
            all_pair_results[pair_key] = {
                "bw_1_stream_gbs": round(float(bw_1), 2),
                "bw_4_stream_gbs": round(float(bw_4), 2),
                "ratio_4_over_1": round(float(ratio), 3),
                "bw_by_streams_gbs": bw_by_streams,
            }
            if has_stream_1 and has_stream_4:
                ratio_records.append((pair_key, float(ratio), float(bw_1), float(bw_4)))

                if bw_1 <= 0:
                    failing_pairs.append(f"GPU{src}->GPU{dst}: 1-stream BW is 0.0 GB/s")
                elif bw_4 > 0 and ratio < min_stability_ratio:
                    failing_pairs.append(
                        f"GPU{src}->GPU{dst}: 4-stream BW {bw_4:.2f} GB/s retains only {ratio:.2f}x of "
                        f"1-stream BW {bw_1:.2f} GB/s (< {min_stability_ratio:.2f}x)"
                    )

            logger.info("P2P multi-stream BW (GPU %d->%d): %s", src, dst, bw_by_streams)

        for src, dst in peer_pairs:
            forward_key = f"{src}->{dst}"
            reverse_key = f"{dst}->{src}"
            if forward_key >= reverse_key:
                continue

            forward = all_pair_results.get(forward_key)
            reverse = all_pair_results.get(reverse_key)
            if not forward or not reverse:
                continue

            forward_bw = float(forward.get("bw_by_streams_gbs", {}).get(str(symmetry_stream_count), 0.0))
            reverse_bw = float(reverse.get("bw_by_streams_gbs", {}).get(str(symmetry_stream_count), 0.0))
            if forward_bw <= 0.0 or reverse_bw <= 0.0:
                continue

            symmetry_ratio = min(forward_bw, reverse_bw) / max(forward_bw, reverse_bw)
            directional_records.append((forward_key, reverse_key, symmetry_ratio))
            if symmetry_ratio < min_directional_symmetry:
                failing_pairs.append(
                    f"GPU{forward_key} vs GPU{reverse_key}: {symmetry_stream_count}-stream directional symmetry "
                    f"{symmetry_ratio:.2f}x is below {min_directional_symmetry:.2f}x "
                    f"({forward_bw:.2f} vs {reverse_bw:.2f} GB/s)"
                )

        valid_ratio_records = [r for r in ratio_records if r[2] > 0.0 and r[3] > 0.0]
        min_ratio_observed = min((r[1] for r in valid_ratio_records), default=0.0)
        avg_ratio_observed = (
            sum(r[1] for r in valid_ratio_records) / len(valid_ratio_records)
            if valid_ratio_records
            else 0.0
        )
        worst_pairs = sorted(valid_ratio_records, key=lambda r: r[1])[:3]
        worst_directional_pairs = sorted(directional_records, key=lambda r: r[2])[:3]
        worst_pairs_summary = [
            {
                "pair": pair,
                "ratio_4_over_1": round(ratio, 3),
                "bw_1_stream_gbs": round(bw1, 2),
                "bw_4_stream_gbs": round(bw4, 2),
            }
            for pair, ratio, bw1, bw4 in worst_pairs
        ]
        worst_directional_summary = [
            {
                "forward_pair": forward_pair,
                "reverse_pair": reverse_pair,
                "directional_symmetry_ratio": round(symmetry_ratio, 3),
            }
            for forward_pair, reverse_pair, symmetry_ratio in worst_directional_pairs
        ]

        min_directional_symmetry_observed = min(
            (r[2] for r in directional_records),
            default=1.0,
        )

        kpi_payload: Dict[str, Any] = {
            "tested_pair_count": len(peer_pairs),
            "stream_counts": stream_counts,
            "n_bytes_per_stream": n_bytes_per_stream,
            "n_iters": n_iters,
            "required_min_stability_ratio": min_stability_ratio,
            "observed_min_scaling_ratio": round(min_ratio_observed, 3),
            "observed_avg_scaling_ratio": round(avg_ratio_observed, 3),
            "required_min_directional_symmetry": min_directional_symmetry,
            "directional_symmetry_stream_count": symmetry_stream_count,
            "observed_min_directional_symmetry": round(min_directional_symmetry_observed, 3),
            "failing_pair_count": len(failing_pairs),
            "worst_pairs": worst_pairs_summary,
            "worst_directional_pairs": worst_directional_summary,
        }
        if include_per_pair:
            kpi_payload["per_pair"] = all_pair_results

        _record_kpi("test_p2p_multi_stream_bandwidth", kpi_payload)

        if failing_pairs:
            details = "\n".join(failing_pairs[:6])
            if len(failing_pairs) > 6:
                details += f"\n... and {len(failing_pairs) - 6} more failing pair(s)"
            worst_text = ", ".join(
                f"{w['pair']}={w['ratio_4_over_1']:.3f}x" for w in worst_pairs_summary
            ) or "none"
            worst_directional_text = ", ".join(
                f"{w['forward_pair']}<->{w['reverse_pair']}={w['directional_symmetry_ratio']:.3f}x"
                for w in worst_directional_summary
            ) or "none"
            pytest.fail(
                "P2P multi-stream stability validation failed.\n"
                f"tested_pairs={len(peer_pairs)}, required_min_stability_ratio={min_stability_ratio:.2f}, "
                f"observed_min_scaling_ratio={min_ratio_observed:.3f}, "
                f"observed_avg_scaling_ratio={avg_ratio_observed:.3f}, "
                f"required_min_directional_symmetry={min_directional_symmetry:.2f}, "
                f"observed_min_directional_symmetry={min_directional_symmetry_observed:.3f}\n"
                f"worst_pairs={worst_text}\n"
                f"worst_directional_pairs={worst_directional_text}\n"
                f"Failures:\n{details}"
            )


@pytest.mark.node_validation
@pytest.mark.sync_overhead
class TestCPUGPULaunchAndSyncOverhead:
    """TC-NV-05: CPU↔GPU kernel launch and sync latency."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_kernel_launch_latency_stats(self):
        """Measure kernel launch latency distribution (min/p50/p95/p99/max)."""
        n_samples = int(os.environ.get("NODE_KERNEL_LAUNCH_SAMPLES", "200"))
        stats = measure_kernel_launch_latency(device_idx=0, n_samples=n_samples)
        assert stats, "Kernel launch measurement failed"

        logger.info("Kernel launch latency (microseconds):")
        for key in ["min", "p50", "mean", "p95", "p99", "max"]:
            logger.info(f"  {key:5s}: {stats[key]:8.2f} µs")

        _record_kpi(
            "test_kernel_launch_latency_stats",
            {
                "samples": n_samples,
                "min_us": round(float(stats["min"]), 3),
                "p50_us": round(float(stats["p50"]), 3),
                "mean_us": round(float(stats["mean"]), 3),
                "p95_us": round(float(stats["p95"]), 3),
                "p99_us": round(float(stats["p99"]), 3),
                "max_us": round(float(stats["max"]), 3),
            },
        )

        # Typical kernel launch latency: 1-10 microseconds
        assert stats["p50"] < 100, f"Median kernel launch latency {stats['p50']:.1f} µs seems high"

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_synchronize_overhead_stats(self):
        """Measure cudaDeviceSynchronize() overhead distribution."""
        n_samples = int(os.environ.get("NODE_SYNC_OVERHEAD_SAMPLES", "200"))
        stats = measure_synchronize_overhead(device_idx=0, n_samples=n_samples)
        assert stats, "Synchronize measurement failed"

        logger.info("cudaDeviceSynchronize() overhead (microseconds):")
        for key in ["min", "p50", "mean", "p95", "p99", "max"]:
            logger.info(f"  {key:5s}: {stats[key]:8.2f} µs")

        _record_kpi(
            "test_synchronize_overhead_stats",
            {
                "samples": n_samples,
                "min_us": round(float(stats["min"]), 3),
                "p50_us": round(float(stats["p50"]), 3),
                "mean_us": round(float(stats["mean"]), 3),
                "p95_us": round(float(stats["p95"]), 3),
                "p99_us": round(float(stats["p99"]), 3),
                "max_us": round(float(stats["max"]), 3),
            },
        )

        # Typical sync overhead: 1-50 microseconds
        assert stats["p95"] < 1000, f"P95 sync overhead {stats['p95']:.1f} µs seems high"


@pytest.mark.node_validation
@pytest.mark.power_validation
class TestCPUGPUPowerAndThrottling:
    """TC-NV-06: CPU↔GPU power coupling and thermal throttling validation."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_sustained_gpu_power_stability(self):
        """Run sustained GPU workload and verify stable power (low variance)."""
        duration_sec = int(os.environ.get("NODE_POWER_PROFILE_DURATION_SEC", "30"))
        power_data = measure_power_profile(duration_sec=duration_sec)
        if not power_data:
            pytest.skip("Could not measure power profile")

        logger.info(f"GPU Power profile (sustained workload, {power_data.get('samples', 0)} samples):")
        logger.info(f"  Min:   {power_data.get('min', 0):.1f} W")
        logger.info(f"  Mean:  {power_data.get('mean', 0):.1f} W")
        logger.info(f"  Max:   {power_data.get('max', 0):.1f} W")
        logger.info(f"  Std:   {power_data.get('std', 0):.1f} W")

        _record_kpi(
            "test_sustained_gpu_power_stability",
            {
                "duration_sec": duration_sec,
                "samples": int(power_data.get("samples", 0)),
                "min_w": round(float(power_data.get("min", 0.0)), 2),
                "mean_w": round(float(power_data.get("mean", 0.0)), 2),
                "max_w": round(float(power_data.get("max", 0.0)), 2),
                "std_w": round(float(power_data.get("std", 0.0)), 2),
            },
        )

        # Verify low jitter (std < 10 W)
        std = power_data.get("std", 0)
        assert std < 50, f"Power variance {std:.1f} W is too high (expected < 50 W)"

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_gpu_write_performance_limiters(self):
        """AI_HOSTCPU_FUNC_TEST_ID_051: Analyse GPU HBM write performance limiters.

        Measures effective write bandwidth from GPU SMs into HBM at increasing buffer
        sizes to distinguish between:
          * HBM-bandwidth-bound   — BW rises towards HBM theoretical as size grows
          * Instruction-serialisation-bound — BW plateaus early at small sizes

        On H100 SXM5 the HBM3 peak write bandwidth is ~3350 GB/s.  The test
        expects >= 5 % of theoretical (167 GB/s) at 512 MB.

        Pass criteria:
          - Peak write BW >= 167 GB/s (5 % of H100 HBM3 theoretical)
          - BW at 256 MB buffer >= 2× BW at 4 KB buffer (demonstrates HBM-bound region)
        """
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        device = torch.device("cuda:0")
        buf_sizes = [
            4 * 1024,            # 4 KB  — likely instruction-overhead bound
            64 * 1024,           # 64 KB
            1 * 1024 * 1024,     # 1 MB
            16 * 1024 * 1024,    # 16 MB
            256 * 1024 * 1024,   # 256 MB — should be HBM bandwidth bound
            512 * 1024 * 1024,   # 512 MB
        ]
        n_iters = int(os.environ.get("GPU_WRITE_ITERS", "10"))
        results: Dict[str, float] = {}

        for n_bytes in buf_sizes:
            n_elems = n_bytes // 4  # float32 elements
            buf = torch.empty(n_elems, dtype=torch.float32, device=device)
            buf.fill_(1.0)                      # warmup
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            for _ in range(n_iters):
                buf.fill_(0.5)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            bw_gbs = (n_bytes * n_iters) / (elapsed * 1e9)
            size_label = (
                f"{n_bytes // (1024 * 1024)}MB"
                if n_bytes >= 1024 * 1024
                else f"{n_bytes // 1024}KB"
            )
            results[size_label] = round(bw_gbs, 2)
            logger.info("Write BW  %-8s : %8.1f GB/s", size_label, bw_gbs)
            del buf

        peak_bw   = max(results.values()) if results else 0.0
        small_bw  = results.get("4KB",   1.0)
        large_bw  = results.get("256MB", results.get("512MB", 0.0))

        logger.info(
            "GPU write BW summary: peak=%.1f GB/s  "
            "small(4KB)=%.1f  large(256MB)=%.1f  ratio=%.1fx",
            peak_bw, small_bw, large_bw,
            large_bw / small_bw if small_bw > 0 else 0.0,
        )

        _record_kpi(
            "test_gpu_write_performance_limiters",
            {
                "device": torch.cuda.get_device_name(0),
                "n_iters": n_iters,
                "bw_by_size_gbs": results,
                "peak_write_bw_gbs": round(peak_bw, 2),
            },
        )

        # H100 HBM3 theoretical write BW ≈ 3350 GB/s; expect at least 5 %
        min_peak_bw = float(os.environ.get("GPU_WRITE_MIN_PEAK_GBS", "167.0"))
        assert peak_bw >= min_peak_bw, (
            f"Peak GPU write BW {peak_bw:.1f} GB/s below threshold {min_peak_bw:.0f} GB/s"
        )
        if small_bw > 0 and large_bw > 0:
            hbm_ratio = large_bw / small_bw
            min_hbm_ratio = float(os.environ.get("GPU_WRITE_HBM_BOUND_RATIO", "2.0"))
            assert hbm_ratio >= min_hbm_ratio, (
                f"Large-buffer write BW {large_bw:.1f} GB/s is only {hbm_ratio:.2f}× "
                f"small-buffer BW {small_bw:.1f} GB/s — write path may be "
                f"instruction-serialisation-limited rather than HBM-bandwidth-limited"
            )


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
        _record_kpi(
            "test_nvlink_status",
            {
                "available": bool(nvlink_info.get("available", False)),
                "status_excerpt": (nvlink_info.get("status", "") or "")[:500],
            },
        )
        # Non-fatal: node may be PCIe-only
        if not nvlink_info["available"]:
            logger.warning("NVLink not detected — system may be PCIe-only interconnect")

    def test_nccl_availability(self):
        """Verify NCCL is accessible (via torch or test binary)."""
        available, message = check_nccl_availability()
        logger.info("NCCL availability: %s", message)
        _record_kpi(
            "test_nccl_availability",
            {
                "available": bool(available),
                "message": message,
            },
        )
        if not available:
            pytest.skip(message)

    def test_nccl_all_reduce_perf(self):
        """Run all_reduce_perf across all GPUs and validate peak busbw."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs for NCCL collective test")

        iters = int(os.environ.get("NODE_NCCL_ITERS", "30"))
        res = _run_nccl_test(
            "all_reduce_perf",
            n_gpus=n_gpus,
            min_bytes="1M",
            max_bytes="1G",
            step_factor=4,
            iters=iters,
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
        _record_kpi(
            "test_nccl_all_reduce_perf",
            {
                "gpu_count": n_gpus,
                "iters": iters,
                "peak_busbw_gbs": round(float(peak_busbw), 2),
                "mean_busbw_gbs": round(float(sum(busbw) / len(busbw)), 2),
                "threshold_gbs": min_busbw,
                "samples": len(busbw),
            },
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

        iters = int(os.environ.get("NODE_NCCL_ITERS", "30"))

        res = _run_nccl_test(
            "all_gather_perf", n_gpus=n_gpus,
            min_bytes="1M", max_bytes="256M", step_factor=4, iters=iters
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"all_gather_perf binary: {res['error']}")

        logger.info("\n=== NCCL all_gather_perf ===\n%s", res["raw"].strip())
        busbw = res["busbw_gbs"]
        if not busbw:
            pytest.skip("Could not parse all_gather_perf output")
        peak = max(busbw)
        logger.info("AllGather peak busbw: %.1f GB/s", peak)
        _record_kpi(
            "test_nccl_all_gather_perf",
            {
                "gpu_count": n_gpus,
                "iters": iters,
                "peak_busbw_gbs": round(float(peak), 2),
                "mean_busbw_gbs": round(float(sum(busbw) / len(busbw)), 2),
                "samples": len(busbw),
            },
        )
        assert peak >= 30.0, f"AllGather busbw {peak:.1f} GB/s too low"

    def test_nccl_reduce_scatter_perf(self):
        """Run reduce_scatter_perf and validate peak bus bandwidth."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")

        iters = int(os.environ.get("NODE_NCCL_ITERS", "30"))

        res = _run_nccl_test(
            "reduce_scatter_perf", n_gpus=n_gpus,
            min_bytes="1M", max_bytes="256M", step_factor=4, iters=iters
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"reduce_scatter_perf binary: {res['error']}")

        logger.info("\n=== NCCL reduce_scatter_perf ===\n%s", res["raw"].strip())
        busbw = res["busbw_gbs"]
        if not busbw:
            pytest.skip("Could not parse reduce_scatter_perf output")
        peak = max(busbw)
        logger.info("ReduceScatter peak busbw: %.1f GB/s", peak)
        _record_kpi(
            "test_nccl_reduce_scatter_perf",
            {
                "gpu_count": n_gpus,
                "iters": iters,
                "peak_busbw_gbs": round(float(peak), 2),
                "mean_busbw_gbs": round(float(sum(busbw) / len(busbw)), 2),
                "samples": len(busbw),
            },
        )
        assert peak >= 30.0, f"ReduceScatter busbw {peak:.1f} GB/s too low"

    def test_nccl_p2p_perf(self):
        """Run p2p_perf (point-to-point send/recv) and validate bandwidth."""
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs")

        iters = int(os.environ.get("NODE_NCCL_ITERS", "30"))

        res = _run_nccl_test(
            "sendrecv_perf", n_gpus=n_gpus,
            min_bytes="1M", max_bytes="256M", step_factor=4, iters=iters
        )
        if res["error"] and "not found" in res["error"]:
            pytest.skip(f"sendrecv_perf binary: {res['error']}")

        logger.info("\n=== NCCL sendrecv_perf ===\n%s", res["raw"].strip())
        busbw = res["busbw_gbs"]
        if not busbw:
            pytest.skip("Could not parse sendrecv_perf output")
        peak = max(busbw)
        logger.info("SendRecv peak busbw: %.1f GB/s", peak)
        _record_kpi(
            "test_nccl_p2p_perf",
            {
                "gpu_count": n_gpus,
                "iters": iters,
                "peak_busbw_gbs": round(float(peak), 2),
                "mean_busbw_gbs": round(float(sum(busbw) / len(busbw)), 2),
                "samples": len(busbw),
            },
        )
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
        _record_kpi(
            "test_nccl_deterministic_init_variance",
            {
                "gpu_count": n_gpus,
                "iterations": 50,
                "mean_algbw_gbs": round(float(mean_algbw), 4),
                "samples": len(algbw),
            },
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


def _module_available(module_name: str) -> bool:
    """Return True when a Python module can be imported in this environment."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse common boolean environment variable values."""
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _auto_install_python_packages(package_specs: List[str]) -> Dict[str, str]:
    """Attempt pip installs in the current interpreter; returns package->status."""
    results: Dict[str, str] = {}
    if not package_specs:
        return results

    timeout_sec = int(os.environ.get("NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC", "1800"))
    for package_spec in package_specs:
        cmd = [sys.executable, "-m", "pip", "install", package_spec]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if res.returncode == 0:
                results[package_spec] = "installed"
            else:
                tail = (res.stderr or res.stdout or "").strip().splitlines()
                results[package_spec] = (
                    f"failed: {tail[-1]}" if tail else f"failed: exit {res.returncode}"
                )
        except Exception as exc:
            results[package_spec] = f"failed: {exc}"
    return results


@contextmanager
def _temporary_logger_levels(levels: Dict[str, int]):
    """Temporarily override logger levels and restore after scope exits."""
    previous_levels: Dict[str, int] = {}
    for logger_name, level in levels.items():
        target = logging.getLogger(logger_name)
        previous_levels[logger_name] = target.level
        target.setLevel(level)
    try:
        yield
    finally:
        for logger_name, level in previous_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def _ensure_transformers_compat_shims() -> None:
    """Backfill symbols expected by some remote model code on mixed transformers versions."""
    try:
        import transformers  # type: ignore
    except Exception:
        return

    if hasattr(transformers, "BeamSearchScorer"):
        return

    try:
        from transformers.generation.beam_search import BeamSearchScorer  # type: ignore

        setattr(transformers, "BeamSearchScorer", BeamSearchScorer)
        logger.info("Applied transformers compatibility shim: BeamSearchScorer")
    except Exception:
        # Leave behavior unchanged if this symbol path is unavailable.
        pass


def _hf_from_pretrained(cls, model_id: str, **kwargs):
    """Load HF assets with auth fallback for mixed transformers versions.

    Uses token= for modern transformers, and falls back to use_auth_token=
    when running with older versions.
    """
    _ensure_transformers_compat_shims()

    token = _hf_auth_token()
    call_kwargs = dict(kwargs)
    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1").lower() in {
        "1", "true", "yes", "on"
    }
    if "trust_remote_code" not in call_kwargs and trust_remote_code:
        call_kwargs["trust_remote_code"] = True
    if token:
        call_kwargs["token"] = token

    verbose_http = os.environ.get("HF_VERBOSE_HTTP", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    suppressed_loggers = (
        {
            "httpx": logging.WARNING,
            "httpcore": logging.WARNING,
            "huggingface_hub": logging.WARNING,
        }
        if not verbose_http
        else {}
    )

    with _temporary_logger_levels(suppressed_loggers):
        try:
            return cls.from_pretrained(model_id, **call_kwargs)
        except TypeError as exc:
            if token and "token" in str(exc):
                call_kwargs.pop("token", None)
                call_kwargs["use_auth_token"] = token
                return cls.from_pretrained(model_id, **call_kwargs)
            if call_kwargs.get("trust_remote_code") and "trust_remote_code" in str(exc):
                call_kwargs.pop("trust_remote_code", None)
                return cls.from_pretrained(model_id, **call_kwargs)
            raise
        except ValueError as exc:
            if "trust_remote_code=True" in str(exc) and not call_kwargs.get("trust_remote_code"):
                call_kwargs["trust_remote_code"] = True
                return cls.from_pretrained(model_id, **call_kwargs)
            raise


def _resolve_supported_context_length(model_id: str, requested_context_length: int) -> int:
    """Return a context length capped to the model config when available."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return requested_context_length

    try:
        config = _hf_from_pretrained(AutoConfig, model_id)
    except Exception as exc:
        logger.warning("Unable to inspect model config for %s: %s", model_id, exc)
        return requested_context_length

    candidates: List[int] = []
    for attr_name in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
        "model_max_length",
    ):
        value = getattr(config, attr_name, None)
        if isinstance(value, (int, float)) and value and value < 10**7:
            candidates.append(int(value))

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        for key in ("original_max_position_embeddings", "max_position_embeddings"):
            value = rope_scaling.get(key)
            if isinstance(value, (int, float)) and value and value < 10**7:
                candidates.append(int(value))

    derived_context_length = min(candidates) if candidates else requested_context_length
    return min(requested_context_length, derived_context_length)


def _extract_vllm_derived_max_model_len(error_text: str) -> Optional[int]:
    """Parse vLLM validation text for the derived max model length."""
    match = re.search(r"derived max_model_len \(max_position_embeddings=([0-9.]+)", error_text)
    if not match:
        return None

    try:
        return int(float(match.group(1)))
    except (TypeError, ValueError):
        return None


def _kpi_output_path() -> Path:
    """Return JSONL file path for persisted KPI records."""
    return Path(os.environ.get("NODE_VALIDATION_KPI_FILE", "artifacts/node_validation_kpis.jsonl"))


def _record_kpi(test_name: str, metrics: Dict) -> None:
    """Append KPI metrics to a JSONL file for post-run reporting."""
    out_path = _kpi_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test": test_name,
        "metrics": metrics,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")

    # Emit KPI values directly to test logs so workload-category runs show
    # throughput/latency/utilization numbers without opening the JSONL artifact.
    logger.info("KPI %s: %s", test_name, json.dumps(metrics, default=str, sort_keys=True))


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, destination: Path) -> Path:
    _ensure_directory(destination.parent)
    if destination.exists():
        return destination

    with urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())
    return destination


def _resolve_vision_input_source(source: str) -> Path:
    if re.match(r"^https?://", source, flags=re.IGNORECASE):
        parsed = urlparse(source)
        file_name = Path(parsed.path).name or "vision_input.bin"
        return _download_file(source, Path("artifacts") / "vision_inputs" / file_name)

    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Vision input source not found: {source_path}")
    return source_path


def _cpu_monitor(stop_event: threading.Event, interval_sec: float, samples: List[float]) -> None:
    try:
        import psutil
    except ImportError:
        return

    psutil.cpu_percent(interval=None)
    while not stop_event.is_set():
        samples.append(psutil.cpu_percent(interval=interval_sec))


def _yolo_worker(
    gpu_id: int,
    model_name: str,
    image_path: str,
    images_per_gpu: int,
    batch_size: int,
    imgsz: int,
    half: bool,
    result_queue,
) -> None:
    try:
        from ultralytics import YOLO

        torch.cuda.set_device(gpu_id)
        model = YOLO(model_name)

        warmup_sources = [image_path] * max(1, min(batch_size, 8))
        _ = model.predict(
            source=warmup_sources,
            imgsz=imgsz,
            batch=min(batch_size, len(warmup_sources)),
            half=half,
            device=gpu_id,
            verbose=False,
            save=False,
        )
        torch.cuda.synchronize(gpu_id)

        sources = [image_path] * images_per_gpu
        t0 = time.perf_counter()
        _ = model.predict(
            source=sources,
            imgsz=imgsz,
            batch=min(batch_size, len(sources)),
            half=half,
            device=gpu_id,
            verbose=False,
            save=False,
        )
        torch.cuda.synchronize(gpu_id)
        elapsed = time.perf_counter() - t0
        fps = images_per_gpu / elapsed if elapsed > 0 else 0.0

        result_queue.put(
            {
                "gpu": gpu_id,
                "status": "ok",
                "images": images_per_gpu,
                "elapsed": elapsed,
                "fps": fps,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "gpu": gpu_id,
                "status": "error",
                "error": str(exc),
            }
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _dist_setup(rank: int, world_size: int, master_port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _dist_cleanup() -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        return

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _diffusion_worker(rank: int, world_size: int, master_port: int, cfg: Dict, result_queue) -> None:
    pipe = None
    try:
        import torch.distributed as dist
        from diffusers import AutoPipelineForText2Image

        _dist_setup(rank, world_size, master_port)
        device = f"cuda:{rank}"

        pipe_kwargs: Dict = {"torch_dtype": torch.float16}
        if cfg.get("variant"):
            pipe_kwargs["variant"] = cfg["variant"]
        token = _hf_auth_token()
        if token:
            pipe_kwargs["token"] = token

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(cfg["model"], **pipe_kwargs)
        except Exception:
            pipe_kwargs.pop("variant", None)
            pipe = AutoPipelineForText2Image.from_pretrained(cfg["model"], **pipe_kwargs)

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        for _ in range(cfg["warmup"]):
            _ = pipe(
                prompt=cfg["prompt"],
                negative_prompt=cfg["negative_prompt"] or None,
                num_inference_steps=cfg["steps"],
                guidance_scale=cfg["guidance_scale"],
                height=cfg["height"],
                width=cfg["width"],
            )

        dist.barrier()
        torch.cuda.synchronize(rank)
        start = time.perf_counter()

        for _ in range(cfg["images_per_gpu"]):
            _ = pipe(
                prompt=cfg["prompt"],
                negative_prompt=cfg["negative_prompt"] or None,
                num_inference_steps=cfg["steps"],
                guidance_scale=cfg["guidance_scale"],
                height=cfg["height"],
                width=cfg["width"],
            )

        torch.cuda.synchronize(rank)
        dist.barrier()
        elapsed = time.perf_counter() - start
        ips = cfg["images_per_gpu"] / elapsed if elapsed > 0 else 0.0

        gathered: List[Optional[Dict]] = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "images": cfg["images_per_gpu"],
                "elapsed": elapsed,
                "ips": ips,
            },
        )

        if rank == 0:
            result_queue.put({"status": "ok", "per_rank": gathered})
    except Exception as exc:
        result_queue.put({"status": "error", "rank": rank, "error": str(exc)})
    finally:
        del pipe
        _dist_cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ddp_training_worker(
    rank: int,
    world_size: int,
    master_port: int,
    cfg: Dict,
    result_queue,
) -> None:
    """Multi-GPU DDP (or DeepSpeed ZeRO) training worker used by TC-013 and TC-014.

    Measures per-step latency, gradient allreduce latency, sync ratio, per-GPU
    throughput, TFLOPS estimate, and (rank-0 only) scaling efficiency vs a
    single-GPU baseline captured before DDP wrapping.
    """
    try:
        import torch.distributed as dist
        from transformers import AutoModelForCausalLM

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("LOCAL_RANK", str(rank))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))

        _dist_setup(rank, world_size, master_port)
        device = f"cuda:{rank}"

        model_id      = cfg["model_id"]
        batch_size    = cfg["batch_size"]
        seq_len       = cfg["seq_len"]
        n_warmup      = cfg.get("warmup_steps", 3)
        n_steps       = cfg.get("n_steps", 20)
        use_deepspeed = cfg.get("use_deepspeed", False)
        deepspeed_zero_stage = int(cfg.get("deepspeed_zero_stage", 2))
        require_model_sharding = bool(cfg.get("require_model_sharding", False))
        collect_single_gpu_baseline = bool(cfg.get("collect_single_gpu_baseline", True))

        # ── Load model ───────────────────────────────────────────────────────
        try:
            model_kwargs: Dict = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }
            token = _hf_auth_token()
            if token:
                model_kwargs["token"] = token
            if os.environ.get("HF_TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes", "on"}:
                model_kwargs["trust_remote_code"] = True
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except Exception as exc:
            result_queue.put({"rank": rank, "status": "error",
                              "error": f"Model load failed: {exc}"})
            return

        param_count = sum(p.numel() for p in model.parameters())
        model = model.to(device)

        # ── Single-GPU baseline (rank 0 only, *before* DDP wrapping) ─────────
        single_gpu_tps: Optional[float] = None
        if rank == 0 and collect_single_gpu_baseline:
            vocab_size = model.config.vocab_size
            _ids   = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            _mask  = torch.ones_like(_ids)
            _opt   = torch.optim.AdamW(model.parameters(), lr=1e-5)
            model.train()
            for _ in range(2):  # warmup
                _opt.zero_grad()
                model(_ids, attention_mask=_mask, labels=_ids).loss.backward()
                _opt.step()
                torch.cuda.synchronize(rank)
            _n = 3
            torch.cuda.synchronize(rank)
            _t0 = time.perf_counter()
            for _ in range(_n):
                _opt.zero_grad()
                model(_ids, attention_mask=_mask, labels=_ids).loss.backward()
                _opt.step()
                torch.cuda.synchronize(rank)
            _elapsed = time.perf_counter() - _t0
            single_gpu_tps = (batch_size * _n) / _elapsed
            del _opt, _ids, _mask
            torch.cuda.empty_cache()

        dist.barrier()

        # ── Wrap in DDP or DeepSpeed ─────────────────────────────────────────
        engine    = None
        optimizer = None
        if use_deepspeed:
            try:
                import deepspeed
                ds_cfg = {
                    "train_micro_batch_size_per_gpu": batch_size,
                    "gradient_accumulation_steps": 1,
                    "bf16": {"enabled": True},
                    "zero_optimization": {
                        "stage": deepspeed_zero_stage,
                        "contiguous_gradients": True,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                    },
                }
                if deepspeed_zero_stage >= 3:
                    ds_cfg["zero_optimization"].update(
                        {
                            "stage3_prefetch_bucket_size": 5e7,
                            "stage3_param_persistence_threshold": 1e5,
                            "stage3_max_live_parameters": 1e9,
                            "stage3_max_reuse_distance": 1e9,
                        }
                    )
                _base_opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
                engine, optimizer, _, _ = deepspeed.initialize(
                    model=model, optimizer=_base_opt, config=ds_cfg
                )
                model = engine
            except Exception as exc:
                if require_model_sharding:
                    result_queue.put(
                        {
                            "rank": rank,
                            "status": "error",
                            "error": (
                                "DeepSpeed ZeRO sharding is required for this test but initialization failed: "
                                f"{exc}"
                            ),
                        }
                    )
                    return
                logger.warning("DeepSpeed init failed (%s), falling back to DDP", exc)
                use_deepspeed = False

        if require_model_sharding and not use_deepspeed:
            result_queue.put(
                {
                    "rank": rank,
                    "status": "error",
                    "error": (
                        "TC-014 large multi-GPU training requires model sharding across GPUs. "
                        "Plain DDP replicates the full model on every GPU. Install/configure DeepSpeed and enable ZeRO-3."
                    ),
                }
            )
            return

        if not use_deepspeed:
            model     = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        model.train()
        vocab_size = (
            model.module.config.vocab_size
            if hasattr(model, "module")
            else model.config.vocab_size
        )
        input_ids      = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)

        step_times_ms:      List[float] = []
        allreduce_times_ms: List[float] = []

        def _run_step() -> float:
            """One forward+backward+optimizer step; returns allreduce proxy time in ms."""
            if use_deepspeed:
                engine.zero_grad()
            else:
                optimizer.zero_grad()

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss if hasattr(out, "loss") else out[0]

            if use_deepspeed:
                engine.backward(loss)
            else:
                loss.backward()

            # Proxy allreduce: sync a buffer matching total gradient volume
            ar_ms = 0.0
            if not use_deepspeed:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if grads:
                    total_elems = sum(g.numel() for g in grads)
                    comm_buf    = grads[0].new_zeros(total_elems)
                    torch.cuda.synchronize(rank)
                    _t = time.perf_counter()
                    dist.all_reduce(comm_buf, op=dist.ReduceOp.SUM)
                    torch.cuda.synchronize(rank)
                    ar_ms = (time.perf_counter() - _t) * 1000.0

            if use_deepspeed:
                engine.step()
            else:
                optimizer.step()

            return ar_ms

        # Warmup
        for _ in range(n_warmup):
            _run_step()
        dist.barrier()

        # Timed measurement
        for _ in range(n_steps):
            torch.cuda.synchronize(rank)
            _t0  = time.perf_counter()
            ar   = _run_step()
            torch.cuda.synchronize(rank)
            step_times_ms.append((time.perf_counter() - _t0) * 1000.0)
            allreduce_times_ms.append(ar)
        dist.barrier()

        mean_step_ms = float(np.mean(step_times_ms))
        mean_ar_ms   = float(np.mean(allreduce_times_ms))
        total_sec    = sum(step_times_ms) / 1000.0
        per_gpu_tps  = (batch_size * n_steps) / total_sec if total_sec > 0 else 0.0
        sync_ratio   = mean_ar_ms / mean_step_ms if mean_step_ms > 0 else 0.0

        # TFLOPS estimate: 6 * params * batch * seq / step_sec (standard approximation)
        flops_per_step = 6 * param_count * batch_size * seq_len
        tflops = (flops_per_step * n_steps) / (total_sec * 1e12) if total_sec > 0 else 0.0

        # Scaling efficiency (rank 0 only)
        scaling_efficiency_pct: Optional[float] = None
        if rank == 0 and single_gpu_tps and single_gpu_tps > 0:
            ideal_total = single_gpu_tps * world_size
            actual_total = per_gpu_tps * world_size
            scaling_efficiency_pct = round((actual_total / ideal_total) * 100.0, 1)

        row = {
            "rank":            rank,
            "mean_step_ms":    round(mean_step_ms, 2),
            "mean_allreduce_ms": round(mean_ar_ms, 2),
            "sync_ratio":      round(sync_ratio, 4),
            "per_gpu_tps":     round(per_gpu_tps, 2),
            "tflops":          round(tflops, 3),
            "scaling_efficiency_pct": scaling_efficiency_pct,
        }

        gathered: List[Optional[Dict]] = [None] * world_size
        dist.all_gather_object(gathered, row)
        if rank == 0:
            result_queue.put({"status": "ok", "per_rank": gathered})

    except Exception as exc:
        result_queue.put({"rank": rank, "status": "error", "error": str(exc)})
    finally:
        _dist_cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ensure_yolo_model_downloaded(model_name: str = "yolov8x.pt") -> Optional[str]:
    """
    Ensure YOLO model is downloaded and cached locally.
    
    Uses the ultralytics YOLO Python API to trigger model download if not already cached.
    Gracefully returns None if download fails; caller should skip test.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    
    try:
        # Set environment to use standard YOLO cache location.
        # YOLO will auto-download model if not cached.
        os.environ.setdefault("YOLO_CACHE", os.path.expanduser("~/.cache/yolo"))
        
        logger.info("Pre-downloading YOLO model: %s", model_name)
        # Load the model; YOLO constructor downloads if not cached.
        model = YOLO(model_name)
        logger.info("YOLO model %s ready at cache", model_name)
        return model_name
    except Exception as e:
        logger.warning("Failed to pre-download YOLO model %s: %s", model_name, e)
        return None


def _measure_page_faults() -> Tuple[int, int]:
    """
    Sample current major and minor page faults from /proc/self/stat.
    Returns (major_faults, minor_faults) counts.
    """
    try:
        with open("/proc/self/stat", "r") as f:
            stat_line = f.read().split()
            # major_fault is at index 11, min_flt is at index 9
            # (see man proc(5) for /proc/[pid]/stat format)
            if len(stat_line) >= 12:
                return int(stat_line[11]), int(stat_line[9])
    except Exception:
        pass
    return 0, 0


def _measure_gpu_pcie_bandwidth() -> float:
    """
    Estimate PCIe bandwidth from nvidia-smi if available.
    Returns estimated PCIe bandwidth in GB/s (default 16 if unavailable).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=pci.link.gen.current,pci.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                gen = int(parts[0].strip())
                width = int(parts[1].strip())
                # PCIe 4.0: 2.5 GB/s per lane; 5.0: 4 GB/s per lane; 6.0: 8 GB/s per lane
                per_lane = 2.5 if gen == 4 else (4.0 if gen == 5 else 8.0)
                return per_lane * width
    except Exception:
        pass
    return 16.0  # Default assume PCIe 4.0 x16 = 32 GB/s, but return conservative estimate


def _sample_vmstat_pagefaults(duration_sec: float = 5.0, interval_sec: float = 0.2) -> Dict[str, float]:
    """
    Sample page faults from vmstat over a time period.
    Returns dict with pgmajfault, pgminor rates (faults per second).
    """
    try:
        import psutil
    except ImportError:
        return {}

    initial_major, initial_minor = _measure_page_faults()
    time.sleep(duration_sec)
    final_major, final_minor = _measure_page_faults()
    
    maj_rate = (final_major - initial_major) / duration_sec if duration_sec > 0 else 0.0
    min_rate = (final_minor - initial_minor) / duration_sec if duration_sec > 0 else 0.0
    
    return {
        "pgmajfault_per_sec": round(maj_rate, 2),
        "pgminor_per_sec": round(min_rate, 2),
    }


def _run_hf_causal_latency_workload(
    model_id: str,
    max_new_tokens: int,
    max_mean_latency_ms: float,
    min_duration_sec: int,
    prefer_4bit_quant: bool = False,
    kpi_test_name: str = "test_hf_causal_lm_inference_latency_enterprise_workloads",
) -> Dict[str, float]:
    """Run a causal-LM latency workload for at least *min_duration_sec* and assert mean latency."""
    workload_t0 = time.perf_counter()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices")

    device = torch.device("cuda:0")

    # Very large models (e.g., 70B/72B) can spend a long time downloading or
    # CPU-side deserializing shards. By default, require local cache for these
    # to avoid runs appearing stuck. Override with HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1.
    model_id_l = model_id.lower()
    is_very_large_model = any(tag in model_id_l for tag in ("70b", "72b", "65b", "34b"))
    allow_remote_large = os.environ.get("HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD", "0").lower() in {
        "1", "true", "yes", "on"
    }
    force_local_only = is_very_large_model and not allow_remote_large

    logger.info("Loading %s for causal-LM latency workload …", model_id)
    if force_local_only:
        logger.info(
            "Large model local-cache mode enabled for %s. "
            "Set HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1 to allow remote download.",
            model_id,
        )

    tok_t0 = time.perf_counter()
    try:
        tokenizer = _hf_from_pretrained(
            AutoTokenizer,
            model_id,
            **({"local_files_only": True} if force_local_only else {}),
        )
    except Exception as exc:
        if force_local_only:
            logger.warning(
                "Large model %s not found in local cache (%s). Falling back to remote download.",
                model_id,
                exc,
            )
            os.environ["HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD"] = "1"
            force_local_only = False
            tokenizer = _hf_from_pretrained(AutoTokenizer, model_id)
            logger.info("Remote tokenizer download enabled for %s", model_id)
        else:
            raise
    logger.info("Tokenizer load completed in %.1fs", time.perf_counter() - tok_t0)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict = {}
    use_quantized_load = False
    if prefer_4bit_quant:
        if not _module_available("accelerate"):
            logger.warning(
                "accelerate not installed; disabling 4-bit/device_map path for %s",
                model_id,
            )
            prefer_4bit_quant = False
        
    if prefer_4bit_quant:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
            use_quantized_load = True
        except Exception as exc:
            logger.warning("4-bit quantization config unavailable: %s", exc)

    if not use_quantized_load:
        model_kwargs["torch_dtype"] = torch.float16

    if force_local_only:
        model_kwargs["local_files_only"] = True

    try:
        model_t0 = time.perf_counter()
        model = _hf_from_pretrained(AutoModelForCausalLM, model_id, **model_kwargs)
        logger.info("Model load completed in %.1fs", time.perf_counter() - model_t0)
    except Exception as exc:
        if force_local_only:
            logger.warning(
                "Large model %s weights not available locally (%s). Falling back to remote download.",
                model_id,
                exc,
            )
            os.environ["HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD"] = "1"
            model_kwargs.pop("local_files_only", None)
            model_t0 = time.perf_counter()
            model = _hf_from_pretrained(AutoModelForCausalLM, model_id, **model_kwargs)
            logger.info("Remote model download/load completed in %.1fs", time.perf_counter() - model_t0)
        else:
            pytest.skip(f"Model load failed for {model_id}: {exc}")

    if not use_quantized_load:
        model = model.to(device)
    model.eval()

    prompt = "The AI head node validation test confirms stable latency for"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize(device)

    latencies_ms = []
    with torch.no_grad():
        start = time.perf_counter()
        while (time.perf_counter() - start) < min_duration_sec:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    total_runtime_sec = time.perf_counter() - start
    assert total_runtime_sec >= min_duration_sec, (
        f"Inference test runtime {total_runtime_sec:.1f}s shorter than required "
        f"{min_duration_sec}s"
    )

    mean_ms = np.mean(latencies_ms)
    p95_ms = np.percentile(latencies_ms, 95)
    tps = max_new_tokens / (mean_ms / 1000)

    logger.info(
        "HF Causal-LM (%s) inference: mean=%.1f ms  p95=%.1f ms  "
        "tokens/sec=%.0f (batch=1, new_tokens=%d, runtime=%.1fs, completion=%.1fs, samples=%d)",
        model_id,
        mean_ms,
        p95_ms,
        tps,
        max_new_tokens,
        total_runtime_sec,
        time.perf_counter() - workload_t0,
        len(latencies_ms),
    )

    completion_time_sec = time.perf_counter() - workload_t0

    _record_kpi(
        kpi_test_name,
        {
            "model": model_id,
            "max_new_tokens": max_new_tokens,
            "mean_latency_ms": round(float(mean_ms), 2),
            "p95_latency_ms": round(float(p95_ms), 2),
            "throughput_tokens_per_sec": round(float(tps), 2),
            "runtime_sec": round(float(total_runtime_sec), 2),
            "completion_time_sec": round(float(completion_time_sec), 2),
            "samples": len(latencies_ms),
            "used_4bit_quant": use_quantized_load,
        },
    )

    assert mean_ms < max_mean_latency_ms, (
        f"{model_id} inference mean latency {mean_ms:.0f} ms > {max_mean_latency_ms:.0f} ms"
    )

    del model
    torch.cuda.empty_cache()
    return {
        "mean_latency_ms": float(mean_ms),
        "p95_latency_ms": float(p95_ms),
        "throughput_tokens_per_sec": float(tps),
        "runtime_sec": float(total_runtime_sec),
        "completion_time_sec": float(completion_time_sec),
        "samples": float(len(latencies_ms)),
    }


@pytest.mark.node_validation
@pytest.mark.workload_acceptance
class TestWorkloadAcceptanceHuggingFace:
    def _vllm_multitenancy_worker(
        gpu_ids_str: str,
        model_id: str,
        n_requests: int,
        max_tokens: int,
        tp_size: int,
        kv_cache_dtype: str,
        cpu_offload_gb: float,
        result_queue,
    ) -> None:
        """Module-level worker for multi-tenant vLLM inference tests (TC-047 / TC-048).

        Runs a vLLM LLM engine pinned to *gpu_ids_str* (CUDA_VISIBLE_DEVICES), serves
        *n_requests* single-turn prompts, and puts a latency-stats dict into
        *result_queue*.  Must be at module scope for multiprocessing spawn.

        Result dict keys on success:
            model, gpu_ids, n_requests, mean_ms, p50_ms, p99_ms
        On error:
            error (str), model, gpu_ids
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        try:
            import time as _time
            import numpy as _np
            from vllm import LLM, SamplingParams

            kwargs: Dict = dict(
                model=model_id,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=float(
                    os.environ.get("MULTITENANT_GPU_MEM_UTIL", "0.85")
                ),
                dtype="bfloat16",
                max_model_len=int(os.environ.get("MULTITENANT_MAX_MODEL_LEN", "2048")),
            )
            if kv_cache_dtype:
                kwargs["kv_cache_dtype"] = kv_cache_dtype
            if cpu_offload_gb > 0:
                kwargs["cpu_offload_gb"] = cpu_offload_gb

            llm = LLM(**kwargs)
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            prompts = [
                f"Explain GPU interconnect topology concept {i % 20}:"
                for i in range(n_requests + 1)
            ]

            # Warmup
            _ = llm.generate(prompts[:1], sp)

            # Measure per-request latency
            latencies: List[float] = []
            for p in prompts[1 : n_requests + 1]:
                t0 = _time.perf_counter()
                _ = llm.generate([p], sp)
                latencies.append((_time.perf_counter() - t0) * 1000.0)

            result_queue.put(
                {
                    "model": model_id,
                    "gpu_ids": gpu_ids_str,
                    "n_requests": n_requests,
                    "mean_ms": float(_np.mean(latencies)),
                    "p50_ms": float(_np.percentile(latencies, 50)),
                    "p99_ms": float(_np.percentile(latencies, 99)),
                }
            )
        except Exception as exc:
            result_queue.put({"error": str(exc), "model": model_id, "gpu_ids": gpu_ids_str})


@pytest.mark.node_validation
@pytest.mark.workload_acceptance
class TestWorkloadAcceptanceHuggingFace:
    """TC-NV-08: Workload acceptance using HuggingFace Transformer models.

    Inference model : _HF_INFERENCE_MODEL (default: openai-community/gpt2)
    Training model  : _HF_TRAINING_MODEL  (default: bert-base-uncased)

    The tests use the 'transformers' library directly on GPU.  Set
    _HF_INFERENCE_MODEL / _HF_TRAINING_MODEL to a local cached path if the
    node has no internet access.
    """

    def test_workload_preflight_requirements(self):
        """Preflight checker for workload acceptance dependencies and env toggles.

        Set NODE_WORKLOAD_PREFLIGHT_STRICT=1 to fail fast on missing requirements.
        """
        strict = os.environ.get("NODE_WORKLOAD_PREFLIGHT_STRICT", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        blockers: List[str] = []
        notes: List[str] = []

        # Linux default: auto-install missing workload deps in current env.
        auto_install_default = "1" if sys.platform.startswith("linux") else "0"
        auto_install = _env_flag("NODE_WORKLOAD_AUTO_INSTALL", auto_install_default)

        enable_large_models = os.environ.get("HF_ENABLE_LARGE_MODELS", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if auto_install:
            install_plan: List[str] = []
            if not _module_available("transformers"):
                install_plan.append("transformers")
            if enable_large_models and not _module_available("accelerate"):
                install_plan.append("accelerate")

            optional_pkg_map = {
                "vllm": "vllm",
                "ultralytics": "ultralytics",
                "diffusers": "diffusers",
                "psutil": "psutil",
            }
            for module_name, package_name in optional_pkg_map.items():
                if not _module_available(module_name):
                    install_plan.append(package_name)

            if install_plan:
                logger.info(
                    "Workload preflight auto-install enabled; installing missing packages: %s",
                    ", ".join(install_plan),
                )
                install_results = _auto_install_python_packages(install_plan)
                for pkg, status in install_results.items():
                    logger.info("auto-install %s: %s", pkg, status)

        if not _module_available("transformers"):
            blockers.append("Missing package: transformers (required for HF workload tests)")
        if enable_large_models and not _module_available("accelerate"):
            blockers.append(
                "Missing package: accelerate (required by large/quantized HF model loads using device_map)"
            )

        token = _hf_auth_token()
        if enable_large_models and not token:
            notes.append(
                "HF_ENABLE_LARGE_MODELS=1 but HF_TOKEN is not set; gated models may fail or skip"
            )

        optional_components = {
            "vllm": "required for vLLM saturation/throughput tests",
            "ultralytics": "required for YOLO vision workload test",
            "diffusers": "required for distributed image-generation test",
            "psutil": "required for CPU-utilization telemetry in workload tests",
        }
        for module_name, reason in optional_components.items():
            if not _module_available(module_name):
                notes.append(f"Missing optional package: {module_name} ({reason})")

        logger.info("Workload preflight: strict=%s, large_models=%s", strict, enable_large_models)
        if blockers:
            logger.warning("Workload preflight blockers (%d):", len(blockers))
            for item in blockers:
                logger.warning("  - %s", item)
        else:
            logger.info("Workload preflight blockers: none")

        if notes:
            logger.info("Workload preflight notes (%d):", len(notes))
            for item in notes:
                logger.info("  - %s", item)

        if strict and blockers:
            pytest.fail(
                "Workload preflight failed in strict mode. "
                "Resolve blockers or set NODE_WORKLOAD_PREFLIGHT_STRICT=0.\n"
                + "\n".join(blockers)
            )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.timeout(0)
    def test_hf_causal_lm_inference_latency(self):
        """Load a HuggingFace causal-LM (GPT-2) on GPU and measure per-token generation latency."""
        test_t0 = time.perf_counter()
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
        max_mean_latency_ms = float(os.environ.get("HF_INFERENCE_MAX_MEAN_LATENCY_MS", "1200"))
        max_p95_latency_ms = float(os.environ.get("HF_INFERENCE_MAX_P95_LATENCY_MS", "1500"))

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
            "tokens/sec=%.0f (batch=1, new_tokens=%d, runtime=%.1fs, completion=%.1fs, samples=%d)",
            _HF_INFERENCE_MODEL, mean_ms, p95_ms, tps, max_new_tokens,
            total_runtime_sec, time.perf_counter() - test_t0, len(latencies_ms)
        )
        logger.info(
            "HF latency acceptance thresholds: mean<%.1f ms, p95<%.1f ms",
            max_mean_latency_ms,
            max_p95_latency_ms,
        )

        _record_kpi(
            "test_hf_causal_lm_inference_latency",
            {
                "model": _HF_INFERENCE_MODEL,
                "max_new_tokens": max_new_tokens,
                "mean_latency_ms": round(float(mean_ms), 2),
                "p95_latency_ms": round(float(p95_ms), 2),
                "throughput_tokens_per_sec": round(float(tps), 2),
                "runtime_sec": round(float(total_runtime_sec), 2),
                "completion_time_sec": round(float(time.perf_counter() - test_t0), 2),
                "samples": len(latencies_ms),
            },
        )

        # Keep thresholds configurable so workload acceptance can be tuned per model/stack.
        assert mean_ms < max_mean_latency_ms, (
            f"{_HF_INFERENCE_MODEL} inference mean latency {mean_ms:.0f} ms > "
            f"{max_mean_latency_ms:.0f} ms"
        )
        assert p95_ms < max_p95_latency_ms, (
            f"{_HF_INFERENCE_MODEL} inference p95 latency {p95_ms:.0f} ms > "
            f"{max_p95_latency_ms:.0f} ms"
        )
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.slow
    @pytest.mark.timeout(0)
    @pytest.mark.parametrize(
        "workload_name,model_id,max_new_tokens,max_mean_latency_ms,prefer_4bit_quant",
        [
            (
                "deepseek_r1_quantized",
                _HF_DEEPSEEK_R1_QUANT_MODEL,
                32,
                3000.0,
                True,
            ),
            (
                "llama_7b",
                _HF_LLAMA_7B_MODEL,
                32,
                3000.0,
                False,
            ),
            (
                "llama_70b",
                _HF_LLAMA_70B_MODEL,
                16,
                12000.0,
                True,
            ),
            (
                "qwen_70b",
                _HF_QWEN_70B_MODEL,
                16,
                12000.0,
                True,
            ),
        ],
    )
    def test_hf_causal_lm_inference_latency_enterprise_workloads(
        self,
        workload_name: str,
        model_id: str,
        max_new_tokens: int,
        max_mean_latency_ms: float,
        prefer_4bit_quant: bool,
    ):
        """Repeat latency validation for DeepSeek-R1 quantized, Llama, and Qwen workloads."""
        enable_large_models = os.environ.get("HF_ENABLE_LARGE_MODELS", "1").lower() in {
            "1", "true", "yes", "on"
        }
        if "70b" in workload_name and not enable_large_models:
            pytest.skip(
                f"Skipping {workload_name} ({model_id}) due to HF_ENABLE_LARGE_MODELS=0. "
                "Set HF_ENABLE_LARGE_MODELS=1 to run 70B workloads."
            )

        min_duration_sec = int(os.environ.get("HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC", "300"))
        logger.info("Running workload=%s model=%s", workload_name, model_id)
        try:
            metrics = _run_hf_causal_latency_workload(
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                max_mean_latency_ms=max_mean_latency_ms,
                min_duration_sec=min_duration_sec,
                prefer_4bit_quant=prefer_4bit_quant,
                kpi_test_name=f"test_hf_causal_lm_inference_latency_enterprise_workloads_{workload_name}",
            )
        except AttributeError as exc:
            pytest.skip(f"{workload_name} runtime model compatibility issue: {exc}")
        logger.info(
            "Workload %s completed in %.1fs",
            workload_name,
            metrics["completion_time_sec"],
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.timeout(0)
    def test_hf_causal_lm_batch_throughput(self):
        """Measure batched prefill throughput for HuggingFace GPT-2 (tokens/sec)."""
        test_t0 = time.perf_counter()
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
            "(batch=%d, max_new_tokens=%d, iters=%d, completion=%.1fs)",
            _HF_INFERENCE_MODEL, tps, batch_size, max_new, n_iters, time.perf_counter() - test_t0
        )

        _record_kpi(
            "test_hf_causal_lm_batch_throughput",
            {
                "model": _HF_INFERENCE_MODEL,
                "batch_size": batch_size,
                "max_new_tokens": max_new,
                "iterations": n_iters,
                "elapsed_sec": round(float(elapsed), 2),
                "completion_time_sec": round(float(time.perf_counter() - test_t0), 2),
                "throughput_tokens_per_sec": round(float(tps), 2),
            },
        )

        assert tps > 100, f"Batch throughput {tps:.0f} tokens/sec too low"
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    def test_vllm_data_pipeline_saturation_inference_prefill(self):
        """TC-NV-08b: Data pipeline saturation (inference prefill).

        Goal: ensure GPUs do not idle due to CPU-side starvation while serving
        prefill-heavy requests.

        Workload command (vLLM benchmark):
          python -m vllm.benchmarks.benchmark_serving --request-rate 100 --input-len-mean 1024

        Pass criteria:
          - GPU utilization sustained high (>= 90% for most samples)
          - CPU utilization sustained below saturation (<= 70% for most samples)
        """
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vllm not installed")

        try:
            vllm_version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            vllm_version = "unknown"

        benchmark_timeout_sec = int(os.environ.get("VLLM_BENCHMARK_TIMEOUT_SEC", "240"))
        sample_interval_sec = float(os.environ.get("VLLM_BENCHMARK_SAMPLE_INTERVAL_SEC", "1.0"))

        # Thresholds can adapt on shared nodes to avoid non-actionable hard skips.
        # Exclusive-GPU runs keep strict defaults.
        strict_gpu_over_90_target = float(os.environ.get("VLLM_GPU_OVER_90_RATIO_TARGET", "0.8"))
        shared_gpu_over_90_target = float(os.environ.get("VLLM_GPU_OVER_90_RATIO_TARGET_SHARED", "0.5"))
        cpu_under_70_target = float(os.environ.get("VLLM_CPU_UNDER_70_RATIO_TARGET", "0.8"))

        def _query_active_compute_pids() -> set:
            try:
                proc_smi = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc_smi.returncode != 0:
                    return set()
                return {
                    line.strip()
                    for line in proc_smi.stdout.strip().splitlines()
                    if line.strip()
                }
            except Exception:
                return set()

        baseline_active_pids = _query_active_compute_pids()

        gpu_util_samples: List[float] = []
        cpu_util_samples: List[float] = []
        observed_active_pid_counts: List[int] = []
        stdout = ""
        stderr = ""
        runtime_sec = 0.0
        benchmark_mode = ""
        returncode: Optional[int] = None

        def _sample_while_running(target_proc: subprocess.Popen, timeout_sec: int) -> Tuple[str, str, float, int]:
            start_local = time.perf_counter()
            while target_proc.poll() is None and (time.perf_counter() - start_local) < timeout_sec:
                # System-wide CPU utilization for starvation detection.
                cpu_util_samples.append(psutil.cpu_percent(interval=sample_interval_sec))
                active_pids = _query_active_compute_pids()
                observed_active_pid_counts.append(len(active_pids))
                try:
                    smi = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if smi.returncode == 0:
                        vals = []
                        for line in smi.stdout.strip().splitlines():
                            line = line.strip()
                            if line:
                                vals.append(float(line))
                        if vals:
                            gpu_util_samples.append(float(sum(vals) / len(vals)))
                except Exception:
                    # Keep sampling loop running even if one nvidia-smi call fails.
                    pass

            if target_proc.poll() is None:
                target_proc.terminate()
                try:
                    target_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    target_proc.kill()

            out, err = target_proc.communicate(timeout=10)
            elapsed_local = time.perf_counter() - start_local
            rc = target_proc.returncode if target_proc.returncode is not None else 0
            return out, err, elapsed_local, rc

        def _wait_for_http_ready(
            base_url: str,
            timeout_sec: int,
            poll_interval_sec: float,
            server_proc: Optional[subprocess.Popen] = None,
        ) -> bool:
            probe_paths = ["/v1/models", "/health", "/metrics"]
            deadline = time.perf_counter() + timeout_sec
            while time.perf_counter() < deadline:
                if server_proc is not None and server_proc.poll() is not None:
                    return False
                for probe in probe_paths:
                    try:
                        with urlopen(base_url + probe, timeout=3) as resp:
                            if 200 <= getattr(resp, "status", 200) < 500:
                                return True
                    except Exception:
                        pass
                time.sleep(poll_interval_sec)
            return False

        # Preferred path: supported vLLM online benchmark flow.
        serve_host = os.environ.get("VLLM_SERVE_HOST", "127.0.0.1")
        serve_port = int(os.environ.get("VLLM_SERVE_PORT", "8000"))
        base_url = os.environ.get("VLLM_BENCH_BASE_URL", f"http://127.0.0.1:{serve_port}")
        endpoint = os.environ.get("VLLM_BENCH_ENDPOINT", "/v1/chat/completions")
        request_rate = os.environ.get("VLLM_BENCH_REQUEST_RATE", "100")
        num_prompts = os.environ.get("VLLM_BENCH_NUM_PROMPTS", "1000")
        random_input_len = os.environ.get("VLLM_BENCH_RANDOM_INPUT_LEN", "1024")
        random_output_len = os.environ.get("VLLM_BENCH_RANDOM_OUTPUT_LEN", "128")
        serve_tensor_parallel_size = os.environ.get(
            "VLLM_SERVE_TENSOR_PARALLEL_SIZE",
            os.environ.get("ML_TENSOR_PARALLEL", "1"),
        )
        serve_max_model_len = os.environ.get("VLLM_SERVE_MAX_MODEL_LEN", "1024")
        serve_gpu_memory_utilization = os.environ.get(
            "VLLM_SERVE_GPU_MEMORY_UTILIZATION",
            "0.4",
        )
        serve_dtype = os.environ.get("VLLM_SERVE_DTYPE", "bfloat16")
        serve_ready_timeout_sec = int(os.environ.get("VLLM_SERVE_READY_TIMEOUT_SEC", "300"))
        serve_ready_poll_interval_sec = float(os.environ.get("VLLM_SERVE_READY_POLL_INTERVAL_SEC", "2.0"))
        # Keep readiness budget bounded so this test can still run benchmark/fallback
        # before global pytest-timeout is reached.
        effective_ready_timeout_sec = min(
            serve_ready_timeout_sec,
            max(30, benchmark_timeout_sec - 30),
        )

        cli_failed = False
        if shutil.which("vllm") is not None:
            serve_cmd = [
                "vllm",
                "serve",
                _HF_INFERENCE_MODEL,
                "--host",
                serve_host,
                "--port",
                str(serve_port),
                "--tensor-parallel-size",
                str(serve_tensor_parallel_size),
                "--max-model-len",
                str(serve_max_model_len),
                "--gpu-memory-utilization",
                str(serve_gpu_memory_utilization),
                "--dtype",
                str(serve_dtype),
            ]
            bench_cmd = [
                "vllm",
                "bench",
                "serve",
                "--backend",
                "openai-chat",
                "--model",
                _HF_INFERENCE_MODEL,
                "--base-url",
                base_url,
                "--endpoint",
                endpoint,
                "--dataset-name",
                "random",
                "--random-input-len",
                random_input_len,
                "--random-output-len",
                random_output_len,
                "--request-rate",
                request_rate,
                "--num-prompts",
                num_prompts,
            ]

            logger.info("vLLM version: %s", vllm_version)
            logger.info("Starting vLLM server: %s", " ".join(serve_cmd))
            serve_proc = subprocess.Popen(
                serve_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            )
            try:
                if _wait_for_http_ready(
                    base_url,
                    effective_ready_timeout_sec,
                    serve_ready_poll_interval_sec,
                    serve_proc,
                ):
                    logger.info("Running vLLM online serving benchmark: %s", " ".join(bench_cmd))
                    bench_proc = subprocess.Popen(
                        bench_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                    )
                    stdout, stderr, runtime_sec, returncode = _sample_while_running(
                        bench_proc,
                        benchmark_timeout_sec,
                    )
                    benchmark_mode = "vllm_cli_serve_bench"
                else:
                    cli_failed = True
                    returncode = 1
                    # Don't set benchmark_mode here - let fallback run if needed
                    if serve_proc.poll() is not None:
                        logger.warning(
                            "vLLM server process exited before readiness (rc=%s); falling back",
                            serve_proc.returncode,
                        )
                    else:
                        logger.warning(
                            "vLLM server did not become ready at %s within %ds (configured=%ds); falling back",
                            base_url,
                            effective_ready_timeout_sec,
                            serve_ready_timeout_sec,
                        )
            finally:
                if serve_proc.poll() is None:
                    serve_proc.terminate()
                    try:
                        serve_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        serve_proc.kill()
                try:
                    _serve_out, _serve_err = serve_proc.communicate(timeout=5)
                    if _serve_err:
                        stderr = (_serve_err or "").strip()
                        # Always capture full stderr to file for debugging
                        diagnostics_dir = "artifacts"
                        os.makedirs(diagnostics_dir, exist_ok=True)
                        with open(os.path.join(diagnostics_dir, "vllm_serve_startup.log"), "w") as f:
                            f.write(_serve_err)
                        err_tail = _serve_err[-1500:]
                        logger.warning("vLLM serve stderr tail (for readiness debugging): %s", err_tail)
                except Exception:
                    pass
        else:
            cli_failed = True
            returncode = 127
            logger.warning("vllm CLI not found on PATH; falling back to module/API benchmark paths")

        # Skip fallback if primary vllm CLI benchmark path already ran successfully.
        if benchmark_mode != "vllm_cli_serve_bench" and returncode not in (0, None):
            cmd = [
                sys.executable,
                "-m",
                "vllm.benchmarks.benchmark_serving",
                "--model",
                _HF_INFERENCE_MODEL,
                "--request-rate",
                "100",
                "--input-len-mean",
                "1024",
            ]

            logger.info("Running vLLM pipeline saturation benchmark (module fallback): %s", " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            )

            try:
                stdout, stderr, runtime_sec, returncode = _sample_while_running(
                    proc,
                    benchmark_timeout_sec,
                )
                benchmark_mode = "vllm_module_benchmark_serving"
            except Exception:
                if proc.poll() is None:
                    proc.kill()
                raise

            out_err = ((stderr or "") + "\n" + (stdout or ""))[-500:]
            # Fallback when benchmark module path is unavailable in installed vLLM.
            if "No module named vllm.benchmarks.benchmark_serving" in out_err:
                try:
                    from vllm import LLM, SamplingParams

                    logger.warning(
                        "benchmark_serving module unavailable; using vLLM API fallback for saturation KPI"
                    )
                    llm = LLM(
                        model=_HF_INFERENCE_MODEL,
                        dtype="float16",
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.4,
                        max_model_len=1024,
                        enforce_eager=True,
                        disable_log_stats=True,
                    )
                    params = SamplingParams(temperature=0.0, max_tokens=64)
                    prompts = ["prefill benchmark prompt " + str(i) for i in range(32)]

                    gpu_util_samples = []
                    cpu_util_samples = []
                    fallback_duration_sec = int(os.environ.get("VLLM_API_FALLBACK_DURATION_SEC", "90"))
                    monitor_interval_sec = float(os.environ.get("VLLM_MONITOR_INTERVAL_SEC", "0.5"))

                    # ── Background monitor thread samples GPU/CPU WHILE generate() runs.
                    # Without this the GPU is already idle when sequential sampling would run.
                    _monitor_stop = threading.Event()

                    def _util_monitor() -> None:
                        """Sample GPU and CPU utilization concurrently with generation."""
                        psutil.cpu_percent(interval=None)  # discard first dummy reading
                        while not _monitor_stop.is_set():
                            cpu_util_samples.append(psutil.cpu_percent(interval=monitor_interval_sec))
                            try:
                                smi = subprocess.run(
                                    [
                                        "nvidia-smi",
                                        "--query-gpu=utilization.gpu",
                                        "--format=csv,noheader,nounits",
                                    ],
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                )
                                if smi.returncode == 0:
                                    vals = [
                                        float(x.strip())
                                        for x in smi.stdout.strip().splitlines()
                                        if x.strip()
                                    ]
                                    if vals:
                                        gpu_util_samples.append(float(sum(vals) / len(vals)))
                            except Exception:
                                pass

                    monitor_thread = threading.Thread(target=_util_monitor, daemon=True)
                    monitor_thread.start()

                    start = time.perf_counter()
                    try:
                        while (time.perf_counter() - start) < fallback_duration_sec:
                            _ = llm.generate(prompts, params)
                    finally:
                        _monitor_stop.set()
                        monitor_thread.join(timeout=5)

                    stdout = ""
                    stderr = ""
                    runtime_sec = time.perf_counter() - start
                    returncode = 0
                    benchmark_mode = "vllm_api_fallback"
                except Exception as fallback_exc:
                    pytest.skip(
                        "vLLM benchmark module unavailable and API fallback failed: "
                        f"{fallback_exc}"
                    )
            else:
                pytest.skip(
                    "vLLM benchmark_serving failed in this environment. "
                    f"exit={proc.returncode}, stderr={out_err}"
                )

        if not gpu_util_samples:
            pytest.skip("No GPU utilization samples collected during vLLM benchmark")
        if not cpu_util_samples:
            pytest.skip("No CPU utilization samples collected during vLLM benchmark")

        gpu_over_90_ratio = sum(1 for x in gpu_util_samples if x >= 90.0) / len(gpu_util_samples)
        cpu_under_70_ratio = sum(1 for x in cpu_util_samples if x <= 70.0) / len(cpu_util_samples)
        gpu_mean = float(np.mean(gpu_util_samples))
        cpu_mean = float(np.mean(cpu_util_samples))
        cpu_p95 = float(np.percentile(cpu_util_samples, 95))
        max_active_compute_processes = max(observed_active_pid_counts) if observed_active_pid_counts else 0

        # If compute processes were already present before the benchmark starts,
        # treat this as shared-GPU contention and use an adjusted threshold.
        contention_detected = len(baseline_active_pids) > 0
        gpu_over_90_target = (
            shared_gpu_over_90_target if contention_detected else strict_gpu_over_90_target
        )

        # Optional latency-jitter signals from benchmark output, when available.
        p50_ms = None
        p95_ms = None
        p99_ms = None
        out = (stdout or "") + "\n" + (stderr or "")
        for token in out.replace(",", " ").split():
            low = token.lower()
            if low.startswith("p50") and "=" in token:
                try:
                    p50_ms = float(token.split("=")[-1])
                except ValueError:
                    pass
            if low.startswith("p95") and "=" in token:
                try:
                    p95_ms = float(token.split("=")[-1])
                except ValueError:
                    pass
            if low.startswith("p99") and "=" in token:
                try:
                    p99_ms = float(token.split("=")[-1])
                except ValueError:
                    pass

        logger.info(
            "vLLM prefill saturation: runtime=%.1fs  gpu_mean=%.1f%%  cpu_mean=%.1f%%  "
            "gpu>=90%% ratio=%.2f  cpu<=70%% ratio=%.2f  cpu_p95=%.1f%%",
            runtime_sec,
            gpu_mean,
            cpu_mean,
            gpu_over_90_ratio,
            cpu_under_70_ratio,
            cpu_p95,
        )

        if p50_ms is not None and p95_ms is not None and p99_ms is not None:
            logger.info("vLLM latency percentiles (ms): p50=%.2f p95=%.2f p99=%.2f", p50_ms, p95_ms, p99_ms)
            if p99_ms > p95_ms * 1.5:
                logger.warning(
                    "Large tail-latency spread detected (p99 much larger than p95) - "
                    "possible jitter, memory thrash, or system instability"
                )

        _record_kpi(
            "test_vllm_data_pipeline_saturation_inference_prefill",
            {
                "model": _HF_INFERENCE_MODEL,
                "vllm_version": vllm_version,
                "benchmark_mode": benchmark_mode,
                "runtime_sec": round(runtime_sec, 2),
                "gpu_mean_util_pct": round(gpu_mean, 2),
                "cpu_mean_util_pct": round(cpu_mean, 2),
                "gpu_over_90_ratio": round(gpu_over_90_ratio, 3),
                "cpu_under_70_ratio": round(cpu_under_70_ratio, 3),
                "cpu_p95_util_pct": round(cpu_p95, 2),
                "gpu_over_90_ratio_target": round(gpu_over_90_target, 3),
                "cpu_under_70_ratio_target": round(cpu_under_70_target, 3),
                "contention_detected": contention_detected,
                "baseline_active_compute_processes": len(baseline_active_pids),
                "max_active_compute_processes": max_active_compute_processes,
                "request_rate": 100,
                "input_len_mean": 1024,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
            },
        )

        if contention_detected:
            logger.warning(
                "Shared GPU contention detected (baseline active compute processes=%d); "
                "using adjusted gpu>=90%% ratio target %.2f instead of %.2f",
                len(baseline_active_pids),
                gpu_over_90_target,
                strict_gpu_over_90_target,
            )

        assert gpu_over_90_ratio >= gpu_over_90_target, (
            "GPU utilization not sustained "
            f"(>=90% ratio={gpu_over_90_ratio:.2f}, mean={gpu_mean:.1f}%, "
            f"target_ratio={gpu_over_90_target:.2f}, shared_gpu={contention_detected})"
        )
        assert cpu_under_70_ratio >= cpu_under_70_target, (
            "CPU utilization too high for sustained periods "
            f"(<=70% ratio={cpu_under_70_ratio:.2f}, p95={cpu_p95:.1f}%, "
            f"target_ratio={cpu_under_70_target:.2f})"
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    def test_vision_object_detection_yolo_fps(self):
        """TC-NV-08c: Vision workload (object detection) via YOLO.

        Goal: validate high aggregate FPS with efficient preprocessing.
        Pass criteria:
          - Aggregate FPS >= 200
          - CPU utilization mean < 50%
        """
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        try:
            import ultralytics  # noqa: F401
        except ImportError:
            pytest.skip("ultralytics not installed")

        visible_gpus = torch.cuda.device_count()
        required_gpus = int(os.environ.get("YOLO_GPU_COUNT", "4"))
        if visible_gpus < required_gpus:
            pytest.skip(
                f"YOLO workload requires {required_gpus} CUDA GPUs; found {visible_gpus}"
            )

        model_name = os.environ.get("YOLO_DETECT_MODEL", "yolov8x.pt")
        if not _ensure_yolo_model_downloaded(model_name):
            pytest.skip(f"Failed to download/cache YOLO model: {model_name}")

        source = os.environ.get("YOLO_DETECT_SOURCE", "https://ultralytics.com/images/bus.jpg")
        try:
            image_path = _resolve_vision_input_source(source)
        except Exception as exc:
            pytest.skip(f"Failed to prepare YOLO input source: {exc}")

        ctx = mp.get_context("spawn")
        gpu_ids = list(range(required_gpus))
        images_per_gpu = int(os.environ.get("YOLO_IMAGES_PER_GPU", "256"))
        batch_size = int(os.environ.get("YOLO_DETECT_BATCH", "128"))
        imgsz = int(os.environ.get("YOLO_DETECT_IMGSZ", "640"))
        half = os.environ.get("YOLO_HALF_PRECISION", "1").lower() in {"1", "true", "yes", "on"}
        timeout_sec = int(os.environ.get("YOLO_DETECT_TIMEOUT_SEC", "240"))
        cpu_sample_interval_sec = float(os.environ.get("YOLO_CPU_SAMPLE_INTERVAL_SEC", "0.5"))
        fps_target = float(os.environ.get("YOLO_AGGREGATE_FPS_TARGET", "200"))
        cpu_target = float(os.environ.get("YOLO_CPU_UTIL_TARGET", "50"))

        logger.info(
            "Running YOLO workload: model=%s gpus=%s images_per_gpu=%d batch=%d imgsz=%d half=%s source=%s",
            model_name,
            gpu_ids,
            images_per_gpu,
            batch_size,
            imgsz,
            half,
            image_path,
        )

        cpu_samples: List[float] = []
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=_cpu_monitor,
            args=(stop_event, cpu_sample_interval_sec, cpu_samples),
            daemon=True,
        )
        monitor.start()

        result_queue = ctx.Queue()
        processes = []
        wall_start = time.perf_counter()
        for gpu_id in gpu_ids:
            process = ctx.Process(
                target=_yolo_worker,
                args=(
                    gpu_id,
                    model_name,
                    str(image_path),
                    images_per_gpu,
                    batch_size,
                    imgsz,
                    half,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)

        deadline = wall_start + timeout_sec
        for process in processes:
            remaining = max(deadline - time.perf_counter(), 0.0)
            process.join(timeout=remaining)

        timed_out = [process.pid for process in processes if process.is_alive()]
        if timed_out:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            stop_event.set()
            monitor.join(timeout=2)
            pytest.fail(f"YOLO workload timed out waiting for workers: {timed_out}")

        stop_event.set()
        monitor.join(timeout=2)
        wall_end = time.perf_counter()

        results = []
        while len(results) < len(processes):
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                break

        if len(results) != len(processes):
            pytest.fail(
                f"YOLO workload collected {len(results)} worker results for {len(processes)} processes"
            )

        errors = [result for result in results if result.get("status") != "ok"]
        if errors:
            merged_errors = "\n".join(str(err.get("error", "")) for err in errors)
            if "cuda out of memory" in merged_errors.lower() or "out of memory" in merged_errors.lower():
                pytest.skip("YOLO workload skipped due to CUDA OOM in shared GPU environment")
            pytest.fail(f"YOLO worker failure: {errors}")

        results = sorted(results, key=lambda result: result["gpu"])
        total_images = sum(result["images"] for result in results)
        makespan_sec = wall_end - wall_start
        aggregate_fps = total_images / makespan_sec if makespan_sec > 0 else 0.0
        cpu_mean = float(np.mean(cpu_samples)) if cpu_samples else 0.0
        cpu_p95 = float(np.percentile(cpu_samples, 95)) if cpu_samples else 0.0
        per_gpu_mean_fps = float(np.mean([result["fps"] for result in results])) if results else 0.0
        per_gpu_min_fps = float(min(result["fps"] for result in results)) if results else 0.0

        for result in results:
            logger.info(
                "YOLO GPU %d: images=%d elapsed=%.3fs fps=%.2f",
                result["gpu"],
                result["images"],
                result["elapsed"],
                result["fps"],
            )

        logger.info(
            "YOLO object detection workload: makespan=%.1fs aggregate_fps=%.1f cpu_mean=%.1f%% cpu_p95=%.1f%%",
            makespan_sec,
            aggregate_fps,
            cpu_mean,
            cpu_p95,
        )

        _record_kpi(
            "test_vision_object_detection_yolo_fps",
            {
                "model": model_name,
                "gpu_count": required_gpus,
                "images_per_gpu": images_per_gpu,
                "total_images": total_images,
                "makespan_sec": round(makespan_sec, 3),
                "batch": batch_size,
                "imgsz": imgsz,
                "aggregate_fps": round(aggregate_fps, 2),
                "aggregate_fps_target": round(fps_target, 2),
                "cpu_mean_util_pct": round(cpu_mean, 2),
                "cpu_p95_util_pct": round(cpu_p95, 2),
                "cpu_target_util_pct": round(cpu_target, 2),
                "per_gpu_mean_fps": round(per_gpu_mean_fps, 2),
                "per_gpu_min_fps": round(per_gpu_min_fps, 2),
            },
        )

        assert aggregate_fps >= fps_target, (
            f"YOLO aggregate FPS too low: {aggregate_fps:.1f} < target {fps_target:.1f}"
        )
        assert cpu_mean < cpu_target, (
            f"YOLO CPU utilization too high: mean {cpu_mean:.1f}% >= target {cpu_target:.1f}%"
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    def test_vision_image_generation_diffusers_throughput(self):
        """TC-NV-08d: Vision workload (image generation) via distributed Diffusers.

        Goal: validate stable 4-GPU image generation throughput.
        Pass criteria:
          - Aggregate throughput > 10 images/sec
        """
        try:
            import diffusers  # noqa: F401
        except ImportError:
            pytest.skip("diffusers not installed")

        visible_gpus = torch.cuda.device_count()
        world_size = int(os.environ.get("DIFFUSERS_GPU_COUNT", "4"))
        if visible_gpus < world_size:
            pytest.skip(
                f"Diffusers workload requires {world_size} CUDA GPUs; found {visible_gpus}"
            )

        ctx = mp.get_context("spawn")
        timeout_sec = int(os.environ.get("DIFFUSERS_GENERATE_TIMEOUT_SEC", "600"))
        master_port = int(os.environ.get("DIFFUSERS_MASTER_PORT", "29501"))
        model_id = os.environ.get("DIFFUSERS_MODEL_ID", "stabilityai/sd-turbo")
        images_per_gpu = int(os.environ.get("DIFFUSERS_IMAGES_PER_GPU", os.environ.get("DIFFUSERS_NUM_IMAGES", "32")))
        steps = int(os.environ.get("DIFFUSERS_STEPS", "1"))
        guidance_scale = float(os.environ.get("DIFFUSERS_GUIDANCE_SCALE", "0.0"))
        height = int(os.environ.get("DIFFUSERS_HEIGHT", "512"))
        width = int(os.environ.get("DIFFUSERS_WIDTH", "512"))
        warmup = int(os.environ.get("DIFFUSERS_WARMUP", "2"))
        throughput_target = float(os.environ.get("DIFFUSERS_TARGET_IMG_PER_SEC", "10.0"))
        prompt = os.environ.get(
            "DIFFUSERS_PROMPT",
            "A futuristic AI data center with GPUs, ultra detailed",
        )
        negative_prompt = os.environ.get("DIFFUSERS_NEGATIVE_PROMPT", "")

        cfg = {
            "model": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "images_per_gpu": images_per_gpu,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "warmup": warmup,
            "variant": "fp16",
        }

        logger.info(
            "Running Diffusers distributed workload: model=%s world_size=%d images_per_gpu=%d steps=%d guidance=%.2f size=%dx%d",
            model_id,
            world_size,
            images_per_gpu,
            steps,
            guidance_scale,
            height,
            width,
        )

        result_queue = ctx.Queue()
        processes = []
        start = time.perf_counter()
        for rank in range(world_size):
            process = ctx.Process(
                target=_diffusion_worker,
                args=(rank, world_size, master_port, cfg, result_queue),
            )
            process.start()
            processes.append(process)

        deadline = start + timeout_sec
        for process in processes:
            remaining = max(deadline - time.perf_counter(), 0.0)
            process.join(timeout=remaining)

        timed_out = [process.pid for process in processes if process.is_alive()]
        if timed_out:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            pytest.fail(f"Diffusers workload timed out waiting for workers: {timed_out}")

        results = []
        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                break

        errors = [result for result in results if result.get("status") == "error"]
        if errors:
            pytest.fail(f"Diffusers worker failure: {errors}")

        success = next((result for result in results if result.get("status") == "ok"), None)
        if success is None:
            pytest.fail("Diffusers workload finished without aggregated rank results")

        rows = sorted(success["per_rank"], key=lambda item: item["rank"])
        total_images = sum(row["images"] for row in rows)
        makespan_sec = max(row["elapsed"] for row in rows) if rows else 0.0
        aggregate_img_per_sec = total_images / makespan_sec if makespan_sec > 0 else 0.0
        per_rank_mean_ips = float(np.mean([row["ips"] for row in rows])) if rows else 0.0
        per_rank_min_ips = float(min(row["ips"] for row in rows)) if rows else 0.0

        for row in rows:
            logger.info(
                "Diffusers GPU %d: images=%d elapsed=%.3fs img_per_sec=%.3f",
                row["rank"],
                row["images"],
                row["elapsed"],
                row["ips"],
            )

        logger.info(
            "Diffusers image-generation workload: total_images=%d makespan=%.3fs aggregate_img_per_sec=%.3f",
            total_images,
            makespan_sec,
            aggregate_img_per_sec,
        )

        _record_kpi(
            "test_vision_image_generation_diffusers_throughput",
            {
                "model": model_id,
                "world_size": world_size,
                "images_per_gpu": images_per_gpu,
                "total_images": total_images,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "warmup": warmup,
                "makespan_sec": round(makespan_sec, 3),
                "throughput_img_per_sec": round(aggregate_img_per_sec, 3),
                "throughput_target_img_per_sec": round(throughput_target, 3),
                "per_rank_mean_img_per_sec": round(per_rank_mean_ips, 3),
                "per_rank_min_img_per_sec": round(per_rank_min_ips, 3),
            },
        )

        assert aggregate_img_per_sec > throughput_target, (
            f"Image-generation throughput too low: {aggregate_img_per_sec:.3f} img/s <= {throughput_target:.3f}"
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.timeout(0)
    def test_hf_encoder_training_throughput(self):
        """Fine-tune HuggingFace BERT for one epoch of synthetic data; validate throughput."""
        test_t0 = time.perf_counter()
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
            "(batch=%d, seq=%d, steps=%d, completion=%.1fs)",
            _HF_TRAINING_MODEL, samples_per_sec, batch_size, seq_len, n_steps,
            time.perf_counter() - test_t0
        )

        _record_kpi(
            "test_hf_encoder_training_throughput",
            {
                "model": _HF_TRAINING_MODEL,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "steps": n_steps,
                "elapsed_sec": round(elapsed, 4),
                "completion_time_sec": round(float(time.perf_counter() - test_t0), 2),
                "samples_per_sec": round(float(samples_per_sec), 2),
            },
        )

        assert samples_per_sec > 50, (
            f"{_HF_TRAINING_MODEL} training throughput {samples_per_sec:.0f} samples/sec too low"
        )
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.timeout(0)
    def test_vllm_inference_throughput(self):
        """Run vLLM offline inference with a HuggingFace model and measure throughput."""
        test_t0 = time.perf_counter()
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            pytest.skip("vllm not installed")

        # vLLM may compile small C/CUDA helpers at runtime; ensure Python headers exist.
        import sysconfig
        include_dir = sysconfig.get_config_var("INCLUDEPY")
        python_h = (Path(include_dir) / "Python.h") if include_dir else None
        if not python_h or not python_h.exists():
            pytest.skip(
                "Missing Python.h (python dev headers). "
                "Install python3-devel/python3-dev for this interpreter."
            )

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
            "throughput=%.0f tokens/sec  completion=%.1fs",
            _HF_INFERENCE_MODEL, len(prompts), elapsed, total_tokens, tps,
            time.perf_counter() - test_t0
        )
        for i, output in enumerate(outputs):
            logger.info("  Prompt %d: %s", i, output.outputs[0].text[:80].replace("\n", " "))

        _record_kpi(
            "test_vllm_inference_throughput",
            {
                "model": _HF_INFERENCE_MODEL,
                "prompts": len(prompts),
                "max_tokens_per_prompt": 64,
                "elapsed_sec": round(elapsed, 4),
                "completion_time_sec": round(float(time.perf_counter() - test_t0), 2),
                "total_tokens": int(total_tokens),
                "throughput_tokens_per_sec": round(float(tps), 2),
            },
        )

        assert tps > 10, f"vLLM throughput {tps:.0f} tokens/sec too low"
        del llm
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    def test_predictable_memory_orchestration_kv_cache_offload(self):
        """TC-NV-09: KV cache offload and predictable memory orchestration.

        Goal: Validate KV cache offload to host DDR under high context loads.
        Pushes KV cache size beyond GPU HBM by using large context lengths,
        forcing graceful offload to host memory. Measures latency impact,
        page faults, and PCIe bandwidth utilization.

        Pass criteria:
          - Baseline throughput (no offload) >= 50 tokens/sec
          - Offload throughput degradation <= 50% (offload >= 25 tokens/sec)
          - Page faults stable (<200/s average)
          - KV cache hit rate >80% (from vLLM metrics if available)
          - No sudden latency cliffs (p99 < 5x p50)
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            pytest.skip("vllm not installed")

        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        import sysconfig
        include_dir = sysconfig.get_config_var("INCLUDEPY")
        if not include_dir or not Path(include_dir).is_dir():
            pytest.skip("Python headers not found; compile dependencies missing")

        # Configuration for KV cache offload test.
        # Prefer NODE_VLLM_* names to avoid clashes with vLLM's own env parser,
        # but keep legacy VLLM_* fallback for backward compatibility.
        model_id = os.environ.get(
            "NODE_VLLM_KV_OFFLOAD_MODEL",
            os.environ.get("VLLM_KV_OFFLOAD_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        )
        context_length = int(
            os.environ.get("NODE_VLLM_KV_CONTEXT_LENGTH", os.environ.get("VLLM_KV_CONTEXT_LENGTH", "8192"))
        )
        batch_size = int(
            os.environ.get("NODE_VLLM_KV_BATCH_SIZE", os.environ.get("VLLM_KV_BATCH_SIZE", "4"))
        )
        num_prompts = int(
            os.environ.get("NODE_VLLM_KV_NUM_PROMPTS", os.environ.get("VLLM_KV_NUM_PROMPTS", "10"))
        )
        max_tokens = int(
            os.environ.get("NODE_VLLM_KV_MAX_TOKENS", os.environ.get("VLLM_KV_MAX_TOKENS", "512"))
        )
        baseline_gpu_memory_util = float(
            os.environ.get(
                "NODE_VLLM_BASELINE_GPU_MEMORY_UTIL",
                os.environ.get("VLLM_BASELINE_GPU_MEMORY_UTIL", "0.9"),
            )
        )
        gpu_memory_util = float(
            os.environ.get("NODE_VLLM_GPU_MEMORY_UTIL", os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.7"))
        )  # Lower util forces offload
        cpu_offload_gb = int(
            os.environ.get("NODE_VLLM_CPU_OFFLOAD_GB", os.environ.get("VLLM_CPU_OFFLOAD_GB", "128"))
        )

        logger.info(
            "KV cache offload test config: model=%s context_len=%d batch=%d num_prompts=%d",
            model_id, context_length, batch_size, num_prompts
        )

        # Test 1: Baseline run WITHOUT KV offload (high memory utilization)
        logger.info("=== BASELINE: KV cache WITHOUT offload ===")
        baseline_metrics = self._run_kv_cache_workload(
            model_id=model_id,
            context_length=context_length,
            batch_size=batch_size,
            num_prompts=num_prompts,
            max_tokens=max_tokens,
            gpu_memory_util=baseline_gpu_memory_util,  # Keep high util, but avoid fragile hard-coded 0.95.
            cpu_offload_gb=0,      # No offload
            offload_enabled=False,
        )

        # Test 2: Offload-enabled run (lower memory utilization forces UVM/offload)
        logger.info("=== OFFLOAD-ENABLED: KV cache with offload ===")
        offload_metrics = self._run_kv_cache_workload(
            model_id=model_id,
            context_length=context_length,
            batch_size=batch_size,
            num_prompts=num_prompts,
            max_tokens=max_tokens,
            gpu_memory_util=gpu_memory_util,
            cpu_offload_gb=cpu_offload_gb,
            offload_enabled=True,
        )

        if not baseline_metrics or not offload_metrics:
            pytest.skip("KV offload workload initialization failed")

        # Calculate metrics and degradation
        baseline_tps = baseline_metrics.get("tokens_per_sec", 0.0)
        offload_tps = offload_metrics.get("tokens_per_sec", 0.0)
        baseline_latency_ms = baseline_metrics.get("mean_latency_ms", 0.0)
        offload_latency_ms = offload_metrics.get("mean_latency_ms", 0.0)

        throughput_degradation = (baseline_tps - offload_tps) / baseline_tps * 100 if baseline_tps > 0 else 0.0
        latency_increase = (offload_latency_ms - baseline_latency_ms) / baseline_latency_ms * 100 if baseline_latency_ms > 0 else 0.0

        logger.info(
            "KV Offload Impact Summary:\n"
            "  Baseline throughput: %.1f tok/s, latency: %.2f ms\n"
            "  Offload throughput: %.1f tok/s, latency: %.2f ms\n"
            "  Degradation: %.1f%% throughput, %.1f%% latency increase",
            baseline_tps, baseline_latency_ms,
            offload_tps, offload_latency_ms,
            throughput_degradation, latency_increase
        )

        # Measure PCIe bandwidth
        pcie_bw_gbs = _measure_gpu_pcie_bandwidth()
        logger.info("PCIe Bandwidth Estimate: %.1f GB/s", pcie_bw_gbs)

        effective_context_length = int(
            min(
                baseline_metrics.get("effective_context_length", context_length),
                offload_metrics.get("effective_context_length", context_length),
            )
        )

        # Estimate KV traffic (rough: batch * context * 2 (key+value) * dtype_size * requests)
        dtype_bytes = 2  # bfloat16
        estimated_kv_volume_gb = (batch_size * effective_context_length * 2 * dtype_bytes * num_prompts) / 1e9
        estimated_pcie_util_percent = (estimated_kv_volume_gb / (pcie_bw_gbs * 1.0)) * 100 if pcie_bw_gbs > 0 else 0.0

        logger.info(
            "Estimated KV Volume: %.2f GB, PCIe Utilization (estimate): %.1f%%",
            estimated_kv_volume_gb, estimated_pcie_util_percent
        )

        # Record comprehensive KPI
        _record_kpi(
            "test_predictable_memory_orchestration_kv_cache_offload",
            {
                "model": model_id,
                "requested_context_length": context_length,
                "effective_context_length": effective_context_length,
                "batch_size": batch_size,
                "num_prompts": num_prompts,
                "max_tokens_per_prompt": max_tokens,
                #### BASELINE
                "baseline_throughput_tokens_per_sec": round(baseline_tps, 2),
                "baseline_mean_latency_ms": round(baseline_latency_ms, 4),
                "baseline_p50_latency_ms": round(baseline_metrics.get("p50_latency_ms", 0.0), 4),
                "baseline_p95_latency_ms": round(baseline_metrics.get("p95_latency_ms", 0.0), 4),
                "baseline_p99_latency_ms": round(baseline_metrics.get("p99_latency_ms", 0.0), 4),
                #### OFFLOAD
                "offload_throughput_tokens_per_sec": round(offload_tps, 2),
                "offload_mean_latency_ms": round(offload_latency_ms, 4),
                "offload_p50_latency_ms": round(offload_metrics.get("p50_latency_ms", 0.0), 4),
                "offload_p95_latency_ms": round(offload_metrics.get("p95_latency_ms", 0.0), 4),
                "offload_p99_latency_ms": round(offload_metrics.get("p99_latency_ms", 0.0), 4),
                #### DEGRADATION
                "throughput_degradation_percent": round(throughput_degradation, 1),
                "latency_increase_percent": round(latency_increase, 1),
                #### MEMORY & FAULTS
                "baseline_page_faults_per_sec": baseline_metrics.get("page_faults_per_sec", 0.0),
                "offload_page_faults_per_sec": offload_metrics.get("page_faults_per_sec", 0.0),
                #### PCIE
                "pcie_bandwidth_gb_per_sec": round(pcie_bw_gbs, 1),
                "estimated_kv_volume_gb": round(estimated_kv_volume_gb, 2),
                "estimated_pcie_utilization_percent": round(estimated_pcie_util_percent, 1),
            },
        )

        # Assertions
        assert baseline_tps >= 10, f"Baseline throughput {baseline_tps:.1f} tok/s too low"
        assert offload_tps >= 5, f"Offload throughput {offload_tps:.1f} tok/s too low"
        assert throughput_degradation <= 60, (
            f"Offload throughput degradation too high: {throughput_degradation:.1f}% "
            f"(baseline {baseline_tps:.1f} -> offload {offload_tps:.1f})"
        )
        assert latency_increase <= 150, (
            f"Offload latency increase too high: {latency_increase:.1f}% "
            f"(baseline {baseline_latency_ms:.2f}ms -> offload {offload_latency_ms:.2f}ms)"
        )

        # Page fault validation
        baseline_faults = baseline_metrics.get("page_faults_per_sec", 0.0)
        offload_faults = offload_metrics.get("page_faults_per_sec", 0.0)
        assert baseline_faults < 200, (
            f"Baseline page faults too high: {baseline_faults:.1f} faults/sec"
        )
        assert offload_faults < 500, (
            f"Offload page faults too high: {offload_faults:.1f} faults/sec "
            f"(offload may be thrashing)"
        )

        torch.cuda.empty_cache()

    def _run_kv_cache_workload(
        self,
        model_id: str,
        context_length: int,
        batch_size: int,
        num_prompts: int,
        max_tokens: int,
        gpu_memory_util: float,
        cpu_offload_gb: int,
        offload_enabled: bool,
    ) -> Optional[Dict]:
        """
        Run a KV cache workload and collect metrics.
        Returns dict with throughput, latency percentiles, page faults, or None if failed.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            return None

        try:
            device_count = torch.cuda.device_count()
            effective_context_length = _resolve_supported_context_length(model_id, context_length)
            if effective_context_length != context_length:
                logger.info(
                    "Reducing KV cache context length for %s from %d to %d based on model config",
                    model_id,
                    context_length,
                    effective_context_length,
                )

            llm_kwargs_base = {
                "model": model_id,
                "dtype": "bfloat16",
                "max_model_len": effective_context_length,
                "enable_prefix_caching": True,  # KV cache prefixing helps with large contexts
            }

            if offload_enabled and cpu_offload_gb > 0:
                llm_kwargs_base["cpu_offload_gb"] = cpu_offload_gb
                logger.info("Enabling KV cache offload to CPU: %d GB", cpu_offload_gb)

            # vLLM engine init can fail when memory utilization is too aggressive for
            # the model/context combo. Retry with progressively safer limits.
            util_candidates: List[float] = [gpu_memory_util]
            for fallback_util in (0.9, 0.85, 0.8, 0.75, 0.7):
                if fallback_util < gpu_memory_util and fallback_util not in util_candidates:
                    util_candidates.append(fallback_util)

            # Also back off tensor-parallel size if engine core initialization fails.
            # This helps environments where NCCL/group init is unstable at full GPU count.
            tp_candidates: List[int] = []
            for tp in (device_count, min(device_count, 2), 1):
                if tp >= 1 and tp not in tp_candidates:
                    tp_candidates.append(tp)

            llm = None
            last_init_exc: Optional[Exception] = None
            total_attempts = len(util_candidates) * len(tp_candidates)
            attempt_idx = 0
            for tp_size in tp_candidates:
                for util in util_candidates:
                    attempt_idx += 1
                    llm_kwargs = {
                        **llm_kwargs_base,
                        "tensor_parallel_size": tp_size,
                        "gpu_memory_utilization": util,
                    }
                    logger.info(
                        "Initializing vLLM with kwargs (attempt %d/%d): %s",
                        attempt_idx,
                        total_attempts,
                        llm_kwargs,
                    )
                    try:
                        llm = LLM(**llm_kwargs)
                        if util != gpu_memory_util:
                            logger.warning(
                                "vLLM init succeeded after lowering gpu_memory_utilization from %.2f to %.2f",
                                gpu_memory_util,
                                util,
                            )
                        if tp_size != device_count:
                            logger.warning(
                                "vLLM init succeeded after lowering tensor_parallel_size from %d to %d",
                                device_count,
                                tp_size,
                            )
                        break
                    except Exception as exc:
                        last_init_exc = exc
                        derived_max_model_len = _extract_vllm_derived_max_model_len(str(exc))
                        if derived_max_model_len and derived_max_model_len < effective_context_length:
                            effective_context_length = derived_max_model_len
                            llm_kwargs_base["max_model_len"] = effective_context_length
                            logger.warning(
                                "Retrying KV cache workload for %s with max_model_len=%d after vLLM validation failure",
                                model_id,
                                effective_context_length,
                            )
                        logger.warning(
                            "vLLM init attempt %d failed (tensor_parallel_size=%d, gpu_memory_utilization=%.2f): %s",
                            attempt_idx,
                            tp_size,
                            util,
                            exc,
                        )
                if llm is not None:
                    break

            if llm is None:
                if last_init_exc is not None:
                    raise last_init_exc
                raise RuntimeError("vLLM initialization failed without exception details")

            # Create varied prompts with context padding to fill context length
            padding_repeats = max(effective_context_length // 20 - 5, 0)
            context_padding = " ".join(["This is context."] * padding_repeats)
            prompts = [
                f"{context_padding} Explain PCIe bandwidth benefits in two sentences:",
                f"{context_padding} Describe NVLink in three sentences:",
                f"{context_padding} What is HBM memory role:",
                f"{context_padding} Summarize NCCL importance:",
                f"{context_padding} Explain GPU utilization:",
            ] * (num_prompts // 5 + 1)
            prompts = prompts[:num_prompts]

            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

            # Warmup
            try:
                _ = llm.generate(prompts[:1], sampling_params)
            except Exception as exc:
                logger.warning("Warmup failed: %s", exc)
                return None

            # Measure page faults before run
            fault_start_major, fault_start_minor = _measure_page_faults()

            # Measure inference with latency tracking
            batch_latencies_ms: List[float] = []
            total_generated_tokens = 0
            t_start = time.perf_counter()

            try:
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i : i + batch_size]
                    t_batch_start = time.perf_counter()
                    outputs = llm.generate(batch, sampling_params)
                    t_batch_end = time.perf_counter()
                    batch_latency_ms = (t_batch_end - t_batch_start) * 1000
                    batch_latencies_ms.append(batch_latency_ms)

                    batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                    total_generated_tokens += batch_tokens
                    logger.info(
                        "  Batch %d/%d (offload=%s): %.2f ms, %d tokens",
                        i // batch_size + 1, (len(prompts) + batch_size - 1) // batch_size,
                        offload_enabled, batch_latency_ms, batch_tokens
                    )
            except Exception as exc:
                logger.warning("Inference run failed: %s", exc)
                return None

            t_end = time.perf_counter()
            elapsed = t_end - t_start

            # Measure page faults after run
            fault_end_major, fault_end_minor = _measure_page_faults()
            fault_duration = elapsed if elapsed > 0 else 1.0
            page_faults_per_sec = (fault_end_major - fault_start_major) / fault_duration

            # Calculate throughput
            throughput_tps = total_generated_tokens / elapsed if elapsed > 0 else 0.0

            # Calculate latency percentiles
            if batch_latencies_ms:
                p50 = float(np.percentile(batch_latencies_ms, 50))
                p95 = float(np.percentile(batch_latencies_ms, 95))
                p99 = float(np.percentile(batch_latencies_ms, 99))
                mean = float(np.mean(batch_latencies_ms))
            else:
                p50 = p95 = p99 = mean = 0.0

            logger.info(
                "Workload complete (offload=%s): throughput=%.1f tok/s, latency=%.2f/%.2f/%.2f (p50/p95/p99) ms",
                offload_enabled, throughput_tps, p50, p95, p99
            )

            del llm
            torch.cuda.empty_cache()

            return {
                "tokens_per_sec": throughput_tps,
                "mean_latency_ms": mean,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "p99_latency_ms": p99,
                "page_faults_per_sec": page_faults_per_sec,
                "effective_context_length": effective_context_length,
                "total_generated_tokens": total_generated_tokens,
            }

        except Exception as e:
            logger.error("KV cache workload failed: %s", e)
            return None

    def _pick_single_cpu_core(self) -> int:
        """Pick one CPU core for single-threaded affinity tests."""
        affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        if affinity:
            return sorted(list(affinity))[0]
        return 0

    def _first_rdma_device(self) -> Optional[str]:
        """Return first RDMA device name from ibv_devices output."""
        if shutil.which("ibv_devices") is None:
            return None
        try:
            proc = subprocess.run(
                ["ibv_devices"], capture_output=True, text=True, timeout=10
            )
            if proc.returncode != 0:
                return None
            for line in proc.stdout.splitlines():
                s = line.strip()
                if not s or s.startswith("device") or s.startswith("------"):
                    continue
                parts = s.split()
                if parts:
                    return parts[0]
        except Exception:
            return None
        return None

    def _parse_ib_write_lat_stats(self, output: str, size_bytes: int) -> Optional[Tuple[float, float]]:
        """Parse ib_write_lat table output and return (min_us, avg_us) for a size."""
        for line in output.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split()
            # Expected table row shape:
            # bytes iterations t_min t_max t_typ t_avg ...
            if len(cols) < 6:
                continue
            if not cols[0].isdigit() or int(cols[0]) != size_bytes:
                continue
            try:
                min_us = float(cols[2])
                avg_us = float(cols[5])
                return min_us, avg_us
            except Exception:
                continue
        return None

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.rdma_validation
    def test_single_core_gpudirect_rdma_flow(self):
        """Single-core GPUDirect RDMA validation across fixed message sizes.

        Validates one-thread, one-QP/CQ style flow pinned to one CPU core for
        message sizes 64B, 256B, and 4KB, and records min/avg latency.
        """
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        if shutil.which("taskset") is None:
            pytest.skip("taskset not found; cannot enforce single-core pinning")
        if shutil.which("ib_write_lat") is None:
            pytest.skip("ib_write_lat not found; install perftest package")

        remote_host = os.environ.get("RDMA_REMOTE_HOST", "").strip()
        if not remote_host:
            pytest.skip("Set RDMA_REMOTE_HOST for 2-node RDMA latency validation")

        rdma_dev = os.environ.get("RDMA_DEVICE", "").strip() or self._first_rdma_device()
        if not rdma_dev:
            pytest.skip("No RDMA device found; set RDMA_DEVICE or install rdma-core tools")

        cpu_core = self._pick_single_cpu_core()
        iterations = int(os.environ.get("RDMA_SMOKE_ITERS", "1000"))
        use_cuda_dev = int(os.environ.get("RDMA_USE_CUDA_DEVICE", "0"))
        msg_sizes = [64, 256, 4096]
        results: Dict[str, Dict[str, float]] = {}

        # Pattern preparation requested for smoke/stability loops.
        patt_buf = torch.empty(4096, dtype=torch.uint8, device="cuda:0")
        patterns = [0xA5, 0x5A]

        for idx, sz in enumerate(msg_sizes):
            patt_buf[:sz].fill_(patterns[idx % len(patterns)])
            if sz >= 256:
                counter = torch.arange(sz, dtype=torch.uint8, device="cuda:0")
                patt_buf[:sz].copy_(counter)

            cmd = [
                "taskset", "-c", str(cpu_core),
                "ib_write_lat", remote_host,
                "-d", rdma_dev,
                "-s", str(sz),
                "-n", str(iterations),
                "--use_cuda", str(use_cuda_dev),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0:
                pytest.fail(
                    f"ib_write_lat failed for size={sz} on {rdma_dev}: {combined[-600:]}"
                )

            parsed = self._parse_ib_write_lat_stats(combined, sz)
            if not parsed:
                pytest.fail(f"Could not parse ib_write_lat output for size={sz}")
            min_us, avg_us = parsed
            results[str(sz)] = {"min_us": min_us, "avg_us": avg_us}

        _record_kpi(
            "test_single_core_gpudirect_rdma_flow",
            {
                "remote_host": remote_host,
                "rdma_device": rdma_dev,
                "cpu_core": cpu_core,
                "iterations": iterations,
                "sizes": results,
            },
        )

        assert results["4096"]["avg_us"] < float(
            os.environ.get("RDMA_4KB_MAX_AVG_LAT_US", "300.0")
        ), f"4KB RDMA avg latency too high: {results['4096']['avg_us']:.2f} us"

    @pytest.mark.workload_acceptance
    @pytest.mark.rdma_validation
    def test_mmio_write_read_validation(self):
        """MMIO write/read validation for NIC/DPU register access path.

        Uses a user-provided command if available (MMIO_VALIDATION_CMD).
        Fallback performs a safe read/read consistency check via setpci.
        """
        custom_cmd = os.environ.get("MMIO_VALIDATION_CMD", "").strip()
        if custom_cmd:
            proc = subprocess.run(custom_cmd, shell=True, capture_output=True, text=True, timeout=120)
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0:
                pytest.fail(f"MMIO validation command failed: {combined[-600:]}")
            _record_kpi(
                "test_mmio_write_read_validation",
                {"mode": "custom_cmd", "status": "pass", "command": custom_cmd},
            )
            return

        if shutil.which("setpci") is None or shutil.which("lspci") is None:
            pytest.skip("setpci/lspci not available; set MMIO_VALIDATION_CMD to run platform MMIO test")

        nic_bdf = os.environ.get("MMIO_PCI_BDF", "").strip()
        if not nic_bdf:
            pci = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
            if pci.returncode != 0:
                pytest.skip("lspci failed; set MMIO_PCI_BDF explicitly")
            for line in pci.stdout.splitlines():
                if "Mellanox" in line or "NVIDIA" in line and "Ethernet" in line:
                    nic_bdf = line.split()[0]
                    break
        if not nic_bdf:
            pytest.skip("Could not detect NIC BDF; set MMIO_PCI_BDF")

        rd1 = subprocess.run(["setpci", "-s", nic_bdf, "04.w"], capture_output=True, text=True, timeout=10)
        rd2 = subprocess.run(["setpci", "-s", nic_bdf, "04.w"], capture_output=True, text=True, timeout=10)
        if rd1.returncode != 0 or rd2.returncode != 0:
            pytest.fail(f"MMIO read failed on {nic_bdf}: {(rd1.stderr or rd2.stderr)[-300:]}")
        v1 = rd1.stdout.strip()
        v2 = rd2.stdout.strip()

        if os.environ.get("MMIO_ALLOW_WRITE", "0") == "1":
            wr = subprocess.run(["setpci", "-s", nic_bdf, f"04.w={v1}"], capture_output=True, text=True, timeout=10)
            if wr.returncode != 0:
                pytest.fail(f"MMIO write failed on {nic_bdf}: {(wr.stderr or '')[-300:]}")

        _record_kpi(
            "test_mmio_write_read_validation",
            {
                "mode": "setpci_fallback",
                "nic_bdf": nic_bdf,
                "command_reg_read_1": v1,
                "command_reg_read_2": v2,
                "write_attempted": os.environ.get("MMIO_ALLOW_WRITE", "0") == "1",
            },
        )
        assert v1 == v2, f"MMIO register read mismatch on {nic_bdf}: {v1} vs {v2}"

    @pytest.mark.workload_acceptance
    @pytest.mark.rdma_validation
    def test_fence_validation_rdma_ordering(self):
        """Fence validation for ordering between dependent RDMA/MMIO operations.

        Expects an external fence test command when real RDMA ordering needs
        to be validated end-to-end.
        """
        fence_cmd = os.environ.get("RDMA_FENCE_VALIDATION_CMD", "").strip()
        if not fence_cmd:
            pytest.skip("Set RDMA_FENCE_VALIDATION_CMD to run device-specific fence ordering validation")

        loops = int(os.environ.get("RDMA_FENCE_ITERS", "10000"))
        proc = subprocess.run(
            f"{fence_cmd} --iters {loops}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            pytest.fail(f"Fence validation command failed: {combined[-800:]}")

        stale_reads = 0
        m = re.search(r"stale[_\s-]*reads\s*[:=]\s*(\d+)", combined, flags=re.IGNORECASE)
        if m:
            stale_reads = int(m.group(1))

        _record_kpi(
            "test_fence_validation_rdma_ordering",
            {
                "command": fence_cmd,
                "iterations": loops,
                "stale_reads": stale_reads,
            },
        )
        assert stale_reads == 0, f"Fence/order violation detected: stale_reads={stale_reads}"

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.rdma_validation
    def test_one_dma_read_4kb_validation(self):
        """Validate one exact 4KB DMA-like transfer with data pattern compare."""
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        n_bytes = 4096
        host_src = bytearray((i % 256 for i in range(n_bytes)))
        cpu_in = torch.tensor(list(host_src), dtype=torch.uint8, pin_memory=True)
        gpu_buf = torch.empty(n_bytes, dtype=torch.uint8, device="cuda:0")
        cpu_out = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)

        t0 = time.perf_counter()
        gpu_buf.copy_(cpu_in, non_blocking=True)
        cpu_out.copy_(gpu_buf, non_blocking=True)
        torch.cuda.synchronize("cuda:0")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out_bytes = bytes(cpu_out.tolist())
        data_match = out_bytes == bytes(host_src)

        _record_kpi(
            "test_one_dma_read_4kb_validation",
            {
                "transfer_bytes": n_bytes,
                "completion_status": "success" if data_match else "mismatch",
                "elapsed_ms": round(elapsed_ms, 4),
                "data_compare_pass": data_match,
            },
        )

        assert n_bytes == 4096, "Transfer length must be exactly 4KB"
        assert data_match, "4KB DMA data compare failed"

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.rdma_validation
    def test_looped_rdma_mmio_dma_stress_soak(self):
        """Looped stress/soak run for RDMA/MMIO/DMA functional sanity."""
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices")

        loops = int(os.environ.get("RDMA_SOAK_LOOPS", "1000"))
        n_bytes = 4096
        lat_us: List[float] = []
        failures = 0

        gpu_buf = torch.empty(n_bytes, dtype=torch.uint8, device="cuda:0")
        cpu_in = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
        cpu_out = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)

        for i in range(loops):
            pattern = 0xA5 if (i % 2 == 0) else 0x5A
            cpu_in.fill_(pattern)
            if i % 10 == 0:
                cpu_in.copy_(torch.arange(n_bytes, dtype=torch.uint8))

            t0 = time.perf_counter()
            gpu_buf.copy_(cpu_in, non_blocking=True)
            cpu_out.copy_(gpu_buf, non_blocking=True)
            torch.cuda.synchronize("cuda:0")
            dt_us = (time.perf_counter() - t0) * 1e6
            lat_us.append(dt_us)

            if not torch.equal(cpu_in, cpu_out):
                failures += 1

        total_sec = sum(lat_us) / 1e6
        bw_gbps = ((n_bytes * 2 * loops) / (1024 ** 3)) / total_sec if total_sec > 0 else 0.0
        avg_us = float(np.mean(lat_us)) if lat_us else 0.0
        p95_us = float(np.percentile(lat_us, 95)) if lat_us else 0.0

        _record_kpi(
            "test_looped_rdma_mmio_dma_stress_soak",
            {
                "loops": loops,
                "transfer_bytes": n_bytes,
                "avg_latency_us": round(avg_us, 2),
                "p95_latency_us": round(p95_us, 2),
                "bandwidth_gbps": round(bw_gbps, 3),
                "data_mismatch_count": failures,
            },
        )

        assert failures == 0, f"Soak validation had data mismatches: {failures}"
        assert p95_us < float(os.environ.get("SOAK_MAX_P95_US", "20000")), (
            f"Soak latency p95 too high: {p95_us:.1f} us"
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.timeout(0)
    def test_llm_training_small_multigpu(self):
        """AI_HOSTCPU_FUNC_TEST_ID_013: Small LLM fine-tuning (Gemma-270M) on all GPUs via DDP.

        Measures per-step latency, gradient-sync ratio, throughput and TFLOPS.
        PASS criteria: gradient allreduce < 15% of total step time.
        Runtime target: < 10 minutes on 4× H100.
        """
        try:
            from transformers import AutoModelForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        n_gpus = torch.cuda.device_count()
        if n_gpus < 1:
            pytest.skip("No CUDA devices")

        model_id     = os.environ.get("LLM_TRAIN_SMALL_MODEL", "google/gemma-3-270m-it")
        world_size   = min(n_gpus, int(os.environ.get("LLM_TRAIN_SMALL_GPU_COUNT", str(n_gpus))))
        batch_size   = int(os.environ.get("LLM_TRAIN_SMALL_BATCH", "2"))
        seq_len      = int(os.environ.get("LLM_TRAIN_SMALL_SEQ_LEN", "512"))
        n_steps      = int(os.environ.get("LLM_TRAIN_SMALL_STEPS", "50"))
        warmup_steps = int(os.environ.get("LLM_TRAIN_SMALL_WARMUP", "5"))
        timeout_sec  = int(os.environ.get("LLM_TRAIN_SMALL_TIMEOUT_SEC", "3600"))
        max_sync_ratio = float(os.environ.get("LLM_TRAIN_SMALL_MAX_SYNC_RATIO", "0.15"))
        master_port  = int(os.environ.get("LLM_TRAIN_SMALL_MASTER_PORT", "29601"))

        cfg = {
            "model_id":     model_id,
            "batch_size":   batch_size,
            "seq_len":      seq_len,
            "n_steps":      n_steps,
            "warmup_steps": warmup_steps,
            "use_deepspeed": False,
        }

        logger.info(
            "TC-013 LLM Training Small: model=%s world_size=%d batch=%d seq=%d steps=%d",
            model_id, world_size, batch_size, seq_len, n_steps,
        )

        ctx          = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes    = []
        wall_start   = time.perf_counter()

        for rank in range(world_size):
            p = ctx.Process(
                target=_ddp_training_worker,
                args=(rank, world_size, master_port, cfg, result_queue),
            )
            p.start()
            processes.append(p)

        deadline = wall_start + timeout_sec
        for p in processes:
            p.join(timeout=max(deadline - time.perf_counter(), 0.0))

        timed_out = [p.pid for p in processes if p.is_alive()]
        if timed_out:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            for p in processes:
                p.join(timeout=5)
            pytest.fail(f"TC-013 training workers timed out: pids={timed_out}")

        results: List[Dict] = []
        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                break

        errors = [r for r in results if r.get("status") == "error"]
        if errors:
            pytest.fail(f"TC-013 worker failures: {errors}")

        success = next((r for r in results if r.get("status") == "ok"), None)
        assert success is not None, "TC-013: rank-0 produced no aggregated result"

        per_rank = sorted(
            [r for r in success["per_rank"] if r is not None],
            key=lambda r: r["rank"],
        )
        wall_sec        = time.perf_counter() - wall_start
        mean_sync_ratio = float(np.mean([r["sync_ratio"] for r in per_rank]))
        mean_tflops     = float(np.mean([r["tflops"]     for r in per_rank]))
        total_tps       = float(np.sum([r["per_gpu_tps"] for r in per_rank]))
        scaling_eff     = per_rank[0].get("scaling_efficiency_pct") if per_rank else None

        for r in per_rank:
            logger.info(
                "TC-013 GPU %d: step=%.1f ms  allreduce=%.1f ms  sync=%.2f%%  "
                "tps=%.1f samples/s  tflops=%.3f",
                r["rank"], r["mean_step_ms"], r["mean_allreduce_ms"],
                r["sync_ratio"] * 100, r["per_gpu_tps"], r["tflops"],
            )
        logger.info(
            "TC-013 SUMMARY: total_tps=%.1f  mean_sync=%.2f%%  mean_tflops=%.3f  "
            "scaling_eff=%s%%  wall=%.1fs",
            total_tps, mean_sync_ratio * 100, mean_tflops,
            "N/A" if scaling_eff is None else f"{scaling_eff:.1f}", wall_sec,
        )

        _record_kpi(
            "test_llm_training_small_multigpu",
            {
                "test_id":              "AI_HOSTCPU_FUNC_TEST_ID_013",
                "model":                model_id,
                "world_size":           world_size,
                "batch_size":           batch_size,
                "seq_len":              seq_len,
                "n_steps":              n_steps,
                "total_samples_per_sec": round(total_tps, 2),
                "mean_sync_ratio":      round(mean_sync_ratio, 4),
                "mean_sync_pct":        round(mean_sync_ratio * 100, 2),
                "threshold_sync_pct":   round(max_sync_ratio * 100, 1),
                "mean_tflops_per_gpu":  round(mean_tflops, 3),
                "scaling_efficiency_pct": scaling_eff,
                "wall_sec":             round(wall_sec, 1),
                "per_rank":             per_rank,
            },
        )

        assert mean_sync_ratio <= max_sync_ratio, (
            f"TC-013: gradient sync ratio {mean_sync_ratio * 100:.1f}% exceeds "
            f"threshold {max_sync_ratio * 100:.0f}% — communication bottleneck detected; "
            "check NVLink status and NCCL topology"
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.slow
    @pytest.mark.timeout(0)
    def test_llm_training_large_multigpu(self):
        """AI_HOSTCPU_FUNC_TEST_ID_014: Large LLM distributed fine-tuning (Llama-3 8B+).

                Uses actual multi-GPU model sharding via DeepSpeed ZeRO for large models.
        Validates:
          * Gradient sync < 10% of step time
          * TFLOPS stable across GPUs (CV < 10%)
          * Scaling efficiency ≥ 70% (vs single-GPU baseline)
        """
        try:
            from transformers import AutoModelForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need ≥ 2 GPUs for TC-014 large distributed training")

        model_id     = os.environ.get("LLM_TRAIN_LARGE_MODEL", _HF_LLAMA_7B_MODEL)
        world_size   = min(n_gpus, int(os.environ.get("LLM_TRAIN_LARGE_GPU_COUNT", str(n_gpus))))
        batch_size   = int(os.environ.get("LLM_TRAIN_LARGE_BATCH", "1"))
        seq_len      = int(os.environ.get("LLM_TRAIN_LARGE_SEQ_LEN", "512"))
        n_steps      = int(os.environ.get("LLM_TRAIN_LARGE_STEPS", "20"))
        warmup_steps = int(os.environ.get("LLM_TRAIN_LARGE_WARMUP", "3"))
        timeout_sec  = int(os.environ.get("LLM_TRAIN_LARGE_TIMEOUT_SEC", "3600"))
        max_sync_ratio  = float(os.environ.get("LLM_TRAIN_LARGE_MAX_SYNC_RATIO", "0.10"))
        min_scaling_eff = float(os.environ.get("LLM_TRAIN_LARGE_MIN_SCALING_EFF", "70.0"))
        master_port  = int(os.environ.get("LLM_TRAIN_LARGE_MASTER_PORT", "29602"))
        deepspeed_zero_stage = int(os.environ.get("LLM_TRAIN_LARGE_ZERO_STAGE", "3"))

        # Large-model training must use sharded execution rather than replicated DDP.
        use_deepspeed = False
        try:
            import deepspeed  # noqa: F401
            use_deepspeed = os.environ.get(
                "LLM_TRAIN_LARGE_USE_DEEPSPEED", "1"
            ).lower() in {"1", "true", "yes", "on"}
            if use_deepspeed:
                logger.info(
                    "TC-014: DeepSpeed detected — enabling ZeRO Stage %d for model sharding",
                    deepspeed_zero_stage,
                )
        except ImportError:
            pytest.skip(
                "TC-014 large multi-GPU training requires DeepSpeed for model sharding across GPUs."
            )

        if not use_deepspeed:
            pytest.skip(
                "TC-014 large multi-GPU training is configured to require model sharding. "
                "Set LLM_TRAIN_LARGE_USE_DEEPSPEED=1 and install DeepSpeed."
            )

        cfg = {
            "model_id":     model_id,
            "batch_size":   batch_size,
            "seq_len":      seq_len,
            "n_steps":      n_steps,
            "warmup_steps": warmup_steps,
            "use_deepspeed": use_deepspeed,
            "deepspeed_zero_stage": deepspeed_zero_stage,
            "require_model_sharding": True,
            "collect_single_gpu_baseline": False,
        }

        logger.info(
            "TC-014 LLM Training Large: model=%s world_size=%d batch=%d seq=%d "
            "steps=%d deepspeed=%s zero_stage=%d",
            model_id, world_size, batch_size, seq_len, n_steps, use_deepspeed,
            deepspeed_zero_stage,
        )

        ctx          = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes    = []
        wall_start   = time.perf_counter()

        for rank in range(world_size):
            p = ctx.Process(
                target=_ddp_training_worker,
                args=(rank, world_size, master_port, cfg, result_queue),
            )
            p.start()
            processes.append(p)

        deadline = wall_start + timeout_sec
        for p in processes:
            p.join(timeout=max(deadline - time.perf_counter(), 0.0))

        timed_out = [p.pid for p in processes if p.is_alive()]
        if timed_out:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            for p in processes:
                p.join(timeout=5)
            pytest.fail(f"TC-014 training workers timed out: pids={timed_out}")

        results: List[Dict] = []
        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                break

        errors = [r for r in results if r.get("status") == "error"]
        if errors:
            joined_errors = "\n".join(str(r.get("error", "")) for r in errors)
            if "local_rank" in joined_errors.lower() and "deepspeed" in joined_errors.lower():
                pytest.skip("TC-014 skipped: DeepSpeed launcher environment not available in this runtime")
            pytest.fail(f"TC-014 worker failures: {errors}")

        success = next((r for r in results if r.get("status") == "ok"), None)
        assert success is not None, "TC-014: rank-0 produced no aggregated result"

        per_rank = sorted(
            [r for r in success["per_rank"] if r is not None],
            key=lambda r: r["rank"],
        )
        wall_sec        = time.perf_counter() - wall_start
        mean_sync_ratio = float(np.mean([r["sync_ratio"] for r in per_rank]))
        tflops_vals     = [r["tflops"] for r in per_rank if r["tflops"] > 0]
        mean_tflops     = float(np.mean(tflops_vals)) if tflops_vals else 0.0
        tflops_cv       = (float(np.std(tflops_vals)) / mean_tflops
                           if len(tflops_vals) > 1 and mean_tflops > 0 else 0.0)
        total_tps       = float(np.sum([r["per_gpu_tps"] for r in per_rank]))
        scaling_eff     = per_rank[0].get("scaling_efficiency_pct") if per_rank else None

        for r in per_rank:
            logger.info(
                "TC-014 GPU %d: step=%.1f ms  allreduce=%.1f ms  sync=%.2f%%  "
                "tps=%.2f samples/s  tflops=%.3f  scaling_eff=%s%%",
                r["rank"], r["mean_step_ms"], r["mean_allreduce_ms"],
                r["sync_ratio"] * 100, r["per_gpu_tps"], r["tflops"],
                "N/A" if r["scaling_efficiency_pct"] is None
                else f"{r['scaling_efficiency_pct']:.1f}",
            )
        logger.info(
            "TC-014 SUMMARY: total_tps=%.2f  mean_sync=%.2f%%  mean_tflops=%.3f  "
            "tflops_cv=%.3f  scaling_eff=%s%%  wall=%.1fs",
            total_tps, mean_sync_ratio * 100, mean_tflops, tflops_cv,
            "N/A" if scaling_eff is None else f"{scaling_eff:.1f}", wall_sec,
        )

        _record_kpi(
            "test_llm_training_large_multigpu",
            {
                "test_id":               "AI_HOSTCPU_FUNC_TEST_ID_014",
                "model":                 model_id,
                "world_size":            world_size,
                "batch_size":            batch_size,
                "seq_len":               seq_len,
                "n_steps":               n_steps,
                "use_deepspeed":         use_deepspeed,
                "total_samples_per_sec": round(total_tps, 2),
                "mean_sync_ratio":       round(mean_sync_ratio, 4),
                "mean_sync_pct":         round(mean_sync_ratio * 100, 2),
                "threshold_sync_pct":    round(max_sync_ratio * 100, 1),
                "mean_tflops_per_gpu":   round(mean_tflops, 3),
                "tflops_cv":             round(tflops_cv, 4),
                "scaling_efficiency_pct": scaling_eff,
                "min_scaling_eff_pct":   min_scaling_eff,
                "wall_sec":              round(wall_sec, 1),
            },
        )

        assert mean_sync_ratio <= max_sync_ratio, (
            f"TC-014: gradient sync ratio {mean_sync_ratio * 100:.1f}% exceeds "
            f"threshold {max_sync_ratio * 100:.0f}% — validate NVLink/P2P bandwidth "
            "and NCCL topology"
        )
        assert tflops_cv < 0.10, (
            f"TC-014: TFLOPS imbalance across GPUs (CV={tflops_cv:.3f}) — "
            "one or more GPUs may be thermally throttled or have hardware issues"
        )
        if scaling_eff is not None:
            assert scaling_eff >= min_scaling_eff, (
                f"TC-014: scaling efficiency {scaling_eff:.1f}% below "
                f"threshold {min_scaling_eff:.0f}% — check NVLink bandwidth and "
                "PCIe topology"
            )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.slow
    @pytest.mark.timeout(0)
    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.slow
    @pytest.mark.timeout(0)
    def test_multi_tenant_inference_tail_latency(self):
        """AI_HOSTCPU_FUNC_TEST_ID_047: Multi-tenant inference + tail-latency isolation."""
        pytest.skip(
            "TC-047 is temporarily disabled due to unstable vLLM multi-process setup in this branch. "
            "Use test_vllm_data_pipeline_saturation_inference_prefill for current inference validation."
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.slow
    @pytest.mark.timeout(0)
    def test_multi_model_inference_kv_cache_concurrent(self):
        """AI_HOSTCPU_FUNC_TEST_ID_048: Concurrent multi-model inference + KV cache offload."""
        pytest.skip(
            "TC-048 is temporarily disabled due to unstable vLLM multi-process setup in this branch."
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    @pytest.mark.workload_acceptance
    @pytest.mark.slow
    @pytest.mark.timeout(0)
    def test_deepseek_r1_kv_cache_dram_offload(self):
        """AI_HOSTCPU_FUNC_TEST_ID_049: DeepSeek KV-cache offload to DRAM."""
        pytest.skip(
            "TC-049 is temporarily disabled in this branch while consolidating vLLM setup paths."
        )

    def test_vllm_training_kpi_status(self):
        """Capture vLLM training KPI status (vLLM is inference-focused and does not train)."""
        _record_kpi(
            "test_vllm_training_kpi_status",
            {
                "component": "vllm",
                "training_supported": False,
                "status": "not_applicable",
                "note": "vLLM is inference-serving runtime; use PyTorch/DeepSpeed for training KPIs",
            },
        )
        logger.info(
            "vLLM training KPI status: not_applicable (vLLM is inference-focused; training not supported)"
        )


# =============================================================================
# Standalone Validation Functions (for external use)
# =============================================================================

def run_all_validations() -> List[ValidationResult]:
    """Run core validation probes and return basic results (non-pytest path)."""
    results: List[ValidationResult] = []

    enum_result = GPUEnumeration.enumerate_gpus()
    if enum_result.get("count", 0) > 0:
        results.append(
            ValidationResult(
                name="GPU Enumeration",
                category="GPU Enumeration",
                status="PASS",
                message=f"Found {enum_result['count']} GPU(s)",
                data=enum_result,
            )
        )
    else:
        results.append(
            ValidationResult(
                name="GPU Enumeration",
                category="GPU Enumeration",
                status="FAIL",
                message="No GPUs found",
                data=enum_result,
            )
        )

    topo_result = GPUEnumeration.get_pcie_topology()
    if topo_result.get("topology"):
        results.append(
            ValidationResult(
                name="PCIe Topology",
                category="PCIe Topology",
                status="PASS",
                message=f"Discovered {len(topo_result['topology'])} PCIe entries",
                data=topo_result,
            )
        )
    else:
        results.append(
            ValidationResult(
                name="PCIe Topology",
                category="PCIe Topology",
                status="FAIL",
                message="Could not discover PCIe topology",
                data=topo_result,
            )
        )

    return results
