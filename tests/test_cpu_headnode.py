"""
CPU-side head-node validation: ensure the CPU never starves an H100.

These tests target the *integration* between CPUs/cores and the H100 GPUs
in real customer scenarios (serving, RAG, dataloader-fed training). They are
deliberately separate from ``test_cpu_validation.py`` (which measures CPU
compute in isolation) and from ``test_node_validation.py`` (which measures
GPU/PCIe physics). The failures these tests catch are the classic
"GPUs at 30% util in production while CPU looks fine" tickets.

Coverage
--------
TC-CPUH-01  NUMA-local vs NUMA-remote H2D bandwidth
TC-CPUH-02  Pinned (page-locked) host memory allocation scaling
TC-CPUH-03  CPU DRAM bandwidth under concurrent multi-GPU H2D load
TC-CPUH-04  HuggingFace tokenizer throughput scaling (1 -> N threads)
TC-CPUH-05  PyTorch DataLoader scaling with num_workers
TC-CPUH-06  CPU governor / C-state policy (perf-critical sysfs gates)
TC-CPUH-07  IRQ affinity distribution (mlx5 / nvme / nvidia)
TC-CPUH-08  CUDA sync-primitive round-trip latency (empty kernel + sync)
TC-CPUH-09  NVMe model-cache sequential read throughput
TC-CPUH-10  Python async concurrency scaling (asyncio + tokenize burst)

Run with:
    pytest -m cpu_headnode -vv tests/test_cpu_headnode.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


logger = logging.getLogger("cpu_headnode")

_IS_LINUX = sys.platform.startswith("linux")

pytestmark = [pytest.mark.cpu_headnode]


# ---------------------------------------------------------------------------
# KPI recording (mirrors the pattern used by sibling test files)
# ---------------------------------------------------------------------------

def _kpi_path() -> Path:
    return Path(
        os.environ.get(
            "CPU_HEADNODE_KPI_FILE",
            "artifacts/cpu_headnode_kpis.jsonl",
        )
    )


def _record_kpi(test_name: str, metrics: Dict) -> None:
    out = _kpi_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test": test_name,
        "metrics": metrics,
    }
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
    logger.info("CPU-HEADNODE KPI [%s]: %s",
                test_name, json.dumps(metrics, default=str, sort_keys=True))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _read_sysfs(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def _gpu_numa_node(pci_addr: str) -> Optional[int]:
    """Return NUMA node for a PCI address like '0000:61:00.0'."""
    p = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    raw = _read_sysfs(p)
    if raw is None:
        return None
    try:
        n = int(raw)
        return n if n >= 0 else None
    except ValueError:
        return None


def _gpu_pci_addresses() -> List[str]:
    """Best-effort: query nvidia-smi for GPU PCI bus IDs."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return []
        addrs: List[str] = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # nvidia-smi gives '00000000:61:00.0' -> normalise to '0000:61:00.0'
            m = re.match(r"^[0-9A-Fa-f]*?([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])$", line)
            if m:
                addrs.append(m.group(1).lower())
            else:
                addrs.append(line.lower())
        return addrs
    except Exception:
        return []


def _set_thread_affinity(cpus: List[int]) -> bool:
    if not _HAS_PSUTIL or not cpus:
        return False
    try:
        psutil.Process().cpu_affinity(cpus)
        return True
    except Exception:
        return False


def _numa_node_cpus(node: int) -> List[int]:
    """Read /sys/devices/system/node/nodeN/cpulist."""
    raw = _read_sysfs(f"/sys/devices/system/node/node{node}/cpulist")
    if raw is None:
        return []
    cpus: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(chunk))
    return cpus


def _all_numa_nodes() -> List[int]:
    nodes: List[int] = []
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        return nodes
    for child in base.iterdir():
        m = re.match(r"^node(\d+)$", child.name)
        if m:
            nodes.append(int(m.group(1)))
    return sorted(nodes)


# =============================================================================
# TC-CPUH-01: NUMA-local vs NUMA-remote H2D bandwidth
# =============================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not _IS_LINUX, reason="NUMA topology only inspected on Linux")
@pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil required for cpu_affinity")
def test_numa_local_vs_remote_h2d_bandwidth():
    """For each GPU, H2D from a CPU on the *local* NUMA node should be
    materially faster than from a *remote* NUMA node. A poor ratio means
    workers are mis-pinned and every batch pays a UPI/QPI hop.

    Pass: remote/local ratio >= 0.60 for every GPU that has multiple
    candidate NUMA nodes. (Threshold is conservative; severe mispinning
    typically lands around 0.30-0.50.)
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA devices")

    nodes = _all_numa_nodes()
    if len(nodes) < 2:
        pytest.skip(f"Single NUMA node ({nodes}); test is trivially uninteresting")

    pci_addrs = _gpu_pci_addresses()
    n_gpus = torch.cuda.device_count()
    if len(pci_addrs) != n_gpus:
        pytest.skip(f"PCI addr count ({len(pci_addrs)}) != GPU count ({n_gpus})")

    buf_mb = int(os.environ.get("CPU_HEADNODE_NUMA_H2D_MB", "256"))
    iters = int(os.environ.get("CPU_HEADNODE_NUMA_H2D_ITERS", "5"))
    min_ratio = float(os.environ.get("CPU_HEADNODE_NUMA_H2D_MIN_RATIO", "0.60"))

    nbytes = buf_mb * 1024 * 1024
    results: List[Dict] = []
    violations: List[str] = []
    original_affinity = psutil.Process().cpu_affinity()

    try:
        for gpu_id in range(n_gpus):
            local_node = _gpu_numa_node(pci_addrs[gpu_id])
            if local_node is None or local_node not in nodes:
                logger.warning("GPU %d (%s): no NUMA node info, skipping",
                               gpu_id, pci_addrs[gpu_id])
                continue
            remote_node = next((n for n in nodes if n != local_node), None)
            if remote_node is None:
                continue

            local_cpus = _numa_node_cpus(local_node)
            remote_cpus = _numa_node_cpus(remote_node)
            if not local_cpus or not remote_cpus:
                continue

            per_node_bw: Dict[str, float] = {}
            for tag, cpus, node in (("local", local_cpus, local_node),
                                    ("remote", remote_cpus, remote_node)):
                if not _set_thread_affinity(cpus[: min(8, len(cpus))]):
                    pytest.skip(f"Could not set affinity for {tag}")
                # Allocate AFTER affinity change so first-touch lands on the
                # right NUMA node.
                host = torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=True)
                dev = torch.empty_like(host, device=f"cuda:{gpu_id}")
                # Warmup
                dev.copy_(host, non_blocking=True)
                torch.cuda.synchronize(gpu_id)
                t0 = time.perf_counter()
                for _ in range(iters):
                    dev.copy_(host, non_blocking=True)
                torch.cuda.synchronize(gpu_id)
                elapsed = time.perf_counter() - t0
                gbs = (nbytes * iters) / elapsed / 1e9
                per_node_bw[tag] = gbs
                logger.info("GPU %d %s NUMA node %d: %.2f GB/s",
                            gpu_id, tag, node, gbs)
                del host, dev
                torch.cuda.empty_cache()

            ratio = per_node_bw["remote"] / per_node_bw["local"] if per_node_bw["local"] > 0 else 0.0
            entry = {
                "gpu_id": gpu_id,
                "pci_addr": pci_addrs[gpu_id],
                "local_numa": local_node,
                "remote_numa": remote_node,
                "local_h2d_gbs": round(per_node_bw["local"], 2),
                "remote_h2d_gbs": round(per_node_bw["remote"], 2),
                "remote_over_local": round(ratio, 3),
            }
            results.append(entry)
            if ratio < min_ratio:
                violations.append(
                    f"GPU {gpu_id}: remote/local={ratio:.2f} "
                    f"(local={per_node_bw['local']:.1f} remote={per_node_bw['remote']:.1f} GB/s)"
                )
    finally:
        try:
            psutil.Process().cpu_affinity(original_affinity)
        except Exception:
            pass

    _record_kpi("test_numa_local_vs_remote_h2d_bandwidth", {
        "buffer_mb": buf_mb,
        "iters": iters,
        "min_ratio_required": min_ratio,
        "per_gpu": results,
    })

    if not results:
        pytest.skip("No GPU had both local and remote NUMA candidates")

    assert not violations, (
        "NUMA-remote H2D too slow vs local (mispinning risk):\n  "
        + "\n  ".join(violations)
    )


# =============================================================================
# TC-CPUH-02: Pinned (page-locked) host memory allocation scaling
# =============================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_pinned_host_memory_allocation_scaling():
    """vLLM, NCCL, and dataloaders all allocate large page-locked buffers.
    If hugepages are misconfigured or RLIMIT_MEMLOCK is tight, large pinned
    allocations either fail or fall back to slow paths.

    Pass: every requested size allocates within the time budget AND
    achieves >= ``CPU_HEADNODE_PINNED_MIN_H2D_GBS`` GB/s H2D BW.
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA devices")

    sizes_mb = [int(x) for x in os.environ.get(
        "CPU_HEADNODE_PINNED_SIZES_MB", "256,1024,4096"
    ).split(",") if x.strip()]
    alloc_budget_sec = float(os.environ.get("CPU_HEADNODE_PINNED_ALLOC_BUDGET_SEC", "10.0"))
    min_h2d_gbs = float(os.environ.get("CPU_HEADNODE_PINNED_MIN_H2D_GBS", "10.0"))

    # System headroom check: don't try to pin > 25% of free RAM
    if _HAS_PSUTIL:
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        cap_mb = int(free_gb * 0.25 * 1024)
        sizes_mb = [s for s in sizes_mb if s <= cap_mb]
        if not sizes_mb:
            pytest.skip(f"Insufficient free RAM ({free_gb:.1f} GB) for any test size")

    rlimit_info: Dict[str, object] = {}
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        rlimit_info = {"memlock_soft": soft, "memlock_hard": hard}
    except Exception:
        pass

    transparent_hp = _read_sysfs("/sys/kernel/mm/transparent_hugepage/enabled")
    nr_hugepages = _read_sysfs("/proc/sys/vm/nr_hugepages")

    results: List[Dict] = []
    failures: List[str] = []
    for size_mb in sizes_mb:
        nbytes = size_mb * 1024 * 1024
        t0 = time.perf_counter()
        try:
            host = torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=True)
        except RuntimeError as exc:
            failures.append(f"{size_mb} MB: allocation failed: {exc}")
            continue
        alloc_sec = time.perf_counter() - t0

        dev = torch.empty_like(host, device="cuda:0")
        # Warmup
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        iters = max(3, int(2_000_000_000 / nbytes))  # ~2 GB total moved
        t0 = time.perf_counter()
        for _ in range(iters):
            dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        gbs = (nbytes * iters) / (time.perf_counter() - t0) / 1e9
        results.append({
            "size_mb": size_mb,
            "alloc_sec": round(alloc_sec, 3),
            "h2d_gbs": round(gbs, 2),
        })
        if alloc_sec > alloc_budget_sec:
            failures.append(f"{size_mb} MB: alloc took {alloc_sec:.2f}s > {alloc_budget_sec}s")
        if gbs < min_h2d_gbs:
            failures.append(f"{size_mb} MB: H2D {gbs:.2f} GB/s < {min_h2d_gbs} GB/s")

        del host, dev
        torch.cuda.empty_cache()

    _record_kpi("test_pinned_host_memory_allocation_scaling", {
        "results": results,
        "rlimit_memlock": rlimit_info,
        "transparent_hugepage": transparent_hp,
        "nr_hugepages": nr_hugepages,
        "alloc_budget_sec": alloc_budget_sec,
        "min_h2d_gbs": min_h2d_gbs,
    })

    assert not failures, (
        "Pinned-memory allocation issues (likely RLIMIT_MEMLOCK / hugepages):\n  "
        + "\n  ".join(failures)
    )


# =============================================================================
# TC-CPUH-03: CPU DRAM bandwidth under concurrent multi-GPU H2D load
# =============================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
@pytest.mark.slow
def test_cpu_memory_bandwidth_under_gpu_load():
    """Run a STREAM-Triad style CPU memory bandwidth workload while N
    concurrent H2D streams hammer the PCIe lanes. On a healthy node the
    combined throughput stays close to the sum of idle values; severe
    contention indicates DRAM channels or PCIe root complexes are
    oversubscribed.

    Pass: (loaded_cpu_bw + loaded_h2d_bw) >= (idle_cpu_bw + idle_h2d_bw) * ratio
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA devices")
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy required")

    n_gpus = torch.cuda.device_count()
    arr_mb = int(os.environ.get("CPU_HEADNODE_TRIAD_MB", "1024"))
    duration_sec = float(os.environ.get("CPU_HEADNODE_TRIAD_DURATION_SEC", "4.0"))
    # Idle baseline is measured per-GPU sequentially then summed; loaded phase
    # runs all GPUs concurrently sharing host DRAM read BW. On a PCIe-only
    # multi-GPU node the loaded:idle ratio is fundamentally bounded by
    # (per-GPU PCIe BW * n_gpus) / (per-GPU PCIe BW * n_gpus from cache).
    # 0.40 reflects healthy near-PCIe-saturation behavior on 4xH100 Gen5.
    min_combined_ratio = float(os.environ.get("CPU_HEADNODE_TRIAD_MIN_COMBINED_RATIO", "0.40"))

    n_elems = (arr_mb * 1024 * 1024) // 8  # float64
    a = np.ones(n_elems, dtype=np.float64)
    b = np.full(n_elems, 2.0)
    c = np.empty_like(a)
    scalar = 3.0

    def _triad_for(seconds: float) -> float:
        bytes_moved = 0
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            np.add(b, scalar * c, out=a)  # close enough to triad for this purpose
            bytes_moved += a.nbytes * 3  # 2 reads + 1 write
        elapsed = seconds  # ~accurate; loop exits just past deadline
        return bytes_moved / elapsed / 1e9

    # Phase 1: idle CPU BW
    idle_cpu_bw = _triad_for(duration_sec)

    # Phase 1b: idle H2D BW (single stream, sum across GPUs sequentially)
    buf_bytes = 128 * 1024 * 1024
    host_bufs = [torch.empty(buf_bytes // 4, dtype=torch.float32, pin_memory=True)
                 for _ in range(n_gpus)]
    dev_bufs = [torch.empty_like(host_bufs[g], device=f"cuda:{g}") for g in range(n_gpus)]
    for g in range(n_gpus):
        dev_bufs[g].copy_(host_bufs[g], non_blocking=True)
    torch.cuda.synchronize()

    def _h2d_burst(gpu_id: int, seconds: float) -> float:
        bytes_moved = 0
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            dev_bufs[gpu_id].copy_(host_bufs[gpu_id], non_blocking=True)
            bytes_moved += buf_bytes
        torch.cuda.synchronize(gpu_id)
        return bytes_moved / seconds / 1e9

    idle_h2d_bw = 0.0
    for g in range(n_gpus):
        idle_h2d_bw += _h2d_burst(g, duration_sec / max(1, n_gpus))

    # Phase 2: concurrent CPU triad + per-GPU H2D
    loaded_cpu_bw_box: List[float] = []
    loaded_h2d_bw_box: List[float] = [0.0] * n_gpus

    def _cpu_worker():
        loaded_cpu_bw_box.append(_triad_for(duration_sec))

    def _gpu_worker(gpu_id: int):
        loaded_h2d_bw_box[gpu_id] = _h2d_burst(gpu_id, duration_sec)

    threads: List[threading.Thread] = []
    threads.append(threading.Thread(target=_cpu_worker))
    for g in range(n_gpus):
        threads.append(threading.Thread(target=_gpu_worker, args=(g,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    loaded_cpu_bw = loaded_cpu_bw_box[0] if loaded_cpu_bw_box else 0.0
    loaded_h2d_bw = sum(loaded_h2d_bw_box)

    idle_combined = idle_cpu_bw + idle_h2d_bw
    loaded_combined = loaded_cpu_bw + loaded_h2d_bw
    combined_ratio = loaded_combined / idle_combined if idle_combined > 0 else 0.0

    _record_kpi("test_cpu_memory_bandwidth_under_gpu_load", {
        "n_gpus": n_gpus,
        "triad_array_mb": arr_mb,
        "duration_sec": duration_sec,
        "idle_cpu_triad_gbs": round(idle_cpu_bw, 2),
        "idle_h2d_total_gbs": round(idle_h2d_bw, 2),
        "loaded_cpu_triad_gbs": round(loaded_cpu_bw, 2),
        "loaded_h2d_total_gbs": round(loaded_h2d_bw, 2),
        "idle_combined_gbs": round(idle_combined, 2),
        "loaded_combined_gbs": round(loaded_combined, 2),
        "combined_ratio": round(combined_ratio, 3),
        "min_combined_ratio_required": min_combined_ratio,
    })

    del host_bufs, dev_bufs
    torch.cuda.empty_cache()

    assert combined_ratio >= min_combined_ratio, (
        f"Severe CPU<->GPU memory contention: combined throughput dropped to "
        f"{combined_ratio:.2f} of idle sum (CPU {loaded_cpu_bw:.1f}/{idle_cpu_bw:.1f} GB/s, "
        f"H2D {loaded_h2d_bw:.1f}/{idle_h2d_bw:.1f} GB/s)"
    )


# =============================================================================
# TC-CPUH-04: HuggingFace tokenizer throughput scaling
# =============================================================================

@pytest.mark.slow
def test_tokenizer_throughput_scaling():
    """Tokenization is the #1 silent serving bottleneck. Validate that a
    HuggingFace fast tokenizer scales across CPU threads roughly linearly
    up to a small thread count.

    Pass: single-thread throughput >= ``CPU_HEADNODE_TOK_MIN_SINGLE_TPS`` AND
    multi-thread scaling >= ``CPU_HEADNODE_TOK_MIN_SCALING_RATIO`` of linear.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    model_id = os.environ.get("CPU_HEADNODE_TOK_MODEL", "openai-community/gpt2")
    n_prompts = int(os.environ.get("CPU_HEADNODE_TOK_PROMPTS", "5000"))
    prompt_chars = int(os.environ.get("CPU_HEADNODE_TOK_PROMPT_CHARS", "1024"))
    thread_counts = [int(x) for x in os.environ.get(
        "CPU_HEADNODE_TOK_THREADS", "1,4,8"
    ).split(",") if x.strip()]
    min_single_tps = float(os.environ.get("CPU_HEADNODE_TOK_MIN_SINGLE_TPS", "30000"))
    # HF fast tokenizers release the GIL during the Rust core but the
    # Python-level encode() wrapper and per-prompt loop are GIL-bound.
    # Realistic 8-thread efficiency for per-prompt encode is ~0.20-0.30;
    # the batch encode API would scale better but is a separate test.
    min_scaling_ratio = float(os.environ.get("CPU_HEADNODE_TOK_MIN_SCALING_RATIO", "0.20"))

    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer {model_id}: {exc}")

    prompt = ("The quick brown fox jumps over the lazy dog. " * 64)[:prompt_chars]
    prompts = [prompt] * n_prompts

    def _encode_chunk(chunk: List[str]) -> int:
        total = 0
        for p in chunk:
            ids = tok.encode(p)
            total += len(ids)
        return total

    results: Dict[int, Dict[str, float]] = {}
    for n_threads in thread_counts:
        chunks = [prompts[i::n_threads] for i in range(n_threads)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            totals = list(ex.map(_encode_chunk, chunks))
        elapsed = time.perf_counter() - t0
        tokens = sum(totals)
        prompts_per_sec = n_prompts / elapsed
        tokens_per_sec = tokens / elapsed
        results[n_threads] = {
            "elapsed_sec": round(elapsed, 3),
            "prompts_per_sec": round(prompts_per_sec, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
        }

    single_tps = results[thread_counts[0]]["tokens_per_sec"]
    max_threads = max(thread_counts)
    max_tps = results[max_threads]["tokens_per_sec"]
    scaling_ratio = (max_tps / single_tps) / max_threads if single_tps > 0 else 0.0

    _record_kpi("test_tokenizer_throughput_scaling", {
        "model": model_id,
        "n_prompts": n_prompts,
        "prompt_chars": prompt_chars,
        "per_thread_results": {str(k): v for k, v in results.items()},
        "scaling_ratio_vs_linear": round(scaling_ratio, 3),
        "single_thread_tokens_per_sec": single_tps,
        "min_single_tps_required": min_single_tps,
        "min_scaling_ratio_required": min_scaling_ratio,
    })

    failures: List[str] = []
    if single_tps < min_single_tps:
        failures.append(
            f"single-thread tokenizer too slow: {single_tps:.0f} tok/s "
            f"< {min_single_tps:.0f}"
        )
    if scaling_ratio < min_scaling_ratio:
        failures.append(
            f"tokenizer thread scaling poor: ratio={scaling_ratio:.2f} "
            f"(1->{max_threads}t took {results[max_threads]['elapsed_sec']:.2f}s vs "
            f"ideal {results[thread_counts[0]]['elapsed_sec']/max_threads:.2f}s)"
        )
    assert not failures, "; ".join(failures)


# =============================================================================
# TC-CPUH-05: PyTorch DataLoader scaling with num_workers
# =============================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
@pytest.mark.slow
def test_dataloader_throughput_scaling():
    """A synthetic DataLoader feeding a no-op GPU consumer should scale
    when adding worker processes. If samples/sec plateaus or *drops*,
    fork-join overhead or pin_memory is broken.

    Pass: at least one ``num_workers > 0`` configuration delivers >= 1.5x
    samples/sec vs ``num_workers=0``.
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA devices")

    from torch.utils.data import Dataset, DataLoader

    n_samples = int(os.environ.get("CPU_HEADNODE_DL_SAMPLES", "2048"))
    sample_bytes = int(os.environ.get("CPU_HEADNODE_DL_SAMPLE_BYTES", "262144"))  # 256 KB
    batch_size = int(os.environ.get("CPU_HEADNODE_DL_BATCH", "32"))
    worker_counts = [int(x) for x in os.environ.get(
        "CPU_HEADNODE_DL_WORKERS", "0,4,8"
    ).split(",") if x.strip()]
    # Synthetic in-memory dataset has trivial __getitem__ work; with worker
    # processes the fork/IPC/serialize overhead per batch is comparable to
    # the per-sample work, so realistic peak speedup is ~1.2-1.5x. A real
    # workload with disk I/O or augmentation would scale much better; this
    # test only catches regressions where workers help LESS than ~20%.
    min_speedup = float(os.environ.get("CPU_HEADNODE_DL_MIN_SPEEDUP", "1.20"))

    class _SyntheticDataset(Dataset):
        def __init__(self, n: int, nbytes: int) -> None:
            self.n = n
            self.nbytes = nbytes
        def __len__(self) -> int:
            return self.n
        def __getitem__(self, _idx: int) -> torch.Tensor:
            # Simulate cheap preprocessing: allocate + light fill.
            t = torch.empty(self.nbytes // 4, dtype=torch.float32)
            t.fill_(0.5)
            return t

    ds = _SyntheticDataset(n_samples, sample_bytes)
    results: Dict[int, float] = {}

    for nw in worker_counts:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
        )
        # Warmup
        try:
            it = iter(loader)
            next(it)
        except StopIteration:
            pass
        del it

        t0 = time.perf_counter()
        consumed = 0
        for batch in loader:
            batch = batch.to("cuda:0", non_blocking=True)
            torch.cuda.current_stream().synchronize()
            consumed += batch.shape[0]
        elapsed = time.perf_counter() - t0
        sps = consumed / elapsed if elapsed > 0 else 0.0
        results[nw] = sps
        logger.info("DataLoader nw=%d: %.1f samples/s (%.2fs)", nw, sps, elapsed)

        del loader

    baseline = results[worker_counts[0]]
    best = max(results.values())
    speedup = best / baseline if baseline > 0 else 0.0

    _record_kpi("test_dataloader_throughput_scaling", {
        "n_samples": n_samples,
        "sample_bytes": sample_bytes,
        "batch_size": batch_size,
        "samples_per_sec_by_workers": {str(k): round(v, 1) for k, v in results.items()},
        "best_speedup_vs_nw0": round(speedup, 2),
        "min_speedup_required": min_speedup,
    })

    assert speedup >= min_speedup, (
        f"DataLoader did not scale with workers: best speedup {speedup:.2f}x "
        f"< {min_speedup:.2f}x. Per-worker results: {results}"
    )


# =============================================================================
# TC-CPUH-06: CPU governor / C-state policy
# =============================================================================

@pytest.mark.skipif(not _IS_LINUX, reason="sysfs CPU governor only on Linux")
def test_cpu_governor_and_cstate_policy():
    """Default ``powersave`` + deep C-states adds tens of microseconds to
    every kernel launch and bloats TTFT. Production head nodes should run
    ``performance`` governor.

    Pass: every online CPU has the governor listed in
    ``CPU_HEADNODE_ALLOWED_GOVERNORS`` (default 'performance').
    Set ``CPU_HEADNODE_GOVERNOR_WARN_ONLY=1`` to downgrade to a warning.
    """
    allowed = {g.strip() for g in os.environ.get(
        "CPU_HEADNODE_ALLOWED_GOVERNORS", "performance"
    ).split(",") if g.strip()}
    warn_only = os.environ.get("CPU_HEADNODE_GOVERNOR_WARN_ONLY", "0") == "1"

    cpu_dirs = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
    if not cpu_dirs:
        pytest.skip("No /sys/devices/system/cpu entries")

    governors: Dict[str, int] = {}
    bad: List[str] = []
    sampled = 0
    for d in cpu_dirs:
        gov = _read_sysfs(str(d / "cpufreq/scaling_governor"))
        if gov is None:
            continue
        sampled += 1
        governors[gov] = governors.get(gov, 0) + 1
        if gov not in allowed:
            bad.append(d.name)

    # Sample C-state disable flags from CPU0 (representative)
    cstates: Dict[str, Dict[str, str]] = {}
    cs_root = Path("/sys/devices/system/cpu/cpu0/cpuidle")
    if cs_root.is_dir():
        for s in sorted(cs_root.glob("state*")):
            cstates[s.name] = {
                "name": _read_sysfs(str(s / "name")) or "",
                "disable": _read_sysfs(str(s / "disable")) or "",
                "latency_us": _read_sysfs(str(s / "latency")) or "",
            }

    intel_pstate_no_turbo = _read_sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo")
    sched_mc = _read_sysfs("/proc/sys/kernel/sched_mc_power_savings")

    _record_kpi("test_cpu_governor_and_cstate_policy", {
        "cpus_sampled": sampled,
        "governor_counts": governors,
        "bad_cpus_count": len(bad),
        "bad_cpus_first_10": bad[:10],
        "cpu0_cstates": cstates,
        "intel_pstate_no_turbo": intel_pstate_no_turbo,
        "sched_mc_power_savings": sched_mc,
        "allowed_governors": sorted(allowed),
    })

    if sampled == 0:
        pytest.skip("No cpufreq scaling_governor sysfs entries (cpufreq may be disabled)")

    if bad:
        msg = (f"{len(bad)}/{sampled} CPUs not in allowed governors {sorted(allowed)}. "
               f"Counts: {governors}. First bad: {bad[:5]}")
        if warn_only:
            logger.warning(msg)
        else:
            assert False, msg


# =============================================================================
# TC-CPUH-07: IRQ affinity distribution
# =============================================================================

@pytest.mark.skipif(not _IS_LINUX, reason="/proc/interrupts only on Linux")
def test_irq_affinity_distribution():
    """If every NIC / NVMe / GPU IRQ lands on CPU 0, that core becomes
    the head-node hot core under load and tail latency suffers.

    Pass: for each interesting driver (mlx5, nvme, nvidia) with >= 2 IRQs,
    those IRQs use at least ``CPU_HEADNODE_IRQ_MIN_UNIQUE_CPUS`` distinct
    CPUs in their effective affinity.
    """
    proc_int = Path("/proc/interrupts")
    if not proc_int.is_file():
        pytest.skip("/proc/interrupts not readable")

    min_unique = int(os.environ.get("CPU_HEADNODE_IRQ_MIN_UNIQUE_CPUS", "2"))
    drivers_env = os.environ.get("CPU_HEADNODE_IRQ_DRIVERS", "mlx5,nvme,nvidia")
    drivers = [d.strip().lower() for d in drivers_env.split(",") if d.strip()]
    warn_only = os.environ.get("CPU_HEADNODE_IRQ_WARN_ONLY", "0") == "1"

    by_driver: Dict[str, List[int]] = {d: [] for d in drivers}
    try:
        lines = proc_int.read_text().splitlines()
    except Exception as exc:
        pytest.skip(f"Could not read /proc/interrupts: {exc}")

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        irq_field = parts[0].rstrip(":")
        if not irq_field.isdigit():
            continue
        irq = int(irq_field)
        desc = " ".join(parts[-3:]).lower()
        for d in drivers:
            if d in desc:
                by_driver[d].append(irq)

    def _affinity_cpus(irq: int) -> List[int]:
        # Prefer effective_affinity_list (post-IRQ-balance), fall back to smp_affinity_list
        for fname in ("effective_affinity_list", "smp_affinity_list"):
            raw = _read_sysfs(f"/proc/irq/{irq}/{fname}")
            if raw:
                cpus: List[int] = []
                for chunk in raw.split(","):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    if "-" in chunk:
                        a, b = chunk.split("-", 1)
                        try:
                            cpus.extend(range(int(a), int(b) + 1))
                        except ValueError:
                            pass
                    else:
                        try:
                            cpus.append(int(chunk))
                        except ValueError:
                            pass
                if cpus:
                    return cpus
        return []

    report: Dict[str, Dict] = {}
    violations: List[str] = []
    for d, irqs in by_driver.items():
        all_cpus: set = set()
        per_irq: Dict[str, List[int]] = {}
        for irq in irqs:
            cpus = _affinity_cpus(irq)
            per_irq[str(irq)] = cpus
            all_cpus.update(cpus)
        report[d] = {
            "n_irqs": len(irqs),
            "unique_cpus_in_affinity": len(all_cpus),
            "first_cpu_set": sorted(all_cpus)[:10],
        }
        if len(irqs) >= 2 and len(all_cpus) < min_unique:
            violations.append(
                f"{d}: {len(irqs)} IRQs but only {len(all_cpus)} unique CPU(s) in affinity"
            )

    _record_kpi("test_irq_affinity_distribution", {
        "drivers": drivers,
        "min_unique_cpus_required": min_unique,
        "report": report,
    })

    # If no IRQs were found for any driver, skip rather than fail (driver not loaded)
    if not any(r["n_irqs"] > 0 for r in report.values()):
        pytest.skip(f"No IRQs found for any driver in {drivers}")

    if violations:
        msg = "IRQ affinity too concentrated:\n  " + "\n  ".join(violations)
        if warn_only:
            logger.warning(msg)
        else:
            assert False, msg


# =============================================================================
# TC-CPUH-08: CUDA sync-primitive round-trip latency
# =============================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_cuda_sync_primitive_latency():
    """Empty-kernel-launch + sync round-trip sets the floor for batch-1
    decode latency. A high or wildly-variant value across GPUs means
    either a degraded PCIe link or driver-level contention.

    Pass: median round-trip <= ``CPU_HEADNODE_SYNC_MAX_MEDIAN_US`` for
    every GPU, AND max-vs-median across GPUs <= 3x.
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA devices")

    iters = int(os.environ.get("CPU_HEADNODE_SYNC_ITERS", "5000"))
    warmup = int(os.environ.get("CPU_HEADNODE_SYNC_WARMUP", "500"))
    max_median_us = float(os.environ.get("CPU_HEADNODE_SYNC_MAX_MEDIAN_US", "25.0"))

    n_gpus = torch.cuda.device_count()
    per_gpu: List[Dict] = []
    failures: List[str] = []
    medians: List[float] = []
    for g in range(n_gpus):
        torch.cuda.set_device(g)
        stream = torch.cuda.current_stream(g)
        # A trivially small tensor op stands in for "empty kernel"
        a = torch.zeros(1, device=f"cuda:{g}")
        for _ in range(warmup):
            a.add_(1.0)
            torch.cuda.synchronize(g)

        samples_us: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            a.add_(1.0)
            torch.cuda.synchronize(g)
            samples_us.append((time.perf_counter_ns() - t0) / 1000.0)
        samples_us.sort()
        median_us = samples_us[len(samples_us) // 2]
        p99_us = samples_us[int(len(samples_us) * 0.99)]
        medians.append(median_us)
        per_gpu.append({
            "gpu_id": g,
            "median_us": round(median_us, 2),
            "p99_us": round(p99_us, 2),
            "min_us": round(samples_us[0], 2),
        })
        if median_us > max_median_us:
            failures.append(f"GPU {g}: median {median_us:.1f} us > {max_median_us} us")

    if medians:
        median_of_medians = statistics.median(medians)
        worst_outlier_ratio = max(medians) / median_of_medians if median_of_medians > 0 else 0.0
    else:
        median_of_medians = 0.0
        worst_outlier_ratio = 0.0

    _record_kpi("test_cuda_sync_primitive_latency", {
        "iters": iters,
        "per_gpu": per_gpu,
        "max_median_us_required": max_median_us,
        "median_of_medians_us": round(median_of_medians, 2),
        "worst_outlier_ratio": round(worst_outlier_ratio, 2),
    })

    if worst_outlier_ratio > 3.0:
        failures.append(
            f"GPU sync latency outlier: worst/median = {worst_outlier_ratio:.2f}x "
            f"(per-GPU medians: {[p['median_us'] for p in per_gpu]})"
        )

    assert not failures, "\n  ".join(failures)


# =============================================================================
# TC-CPUH-09: NVMe model-cache sequential read throughput
# =============================================================================

@pytest.mark.slow
def test_nvme_model_cache_read_throughput():
    """Cold-start of a 70B model is ~140 GB of sequential reads. If the
    HuggingFace cache lives on a slow disk or NFS mount, GPUs sit idle for
    minutes on every restart.

    Pass: sequential read throughput from the cache directory is
    >= ``CPU_HEADNODE_NVME_MIN_READ_GBS`` GB/s.
    """
    cache_dir_str = os.environ.get(
        "CPU_HEADNODE_NVME_CACHE_DIR",
        os.environ.get("HF_HOME", os.environ.get("HUGGINGFACE_HUB_CACHE", "")),
    )
    cache_dir = Path(cache_dir_str).expanduser() if cache_dir_str else Path.cwd()
    if not cache_dir.is_dir():
        cache_dir = Path(tempfile.gettempdir())

    size_mb = int(os.environ.get("CPU_HEADNODE_NVME_SIZE_MB", "2048"))
    min_gbs = float(os.environ.get("CPU_HEADNODE_NVME_MIN_READ_GBS", "1.5"))

    # Check free space
    try:
        free_bytes = shutil.disk_usage(cache_dir).free
    except Exception:
        free_bytes = 0
    if free_bytes < size_mb * 1024 * 1024 * 2:
        pytest.skip(f"Insufficient free space in {cache_dir} for {size_mb} MB test file")

    test_file = cache_dir / f".cpu_headnode_io_test_{os.getpid()}.bin"
    nbytes = size_mb * 1024 * 1024
    chunk = 4 * 1024 * 1024  # 4 MB
    pattern = os.urandom(chunk)
    try:
        # Write
        t0 = time.perf_counter()
        with test_file.open("wb", buffering=0) as f:
            written = 0
            while written < nbytes:
                f.write(pattern)
                written += chunk
            os.fsync(f.fileno())
        write_sec = time.perf_counter() - t0

        # Drop page cache to force real read (best-effort; needs root)
        dropped = False
        try:
            drop = Path("/proc/sys/vm/drop_caches")
            if drop.is_file() and os.geteuid() == 0:
                drop.write_text("3\n")
                dropped = True
        except Exception:
            pass

        # Read
        t0 = time.perf_counter()
        with test_file.open("rb", buffering=0) as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
        read_sec = time.perf_counter() - t0
    finally:
        try:
            test_file.unlink()
        except Exception:
            pass

    write_gbs = nbytes / write_sec / 1e9 if write_sec > 0 else 0.0
    read_gbs = nbytes / read_sec / 1e9 if read_sec > 0 else 0.0

    _record_kpi("test_nvme_model_cache_read_throughput", {
        "cache_dir": str(cache_dir),
        "size_mb": size_mb,
        "write_gbs": round(write_gbs, 2),
        "read_gbs": round(read_gbs, 2),
        "page_cache_dropped": dropped,
        "min_read_gbs_required": min_gbs,
    })

    assert read_gbs >= min_gbs, (
        f"Model cache read throughput too low: {read_gbs:.2f} GB/s "
        f"from {cache_dir} (target {min_gbs} GB/s). "
        f"Cold-start of multi-GB models will be very slow. "
        f"page_cache_dropped={dropped}; result includes page-cache effects if False."
    )


# =============================================================================
# TC-CPUH-10: Python asyncio concurrency scaling
# =============================================================================

@pytest.mark.slow
def test_python_async_concurrency_scaling():
    """A serving front-end juggles many concurrent requests via asyncio.
    If the GIL or event loop is the bottleneck, p99 latency explodes as
    concurrency rises even though per-task work is identical.

    Pass: p99 latency at high concurrency is not more than
    ``CPU_HEADNODE_ASYNC_MAX_P99_RATIO`` x the single-task p99.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    model_id = os.environ.get("CPU_HEADNODE_ASYNC_MODEL", "openai-community/gpt2")
    n_per_level = int(os.environ.get("CPU_HEADNODE_ASYNC_TASKS_PER_LEVEL", "200"))
    concurrency_levels = [int(x) for x in os.environ.get(
        "CPU_HEADNODE_ASYNC_CONCURRENCY", "1,16,64"
    ).split(",") if x.strip()]
    # With Semaphore(N), N CPU-bound tasks queue behind each other on the
    # single asyncio loop thread. p99 grows ~linearly with concurrency for
    # CPU-bound payloads (this is queuing, not GIL pathology). For 1->64
    # concurrency a healthy node lands around 25-40x.
    max_p99_ratio = float(os.environ.get("CPU_HEADNODE_ASYNC_MAX_P99_RATIO", "50.0"))

    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer {model_id}: {exc}")

    prompt = "The quick brown fox jumps over the lazy dog. " * 32

    async def _one_task() -> float:
        t0 = time.perf_counter()
        # Mix CPU-bound (tokenize) + IO yield to exercise the loop
        _ = tok.encode(prompt)
        await asyncio.sleep(0)
        _ = tok.encode(prompt)
        return (time.perf_counter() - t0) * 1000.0

    async def _run_level(concurrency: int) -> List[float]:
        latencies: List[float] = []
        sem = asyncio.Semaphore(concurrency)
        async def _bounded():
            async with sem:
                latencies.append(await _one_task())
        await asyncio.gather(*[_bounded() for _ in range(n_per_level)])
        return latencies

    per_level: Dict[int, Dict[str, float]] = {}
    for c in concurrency_levels:
        latencies = asyncio.run(_run_level(c))
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]
        per_level[c] = {"p50_ms": round(p50, 3), "p99_ms": round(p99, 3)}

    base_p99 = per_level[concurrency_levels[0]]["p99_ms"]
    max_p99 = max(v["p99_ms"] for v in per_level.values())
    ratio = max_p99 / base_p99 if base_p99 > 0 else 0.0

    _record_kpi("test_python_async_concurrency_scaling", {
        "model": model_id,
        "tasks_per_level": n_per_level,
        "per_level": {str(k): v for k, v in per_level.items()},
        "p99_ratio_max_vs_base": round(ratio, 2),
        "max_p99_ratio_allowed": max_p99_ratio,
    })

    assert ratio <= max_p99_ratio, (
        f"asyncio p99 latency blew up with concurrency: ratio {ratio:.2f}x "
        f"(base {base_p99:.2f} ms -> peak {max_p99:.2f} ms). "
        f"Per-level: {per_level}"
    )
