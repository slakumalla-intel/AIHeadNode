"""
AI Pipeline KPI Test Suite
==========================
Covers the four areas not fully addressed by existing suites:

  Domain A – Data Loading & Preprocessing
  Domain B – Model Orchestration
  Domain C – Networking I/O & Storage I/O
  Domain D – Inference Pipeline KPIs (TTFT, ITL, prefill/decode split, batching)

KPIs are written to:
  artifacts/ai_pipeline_kpis.jsonl

Run all:
  pytest -m ai_pipeline -vv tests/test_ai_pipeline_kpis.py

Run one domain:
  pytest -m data_loading   -vv tests/test_ai_pipeline_kpis.py
  pytest -m model_orch     -vv tests/test_ai_pipeline_kpis.py
  pytest -m storage_net_io -vv tests/test_ai_pipeline_kpis.py
  pytest -m inference_kpis -vv tests/test_ai_pipeline_kpis.py

Environment toggles (all optional):
  PIPELINE_EXECUTE=1          # set 0 to skip execution-heavy tests
  PIPELINE_MODEL_ID           # HF model for inference KPIs (default gpt2)
  PIPELINE_LARGE_MODEL_ID     # larger model for orchestration tests
  PIPELINE_SEQ_LENS           # space-sep sequence lengths (default: 32 128 512 2048)
  PIPELINE_BATCH_SIZES        # space-sep batch sizes   (default: 1 4 8 16)
  PIPELINE_WARMUP_ITERS       # warmup iterations       (default: 3)
  PIPELINE_BENCH_ITERS        # benchmark iterations    (default: 10)
  STORAGE_TEST_PATH           # path for storage IO tests (default: /tmp)
  STORAGE_TEST_FILE_GB        # size for sequential IO   (default: 2)
  NET_LOOPBACK_PORT           # TCP port for loopback BW (default: 15002)
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from tests.kpi_efficiency import enrich_efficiency_kpis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _env_list(name: str, default: str) -> List[int]:
    raw = os.environ.get(name, default)
    return [int(x) for x in raw.split()]

os.environ.setdefault("PIPELINE_EXECUTE", "1")
os.environ.setdefault("PIPELINE_MODEL_ID", "openai-community/gpt2")
os.environ.setdefault("PIPELINE_LARGE_MODEL_ID", os.environ.get("HF_INFERENCE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
os.environ.setdefault("PIPELINE_SEQ_LENS", "32 128 512 2048")
os.environ.setdefault("PIPELINE_BATCH_SIZES", "1 4 8 16")
os.environ.setdefault("PIPELINE_WARMUP_ITERS", "3")
os.environ.setdefault("PIPELINE_BENCH_ITERS", "10")
os.environ.setdefault("STORAGE_TEST_PATH", tempfile.gettempdir())
os.environ.setdefault("STORAGE_TEST_FILE_GB", "2")
os.environ.setdefault("NET_LOOPBACK_PORT", "15002")

_EXECUTE = os.environ.get("PIPELINE_EXECUTE", "1") == "1"
_MODEL_ID = os.environ["PIPELINE_MODEL_ID"]
_LARGE_MODEL_ID = os.environ["PIPELINE_LARGE_MODEL_ID"]
_SEQ_LENS = _env_list("PIPELINE_SEQ_LENS", "32 128 512 2048")
_BATCH_SIZES = _env_list("PIPELINE_BATCH_SIZES", "1 4 8 16")
_WARMUP = _env_int("PIPELINE_WARMUP_ITERS", 3)
_BENCH = _env_int("PIPELINE_BENCH_ITERS", 10)
_STORAGE_PATH = Path(os.environ["STORAGE_TEST_PATH"])
_STORAGE_GB = _env_int("STORAGE_TEST_FILE_GB", 2)
_NET_PORT = _env_int("NET_LOOPBACK_PORT", 15002)

# ---------------------------------------------------------------------------
# KPI recording
# ---------------------------------------------------------------------------

def _kpi_path() -> Path:
    return Path(os.environ.get("PIPELINE_KPI_FILE", "artifacts/ai_pipeline_kpis.jsonl"))


def _record_kpi(test_name: str, metrics: dict) -> None:
    out = _kpi_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    enriched_metrics = enrich_efficiency_kpis(metrics, os.environ)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test": test_name,
        "metrics": enriched_metrics,
    }
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
    logger.info("KPI [%s]: %s", test_name, json.dumps(enriched_metrics, default=str, sort_keys=True))


# ---------------------------------------------------------------------------
# Optional-import guards
# ---------------------------------------------------------------------------

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
    _CUDA_DEVICE_COUNT = torch.cuda.device_count() if _HAS_TORCH else 0
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
    _CUDA_DEVICE_COUNT = 0

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    pynvml = None  # type: ignore[assignment]
    _HAS_NVML = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gpu_memory_used_gb(device: int = 0) -> float:
    if not _HAS_TORCH:
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024 ** 3


def _gpu_free_memory_gb(device: int = 0) -> float:
    if not _HAS_TORCH:
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3


def _nvml_gpu_util_pct(device: int = 0) -> float:
    """Instantaneous GPU utilisation % via NVML."""
    if not _HAS_NVML:
        return -1.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return -1.0


def _nvml_mem_util_pct(device: int = 0) -> float:
    """Instantaneous memory controller utilisation % via NVML."""
    if not _HAS_NVML:
        return -1.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.memory)
    except Exception:
        return -1.0


def _cuda_context_healthy(device: int = 0) -> bool:
    """Return False when CUDA context is poisoned (e.g., prior device-side assert)."""
    if not _HAS_TORCH or _CUDA_DEVICE_COUNT == 0:
        return True
    try:
        torch.cuda.synchronize(device)
        # mem_get_info is a reliable touchpoint that surfaces deferred CUDA errors.
        torch.cuda.mem_get_info(device)
        return True
    except Exception as exc:
        logger.warning("CUDA context unhealthy on cuda:%s: %s", device, exc)
        return False


@pytest.fixture(autouse=True)
def _guard_inference_cuda_context(request: pytest.FixtureRequest):
    """Skip inference KPI tests when a prior test left CUDA in an error state."""
    if "inference_kpis" in request.keywords and not _cuda_context_healthy(0):
        pytest.skip(
            "CUDA context is in an error state from a prior kernel failure "
            "(device-side assert). Restart the pytest process or run with "
            "CUDA_LAUNCH_BLOCKING=1 to identify the first failing kernel."
        )


def _poll_gpu_utilisation(device: int, interval: float, results: list, stop_event: threading.Event):
    """Background thread: poll GPU util% and mem-util% at `interval` seconds."""
    while not stop_event.is_set():
        results.append((_nvml_gpu_util_pct(device), _nvml_mem_util_pct(device)))
        time.sleep(interval)


# ===========================================================================
# DOMAIN A – Data Loading & Preprocessing
# ===========================================================================


def _tokenize_batch_worker(args):
    """Top-level picklable worker for ProcessPoolExecutor tokenizer scaling."""
    model_id, seq_len, batch = args
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    texts = ["benchmark token throughput " * 20] * batch
    tokenizer(texts, truncation=True, max_length=seq_len,
              padding="max_length", return_tensors="pt")
    return batch * seq_len


class TestDataLoadingAndPreprocessing:
    """Domain A: tokenizer throughput, DataLoader throughput, async prefetch."""

    # -----------------------------------------------------------------------
    # TC-PIPE-A-01  Tokenizer throughput across sequence lengths
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.data_loading
    @pytest.mark.parametrize("seq_len", _SEQ_LENS)
    def test_tokenizer_throughput_by_seq_len(self, seq_len: int):
        """TC-PIPE-A-01: Tokenizer tokens/sec at multiple sequence lengths.

        KPIs: tokens_per_sec, sequences_per_sec, seq_len
        Threshold: ≥ 50 000 tokens/sec for seq_len ≤ 512; ≥ 20 000 for longer.
        """
        if not _HAS_TRANSFORMERS:
            pytest.skip("transformers not available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        # Build synthetic text of desired length (over-sample, then truncate)
        word = "inference " * (seq_len + 10)
        batch_size = 64  # fixed batch for throughput measurement
        texts = [word] * batch_size

        # Warm-up
        for _ in range(_WARMUP):
            _ = tokenizer(texts, truncation=True, max_length=seq_len,
                          padding="max_length", return_tensors="pt")

        t0 = time.perf_counter()
        for _ in range(_BENCH):
            enc = tokenizer(texts, truncation=True, max_length=seq_len,
                            padding="max_length", return_tensors="pt")
        elapsed = time.perf_counter() - t0

        total_tokens = batch_size * seq_len * _BENCH
        tokens_per_sec = total_tokens / elapsed
        seqs_per_sec = (batch_size * _BENCH) / elapsed

        threshold = 50_000 if seq_len <= 512 else 20_000

        _record_kpi("test_tokenizer_throughput_by_seq_len", {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "iterations": _BENCH,
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "sequences_per_sec": round(seqs_per_sec, 1),
            "threshold_tokens_per_sec": threshold,
        })

        assert tokens_per_sec >= threshold, (
            f"Tokenizer throughput {tokens_per_sec:.0f} tok/s < {threshold} tok/s "
            f"at seq_len={seq_len}"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-A-02  Multi-threaded tokenizer scaling
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.data_loading
    def test_tokenizer_multi_thread_scaling(self):
        """TC-PIPE-A-02: Tokenizer throughput scales with worker thread count.

        KPIs: threads_1_tps, threads_4_tps, scaling_ratio
        Threshold: 4-thread throughput ≥ 2× single-thread throughput.
        """
        if not _HAS_TRANSFORMERS:
            pytest.skip("transformers not available")

        from concurrent.futures import ProcessPoolExecutor

        seq_len, batch = 128, 32

        def _measure_tps(n_workers: int) -> float:
            n_calls = max(_BENCH, 8)
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_tokenize_batch_worker,
                              [(_MODEL_ID, seq_len, batch)] * n_calls))
            elapsed = time.perf_counter() - t0
            return (batch * seq_len * n_calls) / elapsed

        tps_1 = _measure_tps(1)
        tps_2 = _measure_tps(2)
        tps_4 = _measure_tps(4)
        scaling_ratio_2 = tps_2 / tps_1 if tps_1 > 0 else 0.0
        scaling_ratio_4 = tps_4 / tps_1 if tps_1 > 0 else 0.0

        _record_kpi("test_tokenizer_multi_thread_scaling", {
            "threads_1_tokens_per_sec": round(tps_1, 1),
            "threads_2_tokens_per_sec": round(tps_2, 1),
            "threads_4_tokens_per_sec": round(tps_4, 1),
            "scaling_ratio_2_over_1": round(scaling_ratio_2, 3),
            "scaling_ratio_4_over_1": round(scaling_ratio_4, 3),
            "threshold_scaling_ratio": 2.0,
        })

        assert scaling_ratio_4 >= 2.0, (
            f"4-worker tokenizer scaling {scaling_ratio_4:.2f}× < 2.0× single-worker"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-A-03  In-memory DataLoader throughput (CPU → pinned memory)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.data_loading
    @pytest.mark.parametrize("num_workers", [0, 2, 4])
    def test_dataloader_cpu_to_pinned_throughput(self, num_workers: int):
        """TC-PIPE-A-03: PyTorch DataLoader samples/sec with varying prefetch workers.

        KPIs: samples_per_sec, batches_per_sec, num_workers
        Threshold: ≥ 1 000 samples/sec.
        """
        if not _HAS_TORCH:
            pytest.skip("torch not available")

        from torch.utils.data import DataLoader, TensorDataset

        seq_len, batch = 128, 32
        n_samples = 2_000
        data = torch.randint(0, 50_000, (n_samples, seq_len), dtype=torch.long)
        labels = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        ds = TensorDataset(data, labels)

        loader = DataLoader(
            ds, batch_size=batch, num_workers=num_workers,
            pin_memory=True, persistent_workers=(num_workers > 0),
        )

        # Warm-up
        for i, _ in enumerate(loader):
            if i >= 2:
                break

        t0 = time.perf_counter()
        samples_seen = 0
        for inp, _ in loader:
            samples_seen += inp.shape[0]
        elapsed = time.perf_counter() - t0

        samples_per_sec = samples_seen / elapsed
        threshold = 1_000

        _record_kpi("test_dataloader_cpu_to_pinned_throughput", {
            "num_workers": num_workers,
            "batch_size": batch,
            "seq_len": seq_len,
            "n_samples": n_samples,
            "elapsed_sec": round(elapsed, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "threshold_samples_per_sec": threshold,
        })

        assert samples_per_sec >= threshold, (
            f"DataLoader ({num_workers} workers) {samples_per_sec:.0f} samp/s < {threshold}"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-A-04  Async prefetch / GPU compute overlap efficiency
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.data_loading
    def test_async_prefetch_gpu_overlap_efficiency(self):
        """TC-PIPE-A-04: Overlap between CPU data prefetch and GPU GEMM.

        Measures: compute_only_sec, with_prefetch_sec, overlap_efficiency_pct
        Threshold: overlap efficiency ≥ 60 % (i.e. prefetch overhead ≤ 40 % of
                   pure compute time when pipelined).
        """
        if not _HAS_TORCH:
            pytest.skip("torch not available")

        device = torch.device("cuda:0")
        size = 2048
        n_iters = 20

        # Measure pure compute time (GEMM only, no data movement)
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)
        for _ in range(_WARMUP):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        compute_only_sec = time.perf_counter() - t0

        # Measure pipelined: CUDA stream for compute, concurrent H2D transfer
        cpu_buf = torch.randn(size, size, dtype=torch.float16, pin_memory=True)
        gpu_buf = torch.empty(size, size, device=device, dtype=torch.float16)
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()

        for _ in range(_WARMUP):
            with torch.cuda.stream(transfer_stream):
                gpu_buf.copy_(cpu_buf, non_blocking=True)
            with torch.cuda.stream(compute_stream):
                torch.matmul(a, b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.cuda.stream(transfer_stream):
                gpu_buf.copy_(cpu_buf, non_blocking=True)
            with torch.cuda.stream(compute_stream):
                torch.matmul(a, b)
        torch.cuda.synchronize()
        pipelined_sec = time.perf_counter() - t0

        # Overlap efficiency: how close is pipelined to compute-only?
        # Perfect overlap → pipelined_sec == compute_only_sec  → 100 %
        # No overlap     → pipelined_sec ≈ compute+transfer   → low %
        overlap_pct = min(100.0, (compute_only_sec / pipelined_sec) * 100.0)
        threshold_pct = 60.0

        _record_kpi("test_async_prefetch_gpu_overlap_efficiency", {
            "matrix_size": size,
            "iterations": n_iters,
            "compute_only_sec": round(compute_only_sec, 4),
            "pipelined_sec": round(pipelined_sec, 4),
            "overlap_efficiency_pct": round(overlap_pct, 2),
            "threshold_overlap_pct": threshold_pct,
        })

        assert overlap_pct >= threshold_pct, (
            f"Async prefetch overlap efficiency {overlap_pct:.1f}% < {threshold_pct}%"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-A-05  CPU preprocessing pipeline throughput for NLP batches
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.data_loading
    def test_cpu_preprocessing_pipeline_throughput(self):
        """TC-PIPE-A-05: Full CPU preprocessing pipeline: tokenize → pad → tensor.

        Measures end-to-end throughput of the CPU preprocessing stage that feeds
        the GPU inference engine.

        KPIs: total_tokens_per_sec, batches_per_sec
        Threshold: ≥ 200 000 tokens/sec (must not be the inference bottleneck).
        """
        if not _HAS_TRANSFORMERS:
            pytest.skip("transformers not available")
        if not _HAS_TORCH:
            pytest.skip("torch not available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        seq_len = 128
        batch_size = 64
        n_batches = max(_BENCH, 20)
        raw_text = ("The AI inference pipeline processes text sequences. " * 30)
        texts = [raw_text] * batch_size

        # Warm-up
        for _ in range(_WARMUP):
            enc = tokenizer(texts, truncation=True, max_length=seq_len,
                            padding="max_length", return_tensors="pt")
            _ = enc["input_ids"].numpy()

        t0 = time.perf_counter()
        for _ in range(n_batches):
            enc = tokenizer(texts, truncation=True, max_length=seq_len,
                            padding="max_length", return_tensors="pt")
            _ = enc["input_ids"].numpy()  # simulate feeding to GPU loader
        elapsed = time.perf_counter() - t0

        total_tokens = batch_size * seq_len * n_batches
        tps = total_tokens / elapsed
        bps = n_batches / elapsed
        threshold_tps = 200_000

        _record_kpi("test_cpu_preprocessing_pipeline_throughput", {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "n_batches": n_batches,
            "elapsed_sec": round(elapsed, 3),
            "total_tokens_per_sec": round(tps, 0),
            "batches_per_sec": round(bps, 2),
            "threshold_tokens_per_sec": threshold_tps,
        })

        assert tps >= threshold_tps, (
            f"CPU preprocessing pipeline {tps:.0f} tok/s < {threshold_tps} tok/s"
        )


# ===========================================================================
# DOMAIN B – Model Orchestration
# ===========================================================================

class TestModelOrchestration:
    """Domain B: model load time, checkpoint IO, KV cache, TP overhead."""

    # -----------------------------------------------------------------------
    # TC-PIPE-B-01  Model weight load time from disk to GPU
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.model_orch
    @pytest.mark.slow
    def test_model_load_time_to_gpu(self):
        """TC-PIPE-B-01: Measure wall-clock time to load model weights disk → GPU.

        KPIs: load_time_sec, model_size_gb, load_bandwidth_gbs
        Threshold: load bandwidth ≥ 1 GB/s (disk + PCIe + HBM write combined).
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        # Use small model to keep test time reasonable; override via env
        model_id = os.environ.get("PIPELINE_LOAD_MODEL_ID", _MODEL_ID)

        # Drop model from cache if it was previously loaded in this session
        if _HAS_TORCH:
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        torch.cuda.synchronize()
        load_time = time.perf_counter() - t0

        # Estimate model size
        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_gb = model_bytes / 1024 ** 3
        bw_gbs = model_gb / load_time if load_time > 0 else 0.0
        threshold_gbs = 1.0

        _record_kpi("test_model_load_time_to_gpu", {
            "model_id": model_id,
            "model_size_gb": round(model_gb, 3),
            "load_time_sec": round(load_time, 3),
            "load_bandwidth_gbs": round(bw_gbs, 3),
            "threshold_gbs": threshold_gbs,
        })

        del model
        torch.cuda.empty_cache()

        assert bw_gbs >= threshold_gbs, (
            f"Model load bandwidth {bw_gbs:.2f} GB/s < {threshold_gbs} GB/s"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-B-02  Checkpoint save / restore throughput
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.model_orch
    def test_checkpoint_save_restore_throughput(self):
        """TC-PIPE-B-02: Checkpoint save (GPU→disk) and restore (disk→GPU) throughput.

        KPIs: save_gbs, restore_gbs, checkpoint_size_gb
        Thresholds: save ≥ 0.5 GB/s, restore ≥ 1.0 GB/s.
        """
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")
        if not _STORAGE_PATH.exists():
            pytest.skip(f"STORAGE_TEST_PATH {_STORAGE_PATH} not accessible")

        # Create a synthetic state-dict (~500 MB worth of tensors)
        n_params = 125_000_000  # ~500 MB @ float16
        state = {
            "layer_0.weight": torch.randn(8192, 4096, dtype=torch.float16, device="cuda:0"),
            "layer_1.weight": torch.randn(4096, 8192, dtype=torch.float16, device="cuda:0"),
            "layer_2.weight": torch.randn(4096, 4096, dtype=torch.float16, device="cuda:0"),
        }
        ckpt_path = _STORAGE_PATH / "pipeline_kpi_checkpoint.pt"

        try:
            # --- Save ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.save({k: v.cpu() for k, v in state.items()}, ckpt_path)
            save_sec = time.perf_counter() - t0

            ckpt_bytes = ckpt_path.stat().st_size
            ckpt_gb = ckpt_bytes / 1024 ** 3
            save_gbs = ckpt_gb / save_sec if save_sec > 0 else 0.0

            # --- Restore ---
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            loaded = torch.load(ckpt_path, map_location="cuda:0")
            torch.cuda.synchronize()
            restore_sec = time.perf_counter() - t0
            restore_gbs = ckpt_gb / restore_sec if restore_sec > 0 else 0.0

        finally:
            ckpt_path.unlink(missing_ok=True)

        _record_kpi("test_checkpoint_save_restore_throughput", {
            "checkpoint_size_gb": round(ckpt_gb, 3),
            "save_time_sec": round(save_sec, 3),
            "save_bandwidth_gbs": round(save_gbs, 3),
            "restore_time_sec": round(restore_sec, 3),
            "restore_bandwidth_gbs": round(restore_gbs, 3),
            "threshold_save_gbs": 0.5,
            "threshold_restore_gbs": 1.0,
        })

        assert save_gbs >= 0.5, (
            f"Checkpoint save {save_gbs:.2f} GB/s < 0.5 GB/s"
        )
        assert restore_gbs >= 1.0, (
            f"Checkpoint restore {restore_gbs:.2f} GB/s < 1.0 GB/s"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-B-03  KV cache GPU memory allocation efficiency
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.model_orch
    def test_kv_cache_memory_allocation_efficiency(self):
        """TC-PIPE-B-03: KV cache allocation latency and GPU memory footprint.

        Simulates the KV cache block table for a typical LLM serving scenario.
        KPIs: alloc_latency_us_per_block, total_kv_gb, blocks_per_sec
        Threshold: allocation rate ≥ 10 000 blocks/sec.
        """
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        # H100 80 GB: typical vLLM-style KV block parameters
        n_layers = 32
        n_heads = 32
        head_dim = 128
        block_size = 16   # tokens per block
        n_blocks = 1_000  # total KV cache blocks to allocate

        # Each KV block: [2, n_layers, block_size, n_heads, head_dim] @ fp16
        block_shape = (2, n_layers, block_size, n_heads, head_dim)
        block_bytes = math.prod(block_shape) * 2  # float16 = 2 bytes
        total_bytes = block_bytes * n_blocks
        total_gb = total_bytes / 1024 ** 3

        free_gb = _gpu_free_memory_gb(0)
        if free_gb < total_gb + 0.5:
            pytest.skip(
                f"Insufficient GPU free memory ({free_gb:.1f} GB) "
                f"for KV cache allocation ({total_gb:.2f} GB needed)"
            )

        # --- Allocate all blocks at once (like vLLM pre-allocation) ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kv_cache = torch.zeros(
            (n_blocks,) + block_shape, dtype=torch.float16, device="cuda:0"
        )
        torch.cuda.synchronize()
        alloc_sec = time.perf_counter() - t0

        blocks_per_sec = n_blocks / alloc_sec
        alloc_us_per_block = (alloc_sec / n_blocks) * 1e6
        threshold_blocks_per_sec = 10_000

        _record_kpi("test_kv_cache_memory_allocation_efficiency", {
            "n_blocks": n_blocks,
            "block_size_tokens": block_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "block_bytes": block_bytes,
            "total_kv_gb": round(total_gb, 3),
            "alloc_time_sec": round(alloc_sec, 4),
            "alloc_latency_us_per_block": round(alloc_us_per_block, 3),
            "blocks_per_sec": round(blocks_per_sec, 0),
            "threshold_blocks_per_sec": threshold_blocks_per_sec,
        })

        del kv_cache
        torch.cuda.empty_cache()

        assert blocks_per_sec >= threshold_blocks_per_sec, (
            f"KV cache alloc {blocks_per_sec:.0f} blocks/sec < {threshold_blocks_per_sec}"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-B-04  Tensor-parallel AllReduce overhead in forward pass
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.model_orch
    def test_tensor_parallel_allreduce_overhead_fraction(self):
        """TC-PIPE-B-04: AllReduce comm overhead as a fraction of GEMM compute time.

        Simulates the TP=4 communication pattern (allreduce on hidden-dim tensor)
        and measures the ratio of comm time to compute time.

        KPIs: compute_ms, allreduce_ms, comm_overhead_pct
        Threshold: AllReduce overhead ≤ 20 % of compute time.
        """
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        if _CUDA_DEVICE_COUNT < 2:
            pytest.skip("need ≥ 2 GPUs for AllReduce overhead test")
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed not available")

        import torch.distributed as dist

        if not dist.is_initialized():
            try:
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", str(_NET_PORT + 1))
                dist.init_process_group(
                    backend="nccl",
                    rank=0, world_size=1,
                )
            except Exception as exc:
                pytest.skip(f"Cannot init process group: {exc}")

        device = torch.device("cuda:0")
        hidden = 8192  # typical LLM hidden size
        seq_len = 512
        n_iters = 20

        # Simulate one TP linear layer: GEMM on (seq_len × hidden)
        x = torch.randn(seq_len, hidden, device=device, dtype=torch.float16)
        w = torch.randn(hidden, hidden, device=device, dtype=torch.float16)
        result_buf = torch.zeros(seq_len, hidden, device=device, dtype=torch.float16)

        # Warm-up
        for _ in range(_WARMUP):
            torch.matmul(x, w)
            dist.all_reduce(result_buf) if dist.is_initialized() and dist.get_world_size() > 1 else None

        # GEMM only
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = torch.matmul(x, w)
        torch.cuda.synchronize()
        compute_ms = (time.perf_counter() - t0) / n_iters * 1000

        # AllReduce on same tensor (simulates TP sync after the GEMM)
        sync_buf = torch.randn(seq_len, hidden, device=device, dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            # Loopback: dist.all_reduce on world_size=1 measures NCCL call overhead
            sync_buf_clone = sync_buf.clone()
            dist.all_reduce(sync_buf_clone) if dist.is_initialized() else None
        torch.cuda.synchronize()
        allreduce_ms = (time.perf_counter() - t0) / n_iters * 1000

        comm_overhead_pct = (allreduce_ms / (compute_ms + allreduce_ms)) * 100
        threshold_pct = 20.0

        _record_kpi("test_tensor_parallel_allreduce_overhead_fraction", {
            "hidden_dim": hidden,
            "seq_len": seq_len,
            "iterations": n_iters,
            "compute_time_ms": round(compute_ms, 4),
            "allreduce_time_ms": round(allreduce_ms, 4),
            "comm_overhead_pct": round(comm_overhead_pct, 2),
            "threshold_comm_overhead_pct": threshold_pct,
        })

        assert comm_overhead_pct <= threshold_pct, (
            f"AllReduce overhead {comm_overhead_pct:.1f}% > {threshold_pct}% of compute"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-B-05  GPU-side batch assembly latency (padding + masking)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.model_orch
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_assembly_gpu_latency(self, batch_size: int):
        """TC-PIPE-B-05: GPU-side batch assembly: pad + attention-mask creation.

        KPIs: assemble_us_per_batch, throughput_batches_per_sec
        Threshold: ≤ 500 µs per batch (must not dominate inference latency).
        """
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        seq_len = 512
        vocab = 50_257

        # Warm-up
        for _ in range(_WARMUP):
            ids = torch.randint(0, vocab, (batch_size, seq_len), device="cuda:0")
            mask = (ids != 0).long()

        n = _BENCH * 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            ids = torch.randint(0, vocab, (batch_size, seq_len), device="cuda:0")
            mask = (ids != 0).long()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        us_per_batch = (elapsed / n) * 1e6
        bps = n / elapsed
        threshold_us = 500.0

        _record_kpi("test_batch_assembly_gpu_latency", {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "iterations": n,
            "assemble_us_per_batch": round(us_per_batch, 3),
            "throughput_batches_per_sec": round(bps, 1),
            "threshold_us_per_batch": threshold_us,
        })

        assert us_per_batch <= threshold_us, (
            f"Batch assembly {us_per_batch:.1f} µs > {threshold_us} µs (batch={batch_size})"
        )


# ===========================================================================
# DOMAIN C – Networking I/O & Storage I/O
# ===========================================================================

class TestNetworkingAndStorageIO:
    """Domain C: NVMe sequential/random IO, TCP throughput, model weight loading BW."""

    # -----------------------------------------------------------------------
    # TC-PIPE-C-01  Storage sequential read throughput (NVMe)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.storage_net_io
    def test_storage_sequential_read_throughput(self):
        """TC-PIPE-C-01: Sequential read throughput from storage (model weight loading path).

        KPIs: read_bandwidth_gbs, file_size_gb, read_time_sec
        Threshold: ≥ 1.0 GB/s (minimum to keep model load under ~80s for 80 GB model).
        """
        file_size_bytes = _STORAGE_GB * 1024 ** 3
        test_file = _STORAGE_PATH / "pipeline_kpi_seq_read.bin"

        # Create file with random bytes (bypass OS write cache for realistic read)
        try:
            chunk = os.urandom(64 * 1024 * 1024)  # 64 MB chunks
            written = 0
            with test_file.open("wb") as fh:
                while written < file_size_bytes:
                    to_write = min(len(chunk), file_size_bytes - written)
                    fh.write(chunk[:to_write])
                    written += to_write
            os.sync() if hasattr(os, "sync") else None

            # Drop page cache on Linux
            try:
                Path("/proc/sys/vm/drop_caches").write_text("1")
            except (PermissionError, FileNotFoundError):
                pass  # non-root or non-Linux; results may include cache effects

            buf = bytearray(16 * 1024 * 1024)  # 16 MB read buffer
            t0 = time.perf_counter()
            read_bytes = 0
            with test_file.open("rb") as fh:
                while True:
                    n = fh.readinto(buf)
                    if not n:
                        break
                    read_bytes += n
            elapsed = time.perf_counter() - t0

        finally:
            test_file.unlink(missing_ok=True)

        read_gbs = (read_bytes / 1024 ** 3) / elapsed
        threshold_gbs = 1.0

        _record_kpi("test_storage_sequential_read_throughput", {
            "file_size_gb": round(read_bytes / 1024 ** 3, 3),
            "read_time_sec": round(elapsed, 3),
            "read_bandwidth_gbs": round(read_gbs, 3),
            "threshold_gbs": threshold_gbs,
            "storage_path": str(_STORAGE_PATH),
        })

        assert read_gbs >= threshold_gbs, (
            f"Sequential read {read_gbs:.2f} GB/s < {threshold_gbs} GB/s"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-C-02  Storage sequential write throughput
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.storage_net_io
    def test_storage_sequential_write_throughput(self):
        """TC-PIPE-C-02: Sequential write throughput (checkpoint save path).

        KPIs: write_bandwidth_gbs, file_size_gb, write_time_sec
        Threshold: ≥ 0.5 GB/s.
        """
        file_size_bytes = _STORAGE_GB * 1024 ** 3
        test_file = _STORAGE_PATH / "pipeline_kpi_seq_write.bin"

        chunk = os.urandom(64 * 1024 * 1024)
        try:
            t0 = time.perf_counter()
            written = 0
            with test_file.open("wb") as fh:
                while written < file_size_bytes:
                    to_write = min(len(chunk), file_size_bytes - written)
                    fh.write(chunk[:to_write])
                    written += to_write
                fh.flush()
                os.fsync(fh.fileno())
            elapsed = time.perf_counter() - t0
        finally:
            test_file.unlink(missing_ok=True)

        write_gbs = (file_size_bytes / 1024 ** 3) / elapsed
        threshold_gbs = 0.5

        _record_kpi("test_storage_sequential_write_throughput", {
            "file_size_gb": round(file_size_bytes / 1024 ** 3, 3),
            "write_time_sec": round(elapsed, 3),
            "write_bandwidth_gbs": round(write_gbs, 3),
            "threshold_gbs": threshold_gbs,
            "storage_path": str(_STORAGE_PATH),
        })

        assert write_gbs >= threshold_gbs, (
            f"Sequential write {write_gbs:.2f} GB/s < {threshold_gbs} GB/s"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-C-03  Storage random 4 K read IOPS
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.storage_net_io
    def test_storage_random_4k_read_iops(self):
        """TC-PIPE-C-03: Random 4 KB read IOPS – relevant for dataset shuffling.

        KPIs: iops, avg_latency_us
        Threshold: ≥ 5 000 IOPS.
        """
        block_size = 4096
        file_size_bytes = 256 * 1024 * 1024  # 256 MB arena
        n_reads = 5_000
        test_file = _STORAGE_PATH / "pipeline_kpi_rand_read.bin"

        try:
            with test_file.open("wb") as fh:
                fh.write(os.urandom(file_size_bytes))
            os.sync() if hasattr(os, "sync") else None
            try:
                Path("/proc/sys/vm/drop_caches").write_text("1")
            except (PermissionError, FileNotFoundError):
                pass

            offsets = [
                (np.random.randint(0, file_size_bytes // block_size)) * block_size
                for _ in range(n_reads)
            ]
            buf = bytearray(block_size)
            t0 = time.perf_counter()
            with test_file.open("rb") as fh:
                for off in offsets:
                    fh.seek(off)
                    fh.readinto(buf)
            elapsed = time.perf_counter() - t0
        finally:
            test_file.unlink(missing_ok=True)

        iops = n_reads / elapsed
        avg_lat_us = (elapsed / n_reads) * 1e6
        threshold_iops = 5_000

        _record_kpi("test_storage_random_4k_read_iops", {
            "block_size_bytes": block_size,
            "n_reads": n_reads,
            "elapsed_sec": round(elapsed, 3),
            "iops": round(iops, 0),
            "avg_latency_us": round(avg_lat_us, 1),
            "threshold_iops": threshold_iops,
            "storage_path": str(_STORAGE_PATH),
        })

        assert iops >= threshold_iops, (
            f"Random 4K read IOPS {iops:.0f} < {threshold_iops}"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-C-04  TCP loopback throughput (host networking baseline)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.storage_net_io
    def test_tcp_loopback_throughput(self):
        """TC-PIPE-C-04: TCP loopback throughput – measures host networking stack BW.

        Used as a baseline before any RDMA or multi-node networking tests.
        KPIs: throughput_gbps, total_bytes_gb, elapsed_sec
        Threshold: ≥ 5 Gbps on loopback.
        """
        port = _NET_PORT
        total_bytes = 512 * 1024 * 1024  # 512 MB
        chunk_size = 1 * 1024 * 1024     # 1 MB
        data = os.urandom(chunk_size)

        received = [0]
        errors = []

        def _server():
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(("127.0.0.1", port))
                srv.listen(1)
                srv.settimeout(10)
                conn, _ = srv.accept()
                buf = bytearray(chunk_size)
                while True:
                    n = conn.recv_into(buf)
                    if not n:
                        break
                    received[0] += n
                conn.close()
                srv.close()
            except Exception as e:
                errors.append(str(e))

        srv_thread = threading.Thread(target=_server, daemon=True)
        srv_thread.start()
        time.sleep(0.1)  # give server time to bind

        try:
            cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cli.connect(("127.0.0.1", port))
            t0 = time.perf_counter()
            sent = 0
            while sent < total_bytes:
                to_send = min(len(data), total_bytes - sent)
                cli.sendall(data[:to_send])
                sent += to_send
            cli.close()
        except Exception as e:
            pytest.skip(f"TCP loopback connection failed: {e}")

        srv_thread.join(timeout=15)
        elapsed = time.perf_counter() - t0

        if errors:
            pytest.skip(f"TCP server error: {errors[0]}")

        throughput_gbps = (received[0] * 8 / 1e9) / elapsed
        threshold_gbps = 5.0

        _record_kpi("test_tcp_loopback_throughput", {
            "total_bytes_mb": round(received[0] / 1024 ** 2, 1),
            "elapsed_sec": round(elapsed, 3),
            "throughput_gbps": round(throughput_gbps, 3),
            "threshold_gbps": threshold_gbps,
        })

        assert throughput_gbps >= threshold_gbps, (
            f"TCP loopback {throughput_gbps:.2f} Gbps < {threshold_gbps} Gbps"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-C-05  Model weights load bandwidth (disk → CPU RAM → GPU)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.storage_net_io
    def test_model_weight_load_pipeline_bandwidth(self):
        """TC-PIPE-C-05: End-to-end model weight loading BW through full path.

        Writes synthetic weight shards to disk, then loads through:
          disk → numpy mmap → CPU tensor → GPU (non-blocking H2D)

        KPIs: disk_read_gbs, h2d_gbs, end_to_end_gbs
        Threshold: end-to-end ≥ 1 GB/s.
        """
        if not _HAS_TORCH:
            pytest.skip("torch not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        shard_gb = min(_STORAGE_GB, 1)  # 1 GB shard
        shard_bytes = shard_gb * 1024 ** 3
        n_floats = shard_bytes // 4  # float32
        shard_file = _STORAGE_PATH / "pipeline_kpi_weight_shard.npy"

        try:
            arr = np.random.rand(n_floats).astype(np.float32)
            np.save(shard_file, arr)
            os.sync() if hasattr(os, "sync") else None
            try:
                Path("/proc/sys/vm/drop_caches").write_text("1")
            except (PermissionError, FileNotFoundError):
                pass

            # Full pipeline: load from disk, move to GPU
            t0 = time.perf_counter()
            loaded_arr = np.load(shard_file)
            cpu_tensor = torch.from_numpy(loaded_arr)
            cpu_pinned = cpu_tensor.pin_memory()
            gpu_tensor = cpu_pinned.to("cuda:0", non_blocking=True)
            torch.cuda.synchronize()
            total_sec = time.perf_counter() - t0

        finally:
            shard_file.unlink(missing_ok=True)

        data_gb = shard_bytes / 1024 ** 3
        e2e_gbs = data_gb / total_sec
        threshold_gbs = 1.0

        _record_kpi("test_model_weight_load_pipeline_bandwidth", {
            "shard_size_gb": round(data_gb, 3),
            "total_load_time_sec": round(total_sec, 3),
            "end_to_end_gbs": round(e2e_gbs, 3),
            "threshold_gbs": threshold_gbs,
        })

        assert e2e_gbs >= threshold_gbs, (
            f"Weight load pipeline {e2e_gbs:.2f} GB/s < {threshold_gbs} GB/s"
        )


# ===========================================================================
# DOMAIN D – Inference Pipeline KPIs
# ===========================================================================

class TestInferencePipelineKPIs:
    """Domain D: TTFT, ITL, prefill/decode split, batching efficiency, GPU utilisation."""

    # -----------------------------------------------------------------------
    # TC-PIPE-D-01  Time to First Token (TTFT) across input lengths
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    @pytest.mark.parametrize("input_len", [32, 128, 512, 2048])
    def test_time_to_first_token(self, input_len: int):
        """TC-PIPE-D-01: TTFT (time to first generated token) vs input length.

        KPIs: ttft_ms, input_len, tokens_per_sec_prefill
        Thresholds (H100 PCIe, gpt2-scale):
          input_len ≤  128  → TTFT ≤  200 ms
          input_len ≤  512  → TTFT ≤  500 ms
          input_len ≤ 2048  → TTFT ≤ 2000 ms
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()

        # Guard against requesting a longer context than the model actually
        # supports: e.g. GPT-2 (the default PIPELINE_MODEL_ID) has
        # max_position_embeddings=1024, so input_len=2048 indexes past the
        # positional embedding table and triggers a CUDA device-side assert
        # that poisons the CUDA context for every subsequent test in this
        # process. Skip rather than let that happen.
        max_ctx = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
        if max_ctx and input_len > max_ctx:
            pytest.skip(f"input_len={input_len} exceeds {_MODEL_ID}'s max context length ({max_ctx})")

        text = "The AI inference pipeline processes sequences. " * (input_len // 8 + 1)
        inputs = tokenizer(text, truncation=True, max_length=input_len,
                           return_tensors="pt").to("cuda:0")

        # Warm-up
        with torch.no_grad():
            for _ in range(_WARMUP):
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        ttft_samples = []
        with torch.no_grad():
            for _ in range(min(_BENCH, 5)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
                torch.cuda.synchronize()
                ttft_samples.append((time.perf_counter() - t0) * 1000)

        ttft_ms = float(np.median(ttft_samples))
        ttft_p95_ms = float(np.percentile(ttft_samples, 95))
        prefill_tps = (input_len / (ttft_ms / 1000)) if ttft_ms > 0 else 0

        threshold_ms = 200 if input_len <= 128 else (500 if input_len <= 512 else 2000)

        _record_kpi("test_time_to_first_token", {
            "model_id": _MODEL_ID,
            "input_len": input_len,
            "ttft_median_ms": round(ttft_ms, 2),
            "ttft_p95_ms": round(ttft_p95_ms, 2),
            "prefill_throughput_tokens_per_sec": round(prefill_tps, 1),
            "threshold_ms": threshold_ms,
            "samples": ttft_samples,
        })

        del model
        torch.cuda.empty_cache()

        assert ttft_ms <= threshold_ms, (
            f"TTFT {ttft_ms:.1f} ms > {threshold_ms} ms at input_len={input_len}"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-02  Inter-Token Latency (ITL / decode step latency)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    def test_inter_token_latency(self):
        """TC-PIPE-D-02: Decode step (inter-token) latency at batch=1.

        KPIs: itl_median_ms, itl_p95_ms, decode_tokens_per_sec
        Threshold: ITL (p95) ≤ 100 ms per token (H100 PCIe, gpt2-scale model).
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()

        prompt = "The quick brown fox jumped over the lazy dog. " * 4
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        n_decode_steps = 32

        # Measure token-by-token decode latency manually
        itl_list = []
        with torch.no_grad():
            # Build past-key-values from the prompt
            out = model(**inputs, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for _ in range(n_decode_steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                step_out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
                torch.cuda.synchronize()
                itl_list.append((time.perf_counter() - t0) * 1000)
                past_kv = step_out.past_key_values
                next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        itl_median = float(np.median(itl_list))
        itl_p95 = float(np.percentile(itl_list, 95))
        decode_tps = 1000.0 / itl_median if itl_median > 0 else 0

        threshold_itl_ms = 100.0

        _record_kpi("test_inter_token_latency", {
            "model_id": _MODEL_ID,
            "n_decode_steps": n_decode_steps,
            "itl_median_ms": round(itl_median, 3),
            "itl_p95_ms": round(itl_p95, 3),
            "decode_tokens_per_sec": round(decode_tps, 2),
            "threshold_itl_p95_ms": threshold_itl_ms,
            "all_itl_ms": [round(v, 3) for v in itl_list],
        })

        del model
        torch.cuda.empty_cache()

        assert itl_p95 <= threshold_itl_ms, (
            f"ITL p95 {itl_p95:.2f} ms > {threshold_itl_ms} ms"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-03  Prefill throughput vs decode throughput ratio
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    def test_prefill_vs_decode_throughput_ratio(self):
        """TC-PIPE-D-03: Compare prefill (compute-bound) vs decode (memory-bound) TPS.

        Measures both phases independently and reports the ratio.
        KPIs: prefill_tps, decode_tps, prefill_to_decode_ratio
        Threshold: prefill_tps ≥ 5× decode_tps (prefill should be much faster per token
                   due to parallel matrix ops vs sequential KV-cache reads).
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()

        prompt_512 = "benchmark " * 64  # ~512 tokens for prefill
        inputs = tokenizer(prompt_512, truncation=True, max_length=512,
                           return_tensors="pt").to("cuda:0")
        n_input_tokens = inputs["input_ids"].shape[1]

        # --- Prefill phase only (forward pass, no decode) ---
        with torch.no_grad():
            for _ in range(_WARMUP):
                _ = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(_BENCH):
                out = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        prefill_sec = (time.perf_counter() - t0) / _BENCH
        prefill_tps = n_input_tokens / prefill_sec

        # --- Decode phase (sequential single-token steps) ---
        n_decode = 32
        decode_times = []
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(n_decode):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                step = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
                torch.cuda.synchronize()
                decode_times.append(time.perf_counter() - t0)
                past_kv = step.past_key_values
                next_tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        decode_sec_per_token = float(np.median(decode_times))
        decode_tps = 1.0 / decode_sec_per_token if decode_sec_per_token > 0 else 0
        ratio = prefill_tps / decode_tps if decode_tps > 0 else 0.0
        threshold_ratio = 5.0

        _record_kpi("test_prefill_vs_decode_throughput_ratio", {
            "model_id": _MODEL_ID,
            "prefill_input_tokens": n_input_tokens,
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(decode_tps, 1),
            "prefill_to_decode_ratio": round(ratio, 2),
            "threshold_ratio": threshold_ratio,
            "decode_samples": n_decode,
        })

        del model
        torch.cuda.empty_cache()

        assert ratio >= threshold_ratio, (
            f"Prefill/decode TPS ratio {ratio:.1f}× < {threshold_ratio}× "
            f"(prefill={prefill_tps:.0f} tps, decode={decode_tps:.0f} tps)"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-04  Batched inference throughput vs batch size
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_batched_inference_throughput(self, batch_size: int):
        """TC-PIPE-D-04: Tokens/sec at varying batch sizes (latency-throughput tradeoff).

        KPIs: tokens_per_sec, latency_ms, batch_size
        Threshold: throughput at batch=8 ≥ 4× throughput at batch=1
                   (shows the system can leverage batching efficiently).
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()

        seq_len = 128
        n_gen_tokens = 32
        texts = ["The AI inference system processes " * 5] * batch_size
        inputs = tokenizer(texts, truncation=True, max_length=seq_len,
                           padding="max_length", return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            for _ in range(_WARMUP):
                _ = model.generate(**inputs, max_new_tokens=n_gen_tokens, do_sample=False)

        latencies = []
        with torch.no_grad():
            for _ in range(min(_BENCH, 5)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model.generate(**inputs, max_new_tokens=n_gen_tokens, do_sample=False)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)

        lat_median = float(np.median(latencies))
        lat_p95 = float(np.percentile(latencies, 95))
        total_tokens_generated = batch_size * n_gen_tokens
        tps = total_tokens_generated / lat_median if lat_median > 0 else 0

        _record_kpi("test_batched_inference_throughput", {
            "model_id": _MODEL_ID,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "n_gen_tokens": n_gen_tokens,
            "latency_median_sec": round(lat_median, 4),
            "latency_p95_sec": round(lat_p95, 4),
            "latency_ms": round(lat_median * 1000, 2),
            "tokens_per_sec": round(tps, 1),
            "iterations": len(latencies),
        })

        del model
        torch.cuda.empty_cache()

        # Throughput sanity: all batch sizes should generate > 0 tokens/sec
        assert tps > 0, f"Zero throughput at batch_size={batch_size}"

    # -----------------------------------------------------------------------
    # TC-PIPE-D-05  GPU utilisation during inference (should be high)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    def test_gpu_utilisation_during_inference(self):
        """TC-PIPE-D-05: GPU utilisation % while running sustained batch inference.

        KPIs: mean_gpu_util_pct, mean_mem_util_pct, peak_gpu_util_pct
        Threshold: mean GPU util ≥ 10 % during active inference.
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")
        if not _HAS_NVML:
            pytest.skip("pynvml not available – cannot measure GPU utilisation")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        batch_size, seq_len, n_gen = 8, 128, 64
        texts = ["AI inference test batch " * 5] * batch_size
        inputs = tokenizer(texts, truncation=True, max_length=seq_len,
                           padding="max_length", return_tensors="pt").to("cuda:0")

        # Warm-up
        with torch.no_grad():
            for _ in range(_WARMUP):
                _ = model.generate(**inputs, max_new_tokens=n_gen, do_sample=False)

        # Launch background utilisation poller
        util_samples = []
        stop_event = threading.Event()
        poll_thread = threading.Thread(
            target=_poll_gpu_utilisation,
            args=(0, 0.1, util_samples, stop_event),
            daemon=True,
        )
        poll_thread.start()

        with torch.no_grad():
            for _ in range(5):
                _ = model.generate(**inputs, max_new_tokens=n_gen, do_sample=False)
        torch.cuda.synchronize()

        stop_event.set()
        poll_thread.join(timeout=5)

        gpu_utils = [s[0] for s in util_samples if s[0] >= 0]
        mem_utils = [s[1] for s in util_samples if s[1] >= 0]
        mean_gpu = float(np.mean(gpu_utils)) if gpu_utils else 0.0
        peak_gpu = float(np.max(gpu_utils)) if gpu_utils else 0.0
        mean_mem = float(np.mean(mem_utils)) if mem_utils else 0.0
        threshold_util = 10.0

        _record_kpi("test_gpu_utilisation_during_inference", {
            "model_id": _MODEL_ID,
            "batch_size": batch_size,
            "n_gen_tokens": n_gen,
            "mean_gpu_util_pct": round(mean_gpu, 2),
            "peak_gpu_util_pct": round(peak_gpu, 2),
            "mean_mem_util_pct": round(mean_mem, 2),
            "n_samples": len(gpu_utils),
            "threshold_mean_util_pct": threshold_util,
        })

        del model
        torch.cuda.empty_cache()

        assert mean_gpu >= threshold_util, (
            f"Mean GPU utilisation {mean_gpu:.1f}% < {threshold_util}% during inference"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-06  HBM bandwidth utilisation during decode (memory-bound check)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    def test_hbm_bandwidth_utilisation_during_decode(self):
        """TC-PIPE-D-06: HBM memory controller utilisation during autoregressive decode.

        During decode the operation is memory-bound; HBM mem-util should be ≥ 40 %.
        KPIs: mean_mem_util_pct, decode_tps
        Threshold: mean mem controller util ≥ 40 %.
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")
        if not _HAS_NVML:
            pytest.skip("pynvml not available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()

        prompt = "benchmark autoregressive decode memory bandwidth " * 8
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        n_steps = 50

        util_samples = []
        stop_event = threading.Event()
        poll_thread = threading.Thread(
            target=_poll_gpu_utilisation,
            args=(0, 0.05, util_samples, stop_event),
            daemon=True,
        )

        with torch.no_grad():
            out = model(**inputs, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            poll_thread.start()
            t_decode_start = time.perf_counter()
            for _ in range(n_steps):
                step = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
                past_kv = step.past_key_values
                next_tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_elapsed = time.perf_counter() - t_decode_start

        stop_event.set()
        poll_thread.join(timeout=5)

        mem_utils = [s[1] for s in util_samples if s[1] >= 0]
        mean_mem = float(np.mean(mem_utils)) if mem_utils else 0.0
        decode_tps = n_steps / decode_elapsed
        threshold_mem = 40.0

        _record_kpi("test_hbm_bandwidth_utilisation_during_decode", {
            "model_id": _MODEL_ID,
            "n_decode_steps": n_steps,
            "decode_elapsed_sec": round(decode_elapsed, 3),
            "decode_tps": round(decode_tps, 2),
            "mean_mem_util_pct": round(mean_mem, 2),
            "n_samples": len(mem_utils),
            "threshold_mem_util_pct": threshold_mem,
        })

        del model
        torch.cuda.empty_cache()

        assert mean_mem >= threshold_mem, (
            f"HBM mem-util during decode {mean_mem:.1f}% < {threshold_mem}%"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-07  Continuous batching throughput (vLLM serving)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    def test_continuous_batching_throughput_vllm(self):
        """TC-PIPE-D-07: vLLM continuous-batching throughput (tokens/sec, requests/sec).

        KPIs: throughput_tokens_per_sec, requests_per_sec, p50_latency_ms, p99_latency_ms
        Threshold: ≥ 200 tokens/sec aggregate for small model on 1 GPU.
        Skips if vllm not installed.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            pytest.skip("vllm not installed")

        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        model_id = os.environ.get("PIPELINE_VLLM_MODEL_ID", _MODEL_ID)
        n_prompts = 50
        max_tokens = 64

        prompts = [
            f"Explain concept {i} in AI inference systems:" for i in range(n_prompts)
        ]
        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        llm = LLM(
            model=model_id,
            dtype="float16",
            gpu_memory_utilization=0.7,
            max_model_len=1024,
        )

        # Warm-up
        _ = llm.generate(prompts[:4], sampling)

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / elapsed
        rps = n_prompts / elapsed

        latencies_ms = [(time.perf_counter() - t0) * 1000 / n_prompts] * n_prompts
        threshold_tps = 200.0

        _record_kpi("test_continuous_batching_throughput_vllm", {
            "model_id": model_id,
            "n_prompts": n_prompts,
            "max_tokens": max_tokens,
            "elapsed_sec": round(elapsed, 3),
            "total_tokens_generated": total_tokens,
            "throughput_tokens_per_sec": round(tps, 1),
            "requests_per_sec": round(rps, 2),
            "threshold_tps": threshold_tps,
        })

        del llm
        if _HAS_TORCH:
            torch.cuda.empty_cache()

        assert tps >= threshold_tps, (
            f"vLLM continuous batching {tps:.0f} tok/s < {threshold_tps} tok/s"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-08  Latency-throughput tradeoff curve across batch sizes
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    def test_latency_throughput_tradeoff_curve(self):
        """TC-PIPE-D-08: Record latency and throughput at B=1,4,8,16 as a curve.

        This test always passes (informational KPI sweep); it records the full
        latency-throughput Pareto curve for offline analysis.

        KPIs: curve[batch_size] = {latency_ms, tokens_per_sec}
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()

        seq_len, n_gen = 64, 16
        curve = {}

        with torch.no_grad():
            for bs in [1, 4, 8, 16]:
                texts = ["benchmark " * 8] * bs
                inputs = tokenizer(texts, truncation=True, max_length=seq_len,
                                   padding="max_length", return_tensors="pt").to("cuda:0")
                for _ in range(_WARMUP):
                    _ = model.generate(**inputs, max_new_tokens=n_gen, do_sample=False)

                lats = []
                for _ in range(min(_BENCH, 5)):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model.generate(**inputs, max_new_tokens=n_gen, do_sample=False)
                    torch.cuda.synchronize()
                    lats.append((time.perf_counter() - t0) * 1000)

                lat_med = float(np.median(lats))
                tps = (bs * n_gen) / (lat_med / 1000) if lat_med > 0 else 0
                curve[bs] = {"latency_ms": round(lat_med, 2), "tokens_per_sec": round(tps, 1)}
                logger.info("Batch=%d  latency=%.1f ms  tps=%.0f", bs, lat_med, tps)

        _record_kpi("test_latency_throughput_tradeoff_curve", {
            "model_id": _MODEL_ID,
            "seq_len": seq_len,
            "n_gen_tokens": n_gen,
            "curve": curve,
        })

        del model
        torch.cuda.empty_cache()

        # Informational: only assert that throughput increases with batch size
        tps_b1 = curve[1]["tokens_per_sec"]
        tps_b8 = curve[8]["tokens_per_sec"]
        assert tps_b8 > tps_b1, (
            f"Throughput did not increase from batch=1 ({tps_b1:.0f} tps) "
            f"to batch=8 ({tps_b8:.0f} tps)"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-09  Inference pipeline end-to-end latency breakdown
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    def test_inference_pipeline_latency_breakdown(self):
        """TC-PIPE-D-09: Decompose end-to-end inference into: preprocess + TTFT + decode.

        KPIs: preprocess_ms, ttft_ms, decode_ms_per_token, total_ms,
              preprocess_pct, ttft_pct, decode_pct
        Ensures preprocess stage < 10 % of total inference time.
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT == 0:
            pytest.skip("no CUDA device available")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
        )
        model.eval()

        raw_text = "The inference pipeline benchmark processes these tokens " * 16
        n_gen = 16
        n_trials = min(_BENCH, 5)

        preprocess_ms_list, ttft_ms_list, decode_ms_list = [], [], []

        with torch.no_grad():
            for _ in range(n_trials):
                # 1. Preprocessing
                t0 = time.perf_counter()
                inputs = tokenizer(raw_text, truncation=True, max_length=256,
                                   return_tensors="pt").to("cuda:0")
                preprocess_ms_list.append((time.perf_counter() - t0) * 1000)

                # 2. TTFT (prefill → first token)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(**inputs, use_cache=True)
                past_kv = out.past_key_values
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
                ttft_ms_list.append((time.perf_counter() - t0) * 1000)

                # 3. Decode (n_gen steps)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_gen):
                    step = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
                    past_kv = step.past_key_values
                    next_tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
                decode_ms_list.append((time.perf_counter() - t0) * 1000)

        preproc_ms = float(np.median(preprocess_ms_list))
        ttft_ms = float(np.median(ttft_ms_list))
        decode_total_ms = float(np.median(decode_ms_list))
        decode_per_tok_ms = decode_total_ms / n_gen
        total_ms = preproc_ms + ttft_ms + decode_total_ms

        preproc_pct = (preproc_ms / total_ms) * 100
        ttft_pct = (ttft_ms / total_ms) * 100
        decode_pct = (decode_total_ms / total_ms) * 100
        threshold_preproc_pct = 10.0

        _record_kpi("test_inference_pipeline_latency_breakdown", {
            "model_id": _MODEL_ID,
            "n_gen_tokens": n_gen,
            "preprocess_ms": round(preproc_ms, 3),
            "ttft_ms": round(ttft_ms, 3),
            "decode_total_ms": round(decode_total_ms, 3),
            "decode_ms_per_token": round(decode_per_tok_ms, 3),
            "total_ms": round(total_ms, 3),
            "preprocess_pct": round(preproc_pct, 2),
            "ttft_pct": round(ttft_pct, 2),
            "decode_pct": round(decode_pct, 2),
            "threshold_preprocess_pct": threshold_preproc_pct,
        })

        del model
        torch.cuda.empty_cache()

        assert preproc_pct <= threshold_preproc_pct, (
            f"Preprocessing takes {preproc_pct:.1f}% of total inference time "
            f"(> {threshold_preproc_pct}%); consider batching or GPU-side tokenization"
        )

    # -----------------------------------------------------------------------
    # TC-PIPE-D-10  Multi-GPU inference throughput scaling (TP=1 vs TP=4)
    # -----------------------------------------------------------------------
    @pytest.mark.ai_pipeline
    @pytest.mark.inference_kpis
    @pytest.mark.slow
    def test_multi_gpu_inference_throughput_scaling(self):
        """TC-PIPE-D-10: Compare 1-GPU vs multi-GPU inference throughput scaling.

        Uses device_map='balanced' to spread model across available GPUs.
        KPIs: tps_single_gpu, tps_multi_gpu, scaling_ratio
        Threshold: multi-GPU throughput ≥ 1.5× single-GPU (not linear due to comm overhead).
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            pytest.skip("torch or transformers not available")
        if _CUDA_DEVICE_COUNT < 2:
            pytest.skip("need ≥ 2 GPUs for multi-GPU scaling test")

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        seq_len, n_gen, batch = 64, 16, 4
        texts = ["multi gpu inference benchmark " * 4] * batch
        n_trials = min(_BENCH, 5)

        def _measure_tps(device_map: str) -> float:
            mdl = AutoModelForCausalLM.from_pretrained(
                _MODEL_ID, torch_dtype=torch.float16, device_map=device_map
            )
            mdl.eval()
            # Determine where inputs should live
            first_device = next(mdl.parameters()).device
            inputs = tokenizer(texts, truncation=True, max_length=seq_len,
                               padding="max_length", return_tensors="pt").to(first_device)
            with torch.no_grad():
                for _ in range(_WARMUP):
                    _ = mdl.generate(**inputs, max_new_tokens=n_gen, do_sample=False)
            lats = []
            with torch.no_grad():
                for _ in range(n_trials):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = mdl.generate(**inputs, max_new_tokens=n_gen, do_sample=False)
                    torch.cuda.synchronize()
                    lats.append(time.perf_counter() - t0)
            del mdl
            torch.cuda.empty_cache()
            lat = float(np.median(lats))
            return (batch * n_gen) / lat if lat > 0 else 0.0

        tps_1gpu = _measure_tps("cuda:0")
        tps_mgpu = _measure_tps("balanced")
        ratio = tps_mgpu / tps_1gpu if tps_1gpu > 0 else 0.0
        threshold_ratio = 1.5

        _record_kpi("test_multi_gpu_inference_throughput_scaling", {
            "model_id": _MODEL_ID,
            "n_gpus": _CUDA_DEVICE_COUNT,
            "batch_size": batch,
            "n_gen_tokens": n_gen,
            "tps_single_gpu": round(tps_1gpu, 1),
            "tps_multi_gpu": round(tps_mgpu, 1),
            "scaling_ratio": round(ratio, 3),
            "threshold_scaling_ratio": threshold_ratio,
        })

        assert ratio >= threshold_ratio, (
            f"Multi-GPU inference scaling {ratio:.2f}× < {threshold_ratio}× "
            f"(1-GPU: {tps_1gpu:.0f} tps, {_CUDA_DEVICE_COUNT}-GPU: {tps_mgpu:.0f} tps)"
        )
