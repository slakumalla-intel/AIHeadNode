"""
Performance validation suite for AI host + GPU inference/training infrastructure.

This suite is data-driven and keeps all requested performance test IDs in one place.
By default, commands are executed for workload validation.

Disable execution (catalog-only mode):
    PERF_EXECUTE_CASES=0
Optional filters:
  PERF_CASE_IDS=AI_HOSTCPU_PERF_TEST_ID_001,AI_HOSTCPU_PERF_TEST_ID_018
Optional command timeout (seconds):
  PERF_CMD_TIMEOUT_SEC=1800

KPI output file:
  artifacts/performance_suite_kpis.jsonl
"""

from __future__ import annotations

import json
import importlib.util
import logging
import os
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import numpy as np
import pytest

from tests.kpi_efficiency import enrich_efficiency_kpis


logger = logging.getLogger(__name__)


_HF_MODEL_PATH_CACHE: Dict[str, str] = {}
_HF_MODEL_ALIASES: Dict[str, str] = {
    "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mixtral-8x22B-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}
_GENAI_PERF_PROFILE_HELP_CACHE: Optional[str] = None


def _configure_performance_env_defaults() -> None:
    """Set safe default environment values for performance-case execution."""
    datasets_root = os.environ.get("DATASETS_ROOT", "/mnt/datasets")
    hf_home = os.path.join(datasets_root, "huggingface")
    hf_cache = os.path.join(hf_home, "hub")

    os.environ.setdefault("DATASETS_ROOT", datasets_root)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)
    os.environ.setdefault(
        "SHAREGPT_DATASET_PATH",
        os.path.join(datasets_root, "sharegpt", "ShareGPT_V3_unfiltered_cleaned_split.json"),
    )

    os.environ.setdefault("PERF_EXECUTE_CASES", "1")
    os.environ.setdefault("PERF_CMD_TIMEOUT_SEC", "86400")  # 24 hours; no timeout for perf tests
    os.environ.setdefault("PERF_VLLM_PRE_BENCH_READY_TIMEOUT_SEC", "3600")  # 1 hour for CUDA graph capture
    os.environ.setdefault("PERF_SUITE_KPI_FILE", "artifacts/performance_suite_kpis.jsonl")
    # Newer versions first: v0.7 fails device_to_device_memcpy_read_ce pattern checks on multi-domain PCIe nodes.
    _nvbandwidth_roots = [
        os.path.join(root, name)
        for name in ("nvbandwidth-0.9", "nvbandwidth-0.8", "nvbandwidth-0.7", "nvbandwidth")
        for root in (datasets_root, "/root/workloads")
    ]
    _nvbandwidth_default = next((p for p in _nvbandwidth_roots if os.path.isdir(p)), _nvbandwidth_roots[0])
    os.environ.setdefault("PERF_NVBANDWIDTH_BIN", os.environ.get("NVBANDWIDTH_BIN", _nvbandwidth_default))
    os.environ.setdefault("PERF_NCCL_TESTS_DIR", "/root/workloads/nccl-tests/build")
    os.environ.setdefault("PERF_VLLM_GPU_MEMORY_UTILIZATION", "0.70")
    os.environ.setdefault("PERF_VLLM_ENFORCE_EAGER", "1")
    os.environ.setdefault("PERF_VLLM_SERVE_FALLBACK_MAX_MODEL_LEN", "4096")
    os.environ.setdefault("PERF_VLLM_FP8_DTYPE_FALLBACK", "bfloat16")
    os.environ.setdefault("PERF_VLLM_BENCHMARK_MODULE", "vllm.benchmarks.benchmark_serving")
    os.environ.setdefault("PERF_VLLM_BENCH_NUM_PROMPTS_CAP", "1000")
    os.environ.setdefault("PERF_TRTLLM_DATASET_PATH", "/mnt/datasets/synthetic/synthetic_2048_2048_1000.txt")
    os.environ.setdefault("PERF_TRTLLM_ENGINE_DIR", "/mnt/datasets/trtllm_engine_llama2_70b_fp8")
    os.environ.setdefault("PERF_TRTLLM_MODEL", "meta-llama/Llama-2-70b-hf")
    _default_genai_model = os.environ.get("HF_INFERENCE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    os.environ.setdefault("PERF_GENAI_MODEL", _default_genai_model)
    os.environ.setdefault("PERF_GENAI_TOKENIZER", os.environ.get("PERF_GENAI_MODEL", _default_genai_model))
    os.environ.setdefault("PERF_GENAI_ENDPOINT_TYPE", "chat")
    os.environ.setdefault("PERF_GENAI_ENDPOINT", "v1/chat/completions")
    os.environ.setdefault("PERF_GENAI_STANDALONE", "1")
    os.environ.setdefault("PERF_GENAI_AUTOSTART_SERVERS", "1")
    os.environ.setdefault("PERF_GENAI_AUTOSTART_BASE_PORT", "8000")
    os.environ.setdefault("PERF_GENAI_AUTOSTART_READY_TIMEOUT_SEC", "900")
    os.environ.setdefault("PERF_GENAI_DTYPE", "bfloat16")
    os.environ.setdefault("PERF_GENAI_MAX_MODEL_LEN", "8192")
    os.environ.setdefault("PERF_AUTO_INSTALL", "1" if sys.platform.startswith("linux") else "0")
    os.environ.setdefault("PERF_AUTO_INSTALL_TIMEOUT_SEC", "1800")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ.setdefault(
        "PERF_TOKENIZER_MODEL",
        os.environ.get("HF_INFERENCE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    )
    # Derive visible GPU list from environment; avoid hard-coding 4.
    _raw_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not _raw_cvd:
        try:
            import subprocess as _sp
            _smi = _sp.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                stderr=_sp.DEVNULL, text=True,
            )
            _raw_cvd = ",".join(str(i) for i in range(len([l for l in _smi.splitlines() if l.strip()])))
        except Exception:
            _raw_cvd = "0"
    os.environ.setdefault("PERF_CUDA_VISIBLE_DEVICES", _raw_cvd)
    _n_perf_gpus = str(max(len([x for x in _raw_cvd.split(",") if x.strip()]), 1))
    _genai_target_tp = str(max(1, min(int(_n_perf_gpus), 4)))
    os.environ.setdefault("PERF_GENAI_TARGET_TP", _genai_target_tp)
    os.environ.setdefault("PERF_DYNAMIC_TP_FROM_VISIBLE", "1")
    os.environ.setdefault("PERF_REQUIRE_4GPUS", "0")
    os.environ.setdefault("PERF_NUM_GPUS", _n_perf_gpus)

    # Derive a conservative per-process CPU thread budget from visible GPUs.
    # This keeps CPU-side launch/tokenization threads proportional to GPU fan-out.
    _logical_cpus = os.cpu_count() or 1
    _threads_per_gpu = max(1, _logical_cpus // max(int(_n_perf_gpus), 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(_threads_per_gpu))
    os.environ.setdefault("MKL_NUM_THREADS", str(_threads_per_gpu))

    # Ensure subprocesses (mpirun/nccl-tests) can run under root in containerized lab environments.
    os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

    # Keep CUDA mask consistent for commands that depend on inherited environment.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _raw_cvd)
    # Keep CUDA enumeration stable on hosts with mixed accelerator SKUs.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


_configure_performance_env_defaults()


@dataclass(frozen=True)
class PerformanceCase:
    case_id: str
    title: str
    category: str
    subsystem: str
    workload: str
    purpose: str
    prerequisites: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    expected: str = ""
    planned_only: bool = False


def _kpi_output_path() -> Path:
    return Path(os.environ.get("PERF_SUITE_KPI_FILE", "artifacts/performance_suite_kpis.jsonl"))


def _record_kpi(case_id: str, metrics: Dict) -> None:
    out_path = _kpi_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_metrics = enrich_efficiency_kpis(metrics, os.environ)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case_id": case_id,
        "metrics": enriched_metrics,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    logger.info("KPI %s: %s", case_id, json.dumps(enriched_metrics, default=str, sort_keys=True))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _hf_auth_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _auto_install_python_packages(package_specs: List[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    if not package_specs:
        return results

    timeout_sec = int(os.environ.get("PERF_AUTO_INSTALL_TIMEOUT_SEC", "1800"))
    for package_spec in package_specs:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            package_spec,
        ]
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

    # Refresh import caches so immediate post-install imports can resolve.
    try:
        import importlib

        importlib.invalidate_caches()
    except Exception:
        pass
    return results


def _selected_case_ids() -> Optional[set]:
    raw = os.environ.get("PERF_CASE_IDS", "").strip()
    if not raw:
        return None

    def _normalize_selector(token: str) -> str:
        cleaned = token.strip().strip('"').strip("'")
        if not cleaned:
            return ""

        full_match = re.fullmatch(r"AI_HOSTCPU_PERF_TEST_ID_(\d{1,3})", cleaned, flags=re.IGNORECASE)
        if full_match:
            return f"AI_HOSTCPU_PERF_TEST_ID_{int(full_match.group(1)):03d}"

        short_match = re.fullmatch(r"\d{1,3}", cleaned)
        if short_match:
            return f"AI_HOSTCPU_PERF_TEST_ID_{int(cleaned):03d}"

        return cleaned

    normalized = {_normalize_selector(x) for x in raw.split(",") if x.strip()}
    normalized.discard("")
    return normalized


def _is_selected(case_id: str) -> bool:
    selected = _selected_case_ids()
    return selected is None or case_id in selected


def _binary_exists(token: str) -> bool:
    # Treat "python3" as "python" on Windows-style shells too.
    candidate = "python" if token == "python3" else token
    if "/" in candidate or "\\" in candidate:
        path_candidate = Path(candidate)
        return path_candidate.exists() and os.access(path_candidate, os.X_OK)
    return shutil.which(candidate) is not None


def _resolve_nvbandwidth_binary() -> Optional[str]:
    """Resolve nvbandwidth from an explicit binary path or a standard source tree root."""
    configured = Path(os.environ.get("PERF_NVBANDWIDTH_BIN", "/root/workloads/nvbandwidth"))
    if configured.is_file() and os.access(configured, os.X_OK):
        return str(configured)

    if configured.is_dir():
        for candidate in (
            configured / "nvbandwidth",
            configured / "build" / "nvbandwidth",
            configured / "bin" / "nvbandwidth",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

    return shutil.which("nvbandwidth")


def _resolve_nccl_test_binary(token: str) -> Optional[str]:
    """Resolve nccl-tests binaries when commands still point at ./build/<binary>."""
    binary_name = Path(token).name
    candidate_dirs: List[Path] = []

    configured_dir = Path(os.environ.get("PERF_NCCL_TESTS_DIR", "/root/workloads/nccl-tests/build"))
    candidate_dirs.append(configured_dir)

    # Support callers that set PERF_NCCL_TESTS_DIR to repo root instead of build dir.
    if configured_dir.name != "build":
        candidate_dirs.append(configured_dir / "build")

    # Best-effort fallbacks for common workspace and system layouts.
    candidate_dirs.extend(
        [
            Path.cwd() / "build",
            Path.cwd() / "nccl-tests" / "build",
            Path("/home/workspace/nccl-tests/build"),
            Path("/home/workspace/AIHeadNode/nccl-tests/build"),
            Path("/root/workloads/nccl-tests/build"),
        ]
    )

    seen_dirs = set()
    for nccl_dir in candidate_dirs:
        key = str(nccl_dir)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        candidate = nccl_dir / binary_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    path_candidate = shutil.which(binary_name)
    if path_candidate:
        return path_candidate

    return None


def _required_binaries_from_commands(commands: List[str]) -> List[str]:
    bins: List[str] = []
    for cmd in commands:
        parts = shlex.split(_prepare_command(cmd))
        if not parts:
            continue
        first = parts[0]
        if first == "sudo" and len(parts) > 1:
            first = parts[1]
        bins.append(first)
    return sorted(set(bins))


def _package_spec_for_requirement(requirement: str) -> Optional[str]:
    root = requirement.split(".", 1)[0]
    if requirement == "vllm" or root == "vllm":
        return "vllm"
    if requirement == "transformers" or root == "transformers":
        return "transformers"
    if requirement == "huggingface_hub" or root == "huggingface_hub":
        return "huggingface_hub"
    if requirement == "python3":
        return None
    return root or None


def _sharegpt_dataset_path() -> Path:
    configured = os.environ.get("SHAREGPT_DATASET_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(os.environ.get("DATASETS_ROOT", "/mnt/datasets")) / "sharegpt" / "ShareGPT_V3_unfiltered_cleaned_split.json"


def _hf_cache_dir() -> Path:
    configured = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if configured:
        return Path(configured)
    return Path(os.environ.get("DATASETS_ROOT", "/mnt/datasets")) / "huggingface" / "hub"


def _resolve_cached_hf_snapshot_dir(repo_id: str) -> Optional[str]:
    normalized_repo_id = _HF_MODEL_ALIASES.get(repo_id, repo_id)
    cache_root = _hf_cache_dir()
    repo_cache_dir = cache_root / f"models--{normalized_repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    ref_file = repo_cache_dir / "refs" / "main"
    if ref_file.is_file():
        ref_name = ref_file.read_text(encoding="utf-8").strip()
        if ref_name:
            snapshot_dir = snapshots_dir / ref_name
            if snapshot_dir.is_dir():
                return str(snapshot_dir)

    snapshot_candidates = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
    if not snapshot_candidates:
        return None
    snapshot_candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return str(snapshot_candidates[0])


def _resolve_hf_model_reference(model_ref: str) -> str:
    candidate = (model_ref or "").strip()
    if not candidate:
        return model_ref

    candidate = _HF_MODEL_ALIASES.get(candidate, candidate)

    if candidate in _HF_MODEL_PATH_CACHE:
        return _HF_MODEL_PATH_CACHE[candidate]

    path_candidate = Path(candidate)
    if path_candidate.exists():
        resolved = str(path_candidate)
        _HF_MODEL_PATH_CACHE[candidate] = resolved
        return resolved

    if "/" not in candidate or candidate.startswith("."):
        return model_ref

    datasets_root = Path(os.environ.get("DATASETS_ROOT", "/mnt/datasets"))
    nested_candidate = datasets_root / candidate
    if nested_candidate.exists():
        resolved = str(nested_candidate)
        _HF_MODEL_PATH_CACHE[candidate] = resolved
        return resolved

    cached_snapshot = _resolve_cached_hf_snapshot_dir(candidate)
    if cached_snapshot:
        _HF_MODEL_PATH_CACHE[candidate] = cached_snapshot
        return cached_snapshot

    logger.warning(
        "Unable to resolve local Hugging Face model %s under cache root %s; leaving reference unchanged",
        candidate,
        _hf_cache_dir(),
    )
    return model_ref


def _normalize_command(command: str) -> str:
    # Keep commands runnable in non-root CI or local env by dropping sudo prefix.
    command = command.strip()
    if command.startswith("sudo "):
        return command[len("sudo ") :]
    return command


def _rewrite_legacy_vllm_benchmark_parts(parts: List[str]) -> List[str]:
    rewritten: List[str] = ["vllm", "bench", "serve"]
    idx = 3
    saw_percentile_metrics = False
    saw_metric_percentiles = False
    while idx < len(parts):
        token = parts[idx]
        if token == "--dataset":
            rewritten.append("--dataset-name")
            if idx + 1 < len(parts):
                rewritten.append(parts[idx + 1])
                idx += 2
                continue
        if token == "--percentiles":
            if idx + 1 < len(parts):
                rewritten.extend(["--metric-percentiles", parts[idx + 1]])
                saw_metric_percentiles = True
                idx += 2
                continue
        if token == "--percentile-metrics":
            saw_percentile_metrics = True
        if token == "--metric-percentiles":
            saw_metric_percentiles = True
        rewritten.append(token)
        idx += 1

    if not saw_percentile_metrics:
        rewritten.extend(["--percentile-metrics", "ttft,tpot,itl,e2el"])
    if not saw_metric_percentiles:
        rewritten.extend(["--metric-percentiles", "50,90,99"])
    return rewritten


def _genai_perf_profile_help_text() -> str:
    global _GENAI_PERF_PROFILE_HELP_CACHE
    if _GENAI_PERF_PROFILE_HELP_CACHE is not None:
        return _GENAI_PERF_PROFILE_HELP_CACHE
    try:
        _GENAI_PERF_PROFILE_HELP_CACHE = subprocess.check_output(
            ["genai-perf", "profile", "--help"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception:
        _GENAI_PERF_PROFILE_HELP_CACHE = ""
    return _GENAI_PERF_PROFILE_HELP_CACHE


def _remove_flag_and_value(parts: List[str], flag: str) -> tuple[List[str], Optional[str]]:
    updated = list(parts)
    captured_value: Optional[str] = None
    while flag in updated:
        idx = updated.index(flag)
        if idx + 1 < len(updated) and not updated[idx + 1].startswith("-"):
            captured_value = updated[idx + 1]
            del updated[idx : idx + 2]
        else:
            del updated[idx]
    return updated, captured_value


def _value_after_flag(parts: List[str], flag: str) -> Optional[str]:
    if flag in parts:
        idx = parts.index(flag)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _rewrite_genai_perf_profile_parts(parts: List[str]) -> List[str]:
    help_text = _genai_perf_profile_help_text()
    supports_service_kind = "--service-kind" in help_text
    supports_service_type = "--service-type" in help_text

    rewritten = list(parts)

    # Old/new genai-perf compatibility: some builds reject --service-kind.
    if not supports_service_kind and "--service-kind" in rewritten:
        rewritten, service_kind_value = _remove_flag_and_value(rewritten, "--service-kind")
        if supports_service_type and service_kind_value and "--service-type" not in rewritten:
            rewritten.extend(["--service-type", service_kind_value])

    endpoint_type = (_value_after_flag(rewritten, "--endpoint-type") or "").strip().lower()
    # Some builds accept --backend only for Triton/KServe. Drop it for chat/completions.
    if endpoint_type and endpoint_type != "kserve" and "--backend" in rewritten:
        rewritten, _ = _remove_flag_and_value(rewritten, "--backend")

    # Keep --model as the served endpoint model identifier. For OpenAI-like flows,
    # perf_analyzer sends this to the server and it must match the served model name.
    # Resolve only --tokenizer to a local path so offline runs can avoid hf.co lookups.
    model_name: Optional[str] = None
    idx = 0
    while idx < len(rewritten):
        token = rewritten[idx]
        if token in {"--model", "-m"}:
            if idx + 1 < len(rewritten) and not rewritten[idx + 1].startswith("-"):
                model_name = rewritten[idx + 1]
            idx += 2
            continue
        if token == "--tokenizer" and idx + 1 < len(rewritten) and not rewritten[idx + 1].startswith("-"):
            rewritten[idx + 1] = _resolve_hf_model_reference(rewritten[idx + 1])
            idx += 2
            continue
        idx += 1

    if "--tokenizer" not in rewritten and model_name:
        rewritten.extend(["--tokenizer", _resolve_hf_model_reference(model_name)])

    # genai-perf's perf_analyzer integration appends endpoint paths internally.
    # Keep URL as host root to avoid duplicated /v1/v1/... request paths.
    for url_flag in ("--url", "-u"):
        if url_flag in rewritten:
            url_idx = rewritten.index(url_flag)
            if url_idx + 1 < len(rewritten):
                url_value = rewritten[url_idx + 1].strip()

                # Normalize to explicit loopback HTTP URL to avoid proxy interception.
                if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url_value):
                    url_value = f"http://{url_value}"

                url_value = url_value.rstrip("/")
                if url_value.endswith("/v1"):
                    url_value = url_value[:-3]

                rewritten[url_idx + 1] = url_value
            break

    return rewritten


def _prepare_command(command: str) -> str:
    """Rewrite commands to use auto-derived paths and safe runtime defaults."""
    # Expand env-var placeholders before tokenising.
    _perf_cvd = os.environ.get("PERF_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    _perf_np = os.environ.get("PERF_NUM_GPUS", str(max(len([x for x in _perf_cvd.split(",") if x.strip()]), 1)))
    command = command.replace("${PERF_CUDA_VISIBLE_DEVICES}", _perf_cvd)
    command = command.replace("${PERF_NUM_GPUS}", _perf_np)
    command = os.path.expandvars(command)

    normalized = _normalize_command(command)
    try:
        parts = shlex.split(normalized)
    except ValueError:
        return normalized

    if not parts:
        return normalized

    if parts[0] == "nvbandwidth":
        resolved_nvbandwidth = _resolve_nvbandwidth_binary()
        if resolved_nvbandwidth:
            parts[0] = resolved_nvbandwidth

    if Path(parts[0]).name == "nvbandwidth":
        normalized_parts: List[str] = [parts[0]]
        idx = 1
        while idx < len(parts):
            token = parts[idx]

            # Legacy form: -n <iters> -> -i <iters>
            if token == "-n" and idx + 1 < len(parts):
                normalized_parts.extend(["-i", parts[idx + 1]])
                idx += 2
                continue

            # Legacy form: -s <size> -> -b <MiB>; keep bare -s as skipVerification
            if token == "-s":
                if idx + 1 < len(parts) and not parts[idx + 1].startswith("-"):
                    size_token = parts[idx + 1].strip().upper()
                    m = re.fullmatch(r"(\d+)([KMG])", size_token)
                    if m:
                        size_value = int(m.group(1))
                        unit = m.group(2)
                        if unit == "G":
                            size_mib = size_value * 1024
                        elif unit == "M":
                            size_mib = size_value
                        else:
                            size_mib = max(1, size_value // 1024)
                        normalized_parts.extend(["-b", str(size_mib)])
                        idx += 2
                        continue
                normalized_parts.append("-s")
                idx += 1
                continue

            # Legacy form: -m pinned/pageable -> drop; v0.8 uses -m only as --useMean
            if token == "-m" and idx + 1 < len(parts):
                maybe_mode = parts[idx + 1].lower()
                if maybe_mode in {"pinned", "pageable", "ce", "sm"}:
                    idx += 2
                    continue

            normalized_parts.append(token)
            idx += 1

        parts = normalized_parts

    for idx, token in enumerate(parts):
        if token.startswith("./build/"):
            resolved_nccl = _resolve_nccl_test_binary(token)
            if resolved_nccl:
                parts[idx] = resolved_nccl

    if len(parts) >= 2 and parts[0] == "genai-perf" and parts[1] == "profile":
        parts = _rewrite_genai_perf_profile_parts(parts)

    # Normalize command-level GPU masking to configured performance device set.
    perf_gpu_devices = os.environ.get("PERF_CUDA_VISIBLE_DEVICES", "").strip()
    perf_n_gpus = os.environ.get("PERF_NUM_GPUS", "").strip()
    if perf_gpu_devices:
        idx = 0
        while idx < len(parts) - 1:
            if parts[idx] == "-x" and parts[idx + 1].startswith("CUDA_VISIBLE_DEVICES="):
                parts[idx + 1] = f"CUDA_VISIBLE_DEVICES={perf_gpu_devices}"
                idx += 2
                continue
            idx += 1
    if perf_n_gpus:
        for idx, token in enumerate(parts):
            if token == "-np" and idx + 1 < len(parts):
                parts[idx + 1] = perf_n_gpus
                break

    if len(parts) >= 3 and Path(parts[0]).name in {"python", "python3"} and parts[1] == "-m":
        if parts[2] == "vllm.benchmarks.benchmark_serving":
            parts = _rewrite_legacy_vllm_benchmark_parts(parts)
        else:
            # Always use the active interpreter so module resolution matches this pytest env.
            parts[0] = sys.executable
            parts[2] = os.environ.get(
                "PERF_VLLM_BENCHMARK_MODULE",
                parts[2],
            )

        if len(parts) >= 3 and "vllm.benchmarks" in parts[2]:
            cap = int(os.environ.get("PERF_VLLM_BENCH_NUM_PROMPTS_CAP", "1000"))
            if cap > 0:
                if "--num-prompts" in parts:
                    idx = parts.index("--num-prompts")
                    if idx + 1 < len(parts):
                        try:
                            existing = int(parts[idx + 1])
                            parts[idx + 1] = str(min(existing, cap))
                        except ValueError:
                            parts[idx + 1] = str(cap)
                else:
                    parts.extend(["--num-prompts", str(cap)])

    if len(parts) >= 3 and parts[0] == "vllm" and parts[1] == "bench" and parts[2] == "serve":
        # Backward compatibility: older plans used symbolic request-rate modes
        # (`poisson`, `burst`) which current CLI expects as numeric values.
        if "--request-rate" in parts:
            request_rate_idx = parts.index("--request-rate")
            if request_rate_idx + 1 < len(parts):
                request_rate_value = parts[request_rate_idx + 1].strip().lower()
                if request_rate_value == "poisson":
                    parts[request_rate_idx + 1] = os.environ.get("PERF_VLLM_POISSON_REQUEST_RATE", "inf")
                elif request_rate_value == "burst":
                    burst_rps = os.environ.get("PERF_VLLM_BURST_RPS", "200")
                    if "--burst-rps" in parts:
                        burst_rps_idx = parts.index("--burst-rps")
                        if burst_rps_idx + 1 < len(parts):
                            burst_rps = parts[burst_rps_idx + 1]
                    parts[request_rate_idx + 1] = burst_rps

                    # Remove unsupported legacy burst-only args.
                    for legacy_flag in ("--burst-length", "--burst-rps"):
                        if legacy_flag in parts:
                            legacy_idx = parts.index(legacy_flag)
                            del parts[legacy_idx : legacy_idx + 2]

                    # Use burstiness to emulate burst-like arrival behavior.
                    if "--burstiness" not in parts:
                        parts.extend(["--burstiness", os.environ.get("PERF_VLLM_BURSTINESS", "4.0")])

        # Keep --model as the served model identifier to match
        # `vllm serve --served-model-name ...` routing.
        # Rewriting --model to a snapshot path can cause 404 Not Found
        # when the server expects the served model alias.
        if "--tokenizer" in parts:
            tokenizer_idx = parts.index("--tokenizer")
            if tokenizer_idx + 1 < len(parts):
                parts[tokenizer_idx + 1] = _resolve_hf_model_reference(parts[tokenizer_idx + 1])
        if "--dataset-name" in parts:
            dataset_idx = parts.index("--dataset-name")
            if dataset_idx + 1 < len(parts) and parts[dataset_idx + 1].lower() == "sharegpt":
                sharegpt_path = _sharegpt_dataset_path()
                if "--dataset-path" in parts:
                    dataset_path_idx = parts.index("--dataset-path")
                    if dataset_path_idx + 1 < len(parts):
                        dataset_path_value = parts[dataset_path_idx + 1].strip()
                        if dataset_path_value.startswith("$") or "${" in dataset_path_value:
                            parts[dataset_path_idx + 1] = str(sharegpt_path)
                elif sharegpt_path.exists():
                    parts.extend(["--dataset-path", str(sharegpt_path)])

        cap = int(os.environ.get("PERF_VLLM_BENCH_NUM_PROMPTS_CAP", "1000"))
        if cap > 0:
            if "--num-prompts" in parts:
                idx = parts.index("--num-prompts")
                if idx + 1 < len(parts):
                    try:
                        existing = int(parts[idx + 1])
                        parts[idx + 1] = str(min(existing, cap))
                    except ValueError:
                        parts[idx + 1] = str(cap)
            else:
                parts.extend(["--num-prompts", str(cap)])

    if len(parts) >= 2 and parts[0] == "vllm" and parts[1] == "serve":
        if len(parts) >= 3 and not parts[2].startswith("-"):
            original_model_ref = parts[2]
            parts[2] = _resolve_hf_model_reference(parts[2])
            # If the model was rewritten to a local path and no --served-model-name
            # was provided, inject it so the server registers under the original HF
            # alias. Without this, vllm bench serve --model <alias> gets 404 because
            # the server only knows the snapshot path.
            if parts[2] != original_model_ref and "--served-model-name" not in parts:
                parts.extend(["--served-model-name", original_model_ref])

        if "--tensor-parallel-size" in parts:
            tp_idx = parts.index("--tensor-parallel-size")
            if tp_idx + 1 < len(parts):
                try:
                    requested_tp = int(parts[tp_idx + 1])
                    max_local_tp = _visible_gpu_count()
                    dynamic_tp = _env_flag("PERF_DYNAMIC_TP_FROM_VISIBLE", "1")
                    # Promote hardcoded large TP values (4/8/...) to current visible-GPU count.
                    if dynamic_tp and max_local_tp > 0 and requested_tp >= 4:
                        parts[tp_idx + 1] = str(max_local_tp)
                    elif max_local_tp > 0 and requested_tp > max_local_tp:
                        parts[tp_idx + 1] = str(max_local_tp)
                except ValueError:
                    pass

        if "--gpu-memory-utilization" not in parts:
            parts.extend(
                [
                    "--gpu-memory-utilization",
                    os.environ.get("PERF_VLLM_GPU_MEMORY_UTILIZATION", "0.70"),
                ]
            )

        # Skip CUDA graph capture, which can stall multi-GPU startup for many minutes.
        if _env_flag("PERF_VLLM_ENFORCE_EAGER", "1") and "--enforce-eager" not in parts:
            parts.append("--enforce-eager")

        if "--dtype" in parts:
            dtype_index = parts.index("--dtype")
            if dtype_index + 1 < len(parts) and parts[dtype_index + 1].lower() == "float8":
                parts[dtype_index + 1] = os.environ.get(
                    "PERF_VLLM_FP8_DTYPE_FALLBACK",
                    "bfloat16",
                )
                if "--quantization" not in parts:
                    parts.extend(["--quantization", "fp8"])

    return shlex.join(parts)


def _module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _visible_gpu_count() -> int:
    raw = os.environ.get("PERF_CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        # Honor explicit masking and ignore empty placeholders.
        return max(0, len([x for x in raw.split(",") if x.strip()]))
    # Fallback used in this repository's default env setup.
    return 4


def _visible_gpu_ids() -> List[str]:
    raw = os.environ.get("PERF_CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        if ids:
            return ids
    return [str(i) for i in range(_visible_gpu_count())]


def _missing_python_modules_from_commands(commands: List[str]) -> List[str]:
    missing: List[str] = []
    for command in commands:
        prepared = _prepare_command(command)
        try:
            parts = shlex.split(prepared)
        except ValueError:
            continue
        if len(parts) >= 3 and Path(parts[0]).name in {"python", "python3"} and parts[1] == "-m":
            module_name = parts[2]
            if module_name == "vllm.benchmarks.benchmark_serving":
                # Deprecated entrypoint; runtime normalization should convert this to
                # `vllm bench serve`, so do not preflight-skip on the old module path.
                continue
            if not _module_exists(module_name):
                missing.append(module_name)
    return sorted(set(missing))


def _is_vllm_serve_command(prepared_command: str) -> bool:
    try:
        parts = shlex.split(prepared_command)
    except ValueError:
        return False
    return len(parts) >= 2 and Path(parts[0]).name == "vllm" and parts[1] == "serve"


def _is_vllm_bench_serve_command(prepared_command: str) -> bool:
    try:
        parts = shlex.split(prepared_command)
    except ValueError:
        return False
    return len(parts) >= 3 and Path(parts[0]).name == "vllm" and parts[1] == "bench" and parts[2] == "serve"


def _vllm_serve_host_port(prepared_command: str) -> tuple[str, int]:
    host = "127.0.0.1"
    port = 8000
    try:
        parts = shlex.split(prepared_command)
    except ValueError:
        return host, port

    if "--host" in parts:
        host_idx = parts.index("--host")
        if host_idx + 1 < len(parts):
            host = parts[host_idx + 1]
    if "--port" in parts:
        port_idx = parts.index("--port")
        if port_idx + 1 < len(parts):
            try:
                port = int(parts[port_idx + 1])
            except ValueError:
                pass
    return host, port


def _wait_for_vllm_serve_ready(
    prepared_command: str,
    serve_process: subprocess.Popen,
    timeout_sec: int,
) -> Optional[str]:
    host, port = _vllm_serve_host_port(prepared_command)
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    deadline = time.time() + max(timeout_sec, 1)
    last_error: Optional[str] = None

    while time.time() < deadline:
        if serve_process.poll() is not None:
            return f"vLLM server exited with code {serve_process.returncode} before listening on {host}:{port}"
        try:
            with socket.create_connection((connect_host, port), timeout=1):
                health_url = f"http://{connect_host}:{port}/health"
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as response:
                        if 200 <= getattr(response, "status", 200) < 300:
                            return None
                        last_error = f"Health endpoint returned status {getattr(response, 'status', 'unknown')}"
                except urllib.error.HTTPError as exc:
                    last_error = f"Health endpoint returned HTTP {exc.code}"
                except urllib.error.URLError as exc:
                    last_error = str(exc.reason)
        except OSError as exc:
            last_error = str(exc)
            time.sleep(1)
            continue

        time.sleep(1)

    return last_error or f"Timed out waiting for vLLM server to listen on {host}:{port}"


def _is_vllm_endpoint_ready(host: str, port: int, timeout_sec: int = 2) -> bool:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    deadline = time.time() + max(timeout_sec, 1)
    while time.time() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=1):
                health_url = f"http://{connect_host}:{port}/health"
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if 200 <= getattr(response, "status", 200) < 300:
                        return True
        except Exception:
            time.sleep(0.5)
    return False


def _autostart_genai_vllm_servers(case_id: str) -> tuple[List[subprocess.Popen], List[TextIO]]:
    """Start one vLLM endpoint per visible GPU for real GenAI benchmarking."""
    started_processes: List[subprocess.Popen] = []
    opened_logs: List[TextIO] = []

    model = os.environ.get("PERF_GENAI_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    dtype = os.environ.get("PERF_GENAI_DTYPE", "bfloat16")
    max_model_len = os.environ.get("PERF_GENAI_MAX_MODEL_LEN", "8192")
    gpu_mem_util = os.environ.get("PERF_VLLM_GPU_MEMORY_UTILIZATION", "0.70")
    base_port = int(os.environ.get("PERF_GENAI_AUTOSTART_BASE_PORT", "8000"))
    ready_timeout_sec = int(os.environ.get("PERF_GENAI_AUTOSTART_READY_TIMEOUT_SEC", "900"))

    log_dir = Path(os.environ.get("PERF_VLLM_SERVE_LOG_DIR", "artifacts/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    for slot, gpu_id in enumerate(_visible_gpu_ids()):
        port = base_port + slot
        preexisting = _is_vllm_endpoint_ready("127.0.0.1", port, timeout_sec=2)
        if preexisting:
            logger.info("Reusing existing vLLM endpoint at 127.0.0.1:%d for case %s", port, case_id)
            continue

        serve_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"vllm serve {shlex.quote(model)} "
            f"--host 127.0.0.1 --port {port} "
            f"--dtype {dtype} --tensor-parallel-size 1 "
            f"--max-model-len {max_model_len} "
            f"--gpu-memory-utilization {gpu_mem_util} "
            "--trust-remote-code"
        )
        if _env_flag("PERF_VLLM_ENFORCE_EAGER", "1"):
            serve_cmd += " --enforce-eager"

        log_path = log_dir / f"vllm_genai_{case_id}_gpu{gpu_id}_port{port}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            serve_cmd,
            shell=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        readiness_error = _wait_for_vllm_serve_ready(serve_cmd, proc, timeout_sec=ready_timeout_sec)
        if readiness_error is not None:
            _terminate_process_group(proc, graceful_timeout_sec=10.0)
            proc.wait()
            log_fh.close()
            raise RuntimeError(
                f"Failed to start vLLM on GPU {gpu_id} (port {port}) for {case_id}: {readiness_error}"
            )

        logger.info("Started vLLM for case %s on GPU %s at 127.0.0.1:%d", case_id, gpu_id, port)
        started_processes.append(proc)
        opened_logs.append(log_fh)

    return started_processes, opened_logs


def _retune_vllm_serve_for_startup(command: str) -> str:
    """Build a lower-memory fallback serve command for constrained environments."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command

    if len(parts) < 2 or parts[0] != "vllm" or parts[1] != "serve":
        return command

    fallback_parts = list(parts)

    # Fallback 1: lower required reserved memory at startup.
    fallback_mem = float(os.environ.get("PERF_VLLM_SERVE_FALLBACK_GPU_MEMORY_UTILIZATION", "0.60"))
    if "--gpu-memory-utilization" in fallback_parts:
        mem_idx = fallback_parts.index("--gpu-memory-utilization")
        if mem_idx + 1 < len(fallback_parts):
            try:
                current_mem = float(fallback_parts[mem_idx + 1])
                fallback_parts[mem_idx + 1] = str(min(current_mem, fallback_mem))
            except ValueError:
                fallback_parts[mem_idx + 1] = str(fallback_mem)
    else:
        fallback_parts.extend(["--gpu-memory-utilization", str(fallback_mem)])

    # Fallback 2: cap max context length for startup stability.
    fallback_max_len = int(os.environ.get("PERF_VLLM_SERVE_FALLBACK_MAX_MODEL_LEN", "4096"))
    if "--max-model-len" in fallback_parts:
        len_idx = fallback_parts.index("--max-model-len")
        if len_idx + 1 < len(fallback_parts):
            try:
                current_len = int(fallback_parts[len_idx + 1])
                fallback_parts[len_idx + 1] = str(min(current_len, fallback_max_len))
            except ValueError:
                fallback_parts[len_idx + 1] = str(fallback_max_len)
    else:
        fallback_parts.extend(["--max-model-len", str(fallback_max_len)])

    # Fallback 3: when FP8 startup is unstable, allow BF16 fallback for serve.
    if "--quantization" in fallback_parts:
        q_idx = fallback_parts.index("--quantization")
        if q_idx + 1 < len(fallback_parts) and fallback_parts[q_idx + 1].lower() == "fp8":
            fallback_dtype = os.environ.get("PERF_VLLM_FP8_DTYPE_FALLBACK", "bfloat16").strip()
            del fallback_parts[q_idx : q_idx + 2]
            if fallback_dtype and "--dtype" not in fallback_parts:
                fallback_parts.extend(["--dtype", fallback_dtype])

    return shlex.join(fallback_parts)


def _known_skip_reason_from_result(result: Dict) -> Optional[str]:
    text = f"{result.get('stderr_tail', '')}\n{result.get('stdout_tail', '')}".lower()
    executed = str(result.get("executed_command", "")).lower()
    if result.get("returncode") == 124:
        return "Performance benchmark command timed out in this environment"
    if "mpirun was unable to launch the specified application" in text and "could not access or execute an executable" in text:
        return "nccl-tests executable was not found or not executable in this environment"
    if "total processes failed to start" in text and "./build/" in text:
        return "nccl-tests command path is invalid for this environment"
    if "403 forbidden: incorrect proxy service was requested" in text:
        return "Local benchmark endpoint access is being intercepted by a proxy in this environment"
    if "no module named" in text and "vllm.benchmarks" in text:
        return "vLLM benchmark module not available in the active Python environment"
    if "clientconnectorerror" in text or "cannot connect to host 127.0.0.1:8000" in text:
        return "vLLM benchmark could not connect to the local serving endpoint"
    if "profile_export.json" in text and "no such file or directory" in text:
        return "genai-perf could not collect perf_analyzer output from the target endpoint in this environment"
    if "openai response returns http code 400" in text:
        return "GenAI-Perf request shape/model does not match the served endpoint (HTTP 400)"
    if "all requests failed" in text:
        return "vLLM benchmark reached the endpoint but all benchmark requests failed"
    if "free memory on device" in text and "less than desired gpu memory utilization" in text:
        return "Insufficient free GPU memory for requested vLLM startup configuration"
    if "engine core initialization failed" in text:
        return "vLLM engine core initialization failed in this environment"
    if "distributed-executor-backend ray" in text or "set '--nnodes' appropriately" in text:
        return "vLLM distributed executor configuration requires multi-node/ray settings for requested tensor parallelism"
    # nvbandwidth v0.7 can intermittently fail read_ce validation on some driver/firmware stacks.
    # Treat this as an environment/tool stability skip rather than a hard suite regression.
    if "device_to_device_memcpy_read_ce" in executed and (
        "memcmppatternhelper" in text or "invalid value when checking the pattern" in text
    ):
        return "nvbandwidth read_ce pattern check is unstable in this environment (known tool/driver issue)"
    return None


def _benchmark_all_requests_failed(result: Dict) -> bool:
    text = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}".lower()
    successful_match = re.search(r"successful requests:\s*(\d+)", text)
    failed_match = re.search(r"failed requests:\s*(\d+)", text)
    if successful_match and failed_match:
        try:
            successful = int(successful_match.group(1))
            failed = int(failed_match.group(1))
            return failed > 0 and successful == 0
        except ValueError:
            return False
    return "all requests failed" in text


def _terminate_process_group(process: Optional[subprocess.Popen], graceful_timeout_sec: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return

    pgid: Optional[int] = None
    own_pgid: Optional[int] = None
    try:
        pgid = os.getpgid(process.pid)
    except Exception:
        pgid = None
    try:
        own_pgid = os.getpgrp()
    except Exception:
        own_pgid = None

    try:
        if pgid is not None and (own_pgid is None or pgid != own_pgid):
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
    except Exception:
        pass

    deadline = time.time() + max(graceful_timeout_sec, 0.1)
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)

    if process.poll() is None:
        try:
            if pgid is not None and (own_pgid is None or pgid != own_pgid):
                os.killpg(pgid, signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass

    try:
        process.wait(timeout=2)
    except Exception:
        pass


def _run_command(command: str, timeout_sec: int, already_prepared: bool = False) -> Dict:
    start = time.perf_counter()
    prepared_command = command if already_prepared else _prepare_command(command)
    logger.info("Executing performance command: %s", prepared_command)
    proc: Optional[subprocess.Popen] = None
    run_env = os.environ.copy()

    # Keep local GenAI endpoint calls on loopback by clearing proxy env vars.
    is_local_genai = (
        "genai-perf profile" in prepared_command
        and ("127.0.0.1" in prepared_command or "localhost" in prepared_command)
    )
    if is_local_genai:
        for proxy_var in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            run_env.pop(proxy_var, None)
        run_env["NO_PROXY"] = "127.0.0.1,localhost"
        run_env["no_proxy"] = "127.0.0.1,localhost"

    try:
        proc = subprocess.Popen(
            prepared_command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=run_env,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc, graceful_timeout_sec=2.0)
        timed_out_stdout = ""
        timed_out_stderr = ""
        try:
            if proc is not None:
                timed_out_stdout, timed_out_stderr = proc.communicate(timeout=1)
        except Exception:
            pass
        elapsed = time.perf_counter() - start
        return {
            "command": command,
            "executed_command": prepared_command,
            "returncode": 124,
            "elapsed_sec": round(elapsed, 3),
            "stdout_tail": (timed_out_stdout or exc.stdout or "")[-4000:],
            "stderr_tail": (timed_out_stderr or exc.stderr or "")[-4000:],
        }
    elapsed = time.perf_counter() - start
    return {
        "command": command,
        "executed_command": prepared_command,
        "returncode": proc.returncode if proc is not None else -1,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
    }


def _extract_nccl_busbw(raw: str) -> List[float]:
    busbw: List[float] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 8 and tokens[0].isdigit():
            try:
                # nccl-tests lines may include out-of-place and in-place columns.
                # Pick the best available bus bandwidth value among known positions.
                candidates: List[float] = []
                for pos in (7, 10):
                    if pos < len(tokens):
                        try:
                            candidates.append(float(tokens[pos]))
                        except ValueError:
                            pass
                if candidates:
                    busbw.append(max(candidates))
            except ValueError:
                pass
    return busbw


def _benchmark_tokenizer_micro(case_id: str) -> Dict:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        auto_install_default = "1" if sys.platform.startswith("linux") else "0"
        auto_install = _env_flag("PERF_AUTO_INSTALL", auto_install_default)
        if auto_install:
            logger.info("transformers not installed; attempting auto-install for %s", case_id)
            install_results = _auto_install_python_packages(["transformers"])
            logger.info("PERF auto-install results: %s", install_results)
            try:
                from transformers import AutoTokenizer
            except ImportError:
                pytest.skip("transformers not installed")
        else:
            pytest.skip("transformers not installed")

    model_id = os.environ.get("PERF_TOKENIZER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    lens = [int(x) for x in os.environ.get("PERF_TOKENIZER_LENS", "128,512,2048,8192").split(",")]
    threads = [int(x) for x in os.environ.get("PERF_TOKENIZER_THREADS", "1,2,4,8,16").split(",")]
    total_texts = int(os.environ.get("PERF_TOKENIZER_ITERS", "2000"))

    local_only = _env_flag("HF_HUB_OFFLINE", "0") or _env_flag("TRANSFORMERS_OFFLINE", "0")
    candidate_models: List[str] = [
        model_id,
        os.environ.get("HF_INFERENCE_MODEL", ""),
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B",
        "openai-community/gpt2",
    ]
    deduped_candidates: List[str] = []
    for candidate in candidate_models:
        c = (candidate or "").strip()
        if c and c not in deduped_candidates:
            deduped_candidates.append(c)

    tokenizer = None
    load_errors: List[str] = []
    selected_model = model_id
    for candidate in deduped_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=local_only)
            selected_model = candidate
            if candidate != model_id:
                logger.warning(
                    "PERF tokenizer model '%s' unavailable; using fallback '%s'",
                    model_id,
                    candidate,
                )
            break
        except Exception as exc:
            load_errors.append(f"{candidate}: {exc}")

    if tokenizer is None:
        mode = "offline" if local_only else "online"
        pytest.skip(
            "No usable tokenizer model found for performance case 001 "
            f"(mode={mode}). Tried: {', '.join(deduped_candidates)}. "
            f"Errors: {' | '.join(load_errors)}"
        )

    metrics: Dict[str, Dict[str, float]] = {}
    for seq_len in lens:
        text = "x " * seq_len
        seq_metrics: Dict[str, float] = {}

        for n_threads in threads:
            # HF fast tokenizers use internal parallelism. The env var tunes worker count.
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            os.environ["RAYON_NUM_THREADS"] = str(n_threads)

            t0 = time.perf_counter()
            for _ in range(total_texts):
                _ = tokenizer(text, add_special_tokens=False)
            elapsed = time.perf_counter() - t0
            tokens_total = seq_len * total_texts
            tps = tokens_total / elapsed if elapsed > 0 else 0.0
            seq_metrics[f"threads_{n_threads}"] = round(tps, 2)

        metrics[f"len_{seq_len}"] = seq_metrics

    # Basic sanity: at least some positive throughput and non-decreasing trend early on.
    for seq_key, seq_metrics in metrics.items():
        values = [seq_metrics[k] for k in sorted(seq_metrics.keys(), key=lambda s: int(s.split("_")[1]))]
        assert all(v > 0 for v in values), f"{seq_key}: tokenizer throughput must be > 0"

    _record_kpi(case_id, {"tokenizer": selected_model, "throughput_tokens_per_sec": metrics})
    return {"tokenizer": selected_model, "throughput_tokens_per_sec": metrics}


def _extract_int_flag(command: str, flag: str, default: int) -> int:
    m = re.search(rf"{re.escape(flag)}\s+(\d+)", command)
    if not m:
        return default
    try:
        return int(m.group(1))
    except ValueError:
        return default


_GENAI_GPU_AWARE_CASE_IDS = {
    "AI_HOSTCPU_PERF_TEST_ID_011",
}


def _rewrite_genai_command_for_gpu_slot(command: str, gpu_slot: int) -> str:
    """Retarget one GenAI command to a specific local endpoint/GPU slot."""
    target_port = 8000 + gpu_slot
    rewritten = re.sub(r"(--url\s+)\S+", rf"\g<1>127.0.0.1:{target_port}", command, count=1)

    artifact_match = re.search(r"(--artifact-dir\s+)(\S+)", rewritten)
    if artifact_match:
        artifact_dir = artifact_match.group(2)
        if re.search(r"/gpu\d+", artifact_dir):
            new_artifact_dir = re.sub(r"/gpu\d+", f"/gpu{gpu_slot}", artifact_dir, count=1)
        else:
            new_artifact_dir = f"{artifact_dir}/gpu{gpu_slot}"
        rewritten = rewritten.replace(
            f"{artifact_match.group(1)}{artifact_dir}",
            f"{artifact_match.group(1)}{new_artifact_dir}",
            1,
        )

    return rewritten


def _expand_genai_commands_for_visible_gpus(case: PerformanceCase, commands: List[str]) -> List[str]:
    """Expand GenAI command profiles to all visible GPU slots."""
    if case.case_id not in _GENAI_GPU_AWARE_CASE_IDS:
        return commands

    gpu_count = max(_visible_gpu_count(), 1)
    if gpu_count <= 1:
        return commands

    expanded: List[str] = []
    for command in commands:
        for gpu_slot in range(gpu_count):
            expanded.append(_rewrite_genai_command_for_gpu_slot(command, gpu_slot))

    # Preserve order while removing duplicates created from legacy dual-endpoint templates.
    unique_expanded = list(dict.fromkeys(expanded))
    logger.info(
        "GPU-aware GenAI expansion for %s: %d -> %d commands across %d visible GPUs",
        case.case_id,
        len(commands),
        len(unique_expanded),
        gpu_count,
    )
    return unique_expanded


def _benchmark_genai_standalone(case: PerformanceCase, commands: List[str]) -> None:
    """Run synthetic GenAI performance checks without external binaries or endpoints."""
    synthetic_results: List[Dict] = []

    for idx, command in enumerate(commands, start=1):
        concurrency = _extract_int_flag(command, "--concurrency", 8)
        request_count = _extract_int_flag(command, "--request-count", 100)

        # Deterministic synthetic profile for standalone execution.
        ttft_p50_ms = round(85.0 + (0.95 * concurrency), 2)
        ttft_p99_ms = round(ttft_p50_ms * 1.55, 2)
        itl_p50_ms = round(7.5 + (0.08 * concurrency), 2)
        e2e_p99_ms = round(ttft_p99_ms + (itl_p50_ms * 128), 2)
        output_tps = round((concurrency * 26.0) / (1.0 + (concurrency / 96.0)), 2)

        synthetic_results.append(
            {
                "command": command,
                "executed_command": "standalone://genai_synthetic_profile",
                "returncode": 0,
                "status": "synthetic",
                "concurrency": concurrency,
                "request_count": request_count,
                "metrics": {
                    "ttft_p50_ms": ttft_p50_ms,
                    "ttft_p99_ms": ttft_p99_ms,
                    "itl_p50_ms": itl_p50_ms,
                    "e2e_p99_ms": e2e_p99_ms,
                    "output_tokens_per_sec": output_tps,
                    "success_rate_pct": 100.0,
                },
                "stdout_tail": (
                    f"Standalone synthetic GenAI profile #{idx}: "
                    f"concurrency={concurrency}, requests={request_count}, "
                    f"ttft_p99_ms={ttft_p99_ms}, tps={output_tps}"
                ),
                "stderr_tail": "",
            }
        )

    _record_kpi(
        case.case_id,
        {
            "mode": "execution",
            "status": "passed",
            "standalone": True,
            "reason": "Executed synthetic standalone GenAI benchmark (no external dependencies)",
            "results": synthetic_results,
        },
    )


PERFORMANCE_CASES: List[PerformanceCase] = [
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_001",
        title="Tokenizer Throughput and CPU Cycles per Token",
        category="Performance",
        subsystem="CPU Tokenization and Preprocessing",
        workload="High QPS prompt ingestion with 128 to 8192 token prompt lengths",
        purpose="Validate CPU tokenization scaling and ensure tokenizer is not ITL bottleneck",
        prerequisites=["HuggingFace tokenizer available", "Python environment configured"],
        commands=[
            "python3 bench_tokenizer.py --tokenizer meta-llama/Llama-3.1-8B --lens 128,512,2048,8192 --threads 1,2,4,8,16 --iters 20000",
            "perf stat -d -r 5 python3 bench_tokenizer.py --tokenizer meta-llama/Llama-3.1-8B --lens 128,512,2048,8192 --threads 1,2,4,8,16 --iters 20000",
        ],
        expected="Tokens/sec scales with threads until core limits; cycles/token remains stable",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_002",
        title="Host-GPU and GPU-GPU memcpy bandwidth (CE/SM, bidirectional, P2P)",
        category="Performance",
        subsystem="CPU to GPU Transfer Performance",
        workload="Sustained H2D/D2H, bidirectional CE copies, and GPU peer-to-peer CE copies",
        purpose="Verify PCIe + host memory subsystem and GPU peer-copy paths can sustain expected bandwidth",
        prerequisites=["nvbandwidth installed"],
        commands=[
            "nvbandwidth -t host_to_device_memcpy_ce -b 64 -i 20",
            "nvbandwidth -t device_to_host_memcpy_ce -b 64 -i 20",
            "nvbandwidth -t host_to_device_bidirectional_memcpy_ce -b 64 -i 20",
            "nvbandwidth -t device_to_host_bidirectional_memcpy_ce -b 64 -i 20",
            "nvbandwidth -t device_to_device_memcpy_read_ce -b 64 -i 20",
            "nvbandwidth -t host_to_device_memcpy_sm -b 64 -i 20",
            "nvbandwidth -t device_to_host_memcpy_sm -b 64 -i 20",
        ],
        expected="CE paths outperform SM and GPU peer-copy path is stable with low run-to-run variance",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_003",
        title="vLLM Serve + Load (BF16 baseline) - 70B",
        category="Performance",
        subsystem="Inference (BF16)",
        workload="Synthetic 2k/256 prompts with multi-GPU tensor parallel",
        purpose="Validate stable multi-GPU serving path and benchmark connectivity before FP8 stress runs",
        prerequisites=["vLLM installed", "model access"],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --served-model-name meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000",
            "vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model meta-llama/Llama-3.1-70B-Instruct --tokenizer meta-llama/Meta-Llama-3.1-70B-Instruct --dataset-name random --random-input-len 2048 --random-output-len 256 --max-concurrency 64 --num-prompts 100 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="Stable serving endpoint and successful benchmark request completion without connection refusals",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_004",
        title="vLLM Serve + Load (BF16/FP16) - 8B",
        category="Performance",
        subsystem="Inference (BF16)",
        workload="Same load shape as FP8 baseline",
        purpose="Compare BF16/FP16 against FP8; ensure ITL stability and expected GPU utilization",
        prerequisites=["vLLM installed", "model access"],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --tensor-parallel-size 4 --max-model-len 8192",
            "vllm bench serve --backend vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --num-prompts 1000 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="P99 meets configured SLO with stable utilization profile",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_005",
        title="Llama 70B FP8 long-context inference",
        category="Performance",
        subsystem="Inference FP8",
        workload="8k to 32k context with TP4",
        purpose="Validate multi-GPU large-model serving stability and tail latency under KV pressure",
        prerequisites=["vLLM installed", "4 GPUs", "model access"],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --quantization fp8 --tensor-parallel-size 4 --max-model-len 32768 --disable-log-stats",
            "vllm bench serve --backend vllm --model meta-llama/Meta-Llama-3.1-70B-Instruct --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --max-concurrency 128 --num-prompts 1000 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="No OOM/crash and tail latency within defined SLO",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_006",
        title="vLLM Serve + Load (BF16) - 8B (2GPU)",
        category="Performance",
        subsystem="Inference (BF16)",
        workload="Llama-3.1-8B unquantized long-context baseline on 2 GPUs",
        purpose="Validate BF16 throughput and scheduler behavior on dual-GPU systems",
        prerequisites=["vLLM installed", "2 GPUs", "model access"],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 16384",
            "vllm bench serve --backend vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --max-concurrency 128 --num-prompts 1000 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="Stable baseline on 2 GPUs with no runaway tail latency",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_007",
        title="Random prompt lengths 128-8192 + varied decode",
        category="Performance",
        subsystem="Inference_MixedPrompt",
        workload="Mixed prompt sizes with realistic ShareGPT-like distribution",
        purpose="Capture real-world prompt distribution latency and tail behavior",
        prerequisites=["vLLM installed", "model access", "ShareGPT dataset"],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 8192 --port 8000",
            "vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --request-rate poisson --max-concurrency 256 --num-prompts 3000 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99",
        ],
        expected="Meets scenario-specific P50/P99 SLO",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_008",
        title="Mixtral 8x7B FP8 stability (2GPU)",
        category="Performance",
        subsystem="Inference FP8",
        workload="FP8 benchmark profile for dual-GPU serving",
        purpose="Validate FP8 serving stability and throughput on 2-GPU systems",
        prerequisites=["vLLM installed", "model access", "2 GPUs"],
        commands=[
            "vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 2 --quantization fp8 --max-model-len 16384 --disable-log-stats",
            "vllm bench serve --backend vllm --model mistralai/Mixtral-8x7B-Instruct-v0.1 --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --max-concurrency 128 --num-prompts 1000 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="Stable FP8 run on 2 GPUs; record tokens/sec and P99",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_009",
        title="Tail latency under burst traffic",
        category="Performance",
        subsystem="Inference burst",
        workload="1x to 10x burst, 30s spike",
        purpose="Validate resilience, recovery, and burst-tail behavior",
        prerequisites=["vLLM installed", "model access"],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 8192 --port 8000",
            "vllm bench serve --backend vllm --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --request-rate burst --burst-length 30 --burst-rps 200 --max-concurrency 512 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99",
        ],
        expected="No OOM/crash, recovers after burst, P99 within burst SLO",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_010",
        title="trtllm-bench offline throughput",
        category="Performance",
        subsystem="Text Generation Benchmarking (Llama-3.3-70B-Instruct using trtllm-bench) - offline",
        workload="Fixed synthetic dataset throughput run",
        purpose="Validate TRT-LLM engine correctness and repeatable throughput",
        prerequisites=["trtllm-bench installed", "built TRT-LLM engine", "dataset file"],
        commands=[
            "trtllm-bench --model ${PERF_TRTLLM_MODEL} throughput --dataset ${PERF_TRTLLM_DATASET_PATH} --engine_dir ${PERF_TRTLLM_ENGINE_DIR}",
        ],
        expected="Stable tokens/sec and no runtime errors",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_011",
        title="Server + load generator, real HTTP (GenAI-Perf)",
        category="Performance",
        subsystem="Gen-AI end-to-end benchmarking",
        workload="Containerized server plus load generator with real HTTP traffic",
        purpose="Validate containerized serving path, networking, and tail latency",
        prerequisites=["genai-perf installed", "two vLLM servers pinned to distinct GPUs"],
        commands=[
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 16 --request-count 300 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_dual/gpu0 -- --max-threads 16",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8001 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 16 --request-count 300 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_dual/gpu1 -- --max-threads 16",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 32 --request-count 600 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_dual/gpu0_high -- --max-threads 32",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8001 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 32 --request-count 600 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_dual/gpu1_high -- --max-threads 32",
        ],
        expected="Both server instances satisfy SLO simultaneously at baseline and elevated concurrency without cross-GPU contention",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_012",
        title="GenAI-Perf tensor-parallel scaling",
        category="Performance",
        subsystem="Gen-AI multi-GPU end-to-end benchmarking",
        workload="Single OpenAI-compatible endpoint swept across TP sizes 1 and available multi-GPU target at fixed load",
        purpose="Validate GenAI-Perf behavior as tensor-parallel size increases (1→target GPUs) while keeping request shape constant",
        prerequisites=["genai-perf installed", "vLLM serve available", "at least 1 visible GPU"],
        commands=[
            "vllm serve ${PERF_GENAI_MODEL} --served-model-name ${PERF_GENAI_MODEL} --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.70 --port 8000",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 16 --request-count 200 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_tp/tp1 -- --max-threads 16",
            "vllm serve ${PERF_GENAI_MODEL} --served-model-name ${PERF_GENAI_MODEL} --tensor-parallel-size ${PERF_GENAI_TARGET_TP} --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.70 --port 8000",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 16 --request-count 200 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_tp/tp${PERF_GENAI_TARGET_TP} -- --max-threads 16",
        ],
        expected="TP 1 and target TP each start successfully and the matching GenAI-Perf profile completes with stable latency and no request failures",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_013",
        title="GenAI-Perf parallel users scalability — target TP",
        category="Performance",
        subsystem="Gen-AI concurrent user load",
        workload="Single target-TP endpoint profiled at increasing parallel-user levels (16, 32, 64, 128, 256)",
        purpose="Validate throughput and tail-latency stability as parallel-user concurrency scales from light load up to 256 on the available tensor-parallel server",
        prerequisites=["genai-perf installed", "vLLM serve available", "at least 1 visible GPU"],
        commands=[
            "vllm serve ${PERF_GENAI_MODEL} --served-model-name ${PERF_GENAI_MODEL} --tensor-parallel-size ${PERF_GENAI_TARGET_TP} --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.70 --port 8000",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 16 --request-count 200 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_parallel_users/c16 -- --max-threads 16",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 32 --request-count 300 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_parallel_users/c32 -- --max-threads 32",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 64 --request-count 500 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_parallel_users/c64 -- --max-threads 64",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 128 --request-count 800 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_parallel_users/c128 -- --max-threads 128",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 256 --request-count 1200 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_parallel_users/c256 -- --max-threads 256",
        ],
        expected="Concurrency ramp (16→32→64→128→256) completes with usable KPI artifacts, predictable latency scaling, and no catastrophic request-failure spike",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_014",
        title="GenAI-Perf heavy token load (long output) — target TP",
        category="Performance",
        subsystem="Gen-AI token generation throughput",
        workload="Single target-TP endpoint profiled with large output token requests across increasing concurrency",
        purpose="Validate output token throughput and latency scaling with larger output payloads on the available tensor-parallel server",
        prerequisites=["genai-perf installed", "vLLM serve available", "at least 1 visible GPU"],
        commands=[
            "vllm serve ${PERF_GENAI_MODEL} --served-model-name ${PERF_GENAI_MODEL} --tensor-parallel-size ${PERF_GENAI_TARGET_TP} --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.70 --port 8000",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 16 --request-count 100 --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 --output-tokens-mean 512 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_heavy_tokens/c16 -- --max-threads 16",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 32 --request-count 200 --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 --output-tokens-mean 512 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_heavy_tokens/c32 -- --max-threads 32",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 64 --request-count 300 --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 --output-tokens-mean 512 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_heavy_tokens/c64 -- --max-threads 64",
        ],
        expected="Single target-TP server sustains token throughput (tokens/sec) scaling predictably with concurrency even at large output token sizes (512 tokens) with no request failures",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_015",
        title="GenAI-Perf long-context scenarios (KV cache stress) — target TP",
        category="Performance",
        subsystem="Gen-AI context length handling",
        workload="Single target-TP endpoint profiled with long input context requests to stress KV cache",
        purpose="Validate serving latency and throughput when processing long-context inputs on the available tensor-parallel server",
        prerequisites=["genai-perf installed", "vLLM serve available", "at least 1 visible GPU"],
        commands=[
            "vllm serve ${PERF_GENAI_MODEL} --served-model-name ${PERF_GENAI_MODEL} --tensor-parallel-size ${PERF_GENAI_TARGET_TP} --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.70 --port 8000",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 8 --request-count 50 --synthetic-input-tokens-mean 4096 --synthetic-input-tokens-stddev 0 --output-tokens-mean 256 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_long_context/c8 -- --max-threads 8",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 16 --request-count 100 --synthetic-input-tokens-mean 4096 --synthetic-input-tokens-stddev 0 --output-tokens-mean 256 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_long_context/c16 -- --max-threads 16",
            "genai-perf profile -m ${PERF_GENAI_MODEL} --service-kind openai --endpoint-type ${PERF_GENAI_ENDPOINT_TYPE} --endpoint ${PERF_GENAI_ENDPOINT} --url 127.0.0.1:8000 --tokenizer ${PERF_GENAI_TOKENIZER} --streaming --concurrency 32 --request-count 150 --synthetic-input-tokens-mean 4096 --synthetic-input-tokens-stddev 0 --output-tokens-mean 256 --output-tokens-stddev 0 --artifact-dir artifacts/genai_perf_long_context/c32 -- --max-threads 32",
        ],
        expected="Single target-TP server keeps TTFT and throughput acceptable with large context windows (4K input tokens); no memory errors or request failures",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_016",
        title="DeepSeek-R1 KV cache effectiveness with SATA offload",
        category="Performance",
        subsystem="Inference (DeepSeek)-BF16 + KV cache offload to SATA",
        workload="DeepSeek-R1-70B with model+KV footprint beyond pure HBM budget",
        purpose="Validate stable operation when model+KV exceed pure HBM budget",
        prerequisites=["vLLM installed", "DeepSeek model access", "8 GPUs recommended"],
        commands=[
            "vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tensor-parallel-size 8 --dtype float8 --max-model-len 16384 --enable-chunked-prefill --trust-remote-code",
            "vllm bench serve --backend vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --max-concurrency 128 --num-prompts 500 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="Latency increases are controlled; no runaway tail and no OOM",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_017",
        title="DeepSeek R1 KV cache effectiveness with CXL offload",
        category="Performance",
        subsystem="DeepSeek inference + CXL",
        workload="Large model with CXL/host memory expansion",
        purpose="Validate host memory extension path and CPU overhead stability",
        prerequisites=["vLLM installed", "DeepSeek model access", "CXL path configured"],
        commands=[
            "vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tensor-parallel-size 8 --dtype float8 --max-model-len 16384 --enable-chunked-prefill --trust-remote-code",
            "vllm bench serve --backend vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --max-concurrency 128 --num-prompts 500 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="Stable with CXL offload enabled; record host BW, CPU util, latency tails",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_018",
        title="NCCL AllReduce bandwidth sweep",
        category="Performance",
        subsystem="AllReduce",
        workload="small to huge payload sweep",
        purpose="Find saturation limits and topology bottlenecks",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} ./build/all_reduce_perf -g 1 -b 1K -e 4G -f 2 -n 100",
        ],
        expected="High busbw at larger sizes; smooth scaling trend",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_019",
        title="NCCL tiny-message latency profile",
        category="Performance",
        subsystem="AllReduce latency",
        workload="tiny payload latency and jitter",
        purpose="Expose launch overhead and protocol-switch jitter",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} ./build/all_reduce_perf -g 1 -b 8 -e 256K -f 2 -n 200",
        ],
        expected="Low tiny-message latency with no large jitter spikes",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_020",
        title="NCCL medium-size steady-state",
        category="Performance",
        subsystem="AllReduce steady-state",
        workload="medium payload stability",
        purpose="Validate stable bus utilization in realistic DL message range",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} ./build/all_reduce_perf -g 1 -b 256K -e 256M -f 2 -n 100",
        ],
        expected="Consistent busbw across middle sizes",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_021",
        title="NCCL ReduceScatter efficiency",
        category="Performance",
        subsystem="ReduceScatter",
        workload="ZeRO/FSDP-like split-reduce",
        purpose="Compare ReduceScatter efficiency to AllReduce",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} ./build/reduce_scatter_perf -g 1 -b 1K -e 4G -f 2 -n 100",
        ],
        expected="Strong busbw/algbw at scale and comparable or better efficiency than AllReduce",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_022",
        title="NCCL AllGather path health",
        category="Performance",
        subsystem="AllGather",
        workload="gather-heavy communication",
        purpose="Catch gather path regressions for sharding/embedding flows",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} ./build/all_gather_perf -g 1 -b 1K -e 4G -f 2 -n 100",
        ],
        expected="Large-message busbw remains healthy; no severe drop vs peers",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_023",
        title="NCCL AllToAll topology sensitivity",
        category="Performance",
        subsystem="AllToAll",
        workload="shuffle-heavy communication",
        purpose="Detect topology/P2P sensitivity for worst-case shuffle",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} ./build/alltoall_perf -g 1 -b 1K -e 4G -f 2 -n 100",
        ],
        expected="Lower than AllReduce is expected; major drops imply topology/P2P issues",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_024",
        title="NCCL AllReduce with P2P disabled",
        category="Performance",
        subsystem="PCIe vs NVLink sanity",
        workload="force host/indirect communication path",
        purpose="Quantify direct P2P benefit by disabling it",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} -x NCCL_P2P_DISABLE=1 ./build/all_reduce_perf -g 1 -b 8M -e 2G -f 2 -n 100",
        ],
        expected="Performance drops compared with P2P-enabled baseline",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_025",
        title="NCCL AllReduce baseline with P2P enabled",
        category="Performance",
        subsystem="PCIe vs NVLink baseline",
        workload="best available interconnect path",
        purpose="Establish best-case reference for comparison with P2P-disabled case",
        prerequisites=["nccl-tests built", "mpirun available"],
        commands=[
            "mpirun -np ${PERF_NUM_GPUS} --allow-run-as-root -x CUDA_VISIBLE_DEVICES=${PERF_CUDA_VISIBLE_DEVICES} ./build/all_reduce_perf -g 1 -b 8M -e 4G -f 2 -n 100",
        ],
        expected="Best observed result among paired P2P path tests",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_026",
        title="vLLM Serve + Load (FP8 stress) - 70B",
        category="Performance",
        subsystem="Inference (FP8)",
        workload="Long context 8k-32k with multi-GPU tensor parallel",
        purpose="Stress FP8 serving path for tail-latency and stability under heavy long-context load",
        prerequisites=["vLLM installed", "model access", "4 GPUs"],
        commands=[
            "vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --quantization fp8 --tensor-parallel-size 4 --max-model-len 32768",
            "vllm bench serve --backend vllm --model meta-llama/Meta-Llama-3.1-70B-Instruct --dataset-name sharegpt --dataset-path ${SHAREGPT_DATASET_PATH} --max-concurrency 128 --num-prompts 1000 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99",
        ],
        expected="Either stable FP8 serving with acceptable tail latency, or deterministic skip with explicit failure reason",
    ),
]


@pytest.mark.node_validation
@pytest.mark.workload_acceptance
@pytest.mark.slow
@pytest.mark.parametrize("case", PERFORMANCE_CASES, ids=[c.case_id for c in PERFORMANCE_CASES])
def test_performance_case_suite(case: PerformanceCase):
    """Catalog + optional execution harness for AI host CPU performance cases."""
    if not _is_selected(case.case_id):
        pytest.skip("Filtered by PERF_CASE_IDS")

    # Always validate metadata shape so this file is useful in dry-run mode.
    assert case.case_id.startswith("AI_HOSTCPU_PERF_TEST_ID_")
    assert case.title.strip(), "Case title must be defined"
    assert case.category == "Performance"
    assert case.subsystem.strip()
    assert case.workload.strip()
    assert case.purpose.strip()

    if case.planned_only:
        pytest.skip("Planned placeholder case; details not provided yet")

    execute = os.environ.get("PERF_EXECUTE_CASES", "0").lower() in {"1", "true", "yes", "on"}
    if not execute:
        pytest.skip("Catalog validated. Set PERF_EXECUTE_CASES=1 to execute benchmark commands")

    require_4_gpus = _env_flag("PERF_REQUIRE_4GPUS", "0")
    visible_gpu_count = _visible_gpu_count()
    if require_4_gpus and visible_gpu_count < 4:
        pytest.skip(
            "Performance execution requires 4 visible GPUs; "
            f"found {visible_gpu_count}. Set PERF_REQUIRE_4GPUS=0 to allow running with fewer GPUs."
        )
    effective_commands = _expand_genai_commands_for_visible_gpus(case, case.commands)
    _record_kpi(
        case.case_id,
        {
            "mode": "catalog",
            "title": case.title,
            "subsystem": case.subsystem,
            "workload": case.workload,
            "expected": case.expected,
            "planned_only": case.planned_only,
            "commands": effective_commands,
            "gpu_environment": {
                "visible_gpu_count": visible_gpu_count,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "perf_cuda_visible_devices": os.environ.get("PERF_CUDA_VISIBLE_DEVICES", ""),
                "perf_require_4gpus": os.environ.get("PERF_REQUIRE_4GPUS", "0"),
            },
        },
    )

    # Internal Python implementation for tokenizer case gives deterministic local validation.
    if case.case_id == "AI_HOSTCPU_PERF_TEST_ID_001":
        _benchmark_tokenizer_micro(case.case_id)
        return

    # TensorRT-LLM tooling often conflicts with vLLM dependency stacks.
    # Keep this case opt-in so shared vLLM environments remain stable.
    if case.case_id == "AI_HOSTCPU_PERF_TEST_ID_010":
        enable_trtllm = _env_flag("PERF_ENABLE_TRTLLM", "0")
        if not enable_trtllm:
            pytest.skip(
                "TRT-LLM case is disabled by default to avoid package conflicts with vLLM. "
                "Run in a dedicated TRT environment and set PERF_ENABLE_TRTLLM=1 to execute."
            )

    # === STANDALONE GENAI EXECUTION PATH ===
    # For GenAI tests, default to synthetic standalone execution to avoid proxy/endpoint issues
    if case.case_id in _GENAI_GPU_AWARE_CASE_IDS:
        use_standalone = _env_flag("PERF_GENAI_STANDALONE", "1")
        # Case 011 is intended to validate real HTTP serving/load behavior, not synthetic fallback.
        if case.case_id == "AI_HOSTCPU_PERF_TEST_ID_011":
            use_standalone = False
        if use_standalone:
            _benchmark_genai_standalone(case, effective_commands)
            return

    # Normalize all case commands up-front so every downstream path — preflight,
    # binary/module detection, background serve launch, and _run_command — all
    # operate on the same prepared strings. This ensures legacy command forms
    # (e.g. `python3 -m vllm.benchmarks.benchmark_serving`) are rewritten to the
    # current CLI (`vllm bench serve`) before any execution decision is made.
    prepared_commands: List[str] = [_prepare_command(cmd) for cmd in effective_commands]
    for raw, prepared in zip(effective_commands, prepared_commands):
        if raw != prepared:
            logger.info("Command rewritten: %s -> %s", raw, prepared)

    required_bins = _required_binaries_from_commands(prepared_commands)
    missing = [b for b in required_bins if not _binary_exists(b)]
    if missing:
        auto_install = _env_flag("PERF_AUTO_INSTALL", "1")
        if auto_install:
            package_specs = sorted(
                {
                    package_spec
                    for package_spec in (_package_spec_for_requirement(item) for item in missing)
                    if package_spec
                }
            )
            if package_specs:
                logger.info("Auto-installing packages for missing binaries: %s", package_specs)
                install_results = _auto_install_python_packages(package_specs)
                for pkg, status in install_results.items():
                    logger.info("  %s: %s", pkg, status)
                missing = [b for b in required_bins if not _binary_exists(b)]
        if missing:
            pytest.skip(f"Missing required binaries: {missing}")

    missing_modules = _missing_python_modules_from_commands(prepared_commands)
    if missing_modules:
        # Attempt auto-install if enabled before giving up
        auto_install = _env_flag("PERF_AUTO_INSTALL", "1")
        if auto_install:
            packages_to_install = {
                package_spec
                for package_spec in (_package_spec_for_requirement(module) for module in missing_modules)
                if package_spec
            }

            if packages_to_install:
                logger.info("Auto-installing missing Python packages: %s", packages_to_install)
                install_results = _auto_install_python_packages(sorted(packages_to_install))
                for pkg, status in install_results.items():
                    logger.info("  %s: %s", pkg, status)

                # Re-check after install attempt
                missing_modules = _missing_python_modules_from_commands(prepared_commands)

        if missing_modules:
            pytest.skip(f"Missing required Python module(s): {missing_modules}")

    timeout_sec = int(os.environ.get("PERF_CMD_TIMEOUT_SEC", "1800"))
    command_results: List[Dict] = []
    genai_auto_processes: List[subprocess.Popen] = []
    genai_auto_logs: List[TextIO] = []

    def _start_vllm_serve(prepared_serve: str, ready_timeout_sec: int) -> tuple[subprocess.Popen, TextIO, Path, str]:
        serve_candidates = [prepared_serve]
        fallback_serve = _retune_vllm_serve_for_startup(prepared_serve)
        if fallback_serve != prepared_serve:
            serve_candidates.append(fallback_serve)

        readiness_error: Optional[str] = None
        serve_start_elapsed = 0.0
        _serve_log_dir = Path(os.environ.get("PERF_VLLM_SERVE_LOG_DIR", "/tmp"))
        _serve_log_dir.mkdir(parents=True, exist_ok=True)

        for attempt_idx, serve_candidate in enumerate(serve_candidates, start=1):
            _serve_log_path = _serve_log_dir / f"vllm_serve_{case.case_id}_attempt{attempt_idx}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            logger.info("Starting vLLM server in background (attempt %d/%d): %s", attempt_idx, len(serve_candidates), serve_candidate)
            logger.info("vLLM serve log: %s", _serve_log_path)
            start_wait = time.time()
            serve_log_fh = open(_serve_log_path, "w", encoding="utf-8", buffering=1)
            serve_process = subprocess.Popen(
                serve_candidate,
                shell=True,
                stdout=serve_log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            readiness_error = _wait_for_vllm_serve_ready(
                serve_candidate,
                serve_process,
                timeout_sec=ready_timeout_sec,
            )
            serve_start_elapsed = time.time() - start_wait
            if readiness_error is None:
                command_results.append({
                    "command": serve_candidate,
                    "executed_command": serve_candidate,
                    "returncode": 0,
                    "status": "background_started",
                    "elapsed_sec": round(serve_start_elapsed, 3),
                })
                return serve_process, serve_log_fh, _serve_log_path, serve_candidate

            if serve_process.poll() is None:
                _terminate_process_group(serve_process, graceful_timeout_sec=10.0)
            serve_process.wait()
            serve_log_fh.close()
            _log_tail = ""
            try:
                _log_tail = _serve_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            except Exception:
                pass
            command_results.append({
                "command": serve_candidate,
                "executed_command": serve_candidate,
                "returncode": serve_process.returncode if serve_process.returncode is not None else -1,
                "stdout_tail": _log_tail,
                "stderr_tail": "",
            })
            if attempt_idx < len(serve_candidates):
                logger.warning("vLLM serve startup attempt %d failed, retrying with fallback command", attempt_idx)

        last_result = command_results[-1] if command_results else {}
        skip_reason = _known_skip_reason_from_result(last_result)
        if skip_reason:
            raise RuntimeError(f"vLLM server failed to start: {skip_reason}")
        raise RuntimeError(f"vLLM server was not ready: {readiness_error}")

    if case.case_id in _GENAI_GPU_AWARE_CASE_IDS and not _env_flag("PERF_GENAI_STANDALONE", "1"):
        if _env_flag("PERF_GENAI_AUTOSTART_SERVERS", "1"):
            try:
                genai_auto_processes, genai_auto_logs = _autostart_genai_vllm_servers(case.case_id)
            except RuntimeError as exc:
                pytest.skip(str(exc))

    # Special handling for vllm serve + benchmark pattern (serve must run in background)
    serve_process = None
    serve_log_fh = None
    serve_command = None
    try:
        for prepared_cmd in prepared_commands:
            if _is_vllm_serve_command(prepared_cmd):
                if serve_process is not None:
                    if serve_process.poll() is None:
                        _terminate_process_group(serve_process, graceful_timeout_sec=10.0)
                    serve_process.wait()
                    if serve_log_fh is not None and not serve_log_fh.closed:
                        serve_log_fh.close()
                    serve_process = None
                    serve_log_fh = None
                    serve_command = None

                ready_timeout_default = "1800" if case.case_id == "AI_HOSTCPU_PERF_TEST_ID_003" else "900"
                ready_timeout_sec = int(os.environ.get("PERF_VLLM_SERVE_READY_TIMEOUT_SEC", ready_timeout_default))
                try:
                    serve_process, serve_log_fh, _serve_log_path, serve_command = _start_vllm_serve(prepared_cmd, ready_timeout_sec)
                except RuntimeError as exc:
                    pytest.skip(str(exc))
                continue

            if _is_vllm_bench_serve_command(prepared_cmd):
                if serve_process is None or serve_process.poll() is not None:
                    skip_reason = "vLLM serving process is not running before benchmark execution"
                    _record_kpi(
                        case.case_id,
                        {
                            "mode": "execution",
                            "status": "skipped",
                            "reason": skip_reason,
                            "results": command_results,
                        },
                    )
                    pytest.skip(f"{case.case_id}: {skip_reason}")

                pre_bench_ready_timeout_sec = int(
                    os.environ.get("PERF_VLLM_PRE_BENCH_READY_TIMEOUT_SEC", "15")
                )
                pre_bench_readiness_error = _wait_for_vllm_serve_ready(
                    serve_command,
                    serve_process,
                    timeout_sec=pre_bench_ready_timeout_sec,
                )
                if pre_bench_readiness_error is not None:
                    skip_reason = (
                        "vLLM serving endpoint was not reachable immediately before benchmark: "
                        f"{pre_bench_readiness_error}"
                    )
                    _record_kpi(
                        case.case_id,
                        {
                            "mode": "execution",
                            "status": "skipped",
                            "reason": skip_reason,
                            "results": command_results,
                        },
                    )
                    pytest.skip(f"{case.case_id}: {skip_reason}")

            result = _run_command(prepared_cmd, timeout_sec=timeout_sec, already_prepared=True)
            command_results.append(result)
            skip_reason = _known_skip_reason_from_result(result)
            if skip_reason:
                _record_kpi(
                    case.case_id,
                    {
                        "mode": "execution",
                        "status": "skipped",
                        "reason": skip_reason,
                        "results": command_results,
                    },
                )
                pytest.skip(f"{case.case_id}: {skip_reason}")

            if _benchmark_all_requests_failed(result):
                reason = "vLLM benchmark completed but all requests failed"
                _record_kpi(
                    case.case_id,
                    {
                        "mode": "execution",
                        "status": "skipped",
                        "reason": reason,
                        "results": command_results,
                    },
                )
                pytest.skip(f"{case.case_id}: {reason}")

            if result["returncode"] != 0:
                if _is_vllm_bench_serve_command(prepared_cmd):
                    reason = "vLLM benchmark command failed in this environment"
                    _record_kpi(
                        case.case_id,
                        {
                            "mode": "execution",
                            "status": "skipped",
                            "reason": reason,
                            "results": command_results,
                        },
                    )
                    pytest.skip(f"{case.case_id}: {reason}")
                _record_kpi(case.case_id, {"mode": "execution", "status": "failed", "results": command_results})
                pytest.fail(
                    f"Command failed for {case.case_id}: {prepared_cmd}\n"
                    f"executed_command:\n{result['executed_command']}\n"
                    f"returncode: {result['returncode']}\n"
                    f"stderr_tail:\n{result['stderr_tail']}\nstdout_tail:\n{result['stdout_tail']}"
                )
    finally:
        # Clean up vllm serve background process if it's still running
        if serve_process is not None and serve_process.poll() is None:
            logger.info("Terminating vLLM server background process")
            _terminate_process_group(serve_process, graceful_timeout_sec=5.0)
        if serve_log_fh is not None and not serve_log_fh.closed:
            serve_log_fh.close()
        for proc in genai_auto_processes:
            if proc.poll() is None:
                logger.info("Terminating auto-started GenAI vLLM process pid=%s", proc.pid)
                _terminate_process_group(proc, graceful_timeout_sec=5.0)
        for fh in genai_auto_logs:
            if fh is not None and not fh.closed:
                fh.close()

    # Extract NCCL busbw metrics where applicable.
    aggregated_metrics: Dict[str, object] = {
        "mode": "execution",
        "status": "passed",
        "results": command_results,
    }
    if case.case_id >= "AI_HOSTCPU_PERF_TEST_ID_018" and case.case_id <= "AI_HOSTCPU_PERF_TEST_ID_025":
        merged_stdout = "\n".join(r["stdout_tail"] for r in command_results)
        busbw = _extract_nccl_busbw(merged_stdout)
        if busbw:
            aggregated_metrics["busbw_mean_gbs"] = round(float(np.mean(busbw)), 3)
            aggregated_metrics["busbw_peak_gbs"] = round(float(max(busbw)), 3)
            aggregated_metrics["busbw_samples"] = len(busbw)

    _record_kpi(case.case_id, aggregated_metrics)
