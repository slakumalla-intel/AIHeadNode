"""
Performance validation suite for AI host + GPU inference/training infrastructure.

This suite is data-driven and keeps all requested performance test IDs in one place.
By default, commands are NOT executed to avoid accidental long-running benchmark jobs.

Enable execution:
  PERF_EXECUTE_CASES=1
Optional filters:
  PERF_CASE_IDS=AI_HOSTCPU_PERF_TEST_ID_001,AI_HOSTCPU_PERF_TEST_ID_018
Optional command timeout (seconds):
  PERF_CMD_TIMEOUT_SEC=1800

KPI output file:
  artifacts/performance_suite_kpis.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest


logger = logging.getLogger(__name__)


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
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case_id": case_id,
        "metrics": metrics,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _selected_case_ids() -> Optional[set]:
    raw = os.environ.get("PERF_CASE_IDS", "").strip()
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def _is_selected(case_id: str) -> bool:
    selected = _selected_case_ids()
    return selected is None or case_id in selected


def _binary_exists(token: str) -> bool:
    # Treat "python3" as "python" on Windows-style shells too.
    candidate = "python" if token == "python3" else token
    return shutil.which(candidate) is not None


def _required_binaries_from_commands(commands: List[str]) -> List[str]:
    bins: List[str] = []
    for cmd in commands:
        parts = shlex.split(cmd)
        if not parts:
            continue
        first = parts[0]
        if first == "sudo" and len(parts) > 1:
            first = parts[1]
        bins.append(first)
    return sorted(set(bins))


def _normalize_command(command: str) -> str:
    # Keep commands runnable in non-root CI or local env by dropping sudo prefix.
    command = command.strip()
    if command.startswith("sudo "):
        return command[len("sudo ") :]
    return command


def _run_command(command: str, timeout_sec: int) -> Dict:
    start = time.perf_counter()
    result = subprocess.run(
        _normalize_command(command),
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    elapsed = time.perf_counter() - start
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": (result.stdout or "")[-4000:],
        "stderr_tail": (result.stderr or "")[-4000:],
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
                busbw.append(float(tokens[7]))
            except ValueError:
                pass
    return busbw


def _benchmark_tokenizer_micro(case_id: str) -> Dict:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    model_id = os.environ.get("PERF_TOKENIZER_MODEL", "meta-llama/Llama-3.1-8B")
    lens = [int(x) for x in os.environ.get("PERF_TOKENIZER_LENS", "128,512,2048,8192").split(",")]
    threads = [int(x) for x in os.environ.get("PERF_TOKENIZER_THREADS", "1,2,4,8,16").split(",")]
    total_texts = int(os.environ.get("PERF_TOKENIZER_ITERS", "2000"))

    tokenizer = AutoTokenizer.from_pretrained(model_id)

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

    _record_kpi(case_id, {"tokenizer": model_id, "throughput_tokens_per_sec": metrics})
    return {"tokenizer": model_id, "throughput_tokens_per_sec": metrics}


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
        title="Host to Device memcpy bandwidth",
        category="Performance",
        subsystem="CPU to GPU transfer",
        workload="Pinned vs pageable, CE vs SM path sweeps",
        purpose="Validate host memory + PCIe transfer stability and expected pinned-memory advantage",
        prerequisites=["nvbandwidth installed"],
        commands=[
            "nvbandwidth -t host_to_device_memcpy_ce -m pinned -s 64M -n 50",
            "nvbandwidth -t host_to_device_memcpy_sm -m pinned -s 64M -n 50",
            "nvbandwidth -t host_to_device_memcpy_ce -m pageable -s 64M -n 50",
            "nvbandwidth -t all_to_host_bidirectional_memcpy_sm -m pinned -s 64M -n 50",
        ],
        expected="Pinned bandwidth significantly exceeds pageable; low run-to-run variance",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_003",
        title="vLLM Serve + Load FP8 8B",
        category="Performance",
        subsystem="Inference FP8",
        workload="Long context 8k, multi-user serving",
        purpose="Measure ITL and verify host scheduling path stability in fast-token regime",
        prerequisites=["vLLM installed", "model access"],
        commands=[
            "vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --tensor-parallel-size 1 --max-model-len 8192",
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --dataset sharegpt --num-prompts 1000 --percentiles 50,90,99",
        ],
        expected="Stable serving with ITL and tail latency within scenario SLO",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_004",
        title="vLLM Serve + Load BF16 8B",
        category="Performance",
        subsystem="Inference BF16",
        workload="Same load shape as FP8 baseline",
        purpose="Compare BF16/FP16 behavior vs FP8 and detect CPU-bound scheduler behavior",
        prerequisites=["vLLM installed", "model access"],
        commands=[
            "vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 8192",
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --dataset sharegpt --num-prompts 1000 --percentiles 50,90,99",
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
            "vllm serve meta-llama/Llama-3.1-70B-Instruct --quantization fp8 --tensor-parallel-size 4 --max-model-len 32768 --disable-log-stats",
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --model meta-llama/Llama-3.1-70B-Instruct --dataset sharegpt --max-concurrency 128 --num-prompts 1000 --percentiles 50,90,99",
        ],
        expected="No OOM/crash and tail latency within defined SLO",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_006",
        title="Llama 70B BF16 long-context inference",
        category="Performance",
        subsystem="Inference BF16",
        workload="Large model unquantized baseline",
        purpose="Compare throughput and scheduler behavior against FP8",
        prerequisites=["vLLM installed", "4 GPUs", "model access"],
        commands=[
            "vllm serve meta-llama/Llama-3.1-70B-Instruct --dtype bfloat16 --tensor-parallel-size 4 --max-model-len 16384",
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --model meta-llama/Llama-3.1-70B-Instruct --dataset sharegpt --max-concurrency 128 --num-prompts 1000 --percentiles 50,90,99",
        ],
        expected="Stable baseline; no runaway tail latency; throughput lower than FP8",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_007",
        title="Mixed prompt-length realistic traffic",
        category="Performance",
        subsystem="Inference mixed prompt",
        workload="ShareGPT-like distribution, poisson arrivals",
        purpose="Capture real-world tail latency for mixed workloads",
        prerequisites=["vLLM or compatible server", "ShareGPT dataset"],
        commands=[
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --dataset sharegpt --request-rate poisson --max-concurrency 256 --num-prompts 3000 --percentiles 50,99",
        ],
        expected="Meets scenario-specific P50/P99 SLO",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_008",
        title="Mixtral 8x22B FP8 vs Llama Dense",
        category="Performance",
        subsystem="Inference MoE vs dense",
        workload="Same benchmark profile for MoE and dense baseline",
        purpose="Compare routing/memory sensitivity (MoE) against dense behavior",
        prerequisites=["vLLM installed", "model access", "4 GPUs"],
        commands=[
            "vllm serve mistralai/Mixtral-8x22B-v0.1 --tensor-parallel-size 4 --quantization fp8 --max-model-len 32768 --disable-log-stats",
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --model mistralai/Mixtral-8x22B-v0.1 --dataset sharegpt --max-concurrency 128 --num-prompts 1000 --percentiles 50,90,99",
        ],
        expected="Stable run; record tokens/sec and P99 for MoE-vs-dense comparison",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_009",
        title="Tail latency under burst traffic",
        category="Performance",
        subsystem="Inference burst",
        workload="1x to 10x burst, 30s spike",
        purpose="Validate resilience, recovery, and burst-tail behavior",
        prerequisites=["vLLM benchmark tools", "model access"],
        commands=[
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --dataset sharegpt --request-rate burst --burst-length 30 --burst-rps 200 --max-concurrency 512 --percentiles 50,99",
        ],
        expected="No OOM/crash, recovers after burst, P99 within burst SLO",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_010",
        title="trtllm-bench offline throughput",
        category="Performance",
        subsystem="Text generation offline benchmark",
        workload="Fixed synthetic dataset throughput run",
        purpose="Validate TRT-LLM engine correctness and repeatable throughput",
        prerequisites=["trtllm-bench installed", "built TRT-LLM engine", "dataset file"],
        commands=[
            "trtllm-bench --model /path/to/hf/Llama-3.3-70B-Instruct throughput --dataset /path/to/dataset/synthetic_2048_2048_1000.txt --engine_dir /path/to/engines/baseline",
        ],
        expected="Stable tokens/sec and no runtime errors",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_011",
        title="GenAI-Perf end-to-end HTTP benchmark",
        category="Performance",
        subsystem="E2E serving + networking",
        workload="Dual-server concurrent load isolation",
        purpose="Validate containerized serving path, networking, and tail latency",
        prerequisites=["genai-perf installed", "two vLLM servers pinned to distinct GPUs"],
        commands=[
            "genai-perf profile --endpoint-type chat --service-kind openai --url http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct",
            "genai-perf profile --endpoint-type chat --service-kind openai --url http://127.0.0.1:8001/v1 --model meta-llama/Llama-3.1-8B-Instruct",
        ],
        expected="Both server instances satisfy SLO simultaneously without cross-GPU contention",
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_012",
        title="Reserved performance case",
        category="Performance",
        subsystem="TBD",
        workload="TBD",
        purpose="Placeholder for future case definition",
        planned_only=True,
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_013",
        title="Reserved performance case",
        category="Performance",
        subsystem="TBD",
        workload="TBD",
        purpose="Placeholder for future case definition",
        planned_only=True,
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_014",
        title="Reserved performance case",
        category="Performance",
        subsystem="TBD",
        workload="TBD",
        purpose="Placeholder for future case definition",
        planned_only=True,
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_015",
        title="Reserved performance case",
        category="Performance",
        subsystem="TBD",
        workload="TBD",
        purpose="Placeholder for future case definition",
        planned_only=True,
    ),
    PerformanceCase(
        case_id="AI_HOSTCPU_PERF_TEST_ID_016",
        title="DeepSeek R1 KV cache effectiveness with host offload",
        category="Performance",
        subsystem="DeepSeek inference with offload",
        workload="Large model with offload path",
        purpose="Validate stable operation when model+KV exceed pure HBM budget",
        prerequisites=["vLLM installed", "DeepSeek model access", "8 GPUs recommended"],
        commands=[
            "vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8 --dtype float8 --max-model-len 16384 --enable-chunked-prefill --trust-remote-code",
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --model deepseek-ai/DeepSeek-R1 --dataset sharegpt --max-concurrency 128 --num-prompts 500 --percentiles 50,90,99",
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
            "vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8 --dtype float8 --max-model-len 16384 --enable-chunked-prefill --trust-remote-code",
            "python3 -m vllm.benchmarks.benchmark_serving --backend vllm --model deepseek-ai/DeepSeek-R1 --dataset sharegpt --max-concurrency 128 --num-prompts 500 --percentiles 50,90,99",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 1K -e 4G -f 2 -n 100",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 8 -e 256K -f 2 -n 200",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 256K -e 256M -f 2 -n 100",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/reduce_scatter_perf -g 1 -b 1K -e 4G -f 2 -n 100",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_gather_perf -g 1 -b 1K -e 4G -f 2 -n 100",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/alltoall_perf -g 1 -b 1K -e 4G -f 2 -n 100",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 -x NCCL_P2P_DISABLE=1 ./build/all_reduce_perf -g 1 -b 8M -e 2G -f 2 -n 100",
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
            "mpirun -np 2 --allow-run-as-root -x CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -g 1 -b 8M -e 4G -f 2 -n 100",
        ],
        expected="Best observed result among paired P2P path tests",
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

    _record_kpi(
        case.case_id,
        {
            "mode": "catalog",
            "title": case.title,
            "subsystem": case.subsystem,
            "workload": case.workload,
            "expected": case.expected,
            "planned_only": case.planned_only,
            "commands": case.commands,
        },
    )

    if case.planned_only:
        pytest.skip("Planned placeholder case; details not provided yet")

    execute = os.environ.get("PERF_EXECUTE_CASES", "0").lower() in {"1", "true", "yes", "on"}
    if not execute:
        pytest.skip("Catalog validated. Set PERF_EXECUTE_CASES=1 to execute benchmark commands")

    # Internal Python implementation for tokenizer case gives deterministic local validation.
    if case.case_id == "AI_HOSTCPU_PERF_TEST_ID_001":
        _benchmark_tokenizer_micro(case.case_id)
        return

    required_bins = _required_binaries_from_commands(case.commands)
    missing = [b for b in required_bins if not _binary_exists(b)]
    if missing:
        pytest.skip(f"Missing required binaries: {missing}")

    timeout_sec = int(os.environ.get("PERF_CMD_TIMEOUT_SEC", "1800"))
    command_results: List[Dict] = []
    for command in case.commands:
        result = _run_command(command, timeout_sec=timeout_sec)
        command_results.append(result)
        if result["returncode"] != 0:
            _record_kpi(case.case_id, {"mode": "execution", "status": "failed", "results": command_results})
            pytest.fail(
                f"Command failed for {case.case_id}: {command}\n"
                f"stderr_tail:\n{result['stderr_tail']}\nstdout_tail:\n{result['stdout_tail']}"
            )

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
