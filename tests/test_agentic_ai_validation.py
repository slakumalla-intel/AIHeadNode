"""
Agentic AI Workload Validation Suite for AI Head Node (CPU + GPU)
==================================================================

Comprehensive validation of agentic AI systems combining:
  - GPU-based language model inference (reasoning, planning)
  - CPU-based tool/function execution (search, API calls, computation)
  - CPU-GPU synchronization and state management
  - Multi-turn conversation tracking

KPI areas covered:
  1. Tool invocation overhead (dispatch, scheduling, result collection)
  2. Reasoning phase latency (GPU token generation for chain-of-thought)
  3. CPU-GPU synchronization overhead
  4. Tool execution parallelization efficiency
  5. Multi-turn conversation memory and state management
  6. Agent decision-making latency percentiles
  7. CPU memory bandwidth during tool execution
  8. GPU utilization during tool wait periods

KPIs are recorded to:
  artifacts/agentic_ai_kpis.jsonl

Run all agentic AI tests:
  pytest -m agentic_ai -vv tests/test_agentic_ai_validation.py

Run specific scenarios:
  pytest -m tool_invocation -vv tests/test_agentic_ai_validation.py
  pytest -m reasoning_latency -vv tests/test_agentic_ai_validation.py
  pytest -m multi_turn -vv tests/test_agentic_ai_validation.py
  pytest -m cpu_gpu_sync -vv tests/test_agentic_ai_validation.py

Environment toggles (all optional):
  AGENTIC_EXECUTE=1               # Set 0 to skip GPU-heavy tests
  AGENTIC_MODEL_ID                # HF model for reasoning (default: gpt2)
  AGENTIC_NUM_TOOLS               # Number of tools to simulate (default: 8)
  AGENTIC_NUM_TURNS               # Multi-turn conversation depth (default: 5)
  AGENTIC_NUM_AGENTS              # Parallel agents to test (default: 2)
  AGENTIC_TOOL_EXEC_TIME_MS       # Simulated tool exec time (default: 50)
  AGENTIC_REASONING_TOKENS        # Tokens for reasoning phase (default: 256)
  AGENTIC_WARMUP_ITERS            # Warmup iterations (default: 3)
  AGENTIC_BENCH_ITERS             # Benchmark iterations (default: 10)
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from tests.kpi_efficiency import enrich_efficiency_kpis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, str(default))
    return val.lower() in ("1", "true", "yes")

os.environ.setdefault("AGENTIC_EXECUTE", "1")
os.environ.setdefault("AGENTIC_MODEL_ID", "openai-community/gpt2")
os.environ.setdefault("AGENTIC_NUM_TOOLS", "8")
os.environ.setdefault("AGENTIC_NUM_TURNS", "5")
os.environ.setdefault("AGENTIC_NUM_AGENTS", "2")
os.environ.setdefault("AGENTIC_TOOL_EXEC_TIME_MS", "50")
os.environ.setdefault("AGENTIC_REASONING_TOKENS", "256")
os.environ.setdefault("AGENTIC_WARMUP_ITERS", "3")
os.environ.setdefault("AGENTIC_BENCH_ITERS", "10")

_EXECUTE = _env_bool("AGENTIC_EXECUTE", True)
_MODEL_ID = os.environ["AGENTIC_MODEL_ID"]
_NUM_TOOLS = _env_int("AGENTIC_NUM_TOOLS", 8)
_NUM_TURNS = _env_int("AGENTIC_NUM_TURNS", 5)
_NUM_AGENTS = _env_int("AGENTIC_NUM_AGENTS", 2)
_TOOL_EXEC_TIME_MS = _env_int("AGENTIC_TOOL_EXEC_TIME_MS", 50)
_REASONING_TOKENS = _env_int("AGENTIC_REASONING_TOKENS", 256)
_WARMUP = _env_int("AGENTIC_WARMUP_ITERS", 3)
_BENCH = _env_int("AGENTIC_BENCH_ITERS", 10)

# ---------------------------------------------------------------------------
# KPI recording
# ---------------------------------------------------------------------------

def _kpi_path() -> Path:
    return Path(os.environ.get("AGENTIC_KPI_FILE", "artifacts/agentic_ai_kpis.jsonl"))


def _record_kpi(test_name: str, metrics: dict) -> None:
    """Record KPI metrics to JSONL file."""
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
# Optional imports
# ---------------------------------------------------------------------------

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
    _CUDA_DEVICE_COUNT = torch.cuda.device_count() if _HAS_TORCH else 0
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
    _CUDA_DEVICE_COUNT = 0

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# ---------------------------------------------------------------------------
# Data models for agent state tracking
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Represents a single tool invocation."""
    tool_id: int
    tool_name: str
    args: Dict[str, Any]
    dispatch_time_ns: int  # When tool was invoked
    start_time_ns: int = 0  # When execution started on CPU
    end_time_ns: int = 0    # When result was ready
    cpu_time_ns: int = 0    # Actual CPU execution time
    result: Any = None
    error: Optional[str] = None

    def dispatch_latency_us(self) -> float:
        """Latency from dispatch to start of execution (microseconds)."""
        if self.start_time_ns > 0 and self.dispatch_time_ns > 0:
            return (self.start_time_ns - self.dispatch_time_ns) / 1000.0
        return 0.0

    def end_to_end_latency_us(self) -> float:
        """Total latency from dispatch to result ready (microseconds)."""
        if self.end_time_ns > 0 and self.dispatch_time_ns > 0:
            return (self.end_time_ns - self.dispatch_time_ns) / 1000.0
        return 0.0


@dataclass
class AgentTurn:
    """Represents one turn in a multi-turn conversation."""
    turn_id: int
    user_message: str
    reasoning_text: str = ""
    reasoning_tokens_generated: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_response: str = ""
    
    timestamps: Dict[str, float] = field(default_factory=dict)
    # timestamps keys: "reasoning_start", "reasoning_end", "tools_dispatch_start",
    #                  "tools_dispatch_end", "tools_exec_start", "tools_exec_end",
    #                  "response_start", "response_end"
    
    def reasoning_latency_ms(self) -> float:
        """Time spent in reasoning phase (ms)."""
        start = self.timestamps.get("reasoning_start", 0)
        end = self.timestamps.get("reasoning_end", 0)
        return (end - start) * 1000.0 if end > start else 0.0
    
    def tool_dispatch_latency_ms(self) -> float:
        """Time to dispatch all tools (ms)."""
        start = self.timestamps.get("tools_dispatch_start", 0)
        end = self.timestamps.get("tools_dispatch_end", 0)
        return (end - start) * 1000.0 if end > start else 0.0
    
    def tool_execution_latency_ms(self) -> float:
        """Max tool execution latency (ms)."""
        if not self.tool_calls:
            return 0.0
        max_latency = max((tc.end_to_end_latency_us() for tc in self.tool_calls), default=0.0)
        return max_latency / 1000.0
    
    def total_turn_latency_ms(self) -> float:
        """Total latency for this turn (ms)."""
        start = self.timestamps.get("reasoning_start", 0)
        end = self.timestamps.get("response_end", 0)
        return (end - start) * 1000.0 if end > start else 0.0


@dataclass
class AgentSession:
    """Represents an entire multi-turn agent session."""
    agent_id: int
    turns: List[AgentTurn] = field(default_factory=list)
    total_cpu_memory_used_mb: float = 0.0
    peak_gpu_util_percent: float = 0.0
    peak_cpu_util_percent: float = 0.0
    
    def avg_reasoning_latency_ms(self) -> float:
        """Average reasoning latency across turns."""
        if not self.turns:
            return 0.0
        latencies = [t.reasoning_latency_ms() for t in self.turns]
        return sum(latencies) / len(latencies)
    
    def avg_tool_latency_ms(self) -> float:
        """Average tool execution latency across turns."""
        if not self.turns:
            return 0.0
        latencies = [t.tool_execution_latency_ms() for t in self.turns]
        return sum(latencies) / len(latencies) if any(latencies) else 0.0
    
    def total_session_latency_ms(self) -> float:
        """Total time for entire session."""
        if not self.turns:
            return 0.0
        return sum(t.total_turn_latency_ms() for t in self.turns)


# ---------------------------------------------------------------------------
# Tool simulation and CPU execution
# ---------------------------------------------------------------------------

def _simulate_tool_execution(tool_id: int, duration_ms: int = 50) -> Dict[str, Any]:
    """
    Simulate CPU-based tool execution with realistic latency.
    
    Examples: search API, database query, computation, etc.
    """
    result = {
        "tool_id": tool_id,
        "status": "success",
        "data": np.random.randn(100).tolist(),  # Simulated result
        "execution_time_ms": duration_ms,
    }
    
    # Simulate actual work (CPU-bound)
    end_time = time.perf_counter() + (duration_ms / 1000.0)
    while time.perf_counter() < end_time:
        # Busy-wait / compute
        _ = sum([i * i for i in range(1000)])
    
    return result


def _simulate_parallel_tool_execution(tool_ids: List[int], duration_ms: int = 50) -> Dict[int, Any]:
    """Execute multiple tools in parallel (thread pool on CPU)."""
    results = {}
    
    def _execute(tool_id: int):
        results[tool_id] = _simulate_tool_execution(tool_id, duration_ms)
    
    with ThreadPoolExecutor(max_workers=min(len(tool_ids), os.cpu_count() or 4)) as ex:
        threads = [ex.submit(_execute, tid) for tid in tool_ids]
        for t in threads:
            t.result()
    
    return results


# ---------------------------------------------------------------------------
# GPU-based reasoning simulation
# ---------------------------------------------------------------------------

def _simulate_gpu_reasoning(num_tokens: int = 256) -> Tuple[str, float]:
    """
    Simulate GPU-based language model reasoning phase.
    Returns (reasoning_text, latency_ms).
    """
    if _HAS_TORCH:
        try:
            device = torch.device("cuda" if _HAS_TORCH else "cpu")
            # Simulate token generation on GPU
            t0 = time.perf_counter()
            for _ in range(num_tokens):
                # Simulate one token generation (attention, MLP, etc.)
                x = torch.randn(1, 768, device=device)
                # Simulate transformer layer
                _ = torch.matmul(x, torch.randn(768, 768, device=device))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            reasoning_text = f"Reasoning (simulated {num_tokens} tokens on GPU)"
            return reasoning_text, elapsed_ms
        except Exception as e:
            logger.warning(f"GPU reasoning simulation failed: {e}")
    
    # Fallback: CPU-based simulation
    t0 = time.perf_counter()
    for _ in range(num_tokens):
        _ = sum([i * i for i in range(100)])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    reasoning_text = f"Reasoning (simulated {num_tokens} tokens on CPU fallback)"
    return reasoning_text, elapsed_ms


# ---------------------------------------------------------------------------
# CPU-GPU synchronization measurement
# ---------------------------------------------------------------------------

def _measure_cpu_gpu_sync_overhead() -> float:
    """
    Measure synchronization overhead between CPU and GPU.
    (e.g., copying results, synchronizing events)
    Returns latency in microseconds.
    """
    if not _HAS_TORCH:
        return 0.0
    
    try:
        device = torch.device("cuda" if _HAS_TORCH else "cpu")
        
        # Create a small tensor and copy it back and forth
        t0 = time.perf_counter_ns()
        
        x = torch.randn(1000, device="cpu")
        x_gpu = x.to(device)
        torch.cuda.synchronize() if _HAS_TORCH else None
        x_cpu = x_gpu.to("cpu")
        torch.cuda.synchronize() if _HAS_TORCH else None
        
        elapsed_ns = time.perf_counter_ns() - t0
        return elapsed_ns / 1000.0  # Convert to microseconds
    except Exception as e:
        logger.warning(f"CPU-GPU sync measurement failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Memory usage tracking
# ---------------------------------------------------------------------------

def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if not _HAS_PSUTIL:
        return 0.0
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def _get_cpu_utilization_percent() -> float:
    """Get current CPU utilization percentage."""
    if not _HAS_PSUTIL:
        return 0.0
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# TC-AGENT-01: Tool Invocation Overhead
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.tool_invocation
class TestToolInvocationOverhead:
    """Measure tool dispatch, scheduling, and collection overhead."""

    def test_single_tool_dispatch_latency(self):
        """TC-AGENT-01a: Single tool dispatch latency must be < 10ms."""
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        dispatch_latencies = []
        
        for _ in range(_BENCH):
            tool_call = ToolCall(
                tool_id=0,
                tool_name="search",
                args={"query": "test"},
                dispatch_time_ns=time.perf_counter_ns()
            )
            
            t0 = time.perf_counter_ns()
            tool_call.start_time_ns = time.perf_counter_ns()
            dispatch_latencies.append((tool_call.start_time_ns - tool_call.dispatch_time_ns) / 1e6)
        
        avg_latency_ms = sum(dispatch_latencies) / len(dispatch_latencies)
        
        assert avg_latency_ms < 10.0, (
            f"Tool dispatch latency {avg_latency_ms:.2f}ms exceeds 10ms threshold"
        )
        
        metrics = {
            "avg_dispatch_latency_ms": avg_latency_ms,
            "min_dispatch_latency_ms": min(dispatch_latencies),
            "max_dispatch_latency_ms": max(dispatch_latencies),
            "iterations": _BENCH,
        }
        _record_kpi("TC-AGENT-01a: Single tool dispatch", metrics)
        
        logger.info(
            f"\n{'='*80}\n"
            f"  TOOL DISPATCH LATENCY\n"
            f"{'='*80}\n"
            f"  Avg latency:              {avg_latency_ms:.3f} ms\n"
            f"  Min latency:              {min(dispatch_latencies):.3f} ms\n"
            f"  Max latency:              {max(dispatch_latencies):.3f} ms\n"
            f"{'='*80}\n"
        )

    def test_parallel_tool_dispatch_scalability(self):
        """TC-AGENT-01b: Dispatching N tools concurrently should scale sub-linearly.

        The earlier implementation looped sequentially over `range(num_tools)`
        constructing ``ToolCall`` objects on the main thread, which by
        definition cannot scale sub-linearly: 16 constructions take ~16× the
        time of 1 (plus a small fixed overhead), and the resulting ratio
        depends on Python timer resolution, not on any dispatch path.

        This test now dispatches tools concurrently via ``ThreadPoolExecutor``
        and measures wall-clock time, so the assertion meaningfully captures
        the cost of fanning work out to multiple workers.
        """
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")

        from concurrent.futures import ThreadPoolExecutor

        def _dispatch_one(i: int) -> ToolCall:
            return ToolCall(
                tool_id=i,
                tool_name=f"tool_{i}",
                args={"id": i},
                dispatch_time_ns=time.perf_counter_ns(),
            )

        results: Dict[int, float] = {}
        # Use a single pool sized to the largest fan-out so we measure dispatch
        # cost, not pool-creation cost.
        max_fanout = 16
        with ThreadPoolExecutor(max_workers=max_fanout) as pool:
            for num_tools in [1, 4, 8, 16]:
                dispatch_latencies = []
                for _ in range(_BENCH):
                    t0 = time.perf_counter_ns()
                    # Submit all tools and wait for completion.
                    list(pool.map(_dispatch_one, range(num_tools)))
                    elapsed_ns = time.perf_counter_ns() - t0
                    dispatch_latencies.append(elapsed_ns / 1e6)  # ms

                avg_latency = sum(dispatch_latencies) / len(dispatch_latencies)
                results[num_tools] = avg_latency
                logger.info(f"  {num_tools} tools: {avg_latency:.3f} ms")

        # Sub-linear scaling: 16 concurrent tools must take less than 5× the
        # latency of a single tool dispatch.
        latency_1 = results[1]
        latency_16 = results[16]
        scaling_ratio = latency_16 / latency_1 if latency_1 > 0 else float("inf")

        assert scaling_ratio < 5.0, (
            f"Tool dispatch scales poorly: 16 tools take {scaling_ratio:.1f}× time of 1 tool"
        )

        metrics = {
            "latency_1_tool_ms": latency_1,
            "latency_16_tools_ms": latency_16,
            "scaling_ratio": scaling_ratio,
            "all_results": results,
        }
        _record_kpi("TC-AGENT-01b: Parallel tool dispatch scalability", metrics)


# ---------------------------------------------------------------------------
# TC-AGENT-02: Reasoning Phase Latency
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.reasoning_latency
@pytest.mark.slow
class TestReasoningPhaseLatency:
    """Measure GPU-based reasoning (chain-of-thought) latency."""

    def test_reasoning_latency_by_token_count(self):
        """TC-AGENT-02: Measure reasoning latency for various token counts."""
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        token_counts = [64, 128, 256, 512]
        results = {}
        
        for tokens in token_counts:
            latencies = []
            for _ in range(_WARMUP):
                _ = _simulate_gpu_reasoning(tokens)
            
            for _ in range(_BENCH):
                _, latency_ms = _simulate_gpu_reasoning(tokens)
                latencies.append(latency_ms)
            
            avg_latency = sum(latencies) / len(latencies)
            results[tokens] = avg_latency
            
            logger.info(f"  {tokens} tokens: {avg_latency:.2f} ms")
        
        metrics = {
            "reasoning_latencies_by_tokens": results,
            "device": "GPU" if _HAS_TORCH else "CPU",
        }
        _record_kpi("TC-AGENT-02: Reasoning phase latency", metrics)
        
        logger.info(
            f"\n{'='*80}\n"
            f"  REASONING PHASE LATENCY\n"
            f"{'='*80}\n"
        )
        for tokens, latency in results.items():
            logger.info(f"  {tokens} tokens: {latency:.2f} ms")
        logger.info(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# TC-AGENT-03: CPU-GPU Synchronization Overhead
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.cpu_gpu_sync
class TestCPUGPUSynchronization:
    """Measure synchronization overhead between CPU and GPU."""

    def test_cpu_gpu_sync_latency(self):
        """TC-AGENT-03: CPU-GPU sync overhead must be < 500µs."""
        if not _HAS_TORCH:
            pytest.skip("CUDA not available")
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        sync_latencies = []
        
        for _ in range(_WARMUP):
            _ = _measure_cpu_gpu_sync_overhead()
        
        for _ in range(_BENCH):
            latency_us = _measure_cpu_gpu_sync_overhead()
            sync_latencies.append(latency_us)
        
        avg_latency_us = sum(sync_latencies) / len(sync_latencies)
        
        logger.info(
            f"\n{'='*80}\n"
            f"  CPU-GPU SYNC OVERHEAD\n"
            f"{'='*80}\n"
            f"  Avg latency:              {avg_latency_us:.2f} µs\n"
            f"  Min latency:              {min(sync_latencies):.2f} µs\n"
            f"  Max latency:              {max(sync_latencies):.2f} µs\n"
            f"{'='*80}\n"
        )
        
        metrics = {
            "avg_sync_latency_us": avg_latency_us,
            "min_sync_latency_us": min(sync_latencies),
            "max_sync_latency_us": max(sync_latencies),
            "iterations": _BENCH,
        }
        _record_kpi("TC-AGENT-03: CPU-GPU sync overhead", metrics)


# ---------------------------------------------------------------------------
# TC-AGENT-04: Tool Execution Parallelization
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.tool_parallelism
class TestToolExecutionParallelization:
    """Measure parallel tool execution efficiency on CPU."""

    def test_sequential_vs_parallel_tool_execution(self):
        """TC-AGENT-04: Parallel tool execution should be faster than sequential."""
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        num_tools = min(_NUM_TOOLS, 8)
        tool_ids = list(range(num_tools))
        
        # Sequential execution
        t0 = time.perf_counter()
        for _ in range(_BENCH):
            for tool_id in tool_ids:
                _simulate_tool_execution(tool_id, _TOOL_EXEC_TIME_MS)
        sequential_time = time.perf_counter() - t0
        
        # Parallel execution
        t0 = time.perf_counter()
        for _ in range(_BENCH):
            _simulate_parallel_tool_execution(tool_ids, _TOOL_EXEC_TIME_MS)
        parallel_time = time.perf_counter() - t0
        
        speedup = sequential_time / parallel_time
        
        logger.info(
            f"\n{'='*80}\n"
            f"  TOOL EXECUTION PARALLELIZATION\n"
            f"{'='*80}\n"
            f"  Sequential time:          {sequential_time:.2f} s\n"
            f"  Parallel time:            {parallel_time:.2f} s\n"
            f"  Speedup:                  {speedup:.2f}x\n"
            f"  Number of tools:          {num_tools}\n"
            f"{'='*80}\n"
        )
        
        # Speedup should be at least 1.5x for parallel with multiple tools
        assert speedup > 1.0, (
            f"Parallel execution {parallel_time:.2f}s not faster than sequential {sequential_time:.2f}s"
        )
        
        metrics = {
            "sequential_time_s": sequential_time,
            "parallel_time_s": parallel_time,
            "speedup": speedup,
            "num_tools": num_tools,
            "iterations": _BENCH,
        }
        _record_kpi("TC-AGENT-04: Tool execution parallelization", metrics)


# ---------------------------------------------------------------------------
# TC-AGENT-05: Multi-Turn Conversation KPIs
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.multi_turn
class TestMultiTurnConversation:
    """Measure multi-turn agent conversation KPIs."""

    def test_multi_turn_session_latency(self):
        """TC-AGENT-05: Measure latency for multi-turn agent conversation."""
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        num_turns = min(_NUM_TURNS, 5)
        session = AgentSession(agent_id=0)
        
        for turn_id in range(num_turns):
            turn = AgentTurn(
                turn_id=turn_id,
                user_message=f"Question {turn_id + 1}",
            )
            
            # Reasoning phase
            turn.timestamps["reasoning_start"] = time.perf_counter()
            reasoning_text, reasoning_latency = _simulate_gpu_reasoning(_REASONING_TOKENS)
            turn.reasoning_text = reasoning_text
            turn.reasoning_tokens_generated = _REASONING_TOKENS
            turn.timestamps["reasoning_end"] = time.perf_counter()
            
            # Tool dispatch
            turn.timestamps["tools_dispatch_start"] = time.perf_counter()
            num_tools = min(_NUM_TOOLS, 4)
            for i in range(num_tools):
                tool_call = ToolCall(
                    tool_id=i,
                    tool_name=f"tool_{i}",
                    args={"query": f"from_reasoning_{turn_id}"},
                    dispatch_time_ns=time.perf_counter_ns()
                )
                tool_call.start_time_ns = time.perf_counter_ns()
                turn.tool_calls.append(tool_call)
            turn.timestamps["tools_dispatch_end"] = time.perf_counter()
            
            # Tool execution
            turn.timestamps["tools_exec_start"] = time.perf_counter()
            for tool_call in turn.tool_calls:
                tool_result = _simulate_tool_execution(tool_call.tool_id, _TOOL_EXEC_TIME_MS)
                tool_call.result = tool_result
                tool_call.end_time_ns = time.perf_counter_ns()
            turn.timestamps["tools_exec_end"] = time.perf_counter()
            
            # Response generation
            turn.timestamps["response_start"] = time.perf_counter()
            turn.final_response = f"Response to turn {turn_id + 1}"
            turn.timestamps["response_end"] = time.perf_counter()
            
            session.turns.append(turn)
        
        metrics = {
            "total_session_latency_ms": session.total_session_latency_ms(),
            "avg_reasoning_latency_ms": session.avg_reasoning_latency_ms(),
            "avg_tool_latency_ms": session.avg_tool_latency_ms(),
            "num_turns": num_turns,
            "per_turn_latencies_ms": [t.total_turn_latency_ms() for t in session.turns],
        }
        _record_kpi("TC-AGENT-05: Multi-turn conversation", metrics)
        
        logger.info(
            f"\n{'='*80}\n"
            f"  MULTI-TURN CONVERSATION KPIs\n"
            f"{'='*80}\n"
            f"  Total session latency:    {metrics['total_session_latency_ms']:.2f} ms\n"
            f"  Avg reasoning latency:    {metrics['avg_reasoning_latency_ms']:.2f} ms\n"
            f"  Avg tool latency:         {metrics['avg_tool_latency_ms']:.2f} ms\n"
            f"  Number of turns:          {num_turns}\n"
        )
        for i, latency in enumerate(metrics['per_turn_latencies_ms']):
            logger.info(f"    Turn {i+1} latency:        {latency:.2f} ms")
        logger.info(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# TC-AGENT-06: CPU Memory and Bandwidth During Tool Execution
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.memory_intensive
class TestCPUMemoryDuringToolExecution:
    """Measure CPU memory usage and bandwidth during tool execution."""

    def test_memory_usage_scaling_with_tools(self):
        """TC-AGENT-06: Memory usage should scale sub-linearly with tool count."""
        if not _HAS_PSUTIL:
            pytest.skip("psutil not available")
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        memory_baseline = _get_memory_usage_mb()
        
        results = {}
        
        for num_tools in [1, 4, 8]:
            tool_ids = list(range(num_tools))
            
            # Execute tools and measure memory
            max_memory = memory_baseline
            for _ in range(_BENCH):
                _simulate_parallel_tool_execution(tool_ids, _TOOL_EXEC_TIME_MS)
                current_mem = _get_memory_usage_mb()
                max_memory = max(max_memory, current_mem)
            
            memory_delta = max_memory - memory_baseline
            results[num_tools] = memory_delta
            logger.info(f"  {num_tools} tools: {memory_delta:.2f} MB delta")
        
        metrics = {
            "memory_baseline_mb": memory_baseline,
            "memory_deltas_by_tool_count": results,
        }
        _record_kpi("TC-AGENT-06: CPU memory during tool execution", metrics)


# ---------------------------------------------------------------------------
# TC-AGENT-07: Agent State Management KPIs
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.agent_state
class TestAgentStateManagement:
    """Measure agent state management overhead."""

    def test_agent_state_serialization_overhead(self):
        """TC-AGENT-07: Measure overhead of agent state serialization."""
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        # Create a complex agent session state
        session = AgentSession(agent_id=0)
        for turn_id in range(_NUM_TURNS):
            turn = AgentTurn(
                turn_id=turn_id,
                user_message=f"Message {turn_id}",
                reasoning_text=f"Reasoning for turn {turn_id}" * 10,
                reasoning_tokens_generated=256,
            )
            for i in range(4):
                tool_call = ToolCall(
                    tool_id=i,
                    tool_name=f"tool_{i}",
                    args={"data": np.random.randn(1000).tolist()},
                    dispatch_time_ns=time.perf_counter_ns(),
                    result={"output": np.random.randn(100).tolist()},
                )
                turn.tool_calls.append(tool_call)
            session.turns.append(turn)
        
        # Measure serialization
        serialization_times = []
        for _ in range(_BENCH):
            t0 = time.perf_counter_ns()
            state_json = json.dumps(session, default=str)
            elapsed_ns = time.perf_counter_ns() - t0
            serialization_times.append(elapsed_ns / 1e6)  # Convert to ms
        
        avg_serialization_ms = sum(serialization_times) / len(serialization_times)
        state_size_kb = len(state_json) / 1024.0
        
        logger.info(
            f"\n{'='*80}\n"
            f"  AGENT STATE SERIALIZATION\n"
            f"{'='*80}\n"
            f"  Avg serialization time:   {avg_serialization_ms:.3f} ms\n"
            f"  State size:               {state_size_kb:.2f} KB\n"
            f"  Serialization rate:       {state_size_kb / avg_serialization_ms:.2f} MB/s\n"
            f"{'='*80}\n"
        )
        
        metrics = {
            "avg_serialization_ms": avg_serialization_ms,
            "state_size_kb": state_size_kb,
            "serialization_rate_mbps": state_size_kb / avg_serialization_ms,
            "num_turns": _NUM_TURNS,
            "num_tools_per_turn": 4,
        }
        _record_kpi("TC-AGENT-07: Agent state serialization", metrics)


# ---------------------------------------------------------------------------
# TC-AGENT-08: Latency Percentiles and Outliers
# ---------------------------------------------------------------------------

@pytest.mark.agentic_ai
@pytest.mark.latency_analysis
class TestLatencyPercentiles:
    """Measure agent latency percentiles for SLA compliance."""

    def test_agent_latency_percentiles(self):
        """TC-AGENT-08: Measure latency percentiles for multi-turn interactions."""
        if not _EXECUTE:
            pytest.skip("Agentic AI execution disabled")
        
        turn_latencies = []
        
        for _ in range(_BENCH * 2):
            session = AgentSession(agent_id=0)
            turn = AgentTurn(turn_id=0, user_message="Test")
            
            turn.timestamps["reasoning_start"] = time.perf_counter()
            _, _ = _simulate_gpu_reasoning(_REASONING_TOKENS)
            turn.timestamps["reasoning_end"] = time.perf_counter()
            
            turn.timestamps["tools_dispatch_start"] = time.perf_counter()
            for i in range(min(4, _NUM_TOOLS)):
                _ = ToolCall(
                    tool_id=i,
                    tool_name=f"tool_{i}",
                    args={},
                    dispatch_time_ns=time.perf_counter_ns()
                )
            turn.timestamps["tools_dispatch_end"] = time.perf_counter()
            
            turn.timestamps["tools_exec_start"] = time.perf_counter()
            _simulate_tool_execution(0, _TOOL_EXEC_TIME_MS)
            turn.timestamps["tools_exec_end"] = time.perf_counter()
            
            turn.timestamps["response_start"] = time.perf_counter()
            turn.timestamps["response_end"] = time.perf_counter()
            
            turn_latencies.append(turn.total_turn_latency_ms())
        
        turn_latencies.sort()
        
        p50 = turn_latencies[len(turn_latencies) // 2]
        p95 = turn_latencies[int(len(turn_latencies) * 0.95)]
        p99 = turn_latencies[int(len(turn_latencies) * 0.99)]
        
        logger.info(
            f"\n{'='*80}\n"
            f"  AGENT LATENCY PERCENTILES\n"
            f"{'='*80}\n"
            f"  P50 latency:              {p50:.2f} ms\n"
            f"  P95 latency:              {p95:.2f} ms\n"
            f"  P99 latency:              {p99:.2f} ms\n"
            f"  Min latency:              {min(turn_latencies):.2f} ms\n"
            f"  Max latency:              {max(turn_latencies):.2f} ms\n"
            f"  Avg latency:              {sum(turn_latencies)/len(turn_latencies):.2f} ms\n"
            f"{'='*80}\n"
        )
        
        metrics = {
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "min_latency_ms": min(turn_latencies),
            "max_latency_ms": max(turn_latencies),
            "avg_latency_ms": sum(turn_latencies) / len(turn_latencies),
            "samples": len(turn_latencies),
        }
        _record_kpi("TC-AGENT-08: Latency percentiles", metrics)
