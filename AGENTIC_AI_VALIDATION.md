# Agentic AI Workload Validation Suite - CPU+GPU KPIs

## Overview

Comprehensive validation suite for agentic AI systems on the AI Head Node combining CPU-based tool/function execution with GPU-based language model reasoning. This test suite captures industrial-grade KPIs for production agentic workload deployment.

**Test Module:** `tests/test_agentic_ai_validation.py`  
**KPI Output:** `artifacts/agentic_ai_kpis.jsonl`

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  AGENTIC AI SYSTEM                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────┐          ┌──────────────────┐ │
│  │  GPU INFERENCE   │          │  CPU TOOLS       │ │
│  │                  │          │  (Thread Pool)   │ │
│  │ • Reasoning      │◄────────►│ • Search         │ │
│  │ • Planning       │ Sync     │ • APIs           │ │
│  │ • Decision-making│ Overhead │ • Computation    │ │
│  │                  │          │ • I/O            │ │
│  └──────────────────┘          └──────────────────┘ │
│         (4x H100)                  (160 cores)      │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │     STATE MANAGEMENT & MEMORY LAYER          │  │
│  │ • Multi-turn context tracking                │  │
│  │ • KV cache management                        │  │
│  │ • Tool result aggregation                    │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Test Categories and KPIs

### TC-AGENT-01: Tool Invocation Overhead

**Tests:** 
- `test_single_tool_dispatch_latency()`
- `test_parallel_tool_dispatch_scalability()`

**KPIs Captured:**
- Single tool dispatch latency (μs)
- Per-tool dispatch overhead
- N-tool dispatch scalability (1→16 tools)
- Dispatch scaling ratio

**Industrial Standard:** Tool dispatch < 10ms per tool

**Metrics Recorded:**
```json
{
  "avg_dispatch_latency_ms": 0.5,
  "min_dispatch_latency_ms": 0.1,
  "max_dispatch_latency_ms": 1.2,
  "latency_1_tool_ms": 0.4,
  "latency_16_tools_ms": 3.2,
  "scaling_ratio": 8.0
}
```

**Use Case:** Ensures rapid tool dispatch doesn't become bottleneck in agent reasoning loop.

---

### TC-AGENT-02: Reasoning Phase Latency

**Tests:**
- `test_reasoning_latency_by_token_count()`

**KPIs Captured:**
- GPU reasoning latency by token count (64, 128, 256, 512 tokens)
- Tokens per millisecond (throughput)
- Reasoning phase device (GPU vs CPU fallback)

**Industrial Standard:** <10ms per 256 tokens (expected for 4xH100)

**Metrics Recorded:**
```json
{
  "reasoning_latencies_by_tokens": {
    "64": 2.5,
    "128": 5.1,
    "256": 10.2,
    "512": 20.4
  },
  "device": "GPU",
  "tokens_per_ms": 25.0
}
```

**Measurement:** GPU token generation using transformer simulation (attention + MLP)

**Use Case:** SLA planning for agent response time budgets.

---

### TC-AGENT-03: CPU-GPU Synchronization Overhead

**Tests:**
- `test_cpu_gpu_sync_latency()`

**KPIs Captured:**
- CPU-GPU sync latency (μs)
- P50, min, max sync overhead
- Data transfer efficiency

**Industrial Standard:** <500 μs per sync operation

**Metrics Recorded:**
```json
{
  "avg_sync_latency_us": 125.5,
  "min_sync_latency_us": 45.2,
  "max_sync_latency_us": 342.1,
  "iterations": 10
}
```

**Measurement:** Synchronization via tensor copy + CUDA sync

**Impact:** Affects interleaved GPU reasoning + CPU tool execution patterns

---

### TC-AGENT-04: Tool Execution Parallelization

**Tests:**
- `test_sequential_vs_parallel_tool_execution()`

**KPIs Captured:**
- Sequential execution time
- Parallel execution time
- Speedup ratio (seq / par)
- Tool count scalability
- CPU core utilization

**Industrial Standard:** Speedup >1.5x for 8+ tools on 160-core CPU

**Metrics Recorded:**
```json
{
  "sequential_time_s": 4.2,
  "parallel_time_s": 1.1,
  "speedup": 3.8,
  "num_tools": 8,
  "iterations": 10,
  "avg_core_utilization_percent": 67.3
}
```

**Measurement:** ThreadPoolExecutor with CPU-bound tool simulation

**Use Case:** Validates multi-tool agent strategies execute efficiently on available CPU cores.

---

### TC-AGENT-05: Multi-Turn Conversation KPIs

**Tests:**
- `test_multi_turn_session_latency()`

**KPIs Captured:**
- Total session latency (all turns)
- Per-turn latency breakdown:
  - Reasoning phase latency
  - Tool dispatch latency
  - Tool execution latency
  - Response generation latency
- Average latencies across turns
- Cumulative KV cache growth

**Industrial Standard:** <200ms per turn (reasoning + tools + response)

**Metrics Recorded:**
```json
{
  "total_session_latency_ms": 875.3,
  "avg_reasoning_latency_ms": 45.2,
  "avg_tool_latency_ms": 125.6,
  "num_turns": 5,
  "per_turn_latencies_ms": [165.4, 172.1, 168.9, 170.2, 171.8]
}
```

**State Tracked:**
- Turn ID and user message
- Reasoning text and token count
- Tool calls with dispatch/execution times
- Final response
- Timestamps for each phase

---

### TC-AGENT-06: CPU Memory and Bandwidth During Tool Execution

**Tests:**
- `test_memory_usage_scaling_with_tools()`

**KPIs Captured:**
- Baseline memory usage
- Peak memory delta by tool count (1, 4, 8 tools)
- Memory scaling efficiency
- Per-tool memory overhead

**Industrial Standard:** Sub-linear memory growth (N tools ≠ N× memory)

**Metrics Recorded:**
```json
{
  "memory_baseline_mb": 245.3,
  "memory_deltas_by_tool_count": {
    "1": 12.5,
    "4": 35.2,
    "8": 64.1
  },
  "scaling_efficiency": "sub-linear"
}
```

**Measurement:** Process RSS memory via psutil

**Use Case:** Capacity planning for multi-agent deployments.

---

### TC-AGENT-07: Agent State Management KPIs

**Tests:**
- `test_agent_state_serialization_overhead()`

**KPIs Captured:**
- State serialization latency (ms)
- Serialized state size (KB)
- Serialization throughput (MB/s)
- Number of turns and tools in state
- JSON encoding overhead

**Industrial Standard:** >100 MB/s serialization throughput

**Metrics Recorded:**
```json
{
  "avg_serialization_ms": 8.3,
  "state_size_kb": 1245.6,
  "serialization_rate_mbps": 150.2,
  "num_turns": 5,
  "num_tools_per_turn": 4
}
```

**State Components:**
- Multi-turn conversation history
- Tool calls with arguments and results
- Reasoning text and token counts
- Timestamps for each phase
- Agent metadata

**Use Case:** Checkpoint/restore, state persistence, cluster coordination.

---

### TC-AGENT-08: Latency Percentiles and SLA Analysis

**Tests:**
- `test_agent_latency_percentiles()`

**KPIs Captured:**
- P50 (median) latency
- P95 latency (95th percentile)
- P99 latency (99th percentile)
- Min, max, average latencies
- Outlier distribution

**Industrial Standard:**
- P50: <100ms
- P95: <250ms
- P99: <500ms

**Metrics Recorded:**
```json
{
  "p50_latency_ms": 87.3,
  "p95_latency_ms": 156.2,
  "p99_latency_ms": 312.5,
  "min_latency_ms": 45.1,
  "max_latency_ms": 425.8,
  "avg_latency_ms": 105.6,
  "samples": 20
}
```

**Distribution Visualization:**
```
Latency Distribution (20 samples):
│
│ ███                           P50: 87ms
│ ███ ███                       P95: 156ms
│ ███ ███ █                     P99: 312ms
│ ███ ███ █ █ █ █ █ █ █ █
└─────────────────────────────
  50   100  150  200  250  300+ ms
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGENTIC_EXECUTE` | `1` | Enable/disable GPU-heavy tests |
| `AGENTIC_MODEL_ID` | `gpt2` | HuggingFace model for reasoning |
| `AGENTIC_NUM_TOOLS` | `8` | Number of tools in agent system |
| `AGENTIC_NUM_TURNS` | `5` | Multi-turn conversation depth |
| `AGENTIC_NUM_AGENTS` | `2` | Parallel agents for scaling tests |
| `AGENTIC_TOOL_EXEC_TIME_MS` | `50` | Simulated tool execution time (ms) |
| `AGENTIC_REASONING_TOKENS` | `256` | Tokens for reasoning phase |
| `AGENTIC_WARMUP_ITERS` | `3` | Warmup iterations before benchmarking |
| `AGENTIC_BENCH_ITERS` | `10` | Main benchmark iterations |
| `AGENTIC_KPI_FILE` | `artifacts/agentic_ai_kpis.jsonl` | KPI output file |

### Configuration Example:

```bash
# Quick validation run
export AGENTIC_EXECUTE=1
export AGENTIC_NUM_TOOLS=4
export AGENTIC_NUM_TURNS=3
export AGENTIC_WARMUP_ITERS=1
export AGENTIC_BENCH_ITERS=5

# Stress test with many tools
export AGENTIC_NUM_TOOLS=32
export AGENTIC_NUM_TURNS=10
export AGENTIC_BENCH_ITERS=50
```

---

## Running the Tests

### Run all agentic AI tests:
```bash
pytest -m agentic_ai -vv tests/test_agentic_ai_validation.py
```

### Run specific test category:
```bash
pytest -m tool_invocation -vv tests/test_agentic_ai_validation.py
pytest -m reasoning_latency -vv tests/test_agentic_ai_validation.py
pytest -m cpu_gpu_sync -vv tests/test_agentic_ai_validation.py
pytest -m tool_parallelism -vv tests/test_agentic_ai_validation.py
pytest -m multi_turn -vv tests/test_agentic_ai_validation.py
pytest -m memory_intensive -vv tests/test_agentic_ai_validation.py
pytest -m agent_state -vv tests/test_agentic_ai_validation.py
pytest -m latency_analysis -vv tests/test_agentic_ai_validation.py
```

### Run with custom configuration:
```bash
AGENTIC_NUM_TOOLS=16 AGENTIC_NUM_TURNS=8 pytest -m agentic_ai -vv tests/test_agentic_ai_validation.py
```

### Run with detailed logging:
```bash
pytest -m agentic_ai -vv tests/test_agentic_ai_validation.py -s --log-cli-level=DEBUG
```

### Collect tests without running:
```bash
pytest -m agentic_ai --collect-only tests/test_agentic_ai_validation.py
```

---

## KPI Output Format

All KPIs are recorded to a JSONL file with the following structure:

```json
{
  "timestamp": "2026-04-20T14:35:22",
  "test": "TC-AGENT-01a: Single tool dispatch",
  "metrics": {
    "avg_dispatch_latency_ms": 0.45,
    "min_dispatch_latency_ms": 0.12,
    "max_dispatch_latency_ms": 1.05,
    "iterations": 10
  }
}
{
  "timestamp": "2026-04-20T14:35:25",
  "test": "TC-AGENT-02: Reasoning phase latency",
  "metrics": {
    "reasoning_latencies_by_tokens": {
      "64": 2.5,
      "128": 5.1,
      "256": 10.2,
      "512": 20.4
    },
    "device": "GPU"
  }
}
```

### Parse KPIs with Python:

```python
import json
from pathlib import Path

kpi_file = Path("artifacts/agentic_ai_kpis.jsonl")
for line in kpi_file.read_text().strip().split("\n"):
    if line:
        record = json.loads(line)
        print(f"{record['test']}: {record['metrics']}")
```

---

## Data Models

### ToolCall
Represents a single tool invocation with timing:
- `tool_id`: Unique identifier
- `tool_name`: String name of tool
- `args`: Dictionary of arguments
- `dispatch_time_ns`: When invoked (nanoseconds)
- `start_time_ns`: When execution started
- `end_time_ns`: When result ready
- `result`: Execution result
- `error`: Optional error message

Methods:
- `dispatch_latency_us()`: Time from dispatch to execution start
- `end_to_end_latency_us()`: Total time from dispatch to result

### AgentTurn
Represents one turn in a multi-turn conversation:
- `turn_id`: Turn number
- `user_message`: Input from user
- `reasoning_text`: Chain-of-thought generated by LLM
- `tool_calls`: List of ToolCall objects
- `final_response`: Agent's response

Methods:
- `reasoning_latency_ms()`: Time spent reasoning
- `tool_dispatch_latency_ms()`: Time to dispatch tools
- `tool_execution_latency_ms()`: Max tool execution latency
- `total_turn_latency_ms()`: Total turn time

### AgentSession
Represents entire multi-turn agent session:
- `agent_id`: Session identifier
- `turns`: List of AgentTurn objects
- `total_cpu_memory_used_mb`: Peak memory
- `peak_gpu_util_percent`: Max GPU utilization
- `peak_cpu_util_percent`: Max CPU utilization

Methods:
- `avg_reasoning_latency_ms()`: Average reasoning across turns
- `avg_tool_latency_ms()`: Average tool latency
- `total_session_latency_ms()`: Total session duration

---

## Expected Results (4x H100 + 160-core CPU)

| KPI | Target | Expected | Pass Threshold |
|-----|--------|----------|-----------------|
| Tool dispatch latency | <10ms | 0.5ms | <5ms |
| Reasoning latency (256 tokens) | <20ms | 10ms | <25ms |
| CPU-GPU sync overhead | <500μs | 150μs | <1000μs |
| Tool parallelization speedup | >1.5x | 3.5x | >1.2x |
| Multi-turn latency (P50) | <100ms | 87ms | <150ms |
| Multi-turn latency (P95) | <250ms | 156ms | <400ms |
| Memory scaling efficiency | Sub-linear | Yes | Pass |
| State serialization | >100 MB/s | 150 MB/s | >50 MB/s |

---

## CI/CD Integration

### GitLab CI Example:

```yaml
test_agentic_ai:
  stage: validate
  script:
    - pytest -m agentic_ai tests/test_agentic_ai_validation.py -v --tb=short
  artifacts:
    reports:
      junit: test-results.xml
  timeout: 30m
```

### GitHub Actions Example:

```yaml
- name: Run agentic AI validation
  run: |
    pytest -m agentic_ai tests/test_agentic_ai_validation.py -v
  env:
    AGENTIC_NUM_TURNS: 5
    AGENTIC_BENCH_ITERS: 10
```

---

## Troubleshooting

### Issue: "CUDA not available" for sync tests

**Solution:** Ensure PyTorch with CUDA support is installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Tests timeout

**Solution:** Reduce benchmark iterations:
```bash
AGENTIC_BENCH_ITERS=3 pytest -m agentic_ai tests/test_agentic_ai_validation.py
```

### Issue: Memory exhausted

**Solution:** Reduce number of tools and turns:
```bash
AGENTIC_NUM_TOOLS=4 AGENTIC_NUM_TURNS=3 pytest -m agentic_ai tests/test_agentic_ai_validation.py
```

### Issue: psutil not available

**Solution:** Install psutil:
```bash
pip install psutil
```

---

## Performance Tuning Tips

1. **GPU Reasoning:** Use lower precision (BF16/FP8) for faster inference
2. **Tool Execution:** Use process pool for I/O-bound tools, thread pool for CPU-bound
3. **State Management:** Compress large tool results before serialization
4. **Multi-turn:** Implement sliding window to bound context size
5. **CPU-GPU Sync:** Batch tool results to amortize sync overhead

---

## References

- Agentic AI Systems: https://arxiv.org/abs/2309.17288
- LLM Latency Analysis: https://llm.datasette.io/
- GPU Performance Tuning: https://docs.nvidia.com/cuda/best-practices-guide/
- CPU Memory Hierarchy: https://en.wikichip.org/wiki/intel/microarchitectures/cascade_lake
