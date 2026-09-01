# KVCacheClip Manual Mode Configuration Guide

This guide explains how to manually configure and run independent vLLM instances in different KVCacheClip modes for benchmarking and comparison.

## Overview of Modes

KVCacheClip supports four modes for running vLLM servers:

| Mode | Cache Reuse | Compression | Backend | Use Case |
|------|-------------|------------|---------|----------|
| **cold** | ❌ No | — | — | Baseline: full prefill on every request |
| **raw** | ✅ Yes | ❌ None | — | Measure cache restore overhead only |
| **qat_sw** | ✅ Yes | ✅ Software | CPU | Compression impact on CPU (slow, high CPU load) |
| **qat_hw** | ✅ Yes | ✅ Hardware | QAT Device | Production: hardware-accelerated compression |

### Performance Hierarchy

```
qat_hw ≈ raw  ≫  qat_sw  ≫  cold
```

- **qat_hw ≈ raw**: Hardware compression is nearly transparent (adds compression for free)
- **raw ≫ qat_sw**: Software compression creates CPU contention with decode threads
- **qat_sw ≫ cold**: Any prefix reuse beats recomputing full prefill

---

## Quick Start

### Run Single Mode

```bash
# Start a single vLLM in a specific mode
cd /path/to/KVCacheClip

# Cold mode (baseline)
./examples/run_vllm_modes.sh cold Qwen/Qwen2.5-7B-Instruct 8000

# Raw mode (cache reuse, no compression)
./examples/run_vllm_modes.sh raw Qwen/Qwen2.5-7B-Instruct 8001

# QAT software mode (cache reuse + software compression)
./examples/run_vllm_modes.sh qat_sw Qwen/Qwen2.5-7B-Instruct 8002

# QAT hardware mode (cache reuse + hardware compression)
./examples/run_vllm_modes.sh qat_hw Qwen/Qwen2.5-7B-Instruct 8003
```

### Run All Modes in Parallel

```bash
# Start all four modes on ports 8000-8003
./examples/run_all_modes.sh Qwen/Qwen2.5-7B-Instruct

# Stop all servers
./examples/run_all_modes.sh stop
```

---

## Detailed Mode Configuration

### 1. COLD Mode (Baseline)

**What it does**: Disables external cache reuse entirely. Every request triggers a full KV cache prefill computation.

**Configuration**:
```bash
./examples/run_vllm_modes.sh cold MODEL_NAME 8000
```

**Environment Variables**:
```bash
KVCLIP_CACHE_ENABLE=0
```

**Why use it**: Establishes a performance baseline to measure the value of prefix caching.

---

### 2. RAW Mode (Cache Reuse without Compression)

**What it does**: Enables external cache reuse but disables compression. This isolates the cost of cache restore (decompression + data copy) without compression overhead.

**Configuration**:
```bash
./examples/run_vllm_modes.sh raw MODEL_NAME 8001
```

**Environment Variables**:
```bash
KVCLIP_CACHE_COMPRESSION=0
```

**QAT Device State**: UP (but not used since compression is disabled)

**Why use it**: Establishes the performance ceiling — the best possible performance when using external cache.

---

### 3. QAT_SW Mode (Software Compression)

**What it does**: Enables cache reuse with software DEFLATE compression. Compression runs on CPU cores alongside decode, creating CPU contention.

**Configuration**:
```bash
./examples/run_vllm_modes.sh qat_sw MODEL_NAME 8002
```

**Environment Variables**:
```bash
KVCLIP_CACHE_COMPRESSION=1
KVCLIP_CACHE_COMPRESSION_QUANT=0        # No quantization truncation
KVCLIP_QAT_INSTANCE_NUM=4                # Default threads
```

**QAT Device State**: DOWN (forced software fallback via `adf_ctl down`)

**CPU Impact**: HIGH — compression competes with decode threads for CPU cores

**Why use it**: Demonstrates the value of hardware offload by showing CPU contention costs.

---

### 4. QAT_HW Mode (Hardware Compression)

**What it does**: Enables cache reuse with hardware-accelerated DEFLATE compression on the QAT device. Compression is offloaded from the CPU.

**Configuration**:
```bash
./examples/run_vllm_modes.sh qat_hw MODEL_NAME 8003
```

**Environment Variables**:
```bash
KVCLIP_CACHE_COMPRESSION=1
KVCLIP_CACHE_COMPRESSION_QUANT=0        # No quantization truncation
KVCLIP_QAT_INSTANCE_NUM=64               # Higher parallelism for hardware
KVCLIP_QAT_HW_BUFF_KB=256                # Hardware buffer size
```

**QAT Device State**: UP (active hardware acceleration via `adf_ctl up`)

**CPU Impact**: MINIMAL — compression offloaded to QAT accelerator

**Why use it**: Production-ready mode with optimal performance and minimal CPU overhead.

---

## Advanced Configuration

### Custom Quantization Truncation

Reduce memory footprint further by lossy truncation of LSBs:

```bash
# Auto: bf16→2 bits, fp8→1 bit
KVCLIP_CACHE_COMPRESSION_QUANT=auto ./examples/run_vllm_modes.sh qat_hw MODEL 8003

# Manual: truncate 3 LSBs
KVCLIP_CACHE_COMPRESSION_QUANT=3 ./examples/run_vllm_modes.sh qat_hw MODEL 8003
```

### Adjust QAT Threads

For `qat_hw` mode:

```bash
# More parallelism (default is 64)
KVCLIP_QAT_INSTANCE_NUM=128 ./examples/run_vllm_modes.sh qat_hw MODEL 8003

# Fewer threads (for testing)
KVCLIP_QAT_INSTANCE_NUM=8 ./examples/run_vllm_modes.sh qat_hw MODEL 8003
```

### Enable Debugging

Add verbose logging:

```bash
KVCACHE_CLIP_DEBUG=1 ./examples/run_vllm_modes.sh qat_hw MODEL 8003
```

---

## Testing the Servers

### Health Check

```bash
# Check if server is ready
curl http://localhost:8000/health

# List loaded models
curl http://localhost:8000/v1/models
```

### Simple Generation Test

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Benchmark with vLLM Client

Use the standard vLLM benchmarking tools:

```bash
# Sequential requests
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000

# Generate traffic
python examples/benchmark_serving.py \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-prompts 100
```

---

## Comparison Workflow

Run all modes and compare results:

```bash
#!/bin/bash

# Terminal 1: Start all servers
./examples/run_all_modes.sh Qwen/Qwen2.5-7B-Instruct

# Terminal 2: Wait for servers to be ready, then run benchmarks
sleep 60

for port in 8000 8001 8002 8003; do
  mode_name=("cold" "raw" "qat_sw" "qat_hw")
  mode=${mode_name[$((port - 8000))]}
  
  echo "Benchmarking $mode on port $port..."
  python examples/benchmark_serving.py \
    --backend openai \
    --base-url http://localhost:$port/v1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 100 \
    --dataset-path ./examples/sample_requests.jsonl \
    --output-file results_${mode}.jsonl
done

# Terminal 1: Stop all servers
./examples/run_all_modes.sh stop
```

---

## QAT Device Management

### Check QAT Status

```bash
# View all QAT devices
adf_ctl status

# Expected output when active:
#   Available Devices:
#   qat_dev0: up
#   qat_dev1: up
#   ...
```

### Manual QAT Control

```bash
# Bring QAT up (enable hardware)
sudo adf_ctl up

# Bring QAT down (disable hardware, force software fallback)
sudo adf_ctl down
```

### Verify QAT Driver Installation

```bash
# Check if QAT driver is loaded
lsmod | grep qat

# Check QAT library
ls -la /usr/lib64/libqatzip.so*
```

---

## Troubleshooting

### Problem: QAT device not found
```
[WARN] adf_ctl not found, skipping QAT device setup
```

**Solution**: Install QAT driver (see [Getting Started Guide](../doc/en/usage/getting-started.md))

### Problem: Cannot bring QAT device up/down
```
Bringing QAT device up...
Could not bring QAT device up. Will use software fallback.
```

**Solution**: Check QAT driver status and permissions:
```bash
sudo adf_ctl status
# If error, restart QAT service:
sudo systemctl restart qat
```

### Problem: Docker permission denied
```
Got permission denied while trying to connect to the Docker daemon
```

**Solution**: Run with sudo or add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Problem: Port already in use
```
Error response from daemon: driver failed... bind address already in use
```

**Solution**: Change port number or stop existing container:
```bash
docker stop vllm-cold-8000
docker rm vllm-cold-8000

# Or use a different port
./examples/run_vllm_modes.sh cold MODEL 9000
```

---

## Performance Metrics to Track

When benchmarking, capture:

| Metric | Unit | How to Measure |
|--------|------|---|
| **Throughput** | tokens/sec | Total output tokens / elapsed time |
| **TTFT** | ms | Time to first token (latency for first output) |
| **ITL** | ms | Inter-token latency (latency between subsequent tokens) |
| **CPU Utilization** | % | `top`, `htop`, or per-core via `/proc/cpuinfo` |
| **GPU Utilization** | % | `nvidia-smi` |
| **Memory Used** | GB | `nvidia-smi` for GPU, `/proc/meminfo` for CPU |
| **Compression Ratio** | % | Compressed size / original size × 100 |

---

## References

- [KVCacheClip Design Overview](../doc/en/design/kvclip-overview.md)
- [QAT for CPU Inference](../doc/en/design/qat-cpu-inference.md)
- [Getting Started Guide](../doc/en/usage/getting-started.md)
- [TensorZip Design](../doc/en/design/tensorzip.md)

