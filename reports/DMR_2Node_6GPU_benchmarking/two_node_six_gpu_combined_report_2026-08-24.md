# Two-Node Six-GPU Combined Validation Report

**Date:** 2026-08-24  
**Project:** AIHeadNode multi-node GPU, Ray, NCCL, vLLM, and model validation  
**Overall status:** **PASS with follow-up items**

## 1. Executive Status

The two-node cluster was validated with six visible GPUs and successful distributed vLLM inference. Ray scheduling, Open MPI process launch, NCCL socket communication, model loading, API readiness, and benchmark request handling all completed successfully.

The validated production-style topology for this heterogeneous cluster is:

```text
TP=2 x PP=3 = 6 parallel workers
```

The cluster is heterogeneous, with four H100 GPUs on the head and two H200 NVL GPUs on the worker. RDMA was not available in the tested environment, so NCCL used the socket transport over `ens1f1np1`.

## 2. Cluster Inventory

| Role | Host | GPUs | Data IP | Status |
|---|---|---|---|---|
| HEAD | `SC009011168H0070` | 4 x H100 80 GB | `200.0.0.70` | PASS |
| WORKER | `sc00901168s0075` | 2 x H200 NVL 143 GB | `200.0.0.75` | PASS |

| Component | Validated configuration | Status |
|---|---|---|
| Data NIC | `ens1f1np1` | PASS |
| Ray address | `200.0.0.70:6379` | PASS |
| Python | 3.12.13 | PASS |
| PyTorch | 2.13.0+cu130 | PASS |
| CUDA toolkit | 13.2.51 | PASS |
| vLLM | 0.27.1 | PASS |
| Ray | 2.57.0 | PASS |
| Open MPI | 4.1.8 | PASS |

## 3. Ray Cluster Status

The Ray cluster was reset and restarted with the head first and worker second. Environment visibility was corrected by unsetting `CUDA_VISIBLE_DEVICES` before Ray startup.

Validated outcomes:

| Check | Result |
|---|---|
| Head joined as `200.0.0.70` | PASS |
| Worker joined as `200.0.0.75` | PASS |
| Total Ray GPU resources | `6.0 GPU` |
| GPU task visibility | Six one-GPU tasks saw `[1, 1, 1, 1, 1, 1]` |
| Stale-node cleanup | Completed |
| Management NIC avoidance | `ens1f1np1` selected |

The `32.0 TPU` line sometimes reported by Ray is a harmless hardware-detection artifact.

## 4. Topology Decision

### Validated topology

```text
Tensor parallelism: 2
Pipeline parallelism: 3
Total workers: 6
```

This topology uses two-GPU tensor-parallel groups and three pipeline stages. It is the recommended six-GPU configuration for this cluster.

### Rejected or unsuitable configurations

| Configuration | Reason |
|---|---|
| `TP=6` | Llama 3.1 8B has 32 attention heads and Llama 3.1 70B has 64; neither is divisible by 6 |
| `TP=4` | Exceeds the worker's two-GPU local count and creates unsuitable heterogeneous placement |
| `TP=1 x PP=6` | Not the validated topology for this model and cluster layout |

The cluster mixes H100 and H200 GPUs, which is officially unsupported by vLLM for some distributed configurations. KV cache capacity is bounded by the smaller H100 devices.

## 5. NCCL Validation

### Build and runtime repair

The initial `all_reduce_perf` file was zero bytes, causing:

```text
Exec format error
```

The test toolchain was repaired using:

```text
CUDA_HOME=/usr/local/cuda-13.2
MPI_HOME=/opt/openmpi-4.1.8
NCCL_HOME=/root/nccl/build
```

The build initially lacked `mpi.h`; passing `MPI_HOME` corrected the CUDA compilation include path. A valid 27 MB x86-64 executable was verified on both nodes.

NCCL runtime mismatch was then found:

- Head loaded NCCL 2.30.7
- Worker initially loaded NCCL 2.30.4

The runtime was aligned under:

```text
/root/nccl-uniform/lib
```

### Six-GPU NCCL result

The six-rank all-reduce test used all GPUs across both nodes and completed successfully:

| Metric | Result |
|---|---:|
| MPI ranks | 6 |
| Nodes | 2 |
| GPUs | 4 x H100 + 2 x H200 |
| Interface | `ens1f1np1` |
| NCCL mode | Socket, `NCCL_IB_DISABLE=1` |
| Validation errors | 0 |
| Peak algorithm bandwidth | 2.17 GB/s |
| Peak bus bandwidth | 3.62 GB/s |
| Approximate peak bus rate | 28.9 Gb/s |
| Average bus bandwidth | 0.947 GB/s |
| Result | **PASS** |

The NCCL log showed successful rings and socket transport between both node IPs.

### RDMA status

With `NCCL_IB_DISABLE=0`, NCCL reported:

```text
NET/IB : No device found.
Using network Socket
```

Therefore, RDMA/RoCE was not validated. The current bandwidth result is specifically a socket-path result.

## 6. vLLM Service Status

The six-GPU vLLM service was launched on port `8002` with:

```text
Model: Llama 3.1 8B or Llama 3.1 70B
TP=2 x PP=3
Ray distributed executor
bfloat16
max-model-len=8192
gpu-memory-utilization=0.85
```

Validated API checks:

| Check | Result |
|---|---|
| Port `8002` listening | PASS |
| `/health` | HTTP 200 OK |
| `/v1/models` | Expected model returned |
| `/v1/completions` | HTTP 200 responses |
| Engine initialization | PASS |
| Post-benchmark drain | Running and waiting requests reached 0 |

An earlier benchmark failure was caused by the client targeting the default port `8000` while the server was listening on `8002`. Adding `--port 8002` corrected the issue.

## 7. Captured vLLM Benchmark Results

### Llama 3.1 8B, TP=2 x PP=3

| Metric | Result |
|---|---:|
| Requests | 1,024 |
| Successful | 1,024 |
| Failed | 0 |
| Duration | 24.81 s |
| Request throughput | 41.28 req/s |
| Output token throughput | 8,126.18 tok/s |
| Total token throughput | 17,030.44 tok/s |
| P50 TTFT | 2,491.19 ms |
| P99 TTFT | 4,220.20 ms |
| P50 TPOT | 39.00 ms |
| P99 TPOT | 376.01 ms |
| P50 end-to-end latency | 8,152.05 ms |
| P99 end-to-end latency | 23,516.33 ms |
| Result | **PASS** |

### Llama 3.1 70B, TP=1

| Metric | Result |
|---|---:|
| Requests | 10,240 |
| Successful | 10,240 |
| Failed | 0 |
| Duration | 253.08 s |
| Request throughput | 40.46 req/s |
| Output token throughput | 7,997.93 tok/s |
| Total token throughput | 16,931.39 tok/s |
| P50 TTFT | 106,201.09 ms |
| P99 TTFT | 221,918.09 ms |
| P50 TPOT | 118.27 ms |
| P99 TPOT | 322.69 ms |
| P50 end-to-end latency | 130,190.99 ms |
| P99 end-to-end latency | 241,159.06 ms |
| Result | **PASS** |

### Llama 3.1 70B, TP=2 x PP=3

| Metric | Result |
|---|---:|
| Requests | 10,240 |
| Successful | 10,240 |
| Failed | 0 |
| Duration | 118.29 s |
| Request throughput | 86.57 req/s |
| Output token throughput | 17,116.38 tok/s |
| Total token throughput | 36,229.84 tok/s |
| P50 TTFT | 45,283.30 ms |
| P99 TTFT | 84,329.54 ms |
| P50 TPOT | 41.91 ms |
| P99 TPOT | 240.05 ms |
| P50 end-to-end latency | 53,699.17 ms |
| P99 end-to-end latency | 97,362.11 ms |
| Result | **PASS** |

## 8. 70B Scaling Comparison

Both 70B runs used 10,240 requests and the same high-concurrency burst workload.

| Metric | TP=1 | TP=2 x PP=3 | Change |
|---|---:|---:|---:|
| Request throughput | 40.46 req/s | 86.57 req/s | 2.14x |
| Output token throughput | 7,997.93 tok/s | 17,116.38 tok/s | 2.14x |
| Benchmark duration | 253.08 s | 118.29 s | 53.3% shorter |
| P50 TTFT | 106.20 s | 45.28 s | 57.4% lower |
| P99 TTFT | 221.92 s | 84.33 s | 62.0% lower |
| P50 TPOT | 118.27 ms | 41.91 ms | 64.6% lower |
| P50 end-to-end latency | 130.19 s | 53.70 s | 58.7% lower |
| P99 end-to-end latency | 241.16 s | 97.36 s | 59.6% lower |

The result represents combined tensor and pipeline parallel scaling from one worker to six workers. It is not pure tensor-parallel scaling. The 10,240-request burst caused substantial queueing, so a controlled-concurrency rerun is needed for latency-focused conclusions.

## 9. Configuration Fixes Recorded

| Issue | Resolution | Status |
|---|---|---|
| Missing `shared_worker_lock` argument | Use `--mm-processor-cache-type=lru` for text-only workloads | Resolved |
| Invalid cache value `none` | Use supported `lru` or `shm`; prefer `lru` here | Resolved |
| Empty NCCL test binary | Rebuilt `nccl-tests` | Resolved |
| Missing `nvcc` | Use `/usr/local/cuda-13.2/bin/nvcc` | Resolved |
| Missing `mpi.h` | Set `MPI_HOME=/opt/openmpi-4.1.8` | Resolved |
| NCCL version mismatch | Align runtime under `/root/nccl-uniform/lib` | Resolved |
| Open MPI unexpected process identifier | Clean stale MPI processes and bind Open MPI to `ens1f1np1` | Resolved |
| Benchmark connection refused | Match benchmark port to server port `8002` | Resolved |
| Optional plotting notice | Install `termplotlib` and `gnuplot` if plots are needed | Optional |

## 10. Remaining Follow-Up Items

| Priority | Item | Current status |
|---|---|---|
| High | Run TP=1 versus TP=2 x PP=3 with `--max-concurrency 128` | Pending |
| High | Validate RDMA/RoCE device and configuration | Pending; current test falls back to Socket |
| Medium | Capture full 70B launch environment and model path | Partially recorded |
| Medium | Generate saved plots with `termplotlib` and `gnuplot` | Optional |
| Medium | Continue monitoring mixed H100/H200 placement stability | Required for production confidence |

## 11. Source Artifacts

- [Daily work report](daily_work_report_2026-08-24.md)
- [Llama 3.1 8B TP2 x PP3 benchmark](llama31_8b_tp2_pp3_benchmark_2026-08-24.md)
- [Llama 70B TP1 benchmark](llama70b_benchmark_2026-08-24.md)
- [Llama 70B TP2 x PP3 benchmark](llama70b_tp2_pp3_benchmark_2026-08-24.md)
- [Multi-node vLLM runbook](../WIKI_multinode_vllm_runbook.md)
