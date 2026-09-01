# Llama 3.1 8B TP2 x PP3 Benchmark Result

**Date:** 2026-08-24  
**Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`  
**Cluster:** 4 x H100 head + 2 x H200 worker  
**Tensor parallelism:** 2  
**Pipeline parallelism:** 3  
**Endpoint:** `http://127.0.0.1:8002`  
**Dataset:** ShareGPT  
**Requests:** 1,024  
**Maximum model length:** 8,192  
**Distributed backend:** Ray  
**NCCL transport:** Socket over `ens1f1np1`  

## Completion Evidence

```text
POST /v1/completions HTTP/1.1" 200 OK
Successful requests:                     1024
Failed requests:                         0
```

The vLLM server completed initialization and remained idle after the benchmark:

```text
init engine (profile, create kv cache, warmup model) took 55.78 s
API server: HTTP server started
Running: 0 reqs, Waiting: 0 reqs
GPU KV cache usage: 0.0%
```

## Benchmark Results

| Metric | Result |
|---|---:|
| Successful requests | 1,024 |
| Failed requests | 0 |
| Benchmark duration | 24.81 s |
| Total input tokens | 220,902 |
| Total generated tokens | 201,599 |
| Request throughput | 41.28 req/s |
| Output token throughput | 8,126.18 tok/s |
| Peak output token throughput | 24,358.00 tok/s |
| Total token throughput | 17,030.44 tok/s |
| Peak concurrent requests | 1,024 |

## Latency Results

| Metric | Mean | Median / P50 | P90 | P99 |
|---|---:|---:|---:|---:|
| TTFT | 2,553.61 ms | 2,491.19 ms | 3,921.29 ms | 4,220.20 ms |
| TPOT, excluding first token | 78.71 ms | 39.00 ms | 186.97 ms | 376.01 ms |
| ITL | 35.05 ms | 22.86 ms | 42.68 ms | 320.44 ms |
| End-to-end latency | 9,418.11 ms | 8,152.05 ms | 16,629.13 ms | 23,516.33 ms |

## Engine Observations

```text
Avg prompt throughput: 21909.0 tokens/s
Avg generation throughput: 3440.3 tokens/s
Running: 668 reqs, Waiting: 0 reqs
GPU KV cache usage: 4.1%

Avg generation throughput: 14802.3 tokens/s
Running: 127 reqs, Waiting: 0 reqs
GPU KV cache usage: 2.7%
```

## Notes

- The benchmark targeted port `8002`, matching the running vLLM server.
- All 1,024 requests returned successfully with no failed requests.
- The run used a high-concurrency burst, so the tail latency values are stress-test measurements rather than low-concurrency latency targets.
- The message about installing `termplotlib` and `gnuplot` is optional and does not affect benchmark validity.
- NCCL reported successful communicator initialization and socket transport over `ens1f1np1`.
