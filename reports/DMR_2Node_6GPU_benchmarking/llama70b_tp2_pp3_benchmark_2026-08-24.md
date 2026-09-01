# Llama 3.1 70B TP=2 x PP=3 Benchmark Result

**Date:** 2026-08-24  
**Model:** `meta-llama/Meta-Llama-3.1-70B-Instruct`  
**Tensor parallelism:** 2  
**Pipeline parallelism:** 3  
**Total parallel workers:** 6  
**Distributed executor:** Ray  
**Dtype:** `bfloat16`  
**Maximum model length:** 8,192  
**GPU memory utilization:** 0.85  
**Endpoint:** `http://127.0.0.1:8002`  
**Dataset:** ShareGPT  
**Requests:** 10,240  

## Launch Commands

### Server

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 3 \
  --distributed-executor-backend ray \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --enforce-eager \
  --gpu-memory-utilization 0.85 \
  --port 8002 &
```

### Benchmark

```bash
vllm bench serve \
  --backend vllm \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --dataset-name sharegpt \
  --dataset-path "$SHAREGPT_DATASET_PATH" \
  --host 127.0.0.1 \
  --port 8002 \
  --num-prompts 10240 \
  --percentile-metrics ttft,tpot,itl,e2e \
  --metric-percentiles 50,90,99 \
  --save-result \
  --save-detailed
```

## Completion Evidence

```text
Successful requests:                     10240
Failed requests:                         0
Benchmark duration (s):                  118.29
```

The server returned HTTP 200 responses and drained the workload:

```text
POST /v1/completions HTTP/1.1" 200 OK
Running: 0 reqs, Waiting: 0 reqs
GPU KV cache usage: 0.0%
```

## Benchmark Results

| Metric | Result |
|---|---:|
| Successful requests | 10,240 |
| Failed requests | 0 |
| Benchmark duration | 118.29 s |
| Total input tokens | 2,260,903 |
| Total generated tokens | 2,024,671 |
| Request throughput | 86.57 req/s |
| Output token throughput | 17,116.38 tok/s |
| Peak output token throughput | 32,871.00 tok/s |
| Total token throughput | 36,229.84 tok/s |
| Peak concurrent requests | 10,240 |

## Latency Results

| Metric | Mean | Median / P50 | P90 | P99 |
|---|---:|---:|---:|---:|
| TTFT | 44,833.03 ms | 45,283.30 ms | 76,984.19 ms | 84,329.54 ms |
| TPOT, excluding first token | 49.15 ms | 41.91 ms | 53.50 ms | 240.05 ms |
| ITL | 42.16 ms | 36.33 ms | 61.95 ms | 127.04 ms |
| End-to-end latency | 53,127.27 ms | 53,699.17 ms | 85,168.55 ms | 97,362.11 ms |

## Engine Observations

The pasted excerpt does not include intermediate engine-throughput samples. Only the final benchmark summary is recorded in this artifact.

## Interpretation

- The run completed successfully: 10,240 of 10,240 requests succeeded.
- This was a maximum-burst workload with peak concurrency equal to the full request count.
- The median TTFT was 45.28 seconds and median end-to-end latency was 53.70 seconds; these values include queueing from the 10,240-request burst.
- Decode performance was comparatively lower than the earlier TP=1 run's reported values: median TPOT was 41.91 ms and P99 TPOT was 240.05 ms.
- The `termplotlib` and `gnuplot` message is optional and does not indicate benchmark failure.
- The pasted result did not include the NCCL transport/version lines, so those details are not inferred here.
