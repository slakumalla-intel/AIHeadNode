# KV Cache Multi-Model Tier Benchmark Report

## Test Summary

| Field | Value |
|---|---|
| Test date | 2026-09-07 |
| Workload | vLLM ShareGPT serving benchmark (`vllm bench serve`) |
| KV-cache tiers | GDDR (GPU HBM), DDR (LMCache L1 host DRAM), NVMe (LMCache L1+L2) |
| Models under test | 9 configurations across Llama, Qwen, and DeepSeek families |
| Requests per run | 2,048 |
| Client concurrency | 128 |
| Precision | BF16 |
| Decoding | Greedy (`--temperature 0`) |
| Total runs | 27 (9 models x 3 tiers) |
| Failed requests | 0 across all runs |
| Overall status | PASS |

## Models Under Test

The `Model` identifier below reflects the benchmark `model_label`. Tensor-parallel (TP) 2
configurations are the small/mid-size variants; TP=4 configurations are the large variants.

| Model | Model ID family | TP | Max model len |
|---|---|---:|---:|
| llama31_8b | meta-llama | 2 | 8,192 |
| llama2_7b | meta-llama | 2 | 4,096 |
| llama31_70b | meta-llama | 4 | 8,192 |
| llama33_70b | meta-llama | 4 | 8,192 |
| qwen25_7b | Qwen | 2 | 8,192 |
| qwen25_14b | Qwen | 2 | 8,192 |
| qwen25_72b | Qwen | 4 | 8,192 |
| deepseek | deepseek | 2 | 8,192 |
| deepseek | deepseek | 4 | 8,192 |

## Test Method

Each of the 9 model configurations was benchmarked against three KV-cache placements. Every run
issued 2,048 ShareGPT prompts at client concurrency 128 with greedy decoding, and recorded request
throughput, output-token throughput, and P50 TTFT/TPOT/ITL latency percentiles.

- **GDDR / GPU HBM** — KV cache held in vLLM-managed GPU memory (baseline).
- **DDR / LMCache L1** — KV cache offloaded to an LMCache host-DRAM L1 tier.
- **NVMe / LMCache L1+L2** — KV cache backed by an LMCache host-DRAM L1 tier plus an `fs` L2 adapter on NVMe.

## Results — GDDR / GPU HBM

| Model | TP | Max len | Conc | Status | Success | Fail | Req/s | Output tok/s | P50 TTFT (ms) | P50 TPOT (ms) | P50 ITL (ms) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 38.71 | 7,871.83 | 48.07 | 12.58 | 10.40 |
| llama2_7b | 2 | 4,096 | 128 | passed | 2,048 | 0 | 27.44 | 6,366.62 | 55.06 | 17.75 | 15.03 |
| llama31_70b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 9.16 | 1,651.01 | 234.09 | 71.89 | 49.61 |
| llama33_70b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.50 | 1,663.33 | 224.56 | 69.79 | 49.78 |
| qwen25_7b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 42.01 | 8,549.30 | 44.89 | 11.65 | 10.04 |
| qwen25_14b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 23.53 | 4,841.95 | 68.07 | 21.82 | 17.28 |
| qwen25_72b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.23 | 1,666.80 | 228.96 | 70.97 | 50.58 |
| deepseek | 2 | 8,192 | 128 | passed | 2,048 | 0 | 43.00 | 8,686.82 | 44.36 | 11.42 | 9.56 |
| deepseek | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.24 | 1,714.03 | 224.52 | 68.94 | 49.64 |

## Results — DDR / LMCache L1 (host DRAM)

| Model | TP | Max len | Conc | Status | Success | Fail | Req/s | Output tok/s | P50 TTFT (ms) | P50 TPOT (ms) | P50 ITL (ms) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 35.10 | 7,140.67 | 52.38 | 14.01 | 11.87 |
| llama2_7b | 2 | 4,096 | 128 | passed | 2,048 | 0 | 26.19 | 6,073.41 | 57.85 | 18.63 | 15.69 |
| llama31_70b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 9.14 | 1,652.00 | 233.93 | 71.60 | 50.08 |
| llama33_70b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.10 | 1,583.58 | 228.55 | 70.56 | 50.34 |
| qwen25_7b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 38.79 | 7,917.53 | 48.39 | 12.59 | 10.94 |
| qwen25_14b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 22.71 | 4,683.22 | 70.23 | 22.47 | 17.96 |
| qwen25_72b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.12 | 1,644.62 | 234.42 | 71.88 | 51.33 |
| deepseek | 2 | 8,192 | 128 | passed | 2,048 | 0 | 38.78 | 7,824.50 | 49.19 | 12.74 | 10.96 |
| deepseek | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.03 | 1,672.60 | 230.40 | 70.28 | 50.16 |

## Results — NVMe / LMCache L1+L2

| Model | TP | Max len | Conc | Status | Success | Fail | Req/s | Output tok/s | P50 TTFT (ms) | P50 TPOT (ms) | P50 ITL (ms) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 33.72 | 6,855.38 | 57.29 | 14.38 | 11.96 |
| llama2_7b | 2 | 4,096 | 128 | passed | 2,048 | 0 | 26.28 | 6,080.13 | 63.34 | 18.62 | 15.63 |
| llama31_70b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.98 | 1,619.99 | 270.60 | 72.98 | 50.38 |
| llama33_70b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 7.99 | 1,566.01 | 262.80 | 71.45 | 50.23 |
| qwen25_7b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 36.74 | 7,469.78 | 54.50 | 13.32 | 11.29 |
| qwen25_14b | 2 | 8,192 | 128 | passed | 2,048 | 0 | 22.51 | 4,635.92 | 76.99 | 22.50 | 17.79 |
| qwen25_72b | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.07 | 1,630.57 | 264.52 | 72.02 | 51.07 |
| deepseek | 2 | 8,192 | 128 | passed | 2,048 | 0 | 39.33 | 7,930.15 | 54.52 | 13.35 | 11.26 |
| deepseek | 4 | 8,192 | 128 | passed | 2,048 | 0 | 8.05 | 1,679.19 | 252.10 | 70.18 | 50.03 |

## Cross-Tier Comparison (Output Throughput, tok/s)

| Model | TP | GDDR | DDR | NVMe | DDR vs GDDR | NVMe vs GDDR |
|---|---:|---:|---:|---:|---:|---:|
| llama31_8b | 2 | 7,871.83 | 7,140.67 | 6,855.38 | -9.3% | -12.9% |
| llama2_7b | 2 | 6,366.62 | 6,073.41 | 6,080.13 | -4.6% | -4.5% |
| llama31_70b | 4 | 1,651.01 | 1,652.00 | 1,619.99 | +0.1% | -1.9% |
| llama33_70b | 4 | 1,663.33 | 1,583.58 | 1,566.01 | -4.8% | -5.8% |
| qwen25_7b | 2 | 8,549.30 | 7,917.53 | 7,469.78 | -7.4% | -12.6% |
| qwen25_14b | 2 | 4,841.95 | 4,683.22 | 4,635.92 | -3.3% | -4.3% |
| qwen25_72b | 4 | 1,666.80 | 1,644.62 | 1,630.57 | -1.3% | -2.2% |
| deepseek (TP2) | 2 | 8,686.82 | 7,824.50 | 7,930.15 | -9.9% | -8.7% |
| deepseek (TP4) | 4 | 1,714.03 | 1,672.60 | 1,679.19 | -2.4% | -2.0% |

## Interpretation

- All 27 runs completed with 2,048 successful requests and 0 failures, confirming functional
  stability of every KV-cache tier across all model families.
- The GDDR (GPU HBM) baseline delivered the highest output throughput in nearly every case, as
  expected for KV cache held directly in GPU memory.
- The DDR (LMCache L1) tier tracked close to the GDDR baseline. The largest throughput reductions
  appeared on the fast TP=2 small models (e.g. deepseek TP2 -9.9%, llama31_8b -9.3%), where offload
  overhead is proportionally larger relative to already-high token rates.
- The NVMe (LMCache L1+L2) tier remained within roughly 2-13% of the baseline. Large TP=4 models
  (70B/72B class) were least affected (-1.9% to -5.8%) because their lower token rates hide the
  added tier latency, while fast small models saw the largest relative reductions.
- Latency followed the same pattern: P50 TTFT increased modestly from GDDR to DDR to NVMe,
  most visibly on the large TP=4 models (e.g. llama31_70b TTFT 234 -> 234 -> 271 ms).

## Test Outcome

| Tier | Runs | Passed | Failed requests | Status |
|---|---:|---:|---:|---|
| GDDR / GPU HBM | 9 | 9 | 0 | PASS |
| DDR / LMCache L1 | 9 | 9 | 0 | PASS |
| NVMe / LMCache L1+L2 | 9 | 9 | 0 | PASS |

All nine model configurations passed on all three KV-cache tiers with zero request failures. The
DDR and NVMe offload tiers maintained throughput within a modest margin of the GDDR baseline,
with the smallest impact on large TP=4 models and the largest relative impact on fast TP=2 models.
