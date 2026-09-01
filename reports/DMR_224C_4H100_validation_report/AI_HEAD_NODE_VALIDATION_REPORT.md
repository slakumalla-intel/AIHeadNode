# AI Head Node Validation Report

Comprehensive functional, performance, reliability, workload, and AI pipeline assessment

**System:** SC009011168H0070

**Run:** 17 Aug 2026, 01:16-03:17 UTC

**Platform:** 4x NVIDIA H100 PCIe, 224 CPU cores, 3.02 TB RAM

**Source:** supplied pytest live log + AIHeadNode test definitions

## Executive Summary

The report covers the complete selected validation list. The node demonstrates strong core hardware health and usable AI performance: GPU compute, HBM, PCIe, NVLink, NCCL, CPU scaling, agentic AI, AI pipeline, vLLM prefill, and the 26-case performance suite are operational. Five individual tests require follow-up, but every reported subcategory passes the acceptance rule because at least 90% of its executed tests passed.

**Total tests:** 390 - 12 reported test domains

**Passed:** 357 - 91.5% of all collected tests

**Failed:** 5 - Five individual follow-up items

**Skipped:** 28 - Unsupported or unavailable prerequisites

**Subcategory status:** 12 / 12 - PASS when executed pass rate is >=90%

**Executed pass rate:** 98.6% - 357 / 362 non-skipped outcomes

**Hardware confidence: high.** All four GPUs delivered 92.7-92.8% HBM efficiency, 85-89% PCIe bandwidth in repeatable suites, 55-68% practical dense GEMM efficiency, zero uncorrected ECC errors, healthy Gen5 x16 links, and up to 265 GB/s on NVLink-connected pairs.

**Release verdict: subcategories pass with follow-up actions.** All 12 subcategories meet or exceed the 90% executed-test pass threshold. Resolve the single-thread STREAM Triad result, intermittent GPU 0 D2H result, torch.compile regression, vLLM utilization telemetry, and YOLO worker crash to achieve a clean test run.

### Test outcome distribution

- Passed: 357
- Failed: 5
- Skipped: 28
- Total: 390

### Five individual failures by subcategory

- Node workloads 3

- Memory 1

- System health 1

Failures remain visible even though each subcategory exceeds the 90% acceptance threshold.

## Platform Under Test

|  |  |
| --- | --- |
| Host | SC009011168H0070 |
| OS | Linux 6.18.0, glibc 2.39 |
| Python / pytest | 3.12.13 / 9.1.1 |
| CPU | 224 physical cores, 224 logical threads, 1 NUMA node |
| Memory | 3,022.62 GB RAM; 3,005.82 GB available at memory test |
| Storage | 1,675.69 GB detected |
| GPU | 4x NVIDIA H100 PCIe, compute capability 9.0 |
| GPU memory | 79.18 GB each; 316.72 GB aggregate |
| Driver / CUDA runtime | 595.45.04 / 13.2 runtime reported by nvbandwidth |
| PCIe | Gen5 x16 negotiated on all GPUs; 63 GB/s theoretical per direction |
| GPU fabric | NV12 pairs 0-1 and 2-3; NODE path across pairs |
| Offline mode | HF\_HUB\_OFFLINE=1; remote large-model download disabled |

## Test Coverage and Domain Results

| Domain | Description | Status | Tests | Pass | Fail | Skip | Executed pass rate | Duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bottleneck | Roofline and resource bottleneck analysis | PASS | 25 | 25 | 0 | 0 | 100.0% | 38.2 s |
| CPU | Topology, compute, bandwidth, scaling, scheduling | PASS | 33 | 32 | 0 | 1 | 100.0% | 10m 36s |
| Memory | Capacity, STREAM, latency, cache, ECC, pressure | PASS | 28 | 27 | 1 | 0 | 96.4% | 28.0 s |
| GPU | Discovery, compute, HBM, PCIe, ECC, health | PASS | 53 | 53 | 0 | 0 | 100.0% | 29.7 s |
| Interconnect | PCIe, NVLink, P2P, multistream, latency | PASS | 77 | 69 | 0 | 8 | 100.0% | 1m 38s |
| Node validation | Bring-up, workload acceptance, inference, RDMA | PASS | 51 | 39 | 3 | 9 | 92.9% | 38m 52s |
| Agentic AI | Tool dispatch, sync, parallelism, state, latency | PASS | 9 | 9 | 0 | 0 | 100.0% | 21.3 s |
| AI pipeline | Tokenization, I/O, orchestration, inference KPIs | PASS | 38 | 34 | 0 | 4 | 100.0% | 1m 29s |
| System health | Thermals, clocks, ECC, numerical correctness | PASS | 39 | 36 | 1 | 2 | 97.3% | 1m 2s |
| System config | Captured hardware and software conformance | PASS | 8 | 8 | 0 | 0 | 100.0% | 6.5 s |
| vLLM prefill | 4-H100 prefill and quick validation | PASS | 3 | 2 | 0 | 1 | 100.0% | 4m 27s |
| Performance suite | LLM serving, GenAI-Perf, NCCL catalog | PASS | 26 | 23 | 0 | 3 | 100.0% | 28m 57s |

Pass counts are tests minus failures, errors, and skips. Subcategory status is PASS when passed / (passed + failed + errors) is at least 90%; skipped tests are excluded from that percentage. Individual failures remain reported in the detailed KPI and action tables.

## CPU and GPU Compute KPIs

### GPU practical efficiency vs theoretical peak

- FP16 GPU 0 62.4%

- FP16 GPU 1 63.2%

- FP16 GPU 2 65.9%

- FP16 GPU 3 67.8%

- HBM average 92.8%

### CPU scaling efficiency

- 1 core 99.4%

- 2 cores 89.5%

- 4 cores 99.2%

- 8 cores 94.0%

- 16 / 32 / 64 94-99%

### Compute result table

| Test / KPI | Measured | Reference | Efficiency | Threshold | Status | Threshold justification |
| --- | --- | --- | --- | --- | --- | --- |
| CPU FP32 GEMM, 1x32 | 5.125 TFLOPS | 27.238 | 18.82% | >=5% | PASS | Low floor validates vectorized compute without treating Python/BLAS process topology as vendor-peak parity. |
| CPU FP32 GEMM, 4x52 | 7.013 TFLOPS | 27.238 | 25.75% | >=5% | PASS | Same practical smoke threshold; higher process count improves utilization. |
| CPU all-core extrapolated FP32 | 28.536 TFLOPS | 27.238 | 104.76% | >=20% | PASS | Sampled 16-core BLAS throughput is extrapolated; values over 100% indicate turbo/library effects and are not literal sustained whole-socket proof. |
| CPU 8-core scaling | 941.742 GFLOPS | 940.503 | 100.13% | >=85% | PASS | 85% allows scheduling and synchronization overhead while requiring useful parallel scaling. |
| GPU 0 FP32 | 37.122 TFLOPS | 67.0 | 55.41% | >=52% | PASS | PCIe H100 practical eager-GEMM floor accounts for framework and launch overhead. |
| GPU 1 / 2 / 3 FP32 | 37.405 / 37.047 / 37.103 | 67.0 each | 55.83 / 55.29 / 55.38% | >=52% | PASS | Cross-GPU spread is under 1%, indicating balanced devices. |
| GPU 0 / 1 FP16 | 471.80 / 477.71 TFLOPS | 756 each | 62.41 / 63.19% | >=40% | PASS | PyTorch eager matrix multiplication under-represents vendor peak due to runtime and kernel overhead. |
| GPU 2 / 3 FP16 | 498.21 / 512.69 TFLOPS | 756 each | 65.90 / 67.82% | >=40% | PASS | Healthy margin; GPU 3 is the best FP16 result. |
| 4-GPU concurrent FP16 | 1,881.10 TFLOPS | 1,849.09 measured baseline sum | 101.73% | >=80% | PASS | Validates concurrent execution against observed single-GPU baselines, not 3,024 TFLOPS vendor peak. |
| 4-GPU aggregate bottleneck test | 1,904.81 TFLOPS | 3,024 | 62.99% | >=60% | PASS | 60% selected for 2.0 TB/s PCIe H100 runtime path; separates practical GEMM from ideal tensor-core peak. |
| BF16 GPU 0-3 | 500.56-533.23 TFLOPS | 756 test reference | 66.2-70.5% | >=40% | PASS | Matches FP16 practical eager-GEMM threshold. |
| INT8 proxy GPU 0-3 | 552.29-562.78 TOPS | 756 proxy ref. | 73.1-74.4% | >=40% | PASS | Proxy test uses FP16-derived reference; it is not a direct claim against 3,958 TOPS vendor INT8 peak. |
| FP64 GPU 0-3 | 35.80-37.89 TFLOPS | 34 each | 105.3-111.4% | >=50% | PASS | Above-reference results reflect clock and implementation differences; all devices pass comfortably. |

**Interpretation guardrail:** several metrics exceed 100% because the tests compare sampled, extrapolated, proxy, or conservative reference values. They demonstrate threshold compliance, not a violation of physical peak limits.

## Memory Hierarchy and Bandwidth KPIs

| KPI | Measured | Acceptance threshold | Status | Description / justification |
| --- | --- | --- | --- | --- |
| Total RAM | 3,022.62 GB | >=1,024 GB; within 5% detected baseline | PASS | Capacity and stable baseline validation. |
| Available RAM | 3,005.82 GB (99.44%) | >=50% available | PASS | Ensures performance tests are not distorted by memory pressure. |
| Single-thread Copy | 40.37-40.50 GB/s | >=30 GB/s | PASS | Baseline single-thread memory-path floor. |
| Single-thread Scale | 27.63 GB/s | >=17.94 GB/s (40% Copy, 12 GB/s floor) | PASS | Scale has an additional arithmetic operation and three memory references; 40% of observed Copy adapts to the active platform. |
| Single-thread Triad | 11.30 GB/s | >=16.21 GB/s (40% Copy; floor 12) | FAIL | Measured 30.3% below adaptive threshold and 5.8% below absolute floor. Re-run pinned to a quiet core and verify vectorization. |
| Aggregate STREAM Triad | 596.33 GB/s | >=75% of 307.2 GB/s | PASS | Full-chip floor allows NUMA and synchronization overhead; result above reference likely includes cache effects. |
| Aggregate DRAM Copy | 1,812.60 GB/s | >=75% of 307.2 GB/s | PASS | 590% efficiency is not credible as sustained DRAM-only traffic; treat as cache-assisted benchmark throughput. |
| L3 aggregate bandwidth | 1,365.65 GB/s | >=200 GB/s scaled floor | PASS | 224-thread cache bandwidth aggregate. |
| Latency 32 KB / 512 KB | 0.94 / 0.95 ns | <=4 / 8 ns | PASS | L1/L2-size working-set checks. |
| Latency 16 MB / 512 MB | 6.12 / 7.24 ns | <=30 / 120 ns | PASS | L3/DRAM-size working-set checks; very low 512 MB result warrants methodology review but passes configured test. |
| Cache BW L1 / L2 / L3 / DRAM | 33.55 / 38.88 / 20.79 / 39.19 GB/s | >=5 / 30 / 20 / 15 | PASS | Per-working-set floors detect collapsed hierarchy performance. |
| Sustained bandwidth | 43.59-45.37 GB/s | <=15% degradation | PASS | Observed maximum degradation 1.52%. |
| Large allocation | 8 GB in 1.888 s | <=30 s | PASS | Allocation/free pressure check. |
| HBM3 GPU 0-3 | 1.8548-1.8565 TB/s | >=80% of 2.0 TB/s | PASS | 80% allows memory-controller, driver, ECC, and kernel overhead; measured 92.7-92.8%. |

**Memory conclusion:** aggregate and sustained tests are healthy, while one single-thread Triad sample fails. Because the aggregate Triad passes strongly, the immediate hypothesis is a local benchmark/vectorization/affinity issue rather than a node-wide DRAM fault; this must be confirmed by a controlled rerun.

## PCIe, NVLink, P2P, and NCCL KPIs

### PCIe efficiency by GPU

- H2D GPU 0-3 89.3-89.4%

- D2H GPU 0-3 86.8-87.1%

- Bidirectional 79.0-79.2%

- All-GPU H2D 81.4%

### NCCL peak bus bandwidth

- AllReduce sweep 96.17

- AllGather 83.26

- ReduceScatter 78.57

- AllToAll 62.03

| KPI | Measured | Reference | Threshold | Status | Interpretation |
| --- | --- | --- | --- | --- | --- |
| PCIe H2D GPU 0-3 | 56.26-56.33 GB/s | 63 GB/s | >=65% | PASS | 89.3-89.4%; strong Gen5 x16 utilization. |
| PCIe D2H GPU 0-3 | 54.66-54.89 GB/s | 63 GB/s | >=65% | PASS | 86.8-87.1%; repeatable interconnect suite result. |
| Node direct D2H GPU 0 | 38.99 GB/s | 63 GB/s | >=40.95 GB/s | FAIL | Isolated low sample; later nvbandwidth D2H measured 55.36 GB/s. Treat as intermittent until reproduced. |
| Bidirectional per GPU | 99.54-99.83 GB/s combined | 126 GB/s | configured suite pass | PASS | 79% of sum-of-directions reference with simultaneous streams. |
| Concurrent H2D, 4 GPUs | 205.21 GB/s | 252 GB/s | >=65% | PASS | 81.43% aggregate host-to-device efficiency. |
| NVLink pair 0-1 / 2-3 | 265.01 / 265.20 GB/s | 300 GB/s | >=70% | PASS | 88.3-88.4% of configured pair reference. |
| NODE-path P2P pairs | 69.97-71.56 GB/s | 63 GB/s PCIe ref. | >=50% | PASS | Non-NVLink pairs route through PCIe/NODE topology. |
| All-to-all P2P aggregate | 526.0 GB/s | 1,704 GB/s mixed-path theoretical | >=20% | PASS | 30.87%; topology contains both NVLink and NODE paths. |
| NCCL AllReduce benchmark | peak 96.17, mean 36.80 GB/s | size-dependent | successful, no wrong values | PASS | Peak occurs at large payloads; small-message averages reduce overall mean. |
| NCCL P2P disabled | peak 36.71 GB/s | P2P baseline 89.33 | expected degradation | PASS | 59% peak reduction confirms benefit of direct P2P paths. |
| Kernel launch / sync | p50 18.89 us / 3.09 us | observational | suite acceptance | PASS | Launch and cudaDeviceSynchronize overhead characterization. |

## AI Workload Acceptance KPIs

### Hugging Face batch-1 latency and throughput

- Llama 8B 69 tok/s

- DeepSeek 7B 4-bit 39.3 tok/s

- Llama 70B 4-bit 13.4 tok/s

- Qwen 72B 4-bit 13.4 tok/s

### GenAI-Perf parallel-user scaling (TP4)

- Concurrency 16 1,239 tok/s

- Concurrency 32 2,652 tok/s

- Concurrency 64 5,217 tok/s

- Concurrency 128 7,990 tok/s

- Concurrency 256 11,251 tok/s

| Workload | Configuration | Mean / p95 latency | Throughput | Threshold / expectation | Status |
| --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B HF inference | batch 1, 32 new tokens | 465.2 / 558.7 ms | 68.79 tok/s | mean <1,200 ms; p95 <1,500 ms | PASS |
| DeepSeek R1 Distill Qwen 7B | 4-bit, batch 1, 32 tokens | 813.9 / 920.8 ms | 39.32 tok/s | <3,000 ms configured | PASS |
| Llama 3.1 70B | 4-bit, batch 1, 16 tokens | 1,191.7 / 1,290.7 ms | 13.43 tok/s | <12,000 ms configured | PASS |
| Qwen 2.5 72B | 4-bit, batch 1, 16 tokens | 1,192.4 / 1,278.1 ms | 13.42 tok/s | <12,000 ms configured | PASS |
| Llama 8B HF batch | batch 8, 64 tokens, 5 iterations | 4.91 s measured generation window | 521.16 tok/s | workload acceptance | PASS |
| vLLM engine inference | 4 prompts, 64 tokens | 0.673 s | 380.38 tok/s | successful runtime | PASS |
| KV cache CPU offload | TP4, context 8,192, batch 4 | 6,771.7 ms offload vs 6,831.5 baseline | 252.0 vs 249.8 tok/s | controlled degradation | PASS |
| Diffusers SD-Turbo | 4 GPUs, 128 images, 512x512 | 2.445 s makespan | 52.36 img/s | >=10 img/s | PASS |
| BERT encoder training | batch 16, seq 128, 10 steps | 0.563 s training window | 284.27 samples/s | successful throughput test | PASS |
| vLLM prefill saturation | 1,000 prompts, 1,024/128 tokens | 240.2 s runtime | GPU mean 0% | GPU >=90% for >=50% samples | FAIL |
| YOLOv8x object detection | 4 workers, batch 128 | no worker KPI | 0 results | 4 worker results required | FAIL |

## Agentic AI and AI Pipeline KPIs

### Agentic AI

| KPI | Result | Meaning |
| --- | --- | --- |
| Single tool dispatch | 0.00094 ms avg | In-process dispatch overhead is negligible. |
| 16-tool dispatch | 0.00760 ms | 6.44x time for 16x dispatch count. |
| Reasoning latency | 1.35-10.19 ms | 64-512 token synthetic reasoning path. |
| CPU-GPU sync | 36.97 us avg | Synchronization overhead. |
| Parallel tool speedup | 1.97x | 8 tools: 4.08 s sequential vs 2.07 s parallel. |
| Five-turn session | 1,028.09 ms | 205-207 ms per turn. |
| State serialization | 7.91 ms | 448.46 KB at 56.72 MB/s. |
| Agent latency p50/p95/p99 | 55.19 / 107.46 / 107.46 ms | 20-sample latency distribution. |

### Pipeline highlights

| KPI | Result | Threshold |
| --- | --- | --- |
| Tokenizer, seq 32 / 128 | 430k / 841k tok/s | >=50k |
| Tokenizer, seq 512 / 2,048 | 1.46M / 1.73M tok/s | >=50k / 20k |
| Tokenizer 4-thread scaling | 1.835x | >=1.03x |
| Async prefetch overlap | 99.5% | >=60% |
| CPU preprocessing | 746,252 tok/s | >=200,000 |
| Storage read / write | 1.401 / 1.171 GB/s | >=1.0 / 0.5 |
| Random 4K read | 10,723 IOPS | >=5,000 |
| TCP loopback | 99.80 Gbps | >=5 |
| Model weight pipeline | 4.456 GB/s | >=1.0 |
| TTFT p95, 32-512 input | 4.74-5.06 ms | <=200-500 |
| ITL p95 | 57.60 ms | <=100 |
| Continuous batching | 28,128 tok/s | >=200 |

### Batched inference curve

- Batch 1 250.5 tok/s

- Batch 4 986.7 tok/s

- Batch 8 2,009 tok/s

- Batch 16 3,838 tok/s

Median latency remains approximately 127-133 ms while throughput scales 15.3x from batch 1 to 16.

## System Health, Reliability, and Numerical Correctness

| KPI | Measured | Threshold | Status | Assessment |
| --- | --- | --- | --- | --- |
| Idle GPU temperature 0-3 | 38 / 35 / 34 / 33 C | <83 C | PASS | Large thermal margin. |
| Peak observed clock ratio | 100 / 100 / 94.9 / 91.5% | >=70% | PASS | All devices boost adequately. |
| Sustained thermal test | 59 C max; 47.9% clock drop | drop <=55% | PASS | Configured throttle criterion met. |
| Power headroom | 85.8-252.8 W / 350 W | <=limit | PASS | No power-limit violation. |
| Uncorrected / corrected HBM ECC | 0 / 0 all GPUs | 0 / <100 | PASS | No logged memory errors. |
| PCIe replay counters | 0 all GPUs | 0 | PASS | Clean PCIe link health. |
| FP16 energy efficiency | 5.00-5.59 TFLOPS/W | >=1.0 | PASS | Power sample appears near idle for some GPUs; useful for test compliance, not audited energy benchmarking. |
| INT8 vs FP16 error | 1.534% | <=5% | PASS | Quantized matrix result within tolerance. |
| FP8 vs FP16 error | 3.745% | <=15% | PASS | Within configured E4M3 tolerance. |
| BF16 vs FP16 error | 0.287% | <=10% | PASS | Strong numerical agreement. |
| Softmax max absolute error | 0.002563 | <=0.003 | PASS | Pass with 14.6% margin. |
| torch.compile speedup | 0.231x | >=0.75x effective floor | FAIL | Compiled path is 4.33x slower for this workload shape; investigate compilation mode, warmup, and graph breaks. |
| Stream concurrency ratio | 1.050x | <=1.5x | PASS | Concurrent execution not materially worse than sequential. |

## Performance Suite Results

The dedicated 26-case suite completed with **23 passed and 3 skipped**. This is the strongest evidence that the available LLM serving and collective-communication toolchain is usable.

### LLM serving benchmark table

| Case | Model / mode | Output tok/s | P50 / P99 TTFT | P50 / P99 ITL | P99 E2E | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 003 | Llama 3.1 70B BF16, TP4, random 2K/256 | 520.2 | 4,793 / 17,734 ms | 45.3 / 1,137 ms | 39,483 ms | PASS |
| 004 | Llama 3.1 8B BF16, TP4, ShareGPT | 2,307.2 | 453 / 593 ms | 10.2 / 73.9 ms | 9,003 ms | PASS |
| 005 | Llama 3.1 70B FP8, TP4, ShareGPT | 723.5 | 2,023 / 2,792 ms | 28.1 / 94.2 ms | 25,586 ms | PASS |
| 006 | Llama 3.1 8B BF16, TP2 | 2,336.4 | 506 / 696 ms | 10.1 / 73.2 ms | 8,789 ms | PASS |
| 007 | Llama 3.1 8B BF16, TP1, mixed prompt | 2,073.6 | 939 / 1,007 ms | 11.5 / 71.3 ms | 10,034 ms | PASS |
| 008 | Mixtral 8x7B FP8, TP2 | 1,080.5 | 717 / 1,026 ms | 21.2 / 85.6 ms | 22,160 ms | PASS |
| 009 | Llama 8B burst, 200 RPS configured | 2,109.8 | 500 / 697 ms | 11.5 / 51.9 ms | 9,668 ms | PASS |
| 016 | DeepSeek R1 Distill Llama 70B FP8 | 761.8 | 1,657 / 2,844 ms | 29.6 / 97.7 ms | 26,104 ms | PASS |
| 017 | DeepSeek R1 70B FP8, alternate profile | 762.6 | 1,957 / 2,811 ms | 28.8 / 98.3 ms | 26,482 ms | PASS |
| 026 | Llama 3.1 70B FP8 stress | 735.7 | 2,039 / 2,633 ms | 27.5 / 101.3 ms | 24,121 ms | PASS |

### GenAI-Perf scenarios

| Scenario | Concurrency | Output tok/s | Average TTFT | P99 TTFT | Average ITL |
| --- | --- | --- | --- | --- | --- |
| TP scaling: TP1 | 16 | 1,038.2 | 258.8 ms | 731.6 ms | 12.88 ms |
| TP scaling: TP4 | 16 | 1,221.1 | 188.7 ms | 486.8 ms | 11.25 ms |
| Parallel users | 16 / 32 / 64 / 128 / 256 | 1,239 / 2,652 / 5,217 / 7,990 / 11,251 | 183 / 68 / 101 / 142 / 322 ms | 483 / 159 / 177 / 281 / 770 ms | 11.1 / 10.8 / 11.0 / 13.2 / 18.3 ms |
| Heavy output, 512 tokens | 16 / 32 / 64 | 1,039 / 2,355 / 4,110 | 73.7 / 43.8 / 53.2 ms | 274 / 119 / 158 ms | 10.96 / 10.80 / 11.17 ms |
| Long context, 4K input | 8 / 16 / 32 | 526 / 930 / 2,421 | 353 / 177 / 124 ms | 833 / 527 / 278 ms | 13.05 / 13.75 / 10.90 ms |

## Failure Analysis and Recommended Actions

| Priority | Finding | Evidence | Likely cause / justification | Recommended action |
| --- | --- | --- | --- | --- |
| P1 | vLLM prefill utilization telemetry invalid | GPU mean 0%, no latency percentiles, stale EngineCore cleanup | Benchmark/server monitoring did not observe the active process or the benchmark failed to produce metrics. | Capture server and bench exit codes/output; bind NVML sampling to serving GPU PID; fail fast if p50/p95/p99 are null. |
| P1 | YOLO multiprocessing produced no results | 0 worker results for 4 processes | Worker exception was not propagated; deprecated half option may be related but is not sufficient evidence. | Capture worker tracebacks through the result queue, lower batch from 128, and validate one GPU before four-GPU spawn. |
| P1 | Intermittent GPU 0 D2H dip | 38.99 GB/s vs 40.95 threshold; later 54-55 GB/s | Transient contention, first-iteration state, affinity, or pinned-memory variability. | Run 10 repeated D2H trials per GPU, report median/p05, and correlate CPU NUMA affinity; alert only on reproducible p05 failure. |
| P1 | Single-thread STREAM Triad below floor | 11.30 vs 16.21 GB/s adaptive threshold | Possible vectorization, CPU affinity, frequency, or allocation alignment issue; aggregate bandwidth is healthy. | Pin to a physical core, inspect compiler/NumPy SIMD path, increase warmup, and compare Copy/Scale/Triad under identical allocation. |
| P2 | torch.compile regression | 0.231x vs 0.75x effective floor | Workload too small, insufficient steady-state iterations, graph breaks, or unsuitable compile mode. | Separate compile cost from steady-state timing; inspect graph-break logs; test reduce-overhead and max-autotune modes. |
| P2 | Misleading over-100% efficiencies | DRAM 590%, CPU 653%, scaling 271% | Cache effects, extrapolation, or reference mismatch. | Label each denominator explicitly; add cache-flush/working-set controls; cap visualization at 100% while preserving raw values. |
| P3 | Fork deprecation warnings | 15 bottleneck warnings on multi-threaded fork | Python 3.12 warns that fork from multithreaded process can deadlock. | Use spawn/forkserver multiprocessing context for benchmark workers. |

### Acceptance gates for the next run

**Hardware gate**

- All 4 GPUs Gen5 x16
- HBM >=80%
- PCIe median >=65%
- Zero uncorrected ECC

**Runtime gate**

- No stale vLLM processes
- Non-null latency percentiles
- YOLO workers return metrics
- torch.compile >=0.75x

**Clean-run gate**

- All five failed tests rerun
- No stale service processes
- No null workload metrics
- 390-test counts reconciled

Methodology and interpretation notes

- Results are transcribed from the supplied live pytest log and cross-checked against workload\_status\_report.json and test descriptions in the AIHeadNode workspace.
- The run used Python 3.12.13, pytest 9.1.1, a 1,800-second pytest timeout, offline Hugging Face mode, and remote large-model downloads disabled.
- Theoretical values are suite-configured references, not independently audited vendor specifications. HBM is treated as 2.0 TB/s per GPU and PCIe as 63 GB/s per direction.
- Thresholds are acceptance floors designed to catch regressions under framework overhead. They are not equivalent to vendor benchmark targets unless explicitly stated.
- Skipped tests are excluded from the 90% subcategory pass calculation and do not establish compliance for the skipped behavior.

AI Head Node Validation Report | 390 selected tests from the 17 Aug 2026 run log | Self-contained HTML; no external scripts, fonts, or images
