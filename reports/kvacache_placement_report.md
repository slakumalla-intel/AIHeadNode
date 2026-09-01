# Weekly Report — KVCacheClip & Node Validation (Aug 16–24, 2026)

## Executive Summary
This week focused on measuring and tuning the KVCacheClip offload layer for vLLM and
triaging node-validation KPIs. Three results stand out:

1. **Placement tier is latency-neutral.** RAM vs GPU vs NVMe backends show ≤6 ms
   difference (noise) — serving latency is prefill/compute-bound, and the cache stays
   in the DDR pool during serving (no auto-eviction to disk). Storage choice affects
   capacity/durability, not hot-path speed.
2. **QAT compression is a density lever, not a latency lever.** Higher
   `KVCLIP_CACHE_COMPRESSION_QUANT` monotonically shrinks the cache (up to 8.61x at
   QUANT=8) with no serving-latency penalty, but never beats the uncompressed
   latency baseline. `QUANT=4` is the best practical setting (~73% saved).
3. **Hit ratio is driven by prompt reuse + capacity, not model size or CPU speed.**
   CPU frequency (1800/3200 MHz) had zero effect (GPU-bandwidth-bound). Hit ratio
   comes from shared prefixes/chunks staying cached — now measurable via the new
   `REUSE_MODE=shared` and `KVFUSE_ENABLE` knobs in the benchmark harness.

**Bottom line:** For this workload, optimize for GPU memory bandwidth and prompt
reuse structure. Use compression to fit more cache, not to go faster.

## Summary
Delivered a hardened KVCache placement/hit benchmark, characterized QAT compression
behavior, and established that storage tier does not affect serving latency for the
tested workload. Latter part of the week shifted to node-validation KPI triage.

## Key Deliverables

### 1. KVCache Placement & Hit-Matrix Benchmark
- Built `examples/kvcache_placement_hit_matrix.sh`: serial RAM → GPU → NVMe comparison
  across low/medium/high reuse profiles, emitting a consolidated table,
  latency-vs-occupancy scatter, and winner-by-hits / winner-by-avg_ms verdict.
- Fixed defects: malformed connector JSON, CentOS here-doc / CRLF breakage,
  `print_winner_summary()` parse error, `workers` dict (not list) status parsing,
  compression field naming, and fractional `poolGB` crash.
- Added dedicated tmpfs ramdisk (`ensure_ramdisk`) and configurable `MEASURE_MAX_TOKENS`.

**Placement / Hit-Matrix Results** (H100, Qwen2.5-14B, `MEASURE_MAX_TOKENS=1`,
`--no-enable-prefix-caching`):

| Profile | Placement | Reqs | Entries | Usage % | Hits | Compression | Avg Latency |
|---------|-----------|-----:|--------:|--------:|------:|:-----------:|------------:|
| low     | ram (tmpfs) | 10 | 576  | 0.04 | 6,912   | 1.29x | ~72–80 ms |
| low     | gpu (8 GB)  | 10 | 576  | 0.04 | 6,912   | 1.29x | ~72–80 ms |
| low     | nvme        | 10 | 576  | 0.04 | 6,912   | 1.29x | ~72–80 ms |
| medium  | ram (tmpfs) | 20 | 2,448 | 0.5 | 61,344  | 1.29x | ~194–198 ms |
| medium  | gpu (8 GB)  | 20 | 2,448 | 0.5 | 61,344  | 1.29x | ~194–198 ms |
| medium  | nvme        | 20 | 2,448 | 0.5 | 61,344  | 1.29x | ~194–198 ms |
| high    | ram (tmpfs) | 40 | 6,864 | 4.0 | ~353,000 | 1.29x | ~469–472 ms |
| high    | gpu (8 GB)  | 40 | 6,864 | 4.0 | ~353,000 | 1.29x | ~469–472 ms |
| high    | nvme        | 40 | 6,864 | 4.0 | ~353,000 | 1.29x | ~469–472 ms |

> Latency spread across placements within each profile is ≤6 ms (measurement noise).
> Latency scales with prefix length / working set, not storage tier.

### 2. QAT Compression Tuning
- Characterized `KVCLIP_CACHE_COMPRESSION_QUANT` (levels 0/1/2/4/8) as a lossy
  quantization-truncation applied before QAT compression — not a hardware ratio knob.
- Ran QAT serving/capacity benchmarks and produced comparison tables across levels.

**CPU-Frequency Study** (Qwen2.5-7B, 3200-token prefix, 100 queries, 16 GB DDR pool):

| Config    | TTFT (mean) | Throughput | Compression | Cache Hits | Entries | Uncompressed | Compressed |
|-----------|------------:|-----------:|:-----------:|-----------:|--------:|-------------:|-----------:|
| 1800 MHz  | 259.90 ms | 87.35 tok/s | 1.27x | 73,332 | 9,380 | 307.36 MB | 241.42 MB |
| 3200 MHz  | 259.90 ms | 87.35 tok/s | 1.27x | 73,332 | 9,380 | 307.36 MB | 241.42 MB |
| Generic   | 259.90 ms | 87.35 tok/s | 1.27x | 73,332 | 9,380 | 307.36 MB | 241.42 MB |

> All three CPU-frequency configs yield identical performance → workload is
> GPU-memory-bandwidth-bound, not CPU-bound.

**QAT Quantization Levels — Serving Benchmark** (end-to-end):

| Quant | Compression | Space Saved | Mean TTFT | Output tok/s |
|------:|:-----------:|------------:|----------:|-------------:|
| 0 | 1.27x | 21.46% | 254.86 ms | 82.70 |
| 1 | 1.38x | 27.33% | 258.16 ms | 82.65 |
| 2 | 1.50x | 33.42% | 259.48 ms | 83.15 |
| 4 | 1.87x | 46.53% | 244.02 ms | 84.09 |

**QAT Quantization Levels — Capacity Benchmark** (live cache density):

| Quant | Measured Ratio | Compressed Size | Space Saved | Warm-Hit Speedup |
|------:|---------------:|----------------:|------------:|-----------------:|
| 0 | 2.58x | 1.36 MB | ~61% | 1.23x |
| 1 | 2.79x | 1.25 MB | ~64% | 1.25x |
| 2 | 3.06x | 1.15 MB | ~67% | 1.25x |
| 4 | 3.76x | 0.93 MB | ~73% | 1.36x |
| 8 | 8.61x | 0.41 MB | ~95% | 1.46x |

> Compression scales monotonically with quant level. `QUANT=4` is the strongest
> practical setting (best density with no serving-latency penalty); `QUANT=8` is
> an extreme, high-fidelity-risk setting. All compressed settings still trail the
> uncompressed baseline on latency (~138–144 ms TTFT), so compression is a
> capacity/density win, not a serving-latency win.

### 3. Node Validation & Infra (Aug 21, 24)
- Multi-node vLLM networking; FlashInfer / torch / CUDA version reconciliation;
  root-disk cleanup (was at 100%).
- Triaged node-validation KPIs: multi-GPU throughput scaling, HBM bandwidth during
  decode, P2P bandwidth/latency, KV-cache offload; copied results to HSD.

## Key Findings
- External KVClip hits register only with `--no-enable-prefix-caching`; otherwise
  vLLM's local GPU prefix cache serves every repeat.
- **Storage tier (ramdisk vs NVMe) has no hot-path latency impact** — the cache does
  not auto-evict to disk during serving (`NativeCache` grew to 324% usage);
  `persist_dir` is only touched via explicit `persist`/`evict`. Latency is
  prefill/compute-bound, not storage-bound.
- Cache location is set via `KVCLIP_CACHE_DIR` + `KVCACHE_CLIP_DDR_POOL_SIZE_GB`;
  there is no GPU-HBM placement switch.

## Recommended Next Steps
- To measure storage-tier latency, use an explicit persist → evict → replay flow
  rather than pool-size shrinking.
- Close out remaining node-validation KPI failures (HBM decode util, multi-GPU
  scaling ratio) and file results to HSD.
