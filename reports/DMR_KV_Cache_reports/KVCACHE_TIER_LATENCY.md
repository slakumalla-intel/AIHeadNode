# KV Cache Tier Latency Benchmark

`kvcache_tier_latency.sh` measures how long it actually takes to get KV blocks
back from each storage tier: GPU HBM, the KVClip DDR pool, tmpfs, and NVMe.

## Why the obvious approach does not work

The KVClip `NativeCache` does **not** auto-spill to `persist_dir` while serving.
Shrinking `KVCACHE_CLIP_DDR_POOL_SIZE_GB` does not push blocks to disk — the
pool simply grows past `capacity_bytes`. A "DDR vs NVMe" test built on pool
sizing therefore measures nothing but run-to-run noise.

Two mechanisms are needed to move the data:

| comparison | mechanism |
| --- | --- |
| HBM vs DDR | `POST /reset_prefix_cache` drops vLLM's local cache, forcing the connector path |
| DDR vs disk | `POST /v1/cache/persist` then `POST /v1/cache/evict`, then replay |

## Tiers

| tier | how it is forced | blocks come from |
| --- | --- | --- |
| `cold` | unique nonce per request | full prefill (upper bound) |
| `gpu` | replay with prefix caching on | vLLM local cache in HBM |
| `ddr` | `reset_prefix_cache` before every request | KVClip in-memory pool |
| `ram` | persist + evict, `persist_dir` on tmpfs | tmpfs read |
| `nvme` | persist + evict + `drop_caches`, `persist_dir` on NVMe | device read |

`gpu` and `ddr` run in the **same** server process with identical flags, so the
delta between those two rows is the HBM↔DDR transfer plus decompression.

## Usage

```bash
sed -i 's/\r$//' kvcache_tier_latency.sh   # only if edited on Windows

# full sweep
NVME_DIR=/mnt/nvme/kvclip_tier ./kvcache_tier_latency.sh Qwen/Qwen2.5-14B-Instruct

# HBM vs DDR only - one server, no restarts
MODE=gpu_ddr REQS=30 ./kvcache_tier_latency.sh Qwen/Qwen2.5-14B-Instruct
```

Root is required for the tmpfs mount and for `drop_caches`. Without it the NVMe
row is optimistic because reads are served from the OS page cache.

## Configuration

| variable | default | purpose |
| --- | --- | --- |
| `MODE` | `all` | `all` or `gpu_ddr` |
| `REQS` | `30` | measured requests per phase |
| `PREFIX_CHARS` | `16000` | shared prefix length |
| `MAX_TOKENS` | `1` | keep at 1 so latency reflects prefill + fetch, not decode |
| `POOL_GB` | `64` | `KVCACHE_CLIP_DDR_POOL_SIZE_GB`; must hold the working set |
| `RAMDISK_DIR` | `/mnt/ramdisk/kvclip_tier` | tmpfs persist dir |
| `NVME_DIR` | `/mnt/nvme/kvclip_tier` | NVMe persist dir |
| `PORT` | `8000` | vLLM port |
| `CACHE_API` | `http://localhost:18700` | KVClip controller |
| `LOG_DIR` | `/tmp/kvcache_tier_latency` | logs, samples, `results.tsv` |

## Output

`${LOG_DIR}/results.tsv`:

```
tier  mean_ms  p50_ms  p95_ms  hits  hits_in_storage  misses
```

## Reading the results

A row is only meaningful if the cache counters moved the right way:

- `ddr` must show `hits` increasing; `gpu` must show it flat.
- `ram` / `nvme` must show `hits_in_storage` increasing.

If those stay at zero, the request was served from a faster tier and the row
measures nothing. Cross-check the server log in `${LOG_DIR}`: a GPU hit logs
`locally-cached tokens: N, externally-cached tokens: 0`, an external hit logs
`externally-cached tokens: N`.

## Caveats

- Requests are serial and single-stream by design; concurrency blends queueing
  delay into the tier signal.
- Resetting the prefix cache once per phase is not enough — the first request
  repopulates HBM and every later one becomes a GPU hit again. The script
  resets before each request in the `ddr` phase.
- Latency scales with prefix length and working-set size; compare rows only
  within a single run.
