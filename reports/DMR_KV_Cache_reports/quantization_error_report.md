# KVCacheClip Quantization Error Report

Date: 2026-08-27

## Test Setup

This report summarizes BF16 quantization-truncation tests run with `examples/test_quantization_error.py` against `kv_tensor.pt`.

| Field | Value |
|---|---:|
| Data type | `bfloat16` |
| Number of values | `1048576` |
| Raw BF16 tensor size | `2097152` bytes / `2.00 MiB` |
| Device | `cuda` |
| Source | `kv_tensor.pt` |
| Tensor file | `examples/kv_tensor.pt` |

The test compares each original tensor value with the value after KVCacheClip-style bit truncation:

```text
quantized = original_bits & ~((1 << bits) - 1)
```

Percentage error is computed per value as:

```text
pct_error = abs(original - quantized) / max(abs(original), 1e-12) * 100
```

The updated `test_quantization_error.py` script also reports zlib-compressed byte savings (`zlib_saved_bytes`, `zlib_saved_pct`) as a standalone proxy for QAT deflate savings.

## Graph Summary

![KVCacheClip quantization space saving vs error](quant_space_error_graphs.svg)

## Single Comparison Report

| Quant bits | Effective BF16 bits | Effective packed size | Mean abs error | P99 abs error | Mean pct error | P99 pct error | RMSE | Cosine similarity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 2.00 MiB | 0 | 0 | 0% | 0% | 0 | 1.00000000 |
| 1 | 15 | 1.88 MiB | 0.00224624 | 0.015625 | 0.281576% | 0.775194% | 0.00406796 | 0.99999571 |
| 2 | 14 | 1.75 MiB | 0.00671938 | 0.046875 | 0.842384% | 2.290077% | 0.01071140 | 0.99997777 |
| 3 | 13 | 1.62 MiB | 0.01556845 | 0.09375 | 1.950904% | 5.185185% | 0.02374791 | 0.99990535 |
| 4 | 12 | 1.50 MiB | 0.03277579 | 0.171875 | 4.108308% | 10.489511% | 0.04896548 | 0.99961650 |

## Quantization Benchmark Results

The following results summarize the benchmarked impact of quantization on cache compression, space savings, latency, and throughput.

### Serving Benchmark

Workload: Qwen2.5-7B, 3200-token prefix, 100 queries, 16 GB DDR pool.

| Quant | Compression ratio | Space saved | Mean TTFT | Output throughput |
|---:|---:|---:|---:|---:|
| 0 | 1.27x | 21.46% | 254.86 ms | 82.70 tok/s |
| 1 | 1.38x | 27.33% | 258.16 ms | 82.65 tok/s |
| 2 | 1.50x | 33.42% | 259.48 ms | 83.15 tok/s |
| 4 | 1.87x | 46.53% | 244.02 ms | 84.09 tok/s |

### Capacity Benchmark

| Quant | Measured ratio | Compressed size | Space saved | Warm-hit speedup |
|---:|---:|---:|---:|---:|
| 0 | 2.58x | 1.36 MB | ~61% | 1.23x |
| 1 | 2.79x | 1.25 MB | ~64% | 1.25x |
| 2 | 3.06x | 1.15 MB | ~67% | 1.25x |
| 4 | 3.76x | 0.93 MB | ~73% | 1.36x |
| 8 | 8.61x | 0.41 MB | ~95% | 1.46x |

### Combined Space And Error View

The serving and capacity benchmarks measured cache density and latency. The `kv_tensor.pt` analysis measured quantization error. Rows below combine matching quant levels.

| Quant | Serving space saved | Serving compression | Capacity space saved | Capacity ratio | Mean pct error | P99 pct error | Cosine similarity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21.46% | 1.27x | ~61% | 2.58x | 0% | 0% | 1.00000000 |
| 1 | 27.33% | 1.38x | ~64% | 2.79x | 0.281576% | 0.775194% | 0.99999571 |
| 2 | 33.42% | 1.50x | ~67% | 3.06x | 0.842384% | 2.290077% | 0.99997777 |
| 4 | 46.53% | 1.87x | ~73% | 3.76x | 4.108308% | 10.489511% | 0.99961650 |
| 8 | Not run in serving table | Not run in serving table | ~95% | 8.61x | Not measured | Not measured | Not measured |

Result summary: compression and space saving increased monotonically with quantization level. `QUANT=4` was the strongest practical high-density setting, reaching ~73% capacity-benchmark space saving with no serving-latency penalty. `QUANT=8` reached ~95% capacity-benchmark space saving, but tensor error was not measured and should be treated as high fidelity risk until measured.

## Full Error Metrics

| Quant bits | Max abs error | Mean abs error | P50 abs error | P95 abs error | P99 abs error | Max pct error | Mean pct error | P50 pct error | P95 pct error | P99 pct error | MSE | RMSE | Cosine similarity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0% | 0% | 0% | 0% | 0% | 0 | 0 | 1.00000000 |
| 1 | 0.03125 | 0.00224624 | 0 | 0.0078125 | 0.015625 | 0.775194% | 0.281576% | 0% | 0.729927% | 0.775194% | 0.00001655 | 0.00406796 | 0.99999571 |
| 2 | 0.09375 | 0.00671938 | 0.00390625 | 0.0234375 | 0.046875 | 2.290077% | 0.842384% | 0.775194% | 2.040816% | 2.290077% | 0.00011473 | 0.01071140 | 0.99997777 |
| 3 | 0.21875 | 0.01556845 | 0.0078125 | 0.0546875 | 0.09375 | 5.185185% | 1.950904% | 1.886792% | 4.402516% | 5.185185% | 0.00056396 | 0.02374791 | 0.99990535 |
| 4 | 0.46875 | 0.03277579 | 0.02148438 | 0.109375 | 0.171875 | 10.489511% | 4.108308% | 4.000000% | 8.571429% | 10.489511% | 0.00239762 | 0.04896548 | 0.99961650 |

## Interpretation

| Quant bits | Interpretation |
|---:|---|
| 0 | Baseline. No quantization and no data loss. |
| 1 | Very small loss. Mean percentage error is about `0.28%`, and cosine similarity remains almost exactly `1.0`. |
| 2 | Practical BF16 setting. Mean percentage error is about `0.84%`, P99 percentage error is about `2.29%`, and cosine similarity remains very high at `0.99997777`. |
| 3 | Moderate truncation. Mean percentage error rises to about `1.95%`, with P99 percentage error around `5.19%`. |
| 4 | Aggressive truncation. Mean percentage error rises to about `4.11%`, with P99 percentage error around `10.49%`. This may improve compression but has visibly higher precision loss. |

For this BF16 KV tensor, `2` truncated bits is the best tradeoff from these results: it introduces measurable compression-friendly truncation while preserving strong numerical similarity.

## Reproduction Commands

Run from the `examples` directory:

```bash
python3 test_quantization_error.py \
  --tensor-path ./kv_tensor.pt \
  --bits 0 \
  --csv-out ./quant_values_bits0.csv \
  --json-out ./quant_summary_bits0.json

python3 test_quantization_error.py \
  --tensor-path ./kv_tensor.pt \
  --bits 1 \
  --csv-out ./quant_values_bits1.csv \
  --json-out ./quant_summary_bits1.json

python3 test_quantization_error.py \
  --tensor-path ./kv_tensor.pt \
  --bits 2 \
  --csv-out ./quant_values_bits2.csv \
  --json-out ./quant_summary_bits2.json

python3 test_quantization_error.py \
  --tensor-path ./kv_tensor.pt \
  --bits 3 \
  --csv-out ./quant_values_bits3.csv \
  --json-out ./quant_summary_bits3.json

python3 test_quantization_error.py \
  --tensor-path ./kv_tensor.pt \
  --bits 4 \
  --csv-out ./quant_values_bits4.csv \
  --json-out ./quant_summary_bits4.json
```

To test a real KV tensor instead of synthetic data, pass an existing `.pt` or `.pth` tensor file:

```bash
python3 test_quantization_error.py \
  --tensor-path ./kv_tensor.pt \
  --bits auto \
  --csv-out ./kv_quant_values.csv \
  --json-out ./kv_quant_summary.json
```

## Notes

Percentage error can be unstable for values close to zero, so `MAE`, `RMSE`, percentile errors, and cosine similarity should be considered together. For end-to-end model quality, this numeric report should be paired with generation-quality checks on cached KV reuse.

## Final Conclusion

The quantization results show a clear tradeoff: increasing `KVCLIP_CACHE_COMPRESSION_QUANT` improves cache compression and space savings, but also increases numerical error in the KV tensor.

For BF16 KV cache, `QUANT=2` is the best practical balance from these results. In the serving benchmark, `QUANT=2` reached `1.50x` compression, `33.42%` space saved, `259.48 ms` mean TTFT, and `83.15 tok/s` output throughput. In the tensor error test, it kept mean percentage error to `0.842%`, P99 percentage error to `2.29%`, RMSE to `0.0107`, and cosine similarity to `0.99997777`.

`QUANT=1` is safer and has very low error, but gives smaller compression improvement. `QUANT=4` gives stronger space savings, including `46.53%` saved in the serving benchmark and about `~73%` saved in the capacity benchmark, but the tensor error rises sharply: mean percentage error reaches `4.11%` and P99 percentage error reaches `10.49%`. This may be acceptable for aggressive capacity testing, but it carries higher model-quality risk.

`QUANT=8` had the best capacity result, with about `~95%` space saved and `8.61x` measured compression ratio, but tensor error was not measured for this level. It should be treated as experimental or high-risk until both numerical error and generation quality are validated.

Recommended default: use `KVCLIP_CACHE_COMPRESSION_QUANT=2` for BF16 KV cache. Use `QUANT=4` only when cache capacity is more important than fidelity, and validate model output quality before relying on it.