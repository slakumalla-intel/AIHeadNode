# Performance Suite Analysis Report

Generated from `artifacts/reports/logs/performance_suite.log`

## Executive View

- Performance suite execution showed broad functional coverage across inference, tokenizer, host-device transfer, and NCCL collectives.
- The strongest results came from serve-and-benchmark inference runs and large-message collective bandwidth sweeps.
- The clean-pass story is limited by one hard tool-compatibility failure, two endpoint-availability skips, one missing dependency, and repeated stale vLLM worker cleanup warnings.

## Overall Assessment

- Stable inference runs completed for Llama 8B BF16, Llama 70B BF16, Llama 70B FP8, Mixtral FP8, DeepSeek-R1 offload variants, and the 70B FP8 stress case.
- NCCL collective tests completed successfully and showed healthy large-message bandwidth across AllReduce, ReduceScatter, AllGather, and AllToAll.
- Operational reliability remains below production-ready because the suite still depends on fragile local endpoint assumptions, a broken `genai-perf` CLI invocation, and incomplete post-test process cleanup.

## Good Data

### Inference Throughput and Latency

Representative successful serve benchmarks from the log:

| Case | Workload | Output tok/s | P99 TTFT (ms) | Notes |
|---|---|---:|---:|---|
| AI_HOSTCPU_PERF_TEST_ID_003 | Llama 70B BF16 baseline | 121.35 | 54566.34 | Stable but slow tail latency at 64 concurrency |
| AI_HOSTCPU_PERF_TEST_ID_004 | Llama 8B BF16 | 428.31 | 2613.05 | Best latency profile among logged inference runs |
| AI_HOSTCPU_PERF_TEST_ID_005 | Llama 70B FP8 | 392.75 | 6535.19 | Strong throughput for 70B FP8 |
| AI_HOSTCPU_PERF_TEST_ID_006 | Llama 70B BF16 long-context | 337.29 | 6764.58 | Lower throughput than FP8 as expected |
| AI_HOSTCPU_PERF_TEST_ID_008 | Mixtral 8x22B FP8 | 419.60 | 4183.21 | Healthy MoE baseline |
| AI_HOSTCPU_PERF_TEST_ID_016 | DeepSeek-R1 SATA offload | 394.96 | 6536.11 | Offload path remained stable |
| AI_HOSTCPU_PERF_TEST_ID_017 | DeepSeek-R1 CXL offload | 392.33 | 6576.02 | Similar profile to SATA offload |
| AI_HOSTCPU_PERF_TEST_ID_026 | Llama 70B FP8 stress | 388.89 | 6280.82 | Stress case still completed cleanly |

Key inference observations:

- The 8B BF16 run produced the best tail latency and the highest single-model output throughput in the log.
- 70B FP8 and DeepSeek runs clustered around roughly 389 to 395 output tok/s, which indicates the platform can sustain large-model serving when startup succeeds.
- The 70B BF16 baseline completed reliably but showed materially worse throughput than the FP8 path, which is directionally correct.

### Interconnect and Collective Health

Representative NCCL bandwidth signals:

| Case | Test | Mean bus bandwidth | Peak bus bandwidth |
|---|---|---:|---:|
| AI_HOSTCPU_PERF_TEST_ID_018 | AllReduce sweep | 15076750.067 | 60000000.0 |
| AI_HOSTCPU_PERF_TEST_ID_021 | ReduceScatter | 13407556.8 | 60000000.0 |
| AI_HOSTCPU_PERF_TEST_ID_022 | AllGather | 14206755.6 | 60000000.0 |
| AI_HOSTCPU_PERF_TEST_ID_023 | AllToAll | 14420167.967 | 60000000.0 |
| AI_HOSTCPU_PERF_TEST_ID_025 | AllReduce with P2P enabled | 14242982.833 | 60000000.0 |

Important topology signal:

- Disabling P2P reduced AllReduce mean bus bandwidth from 14242982.833 in AI_HOSTCPU_PERF_TEST_ID_025 to 6757240.067 in AI_HOSTCPU_PERF_TEST_ID_024.
- That drop confirms the suite can distinguish between preferred and degraded communication paths, which is useful for topology validation.

### Host-Side Coverage

- Tokenizer throughput and host-to-device / device-to-host bandwidth commands completed without failure signatures.
- nvbandwidth execution completed cleanly and showed consistent transfer behavior across CE and SM copy paths.

## Bad Data

### Hard Failure

| Case | Failure | Impact |
|---|---|---|
| AI_HOSTCPU_PERF_TEST_ID_011 | `genai-perf` rejects `--service-kind openai` | End-to-end HTTP benchmarking path is currently invalid in this environment |

Evidence from the log:

- `genai-perf: error: unrecognized arguments: --service-kind openai`
- The case exits almost immediately with return code 2, so this is a deterministic tooling or command-version mismatch rather than a load-related failure.

### Skipped or Blocked Executions

| Case | Reason | Impact |
|---|---|---|
| AI_HOSTCPU_PERF_TEST_ID_007 | Could not connect to `127.0.0.1:8000` | Mixed-prompt serving scenario was not actually exercised |
| AI_HOSTCPU_PERF_TEST_ID_009 | Could not connect to `127.0.0.1:8000` | Burst-traffic recovery scenario was not exercised |
| AI_HOSTCPU_PERF_TEST_ID_010 | `trtllm-bench` missing and auto-install failed | TRT-LLM offline throughput path remains unvalidated |
| AI_HOSTCPU_PERF_TEST_ID_012-AI_HOSTCPU_PERF_TEST_ID_015 | Planned-only placeholders | Reserved IDs reduce effective executed coverage |

Additional evidence:

- Both skipped vLLM benchmark cases fail while fetching model metadata from the local server because no serving endpoint is available.
- The TRT-LLM auto-install path reports `ERROR: No matching distribution found for trtllm-bench`, so the suite cannot self-heal this dependency.

### Operational Reliability Warnings

- Multiple serve-based cases emit post-test cleanup warnings for stale vLLM worker PIDs.
- The warning appears after AI_HOSTCPU_PERF_TEST_ID_003, `_005`, `_006`, `_008`, `_016`, `_017`, and `_026`.
- The tests still complete, but stale workers create a risk of GPU-memory leakage, port conflicts, and cross-test contamination in longer soak runs.

### Configuration and Benchmarking Caveats

- Every `vllm bench serve` block warns that greedy decoding is no longer implicit and requires explicit `--temperature=0` for the old behavior.
- This does not fail the suite, but it weakens apples-to-apples comparison against older baseline data if historical runs assumed greedy decode.

## Failure Categories

| Category | Count |
|---|---:|
| Tool CLI incompatibility | 1 |
| Local endpoint unavailable | 2 |
| Missing external dependency | 1 |
| Planned-only placeholder coverage gaps | 4 |
| Post-test cleanup hygiene issue | 7 warnings |

## Top Risks

- The end-to-end benchmarking path is not trustworthy until the `genai-perf` command line is aligned with the installed version.
- Local-endpoint assumptions for standalone `vllm bench serve` tests make parts of the suite environment-sensitive rather than self-contained.
- Repeated stale vLLM worker warnings indicate cleanup is still incomplete under success paths, not just failure paths.
- Placeholder cases inflate nominal suite breadth without increasing executed validation coverage.

## Recommended Actions

1. Fix AI_HOSTCPU_PERF_TEST_ID_011 by detecting the installed `genai-perf` version and generating version-compatible CLI arguments instead of hard-coding `--service-kind openai`.
2. Convert AI_HOSTCPU_PERF_TEST_ID_007 and AI_HOSTCPU_PERF_TEST_ID_009 into self-contained tests that launch and validate their own serving endpoint before load generation, or explicitly preflight and skip before issuing benchmark traffic.
3. Decide whether TRT-LLM benchmarking is required in this environment; if yes, provision `trtllm-bench` out-of-band and verify the binary path rather than relying on pip auto-install.
4. Tighten vLLM post-test cleanup so success-path worker processes are fully reaped before the next case starts.
5. Make decode behavior explicit by setting `--temperature=0` where historical comparability matters.
6. Replace placeholder cases 012 to 015 with implemented workloads or remove them from pass-rate discussions until they execute real work.

## Bottom Line

The log supports a positive conclusion on core platform performance: large-model serving and NCCL collective paths are broadly healthy, and the platform can sustain demanding FP8 and BF16 workloads. The main problems are not raw performance bottlenecks in this log; they are harness reliability issues, dependency gaps, and one command-version mismatch that prevent the suite from being considered fully production-grade.