# AI HeadNode Validation — Executive Summary

- **Generated:** 2026-04-27
- **System:** sc00901168s0075 (4× H100 PCIe)
- **Source:** artifacts/reports/bugtool_testcase_report_new.csv

---

## Top-Line Results

| Metric | Count |
|---|---:|
| Total mapped tests | 285 |
| Executed | 246 |
| **PASS** | **152** |
| **FAIL** | **62** |
| SKIP | 32 |
| NOT_EXECUTED | 39 |
| Pass rate (executed) | **61.8%** |

---

## Suite-Level Summary

| Suite | Total | PASS | FAIL | SKIP | NOT_EXEC |
|---|---:|---:|---:|---:|---:|
| agentic_ai.xml | 9 | 8 | 1 | 0 | 0 |
| ai_pipeline.xml | 20 | 7 | 13 | 0 | 0 |
| bottleneck.xml | 20 | 16 | 4 | 0 | 0 |
| cpu.xml | 26 | 21 | 4 | 0 | 0 |
| gpu.xml | 21 | 19 | 2 | 0 | 0 |
| interconnect.xml | 7 | 2 | 0 | 4 | 0 |
| memory.xml | 20 | 16 | 4 | 0 | 0 |
| mlperf_suite.xml | 21 | 0 | 19 | 2 | 0 |
| node_validation.xml | 47 | 32 | 9 | 6 | 0 |
| performance_suite.xml | 25 | 10 | 0 | 15 | 0 |
| system_config.xml | 8 | 8 | 0 | 0 | 0 |
| system_health.xml | 19 | 12 | 4 | 3 | 0 |
| vllm_prefill.xml | 3 | 0 | 0 | 1 | 0 |
| *(not yet executed)* | 39 | — | — | — | 39 |

---

## Failure Root-Cause Breakdown

| Root Cause | # Failures | Impacted Suites |
|---|---:|---|
| MLPerf workload command errors (rc=2 / dataset missing) | 16 | mlperf_suite |
| MLPerf / vLLM timeouts (>600 s) | 4 | mlperf_suite, vllm_prefill |
| CUDA device-side assert (inference pipeline) | 8 | ai_pipeline |
| Performance / compute efficiency below threshold | 5 | bottleneck, cpu, ai_pipeline |
| Bandwidth below threshold | 9 | ai_pipeline, memory, node_validation |
| Assertion failures (correctness / config) | 15 | agentic_ai, ai_pipeline, cpu, memory, node_validation, system_health |
| Hardware / config issues (ECC, PCIe Gen5, bus collision) | 3 | gpu, interconnect |
| Code / config error (missing attribute) | 1 | cpu |
| Other | 1 | node_validation |

---

## Key Blockers

### BLK-1 — MLPerf Suite: All inference workloads failing (16/21 cases)
- **Symptom:** `rc=2` (missing datasets/models) or timeout >600 s
- **Affected:** AI_HOSTCPU_ML_TEST_ID_003 through ML_TEST_ID_021
- **Root cause:** MLCommons inference scripts missing datasets at `/root/workloads/AIHeadNode/datasets/` and model checkpoints not downloaded
- **Action:** Run `scripts/download_mlperf_datasets.sh` and `scripts/download_mlperf_models.sh` before re-executing

### BLK-2 — AI Pipeline: CUDA device-side assert cascades (8 tests)
- **Symptom:** `torch.AcceleratorError: CUDA error: device-side assert triggered` across all `TestInferencePipelineKPIs` tests
- **Affected:** AI_HOSTCPU_AGENTIC_TEST_ID_016–023
- **Root cause:** Single upstream assertion failure in `test_batched_inference_throughput` likely causing CUDA state corruption; all dependent inference tests cascade
- **Action:** Fix root assertion in TestInferencePipelineKPIs, then re-run suite

### BLK-3 — GPU Compute Efficiency Low (FP16/BF16/INT8 ~25–32%)
- **Symptom:** Efficiency well below 75% floor across all 4 GPUs
- **Affected:** test_gpu_validation (TestFP16ThroughputGPU, TestBF16Throughput, TestINT8Throughput)
- **Root cause:** H100 PCIe boards running under power cap — GPU0 at 99% power limit at rest (BLK-5). Clock throttling of 47% observed under load.
- **Action:** Resolve power limit / thermal throttle issue (BLK-5) first

### BLK-4 — vLLM Prefill: test_vllm_4h100_prefill_validation timeout
- **Symptom:** Timeout >600 s — pytest-timeout reached before vLLM process completed
- **Affected:** AI_HOSTCPU_AGENTIC_TEST_ID_188
- **Root cause:** vLLM 4× H100 prefill benchmark exceeds 600 s timeout; likely model load latency or CUDA initialisation taking too long under power throttle
- **Action:** Increase pytest timeout for this test, or resolve GPU power/clock issue first

### BLK-5 — GPU Power Throttling (Hardware / Thermal)
- **Symptom:**
  - `GPU0 at 99% power limit at rest` (345.9 W / 350.0 W limit)
  - SM clock dropped 47% under sustained load (1755 → 930 MHz)
- **Affected:** AI_HOSTCPU_FUNC_TEST_ID_027, AI_HOSTCPU_AGENTIC_TEST_ID_181
- **Root cause:** GPUs running near power limit even at idle; sustained compute triggers power-based clock throttle. Possibly: power delivery issue, incorrect power limit set, or cooling/ambient temperature
- **Action:** Check power delivery rail, verify power limit is set to max (e.g., `nvidia-smi -pl 400`), inspect chassis thermals

### BLK-6 — PCIe / NVLink Connectivity Issues
- **Symptom:**
  - No PCIe Gen 5 (32 GT/s) links found in sysfs
  - GPU 2 PCIe bus number collision (0x61)
  - NCCL P2P busbw 25.1 GB/s < 30 GB/s threshold
  - P2P bandwidth low: 17.3 GB/s on some pairs
- **Affected:** AI_HOSTCPU_AGENTIC_TEST_ID_126, AI_HOSTCPU_AGENTIC_TEST_ID_092, AI_HOSTCPU_FUNC_TEST_ID_008, AI_HOSTCPU_FUNC_TEST_ID_005
- **Root cause:** PCIe topology not Gen5; GPU 2 may have a bus enumeration issue in firmware/BIOS. P2P bandwidth degradation consistent with PCIe Gen4 or misconfigured NVLink
- **Action:** Check BIOS PCIe settings, verify GPU2 slot assignment, validate NVLink topology

### BLK-7 — Memory: RAM Capacity Just Under 1 TB Threshold
- **Symptom:** Total RAM 1007 GB < 1024 GB (1 TiB)
- **Affected:** AI_HOSTCPU_AGENTIC_TEST_ID_143
- **Root cause:** System has <1 TiB physical RAM or some DIMMs not detected / reserved by OS
- **Action:** Verify DIMM population in all slots; check `dmidecode --type 17` and BIOS memory settings

---

## Passed Highlights

- ✅ GPU enumeration and topology discovery — all 4 GPUs visible
- ✅ NCCL AllReduce, AllGather, ReduceScatter — within spec
- ✅ Host-to-Device PCIe bandwidth — within spec
- ✅ NVLink status — detected and enabled
- ✅ System config (BIOS, OS, driver, CUDA versions) — all pass
- ✅ GPU ECC DRAM (memory rows) — no uncorrectable errors
- ✅ BF16/FP8/INT8 numerical correctness — within error bounds
- ✅ Agentic AI latency tests — 8/9 passing
- ✅ CPU topology, affinity, scaling — passing
- ✅ Performance suite — 10/25 executed, 15 skipped (environment setup needed)

---

## Recommended Action Priority

| Priority | Blocker | Expected Recovery |
|---|---|---|
| P0 | BLK-5 — GPU power throttle | Fixes BLK-3, BLK-4 automatically |
| P0 | BLK-2 — CUDA device assert | Clears 8 ai_pipeline failures |
| P1 | BLK-1 — MLPerf datasets/models | Clears 16 mlperf failures |
| P1 | BLK-6 — PCIe/NVLink topology | Fixes interconnect + P2P failures |
| P2 | BLK-4 — vLLM timeout | After BLK-5 resolved |
| P2 | BLK-7 — RAM capacity | DIMM population check |
| P3 | BLK-6 — PCIe Gen5 detection | BIOS/firmware update |
