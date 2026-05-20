# Performance Test Suite - Complete Report Index
**AI Head Node Validation | Generated: May 19, 2026**

---

## 📋 Quick Navigation

### Start Here
- **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** ⭐ **START HERE** - 5-minute overview of key findings and recommendations

### Detailed Analysis
- **[COMPARISON_TABLES.md](COMPARISON_TABLES.md)** - All 26 test cases with side-by-side metrics
- **[comparison_dashboard.html](comparison_dashboard.html)** - Interactive visualization (open in browser)
- **[KPI_ANALYSIS.md](KPI_ANALYSIS.md)** - Statistical metrics (min/max/percentiles per case)
- **[PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)** - Full test definitions and detailed analysis

### Raw Data
- **performance_suite_kpis.jsonl** - 154 JSONL records with complete metrics for all test cases

---

## 🎯 Report at a Glance

### Test Results Summary
| Metric | Value |
|--------|-------|
| **Total Cases** | 26 |
| **Passed** | 17 ✅ |
| **Failed** | 3 ❌ (legacy CLI args) |
| **Planned** | 6 📋 |
| **Pass Rate** | 85% |

### Key Performance Findings
| Subsystem | Status | Key Metric |
|-----------|--------|-----------|
| **LLM Inference** | ✅ | FP8 achieves 50% memory savings with no throughput penalty |
| **Collective Comm.** | ✅ | 13-15 TB/s sustained bandwidth across NCCL operations |
| **GPU P2P** | ✅ | Multi-stream and bandwidth paths validated |
| **CPU** | ✅ | Stable compute performance (4.3-6.2s GEMM) |

---

## 📊 How to Use This Report

### For Executives/Decision Makers
**Read:** [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) (5 min)
- What's working, what needs fixing
- Impact on production deployments
- Immediate action items

### For Performance Engineers
**Read:** 
1. [COMPARISON_TABLES.md](COMPARISON_TABLES.md) (15 min) - Detailed test results
2. Open [comparison_dashboard.html](comparison_dashboard.html) in browser (10 min) - Visual analysis
3. [KPI_ANALYSIS.md](KPI_ANALYSIS.md) (10 min) - Statistical breakdown

**Key Questions Answered:**
- Which tests passed? Which failed and why?
- What are the performance characteristics?
- Are there any bottlenecks?
- What's production-ready vs needs work?

### For Developers/DevOps
**Read:**
1. [COMPARISON_TABLES.md](COMPARISON_TABLES.md) - Failure root causes
2. [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md#recommendations) - Immediate fixes needed
3. Source code: `tests/test_performance_suite.py` - Implementation details

**Action Items:**
- Fix cases 007/009: vLLM legacy argument translation
- Fix case 011: genai-perf flag compatibility
- Enable continuous regression testing

### For Integration Teams
**Reference:**
- [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) - Full test definitions and data models
- [comparison_dashboard.html](comparison_dashboard.html) - Baseline metrics for comparison
- performance_suite_kpis.jsonl - Raw data for custom analysis

---

## 📈 Test Categories Covered

### LLM Inference (Cases 003, 006, 008, 026)
- ✅ Full precision (BF16) serving
- ✅ Quantized (FP8) serving with memory optimization
- ✅ Long-context support (up to 32k tokens)
- ❌ Legacy traffic patterns (cases 007, 009)
- ❌ GenAI-Perf HTTP load (case 011)

### Collective Communication (Cases 018-025)
- ✅ AllReduce (primary all-reduce path)
- ✅ AllGather (embeddings/gather-heavy workloads)
- ✅ ReduceScatter (ZeRO/FSDP gradient synchronization)
- ✅ AllToAll (shuffle-heavy communication)
- ✅ P2P optimizations for reduce/scatter

### GPU Performance (Cases 004, 005)
- ✅ Multi-stream P2P bandwidth
- ✅ Bandwidth sweep validation
- ✅ H100 PCIe interconnect health

### CPU Performance (Case 002)
- ✅ GEMM compute validation
- ✅ Stability and consistency checks

### Future/Planned (Cases 001, 010, 012-017)
- 📋 CPU tokenizer micro (case 001)
- 📋 TensorRT-LLM offline throughput (case 010)
- 📋 DeepSeek KV cache with SATA offload (case 016)
- 📋 DeepSeek KV cache with CXL offload (case 017)
- 📋 Reserved placeholders (cases 012-015)

---

## 🔧 Understanding Test Failures

### Legacy CLI Compatibility Issues (Cases 007, 009)
```
Error: vllm bench serve: unrecognized arguments: --request-rate burst
```
- **Impact:** Non-critical (vLLM serving still works for standard workloads)
- **Status:** Fixable through command normalization
- **Timeline:** Address in next harness update

### GenAI-Perf Version Mismatch (Case 011)
```
Error: genai-perf: error: unrecognized arguments: --service-kind openai
```
- **Impact:** Cannot validate containerized HTTP load generation path
- **Status:** Requires flag mapping or version compatibility layer
- **Timeline:** Low priority (not on critical path)

---

## 📊 How Metrics Are Calculated

### Bandwidth Metrics
- Extracted from nvbandwidth and NCCL test stdout
- Reported in GB/s or TB/s
- Represents sustained bandwidth across all message sizes

### Throughput Metrics (vLLM)
- Tokens per second (tok/s)
- Measured under specified concurrency and batch size
- Varies with quantization and context length

### Latency Metrics
- End-to-end latency (E2EL)
- Time-to-first-token (TTFT)
- Percentiles: P50, P99, P99.9

### Execution Time
- Wall-clock time for entire test (seconds)
- Includes setup, warmup, and actual benchmark runs

---

## 💾 Data Handling

### Artifact Location
```
artifacts/
├── EXECUTIVE_SUMMARY.md            # This report (updated May 19)
├── COMPARISON_TABLES.md             # Detailed metrics (updated May 19)
├── comparison_dashboard.html        # Interactive dashboard (updated May 19)
├── KPI_ANALYSIS.md                 # Statistical analysis
├── PERFORMANCE_REPORT.md           # Full definitions
└── performance_suite_kpis.jsonl    # Raw 154-record JSONL file
```

### JSONL Record Format
```json
{
  "timestamp": "2026-05-19T14:23:45",
  "case_id": "AI_HOSTCPU_PERF_TEST_ID_003",
  "metrics": {
    "mode": "execution",
    "status": "passed",
    "title": "vLLM Serve + Load (BF16 stress) - 70B",
    "subsystem": "Inference (BF16)",
    "commands": [...],
    "results": [
      {
        "command": "vllm serve ...",
        "executed_command": "vllm serve ...",
        "returncode": 0,
        "elapsed_sec": 66.091,
        "stdout_tail": "...",
        "stderr_tail": ""
      }
    ]
  }
}
```

---

## 🚀 Reproducing Results

### Quick Run (Catalog Only - No Execution)
```bash
cd AIHeadNode
pytest tests/test_performance_suite.py -v
```
Output: Test definitions validated, no commands executed

### Full Run (With Execution)
```bash
export PERF_EXECUTE_CASES=1
export PERF_AUTO_INSTALL=1
pytest tests/test_performance_suite.py -v
```
Output: Full execution with metrics recorded to JSONL

### Specific Cases Only
```bash
export PERF_CASE_IDS="AI_HOSTCPU_PERF_TEST_ID_003,AI_HOSTCPU_PERF_TEST_ID_026"
export PERF_EXECUTE_CASES=1
pytest tests/test_performance_suite.py -v
```
Output: Run only specified cases (e.g., vLLM BF16 and FP8)

### Environment Variables
```bash
export PERF_CUDA_VISIBLE_DEVICES=0,1,2,3       # GPU selection
export PERF_NUM_GPUS=4                          # GPU count
export PERF_EXECUTE_CASES=1                     # Enable execution
export PERF_AUTO_INSTALL=1                      # Auto-install missing deps
export PERF_VLLM_BENCH_NUM_PROMPTS_CAP=500      # Cap prompts
export PERF_CMD_TIMEOUT_SEC=1800                # Timeout (30 min)
```

---

## ✅ Deployment Readiness Checklist

### Production Ready ✓
- [x] vLLM FP8 inference serving
- [x] Collective communication (NCCL)
- [x] GPU P2P functionality
- [x] CPU compute validation

### Needs Minor Fixes ⚠️
- [ ] Fix case 007 legacy burst traffic args
- [ ] Fix case 009 legacy poisson traffic args
- [ ] Fix case 011 genai-perf flag compatibility

### Future Work 📋
- [ ] Validate TensorRT-LLM offline throughput (case 010)
- [ ] Test DeepSeek KV cache with SATA offload (case 016)
- [ ] Test DeepSeek KV cache with CXL offload (case 017)

---

## 📞 Support & Questions

### Common Questions

**Q: Why did case 011 fail?**
A: genai-perf CLI uses different flag names than the test definition. See [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md#case-011-genai-perf-cli-incompatibility).

**Q: Is the system production-ready?**
A: Yes for LLM inference, collective communication, and GPU operations. Pending minor CLI compatibility fixes.

**Q: How do I update the report?**
A: Run the test suite with `PERF_EXECUTE_CASES=1` and the JSONL will auto-update. Then regenerate from `scripts/analyze_kpis.py`.

**Q: Can I share this report?**
A: Yes. All reports in `artifacts/` are HTML/Markdown and can be shared or included in documentation.

---

## 📜 Version History

- **v1.0** (May 19, 2026) - Initial comprehensive report
  - 26 test cases analyzed
  - 17 passed, 3 failed, 6 planned
  - Executive summary, comparison tables, and interactive dashboard
  - Raw JSONL data with 154 records

---

**Last Updated:** May 19, 2026  
**Test Environment:** 4× NVIDIA H100 GPUs  
**NCCL Version:** 2.21  
**vLLM Version:** Latest  
**Data Source:** performance_suite_kpis.jsonl

---

## 📚 Additional Resources

- [Performance Test Suite Source](tests/test_performance_suite.py)
- [Test Configuration](tests/conftest.py)
- [vLLM Serving Guide](https://docs.vllm.ai)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [nvbandwidth GitHub](https://github.com/NVIDIA/nvbandwidth)

