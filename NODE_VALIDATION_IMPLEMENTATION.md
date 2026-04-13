# Node Validation Test Suite — Implementation Summary

## Overview

A comprehensive node validation and bring-up test suite has been added to validate hardware configuration, interconnects, and system readiness. These tests are separate from the existing bottleneck analysis suite and generate their own dedicated report.

## Files Created and Modified

### 1. **tests/test_node_validation.py** (NEW)
Comprehensive test suite covering:
- **GPU Enumeration & PCIe Topology** (TC-NV-01)
- **PCIe Bandwidth Validation** (TC-NV-02)
- **Pinned vs Pageable Memory** (TC-NV-03)
- **GPU P2P Bandwidth** (TC-NV-04)
- **CPU↔GPU Launch & Sync Overhead** (TC-NV-05)
- **CPU↔GPU Power & Throttling** (TC-NV-06)
- **NCCL Initialization & Validation** (TC-NV-07)
- **Workload Acceptance Tests** (TC-NV-08)

**Test Classes**:
```
TestGPUEnumerationAndTopology
TestPCIeBandwidthDirect
TestPinnedVsPageableMemory
TestGPUP2PBandwidth
TestCPUGPULaunchAndSyncOverhead
TestCPUGPUPowerAndThrottling
TestNCCLInitializationAndValidation
TestWorkloadAcceptanceBasic
```

**Key Features**:
- Graceful handling of missing dependencies (torch, NCCL, NVIDIA drivers)
- Standalone functions for external integration
- Comprehensive logging of measured values
- Configurable thresholds and measurement parameters

### 2. **tests/conftest.py** (MODIFIED)
Updated pytest configuration to support dual-report generation:
- Added `_NODE_VALIDATION_REPORT_PATH` for node validation tests
- Added `_node_validation_report_rows` dictionary to track node validation tests separately
- Updated `pytest_runtest_makereport()` to detect `node_validation` marker and route tests to correct report
- Updated `pytest_sessionfinish()` to write both reports
- Added new marker definitions in `pytest_configure()`

### 3. **pytest.ini** (MODIFIED)
Added new test markers:
- `node_validation`: Main marker for all node validation tests
- `gpu_enumeration`: GPU discovery and topology
- `pcie_validation`: PCIe bandwidth tests
- `memory_validation`: Memory type comparison
- `p2p_validation`: GPU P2P connectivity
- `sync_overhead`: Launch and sync latency
- `power_validation`: Power and thermal stability
- `nccl_validation`: NCCL initialization
- `workload_acceptance`: Inference/training acceptance

### 4. **NODE_VALIDATION_REPORT.md** (NEW)
Comprehensive template report with:
- Auto-populated highlights (pass/fail/skip counts)
- Execution status table
- Test category descriptions and expected outputs
- Common failure modes and remediation
- Execution command examples
- Interpretation guide

## Test Coverage Matrix

| Test Category | Sub-Tests | Status | Purpose |
|---|---|---|---|
| GPU Enumeration | GPU count • PCIe topology • Topology consistency | Functional | Verify GPU discovery and PCIe topology correctness |
| PCIe Bandwidth (Direct) | H2D bandwidth all GPUs • D2H bandwidth all GPUs | Functional | Validate PCIe Gen5 BW ≥ 65% theoretical (63 GB/s) |
| Pinned vs Pageable | CE performance • Pinned vs pageable comparison | Functional | Compare DMA engine copy performance |
| GPU P2P | P2P BW all pairs • NVLink/PCIe validation | Functional | Validate GPU inter-connect bandwidth |
| Launch & Sync | Launch latency distribution • Sync overhead | Functional | Measure CPU-GPU control-plane latency |
| Power & Thermal | Sustained power profile • Power stability/jitter | Functional | Verify stable power under workload |
| NCCL | NCCL availability • NVLink status • Init variance | Functional | Validate NCCL readiness and determinism |
| Workload Acceptance | Inference latency • Training throughput | Functional | Verify system ready for real workloads |

## Execution Commands

### Run ALL node validation tests:
```powershell
cd "c:\Users\slakumal\OneDrive - Intel Corporation\Documents\GitHub\AIHeadNode"
pytest -m node_validation -vv
```

### Run specific categories:

**GPU Enumeration & Topology:**
```powershell
pytest -m gpu_enumeration -vv
```

**PCIe Bandwidth Validation:**
```powershell
pytest -m pcie_validation -vv
```

**Memory Type Validation:**
```powershell
pytest -m memory_validation -vv
```

**GPU P2P Bandwidth:**
```powershell
pytest -m p2p_validation -vv
```

**CPU↔GPU Overhead:**
```powershell
pytest -m sync_overhead -vv
```

**Power & Thermal Tests:**
```powershell
pytest -m power_validation -vv
```

**NCCL Validation:**
```powershell
pytest -m nccl_validation -vv
```

**Workload Acceptance:**
```powershell
./run_workload_acceptance.sh
# Skip 70B/72B models intentionally:
./run_workload_acceptance.sh --skip-large-models
```

Default workload behavior (Linux path):
- `HF_ENABLE_LARGE_MODELS=1`
- `HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1`
- `NODE_WORKLOAD_AUTO_INSTALL=1` (`NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=1800`)

### Run with custom output:
```powershell
# Run with detailed output and custom log file
pytest -m node_validation -vv -s --log-file=node_validation.log

# Run specific test class
pytest tests/test_node_validation.py::TestGPUEnumerationAndTopology -vv

# Run specific test method
pytest tests/test_node_validation.py::TestPCIeBandwidthDirect::test_h2d_bandwidth_all_gpus -vv

# Run with timeout (in case of hanging)
pytest -m node_validation --timeout=300 -vv
```

### Run BOTH suites (node validation + bottleneck analysis):
```powershell
# This will generate both reports: NODE_VALIDATION_REPORT.md and TESTCASE_EXECUTION_FINAL_REPORT.md
pytest tests/test_node_validation.py tests/test_bottleneck_analysis.py -vv
```

## Report Files

### NODE_VALIDATION_REPORT.md
- **Location**: `AIHeadNode/NODE_VALIDATION_REPORT.md`
- **Updated**: Automatically after each `pytest -m node_validation` run
- **Content**: GPU/PCIe/bandwidth/power/NCCL tests only
- **Columns**: Test | Purpose | Result | Status | Analysis

### TESTCASE_EXECUTION_FINAL_REPORT.md
- **Location**: `AIHeadNode/TESTCASE_EXECUTION_FINAL_REPORT.md`
- **Updated**: Automatically after each `pytest` run (excluding node_validation tests)
- **Content**: Bottleneck analysis, roofline, multi-GPU scaling, etc.
- **Columns**: Test | Purpose | Result | Status | Analysis

## How It Works

### Test Routing Logic
1. When pytest runs test_node_validation.py, each test has `@pytest.mark.node_validation`
2. The `pytest_runtest_makereport()` hook detects this marker
3. Test results go to `config._node_validation_report_rows` (instead of `config._exec_report_rows`)
4. On session finish, both report dictionaries are written to separate files

### Graceful Degradation
Tests use optional imports and skip/warn appropriately:
```python
if not _HAS_TORCH:
    pytest.skip("torch not available")
```

This means:
- If CUDA/torch missing → GPU tests skip gracefully
- If NCCL missing → NCCL tests skip (but don't fail)
- If nvidia-smi unavailable → enumeration tests skip (but don't fail)

## Test Output Interpretation

### PASS ✓
- Test assertion succeeded
- Measured value met threshold
- Example: "GPU 0 H2D BW: 42.1 GB/s (66.8% of 63 GB/s)" ✓

### FAIL ✗
- Test assertion failed
- Measured value below threshold
- Example: "GPU 0 H2D BW (35.2 GB/s) below threshold 40.95 GB/s" ✗

### SKIP ⊘
- Test skipped due to missing prerequisites
- Example: "Need ≥ 2 GPUs for P2P test" or "torch not available"

## Typical Baseline Values

For H100 SXM5 systems:

| Measurement | Target | Min (60% of Gen5 nominal) |
|---|---|---|
| PCIe Gen5 H2D BW | 63 GB/s | 41 GB/s |
| PCIe Gen5 D2H BW | 63 GB/s | 41 GB/s |
| NVLink P2P BW | 400 GB/s | 240 GB/s |
| Kernel Launch Latency (p50) | 1-10 µs | < 100 µs |
| Sync Overhead (p95) | 1-50 µs | < 1000 µs |
| Power Variance (std) | < 10W | < 50W |

## Troubleshooting

### Issue: Tests fail with "nvidia-smi not found"
**Solution**: Ensure CUDA drivers installed and NVIDIA Management Library available

### Issue: "torch not available"
**Solution**: Install PyTorch: `pip install torch`

### Issue: P2P tests fail or skip
**Solution**: 
- Need ≥ 2 GPUs on same PCIe switch
- Verify P2P access: Check with `nvidia-smi`

### Issue: Power/thermal tests show high variance
**Solution**: Reduce background system load; disable CPU boost/turbo during measurement

### Issue: NCCL tests skip
**Solution**: Optional; NCCL not required for basic node validation. Install if needed: `pip install nccl`

## Integration with CI/CD

### GitHub Actions Example:
```yaml
- name: Run Node Validation Tests
  run: |
    pytest tests/test_node_validation.py -m node_validation -v \
      --tb=short --junit-xml=node-validation-results.xml
      
- name: Upload Report
  uses: actions/upload-artifact@v3
  with:
    name: node-validation-report
    path: NODE_VALIDATION_REPORT.md
```

### Jenkins Example:
```groovy
stage('Node Validation') {
  steps {
    sh 'pytest -m node_validation -vv --json-report --json-report-file=node-validation.json'
    publishHTML([
      reportDir: 'htmlcov',
      reportFiles: 'NODE_VALIDATION_REPORT.md',
      reportName: 'Node Validation Report'
    ])
  }
}
```

## Next Steps

1. **Run the suite for the first time**:
   ```powershell
   pytest -m node_validation -vv
   ```

2. **Review NODE_VALIDATION_REPORT.md** for any failures

3. **Run specific failed categories** for deeper investigation:
   ```powershell
   pytest -m pcie_validation -vv -s
   ```

4. **Document baseline values** for your specific node in a README or wiki

5. **Integrate into CI/CD** to automatically validate every new deployment

---

**Test Suite Version**: v1.0
**Created**: April 2026
**Hardware Target**: 160-core CPU + 4× H100 SXM5
