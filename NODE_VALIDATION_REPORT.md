# AI Head Node — Node Validation Report

Generated: (auto-updated on pytest run)
Pytest command line: (auto-updated on pytest run)

## Highlights

- Total testcases captured: 0
- Passed: 0
- Failed: 0
- Skipped: 0
- Validation Status: Pending first test execution

## Validation Categories

This report covers node bring-up and validation across multiple subsystems:

### 1. GPU Enumeration and PCIe Topology
- **Purpose**: Verify GPU enumeration accuracy and PCIe topology discovery
- **Key Tests**: GPU count validation, PCIe link speed/width verification, topology matrix consistency
- **Expected Output**: List of all GPUs with PCI bus IDs, PCIe generation, and link widths

### 2. PCIe Bandwidth Validation
- **Purpose**: Validate Host↔Device (H2D/D2H) bandwidths for direct-attached and switch-attached GPUs
- **Key Tests**: H2D bandwidth ≥ 65% of 63 GB/s theoretical, D2H bandwidth ≥ 65% of theoretical
- **Expected Output**: Bandwidth measurements for each GPU across multiple transfer sizes

### 3. Pinned vs Pageable Memory
- **Purpose**: Compare DMA Copy Engine (CE) and SM copy performance with different memory types
- **Key Tests**: Pinned memory performance, pageable memory performance, speedup comparison
- **Expected Output**: Bandwidth comparison showing pinned > pageable by typical 1.5–2.5× factor

### 4. GPU P2P Data Bandwidth
- **Purpose**: Validate GPU-to-GPU P2P connectivity via NVLink or PCIe
- **Key Tests**: P2P bandwidth for all GPU pairs, verify minimum threshold performance
- **Expected Output**: NVLink: ~400 GB/s per direction; PCIe P2P: ~63 GB/s per direction

### 5. CPU↔GPU Launch and Sync Overhead
- **Purpose**: Measure kernel launch latency and synchronization overhead under idle and loaded conditions
- **Key Tests**: Kernel launch latency (min/p50/p95/p99/max), cudaDeviceSynchronize() overhead
- **Expected Output**: Launch latency: 1–10 µs typical; sync overhead: 1–50 µs typical

### 6. CPU↔GPU Power and Throttling
- **Purpose**: Verify stable power delivery and thermal stability under sustained GPU workload
- **Key Tests**: Power profile under sustained workload, jitter analysis, no thermal throttling
- **Expected Output**: Power variance < 50W, stable sustained power envelope

### 7. NCCL Initialization and Validation
- **Purpose**: Verify NCCL library availability and deterministic initialization
- **Key Tests**: NCCL version check, NVLink status, init variance over 50 runs
- **Expected Output**: NCCL available, NVLink functional, init variance < 50% of mean time

### 8. Workload Acceptance Tests
- **Purpose**: Run sample inference and training workloads to validate overall system readiness
- **Key Tests**: Simple inference latency (MLP), training throughput validation
- **Expected Output**: Inference latency < 100ms, training throughput > 1000 samples/sec

## Execution Status Table

| Test | Purpose | Result | Status | Analysis |
|---|---|---|---|---|
| (Pending) | (Pending) | (Pending) | (Pending) | Run pytest with `-m node_validation` to populate this table. |

## Execution Commands

### Run all node validation tests:
```bash
cd AIHeadNode
pytest -m node_validation -vv
```

### Run specific category (e.g., GPU enumeration only):
```bash
pytest -m node_validation -k "gpu_enumeration" -vv
```

### Run PCIe bandwidth validation:
```bash
pytest -m node_validation -k "pcie_validation" -vv
```

### Run P2P bandwidth tests:
```bash
pytest -m node_validation -k "p2p_validation" -vv
```

### Run power and thermal tests:
```bash
pytest -m node_validation -k "power_validation" -vv
```

### Run NCCL and inter-GPU tests:
```bash
pytest -m node_validation -k "nccl_validation" -vv
```

## Report Interpretation

### Status Meanings
- **PASS**: Test executed successfully and met all assertions
- **FAIL**: Test assertion failed; check the "Result" column for details
- **SKIP**: Test was skipped due to missing prerequisites (e.g., no NCCL, no dual GPU)

### Common Failure Reasons
1. **PCIe Bandwidth**: Host→Device or Device→Host BW below 65% threshold
   - **Action**: Check PCIe link speed/width via `nvidia-smi`; verify no PCIe switch or bifurcation issues
2. **P2P Bandwidth**: GPU-to-GPU BW below expected threshold
   - **Action**: Verify NVLink functional via `nvidia-smi nvlink -s`; check for proper GPU pairing
3. **Power Instability**: Power variance > 50W during sustained workload
   - **Action**: Check power delivery system, thermal paste, and cooling; verify no thermal throttling
4. **NCCL Init Variance**: Init time has > 50% variance
   - **Action**: Reduce background system load; recheck NCCL pins and process affinity

## Next Steps After Validation

If all tests **PASS**:
1. Proceed to workload bring-up (inference serving, training)
2. Run full benchmark suite (tests/test_bottleneck_analysis.py)
3. Validate performance against published baselines

If any tests **FAIL**:
1. Document failure details in ticket/issue
2. Contact hardware vendor for diagnostics
3. Check system logs for thermal throttling, power events, or CUDA errors

---

**Generated by**: pytest node_validation test suite
**Last Run**: (auto-updated)
