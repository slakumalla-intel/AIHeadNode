# Quick Start: Node Validation Tests

## 30-Second TL;DR

```powershell
cd AIHeadNode
pytest -m node_validation -vv
```

Check the generated `TESTCASE_EXECUTION_FINAL_REPORT.md` for results.

---

## What Gets Tested?

✓ GPU enumeration & PCIe topology  
✓ PCIe H2D/D2H bandwidth (all GPUs)  
✓ Pinned vs pageable memory performance  
✓ GPU-to-GPU P2P bandwidth  
✓ Kernel launch latency  
✓ Sync overhead  
✓ Power stability  
✓ NCCL initialization  
✓ Basic inference/training workloads

---

## Report Location

**Main Report**: `TESTCASE_EXECUTION_FINAL_REPORT.md`  
**Auto-updated**: Every time you run pytest for this repository

---

## Key Test Markers

Run tests by category:

```powershell
pytest -m gpu_enumeration -vv          # GPU discovery
pytest -m pcie_validation -vv          # PCIe bandwidth
pytest -m p2p_validation -vv           # GPU P2P
pytest -m nccl_validation -vv          # NCCL checks
./run_workload_acceptance.sh           # Linux workload acceptance (large models enabled by default)
./run_workload_acceptance.sh --skip-large-models   # Disable 70B/72B workloads
```

Linux workload defaults:
- `HF_ENABLE_LARGE_MODELS=1`
- `HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1`
- `NODE_WORKLOAD_AUTO_INSTALL=1` (with `NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=1800`)

---

## Expected Pass Criteria

| Test | Min Threshold | Typical Value |
|---|---|---|
| PCIe H2D BW | 41 GB/s (65% of 63) | 55–62 GB/s |
| PCIe D2H BW | 41 GB/s (65% of 63) | 55–62 GB/s |
| NVLink P2P BW | 240 GB/s | 350–400 GB/s |
| Kernel Launch Latency (p50) | < 100 µs | 1–10 µs |
| Power Variance | < 50W | < 10W |

---

## Files Modified

1. **tests/test_node_validation.py** — NEW test suite
2. **tests/conftest.py** — Updated for dual-report generation
3. **pytest.ini** — Added node_validation markers
4. **TESTCASE_EXECUTION_FINAL_REPORT.md** — consolidated execution report
5. **NODE_VALIDATION_IMPLEMENTATION.md** — This guide

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "nvidia-smi not found" | Install NVIDIA drivers |
| "torch not available" | `pip install torch` |
| P2P tests skip | Need 2+ GPUs on same PCIe switch |
| Power tests fail | Reduce background load |

---

## Separate from Bottleneck Tests

- Node validation: `pytest -m node_validation`
- Bottleneck analysis: `pytest -m bottleneck`
- Both: `pytest` (runs everything)

Each generates its own report file.

---

## Next: Run Your First Test

```powershell
cd c:\Users\slakumal\OneDrive\ -\ Intel\ Corporation\Documents\GitHub\AIHeadNode
pytest tests/test_node_validation.py::TestGPUEnumerationAndTopology -vv
```

Then:
```powershell
pytest -m node_validation -vv
```

View `TESTCASE_EXECUTION_FINAL_REPORT.md` for results.
