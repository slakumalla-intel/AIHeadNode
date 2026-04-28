# Quick Start: Node Validation (Linux)

## 1-Minute Start

```bash
cd AIHeadNode
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all_workloads.py --only node_validation
```

Check:

- `TESTCASE_EXECUTION_FINAL_REPORT.md`
- `artifacts/node_validation_kpis.jsonl`

## What This Covers

- GPU enumeration and PCIe topology checks
- PCIe host-device bandwidth validation
- pinned vs pageable transfer validation
- GPU peer-to-peer bandwidth checks
- launch and synchronization overhead checks
- power and thermal coupling checks
- NCCL validation checks
- workload acceptance tests (model- and dependency-gated)

## Recommended Execution Paths

Run node validation through the workload orchestrator (recommended on Linux SUT):

```bash
python scripts/run_all_workloads.py --only node_validation
```

Run workload acceptance tests directly:

```bash
pytest -m workload_acceptance -vv tests/test_node_validation.py
```

Run only node validation tests:

```bash
pytest -m node_validation -vv tests/test_node_validation.py
```

## Linux SUT Preflight

Before running workload-heavy tests, verify GPU tools and binaries expected by default:

```bash
nvidia-smi
test -x /root/workloads/nvbandwidth
test -d /root/workloads/nccl-tests/build
```

If your paths differ, override before execution:

```bash
export NVBANDWIDTH_BIN=/opt/nvbandwidth/nvbandwidth
export PERF_NVBANDWIDTH_BIN=/opt/nvbandwidth/nvbandwidth
export PERF_NCCL_TESTS_DIR=/opt/nccl-tests/build
```

## Key Markers

```bash
pytest -m gpu_enumeration -vv tests/test_node_validation.py
pytest -m pcie_validation -vv tests/test_node_validation.py
pytest -m p2p_validation -vv tests/test_node_validation.py
pytest -m nccl_validation -vv tests/test_node_validation.py
pytest -m workload_acceptance -vv tests/test_node_validation.py
```

## Save Console Logs

Save pytest logger output to file:

```bash
pytest -m workload_acceptance -vv --log-file=node_validation.log --log-file-level=INFO
```

Save full console output:

```bash
pytest -m workload_acceptance -vv 2>&1 | tee node_validation_full.log
```

## Important Defaults

Large-model workload defaults:

- `HF_ENABLE_LARGE_MODELS=1`
- `HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1`
- `NODE_WORKLOAD_AUTO_INSTALL=1` on Linux

Useful overrides:

```bash
export HF_ENABLE_LARGE_MODELS=0
export HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0
export HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=300
export NODE_WORKLOAD_AUTO_INSTALL=0
export NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=1800
```

## Quick Troubleshooting

- If `nvidia-smi` is missing: install NVIDIA driver/toolkit on the node.
- If `torch` is missing: install PyTorch in the active virtual environment.
- If 70B/72B workloads appear stuck after tokenizer load: model shard loading is in progress; monitor `nvidia-smi` memory usage and process state.
- If you want shorter enterprise latency tests: lower `HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC`.

## Next Steps

After node validation is stable, run performance and MLPerf flows:

```bash
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite
```
