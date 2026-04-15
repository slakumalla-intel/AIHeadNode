# AIHeadNode

Linux-first validation suite for AI head-node bring-up, performance qualification,
and workload acceptance on multi-GPU systems (for example 4x H100).

## Scope

Main Linux execution paths in this repo:

- `run_all_validation_tests.sh` for staged validation (bottleneck + node validation + performance suite)
- `run_workload_acceptance.sh` for workload acceptance validation
- `scripts/download_mlperf_models.sh` for MLPerf model downloads
- `scripts/download_mlperf_datasets.sh` for MLPerf dataset downloads
- `scripts/run_mlperf_h100x4.sh` for orchestrated MLPerf runs on 4 GPUs

## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional if PyTorch is not already present in your image:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

System packages often needed for native builds:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# RHEL / Rocky / CentOS
sudo dnf install -y python3-devel gcc gcc-c++ make
```

## Core Validation Procedure

### 1) Run All Validation Stages

```bash
chmod +x run_all_validation_tests.sh
./run_all_validation_tests.sh
```

Stages executed in order:

1. Bottleneck tests
2. Node validation tests
3. Performance suite

### 2) Run Workload Acceptance Only

```bash
chmod +x run_workload_acceptance.sh
./run_workload_acceptance.sh
```

### 3) Save Console Logs to File

Persist pytest logs with timestamps and logger names:

```bash
pytest -m workload_acceptance -vv --log-file=node_validation.log --log-file-level=INFO
```

Capture full console output (stdout + stderr) as well:

```bash
pytest -m workload_acceptance -vv 2>&1 | tee node_validation_full.log
```

## Useful Runner Options

For staged validation:

```bash
./run_all_validation_tests.sh --performance-catalog-only
./run_all_validation_tests.sh --performance-case-ids 001,002,018
./run_all_validation_tests.sh --stop-on-failure
./run_all_validation_tests.sh --skip-consolidated-report
./run_all_validation_tests.sh -- -s --log-cli-level=INFO
```

By default, the staged runner executes a final consolidated pytest pass to refresh
`TESTCASE_EXECUTION_FINAL_REPORT.md` with a single unified report snapshot.

For workload acceptance:

```bash
./run_workload_acceptance.sh --min-duration-sec 600
./run_workload_acceptance.sh --kv-context-length 16384 --kv-batch-size 8
./run_workload_acceptance.sh --skip-large-models
```

## Direct Pytest Commands

```bash
pytest -m bottleneck -vv tests/test_bottleneck_analysis.py
pytest -m node_validation -vv tests/test_node_validation.py
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite
```

Other useful selectors:

```bash
pytest -m workload_acceptance -vv
pytest -m rdma_validation -vv tests/test_node_validation.py
pytest -m "not slow" -vv
```

## Environment Defaults (Workload Path)

Default behavior for large-model workload tests:

- `HF_ENABLE_LARGE_MODELS=1`
- `HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1`
- `NODE_WORKLOAD_AUTO_INSTALL=1` on Linux

Common overrides:

```bash
export NODE_WORKLOAD_AUTO_INSTALL=0
export NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=1800
export HF_ENABLE_LARGE_MODELS=0
export HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0
export HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=300
```

## Performance Suite Controls

`tests/test_performance_suite.py` uses these controls:

```bash
export PERF_EXECUTE_CASES=1
export PERF_CASE_IDS=001,002,018
export PERF_CMD_TIMEOUT_SEC=1800
export PERF_SUITE_KPI_FILE=artifacts/performance_suite_kpis.jsonl
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
```

## MLPerf Workflow (4x H100)

Download models:

```bash
chmod +x scripts/download_mlperf_models.sh
scripts/download_mlperf_models.sh --model-dir /models/mlperf
```

Download datasets:

```bash
chmod +x scripts/download_mlperf_datasets.sh
scripts/download_mlperf_datasets.sh --data-dir /data/mlperf
```

Run MLPerf orchestration script:

```bash
chmod +x scripts/run_mlperf_h100x4.sh
scripts/run_mlperf_h100x4.sh --data-dir /data/mlperf --model-dir /models/mlperf --num-gpus 4 --tp-size 4
```

Or run the MLPerf pytest suite directly:

```bash
export ML_EXECUTE_CASES=1
export ML_CASE_IDS=001,006,014
export ML_CMD_TIMEOUT_SEC=3600
pytest -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite
```

## Artifacts and Reports

Main generated outputs:

- `TESTCASE_EXECUTION_FINAL_REPORT.md`
- `artifacts/node_validation_kpis.jsonl`
- `artifacts/performance_suite_kpis.jsonl`
- `artifacts/mlperf_suite_kpis.jsonl`

Node KPI output path can be overridden with:

```bash
export NODE_VALIDATION_KPI_FILE=artifacts/node_validation_kpis.jsonl
```

## Reference

- `.env.example` for environment variable examples.
- Linux is the supported execution path for this workflow.
