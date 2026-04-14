# AIHeadNode

Linux-first validation suite for AI head-node bring-up, performance qualification,
and workload acceptance.

Primary target platform: multi-GPU Linux nodes (for example 4x NVIDIA H100 class systems).

## What Changed

Execution is now centered on Linux shell runners:

- `run_all_validation_tests.sh` for end-to-end staged validation
- `run_workload_acceptance.sh` for workload-only validation flows

This README reflects that Linux-first workflow.

## Repository Layout

```text
.
├── requirements.txt
├── pytest.ini
├── run_all_validation_tests.sh
├── run_all_validation_tests.ps1
├── run_workload_acceptance.sh
├── run_workload_acceptance.ps1
├── tests/
│   ├── test_bottleneck_analysis.py
│   ├── test_node_validation.py
│   ├── test_performance_suite.py
│   ├── test_cpu_validation.py
│   ├── test_memory_validation.py
│   ├── test_gpu_validation.py
│   ├── test_interconnect_validation.py
│   └── test_system_config_report.py
└── artifacts/ (generated)
```

## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU stack install (if not already available in your image):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

System packages commonly needed for vLLM/native extensions:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# RHEL / Rocky / CentOS
sudo dnf install -y python3-devel gcc gcc-c++ make
```

## Main Execution Paths

### 1) Run Everything (Recommended)

Use the all-in-one Linux runner:

```bash
chmod +x run_all_validation_tests.sh
./run_all_validation_tests.sh
```

This runs three stages in order:

1. Bottleneck tests
2. Node validation tests
3. Performance suite

### 2) Run Workload Acceptance Only

```bash
chmod +x run_workload_acceptance.sh
./run_workload_acceptance.sh
```

## `run_all_validation_tests.sh` Options

```bash
./run_all_validation_tests.sh --performance-catalog-only
./run_all_validation_tests.sh --performance-case-ids 001,002,018
./run_all_validation_tests.sh --stop-on-failure
./run_all_validation_tests.sh -- -s --log-cli-level=INFO
```

Meaning:

- `--performance-catalog-only`: sets `PERF_EXECUTE_CASES=0`
- `--performance-case-ids`: sets `PERF_CASE_IDS` (comma-separated IDs)
- `--stop-on-failure`: stops after first failed stage
- `--`: pass-through extra pytest args to each stage

## Direct Pytest Commands

Run the same suites manually if preferred:

```bash
pytest -m bottleneck -vv tests/test_bottleneck_analysis.py
pytest -m node_validation -vv tests/test_node_validation.py
pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
```

Other useful commands:

```bash
pytest -m workload_acceptance -vv
pytest -m rdma_validation -vv tests/test_node_validation.py
pytest -m "not slow" -vv
```

## Performance Suite Controls

`tests/test_performance_suite.py` supports execution gating through environment variables:

```bash
export PERF_EXECUTE_CASES=1
export PERF_CASE_IDS=001,002,018
export PERF_CMD_TIMEOUT_SEC=600
export PERF_SUITE_KPI_FILE=artifacts/performance_suite_kpis.jsonl

pytest -vv tests/test_performance_suite.py -k test_performance_case_suite
```

Defaults:

- KPI file: `artifacts/performance_suite_kpis.jsonl`
- Catalog-only behavior can be enabled via runner option or `PERF_EXECUTE_CASES=0`
- Command timeout defaults to `PERF_CMD_TIMEOUT_SEC=1800`
- `vllm serve` commands automatically add `--gpu-memory-utilization` using
	`PERF_VLLM_GPU_MEMORY_UTILIZATION` (default `0.70`) when not already set
- Invalid `--dtype float8` cases are rewritten to use
	`PERF_VLLM_FP8_DTYPE_FALLBACK` (default `bfloat16`) plus `--quantization fp8`
- `./build/<nccl-test>` commands are automatically resolved from
	`PERF_NCCL_TESTS_DIR` (default `/root/workloads/nccl-tests/build`)
- `nvbandwidth` commands are automatically resolved from `PERF_NVBANDWIDTH_BIN`
	(default `/root/workloads/nvbandwidth` or `NVBANDWIDTH_BIN` when already set)
- vLLM benchmark module lookup can be overridden with
	`PERF_VLLM_BENCHMARK_MODULE`

## Workload Acceptance Controls

The workload runner exposes common knobs:

```bash
./run_workload_acceptance.sh --min-duration-sec 600
./run_workload_acceptance.sh --kv-context-length 16384 --kv-batch-size 8
./run_workload_acceptance.sh --skip-large-models
```

Reference file for env configuration:

- `.env.example`

## Automatic Linux Defaults (Workload Path)

Workload preflight now supports Linux-first automatic setup behavior in
`tests/test_node_validation.py`:

- Large-model workloads are enabled by default:
	- `HF_ENABLE_LARGE_MODELS=1` (default)
- Remote model download is enabled by default when local cache is missing:
	- `HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=1` (default)
- Workload dependency auto-install is enabled by default on Linux:
	- `NODE_WORKLOAD_AUTO_INSTALL=1` (Linux default)

Auto-install attempts use the active Python interpreter (`python -m pip install`)
and can install missing workload dependencies before strict preflight checks:

- Required / common: `transformers`
- Large-model path: `accelerate`
- Optional workload components: `vllm`, `ultralytics`, `diffusers`, `psutil`

Override controls:

```bash
# Disable dependency auto-install
export NODE_WORKLOAD_AUTO_INSTALL=0

# Change per-package auto-install timeout (seconds)
export NODE_WORKLOAD_AUTO_INSTALL_TIMEOUT_SEC=1800

# Disable large-model execution gate
export HF_ENABLE_LARGE_MODELS=0

# Force local-only large-model behavior
export HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD=0
```

Notes:

- If `accelerate` is unavailable, quantized/device-map model load paths fall back
	to non-quantized load behavior with a warning when possible.
- Gated Hugging Face repositories may still require `HF_TOKEN`.

## Reports and KPI Artifacts

Generated outputs include:

- `TESTCASE_EXECUTION_FINAL_REPORT.md`
- `artifacts/node_validation_kpis.jsonl`
- `artifacts/performance_suite_kpis.jsonl`

`TESTCASE_EXECUTION_FINAL_REPORT.md` is the single consolidated markdown report for the full test run.
It includes:

- overall summary counts for executed, passed, failed, and skipped tests
- a brief issues table near the top
- grouped sections for Workload Acceptance, Node Validation, Bottleneck, Performance, and MLPerf
- color-coded status rendering: pass in green, fail in red, skip in orange

Default node KPI file can be overridden with:

```bash
export NODE_VALIDATION_KPI_FILE=artifacts/node_validation_kpis.jsonl
```

## Notes

- Linux is the supported execution path for this workflow.
- PowerShell scripts are retained in-repo but are not the primary path.
- Some workload tests are hardware- and dependency-sensitive and may skip when prerequisites are missing.
