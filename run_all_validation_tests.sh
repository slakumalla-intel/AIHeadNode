#!/usr/bin/env bash
set -euo pipefail

PERFORMANCE_CATALOG_ONLY=0
PERFORMANCE_CASE_IDS=""
STOP_ON_FAILURE=0
EXTRA_PYTEST_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./run_all_validation_tests.sh [options] [-- <extra pytest args>]

Runs all validation stages:
  1) Bottleneck tests
  2) Node validation tests
  3) Performance suite

Options:
  --performance-catalog-only     Do not execute performance commands (catalog mode)
  --performance-case-ids IDS     Comma-separated performance case IDs (e.g. 001,002,018)
  --stop-on-failure              Stop immediately when a stage fails
  -h, --help                     Show this help

Any arguments after -- are passed through to each pytest invocation.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --performance-catalog-only)
      PERFORMANCE_CATALOG_ONLY=1
      shift
      ;;
    --performance-case-ids)
      PERFORMANCE_CASE_IDS="${2:-}"
      if [[ -z "$PERFORMANCE_CASE_IDS" ]]; then
        echo "Error: --performance-case-ids requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --stop-on-failure)
      STOP_ON_FAILURE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_PYTEST_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v pytest >/dev/null 2>&1; then
  echo "Error: pytest not found in PATH. Activate your environment and install dependencies first." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$PERFORMANCE_CATALOG_ONLY" -eq 1 ]]; then
  export PERF_EXECUTE_CASES="0"
else
  export PERF_EXECUTE_CASES="1"
fi

if [[ -n "$PERFORMANCE_CASE_IDS" ]]; then
  export PERF_CASE_IDS="$PERFORMANCE_CASE_IDS"
else
  unset PERF_CASE_IDS || true
fi

STAGE_NAMES=(
  "Bottleneck Tests"
  "Node Validation Tests"
  "Performance Suite"
)

STAGE_COMMANDS=(
  "pytest -m bottleneck -vv tests/test_bottleneck_analysis.py"
  "pytest -m node_validation -vv tests/test_node_validation.py"
  "pytest -vv tests/test_performance_suite.py -k test_performance_case_suite"
)

RESULT_CODES=()
RESULT_DURATIONS=()

run_stage() {
  local stage_name="$1"
  local stage_cmd="$2"

  echo
  echo "============================================================"
  echo "Running stage: ${stage_name}"
  echo "Command: ${stage_cmd} ${EXTRA_PYTEST_ARGS[*]:-}"
  echo "============================================================"

  local start_ts
  start_ts="$(date +%s)"

  set +e
  bash -lc "${stage_cmd} ${EXTRA_PYTEST_ARGS[*]:-}"
  local exit_code=$?
  set -e

  local end_ts
  end_ts="$(date +%s)"
  local duration=$((end_ts - start_ts))

  RESULT_CODES+=("$exit_code")
  RESULT_DURATIONS+=("$duration")

  return "$exit_code"
}

for i in "${!STAGE_NAMES[@]}"; do
  if ! run_stage "${STAGE_NAMES[$i]}" "${STAGE_COMMANDS[$i]}"; then
    if [[ "$STOP_ON_FAILURE" -eq 1 ]]; then
      break
    fi
  fi
done

echo
echo "==================== Overall Summary ===================="
any_failed=0
for i in "${!RESULT_CODES[@]}"; do
  status="PASS"
  if [[ "${RESULT_CODES[$i]}" -ne 0 ]]; then
    status="FAIL"
    any_failed=1
  fi
  printf "%-24s %-6s %8ss\n" "${STAGE_NAMES[$i]}" "$status" "${RESULT_DURATIONS[$i]}"
done
echo "========================================================="

exit "$any_failed"
