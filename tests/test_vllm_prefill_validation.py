"""
vLLM 4xH100 Prefill Validation Tests

This test module wraps the vllm_4h100_prefill_validation.sh script to provide
pytest integration for comprehensive prefill performance validation across
multiple input lengths and batch sizes on 4 H100 GPUs.

Metrics recorded to: artifacts/vllm_prefill_kpis.jsonl

Run with:
  pytest tests/test_vllm_prefill_validation.py -v
  
Or with custom parameters:
  MODEL="meta-llama/Llama-2-70B" TP=4 DTYPE=bfloat16 pytest tests/test_vllm_prefill_validation.py -v

Environment Variables (all optional):
  MODEL                : HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
  TP                   : Tensor parallelism degree (default: 4)
  DTYPE                : Data type (default: bfloat16)
  GPU_UTIL             : GPU memory utilization (default: 0.90)
  INPUT_LENS           : Space-separated input lengths (default: 128 512 2048 8192)
  BATCH_SIZES          : Space-separated batch sizes (default: 1 2 4 8 16)
  OUTPUT_LEN           : Output length (default: 1)
  WARMUP               : Warmup iterations (default: 5)
  ITERS                : Benchmark iterations (default: 20)
  RUN_PERF             : Whether to run perf stat (default: 1)
  OUTDIR               : Output directory (auto-generated if not set)
  VLLM_PREFILL_KPI_FILE: KPI output file (default: artifacts/vllm_prefill_kpis.jsonl)
"""

import os
import subprocess
import json
import time
import logging
import pytest
from pathlib import Path

from tests.kpi_efficiency import enrich_efficiency_kpis

logger = logging.getLogger(__name__)


def _kpi_output_path() -> Path:
    """Return JSONL file path for persisted KPI records."""
    return Path(os.environ.get("VLLM_PREFILL_KPI_FILE", "artifacts/vllm_prefill_kpis.jsonl"))


def _record_kpi(test_name: str, metrics: dict) -> None:
    """Append KPI metrics to a JSONL file for post-run reporting."""
    out_path = _kpi_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_metrics = enrich_efficiency_kpis(metrics, os.environ)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test": test_name,
        "metrics": enriched_metrics,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")

    logger.info("KPI %s: %s", test_name, json.dumps(enriched_metrics, default=str, sort_keys=True))


def _extract_vllm_metrics(json_file: Path) -> dict:
    """Extract latency metrics from vLLM JSON output."""
    if not json_file.exists():
        return {}
    
    try:
        with json_file.open("r") as f:
            data = json.load(f)
        
        metrics = {}
        
        # Extract latency percentiles if available.
        # vLLM's benchmark_latency.py emits latencies in SECONDS, so converting
        # to milliseconds requires multiplication by 1000, not division.
        if "latency_percentiles" in data:
            percs = data["latency_percentiles"]
            metrics["mean_latency_ms"] = percs.get("mean", 0) * 1000  # s → ms
            metrics["p50_latency_ms"] = percs.get("50", 0) * 1000
            metrics["p90_latency_ms"] = percs.get("90", 0) * 1000
            metrics["p95_latency_ms"] = percs.get("95", 0) * 1000
            metrics["p99_latency_ms"] = percs.get("99", 0) * 1000
        
        # Extract throughput if available
        if "throughput" in data:
            metrics["throughput_tokens_per_sec"] = data["throughput"]
        
        # Extract other top-level metrics
        for key in ["num_generated_tokens", "num_prompts", "total_input_length"]:
            if key in data:
                metrics[key] = data[key]
        
        return metrics
    except Exception as exc:
        logger.warning("Failed to extract metrics from %s: %s", json_file, exc)
        return {}


def _extract_aggregate_metrics(output_dir: Path, model: str, tp: int, dtype: str) -> dict:
    """Extract and aggregate metrics from all JSON files in output directory."""
    if not output_dir.exists():
        return {}
    
    all_mean_latencies = []
    all_p95_latencies = []
    all_throughputs = []
    
    json_files = sorted(output_dir.glob("len*.json"))
    if not json_files:
        return {}
    
    for json_file in json_files:
        metrics = _extract_vllm_metrics(json_file)
        if metrics:
            if "mean_latency_ms" in metrics:
                all_mean_latencies.append(metrics["mean_latency_ms"])
            if "p95_latency_ms" in metrics:
                all_p95_latencies.append(metrics["p95_latency_ms"])
            if "throughput_tokens_per_sec" in metrics:
                all_throughputs.append(metrics["throughput_tokens_per_sec"])
    
    # Compute aggregates
    aggregate = {
        "model": model,
        "tensor_parallel_size": tp,
        "dtype": dtype,
        "num_test_cases": len(json_files),
    }
    
    if all_mean_latencies:
        aggregate["mean_latency_ms_min"] = min(all_mean_latencies)
        aggregate["mean_latency_ms_max"] = max(all_mean_latencies)
        aggregate["mean_latency_ms_avg"] = sum(all_mean_latencies) / len(all_mean_latencies)
    
    if all_p95_latencies:
        aggregate["p95_latency_ms_min"] = min(all_p95_latencies)
        aggregate["p95_latency_ms_max"] = max(all_p95_latencies)
        aggregate["p95_latency_ms_avg"] = sum(all_p95_latencies) / len(all_p95_latencies)
    
    if all_throughputs:
        aggregate["throughput_tokens_per_sec_min"] = min(all_throughputs)
        aggregate["throughput_tokens_per_sec_max"] = max(all_throughputs)
        aggregate["throughput_tokens_per_sec_avg"] = sum(all_throughputs) / len(all_throughputs)
    
    return aggregate


def get_script_path():
    """Get the path to the vLLM prefill validation script."""
    return Path(__file__).parent.parent / "scripts" / "vllm_4h100_prefill_validation.sh"


def skip_if_insufficient_gpus():
    """Skip test if fewer than 4 GPUs available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5
        )
        gpu_count = len(result.stdout.strip().split("\n"))
        if gpu_count < 4:
            pytest.skip(f"Requires 4 GPUs; found {gpu_count}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("nvidia-smi not available or timed out")


def skip_if_vllm_not_installed():
    """Skip test if vLLM is not installed."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM not installed")


def skip_if_script_missing():
    """Skip test if validation script is missing."""
    script_path = get_script_path()
    if not script_path.exists():
        pytest.skip(f"Validation script not found: {script_path}")


@pytest.mark.workload_acceptance
@pytest.mark.vllm_prefill
class TestVLLMPrefillValidation:
    """Test suite for vLLM 4xH100 prefill validation."""

    def setup_method(self):
        """Setup for each test method."""
        skip_if_insufficient_gpus()
        skip_if_vllm_not_installed()
        skip_if_script_missing()

    def test_vllm_4h100_prefill_validation(self):
        """
        Run the vLLM 4xH100 prefill validation script.
        
        This test executes comprehensive prefill performance benchmarks
        across multiple input lengths and batch sizes on 4 H100 GPUs.
        Results are collected in CSV and JSON formats with hardware metrics.
        
        Output artifacts:
          - summary.csv: Aggregated results table
          - {case}.json: JSON latency output for each case
          - {case}.perf.txt: Linux perf stats (if available)
          - {case}.nvidia_smi.csv: GPU metrics snapshot
          - run.log: Comprehensive execution log
        """
        script_path = get_script_path()
        model = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B")
        tp = os.environ.get("TP", "4")
        dtype = os.environ.get("DTYPE", "bfloat16")
        
        # Prepare environment (inherit from pytest env, allow overrides)
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        env.setdefault("TOKENIZERS_PARALLELISM", "true")
        
        # Set OMPI variables for containerized environments
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
        
        # Run the validation script
        result = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout for full benchmark
        )
        
        # Assert successful execution
        assert result.returncode == 0, f"Validation script failed with rc={result.returncode}"
        
        # Extract and record KPIs from the output directory
        # The script creates vllm_4h100_prefill_results_* directory
        output_dirs = sorted(Path.cwd().glob("vllm_4h100_prefill_results_*"))
        if output_dirs:
            output_dir = output_dirs[-1]  # Use the most recent
            metrics = _extract_aggregate_metrics(output_dir, model, int(tp), dtype)
            if metrics:
                _record_kpi("test_vllm_4h100_prefill_validation", metrics)

    def test_vllm_prefill_with_custom_model(self):
        """
        Run prefill validation with a custom model via MODEL env var.
        
        Example:
          MODEL="meta-llama/Llama-2-70B" pytest tests/test_vllm_prefill_validation.py::TestVLLMPrefillValidation::test_vllm_prefill_with_custom_model -v
        """
        model = os.environ.get("CUSTOM_MODEL")
        if not model:
            pytest.skip("CUSTOM_MODEL environment variable not set")
        
        script_path = get_script_path()
        tp = os.environ.get("TP", "4")
        dtype = os.environ.get("DTYPE", "bfloat16")
        env = os.environ.copy()
        env["MODEL"] = model
        env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        env.setdefault("TOKENIZERS_PARALLELISM", "true")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
        
        result = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            capture_output=False,
            text=True,
            timeout=3600
        )
        
        assert result.returncode == 0, f"Validation script failed with rc={result.returncode}"
        
        # Extract and record KPIs
        output_dirs = sorted(Path.cwd().glob("vllm_4h100_prefill_results_*"))
        if output_dirs:
            output_dir = output_dirs[-1]
            metrics = _extract_aggregate_metrics(output_dir, model, int(tp), dtype)
            if metrics:
                _record_kpi("test_vllm_prefill_with_custom_model", metrics)

    def test_vllm_prefill_quick(self):
        """
        Run a quick prefill validation with reduced iterations and batch sizes.
        
        Useful for rapid CI/CD validation without full benchmarking.
        """
        script_path = get_script_path()
        model = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B")
        tp = os.environ.get("TP", "4")
        dtype = os.environ.get("DTYPE", "bfloat16")
        env = os.environ.copy()
        env["INPUT_LENS"] = "512 2048"  # Fewer input lengths
        env["BATCH_SIZES"] = "1 4"  # Fewer batch sizes
        env["WARMUP"] = "2"
        env["ITERS"] = "5"
        env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        env.setdefault("TOKENIZERS_PARALLELISM", "true")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
        
        result = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            capture_output=False,
            text=True,
            timeout=1200  # 20 min timeout for quick run
        )
        
        assert result.returncode == 0, f"Quick validation script failed with rc={result.returncode}"
        
        # Extract and record KPIs
        output_dirs = sorted(Path.cwd().glob("vllm_4h100_prefill_results_*"))
        if output_dirs:
            output_dir = output_dirs[-1]
            metrics = _extract_aggregate_metrics(output_dir, model, int(tp), dtype)
            if metrics:
                _record_kpi("test_vllm_prefill_quick", metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
