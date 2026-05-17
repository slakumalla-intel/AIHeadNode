"""Helpers for derived KPI efficiency metrics.

This module adds normalized value KPIs when source metrics are available:
  - performance_per_watt
  - performance_per_dollar
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Tuple


_PERFORMANCE_KEYS = (
    "throughput_tokens_per_sec",
    "tokens_per_sec",
    "samples_per_sec",
    "requests_per_sec",
    "throughput_qps",
    "qps",
    "aggregate_fps",
    "throughput_fps",
    "tflops",
    "gflops",
)

_POWER_KEYS = (
    "mean_power_watts",
    "avg_power_watts",
    "power_watts",
    "gpu_power_watts",
    "board_power_watts",
    "mean_w",
    "avg_w",
    "power_w",
    "gpu_power_w",
)

_COST_PER_HOUR_KEYS = (
    "cost_per_hour_usd",
    "hourly_cost_usd",
    "node_cost_per_hour_usd",
    "total_cost_per_hour_usd",
)

_GPU_COUNT_KEYS = (
    "gpu_count",
    "num_gpus",
    "n_gpus",
    "tensor_parallel",
    "tp",
)


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result <= 0:
        return None
    return result


def _first_numeric(metrics: Mapping[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[float]]:
    for key in keys:
        if key not in metrics:
            continue
        value = _as_float(metrics.get(key))
        if value is not None:
            return key, value
    return None, None


def _infer_gpu_count(metrics: Mapping[str, Any], environ: Mapping[str, str]) -> int:
    for key in _GPU_COUNT_KEYS:
        value = metrics.get(key)
        parsed = _as_float(value)
        if parsed is not None:
            return max(1, int(parsed))

    for env_key in ("PERF_NUM_GPUS", "ML_NUM_GPUS"):
        parsed = _as_float(environ.get(env_key))
        if parsed is not None:
            return max(1, int(parsed))

    return 1


def _infer_power_watts(metrics: Mapping[str, Any], environ: Mapping[str, str]) -> Optional[float]:
    _, power = _first_numeric(metrics, _POWER_KEYS)
    if power is not None:
        return power

    tdp = _as_float(environ.get("GPU_TDP_WATTS"))
    if tdp is None:
        return None

    return tdp * _infer_gpu_count(metrics, environ)


def _infer_cost_per_hour_usd(metrics: Mapping[str, Any], environ: Mapping[str, str]) -> Optional[float]:
    _, cost_per_hour = _first_numeric(metrics, _COST_PER_HOUR_KEYS)
    if cost_per_hour is not None:
        return cost_per_hour

    shared = _as_float(environ.get("KPI_COST_PER_HOUR_USD"))
    if shared is not None:
        return shared

    node_cost = _as_float(environ.get("NODE_COST_PER_HOUR_USD"))
    if node_cost is not None:
        return node_cost

    gpu_cost = _as_float(environ.get("GPU_COST_PER_HOUR_USD"))
    if gpu_cost is None:
        return None

    return gpu_cost * _infer_gpu_count(metrics, environ)


def enrich_efficiency_kpis(metrics: Mapping[str, Any], environ: Mapping[str, str]) -> MutableMapping[str, Any]:
    """Return a copy of metrics with derived value KPIs when derivable."""
    out = dict(metrics)

    _, performance = _first_numeric(out, _PERFORMANCE_KEYS)
    if performance is None:
        return out

    if "performance_per_watt" not in out:
        power_watts = _infer_power_watts(out, environ)
        if power_watts is not None:
            out["performance_per_watt"] = round(performance / power_watts, 6)

    if "performance_per_dollar" not in out:
        cost_per_hour = _infer_cost_per_hour_usd(out, environ)
        if cost_per_hour is not None:
            out["performance_per_dollar"] = round((performance * 3600.0) / cost_per_hour, 6)

    return out