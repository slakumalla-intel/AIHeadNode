"""Functional test-plan ID catalog for AI host CPU validation.

This file provides an explicit, code-owned mapping from
AI_HOSTCPU_FUNC_TEST_ID_001..054 to test selectors in this repository.
It is intentionally metadata-only and does not execute heavy workloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pytest


@dataclass(frozen=True)
class FunctionalPlanCase:
    case_id: str
    title: str
    status: str
    mapped_selectors: List[str] = field(default_factory=list)
    note: str = ""


_PLAN_TITLES: Dict[int, str] = {
    1: "Verify GPU enumeration and PCIe topology",
    2: "CPU to GPU PCIe bandwidth for direct root-complex attached GPUs",
    3: "CPU to GPU PCIe bandwidth for PCIe-switch attached GPUs",
    4: "Pinned vs pageable copy path validation for CE and SM",
    5: "Validate GPU P2P data bandwidth",
    6: "CPU-GPU launch and synchronize overhead",
    7: "CPU-GPU power and throttling coupling",
    8: "NCCL and NVLink validation",
    9: "Deterministic NCCL init behavior",
    10: "Data pipeline saturation for inference prefill",
    11: "Low-jitter NCCL communication",
    12: "Predictable memory orchestration with KV offload",
    13: "LLM training small",
    14: "LLM training large",
    15: "LLM inference",
    16: "Vision workload object detection",
    17: "Vision workload image generation",
    18: "Multi-node NCCL",
    19: "Multi-node training",
    20: "Two-node networking",
    21: "Single-core GPUDirect RDMA flow",
    22: "WCIL MMIO write/read validation",
    23: "Fence validation",
    24: "One DMA read of 4KB validation",
    25: "Looped read/write stress and soak",
    26: "Storage throughput",
    27: "Idle power",
    28: "Peak power",
    29: "Performance per watt",
    30: "MLPerf ResNet train",
    31: "MLPerf BERT train",
    32: "MLPerf ResNet infer",
    33: "MLPerf BERT infer",
    34: "72h stability soak",
    35: "RAS and ECC error checking",
    36: "Inference thermal stability",
    37: "Cold start latency",
    38: "Oversubscription in VM",
    39: "Near-OOM behavior",
    40: "48h soak stability",
    41: "CPU scheduling",
    42: "MPI scaling",
    43: "Core and IO frequency impact",
    44: "UPIx lines impact",
    45: "Turbo vs non-turbo",
    46: "Multi-model allocation and isolation",
    47: "Inference plus multi-user",
    48: "Inference plus multi-user plus KV cache",
    49: "DeepSeek BF16 inference plus KV cache DRAM offload",
    50: "PCIe bandwidth and P2P latency with 2MB hugepages",
    51: "GPU write performance limiters vs memory speed",
    52: "P2P performance with multiple streams",
    53: "JVM readiness for SERT validation",
    54: "Disable PCT SKU Intel SST",
}

_MAPPED_SELECTORS: Dict[int, List[str]] = {
    1: [
        "tests/test_node_validation.py::TestGPUEnumerationAndTopology::test_gpu_enumeration",
        "tests/test_node_validation.py::TestGPUEnumerationAndTopology::test_pcie_topology_discovery",
    ],
    2: ["tests/test_node_validation.py::TestPCIeBandwidthDirect::test_h2d_bandwidth_all_gpus"],
    3: ["tests/test_node_validation.py::TestPCIeBandwidthDirect::test_d2h_bandwidth_all_gpus"],
    4: [
        "tests/test_node_validation.py::TestNVBandwidthMemcpy::test_host_to_device_memcpy_ce",
        "tests/test_node_validation.py::TestNVBandwidthMemcpy::test_device_to_host_memcpy_ce",
        "tests/test_node_validation.py::TestNVBandwidthMemcpy::test_host_to_device_memcpy_sm",
        "tests/test_node_validation.py::TestNVBandwidthMemcpy::test_device_to_host_memcpy_sm",
    ],
    5: [
        "tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary",
        "tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback",
    ],
    6: [
        "tests/test_node_validation.py::TestCPUGPULaunchAndSyncOverhead::test_kernel_launch_latency_stats",
        "tests/test_node_validation.py::TestCPUGPULaunchAndSyncOverhead::test_synchronize_overhead_stats",
    ],
    7: ["tests/test_node_validation.py::TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability"],
    8: [
        "tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nvlink_status",
        "tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_all_reduce_perf",
        "tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_all_gather_perf",
        "tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_reduce_scatter_perf",
        "tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_p2p_perf",
    ],
    9: ["tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_deterministic_init_variance"],
    10: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill"],
    11: ["tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_deterministic_init_variance"],
    12: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload"],
    13: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu"],
    14: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_llm_training_large_multigpu"],
    15: [
        "tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vllm_inference_throughput",
        "tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency",
        "tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_batch_throughput",
    ],
    16: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps"],
    17: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vision_image_generation_diffusers_throughput"],
    21: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_single_core_gpudirect_rdma_flow"],
    22: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_mmio_write_read_validation"],
    23: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_fence_validation_rdma_ordering"],
    24: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_one_dma_read_4kb_validation"],
    25: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_looped_rdma_mmio_dma_stress_soak"],
    26: [
        "tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_storage_sequential_read_throughput",
        "tests/test_ai_pipeline_kpis.py::TestNetworkingAndStorageIO::test_storage_sequential_write_throughput",
    ],
    27: ["tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_power_limit_headroom"],
    28: ["tests/test_node_validation.py::TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability"],
    29: [
        "tests/test_system_health_and_reliability.py::TestEnergyEfficiency::test_fp16_gflops_per_watt",
        "tests/test_system_health_and_reliability.py::TestEnergyEfficiency::test_aggregate_4gpu_fp16_tflops_per_watt",
    ],
    30: ["tests/test_mlperf_suite.py::test_mlperf_case_suite"],
    31: ["tests/test_mlperf_suite.py::test_mlperf_case_suite"],
    32: ["tests/test_mlperf_suite.py::test_mlperf_case_suite"],
    33: ["tests/test_mlperf_suite.py::test_mlperf_case_suite"],
    35: [
        "tests/test_system_health_and_reliability.py::TestGPUECCAndPCIeLinkHealth::test_hbm_ecc_uncorrected_errors_zero",
        "tests/test_system_health_and_reliability.py::TestGPUECCAndPCIeLinkHealth::test_hbm_ecc_corrected_errors_rate",
    ],
    36: [
        "tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_temperature_within_safe_limits",
        "tests/test_system_health_and_reliability.py::TestGPUThermalAndClockStability::test_gpu_clock_stability_under_load",
    ],
    37: ["tests/test_ai_pipeline_kpis.py::TestInferencePipelineKPIs::test_time_to_first_token"],
    39: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload"],
    41: ["tests/test_cpu_validation.py::TestCPUAffinity::test_process_can_set_affinity"],
    47: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_multi_tenant_inference_tail_latency"],
    48: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_multi_model_inference_kv_cache_concurrent"],
    49: ["tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_deepseek_r1_kv_cache_dram_offload"],
    50: ["tests/test_node_validation.py::TestNVBandwidthMemcpy::test_pcie_bw_p2p_latency_hugepages_2mb"],
    51: ["tests/test_node_validation.py::TestCPUGPUPowerAndThrottling::test_gpu_write_performance_limiters"],
    52: ["tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_multi_stream_bandwidth"],
}

_PARTIAL_IDS = {11, 27, 30, 31, 32, 33, 39}
_PLANNED_ONLY_IDS = {18, 19, 20, 34, 38, 40, 42, 43, 44, 45, 46, 53, 54}


def _build_cases() -> List[FunctionalPlanCase]:
    cases: List[FunctionalPlanCase] = []
    for num in range(1, 55):
        status = "implemented"
        if num in _PLANNED_ONLY_IDS:
            status = "planned_only"
        elif num in _PARTIAL_IDS:
            status = "partial"
        selectors = _MAPPED_SELECTORS.get(num, [])
        note = ""
        if num == 46:
            note = "Mapped from source row marked as #REF! in the supplied plan."
        if num in {30, 31, 32, 33}:
            note = "Executed through tests/test_mlperf_suite.py::test_mlperf_case_suite with ML_CASE_IDS filtering."
        cases.append(
            FunctionalPlanCase(
                case_id=f"AI_HOSTCPU_FUNC_TEST_ID_{num:03d}",
                title=_PLAN_TITLES[num],
                status=status,
                mapped_selectors=selectors,
                note=note,
            )
        )
    return cases


FUNCTIONAL_PLAN_CASES = _build_cases()


@pytest.mark.node_validation
@pytest.mark.parametrize("case", FUNCTIONAL_PLAN_CASES, ids=[c.case_id for c in FUNCTIONAL_PLAN_CASES])
def test_functional_plan_mapping_catalog(case: FunctionalPlanCase):
    """Validate completeness and shape of functional test-plan mapping metadata."""
    assert re.fullmatch(r"AI_HOSTCPU_FUNC_TEST_ID_\d{3}", case.case_id)
    assert case.title.strip()
    assert case.status in {"implemented", "partial", "planned_only"}

    for selector in case.mapped_selectors:
        file_part = selector.split("::", 1)[0]
        assert Path(file_part).exists(), f"Mapped selector file does not exist: {selector}"


def test_functional_plan_mapping_is_complete_and_unique():
    """Ensure all functional IDs 001..054 are represented exactly once."""
    ids = [c.case_id for c in FUNCTIONAL_PLAN_CASES]
    assert len(ids) == 54
    assert len(set(ids)) == 54
    assert ids[0] == "AI_HOSTCPU_FUNC_TEST_ID_001"
    assert ids[-1] == "AI_HOSTCPU_FUNC_TEST_ID_054"
