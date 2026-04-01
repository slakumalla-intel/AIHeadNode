"""
CPU + GPU System Configuration Report for the AI Head Node.

Test categories
---------------
TC-REPORT-01  Collect and log full CPU topology & performance specs
TC-REPORT-02  Collect and log per-GPU configuration (memory, clocks, ECC, NVLink)
TC-REPORT-03  Collect and log PCIe topology per GPU
TC-REPORT-04  Generate combined summary and persist report to JSON

Running
-------
Print the human-readable report to stdout::

    pytest tests/test_system_config_report.py -v -s

Save to a file in addition::

    pytest tests/test_system_config_report.py -v -s | tee system_report.txt

The JSON report is always written to ``system_config_report.json`` in the
repository root when TC-REPORT-04 runs.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests.theoretical_specs import CPU_SPECS, H100_SPECS, SYSTEM_SPECS

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None          # type: ignore[assignment]
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None           # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    pynvml = None          # type: ignore[assignment]
    _HAS_NVML = False

# ---------------------------------------------------------------------------
# Report output path
# ---------------------------------------------------------------------------

_REPORT_PATH = Path(__file__).resolve().parents[1] / "system_config_report.json"

# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _collect_cpu_info() -> Dict[str, Any]:
    """Return a dict with CPU topology, clock, memory, and theoretical spec data."""
    info: Dict[str, Any] = {}

    info["os_platform"]    = platform.platform()
    info["python_version"] = sys.version.split()[0]
    info["cpu_model"]      = platform.processor() or "unknown"
    info["logical_cpus"]   = os.cpu_count()

    if _HAS_PSUTIL:
        info["physical_cores"]   = psutil.cpu_count(logical=False)
        info["logical_threads"]  = psutil.cpu_count(logical=True)
        info["smt_enabled"]      = psutil.cpu_count(logical=True) > psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_current_mhz"] = round(freq.current, 1)
            info["cpu_freq_max_mhz"]     = round(freq.max, 1)
            info["cpu_freq_min_mhz"]     = round(freq.min, 1)
        vm = psutil.virtual_memory()
        info["total_ram_gb"]     = round(vm.total     / 1024 ** 3, 2)
        info["available_ram_gb"] = round(vm.available / 1024 ** 3, 2)

        numa_root = "/sys/devices/system/node"
        if os.path.isdir(numa_root):
            nodes = sorted(
                int(d.replace("node", ""))
                for d in os.listdir(numa_root)
                if d.startswith("node") and d[4:].isdigit()
            )
            info["numa_nodes"]      = nodes
            info["numa_node_count"] = len(nodes)
    else:
        info["physical_cores"]  = None
        info["logical_threads"] = os.cpu_count()
        info["total_ram_gb"]    = None

    # Theoretical specification values from theoretical_specs.py
    info["spec"] = {
        "sockets":               CPU_SPECS.sockets,
        "cores_per_socket":      CPU_SPECS.cores_per_socket,
        "physical_cores":        CPU_SPECS.total_physical_cores,
        "logical_threads":       CPU_SPECS.total_logical_threads,
        "base_clock_ghz":        CPU_SPECS.base_clock_ghz,
        "max_turbo_ghz":         CPU_SPECS.max_turbo_ghz,
        "l3_cache_mb_total":     CPU_SPECS.l3_cache_mb_total,
        "peak_fp32_tflops_base": round(CPU_SPECS.peak_fp32_tflops_base,  2),
        "peak_fp32_tflops_turbo":round(CPU_SPECS.peak_fp32_tflops_turbo, 2),
        "peak_memory_bw_gbs":    round(CPU_SPECS.peak_memory_bandwidth_gbs, 1),
    }
    return info


def _collect_gpu_info() -> List[Dict[str, Any]]:
    """Return a list of dicts – one per detected GPU – with all available metadata."""
    gpus: List[Dict[str, Any]] = []
    if not _HAS_TORCH:
        return gpus

    for i in range(torch.cuda.device_count()):
        g: Dict[str, Any] = {"gpu_index": i}
        g["name"] = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        g["compute_capability"]    = f"{props.major}.{props.minor}"
        g["total_memory_gb"]       = round(props.total_memory / 1024 ** 3, 2)
        g["multi_processor_count"] = props.multi_processor_count

        if _HAS_NVML:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Driver
            try:
                g["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
            except Exception:
                g["driver_version"] = "n/a"

            # PCI topology
            try:
                pci = pynvml.nvmlDeviceGetPciInfo(handle)
                raw = pci.busId
                g["pci_bus_id"] = raw.decode() if isinstance(raw, bytes) else raw
                g["pci_domain"] = hex(pci.domain)
                g["pci_bus"]    = hex(pci.bus)
                g["pci_device"] = hex(pci.device)
            except Exception:
                pass

            # PCIe link details
            try:
                g["pcie_max_gen"]    = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                g["pcie_curr_gen"]   = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                g["pcie_max_width"]  = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                g["pcie_curr_width"] = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                g["pcie_gen_healthy"]   = g["pcie_curr_gen"]   >= g["pcie_max_gen"]
                g["pcie_width_healthy"] = g["pcie_curr_width"] >= g["pcie_max_width"]
            except Exception:
                pass

            # Thermals / power
            try:
                g["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                g["temperature_c"] = "n/a"

            try:
                g["power_draw_w"]  = round(pynvml.nvmlDeviceGetPowerUsage(handle)         / 1000, 1)
                g["power_limit_w"] = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000, 1)
            except Exception:
                g["power_draw_w"]  = "n/a"
                g["power_limit_w"] = "n/a"

            # ECC
            try:
                mode = pynvml.nvmlDeviceGetEccMode(handle)
                curr = mode[0] if isinstance(mode, tuple) else mode
                g["ecc_enabled"] = curr == pynvml.NVML_FEATURE_ENABLED
            except Exception:
                g["ecc_enabled"] = "n/a"

            # Clocks
            try:
                g["clock_sm_mhz"]     = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                g["clock_mem_mhz"]    = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                g["max_clock_sm_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except Exception:
                pass

            # NVLink
            try:
                active = 0
                for link in range(H100_SPECS.nvlink_links_per_gpu):
                    try:
                        state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                        if state == pynvml.NVML_FEATURE_ENABLED:
                            active += 1
                    except pynvml.NVMLError:
                        break
                g["nvlink_active_links"] = active
                g["nvlink_total_links"]  = H100_SPECS.nvlink_links_per_gpu
            except Exception:
                pass

        # Theoretical spec values
        g["spec"] = {
            "fp64_tflops":          H100_SPECS.fp64_tflops,
            "fp32_tflops":          H100_SPECS.fp32_tflops,
            "fp16_tflops":          H100_SPECS.fp16_tflops,
            "bf16_tflops":          H100_SPECS.bf16_tflops,
            "int8_tops":            H100_SPECS.int8_tops,
            "hbm3_capacity_gb":     H100_SPECS.hbm3_capacity_gb,
            "hbm3_bandwidth_tbs":   H100_SPECS.hbm3_bandwidth_tbs,
            "nvlink_total_bw_gbs":  H100_SPECS.nvlink_total_bw_gbs,
            "pcie_gen":             H100_SPECS.pcie_gen,
            "pcie_lanes":           H100_SPECS.pcie_lanes,
            "pcie_unidirectional_bw_gbs": H100_SPECS.pcie_unidirectional_bw_gbs,
        }
        gpus.append(g)
    return gpus


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def _print_cpu_section(cpu: Dict[str, Any]) -> None:
    spec = cpu["spec"]
    lines = [
        "",
        _sep("═"),
        "  CPU CONFIGURATION REPORT",
        _sep("═"),
        f"  OS / Platform        : {cpu.get('os_platform', 'n/a')}",
        f"  Python version       : {cpu.get('python_version', 'n/a')}",
        f"  CPU model            : {cpu.get('cpu_model', 'n/a')}",
        _sep(),
        "  Topology",
        f"    Physical cores     : {cpu.get('physical_cores', 'n/a')}  "
        f"(spec: {spec['physical_cores']})",
        f"    Logical threads    : {cpu.get('logical_threads', 'n/a')}  "
        f"(spec: {spec['logical_threads']})",
        f"    SMT enabled        : {cpu.get('smt_enabled', 'n/a')}",
        f"    NUMA nodes         : {cpu.get('numa_node_count', 'n/a')}  "
        f"IDs: {cpu.get('numa_nodes', 'n/a')}",
        _sep(),
        "  Clock",
        f"    Current freq       : {cpu.get('cpu_freq_current_mhz', 'n/a')} MHz",
        f"    Max turbo          : {cpu.get('cpu_freq_max_mhz', 'n/a')} MHz  "
        f"(spec: {spec['max_turbo_ghz'] * 1000:.0f} MHz)",
        f"    Base clock (spec)  : {spec['base_clock_ghz'] * 1000:.0f} MHz",
        _sep(),
        "  Memory",
        f"    Total RAM          : {cpu.get('total_ram_gb', 'n/a')} GB",
        f"    Available RAM      : {cpu.get('available_ram_gb', 'n/a')} GB",
        f"    Peak DRAM BW (spec): {spec['peak_memory_bw_gbs']} GB/s",
        _sep(),
        "  Theoretical Performance (spec)",
        f"    L3 cache total     : {spec['l3_cache_mb_total']} MB",
        f"    FP32 @ base clock  : {spec['peak_fp32_tflops_base']} TFLOPS",
        f"    FP32 @ max turbo   : {spec['peak_fp32_tflops_turbo']} TFLOPS",
        _sep("═"),
    ]
    print("\n".join(lines))


def _print_gpu_section(gpus: List[Dict[str, Any]]) -> None:
    lines = [
        "",
        _sep("═"),
        "  GPU CONFIGURATION REPORT",
        _sep("═"),
        f"  GPUs detected        : {len(gpus)}  (spec: {SYSTEM_SPECS.gpu_count})",
        f"  Total HBM (spec)     : {SYSTEM_SPECS.total_gpu_memory_gb:.0f} GB",
        f"  Total FP16 (spec)    : {SYSTEM_SPECS.total_fp16_tflops_gpu:.0f} TFLOPS",
        f"  Total INT8 (spec)    : {SYSTEM_SPECS.total_int8_tops_gpu:.0f} TOPS",
        f"  NVLink fabric BW(spec): {SYSTEM_SPECS.nvlink_fabric_bw_gbs:.0f} GB/s",
        f"  PCIe host BW (spec)  : {SYSTEM_SPECS.pcie_host_bw_gbs:.0f} GB/s (all GPUs)",
        _sep(),
    ]

    for g in gpus:
        spec = g["spec"]
        lines += [
            f"  GPU {g['gpu_index']}  ─  {g['name']}",
            f"    Compute capability : {g.get('compute_capability', 'n/a')}  "
            f"(expected 9.0 for H100)",
            f"    Total HBM memory   : {g.get('total_memory_gb', 'n/a')} GB  "
            f"(spec: {spec['hbm3_capacity_gb']} GB)",
            f"    SM count           : {g.get('multi_processor_count', 'n/a')}",
        ]

        if "driver_version" in g:
            lines.append(f"    Driver version     : {g['driver_version']}")

        if "pci_bus_id" in g:
            lines.append(f"    PCI Bus ID         : {g['pci_bus_id']}")
            lines.append(
                f"    PCI domain/bus/dev : {g.get('pci_domain','?')} / "
                f"{g.get('pci_bus','?')} / {g.get('pci_device','?')}"
            )

        if "pcie_max_gen" in g:
            gen_status = "OK" if g.get("pcie_gen_healthy") else "DEGRADED !"
            wid_status = "OK" if g.get("pcie_width_healthy") else "DEGRADED !"
            lines += [
                f"    PCIe max gen       : Gen {g['pcie_max_gen']}  "
                f"(spec: Gen {spec['pcie_gen']})",
                f"    PCIe current gen   : Gen {g['pcie_curr_gen']}  [{gen_status}]",
                f"    PCIe max width     : ×{g['pcie_max_width']}  "
                f"(spec: ×{spec['pcie_lanes']})",
                f"    PCIe current width : ×{g['pcie_curr_width']}  [{wid_status}]",
                f"    PCIe BW (spec)     : {spec['pcie_unidirectional_bw_gbs']} GB/s "
                f"unidirectional",
            ]

        _add_if(lines, g, "temperature_c", "Temperature",    "°C")
        if "power_draw_w" in g:
            lines.append(
                f"    Power draw / limit : {g['power_draw_w']} W / {g['power_limit_w']} W  "
                f"(TDP spec: 700 W)"
            )

        _add_if(lines, g, "ecc_enabled",   "ECC enabled",    "")

        if "clock_sm_mhz" in g:
            lines.append(
                f"    SM clock curr/max  : {g['clock_sm_mhz']} / "
                f"{g.get('max_clock_sm_mhz', 'n/a')} MHz"
            )
        _add_if(lines, g, "clock_mem_mhz", "Mem clock",      "MHz")

        if "nvlink_active_links" in g:
            lines.append(
                f"    NVLink links       : {g['nvlink_active_links']} active / "
                f"{g['nvlink_total_links']} total  "
                f"(spec BW: {spec['nvlink_total_bw_gbs']} GB/s)"
            )

        lines.append(_sep())

    lines += [
        "  Theoretical compute totals (4 × H100)",
        f"    FP64  : {SYSTEM_SPECS.gpu_count * H100_SPECS.fp64_tflops:.0f} TFLOPS",
        f"    FP32  : {SYSTEM_SPECS.total_fp32_tflops_gpu:.0f} TFLOPS",
        f"    TF32  : {SYSTEM_SPECS.gpu_count * H100_SPECS.tf32_tflops:.0f} TFLOPS",
        f"    FP16  : {SYSTEM_SPECS.total_fp16_tflops_gpu:.0f} TFLOPS",
        f"    BF16  : {SYSTEM_SPECS.total_bf16_tflops_gpu:.0f} TFLOPS",
        f"    INT8  : {SYSTEM_SPECS.total_int8_tops_gpu:.0f} TOPS",
        f"    FP8   : {SYSTEM_SPECS.total_fp8_tflops_gpu:.0f} TFLOPS",
        _sep("═"),
    ]
    print("\n".join(lines))


def _add_if(lines: List[str], g: Dict[str, Any], key: str,
            label: str, unit: str) -> None:
    if key in g:
        val = g[key]
        lines.append(f"    {label:<19}: {val} {unit}".rstrip())


# ---------------------------------------------------------------------------
# TC-REPORT-01  CPU section
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestCPUConfigReport:

    def test_collect_and_print_cpu_config(self):
        """TC-REPORT-01: Collect full CPU topology and print to test output."""
        cpu_info = _collect_cpu_info()
        _print_cpu_section(cpu_info)

        # Basic sanity: report must contain required keys
        assert "os_platform"   in cpu_info
        assert "logical_cpus"  in cpu_info
        assert "spec"          in cpu_info
        assert cpu_info["spec"]["physical_cores"] == CPU_SPECS.total_physical_cores

    def test_cpu_spec_conforms_to_theoretical(self):
        """TC-REPORT-01b: Detected CPU core count must match theoretical spec (if psutil present)."""
        if not _HAS_PSUTIL:
            pytest.skip("psutil not installed")
        cpu_info = _collect_cpu_info()
        assert cpu_info.get("physical_cores") == CPU_SPECS.total_physical_cores, (
            f"Physical cores: detected {cpu_info.get('physical_cores')}, "
            f"spec {CPU_SPECS.total_physical_cores}"
        )
        assert cpu_info.get("logical_threads") == CPU_SPECS.total_logical_threads, (
            f"Logical threads: detected {cpu_info.get('logical_threads')}, "
            f"spec {CPU_SPECS.total_logical_threads}"
        )


# ---------------------------------------------------------------------------
# TC-REPORT-02  GPU section
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestGPUConfigReport:

    def test_collect_and_print_gpu_config(self):
        """TC-REPORT-02: Collect full GPU configuration and print to test output."""
        if not _HAS_TORCH:
            pytest.skip("PyTorch / CUDA not available")
        gpu_info = _collect_gpu_info()
        _print_gpu_section(gpu_info)

        assert len(gpu_info) > 0, "No GPUs detected"
        for g in gpu_info:
            assert "name"          in g
            assert "total_memory_gb" in g
            assert "spec"          in g

    def test_gpu_model_names_in_report(self):
        """TC-REPORT-02b: Each GPU entry must name an H100 or H200 class device."""
        if not _HAS_TORCH:
            pytest.skip("PyTorch / CUDA not available")
        gpu_info = _collect_gpu_info()
        for g in gpu_info:
            name = g["name"]
            assert any(kw in name for kw in ("H100", "H200")), (
                f"GPU {g['gpu_index']} name '{name}' is not H100/H200"
            )

    def test_total_gpu_memory_in_report(self):
        """TC-REPORT-02c: Total HBM across all GPUs must be ≥ 315 GB."""
        if not _HAS_TORCH:
            pytest.skip("PyTorch / CUDA not available")
        gpu_info = _collect_gpu_info()
        total_gb = sum(g.get("total_memory_gb", 0) for g in gpu_info)
        assert total_gb >= 315.0, (
            f"Total GPU memory {total_gb:.1f} GB < 315 GB"
        )


# ---------------------------------------------------------------------------
# TC-REPORT-03  PCIe topology section
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.pcie
class TestPCIeTopologyReport:

    def test_pcie_topology_present_in_report(self):
        """TC-REPORT-03: PCIe topology data must be populated for all GPUs."""
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        gpu_info = _collect_gpu_info()
        for g in gpu_info:
            assert "pci_bus_id" in g, (
                f"GPU {g['gpu_index']}: PCIe bus ID missing from report"
            )
            assert "pcie_max_gen" in g, (
                f"GPU {g['gpu_index']}: PCIe gen info missing from report"
            )

    def test_pcie_links_healthy_in_report(self):
        """TC-REPORT-03b: No PCIe gen or width downgrade must appear in the report."""
        if not _HAS_NVML:
            pytest.skip("pynvml not installed")
        gpu_info = _collect_gpu_info()
        for g in gpu_info:
            if "pcie_gen_healthy" in g:
                assert g["pcie_gen_healthy"], (
                    f"GPU {g['gpu_index']}: PCIe gen degraded "
                    f"(curr Gen {g.get('pcie_curr_gen')} < max Gen {g.get('pcie_max_gen')})"
                )
            if "pcie_width_healthy" in g:
                assert g["pcie_width_healthy"], (
                    f"GPU {g['gpu_index']}: PCIe width degraded "
                    f"(curr ×{g.get('pcie_curr_width')} < max ×{g.get('pcie_max_width')})"
                )


# ---------------------------------------------------------------------------
# TC-REPORT-04  Persist combined report to JSON
# ---------------------------------------------------------------------------
class TestCombinedConfigReport:

    def test_generate_and_save_combined_report(self):
        """TC-REPORT-04: Build the full CPU + GPU report and write it to JSON."""
        report: Dict[str, Any] = {
            "report_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cpu": _collect_cpu_info(),
            "gpus": _collect_gpu_info(),
            "system_spec_totals": {
                "gpu_count":              SYSTEM_SPECS.gpu_count,
                "total_gpu_memory_gb":    SYSTEM_SPECS.total_gpu_memory_gb,
                "total_hbm_bandwidth_tbs":SYSTEM_SPECS.total_hbm_bandwidth_tbs,
                "total_fp16_tflops":      SYSTEM_SPECS.total_fp16_tflops_gpu,
                "total_bf16_tflops":      SYSTEM_SPECS.total_bf16_tflops_gpu,
                "total_int8_tops":        SYSTEM_SPECS.total_int8_tops_gpu,
                "nvlink_fabric_bw_gbs":   SYSTEM_SPECS.nvlink_fabric_bw_gbs,
                "pcie_host_bw_gbs":       SYSTEM_SPECS.pcie_host_bw_gbs,
                "roofline_ai_threshold":  round(SYSTEM_SPECS.arithmetic_intensity_roofline, 2),
            },
        }

        _REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved → {_REPORT_PATH}")

        # Verify the file round-trips correctly
        loaded = json.loads(_REPORT_PATH.read_text())
        assert "cpu"  in loaded
        assert "gpus" in loaded
        assert "report_timestamp" in loaded
