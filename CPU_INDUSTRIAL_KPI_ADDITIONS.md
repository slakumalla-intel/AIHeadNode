# Industrial Standard CPU KPIs - Test Suite Additions

## Overview
Added comprehensive industrial standard CPU Key Performance Indicators (KPIs) to the test suite in `tests/test_cpu_validation.py`. These additions capture enterprise-grade performance metrics essential for production workload validation on AI Head Node systems.

## New Industrial Standard KPI Tests

### TC-CPU-16: CPU Utilization and Per-Core Metrics
- **Test Class:** `TestCPUUtilization`
- **Metrics Captured:**
  - Average CPU utilization (%)
  - Per-core utilization breakdown
  - Minimum and maximum utilization
  - Context switch rate
  - Interrupt count
  - System call frequency
- **Purpose:** Validates CPU utilization is balanced and no cores are idle
- **Requirements:** psutil library
- **Threshold:** Max utilization > 0% (sanity check)

### TC-CPU-17: Context Switch Rate and Scheduling Metrics
- **Test Class:** `TestCPUUtilization::test_context_switch_rate()`
- **Metrics Captured:**
  - Context switch delta during workload
  - Scheduling efficiency
- **Purpose:** Ensures proper process scheduling and context switching behavior
- **Requirements:** psutil library, multi-threaded workload
- **Expected Behavior:** Context switches should increase under load

### TC-CPU-18: CPU Frequency Scaling and Turbo Boost Tracking
- **Test Class:** `TestCPUFrequencyScaling`
- **Metrics Captured:**
  - Current CPU frequency (MHz)
  - Minimum frequency (MHz)
  - Maximum frequency (MHz)
  - Frequency scaling percentage
  - Turbo boost verification
- **Purpose:** Validates dynamic frequency scaling (P-states) and turbo boost functionality
- **Requirements:** psutil library, CPU frequency scaling support
- **Threshold:** Turbo boost > 2000 MHz (for Xeon Platinum)

### TC-CPU-19: Cache Miss Rates (L1, L2, L3) via Performance Counters
- **Test Class:** `TestPerformanceCounters`
- **Metrics Captured:**
  - Cache-references count
  - Cache-misses count
  - Cache miss rate (%)
  - LLC (Last Level Cache) loads/misses
- **Purpose:** Identifies cache efficiency and memory access patterns
- **Requirements:** Linux with `perf` tool available, elevated privileges may be needed
- **Data Source:** Linux perf subsystem

### TC-CPU-20: Instructions Per Cycle (IPC) and Execution Efficiency
- **Test Class:** `TestPerformanceCounters`
- **Metrics Captured:**
  - IPC (Instructions Per Cycle)
  - Total instructions executed
  - Total CPU cycles
  - Instructions per cycle ratio
- **Purpose:** Measures instruction-level parallelism and CPU execution efficiency
- **Requirements:** Linux with `perf` tool
- **Expected Range:** 0.5-3.0 for typical workloads, >2.0 for optimized code
- **Interpretation:**
  - Low IPC (<1.0): Possible branch misses, memory stalls
  - High IPC (>2.0): Good instruction parallelism, cache locality

### TC-CPU-21: System Load Average and Queue Depth
- **Test Class:** `TestSystemLoad`
- **Metrics Captured:**
  - 1-minute load average
  - 5-minute load average
  - 15-minute load average
  - Load ratio per CPU (load / num_cpus)
  - Logical CPU count
- **Purpose:** Validates system scheduler queue depth and responsiveness
- **Requirements:** Unix-like systems (Linux, macOS) with `os.getloadavg()`
- **Interpretation:**
  - Ratio > 1.0: System overloaded
  - Ratio < 0.7: System has spare capacity

### TC-CPU-22: CPU Thermal Metrics and Temperature Stability
- **Test Class:** `TestCPUThermal`
- **Metrics Captured:**
  - Maximum CPU temperature (°C)
  - Average CPU temperature (°C)
  - Minimum CPU temperature (°C)
  - Per-sensor temperature readings
  - Thermal thresholds (high, critical)
- **Purpose:** Ensures thermal stability and detects thermal throttling conditions
- **Requirements:** psutil library with thermal sensor access
- **Thresholds:**
  - Normal: 20-80°C
  - Warning: 80-100°C
  - Critical: >100°C

### TC-CPU-23: Power Consumption via RAPL (Running Average Power Limit)
- **Test Class:** `TestCPUPower`
- **Metrics Captured:**
  - Per-socket energy consumption (microjoules)
  - Package power draw
  - RAPL energy counter readings
- **Purpose:** Measures actual CPU power consumption for efficiency analysis
- **Requirements:** Intel RAPL subsystem available (Xeon Platinum supports)
- **Data Source:** Linux sysfs `/sys/devices/virtual/powercap/`
- **Interpretation:**
  - Track energy delta over time to calculate wattage
  - Compare against theoretical TDP for validation

### TC-CPU-24: Memory Latency Percentiles (P50, P95, P99)
- **Test Class:** `TestMemoryLatency`
- **Metrics Captured:**
  - P50 latency (nanoseconds)
  - P95 latency (nanoseconds)
  - P99 latency (nanoseconds)
  - Minimum latency (nanoseconds)
  - Maximum latency (nanoseconds)
  - Average latency (nanoseconds)
- **Purpose:** Characterizes memory access latency distribution for performance analysis
- **Requirements:** numpy library
- **Expected Range:** 1-100 ns (for L1/L2 cache hits)
- **Interpretation:**
  - P50 ~5-10 ns: L1/L2 cache hit
  - P95 ~50-100 ns: L3 cache hit / DRAM miss beginning
  - P99 >100 ns: DRAM access

### TC-CPU-25: Interrupt Rate and System Call Frequency
- **Test Class:** `TestCPUUtilization`
- **Metrics Captured:**
  - Interrupt count
  - System call count
  - Context switches
- **Purpose:** Monitors system call overhead and interrupt activity
- **Requirements:** psutil library
- **Interpretation:**
  - High interrupt rate: Possible I/O contention
  - High syscall rate: Possible kernel overhead

## Helper Functions Added

### `_get_cpu_utilization_stats()`
Collects per-core and aggregate CPU utilization using psutil.

### `_get_load_average()`
Retrieves system load average and queue depth metrics.

### `_get_cpu_freq_stats()`
Extracts CPU frequency scaling information including turbo boost state.

### `_get_cpu_thermal_stats()`
Gathers CPU temperature and thermal sensor readings.

### `_measure_memory_latency()`
Measures memory access latency percentiles using NumPy arrays.

### `_try_run_perf_stat()`
Attempts to run Linux `perf stat` for performance counter collection.

### `_parse_perf_stat()`
Parses perf output to extract cache miss rates and IPC metrics.

### `_estimate_power_consumption()`
Reads RAPL energy counters for power consumption estimation.

## Markers Used
- `@pytest.mark.cpu`: All tests marked as CPU tests
- `@pytest.mark.slow`: Slow tests (frequency scaling, thermal measurements)

## Usage

### Run all CPU KPI tests:
```bash
pytest -m cpu tests/test_cpu_validation.py -v
```

### Run only industrial standard KPI tests:
```bash
pytest tests/test_cpu_validation.py::TestCPUUtilization -v
pytest tests/test_cpu_validation.py::TestCPUFrequencyScaling -v
pytest tests/test_cpu_validation.py::TestSystemLoad -v
pytest tests/test_cpu_validation.py::TestCPUThermal -v
pytest tests/test_cpu_validation.py::TestMemoryLatency -v
pytest tests/test_cpu_validation.py::TestPerformanceCounters -v
pytest tests/test_cpu_validation.py::TestCPUPower -v
```

### Run with detailed logging:
```bash
pytest -m cpu tests/test_cpu_validation.py -v -s --log-cli-level=DEBUG
```

## System Requirements

### Minimum:
- Python 3.8+
- psutil library (for utilization, frequency, thermal metrics)
- numpy library (for latency measurements)

### For Full Functionality:
- Linux system (for load average, perf, RAPL)
- `perf` tool installed for cache miss / IPC collection
- Elevated privileges for RAPL/perf access
- CPU thermal sensors (not all systems expose)

### Platform Support:
| Metric | Linux | macOS | Windows |
|--------|-------|-------|---------|
| CPU Utilization | ✓ | ✓ | ✓ |
| Context Switches | ✓ | ✓ | ✓ |
| Frequency Scaling | ✓ | ✓ | ✓ |
| Cache Misses (perf) | ✓ | ✗ | ✗ |
| IPC (perf) | ✓ | ✗ | ✗ |
| Load Average | ✓ | ✓ | ✗ |
| Temperature | ✓ | ~ | ~ |
| RAPL Power | ✓ | ✗ | ✗ |

## Expected Output

Tests will log structured KPI information:

```
================================================================================
  CPU UTILIZATION SNAPSHOT
================================================================================
  Average utilization:      45.32%
  Min per-core util:        12.50%
  Max per-core util:        78.90%
  Cores reporting:          160
  Context switches:         45821
  Interrupts:               123456
================================================================================

================================================================================
  CPU FREQUENCY SCALING
================================================================================
  Current frequency:        3400 MHz
  Min frequency:            1900 MHz
  Max frequency:            3900 MHz
  Scaling %:                87.2%
================================================================================

================================================================================
  MEMORY LATENCY PERCENTILES
================================================================================
  P50 latency:              8.5 ns
  P95 latency:              45.2 ns
  P99 latency:              127.3 ns
  Min latency:              2.1 ns
  Max latency:              1842.5 ns
  Avg latency:              38.7 ns
================================================================================
```

## Integration with CI/CD

These tests can be integrated into automated validation pipelines:

```yaml
# Example CI job
test_cpu_kpis:
  stage: validate
  script:
    - pytest -m cpu tests/test_cpu_validation.py -v --tb=short
  artifacts:
    reports:
      junit: test-results.xml
```

## References

- Intel Running Average Power Limit (RAPL): https://github.com/torvalds/linux/blob/master/Documentation/admin-guide/pm/intel_rapl.rst
- Linux Performance Counters: https://perf.wiki.kernel.org/
- psutil Documentation: https://psutil.readthedocs.io/
- Memory Latency Definition: https://en.wikipedia.org/wiki/Memory_latency
