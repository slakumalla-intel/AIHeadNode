param(
    [int]$DurationHours = 12,
    [int]$CpuCores = 224,
    [int]$TotalMemGB = 371,
    [string]$DiskPath = "/",
    [int]$FioSizeGB = 600,
    [int]$FioNumJobs = 8,
    [int]$FioIoDepth = 64,
    [string]$FioBlockSize = "1M",
    [int]$SnapshotIntervalSec = 60,
    [string]$LogDir = "artifacts/soak_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$LlamaCmd = ""
)

# Wrapper script intentionally delegates execution to the Linux runner.
# PowerShell wrapper for Linux soak runner. Use this from a PowerShell session
# that has access to the Linux node (native Linux PowerShell, WSL, or SSH shell).

if ([string]::IsNullOrWhiteSpace($LlamaCmd)) {
    $LlamaCmd = $env:LLAMA_TRAIN_CMD
}
if ([string]::IsNullOrWhiteSpace($LlamaCmd)) {
    Write-Error "Llama training command is required. Set LLAMA_TRAIN_CMD or pass -LlamaCmd."
    exit 2
}

if (-not (Get-Command bash -ErrorAction SilentlyContinue)) {
    Write-Error "bash not found. Run on Linux/WSL or install bash in your environment."
    exit 2
}

$env:LLAMA_TRAIN_CMD = $LlamaCmd

$bashArgs = @(
    "scripts/run_parallel_soak_12h.sh",
    "--duration-hours", "$DurationHours",
    "--cpu-cores", "$CpuCores",
    "--total-mem-gb", "$TotalMemGB",
    "--disk-path", "$DiskPath",
    "--fio-size-gb", "$FioSizeGB",
    "--fio-numjobs", "$FioNumJobs",
    "--fio-iodepth", "$FioIoDepth",
    "--fio-bs", "$FioBlockSize",
    "--snapshot-interval-sec", "$SnapshotIntervalSec",
    "--log-dir", "$LogDir"
)

Write-Host "[INFO] Delegating to Linux soak runner with 12h parallel workload profile..."
& bash @bashArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
