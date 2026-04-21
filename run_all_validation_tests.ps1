param(
    [switch]$PerformanceCatalogOnly,
    [string]$PerformanceCaseIds = "",
    [switch]$StopOnFailure,
    [switch]$SkipConsolidatedReport,
    [string]$ExtraPytestArgs = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Stage {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Running stage: $Name" -ForegroundColor Cyan
    Write-Host "Command: $Command" -ForegroundColor DarkGray
    Write-Host "============================================================" -ForegroundColor Cyan

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & powershell -NoLogo -NoProfile -Command $Command
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    return [PSCustomObject]@{
        Name = $Name
        ExitCode = $exitCode
        DurationSec = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
    }
}

if (-not (Get-Command pytest -ErrorAction SilentlyContinue)) {
    throw "pytest not found in PATH. Activate your environment and install dependencies first."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

try {
    $env:PERF_EXECUTE_CASES = if ($PerformanceCatalogOnly) { "0" } else { "1" }
    $env:ML_EXECUTE_CASES = if ($PerformanceCatalogOnly) { "0" } else { "1" }
    if ([string]::IsNullOrWhiteSpace($PerformanceCaseIds)) {
        Remove-Item Env:PERF_CASE_IDS -ErrorAction SilentlyContinue
    } else {
        $env:PERF_CASE_IDS = $PerformanceCaseIds
    }

    $stages = @()
    $stages += [PSCustomObject]@{
        Name = "Bottleneck Tests"
        Command = "pytest -m bottleneck -vv tests/test_bottleneck_analysis.py $ExtraPytestArgs"
    }
    $stages += [PSCustomObject]@{
        Name = "Node Validation Tests"
        Command = "pytest -m node_validation -vv tests/test_node_validation.py $ExtraPytestArgs"
    }
    $stages += [PSCustomObject]@{
        Name = "Workload Acceptance"
        Command = "pytest -m workload_acceptance -vv tests/test_node_validation.py tests/test_vllm_prefill_validation.py $ExtraPytestArgs"
    }
    $stages += [PSCustomObject]@{
        Name = "Performance Suite"
        Command = "pytest -vv tests/test_performance_suite.py -k test_performance_case_suite $ExtraPytestArgs"
    }
    $stages += [PSCustomObject]@{
        Name = "MLPerf Suite"
        Command = "pytest -m mlperf -vv tests/test_mlperf_suite.py -k test_mlperf_case_suite $ExtraPytestArgs"
    }

    $results = @()
    foreach ($stage in $stages) {
        $result = Invoke-Stage -Name $stage.Name -Command $stage.Command
        $results += $result

        if ($result.ExitCode -ne 0 -and $StopOnFailure) {
            break
        }
    }

    if (-not $SkipConsolidatedReport) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Refreshing consolidated markdown report" -ForegroundColor Cyan
        $consolidatedCmd = "pytest -vv tests/ $ExtraPytestArgs"
        Write-Host "Command: $consolidatedCmd" -ForegroundColor DarkGray
        Write-Host "============================================================" -ForegroundColor Cyan

        & powershell -NoLogo -NoProfile -Command $consolidatedCmd
        $consolidatedExitCode = $LASTEXITCODE
        $results += [PSCustomObject]@{
            Name = "Consolidated Report"
            ExitCode = $consolidatedExitCode
            DurationSec = "n/a"
        }
    }

    Write-Host ""
    Write-Host "==================== Overall Summary ====================" -ForegroundColor Yellow
    $failed = $false
    foreach ($r in $results) {
        $status = if ($r.ExitCode -eq 0) { "PASS" } else { "FAIL" }
        $color = if ($r.ExitCode -eq 0) { "Green" } else { "Red" }
        Write-Host ("{0,-24} {1,-6} {2,8}s" -f $r.Name, $status, $r.DurationSec) -ForegroundColor $color
        if ($r.ExitCode -ne 0) {
            $failed = $true
        }
    }

    Write-Host "=========================================================" -ForegroundColor Yellow

    if ($failed) {
        exit 1
    }

    exit 0
}
finally {
    Pop-Location
}
