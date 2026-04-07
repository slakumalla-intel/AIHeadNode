param(
    [switch]$IncludeLargeModels,
    [int]$MinDurationSec = 300,
    [string]$KpiFile = "artifacts/node_validation_kpis.jsonl",
    [string[]]$ExtraPytestArgs = @()
)

$env:HF_ENABLE_LARGE_MODELS = if ($IncludeLargeModels) { "1" } else { "0" }
$env:HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC = [string]$MinDurationSec
$env:NODE_VALIDATION_KPI_FILE = $KpiFile

Write-Host "Running workload acceptance tests with settings:"
Write-Host "  HF_ENABLE_LARGE_MODELS=$($env:HF_ENABLE_LARGE_MODELS)"
Write-Host "  HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC=$($env:HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC)"
Write-Host "  NODE_VALIDATION_KPI_FILE=$($env:NODE_VALIDATION_KPI_FILE)"
Write-Host ""
Write-Host "Tip: Override model IDs via env vars (HF_LLAMA_7B_MODEL, HF_LLAMA_70B_MODEL, HF_QWEN_70B_MODEL, HF_DEEPSEEK_R1_QUANT_MODEL)."

& pytest -m workload_acceptance -vv @ExtraPytestArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
