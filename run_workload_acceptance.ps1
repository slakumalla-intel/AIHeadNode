param(
    [switch]$IncludeLargeModels,
    [switch]$SkipLargeModels,
    [int]$MinDurationSec = 300,
    [string]$KpiFile = "artifacts/node_validation_kpis.jsonl",
    [string[]]$ExtraPytestArgs = @(),
    [string]$KVCacheContextLength = "8192",
    [string]$KVCacheBatchSize = "4",
    [string]$KVCacheNumPrompts = "10",
    [string]$YOLOBatchSize = "128",
    [string]$YOLOImgSize = "640"
)

function Set-DefaultEnv {
    param(
        [string]$Name,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($Name))) {
        [Environment]::SetEnvironmentVariable($Name, $Value)
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

# =============================================================================
# HuggingFace & Large Model Configuration
# =============================================================================
if ($SkipLargeModels) {
    $env:HF_ENABLE_LARGE_MODELS = "0"
} else {
    # Default to enabled so 70B enterprise workloads execute unless explicitly skipped.
    $env:HF_ENABLE_LARGE_MODELS = "1"
}

# Backward-compatible explicit enable switch (no-op when already enabled).
if ($IncludeLargeModels) {
    $env:HF_ENABLE_LARGE_MODELS = "1"
}

$env:HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC = [string]$MinDurationSec
$env:NODE_VALIDATION_KPI_FILE = $KpiFile

# Model IDs (override these via environment before calling script)
Set-DefaultEnv "HF_LLAMA_7B_MODEL" "meta-llama/Llama-2-7b-hf"
Set-DefaultEnv "HF_LLAMA_70B_MODEL" "meta-llama/Llama-2-70b-hf"
Set-DefaultEnv "HF_QWEN_70B_MODEL" "Qwen/Qwen-72B"
Set-DefaultEnv "HF_DEEPSEEK_R1_QUANT_MODEL" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
Set-DefaultEnv "HF_INFERENCE_MODEL" "meta-llama/Llama-2-7b-hf"
Set-DefaultEnv "HF_TRAINING_MODEL" "bert-base-uncased"

# =============================================================================
# KV Cache Offload Configuration
# =============================================================================
Set-DefaultEnv "NODE_VLLM_KV_OFFLOAD_MODEL" "meta-llama/Llama-2-7b-hf"
$env:NODE_VLLM_KV_CONTEXT_LENGTH = $KVCacheContextLength
$env:NODE_VLLM_KV_BATCH_SIZE = $KVCacheBatchSize
$env:NODE_VLLM_KV_NUM_PROMPTS = $KVCacheNumPrompts
Set-DefaultEnv "NODE_VLLM_KV_MAX_TOKENS" "512"
Set-DefaultEnv "NODE_VLLM_BASELINE_GPU_MEMORY_UTIL" "0.9"
Set-DefaultEnv "NODE_VLLM_GPU_MEMORY_UTIL" "0.7"
Set-DefaultEnv "NODE_VLLM_CPU_OFFLOAD_GB" "128"

# Remove legacy names to avoid vLLM unknown-environment warnings.
Remove-Item Env:VLLM_KV_OFFLOAD_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_KV_CONTEXT_LENGTH -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_KV_BATCH_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_KV_NUM_PROMPTS -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_KV_MAX_TOKENS -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_BASELINE_GPU_MEMORY_UTIL -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_GPU_MEMORY_UTIL -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_CPU_OFFLOAD_GB -ErrorAction SilentlyContinue

# RDMA / MMIO / DMA validation defaults
Set-DefaultEnv "RDMA_SMOKE_ITERS" "1000"
Set-DefaultEnv "RDMA_USE_CUDA_DEVICE" "0"
Set-DefaultEnv "RDMA_4KB_MAX_AVG_LAT_US" "300.0"
Set-DefaultEnv "RDMA_SOAK_LOOPS" "1000"
Set-DefaultEnv "SOAK_MAX_P95_US" "20000"

# =============================================================================
# Vision Workload Configuration (YOLO)
# =============================================================================
Set-DefaultEnv "YOLO_DETECT_MODEL" "yolov8x.pt"
$env:YOLO_DETECT_BATCH = $YOLOBatchSize
$env:YOLO_DETECT_IMGSZ = $YOLOImgSize
Set-DefaultEnv "YOLO_DETECT_SOURCE" "https://ultralytics.com/images/bus.jpg"
Set-DefaultEnv "YOLO_DETECT_TIMEOUT_SEC" "240"
Set-DefaultEnv "YOLO_CPU_SAMPLE_INTERVAL_SEC" "0.5"
Set-DefaultEnv "YOLO_AGGREGATE_FPS_TARGET" "100"

# =============================================================================
# Vision Workload Configuration (Diffusers)
# =============================================================================
Set-DefaultEnv "DIFFUSERS_NUM_IMAGES" "50"
Set-DefaultEnv "DIFFUSERS_HEIGHT" "512"
Set-DefaultEnv "DIFFUSERS_WIDTH" "512"
Set-DefaultEnv "DIFFUSERS_BATCH_SIZE" "4"

# =============================================================================
# HuggingFace Cache and Download Configuration
# =============================================================================
Set-DefaultEnv "HF_HOME" (Join-Path $env:USERPROFILE ".cache" "huggingface")
Set-DefaultEnv "HF_TOKEN" ""
Set-DefaultEnv "HUGGINGFACE_HUB_CACHE" (Join-Path $env:HF_HOME "hub")

# Enable remote model downloads (large models won't download by default without this)
Set-DefaultEnv "HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD" "1"

# =============================================================================
# PyTorch Configuration
# =============================================================================
Set-DefaultEnv "TORCH_HOME" (Join-Path $env:USERPROFILE ".cache" "torch")

# =============================================================================
# CUDA Configuration
# =============================================================================
Set-DefaultEnv "CUDA_VISIBLE_DEVICES" "0,1,2,3"
Set-DefaultEnv "CUDA_DEVICE_ORDER" "PCI_BUS_ID"

# =============================================================================
# Logging Configuration
# =============================================================================
$env:PYTHONUNBUFFERED = "1"
Set-DefaultEnv "PYTEST_TIMEOUT" "600"

# =============================================================================
# Display Configuration Summary
# =============================================================================
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Workload Acceptance Test Configuration" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "HuggingFace & Large Models:" -ForegroundColor Yellow
Write-Host "  HF_ENABLE_LARGE_MODELS:                 $($env:HF_ENABLE_LARGE_MODELS)"
Write-Host "  HF_HOME:                                 $($env:HF_HOME)"
Write-Host "  HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD:   $($env:HF_ALLOW_REMOTE_LARGE_MODEL_DOWNLOAD)"
Write-Host ""

Write-Host "KPI & Test Configuration:" -ForegroundColor Yellow
Write-Host "  NODE_VALIDATION_KPI_FILE:                $($env:NODE_VALIDATION_KPI_FILE)"
Write-Host "  HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC:  $($env:HF_ENTERPRISE_LATENCY_MIN_DURATION_SEC)"
Write-Host ""

Write-Host "KV Cache Offload Test:" -ForegroundColor Yellow
Write-Host "  NODE_VLLM_KV_OFFLOAD_MODEL:              $($env:NODE_VLLM_KV_OFFLOAD_MODEL)"
Write-Host "  NODE_VLLM_KV_CONTEXT_LENGTH:             $($env:NODE_VLLM_KV_CONTEXT_LENGTH)"
Write-Host "  NODE_VLLM_KV_BATCH_SIZE:                 $($env:NODE_VLLM_KV_BATCH_SIZE)"
Write-Host "  NODE_VLLM_KV_NUM_PROMPTS:                $($env:NODE_VLLM_KV_NUM_PROMPTS)"
Write-Host "  NODE_VLLM_KV_MAX_TOKENS:                 $($env:NODE_VLLM_KV_MAX_TOKENS)"
Write-Host "  NODE_VLLM_BASELINE_GPU_MEMORY_UTIL:      $($env:NODE_VLLM_BASELINE_GPU_MEMORY_UTIL)"
Write-Host "  NODE_VLLM_GPU_MEMORY_UTIL:               $($env:NODE_VLLM_GPU_MEMORY_UTIL)"
Write-Host "  NODE_VLLM_CPU_OFFLOAD_GB:                $($env:NODE_VLLM_CPU_OFFLOAD_GB)"
Write-Host ""

Write-Host "YOLO Vision Workload:" -ForegroundColor Yellow
Write-Host "  YOLO_DETECT_MODEL:                       $($env:YOLO_DETECT_MODEL)"
Write-Host "  YOLO_DETECT_BATCH:                       $($env:YOLO_DETECT_BATCH)"
Write-Host "  YOLO_DETECT_IMGSZ:                       $($env:YOLO_DETECT_IMGSZ)"
Write-Host ""

Write-Host "Diffusers Vision Workload:" -ForegroundColor Yellow
Write-Host "  DIFFUSERS_NUM_IMAGES:                    $($env:DIFFUSERS_NUM_IMAGES)"
Write-Host "  DIFFUSERS_HEIGHT:                        $($env:DIFFUSERS_HEIGHT)"
Write-Host "  DIFFUSERS_WIDTH:                         $($env:DIFFUSERS_WIDTH)"
Write-Host ""

Write-Host "Computing:" -ForegroundColor Yellow
Write-Host "  CUDA_VISIBLE_DEVICES:                    $($env:CUDA_VISIBLE_DEVICES)"
Write-Host "  TORCH_HOME:                              $($env:TORCH_HOME)"
Write-Host ""

Write-Host "Tips:" -ForegroundColor Cyan
Write-Host "  - Override defaults via environment variables before calling this script"
Write-Host "  - Example: `$env:NODE_VLLM_KV_CONTEXT_LENGTH = '16384' before running"
Write-Host "  - Use -SkipLargeModels to disable 70B+ models"
Write-Host "  - Use -KVCacheContextLength to customize KV cache test context length"
Write-Host ""
Write-Host "Example Usage:" -ForegroundColor Cyan
Write-Host "  # Default: Llama-2-7B with 8K context"
Write-Host "  .\run_workload_acceptance.ps1"
Write-Host ""
Write-Host "  # Custom: 32K context, larger batch"
Write-Host "  .\run_workload_acceptance.ps1 -KVCacheContextLength 32768 -KVCacheBatchSize 8"
Write-Host ""
Write-Host "  # Custom: Skip large models, run quick validation"
Write-Host "  .\run_workload_acceptance.ps1 -SkipLargeModels -MinDurationSec 60"
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

& pytest -m workload_acceptance -vv @ExtraPytestArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
