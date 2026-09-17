$ErrorActionPreference = "Continue"

$projectDir = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $PSScriptRoot "runs\p86_visual_mc3_spatial_region_proxy_v15"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Set-Location $projectDir

$arguments = @(
    "aligned_multimodal/train_p86_visual_pixel_oof.py"
    "--mode", "hybrid"
    "--backbone", "mc3_18_temporal"
    "--fusion-mode", "gated"
    "--freeze-through", "layer2"
    "--frames", "16"
    "--input-resolution", "160"
    "--pixel-cache", "aligned_multimodal/runs/p86_visual_pixel_cache_t16_r160_v12"
    "--output-dir", "aligned_multimodal/runs/p86_visual_mc3_spatial_region_proxy_v15"
    "--proxy-fixed-epochs", "16"
    "--batch-size", "4"
    "--gradient-accumulation", "4"
    "--workers", "0"
    "--head-learning-rate", "0.0002"
    "--backbone-learning-rate", "0.00001"
    "--minimum-learning-rate", "0.00001"
    "--weight-decay", "0.08"
    "--class-weight-power", "0.35"
    "--distillation-temperature", "2.0"
    "--distillation-weight", "1.0"
    "--stage-distillation-weight", "0.0"
    "--relation-weight", "0.2"
    "--label-smoothing", "0.1"
    "--augmentation-mode", "subject_robust"
    "--seed", "20260811"
    "--spatial-region-modeling"
)

$logPath = Join-Path $outputDir "live_training_retry.log"
& python @arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logPath
$trainingExitCode = $LASTEXITCODE
Write-Host "P86 spatial-region proxy finished with exit code $trainingExitCode"
if ($trainingExitCode -ne 0) {
    throw "P86 spatial-region proxy failed with exit code $trainingExitCode"
}
