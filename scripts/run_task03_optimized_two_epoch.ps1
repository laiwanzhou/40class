$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\Anaconda\envs\pyTorch2.7\python.exe"
$OutputRoot = Join-Path $ProjectRoot "outputs\task03_optimized_two_epoch"
$Modalities = @("imu", "skeleton", "radar", "ir", "thermal", "depth_color")

Set-Location $ProjectRoot
foreach ($Modality in $Modalities) {
    $RunId = "optimized_{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $Modality
    & $Python -m src.train_unimodal `
        --config (Join-Path $ProjectRoot "configs\task03\$Modality.yaml") `
        --output-root $OutputRoot `
        --run-id $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "Task 03 optimized run failed for $Modality with exit code $LASTEXITCODE"
    }
    $RunDir = Join-Path $OutputRoot "$Modality\$RunId"
    foreach ($Required in @("best_model.pt", "metrics.json", "fold_0_val_predictions.npz")) {
        if (-not (Test-Path -LiteralPath (Join-Path $RunDir $Required))) {
            throw "Missing $Required after optimized run for $Modality at $RunDir"
        }
    }
}

& $Python scripts\summarize_task03_optimized.py
if ($LASTEXITCODE -ne 0) {
    throw "Optimized Task 03 summary generation failed with exit code $LASTEXITCODE"
}
