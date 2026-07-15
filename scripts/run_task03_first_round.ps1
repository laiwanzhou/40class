param(
    [int]$Epochs = 2
)

$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\pyTorch2.7\python.exe"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$OutputRoot = Join-Path $ProjectRoot "outputs\task03"
$Modalities = @("imu", "skeleton", "radar", "ir", "thermal", "depth_color")

if (-not (Test-Path -LiteralPath $PY)) {
    throw "Required Python interpreter not found: $PY"
}
if ($Epochs -le 0) {
    throw "Epochs must be positive."
}

Push-Location $ProjectRoot
try {
    foreach ($Modality in $Modalities) {
        $Config = Join-Path $ProjectRoot "configs\task03\$Modality.yaml"
        Write-Host "Starting $Modality for $Epochs epochs with num_workers=0"
        & $PY -m src.train_unimodal `
            --config $Config `
            --output-root $OutputRoot `
            --max-epochs $Epochs `
            --num-workers 0
        if ($LASTEXITCODE -ne 0) {
            throw "$Modality training failed with exit code $LASTEXITCODE"
        }
        $Run = Get-ChildItem -LiteralPath (Join-Path $OutputRoot $Modality) -Directory |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        foreach ($Required in @("best_model.pt", "metrics.json", "fold_0_val_predictions.npz")) {
            $RequiredPath = Join-Path $Run.FullName $Required
            if (-not (Test-Path -LiteralPath $RequiredPath)) {
                throw "$Modality did not produce $RequiredPath"
            }
        }
        Write-Host "Completed ${Modality}: $($Run.FullName)"
    }
    & $PY scripts\summarize_task03.py --output-root $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Task 03 summary generation failed."
    }
}
finally {
    Pop-Location
}
