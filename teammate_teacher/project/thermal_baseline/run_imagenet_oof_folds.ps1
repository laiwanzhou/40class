$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\ncy\.conda\envs\deeplearning\python.exe"
$config = Join-Path $PSScriptRoot "configs\resnet18_tsm_imagenet_12f.json"
$runRoot = Join-Path $PSScriptRoot "runs\p11_thermal_imagenet"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$runnerLog = Join-Path $runRoot "runner.log"
"Started $(Get-Date -Format o)" | Set-Content -LiteralPath $runnerLog

foreach ($fold in 0..2) {
    $manifest = Join-Path $PSScriptRoot "data\subject_folds\fold_$fold.csv"
    $output = Join-Path $runRoot "fold_$fold"
    $log = Join-Path $runRoot "fold_$fold.log"
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    "Fold $fold started $(Get-Date -Format o)" | Add-Content -LiteralPath $runnerLog
    & $python -u (Join-Path $PSScriptRoot "train_imagenet_oof.py") `
        --config $config `
        --manifest $manifest `
        --output-dir $output *>&1 |
        Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        "Fold $fold failed with exit code $LASTEXITCODE" | Add-Content -LiteralPath $runnerLog
        exit $LASTEXITCODE
    }
    "Fold $fold completed $(Get-Date -Format o)" | Add-Content -LiteralPath $runnerLog
}

"Completed $(Get-Date -Format o)" | Add-Content -LiteralPath $runnerLog
