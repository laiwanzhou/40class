$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PY = "D:\Anaconda\envs\pyTorch2.7\python.exe"
$OutputRoot = Join-Path $ProjectRoot "outputs\task03_baseline_fold0"
$Modalities = @("imu", "skeleton", "radar", "ir", "thermal", "depth_color")
$TemporalModalities = @("imu", "skeleton", "radar")

function Read-YamlInteger {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Key
    )
    $Match = Select-String -LiteralPath $Path -Pattern ("^{0}:\s*(\d+)\s*$" -f [regex]::Escape($Key))
    if ($null -eq $Match -or $Match.Matches.Count -ne 1) {
        throw "Could not read one integer '$Key' from $Path"
    }
    return [int]$Match.Matches[0].Groups[1].Value
}

Set-Location $ProjectRoot
foreach ($Modality in $Modalities) {
    $ConfigPath = Join-Path $ProjectRoot "configs\task03\$Modality.yaml"
    $Epochs = Read-YamlInteger -Path $ConfigPath -Key "epochs"
    $Patience = Read-YamlInteger -Path $ConfigPath -Key "early_stopping_patience"
    $ExpectedEpochs = if ($TemporalModalities -contains $Modality) { 40 } else { 30 }
    $ExpectedPatience = if ($TemporalModalities -contains $Modality) { 8 } else { 6 }
    if ($Epochs -ne $ExpectedEpochs -or $Patience -ne $ExpectedPatience) {
        throw "Formal config mismatch for ${Modality}: epochs=$Epochs patience=$Patience"
    }

    $RunId = "baseline_fold0_20260716_$Modality"
    Write-Host "Starting ${Modality}: epochs=$Epochs patience=$Patience run_id=$RunId"
    $CommandOutput = & $PY -m src.train_unimodal `
        --config $ConfigPath `
        --output-root $OutputRoot `
        --run-id $RunId 2>&1 | Tee-Object -Variable CapturedOutput
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Formal baseline failed for ${Modality} with exit code $ExitCode`n$($CapturedOutput -join [Environment]::NewLine)"
    }

    $ResultLine = @($CapturedOutput | Where-Object { "$_" -like "RESULT_JSON=*" }) | Select-Object -Last 1
    if ($null -eq $ResultLine) {
        throw "Formal baseline for $Modality did not emit RESULT_JSON."
    }
    $Result = ("$ResultLine" -replace '^RESULT_JSON=', '') | ConvertFrom-Json
    $RunDir = [System.IO.Path]::GetFullPath([string]$Result.output_dir)
    $ExpectedRoot = [System.IO.Path]::GetFullPath($OutputRoot) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $RunDir.StartsWith($ExpectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected output directory for ${Modality}: $RunDir"
    }

    $RequiredFiles = @(
        "best_model.pt",
        "last_model.pt",
        "config.yaml",
        "history.csv",
        "metrics.json",
        "fold_0_val_predictions.npz",
        "confusion_matrix.png"
    )
    if ($TemporalModalities -contains $Modality) {
        $RequiredFiles += "normalization_stats.json"
    }
    foreach ($RequiredFile in $RequiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $RunDir $RequiredFile))) {
            throw "Missing $RequiredFile after formal baseline for $Modality at $RunDir"
        }
    }
    Write-Host "Completed $Modality at $RunDir"
}

& $PY scripts\summarize_task03_baseline_fold0.py
if ($LASTEXITCODE -ne 0) {
    throw "Formal baseline summary generation failed with exit code $LASTEXITCODE"
}
