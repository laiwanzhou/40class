param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunsRoot = Join-Path $PSScriptRoot "runs"
$PythonExe = (Get-Command python).Source

function Invoke-P86Python {
    param(
        [string[]]$Arguments,
        [string]$StdoutName,
        [string]$StderrName
    )
    $stdoutPath = Join-Path $RunsRoot $StdoutName
    $stderrPath = Join-Path $RunsRoot $StderrName
    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Python command failed with exit code $($process.ExitCode). See $stderrPath"
    }
}

if ($WaitForPid -gt 0) {
    $existing = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Wait-Process -Id $WaitForPid
    }
}

Push-Location $ProjectRoot
try {
    $common = @(
        "aligned_multimodal\train_p86_visual_pixel_oof.py",
        "--backbone", "mc3_18",
        "--frames", "12",
        "--pixel-cache", "aligned_multimodal\runs\p86_visual_pixel_cache_t12_v8",
        "--mode", "hybrid",
        "--fusion-mode", "gated",
        "--freeze-through", "layer2",
        "--batch-size", "2",
        "--gradient-accumulation", "8",
        "--head-learning-rate", "2e-4",
        "--backbone-learning-rate", "2e-5"
    )

    Invoke-P86Python `
        -Arguments ($common + @(
            "--outer-fold", "1",
            "--output-dir", "aligned_multimodal\runs\p86_visual_mc3_fold1_v9_fixed_smoke",
            "--fixed-refit-epochs", "1",
            "--max-train-batches", "1",
            "--max-eval-batches", "1",
            "--smoke"
        )) `
        -StdoutName "p86_visual_mc3_fold1_v9_fixed_smoke.out.log" `
        -StderrName "p86_visual_mc3_fold1_v9_fixed_smoke.err.log"

    foreach ($fold in @(1, 2)) {
        Invoke-P86Python `
            -Arguments ($common + @(
                "--outer-fold", "$fold",
                "--output-dir", "aligned_multimodal\runs\p86_visual_mc3_fold${fold}_v9",
                "--fixed-refit-epochs", "16"
            )) `
            -StdoutName "p86_visual_mc3_fold${fold}_v9.out.log" `
            -StderrName "p86_visual_mc3_fold${fold}_v9.err.log"
    }

    Invoke-P86Python `
        -Arguments @(
            "aligned_multimodal\analyze_p86_visual_threefold_oof.py"
        ) `
        -StdoutName "p86_visual_mc3_oof_v9_audit.out.log" `
        -StderrName "p86_visual_mc3_oof_v9_audit.err.log"
}
finally {
    Pop-Location
}
