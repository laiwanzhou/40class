$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "D:\Anaconda\envs\pyTorch2.7\python.exe"
Set-Location $root
& $python -m src.train_target16_linear_residual_e2 @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts\analyze_target16_hierarchical_e2.py `
    --e2-dir outputs\depth_ir_target16_linear_residual_e2_fold0\target16_linear_residual_e2_14train_4val `
    --report-prefix target16_linear_residual_e2 `
    --experiment-label "Target16 linear residual E2"
exit $LASTEXITCODE
