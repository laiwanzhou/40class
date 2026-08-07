$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "D:\Anaconda\envs\pyTorch2.7\python.exe"
Set-Location $root
& $python -m src.train_target16_conditional_e2 @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts\analyze_target16_hierarchical_e2.py
exit $LASTEXITCODE
