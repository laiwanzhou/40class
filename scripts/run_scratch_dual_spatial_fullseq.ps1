$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "D:\Anaconda\envs\pyTorch2.7\python.exe"
Set-Location $root
& $python -m src.train_scratch_dual_spatial_fullseq @args
exit $LASTEXITCODE
