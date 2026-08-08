$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "D:\Anaconda\envs\pyTorch2.7\python.exe"
Set-Location $root
& $python -m src.train_full_sequence_multiscale_tcn @args
exit $LASTEXITCODE
