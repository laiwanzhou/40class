$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\Anaconda\envs\pyTorch2.7\python.exe"

Push-Location $ProjectRoot
try {
    & $Python -m src.train_depth_ir_pose_roi_40class @args
    if ($LASTEXITCODE -ne 0) { throw "40-class training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
