$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PY = "D:\Anaconda\envs\pyTorch2.7\python.exe"
$GlobalConfig = Join-Path $ProjectRoot "configs\experiments\depth_hard_global_expert.yaml"
$RoiConfig = Join-Path $ProjectRoot "configs\experiments\depth_pose_roi_expert.yaml"
$PoseCache = Join-Path $ProjectRoot "outputs\depth_pose_roi_probe\pose_tracks.npz"
$GlobalRun = Join-Path $ProjectRoot "outputs\depth_hard_global_expert_fold0\depth_color\depth_hard_global_expert_fold0"
$RoiRun = Join-Path $ProjectRoot "outputs\depth_pose_roi_expert_fold0\depth_color\depth_pose_roi_expert_fold0"

Set-Location $ProjectRoot
& $PY -m compileall -q src scripts\build_pose_roi_cache.py scripts\visualize_pose_roi.py scripts\probe_pose_roi_training.py scripts\summarize_pose_roi_experiment.py
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

if (-not (Test-Path -LiteralPath $PoseCache)) {
    & $PY scripts\build_pose_roi_cache.py
    if ($LASTEXITCODE -ne 0) { throw "pose cache build failed" }
}
& $PY scripts\visualize_pose_roi.py
if ($LASTEXITCODE -ne 0) { throw "ROI visualization failed" }
& $PY scripts\probe_pose_roi_training.py
if ($LASTEXITCODE -ne 0) { throw "batch=4 training probe failed" }

if (Test-Path -LiteralPath $GlobalRun) { throw "E0 formal run already exists: $GlobalRun" }
& $PY -m src.train_unimodal --config $GlobalConfig --output-root (Join-Path $ProjectRoot "outputs\depth_hard_global_expert_fold0") --run-id "depth_hard_global_expert_fold0"
if ($LASTEXITCODE -ne 0) { throw "E0 formal training failed" }

if (Test-Path -LiteralPath $RoiRun) { throw "E1 formal run already exists: $RoiRun" }
& $PY -m src.train_unimodal --config $RoiConfig --output-root (Join-Path $ProjectRoot "outputs\depth_pose_roi_expert_fold0") --run-id "depth_pose_roi_expert_fold0"
if ($LASTEXITCODE -ne 0) { throw "E1 formal training failed" }

& $PY scripts\summarize_pose_roi_experiment.py
if ($LASTEXITCODE -ne 0) { throw "experiment summary failed" }
