$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PY = "D:\Anaconda\envs\pyTorch2.7\python.exe"
$Config = Join-Path $ProjectRoot "configs\experiments\depth_six_patch.yaml"
$OutputRoot = Join-Path $ProjectRoot "outputs\depth_six_patch_fold0_14train_4val"
$FormalRun = Join-Path $OutputRoot "depth_color\depth_six_patch_fold0_14train_4val"

Set-Location $ProjectRoot
& $PY -m compileall -q src scripts\visualize_depth_six_patch.py scripts\probe_depth_six_patch.py scripts\summarize_depth_six_patch.py
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

& $PY scripts\visualize_depth_six_patch.py
if ($LASTEXITCODE -ne 0) { throw "visual checks failed" }

& $PY scripts\probe_depth_six_patch.py
if ($LASTEXITCODE -ne 0) { throw "batch=4 probe failed; inspect whether this was a real CUDA OOM" }

& $PY -m src.train_unimodal `
    --config $Config `
    --output-root (Join-Path $OutputRoot "smoke") `
    --run-id "smoke_depth_six_patch" `
    --smoke-test `
    --max-epochs 1 `
    --max-train-batches 1 `
    --max-val-batches 1 `
    --num-workers 0
if ($LASTEXITCODE -ne 0) { throw "smoke run failed" }

if (Test-Path -LiteralPath $FormalRun) {
    throw "Formal run directory already exists: $FormalRun"
}
& $PY -m src.train_unimodal `
    --config $Config `
    --output-root $OutputRoot `
    --run-id "depth_six_patch_fold0_14train_4val"
if ($LASTEXITCODE -ne 0) { throw "formal training failed" }

& $PY scripts\summarize_depth_six_patch.py --run-dir $FormalRun
if ($LASTEXITCODE -ne 0) { throw "result summary failed" }
