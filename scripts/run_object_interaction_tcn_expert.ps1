param(
    [switch]$SmokeTest,
    [switch]$Probe
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\Anaconda\envs\pyTorch2.7\python.exe"
$Arguments = @("-m", "src.train_object_interaction_tcn_expert")
if ($SmokeTest) { $Arguments += "--smoke-test" }
if ($Probe) { $Arguments += "--probe" }

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
