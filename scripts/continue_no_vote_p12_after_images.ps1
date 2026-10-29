param([Parameter(Mandatory=$true)][int]$WorkerPid)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskOutput=Join-Path $taskRoot 'outputs/task5_audit'
$taskProtocol=Join-Path $taskRoot 'outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops/protocol/task11_reconstruction_v2'
$taskPython='D:/Anaconda/envs/PyTorch2.7/python.exe'
$taskLog=Join-Path $taskOutput 'P42-p12-global-continuation.log'
$taskErrorLog=Join-Path $taskOutput 'P42-p12-global-continuation-error.log'
Set-Location -LiteralPath $taskRoot
$env:PYTHONDONTWRITEBYTECODE='1'
try {
    $taskProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$WorkerPid"
    if ($taskProcess) {
        if ($taskProcess.CommandLine -notlike '*scripts/run_no_vote_p12_images.py*') { throw 'Worker PID belongs to another command' }
        Write-Output "Waiting for verified P12 image worker $WorkerPid"
        Wait-Process -Id $WorkerPid -ErrorAction SilentlyContinue
    }
    $taskState=Get-Content -LiteralPath (Join-Path $taskProtocol 'progress.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskState.status -eq 'paused_by_user') { Write-Output 'User paused the task; continuation stopped'; exit 0 }
    foreach ($taskRole in @('skeleton','depth','thermal')) {
        $taskMarker=Join-Path $taskProtocol ($taskRole+'_fp16_component.json')
        if (-not (Test-Path -LiteralPath $taskMarker)) { throw "Incomplete image components: $taskRole; will not fit P12/global" }
    }
    Write-Output 'Fresh image component markers exist; building phase-correct P12/global/KNN'
    $taskProducer=Start-Process -FilePath $taskPython -ArgumentList @('-B','-u','scripts/build_no_vote_p12_global.py','--config','configs/experiments/teammate_single_teacher_fixed_split_v3.yaml') -WorkingDirectory $taskRoot -RedirectStandardOutput $taskLog -RedirectStandardError $taskErrorLog -WindowStyle Hidden -PassThru
    $taskProducer.Id | Set-Content -LiteralPath (Join-Path $taskOutput 'P42-p12-global.pid')
    $taskProducer.WaitForExit()
    if ($taskProducer.ExitCode -ne 0) { throw "P12/global producer failed with exit code $($taskProducer.ExitCode); see $taskErrorLog" }
    if (-not (Test-Path -LiteralPath (Join-Path $taskProtocol 'p12_global_knn_component.json'))) { throw 'P12/global completion marker absent' }
    Write-Output 'P12/global/KNN component finished; Task11 source-base/bank/A8 still pending; Task12 not started'
} catch {
    Write-Error $_
    exit 1
}
