param(
    [string]$Python = "C:\ProgramData\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runs = Join-Path $PSScriptRoot "runs"
$RunnerLog = Join-Path $Runs "p86v2_h01_runner.log"

function Write-RunnerLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $RunnerLog -Value $line -Encoding UTF8
}

function Get-OtherTrainingProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -match 'aligned_multimodal[/\\](train|run)_' -and
        $_.CommandLine -notmatch 'train_p86v2_visual\.py'
    })
}

function Wait-ForGpuResearchSlot {
    Write-RunnerLog "waiting for other repository training processes"
    while ($true) {
        while (@(Get-OtherTrainingProcesses).Count -gt 0) {
            Start-Sleep -Seconds 10
        }
        # Require a quiet interval so another research window can launch its
        # already-planned follow-up without racing this queue.
        Start-Sleep -Seconds 20
        if (@(Get-OtherTrainingProcesses).Count -eq 0) {
            break
        }
    }
    Write-RunnerLog "GPU research slot acquired"
}

function Invoke-P86V2Run(
    [string]$Name,
    [string]$Architecture,
    [bool]$Smoke
) {
    $Output = Join-Path $Runs $Name
    $Stdout = Join-Path $Runs "${Name}_stdout.log"
    $Stderr = Join-Path $Runs "${Name}_stderr.log"
    $Arguments = @(
        "aligned_multimodal/train_p86v2_visual.py",
        "--architecture", $Architecture,
        "--split", "development",
        "--output-dir", "aligned_multimodal/runs/$Name"
    )
    if ($Smoke) {
        $Arguments += "--smoke"
    }
    Write-RunnerLog "starting $Name"
    $process = Start-Process -FilePath $Python -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Write-RunnerLog "completed $Name"
}

Wait-ForGpuResearchSlot
Invoke-P86V2Run "p86v2_visual_baseline_smoke_v1" "p86v1_baseline" $true
Invoke-P86V2Run "p86v2_visual_temporal_dedup_smoke_v1" "temporal_dedup" $true
Invoke-P86V2Run "p86v2_visual_baseline_dev_a_v1" "p86v1_baseline" $false
Invoke-P86V2Run "p86v2_visual_temporal_dedup_dev_a_v1" "temporal_dedup" $false
Write-RunnerLog "H0/H1 queue complete"
