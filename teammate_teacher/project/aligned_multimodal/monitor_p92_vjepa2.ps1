param(
    [switch]$Once
)

$ErrorActionPreference = 'SilentlyContinue'

$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runDir = Join-Path $workspace 'runs\p92_vjepa2_vitl_ssv2_12view_fold0_v1'
$cacheSummaryPath = Join-Path $runDir 'cache_summary.json'
$stdoutPath = Join-Path $runDir 'extract.stdout.log'
$stderrPath = Join-Path $runDir 'extract.stderr.log'
$summaryPath = Join-Path $runDir 'fold0_summary.json'

function Show-P92Status {
    Clear-Host
    Write-Host 'P92 V-JEPA2 Fold0 Teacher-Ceiling Monitor' -ForegroundColor Cyan
    Write-Host ('Updated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host ''

    $teacher = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^python' -and
            $_.CommandLine -like '*p92_vjepa2_visual_teacher.py*'
        } |
        Select-Object -First 1

    if ($teacher) {
        Write-Host ('Experiment: RUNNING  PID=' + $teacher.ProcessId) -ForegroundColor Green
    }
    else {
        Write-Host 'Experiment: PROCESS FINISHED' -ForegroundColor Yellow
    }

    if (Test-Path -LiteralPath $cacheSummaryPath) {
        $cacheSummary = Get-Content -LiteralPath $cacheSummaryPath -Raw | ConvertFrom-Json
        if ($cacheSummary -and $cacheSummary.total_samples -gt 0) {
            $percent = 100.0 * [double]$cacheSummary.completed_samples / [double]$cacheSummary.total_samples
            $progress = '{0}/{1} ({2:N1}%)' -f $cacheSummary.completed_samples, $cacheSummary.total_samples, $percent
            Write-Host ('Feature extraction: ' + $progress) -ForegroundColor Green
        }
        else {
            Write-Host 'Feature extraction: status file is being updated; waiting for next refresh'
        }
    }
    else {
        Write-Host 'Feature extraction: waiting for status file'
    }

    if (Test-Path -LiteralPath $summaryPath) {
        Write-Host ''
        Write-Host 'Fold0 audit result is ready:' -ForegroundColor Green
        Write-Host $summaryPath
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        if ($summary) {
            $summary | ConvertTo-Json -Depth 5
        }
    }

    Write-Host ''
    Write-Host 'Latest standard output:' -ForegroundColor Cyan
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath -Tail 12
    }
    else {
        Write-Host 'No standard output yet.'
    }

    if (Test-Path -LiteralPath $stderrPath) {
        $notices = Get-Content -LiteralPath $stderrPath -Tail 8 |
            Where-Object {
                $_ -and
                $_ -notmatch 'Loading weights' -and
                $_ -notmatch '^\s*\d+%\|' -and
                $_ -notmatch 'it/s'
            }
        if ($notices) {
            Write-Host ''
            Write-Host 'Runtime notices (model-loading progress is hidden):' -ForegroundColor DarkYellow
            $notices
        }
    }

    Write-Host ''
    if ($teacher) {
        Write-Host 'Auto-refresh: 10 seconds. Closing this window will NOT stop the experiment.' -ForegroundColor DarkGray
    }
    elseif (Test-Path -LiteralPath $summaryPath) {
        Write-Host 'Experiment and automatic audit are complete. This window may be closed.' -ForegroundColor Green
    }
    else {
        Write-Host 'Process ended without an audit summary. Inspect the latest output above.' -ForegroundColor Yellow
    }

    return [bool]$teacher
}

do {
    $isRunning = Show-P92Status
    if ($Once -or -not $isRunning) {
        break
    }
    Start-Sleep -Seconds 10
} while ($true)
