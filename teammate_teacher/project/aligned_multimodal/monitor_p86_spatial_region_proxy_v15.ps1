param(
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$outputDir = Join-Path $PSScriptRoot "runs\p86_visual_mc3_spatial_region_proxy_v15"
$logPath = Join-Path $outputDir "live_training_retry.log"
$summaryPath = Join-Path $outputDir "summary.json"
$maximumEpochs = 16

do {
    if (-not $Once) {
        Clear-Host
    }
    Write-Host "P86 MC3 2x2 Spatial-Region Raw Confirmation" -ForegroundColor Cyan
    Write-Host "Protocol: fixed 1497 train -> 973 proxy, permanent 444 untouched"
    Write-Host "Cached screen reference: Accuracy 64.44% | Macro-F1 59.31% | Worst 50.34%"
    Write-Host ""

    $epochRecord = $null
    if (Test-Path $logPath) {
        $epochLine = Get-Content $logPath | Where-Object { $_ -match '^\{"epoch"' } |
            Select-Object -Last 1
        if ($epochLine) {
            $epochRecord = $epochLine | ConvertFrom-Json
        }
    }

    if (Test-Path $summaryPath) {
        $summary = Get-Content -Raw -Encoding UTF8 $summaryPath | ConvertFrom-Json
        $metrics = $summary.proxy_metrics
        Write-Host "Status: COMPLETED" -ForegroundColor Green
        Write-Host ("Epoch: {0}/{1}" -f $summary.fixed_epochs, $maximumEpochs)
        Write-Host ("Accuracy: {0:P2} ({1}/{2})" -f $metrics.accuracy, $metrics.correct, $metrics.total)
        Write-Host ("Macro-F1: {0:P2}" -f $metrics.macro_f1)
        Write-Host ("Balanced Accuracy: {0:P2}" -f $metrics.balanced_accuracy)
        Write-Host ("Worst-subject Accuracy: {0:P2}" -f $metrics.worst_subject_accuracy)
        Write-Host ("Selection Score: {0:F5}" -f $summary.selection_score)
        if (-not $Once) {
            Write-Host ""
            Read-Host "Training finished. Press Enter to close"
        }
        break
    }

    $completedEpoch = if ($epochRecord) { [int]$epochRecord.epoch } else { 0 }
    $activeEpoch = [Math]::Min($completedEpoch + 1, $maximumEpochs)
    Write-Host "Status: RUNNING" -ForegroundColor Yellow
    Write-Host ("Active epoch: {0}/{1} | completed: {2}/{1} | progress: {3:N1}%" -f `
        $activeEpoch, $maximumEpochs, $completedEpoch, (100.0 * $completedEpoch / $maximumEpochs))
    if ($epochRecord) {
        Write-Host ("Latest train loss: {0:F4} | CE: {1:F4} | KD: {2:F4}" -f `
            $epochRecord.train_loss, $epochRecord.train_ce, $epochRecord.train_kd)
        Write-Host ("Relation loss: {0:F4} | epoch time: {1:N1}s" -f `
            $epochRecord.train_relation, $epochRecord.seconds)
    } else {
        Write-Host "Latest train metrics: waiting for epoch 1 to finish"
    }
    Write-Host "Raw-run Accuracy/Macro-F1: final one-shot evaluation after epoch 16"
    Write-Host "Reason: no per-epoch proxy-set tuning; permanent validation remains untouched"
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw `
        --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) {
        Write-Host ("GPU: " + $gpu)
    }
    Write-Host ("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    if ($Once) {
        break
    }
    Start-Sleep -Seconds 5
} while ($true)
