param(
    [string]$RunDirectory = "C:\kaggle_CUNK_X\small model\Small-Model-Track\aligned_multimodal\runs\p46_unified_repair_v2",
    [string]$ProtocolLabel = "V2",
    [string]$ProcessPattern = "p46_unified_repair_v2",
    [switch]$Once
)

$ErrorActionPreference = "SilentlyContinue"
$Host.UI.RawUI.WindowTitle = "P46 Unified Repair $ProtocolLabel - Live Training Monitor"

function Format-Percent([object]$Value) {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return "-"
    }
    return ("{0:N2}%" -f (100.0 * [double]$Value))
}

while ($true) {
    Clear-Host
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "P46 Unified Repair $ProtocolLabel - LIVE" -ForegroundColor Cyan
    Write-Host "Updated: $now    Refresh: 2s" -ForegroundColor DarkGray
    Write-Host "Closing this window does NOT stop training." -ForegroundColor Yellow
    Write-Host "Run: $RunDirectory"
    Write-Host ""

    $trainingProcesses = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match $ProcessPattern }
    if ($trainingProcesses) {
        $processIds = ($trainingProcesses | ForEach-Object { $_.ProcessId }) -join ", "
        Write-Host "TRAINING: RUNNING   PID(s): $processIds" -ForegroundColor Green
    } else {
        Write-Host "TRAINING: NOT RUNNING" -ForegroundColor Red
    }

    $os = Get-CimInstance Win32_OperatingSystem
    $freeGiB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    $totalGiB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    Write-Host "System RAM free: $freeGiB / $totalGiB GiB"

    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia) {
        $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
        if ($gpu) {
            $parts = $gpu -split ",\s*"
            Write-Host "GPU: $($parts[0])%   VRAM: $($parts[1])/$($parts[2]) MiB   Temp: $($parts[3]) C   Power: $($parts[4]) W"
        }
    }
    Write-Host ""

    $stageAPath = Join-Path $RunDirectory "stage_a_history.csv"
    $stageBPath = Join-Path $RunDirectory "history.csv"
    $summaryPath = Join-Path $RunDirectory "summary.json"

    $stageA = @()
    if (Test-Path -LiteralPath $stageAPath) {
        $stageA = @(Import-Csv -LiteralPath $stageAPath)
    }
    if ($stageA.Count -lt 4) {
        Write-Host "STAGE A: $($stageA.Count) / 4 completed" -ForegroundColor Magenta
        if ($stageA.Count -gt 0) {
            $lastA = $stageA[-1]
            Write-Host ("Latest A epoch {0}: loss={1:N4}, alignment={2:N4}, seconds={3:N1}, exact_once={4}" -f [int]$lastA.epoch, [double]$lastA.total, [double]$lastA.alignment, [double]$lastA.seconds, $lastA.coverage_exact_once)
        }
        Write-Host "Next: Stage A 4/4 -> process recycle -> Stage B 1/30"
    } else {
        $stageB = @()
        if (Test-Path -LiteralPath $stageBPath) {
            $stageB = @(Import-Csv -LiteralPath $stageBPath)
        }
        Write-Host "STAGE A: 4 / 4 completed" -ForegroundColor DarkGreen
        Write-Host "STAGE B: $($stageB.Count) / 30 completed (early stop: min 8, patience 5)" -ForegroundColor Magenta

        if ($stageB.Count -gt 0) {
            $bestScore = [double]::NegativeInfinity
            $bestEpoch = 0
            $stale = 0
            foreach ($row in $stageB) {
                $score = [double]$row.val_accuracy + 0.5 * [double]$row.val_macro_f1 + 0.25 * [double]$row.val_worst_user_accuracy
                if ($score -gt ($bestScore + 0.001)) {
                    $bestScore = $score
                    $bestEpoch = [int]$row.epoch
                    $stale = 0
                } else {
                    $stale += 1
                }
            }
            $last = $stageB[-1]
            Write-Host ""
            Write-Host ("Latest epoch: {0}   train acc: {1}   val acc: {2}" -f [int]$last.epoch, (Format-Percent $last.train_accuracy), (Format-Percent $last.val_accuracy)) -ForegroundColor White
            Write-Host ("Val balanced: {0}   macro F1: {1}   worst-user: {2}" -f (Format-Percent $last.val_balanced_accuracy), (Format-Percent $last.val_macro_f1), (Format-Percent $last.val_worst_user_accuracy))
            Write-Host ("Best selection score: {0:N5} at epoch {1}" -f $bestScore, $bestEpoch) -ForegroundColor Green
            Write-Host ("Early-stop stale counter: {0} / 5 (active after epoch 8)" -f $stale) -ForegroundColor Yellow
            Write-Host ("Offset acc: {0}   effective offset weight: {1}   localization weight: {2}" -f (Format-Percent $last.train_offset_accuracy), $last.train_effective_offset_weight, $last.train_effective_localization_weight)
            Write-Host ""
            Write-Host "Last five Stage-B epochs:" -ForegroundColor Cyan
            $stageB | Select-Object -Last 5 |
                Select-Object epoch,
                    @{Name="train_acc"; Expression={ Format-Percent $_.train_accuracy }},
                    @{Name="val_acc"; Expression={ Format-Percent $_.val_accuracy }},
                    @{Name="macro_f1"; Expression={ Format-Percent $_.val_macro_f1 }},
                    @{Name="worst_user"; Expression={ Format-Percent $_.val_worst_user_accuracy }},
                    @{Name="seconds"; Expression={ "{0:N0}" -f [double]$_.train_seconds }} |
                Format-Table -AutoSize
        } else {
            Write-Host "Stage B first epoch is starting..." -ForegroundColor Yellow
        }
    }

    if (Test-Path -LiteralPath $summaryPath) {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        if ($summary.stop_reason -eq "validation_plateau") {
            Write-Host ""
            Write-Host "EARLY STOP COMPLETE" -ForegroundColor Green
        } elseif ($summary.stage -eq "P46_unified_repair_complete") {
            Write-Host ""
            Write-Host "TRAINING COMPLETE" -ForegroundColor Green
        }
    }

    if ($Once) {
        break
    }
    Start-Sleep -Seconds 2
}
