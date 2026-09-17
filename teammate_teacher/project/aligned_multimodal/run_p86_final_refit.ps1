param(
    [string]$Python = "C:\ProgramData\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runs = Join-Path $PSScriptRoot "runs"
$VisualRun = Join-Path $Runs "p86_visual_mc3_temporal_t16_r160_layer2_final2470_v2"
$SequenceRun = Join-Path $Runs "p86_mc3_sequence_layer2_final2470_v2"
$MotionRun = Join-Path $Runs "p86_mobind_lite_pretrain_final2470_v1"
$FusionRun = Join-Path $Runs "p86_mobind_fusion_separate_final2470_v1"
$PipelineLog = Join-Path $Runs "p86_final2470_pipeline.log"

function Write-PipelineStatus {
    param([string]$Message)
    $line = "{0:o} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $PipelineLog -Value $line -Encoding utf8
}

function Invoke-PythonStage {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$OutputDirectory
    )
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $stdout = Join-Path $OutputDirectory "${Name}_stdout.log"
    $stderr = Join-Path $OutputDirectory "${Name}_stderr.log"
    Write-PipelineStatus "START $Name"
    # Windows PowerShell wraps any native stderr line as an ErrorRecord.  Torch
    # emits benign warnings there, so temporarily avoid terminating on stderr
    # and use the native exit code as the only success criterion.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 1> $stdout 2> $stderr
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($nativeExitCode -ne 0) {
        Write-PipelineStatus "FAILED $Name exit=$nativeExitCode"
        throw "$Name failed with exit code $nativeExitCode"
    }
    Write-PipelineStatus "DONE $Name"
}

Set-Location $Repo
Write-PipelineStatus "Pipeline attached to frozen P86 final-refit recipe"

$visualPidPath = Join-Path $VisualRun "process.pid"
if (-not (Test-Path (Join-Path $VisualRun "visual_student.pt"))) {
    if (-not (Test-Path $visualPidPath)) {
        throw "Visual checkpoint and process.pid are both missing"
    }
    $visualPid = [int](Get-Content $visualPidPath -Raw).Trim()
    Write-PipelineStatus "WAIT visual pid=$visualPid"
    while (Get-Process -Id $visualPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }
}
if (
    -not (Test-Path (Join-Path $VisualRun "visual_student.pt")) -or
    -not (Test-Path (Join-Path $VisualRun "summary.json"))
) {
    throw "Visual process ended without a complete checkpoint and summary"
}
Write-PipelineStatus "READY visual checkpoint"

if (-not (Test-Path (Join-Path $SequenceRun "summary.json"))) {
    Invoke-PythonStage -Name "cache" -OutputDirectory $SequenceRun -Arguments @(
        "aligned_multimodal/build_p86_mc3_sequence_cache.py",
        "--checkpoint", (Join-Path $VisualRun "visual_student.pt"),
        "--pixel-cache", "aligned_multimodal/runs/p86_visual_pixel_cache_t16_r160_v12",
        "--output-dir", $SequenceRun,
        "--batch-size", "4",
        "--workers", "0"
    )
}

if (-not (Test-Path (Join-Path $MotionRun "mobind_lite.pt"))) {
    Invoke-PythonStage -Name "motion" -OutputDirectory $MotionRun -Arguments @(
        "aligned_multimodal/train_p86_mobind_pretrain.py",
        "--output-dir", $MotionRun,
        "--final-refit",
        "--defer-final-validation",
        "--epochs", "24",
        "--batch-size", "64",
        "--workers", "0",
        "--learning-rate", "0.0004",
        "--minimum-learning-rate", "0.00002",
        "--token-weight", "0.25",
        "--imu-teacher-logits", "aligned_multimodal/runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz",
        "--imu-teacher-weight", "1.0"
    )
}

if (-not (Test-Path (Join-Path $FusionRun "unified_student.pt"))) {
    Invoke-PythonStage -Name "fusion" -OutputDirectory $FusionRun -Arguments @(
        "aligned_multimodal/train_p86_mobind_fusion_proxy.py",
        "--visual-checkpoint", (Join-Path $VisualRun "visual_student.pt"),
        "--sequence-cache", $SequenceRun,
        "--pretrain-checkpoint", (Join-Path $MotionRun "mobind_lite.pt"),
        "--output-dir", $FusionRun,
        "--modality", "separate",
        "--final-refit",
        "--stage-a-epochs", "4",
        "--stage-b-epochs", "20",
        "--batch-size", "64",
        "--workers", "0",
        "--fusion-learning-rate", "0.0004",
        "--encoder-learning-rate", "0.0001",
        "--visual-learning-rate", "0.00002",
        "--minimum-learning-rate", "0.00001",
        "--weight-decay", "0.05",
        "--class-weight-power", "0.35",
        "--label-smoothing", "0.08",
        "--distillation-temperature", "2.0",
        "--distillation-weight", "1.0",
        "--relation-weight", "0.1",
        "--motion-aux-weight", "0.35",
        "--selective-anchor-weight", "0.3",
        "--reliability-weight", "0.0",
        "--visual-corruption-probability", "0.75",
        "--visual-feature-dropout", "0.3",
        "--visual-view-dropout", "0.4",
        "--joint-freeze-pretrained-encoders"
    )
}

Invoke-PythonStage -Name "audit" -OutputDirectory $FusionRun -Arguments @(
    "aligned_multimodal/audit_p86_fusion_predictions.py",
    "--run-dir", $FusionRun
)
Write-PipelineStatus "PIPELINE_COMPLETE"
