[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$OutputsPath = Join-Path $ProjectRoot "outputs"
$KeepDirectories = @("task03_baseline_fold0")
$DeleteDirectories = @(
    "task03",
    "task03_smoke",
    "task03_optimized_smoke",
    "task03_optimized_two_epoch"
)
$Modalities = @("imu", "skeleton", "radar", "ir", "thermal", "depth_color")
$RequiredFiles = @(
    "best_model.pt",
    "last_model.pt",
    "config.yaml",
    "history.csv",
    "metrics.json",
    "fold_0_val_predictions.npz",
    "confusion_matrix.png"
)

function Get-DirectoryBytes {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return [int64]0
    }
    $measurement = Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction Stop |
        Measure-Object -Property Length -Sum
    return [int64]($measurement.Sum)
}

function Test-FormalBaselineStructure {
    param([Parameter(Mandatory = $true)][string]$BaselinePath)
    if (-not (Test-Path -LiteralPath $BaselinePath -PathType Container)) {
        throw "Protected formal Baseline directory is missing: $BaselinePath"
    }
    foreach ($modality in $Modalities) {
        $modalityPath = Join-Path $BaselinePath $modality
        $runs = @(Get-ChildItem -LiteralPath $modalityPath -Directory -ErrorAction Stop)
        if ($runs.Count -ne 1) {
            throw "Expected exactly one formal run for $modality, found $($runs.Count)."
        }
        foreach ($fileName in $RequiredFiles) {
            $requiredPath = Join-Path $runs[0].FullName $fileName
            if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
                throw "Formal Baseline file is missing: $requiredPath"
            }
        }
        if ($modality -in @("imu", "skeleton", "radar")) {
            $statsPath = Join-Path $runs[0].FullName "normalization_stats.json"
            if (-not (Test-Path -LiteralPath $statsPath -PathType Leaf)) {
                throw "Formal Baseline normalization file is missing: $statsPath"
            }
        }
    }
}

if (-not (Test-Path -LiteralPath $OutputsPath -PathType Container)) {
    Write-Host "Outputs directory does not exist; nothing to clean: $OutputsPath"
    return
}

$ResolvedOutputs = (Resolve-Path -LiteralPath $OutputsPath).Path
$OutputsPrefix = $ResolvedOutputs.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$BaselinePath = Join-Path $ResolvedOutputs "task03_baseline_fold0"
Test-FormalBaselineStructure -BaselinePath $BaselinePath

$BeforeBytes = Get-DirectoryBytes -Path $ResolvedOutputs
$ExistingNames = @(Get-ChildItem -LiteralPath $ResolvedOutputs -Directory | Select-Object -ExpandProperty Name)
$UnknownDirectories = @($ExistingNames | Where-Object {
    $_ -notin $KeepDirectories -and $_ -notin $DeleteDirectories
})

Write-Host "Mode: $(if ($Execute) { 'EXECUTE' } else { 'DRY-RUN' })"
Write-Host "Outputs before: $BeforeBytes bytes ($([math]::Round($BeforeBytes / 1MB, 3)) MiB)"
foreach ($name in $DeleteDirectories) {
    $target = Join-Path $ResolvedOutputs $name
    if (-not (Test-Path -LiteralPath $target -PathType Container)) {
        Write-Host "Already absent: $target"
        continue
    }
    $resolvedTarget = (Resolve-Path -LiteralPath $target).Path
    $leafName = Split-Path -Leaf $resolvedTarget
    if (-not $resolvedTarget.StartsWith($OutputsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside outputs: $resolvedTarget"
    }
    if ($leafName -ne $name -or $leafName -in $KeepDirectories -or $leafName -like "*baseline*") {
        throw "Refusing protected or unexpected path: $resolvedTarget"
    }
    $tracked = @(git -C $ProjectRoot ls-files -- "outputs/$leafName")
    if ($tracked.Count -gt 0) {
        throw "Refusing to remove Git-tracked paths under $resolvedTarget"
    }
    $bytes = Get-DirectoryBytes -Path $resolvedTarget
    Write-Host "Allowlisted: $resolvedTarget ($bytes bytes, $([math]::Round($bytes / 1MB, 3)) MiB)"
    if ($Execute -and $PSCmdlet.ShouldProcess($resolvedTarget, "Remove allowlisted Task 03 local output")) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
}

foreach ($name in $UnknownDirectories) {
    Write-Warning "manual_review_required: $(Join-Path $ResolvedOutputs $name)"
}

$AfterBytes = Get-DirectoryBytes -Path $ResolvedOutputs
Write-Host "Outputs after: $AfterBytes bytes ($([math]::Round($AfterBytes / 1MB, 3)) MiB)"
Write-Host "Space released: $($BeforeBytes - $AfterBytes) bytes ($([math]::Round(($BeforeBytes - $AfterBytes) / 1MB, 3)) MiB)"
if (-not $Execute) {
    Write-Host "Dry-run only. Re-run with -Execute to remove the explicit allowlist."
}
