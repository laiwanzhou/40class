$monitor = Join-Path $PSScriptRoot "monitor_p46_unified_repair_v2.ps1"
$runDirectory = Join-Path $PSScriptRoot "runs\p46_unified_repair_v3_clean"

& $monitor `
    -RunDirectory $runDirectory `
    -ProtocolLabel "V3-CLEAN" `
    -ProcessPattern "p46_unified_repair_v3"
