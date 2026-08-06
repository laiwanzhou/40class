param(
    [string]$Python = "D:\Anaconda\envs\pyTorch2.7\python.exe",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

& $Python -m src.train_depth_ir_person_crop_cross_user_supcon @RemainingArgs
exit $LASTEXITCODE
