$ErrorActionPreference = "Stop"

$python = "D:\Anaconda\envs\pyTorch2.7\python.exe"
$config = "configs\experiments\depth_ir_person_crop_cross_user_prototype_supcon.yaml"
$runId = "depth_ir_person_crop_cross_user_prototype_supcon_14train_4val"

& $python -u -m src.train_depth_ir_person_crop_cross_user_supcon `
    --config $config --run-id $runId
