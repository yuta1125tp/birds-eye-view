@echo off
setlocal enabledelayedexpansion
set img_path=images\DSC_3895.JPG
set model_name=inception-v4

for %%i in (./images/*) do (
    set img_path=./images/%%i
    python scripts\predict_horizon_vpz_homography.py ^
        --img_path !img_path! ^
        --model_name %model_name%

)
    
