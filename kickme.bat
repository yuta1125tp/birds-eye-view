@echo off
setlocal enabledelayedexpansion
@REM set img_path=images\DSC_3895.JPG
@REM set img_path=C:\Users\yshira\Work\birdeye\SuperGluePretrainedNetwork\assets\scannet_sample_images
@REM set img_dir=images
set img_dir=C:\Users\yshira\Work\birdeye\SuperGluePretrainedNetwork\assets\scannet_sample_images
set model_name=inception-v4

for %%i in (%img_dir%\*) do (
    set img_path=%%i
    set img_stem=%%~ni
    set img_ext=%%~xi
    python scripts\predict_horizon_vpz_homography.py ^
        --img_path %img_dir%\!img_stem!!img_ext! ^
        --model_name %model_name%
)
    
