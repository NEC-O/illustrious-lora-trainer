@echo off
REM Windows SD LoRA 训练启动脚本 (3060 6G优化)

REM 设置模型路径（需要根据实际情况修改）
set MODEL_PATH=./model/Illustrious-XL-v2.0-FP16-Diffusers

REM 设置训练数据目录
set TRAIN_DATA_DIR=./train_data/12_style

REM 设置输出目录
set OUTPUT_DIR=./output_lora

REM 训练参数（针对 3060 6GB显存优化，目标 <2h 完成）
.\.venv\Scripts\python.exe -u train_lora.py ^
    --pretrained_model_name_or_path "%MODEL_PATH%" ^
    --train_data_dir "%TRAIN_DATA_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --resolution 512,768 ^
    --gradient_accumulation_steps 2 ^
    --learning_rate 3.5e-4 ^
    --max_train_steps 1000 ^
    --save_every_n_steps 600 ^
    --log_every_n_steps 10 ^
    --lr_warmup_steps 100 ^
    --network_rank 4 ^
    --network_alpha 8 ^
    --seed 42

pause
