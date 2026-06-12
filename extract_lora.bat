@echo off
REM ============================================================
REM LoRA 转换脚本: diffusers 风格 → ComfyUI kohya 通用格式
REM ============================================================
REM 用法（双击运行默认参数即可）:
REM   1. 直接双击: 转换默认的 output_lora\checkpoint-400
REM   2. 命令行带参: extract_lora.bat <checkpoint_dir> [output_path]
REM
REM 改下面的 DEFAULT_CHECKPOINT 即可改变默认要提取的 checkpoint 目录
REM ============================================================

setlocal enabledelayedexpansion

REM ---------- 默认配置（按需修改） ----------
set DEFAULT_CHECKPOINT=output_lora\checkpoint-final
set DEFAULT_OUTPUT=output_lora\ill_style_final.safetensors
set DEFAULT_ALPHA=8
set DEFAULT_RANK=4
set DEFAULT_DTYPE=float16

REM ---------- 解析参数 ----------
if not "%~1"=="" (
    set CHECKPOINT_DIR=%~1
) else (
    set CHECKPOINT_DIR=%DEFAULT_CHECKPOINT%
)

if not "%~2"=="" (
    set OUTPUT_PATH=%~2
) else (
    set OUTPUT_PATH=%DEFAULT_OUTPUT%
)

set ALPHA=%DEFAULT_ALPHA%
set RANK=%DEFAULT_RANK%
set DTYPE=%DEFAULT_DTYPE%

REM ---------- 检查输入文件 ----------
set INPUT_FILE=%CHECKPOINT_DIR%\diffusion_pytorch_model.safetensors

if not exist "%INPUT_FILE%" (
    echo [ERROR] 找不到输入文件: %INPUT_FILE%
    echo.
    echo 当前默认检查的目录: %CHECKPOINT_DIR%
    echo 请确认 checkpoint 目录存在且包含 diffusion_pytorch_model.safetensors
    echo.
    echo 用法: %~nx0 ^<checkpoint_dir^> [output_path]
    echo 示例: %~nx0 output_lora\checkpoint-400 output_lora\ill_style.safetensors
    pause
    exit /b 1
)

REM ---------- 输出报告 ----------
echo ============================================================
echo LoRA ^-^> ComfyUI 转换
echo ============================================================
echo checkpoint 目录:  %CHECKPOINT_DIR%
echo 输入 safetensors: %INPUT_FILE%
echo 输出 safetensors: %OUTPUT_PATH%
echo LoRA alpha/rank:  %ALPHA% / %RANK%
echo 输出 dtype:       %DTYPE%
echo ============================================================
echo.

REM ---------- 执行转换 ----------
.\.venv\Scripts\python.exe -u lora_to_comfyui.py ^
    --input  "%INPUT_FILE%" ^
    --output "%OUTPUT_PATH%" ^
    --alpha  %ALPHA% ^
    --rank   %RANK% ^
    --dtype  %DTYPE%

set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% equ 0 (
    echo ============================================================
    echo [OK] 转换完成
    echo 输出文件: %OUTPUT_PATH%
    echo 把这个文件复制到 ComfyUI 的 models\loras\ 目录下即可使用
    echo ============================================================
) else (
    echo ============================================================
    echo [FAILED] 转换失败，退出码 %EXIT_CODE%
    echo ============================================================
)

pause
endlocal
