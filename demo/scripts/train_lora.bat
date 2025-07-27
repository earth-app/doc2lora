@echo off
chcp 65001>nul
REM train_lora.bat - Windows batch script to train a LoRA adapter using demo data

echo 🚀 Starting LoRA Training with Demo Data
echo ========================================

REM Check GPU availability
python -c "import torch; print('🚀 GPU Available:' if torch.cuda.is_available() else '💻 Using CPU'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>nul

REM Configuration (relative to demo directory)
set DATA_DIR=.\data
set OUTPUT_DIR=.\output
set ADAPTER_NAME=software_dev_adapter

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Check if we're in the demo directory
if not exist "data" (
    echo ❌ Error: Please run this script from the demo directory
    echo Usage: cd demo && .\scripts\train_lora.bat
    exit /b 1
)

REM Check if HF_API_KEY is set
if "%HF_API_KEY%"=="" (
    echo ⚠️  Warning: HF_API_KEY not set. This may cause authentication issues with gated models.
    echo Please set your HuggingFace token: set HF_API_KEY=your_token_here
    pause
)

REM Check if demo data exists
if not exist "%DATA_DIR%" (
    echo ❌ Error: Demo data directory not found at %DATA_DIR%
    exit /b 1
)

echo 📁 Using data from: %DATA_DIR%
echo 📁 Output directory: %OUTPUT_DIR%
echo 🏷️  Adapter name: %ADAPTER_NAME%
echo.

REM Run doc2lora training
echo 🔥 Training LoRA adapter...
python -m doc2lora.cli convert "%DATA_DIR%" ^
    --output "%OUTPUT_DIR%\%ADAPTER_NAME%" ^
    --model "mistralai/Mistral-7B-Instruct-v0.2" ^
    --epochs 3 ^
    --learning-rate 2e-4 ^
    --batch-size 2 ^
    --lora-r 8 ^
    --lora-alpha 16 ^
    --lora-dropout 0.1 ^
    --verbose

if %errorlevel% neq 0 (
    echo ❌ Error: Training failed!
    echo Common solutions:
    echo 1. Set HF_API_KEY: set HF_API_KEY=your_huggingface_token
    echo 2. Login to HuggingFace: huggingface-cli login
    echo 3. Request access to the model at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    exit /b 1
)

echo.
echo ✅ Training completed!
echo 📦 Adapter saved to: %OUTPUT_DIR%\%ADAPTER_NAME%
echo.
echo Next steps:
echo 1. Run '.\scripts\deploy_to_r2.bat' to upload to Cloudflare AI
echo 2. Run 'wrangler deploy' to deploy the Cloudflare Worker
