@echo off
chcp 65001>nul
REM deploy_to_r2.bat - Windows batch script to deploy LoRA adapter to Cloudflare AI

echo ☁️  Deploying LoRA Adapter to Cloudflare AI
echo ===========================================

REM Configuration (relative to demo directory)
set OUTPUT_DIR=.\output
set ADAPTER_NAME=software_dev_adapter
set ADAPTER_PATH=%OUTPUT_DIR%\%ADAPTER_NAME%

REM Check if adapter exists
if not exist "%ADAPTER_PATH%" (
    echo ❌ Error: LoRA adapter not found at %ADAPTER_PATH%
    echo Please run '.\scripts\train_lora.bat' first
    exit /b 1
)

echo 📦 Adapter path: %ADAPTER_PATH%
echo.

REM Check if wrangler is installed
wrangler --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: wrangler CLI not found
    echo Please install wrangler: npm install -g wrangler
    echo Then login with: wrangler login
    exit /b 1
)

REM Create finetune and upload adapter
echo 📤 Creating finetune and uploading adapter files...
wrangler ai finetune create "@cf/mistralai/mistral-7b-instruct-v0.2-lora" "%ADAPTER_NAME%" "%ADAPTER_PATH%"

if %errorlevel% equ 0 (
    echo.
    echo ✅ Deployment completed!
    echo 🌐 Your LoRA adapter '%ADAPTER_NAME%' is now available in Cloudflare AI
    echo.
    echo Next steps:
    echo 1. Update your Cloudflare Worker to use adapter: '%ADAPTER_NAME%'
    echo 2. Run 'cd demo && wrangler deploy' to deploy the worker
) else (
    echo.
    echo ❌ Upload failed!
    echo Common solutions:
    echo 1. Run 'wrangler login' to authenticate
    echo 2. Check your account has Workers AI enabled
    echo 3. Verify the adapter files are valid
    exit /b 1
)
