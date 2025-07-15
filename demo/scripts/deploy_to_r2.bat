@echo o# Configuration (rel# Check if adapter exists
if not exist "%ADAPTER_PATH%" (
    echo ❌ Error: LoRA adapter not found at %ADAPTER_PATH%
    echo Please run '.\scripts\train_lora.bat' first
    exit /b 1
)to demo directory)
set OUTPUT_DIR=.\output
set ADAPTER_NAME=software_dev_adapter
set ADAPTER_PATH=%OUTPUT_DIR%\%ADAPTER_NAME%chcp 65001>nul
REM deploy_to_r2.bat - Windows batch script to deploy LoRA adapter to Cloudflare R2

echo ☁️  Deploying LoRA Adapter to Cloudflare R2
echo ===========================================

REM Configuration
set OUTPUT_DIR=.\demo\output
set ADAPTER_NAME=software_dev_adapter
set ADAPTER_PATH=%OUTPUT_DIR%\%ADAPTER_NAME%

REM Check if .env file exists
if not exist ".env" (
    echo ❌ Error: .env file not found
    echo Please create a .env file with your R2 credentials:
    echo.
    echo R2_ACCESS_KEY_ID=your_access_key
    echo R2_SECRET_ACCESS_KEY=your_secret_key
    echo R2_BUCKET_NAME=your_bucket_name
    echo R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
    echo.
    exit /b 1
)

REM Check if adapter exists
if not exist "%ADAPTER_PATH%" (
    echo ❌ Error: LoRA adapter not found at %ADAPTER_PATH%
    echo Please run '.\demo\scripts\train_lora.bat' first
    exit /b 1
)

echo 📦 Adapter path: %ADAPTER_PATH%
echo.

REM Upload adapter using doc2lora R2 integration
echo 🚀 Uploading adapter to R2...
python -c "
import os
from dotenv import load_dotenv
from doc2lora.utils import upload_to_r2

load_dotenv()
adapter_path = '%ADAPTER_PATH%'
bucket_name = os.getenv('R2_BUCKET_NAME')
adapter_name = '%ADAPTER_NAME%'

if not bucket_name:
    print('❌ Error: R2_BUCKET_NAME not found in environment')
    exit(1)

print(f'Uploading {adapter_name} to R2 bucket: {bucket_name}')
result = upload_to_r2(adapter_path, bucket_name, adapter_name)
print(f'✅ Upload completed: {result}')
"

echo.
echo ✅ Deployment completed!
echo 🌐 Your LoRA adapter is now available in R2
echo.
echo Next steps:
echo 1. Update the R2_ADAPTER_URL in your Cloudflare Worker
echo 2. Run 'cd demo && wrangler deploy' to deploy the worker
