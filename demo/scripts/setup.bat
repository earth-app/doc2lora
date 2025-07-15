@echo off
chcp 65001>nul
REM setup.bat - Setup script for doc2lora demo

echo 🛠️  doc2lora Demo Setup
echo =====================

echo.
echo 📋 Checking prerequisites...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ from https://python.org
    exit /b 1
) else (
    echo ✅ Python found
)

REM Check if doc2lora is installed
python -c "import doc2lora" >nul 2>&1
if errorlevel 1 (
    echo ❌ doc2lora not installed
    echo Installing doc2lora...
    cd .. && pip install -e .
    cd demo
    if errorlevel 1 (
        echo ❌ Failed to install doc2lora
        exit /b 1
    )
    echo ✅ doc2lora installed
) else (
    echo ✅ doc2lora already installed
)

REM Check if HuggingFace CLI is installed
huggingface-cli --version >nul 2>&1
if errorlevel 1 (
    echo ❌ HuggingFace CLI not found
    echo Installing HuggingFace CLI...
    pip install huggingface_hub[cli]
    if errorlevel 1 (
        echo ❌ Failed to install HuggingFace CLI
        exit /b 1
    )
    echo ✅ HuggingFace CLI installed
) else (
    echo ✅ HuggingFace CLI found
)

echo.
echo 🔐 Setting up HuggingFace authentication...

REM Check if HF_API_KEY is set
if "%HF_API_KEY%"=="" (
    echo ⚠️  HF_API_KEY not set
    echo.
    echo To access gated models like Mistral, you need a HuggingFace token:
    echo 1. Go to https://huggingface.co/settings/tokens
    echo 2. Create a new token with 'read' permissions
    echo 3. Set it as an environment variable:
    echo    set HF_API_KEY=your_token_here
    echo.
    echo Or you can login interactively:
    set /p choice="Login to HuggingFace now? (y/N): "
    if /i "%choice%"=="y" (
        huggingface-cli login
    ) else (
        echo ⚠️  Skipping HuggingFace login. You may encounter authentication issues.
    )
) else (
    echo ✅ HF_API_KEY is set
    echo Testing HuggingFace authentication...
    python -c "from huggingface_hub import HfApi; api = HfApi(); print('✅ HuggingFace authentication successful')" 2>nul
    if errorlevel 1 (
        echo ⚠️  HuggingFace authentication failed. Please check your token.
    )
)

echo.
echo 📦 Checking Node.js and Wrangler (for deployment)...

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Node.js not found (needed for Cloudflare deployment)
    echo Install from: https://nodejs.org/
) else (
    echo ✅ Node.js found

    REM Check if Wrangler is installed
    wrangler --version >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  Wrangler not found
        set /p install_wrangler="Install Wrangler CLI? (y/N): "
        if /i "%install_wrangler%"=="y" (
            npm install -g wrangler
            if errorlevel 1 (
                echo ❌ Failed to install Wrangler
            ) else (
                echo ✅ Wrangler installed
            )
        )
    ) else (
        echo ✅ Wrangler found
    )
)

echo.
echo 📄 Creating .env file template...

if not exist ".env" (
    echo # doc2lora Demo Environment Variables > .env
    echo # Copy from .env.example and fill in your credentials >> .env
    echo. >> .env
    echo # HuggingFace API Key >> .env
    echo HF_API_KEY=your_huggingface_token_here >> .env
    echo. >> .env
    echo # Cloudflare R2 Configuration >> .env
    echo R2_ACCESS_KEY_ID=your_r2_access_key >> .env
    echo R2_SECRET_ACCESS_KEY=your_r2_secret_key >> .env
    echo R2_BUCKET_NAME=doc2lora-adapters >> .env
    echo R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com >> .env
    echo ✅ Created .env file template
    echo Please edit .env with your actual credentials
) else (
    echo ✅ .env file already exists
)

echo.
echo 🎯 Setup Summary:
echo ================
echo.
if "%HF_API_KEY%"=="" (
    echo ⚠️  HuggingFace: Please set HF_API_KEY or run 'huggingface-cli login'
) else (
    echo ✅ HuggingFace: Configured
)

node --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Cloudflare: Install Node.js for deployment
) else (
    wrangler --version >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  Cloudflare: Install Wrangler CLI
    ) else (
        echo ✅ Cloudflare: Ready for deployment
    )
)

echo.
echo 🚀 Next Steps:
echo 1. Edit .env file with your credentials
echo 2. Run: .\scripts\train_lora.bat
echo 3. Run: .\scripts\deploy_to_r2.bat
echo 4. Run: .\scripts\wrangler_deploy.bat
