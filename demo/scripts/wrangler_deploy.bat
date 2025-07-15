@echo o# Check if we're in the demo directory
if not exist "worker.js" (
    echo ❌ Error: Please run this script from the demo directory
    echo Usage: cd demo && .\scripts\wrangler_deploy.bat
    exit /b 1
)

if not exist "wrangler.toml" (
    echo ❌ Error: wrangler.toml not found
    echo Please run this script from the demo directory
    exit /b 1
)

REM Check if wrangler is installed>nul
REM wrangler_deploy.bat - Complete deployment script for Cloudflare Worker (Windows)

echo 🌐 Deploying Software Developer Assistant to Cloudflare
echo =======================================================

REM Check if we're in the demo directory
if not exist "worker.js" (
    echo ❌ Error: Please run this script from the demo directory
    echo Usage: cd demo && .\scripts\wrangler_deploy.bat
    exit /b 1
)

if not exist "wrangler.toml" (
    echo ❌ Error: wrangler.toml not found
    echo Please run this script from the demo directory
    exit /b 1
)

REM Check if wrangler is installed
wrangler --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Wrangler CLI not found
    echo Please install it with: npm install -g wrangler
    echo Or install Node.js first from: https://nodejs.org/
    exit /b 1
)

REM Check if user is logged in to Cloudflare
echo 🔐 Checking Cloudflare authentication...
wrangler whoami >nul 2>&1
if errorlevel 1 (
    echo ❌ Not logged in to Cloudflare
    echo Please run: wrangler login
    exit /b 1
)

echo ✅ Authenticated with Cloudflare

REM Check if LoRA adapter is available
if not exist "output\software_dev_adapter" (
    if "%R2_ADAPTER_URL%"=="" (
        echo ⚠️  Warning: No local LoRA adapter found and R2_ADAPTER_URL not set
        echo Please either:
        echo 1. Run '.\scripts\train_lora.bat' to train an adapter locally
        echo 2. Set R2_ADAPTER_URL environment variable in wrangler.toml
    )
)

echo.
echo 🚀 Deploying worker to Cloudflare...
wrangler deploy

echo.
echo ✅ Deployment completed!
echo.
echo 🌐 Your Software Developer Assistant is now live!
echo.
echo Test endpoints:
echo 1. Health check: curl https://your-worker.your-subdomain.workers.dev/health
echo 2. Chat: curl -X POST https://your-worker.your-subdomain.workers.dev/chat -H "Content-Type: application/json" -d "{\"message\":\"How do I debug memory leaks?\"}"
echo 3. Docs: https://your-worker.your-subdomain.workers.dev/docs
