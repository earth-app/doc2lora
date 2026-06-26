#!/bin/bash

# deploy_to_r2.sh - Script to deploy LoRA adapter to Cloudflare R2

set -e

echo "☁️  Deploying LoRA Adapter to Cloudflare R2"
echo "==========================================="

# Configuration (relative to demo directory)
OUTPUT_DIR="./output"
ADAPTER_NAME="software_dev_adapter"
ADAPTER_PATH="$OUTPUT_DIR/$ADAPTER_NAME"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "Please create a .env file with your R2 credentials:"
    echo ""
    echo "R2_ACCESS_KEY_ID=your_access_key"
    echo "R2_SECRET_ACCESS_KEY=your_secret_key"
    echo "R2_BUCKET_NAME=your_bucket_name"
    echo "R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com"
    echo ""
    exit 1
fi

# Check if adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Error: LoRA adapter not found at $ADAPTER_PATH"
    echo "Please run './scripts/train_lora.sh' first"
    exit 1
fi

echo "📦 Adapter path: $ADAPTER_PATH"
echo ""

# Load environment variables
source .env

# Validate required environment variables
if [ -z "$R2_BUCKET_NAME" ] || [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ] || [ -z "$R2_ENDPOINT_URL" ]; then
    echo "❌ Error: Missing required R2 environment variables"
    echo "Please check your .env file contains all required variables"
    exit 1
fi

echo "🔧 R2 Configuration:"
echo "   Bucket: $R2_BUCKET_NAME"
echo "   Endpoint: $R2_ENDPOINT_URL"
echo ""

# Upload adapter using Cloudflare AI finetunes API
echo "🚀 Uploading adapter to Cloudflare AI..."

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Error: wrangler CLI not found"
    echo "Please install wrangler: npm install -g wrangler"
    echo "Then login with: wrangler login"
    exit 1
fi

# Create finetune and upload adapter via doc2lora (validates the adapter first,
# then shells out to wrangler). Equivalent to the raw command:
#   wrangler ai finetune create "@cf/mistralai/mistral-7b-instruct-v0.2-lora" "$ADAPTER_NAME" "$ADAPTER_PATH"
echo "📤 Creating finetune and uploading adapter files..."
doc2lora deploy "$ADAPTER_PATH" "$ADAPTER_NAME" \
    --cf-model "@cf/mistralai/mistral-7b-instruct-v0.2-lora"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment completed!"
    echo "🌐 Your LoRA adapter '$ADAPTER_NAME' is now available in Cloudflare AI"
    echo ""
    echo "Next steps:"
    echo "1. Update your Cloudflare Worker to use adapter: '$ADAPTER_NAME'"
    echo "2. Run 'cd demo && wrangler deploy' to deploy the worker"
else
    echo ""
    echo "❌ Upload failed!"
    echo "Common solutions:"
    echo "1. Run 'wrangler login' to authenticate"
    echo "2. Check your account has Workers AI enabled"
    echo "3. Verify the adapter files are valid"
    exit 1
fi
