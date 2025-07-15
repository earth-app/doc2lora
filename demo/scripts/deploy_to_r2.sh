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

# Upload adapter using doc2lora R2 integration
echo "🚀 Uploading adapter to R2..."
python -c "
import os
from doc2lora.utils import upload_to_r2

adapter_path = '$ADAPTER_PATH'
bucket_name = os.getenv('R2_BUCKET_NAME')
adapter_name = '$ADAPTER_NAME'

print(f'Uploading {adapter_name} to R2 bucket: {bucket_name}')
result = upload_to_r2(adapter_path, bucket_name, adapter_name)
print(f'✅ Upload completed: {result}')
"

echo ""
echo "✅ Deployment completed!"
echo "🌐 Your LoRA adapter is now available in R2"
echo ""
echo "Next steps:"
echo "1. Update the R2_ADAPTER_URL in your Cloudflare Worker"
echo "2. Run 'cd demo && wrangler deploy' to deploy the worker"
