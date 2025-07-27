#!/bin/bash

# train_lora.sh - Script to train a LoRA adapter using demo data

set -e

echo "🚀 Starting LoRA Training with Demo Data"
echo "========================================"

# Check GPU availability
python3 -c "import torch; print('🚀 GPU Available:' if torch.cuda.is_available() else '💻 Using CPU'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || echo "💻 PyTorch not installed yet"

# Configuration (relative to demo directory)
DATA_DIR="./data"
OUTPUT_DIR="./output"
ADAPTER_NAME="software_dev_adapter"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if we're in the demo directory
if [ ! -d "data" ]; then
    echo "❌ Error: Please run this script from the demo directory"
    echo "Usage: cd demo && ./scripts/train_lora.sh"
    exit 1
fi

# Check if HF_API_KEY is set
if [ -z "$HF_API_KEY" ]; then
    echo "⚠️  Warning: HF_API_KEY not set. This may cause authentication issues with gated models."
    echo "Please set your HuggingFace token: export HF_API_KEY=your_token_here"
    echo "Or login with: huggingface-cli login"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if demo data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Demo data directory not found at $DATA_DIR"
    exit 1
fi

echo "📁 Using data from: $DATA_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🏷️  Adapter name: $ADAPTER_NAME"
echo ""

# Run doc2lora training
echo "🔥 Training LoRA adapter..."
python -m doc2lora.cli convert "$DATA_DIR" \
    --output "$OUTPUT_DIR/$ADAPTER_NAME" \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --epochs 3 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --verbose

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: Training failed!"
    echo "Common solutions:"
    echo "1. Set HF_API_KEY: export HF_API_KEY=your_huggingface_token"
    echo "2. Login to HuggingFace: huggingface-cli login"
    echo "3. Request access to the model at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
    exit 1
fi

echo ""
echo "✅ Training completed!"
echo "📦 Adapter saved to: $OUTPUT_DIR/$ADAPTER_NAME"
echo ""
echo "Next steps:"
echo "1. Run './scripts/deploy_to_r2.sh' to upload to R2 bucket"
echo "2. Run 'wrangler deploy' to deploy the Cloudflare Worker"
