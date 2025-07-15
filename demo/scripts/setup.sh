#!/bin/bash

# setup.sh - Setup script for doc2lora demo

set -e

echo "🛠️  doc2lora Demo Setup"
echo "====================="

echo ""
echo "📋 Checking prerequisites..."

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ from https://python.org"
    exit 1
else
    echo "✅ Python found"
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if doc2lora is installed
if ! $PYTHON_CMD -c "import doc2lora" &> /dev/null; then
    echo "❌ doc2lora not installed"
    echo "Installing doc2lora..."
    cd .. && pip install -e .
    cd demo
    echo "✅ doc2lora installed"
else
    echo "✅ doc2lora already installed"
fi

# Check if HuggingFace CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ HuggingFace CLI not found"
    echo "Installing HuggingFace CLI..."
    pip install "huggingface_hub[cli]"
    echo "✅ HuggingFace CLI installed"
else
    echo "✅ HuggingFace CLI found"
fi

echo ""
echo "🔐 Setting up HuggingFace authentication..."

# Check if HF_API_KEY is set
if [ -z "$HF_API_KEY" ]; then
    echo "⚠️  HF_API_KEY not set"
    echo ""
    echo "To access gated models like Mistral, you need a HuggingFace token:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'read' permissions"
    echo "3. Set it as an environment variable:"
    echo "   export HF_API_KEY=your_token_here"
    echo ""
    read -p "Login to HuggingFace now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "⚠️  Skipping HuggingFace login. You may encounter authentication issues."
    fi
else
    echo "✅ HF_API_KEY is set"
    echo "Testing HuggingFace authentication..."
    if $PYTHON_CMD -c "from huggingface_hub import HfApi; api = HfApi(); print('✅ HuggingFace authentication successful')" 2>/dev/null; then
        echo "✅ Authentication successful"
    else
        echo "⚠️  HuggingFace authentication failed. Please check your token."
    fi
fi

echo ""
echo "📦 Checking Node.js and Wrangler (for deployment)..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found (needed for Cloudflare deployment)"
    echo "Install from: https://nodejs.org/"
else
    echo "✅ Node.js found"

    # Check if Wrangler is installed
    if ! command -v wrangler &> /dev/null; then
        echo "⚠️  Wrangler not found"
        read -p "Install Wrangler CLI? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            npm install -g wrangler
            echo "✅ Wrangler installed"
        fi
    else
        echo "✅ Wrangler found"
    fi
fi

echo ""
echo "📄 Creating .env file template..."

if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# doc2lora Demo Environment Variables
# Copy from .env.example and fill in your credentials

# HuggingFace API Key
HF_API_KEY=your_huggingface_token_here

# Cloudflare R2 Configuration
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=doc2lora-adapters
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
EOF
    echo "✅ Created .env file template"
    echo "Please edit .env with your actual credentials"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎯 Setup Summary:"
echo "================"
echo ""

if [ -z "$HF_API_KEY" ]; then
    echo "⚠️  HuggingFace: Please set HF_API_KEY or run 'huggingface-cli login'"
else
    echo "✅ HuggingFace: Configured"
fi

if ! command -v node &> /dev/null; then
    echo "⚠️  Cloudflare: Install Node.js for deployment"
else
    if ! command -v wrangler &> /dev/null; then
        echo "⚠️  Cloudflare: Install Wrangler CLI"
    else
        echo "✅ Cloudflare: Ready for deployment"
    fi
fi

echo ""
echo "🚀 Next Steps:"
echo "1. Edit .env file with your credentials"
echo "2. Run: ./scripts/train_lora.sh"
echo "3. Run: ./scripts/deploy_to_r2.sh"
echo "4. Run: ./scripts/wrangler_deploy.sh"
