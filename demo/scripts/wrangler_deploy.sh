#!/bin/bash

# wrangler_deploy.sh - Complete deployment script for Cloudflare Worker

set -e

echo "🌐 Deploying Software Developer Assistant to Cloudflare"
echo "======================================================="

# Check if we're in the demo directory
if [ ! -f "worker.js" ] || [ ! -f "wrangler.toml" ]; then
    echo "❌ Error: Please run this script from the demo directory"
    echo "Usage: cd demo && ./scripts/wrangler_deploy.sh"
    exit 1
fi

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Error: Wrangler CLI not found"
    echo "Please install it with: npm install -g wrangler"
    exit 1
fi

# Check if user is logged in to Cloudflare
echo "🔐 Checking Cloudflare authentication..."
if ! wrangler whoami &> /dev/null; then
    echo "❌ Not logged in to Cloudflare"
    echo "Please run: wrangler login"
    exit 1
fi

echo "✅ Authenticated with Cloudflare"

# Check if LoRA adapter is available
if [ ! -d "output/software_dev_adapter" ] && [ -z "$R2_ADAPTER_URL" ]; then
    echo "⚠️  Warning: No local LoRA adapter found and R2_ADAPTER_URL not set"
    echo "Please either:"
    echo "1. Run './scripts/train_lora.sh' to train an adapter locally"
    echo "2. Set R2_ADAPTER_URL environment variable in wrangler.toml"
fi

echo ""
echo "🚀 Deploying worker to Cloudflare..."
wrangler deploy

echo ""
echo "✅ Deployment completed!"
echo ""
echo "🌐 Your Software Developer Assistant is now live!"
echo ""
echo "Test endpoints:"
echo "1. Health check: curl https://your-worker.your-subdomain.workers.dev/health"
echo "2. Chat: curl -X POST https://your-worker.your-subdomain.workers.dev/chat -H \"Content-Type: application/json\" -d '{\"message\":\"How do I debug memory leaks?\"}'"
echo "3. Docs: https://your-worker.your-subdomain.workers.dev/docs"
