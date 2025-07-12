#!/usr/bin/env python3
"""
Example usage of doc2lora with R2 bucket.

This example demonstrates how to use the doc2lora library to convert
documents stored in a Cloudflare R2 bucket into a LoRA adapter.

Prerequisites:
- boto3 library installed: pip install boto3
- Valid R2 credentials and bucket
- Documents uploaded to your R2 bucket

Usage:
    python r2_usage.py
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import doc2lora
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc2lora import convert_from_r2


def main():
    """Main function to demonstrate R2 bucket usage."""
    print("🚀 doc2lora R2 Bucket Example")
    print("=" * 50)

    # R2 configuration
    # In production, these should be set as environment variables
    BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "my-documents-bucket")
    FOLDER_PREFIX = os.getenv("R2_FOLDER_PREFIX", "training-docs")  # Optional
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")

    # Check if credentials are provided
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_ENDPOINT_URL]):
        print("❌ Missing R2 credentials!")
        print("\nPlease set the following environment variables:")
        print("  AWS_ACCESS_KEY_ID=your-r2-access-key-id")
        print("  AWS_SECRET_ACCESS_KEY=your-r2-secret-access-key")
        print("  R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com")
        print("  R2_BUCKET_NAME=your-bucket-name (optional, defaults to 'my-documents-bucket')")
        print("  R2_FOLDER_PREFIX=folder-prefix (optional)")
        print("\nExample:")
        print("  export AWS_ACCESS_KEY_ID=abc123...")
        print("  export AWS_SECRET_ACCESS_KEY=xyz789...")
        print("  export R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com")
        print("  export R2_BUCKET_NAME=training-documents")
        print("  export R2_FOLDER_PREFIX=legal-docs")
        return 1

    print(f"📦 Bucket: {BUCKET_NAME}")
    if FOLDER_PREFIX:
        print(f"📁 Folder: {FOLDER_PREFIX}")
    print(f"🔗 Endpoint: {R2_ENDPOINT_URL}")
    print()

    try:
        print("🔄 Converting documents from R2 bucket...")

        # Convert documents from R2 bucket
        adapter_path = convert_from_r2(
            bucket_name=BUCKET_NAME,
            folder_prefix=FOLDER_PREFIX,
            output_path="r2_lora_adapter.json",
            model_name="microsoft/DialoGPT-small",  # Use small model for demo
            max_length=256,  # Smaller context for faster training
            batch_size=2,    # Small batch size for demo
            num_epochs=1,    # Just 1 epoch for demo
            learning_rate=5e-4,
            lora_r=8,        # Smaller LoRA rank for demo
            lora_alpha=16,
            lora_dropout=0.1,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            endpoint_url=R2_ENDPOINT_URL,
            cleanup_temp=True,  # Clean up downloaded files
        )

        print(f"✅ Success! LoRA adapter created at: {adapter_path}")
        print("\n📊 Next steps:")
        print("  1. Load the adapter in your LLM framework")
        print("  2. Use it for inference or further fine-tuning")
        print("  3. Deploy with Cloudflare Workers AI or similar platforms")

        return 0

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install boto3 with: pip install boto3")
        return 1

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Check your bucket name, credentials, and permissions")
        return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check your network connection and R2 configuration")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
