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
    # Option 1: Use environment variables directly
    # Option 2: Load from .env file (recommended for security)

    # Try to load from .env file first if it exists
    env_file_path = os.getenv("DOC2LORA_ENV_FILE", ".env")
    if Path(env_file_path).exists():
        print(f"📄 Found .env file at: {env_file_path}")
        try:
            from doc2lora.utils import load_env_file

            load_env_file(env_file_path)
            print("✅ Loaded credentials from .env file")
        except ImportError:
            print(
                "⚠️  python-dotenv not installed, falling back to environment variables"
            )

    BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "my-documents-bucket")
    FOLDER_PREFIX = os.getenv("R2_FOLDER_PREFIX", "training-docs")  # Optional
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")

    # Check if credentials are provided
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL]):
        print("❌ Missing R2 credentials!")
        print("\nYou can provide credentials in several ways:")
        print("\n1. Environment variables (preferred):")
        print("  R2_ACCESS_KEY_ID=your-r2-access-key-id")
        print("  R2_SECRET_ACCESS_KEY=your-r2-secret-access-key")
        print("  R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com")
        print(
            "  R2_BUCKET_NAME=your-bucket-name (optional, defaults to 'my-documents-bucket')"
        )
        print("  R2_FOLDER_PREFIX=folder-prefix (optional)")
        print("\n2. Legacy AWS environment variables (still supported):")
        print("  AWS_ACCESS_KEY_ID=your-r2-access-key-id")
        print("  AWS_SECRET_ACCESS_KEY=your-r2-secret-access-key")
        print("  R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com")
        print("\n3. .env file (recommended):")
        print("  Create a .env file with the following content:")
        print("  R2_ACCESS_KEY_ID=your-r2-access-key-id")
        print("  R2_SECRET_ACCESS_KEY=your-r2-secret-access-key")
        print("  R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com")
        print("  R2_BUCKET_NAME=my-documents-bucket")
        print("  R2_FOLDER_PREFIX=training-docs")
        print("\n4. Pass env_file parameter to convert_from_r2():")
        print("  convert_from_r2(bucket_name='my-bucket', env_file='.env')")
        print("\nExample:")
        print("  export R2_ACCESS_KEY_ID=abc123...")
        print("  export R2_SECRET_ACCESS_KEY=xyz789...")
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

                # Method 1: Convert documents from R2 bucket using explicit credentials
        adapter_path = convert_from_r2(
            bucket_name=BUCKET_NAME,
            folder_prefix=FOLDER_PREFIX,
            output_path="r2_lora_adapter.json",
            model_name="microsoft/DialoGPT-small",  # Use small model for demo
            max_length=256,  # Smaller context for faster training
            batch_size=1,  # Smaller batch for demo
            num_epochs=1,  # Single epoch for demo
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            endpoint_url=R2_ENDPOINT_URL,
        )

        # Method 2: Alternative - using .env file (uncomment to use)
        # adapter_path = convert_from_r2(
        #     bucket_name=BUCKET_NAME,
        #     folder_prefix=FOLDER_PREFIX,
        #     output_path="r2_lora_adapter.json",
        #     env_file=".env",  # Load credentials from .env file
        #     model_name="microsoft/DialoGPT-small",
        #     max_length=256,
        #     batch_size=2,
        #     num_epochs=1,
        #     learning_rate=5e-4,
        #     lora_r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     cleanup_temp=True,
        # )

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
