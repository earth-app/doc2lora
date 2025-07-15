"""Example usage of doc2lora with Mistral models."""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import doc2lora
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_hf_authentication():
    """Setup HuggingFace authentication from environment variable."""
    hf_api_key = os.getenv("HF_API_KEY")
    if hf_api_key:
        print(f"✅ HuggingFace API key found: {hf_api_key[:8]}...")
        # Set the token for transformers library
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_api_key
        return True
    else:
        print("⚠️  HF_API_KEY environment variable not found.")
        print("   For some models, you may need a HuggingFace API key.")
        print("   Set it with: export HF_API_KEY=your_huggingface_token")
        return False


def create_sample_documents():
    """Create sample documents if they don't exist."""
    docs_path = Path(__file__).parent / "example_documents"
    docs_path.mkdir(exist_ok=True)

    # Create sample documents for training
    samples = {
        "sample_conversation.txt": """User: How do I create a REST API?
Assistant: To create a REST API, you can use frameworks like FastAPI for Python or Express.js for Node.js. Here's a basic FastAPI example:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

This creates a simple endpoint that returns JSON when accessed.""",
        "sample_documentation.md": """# API Documentation

## Getting Started

This API provides endpoints for managing user data and authentication.

### Authentication

All endpoints require a valid API key in the header:
```
Authorization: Bearer YOUR_API_KEY
```

### Endpoints

#### GET /users
Returns a list of all users.

#### POST /users
Creates a new user.

#### GET /users/{id}
Returns a specific user by ID.
""",
        "sample_instructions.txt": """How to set up the development environment:

1. Clone the repository
2. Install dependencies with pip install -r requirements.txt
3. Set up environment variables in .env file
4. Run the application with python app.py

For production deployment:
- Use a WSGI server like Gunicorn
- Set up a reverse proxy with Nginx
- Configure SSL certificates
- Set up monitoring and logging
""",
    }

    for filename, content in samples.items():
        file_path = docs_path / filename
        if not file_path.exists():
            file_path.write_text(content)
            print(f"Created sample file: {filename}")

    return docs_path


def demo_mistral_training():
    """Demonstrate training with Mistral model."""
    print("=== doc2lora Mistral Training Demo ===")

    # Check for HuggingFace API key
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        print("⚠️  HF_API_KEY environment variable not found")
        print("   For better performance and to avoid rate limits, set:")
        print("   export HF_API_KEY=your_huggingface_api_key")
        print("   You can get one at: https://huggingface.co/settings/tokens")
    else:
        print("✅ HuggingFace API key found")
        # Set the token for transformers
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_api_key

    # Create sample documents
    docs_path = create_sample_documents()

    try:
        from doc2lora import convert

        print("✅ ML dependencies found! Running Mistral LoRA training...")

        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Configure for Mistral model
        print("🚀 Training LoRA adapter for Mistral-7B-Instruct...")

        adapter_path = convert(
            documents_path=str(docs_path),
            output_path=str(output_dir / "mistral_adapter.json"),
            model_name="mistralai/Mistral-7B-Instruct-v0.2",  # Full model name
            num_epochs=2,  # Reduced for demo
            batch_size=1,  # Small batch for memory efficiency
            max_length=512,
            learning_rate=1e-4,  # Lower learning rate for Mistral
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        print(f"🎉 Mistral LoRA adapter successfully created at: {adapter_path}")

        # Show what the adapter contains
        import json

        with open(adapter_path, "r") as f:
            metadata = json.load(f)

        print("\n📋 Adapter Information:")
        print(f"   Base Model: {metadata['base_model']}")
        print(f"   Target Modules: {metadata['lora_config']['target_modules']}")
        print(f"   LoRA Rank: {metadata['lora_config']['r']}")
        print(f"   LoRA Alpha: {metadata['lora_config']['alpha']}")
        print(f"   Adapter Path: {metadata['adapter_path']}")

        return adapter_path

    except ImportError as e:
        print(f"⚠️  ML dependencies not available: {e}")
        print("   Install with: pip install torch transformers peft datasets")
        return None
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("   This might be due to:")
        print("   - Insufficient GPU memory (try smaller batch_size)")
        print("   - Model download issues (check internet connection)")
        print("   - CUDA/PyTorch compatibility issues")
        return None


def show_cloudflare_usage(adapter_path):
    """Show how to use the adapter with Cloudflare Workers AI."""
    if not adapter_path:
        return

    print(f"\n{'='*60}")
    print("🌐 Using with Cloudflare Workers AI")
    print(f"{'='*60}")

    print(
        """
To use your trained LoRA adapter with Cloudflare Workers AI:

1. Upload your adapter to Cloudflare:
   ```bash
   # Upload the adapter directory to Cloudflare R2 or similar storage
   wrangler r2 object put my-bucket/mistral-adapter ./output/mistral_adapter_adapter/
   ```

2. Reference in your Worker:
   ```javascript
   export default {
     async fetch(request, env) {
       const ai = new Ai(env.AI);

       const response = await ai.run('@cf/mistralai/mistral-7b-instruct-v0.2-lora', {
         messages: [
           { role: "user", content: "How do I create a REST API?" }
         ],
         lora: "mistral-adapter"  // Reference your uploaded adapter
       });

       return new Response(JSON.stringify(response));
     }
   }
   ```

3. The model will now use your fine-tuned knowledge from the documents!
"""
    )


def main():
    """Main demo function."""
    try:
        # Setup HuggingFace authentication
        setup_hf_authentication()

        # Run the Mistral training demo
        adapter_path = demo_mistral_training()

        # Show Cloudflare usage
        show_cloudflare_usage(adapter_path)

        print(f"\n{'='*60}")
        print("📚 Additional Tips")
        print(f"{'='*60}")
        print(
            """
For better results with Mistral:
- Use more training data (100+ documents)
- Increase training epochs (3-5)
- Fine-tune learning rate based on your data
- Monitor training loss to avoid overfitting
- Use validation data to check generalization

Supported Mistral models:
- mistralai/Mistral-7B-v0.1
- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mistral-7B-Instruct-v0.2
- mistralai/Mistral-7B-Instruct-v0.3
"""
        )

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
