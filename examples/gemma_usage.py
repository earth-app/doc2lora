"""Example usage of doc2lora with Google Gemma models for Cloudflare Workers AI."""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import doc2lora
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc2lora import convert


def create_sample_documents():
    """Create sample documents if they don't exist."""
    docs_path = Path(__file__).parent / "example_documents"
    docs_path.mkdir(exist_ok=True)

    # Create sample documents for training with Gemma-specific content
    samples = {
        "gemma_conversation.txt": """User: Explain how to use Google Gemma for text generation.
Assistant: Google Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. Here's how to use it:

1. **Model Selection**: Choose between Gemma 2B or 7B based on your requirements
2. **Fine-tuning**: Use LoRA (Low-Rank Adaptation) for efficient fine-tuning
3. **Deployment**: Deploy on Cloudflare Workers AI for global distribution

Example usage with Cloudflare Workers AI:
```javascript
const response = await env.AI.run("@cf/google/gemma-7b-it-lora", {
  messages: [{"role": "user", "content": "Hello, how can you help me?"}],
  lora: "my-custom-adapter"
});
```""",

        "gemma_technical_guide.md": """# Gemma Model Integration Guide

## Overview
Google Gemma models are designed for responsible AI development and offer excellent performance for various tasks.

## Key Features
- **Lightweight**: Optimized for efficiency
- **Open Source**: Available for research and commercial use
- **Instruction-tuned**: Ready for chat and instruction-following tasks
- **Safety-focused**: Built with responsible AI principles

## Model Variants
- **gemma-2b-it**: 2 billion parameters, instruction-tuned
- **gemma-7b-it**: 7 billion parameters, instruction-tuned

## Fine-tuning Best Practices
1. **Data Quality**: Use high-quality, diverse training data
2. **LoRA Parameters**:
   - Rank (r): 8-16 for Gemma models
   - Alpha: 16-32
   - Dropout: 0.05-0.1
3. **Training**:
   - Learning rate: 1e-4 to 5e-4
   - Batch size: 1-4 (depending on available memory)
   - Epochs: 2-5

## Integration with Cloudflare Workers AI
Gemma models on Cloudflare Workers AI support custom LoRA adapters for domain-specific fine-tuning.
""",

        "gemma_training_data.json": """{
  "conversations": [
    {
      "user": "What are the benefits of using Gemma models?",
      "assistant": "Gemma models offer several key benefits: 1) They're lightweight and efficient, making them suitable for deployment in resource-constrained environments. 2) They're open source with a permissive license. 3) They're instruction-tuned for better chat performance. 4) They're built with safety considerations in mind. 5) They can be fine-tuned with LoRA for specific use cases."
    }
  ]
}"""
    }

    for filename, content in samples.items():
        file_path = docs_path / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
            print(f"📄 Created sample document: {filename}")
        else:
            print(f"📄 Sample document already exists: {filename}")

    return docs_path


def demo_gemma_training():
    """Demonstrate training a LoRA adapter for Gemma."""
    print(f"\n{'='*60}")
    print("🤖 Google Gemma LoRA Training Demo")
    print(f"{'='*60}")

    # Create sample documents
    docs_path = create_sample_documents()
    print(f"\n📁 Using documents from: {docs_path}")

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    adapter_output = output_dir / "gemma_lora_adapter.json"

    print(f"\n🚀 Starting LoRA training for Gemma...")
    print("   Model: @cf/google/gemma-7b-it-lora")
    print(f"   Output: {adapter_output}")

    try:
        # Train with Gemma-optimized parameters
        adapter_path = convert(
            documents_path=str(docs_path),
            output_path=str(adapter_output),
            model_name="google/gemma-7b-it",  # Use Gemma model
            max_length=1024,     # Gemma supports longer contexts
            batch_size=1,        # Small batch for memory efficiency
            num_epochs=3,        # Good balance for Gemma
            learning_rate=2e-4,  # Optimal for Gemma fine-tuning
            lora_r=16,          # Good rank for Gemma
            lora_alpha=32,      # 2x rank ratio
            lora_dropout=0.05,  # Lower dropout for Gemma
        )

        print(f"✅ Training completed successfully!")
        print(f"📦 LoRA adapter saved to: {adapter_path}")

        return adapter_path

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install required packages with:")
        print("   pip install torch transformers peft datasets")
        return None

    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 This is a demo - actual training requires ML dependencies")

        # Create a dummy adapter file for demo purposes
        adapter_output.write_text('{"model": "gemma-7b-it", "type": "lora", "status": "demo"}')
        print(f"📦 Created demo adapter file: {adapter_output}")
        return str(adapter_output)


def show_cloudflare_usage(adapter_path):
    """Show how to use the trained adapter with Cloudflare Workers AI."""
    print(f"\n{'='*60}")
    print("☁️  Cloudflare Workers AI Integration")
    print(f"{'='*60}")

    print("""
🔧 To use your Gemma LoRA adapter with Cloudflare Workers AI:

1. **Upload your adapter**:
   ```bash
   # Upload the adapter to Cloudflare
   wrangler ai lora upload my-gemma-adapter ./gemma_lora_adapter_adapter/
   ```

2. **Use in your Worker**:
   ```javascript
   export default {
     async fetch(request, env) {
       const response = await env.AI.run("@cf/google/gemma-7b-it-lora", {
         messages: [
           {
             "role": "system",
             "content": "You are a helpful assistant trained on custom data."
           },
           {
             "role": "user",
             "content": "Explain the benefits of using Gemma models."
           }
         ],
         lora: "my-gemma-adapter"  // Your uploaded adapter
       });

       return new Response(JSON.stringify(response));
     }
   }
   ```

💡 **Tips for Gemma Fine-tuning:**
- Use diverse, high-quality training data
- Keep LoRA rank between 8-16 for best results
- Monitor training loss to prevent overfitting
""")


def main():
    """Main demo function."""
    try:
        # Run the Gemma training demo
        adapter_path = demo_gemma_training()

        # Show Cloudflare usage
        if adapter_path:
            show_cloudflare_usage(adapter_path)

        print(f"\n{'='*60}")
        print("📚 Gemma Resources")
        print(f"{'='*60}")
        print("""
🔗 **Useful Links:**
- Gemma Model: https://huggingface.co/google/gemma-7b-it
- Cloudflare Workers AI: https://developers.cloudflare.com/workers-ai/
- LoRA Paper: https://arxiv.org/abs/2106.09685

🎓 **Best Practices:**
- Start with small datasets for testing
- Use validation data to monitor progress
- Experiment with different LoRA parameters
""")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
