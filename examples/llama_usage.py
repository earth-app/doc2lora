"""Example usage of doc2lora with Meta Llama models for Cloudflare Workers AI."""

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

    # Create sample documents for training with Llama-specific content
    samples = {
        "llama_conversation.txt": """User: What makes Llama 2 models effective for chat applications?
Assistant: Llama 2 models are particularly effective for chat applications due to several key design decisions:

1. **Chat Fine-tuning**: Llama 2-Chat models are specifically fine-tuned for dialogue use cases with human feedback (RLHF)
2. **Safety Focus**: Built with extensive safety measures and red-teaming
3. **Context Understanding**: Strong ability to maintain context across long conversations
4. **Instruction Following**: Excellent at following complex instructions and maintaining conversational flow

Example implementation with Cloudflare Workers AI:
```javascript
const response = await env.AI.run("@cf/meta-llama/llama-2-7b-chat-hf-lora", {
  messages: [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the benefits of LoRA fine-tuning."}
  ],
  lora: "my-custom-llama-adapter"
});
```""",
        "llama_technical_guide.md": """# Llama 2 Model Integration Guide

## Overview
Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7B to 70B parameters. The fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases.

## Key Features
- **Chat Optimization**: Llama 2-Chat models are specifically optimized for dialogue
- **Safety**: Extensive safety fine-tuning with human feedback
- **Performance**: State-of-the-art performance on many NLP benchmarks
- **Open Source**: Available for research and commercial use

## Fine-tuning Best Practices for Llama 2
1. **Data Format**: Use proper chat format with system/user/assistant roles
2. **LoRA Parameters**:
   - Rank (r): 16-32 for Llama models
   - Alpha: 32-64 (typically 2x rank)
   - Dropout: 0.05-0.1
3. **Training Configuration**:
   - Learning rate: 1e-4 to 3e-4
   - Batch size: 1-2 (memory intensive)
   - Epochs: 2-4
   - Context length: Up to 4096 tokens
""",
        "llama_training_data.json": """{
  "conversations": [
    {
      "system": "You are a helpful AI assistant specialized in explaining AI concepts.",
      "user": "What are the advantages of using Llama 2 for chat applications?",
      "assistant": "Llama 2 offers several advantages for chat applications: 1) It's specifically fine-tuned for dialogue with human feedback (RLHF), making responses more natural and helpful. 2) It has strong safety measures built-in through extensive red-teaming. 3) It maintains context well across long conversations. 4) It follows instructions accurately and can handle complex, multi-turn dialogues."
    }
  ]
}""",
    }

    for filename, content in samples.items():
        file_path = docs_path / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
            print(f"📄 Created sample document: {filename}")
        else:
            print(f"📄 Sample document already exists: {filename}")

    return docs_path


def demo_llama_training():
    """Demonstrate training a LoRA adapter for Llama 2."""
    print(f"\n{'='*60}")
    print("🦙 Meta Llama 2 LoRA Training Demo")
    print(f"{'='*60}")

    # Create sample documents
    docs_path = create_sample_documents()
    print(f"\n📁 Using documents from: {docs_path}")

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    adapter_output = output_dir / "llama_lora_adapter.json"

    print(f"\n🚀 Starting LoRA training for Llama 2...")
    print("   Model: @cf/meta-llama/llama-2-7b-chat-hf-lora")
    print(f"   Output: {adapter_output}")

    try:
        # Train with Llama 2-optimized parameters
        adapter_path = convert(
            documents_path=str(docs_path),
            output_path=str(adapter_output),
            model_name="meta-llama/Llama-2-7b-chat-hf",  # Use Llama 2 Chat model
            max_length=2048,  # Llama 2 supports up to 4096, using 2048 for efficiency
            batch_size=1,  # Small batch due to memory requirements
            num_epochs=3,  # Good balance for Llama 2
            learning_rate=2e-4,  # Optimal for Llama 2 fine-tuning
            lora_r=8,  # Max 8 for Cloudflare Workers AI compatibility
            lora_alpha=16,  # 2x rank ratio
            lora_dropout=0.05,  # Low dropout for stable training
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
        adapter_output.write_text(
            '{"model": "llama-2-7b-chat-hf", "type": "lora", "status": "demo"}'
        )
        print(f"📦 Created demo adapter file: {adapter_output}")
        return str(adapter_output)


def show_cloudflare_usage(adapter_path):
    """Show how to use the trained adapter with Cloudflare Workers AI."""
    print(f"\n{'='*60}")
    print("☁️  Cloudflare Workers AI Integration")
    print(f"{'='*60}")

    print(
        """
🔧 To use your Llama 2 LoRA adapter with Cloudflare Workers AI:

1. **Upload your adapter**:
   ```bash
   wrangler ai lora upload my-llama-adapter ./llama_lora_adapter_adapter/
   ```

2. **Use in your Worker**:
   ```javascript
   export default {
     async fetch(request, env) {
       const response = await env.AI.run("@cf/meta-llama/llama-2-7b-chat-hf-lora", {
         messages: [
           {
             "role": "system",
             "content": "You are a helpful assistant specialized in your domain."
           },
           {
             "role": "user",
             "content": "Explain the benefits of using Llama 2 for chat applications."
           }
         ],
         lora: "my-llama-adapter",
         max_tokens: 512,
         temperature: 0.7
       });

       return new Response(JSON.stringify(response));
     }
   }
   ```

💡 **Tips for Llama 2 Fine-tuning:**
- Use proper chat format in training data
- Include diverse conversation examples
- Monitor memory usage (Llama 2 is memory-intensive)
"""
    )


def main():
    """Main demo function."""
    try:
        # Run the Llama 2 training demo
        adapter_path = demo_llama_training()

        # Show Cloudflare usage
        if adapter_path:
            show_cloudflare_usage(adapter_path)

        print(f"\n{'='*60}")
        print("📚 Llama 2 Resources")
        print(f"{'='*60}")
        print(
            """
🔗 **Useful Links:**
- Llama 2 Model: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- Cloudflare Workers AI: https://developers.cloudflare.com/workers-ai/
- LoRA Paper: https://arxiv.org/abs/2106.09685

🎓 **Best Practices:**
- Always use the chat format for Llama 2-Chat models
- Include system messages to guide behavior
- Use diverse training examples for better generalization
"""
        )

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
