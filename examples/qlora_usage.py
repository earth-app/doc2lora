"""Example: memory-efficient training with 4-bit QLoRA + Cloudflare deploy.

QLoRA loads the base model in 4-bit (nf4) so large models fit on a single
consumer GPU. It requires CUDA and bitsandbytes:

    pip install "doc2lora[quant]"

On CPU/MPS (or without bitsandbytes) the 4-bit request is ignored and doc2lora
falls back to standard LoRA - the rest of the example still runs.
"""

import sys
from pathlib import Path

# import doc2lora from the repo without installing it
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc2lora import convert  # noqa: E402


def main():
    docs_path = Path(__file__).parent / "example_documents"

    # rank up to 32 is now accepted by Cloudflare Workers AI (300MB limit);
    # 16 is a good middle ground when you have the QLoRA memory headroom
    adapter_path = convert(
        documents_path=str(docs_path),
        output_path="qlora_adapter.json",
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        lora_r=16,
        lora_alpha=32,
        batch_size=1,
        gradient_accumulation_steps=8,  # effective batch of 8 with batch_size 1
        gradient_checkpointing=True,
        load_in_4bit=True,  # 4-bit QLoRA (CUDA only; ignored elsewhere)
        num_epochs=3,
        learning_rate=2e-4,
    )

    print(f"✅ Adapter written to: {adapter_path}")
    print("\nDeploy it to Cloudflare Workers AI with:")
    print(
        "  doc2lora deploy qlora_adapter.json my-finetune "
        '--cf-model "@cf/mistralai/mistral-7b-instruct-v0.2-lora"'
    )


if __name__ == "__main__":
    main()
