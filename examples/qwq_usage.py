"""Example: fine-tuning QwQ-32B (Qwen reasoning model) for Cloudflare Workers AI.

QwQ-32B is a 32B reasoning model. On Workers AI it is served as
``@cf/qwen/qwq-32b`` and accepts bring-your-own LoRA adapters.

Training a 32B model locally is heavy: use 4-bit QLoRA and a GPU with plenty of
VRAM (roughly 24GB+ for QwQ-32B in 4-bit). QLoRA is CUDA-only, so this realistically
needs an NVIDIA card (e.g. RTX 3090/4090/5090). doc2lora detects the Qwen
architecture and tags the adapter with ``model_type: qwen`` for Cloudflare.

    pip install "doc2lora[quant]"   # bitsandbytes for 4-bit QLoRA (CUDA)
"""

import sys
from pathlib import Path

# import doc2lora from the repo without installing it
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc2lora import convert  # noqa: E402

# QwQ-32B base model on HuggingFace; the Workers AI endpoint is @cf/qwen/qwq-32b
BASE_MODEL = "Qwen/QwQ-32B"
CF_MODEL = "@cf/qwen/qwq-32b"


def main():
    docs_path = Path(__file__).parent / "example_documents"

    # rank up to 32 is accepted by Cloudflare (a rank-32 adapter on a 32B model is
    # still well under the 300MB safetensors limit)
    adapter_path = convert(
        documents_path=str(docs_path),
        output_path="qwq_adapter.json",
        model_name=BASE_MODEL,
        lora_r=16,
        lora_alpha=32,
        batch_size=1,
        gradient_accumulation_steps=16,  # large effective batch, tiny memory footprint
        gradient_checkpointing=True,
        load_in_4bit=True,  # required to fit 32B on a single GPU
        num_epochs=3,
        learning_rate=1e-4,
        max_length=1024,  # QwQ supports long reasoning context (up to 24k on CF)
    )

    print(f"✅ Adapter written to: {adapter_path}")
    print("\nDeploy it to Cloudflare Workers AI with:")
    print(f'  doc2lora deploy qwq_adapter.json my-qwq-finetune --cf-model "{CF_MODEL}"')
    print(
        "\nThen call it from a Worker (QwQ emits <think> reasoning, then an answer):\n"
        '  env.AI.run("@cf/qwen/qwq-32b", {\n'
        '    messages: [{ role: "user", content: "..." }],\n'
        '    lora: "my-qwq-finetune",\n'
        "  })"
    )


if __name__ == "__main__":
    main()
