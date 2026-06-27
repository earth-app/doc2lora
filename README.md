# doc2lora

This repository is a small library for fine-tuning LLMs using LoRA (Low-Rank Adaptation) by using a folder of documents as input. It is designed to be simple and easy to use, allowing users to quickly adapt large language models to specific tasks or domains.

The library allows you to pass a folder of documents (local or from R2 bucket) and turn them into a LoRA Adapter. It is particularly useful for fine-tuning models on domain-specific data, such as legal documents, medical texts, or any other specialized corpus. It is intended to be used with Cloudflare Workers AI or similar platforms that support LLM fine-tuning.

It supports the following formats:

- **Markdown / reStructuredText**: `.md`, `.rst` files
- **Text**: `.txt` files or blank text files
- **PDF**: `.pdf` files
- **HTML**: `.html` files
- **Word Documents**: `.docx` files
- **PowerPoint**: `.pptx` files (slide text + speaker notes)
- **OpenDocument**: `.odt`, `.ods` files
- **Rich Text**: `.rtf` files
- **EPUB e-books**: `.epub` files
- **Excel Spreadsheets**: `.xlsx` files
- **CSV**: `.csv` files
- **JSON**: `.json` files
- **Jupyter notebooks**: `.ipynb` files (markdown + code cells)
- **YAML**: `.yaml` / `.yml` files
- **XML**: `.xml` files
- **LaTeX**: `.tex` files
- **Source code** (read as plaintext): `.py`, `.js`, `.ts`, `.java`, `.kt`, `.rs`, `.c`/`.cpp`, `.go`, `.rb`, `.php`, `.swift`, `.dart`, `.scala`, and more
- **Audio** (speech-to-text via Whisper): `.wav`, `.mp3`, `.m4a`, `.flac`, `.aac`, `.ogg`, and more
- **Images** (OCR text recognition): `.png`, `.jpg`, `.bmp`, `.gif`, `.tiff`, `.webp`, and more; `.svg` text is read from the markup
- **Video** (audio transcript + on-screen-text OCR): `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, and more
- **Archive Formats**: `.zip`, `.tar.gz`, `.tar.xz`, `.7z`, single-file `.gz`/`.bz2`/`.xz`, etc with supported documents inside

Run `doc2lora formats` to print the full list at any time.

## Quick Start

### Installation

```bash
# Core install (training only):
pip install doc2lora

# Everything (all document formats, audio, R2, QLoRA):
pip install "doc2lora[all]"

# Or pick what you need via extras:
pip install "doc2lora[docs]"    # pdf, docx, pptx, rtf, epub, xlsx, 7z
pip install "doc2lora[image]"   # image OCR (needs the system tesseract-ocr binary)
pip install "doc2lora[audio]"   # speech-to-text via Whisper (needs the ffmpeg binary)
pip install "doc2lora[video]"   # video: per-frame OCR + audio transcript
pip install "doc2lora[r2]"      # Cloudflare R2 ingestion
pip install "doc2lora[quant]"   # 4-bit QLoRA (CUDA only)

# For local development (editable + dev tools):
pip install -e ".[all,dev]"
```

**System dependencies** - the image/audio/video extras shell out to native binaries
that pip can't install:

| Feature | Needs | macOS (Homebrew) | Debian/Ubuntu | Fedora |
| ------- | ----- | ---------------- | ------------- | ------ |
| Image / video OCR (`[image]`, `[video]`) | `tesseract-ocr` | `brew install tesseract` | `sudo apt-get install tesseract-ocr` | `sudo dnf install tesseract` |
| Audio / video transcription (`[audio]`, `[video]`) | `ffmpeg` | `brew install ffmpeg` | `sudo apt-get install ffmpeg` | `sudo dnf install ffmpeg` |

`opencv-python` bundles its own libraries in the wheel (no system package), and
ODT/ODS and SVG are parsed with the stdlib (no extra or system binary). Audio/video transcription defaults to
`faster-whisper` (falls back to `openai-whisper`, then `SpeechRecognition`); choose a
backend with `--audio-backend` and a model size with `--whisper-model`. For more OCR
languages, install the tesseract language pack (e.g. `brew install tesseract-lang`,
`sudo apt-get install tesseract-ocr-fra`) and pass `--ocr-languages eng+fra`.

### Basic Usage

```bash
# Test the example
cd examples
python basic_usage.py
```

## Library Usage

To use the library, you can import it into your project and call the `convert` function with the path to the folder containing your documents, or use `convert_from_r2` to process documents from an R2 bucket. The library will handle the parsing and conversion of the documents into a format suitable for LoRA fine-tuning.

The `convert` function now supports multiple input types:

- **Folder path**: Pass a path to a folder containing documents
- **Array of strings**: Pass document content directly as strings
- **Array of bytes**: Pass document content as byte arrays
- **Single string**: Pass individual document content
- **Single bytes**: Pass individual document as bytes

### Subdirectory-Based Labeling

`doc2lora` now automatically uses subdirectory structure combined with filenames to create detailed labels, making it easy to organize training data by category.

When processing a folder, each document is automatically labeled by combining its subdirectory and filename:

```text
training_data/
├── legal/              # Documents labeled as "legal_[filename]"
│   ├── contract1.pdf   # -> "legal_contract1"
│   └── agreement.docx  # -> "legal_agreement"
├── technical/          # Documents labeled as "technical_[filename]"
│   ├── spec.md         # -> "technical_spec"
│   └── guide.txt       # -> "technical_guide"
├── marketing/          # Documents labeled as "marketing_[filename]"
│   ├── campaign.html   # -> "marketing_campaign"
│   └── copy.txt        # -> "marketing_copy"
└── overview.txt        # Root-level files → "root_overview"
```

**Generated metadata includes:**

```json
{
  "content": "Document content...",
  "filename": "contract1.pdf",
  "label": "legal_contract1",
  "category_path": "legal",
  "extension": ".pdf",
  "size": 1024
}
```

**Use Cases:**

- **Domain + Document type**: legal_contract, legal_agreement, technical_spec, technical_guide
- **Difficulty + Topic**: beginner_python, intermediate_javascript, advanced_algorithms
- **Type + Content**: manual_installation, faq_troubleshooting, tutorial_setup
- **Language + Region**: en_privacy_policy, es_terms_service, fr_user_guide
- **Time + Event**: 2023_quarterly_report, 2024_annual_summary, current_status

```bash
# See the labeling feature in action
cd examples
python subdirectory_labeling_demo.py
```

### Local Documents

```py
from doc2lora import convert

# Method 1: Convert a folder of documents
convert(documents_path="path/to/documents", output_path="path/to/output.json")

# Method 2: Convert array of strings directly
documents = [
    "This is document 1 content...",
    "This is document 2 content...",
    "This is document 3 content..."
]
convert(input_data=documents, output_path="path/to/output.json")

# Method 3: Convert single string
document_content = "This is my document content..."
convert(input_data=document_content, output_path="path/to/output.json")

# Method 4: Convert array of bytes
with open("doc1.txt", "rb") as f1, open("doc2.txt", "rb") as f2:
    byte_documents = [f1.read(), f2.read()]
convert(input_data=byte_documents, output_path="path/to/output.json")
```

### R2 Bucket Documents

```py
from doc2lora import convert_from_r2

# Method 1: Direct credentials
convert_from_r2(
    bucket_name="my-documents-bucket",
    folder_prefix="training-docs",  # optional
    output_path="path/to/output.json",
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    endpoint_url="https://your-account.r2.cloudflarestorage.com"
)

# Method 2: Using .env file (recommended)
convert_from_r2(
    bucket_name="my-documents-bucket",
    folder_prefix="training-docs",  # optional
    output_path="path/to/output.json",
    env_file=".env"  # Load credentials from .env file
)

# The output will be a JSON file containing the LoRA adapter data
# You can then use this output with your LLM fine-tuning framework
# For example, with Cloudflare Workers AI:
from cloudflare_workers_ai import LLM
llm = LLM(model="your-model-name")
llm.load_lora_adapter("path/to/output.json")
```

## CLI

You can also use the library from the command line. The CLI allows you to convert a folder of documents or R2 bucket contents into a LoRA adapter JSON file.

### CLI for Local Documents

```bash
doc2lora convert path/to/documents --output path/to/output.json

# scan first to preview files + per-file and total training-time estimates
doc2lora scan path/to/documents --device cpu

# low-memory machine: smaller batch + gradient accumulation (on by default:
# gradient checkpointing). 4-bit QLoRA is available on CUDA via --load-in-4bit
doc2lora convert path/to/documents \
    --batch-size 1 --gradient-accumulation-steps 8 \
    --output adapter.json
```

### Deploy to Cloudflare Workers AI

Once you have an adapter, upload it as a Workers AI finetune with one command:

```bash
# uses the wrangler CLI under the hood (validates the adapter first)
doc2lora deploy adapter.json my-finetune-name \
    --cf-model "@cf/mistralai/mistral-7b-instruct-v0.2-lora"

# or upload via the REST API (no wrangler needed)
doc2lora deploy adapter.json my-finetune-name --backend rest \
    --account-id "$CLOUDFLARE_ACCOUNT_ID" --api-token "$CLOUDFLARE_API_TOKEN"
```

Then reference it at inference time with the `lora` parameter
(`env.AI.run("@cf/mistralai/mistral-7b-instruct-v0.2-lora", { ..., lora: "my-finetune-name" })`).

### CLI for R2 Bucket Documents

```bash
# Method 1: Set environment variables for credentials
export R2_ACCESS_KEY_ID="your-access-key"
export R2_SECRET_ACCESS_KEY="your-secret-key"
export R2_ENDPOINT_URL="https://your-account.r2.cloudflarestorage.com"

# Convert documents from R2 bucket
doc2lora convert-r2 my-documents-bucket --folder-prefix training-docs --output path/to/output.json

# Method 2: Use .env file (recommended)
doc2lora convert-r2 my-documents-bucket \
    --env-file .env \
    --folder-prefix training-docs \
    --output path/to/output.json

# Method 3: Pass credentials directly
doc2lora convert-r2 my-documents-bucket \
    --r2-access-key-id "your-access-key" \
    --r2-secret-access-key "your-secret-key" \
    --endpoint-url "https://your-account.r2.cloudflarestorage.com" \
    --output path/to/output.json
```

## Project Structure

```text
doc2lora/
├── doc2lora/             # Main package
│   ├── __init__.py       # Package init + single-source __version__
│   ├── core.py           # convert() / convert_from_r2() entry points
│   ├── parsers.py        # Document / image / audio / video parsing
│   ├── lora_trainer.py   # LoRA training, device/precision, speedups
│   ├── deploy.py         # Upload adapters to Cloudflare Workers AI
│   ├── cli.py            # Command-line interface
│   └── utils.py          # R2 download + helpers
├── examples/             # Example usage scripts
│   ├── basic_usage.py
│   ├── media_and_optimization.py   # images/audio/video + speed knobs
│   ├── mistral_usage.py            # Mistral (needs HF_API_KEY)
│   ├── gemma_usage.py              # Gemma
│   ├── llama_usage.py              # Llama 2
│   ├── qwq_usage.py                # QwQ-32B (4-bit QLoRA)
│   ├── qlora_usage.py              # 4-bit QLoRA + deploy
│   ├── r2_usage.py                 # R2 bucket integration
│   ├── subdirectory_labeling_demo.py
│   └── example_documents/          # Sample documents
├── demo/                 # Complete Cloudflare Workers AI demo
│   ├── data/             # Sample training corpus
│   ├── scripts/          # train_lora.sh/.bat, deploy_to_r2.sh/.bat
│   ├── worker.js         # Worker (loads adapter, /chat endpoints)
│   ├── wrangler.toml     # Worker configuration
│   ├── index.html        # Browser UI
│   └── README.md         # Demo documentation
├── tests/                # Test suite (pytest)
├── pyproject.toml        # Packaging, dependencies/extras, tool config
├── requirements.txt      # Full install (equivalent to the [all] extra)
├── setup.py              # Thin shim (metadata lives in pyproject.toml)
├── README.md             # This file
├── USAGE.md              # Usage guide
├── INSTALL_GUIDE.md      # Install + Mistral guide
└── CLAUDE.md             # Repo guide for Claude Code
```

## Examples

The `examples/` directory contains usage examples for different models and scenarios:

### Model-Specific Examples

1. **`mistral_usage.py`** - Complete example for Mistral models with HuggingFace authentication

   ```bash
   cd examples
   export HF_API_KEY="your_huggingface_token"  # Required for Mistral models
   python mistral_usage.py
   ```

2. **`gemma_usage.py`** - Google Gemma model fine-tuning for Cloudflare Workers AI

   ```bash
   cd examples
   python gemma_usage.py
   ```

3. **`llama_usage.py`** - Meta Llama 2 model fine-tuning with optimized parameters

   ```bash
   cd examples
   python llama_usage.py
   ```

4. **`r2_usage.py`** - R2 bucket integration with .env file support

   ```bash
   cd examples
   python r2_usage.py
   ```

5. **`qlora_usage.py`** - Memory-efficient 4-bit QLoRA training (CUDA) + deploy

   ```bash
   cd examples
   python qlora_usage.py
   ```

6. **`qwq_usage.py`** - Fine-tuning the QwQ-32B reasoning model
   (`@cf/qwen/qwq-32b`) with 4-bit QLoRA; needs a 24 GB+ NVIDIA GPU

   ```bash
   cd examples
   python qwq_usage.py
   ```

7. **`media_and_optimization.py`** - Ingest images / audio / video and tune the
   training-speed knobs (the auto defaults plus the opt-in flags)

   ```bash
   cd examples
   python media_and_optimization.py
   ```

### Demo Application

The `demo/` folder contains a complete working demonstration of a Cloudflare Worker using a custom LoRA adapter:

```bash
# 1. Train a LoRA adapter on software development data
cd demo
./scripts/train_lora.sh  # or train_lora.bat on Windows

# 2. Deploy the adapter to R2 bucket
./scripts/deploy_to_r2.sh  # or deploy_to_r2.bat on Windows

# 3. Deploy the Cloudflare Worker
./scripts/wrangler_deploy.sh  # or wrangler_deploy.bat on Windows
```

The demo creates a **Software Developer Assistant** AI that provides guidance on:

- Code development and architecture
- Debugging and troubleshooting
- Team collaboration and communication
- Professional growth and career development
- Technical decision-making

**API Endpoints:**

- `GET /health` - Health check
- `POST /chat` - Send message and get response
- `POST /chat/stream` - Streaming responses
- `GET /docs` - API documentation

## Configuration

### GPU Support

🚀 **Automatic GPU Detection**: doc2lora now automatically detects and uses the best available device for training:

**Device Priority (Automatic):**

1. 🚀 **NVIDIA GPU (CUDA)** - Fastest; bf16 on Ampere+ (else fp16), TF32 matmul, and fused AdamW
2. 🍎 **Apple Silicon (MPS)** - Good performance on Mac M1/M2/M3 (bf16 on macOS 14+, else fp32; fp16 autocast on MPS is NaN-prone and is never auto-enabled)
3. 💻 **CPU** - Reliable fallback, works everywhere (fp32)

**Automatic Detection (Recommended):**

```bash
# Will automatically use GPU if available, fallback to CPU
doc2lora convert ./docs --output adapter.json
```

**Manual Device Selection:**

```bash
# Force GPU usage
doc2lora convert ./docs --output adapter.json --device cuda

# Force CPU usage (useful for troubleshooting)
doc2lora convert ./docs --output adapter.json --device cpu

# Use Apple Silicon GPU (Mac M1/M2/M3)
doc2lora convert ./docs --output adapter.json --device mps
```

**Python API:**

```python
from doc2lora import convert

# Auto-detect device (recommended)
convert(documents_path="./docs", output_path="adapter.json")

# Specify device manually
convert(documents_path="./docs", output_path="adapter.json", device="cuda")
convert(documents_path="./docs", output_path="adapter.json", device="cpu")
convert(documents_path="./docs", output_path="adapter.json", device="mps")  # Apple Silicon
```

**GPU Requirements:**

- **NVIDIA GPUs**: Requires CUDA-compatible PyTorch installation
- **Apple Silicon**: Requires PyTorch with MPS support (automatically included on macOS)
- **Memory**: 8GB+ GPU memory recommended for larger models

### Training Parameters

Common configuration options:

```bash
doc2lora convert ./docs \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --batch-size 2 \
    --epochs 3 \
    --learning-rate 2e-4 \
    --lora-r 8 \
    --lora-alpha 16 \
    --gradient-accumulation-steps 4 \
    --device auto  # or cuda/mps/cpu
```

**LoRA rank:** the default is `8` (broadest compatibility). Cloudflare Workers AI
now accepts adapters up to **rank 32** (with a 300MB safetensors limit), so you can
raise `--lora-r` up to 32 for more capacity; doc2lora only warns above 32.

**Performance / low-resource options:**

- ⚡ **Gradient checkpointing** (on by default): trades ~20% compute for a large
  memory saving. Disable with `--no-gradient-checkpointing`.
- 🧮 **Gradient accumulation**: `--gradient-accumulation-steps N` emulates a larger
  effective batch (`batch_size * N`) without the memory cost - ideal on weak machines.
- 🪶 **4-bit QLoRA**: `--load-in-4bit` (CUDA + `pip install "doc2lora[quant]"`) loads
  the base model in 4-bit (nf4) so large models fit on small GPUs.
- 🚀 **Precision**: bf16 on bf16-capable CUDA and Apple MPS (macOS 14+), fp16 on other
  CUDA GPUs, fp32 on CPU and older MPS (fp16 autocast on MPS is NaN-prone).
- 💻 **Out of Memory**: reduce `--batch-size`, raise `--gradient-accumulation-steps`,
  or fall back with `--device cpu` (CUDA OOM also auto-falls back to CPU).

### Training speed optimizations

doc2lora applies a set of **hardware-aware speedups automatically** - most are
no-ops where they don't apply - and exposes a few **opt-in** ones for power users.

**Applied automatically:**

| Optimization | What it does | Where it helps |
| ------------ | ------------ | -------------- |
| **TF32 matmul** (`set_float32_matmul_precision("high")`) | runs fp32 matmuls on Tensor Cores | NVIDIA **Ampere+** (no-op on older CUDA / MPS / CPU) |
| **bf16 / fp16 precision** | bf16 on bf16-capable CUDA & MPS (macOS 14+), fp16 on other CUDA, fp32 elsewhere | CUDA, Apple Silicon |
| **Fused AdamW** (`optim="adamw_torch_fused"`) | single fused optimizer kernel | CUDA with PyTorch >= 2.8 (else plain AdamW) |
| **SDPA attention** | PyTorch scaled-dot-product attention auto-selects the fastest kernel | all (CUDA fused kernels; math fallback on CPU/MPS) |
| **`pad_to_multiple_of=8`** | aligns padded batches to Tensor-Core tiles | CUDA (harmless elsewhere) |
| **CUDA-gated pinned memory** | `dataloader_pin_memory` only on CUDA | avoids wasted host RAM on CPU/MPS |
| **Gradient checkpointing** (`use_reentrant=False`) | recompute activations to save memory | low-RAM machines (on by default; ~20% slower - disable with `--no-gradient-checkpointing` if you have memory headroom) |
| **`torch.compile`** | fuses the model graph (~20-50% faster steps) | **auto on CUDA when the corpus is >= ~10 MB of text** (compile cost amortizes on long runs only; CUDA-only). Force with `--torch-compile` / `--no-torch-compile` |
| **Length-grouped batching** | groups similar-length samples to cut padding | **auto when batch_size >= 2** (hardware-agnostic; nothing to cut at batch 1). Force with `--group-by-length` / `--no-group-by-length` |
| **Parallel parsing** | thread pool over the document folder (PDF / OCR / transcription) | auto: ~`min(8, CPU count)` threads; tune with `--max-workers N` |

**Opt-in (you choose when it's worth it):**

| Flag | What it does | When to use |
| ---- | ------------ | ----------- |
| `--no-torch-compile` | force `torch.compile` off (it auto-enables on CUDA for large corpora) | short runs, debugging, or if compile graph-breaks |
| `--no-group-by-length` | force length-grouped batching off (it auto-enables at batch >= 2) | if you need strict shuffle order or hit a convergence quirk |
| `--attn-implementation flash_attention_2` | FlashAttention-2 kernel | Ampere+ CUDA with `flash-attn` + bf16/fp16 (falls back to eager; SDPA already uses flash kernels by default) |
| `--optim adamw_bnb_8bit` | 8-bit Adam (~75% less optimizer memory) | full fine-tuning on CUDA; **little benefit for LoRA** (optimizer state is just the tiny adapter) - needs `[quant]`/bitsandbytes |
| `--dataloader-num-workers N` | extra DataLoader worker processes | large corpora on Linux/CUDA only (default 0; keep 0 on macOS / in-memory data) |

**Per-platform notes:**

- **NVIDIA CUDA**: prefer **bf16 on Ampere+** (no loss scaling, no NaNs); TF32, the
  fused optimizer, and (for corpora >= ~10 MB of text) `torch.compile` are on
  automatically. For very long sequences, `--attn-implementation flash_attention_2`
  (with `pip install flash-attn`) is the biggest single win.
- **Apple Silicon (MPS)**: bf16 is used on **macOS 14+** (same memory as fp16, but
  stable), else fp32. `torch.compile`, FlashAttention, and 4-bit QLoRA do **not** help
  on MPS; length-grouped batching (auto at batch_size >= 2) is the main extra lever that
  does. Keep the whole model in unified memory (no CPU offload exists on MPS).
- **CPU**: training is a slow fallback. PyTorch already defaults to your physical-core
  count; you can pin it with `OMP_NUM_THREADS`. Use a small base model.

### How long will training take?

All numbers below are **order-of-magnitude estimates** and vary widely with
sequence length, batch size, LoRA rank, and data shape. `doc2lora scan <dir>
--device <d>` prints an estimate for your own corpus.

#### Small base model (DialoGPT-small / GPT-2 class), 3 epochs

| Corpus size | CPU       | Apple MPS | NVIDIA CUDA |
| ----------- | --------- | --------- | ----------- |
| ~1 MB       | minutes   | ~1 min    | seconds     |
| ~10 MB      | ~1 hour   | ~10 min   | ~2 min      |
| ~100 MB     | many hrs  | ~1-2 hrs  | ~20 min     |

#### 7B-class model (Mistral / Gemma / Llama) vs hardware and VRAM

Times below are for **3 epochs** at ~512-token sequences. The "approach" column
reflects what fits in memory:

- **>= 24 GB VRAM**: full fp16/bf16 LoRA fits comfortably.
- **12 GB VRAM**: use 4-bit QLoRA (`--load-in-4bit`) to fit a 7B model.
- **Apple Silicon**: 4-bit QLoRA is CUDA-only (bitsandbytes), so MPS runs **bf16
  LoRA** (macOS 14+, else fp32) and needs ~18 GB+ unified memory for a 7B model;
  8 GB Macs cannot train 7B (use a smaller base model). MPS is also much slower
  than a discrete GPU.

| Hardware | Memory            | 7B approach               | 1 MB    | 10 MB    | 10 MB +optimizations&dagger; | 100 MB    | 100 MB +optimizations&dagger; |
| -------- | ----------------- | ------------------------- | ------- | -------- | ---------------------------- | --------- | ----------------------------- |
| Apple M2 | 8-24 GB unified   | bf16 LoRA (16 GB+ for 7B) | ~1 hr   | ~11 hrs  | n/a (CUDA only)              | ~4-5 days | n/a (CUDA only)               |
| Apple M3 | 8-128 GB unified  | bf16 LoRA                 | ~40 min | ~6 hrs   | n/a                          | ~2-3 days | n/a                           |
| Apple M4 | 16-128 GB unified | bf16 LoRA                 | ~25 min | ~4 hrs   | n/a                          | ~1.5 days | n/a                           |
| RTX 4070 | 12 GB             | QLoRA (4-bit) required    | ~10 min | ~1.5 hrs | ~1 hr                        | ~17 hrs   | ~12 hrs                       |
| RTX 5070 | 12 GB             | QLoRA (4-bit) required    | ~7 min  | ~1.2 hrs | ~50 min                      | ~12 hrs   | ~8-9 hrs                      |
| RTX 3090 | 24 GB             | full bf16 LoRA            | ~7 min  | ~1 hr    | ~40 min                      | ~11 hrs   | ~7-8 hrs                      |
| RTX 4090 | 24 GB             | full bf16 LoRA            | ~4 min  | ~35 min  | ~25 min                      | ~6 hrs    | ~4 hrs                        |
| RTX 5090 | 32 GB             | full bf16 LoRA            | ~2 min  | ~20 min  | ~15 min                      | ~3-4 hrs  | ~2-3 hrs                      |

&dagger; **+optimizations** = the dynamic speedups doc2lora turns on for you, on top of the
always-on defaults (TF32, fused AdamW, bf16, SDPA, `pad_to_multiple_of=8` - already in the
base columns). For these CUDA rows it is dominated by **`torch.compile`** (auto on CUDA once
the corpus is >= ~10 MB of text; ~20-40% faster steps per HuggingFace, the column applies
~30%) plus **length-grouped batching** (auto at batch_size >= 2). The compile cost (seconds
to minutes) only amortizes on long runs, so the 10 MB and 100 MB columns reflect it while the
~1 MB / minutes-scale column does not. Override with `--torch-compile` / `--no-torch-compile`
(and `--group-by-length`). It does **not** help on **Apple MPS** - compile and FlashAttention
are CUDA-only, and 7B on Apple runs at batch_size 1 (no length-grouping win), so Apple's gains
are the bf16/etc. already in the base columns. For long sequences on Ampere+ GPUs, stack
`--attn-implementation flash_attention_2` (needs `pip install flash-attn`).

> **What counts as an "example"?** Each document becomes **one or more** training
> examples: a file <= `--max-length` tokens (default 512, ~2 KB of text) is one example,
> and a longer file is **auto-chunked** into consecutive `--max-length`-token windows -
> one example each - so **all of its content is trained on** (use `--chunk-overlap N` to
> overlap windows, or `--no-chunk` to revert to truncating each file to its first window).
> So "a few hundred to a few thousand examples" is really *total tokens / max_length*.
>
> **How this impacts performance:** training time scales with **total tokens**, so
> chunking a few huge files can balloon the run - a dozen 1 MB files is ~6,000 chunks
> (~500x more steps than the old truncate-to-12-examples behavior). Bound it with
> `--max-steps`, fewer `--epochs`, a smaller/curated corpus, or `--no-chunk`. Curated
> quality still beats raw quantity. `doc2lora scan` estimates time from total bytes - a
> rough figure, but its all-content basis now matches what's actually trained (chunking
> on); under the old truncation default it over-counted for big files. The small-model
> table above is ~20-40x faster if you only need a lightweight adapter.

#### 32B-class model (QwQ-32B) vs hardware and VRAM

QwQ-32B (`@cf/qwen/qwq-32b`) also accepts BYO LoRA adapters. A 32B base is roughly
4-5x slower than 7B and only fits with **4-bit QLoRA**, which needs ~20-24 GB of
VRAM - so it is realistically a 24 GB+ NVIDIA job. Times are for **3 epochs** at
~512-token sequences (see `examples/qwq_usage.py`).

| Hardware        | Memory   | 32B approach             | 1 MB     | 10 MB    | 100 MB   |
| --------------- | -------- | ------------------------ | -------- | -------- | -------- |
| Apple M2/M3/M4  | unified  | not practical (no 4-bit) | -        | -        | -        |
| RTX 4070 / 5070 | 12 GB    | too small for 32B        | -        | -        | -        |
| RTX 3090        | 24 GB    | QLoRA (4-bit), tight     | ~30 min  | ~4.5 hrs | ~2 days  |
| RTX 4090        | 24 GB    | QLoRA (4-bit)            | ~18 min  | ~2.5 hrs | ~1 day   |
| RTX 5090        | 32 GB    | QLoRA (4-bit), roomy     | ~9 min   | ~1.5 hrs | ~15 hrs  |

> A rank-8..32 adapter on a 32B model is still well under Cloudflare's 300 MB
> safetensors limit. doc2lora tags Qwen/QwQ adapters with `model_type: qwen`
> automatically; deploy with `--cf-model "@cf/qwen/qwq-32b"`.

## Features

- ✅ **Document Parsing**: Recursively scan directories for supported document types
- ✅ **Subdirectory Labeling**: Automatically label documents based on directory structure and filename
- ✅ **Multiple Formats**: Support for 20+ document, image, audio, and video formats including archives
- ✅ **Archive Support**: Extract and parse documents from ZIP and TAR archives
- ✅ **R2 Bucket Support**: Direct integration with Cloudflare R2 storage buckets
- ✅ **CLI Interface**: Easy-to-use command-line interface
- ✅ **Image / Audio / Video**: OCR images (tesseract), transcribe audio & video with Whisper, and OCR on-screen text from frames
- ✅ **Parallel Parsing**: multithreaded parsing of the document folder
- ✅ **Hardware-aware Speedups**: TF32, bf16, fused AdamW, SDPA, and auto `torch.compile` / length-grouped batching selected by device + corpus size
- ✅ **One-command Deploy**: `doc2lora deploy` uploads adapters to Cloudflare Workers AI (wrangler or REST)
- ✅ **Flexible Configuration**: Customizable LoRA parameters
- 🔄 **LoRA Training**: Fine-tune models using LoRA adaptation (requires ML dependencies)
- 🔄 **Export Options**: JSON format compatible with various platforms

## Status

- **Document Parsing**: ✅ Fully working
- **CLI Interface**: ✅ Basic functionality working
- **LoRA Training**: 🔄 Requires ML dependencies (torch, transformers, peft, datasets)

The core document parsing functionality works out of the box. For full LoRA training capabilities, install the ML dependencies listed above.
