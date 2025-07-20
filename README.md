# doc2lora

This repository is a small library for fine-tuning LLMs using LoRA (Low-Rank Adaptation) by using a folder of documents as input. It is designed to be simple and easy to use, allowing users to quickly adapt large language models to specific tasks or domains.

The library allows you to pass a folder of documents (local or from R2 bucket) and turn them into a LoRA Adapter. It is particularly useful for fine-tuning models on domain-specific data, such as legal documents, medical texts, or any other specialized corpus. It is intended to be used with Cloudflare Workers AI or similar platforms that support LLM fine-tuning.

It supports the following formats:

- **Markdown**: `.md` files
- **Text**: `.txt` files or blank text files
- **PDF**: `.pdf` files
- **HTML**: `.html` files
- **Word Documents**: `.docx` files
- **CSV**: `.csv` files
- **JSON**: `.json` files
- **YAML**: `.yaml` files
- **XML**: `.xml` files
- **LaTeX**: `.tex` files

## Quick Start

### Installation

```bash
# Install the package
pip install -e .

# For full functionality with ML training, install additional dependencies:
pip install torch transformers peft datasets

# For additional document format support:
pip install PyPDF2 python-docx beautifulsoup4 PyYAML

# For R2 bucket support:
pip install boto3
```

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
```

### CLI for R2 Bucket Documents

```bash
# Method 1: Set environment variables for credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
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
    --aws-access-key-id "your-access-key" \
    --aws-secret-access-key "your-secret-key" \
    --endpoint-url "https://your-account.r2.cloudflarestorage.com" \
    --output path/to/output.json
```

## Project Structure

```text
doc2lora/
├── doc2lora/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── core.py         # Main convert function
│   ├── parsers.py      # Document parsing logic
│   ├── lora_trainer.py # LoRA training implementation
│   ├── cli.py          # Command-line interface
│   └── utils.py        # Utility functions
├── examples/           # Example usage
│   ├── basic_usage.py  # Working example script
│   ├── mistral_usage.py # Mistral model example with HF API key
│   ├── gemma_usage.py  # Gemma model example for Cloudflare AI
│   ├── llama_usage.py  # Llama model example for Cloudflare AI
│   ├── r2_usage.py     # R2 bucket integration example
│   └── example_documents/  # Sample documents
│       ├── sample.md
│       ├── sample.txt
│       ├── sample.json
│       └── sample.csv
├── demo/              # Complete working demonstration
│   ├── data/          # Sample training documents about software development
│   ├── scripts/       # Automation scripts (train_lora.sh/.bat, deploy_to_r2.sh/.bat)
│   ├── worker.js      # Cloudflare Worker implementation
│   ├── wrangler.toml  # Cloudflare Worker configuration
│   ├── index.html     # Web interface for testing
│   └── README.md      # Demo documentation
├── tests/             # Test suite
├── requirements.txt   # Dependencies
├── setup.py          # Package setup
└── README.md         # This file
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

1. 🚀 **NVIDIA GPU (CUDA)** - Fastest training with fp16 precision and optimal memory usage
2. 🍎 **Apple Silicon (MPS)** - Good performance on Mac M1/M2/M3
3. 💻 **CPU** - Reliable fallback, works everywhere

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
    --lora-r 16 \
    --lora-alpha 32 \
    --device auto  # or cuda/mps/cpu
```

**Memory Management:**

- 🚀 **GPU Training**: Automatically uses fp16 precision on CUDA GPUs to save memory
- 🔧 **Out of Memory**: Reduce `--batch-size` if you encounter GPU memory errors
- 💻 **CPU Fallback**: Use `--device cpu` if GPU memory is insufficient
- ⚡ **Automatic Optimization**: The system automatically chooses optimal settings per device

## Features

- ✅ **Document Parsing**: Recursively scan directories for supported document types
- ✅ **Multiple Formats**: Support for 10+ document formats
- ✅ **R2 Bucket Support**: Direct integration with Cloudflare R2 storage buckets
- ✅ **CLI Interface**: Easy-to-use command-line interface
- ✅ **Flexible Configuration**: Customizable LoRA parameters
- 🔄 **LoRA Training**: Fine-tune models using LoRA adaptation (requires ML dependencies)
- 🔄 **Export Options**: JSON format compatible with various platforms

## Status

- **Document Parsing**: ✅ Fully working
- **CLI Interface**: ✅ Basic functionality working
- **LoRA Training**: 🔄 Requires ML dependencies (torch, transformers, peft, datasets)

The core document parsing functionality works out of the box. For full LoRA training capabilities, install the ML dependencies listed above.
