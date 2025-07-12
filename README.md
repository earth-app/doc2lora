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

### Local Documents

```py
from doc2lora import convert

# Convert a folder of documents to LoRA format
convert("path/to/documents", output_path="path/to/output.json")
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
│   └── example_documents/  # Sample documents
│       ├── sample.md
│       ├── sample.txt
│       ├── sample.json
│       └── sample.csv
├── tests/             # Test suite
├── requirements.txt   # Dependencies
├── setup.py          # Package setup
└── README.md         # This file
```

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
