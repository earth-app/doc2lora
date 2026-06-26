# doc2lora Development

## Installation

### For Development

1. Clone the repository:

```bash
git clone https://github.com/earth-app/doc2lora.git
cd doc2lora
```

2. Install in development mode:

```bash
pip install -e .
```

### For Production

Install from PyPI (when published). The core install is training-only; document
parsers and extras are pulled in via pip extras:

```bash
# Core (training only)
pip install doc2lora

# Everything
pip install "doc2lora[all]"

# Pick what you need:
pip install "doc2lora[docs]"   # pdf/docx/pptx/odt/ods/rtf/epub/xlsx/7z parsers
pip install "doc2lora[audio]"  # speech-to-text (also needs the system ffmpeg binary)
pip install "doc2lora[r2]"     # Cloudflare R2 ingestion
pip install "doc2lora[quant]"  # 4-bit QLoRA (CUDA only)
pip install "doc2lora[dev]"    # dev/test tooling
```

You can combine extras, e.g. `pip install "doc2lora[docs,r2]"`.

## Quick Start

### Using as a Library

#### Working with Local Documents

```python
from doc2lora import convert

# Convert documents to LoRA adapter
adapter_path = convert(
    documents_path="path/to/your/documents",
    output_path="my_lora_adapter.json"
)

print(f"LoRA adapter created at: {adapter_path}")
```

#### Working with R2 Bucket Documents

```python
from doc2lora import convert_from_r2

# Convert documents from R2 bucket to LoRA adapter
adapter_path = convert_from_r2(
    bucket_name="my-documents-bucket",
    folder_prefix="training-docs",  # optional
    output_path="my_lora_adapter.json",
    aws_access_key_id="your-access-key-id",
    aws_secret_access_key="your-secret-access-key",
    endpoint_url="https://your-account.r2.cloudflarestorage.com"
)

print(f"LoRA adapter created at: {adapter_path}")
```

### Using the CLI

#### CLI with Local Documents

```bash
# Basic usage
doc2lora convert path/to/documents --output adapter.json

# With custom parameters
doc2lora convert path/to/documents \
    --output adapter.json \
    --model microsoft/DialoGPT-medium \
    --epochs 5 \
    --batch-size 2
```

#### CLI with R2 Bucket Documents

```bash
# Method 1: Using environment variables for credentials
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export R2_ENDPOINT_URL="https://your-account.r2.cloudflarestorage.com"

# Basic usage with R2 bucket
doc2lora convert-r2 my-documents-bucket --output adapter.json

# Method 2: Using .env file (recommended)
doc2lora convert-r2 my-documents-bucket \
    --env-file .env \
    --output adapter.json

# With folder prefix and custom parameters
doc2lora convert-r2 my-documents-bucket \
    --env-file .env \
    --folder-prefix training-docs \
    --output adapter.json \
    --model microsoft/DialoGPT-medium \
    --epochs 5 \
    --batch-size 2

# Method 3: Passing credentials directly (not recommended for production)
doc2lora convert-r2 my-documents-bucket \
    --aws-access-key-id "your-access-key-id" \
    --aws-secret-access-key "your-secret-access-key" \
    --endpoint-url "https://your-account.r2.cloudflarestorage.com" \
    --output adapter.json
```

#### Other CLI Commands

```bash
# Scan directory for supported files (prints a rough training-time estimate)
doc2lora scan path/to/documents
doc2lora scan path/to/documents --device cpu   # estimate for a specific device

# List supported formats
doc2lora formats
```

#### Deploy to Cloudflare Workers AI

After training, upload the adapter to Cloudflare Workers AI with `doc2lora deploy`.
It validates the adapter first (rank <= 32, < 300MB, required filenames, model_type).

```bash
# Default: shell out to the wrangler CLI (wrangler ai finetune create)
doc2lora deploy adapter.json my-finetune \
    --cf-model "@cf/mistralai/mistral-7b-instruct-v0.2-lora"

# Alternative: Cloudflare REST API (reads CLOUDFLARE_ACCOUNT_ID /
# CLOUDFLARE_API_TOKEN env vars if the flags are omitted)
doc2lora deploy adapter.json my-finetune \
    --backend rest \
    --account-id "your-account-id" \
    --api-token "your-api-token" \
    --cf-model "@cf/mistralai/mistral-7b-instruct-v0.2-lora"
```

At inference, reference the uploaded finetune via the `lora` parameter:

```javascript
const response = await env.AI.run('@cf/mistralai/mistral-7b-instruct-v0.2-lora', {
  messages: [{ role: "user", content: "..." }],
  lora: "my-finetune"
});
```

## Configuration Options

### Training Parameters

- `--model`: Base model for fine-tuning (default: microsoft/DialoGPT-small)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--max-length`: Maximum sequence length (default: 512)
- `--device`: Device for training - `auto` (default), `cuda`, `mps`, or `cpu`
- `--gradient-accumulation-steps`: Accumulate gradients over N steps to emulate a
  larger batch on low-memory machines (default: 1)
- `--gradient-checkpointing` / `--no-gradient-checkpointing`: Trade ~20% compute for
  lower memory use (default: on)
- `--load-in-4bit`: Use 4-bit QLoRA (CUDA + bitsandbytes only; install the `quant` extra)

Precision is selected automatically: bf16 on capable CUDA GPUs, fp16 on other GPUs,
fp32 on CPU.

### LoRA Parameters

- `--lora-r`: LoRA rank parameter (default: 8; Cloudflare Workers AI supports up to 32)
- `--lora-alpha`: LoRA alpha parameter (default: 32)
- `--lora-dropout`: LoRA dropout rate (default: 0.1)

## Supported Document Formats

- **Markdown** (.md): Full markdown parsing
- **Text** (.txt): Plain text files
- **PDF** (.pdf): Text extraction from PDF documents (via `pypdf`, with PyPDF2 fallback)
- **HTML** (.html): Text extraction from HTML
- **Word** (.docx): Microsoft Word documents
- **PowerPoint** (.pptx): Microsoft PowerPoint presentations
- **OpenDocument** (.odt, .ods): OpenDocument text and spreadsheets
- **Rich Text** (.rtf): Rich Text Format documents
- **EPUB** (.epub): EPUB e-books
- **Excel** (.xlsx): Excel spreadsheet files
- **CSV** (.csv): Comma-separated values
- **JSON** (.json): JSON data files
- **YAML** (.yaml, .yml): YAML configuration files
- **XML** (.xml): XML documents
- **LaTeX** (.tex): LaTeX source files
- **reStructuredText** (.rst): reStructuredText documents
- **Jupyter notebooks** (.ipynb): Notebook cells extracted as text
- **Source code** (.py, .js, .ts, .java, .kt, .rs, .c, .cpp, .go, .rb, .php, .swift,
  .dart, .scala, and more): Read as plaintext
- **Audio** (.wav, .mp3, .m4a, .flac, .aac, .ogg, and more): Transcribed via
  speech-to-text (non-wav formats need the system ffmpeg binary)
- **ZIP** (.zip): ZIP archives containing supported documents
- **TAR** (.tar): TAR archives containing supported documents
- **Compressed TAR** (.tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, .txz): Compressed TAR archives
- **7-Zip** (.7z): 7-Zip archives containing supported documents
- **Single-file compressed** (.gz, .bz2, .xz): Standalone compressed documents

### Archive Support

Archive formats (.zip, .tar, and compressed variants) are supported by extracting and parsing any supported document files they contain. The parser will:

1. Extract all files from the archive to a temporary location
2. Identify files with supported document extensions
3. Parse each supported document using the appropriate parser
4. Combine all parsed content into a single output with clear file separators
5. Skip unsupported file types with a notification

This allows you to include entire project documentation, collections of related documents, or backup archives in your LoRA training data.

## Examples

See the `examples/` directory for sample usage and test documents.

## Testing

Run tests with:

```bash
python -m pytest tests/
```

## Hardware Requirements

- **GPU Recommended**: NVIDIA CUDA GPU or Apple Silicon (M1/M2) for faster training
- **Memory**: At least 8GB RAM, 16GB+ recommended for larger models
- **Storage**: Adequate space for model weights and training data

**Automatic Device Detection:**
doc2lora automatically detects and uses the best available device:
1. 🚀 NVIDIA GPU (CUDA) - Uses fp16 precision for memory efficiency
2. 🍎 Apple Silicon (MPS) - Good performance on Mac M1/M2
3. 💻 CPU - Reliable fallback, works everywhere

### Training Time Estimates

Approximate, for a small base model over 3 epochs (use `doc2lora scan` for a
per-machine estimate):

| Corpus size | CPU       | Apple MPS  | NVIDIA CUDA |
| ----------- | --------- | ---------- | ----------- |
| ~1 MB       | minutes   | ~1 min     | seconds     |
| ~10 MB      | ~1 hour   | ~10 min    | ~2 min      |
| ~100 MB     | many hrs  | ~1-2 hrs   | ~20 min     |

7B-class models are roughly 20-40x slower than these figures; on CUDA, use
`--load-in-4bit` (QLoRA) to fit them in memory and speed training up.

## Model Recommendations

### Small Models (Good for testing)

- `microsoft/DialoGPT-small`
- `gpt2`

### Medium Models (Balanced performance)

- `microsoft/DialoGPT-medium`
- `gpt2-medium`

### Large Models (Better quality, requires more resources)

- `microsoft/DialoGPT-large`
- `gpt2-large`

## Troubleshooting

### Common Issues

1. **GPU out of memory**: Reduce batch size (`--batch-size 1`) or use CPU (`--device cpu`)
2. **Missing dependencies**: Install the relevant extra, e.g. `pip install "doc2lora[docs]"` (or `[all]`)
3. **PDF parsing errors**: Ensure `pypdf` is installed (part of the `docs` extra; PyPDF2 also works as a fallback)
4. **Slow training**: GPU is automatically used when available; reduce dataset size for testing

### Performance Tips

- GPU acceleration is automatically enabled when available
- Use `--device cpu` to force CPU usage for troubleshooting
- Start with smaller batch sizes and increase gradually
- Monitor memory usage during training
- Use smaller models for initial testing

### Device-Specific Tips

**NVIDIA GPU (CUDA):**

- Automatically uses fp16 precision for memory efficiency
- Best performance for training

**Apple Silicon (MPS):**

- Good performance on Mac M1/M2
- Automatically detected and used

**CPU Fallback:**

- Automatically used when GPU is not available
- Slower but reliable
- Use `--device cpu` to force CPU usage

## R2 Bucket Setup

### Prerequisites for R2 Bucket Support

To use R2 bucket functionality, you need:

1. **Cloudflare R2 Bucket**: A configured R2 bucket containing your documents
2. **R2 API Token**: API credentials with R2 read permissions
3. **boto3 Library**: Install with `pip install boto3`

### Obtaining R2 Credentials

1. **Create an R2 API Token**:
   - Go to Cloudflare Dashboard → R2 Object Storage → Manage R2 API tokens
   - Create a new API token with "Read" permissions for your bucket
   - Note down the Access Key ID and Secret Access Key

2. **Get your R2 Endpoint URL**:
   - Format: `https://your-account-id.r2.cloudflarestorage.com`
   - Replace `your-account-id` with your actual Cloudflare account ID

### Authentication Methods

#### Method 1: Environment Variables (Recommended)

```bash
# Set these environment variables
export AWS_ACCESS_KEY_ID="your-r2-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-r2-secret-access-key"
export R2_ENDPOINT_URL="https://your-account.r2.cloudflarestorage.com"

# Then use the CLI without additional parameters
doc2lora convert-r2 my-bucket --output adapter.json
```

#### Method 2: .env File (Recommended for Security)

Create a `.env` file in your project directory:

```env
# .env file
AWS_ACCESS_KEY_ID=your-r2-access-key-id
AWS_SECRET_ACCESS_KEY=your-r2-secret-access-key
R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com
R2_BUCKET_NAME=my-documents-bucket
R2_FOLDER_PREFIX=training-docs
```

Then use the CLI or library:

```bash
# CLI with .env file
doc2lora convert-r2 my-bucket --env-file .env --output adapter.json

# Or specify a custom .env file path
doc2lora convert-r2 my-bucket --env-file /path/to/my-credentials.env
```

```python
# Python API with .env file
from doc2lora import convert_from_r2

convert_from_r2(
    bucket_name="my-bucket",
    env_file=".env"  # Loads credentials from .env file
)

# The function will automatically load credentials from the .env file
# You can still override specific parameters if needed
convert_from_r2(
    bucket_name="my-bucket",
    env_file=".env",
    output_path="custom_output.json"  # Override specific settings
)
```

#### Method 3: Direct Parameters

```python
from doc2lora import convert_from_r2

convert_from_r2(
    bucket_name="my-bucket",
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    endpoint_url="https://your-account.r2.cloudflarestorage.com"
)
```

#### Method 4: CLI Parameters

```bash
doc2lora convert-r2 my-bucket \
    --aws-access-key-id "your-access-key" \
    --aws-secret-access-key "your-secret-key" \
    --endpoint-url "https://your-account.r2.cloudflarestorage.com"
```

### R2 Bucket Structure

Your R2 bucket can contain documents in any supported format:

```text
my-documents-bucket/
├── training-docs/          # Optional folder prefix
│   ├── legal/
│   │   ├── contract1.pdf
│   │   └── policy.md
│   ├── medical/
│   │   ├── research.docx
│   │   └── guidelines.txt
│   └── general/
│       ├── faq.json
│       └── manual.html
└── other-files/
    └── readme.txt
```

### Usage Examples

#### Convert Entire Bucket

```bash
doc2lora convert-r2 my-documents-bucket --output full_dataset_adapter.json
```

#### Convert Specific Folder

```bash
doc2lora convert-r2 my-documents-bucket \
    --folder-prefix training-docs/legal \
    --output legal_adapter.json
```

#### Programmatic Usage with Error Handling

```python
from doc2lora import convert_from_r2
import os

try:
    adapter_path = convert_from_r2(
        bucket_name="my-documents-bucket",
        folder_prefix="training-docs",
        output_path="my_adapter.json",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        # Optional: customize training parameters
        model_name="microsoft/DialoGPT-medium",
        epochs=5,
        batch_size=2
    )
    print(f"Successfully created adapter: {adapter_path}")
except Exception as e:
    print(f"Error: {e}")
```
