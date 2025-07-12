# doc2lora Installation and Usage Guide

## Installation

### Option 1: From Source (Recommended for development)

```bash
# Clone the repository
git clone https://github.com/earth-app/doc2lora.git
cd doc2lora

# Install in development mode
pip install -e .

# Verify installation
doc2lora --help
```

### Option 2: Direct Installation

```bash
# Install all dependencies
pip install torch transformers peft datasets
pip install PyPDF2 python-docx beautifulsoup4 PyYAML click

# Install the package
pip install -e .
```

### Option 3: Minimal Installation (for testing parsing only)

```bash
pip install click PyYAML
# Then install just the core package without ML dependencies
```

## Using doc2lora with Mistral Models

### Command Line Interface

#### Basic Usage with Mistral-7B-Instruct

```bash
# Convert documents to LoRA adapter for Mistral
doc2lora convert ./my_documents \
  --output mistral_adapter.json \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32
```

#### Advanced Configuration

```bash
# Fine-tuned configuration for Mistral
doc2lora convert ./training_docs \
  --output ./output/mistral_custom_adapter.json \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --max-length 1024 \
  --batch-size 1 \
  --epochs 5 \
  --learning-rate 5e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --verbose
```

#### Scan Documents Before Training

```bash
# Check what documents will be processed
doc2lora scan ./my_documents

# List supported formats
doc2lora formats
```

### Python API Usage

#### Basic Python Usage

```python
from doc2lora import convert

# Convert documents for Mistral model
adapter_path = convert(
    documents_path="./training_documents",
    output_path="./mistral_adapter.json",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    num_epochs=3,
    batch_size=2,
    learning_rate=1e-4,
    max_length=1024
)

print(f"Adapter saved to: {adapter_path}")
```

#### Advanced Python Usage

```python
from doc2lora.core import convert
from doc2lora.parsers import DocumentParser
from doc2lora.lora_trainer import LoRATrainer

# First, parse and inspect documents
parser = DocumentParser()
documents = parser.parse_directory("./my_docs")

print(f"Found {len(documents)} documents")
for doc in documents[:3]:  # Show first 3
    print(f"- {doc['filename']}: {len(doc['content'])} characters")

# Train with custom configuration
adapter_path = convert(
    documents_path="./my_docs",
    output_path="./custom_mistral.json",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_length=1024,
    batch_size=1,  # Reduce if you get memory errors
    num_epochs=3,
    learning_rate=1e-4,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
```

## Using with Cloudflare Workers AI

### 1. Prepare Your Adapter

After training, you'll have two outputs:
- `mistral_adapter.json` - Metadata file
- `mistral_adapter_adapter/` - Directory with actual adapter weights

### 2. Upload to Cloudflare

```bash
# Upload adapter to R2 storage
wrangler r2 object put my-bucket/adapters/mistral-custom ./mistral_adapter_adapter/ --recursive

# Or use the CF dashboard to upload the adapter directory
```

### 3. Use in Worker

```javascript
export default {
  async fetch(request, env, ctx) {
    const ai = new Ai(env.AI);

    const messages = [
      {
        role: "system",
        content: "You are a helpful assistant trained on my custom documents."
      },
      {
        role: "user",
        content: "How do I implement the features described in the documentation?"
      }
    ];

    const response = await ai.run('@cf/mistralai/mistral-7b-instruct-v0.2-lora', {
      messages: messages,
      lora: "mistral-custom"  // Your adapter name
    });

    return new Response(JSON.stringify(response), {
      headers: { "Content-Type": "application/json" }
    });
  }
}
```

## Best Practices for Mistral Training

### 1. Data Preparation

- Use 50-500 documents for good results
- Ensure documents are well-formatted and relevant
- Mix different types of content (docs, examples, Q&A)

### 2. Hyperparameter Tuning

```python
# Conservative settings (recommended starting point)
conservative_config = {
    "learning_rate": 1e-4,
    "batch_size": 1,
    "num_epochs": 3,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_length": 512
}

# Aggressive settings (for larger datasets)
aggressive_config = {
    "learning_rate": 5e-5,
    "batch_size": 2,
    "num_epochs": 5,
    "lora_r": 32,
    "lora_alpha": 64,
    "max_length": 1024
}
```

### 3. Memory Management

```bash
# If you get CUDA out of memory errors:
doc2lora convert ./docs \
  --batch-size 1 \
  --max-length 512 \
  --model mistralai/Mistral-7B-Instruct-v0.2

# For CPU-only training (slower but works on any machine):
CUDA_VISIBLE_DEVICES="" doc2lora convert ./docs --model mistralai/Mistral-7B-Instruct-v0.2
```

### 4. Monitoring Training

```python
import logging
logging.basicConfig(level=logging.INFO)

# The trainer will show:
# - Loading progress
# - Training loss
# - Trainable parameters count
# - Checkpoint saves
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce batch size and sequence length
   doc2lora convert ./docs --batch-size 1 --max-length 256
   ```

2. **Model Download Issues**

   ```python
   # Pre-download the model
   from transformers import AutoTokenizer, AutoModelForCausalLM

   model_name = "mistralai/Mistral-7B-Instruct-v0.2"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   ```

3. **Permission/Access Issues**

   ```bash
   # Make sure you have access to the Mistral model
   huggingface-cli login
   ```

### Supported Mistral Models

- `mistralai/Mistral-7B-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.2` (recommended)
- `mistralai/Mistral-7B-Instruct-v0.3`

All Mistral models will automatically use the correct target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
