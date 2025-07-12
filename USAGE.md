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

Install from PyPI (when published):

```bash
pip install doc2lora
```

## Quick Start

### Using as a Library

```python
from doc2lora import convert

# Convert documents to LoRA adapter
adapter_path = convert(
    documents_path="path/to/your/documents",
    output_path="my_lora_adapter.json"
)

print(f"LoRA adapter created at: {adapter_path}")
```

### Using the CLI

```bash
# Basic usage
doc2lora convert path/to/documents --output adapter.json

# With custom parameters
doc2lora convert path/to/documents \
    --output adapter.json \
    --model microsoft/DialoGPT-medium \
    --epochs 5 \
    --batch-size 2

# Scan directory for supported files
doc2lora scan path/to/documents

# List supported formats
doc2lora formats
```

## Configuration Options

### Training Parameters

- `--model`: Base model for fine-tuning (default: microsoft/DialoGPT-small)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--max-length`: Maximum sequence length (default: 512)

### LoRA Parameters

- `--lora-r`: LoRA rank parameter (default: 16)
- `--lora-alpha`: LoRA alpha parameter (default: 32)
- `--lora-dropout`: LoRA dropout rate (default: 0.1)

## Supported Document Formats

- **Markdown** (.md): Full markdown parsing
- **Text** (.txt): Plain text files
- **PDF** (.pdf): Text extraction from PDF documents
- **HTML** (.html): Text extraction from HTML
- **Word** (.docx): Microsoft Word documents
- **CSV** (.csv): Comma-separated values
- **JSON** (.json): JSON data files
- **YAML** (.yaml, .yml): YAML configuration files
- **XML** (.xml): XML documents
- **LaTeX** (.tex): LaTeX source files

## Examples

See the `examples/` directory for sample usage and test documents.

## Testing

Run tests with:

```bash
python -m pytest tests/
```

## Hardware Requirements

- **GPU Recommended**: CUDA-compatible GPU for faster training
- **Memory**: At least 8GB RAM, 16GB+ recommended for larger models
- **Storage**: Adequate space for model weights and training data

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

1. **CUDA out of memory**: Reduce batch size or max length
2. **Missing dependencies**: Install with `pip install -e .`
3. **PDF parsing errors**: Ensure PyPDF2 is installed
4. **Slow training**: Use GPU and reduce dataset size for testing

### Performance Tips

- Use GPU acceleration when available
- Start with smaller batch sizes and increase gradually
- Monitor memory usage during training
- Use smaller models for initial testing
