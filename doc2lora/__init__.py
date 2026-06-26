"""doc2lora: A library for fine-tuning LLMs using LoRA by using a folder of documents as input."""

from .core import convert, convert_from_r2

# single source of truth for the package version (read by pyproject + cli)
__version__ = "1.0.0"
__all__ = ["convert", "convert_from_r2", "__version__"]
