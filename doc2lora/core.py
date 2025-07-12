"""Core functionality for doc2lora library."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .parsers import DocumentParser
from .lora_trainer import LoRATrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert(
    documents_path: str,
    output_path: str = "lora_adapter.json",
    model_name: str = "microsoft/DialoGPT-small",
    max_length: int = 512,
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    **kwargs
) -> str:
    """
    Convert a folder of documents to LoRA adapter format.
    
    Args:
        documents_path: Path to folder containing documents
        output_path: Path to save the LoRA adapter JSON file
        model_name: Base model name for fine-tuning
        max_length: Maximum sequence length for tokenization
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        **kwargs: Additional arguments
        
    Returns:
        Path to the generated LoRA adapter file
    """
    logger.info(f"Starting document conversion from: {documents_path}")
    
    # Parse documents
    parser = DocumentParser()
    documents = parser.parse_directory(documents_path)
    
    if not documents:
        raise ValueError(f"No supported documents found in {documents_path}")
    
    logger.info(f"Found {len(documents)} documents to process")
    
    # Create LoRA trainer and train
    trainer = LoRATrainer(
        model_name=model_name,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Train the model
    trainer.train(
        documents=documents,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    # Save the LoRA adapter
    adapter_path = trainer.save_adapter(output_path)
    
    logger.info(f"LoRA adapter saved to: {adapter_path}")
    return adapter_path
