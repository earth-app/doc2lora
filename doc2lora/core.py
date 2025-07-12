"""Core functionality for doc2lora library."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lora_trainer import LoRATrainer
from .parsers import DocumentParser
from .utils import download_from_r2_bucket, cleanup_temp_directory

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
    **kwargs,
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
        lora_dropout=lora_dropout,
    )

    # Train the model
    trainer.train(
        documents=documents,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    # Save the LoRA adapter
    adapter_path = trainer.save_adapter(output_path)

    logger.info(f"LoRA adapter saved to: {adapter_path}")
    return adapter_path


def convert_from_r2(
    bucket_name: str,
    folder_prefix: str = None,
    output_path: str = "lora_adapter.json",
    model_name: str = "microsoft/DialoGPT-small",
    max_length: int = 512,
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    endpoint_url: str = None,
    region_name: str = "auto",
    cleanup_temp: bool = True,
    **kwargs,
) -> str:
    """
    Convert documents from an R2 bucket to LoRA adapter format.

    Args:
        bucket_name: Name of the R2 bucket
        folder_prefix: Optional folder prefix within the bucket
        output_path: Path to save the LoRA adapter JSON file
        model_name: Base model name for fine-tuning
        max_length: Maximum sequence length for tokenization
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        aws_access_key_id: AWS access key ID for R2
        aws_secret_access_key: AWS secret access key for R2
        endpoint_url: R2 endpoint URL
        region_name: Region name (default: "auto" for R2)
        cleanup_temp: Whether to clean up temporary directory after processing
        **kwargs: Additional arguments

    Returns:
        Path to the generated LoRA adapter file
    """
    logger.info(f"Starting R2 bucket conversion from: {bucket_name}")

    # Download files from R2 bucket
    try:
        temp_dir = download_from_r2_bucket(
            bucket_name=bucket_name,
            folder_prefix=folder_prefix,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

        # Use the existing convert function with the temporary directory
        adapter_path = convert(
            documents_path=temp_dir,
            output_path=output_path,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            **kwargs,
        )

        return adapter_path

    finally:
        # Clean up temporary directory
        if cleanup_temp and 'temp_dir' in locals():
            cleanup_temp_directory(temp_dir)
