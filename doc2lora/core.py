"""Core functionality for doc2lora library."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .lora_trainer import LoRATrainer
from .parsers import DocumentParser
from .utils import cleanup_temp_directory, download_from_r2_bucket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert(
    input_data: Union[str, List[str], bytes, List[bytes], None] = None,
    documents_path: Optional[str] = None,
    output_path: str = "lora_adapter.json",
    model_name: str = "microsoft/DialoGPT-small",
    max_length: int = 512,
    batch_size: int = 4,
    num_epochs: Optional[int] = 3,
    max_steps: Optional[int] = None,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    **kwargs,
) -> str:
    """
    Convert documents to LoRA adapter format.

    Args:
        input_data: Content to convert - can be:
            - str: Single document content
            - List[str]: Multiple document contents
            - bytes: Single document as bytes
            - List[bytes]: Multiple documents as bytes
            - None: Use documents_path instead
        documents_path: Path to folder containing documents (used if input_data is None)
        output_path: Path to save the LoRA adapter JSON file
        model_name: Base model name for fine-tuning
        max_length: Maximum sequence length for tokenization
        batch_size: Training batch size
        num_epochs: Number of training epochs (ignored if max_steps is set)
        max_steps: Maximum number of training steps (overrides num_epochs if set)
        learning_rate: Learning rate for training
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        **kwargs: Additional arguments

    Returns:
        Path to the generated LoRA adapter file
    """
    if input_data is None and documents_path is None:
        raise ValueError("Either input_data or documents_path must be provided")

    if input_data is not None and documents_path is not None:
        raise ValueError("Cannot provide both input_data and documents_path")

    documents = []

    if input_data is not None:
        logger.info("Processing provided input data")
        documents = _process_input_data(input_data)
    else:
        logger.info(f"Starting document conversion from: {documents_path}")
        # Parse documents from directory
        parser = DocumentParser()
        documents = parser.parse_directory(documents_path)

    if not documents:
        raise ValueError("No documents to process")

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
        max_steps=max_steps,
        learning_rate=learning_rate,
    )

    # Save the LoRA adapter
    adapter_path = trainer.save_adapter(output_path)

    logger.info(f"LoRA adapter saved to: {adapter_path}")
    return adapter_path


def _process_input_data(input_data: Union[str, List[str], bytes, List[bytes]]) -> List[Dict[str, Any]]:
    """
    Process input data into document format.

    Args:
        input_data: Input data in various formats

    Returns:
        List of document dictionaries
    """
    documents = []

    # Handle single string
    if isinstance(input_data, str):
        documents.append({
            "content": input_data,
            "filename": "input_document_0.txt",
            "filepath": "memory://input_document_0.txt",
            "extension": ".txt",
            "size": len(input_data.encode('utf-8'))
        })

    # Handle list of strings
    elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
        for i, content in enumerate(input_data):
            documents.append({
                "content": content,
                "filename": f"input_document_{i}.txt",
                "filepath": f"memory://input_document_{i}.txt",
                "extension": ".txt",
                "size": len(content.encode('utf-8'))
            })

    # Handle single bytes
    elif isinstance(input_data, bytes):
        try:
            content = input_data.decode('utf-8')
            documents.append({
                "content": content,
                "filename": "input_document_0.txt",
                "filepath": "memory://input_document_0.txt",
                "extension": ".txt",
                "size": len(input_data)
            })
        except UnicodeDecodeError:
            raise ValueError("Unable to decode bytes input as UTF-8")

    # Handle list of bytes
    elif isinstance(input_data, list) and all(isinstance(item, bytes) for item in input_data):
        for i, byte_content in enumerate(input_data):
            try:
                content = byte_content.decode('utf-8')
                documents.append({
                    "content": content,
                    "filename": f"input_document_{i}.txt",
                    "filepath": f"memory://input_document_{i}.txt",
                    "extension": ".txt",
                    "size": len(byte_content)
                })
            except UnicodeDecodeError:
                raise ValueError(f"Unable to decode bytes input {i} as UTF-8")

    else:
        raise ValueError(f"Unsupported input_data type: {type(input_data)}")

    return documents


def convert_from_r2(
    bucket_name: str,
    folder_prefix: str = None,
    output_path: str = "lora_adapter.json",
    model_name: str = "microsoft/DialoGPT-small",
    max_length: int = 512,
    batch_size: int = 4,
    num_epochs: Optional[int] = 3,
    max_steps: Optional[int] = None,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    endpoint_url: str = None,
    region_name: str = "auto",
    cleanup_temp: bool = True,
    env_file: str = None,
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
        env_file: Path to .env file to load credentials from
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
            env_file=env_file,
        )

        # Use the existing convert function with the temporary directory
        adapter_path = convert(
            documents_path=temp_dir,
            output_path=output_path,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            num_epochs=num_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            **kwargs,
        )

        return adapter_path

    finally:
        # Clean up temporary directory
        if cleanup_temp and "temp_dir" in locals():
            cleanup_temp_directory(temp_dir)
