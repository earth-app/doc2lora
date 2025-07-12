"""Utility functions for doc2lora."""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_documents_path(path: str) -> Path:
    """
    Validate that the documents path exists and is a directory.
    
    Args:
        path: Path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If path doesn't exist
        NotADirectoryError: If path is not a directory
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if not path_obj.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    return path_obj


def estimate_training_time(num_documents: int, total_size_mb: float, batch_size: int = 4) -> str:
    """
    Estimate training time based on document count and size.
    
    Args:
        num_documents: Number of documents to process
        total_size_mb: Total size of documents in MB
        batch_size: Training batch size
        
    Returns:
        Estimated training time as a string
    """

    # Rough estimation based on empirical data
    # These are very rough estimates and will vary significantly based on hardware
    base_time_per_doc = 30  # seconds per document
    size_factor = max(1, total_size_mb / 10)  # additional time for large documents
    batch_factor = max(1, 4 / batch_size)  # batch size impact
    
    estimated_seconds = (num_documents * base_time_per_doc * size_factor * batch_factor)
    
    if estimated_seconds < 60:
        return f"{int(estimated_seconds)} seconds"
    elif estimated_seconds < 3600:
        return f"{int(estimated_seconds / 60)} minutes"
    else:
        return f"{estimated_seconds / 3600:.1f} hours"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def create_training_summary(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of the documents for training.
    
    Args:
        documents: List of parsed documents
        
    Returns:
        Summary dictionary with statistics
    """
    
    if not documents:
        return {
            'total_documents': 0,
            'total_size': 0,
            'file_types': {},
            'avg_content_length': 0
        }
    
    total_size = sum(doc['size'] for doc in documents)
    file_types = {}
    content_lengths = []
    
    for doc in documents:
        ext = doc['extension']
        file_types[ext] = file_types.get(ext, 0) + 1
        content_lengths.append(len(doc.get('content', '')))
    
    avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    
    return {
        'total_documents': len(documents),
        'total_size': total_size,
        'total_size_formatted': format_file_size(total_size),
        'file_types': file_types,
        'avg_content_length': int(avg_content_length),
        'estimated_training_time': estimate_training_time(
            len(documents), 
            total_size / (1024 * 1024)
        )
    }
