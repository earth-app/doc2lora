"""Utility functions for doc2lora."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


def load_env_file(env_file_path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_file_path: Path to the .env file. If None, looks for .env in current directory.
    """
    if load_dotenv is None:
        logger.warning(
            "python-dotenv not installed. Cannot load .env file. Install with: pip install python-dotenv"
        )
        return

    if env_file_path:
        env_path = Path(env_file_path)
        if not env_path.exists():
            raise FileNotFoundError(f".env file not found: {env_file_path}")
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from: {env_file_path}")
    else:
        # Look for .env in current directory
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            logger.debug("No .env file found in current directory")


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


def estimate_training_time(
    num_documents: int, total_size_mb: float, batch_size: int = 4
) -> str:
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

    estimated_seconds = num_documents * base_time_per_doc * size_factor * batch_factor

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
    for unit in ["B", "KB", "MB", "GB"]:
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
            "total_documents": 0,
            "total_size": 0,
            "file_types": {},
            "avg_content_length": 0,
        }

    total_size = sum(doc["size"] for doc in documents)
    file_types = {}
    content_lengths = []

    for doc in documents:
        ext = doc["extension"]
        file_types[ext] = file_types.get(ext, 0) + 1
        content_lengths.append(len(doc.get("content", "")))

    avg_content_length = (
        sum(content_lengths) / len(content_lengths) if content_lengths else 0
    )

    return {
        "total_documents": len(documents),
        "total_size": total_size,
        "total_size_formatted": format_file_size(total_size),
        "file_types": file_types,
        "avg_content_length": int(avg_content_length),
        "estimated_training_time": estimate_training_time(
            len(documents), total_size / (1024 * 1024)
        ),
    }


def download_from_r2_bucket(
    bucket_name: str,
    folder_prefix: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region_name: str = "auto",
    env_file: Optional[str] = None,
) -> str:
    """
    Download files from an R2 bucket to a temporary directory.

    Args:
        bucket_name: Name of the R2 bucket
        folder_prefix: Optional folder prefix within the bucket
        aws_access_key_id: AWS access key ID for R2
        aws_secret_access_key: AWS secret access key for R2
        endpoint_url: R2 endpoint URL (e.g., https://your-account.r2.cloudflarestorage.com)
        region_name: Region name (default: "auto" for R2)
        env_file: Path to .env file to load credentials from

    Returns:
        Path to temporary directory containing downloaded files

    Raises:
        ImportError: If boto3 is not installed
        NoCredentialsError: If credentials are not provided
        ClientError: If there's an error accessing the bucket
    """
    # Load environment variables from .env file if provided
    if env_file:
        load_env_file(env_file)

    # Get credentials from environment variables if not provided directly
    if not aws_access_key_id:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    if not aws_secret_access_key:
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not endpoint_url:
        endpoint_url = os.getenv("R2_ENDPOINT_URL")

    if boto3 is None:
        raise ImportError(
            "boto3 is required for R2 bucket support. Install with: pip install boto3"
        )

    # Create S3 client for R2
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
        region_name=region_name,
    )

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="doc2lora_r2_")
    temp_path = Path(temp_dir)

    logger.info(f"Downloading files from R2 bucket '{bucket_name}' to {temp_dir}")

    try:
        # List objects in bucket
        paginator = s3_client.get_paginator("list_objects_v2")

        # Configure pagination parameters
        page_kwargs = {"Bucket": bucket_name}
        if folder_prefix:
            page_kwargs["Prefix"] = folder_prefix.rstrip("/") + "/"

        downloaded_count = 0

        for page in paginator.paginate(**page_kwargs):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]

                # Skip directory markers
                if key.endswith("/"):
                    continue

                # Create local file path
                if folder_prefix:
                    # Remove folder prefix from the key for local path
                    relative_key = key[len(folder_prefix.rstrip("/") + "/") :]
                else:
                    relative_key = key

                local_file_path = temp_path / relative_key

                # Create directories if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Download file
                try:
                    s3_client.download_file(bucket_name, key, str(local_file_path))
                    downloaded_count += 1
                    logger.debug(f"Downloaded: {key} -> {local_file_path}")
                except ClientError as e:
                    logger.error(f"Error downloading {key}: {e}")
                    continue

        if downloaded_count == 0:
            logger.warning("No files were downloaded from the bucket")
        else:
            logger.info(f"Downloaded {downloaded_count} files from R2 bucket")

        return str(temp_path)

    except NoCredentialsError:
        raise NoCredentialsError(
            "AWS credentials not found. Please provide aws_access_key_id and aws_secret_access_key"
        )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            raise ValueError(f"Bucket '{bucket_name}' does not exist")
        elif error_code == "AccessDenied":
            raise ValueError("Access denied. Check your credentials and permissions")
        else:
            raise ClientError(f"Error accessing R2 bucket: {e}", e.operation_name)


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Clean up a temporary directory and all its contents.

    Args:
        temp_dir: Path to the temporary directory to clean up
    """
    import shutil

    try:
        shutil.rmtree(temp_dir)
        logger.debug(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Error cleaning up temporary directory {temp_dir}: {e}")
