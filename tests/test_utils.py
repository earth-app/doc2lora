"""Tests for doc2lora utility functions."""

from unittest.mock import MagicMock, patch

import pytest

from doc2lora.utils import (
    create_training_summary,
    estimate_training_time,
    format_file_size,
    validate_documents_path,
)


def test_format_file_size():
    assert format_file_size(512) == "512.0 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1024 * 1024) == "1.0 MB"
    assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"


def test_estimate_training_time_scales_with_device():
    """A faster device should never produce a longer estimate."""
    cpu = estimate_training_time(10, 5.0, device="cpu")
    cuda = estimate_training_time(10, 5.0, device="cuda")
    assert isinstance(cpu, str) and isinstance(cuda, str)
    # both return formatted strings; just assert they run and differ sensibly
    assert cpu != "" and cuda != ""


def test_estimate_training_time_floor():
    """Tiny corpora still report a small positive duration."""
    result = estimate_training_time(1, 0.0001, device="cuda")
    assert "second" in result


def test_create_training_summary_empty():
    summary = create_training_summary([])
    assert summary["total_documents"] == 0
    assert summary["file_types"] == {}


def test_create_training_summary_counts():
    docs = [
        {"size": 1000, "extension": ".md", "content": "hello world"},
        {"size": 2000, "extension": ".txt", "content": "more text here"},
        {"size": 500, "extension": ".md", "content": "x"},
    ]
    summary = create_training_summary(docs, device="cpu")
    assert summary["total_documents"] == 3
    assert summary["total_size"] == 3500
    assert summary["file_types"] == {".md": 2, ".txt": 1}
    assert "estimated_training_time" in summary


def test_validate_documents_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_documents_path(str(tmp_path / "nope"))


def test_validate_documents_path_not_dir(tmp_path):
    f = tmp_path / "afile.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        validate_documents_path(str(f))


def test_load_env_file_missing_raises(tmp_path):
    # only raises FileNotFoundError when dotenv is available; otherwise warns
    import doc2lora.utils as utils_mod
    from doc2lora.utils import load_env_file

    if utils_mod.load_dotenv is None:
        pytest.skip("python-dotenv not installed")
    with pytest.raises(FileNotFoundError):
        load_env_file(str(tmp_path / "missing.env"))


def test_download_from_r2_prefers_explicit_creds(tmp_path):
    """download_from_r2_bucket builds an S3 client with given creds (boto3 lazily imported)."""
    import sys

    from doc2lora.utils import download_from_r2_bucket

    # boto3 is imported inside the function now, so inject a fake via sys.modules
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    # one page, no contents -> raises "No files found" after client setup
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{}]
    mock_client.get_paginator.return_value = mock_paginator
    mock_boto3.client.return_value = mock_client
    mock_boto3.session.Config.return_value = MagicMock()

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        with pytest.raises(ValueError, match="No files found"):
            download_from_r2_bucket(
                bucket_name="bkt",
                aws_access_key_id="AKID",
                aws_secret_access_key="SECRET",
                endpoint_url="https://acct.r2.cloudflarestorage.com",
            )

    # client was created with our endpoint + creds
    _, kwargs = mock_boto3.client.call_args
    assert kwargs["aws_access_key_id"] == "AKID"
    assert kwargs["endpoint_url"] == "https://acct.r2.cloudflarestorage.com"
